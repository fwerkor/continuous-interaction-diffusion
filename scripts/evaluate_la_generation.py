from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import torch

from cid.language_alignment import load_la_backbone
from cid.model.diffusion import CIDDiffusionScheduler

PROMPTS = [
    ("greeting", "Hello!"),
    ("identity", "Who are you?"),
    ("help", "How can you help me?"),
    ("follow_instructions", "Respond using exactly three words: one, two, three."),
    ("definition", "Explain what TCP is in simple English."),
    ("networking", "Explain the difference between TCP flow control and congestion control."),
    ("history", "Give a short overview of Chinese history in English."),
    (
        "summary",
        "Summarize: The server was overloaded, requests timed out, "
        "and an engineer increased capacity. What happened?",
    ),
    ("rewriting", "Rewrite this sentence clearly: Me and him wasn't going to no store."),
    ("translation", "Translate this sentence into English: 今天的天气很好。"),
    ("math", "What is 17 multiplied by 23?"),
    ("arithmetic", "What is 49 plus 52?"),
    ("reasoning", "If Alice is older than Bob and Bob is older than Carol, who is the youngest?"),
    ("list", "List three advantages of testing software."),
    ("algorithm", "Explain binary search in two sentences."),
    (
        "python_factorial",
        "Write a correct Python function def factorial(n) that returns n! for n >= 0.",
    ),
    ("python_reverse", "Write a Python function def reverse_string(s) that reverses a string."),
    ("python_even", "Write a Python function def is_even(n) that returns True for even integers."),
    ("format", "Reply with a JSON object with keys name and age, using name Alice and age 20."),
]


def diagnostics(label: str, output: str) -> dict[str, bool]:
    words = re.findall(r"[A-Za-z0-9_]+", output.lower())
    repeated = sum(words[i] == words[i - 1] for i in range(1, len(words)))
    triple = any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2))
    seen = Counter(words)
    chars = len(output.strip())
    result = {
        "empty": chars == 0,
        "repetition": triple
        or repeated >= 3
        or (len(words) >= 12 and seen.most_common(1)[0][1] / len(words) >= 0.4),
        "leaked_template": "[/Answer]" in output or "[Question]" in output or "[Answer]" in output,
    }
    if label == "math":
        result["numeric_correct"] = bool(re.search(r"\b391\b", output))
    elif label == "arithmetic":
        result["numeric_correct"] = bool(re.search(r"\b101\b", output))
    elif label in {"python_factorial", "python_reverse", "python_even"}:
        code = output
        fence = re.escape(chr(96) * 3)
        fenced = re.search(
            fence + r"(?:python)?\s*(.*?)" + fence, output, flags=re.DOTALL | re.IGNORECASE
        )
        if fenced:
            code = fenced.group(1)
        try:
            ast.parse(code)
            result["syntactically_valid_code"] = True
        except SyntaxError:
            result["syntactically_valid_code"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--canvas", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model, tok = load_la_backbone(args.model, dtype=torch.float32)
    model.to("cuda").eval()
    scheduler = CIDDiffusionScheduler(
        mask_token_id=int(tok.mask_token_id),
        eos_token_id=int(tok.eos_token_id),
        display_initialization="preallocated",
    )
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, prompt in PROMPTS:
            prompt_ids = tok.encode(prompt, add_special_tokens=True)
            prefix = torch.tensor([prompt_ids], device="cuda")
            display = torch.full(
                (1, args.canvas), tok.mask_token_id, device="cuda", dtype=torch.long
            )
            display[:, -1] = tok.eos_token_id
            stable = 0
            for _step in range(args.steps):
                ids = torch.cat((prefix, display), dim=1)
                logits = model(ids)[:, len(prompt_ids) :].float()
                revised = scheduler.refine_display(
                    display,
                    logits,
                    reveal_fraction=0.125,
                    revision_fraction=0.25,
                    revision_margin=0.1,
                )
                stable = stable + 1 if torch.equal(revised, display) else 0
                display = revised
                if stable >= 3:
                    break
            tokens = display[0].tolist()
            if tok.eos_token_id in tokens:
                tokens = tokens[: tokens.index(tok.eos_token_id)]
            answer = tok.decode(tokens, skip_special_tokens=True)
            row = {
                "task": name,
                "prompt": prompt,
                "steps": _step + 1,
                "answer": answer,
                "diagnostics": diagnostics(name, answer),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    counts = Counter()
    for row in rows:
        for key, value in row["diagnostics"].items():
            if value:
                counts[key] += 1
    result = {
        "checkpoint": args.model,
        "n_tasks": len(rows),
        "failure_signals": dict(counts),
        "notes": "Heuristic quality signals are not semantic evaluation or benchmark accuracy.",
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

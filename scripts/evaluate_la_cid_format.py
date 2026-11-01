from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cid.language_alignment import load_la_backbone
from cid.model.diffusion import CIDDiffusionScheduler

PROMPTS = [
    ("greeting", "Hello!"),
    ("identity", "Who are you?"),
    ("definition", "What is a computer?"),
    ("tcp", "Explain TCP flow control in two sentences."),
    ("history", "Briefly describe Chinese history."),
    ("math", "What is 17 multiplied by 23?"),
    ("reverse", "Write a Python function to reverse a string."),
    ("sort", "Write a Python function that sorts a list of integers."),
    (
        "summary",
        "Summarize: the server was overloaded, requests timed out, and capacity was increased.",
    ),
    ("format", 'Reply in JSON: name Alice, age 20.'),
    ("zh_hello", "你好"),
    ("zh_identity", "请用中文介绍一下自己。"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--canvas", type=int, default=64)
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--reveal-fraction", type=float, default=0.25)
    parser.add_argument("--revision-fraction", type=float, default=0.0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    model, tok = load_la_backbone(args.model, dtype=torch.float32)
    model.cuda().eval()
    sep = int(tok.convert_tokens_to_ids("<|cid_answer|>"))
    assert sep != tok.unk_token_id
    scheduler = CIDDiffusionScheduler(
        mask_token_id=int(tok.mask_token_id),
        eos_token_id=int(tok.eos_token_id),
        display_initialization="preallocated",
        allow_pending_eos=True,
    )
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, question in PROMPTS:
            prefix_ids = tok.encode(question, add_special_tokens=True) + [sep]
            prefix = torch.tensor([prefix_ids], device="cuda")
            display = torch.full(
                (1, args.canvas), int(tok.mask_token_id), dtype=torch.long, device="cuda"
            )
            display[0, -1] = int(tok.eos_token_id)
            stable = 0
            for _step in range(args.steps):
                backbone_display = display.clone()
                backbone_display[backbone_display == tok.eos_token_id] = tok.mask_token_id
                logits = model(torch.cat((prefix, backbone_display), dim=1))[
                    :, prefix.shape[1] :
                ].float()
                updated = scheduler.refine_display(
                    display,
                    logits,
                    reveal_fraction=args.reveal_fraction,
                    revision_fraction=args.revision_fraction,
                    revision_margin=0.1,
                )
                stable = stable + 1 if torch.equal(updated, display) else 0
                display = updated
                if stable >= 3:
                    break
            ids = display[0].tolist()
            stop = ids.index(tok.eos_token_id)
            answer = tok.decode(ids[:stop], skip_special_tokens=True)
            row = {
                "task": name,
                "prompt": question,
                "steps": _step + 1,
                "eos_position": stop,
                "unresolved_before_eos": ids[:stop].count(tok.mask_token_id),
                "native_marker_leak": any(
                    tag in answer for tag in ("[/Answer]", "[Answer]", "[Question]", "[/Question]")
                ),
                "answer": answer,
            }
            rows.append(row)
            print("PROBE", json.dumps(row, ensure_ascii=False), flush=True)
    output = {
        "checkpoint": args.model,
        "canvas": args.canvas,
        "steps": args.steps,
        "reveal_fraction": args.reveal_fraction,
        "revision_fraction": args.revision_fraction,
        "n_probes": len(rows),
        "native_marker_leaks": sum(x["native_marker_leak"] for x in rows),
        "incomplete": sum(x["unresolved_before_eos"] > 0 for x in rows),
        "rows": rows,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({
        k: v for k, v in output.items() if k != "rows"
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

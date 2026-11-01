from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cid.language_alignment import load_la_backbone

TASKS = [
    ("hello", "Hello!"),
    ("identity", "Who are you?"),
    ("tip", "Give one short tip for writing clearer code."),
    ("history", "Give a short overview of Chinese history in English."),
    ("explanation", "Explain TCP congestion control in two sentences."),
    ("reason", "What is 17 times 23?"),
    (
        "code",
        "Write def factorial(n) computing n factorial for n >= 0.",
    ),
    ("rewrite", "Rewrite this sentence correctly: Me and him wasn't going to no store."),
    ("list", "List three benefits of unit testing."),
    ("format", "Provide a JSON object with name Alice and age 20."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--native-template", action="store_true")
    ap.add_argument("--max-answer-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(3)
    model, tok = load_la_backbone(args.model, dtype=torch.float32)
    model.to("cuda").eval()
    output = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for label, question in TASKS:
            prompt = (
                tok.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if args.native_template
                else question
            )
            prefix = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device="cuda")
            canvas = torch.full(
                (1, args.max_answer_tokens), tok.mask_token_id, dtype=torch.long, device="cuda"
            )
            for _ in range(args.max_answer_tokens):
                positions = (canvas[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
                if len(positions) == 0:
                    break
                inp = torch.cat((prefix, canvas), dim=1)
                out = model(inp)[0, prefix.shape[1] :, :][positions].float()
                out[:, len(tok) :] = -torch.inf
                for token_id in tok.all_special_ids:
                    if token_id != tok.eos_token_id and token_id < out.shape[1]:
                        out[:, token_id] = -torch.inf
                probs = out.softmax(-1)
                confidence, ids = probs.max(-1)
                best = confidence.argmax()
                canvas[0, positions[best]] = ids[best]
            decoded = tok.decode(canvas[0], skip_special_tokens=True)
            text = decoded.split("[/Answer]")[0].strip()
            row = {
                "model": args.model,
                "native": args.native_template,
                "task": label,
                "text": text,
                "end_marker": "[/Answer]" in decoded,
            }
            output.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

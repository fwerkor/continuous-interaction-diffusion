from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from research_native_decode import TASKS

from cid.language_alignment import load_la_backbone
from cid.model.diffusion import CIDDiffusionScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--reveal", type=float, default=0.015625)
    ap.add_argument("--revision", type=float, default=0.0)
    ap.add_argument("--native-template", action="store_true")
    ap.add_argument("--ban-special", action="store_true")
    ap.add_argument("--max-steps", type=int, default=64)
    args = ap.parse_args()
    model, tok = load_la_backbone(args.model, dtype=torch.float32)
    model.to("cuda").eval()
    scheduler = CIDDiffusionScheduler(
        mask_token_id=tok.mask_token_id,
        eos_token_id=tok.eos_token_id,
        display_initialization="preallocated",
    )
    samples = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
            display = torch.full(
                (1, args.canvas), tok.mask_token_id, device="cuda", dtype=torch.long
            )
            display[0, -1] = tok.eos_token_id
            stable = 0
            for _step in range(args.max_steps):
                logits = model(torch.cat((prefix, display), dim=1))[:, prefix.shape[1] :].float()
                if args.ban_special:
                    logits[:, :, len(tok) :] = -torch.inf
                    for special in tok.all_special_ids:
                        if special != tok.eos_token_id and special < logits.shape[-1]:
                            logits[:, :, special] = -torch.inf
                after = scheduler.refine_display(
                    display,
                    logits,
                    reveal_fraction=args.reveal,
                    revision_fraction=args.revision,
                    revision_margin=0.1,
                )
                stable = stable + 1 if torch.equal(after, display) else 0
                display = after
                if stable >= 3:
                    break
            ts = display[0].tolist()
            boundary = ts.index(tok.eos_token_id) if tok.eos_token_id in ts else len(ts)
            answer = (
                tok.decode(ts[:boundary], skip_special_tokens=True).split("[/Answer]")[0].strip()
            )
            record = {
                "task": label,
                "text": answer,
                "steps": _step + 1,
                "native_template": args.native_template,
                "reveal": args.reveal,
                "revision": args.revision,
                "ban_special": args.ban_special,
            }
            samples.append(record)
            print("RESULT", json.dumps(record, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

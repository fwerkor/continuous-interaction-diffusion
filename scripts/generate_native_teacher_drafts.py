from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

from cid.language_alignment import load_la_backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--attempts", type=int, default=6000)
    p.add_argument("--canvas", type=int, default=64)
    p.add_argument("--canvas-buckets", default=None)
    p.add_argument("--languages", default="en")
    p.add_argument("--seed", type=int, default=20261101)
    args = p.parse_args()

    random.seed(args.seed)
    buckets = (
        tuple(int(value) for value in args.canvas_buckets.split(","))
        if args.canvas_buckets else (args.canvas,)
    )
    if not buckets or min(buckets) < 16 or max(buckets) > 192:
        raise ValueError("teacher canvas buckets must be in [16, 192]")
    languages = frozenset(s.strip() for s in args.languages.split(","))
    torch.set_num_threads(4)
    model, tok = load_la_backbone(args.model, dtype=torch.float32)
    model.to("cuda").eval()
    with open(args.data, encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    random.shuffle(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = failed = 0
    lengths = Counter()
    mask_id = int(tok.mask_token_id)
    with (
        output.open("w", encoding="utf-8") as stream,
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for row in records[: args.attempts]:
            if selected >= args.max_examples:
                break
            question = row["prompt"]
            if row.get("language") not in languages:
                continue
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device="cuda")
            if prefix.shape[-1] > 240:
                failed += 1
                continue
            width = random.choice(buckets)
            canvas = torch.full((1, width), mask_id, dtype=torch.long, device="cuda")
            for _ in range(width):
                remaining = torch.nonzero(canvas[0].eq(mask_id), as_tuple=False).flatten()
                if not len(remaining):
                    break
                ids = torch.cat((prefix, canvas), dim=-1)
                logits = model(ids)[0, prefix.shape[-1] :, :][remaining].float()
                logits[:, len(tok) :] = -1e9
                for special in tok.all_special_ids:
                    if special != tok.eos_token_id and special < logits.shape[-1]:
                        logits[:, special] = -1e9
                probs = logits.softmax(-1)
                confidence, guesses = probs.max(-1)
                best = int(confidence.argmax())
                canvas[0, remaining[best]] = guesses[best]
            raw = tok.decode(canvas[0], skip_special_tokens=True)
            if "[/Answer]" not in raw:
                failed += 1
                continue
            text = raw.split("[/Answer]")[0].strip()
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            if not 8 <= n_tokens <= width - 4:
                failed += 1
                continue
            if any(tag in text for tag in ("[Question]", "[Answer]", "[/Question]")):
                failed += 1
                continue
            digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
            stream.write(
                json.dumps(
                    {
                        "prompt": question,
                        "answer": text,
                        "source": "native-350m-teacher",
                        "language": row.get("language", "en"),
                        "answer_tokens": n_tokens,
                        "canvas": width,
                        "prompt_sha256": digest,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            selected += 1
            lengths[(n_tokens // 8) * 8] += 1
            if selected % 100 == 0:
                stream.flush()
                print("TEACHER_DISTILLED", selected, "rejected", failed, flush=True)
    print("COMPLETE", json.dumps({
        "written": selected,
        "rejected": failed,
        "output": str(output),
        "length_bins_8": sorted(lengths.items()),
    }))


if __name__ == "__main__":
    main()

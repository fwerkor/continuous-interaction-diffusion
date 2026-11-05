#!/usr/bin/env python3
"""Complete math coverage, split held-out validation, and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cid.language_sft import LanguageSFTExample, validate_language_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=48000)
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be in (0,1)")
    rows = [LanguageSFTExample.from_dict(json.loads(line)) for line in args.input.open()]
    existing = {hashlib.sha256((r.prompt + "\0" + r.answer).encode()).digest() for r in rows}
    to_add = max(0, args.total - len(rows))
    if to_add:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        candidates = []
        for record in ds:
            question = record.get("question", "").strip()
            answer = record.get("answer", "")
            solution = answer.rsplit("####", 1)[0].strip()
            solution = re.sub(r"<<[^<>]*>>", "", solution)
            solution = re.sub(r"\n{3,}", "\n\n", solution).strip()
            if not 50 <= len(solution) <= 3000 or not question:
                continue
            prompt = "User:\n" + question
            h = hashlib.sha256((prompt + "\0" + solution).encode()).digest()
            if h in existing:
                continue
            candidates.append(
                (
                    h,
                    LanguageSFTExample(
                        "gsm8k-" + hashlib.sha256(question.encode()).hexdigest()[:20],
                        prompt,
                        solution,
                        "mathematics",
                        "openai/gsm8k",
                        "mit",
                    ),
                )
            )
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        if len(candidates) < to_add:
            raise ValueError("insufficient GSM8K math examples")
        rows.extend(r for _, r in candidates[:to_add])
    groups = {}
    for row in rows:
        groups.setdefault(row.category, []).append(row)
    train = []
    valid = []
    for category_rows in groups.values():
        category_rows.sort(
            key=lambda row: hashlib.sha256((str(args.seed) + row.example_id).encode()).digest()
        )
        held = max(1, round(len(category_rows) * args.validation_fraction))
        valid.extend(category_rows[:held])
        train.extend(category_rows[held:])
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(valid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split, items in (("train", train), ("validation", valid)):
        output = args.out_dir / f"language-sft-{split}.jsonl"
        tmp = output.with_suffix(".jsonl.building")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for row in items:
                    f.write(
                        json.dumps(row.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
            tmp.replace(output)
        finally:
            tmp.unlink(missing_ok=True)
        results[split] = validate_language_sft(output)
    repo_sha = HfApi(token=False).dataset_info("openai/gsm8k").sha
    report = {
        "dataset_version": "1.2",
        "base_cid_dataset": "fwerkor/CID-Dataset",
        "base_cid_version": "1.1",
        "language_sft_format": "cid-language-sft-v1",
        "train": results["train"],
        "validation": results["validation"],
        "total_examples": len(rows),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "added_gsm8k_math_examples": to_add,
        "math_upstream": {"repo": "openai/gsm8k", "revision": repo_sha, "license": "mit"},
        "recommended_cid_token_fraction": 0.8,
        "recommended_language_sft_token_fraction": 0.2,
        "tct_supervision": False,
        "tool_supervision": False,
    }
    (args.out_dir / "language-sft-v1.2.manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

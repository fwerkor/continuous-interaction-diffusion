#!/usr/bin/env python3
"""Fill the final 300k CID language corpus with multilingual Aya annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expand_language_sft_v12 import prompt_key
from prepare_language_sft_v12 import REJECT, eligible

from cid.language_sft import LanguageSFTExample, validate_language_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--language-cap", type=int, default=1600)
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    original_report = json.loads((args.input_dir / "language-sft-v1.2.manifest.json").read_text())
    old_total = int(original_report["actual_total"])
    missing = args.target_total - old_total
    if missing <= 0:
        raise ValueError("Aya augmentation requires a positive deficit")
    original_train = []
    original_validation = []
    seen = set()
    for split, target in (("validation", original_validation), ("train", original_train)):
        with (args.input_dir / f"language-sft-{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = LanguageSFTExample.from_dict(json.loads(line))
                key = prompt_key(row.prompt)
                if key in seen:
                    raise ValueError("duplicate prompt in parent dataset")
                seen.add(key)
                target.append(row)
    if len(original_train) + len(original_validation) != old_total:
        raise ValueError("parent sample count mismatch")

    repo = "CohereForAI/aya_dataset"
    revision = HfApi(token=False).dataset_info(repo).sha
    print("aya_revision", revision, "needed", missing, flush=True)
    data = load_dataset(repo, split="train", streaming=True, revision=revision)
    data = data.shuffle(seed=args.seed, buffer_size=8192)
    added = []
    languages = Counter()
    audit = Counter()
    for index, example in enumerate(data):
        if index % 25000 == 0 and index:
            print("aya_progress", index, len(added), dict(languages), flush=True)
        lang = str(example.get("language_code", "")).strip().lower()
        if lang in ("en", "eng", ""):
            continue
        if languages[lang] >= args.language_cap:
            continue
        question = str(example.get("inputs") or "").strip()
        answer = str(example.get("targets") or "").strip()
        prompt = "User:\n" + question
        if not eligible(prompt, answer):
            audit["length_or_transport"] += 1
            continue
        if any(marker in prompt or marker in answer for marker in REJECT):
            audit["transport"] += 1
            continue
        pk = prompt_key(prompt)
        if pk in seen:
            audit["duplicate_prompt"] += 1
            continue
        unique_id = hashlib.sha256((lang + "\0" + prompt).encode()).hexdigest()[:20]
        row = LanguageSFTExample(
            "aya-" + unique_id, prompt, answer, "multilingual", repo, "apache-2.0"
        )
        seen.add(pk)
        languages[lang] += 1
        added.append(row)
        if len(added) >= missing:
            break
    if len(added) != missing:
        raise ValueError(f"Aya provided only {len(added)} eligible examples, need {missing}")
    added.sort(
        key=lambda item: hashlib.sha256((str(args.seed) + item.example_id).encode()).digest()
    )
    need_validation = round(args.target_total * 0.03) - len(original_validation)
    if not 0 <= need_validation <= len(added):
        raise ValueError("invalid validation allocation")
    validation = original_validation + added[:need_validation]
    train = original_train + added[need_validation:]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(original_report)
    report["source_revision"][repo] = revision
    report["licenses"][repo] = "apache-2.0"
    report["actual_total"] = args.target_total
    report["target_total"] = args.target_total
    report["accepted_sources"]["aya_dataset"] = len(added)
    report["new_categories"]["multilingual"] += len(added)
    report["language_distribution_aya"] = dict(languages)
    report["aya_filtered"] = dict(audit)
    report["cross_split_prompt_leakage"] = 0
    for split, rows in (("train", train), ("validation", validation)):
        output = args.output_dir / f"language-sft-{split}.jsonl"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
        print("validating", split, len(rows), flush=True)
        report[split] = validate_language_sft(output)
    total = report["train"]["examples"] + report["validation"]["examples"]
    if total != args.target_total:
        raise ValueError("final sample count mismatch")
    (args.output_dir / "language-sft-v1.2.manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "COMPLETE",
        total,
        report["train"]["examples"],
        report["validation"]["examples"],
        "languages",
        dict(languages),
        flush=True,
    )


if __name__ == "__main__":
    main()

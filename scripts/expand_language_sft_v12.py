#!/usr/bin/env python3
"""Create a 300k-example, prompt-disjoint extension of CID v1.2 language_sft."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_language_sft_v12 import SOURCES, classify, dialog_pairs, eligible

from cid.language_sft import (
    LANGUAGE_CATEGORIES,
    LanguageSFTExample,
    validate_language_sft,
)


def prompt_key(text: str) -> bytes:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).digest()


def pair_key(row: LanguageSFTExample) -> bytes:
    return hashlib.sha256((row.prompt + "\0" + row.answer).encode()).digest()


def distribute_total(total: int) -> dict[str, int]:
    # Source composition is heterogeneous; these are safety ceilings rather
    # than hard targets. Fixed 20/20/15/... quotas discarded most UltraChat
    # conversations and could not reach 300k unique prompts.
    ceilings = {
        "conversation": 0.45,
        "knowledge": 0.25,
        "writing": 0.20,
        "summarization": 0.15,
        "coding": 0.15,
        "mathematics": 0.15,
        "instruction_following": 0.15,
        "multilingual": 0.09,
    }
    return {category: round(total * fraction) for category, fraction in ceilings.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--scan-limit", type=int, default=1500000)
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=25000)
    args = parser.parse_args()

    if args.target_total < 48000 or not (0 < args.validation_fraction < 0.2):
        raise ValueError("invalid target or validation fraction")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    rng = random.Random(args.seed)
    quotas = distribute_total(args.target_total)
    originals: dict[bytes, tuple[LanguageSFTExample, str]] = {}
    # Preserve held-out examples first: an existing training duplicate must not
    # contaminate the expanded validation split.
    for split in ("validation", "train"):
        with (args.existing / f"language-sft-{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = LanguageSFTExample.from_dict(json.loads(line))
                originals.setdefault(prompt_key(row.prompt), (row, split))

    seen_prompts = set(originals)
    seen_pairs = {pair_key(row) for row, _ in originals.values()}
    buckets: dict[str, list[LanguageSFTExample]] = {key: [] for key in LANGUAGE_CATEGORIES}
    originals_by_category = Counter(row.category for row, _ in originals.values())
    entries = sum(originals_by_category.values())
    collected_by_source: Counter[str] = Counter()
    scanned = Counter()
    filtered = Counter()
    revisions = {}
    print("preserved_unique_origins", entries, "cross_split_dedup", 48000 - entries, flush=True)

    source_caps = {
        "tulu3": round(args.target_total * 0.63),
        "ultrachat": round(args.target_total * 0.50),
        "oasst1": round(args.target_total * 0.04),
    }
    start_time = time.monotonic()
    for key, (repo, split, license_id) in SOURCES.items():
        if key == "oasst1":
            # The old v1.2 already contains all useful reviewed OASST1 parent-child pairs.
            continue
        revisions[repo] = HfApi(token=False).dataset_info(repo).sha
        print("source", key, repo, "revision", revisions[repo], flush=True)
        dataset = load_dataset(repo, split=split, streaming=True, revision=revisions[repo])
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
        for index, data in enumerate(dataset):
            if index >= args.scan_limit or collected_by_source[key] >= source_caps[key]:
                break
            scanned[key] += 1
            if args.log_every and scanned[key] % args.log_every == 0:
                print(
                    "progress",
                    key,
                    scanned[key],
                    "accepted",
                    collected_by_source[key],
                    "total",
                    entries + sum(map(len, buckets.values())),
                    "categories",
                    {cat: originals_by_category[cat] + len(rows) for cat, rows in buckets.items()},
                    "elapsed_s",
                    int(time.monotonic() - start_time),
                    flush=True,
                )
            messages = data.get("messages")
            if not isinstance(messages, list):
                filtered["no_messages"] += 1
                continue
            dialogs = list(dialog_pairs(messages))
            if not dialogs:
                filtered["no_dialog"] += 1
                continue
            prompt, answer = dialogs[-1]
            if not eligible(prompt, answer):
                filtered["length_transport"] += 1
                continue
            cat = classify(prompt, str(data.get("source", key)), str(data.get("lang", "")))
            if originals_by_category[cat] + len(buckets[cat]) >= quotas[cat]:
                filtered["full_category"] += 1
                continue
            pk = prompt_key(prompt)
            if pk in seen_prompts:
                filtered["duplicate_prompt"] += 1
                continue
            if any(token in answer for token in ("<|cid_", " </think>", "[/Answer]")):
                filtered["hidden_transport"] += 1
                continue
            source_id = str(data.get("id", data.get("prompt_id", index)))
            record_id = (
                f"{key}-" + hashlib.sha256((source_id + "\0" + prompt).encode()).hexdigest()[:20]
            )
            row = LanguageSFTExample(record_id, prompt, answer, cat, repo, license_id)
            pair_hash = pair_key(row)
            if pair_hash in seen_pairs:
                filtered["duplicate_pair"] += 1
                continue
            seen_prompts.add(pk)
            seen_pairs.add(pair_hash)
            buckets[cat].append(row)
            collected_by_source[key] += 1
            if entries + sum(map(len, buckets.values())) >= args.target_total:
                break
        print("source_done", key, scanned[key], collected_by_source[key], flush=True)
        del dataset
        if entries + sum(map(len, buckets.values())) >= args.target_total:
            break

    new_total = sum(map(len, buckets.values()))
    actual_total = entries + new_total
    if actual_total < args.target_total:
        print("WARNING: dataset below target; first-pass category supply exhausted", flush=True)
    if actual_total < round(args.target_total * 0.9):
        raise RuntimeError(
            f"insufficient accepted samples: {actual_total} < 90% of requested {args.target_total}"
        )

    train = [row for row, split in originals.values() if split == "train"]
    validation = [row for row, split in originals.values() if split == "validation"]
    for category, new_rows in buckets.items():
        # Entire prompts are already globally unique; holdout is deterministic and
        # category-stratified, with old validation unchanged.
        new_rows.sort(
            key=lambda row: hashlib.sha256(f"{args.seed}:{row.example_id}".encode()).digest()
        )
        target_valid = round(
            (len(new_rows) + originals_by_category[category]) * args.validation_fraction
        )
        existing_valid = sum(row.category == category for row in validation)
        take = max(0, min(len(new_rows), target_valid - existing_valid))
        validation.extend(new_rows[:take])
        train.extend(new_rows[take:])
    rng.shuffle(train)
    rng.shuffle(validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, items in (("train", train), ("validation", validation)):
        path = args.output_dir / f"language-sft-{name}.jsonl"
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(
                    json.dumps(row.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temp.replace(path)
        print("validating", name, len(items), flush=True)
        reports[name] = validate_language_sft(path)

    validation_keys = {prompt_key(row.prompt) for row in validation}
    train_keys = {prompt_key(row.prompt) for row in train}
    if train_keys & validation_keys:
        raise AssertionError("train/validation prompt leakage")
    report = {
        "dataset_version": "1.2",
        "parent": "CID-Dataset v1.1",
        "source_revision": revisions,
        "licenses": {v[0]: v[2] for v in SOURCES.values()},
        "target_total": args.target_total,
        "actual_total": actual_total,
        "train": reports["train"],
        "validation": reports["validation"],
        "preexisting_unique": entries,
        "cross_split_prompt_duplicates": 48000 - entries,
        "dedup_key": "sha256(whitespace-normalized casefolded prompt)",
        "cross_split_prompt_leakage": 0,
        "category_ceiling": quotas,
        "new_categories": {cat: len(rows) for cat, rows in buckets.items()},
        "scanned": dict(scanned),
        "accepted_sources": dict(collected_by_source),
        "filters": dict(filtered),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "supervision": {"display": True, "thought": False, "tct": False, "tool": False},
        "recommended_language_token_fraction": 0.20,
        "duration_s": round(time.monotonic() - start_time, 1),
    }
    (args.output_dir / "language-sft-v1.2.manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "COMPLETE",
        json.dumps(
            {
                k: report[k]
                for k in (
                    "actual_total",
                    "preexisting_unique",
                    "cross_split_prompt_leakage",
                    "new_categories",
                    "accepted_sources",
                    "duration_s",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

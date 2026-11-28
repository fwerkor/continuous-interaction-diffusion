#!/usr/bin/env python3
"""Build and materialize CID-Dataset v1.3 without duplicating its v1.2 base.

Source-complete annotations preserve existing replies verbatim. They attest
completion format and provenance, not independent factual review. New
counterfactuals are deterministic tool-backed fixtures with verifiable values.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

UNKNOWN = "<|cid_unknown|>"
FORMAT = re.compile(
    r"\b(?:answer|return|output|respond|reply)\s+(?:with\s+)?(?:only|exactly|just)\b"
    r"|\b(?:give|return|report|provide)\s+(?:the\s+)?(?:final\s+)?(?:answer|value|result|balance)"
    r"\s+(?:with\s+)?exactly\s+\w+\s+decimal\s+places"
    r"|\b(?:give|use|round to)\s+exactly\s+\w+\s+decimal\s+places",
    re.I,
)
EXPLANATION = re.compile(
    r"\b(?:explain|justify|show (?:your|the) (?:work|reasoning)|summari[sz]e|supporting evidence)\b",  # noqa: E501
    re.I,
)
# These are upstream conversation and evidence-grounded complete response tasks.
COMPLETE_KINDS = {"natural_instruction", "natural_grounded_interaction"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation(row: dict) -> dict | None:
    meta = row.get("metadata", {})
    if meta.get("native_full_display") or meta.get("native_full_display_by_step"):
        return None
    answer = row["target_display"].strip()
    if not answer or UNKNOWN in answer or "[/Answer]" in answer:
        return None
    prompt = row["prompt"]
    kind = meta.get("task_kind")
    if kind in COMPLETE_KINDS:
        origin = "original_complete_response"
    elif len(answer.split()) <= 8 and FORMAT.search(prompt) and not EXPLANATION.search(prompt):
        origin = "explicit_output_contract"
    else:
        return None
    # Never promote an unresolved/provisional draft or a final answer before
    # the last recorded observation. Existing verified step refs take precedence.
    last_event = max((int(event["arrival_step"]) for event in row.get("events", ())), default=0)
    steps = sorted(
        {
            int(item["step"])
            for item in row.get("display_targets", ())
            if item["text"].strip() == answer and int(item["step"]) >= last_event
        }
    )
    if not steps:
        return None
    return {
        "example_id": row["example_id"],
        "kind": origin,
        "steps": steps,
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
    }


STYLES = (
    ("Read the task-local record {key}. Return only its recorded integer.", "{value}"),
    (
        "Look up {key} and report its measured value in milliseconds.",
        "The measured value is {value} ms.",
    ),
    (
        "What value does record {key} contain? Include a short source attribution.",
        "According to the retrieved record, the value is {value}.",
    ),
    (
        "Use the source to verify the count in {key}. Answer in one sentence.",
        "The record confirms a count of {value}.",
    ),
    (
        "Inspect record {key} and report the result using the format 'Result: number'.",
        "Result: {value}",
    ),
    (
        "What is the final reading stored in {key}? Give the reading and its unit.",
        "{value} ms is the recorded reading.",
    ),
    (
        "Read {key} and state the verified integer. Do not infer a value from the identifier.",
        "The verified integer is {value}.",
    ),
    (
        "Find the stored number in {key}. Report the retrieved number and mention that it came from the task-local record.",  # noqa: E501
        "The number is {value}, as recorded by the task-local source.",
    ),
)


def pair_example(style: int, number: int, variant: int, value: int, other: int, split: str) -> dict:
    prompt, response = STYLES[style]
    key = f"v13-{split}-{style}-{number:05d}"
    answer, alternative = response.format(value=value), response.format(value=other)
    thought = [
        {
            "step": 0,
            "slot": 0,
            "cell_id": "lookup",
            "semantic_text": f"Retrieve the authoritative value for record {key} before answering.",
            "roles": {"information_need": 1.0, "plan": 0.25},
            "uncertainty": 0.8,
            "noise": 0.06,
            "lifecycle": "waiting",
        }
    ]
    for step in (1, 2):
        thought.extend(
            (
                {
                    "step": step,
                    "slot": 0,
                    "cell_id": "lookup",
                    "semantic_text": f"Record {key} returned the verified integer {value}.",
                    "roles": {"percept": 1.0},
                    "uncertainty": 0.02,
                    "noise": 0.0,
                    "lifecycle": "stable",
                },
                {
                    "step": step,
                    "slot": 1,
                    "cell_id": "answer",
                    "semantic_text": f"The available record establishes the result {value}. Use the requested output format.",  # noqa: E501
                    "roles": {"conclusion": 1.0},
                    "uncertainty": 0.01,
                    "noise": 0.0,
                    "lifecycle": "stable",
                },
            )
        )
    return {
        "example_id": f"{key}-variant-{variant}",
        "prompt": prompt.format(key=key),
        "target_display": answer,
        "source_descriptors": [
            {
                "name": "record_lookup",
                "description": "Read an immutable task-local record by key.",
                "arguments": [{"name": "key", "kind": "string", "required": True}],
                "cacheable": True,
                "dynamic": False,
                "versioned": False,
                "promote_results_to_fact": False,
            }
        ],
        "events": [
            {
                "source": "record_lookup",
                "value": {"value": value, "unit": "ms" if "ms" in answer else "integer"},
                "arrival_step": 1,
                "arguments": {"key": key},
                "provenance": "cid.dataset_v13.deterministic_record",
            }
        ],
        "binding_targets": [
            {
                "need_id": "need:record",
                "source": "record_lookup",
                "first_need_step": 0,
                "executable_step": 0,
                "arguments": {"key": key},
                "argument_steps": {"key": 0},
                "confidence": 1.0,
                "freshness": "once",
                "target_cells": [{"kind": "cell", "identifier": "lookup"}],
                "target_display": [],
                "owner_cell_id": "lookup",
            }
        ],
        "thought_targets": thought,
        "display_targets": [
            {"step": 0, "text": UNKNOWN},
            *({"step": step, "text": answer} for step in (1, 2)),
        ],
        "metadata": {
            "task_kind": "evidence_counterfactual",
            "training_mode": "tool_required",
            "semantic_task_id": key,
            "training_weight": 4.0,
            "split": split,
            "native_full_display_by_step": {"1": answer, "2": answer},
            "native_counterfactual_by_step": {"1": alternative, "2": alternative},
            "native_full_display_verified": True,
            "native_full_display_reviewer": "deterministic_record_fixture",
            "event_launch_steps": {"record": 0},
            "event_arrival_steps": {"record": 1},
            "neural_contract_version": 4,
            "display_contract": "continuous-answer-draft-v2",
        },
    }


def build_pairs(path: Path, count: int, split: str) -> None:
    rng = random.Random(27313 if split == "train" else 982143)
    with path.open("w", encoding="utf-8") as out:
        for number in range(count):
            # Comparable digit widths make same-position contrast feasible;
            # the actual tokenizer still verifies equal reference lengths.
            low, high = (100, 999) if number % 2 else (1000, 9999)
            a, b = rng.sample(range(low, high + 1), 2)
            for variant, (value, other) in enumerate(((a, b), (b, a))):
                out.write(
                    json.dumps(
                        pair_example(number % len(STYLES), number, variant, value, other, split),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def build(base: Path, root: Path, pair_count: int, validation_count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    annotations = root / "complete-references.jsonl.gz"
    counts = Counter()
    digest = hashlib.sha256()
    with (
        base.open("rb") as source,
        annotations.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as out,
    ):
        for line in source:
            digest.update(line)
            row = json.loads(line)
            counts["base_examples"] += 1
            item = annotation(row)
            if item is not None:
                counts[item["kind"]] += 1
                counts["annotated_states"] += len(item["steps"])
                out.write((json.dumps(item, separators=(",", ":")) + "\n").encode())
    train = root / "counterfactual-train.jsonl"
    val = root / "counterfactual-validation.jsonl"
    build_pairs(train, pair_count, "train")
    build_pairs(val, validation_count, "validation")
    manifest = {
        "version": "1.3",
        "base": {
            "path": "release/training-trajectories.jsonl",
            "sha256": digest.hexdigest(),
            "examples": counts["base_examples"],
        },
        "counts": dict(counts),
        "counterfactual_train": pair_count * 2,
        "counterfactual_validation": validation_count * 2,
        "language_sft": "unchanged from v1.2",
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (annotations, train, val)
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def materialize(base: Path, root: Path, output: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        if sha256(root / name) != entry["sha256"]:
            raise ValueError(f"v1.3 file checksum mismatch: {name}")
    if base.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the v1.2 base")
    annotations = {}
    with gzip.open(root / "complete-references.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = item["example_id"]
            if key in annotations:
                raise ValueError(f"duplicate annotation {key}")
            annotations[key] = item
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with base.open("rb") as src, temporary.open("wb") as dst:
            for line in src:
                digest.update(line)
                row = json.loads(line)
                item = annotations.pop(row["example_id"], None)
                if item is None:
                    dst.write(line)
                else:
                    if annotation(row) != item:
                        raise ValueError(f"annotation no longer matches {row['example_id']}")
                    meta = row.setdefault("metadata", {})
                    answer = row["target_display"].strip()
                    meta["native_full_display_contract"] = "source-complete-v1"
                    meta["native_full_display_origin"] = item["kind"]
                    meta["native_full_display_by_step"] = {
                        str(step): answer for step in item["steps"]
                    }
                    dst.write(
                        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    )
                count += 1
            if (
                annotations
                or count != manifest["base"]["examples"]
                or digest.hexdigest() != manifest["base"]["sha256"]
            ):
                raise ValueError("v1.3 base or annotations do not match the manifest")
            with (root / "counterfactual-train.jsonl").open("rb") as extra:
                for line in extra:
                    dst.write(line)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"materialized {count + manifest['counterfactual_train']} training trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "materialize"))
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--validation-pairs", type=int, default=128)
    args = parser.parse_args()
    if args.action == "build":
        if args.pairs < 1 or args.validation_pairs < 1:
            parser.error("pair counts must be positive")
        build(args.base, args.release_dir, args.pairs, args.validation_pairs)
    else:
        if args.output is None:
            parser.error("materialize requires --output")
        materialize(args.base, args.release_dir, args.output)


if __name__ == "__main__":
    main()

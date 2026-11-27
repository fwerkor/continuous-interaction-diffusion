#!/usr/bin/env python3
"""Build a small deterministic Thought->Display dependency curriculum.

Two trajectories share the exact Prompt but have different realized Thought
content, so matching the Display reference requires reading that content.
The initial Display remains unresolved until Thought becomes available.
No new model component, synthetic tool response or artificial TCT label is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

UNKNOWN = "<|cid_unknown|>"
STYLES = (
    ("record", "For experiment {key}, report the verified numerical result held in your Thought state.", "The verified result is {value}."),
    ("measurement", "For run {key}, what number did the internal analysis conclude? Use the Thought state.", "The measured value is {value}."),
    ("calculation", "For calculation {key}, state the final computed number from the Thought state.", "The calculation result is {value}."),
    ("checksum", "For check {key}, report the confirmed integer recorded by the internal reasoning process.", "The confirmed integer is {value}."),
)

def example(style: int, task: int, variant: int, value: int, *, split: str) -> dict:
    _, prompt, response = STYLES[style % len(STYLES)]
    key = f"{split[:1].upper()}{style:02d}-{task:05d}"
    answer = response.format(value=value)
    phrase = f"{value}"
    plan = "Locate the finalized result in the Thought state and report it precisely, without guessing from the record identifier."
    conclusion = f"The completed internal calculation for {key} returned {phrase}. This value is authoritative for the Display."
    thought = []
    for step in (0, 1):
        thought.extend((
            {
                "step": step,
                "slot": 0,
                "cell_id": "plan",
                "semantic_text": plan,
                "roles": {"plan": 1.0},
                "uncertainty": 0.05,
                "noise": 0.0,
                "lifecycle": "stable",
            },
            {
                "step": step,
                "slot": 1,
                "cell_id": "conclusion",
                "semantic_text": conclusion,
                "roles": {"conclusion": 1.0},
                "uncertainty": 0.01,
                "noise": 0.0,
                "lifecycle": "stable",
            },
        ))
    return {
        "example_id": f"thought-causal-{split}-{style}-{task:05d}-v{variant}",
        "prompt": prompt.format(key=key),
        "target_display": answer,
        "thought_targets": thought,
        "display_targets": [
            {"step": 0, "text": UNKNOWN},
            {"step": 1, "text": answer},
        ],
        "metadata": {
            "task_kind": "thought_dependency_contrast",
            "semantic_task_id": f"thought-causal-{split}-{style}-{task:05d}",
            "training_mode": "no_tool",
            "training_weight": 4.0,
            "split": split,
            "native_full_display_by_step": {"1": answer},
            "native_full_display_verified": True,
            "native_full_display_reviewer": "deterministic_computation_fixture",
        },
    }

def build(output: Path, *, count: int, seed: int, split: str) -> dict:
    rng = random.Random(seed)
    digest = hashlib.sha256()
    with output.open("wb") as out:
        for i in range(count):
            style = i % len(STYLES)
            value_a = rng.randrange(40, 1800)
            value_b = rng.randrange(40, 1800)
            if value_a == value_b:
                value_b += 1
            for variant, value in enumerate((value_a, value_b)):
                row = example(style, i, variant, value, split=split)
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                out.write(encoded)
                digest.update(encoded)
    return {"pairs": count, "examples": count * 2, "sha256": digest.hexdigest(), "bytes": output.stat().st_size, "split": split}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--holdout", type=Path)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--holdout-count", type=int, default=128)
    args = ap.parse_args()
    if args.count < 1 or args.holdout_count < 1:
        raise SystemExit("counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = args.output.with_suffix(".pairs.jsonl")
    info = build(pairs, count=args.count, seed=2731, split="train")
    if args.source:
        if args.source.resolve() == args.output.resolve():
            raise SystemExit("refusing to overwrite source")
        with args.source.open("rb") as src, args.output.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            with pairs.open("rb") as extra:
                shutil.copyfileobj(extra, dst, length=8 * 1024 * 1024)
    if args.holdout:
        info["holdout"] = build(args.holdout, count=args.holdout_count, seed=98214, split="validation")
    manifest = args.output.with_suffix(".causal-manifest.json")
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, ensure_ascii=False))

if __name__ == "__main__":
    main()

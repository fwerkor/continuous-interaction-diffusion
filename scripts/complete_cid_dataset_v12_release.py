#!/usr/bin/env python3
"""Finalize expanded CID v1.2 metadata without copying the 9 GB v1.1 base."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cid.language_sft import validate_language_sft


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--expanded-dir", type=Path, required=True)
    args = parser.parse_args()

    source_release = json.loads((args.parent_dir / "RELEASE-MANIFEST-v1.2.json").read_text())
    base = source_release["base"]
    component = args.expanded_dir / "components/language-sft"
    expanded_report_path = component / "language-sft-v1.2.manifest.json"
    expanded = json.loads(expanded_report_path.read_text())
    expected_total = int(expanded["actual_total"])
    if expected_total != int(expanded["train"]["examples"]) + int(
        expanded["validation"]["examples"]
    ):
        raise ValueError("train/validation counts do not sum to the expanded total")
    if expanded["cross_split_prompt_leakage"]:
        raise ValueError("validation prompt leakage in expanded language_sft")
    details = {}
    for split in ("train", "validation"):
        relative = f"components/language-sft/language-sft-{split}.jsonl"
        info = validate_language_sft(args.expanded_dir / relative)
        if info["sha256"] != expanded[split]["sha256"]:
            raise ValueError(f"{split} SHA256 changed after validation")
        details[f"{split}_sha256"] = info["sha256"]
        details[f"{split}_examples"] = info["examples"]
        print(f"{split}={info['examples']} sha256={info['sha256']}", flush=True)

    release = {
        "dataset": "fwerkor/CID-Dataset",
        "version": "1.2",
        "parent": "1.1",
        "base": base,
        "additions": {
            "language_sft": {
                "train": "components/language-sft/language-sft-train.jsonl",
                "validation": "components/language-sft/language-sft-validation.jsonl",
                "examples": expected_total,
                "manifest": "components/language-sft/language-sft-v1.2.manifest.json",
                **details,
            }
        },
        "supervision": {
            "cid": "original TCT, Display and tool trajectories",
            "language_sft": "Display-only; no Thought, TCT or tool targets",
        },
        "mix_target": {
            "cid_token_fraction": 0.80,
            "language_sft_token_fraction": 0.20,
            "token_budget": "exact tokenizer reference Prompt and Display tokens",
            "nominal_language_loss_weight_mass": 0.20,
        },
        "mix_implementation": "cid-v1.2-language-token-mix-v2",
        "source_revision": expanded["source_revision"],
        "deduplication": {
            "key": expanded["dedup_key"],
            "train_validation_prompt_overlap": 0,
        },
    }
    release_path = args.expanded_dir / "RELEASE-MANIFEST-v1.2.json"
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n")

    categories = expanded["train"]["categories"]
    category_summary = ", ".join(f"{key}: {count:,}" for key, count in categories.items())
    readme = f"""# CID-Dataset v1.2

This release preserves the original **422,230 CID v1.1 trajectories** byte-for-byte.
The base is available at `release/training-trajectories.jsonl` in the existing
Hugging Face dataset. Its SHA256 remains `{base["sha256"]}`.

## Language SFT component

Added **{expected_total:,} distinct-Prompt instruction/response records**:
{details["train_examples"]:,} training, {details["validation_examples"]:,} held-out validation.
The records have eight category labels, and each sample supervises **Display only**.
There is no Thought, TCT or tool-call target supervision.

Training category distribution: {category_summary}.

Training and validation prompts are disjoint after normalization. The expanded
component retains non-colliding earlier v1.2 examples. New samples use the
pinned Tülu 3 SFT mixture (dataset-level ODC-BY; inspect upstream licenses)
and UltraChat 200k (MIT), with an Aya Dataset (Apache-2.0)
multilingual supplement; retained v1.2 examples also include OASST1
(Apache-2.0) and GSM8K (MIT). Detailed source revisions, SHA256 values,
counts and filtering statistics are recorded in
`components/language-sft/language-sft-v1.2.manifest.json`.

## Training

The CID Stage A and B trainers support a deterministic per-epoch 80/20
mixture of original CID and Language SFT *reference tokenizer tokens*.
Use `--language-sft-data` and
`--language-sft-token-fraction 0.20` with `cid train` or
`cid train-full`. The language component's loss supervises Display only,
and original CID samples retain all their cognitive/tool losses.
Training logs report both reference-token fractions and the nominal
loss-weight share. See the CID code repository for the exact algorithm.

The dataset itself does not imply any particular trained model quality.
"""
    (args.expanded_dir / "README.md").write_text(readme, encoding="utf-8")
    print("FINALIZED", expected_total, "base_sha256", base["sha256"], flush=True)


if __name__ == "__main__":
    main()

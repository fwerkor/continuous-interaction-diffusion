from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import TrajectoryExample, load_jsonl


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    format_version: int
    schema: str
    sha256: str
    bytes: int
    examples: int
    transitions: int
    tag_counts: dict[str, int]
    sources: tuple[str, ...]
    thought_capacity_required: int
    max_trajectory_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "schema": self.schema,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "examples": self.examples,
            "transitions": self.transitions,
            "tag_counts": dict(self.tag_counts),
            "sources": list(self.sources),
            "thought_capacity_required": self.thought_capacity_required,
            "max_trajectory_steps": self.max_trajectory_steps,
        }


def inspect_dataset(path: str | Path) -> DatasetManifest:
    source = Path(path)
    payload = source.read_bytes()
    examples = load_jsonl(source)
    tags = Counter(_dataset_tag(example) for example in examples)
    sources = sorted(
        {
            str(descriptor.get("name", ""))
            for example in examples
            for descriptor in example.source_descriptors
            if str(descriptor.get("name", ""))
        }
    )
    return DatasetManifest(
        format_version=1,
        schema="cid.TrajectoryExample.v1",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        examples=len(examples),
        transitions=sum(_transition_count(example) for example in examples),
        tag_counts=dict(sorted(tags.items())),
        sources=tuple(sources),
        thought_capacity_required=max(
            (target.slot + 1 for example in examples for target in example.thought_targets),
            default=0,
        ),
        max_trajectory_steps=max(
            (len({target.step for target in example.thought_targets}) for example in examples),
            default=0,
        ),
    )


def dump_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _transition_count(example: TrajectoryExample) -> int:
    steps = {target.step for target in example.thought_targets}
    return sum(1 for step in steps if step + 1 in steps)


def _dataset_tag(example: TrajectoryExample) -> str:
    family = example.metadata.get("family")
    if family:
        return f"family:{family}"
    distillation = example.metadata.get("distillation")
    if distillation:
        return f"distillation:{distillation}"
    return "unlabeled"

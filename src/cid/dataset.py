from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import (
    TrajectoryExample,
    adjacent_transition_source_steps,
    training_transition_source_steps,
)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    format_version: int
    schema: str
    sha256: str
    bytes: int
    examples: int
    transitions: int
    bootstrap_transitions: int
    training_transitions: int
    trainable_examples: int
    zero_training_transition_examples: int
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
            "bootstrap_transitions": self.bootstrap_transitions,
            "training_transitions": self.training_transitions,
            "trainable_examples": self.trainable_examples,
            "zero_training_transition_examples": self.zero_training_transition_examples,
            "tag_counts": dict(self.tag_counts),
            "sources": list(self.sources),
            "thought_capacity_required": self.thought_capacity_required,
            "max_trajectory_steps": self.max_trajectory_steps,
        }


def inspect_dataset(path: str | Path) -> DatasetManifest:
    source = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    example_count = 0
    transition_count = 0
    bootstrap_transition_count = 0
    training_transition_count = 0
    trainable_example_count = 0
    zero_training_transition_examples = 0
    tags: Counter[str] = Counter()
    sources: set[str] = set()
    thought_capacity_required = 0
    max_trajectory_steps = 0

    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                example = TrajectoryExample.from_dict(raw)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid CID trajectory at line {line_number}: {exc}") from exc
            example_count += 1
            tags[_dataset_tag(example)] += 1
            sources.update(
                str(descriptor.get("name", ""))
                for descriptor in example.source_descriptors
                if str(descriptor.get("name", ""))
            )
            steps = {target.step for target in example.thought_targets}
            adjacent_sources = adjacent_transition_source_steps(steps)
            training_sources = training_transition_source_steps(steps)
            transition_count += len(adjacent_sources)
            bootstrap_transition_count += int(-1 in training_sources)
            training_transition_count += len(training_sources)
            if training_sources:
                trainable_example_count += 1
            else:
                zero_training_transition_examples += 1
            thought_capacity_required = max(
                thought_capacity_required,
                max((target.slot + 1 for target in example.thought_targets), default=0),
            )
            max_trajectory_steps = max(
                max_trajectory_steps,
                len({target.step for target in example.thought_targets}),
            )

    return DatasetManifest(
        format_version=1,
        schema="cid.TrajectoryExample.v1",
        sha256=digest.hexdigest(),
        bytes=byte_count,
        examples=example_count,
        transitions=transition_count,
        bootstrap_transitions=bootstrap_transition_count,
        training_transitions=training_transition_count,
        trainable_examples=trainable_example_count,
        zero_training_transition_examples=zero_training_transition_examples,
        tag_counts=dict(sorted(tags.items())),
        sources=tuple(sorted(sources)),
        thought_capacity_required=thought_capacity_required,
        max_trajectory_steps=max_trajectory_steps,
    )


def dump_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _dataset_tag(example: TrajectoryExample) -> str:
    family = example.metadata.get("family")
    if family:
        return f"family:{family}"
    distillation = example.metadata.get("distillation")
    if distillation:
        return f"distillation:{distillation}"
    return "unlabeled"

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
    bindings: int
    explicit_owner_bindings: int
    owner_bindings_without_target_cells: int
    multi_cell_bindings: int
    max_bindings_per_owner: int
    max_source_arguments: int
    bindings_with_undeclared_arguments: int
    explicit_display_route_bindings: int
    global_display_fallback_bindings: int

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
            "bindings": self.bindings,
            "explicit_owner_bindings": self.explicit_owner_bindings,
            "owner_bindings_without_target_cells": self.owner_bindings_without_target_cells,
            "multi_cell_bindings": self.multi_cell_bindings,
            "max_bindings_per_owner": self.max_bindings_per_owner,
            "max_source_arguments": self.max_source_arguments,
            "bindings_with_undeclared_arguments": self.bindings_with_undeclared_arguments,
            "explicit_display_route_bindings": self.explicit_display_route_bindings,
            "global_display_fallback_bindings": self.global_display_fallback_bindings,
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
    binding_count = 0
    explicit_owner_bindings = 0
    owner_bindings_without_target_cells = 0
    multi_cell_bindings = 0
    max_bindings_per_owner = 0
    max_source_arguments = 0
    bindings_with_undeclared_arguments = 0
    explicit_display_route_bindings = 0
    global_display_fallback_bindings = 0

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
            descriptors = {
                str(descriptor.get("name", "")): descriptor
                for descriptor in example.source_descriptors
                if str(descriptor.get("name", ""))
            }
            max_source_arguments = max(
                max_source_arguments,
                max(
                    (
                        len(tuple(descriptor.get("arguments", ())))
                        for descriptor in descriptors.values()
                    ),
                    default=0,
                ),
            )
            bindings_by_owner: Counter[str] = Counter()
            for binding in example.binding_targets:
                binding_count += 1
                explicit_owner_bindings += int(binding.owner_cell_id is not None)
                if binding.owner_cell_id is not None:
                    bindings_by_owner[binding.owner_cell_id] += 1
                    owner_bindings_without_target_cells += int(not binding.target_cells)
                multi_cell_bindings += int(len(binding.target_cells) > 1)
                descriptor = descriptors.get(binding.source)
                declared_arguments = (
                    set()
                    if descriptor is None
                    else {
                        str(argument.get("name", ""))
                        for argument in descriptor.get("arguments", ())
                    }
                )
                bindings_with_undeclared_arguments += int(
                    bool(set(binding.arguments) - declared_arguments)
                )
                if binding.target_display:
                    explicit_display_route_bindings += 1
                else:
                    # Empty target_display is the intentional global-display fallback.
                    global_display_fallback_bindings += 1
            max_bindings_per_owner = max(
                max_bindings_per_owner,
                max(bindings_by_owner.values(), default=0),
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
        bindings=binding_count,
        explicit_owner_bindings=explicit_owner_bindings,
        owner_bindings_without_target_cells=owner_bindings_without_target_cells,
        multi_cell_bindings=multi_cell_bindings,
        max_bindings_per_owner=max_bindings_per_owner,
        max_source_arguments=max_source_arguments,
        bindings_with_undeclared_arguments=bindings_with_undeclared_arguments,
        explicit_display_route_bindings=explicit_display_route_bindings,
        global_display_fallback_bindings=global_display_fallback_bindings,
    )


def dump_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_neural_training_contract(
    manifest: DatasetManifest,
    *,
    max_argument_slots: int = 4,
    max_need_slots: int = 4,
) -> None:
    """Reject datasets that cannot supervise the neural-contract-v3 runtime faithfully."""

    if max_argument_slots <= 0 or max_need_slots <= 0:
        raise ValueError("neural training slot capacities must be positive")

    if manifest.bindings and manifest.explicit_owner_bindings != manifest.bindings:
        raise ValueError(
            "training data contains information needs without explicit owner_cell_id; "
            "migrate the materialized trajectories with `cid migrate-dataset-contract-v3` "
            "before neural training"
        )
    if manifest.owner_bindings_without_target_cells:
        raise ValueError(
            "training data contains explicit binding owners without target_cells; "
            "neural runtime routing requires the owner to be included in target_cells"
        )
    if manifest.bindings and manifest.multi_cell_bindings == 0:
        raise ValueError(
            "training data has no multi-cell information-need routing supervision despite "
            f"containing {manifest.bindings} bindings; use the neural-contract-v3 materialization"
        )
    if manifest.max_source_arguments > max_argument_slots:
        raise ValueError(
            "training data source argument capacity exceeds the neural adapter: "
            f"{manifest.max_source_arguments} > {max_argument_slots}"
        )
    if manifest.bindings_with_undeclared_arguments:
        raise ValueError(
            "training data contains binding arguments that are not declared by their source "
            "descriptor"
        )
    if manifest.max_bindings_per_owner > max_need_slots:
        raise ValueError(
            "training data information-need capacity exceeds the neural adapter: "
            f"{manifest.max_bindings_per_owner} > {max_need_slots}"
        )


def _dataset_tag(example: TrajectoryExample) -> str:
    family = example.metadata.get("family")
    if family:
        return f"family:{family}"
    distillation = example.metadata.get("distillation")
    if distillation:
        return f"distillation:{distillation}"
    return "unlabeled"

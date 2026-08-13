from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cid.contracts import FreshnessDemand
from cid.data import TrajectoryExample, training_transition_source_steps


@dataclass(frozen=True, slots=True)
class TrainingDataAudit:
    examples: int
    semantic_tasks: int
    trainable_semantic_tasks: int
    zero_training_transition_semantic_tasks: int
    adjacent_transitions: int
    bootstrap_transitions: int
    training_transitions: int
    mode_semantic_counts: dict[str, int]
    untrainable_first_need_bindings: int
    pre_satisfied_once_bindings: int
    semantic_mode_conflicts: int

    @property
    def ok(self) -> bool:
        return (
            self.zero_training_transition_semantic_tasks == 0
            and self.untrainable_first_need_bindings == 0
            and self.pre_satisfied_once_bindings == 0
            and self.semantic_mode_conflicts == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def audit_training_trajectories(path: str | Path) -> TrainingDataAudit:
    source = Path(path)
    examples = 0
    adjacent_transitions = 0
    bootstrap_transitions = 0
    training_transitions = 0
    untrainable_first_need_bindings = 0
    pre_satisfied_once_bindings = 0
    semantic_modes: dict[str, set[str]] = {}
    semantic_has_training: dict[str, bool] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = TrajectoryExample.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid CID trajectory at line {line_number}: {exc}") from exc
            examples += 1
            steps = {target.step for target in example.thought_targets}
            training_sources = training_transition_source_steps(steps)
            adjacent = tuple(step for step in training_sources if step >= 0)
            adjacent_transitions += len(adjacent)
            bootstrap_transitions += int(-1 in training_sources)
            training_transitions += len(training_sources)

            semantic_id = str(
                example.metadata.get("semantic_task_id") or example.example_id
            )
            mode = _training_mode(example)
            semantic_modes.setdefault(semantic_id, set()).add(mode)
            semantic_has_training[semantic_id] = (
                semantic_has_training.get(semantic_id, False) or bool(training_sources)
            )

            target_steps = {source_step + 1 for source_step in training_sources}
            cells_by_step: dict[int, set[str]] = {}
            for target in example.thought_targets:
                cells_by_step.setdefault(target.step, set()).add(target.cell_id)
            for binding in example.binding_targets:
                target_cells = {
                    ref.identifier
                    for ref in binding.target_cells
                    if ref.identifier is not None
                }
                if (
                    binding.first_need_step not in target_steps
                    or not target_cells.intersection(
                        cells_by_step.get(binding.first_need_step, set())
                    )
                ):
                    untrainable_first_need_bindings += 1
                if (
                    binding.freshness is FreshnessDemand.ONCE
                    and _matching_observation_arrives_by(
                        example, binding.source, binding.arguments, binding.first_need_step
                    )
                ):
                    pre_satisfied_once_bindings += 1

    mode_counts = Counter()
    for modes in semantic_modes.values():
        for mode in modes:
            mode_counts[mode] += 1
    trainable_semantic_tasks = sum(semantic_has_training.values())
    return TrainingDataAudit(
        examples=examples,
        semantic_tasks=len(semantic_modes),
        trainable_semantic_tasks=trainable_semantic_tasks,
        zero_training_transition_semantic_tasks=(
            len(semantic_modes) - trainable_semantic_tasks
        ),
        adjacent_transitions=adjacent_transitions,
        bootstrap_transitions=bootstrap_transitions,
        training_transitions=training_transitions,
        mode_semantic_counts=dict(sorted(mode_counts.items())),
        untrainable_first_need_bindings=untrainable_first_need_bindings,
        pre_satisfied_once_bindings=pre_satisfied_once_bindings,
        semantic_mode_conflicts=sum(len(modes) > 1 for modes in semantic_modes.values()),
    )


def _training_mode(example: TrajectoryExample) -> str:
    raw = str(example.metadata.get("training_mode", "")).strip()
    if raw in {"no_tool", "no_tool_required"}:
        return "no_tool"
    if raw in {"tool_required", "tools_available_unnecessary"}:
        return raw
    if example.binding_targets:
        return "tool_required"
    if example.source_descriptors:
        return "tools_available_unnecessary"
    return "no_tool"


def _matching_observation_arrives_by(
    example: TrajectoryExample,
    source: str,
    arguments: Any,
    step: int,
) -> bool:
    expected = dict(arguments)
    return any(
        event.arrival_step <= step
        and event.source == source
        and dict(event.arguments) == expected
        for event in example.events
    )

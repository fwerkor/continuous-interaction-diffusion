from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.distill import TeacherEvidence, TeacherTask


@dataclass(frozen=True, slots=True)
class CausalTeacherStage:
    phase: str
    arrived_evidence: TeacherEvidence | None
    available_evidence: tuple[Mapping[str, Any], ...]
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "arrived_evidence": (
                None if self.arrived_evidence is None else self.arrived_evidence.to_dict()
            ),
            "available_evidence": [dict(item) for item in self.available_evidence],
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class CausalTeacherJob:
    task_id: str
    task: Mapping[str, Any]
    stages: tuple[CausalTeacherStage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": dict(self.task),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def build_causal_teacher_job(task: TeacherTask) -> CausalTeacherJob:
    """Build a staged teacher job that never exposes future evidence values.

    The job is an orchestration specification, not a prompt to send wholesale to a model. A caller
    processes stages in order and carries the accepted semantic state from one stage into the next.
    Only `arrived_evidence` from the current stage may be shown at that call. `available_evidence`
    contains value-free invocation contracts that may be requested from the current semantic state.
    """

    base = task.teacher_visible_dict()
    base.pop("evidence", None)
    evidence_by_id = {item.evidence_id: item for item in task.evidence}
    persistent_roots = _persistent_binding_roots(task.evidence)
    unlocked: set[str] = set()
    arrived: set[str] = set()

    def newly_available() -> tuple[Mapping[str, Any], ...]:
        contracts: list[Mapping[str, Any]] = []
        for item in task.evidence:
            if item.evidence_id in unlocked or item.evidence_id in arrived:
                continue
            if all(dependency in arrived for dependency in item.depends_on):
                unlocked.add(item.evidence_id)
                if item.requires_need:
                    contracts.append(
                        _evidence_contract(
                            item,
                            persistent=item.evidence_id in persistent_roots,
                        )
                    )
        return tuple(contracts)

    stages: list[CausalTeacherStage] = [
        CausalTeacherStage(
            phase="initial",
            arrived_evidence=None,
            available_evidence=newly_available(),
            terminal=not task.evidence,
        )
    ]
    for index, evidence in enumerate(task.evidence):
        if evidence.evidence_id not in unlocked:
            missing = [
                dependency for dependency in evidence.depends_on if dependency not in arrived
            ]
            raise ValueError(
                f"evidence {evidence.evidence_id!r} is scheduled before dependencies: {missing}"
            )
        unlocked.remove(evidence.evidence_id)
        arrived.add(evidence.evidence_id)
        stages.append(
            CausalTeacherStage(
                phase=f"after:{evidence.evidence_id}",
                arrived_evidence=evidence,
                available_evidence=newly_available(),
                terminal=index == len(task.evidence) - 1,
            )
        )

    if set(evidence_by_id) != arrived:
        raise ValueError("causal teacher job did not schedule every evidence item")
    return CausalTeacherJob(task_id=task.task_id, task=base, stages=tuple(stages))


def dump_causal_teacher_jobs(tasks: tuple[TeacherTask, ...], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    build_causal_teacher_job(task).to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _evidence_contract(
    evidence: TeacherEvidence,
    *,
    persistent: bool = False,
) -> dict[str, Any]:
    contract = {
        "evidence_id": evidence.evidence_id,
        "source": evidence.source,
        "arguments": dict(evidence.arguments),
        "depends_on": list(evidence.depends_on),
        "requires_need": evidence.requires_need,
    }
    if persistent:
        contract["freshness_hint"] = "always"
    return contract


def _persistent_binding_roots(
    evidence: tuple[TeacherEvidence, ...],
) -> set[str]:
    by_id = {item.evidence_id: item for item in evidence}
    roots: set[str] = set()
    for item in evidence:
        if item.requires_need:
            continue
        current = item
        visited: set[str] = set()
        while not current.requires_need and len(current.depends_on) == 1:
            if current.evidence_id in visited:
                break
            visited.add(current.evidence_id)
            parent = by_id[current.depends_on[0]]
            if parent.source != item.source or dict(parent.arguments) != dict(item.arguments):
                break
            current = parent
        if current.requires_need:
            roots.add(current.evidence_id)
    return roots

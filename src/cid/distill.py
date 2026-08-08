from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from cid.contracts import FreshnessDemand
from cid.data import (
    BindingTarget,
    DisplayTarget,
    ExternalEvent,
    GroundingTarget,
    ThoughtTarget,
    TrajectoryExample,
)
from cid.grounding import Anchor, CognitiveLink, GroundingEntry, ObjectRef
from cid.state import CellLifecycle, CognitiveRole


@dataclass(frozen=True, slots=True)
class TeacherEvidence:
    evidence_id: str
    source: str
    value: Any
    arguments: Mapping[str, Any] = field(default_factory=dict)
    version: str | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source:
            raise ValueError("teacher evidence requires non-empty evidence_id and source")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherEvidence:
        return cls(
            evidence_id=str(raw["evidence_id"]),
            source=str(raw["source"]),
            value=raw.get("value"),
            arguments=dict(raw.get("arguments", {})),
            version=None if raw.get("version") is None else str(raw["version"]),
            provenance=(
                None if raw.get("provenance") is None else str(raw["provenance"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "value": self.value,
            "arguments": dict(self.arguments),
            "version": self.version,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class TeacherTask:
    task_id: str
    prompt: str
    protected_facts: Mapping[str, Any] = field(default_factory=dict)
    source_descriptors: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[TeacherEvidence, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("teacher task requires non-empty task_id and prompt")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("teacher task evidence IDs must be unique")
        source_names = {str(item.get("name", "")) for item in self.source_descriptors}
        unknown = {item.source for item in self.evidence} - source_names
        if unknown:
            raise ValueError(f"teacher evidence references unknown sources: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherTask:
        return cls(
            task_id=str(raw["task_id"]),
            prompt=str(raw["prompt"]),
            protected_facts=dict(raw.get("protected_facts", {})),
            source_descriptors=tuple(
                _source_descriptor(item) for item in raw.get("source_descriptors", ())
            ),
            evidence=tuple(
                TeacherEvidence.from_dict(item) for item in raw.get("evidence", ())
            ),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "protected_facts": dict(self.protected_facts),
            "source_descriptors": [dict(item) for item in self.source_descriptors],
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TeacherCellPlan:
    cell_id: str
    semantic_text: str
    roles: Mapping[CognitiveRole, float]
    uncertainty: float = 0.5
    noise: float = 0.5
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE
    anchors: tuple[Anchor, ...] = ()
    links: tuple[CognitiveLink, ...] = ()

    def __post_init__(self) -> None:
        if not self.cell_id or not self.semantic_text:
            raise ValueError("teacher cell requires non-empty cell_id and semantic_text")
        if len(self.semantic_text) > 512:
            raise ValueError("teacher cell semantic_text must remain a concise state summary")
        if not self.roles:
            raise ValueError("teacher cell requires at least one cognitive role")
        if self.lifecycle is CellLifecycle.EMPTY:
            raise ValueError("teacher cells cannot use EMPTY lifecycle")
        if any(not 0.0 <= value <= 1.0 for value in self.roles.values()):
            raise ValueError("teacher cell role weights must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0 or not 0.0 <= self.noise <= 1.0:
            raise ValueError("teacher cell uncertainty/noise must be in [0, 1]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherCellPlan:
        _reject_unknown_keys(
            raw,
            {
                "cell_id",
                "semantic_text",
                "roles",
                "uncertainty",
                "noise",
                "lifecycle",
                "anchors",
                "links",
            },
            "teacher cell",
        )
        return cls(
            cell_id=str(raw["cell_id"]),
            semantic_text=str(raw["semantic_text"]),
            roles={
                CognitiveRole(str(role)): float(weight)
                for role, weight in raw.get("roles", {}).items()
            },
            uncertainty=float(raw.get("uncertainty", 0.5)),
            noise=float(raw.get("noise", 0.5)),
            lifecycle=CellLifecycle(str(raw.get("lifecycle", CellLifecycle.ACTIVE))),
            anchors=tuple(Anchor.from_dict(item) for item in raw.get("anchors", ())),
            links=tuple(CognitiveLink.from_dict(item) for item in raw.get("links", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "semantic_text": self.semantic_text,
            "roles": {role.value: weight for role, weight in self.roles.items()},
            "uncertainty": self.uncertainty,
            "noise": self.noise,
            "lifecycle": self.lifecycle.value,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "links": [link.to_dict() for link in self.links],
        }


@dataclass(frozen=True, slots=True)
class TeacherFrame:
    phase: str
    display: str
    cells: tuple[TeacherCellPlan, ...]

    def __post_init__(self) -> None:
        if not self.phase or not self.display:
            raise ValueError("teacher frame requires non-empty phase and display")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("teacher frame cell IDs must be unique")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherFrame:
        _reject_unknown_keys(raw, {"phase", "display", "cells"}, "teacher frame")
        return cls(
            phase=str(raw["phase"]),
            display=str(raw["display"]),
            cells=tuple(TeacherCellPlan.from_dict(item) for item in raw.get("cells", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "display": self.display,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class TeacherNeed:
    need_id: str
    cell_id: str
    evidence_id: str
    phase: str
    source: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    freshness: FreshnessDemand = FreshnessDemand.ONCE
    max_age_s: float | None = None

    def __post_init__(self) -> None:
        if not all((self.need_id, self.cell_id, self.evidence_id, self.phase, self.source)):
            raise ValueError("teacher need identifiers/source/phase must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("teacher need confidence must be in [0, 1]")
        if self.freshness is FreshnessDemand.MAX_AGE and self.max_age_s is None:
            raise ValueError("MAX_AGE teacher needs require max_age_s")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherNeed:
        _reject_unknown_keys(
            raw,
            {
                "need_id",
                "cell_id",
                "evidence_id",
                "phase",
                "source",
                "arguments",
                "confidence",
                "freshness",
                "max_age_s",
            },
            "teacher need",
        )
        return cls(
            need_id=str(raw["need_id"]),
            cell_id=str(raw["cell_id"]),
            evidence_id=str(raw["evidence_id"]),
            phase=str(raw["phase"]),
            source=str(raw["source"]),
            arguments=dict(raw.get("arguments", {})),
            confidence=float(raw.get("confidence", 1.0)),
            freshness=FreshnessDemand(str(raw.get("freshness", FreshnessDemand.ONCE))),
            max_age_s=None if raw.get("max_age_s") is None else float(raw["max_age_s"]),
        )

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "need_id": self.need_id,
            "cell_id": self.cell_id,
            "evidence_id": self.evidence_id,
            "phase": self.phase,
            "source": self.source,
            "arguments": dict(self.arguments),
            "confidence": self.confidence,
            "freshness": self.freshness.value,
        }
        if self.max_age_s is not None:
            raw["max_age_s"] = self.max_age_s
        return raw


@dataclass(frozen=True, slots=True)
class TeacherPlan:
    task_id: str
    final_answer: str
    frames: tuple[TeacherFrame, ...]
    needs: tuple[TeacherNeed, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.final_answer:
            raise ValueError("teacher plan requires non-empty task_id and final_answer")
        phases = tuple(frame.phase for frame in self.frames)
        if len(phases) != len(set(phases)):
            raise ValueError("teacher plan phases must be unique")
        if not self.frames or self.frames[0].phase != "initial":
            raise ValueError("teacher plan must begin with the initial phase")
        need_ids = tuple(need.need_id for need in self.needs)
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("teacher need IDs must be unique")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherPlan:
        _reject_teacher_timing(raw)
        _reject_unknown_keys(raw, {"task_id", "final_answer", "frames", "needs"}, "teacher plan")
        return cls(
            task_id=str(raw["task_id"]),
            final_answer=str(raw["final_answer"]),
            frames=tuple(TeacherFrame.from_dict(item) for item in raw.get("frames", ())),
            needs=tuple(TeacherNeed.from_dict(item) for item in raw.get("needs", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_answer": self.final_answer,
            "frames": [frame.to_dict() for frame in self.frames],
            "needs": [need.to_dict() for need in self.needs],
        }


@dataclass(frozen=True, slots=True)
class TeacherScheduleConfig:
    thought_capacity: int = 8
    min_delay_steps: int = 1
    max_delay_steps: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.thought_capacity <= 0:
            raise ValueError("thought_capacity must be positive")
        if self.min_delay_steps <= 0 or self.max_delay_steps < self.min_delay_steps:
            raise ValueError("teacher event delays must satisfy 0 < min <= max")


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    task_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class TeacherReview:
    task_id: str
    accepted: bool
    reasons: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "fingerprint": self.fingerprint,
        }


def build_teacher_request(task: TeacherTask) -> TeacherRequest:
    task_json = json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    prompt = f"""You are producing supervision for Continuous Interaction Diffusion (CID).
Return exactly one JSON object and no prose.

Do not write private chain-of-thought. `semantic_text` must be a concise cognitive-state summary,
not a transcript of hidden reasoning. Use typed cognitive roles from:
hypothesis, information_need, percept, plan, constraint, conclusion.

The teacher controls semantic state only. Do NOT emit numeric timesteps, event arrival times,
physical TCT slot numbers, or cache schedules. Those are randomized independently by the compiler.

Output schema:
{{
  "task_id": "{task.task_id}",
  "final_answer": "...",
  "frames": [
    {{"phase":"initial","display":"...","cells":[CELL,...]}},
    {{"phase":"pre","display":"...","cells":[CELL,...]}},
    {{"phase":"after:EVIDENCE_ID","display":"...","cells":[CELL,...]}},
    {{"phase":"final","display":"...","cells":[CELL,...]}}
  ],
  "needs": [
    {{
      "need_id":"...", "cell_id":"...", "evidence_id":"...", "phase":"pre",
      "source":"...", "arguments":{{}}, "confidence":1.0, "freshness":"once"
    }}
  ]
}}

CELL schema:
{{
  "cell_id":"stable-logical-id",
  "semantic_text":"short state summary",
  "roles":{{"plan":1.0}},
  "uncertainty":0.5,
  "noise":0.5,
  "lifecycle":"active",
  "anchors":[],
  "links":[]
}}

Rules:
- Preserve every previously introduced cell in later frames; retire it explicitly if obsolete.
- `initial` must be first. Use `pre` before external evidence if the task needs tools.
- For every supplied evidence item E, emit exactly one `after:E` frame in evidence-list order.
- A final frame is optional; if present it comes after all evidence frames.
- Pre-evidence frames must not reveal values that have not arrived yet.
- Needs refer to evidence IDs and source schemas supplied below.
- Keep cognition compact; create only cells that serve a distinct cognitive role.

TASK:
{task_json}
"""
    return TeacherRequest(task_id=task.task_id, prompt=prompt)


def teacher_tasks_from_trajectories(
    examples: tuple[TrajectoryExample, ...],
) -> tuple[TeacherTask, ...]:
    return tuple(
        TeacherTask(
            task_id=example.example_id,
            prompt=example.prompt,
            protected_facts=dict(example.protected_facts),
            source_descriptors=example.source_descriptors,
            evidence=tuple(
                TeacherEvidence(
                    evidence_id=f"evidence-{index}",
                    source=event.source,
                    value=event.value,
                    arguments=dict(event.arguments),
                    version=event.version,
                    provenance=event.provenance,
                )
                for index, event in enumerate(example.events)
            ),
            metadata={
                **dict(example.metadata),
                "source_example_id": example.example_id,
            },
        )
        for example in examples
    )


def compile_teacher_plans(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
    config: TeacherScheduleConfig | None = None,
) -> tuple[TrajectoryExample, ...]:
    config = config or TeacherScheduleConfig()
    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = "; ".join(
            f"{review.task_id}: {', '.join(review.reasons)}" for review in rejected
        )
        raise ValueError(f"teacher plans failed quality review: {detail}")
    plan_by_id = {plan.task_id: plan for plan in plans}
    if len(plan_by_id) != len(plans):
        raise ValueError("teacher plan task IDs must be unique")
    missing = [task.task_id for task in tasks if task.task_id not in plan_by_id]
    if missing:
        raise ValueError(f"missing teacher plans for tasks: {missing}")
    extra = sorted(set(plan_by_id) - {task.task_id for task in tasks})
    if extra:
        raise ValueError(f"teacher plans contain unknown task IDs: {extra}")

    rng = random.Random(config.seed)
    return tuple(
        _compile_one(task, plan_by_id[task.task_id], config, rng) for task in tasks
    )


def review_teacher_plans(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
) -> tuple[TeacherReview, ...]:
    plan_by_id = {plan.task_id: plan for plan in plans}
    if len(plan_by_id) != len(plans):
        raise ValueError("teacher plan task IDs must be unique")
    task_ids = {task.task_id for task in tasks}
    missing = [task.task_id for task in tasks if task.task_id not in plan_by_id]
    if missing:
        raise ValueError(f"missing teacher plans for tasks: {missing}")
    extra = sorted(set(plan_by_id) - task_ids)
    if extra:
        raise ValueError(f"teacher plans contain unknown task IDs: {extra}")

    reviews: list[TeacherReview] = []
    seen_fingerprints: dict[str, str] = {}
    for task in tasks:
        plan = plan_by_id[task.task_id]
        reasons = list(_teacher_quality_reasons(task, plan))
        fingerprint = _teacher_fingerprint(task, plan)
        duplicate_of = seen_fingerprints.get(fingerprint)
        if duplicate_of is not None:
            reasons.append(f"semantic duplicate of {duplicate_of}")
        else:
            seen_fingerprints[fingerprint] = task.task_id
        reviews.append(
            TeacherReview(
                task_id=task.task_id,
                accepted=not reasons,
                reasons=tuple(reasons),
                fingerprint=fingerprint,
            )
        )
    return tuple(reviews)


def dump_teacher_requests(tasks: Iterable[TeacherTask], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for task in tasks:
            request = build_teacher_request(task)
            handle.write(
                json.dumps(
                    {"task_id": request.task_id, "prompt": request.prompt},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def dump_teacher_tasks(tasks: Iterable[TeacherTask], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    task.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def dump_teacher_plans(plans: Iterable[TeacherPlan], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(
                json.dumps(
                    plan.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def dump_teacher_reviews(reviews: Iterable[TeacherReview], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for review in reviews:
            handle.write(
                json.dumps(
                    review.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_teacher_tasks(path: str | Path) -> tuple[TeacherTask, ...]:
    return tuple(
        TeacherTask.from_dict(item) for item in _load_json_objects(path, "teacher task")
    )


def load_teacher_plans(path: str | Path) -> tuple[TeacherPlan, ...]:
    return tuple(
        TeacherPlan.from_dict(item) for item in _load_json_objects(path, "teacher plan")
    )


def _compile_one(
    task: TeacherTask,
    plan: TeacherPlan,
    config: TeacherScheduleConfig,
    rng: random.Random,
) -> TrajectoryExample:
    if task.task_id != plan.task_id:
        raise ValueError("teacher task and plan IDs do not match")
    frame_by_phase = {frame.phase: frame for frame in plan.frames}
    expected_after = tuple(f"after:{item.evidence_id}" for item in task.evidence)
    for phase in expected_after:
        if phase not in frame_by_phase:
            raise ValueError(f"teacher plan is missing required evidence frame {phase!r}")
    allowed = {"initial", "pre", "final", *expected_after}
    unknown = set(frame_by_phase) - allowed
    if unknown:
        raise ValueError(f"teacher plan contains unsupported phases: {sorted(unknown)}")
    _validate_need_contract(task, plan, frame_by_phase)

    semantic_frames = [frame_by_phase["initial"]]
    if "pre" in frame_by_phase:
        semantic_frames.append(frame_by_phase["pre"])
    semantic_frames.extend(frame_by_phase[phase] for phase in expected_after)
    if "final" in frame_by_phase:
        semantic_frames.append(frame_by_phase["final"])
    _validate_monotonic_cells(semantic_frames)

    ordered_cell_ids: list[str] = []
    for frame in semantic_frames:
        for cell in frame.cells:
            if cell.cell_id not in ordered_cell_ids:
                ordered_cell_ids.append(cell.cell_id)
    if len(ordered_cell_ids) > config.thought_capacity:
        raise ValueError("teacher plan exceeds configured TCT capacity")
    slots = dict(
        zip(
            ordered_cell_ids,
            rng.sample(range(config.thought_capacity), len(ordered_cell_ids)),
            strict=True,
        )
    )

    scheduled: list[tuple[int, TeacherFrame]] = [(0, frame_by_phase["initial"])]
    current_step = 0
    current_frame = frame_by_phase["initial"]
    if "pre" in frame_by_phase:
        current_step += 1
        current_frame = frame_by_phase["pre"]
        scheduled.append((current_step, current_frame))

    arrival_steps: dict[str, int] = {}
    needs_by_evidence = {
        evidence.evidence_id: tuple(
            need for need in plan.needs if need.evidence_id == evidence.evidence_id
        )
        for evidence in task.evidence
    }
    for evidence in task.evidence:
        delay = rng.randint(config.min_delay_steps, config.max_delay_steps)
        arrival_step = current_step + delay
        waiting_ids = {need.cell_id for need in needs_by_evidence[evidence.evidence_id]}
        for step in range(current_step + 1, arrival_step):
            scheduled.append((step, _waiting_frame(current_frame, waiting_ids)))
        current_step = arrival_step
        teacher_after = frame_by_phase[f"after:{evidence.evidence_id}"]
        arrival_frame = _arrival_frame(teacher_after, waiting_ids)
        scheduled.append((current_step, arrival_frame))
        arrival_steps[evidence.evidence_id] = arrival_step
        current_frame = teacher_after
        if arrival_frame != teacher_after:
            current_step += 1
            scheduled.append((current_step, teacher_after))

    if "final" in frame_by_phase:
        current_step += 1
        current_frame = frame_by_phase["final"]
        scheduled.append((current_step, current_frame))

    phase_first_step: dict[str, int] = {}
    for step, frame in scheduled:
        phase_first_step.setdefault(frame.phase, step)

    binding_targets = tuple(
        _binding_target(need, task, phase_first_step) for need in plan.needs
    )
    events = tuple(
        ExternalEvent(
            source=item.source,
            value=item.value,
            arrival_step=arrival_steps[item.evidence_id],
            version=item.version,
            provenance=item.provenance,
            arguments=dict(item.arguments),
        )
        for item in task.evidence
    )
    thought_targets = tuple(
        ThoughtTarget(
            step=step,
            slot=slots[cell.cell_id],
            cell_id=cell.cell_id,
            semantic_text=cell.semantic_text,
            roles=cell.roles,
            uncertainty=cell.uncertainty,
            noise=cell.noise,
            lifecycle=cell.lifecycle,
        )
        for step, frame in scheduled
        for cell in frame.cells
    )
    display_targets = tuple(
        DisplayTarget(step=step, text=frame.display) for step, frame in scheduled
    )
    grounding_targets = tuple(
        GroundingTarget(
            step=step,
            cell_id=cell.cell_id,
            anchors=cell.anchors,
            links=cell.links,
        )
        for step, frame in scheduled
        for cell in frame.cells
        if cell.anchors or cell.links
    )
    grounding_catalog = _grounding_catalog(semantic_frames)
    return TrajectoryExample(
        example_id=task.task_id,
        prompt=task.prompt,
        target_display=plan.final_answer,
        protected_facts=dict(task.protected_facts),
        source_descriptors=task.source_descriptors,
        events=events,
        binding_targets=binding_targets,
        grounding_catalog=grounding_catalog,
        grounding_targets=grounding_targets,
        thought_targets=thought_targets,
        display_targets=display_targets,
        metadata={
            **dict(task.metadata),
            "distillation": "teacher-semantic-plan-v1",
            "event_arrival_steps": arrival_steps,
        },
    )


def _binding_target(
    need: TeacherNeed,
    task: TeacherTask,
    phase_first_step: Mapping[str, int],
) -> BindingTarget:
    first_step = phase_first_step[need.phase]
    descriptor = next(
        item for item in task.source_descriptors if str(item.get("name", "")) == need.source
    )
    required = tuple(
        str(item.get("name", ""))
        for item in descriptor.get("arguments", ())
        if bool(item.get("required", True))
    )
    executable = all(name in need.arguments for name in required)
    return BindingTarget(
        need_id=need.need_id,
        source=need.source,
        first_need_step=first_step,
        executable_step=first_step if executable else None,
        arguments=dict(need.arguments),
        argument_steps={name: first_step for name in need.arguments},
        confidence=need.confidence,
        freshness=need.freshness,
        max_age_s=need.max_age_s,
        target_cells=(ObjectRef.cell(need.cell_id),),
    )


def _validate_need_contract(
    task: TeacherTask,
    plan: TeacherPlan,
    frame_by_phase: Mapping[str, TeacherFrame],
) -> None:
    evidence_by_id = {item.evidence_id: item for item in task.evidence}
    source_names = {str(item.get("name", "")) for item in task.source_descriptors}
    for need in plan.needs:
        if need.evidence_id not in evidence_by_id:
            raise ValueError(f"teacher need references unknown evidence {need.evidence_id!r}")
        if need.source not in source_names:
            raise ValueError(f"teacher need references unknown source {need.source!r}")
        if evidence_by_id[need.evidence_id].source != need.source:
            raise ValueError("teacher need source must match its evidence source")
        if need.phase not in frame_by_phase:
            raise ValueError(f"teacher need references unknown phase {need.phase!r}")
        cells = {cell.cell_id for cell in frame_by_phase[need.phase].cells}
        if need.cell_id not in cells:
            raise ValueError("teacher need target cell must exist in its activation phase")


def _validate_monotonic_cells(frames: list[TeacherFrame]) -> None:
    previous: set[str] = set()
    for frame in frames:
        current = {cell.cell_id for cell in frame.cells}
        missing = previous - current
        if missing:
            raise ValueError(
                f"teacher frame {frame.phase!r} removed cells without retirement: {sorted(missing)}"
            )
        previous = current


def _waiting_frame(frame: TeacherFrame, waiting_ids: set[str]) -> TeacherFrame:
    cells = tuple(
        replace(cell, lifecycle=CellLifecycle.WAITING)
        if cell.cell_id in waiting_ids and cell.lifecycle is not CellLifecycle.RETIRED
        else cell
        for cell in frame.cells
    )
    return TeacherFrame(phase=frame.phase, display=frame.display, cells=cells)


def _arrival_frame(frame: TeacherFrame, waiting_ids: set[str]) -> TeacherFrame:
    cells = tuple(
        replace(cell, lifecycle=CellLifecycle.ACTIVE)
        if cell.cell_id in waiting_ids and cell.lifecycle is not CellLifecycle.RETIRED
        else cell
        for cell in frame.cells
    )
    return TeacherFrame(phase=frame.phase, display=frame.display, cells=cells)


def _grounding_catalog(frames: list[TeacherFrame]) -> tuple[GroundingEntry, ...]:
    anchors: dict[str, Anchor] = {}
    for frame in frames:
        for cell in frame.cells:
            for anchor in cell.anchors:
                existing = anchors.get(anchor.anchor_id)
                if existing is not None and existing != anchor:
                    raise ValueError(
                        f"anchor ID changes meaning across teacher frames: {anchor.anchor_id}"
                    )
                anchors[anchor.anchor_id] = anchor
    return tuple(GroundingEntry(anchor=anchor) for anchor in anchors.values())


def _source_descriptor(raw: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = dict(raw)
    descriptor["arguments"] = tuple(
        dict(argument) for argument in descriptor.get("arguments", ())
    )
    return descriptor


def _load_json_objects(path: str | Path, label: str) -> tuple[Mapping[str, Any], ...]:
    items: list[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("record must be a JSON object")
                items.append(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {label} at line {line_number}: {exc}") from exc
    return tuple(items)


def _reject_teacher_timing(value: Any, path: str = "plan") -> None:
    forbidden = {"step", "slot", "arrival_step", "arrival_time", "event_time"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in forbidden:
                raise ValueError(f"teacher plan must not control {path}.{key_text}")
            _reject_teacher_timing(item, f"{path}.{key_text}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_teacher_timing(item, f"{path}[{index}]")


def _reject_unknown_keys(
    raw: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {unknown}")


def _teacher_quality_reasons(
    task: TeacherTask,
    plan: TeacherPlan,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if task.task_id != plan.task_id:
        return ("task/plan IDs do not match",)

    frame_by_phase = {frame.phase: frame for frame in plan.frames}
    expected_after = tuple(f"after:{item.evidence_id}" for item in task.evidence)
    missing_phases = tuple(phase for phase in expected_after if phase not in frame_by_phase)
    if missing_phases:
        reasons.append(f"missing evidence frames: {list(missing_phases)}")
    allowed = {"initial", "pre", "final", *expected_after}
    unknown_phases = sorted(set(frame_by_phase) - allowed)
    if unknown_phases:
        reasons.append(f"unsupported phases: {unknown_phases}")

    semantic_frames = [frame_by_phase["initial"]]
    if "pre" in frame_by_phase:
        semantic_frames.append(frame_by_phase["pre"])
    semantic_frames.extend(
        frame_by_phase[phase] for phase in expected_after if phase in frame_by_phase
    )
    if "final" in frame_by_phase:
        semantic_frames.append(frame_by_phase["final"])
    try:
        _validate_monotonic_cells(semantic_frames)
    except ValueError as exc:
        reasons.append(str(exc))
    try:
        _validate_need_contract(task, plan, frame_by_phase)
    except (StopIteration, ValueError) as exc:
        reasons.append(str(exc) or "invalid teacher need contract")

    evidence_by_id = {item.evidence_id: item for item in task.evidence}
    source_by_name = {
        str(item.get("name", "")): item for item in task.source_descriptors
    }
    for need in plan.needs:
        evidence = evidence_by_id.get(need.evidence_id)
        descriptor = source_by_name.get(need.source)
        if evidence is None or descriptor is None:
            continue
        for argument in descriptor.get("arguments", ()):
            if not bool(argument.get("required", True)):
                continue
            name = str(argument.get("name", ""))
            if name not in need.arguments:
                reasons.append(f"need {need.need_id} is missing required argument {name!r}")
                continue
            if name not in evidence.arguments:
                reasons.append(
                    f"evidence {evidence.evidence_id} is missing required argument {name!r}"
                )
                continue
            if need.arguments[name] != evidence.arguments[name]:
                reasons.append(
                    f"need {need.need_id} argument {name!r} does not match supplied evidence"
                )

    final_frame = semantic_frames[-1]
    if final_frame.display.strip() != plan.final_answer.strip():
        reasons.append("final frame display does not match final_answer")
    if not any(
        cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0
        for cell in final_frame.cells
    ):
        reasons.append("final frame has no conclusion-role cell")
    reasons.extend(_future_evidence_leaks(task, semantic_frames))
    return tuple(dict.fromkeys(reasons))


def _future_evidence_leaks(
    task: TeacherTask,
    frames: list[TeacherFrame],
) -> tuple[str, ...]:
    baseline = _normalized_text(
        " ".join(
            (
                task.prompt,
                json.dumps(task.protected_facts, ensure_ascii=False, sort_keys=True, default=str),
                json.dumps(
                    task.source_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
        )
    )
    evidence_by_id = {item.evidence_id: item for item in task.evidence}
    evidence_order = tuple(item.evidence_id for item in task.evidence)
    available: set[str] = set()
    reasons: list[str] = []
    for frame in frames:
        if frame.phase.startswith("after:"):
            evidence_id = frame.phase.removeprefix("after:")
            if evidence_id in evidence_by_id:
                available.add(evidence_id)
        elif frame.phase == "final":
            available.update(evidence_order)

        frame_text = _frame_visibility_text(frame)
        for evidence_id in evidence_order:
            if evidence_id in available:
                continue
            evidence = evidence_by_id[evidence_id]
            for marker in _evidence_markers(evidence.value):
                normalized_marker = _normalized_text(marker)
                if normalized_marker in baseline:
                    continue
                if normalized_marker and normalized_marker in frame_text:
                    reasons.append(
                        f"phase {frame.phase!r} leaks future evidence {evidence_id!r}"
                    )
                    break
    return tuple(reasons)


def _frame_visibility_text(frame: TeacherFrame) -> str:
    parts = [frame.display]
    for cell in frame.cells:
        parts.append(cell.semantic_text)
        parts.extend(str(anchor.value) for anchor in cell.anchors)
    return _normalized_text(" ".join(parts))


def _evidence_markers(value: Any) -> tuple[str, ...]:
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, (int, float)):
        return (str(value),)
    if isinstance(value, str):
        marker = value.strip()
        return (marker,) if len(marker) >= 3 else ()
    marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return (marker,) if len(marker) >= 4 else ()


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _teacher_fingerprint(task: TeacherTask, plan: TeacherPlan) -> str:
    task_payload = task.to_dict()
    task_payload.pop("task_id", None)
    task_payload.pop("metadata", None)
    plan_payload = plan.to_dict()
    plan_payload.pop("task_id", None)
    payload = json.dumps(
        {"task": task_payload, "plan": plan_payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()

from __future__ import annotations

import json
import random
import re
import string
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
from cid.python_review import python_public_test_reason
from cid.state import CellLifecycle, CognitiveRole


@dataclass(frozen=True, slots=True)
class TeacherEvidence:
    evidence_id: str
    source: str
    value: Any
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    requires_need: bool = True
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
            depends_on=tuple(str(item) for item in raw.get("depends_on", ())),
            requires_need=bool(raw.get("requires_need", True)),
            version=None if raw.get("version") is None else str(raw["version"]),
            provenance=(None if raw.get("provenance") is None else str(raw["provenance"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "value": self.value,
            "arguments": dict(self.arguments),
            "depends_on": list(self.depends_on),
            "requires_need": self.requires_need,
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
    reference_answer: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("teacher task requires non-empty task_id and prompt")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("teacher task evidence IDs must be unique")
        evidence_id_set = set(evidence_ids)
        seen_evidence: set[str] = set()
        for item in self.evidence:
            unknown_dependencies = set(item.depends_on) - evidence_id_set
            if unknown_dependencies:
                raise ValueError(
                    "teacher evidence dependencies reference unknown evidence IDs: "
                    f"{sorted(unknown_dependencies)}"
                )
            if item.evidence_id in item.depends_on:
                raise ValueError("teacher evidence cannot depend on itself")
            if any(dependency not in seen_evidence for dependency in item.depends_on):
                raise ValueError("teacher evidence must be topologically ordered by dependency")
            seen_evidence.add(item.evidence_id)
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
            evidence=tuple(TeacherEvidence.from_dict(item) for item in raw.get("evidence", ())),
            metadata=dict(raw.get("metadata", {})),
            reference_answer=(
                None if raw.get("reference_answer") is None else str(raw["reference_answer"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "protected_facts": dict(self.protected_facts),
            "source_descriptors": [dict(item) for item in self.source_descriptors],
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": dict(self.metadata),
            "reference_answer": self.reference_answer,
        }

    def teacher_visible_dict(self) -> dict[str, Any]:
        """Return the task payload visible to a semantic teacher.

        The gold answer remains available to deterministic review/auditing but is deliberately
        excluded from teacher input so no-tool cognition must solve the task and evidence-backed
        cognition cannot shortcut the external source path from a leaked label.
        """

        raw = self.to_dict()
        raw.pop("reference_answer", None)
        return raw


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
    variants_per_task: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.thought_capacity <= 0:
            raise ValueError("thought_capacity must be positive")
        if self.min_delay_steps <= 0 or self.max_delay_steps < self.min_delay_steps:
            raise ValueError("teacher event delays must satisfy 0 < min <= max")
        if self.variants_per_task <= 0:
            raise ValueError("variants_per_task must be positive")


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
    task_json = json.dumps(
        task.teacher_visible_dict(), ensure_ascii=False, sort_keys=True, indent=2
    )
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
- Solve the task yourself. No gold/reference answer is included in TASK.

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
            evidence=_teacher_evidence_from_trajectory(example),
            metadata={
                **dict(example.metadata),
                "source_example_id": example.example_id,
                "task_kind": str(example.metadata.get("task_kind", "synthetic_mechanism")),
            },
            reference_answer=example.target_display,
        )
        for example in examples
    )


def _teacher_evidence_from_trajectory(
    example: TrajectoryExample,
) -> tuple[TeacherEvidence, ...]:
    binding_keys = {
        _source_arguments_key(binding.source, binding.arguments)
        for binding in example.binding_targets
    }
    previous_by_binding: dict[str, str] = {}
    evidence: list[TeacherEvidence] = []
    for index, event in enumerate(example.events):
        evidence_id = f"evidence-{index}"
        binding_key = _source_arguments_key(event.source, event.arguments)
        previous = previous_by_binding.get(binding_key)
        requires_need = binding_key in binding_keys and previous is None
        evidence.append(
            TeacherEvidence(
                evidence_id=evidence_id,
                source=event.source,
                value=event.value,
                arguments=dict(event.arguments),
                depends_on=() if previous is None else (previous,),
                requires_need=requires_need,
                version=event.version,
                provenance=event.provenance,
            )
        )
        previous_by_binding[binding_key] = evidence_id
    return tuple(evidence)


def _source_arguments_key(source: str, arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, default=str)
    return f"{source}\0{payload}"


def compile_teacher_plans(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
    config: TeacherScheduleConfig | None = None,
) -> tuple[TrajectoryExample, ...]:
    config = config or TeacherScheduleConfig()
    plan_by_id = {plan.task_id: plan for plan in plans}
    if len(plan_by_id) != len(plans):
        raise ValueError("teacher plan task IDs must be unique")
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("teacher task IDs must be unique")
    extra = sorted(set(plan_by_id) - set(task_by_id))
    if extra:
        raise ValueError(f"teacher plans contain unknown task IDs: {extra}")

    # `review-distillation` intentionally emits only accepted plans.  The
    # compiler therefore treats the supplied plan IDs as the selected task
    # subset while still validating every supplied plan against its task.
    # This keeps the documented full-tasks + accepted-plans workflow usable
    # without weakening review completeness at the review stage itself.
    selected_tasks = tuple(task for task in tasks if task.task_id in plan_by_id)
    reviews = review_teacher_plans(selected_tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = "; ".join(f"{review.task_id}: {', '.join(review.reasons)}" for review in rejected)
        raise ValueError(f"teacher plans failed quality review: {detail}")

    rng = random.Random(config.seed)
    compiled: list[TrajectoryExample] = []
    for task in selected_tasks:
        for variant_index in range(config.variants_per_task):
            example = _compile_one(task, plan_by_id[task.task_id], config, rng)
            if config.variants_per_task > 1:
                example = replace(
                    example,
                    example_id=f"{task.task_id}::schedule-{variant_index:02d}",
                    metadata={
                        **dict(example.metadata),
                        "semantic_task_id": task.task_id,
                        "schedule_variant": variant_index,
                    },
                )
            compiled.append(example)
    return tuple(compiled)


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
    return tuple(TeacherTask.from_dict(item) for item in _load_json_objects(path, "teacher task"))


def load_teacher_plans(path: str | Path) -> tuple[TeacherPlan, ...]:
    return tuple(TeacherPlan.from_dict(item) for item in _load_json_objects(path, "teacher plan"))


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
    slots_by_phase, visible_cells_by_phase = _allocate_teacher_slots(
        semantic_frames,
        config.thought_capacity,
        rng,
    )

    needs_by_evidence = {
        evidence.evidence_id: tuple(
            need for need in plan.needs if need.evidence_id == evidence.evidence_id
        )
        for evidence in task.evidence
    }
    launch_steps, arrival_steps = _event_schedule(
        task,
        plan,
        frame_by_phase,
        config,
        rng,
    )
    scheduled = _schedule_frames(
        task,
        frame_by_phase,
        needs_by_evidence,
        launch_steps,
        arrival_steps,
    )

    phase_first_step: dict[str, int] = {}
    for step, frame in scheduled:
        phase_first_step.setdefault(frame.phase, step)

    binding_targets = tuple(_binding_target(need, task, phase_first_step) for need in plan.needs)
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
            slot=slots_by_phase[frame.phase][cell.cell_id],
            cell_id=cell.cell_id,
            semantic_text=cell.semantic_text,
            roles=cell.roles,
            uncertainty=cell.uncertainty,
            noise=cell.noise,
            lifecycle=cell.lifecycle,
        )
        for step, frame in scheduled
        for cell in frame.cells
        if cell.cell_id in visible_cells_by_phase[frame.phase]
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
        if cell.cell_id in visible_cells_by_phase[frame.phase]
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
            "event_launch_steps": launch_steps,
            "event_arrival_steps": arrival_steps,
        },
    )


def _event_schedule(
    task: TeacherTask,
    plan: TeacherPlan,
    frame_by_phase: Mapping[str, TeacherFrame],
    config: TeacherScheduleConfig,
    rng: random.Random,
) -> tuple[dict[str, int], dict[str, int]]:
    phase_steps: dict[str, int] = {"initial": 0}
    if "pre" in frame_by_phase:
        phase_steps["pre"] = 1
    needs_by_evidence = {
        evidence.evidence_id: tuple(
            need for need in plan.needs if need.evidence_id == evidence.evidence_id
        )
        for evidence in task.evidence
    }
    launch_steps: dict[str, int] = {}
    arrival_steps: dict[str, int] = {}
    previous_arrival = -1
    for evidence in task.evidence:
        evidence_needs = needs_by_evidence[evidence.evidence_id]
        activation_steps: list[int] = []
        for need in evidence_needs:
            if need.phase not in phase_steps:
                raise ValueError(
                    f"teacher need phase {need.phase!r} is not causally available before "
                    f"evidence {evidence.evidence_id!r}"
                )
            activation_steps.append(phase_steps[need.phase])
        dependency_steps = [arrival_steps[item] for item in evidence.depends_on]
        launch_step = max((*activation_steps, *dependency_steps, 0))
        launch_steps[evidence.evidence_id] = launch_step
        proposed_arrival = launch_step + rng.randint(config.min_delay_steps, config.max_delay_steps)
        # Teacher semantic frames are generated in evidence-list order. Sibling calls may launch
        # together, but their sampled arrivals remain strictly ordered so the same semantic plan is
        # valid while I/O still overlaps in flight.
        arrival_step = max(proposed_arrival, previous_arrival + 1)
        arrival_steps[evidence.evidence_id] = arrival_step
        phase_steps[f"after:{evidence.evidence_id}"] = arrival_step
        previous_arrival = arrival_step
    return launch_steps, arrival_steps


def _schedule_frames(
    task: TeacherTask,
    frame_by_phase: Mapping[str, TeacherFrame],
    needs_by_evidence: Mapping[str, tuple[TeacherNeed, ...]],
    launch_steps: Mapping[str, int],
    arrival_steps: Mapping[str, int],
) -> list[tuple[int, TeacherFrame]]:
    arrival_by_step = {step: evidence_id for evidence_id, step in arrival_steps.items()}
    if len(arrival_by_step) != len(arrival_steps):
        raise ValueError("compiled evidence arrivals must be unique by runtime step")
    last_arrival = max(arrival_steps.values(), default=0)
    scheduled: list[tuple[int, TeacherFrame]] = []
    current_semantic = frame_by_phase["initial"]
    initial_pre_step = 1 if "pre" in frame_by_phase else None
    horizon = max(last_arrival, initial_pre_step or 0)

    for step in range(horizon + 1):
        arriving_evidence_id = arrival_by_step.get(step)
        if arriving_evidence_id is not None:
            current_semantic = frame_by_phase[f"after:{arriving_evidence_id}"]
        elif initial_pre_step == step:
            current_semantic = frame_by_phase["pre"]

        waiting_ids: set[str] = set()
        for evidence in task.evidence:
            evidence_id = evidence.evidence_id
            if launch_steps[evidence_id] < step < arrival_steps[evidence_id]:
                waiting_ids.update(need.cell_id for need in needs_by_evidence[evidence_id])
        arriving_ids = (
            set()
            if arriving_evidence_id is None
            else {need.cell_id for need in needs_by_evidence[arriving_evidence_id]}
        )
        runtime_frame = _runtime_frame(current_semantic, waiting_ids, arriving_ids)
        scheduled.append((step, runtime_frame))

    current_step = horizon
    last_runtime_frame = scheduled[-1][1]
    if last_runtime_frame != current_semantic:
        current_step += 1
        scheduled.append((current_step, current_semantic))
    if "final" in frame_by_phase:
        current_step += 1
        scheduled.append((current_step, frame_by_phase["final"]))
    return scheduled


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
    retired: set[str] = set()
    for frame in frames:
        current = {cell.cell_id for cell in frame.cells}
        missing = previous - current
        if missing:
            raise ValueError(
                f"teacher frame {frame.phase!r} removed cells without retirement: {sorted(missing)}"
            )
        reactivated = {
            cell.cell_id
            for cell in frame.cells
            if cell.cell_id in retired and cell.lifecycle is not CellLifecycle.RETIRED
        }
        if reactivated:
            raise ValueError(
                f"teacher frame {frame.phase!r} reactivated retired cells: {sorted(reactivated)}"
            )
        retired.update(
            cell.cell_id for cell in frame.cells if cell.lifecycle is CellLifecycle.RETIRED
        )
        previous = current


def _allocate_teacher_slots(
    frames: list[TeacherFrame],
    capacity: int,
    rng: random.Random,
) -> tuple[dict[str, dict[str, int]], dict[str, frozenset[str]]]:
    """Allocate physical TCT slots while recycling retired semantic cells.

    A retirement transition remains supervised for one semantic frame.  Once a
    cell was already retired in the previous frame, it no longer occupies a
    physical slot and its slot may be reused by a later cell.  This mirrors the
    runtime RETIRED -> EMPTY reclamation contract while preserving an explicit
    retirement target for the model.
    """

    if capacity <= 0:
        raise ValueError("thought capacity must be positive")

    free_slots = list(rng.sample(range(capacity), capacity))
    active_slots: dict[str, int] = {}
    previous_lifecycle: dict[str, CellLifecycle] = {}
    slots_by_phase: dict[str, dict[str, int]] = {}
    visible_by_phase: dict[str, frozenset[str]] = {}

    for frame in frames:
        live_ids = [
            cell.cell_id for cell in frame.cells if cell.lifecycle is not CellLifecycle.RETIRED
        ]
        retiring_ids = [
            cell.cell_id
            for cell in frame.cells
            if cell.lifecycle is CellLifecycle.RETIRED
            and previous_lifecycle.get(cell.cell_id) is not None
            and previous_lifecycle[cell.cell_id] is not CellLifecycle.RETIRED
        ]
        visible_ids = [*live_ids, *retiring_ids]
        if len(visible_ids) > capacity:
            raise ValueError(
                "teacher plan exceeds configured TCT capacity after retired-cell reclamation"
            )

        for cell_id in live_ids:
            if cell_id in active_slots:
                continue
            if not free_slots:
                raise ValueError(
                    "teacher plan exceeds configured TCT capacity after retired-cell reclamation"
                )
            active_slots[cell_id] = free_slots.pop()

        # Retiring cells must still own their old slot for this frame so the
        # retirement transition has a target. They are released immediately
        # after the phase and can be reused by subsequent semantic frames.
        for cell_id in retiring_ids:
            if cell_id not in active_slots:
                raise ValueError(f"retiring teacher cell has no physical slot: {cell_id}")

        slots_by_phase[frame.phase] = {
            cell_id: active_slots[cell_id] for cell_id in visible_ids
        }
        visible_by_phase[frame.phase] = frozenset(visible_ids)

        for cell_id in retiring_ids:
            free_slots.append(active_slots.pop(cell_id))
        previous_lifecycle = {cell.cell_id: cell.lifecycle for cell in frame.cells}

    return slots_by_phase, visible_by_phase


def _runtime_frame(
    frame: TeacherFrame,
    waiting_ids: set[str],
    arriving_ids: set[str],
) -> TeacherFrame:
    cells = tuple(
        (
            replace(cell, lifecycle=CellLifecycle.ACTIVE)
            if cell.cell_id in arriving_ids and cell.lifecycle is not CellLifecycle.RETIRED
            else (
                replace(cell, lifecycle=CellLifecycle.WAITING)
                if cell.cell_id in waiting_ids and cell.lifecycle is not CellLifecycle.RETIRED
                else cell
            )
        )
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
                    # Entity anchors are keyed from a case-insensitive canonical
                    # value, while evidence titles can legitimately vary only in
                    # capitalization across frames (`De Martino`/`de Martino`).
                    # An identical object ID therefore denotes the same grounding;
                    # retain the first display spelling for catalog stability.
                    same_object = (
                        existing.kind is anchor.kind
                        and existing.object_id is not None
                        and existing.object_id == anchor.object_id
                    )
                    if not same_object:
                        raise ValueError(
                            f"anchor ID changes meaning across teacher frames: {anchor.anchor_id}"
                        )
                    continue
                anchors[anchor.anchor_id] = anchor
    return tuple(GroundingEntry(anchor=anchor) for anchor in anchors.values())


def _source_descriptor(raw: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = dict(raw)
    descriptor["arguments"] = tuple(dict(argument) for argument in descriptor.get("arguments", ()))
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
    source_by_name = {str(item.get("name", "")): item for item in task.source_descriptors}
    needs_by_evidence = {
        evidence.evidence_id: tuple(
            need for need in plan.needs if need.evidence_id == evidence.evidence_id
        )
        for evidence in task.evidence
    }
    for evidence in task.evidence:
        evidence_needs = needs_by_evidence[evidence.evidence_id]
        if evidence.requires_need and len(evidence_needs) != 1:
            reasons.append(f"evidence {evidence.evidence_id} requires exactly one activating need")
        if not evidence.requires_need and evidence_needs:
            reasons.append(
                f"evidence {evidence.evidence_id} reuses a persistent binding and must not "
                "create a new need"
            )
        if not evidence.requires_need:
            root_id = _persistent_root_evidence_id(evidence, evidence_by_id)
            root_needs = needs_by_evidence.get(root_id, ())
            if len(root_needs) == 1 and root_needs[0].freshness is not FreshnessDemand.ALWAYS:
                reasons.append(
                    f"need {root_needs[0].need_id} must use freshness='always' for persistent "
                    f"updates ending at {evidence.evidence_id}"
                )
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
    if not any(cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0 for cell in final_frame.cells):
        reasons.append("final frame has no conclusion-role cell")
    reference_reason = _reference_answer_reason(task, plan)
    if reference_reason is not None:
        reasons.append(reference_reason)
    if str(task.metadata.get("task_kind", "")) == "python_programming":
        public_tests = tuple(str(item) for item in task.metadata.get("public_tests", ()))
        setup = str(task.metadata.get("public_test_setup_code", ""))
        python_reason = python_public_test_reason(plan.final_answer, public_tests, setup)
        if python_reason is not None:
            reasons.append(python_reason)
    reasons.extend(_future_evidence_leaks(task, semantic_frames))
    return tuple(dict.fromkeys(reasons))


def _persistent_root_evidence_id(
    evidence: TeacherEvidence,
    evidence_by_id: Mapping[str, TeacherEvidence],
) -> str:
    current = evidence
    visited: set[str] = set()
    while not current.requires_need:
        if current.evidence_id in visited:
            raise ValueError("teacher evidence dependency graph contains a cycle")
        visited.add(current.evidence_id)
        if len(current.depends_on) != 1:
            return current.evidence_id
        parent = evidence_by_id[current.depends_on[0]]
        if parent.source != evidence.source or dict(parent.arguments) != dict(evidence.arguments):
            return current.evidence_id
        current = parent
    return current.evidence_id


def _reference_answer_reason(task: TeacherTask, plan: TeacherPlan) -> str | None:
    reference = task.reference_answer
    if reference is None:
        return None
    kind = str(task.metadata.get("task_kind", ""))
    if kind in {
        "multi_hop_qa",
        "multiple_choice_knowledge_reasoning",
        "science_multiple_choice",
        "synthetic_mechanism",
    }:
        insufficiency = _evidence_insufficiency_match(plan.final_answer)
        if insufficiency is not None:
            supported_prefix = plan.final_answer[: insufficiency.start()].strip(" ,.;:-")
            if not supported_prefix or not _normalized_answer_match(supported_prefix, reference):
                return "final_answer does not match the public reference answer"
        if not _normalized_answer_match(plan.final_answer, reference):
            return "final_answer does not match the public reference answer"
        return None
    if kind == "math_word_problem" and not _numeric_or_normalized_answer_match(
        plan.final_answer, reference
    ):
        return "final_answer does not match the public numerical reference answer"
    if kind == "competition_math":
        if _competition_math_answer_match(plan.final_answer, reference):
            return None
        if _competition_math_choice_match(task.prompt, plan.final_answer, reference):
            return None
        return "final_answer does not match the public competition-math reference answer"
    return None


_QA_INSUFFICIENCY_PATTERN = re.compile(
    r"\b(?:visible|support) evidence\b[^.]{0,160}\b"
    r"(?:does not|cannot|not stated|not establish|gives no)\b|"
    r"\bdid not\b[^.;]{0,120}\bin the visible (?:account|evidence)\b|"
    r"\bno minimum age\b|"
    r"\b(?:cannot be resolved|not established|not reliably established|not explicitly stated|"
    r"does not establish|does not state|does not give)\b",
    re.IGNORECASE,
)


def _evidence_insufficiency_match(value: str) -> re.Match[str] | None:
    return _QA_INSUFFICIENCY_PATTERN.search(value)


def _looks_like_evidence_insufficient_answer(value: str) -> bool:
    return _evidence_insufficiency_match(value) is not None


_QA_COUNTRY_ALIASES = {
    "american": "united states",
    "america": "united states",
    "united states": "united states",
    "united states american": "united states",
    "united states of america": "united states",
    "usa": "united states",
    "u s": "united states",
    "british": "united kingdom",
    "english": "united kingdom",
    "england": "united kingdom",
    "scottish": "united kingdom",
    "welsh": "united kingdom",
    "northern irish": "united kingdom",
    "united kingdom": "united kingdom",
    "uk": "united kingdom",
    "indian": "india",
    "india": "india",
    "canadian": "canada",
    "canada": "canada",
    "australian": "australia",
    "australia": "australia",
    "german": "germany",
    "germany": "germany",
    "french": "france",
    "france": "france",
    "italian": "italy",
    "italy": "italy",
    "spanish": "spain",
    "spain": "spain",
    "norwegian": "norway",
    "norway": "norway",
    "dutch": "netherlands",
    "netherlands": "netherlands",
    "swiss": "switzerland",
    "switzerland": "switzerland",
    "swedish": "sweden",
    "sweden": "sweden",
    "finnish": "finland",
    "finland": "finland",
    "polish": "poland",
    "poland": "poland",
    "irish": "ireland",
    "ireland": "ireland",
    "danish": "denmark",
    "denmark": "denmark",
    "icelandic": "iceland",
    "iceland": "iceland",
    "austrian": "austria",
    "austria": "austria",
    "belgian": "belgium",
    "belgium": "belgium",
    "portuguese": "portugal",
    "portugal": "portugal",
    "greek": "greece",
    "greece": "greece",
    "turkish": "turkey",
    "turkey": "turkey",
    "russian": "russia",
    "russia": "russia",
    "ukrainian": "ukraine",
    "ukraine": "ukraine",
    "iranian": "iran",
    "iran": "iran",
    "syrian": "syria",
    "syria": "syria",
    "israeli": "israel",
    "israel": "israel",
    "egyptian": "egypt",
    "egypt": "egypt",
    "south african": "south africa",
    "south africa": "south africa",
    "argentine": "argentina",
    "argentinian": "argentina",
    "argentina": "argentina",
    "brazilian": "brazil",
    "brazil": "brazil",
    "mexican": "mexico",
    "mexico": "mexico",
    "colombian": "colombia",
    "colombia": "colombia",
    "chilean": "chile",
    "chile": "chile",
    "peruvian": "peru",
    "peru": "peru",
    "japanese": "japan",
    "japan": "japan",
    "chinese": "china",
    "china": "china",
    "south korean": "south korea",
    "south korea": "south korea",
    "pakistani": "pakistan",
    "pakistan": "pakistan",
    "bangladeshi": "bangladesh",
    "bangladesh": "bangladesh",
    "sri lankan": "sri lanka",
    "sri lanka": "sri lanka",
    "indonesian": "indonesia",
    "indonesia": "indonesia",
    "malaysian": "malaysia",
    "malaysia": "malaysia",
    "singaporean": "singapore",
    "singapore": "singapore",
    "thai": "thailand",
    "thailand": "thailand",
    "vietnamese": "vietnam",
    "vietnam": "vietnam",
    "filipino": "philippines",
    "philippine": "philippines",
    "philippines": "philippines",
    "hungarian": "hungary",
    "hungary": "hungary",
    "czech": "czechia",
    "czech republic": "czechia",
    "czechia": "czechia",
    "slovak": "slovakia",
    "slovakia": "slovakia",
    "romanian": "romania",
    "romania": "romania",
    "bulgarian": "bulgaria",
    "bulgaria": "bulgaria",
    "croatian": "croatia",
    "croatia": "croatia",
    "serbian": "serbia",
    "serbia": "serbia",
    "slovenian": "slovenia",
    "slovenia": "slovenia",
    "cypriot": "cyprus",
    "cyprus": "cyprus",
    "cambodian": "cambodia",
    "cambodia": "cambodia",
    "armenian": "armenia",
    "armenia": "armenia",
}


_QA_SURFACE_ALIASES = {
    "abc": "american broadcasting company",
    "american broadcasting company": "american broadcasting company",
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "mumbai india": "mumbai",
    "bombay india": "mumbai",
    "parisian": "paris",
    "paris": "paris",
    "malbork": "malbork",
    "marienburg": "malbork",
    "marienburg german empire": "malbork",
    "wallachia": "wallachia",
    "principality of wallachia": "wallachia",
    "oscar for best picture": "academy award best picture",
    "academy award for best picture": "academy award best picture",
    "jaz": "jazz",
    "jazz": "jazz",
    "north west": "northwest",
    "northwest": "northwest",
}


_QA_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _normalized_answer_match(prediction: str, reference: str) -> bool:
    predicted = _qa_normalize(prediction)
    expected = _qa_normalize(reference)
    if predicted == expected:
        return True
    if not predicted or not expected:
        return False
    if expected in {"yes", "no"} and predicted.startswith(expected):
        suffix = predicted[len(expected) :]
        if (
            not suffix
            or suffix[:1].isspace()
            or prediction.lstrip().casefold().startswith(expected + "—")
        ):
            return True
    predicted_country = _QA_COUNTRY_ALIASES.get(predicted)
    expected_country = _QA_COUNTRY_ALIASES.get(expected)
    if predicted_country is not None and predicted_country == expected_country:
        return True
    predicted_surface = _QA_SURFACE_ALIASES.get(predicted)
    expected_surface = _QA_SURFACE_ALIASES.get(expected)
    if predicted_surface is not None and predicted_surface == expected_surface:
        return True
    predicted_number_word = _QA_NUMBER_WORDS.get(predicted, predicted)
    expected_number_word = _QA_NUMBER_WORDS.get(expected, expected)
    if predicted_number_word == expected_number_word:
        return True
    predicted_date = _canonical_calendar_date(predicted)
    expected_date = _canonical_calendar_date(expected)
    if predicted_date is not None and predicted_date == expected_date:
        return True
    predicted_partial_date = _canonical_partial_calendar_date(predicted)
    expected_partial_date = _canonical_partial_calendar_date(expected)
    if (
        predicted_date is not None
        and expected_partial_date is not None
        and predicted_date[1:] == expected_partial_date
    ):
        return True
    if (
        expected_date is not None
        and predicted_partial_date is not None
        and expected_date[1:] == predicted_partial_date
    ):
        return True
    if (
        predicted_partial_date is not None
        and predicted_partial_date == expected_partial_date
    ):
        return True
    if _numeric_surface_match(prediction, reference):
        return True
    numeric_reference_match = _numeric_reference_leading_match(prediction, reference)
    if numeric_reference_match is not None:
        return numeric_reference_match
    predicted_tokens = predicted.split()
    if expected_date is not None and len(predicted_tokens) >= 3:
        leading_date = _canonical_calendar_date(" ".join(predicted_tokens[:3]))
        if leading_date == expected_date:
            return True
    expected_tokens = expected.split()
    if len(predicted_tokens) <= len(expected_tokens):
        return False
    width = len(expected_tokens)
    return any(
        predicted_tokens[index : index + width] == expected_tokens
        for index in range(len(predicted_tokens) - width + 1)
    )


_QA_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _canonical_calendar_date(normalized: str) -> tuple[int, int, int] | None:
    tokens = normalized.split()
    if len(tokens) != 3:
        return None
    if tokens[0] in _QA_MONTH_NUMBERS:
        month_token, day_token, year_token = tokens
    elif tokens[1] in _QA_MONTH_NUMBERS:
        day_token, month_token, year_token = tokens
    else:
        return None
    day_token = re.sub(r"(?:st|nd|rd|th)$", "", day_token)
    if not day_token.isdigit() or not year_token.isdigit():
        return None
    day = int(day_token)
    year = int(year_token)
    month = _QA_MONTH_NUMBERS[month_token]
    if not 1 <= day <= 31 or year <= 0:
        return None
    return year, month, day


def _canonical_partial_calendar_date(normalized: str) -> tuple[int, int] | None:
    tokens = normalized.split()
    if len(tokens) != 2:
        return None
    if tokens[0] in _QA_MONTH_NUMBERS:
        month_token, day_token = tokens
    elif tokens[1] in _QA_MONTH_NUMBERS:
        day_token, month_token = tokens
    else:
        return None
    day_token = re.sub(r"(?:st|nd|rd|th)$", "", day_token)
    if not day_token.isdigit():
        return None
    day = int(day_token)
    month = _QA_MONTH_NUMBERS[month_token]
    if not 1 <= day <= 31:
        return None
    return month, day


def _numeric_surface_match(prediction: str, reference: str) -> bool:
    def surface_numbers(value: str) -> tuple[str, ...]:
        normalized = re.sub(r"(?<=\d)[-–—−](?=\d)", " ", value)
        return _number_tokens(normalized)

    predicted_numbers = surface_numbers(prediction)
    reference_numbers = surface_numbers(reference)
    if not predicted_numbers or predicted_numbers != reference_numbers:
        return False

    def residue(value: str) -> set[str]:
        normalized = re.sub(r"(?<=\d),(?=\d)", "", value.casefold())
        normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", normalized)
        normalized = re.sub(r"[^a-z]+", " ", normalized)
        return set(normalized.split())

    harmless = {"about", "approximately", "around", "to", "and", "in", "on", "ce", "ad"}
    return residue(prediction) <= harmless and residue(reference) <= harmless


def _numeric_reference_leading_match(prediction: str, reference: str) -> bool | None:
    if not _number_tokens(reference):
        return None
    normalized = re.sub(r"(?<=\d),(?=\d)", "", reference.casefold())
    normalized = re.sub(r"(?<=\d)[-–—−](?=\d)", " ", normalized)
    normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", normalized)
    normalized = re.sub(r"[^a-z]+", " ", normalized)
    if not set(normalized.split()) <= {"to", "and", "in", "on", "ce", "ad"}:
        return None

    reference_numbers = _number_tokens(re.sub(r"(?<=\d)[-–—−](?=\d)", " ", reference))
    prediction_numbers = _number_tokens(re.sub(r"(?<=\d)[-–—−](?=\d)", " ", prediction))
    if not prediction_numbers:
        return False
    if len(reference_numbers) == 1:
        expected = reference_numbers[0]
        if expected.isdigit() and len(expected) == 4 and 1000 <= int(expected) <= 2999:
            prediction_years = tuple(
                number
                for number in prediction_numbers
                if number.isdigit() and len(number) == 4 and 1000 <= int(number) <= 2999
            )
            return bool(prediction_years) and prediction_years[0] == expected
    return prediction_numbers[: len(reference_numbers)] == reference_numbers


def _numeric_or_normalized_answer_match(prediction: str, reference: str) -> bool:
    if _normalized_answer_match(prediction, reference):
        return True
    predicted_numbers = _number_tokens(prediction)
    reference_numbers = _number_tokens(reference)
    if not predicted_numbers or not reference_numbers:
        return False
    return predicted_numbers[-1] == reference_numbers[-1]


def _competition_math_answer_match(prediction: str, reference: str) -> bool:
    predicted = _math_answer_surface(prediction)
    expected = _math_answer_surface(reference)
    if predicted == expected:
        return True
    if _normalized_answer_match(prediction, reference):
        return True
    predicted_parts = _top_level_math_parts(predicted)
    expected_parts = _top_level_math_parts(expected)
    if len(predicted_parts) != len(expected_parts):
        return False
    if len(predicted_parts) == 1:
        return _symbolic_math_part_match(predicted_parts[0], expected_parts[0])
    return _unordered_math_parts_match(predicted_parts, expected_parts)


def _competition_math_choice_match(
    prompt: str, prediction: str, reference: str
) -> bool:
    choice = prediction.strip().upper()
    if not re.fullmatch(r"[A-E]", choice):
        return False
    markers = list(
        re.finditer(
            r"(?:\\(?:textbf|text)\{\(([A-E])\)\s*\}|\(([A-E])\))",
            prompt,
        )
    )
    for index, marker in enumerate(markers):
        marker_choice = marker.group(1) or marker.group(2)
        if marker_choice != choice:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(prompt)
        option = prompt[marker.end() : end]
        option = option.replace("$", "")
        option = option.replace("\\ ", "")
        option = re.sub(r"\\(?:qquad|quad)\b", "", option)
        option = option.replace("\\\\", "")
        option = option.strip(" \n\t.;")
        return _competition_math_answer_match(option, reference)
    return False


def _unordered_math_parts_match(
    predicted_parts: tuple[str, ...], expected_parts: tuple[str, ...]
) -> bool:
    candidates = [
        [
            index
            for index, expected in enumerate(expected_parts)
            if _symbolic_math_part_match(predicted, expected)
        ]
        for predicted in predicted_parts
    ]
    order = sorted(range(len(candidates)), key=lambda index: len(candidates[index]))
    used: set[int] = set()

    def assign(position: int) -> bool:
        if position == len(order):
            return True
        predicted_index = order[position]
        for expected_index in candidates[predicted_index]:
            if expected_index in used:
                continue
            used.add(expected_index)
            if assign(position + 1):
                return True
            used.remove(expected_index)
        return False

    return assign(0)


def _math_answer_surface(value: str) -> str:
    normalized = value.strip().strip("$")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\$", "")
    normalized = normalized.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    normalized = re.sub(r"\\phantom(?:\{[^{}]*\}|.)", "", normalized)
    normalized = normalized.rstrip(" .;")
    normalized = re.sub(
        r"\\(?:text|mathrm)\{\s*,?\s*and\s*\}",
        ",",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r",{2,}", ",", normalized)
    text_only = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]+)\}", normalized)
    if text_only is not None:
        normalized = text_only.group(1)
    normalized = re.sub(
        r"(?<=.)\\(?:mbox|text|mathrm)\{[^{}]*[A-Za-z][^{}]*\}(?:\^\{?\d+\}?)?$",
        "",
        normalized,
    )
    normalized = normalized.replace("{,}", "")
    normalized = normalized.replace(",\\!", "")
    normalized = re.sub(
        r"\\text\{\s*degrees?\s*\}",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.replace("\\,", "").replace("\\!", "")
    normalized = normalized.replace("\\;", "").replace("\\:", "")
    normalized = normalized.replace("\\%", "%")
    base_literal = re.fullmatch(r"([0-9A-Z]+)_\{?(\d+)\}?", normalized, flags=re.IGNORECASE)
    if base_literal is not None:
        normalized = base_literal.group(1)
    normalized = re.sub(r"\^\{?\\circ\}?", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _top_level_math_parts(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "{([":
            depth += 1
        elif char in "})]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return tuple(part for part in parts if part)


def _symbolic_math_part_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    predicted_union = _top_level_union_parts(prediction)
    reference_union = _top_level_union_parts(reference)
    if predicted_union is not None or reference_union is not None:
        if predicted_union is None or reference_union is None:
            return False
        if len(predicted_union) != len(reference_union):
            return False
        return _unordered_math_parts_match(predicted_union, reference_union)
    predicted_matrix = _latex_matrix_entries(prediction)
    reference_matrix = _latex_matrix_entries(reference)
    if predicted_matrix is not None or reference_matrix is not None:
        predicted_vector = predicted_matrix or _coordinate_tuple_entries(prediction)
        reference_vector = reference_matrix or _coordinate_tuple_entries(reference)
        if predicted_vector is None or reference_vector is None:
            return False
        if len(predicted_vector) != len(reference_vector):
            return False
        return all(
            _symbolic_math_part_match(left, right)
            for left, right in zip(predicted_vector, reference_vector, strict=True)
        )
    predicted_coordinates = _coordinate_tuple_entries(prediction)
    reference_coordinates = _coordinate_tuple_entries(reference)
    if predicted_coordinates is not None or reference_coordinates is not None:
        if predicted_coordinates is None or reference_coordinates is None:
            return False
        if len(predicted_coordinates) != len(reference_coordinates):
            return False
        return all(
            _symbolic_math_part_match(left, right)
            for left, right in zip(
                predicted_coordinates, reference_coordinates, strict=True
            )
        )
    predicted_interval = _interval_answer_parts(prediction)
    reference_interval = _interval_answer_parts(reference)
    if predicted_interval is not None or reference_interval is not None:
        if predicted_interval is None or reference_interval is None:
            return False
        predicted_left, predicted_right, predicted_endpoints = predicted_interval
        reference_left, reference_right, reference_endpoints = reference_interval
        if predicted_left != reference_left or predicted_right != reference_right:
            return False
        return all(
            _symbolic_math_part_match(left, right)
            for left, right in zip(predicted_endpoints, reference_endpoints, strict=True)
        )
    predicted_assignment = _scalar_assignment_value(prediction)
    reference_assignment = _scalar_assignment_value(reference)
    if predicted_assignment is not None and "=" not in reference:
        return _symbolic_math_part_match(predicted_assignment, reference)
    if reference_assignment is not None and "=" not in prediction:
        return _symbolic_math_part_match(prediction, reference_assignment)
    relation_match = _symbolic_equation_match(prediction, reference)
    if relation_match is not None:
        return relation_match
    predicted_expr = _parse_simple_latex_expression(prediction)
    reference_expr = _parse_simple_latex_expression(reference)
    if predicted_expr is None or reference_expr is None:
        return False
    try:
        from sympy import simplify

        return simplify(predicted_expr - reference_expr) == 0
    except (ArithmeticError, TypeError, ValueError):
        return False


def _scalar_assignment_value(value: str) -> str | None:
    if value.count("=") != 1:
        return None
    left, right = value.split("=", maxsplit=1)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", left) and right:
        return right
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", right) and left:
        return left
    return None


def _interval_answer_parts(
    value: str,
) -> tuple[str, str, tuple[str, str]] | None:
    if len(value) < 5 or value[0] not in "([" or value[-1] not in ")]":
        return None
    parts = _top_level_math_parts(value[1:-1])
    if len(parts) != 2:
        return None
    return value[0], value[-1], (parts[0], parts[1])


def _top_level_union_parts(value: str) -> tuple[str, ...] | None:
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    found = False
    while index < len(value):
        char = value[index]
        if char in "{([":
            depth += 1
            index += 1
            continue
        if char in "})]":
            depth = max(0, depth - 1)
            index += 1
            continue
        token_width = 0
        if depth == 0 and value.startswith("\\cup", index):
            token_width = 4
        elif depth == 0 and char == "∪":
            token_width = 1
        if token_width:
            part = value[start:index]
            if not part:
                return None
            parts.append(part)
            index += token_width
            start = index
            found = True
            continue
        index += 1
    if not found:
        return None
    tail = value[start:]
    if not tail:
        return None
    parts.append(tail)
    return tuple(parts)


def _coordinate_tuple_entries(value: str) -> tuple[str, ...] | None:
    if len(value) < 3 or value[0] != "(" or value[-1] != ")":
        return None
    parts = _top_level_math_parts(value[1:-1])
    return parts if len(parts) > 1 else None


def _latex_matrix_entries(value: str) -> tuple[str, ...] | None:
    match = re.fullmatch(
        r"(?P<scale>.*?)\\begin\{(?P<env>p?matrix|bmatrix|matrix)\}"
        r"(?P<body>.*)\\end\{(?P=env)\}",
        value,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    scale = match.group("scale")
    body = match.group("body")
    rows = re.split(r"\\\\", body)
    entries: list[str] = []
    for row in rows:
        for part in row.split("&"):
            if not part:
                continue
            entries.append(f"({scale})*({part})" if scale else part)
    return tuple(entries)


def _symbolic_equation_match(prediction: str, reference: str) -> bool | None:
    if prediction.count("=") != 1 or reference.count("=") != 1:
        return None
    predicted_left, predicted_right = prediction.split("=", maxsplit=1)
    reference_left, reference_right = reference.split("=", maxsplit=1)
    predicted_left_expr = _parse_simple_latex_expression(predicted_left)
    predicted_right_expr = _parse_simple_latex_expression(predicted_right)
    reference_left_expr = _parse_simple_latex_expression(reference_left)
    reference_right_expr = _parse_simple_latex_expression(reference_right)
    if any(
        expr is None
        for expr in (
            predicted_left_expr,
            predicted_right_expr,
            reference_left_expr,
            reference_right_expr,
        )
    ):
        return False
    try:
        from sympy import simplify

        predicted_zero = simplify(predicted_left_expr - predicted_right_expr)
        reference_zero = simplify(reference_left_expr - reference_right_expr)
        if simplify(predicted_zero - reference_zero) == 0:
            return True
        if predicted_zero == 0 or reference_zero == 0:
            return False
        ratio = simplify(predicted_zero / reference_zero)
        return bool(ratio.is_number and ratio != 0)
    except (ArithmeticError, TypeError, ValueError):
        return False


def _parse_simple_latex_expression(value: str) -> Any | None:
    if not value or len(value) > 512:
        return None
    if any(token in value for token in ("\\begin", "\\end", "\\text", "=", "<", ">")):
        return None
    try:
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return None

    expression = value
    expression = _replace_latex_group_command(expression, "\\sqrt", "sqrt")
    expression = re.sub(r"\\sqrt([A-Za-z0-9]+)", r"sqrt(\1)", expression)
    expression = _replace_latex_fractions(expression)
    if expression is None:
        return None
    expression = expression.replace("\\pi", "pi")
    expression = expression.replace("\\cdot", "*").replace("\\times", "*")
    expression = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", expression)
    expression = expression.replace("{", "(").replace("}", ")")
    expression = expression.replace("\\", "")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/^().]+", expression):
        return None
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    try:
        return parse_expr(expression, transformations=transformations, evaluate=True)
    except (SyntaxError, TypeError, ValueError):
        return None


def _replace_latex_group_command(value: str, command: str, replacement: str) -> str:
    output = value
    while command in output:
        index = output.rfind(command)
        group_start = index + len(command)
        group = _latex_braced_group(output, group_start)
        if group is None:
            return output
        content, end = group
        output = output[:index] + f"{replacement}({content})" + output[end:]
    return output


def _replace_latex_fractions(value: str) -> str | None:
    output = value
    while "\\frac" in output:
        index = output.rfind("\\frac")
        numerator = _latex_argument(output, index + len("\\frac"))
        if numerator is None:
            return None
        numerator_text, numerator_end = numerator
        denominator = _latex_argument(output, numerator_end)
        if denominator is None:
            return None
        denominator_text, denominator_end = denominator
        replacement = f"(({numerator_text})/({denominator_text}))"
        output = output[:index] + replacement + output[denominator_end:]
    return output


def _latex_argument(value: str, start: int) -> tuple[str, int] | None:
    grouped = _latex_braced_group(value, start)
    if grouped is not None:
        return grouped
    if start >= len(value):
        return None
    return value[start], start + 1


def _latex_braced_group(value: str, start: int) -> tuple[str, int] | None:
    if start >= len(value) or value[start] != "{":
        return None
    depth = 0
    for index in range(start, len(value)):
        char = value[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    return None


def _qa_normalize(value: str) -> str:
    lowered = (
        value.casefold()
        .replace("_", " ")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    lowered = re.sub(r"(?<=\d),(?=\d)", "", lowered)
    no_punctuation = "".join(" " if char in string.punctuation else char for char in lowered)
    tokens = [token for token in no_punctuation.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _number_tokens(value: str) -> tuple[str, ...]:
    normalized = value.replace(",", "")
    return tuple(
        match.group(0).lstrip("+") for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", normalized)
    )


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
        # If a later observation has exactly the same visible value as one
        # that has already arrived, mentioning that value cannot establish a
        # causal leak.  This occurs naturally for refreshes and streams whose
        # next version repeats the previous payload.
        available_markers = {
            _normalized_text(marker)
            for available_id in available
            for marker in _evidence_markers(evidence_by_id[available_id].value)
            if _normalized_text(marker)
        }
        for evidence_id in evidence_order:
            if evidence_id in available:
                continue
            evidence = evidence_by_id[evidence_id]
            for marker in _evidence_markers(evidence.value):
                normalized_marker = _normalized_text(marker)
                if normalized_marker in baseline:
                    continue
                if normalized_marker in available_markers:
                    continue
                if any(
                    _visibility_contains_marker(available_marker, normalized_marker)
                    for available_marker in available_markers
                ):
                    continue
                if normalized_marker and _visibility_contains_marker(
                    frame_text, normalized_marker
                ):
                    reasons.append(f"phase {frame.phase!r} leaks future evidence {evidence_id!r}")
                    break
    return tuple(reasons)


def _visibility_contains_marker(frame_text: str, marker: str) -> bool:
    """Match numeric evidence as a token, not as a substring of another number."""
    numeric = marker.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
        return numeric.lstrip("+") in _number_tokens(frame_text)
    return marker in frame_text


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
    if isinstance(value, dict):
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        nested = tuple(
            child
            for item in value.values()
            for child in _evidence_markers(item)
        )
        return ((marker,) if len(marker) >= 4 else ()) + nested
    if isinstance(value, (list, tuple)):
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        nested = tuple(child for item in value for child in _evidence_markers(item))
        return ((marker,) if len(marker) >= 4 else ()) + nested
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

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from cid.contracts import FreshnessDemand
from cid.data import DISPLAY_UNKNOWN_MARKER, is_legacy_display_status
from cid.distill import (
    TeacherCellPlan,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherTask,
    dump_teacher_plans,
)
from cid.grounding import LinkRelation, ObjectKind
from cid.state import CellLifecycle, CognitiveRole
from cid.teacher_semantics import (
    TEACHER_SEMANTIC_TEXT_MAX_CHARS,
    TEACHER_SEMANTIC_TEXT_TARGET_CHARS,
)

TEACHER_SOFT_QUALITY_GUIDANCE = """TCT supervision quality rules:
- `semantic_text` represents compressed cognitive state, not evidence storage.
  Target <=144 characters; interactive teacher stages are rejected above 192 characters.
  Never paste a whole source sentence or paragraph into a cell. Rewrite evidence as
  task-relevant facts, e.g. `Ada — born: 1815; country: UK.`
- Prefer about 3-6 live cells for an ordinary task. Add a cell only for a distinct
  cognitive role.
- Evidence-bearing percept cells must ground salient objects with `anchors` and carry
  an `observes` link to the source that produced the arrived evidence. Reuse stable
  anchor IDs for the same object.
- A cell that requests external evidence must carry a typed `requests` link to the
  contracted source. Use `depends_on` for prerequisite cells when useful.
- Terminal conclusion cells should link back to supporting percept cells with
  `derived_from`. Use `conflicts` when visible evidence disagrees. Links encode the
  cognitive graph; empty `links` on all cells are not acceptable supervision.

Anchor example:
`{\"anchor_id\":\"entity:alice-example\",\"kind\":\"entity\",`
`\"value\":\"Alice Example\",\"confidence\":1.0}`
Link example:
`{\"relation\":\"observes\",\"target\":{\"kind\":\"source\",`
`\"identifier\":\"workspace_read\"},\"confidence\":1.0}`
"""

TEACHER_SEMANTIC_TEXT_PREFERRED_MAX_CHARS = TEACHER_SEMANTIC_TEXT_MAX_CHARS
TEACHER_PREFERRED_LIVE_CELLS = 6

TEACHER_DISPLAY_GUIDANCE = f"""Display supervision rules:
- `display` is the current user-visible answer draft. It must not narrate hidden reasoning,
  retrieval, tool progress, or generic status such as `pending`, `Reasoning`, `Planning`,
  `Retrieving`, `Gathering`, or `Evidence integrated`.
- Represent answer content that is not known yet with the exact marker
  `{DISPLAY_UNKNOWN_MARKER}`. Preserve any answer content already supported around that marker.
- A terminal stage must contain the fully resolved final answer and must not contain the unresolved
  marker.
"""


@dataclass(frozen=True, slots=True)
class TeacherWaveRequest:
    request_id: str
    task_id: str
    stage_index: int
    phase: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "stage_index": self.stage_index,
            "phase": self.phase,
            "prompt": self.prompt,
        }


@dataclass(frozen=True, slots=True)
class StageNeedChoice:
    evidence_id: str
    cell_id: str
    confidence: float = 1.0
    freshness: FreshnessDemand = FreshnessDemand.ONCE
    max_age_s: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StageNeedChoice:
        _reject_unknown_keys(
            raw,
            {"evidence_id", "cell_id", "confidence", "freshness", "max_age_s"},
            "stage need",
        )
        return cls(
            evidence_id=str(raw["evidence_id"]),
            cell_id=str(raw["cell_id"]),
            confidence=float(raw.get("confidence", 1.0)),
            freshness=FreshnessDemand(str(raw.get("freshness", FreshnessDemand.ONCE))),
            max_age_s=None if raw.get("max_age_s") is None else float(raw["max_age_s"]),
        )

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.cell_id:
            raise ValueError("stage need requires evidence_id and cell_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("stage need confidence must be in [0, 1]")
        if self.freshness is FreshnessDemand.MAX_AGE and self.max_age_s is None:
            raise ValueError("MAX_AGE stage need requires max_age_s")

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "evidence_id": self.evidence_id,
            "cell_id": self.cell_id,
            "confidence": self.confidence,
            "freshness": self.freshness.value,
        }
        if self.max_age_s is not None:
            raw["max_age_s"] = self.max_age_s
        return raw


@dataclass(frozen=True, slots=True)
class TeacherStageOutput:
    display: str
    cells: tuple[TeacherCellPlan, ...]
    needs: tuple[StageNeedChoice, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TeacherStageOutput:
        _reject_unknown_keys(raw, {"display", "cells", "needs"}, "teacher stage output")
        return cls(
            display=str(raw["display"]),
            cells=tuple(TeacherCellPlan.from_dict(item) for item in raw.get("cells", ())),
            needs=tuple(StageNeedChoice.from_dict(item) for item in raw.get("needs", ())),
        )

    def __post_init__(self) -> None:
        if not self.display.strip():
            raise ValueError("teacher stage output requires non-empty display")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("teacher stage output cell IDs must be unique")
        need_ids = tuple(need.evidence_id for need in self.needs)
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("teacher stage output may request each evidence item at most once")

    def to_dict(self) -> dict[str, Any]:
        return {
            "display": self.display,
            "cells": [cell.to_dict() for cell in self.cells],
            "needs": [need.to_dict() for need in self.needs],
        }


@dataclass(frozen=True, slots=True)
class TeacherStageState:
    request_id: str
    task_id: str
    stage_index: int
    phase: str
    output: TeacherStageOutput

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "stage_index": self.stage_index,
            "phase": self.phase,
            "output": self.output.to_dict(),
        }


def export_teacher_wave(
    jobs_path: str | Path,
    state_path: str | Path,
    output_path: str | Path,
    *,
    max_requests: int | None = None,
) -> dict[str, int]:
    jobs = _load_jobs(jobs_path)
    state = load_teacher_wave_state(state_path)
    requests: list[TeacherWaveRequest] = []
    complete_tasks = 0
    for job in jobs:
        task_id = str(job["task_id"])
        stages = list(job["stages"])
        completed = {
            stage_index
            for (record_task_id, stage_index), _ in state.items()
            if record_task_id == task_id
        }
        if len(completed) == len(stages):
            complete_tasks += 1
            continue
        stage_index = 0
        while stage_index in completed:
            stage_index += 1
        if stage_index > 0 and stage_index - 1 not in completed:
            raise ValueError(f"teacher state for {task_id} contains a non-contiguous stage prefix")
        stage = stages[stage_index]
        previous = None if stage_index == 0 else state[(task_id, stage_index - 1)].output
        request = TeacherWaveRequest(
            request_id=_request_id(task_id, stage_index, str(stage["phase"])),
            task_id=task_id,
            stage_index=stage_index,
            phase=str(stage["phase"]),
            prompt=_build_stage_prompt(job, stage_index, previous),
        )
        requests.append(request)
        if max_requests is not None and len(requests) >= max_requests:
            break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(
                json.dumps(request.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return {
        "jobs": len(jobs),
        "complete_tasks": complete_tasks,
        "exported_requests": len(requests),
    }


def import_teacher_wave(
    jobs_path: str | Path,
    requests_path: str | Path,
    responses_path: str | Path,
    state_path: str | Path,
    *,
    rejects_path: str | Path | None = None,
) -> dict[str, int]:
    jobs = {str(job["task_id"]): job for job in _load_jobs(jobs_path)}
    requests = {
        str(item["request_id"]): item
        for item in _load_json_objects(requests_path, "teacher request")
    }
    state = load_teacher_wave_state(state_path)
    imported = 0
    unchanged = 0
    rejected: list[dict[str, Any]] = []
    for raw in _load_json_objects(responses_path, "teacher response"):
        try:
            status = _import_one_teacher_response(raw, requests, jobs, state)
        except (KeyError, TypeError, ValueError) as exc:
            if rejects_path is None:
                raise
            rejected.append(
                {
                    "request_id": raw.get("request_id"),
                    "error": str(exc),
                    "response": raw,
                }
            )
            continue
        if status == "imported":
            imported += 1
        else:
            unchanged += 1

    dump_teacher_wave_state(state.values(), state_path)
    if rejects_path is not None:
        reject_output = Path(rejects_path)
        reject_output.parent.mkdir(parents=True, exist_ok=True)
        with reject_output.open("w", encoding="utf-8") as handle:
            for item in rejected:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "imported": imported,
        "unchanged": unchanged,
        "rejected": len(rejected),
        "state_records": len(state),
    }


def teacher_wave_status(
    jobs_path: str | Path,
    state_path: str | Path,
) -> dict[str, Any]:
    jobs = _load_jobs(jobs_path)
    state = load_teacher_wave_state(state_path)
    total_stages = sum(len(job["stages"]) for job in jobs)
    completed_stages = 0
    complete_tasks = 0
    next_phases: dict[str, int] = {}
    for job in jobs:
        task_id = str(job["task_id"])
        stages = list(job["stages"])
        prefix = 0
        while (task_id, prefix) in state:
            prefix += 1
        completed_stages += prefix
        extras = [
            stage_index
            for record_task_id, stage_index in state
            if record_task_id == task_id and stage_index >= prefix
        ]
        if extras:
            raise ValueError(f"teacher state for {task_id} is not a contiguous stage prefix")
        if prefix == len(stages):
            complete_tasks += 1
        else:
            phase = str(stages[prefix]["phase"])
            next_phases[phase] = next_phases.get(phase, 0) + 1
    return {
        "jobs": len(jobs),
        "total_stages": total_stages,
        "completed_stages": completed_stages,
        "complete_tasks": complete_tasks,
        "incomplete_tasks": len(jobs) - complete_tasks,
        "next_phase_counts": dict(sorted(next_phases.items())),
    }


def _import_one_teacher_response(
    raw: Mapping[str, Any],
    requests: Mapping[str, Mapping[str, Any]],
    jobs: Mapping[str, Mapping[str, Any]],
    state: dict[tuple[str, int], TeacherStageState],
) -> str:
    _reject_unknown_keys(raw, {"request_id", "output"}, "teacher response")
    request_id = str(raw["request_id"])
    if request_id not in requests:
        raise ValueError(f"teacher response references unknown request_id {request_id!r}")
    request = requests[request_id]
    task_id = str(request["task_id"])
    stage_index = int(request["stage_index"])
    job = jobs.get(task_id)
    if job is None:
        raise ValueError(f"teacher request references unknown task {task_id!r}")
    stages = list(job["stages"])
    if not 0 <= stage_index < len(stages):
        raise ValueError("teacher request stage index is out of range")
    phase = str(stages[stage_index]["phase"])
    expected_request_id = _request_id(task_id, stage_index, phase)
    if request_id != expected_request_id:
        raise ValueError("teacher request ID does not match task/stage identity")
    previous = None if stage_index == 0 else state.get((task_id, stage_index - 1))
    if stage_index > 0 and previous is None:
        raise ValueError(f"cannot import {task_id} stage {stage_index} before its predecessor")
    output = TeacherStageOutput.from_dict(raw["output"])
    _validate_stage_output(
        stages[stage_index],
        output,
        None if previous is None else previous.output,
    )
    record = TeacherStageState(
        request_id=request_id,
        task_id=task_id,
        stage_index=stage_index,
        phase=phase,
        output=output,
    )
    key = (task_id, stage_index)
    existing = state.get(key)
    if existing is not None:
        if existing != record:
            raise ValueError(f"teacher state already contains a different output for {key}")
        return "unchanged"
    state[key] = record
    return "imported"


def finalize_teacher_wave(
    tasks: tuple[TeacherTask, ...],
    jobs_path: str | Path,
    state_path: str | Path,
    output_path: str | Path,
) -> tuple[TeacherPlan, ...]:
    task_by_id = {task.task_id: task for task in tasks}
    jobs = _load_jobs(jobs_path)
    state = load_teacher_wave_state(state_path)
    plans: list[TeacherPlan] = []
    for job in jobs:
        task_id = str(job["task_id"])
        task = task_by_id.get(task_id)
        if task is None:
            raise ValueError(f"causal teacher job references unknown task {task_id!r}")
        stages = list(job["stages"])
        records: list[TeacherStageState] = []
        for stage_index in range(len(stages)):
            record = state.get((task_id, stage_index))
            if record is None:
                raise ValueError(f"teacher task {task_id!r} is incomplete at stage {stage_index}")
            records.append(record)
        frames = tuple(
            TeacherFrame(
                phase=record.phase,
                display=record.output.display,
                cells=record.output.cells,
            )
            for record in records
        )
        needs: list[TeacherNeed] = []
        for record, stage in zip(records, stages, strict=True):
            contracts = {
                str(item["evidence_id"]): item for item in stage.get("available_evidence", ())
            }
            for choice in record.output.needs:
                contract = contracts[choice.evidence_id]
                needs.append(
                    TeacherNeed(
                        need_id=f"need:{choice.evidence_id}",
                        cell_id=choice.cell_id,
                        evidence_id=choice.evidence_id,
                        phase=record.phase,
                        source=str(contract["source"]),
                        arguments=dict(contract.get("arguments", {})),
                        confidence=choice.confidence,
                        freshness=choice.freshness,
                        max_age_s=choice.max_age_s,
                    )
                )
        plans.append(
            TeacherPlan(
                task_id=task_id,
                final_answer=records[-1].output.display.strip(),
                frames=frames,
                needs=tuple(needs),
            )
        )
    result = tuple(plans)
    dump_teacher_plans(result, output_path)
    return result


def load_teacher_wave_state(path: str | Path) -> dict[tuple[str, int], TeacherStageState]:
    source = Path(path)
    if not source.exists():
        return {}
    state: dict[tuple[str, int], TeacherStageState] = {}
    for raw in _load_json_objects(source, "teacher state"):
        _reject_unknown_keys(
            raw,
            {"request_id", "task_id", "stage_index", "phase", "output"},
            "teacher state",
        )
        record = TeacherStageState(
            request_id=str(raw["request_id"]),
            task_id=str(raw["task_id"]),
            stage_index=int(raw["stage_index"]),
            phase=str(raw["phase"]),
            output=TeacherStageOutput.from_dict(raw["output"]),
        )
        key = (record.task_id, record.stage_index)
        if key in state:
            raise ValueError(f"duplicate teacher state record for {key}")
        state[key] = record
    return state


def dump_teacher_wave_state(records: Iterable[TeacherStageState], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: (item.task_id, item.stage_index))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in ordered:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _build_stage_prompt(
    job: Mapping[str, Any],
    stage_index: int,
    previous: TeacherStageOutput | None,
) -> str:
    stages = list(job["stages"])
    stage = stages[stage_index]
    task_json = json.dumps(job["task"], ensure_ascii=False, sort_keys=True, indent=2)
    previous_json = (
        "null"
        if previous is None
        else json.dumps(previous.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    )
    arrived_json = json.dumps(
        stage.get("arrived_evidence"), ensure_ascii=False, sort_keys=True, indent=2
    )
    available_json = json.dumps(
        stage.get("available_evidence", ()), ensure_ascii=False, sort_keys=True, indent=2
    )
    terminal = bool(stage.get("terminal", False))
    terminal_rule = (
        "This is the terminal stage. Emit no new needs; `display` must be the concise final "
        "answer, "
        "and at least one cell must carry a positive `conclusion` role."
        if terminal
        else "This is not terminal. Do not claim convergence while required evidence remains."
    )
    return f"""You are producing one causal semantic-supervision stage for Continuous Interaction
Diffusion (CID).
Return exactly one JSON object and no prose. Do not write private chain-of-thought. Cell
`semantic_text` is a concise cognitive-state summary, never a reasoning transcript.

You physically cannot see future evidence in this request. Use only TASK, PREVIOUS_STATE, and
ARRIVED_EVIDENCE below. Preserve every previous cell ID in the new `cells` list; retire obsolete
cells explicitly instead of deleting them. Create new cells only when they serve a distinct role.
Allowed roles: hypothesis, information_need, percept, plan, constraint, conclusion.

Every AVAILABLE_EVIDENCE_CONTRACT is a gold external information need that becomes executable at
this stage. Emit exactly one `needs` entry for each contract, attached to a current cell with a
positive `information_need` role. Do not invent needs for unavailable evidence. Source name and
arguments are fixed by the contract and therefore are omitted from your output. If a contract has
`freshness_hint`, copy that value into the need's `freshness` field; `always` means the binding must
remain live for later refreshes or stream chunks.

{TEACHER_SOFT_QUALITY_GUIDANCE}

{TEACHER_DISPLAY_GUIDANCE}

Output schema:
{{
  "display":"non-empty current display state",
  "cells":[
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
  ],
  "needs":[
    {{"evidence_id":"...","cell_id":"...","confidence":1.0,"freshness":"once"}}
  ]
}}

{terminal_rule}

TASK:
{task_json}

PHASE: {stage["phase"]}

PREVIOUS_STATE:
{previous_json}

ARRIVED_EVIDENCE:
{arrived_json}

AVAILABLE_EVIDENCE_CONTRACTS:
{available_json}
"""


def _validate_stage_output(
    stage: Mapping[str, Any],
    output: TeacherStageOutput,
    previous: TeacherStageOutput | None,
) -> None:
    _validate_stage_display(stage, output)
    current_ids = {cell.cell_id for cell in output.cells}
    if previous is not None:
        previous_ids = {cell.cell_id for cell in previous.cells}
        missing = sorted(previous_ids - current_ids)
        if missing:
            raise ValueError(
                f"teacher stage dropped existing cells instead of retiring them: {missing}"
            )
    cell_by_id = {cell.cell_id: cell for cell in output.cells}
    available = {str(item["evidence_id"]): item for item in stage.get("available_evidence", ())}
    requested = {need.evidence_id for need in output.needs}
    if requested != set(available):
        raise ValueError(
            "teacher stage must request every and only currently available evidence contract: "
            f"requested={sorted(requested)} available={sorted(available)}"
        )
    for need in output.needs:
        cell = cell_by_id.get(need.cell_id)
        if cell is None:
            raise ValueError(f"teacher stage need references missing cell {need.cell_id!r}")
        if cell.roles.get(CognitiveRole.INFORMATION_NEED, 0.0) <= 0.0:
            raise ValueError("teacher stage need cell must carry a positive information_need role")
        freshness_hint = available[need.evidence_id].get("freshness_hint")
        if freshness_hint is not None and need.freshness.value != str(freshness_hint):
            raise ValueError(
                f"teacher stage need {need.evidence_id} must use freshness={freshness_hint!r}"
            )
    if bool(stage.get("terminal", False)):
        if output.needs:
            raise ValueError("terminal teacher stage cannot emit new needs")
        if not any(cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0 for cell in output.cells):
            raise ValueError("terminal teacher stage requires a conclusion-role cell")
    retired = {cell.cell_id for cell in output.cells if cell.lifecycle is CellLifecycle.RETIRED}
    if any(need.cell_id in retired for need in output.needs):
        raise ValueError("teacher stage cannot attach a new need to a retired cell")


def validate_teacher_stage_tct_quality(
    stage: Mapping[str, Any],
    output: TeacherStageOutput,
) -> None:
    """Hard TCT quality guard used by the interactive teacher adapter."""

    _validate_stage_display(stage, output)

    for cell in output.cells:
        if len(cell.semantic_text) > TEACHER_SEMANTIC_TEXT_MAX_CHARS:
            raise ValueError(
                "teacher cell semantic_text exceeds interactive TCT limit "
                f"({len(cell.semantic_text)} > {TEACHER_SEMANTIC_TEXT_MAX_CHARS})"
            )

    live_cells = [cell for cell in output.cells if cell.lifecycle is not CellLifecycle.RETIRED]
    cell_by_id = {cell.cell_id: cell for cell in live_cells}
    available = {str(item["evidence_id"]): item for item in stage.get("available_evidence", ())}
    for need in output.needs:
        contract = available.get(need.evidence_id)
        if contract is None:
            continue
        source = str(contract["source"])
        cell = cell_by_id.get(need.cell_id)
        if cell is None or not _has_source_link(cell, LinkRelation.REQUESTS, source):
            raise ValueError(
                f"teacher need cell {need.cell_id!r} must carry requests link to source {source!r}"
            )

    arrived = stage.get("arrived_evidence")
    if arrived is not None:
        source = str(arrived.get("source", ""))
        percepts = [cell for cell in live_cells if cell.roles.get(CognitiveRole.PERCEPT, 0.0) > 0.0]
        if not percepts:
            raise ValueError("arrived evidence must be represented by a percept cell")
        if not any(cell.anchors for cell in percepts):
            raise ValueError("arrived evidence percept must carry at least one grounding anchor")
        if source and not any(
            _has_source_link(cell, LinkRelation.OBSERVES, source) for cell in percepts
        ):
            raise ValueError(
                f"arrived evidence percept must carry observes link to source {source!r}"
            )
        raw_text = _arrived_evidence_text(arrived.get("value"))
        if raw_text:
            normalized_raw = " ".join(raw_text.casefold().split())
            for cell in percepts:
                normalized_cell = " ".join(cell.semantic_text.casefold().split())
                if len(normalized_cell) >= 96 and normalized_cell in normalized_raw:
                    raise ValueError(
                        "percept semantic_text appears to copy arrived evidence verbatim; "
                        "rewrite it as compact task-relevant state"
                    )

    if bool(stage.get("terminal", False)):
        percept_ids = {
            cell.cell_id for cell in live_cells if cell.roles.get(CognitiveRole.PERCEPT, 0.0) > 0.0
        }
        conclusions = [
            cell for cell in live_cells if cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0
        ]
        if percept_ids and not any(
            link.relation is LinkRelation.DERIVED_FROM
            and link.target.kind is ObjectKind.CELL
            and link.target.identifier in percept_ids
            for cell in conclusions
            for link in cell.links
        ):
            raise ValueError(
                "terminal conclusion must carry a derived_from link to visible percept evidence"
            )


def _validate_stage_display(stage: Mapping[str, Any], output: TeacherStageOutput) -> None:
    if is_legacy_display_status(output.display):
        raise ValueError(
            "teacher stage display narrates process status instead of the current answer draft"
        )
    if bool(stage.get("terminal", False)) and DISPLAY_UNKNOWN_MARKER in output.display:
        raise ValueError("terminal teacher stage display cannot contain unresolved answer content")


def _has_source_link(cell: TeacherCellPlan, relation: LinkRelation, source: str) -> bool:
    return any(
        link.relation is relation
        and link.target.kind is ObjectKind.SOURCE
        and link.target.identifier == source
        for link in cell.links
    )


def _arrived_evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        sentences = value.get("sentences")
        if isinstance(sentences, list):
            return " ".join(str(item) for item in sentences)
        return str(value.get("text") or value.get("value") or "")
    if isinstance(value, list):
        return " ".join(
            str(item.get("title") or item.get("resource_id") or "")
            for item in value
            if isinstance(item, dict)
        )
    return str(value or "")


def teacher_stage_soft_warning_codes(
    stage: Mapping[str, Any],
    output: TeacherStageOutput,
) -> tuple[str, ...]:
    """Return non-blocking quality warnings for teacher supervision.

    These warnings intentionally do not participate in validation. They make compactness and
    grounding quality observable without turning stylistic preferences into brittle data gates.
    """

    warnings: list[str] = []
    lengths = [len(cell.semantic_text) for cell in output.cells]
    if any(length > TEACHER_SEMANTIC_TEXT_TARGET_CHARS for length in lengths):
        warnings.append("semantic_text_over_target")
    if any(length > TEACHER_SEMANTIC_TEXT_PREFERRED_MAX_CHARS for length in lengths):
        warnings.append("semantic_text_over_preferred_max")

    live_cells = [cell for cell in output.cells if cell.lifecycle is not CellLifecycle.RETIRED]
    if len(live_cells) > TEACHER_PREFERRED_LIVE_CELLS:
        warnings.append("live_cells_over_preferred_count")

    if stage.get("arrived_evidence") is not None:
        evidence_cells = [
            cell
            for cell in live_cells
            if cell.roles.get(CognitiveRole.PERCEPT, 0.0) > 0.0
            or cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0
        ]
        if evidence_cells and not any(cell.anchors for cell in evidence_cells):
            warnings.append("arrived_evidence_without_anchor")
        if evidence_cells and not any(cell.links for cell in evidence_cells):
            warnings.append("arrived_evidence_without_link")
    return tuple(warnings)


def _request_id(task_id: str, stage_index: int, phase: str) -> str:
    payload = f"teacher-wave-v1|{task_id}|{stage_index}|{phase}"
    return "tw-" + sha256(payload.encode()).hexdigest()[:24]


def _load_jobs(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    jobs = _load_json_objects(path, "causal teacher job")
    task_ids = [str(job["task_id"]) for job in jobs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("causal teacher job task IDs must be unique")
    return jobs


def _load_json_objects(path: str | Path, label: str) -> tuple[Mapping[str, Any], ...]:
    source = Path(path)
    items: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {label} at line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"invalid {label} at line {line_number}: record must be an object")
            items.append(raw)
    return tuple(items)


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {unknown}")

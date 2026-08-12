from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.causal_distill import build_causal_teacher_job, dump_causal_teacher_jobs
from cid.distill import (
    TeacherEvidence,
    TeacherPlan,
    TeacherTask,
    dump_teacher_requests,
    dump_teacher_tasks,
)

CORRECTION_FAMILIES = (
    "stale_registry",
    "entity_disambiguation",
    "unit_assumption",
    "boundary_classification",
    "numeric_estimate",
    "version_default",
    "category_exception",
    "operation_order",
    "source_priority",
    "temporal_status",
)

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True, slots=True)
class CorrectionTrainingConfig:
    count_per_family: int = 1000
    seed: int = 20260812

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")

    @property
    def total_tasks(self) -> int:
        return self.count_per_family * len(CORRECTION_FAMILIES)


def build_correction_training(
    tasks_output: str | Path,
    requests_output: str | Path,
    causal_jobs_output: str | Path,
    manifest_output: str | Path,
    config: CorrectionTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or CorrectionTrainingConfig()
    tasks = generate_correction_tasks(config)

    task_path = Path(tasks_output)
    request_path = Path(requests_output)
    jobs_path = Path(causal_jobs_output)
    manifest_path = Path(manifest_output)
    for path in (task_path, request_path, jobs_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    dump_teacher_tasks(tasks, task_path)
    dump_teacher_requests(tasks, request_path)
    dump_causal_teacher_jobs(tasks, jobs_path)

    family_counts = Counter(str(task.metadata["family"]) for task in tasks)
    stage_counts = Counter(len(build_causal_teacher_job(task).stages) for task in tasks)
    manifest = {
        "format_version": 1,
        "name": "speculative-local-correction-v1",
        "seed": config.seed,
        "tasks": len(tasks),
        "count_per_family": config.count_per_family,
        "family_counts": dict(sorted(family_counts.items())),
        "causal_stage_histogram": {
            str(stages): count for stages, count in sorted(stage_counts.items())
        },
        "dependency_depth": 2,
        "training_mode": "tool_required",
        "capability": "wrong_hypothesis_then_local_evidence_correction",
        "tasks_sha256": _file_sha256(task_path),
        "requests_sha256": _file_sha256(request_path),
        "causal_jobs_sha256": _file_sha256(jobs_path),
        "tasks_path": str(task_path),
        "causal_jobs": str(jobs_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_correction_tasks(
    config: CorrectionTrainingConfig | None = None,
) -> tuple[TeacherTask, ...]:
    config = config or CorrectionTrainingConfig()
    rng = random.Random(config.seed)
    generators = {
        "stale_registry": _stale_registry,
        "entity_disambiguation": _entity_disambiguation,
        "unit_assumption": _unit_assumption,
        "boundary_classification": _boundary_classification,
        "numeric_estimate": _numeric_estimate,
        "version_default": _version_default,
        "category_exception": _category_exception,
        "operation_order": _operation_order,
        "source_priority": _source_priority,
        "temporal_status": _temporal_status,
    }
    tasks: list[TeacherTask] = []
    for family in CORRECTION_FAMILIES:
        generator = generators[family]
        for index in range(config.count_per_family):
            tasks.append(generator(rng, index))
    tasks.sort(key=lambda item: item.task_id)
    return tuple(tasks)


def correction_teacher_response(request: Mapping[str, Any]) -> dict[str, Any]:
    """Produce one causally valid semantic teacher stage for a correction task.

    The first stage deliberately carries a plausible but wrong hypothesis. The first observation
    contradicts and reopens only the hypothesis and its dependent answer; a second independent
    observation stabilizes the corrected state. Unrelated context/scope cells are copied unchanged.
    """

    task = dict(request["task"])
    metadata = dict(task.get("metadata", {}))
    if metadata.get("capability") != "speculative_local_correction":
        raise ValueError("correction teacher received a non-correction task")

    provisional = str(metadata["provisional_guess"])
    stable_context = str(metadata["stable_context"])
    scope_text = str(metadata["scope_text"])
    previous = request.get("previous_state")
    arrived = request.get("arrived_evidence")
    contracts = [dict(item) for item in request.get("available_evidence_contracts", ())]
    terminal = bool(request.get("terminal", False))
    task_id = str(request["task_id"])

    if previous is None:
        if len(contracts) != 1 or str(contracts[0]["evidence_id"]) != "correction":
            raise ValueError("initial correction stage requires exactly the correction contract")
        contract = contracts[0]
        source = str(contract["source"])
        context = _cell(
            "context",
            stable_context,
            {"constraint": 0.75, "percept": 0.35},
            uncertainty=0.08,
            noise=0.03,
            lifecycle="stable",
            anchors=[_anchor(_contract_key(contract), f"{task_id}|context")],
            links=[],
        )
        scope = _cell(
            "scope",
            scope_text,
            {"constraint": 1.0},
            uncertainty=0.04,
            noise=0.02,
            lifecycle="stable",
            anchors=[],
            links=[_link("depends_on", "cell", "context")],
        )
        hypothesis = _cell(
            "hypothesis",
            f"Provisional hypothesis: {provisional}.",
            {"hypothesis": 1.0},
            uncertainty=0.38,
            noise=0.12,
            lifecycle="active",
            anchors=[_anchor(provisional, f"{task_id}|provisional")],
            links=[_link("derived_from", "cell", "context")],
        )
        verify = _cell(
            "verify",
            "Need authoritative evidence before accepting the provisional hypothesis.",
            {"information_need": 1.0, "plan": 0.25},
            uncertainty=0.88,
            noise=0.08,
            lifecycle="active",
            anchors=[],
            links=[
                _link("requests", "source", source),
                _link("depends_on", "cell", "scope"),
            ],
        )
        answer = _cell(
            "answer",
            f"Provisional answer: {provisional}; verification pending.",
            {"conclusion": 0.55, "hypothesis": 0.45},
            uncertainty=0.42,
            noise=0.12,
            lifecycle="active",
            anchors=[_anchor(provisional, f"{task_id}|answer-provisional")],
            links=[_link("derived_from", "cell", "hypothesis")],
        )
        return {
            "display": f"Provisional: {provisional}; verifying.",
            "cells": [context, scope, hypothesis, verify, answer],
            "needs": [
                {
                    "evidence_id": "correction",
                    "cell_id": "verify",
                    "confidence": 1.0,
                    "freshness": str(contract.get("freshness_hint", "once")),
                }
            ],
        }

    cells = [dict(item) for item in previous["cells"]]
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    for stable_id in ("context", "scope"):
        if stable_id not in by_id:
            raise ValueError(f"previous correction state lost stable cell {stable_id}")

    if arrived is None:
        raise ValueError("non-initial correction stage requires arrived evidence")
    evidence_id = str(arrived["evidence_id"])
    correct = str(arrived.get("value"))

    if evidence_id == "correction":
        if terminal:
            raise ValueError("correction observation must be followed by confirmation")
        if len(contracts) != 1 or str(contracts[0]["evidence_id"]) != "confirmation":
            raise ValueError("correction stage requires exactly the confirmation contract")
        contract = contracts[0]
        source = str(arrived["source"])
        confirmation_source = str(contract["source"])
        by_id["verify"].update(
            _cell_fields(
                f"Authoritative evidence reports {correct}.",
                {"percept": 1.0},
                uncertainty=0.03,
                noise=0.02,
                lifecycle="stable",
                anchors=[_anchor(correct, f"{task_id}|correction")],
                links=[_link("observes", "source", source)],
            )
        )
        by_id["hypothesis"].update(
            _cell_fields(
                f"Evidence contradicts {provisional}; corrected candidate is {correct}.",
                {"hypothesis": 0.55, "percept": 0.65},
                uncertainty=0.24,
                noise=0.85,
                lifecycle="active",
                anchors=[
                    _anchor(provisional, f"{task_id}|provisional"),
                    _anchor(correct, f"{task_id}|correction"),
                ],
                links=[
                    _link("conflicts", "cell", "verify"),
                    _link("derived_from", "cell", "verify"),
                ],
            )
        )
        by_id["answer"].update(
            _cell_fields(
                f"Revised answer candidate: {correct}; independent confirmation pending.",
                {"conclusion": 0.65, "hypothesis": 0.25},
                uncertainty=0.22,
                noise=0.85,
                lifecycle="active",
                anchors=[_anchor(correct, f"{task_id}|answer-corrected")],
                links=[
                    _link("derived_from", "cell", "hypothesis"),
                    _link("derived_from", "cell", "verify"),
                ],
            )
        )
        confirm = _cell(
            "confirm",
            "Need independent confirmation of the corrected candidate.",
            {"information_need": 1.0, "plan": 0.25},
            uncertainty=0.72,
            noise=0.08,
            lifecycle="active",
            anchors=[_anchor(correct, f"{task_id}|confirm-target")],
            links=[
                _link("requests", "source", confirmation_source),
                _link("depends_on", "cell", "hypothesis"),
            ],
        )
        cells.append(confirm)
        return {
            "display": f"Evidence contradicts {provisional}; revising to {correct} and confirming.",
            "cells": cells,
            "needs": [
                {
                    "evidence_id": "confirmation",
                    "cell_id": "confirm",
                    "confidence": 1.0,
                    "freshness": str(contract.get("freshness_hint", "once")),
                }
            ],
        }

    if evidence_id != "confirmation" or not terminal or contracts:
        raise ValueError(
            "final correction stage must be terminal confirmation with no new contracts"
        )
    if "confirm" not in by_id:
        raise ValueError("confirmation stage is missing its need cell")

    confirmation_source = str(arrived["source"])
    by_id["confirm"].update(
        _cell_fields(
            f"Independent confirmation also reports {correct}.",
            {"percept": 1.0},
            uncertainty=0.02,
            noise=0.01,
            lifecycle="stable",
            anchors=[_anchor(correct, f"{task_id}|confirmation")],
            links=[_link("observes", "source", confirmation_source)],
        )
    )
    by_id["hypothesis"].update(
        _cell_fields(
            f"Corrected hypothesis confirmed: {correct}.",
            {"hypothesis": 0.3, "percept": 0.7},
            uncertainty=0.03,
            noise=0.02,
            lifecycle="stable",
            anchors=[_anchor(correct, f"{task_id}|confirmed")],
            links=[
                _link("derived_from", "cell", "verify"),
                _link("derived_from", "cell", "confirm"),
            ],
        )
    )
    by_id["answer"].update(
        _cell_fields(
            f"Final verified answer: {correct}.",
            {"conclusion": 1.0},
            uncertainty=0.01,
            noise=0.01,
            lifecycle="stable",
            anchors=[_anchor(correct, f"{task_id}|final-answer")],
            links=[
                _link("derived_from", "cell", "hypothesis"),
                _link("derived_from", "cell", "confirm"),
            ],
        )
    )
    return {"display": correct, "cells": cells, "needs": []}


def audit_correction_plans(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    plan_by_id = {plan.task_id: plan for plan in plans}
    failures: list[dict[str, Any]] = []
    max_semantic_chars = 0
    local_reopens = 0
    local_stabilizations = 0
    anchored_hypotheses = 0
    linked_hypotheses = 0

    for task_id, task in task_by_id.items():
        plan = plan_by_id.get(task_id)
        reasons: list[str] = []
        if plan is None:
            failures.append({"task_id": task_id, "reasons": ["missing plan"]})
            continue
        if plan.final_answer.strip() != str(task.reference_answer).strip():
            reasons.append("final answer does not match reference")
        frame_by_phase = {frame.phase: frame for frame in plan.frames}
        required = ("initial", "after:correction", "after:confirmation")
        if any(phase not in frame_by_phase for phase in required):
            reasons.append("missing required correction phases")
        else:
            initial, correction, confirmation = (frame_by_phase[phase] for phase in required)
            initial_by = {cell.cell_id: cell for cell in initial.cells}
            correction_by = {cell.cell_id: cell for cell in correction.cells}
            confirmation_by = {cell.cell_id: cell for cell in confirmation.cells}
            provisional = str(task.metadata["provisional_guess"])
            correct = str(task.reference_answer)
            for stable_id in ("context", "scope"):
                if not (
                    stable_id in initial_by
                    and stable_id in correction_by
                    and stable_id in confirmation_by
                    and initial_by[stable_id]
                    == correction_by[stable_id]
                    == confirmation_by[stable_id]
                ):
                    reasons.append(f"stable cell {stable_id} changed across local correction")
            hypothesis = initial_by.get("hypothesis")
            corrected = correction_by.get("hypothesis")
            confirmed = confirmation_by.get("hypothesis")
            if not hypothesis or not corrected or not confirmed:
                reasons.append("hypothesis cell is not preserved")
            else:
                if not any(
                    role.value == "hypothesis" and weight > 0.0
                    for role, weight in hypothesis.roles.items()
                ):
                    reasons.append("initial hypothesis has no positive hypothesis role")
                if provisional not in hypothesis.semantic_text:
                    reasons.append("initial hypothesis does not contain provisional guess")
                if (
                    corrected.semantic_text == hypothesis.semantic_text
                    or correct not in corrected.semantic_text
                ):
                    reasons.append("contradiction does not revise hypothesis to observed candidate")
                if not corrected.noise > hypothesis.noise + 0.2:
                    reasons.append("corrective evidence does not reopen hypothesis")
                else:
                    local_reopens += 1
                if not confirmed.noise < corrected.noise - 0.2:
                    reasons.append("confirmation does not stabilize corrected hypothesis")
                else:
                    local_stabilizations += 1
                if hypothesis.anchors and corrected.anchors and confirmed.anchors:
                    anchored_hypotheses += 1
                else:
                    reasons.append("hypothesis grounding anchors are incomplete")
                if hypothesis.links and corrected.links and confirmed.links:
                    linked_hypotheses += 1
                else:
                    reasons.append("hypothesis cognitive links are incomplete")
            initial_answer = initial_by.get("answer")
            corrected_answer = correction_by.get("answer")
            final_answer = confirmation_by.get("answer")
            if not initial_answer or not corrected_answer or not final_answer:
                reasons.append("dependent answer cell is not preserved")
            else:
                if provisional not in initial_answer.semantic_text:
                    reasons.append("provisional answer does not carry the wrong guess")
                if (
                    correct not in corrected_answer.semantic_text
                    or correct not in final_answer.semantic_text
                ):
                    reasons.append("dependent answer is not locally revised")
                if not corrected_answer.noise > initial_answer.noise + 0.2:
                    reasons.append("dependent answer is not reopened with hypothesis")
                else:
                    local_reopens += 1
                if not final_answer.noise < corrected_answer.noise - 0.2:
                    reasons.append("dependent answer is not stabilized after confirmation")
                else:
                    local_stabilizations += 1
            if confirmation.display.strip() != correct:
                reasons.append("terminal display is not the verified answer")

        for frame in plan.frames:
            for cell in frame.cells:
                max_semantic_chars = max(max_semantic_chars, len(cell.semantic_text))
        if reasons:
            failures.append({"task_id": task_id, "reasons": reasons})

    extra_plans = sorted(set(plan_by_id) - set(task_by_id))
    for task_id in extra_plans:
        failures.append({"task_id": task_id, "reasons": ["plan has no task"]})

    accepted = len(task_by_id) - sum(1 for item in failures if item["task_id"] in task_by_id)
    return {
        "tasks": len(task_by_id),
        "plans": len(plan_by_id),
        "accepted": accepted,
        "rejected": len(failures),
        "local_reopen_cells": local_reopens,
        "local_stabilization_cells": local_stabilizations,
        "anchored_hypothesis_tasks": anchored_hypotheses,
        "linked_hypothesis_tasks": linked_hypotheses,
        "max_semantic_text_chars": max_semantic_chars,
        "failures": failures[:50],
    }


def _task(
    *,
    family: str,
    index: int,
    prompt: str,
    provisional: str | int,
    correct: str | int,
    key: str,
    stable_context: str,
    scope_text: str,
    correction_source: tuple[str, str],
    confirmation_source: tuple[str, str],
    arguments: Mapping[str, Any],
) -> TeacherTask:
    if str(provisional) == str(correct):
        raise ValueError("correction task requires a genuinely wrong provisional guess")
    first_name, first_description = correction_source
    second_name, second_description = confirmation_source
    return TeacherTask(
        task_id=f"correction-{family}-{index:06d}",
        prompt=prompt,
        protected_facts={"output_rule": "return only the verified final value"},
        source_descriptors=(
            _descriptor(first_name, first_description, tuple(arguments)),
            _descriptor(second_name, second_description, tuple(arguments)),
        ),
        evidence=(
            TeacherEvidence(
                evidence_id="correction",
                source=first_name,
                value=correct,
                arguments=dict(arguments),
                provenance="cid.correction_training.v1",
            ),
            TeacherEvidence(
                evidence_id="confirmation",
                source=second_name,
                value=correct,
                arguments=dict(arguments),
                depends_on=("correction",),
                provenance="cid.correction_training.v1",
            ),
        ),
        metadata={
            "task_kind": "speculative_local_correction",
            "capability": "speculative_local_correction",
            "family": family,
            "training_mode": "tool_required",
            "interaction_pattern": "wrong_hypothesis_conflict_confirm",
            "dependency_depth": 2,
            "provisional_guess": str(provisional),
            "stable_context": stable_context,
            "scope_text": scope_text,
            "task_key": key,
            "generated_by": "cid.correction_training.v1",
        },
        reference_answer=str(correct),
    )


def _descriptor(name: str, description: str, argument_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "arguments": tuple(
            {"name": argument, "kind": "string", "required": True} for argument in argument_names
        ),
        "cacheable": True,
        "dynamic": "live" in name or "current" in name,
        "versioned": True,
    }


def _stale_registry(rng: random.Random, index: int) -> TeacherTask:
    key = f"service-{index:05d}"
    provisional, correct = rng.choice((("disabled", "enabled"), ("enabled", "disabled")))
    return _task(
        family="stale_registry",
        index=index,
        prompt=(
            f"A cached note says {key} is {provisional}. Treat that as provisional, verify the "
            "authoritative registry, and report the verified current value."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"The cached note for {key} may be stale.",
        scope_text=f"Only the authoritative current value of {key} is unresolved.",
        correction_source=("registry_read", "Read the authoritative service registry by key."),
        confirmation_source=("registry_audit", "Independently confirm a service registry value."),
        arguments={"key": key},
    )


def _entity_disambiguation(rng: random.Random, index: int) -> TeacherTask:
    key = f"entity-{index:05d}"
    suffix = rng.choice(("North", "South", "Labs", "Systems"))
    provisional = f"Orion {index % 97}"
    correct = f"Orion {index % 97} {suffix}"
    return _task(
        family="entity_disambiguation",
        index=index,
        prompt=(
            f"The common-name match for {key} suggests {provisional}, but the identifier may "
            "resolve to a different entity. Verify the directory entry and report its exact name."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"{key} is an identifier, so common-name frequency is not authoritative.",
        scope_text=f"Resolve only the exact entity bound to {key}.",
        correction_source=("entity_directory", "Resolve an immutable entity identifier."),
        confirmation_source=("id_crosscheck", "Cross-check an entity identifier independently."),
        arguments={"key": key},
    )


def _unit_assumption(rng: random.Random, index: int) -> TeacherTask:
    key = f"timer-{index:05d}"
    magnitude = rng.randint(2, 900)
    provisional_unit, correct_unit = rng.choice((("ms", "us"), ("s", "ms"), ("KB", "KiB")))
    provisional = f"{magnitude} {provisional_unit}"
    correct = f"{magnitude} {correct_unit}"
    return _task(
        family="unit_assumption",
        index=index,
        prompt=(
            f"Most nearby fields use {provisional_unit}, so a first guess for {key} is "
            f"{provisional}. Verify the exact specification and report the value with its "
            "documented unit."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"Unit conventions near {key} are only a heuristic, not a specification.",
        scope_text=f"Preserve the magnitude while verifying the documented unit for {key}.",
        correction_source=(
            "spec_sheet",
            "Read the exact field value and unit from the specification.",
        ),
        confirmation_source=(
            "spec_mirror",
            "Confirm a specification field from an independent mirror.",
        ),
        arguments={"key": key},
    )


def _boundary_classification(rng: random.Random, index: int) -> TeacherTask:
    key = f"case-{index:05d}"
    provisional, correct = rng.choice((("eligible", "ineligible"), ("ineligible", "eligible")))
    approximate = rng.randint(480, 520) / 10
    return _task(
        family="boundary_classification",
        index=index,
        prompt=(
            f"A rough score near the decision boundary ({approximate:.1f}) suggests "
            f"{provisional} for {key}. Verify the exact adjudicated classification and report it."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=(
            f"{key} lies near a classification boundary, so the rough score is uncertain."
        ),
        scope_text=f"Only the final adjudicated class for {key} should be revised.",
        correction_source=("decision_service", "Read the exact adjudicated classification."),
        confirmation_source=("decision_audit", "Independently confirm the adjudicated class."),
        arguments={"key": key},
    )


def _numeric_estimate(rng: random.Random, index: int) -> TeacherTask:
    a, b, c = rng.randint(1000, 9000), rng.randint(17, 99), rng.randint(100, 900)
    correct = a * b + c
    delta = rng.choice((-9, -7, -3, 3, 7, 9))
    provisional = correct + delta
    expression = f"{a}*{b}+{c}"
    key = f"expr-{index:05d}"
    return _task(
        family="numeric_estimate",
        index=index,
        prompt=(
            f"A quick mental pass estimates {expression} as {provisional}. Keep that only as a "
            "provisional result, verify exact arithmetic, and return the exact value."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=(
            f"The expression {expression} is fixed; only its evaluated value is uncertain."
        ),
        scope_text="Revise the numeric result if exact arithmetic contradicts the estimate.",
        correction_source=("calculator", "Evaluate a deterministic arithmetic expression exactly."),
        confirmation_source=(
            "checksum_calculator",
            "Independently recompute an arithmetic expression.",
        ),
        arguments={"expression": expression},
    )


def _version_default(rng: random.Random, index: int) -> TeacherTask:
    major = rng.randint(2, 9)
    old_minor = rng.randint(0, 8)
    key = f"feature-{index:05d}"
    provisional = f"v{major}.{old_minor}"
    correct = f"v{major}.{old_minor + 1}"
    return _task(
        family="version_default",
        index=index,
        prompt=(
            f"An older deployment note associates {key} with {provisional}. Verify the current "
            "release registry and report the active version."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"The deployment note for {key} predates the current release registry.",
        scope_text=f"Only the active version attached to {key} is subject to revision.",
        correction_source=("release_registry", "Read the active release version for a feature."),
        confirmation_source=(
            "release_manifest",
            "Confirm the active version from the release manifest.",
        ),
        arguments={"key": key},
    )


def _category_exception(rng: random.Random, index: int) -> TeacherTask:
    key = f"item-{index:05d}"
    provisional, correct = rng.choice(
        (("standard", "restricted"), ("general", "special"), ("tier-1", "tier-2"))
    )
    return _task(
        family="category_exception",
        index=index,
        prompt=(
            f"Most items in this series are labeled {provisional}, so that is a plausible first "
            f"guess for {key}. Check the catalog exception record and report the actual category."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"Series-level frequency is not authoritative for exceptional item {key}.",
        scope_text=f"Only the catalog category of {key} needs correction.",
        correction_source=("catalog_lookup", "Read the authoritative item category."),
        confirmation_source=(
            "taxonomy_audit",
            "Confirm an item category against taxonomy records.",
        ),
        arguments={"key": key},
    )


def _operation_order(rng: random.Random, index: int) -> TeacherTask:
    key = f"pipeline-{index:05d}"
    orders = (
        ("normalize→clip→scale", "clip→normalize→scale"),
        ("decode→filter→rank", "filter→decode→rank"),
        ("parse→validate→expand", "parse→expand→validate"),
    )
    provisional, correct = rng.choice(orders)
    return _task(
        family="operation_order",
        index=index,
        prompt=(
            f"A remembered ordering for {key} is {provisional}. Verify the current policy "
            "definition and report the exact operation order."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"The same three operations belong to {key}; only their order is uncertain.",
        scope_text=f"Revise only the operation ordering for {key} when policy evidence arrives.",
        correction_source=("policy_lookup", "Read the authoritative operation-order policy."),
        confirmation_source=("policy_mirror", "Confirm an operation-order policy independently."),
        arguments={"key": key},
    )


def _source_priority(rng: random.Random, index: int) -> TeacherTask:
    key = f"metric-{index:05d}"
    correct = rng.randint(100, 9999)
    provisional = correct + rng.choice((-31, -17, 19, 43))
    return _task(
        family="source_priority",
        index=index,
        prompt=(
            f"A secondary snapshot lists {key} as {provisional}. Use it only as a provisional "
            "value, verify the primary source, and report the authoritative value."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"Primary-source evidence outranks the secondary snapshot for {key}.",
        scope_text=f"Only the resolved value of {key} depends on the conflicting source.",
        correction_source=("primary_source", "Read the authoritative primary value by key."),
        confirmation_source=("primary_replica", "Confirm a primary-source value from a replica."),
        arguments={"key": key},
    )


def _temporal_status(rng: random.Random, index: int) -> TeacherTask:
    key = f"job-{index:05d}"
    provisional, correct = rng.choice(
        (("running", "completed"), ("queued", "running"), ("healthy", "degraded"))
    )
    return _task(
        family="temporal_status",
        index=index,
        prompt=(
            f"The last observed state of {key} was {provisional}. Treat that as a stale "
            "hypothesis, query the live status, and report the current state."
        ),
        provisional=provisional,
        correct=correct,
        key=key,
        stable_context=f"{key} has time-varying status, so the previous observation can expire.",
        scope_text=f"Only the current status field for {key} should change after refresh.",
        correction_source=("live_status", "Read the current live status for a job."),
        confirmation_source=(
            "current_status_replica",
            "Confirm current status from an independent live replica.",
        ),
        arguments={"key": key},
    )


def _cell(
    cell_id: str,
    semantic_text: str,
    roles: Mapping[str, float],
    *,
    uncertainty: float,
    noise: float,
    lifecycle: str,
    anchors: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        **_cell_fields(
            semantic_text,
            roles,
            uncertainty=uncertainty,
            noise=noise,
            lifecycle=lifecycle,
            anchors=anchors,
            links=links,
        ),
    }


def _cell_fields(
    semantic_text: str,
    roles: Mapping[str, float],
    *,
    uncertainty: float,
    noise: float,
    lifecycle: str,
    anchors: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "semantic_text": semantic_text,
        "roles": dict(roles),
        "uncertainty": uncertainty,
        "noise": noise,
        "lifecycle": lifecycle,
        "anchors": anchors,
        "links": links,
    }


def _link(relation: str, kind: str, identifier: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "target": {"kind": kind, "identifier": identifier},
        "confidence": 1.0,
    }


def _anchor(value: Any, scope: str) -> dict[str, Any]:
    if isinstance(value, bool):
        value = str(value).lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {
            "anchor_id": f"number:{_short_id(scope + '|' + str(value))}",
            "kind": "number",
            "value": value,
            "confidence": 1.0,
        }
    text = str(value)
    if re.fullmatch(r"[-+]?\d+", text):
        return {
            "anchor_id": f"number:{_short_id(scope + '|' + text)}",
            "kind": "number",
            "value": int(text),
            "confidence": 1.0,
        }
    return {
        "anchor_id": f"text:{_short_id(scope + '|' + text)}",
        "kind": "text",
        "value": text[:192],
        "confidence": 1.0,
    }


def _contract_key(contract: Mapping[str, Any]) -> str:
    arguments = dict(contract.get("arguments", {}))
    if not arguments:
        return str(contract["evidence_id"])
    return str(next(iter(arguments.values())))


def _short_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def safe_request_filename(text: str) -> str:
    return _SAFE.sub("-", text).strip("-")[:96]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

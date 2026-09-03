from __future__ import annotations

import ast
import hashlib
import json
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import sympy as sp

from cid.causal_distill import build_causal_teacher_job, dump_causal_teacher_jobs
from cid.computational_training import calculator_descriptor, record_lookup_descriptor
from cid.data import DISPLAY_UNKNOWN_MARKER, dump_jsonl
from cid.distill import (
    TeacherCellPlan,
    TeacherEvidence,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_reviews,
    dump_teacher_tasks,
    review_teacher_plans,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.symbolic_training import symbolic_math_descriptor

COMPOSED_FAMILIES = (
    "policy_filter_calculate",
    "lookup_then_symbolic",
    "parallel_lookup_system",
    "stale_correct_calculate",
    "alias_disambiguate_calculate",
    "parallel_branch_symbolic",
)

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True, slots=True)
class ComposedTrainingConfig:
    count_per_family: int = 2000
    seed: int = 20260813
    variants_per_task: int = 2
    thought_capacity: int = 8
    min_delay_steps: int = 1
    max_delay_steps: int = 4

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")
        if self.variants_per_task <= 0:
            raise ValueError("variants_per_task must be positive")
        if self.thought_capacity < 7:
            raise ValueError("composed training requires thought_capacity >= 7")

    @property
    def total_tasks(self) -> int:
        return self.count_per_family * len(COMPOSED_FAMILIES)


@dataclass(frozen=True, slots=True)
class _Case:
    task: TeacherTask
    goal: str
    summaries: tuple[str, ...]
    need_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.summaries) != len(self.task.evidence) + 1:
            raise ValueError("case summaries must contain initial plus one entry per evidence")
        if len(self.need_texts) != len(self.task.evidence):
            raise ValueError("case need texts must align with evidence")


def build_composed_distillation(
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: ComposedTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or ComposedTrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    cases = generate_composed_cases(config)
    tasks = tuple(case.task for case in cases)
    plans = tuple(_plan_for(case) for case in cases)
    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = [review.to_dict() for review in rejected[:10]]
        raise RuntimeError(f"composed teacher review rejected {len(rejected)} plans: {detail}")

    verification_failures = [task.task_id for task in tasks if not verify_composed_task(task)]
    if verification_failures:
        raise RuntimeError(f"composed exact verification failed: {verification_failures[:10]}")

    schedule = TeacherScheduleConfig(
        thought_capacity=config.thought_capacity,
        min_delay_steps=config.min_delay_steps,
        max_delay_steps=config.max_delay_steps,
        variants_per_task=config.variants_per_task,
        seed=config.seed,
    )
    trajectories = compile_teacher_plans(tasks, plans, schedule)

    paths = {
        "tasks": output / "composed-teacher-tasks-v1.jsonl",
        "causal_jobs": output / "composed-teacher-causal-v1.jsonl",
        "plans": output / "composed-teacher-plans-v1.jsonl",
        "accepted": output / "composed-teacher-plans-v1.accepted.jsonl",
        "reviews": output / "composed-teacher-review-v1.jsonl",
        "trajectories": output / "composed-trajectories-v1.jsonl",
        "trajectory_manifest": output / "composed-trajectories-v1.manifest.json",
    }
    dump_teacher_tasks(tasks, paths["tasks"])
    dump_causal_teacher_jobs(tasks, paths["causal_jobs"])
    dump_teacher_plans(plans, paths["plans"])
    dump_teacher_plans(plans, paths["accepted"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])

    family_counts = Counter(str(task.metadata["family"]) for task in tasks)
    pattern_counts = Counter(str(task.metadata["interaction_pattern"]) for task in tasks)
    depth_counts = Counter(int(task.metadata["dependency_depth"]) for task in tasks)
    stage_counts = Counter(len(build_causal_teacher_job(task).stages) for task in tasks)
    source_counts: Counter[str] = Counter()
    for task in tasks:
        for evidence in task.evidence:
            source_counts[evidence.source] += 1

    max_semantic_text_chars = max(
        len(cell.semantic_text) for plan in plans for frame in plan.frames for cell in frame.cells
    )
    if max_semantic_text_chars > 112:
        raise RuntimeError(f"composed semantic text cap exceeded: {max_semantic_text_chars}")

    tasks_with_anchor = sum(
        any(cell.anchors for frame in plan.frames for cell in frame.cells) for plan in plans
    )
    tasks_with_link = sum(
        any(cell.links for frame in plan.frames for cell in frame.cells) for plan in plans
    )
    if tasks_with_anchor != len(tasks) or tasks_with_link != len(tasks):
        raise RuntimeError("every composed task must contain both grounding anchors and links")

    compiled_transitions = sum(
        max(
            [target.step for target in trajectory.thought_targets]
            + [target.step for target in trajectory.display_targets]
            + [0]
        )
        for trajectory in trajectories
    )
    manifest = {
        "format_version": 1,
        "name": "composed-tool-reasoning-v1",
        "version": 1,
        "generator": "GPT-5.6-Sol-authored deterministic mixed-capability self-distill",
        "self_distilled_teacher": "GPT-5.6 Sol",
        "seed": config.seed,
        "semantic_tasks": len(tasks),
        "accepted_plans": len(plans),
        "review_rejected": 0,
        "compiled_trajectories": len(trajectories),
        "compiled_transitions": compiled_transitions,
        "family_counts": dict(sorted(family_counts.items())),
        "interaction_pattern_counts": dict(sorted(pattern_counts.items())),
        "dependency_depth_histogram": {str(k): v for k, v in sorted(depth_counts.items())},
        "causal_stage_histogram": {str(k): v for k, v in sorted(stage_counts.items())},
        "tool_call_target_counts": dict(sorted(source_counts.items())),
        "mode_counts": {"tool_required": len(tasks)},
        "thought_capacity_required": config.thought_capacity,
        "semantic_text_cap": 112,
        "max_semantic_text_chars": max_semantic_text_chars,
        "exact_verifier_failures": 0,
        "tasks_with_anchor": tasks_with_anchor,
        "tasks_with_link": tasks_with_link,
        "compiler": {
            "variants_per_task": config.variants_per_task,
            "min_delay_steps": config.min_delay_steps,
            "max_delay_steps": config.max_delay_steps,
            "seed": config.seed,
        },
        "tasks_sha256": _sha256(paths["tasks"]),
        "causal_jobs_sha256": _sha256(paths["causal_jobs"]),
        "plans_sha256": _sha256(paths["plans"]),
        "accepted_plans_sha256": _sha256(paths["accepted"]),
        "review_sha256": _sha256(paths["reviews"]),
        "compiled_sha256": _sha256(paths["trajectories"]),
    }
    reference_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "composed-trajectories-v1",
                "schema": "cid.TrajectoryExample.v1",
                "examples": len(trajectories),
                "transitions": compiled_transitions,
                "thought_capacity_required": config.thought_capacity,
                "max_trajectory_steps": max(
                    max(
                        [target.step for target in trajectory.thought_targets]
                        + [target.step for target in trajectory.display_targets]
                        + [0]
                    )
                    for trajectory in trajectories
                ),
                "sha256": manifest["compiled_sha256"],
                "reference_manifest": str(reference_path),
                "tag_counts": {
                    f"family:{family}": count * config.variants_per_task
                    for family, count in sorted(family_counts.items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_composed_cases(
    config: ComposedTrainingConfig | None = None,
) -> tuple[_Case, ...]:
    config = config or ComposedTrainingConfig()
    rng = random.Random(config.seed)
    generators = {
        "policy_filter_calculate": _policy_filter_calculate,
        "lookup_then_symbolic": _lookup_then_symbolic,
        "parallel_lookup_system": _parallel_lookup_system,
        "stale_correct_calculate": _stale_correct_calculate,
        "alias_disambiguate_calculate": _alias_disambiguate_calculate,
        "parallel_branch_symbolic": _parallel_branch_symbolic,
    }
    cases: list[_Case] = []
    for family in COMPOSED_FAMILIES:
        generator = generators[family]
        for index in range(config.count_per_family):
            cases.append(generator(rng, index))
    cases.sort(key=lambda case: case.task.task_id)
    return tuple(cases)


def verify_composed_task(task: TeacherTask) -> bool:
    if not task.evidence or task.reference_answer is None:
        return False
    final = task.evidence[-1]
    try:
        if final.source == "calculator":
            value = _eval_arithmetic(str(final.arguments["expression"]))
            return Decimal(str(value)) == Decimal(str(task.reference_answer)) and str(
                final.value
            ) == str(task.reference_answer)
        if final.source == "symbolic_math":
            operation = str(final.arguments["operation"])
            expression = str(final.arguments["expression"])
            variables = str(final.arguments["variables"])
            return _verify_symbolic_solution(
                operation,
                expression,
                variables,
                str(task.reference_answer),
            ) and str(final.value) == str(task.reference_answer)
    except (
        InvalidOperation,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        sp.SympifyError,
    ):
        return False
    return False


def _policy_filter_calculate(rng: random.Random, index: int) -> _Case:
    threshold = rng.randint(45, 88)
    multiplier_hundredths = rng.randint(108, 185)
    multiplier = multiplier_hundredths / 100.0
    winner = index % 2
    score_good = threshold + rng.randint(2, 12)
    score_bad = threshold - rng.randint(2, 12)
    bases = [rng.randint(900, 9000), rng.randint(900, 9000)]
    candidates = [
        {"status": "active", "score": score_good, "base": bases[0]},
        {"status": "inactive" if index % 4 < 2 else "active", "score": score_bad, "base": bases[1]},
    ]
    if winner:
        candidates.reverse()
    selected = candidates[winner]
    # Recompute the winner after the optional reversal so exactly one candidate qualifies.
    qualifying = [
        i
        for i, item in enumerate(candidates)
        if item["status"] == "active" and int(item["score"]) >= threshold
    ]
    if len(qualifying) != 1:
        raise AssertionError("policy generator must create exactly one qualifying candidate")
    winner = qualifying[0]
    selected = candidates[winner]
    answer = f"{selected['base'] * multiplier:.2f}"
    policy_key = f"policy-{index:06d}"
    candidate_keys = (f"candidate-a-{index:06d}", f"candidate-b-{index:06d}")
    evidence = (
        _evidence(
            "policy",
            "record_lookup",
            {"required_status": "active", "min_score": threshold, "multiplier": multiplier},
            {"key": policy_key},
        ),
        _evidence(
            "candidate-a",
            "record_lookup",
            candidates[0],
            {"key": candidate_keys[0]},
            depends_on=("policy",),
        ),
        _evidence(
            "candidate-b",
            "record_lookup",
            candidates[1],
            {"key": candidate_keys[1]},
            depends_on=("policy",),
        ),
        _evidence(
            "adjusted-value",
            "calculator",
            answer,
            {"expression": f"round({selected['base']}*{multiplier_hundredths}/100,2)"},
            depends_on=("candidate-a", "candidate-b"),
        ),
    )
    prompt_variants = (
        (
            "Read policy record {p}, compare candidate records {a} and {b}, and identify the only "
            "candidate that is active and meets the policy score threshold. Return that "
            "candidate's "
            "base value multiplied by the policy multiplier, with exactly two decimal places."
        ),
        (
            "Use task-local policy {p} to screen records {a} and {b}. Exactly one record "
            "qualifies. "
            "Compute its policy-adjusted base and return exactly two decimal places."
        ),
        (
            "Resolve policy {p} first, then inspect {a} and {b}. Apply the policy multiplier only "
            "to the uniquely eligible record and return the adjusted value to two decimals."
        ),
    )
    prompt = prompt_variants[index % len(prompt_variants)].format(
        p=policy_key, a=candidate_keys[0], b=candidate_keys[1]
    )
    task = _task(
        family="policy_filter_calculate",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), calculator_descriptor()),
        pattern="lookup_then_parallel_filter_then_calculate",
        depth=3,
    )
    winner_label = "A" if winner == 0 else "B"
    return _Case(
        task=task,
        goal="Resolve the policy, screen both records, then calculate only the qualifying value.",
        summaries=(
            "Eligibility and multiplier are unresolved until policy and candidate records arrive.",
            (
                f"Policy requires active status with score at least {threshold}; multiplier is "
                f"{multiplier:.2f}."
            ),
            (
                f"Candidate A has status {candidates[0]['status']}, score "
                f"{candidates[0]['score']}, base {candidates[0]['base']}."
            ),
            (
                f"Candidate {winner_label} uniquely qualifies; its base {selected['base']} is "
                "ready "
                "for adjustment."
            ),
            f"Adjusted qualifying value is {answer}.",
        ),
        need_texts=(
            "Need the policy threshold and multiplier before screening candidates.",
            "Need candidate A after the policy rule is known.",
            "Need candidate B after the policy rule is known.",
            "Need exact arithmetic for the selected candidate's adjusted value.",
        ),
    )


def _lookup_then_symbolic(rng: random.Random, index: int) -> _Case:
    solution = rng.randint(-80, 120)
    a = rng.randint(2, 17)
    b = rng.randint(-60, 60)
    target = a * solution + b
    key = f"equation-spec-{index:06d}"
    expression = f"{a}*x+({b})={target}"
    answer = str(solution)
    evidence = (
        _evidence(
            "equation-spec",
            "record_lookup",
            {"coefficient": a, "offset": b, "target": target},
            {"key": key},
        ),
        _evidence(
            "solution",
            "symbolic_math",
            answer,
            {"operation": "solve", "expression": expression, "variables": "x"},
            depends_on=("equation-spec",),
        ),
    )
    prompt = (
        f"Read task-local record {key}. Interpret its coefficient, offset, and target as "
        "coefficient*x + offset = target. Solve exactly for x and return the integer only."
    )
    task = _task(
        family="lookup_then_symbolic",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), symbolic_math_descriptor()),
        pattern="lookup_then_symbolic",
        depth=2,
    )
    return _Case(
        task=task,
        goal=(
            "Load the hidden equation specification, construct the equation, then solve it exactly."
        ),
        summaries=(
            "The equation coefficients are external; no symbolic expression is valid yet.",
            f"Resolved equation is {a}*x + ({b}) = {target}; exact solving can now run.",
            f"Exact symbolic solution is x={answer}.",
        ),
        need_texts=(
            "Need the equation specification before constructing the symbolic call.",
            "Need exact symbolic solving of the resolved linear equation.",
        ),
    )


def _parallel_lookup_system(rng: random.Random, index: int) -> _Case:
    x0 = rng.randint(-30, 40)
    y0 = rng.randint(-30, 40)
    while True:
        a, b, c, d = (rng.randint(1, 9) for _ in range(4))
        if a * d != b * c:
            break
    rhs1 = a * x0 + b * y0
    rhs2 = c * x0 + d * y0
    left_key = f"system-left-{index:06d}"
    right_key = f"system-right-{index:06d}"
    answer = f"x={x0}, y={y0}"
    expression = f"{a}*x+{b}*y={rhs1};{c}*x+{d}*y={rhs2}"
    evidence = (
        _evidence(
            "left-equation",
            "record_lookup",
            {"x": a, "y": b, "rhs": rhs1},
            {"key": left_key},
        ),
        _evidence(
            "right-equation",
            "record_lookup",
            {"x": c, "y": d, "rhs": rhs2},
            {"key": right_key},
        ),
        _evidence(
            "system-solution",
            "symbolic_math",
            answer,
            {"operation": "solve_system", "expression": expression, "variables": "x,y"},
            depends_on=("left-equation", "right-equation"),
        ),
    )
    prompt = (
        f"Read independent records {left_key} and {right_key}. Each stores x coefficient, y "
        "coefficient, and right-hand side for one linear equation. Solve the resulting "
        "two-equation system exactly. "
        "Return `x=<integer>, y=<integer>`."
    )
    task = _task(
        family="parallel_lookup_system",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), symbolic_math_descriptor()),
        pattern="parallel_lookup_then_symbolic_merge",
        depth=2,
    )
    return _Case(
        task=task,
        goal=(
            "Read both independent equations in parallel, merge them, then solve the system "
            "exactly."
        ),
        summaries=(
            "Both equation records are unresolved; either lookup can proceed independently.",
            f"First equation resolved as {a}*x + {b}*y = {rhs1}; the second is still pending.",
            (
                f"Both equations are resolved; determinant {a * d - b * c} is nonzero, so the "
                "system has one solution."
            ),
            f"Exact system solution is {answer}.",
        ),
        need_texts=(
            "Need the first equation record.",
            "Need the independent second equation record.",
            "Need exact symbolic solving after both equations are available.",
        ),
    )


def _stale_correct_calculate(rng: random.Random, index: int) -> _Case:
    amount = rng.randint(1200, 18000)
    true_rate = rng.randint(3, 19)
    stale_rate = true_rate + rng.choice((-4, -3, -2, 2, 3, 4))
    if stale_rate <= 0:
        stale_rate = true_rate + 2
    cache_key = f"cached-rate-{index:06d}"
    authoritative_key = f"authoritative-rate-{index:06d}"
    answer = f"{amount * (100 + true_rate) / 100:.2f}"
    evidence = (
        _evidence(
            "cached-record",
            "record_lookup",
            {"amount": amount, "rate": stale_rate},
            {"key": cache_key},
        ),
        _evidence(
            "authoritative-record",
            "record_lookup",
            {"amount": amount, "rate": true_rate},
            {"key": authoritative_key},
            depends_on=("cached-record",),
        ),
        _evidence(
            "corrected-total",
            "calculator",
            answer,
            {"expression": f"round({amount}*(100+{true_rate})/100,2)"},
            depends_on=("authoritative-record",),
        ),
    )
    prompt = (
        f"Record {cache_key} is a possibly stale cached rate for an amount. Record "
        f"{authoritative_key} is authoritative and must override any conflicting cached rate while "
        "leaving the amount unchanged. "
        "Return amount*(1+authoritative_rate/100) with exactly two decimal places."
    )
    task = _task(
        family="stale_correct_calculate",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), calculator_descriptor()),
        pattern="stale_hypothesis_then_authoritative_correction_then_calculate",
        depth=3,
    )
    return _Case(
        task=task,
        goal=(
            "Treat the cache as provisional, correct only conflicting rate state, then recompute "
            "the total."
        ),
        summaries=(
            "The cached rate may be stale; no rate-dependent result is stable yet.",
            (
                f"Cached state suggests rate {stale_rate}% for amount {amount}; authoritative "
                "verification is required."
            ),
            (
                f"Authoritative rate {true_rate}% conflicts with cached {stale_rate}%; keep amount "
                f"{amount} unchanged."
            ),
            f"Corrected total using the authoritative rate is {answer}.",
        ),
        need_texts=(
            "Need the cached state to form a provisional hypothesis.",
            "Need authoritative state before accepting or revising the cached rate.",
            "Need exact recomputation from the corrected rate while preserving the stable amount.",
        ),
    )


def _alias_disambiguate_calculate(rng: random.Random, index: int) -> _Case:
    alias_key = f"alias-{index:06d}"
    entity_key = f"entity-{rng.randint(100000, 999999)}"
    unit_cents = rng.randint(125, 9900)
    quantity = rng.randint(5, 300)
    answer = f"{unit_cents * quantity / 100:.2f}"
    evidence = (
        _evidence(
            "alias-resolution",
            "record_lookup",
            {"entity_key": entity_key},
            {"key": alias_key},
        ),
        _evidence(
            "entity-record",
            "record_lookup",
            {"unit_price": unit_cents / 100, "quantity": quantity},
            {"key": entity_key},
            depends_on=("alias-resolution",),
        ),
        _evidence(
            "entity-total",
            "calculator",
            answer,
            {"expression": f"round({unit_cents}*{quantity}/100,2)"},
            depends_on=("entity-record",),
        ),
    )
    prompt = (
        f"Resolve alias record {alias_key} to its canonical entity key, read that entity's unit "
        "price and quantity, then return unit_price*quantity with exactly two decimal places. Do "
        "not guess the entity "
        "before the alias record is read."
    )
    task = _task(
        family="alias_disambiguate_calculate",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), calculator_descriptor()),
        pattern="entity_disambiguation_then_lookup_then_calculate",
        depth=3,
    )
    return _Case(
        task=task,
        goal=(
            "Resolve the alias first, bind the canonical entity, then calculate from that entity "
            "only."
        ),
        summaries=(
            (
                "The canonical entity is unresolved; downstream record access must wait for alias "
                "binding."
            ),
            f"Alias resolves to canonical key {entity_key}; only that entity record is relevant.",
            (
                f"Canonical entity has unit price {unit_cents / 100:.2f} and quantity {quantity}; "
                "total is ready to compute."
            ),
            f"Canonical entity total is {answer}.",
        ),
        need_texts=(
            "Need alias resolution before selecting an entity record.",
            "Need the canonical entity record produced by alias resolution.",
            "Need exact multiplication of resolved unit price and quantity.",
        ),
    )


def _parallel_branch_symbolic(rng: random.Random, index: int) -> _Case:
    solution_primary = rng.randint(-50, 70)
    solution_secondary = rng.randint(-50, 70)
    while solution_secondary == solution_primary:
        solution_secondary = rng.randint(-50, 70)
    a = rng.randint(2, 13)
    b = rng.randint(-35, 35)
    c = a * solution_primary + b
    d = rng.randint(2, 13)
    e = rng.randint(-35, 35)
    f = d * solution_secondary + e
    branch = "primary" if index % 2 == 0 else "secondary"
    selector_key = f"branch-selector-{index:06d}"
    formulas_key = f"branch-formulas-{index:06d}"
    answer = str(solution_primary if branch == "primary" else solution_secondary)
    expression = f"{a}*x+({b})={c}" if branch == "primary" else f"{d}*x+({e})={f}"
    evidence = (
        _evidence(
            "selector",
            "record_lookup",
            {"branch": branch},
            {"key": selector_key},
        ),
        _evidence(
            "formulas",
            "record_lookup",
            {
                "primary": f"{a}*x+({b})={c}",
                "secondary": f"{d}*x+({e})={f}",
            },
            {"key": formulas_key},
        ),
        _evidence(
            "branch-solution",
            "symbolic_math",
            answer,
            {"operation": "solve", "expression": expression, "variables": "x"},
            depends_on=("selector", "formulas"),
        ),
    )
    prompt = (
        f"Read branch selector {selector_key} and formula record {formulas_key} independently. The "
        "formula record contains primary and secondary linear equations. Solve only the equation "
        "named by the selector "
        "and return x as an integer."
    )
    task = _task(
        family="parallel_branch_symbolic",
        index=index,
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        descriptors=(record_lookup_descriptor(), symbolic_math_descriptor()),
        pattern="parallel_lookup_then_logic_branch_then_symbolic",
        depth=2,
    )
    return _Case(
        task=task,
        goal=(
            "Read selector and formulas in parallel, choose one branch, then solve only that "
            "equation."
        ),
        summaries=(
            (
                "Selector and formula records are independent; neither branch should be assumed "
                "in advance."
            ),
            (
                f"Selector chooses the {branch} branch; the branch equation itself is still "
                "unresolved."
            ),
            f"Formula record is available; choose only the {branch} equation for symbolic solving.",
            f"Selected branch has exact solution x={answer}.",
        ),
        need_texts=(
            "Need the branch selector.",
            "Need the independent formula record.",
            "Need exact solving of the branch selected from the two resolved records.",
        ),
    )


def _task(
    *,
    family: str,
    index: int,
    prompt: str,
    answer: str,
    evidence: tuple[TeacherEvidence, ...],
    descriptors: tuple[dict[str, Any], ...],
    pattern: str,
    depth: int,
) -> TeacherTask:
    return TeacherTask(
        task_id=f"composed-{family}-{index:06d}",
        prompt=prompt,
        source_descriptors=descriptors,
        evidence=evidence,
        metadata={
            "task_kind": "composed_tool_reasoning",
            "family": family,
            "interaction_pattern": pattern,
            "dependency_depth": depth,
            "training_mode": "tool_required",
            "generated_by": "cid.composed_training.v1",
        },
        reference_answer=answer,
    )


def _evidence(
    evidence_id: str,
    source: str,
    value: Any,
    arguments: dict[str, Any],
    *,
    depends_on: tuple[str, ...] = (),
) -> TeacherEvidence:
    return TeacherEvidence(
        evidence_id=evidence_id,
        source=source,
        value=value,
        arguments=arguments,
        depends_on=depends_on,
        requires_need=True,
        provenance="cid.composed_training.v1",
    )


def _plan_for(case: _Case) -> TeacherPlan:
    task = case.task
    evidence_index = {item.evidence_id: i for i, item in enumerate(task.evidence)}
    activation_phase: dict[str, str] = {}
    for item in task.evidence:
        if not item.depends_on:
            activation_phase[item.evidence_id] = "pre"
        else:
            latest = max(item.depends_on, key=evidence_index.__getitem__)
            activation_phase[item.evidence_id] = f"after:{latest}"

    needs = tuple(
        TeacherNeed(
            need_id=f"need:{item.evidence_id}",
            cell_id=_evidence_cell_id(item.evidence_id),
            evidence_id=item.evidence_id,
            phase=activation_phase[item.evidence_id],
            source=item.source,
            arguments=dict(item.arguments),
        )
        for item in task.evidence
    )

    frames: list[TeacherFrame] = [
        TeacherFrame(
            phase="initial",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(
                _goal_cell(case.goal),
                _state_cell(case.summaries[0], arrived_ids=()),
            ),
        )
    ]

    frames.append(
        TeacherFrame(
            phase="pre",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=_cells_for_phase(case, activation_phase, arrived_count=0, phase="pre"),
        )
    )
    for index, item in enumerate(task.evidence):
        phase = f"after:{item.evidence_id}"
        display = (
            task.reference_answer
            if index == len(task.evidence) - 1
            else DISPLAY_UNKNOWN_MARKER
        )
        frames.append(
            TeacherFrame(
                phase=phase,
                display=str(display),
                cells=_cells_for_phase(
                    case,
                    activation_phase,
                    arrived_count=index + 1,
                    phase=phase,
                ),
            )
        )

    final_cells = list(
        _cells_for_phase(
            case,
            activation_phase,
            arrived_count=len(task.evidence),
            phase=f"after:{task.evidence[-1].evidence_id}",
        )
    )
    final_cells.append(
        TeacherCellPlan(
            cell_id="answer",
            semantic_text=_short(f"Final answer: {task.reference_answer}."),
            roles={CognitiveRole.CONCLUSION: 1.0},
            uncertainty=0.02,
            noise=0.01,
            lifecycle=CellLifecycle.STABLE,
            anchors=(_anchor(task.reference_answer, f"{task.task_id}|answer"),),
            links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("state"), 1.0),),
        )
    )
    frames.append(
        TeacherFrame(
            phase="final",
            display=str(task.reference_answer),
            cells=tuple(final_cells),
        )
    )
    return TeacherPlan(
        task_id=task.task_id,
        final_answer=str(task.reference_answer),
        frames=tuple(frames),
        needs=needs,
    )


def _cells_for_phase(
    case: _Case,
    activation_phase: dict[str, str],
    *,
    arrived_count: int,
    phase: str,
) -> tuple[TeacherCellPlan, ...]:
    task = case.task
    cells: list[TeacherCellPlan] = [
        _goal_cell(case.goal),
        _state_cell(
            case.summaries[arrived_count],
            arrived_ids=tuple(item.evidence_id for item in task.evidence[:arrived_count]),
        ),
    ]
    for index, item in enumerate(task.evidence):
        if index < arrived_count:
            cells.append(_percept_cell(task.task_id, item))
        elif _phase_has_reached(activation_phase[item.evidence_id], phase, task.evidence):
            cells.append(_need_cell(case.need_texts[index], item))
    return tuple(cells)


def _phase_has_reached(
    activation: str,
    current: str,
    evidence: tuple[TeacherEvidence, ...],
) -> bool:
    if activation == "pre":
        return current == "pre" or current.startswith("after:")
    if current == "pre":
        return False
    order = {f"after:{item.evidence_id}": index for index, item in enumerate(evidence)}
    return order[current] >= order[activation]


def _goal_cell(text: str) -> TeacherCellPlan:
    return TeacherCellPlan(
        cell_id="goal",
        semantic_text=_short(text),
        roles={CognitiveRole.PLAN: 1.0, CognitiveRole.CONSTRAINT: 0.35},
        uncertainty=0.35,
        noise=0.06,
        lifecycle=CellLifecycle.STABLE,
    )


def _state_cell(text: str, *, arrived_ids: tuple[str, ...]) -> TeacherCellPlan:
    links: tuple[CognitiveLink, ...]
    if arrived_ids:
        links = tuple(
            CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell(_evidence_cell_id(item)), 0.95)
            for item in arrived_ids[-3:]
        )
    else:
        links = (CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("goal"), 1.0),)
    uncertainty = 0.58 if not arrived_ids else max(0.08, 0.42 - 0.08 * len(arrived_ids))
    return TeacherCellPlan(
        cell_id="state",
        semantic_text=_short(text),
        roles={CognitiveRole.HYPOTHESIS: 0.7, CognitiveRole.PLAN: 0.45},
        uncertainty=uncertainty,
        noise=max(0.04, uncertainty * 0.3),
        lifecycle=CellLifecycle.ACTIVE,
        links=links,
    )


def _need_cell(text: str, evidence: TeacherEvidence) -> TeacherCellPlan:
    links = [CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(evidence.source), 1.0)]
    links.extend(
        CognitiveLink(
            LinkRelation.DEPENDS_ON,
            ObjectRef.cell(_evidence_cell_id(parent)),
            1.0,
        )
        for parent in evidence.depends_on
    )
    return TeacherCellPlan(
        cell_id=_evidence_cell_id(evidence.evidence_id),
        semantic_text=_short(text),
        roles={CognitiveRole.INFORMATION_NEED: 1.0, CognitiveRole.PLAN: 0.25},
        uncertainty=0.84,
        noise=0.08,
        lifecycle=CellLifecycle.WAITING,
        links=tuple(links),
    )


def _percept_cell(task_id: str, evidence: TeacherEvidence) -> TeacherCellPlan:
    links = [CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(evidence.source), 1.0)]
    links.extend(
        CognitiveLink(
            LinkRelation.DEPENDS_ON,
            ObjectRef.cell(_evidence_cell_id(parent)),
            1.0,
        )
        for parent in evidence.depends_on
    )
    return TeacherCellPlan(
        cell_id=_evidence_cell_id(evidence.evidence_id),
        semantic_text=_short(_percept_text(evidence)),
        roles={CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.04,
        noise=0.02,
        lifecycle=CellLifecycle.STABLE,
        anchors=_anchors_for_evidence(task_id, evidence),
        links=tuple(links),
    )


def _percept_text(evidence: TeacherEvidence) -> str:
    value = evidence.value
    if isinstance(value, dict):
        fields = "; ".join(f"{key}={item}" for key, item in value.items())
        return f"{evidence.evidence_id}: {fields}."
    return f"{evidence.evidence_id}: {value}."


def _anchors_for_evidence(task_id: str, evidence: TeacherEvidence) -> tuple[Anchor, ...]:
    values: list[Any] = []
    if "key" in evidence.arguments:
        values.append(evidence.arguments["key"])
    if isinstance(evidence.value, dict):
        values.extend(evidence.value.values())
    else:
        values.append(evidence.value)
    result: list[Anchor] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(values[:4]):
        anchor = _anchor(value, f"{task_id}|{evidence.evidence_id}|{index}")
        key = (anchor.kind.value, str(anchor.value))
        if key in seen:
            continue
        seen.add(key)
        result.append(anchor)
    return tuple(result)


def _anchor(value: Any, scope: str) -> Anchor:
    if isinstance(value, bool):
        value = str(value).lower()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return Anchor(
            anchor_id=f"number:{_short_id(scope + '|' + str(value))}",
            kind=AnchorKind.NUMBER,
            value=value,
            confidence=1.0,
        )
    text = str(value)
    numeric = _parse_number(text)
    if numeric is not None:
        return Anchor(
            anchor_id=f"number:{_short_id(scope + '|' + text)}",
            kind=AnchorKind.NUMBER,
            value=numeric,
            confidence=1.0,
        )
    return Anchor(
        anchor_id=f"text:{_short_id(scope + '|' + text)}",
        kind=AnchorKind.TEXT,
        value=text[:192],
        confidence=1.0,
    )


def _parse_number(text: str) -> int | float | None:
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+\.)", text):
        return float(text)
    return None


def _evidence_cell_id(evidence_id: str) -> str:
    return "ext-" + _SAFE.sub("-", evidence_id).strip("-")[:48]


def _short(text: str) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= 112 else normalized[:109].rstrip() + "..."


def _short_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_arithmetic(expression: str) -> int | float:
    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
        ):
            if len(node.args) not in (1, 2):
                raise ValueError("invalid round call")
            value = visit(node.args[0])
            digits = int(visit(node.args[1])) if len(node.args) == 2 else 0
            return round(value, digits)
        raise ValueError("unsupported arithmetic expression")

    return visit(ast.parse(expression, mode="eval"))


def _verify_symbolic_solution(
    operation: str,
    expression: str,
    variables: str,
    answer: str,
) -> bool:
    symbols = {name: sp.Symbol(name) for name in variables.split(",")}
    if operation == "solve":
        variable = next(iter(symbols.values()))
        lhs, rhs = expression.split("=", 1)
        equation = sp.Eq(
            sp.sympify(lhs, locals=symbols),
            sp.sympify(rhs, locals=symbols),
        )
        candidate = sp.sympify(answer)
        residual = equation.lhs.subs(variable, candidate) - equation.rhs.subs(variable, candidate)
        return sp.simplify(residual) == 0
    if operation == "solve_system":
        assignments: dict[sp.Symbol, sp.Expr] = {}
        for part in answer.split(","):
            name, value = part.strip().split("=", 1)
            assignments[symbols[name.strip()]] = sp.sympify(value.strip())
        equations = []
        for item in expression.split(";"):
            lhs, rhs = item.split("=", 1)
            equations.append(
                sp.simplify(
                    sp.sympify(lhs, locals=symbols).subs(assignments)
                    - sp.sympify(rhs, locals=symbols).subs(assignments)
                )
            )
        return all(value == 0 for value in equations) and len(assignments) == len(symbols)
    return False

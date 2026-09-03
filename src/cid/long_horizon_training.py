from __future__ import annotations

import ast
import hashlib
import json
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from cid.causal_distill import build_causal_teacher_job, dump_causal_teacher_jobs
from cid.data import DISPLAY_UNKNOWN_MARKER, dump_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
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

LONG_HORIZON_FAMILIES = (
    "serial_directory_chain",
    "serial_then_calculate",
    "alias_policy_chain",
    "stale_refresh_chain",
    "fork_join_calculate",
    "triple_branch_join",
)

_LOOKUP_SOURCES = (
    "catalog_lookup",
    "entity_directory",
    "record_lookup",
    "release_registry",
    "policy_lookup",
    "current_status",
)
_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")
_STAGE_LABELS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


@dataclass(frozen=True, slots=True)
class LongHorizonTrainingConfig:
    count_per_family: int = 2000
    seed: int = 20260813
    variants_per_task: int = 2
    thought_capacity: int = 12
    min_delay_steps: int = 1
    max_delay_steps: int = 5

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")
        if self.variants_per_task <= 0:
            raise ValueError("variants_per_task must be positive")
        if self.thought_capacity < 10:
            raise ValueError("long-horizon tool training requires thought_capacity >= 10")
        if self.min_delay_steps <= 0 or self.max_delay_steps < self.min_delay_steps:
            raise ValueError("invalid tool delay range")

    @property
    def total_tasks(self) -> int:
        return self.count_per_family * len(LONG_HORIZON_FAMILIES)


@dataclass(frozen=True, slots=True)
class _Case:
    task: TeacherTask
    goal: str
    state_summaries: tuple[str, ...]
    need_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.state_summaries) != len(self.task.evidence) + 1:
            raise ValueError("state summaries must include initial plus one per evidence")
        if len(self.need_texts) != len(self.task.evidence):
            raise ValueError("need text count must match evidence count")


def build_long_horizon_distillation(
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: LongHorizonTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or LongHorizonTrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    cases = generate_long_horizon_cases(config)
    tasks = tuple(case.task for case in cases)
    plans = tuple(_plan_for(case) for case in cases)
    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = [review.to_dict() for review in rejected[:8]]
        raise RuntimeError(f"long-horizon teacher review rejected {len(rejected)} plans: {detail}")

    verifier_failures = tuple(task.task_id for task in tasks if not verify_long_horizon_task(task))
    if verifier_failures:
        raise RuntimeError(f"long-horizon exact verifier failed: {verifier_failures[:8]}")

    causal_jobs = tuple(build_causal_teacher_job(task) for task in tasks)
    schedule = TeacherScheduleConfig(
        thought_capacity=config.thought_capacity,
        min_delay_steps=config.min_delay_steps,
        max_delay_steps=config.max_delay_steps,
        variants_per_task=config.variants_per_task,
        seed=config.seed,
    )
    trajectories = compile_teacher_plans(tasks, plans, schedule)

    paths = {
        "tasks": output / "long-horizon-teacher-tasks-v1.jsonl",
        "causal_jobs": output / "long-horizon-teacher-causal-v1.jsonl",
        "plans": output / "long-horizon-teacher-plans-v1.accepted.jsonl",
        "reviews": output / "long-horizon-teacher-review-v1.jsonl",
        "trajectories": output / "long-horizon-trajectories-v1.jsonl",
        "trajectory_manifest": output / "long-horizon-trajectories-v1.manifest.json",
    }
    dump_teacher_tasks(tasks, paths["tasks"])
    dump_causal_teacher_jobs(tasks, paths["causal_jobs"])
    dump_teacher_plans(plans, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])
    trajectory_manifest = inspect_dataset(paths["trajectories"])
    dump_dataset_manifest(trajectory_manifest, paths["trajectory_manifest"])

    family_counts = Counter(str(task.metadata["family"]) for task in tasks)
    depth_counts = Counter(int(task.metadata["dependency_depth"]) for task in tasks)
    evidence_counts = Counter(len(task.evidence) for task in tasks)
    stage_counts = Counter(len(job.stages) for job in causal_jobs)
    source_counts: Counter[str] = Counter(
        evidence.source for task in tasks for evidence in task.evidence
    )
    max_semantic = max(
        len(cell.semantic_text) for plan in plans for frame in plan.frames for cell in frame.cells
    )
    if max_semantic > 112:
        raise RuntimeError(f"long-horizon semantic text cap exceeded: {max_semantic}")
    anchored = sum(any(c.anchors for f in p.frames for c in f.cells) for p in plans)
    linked = sum(any(c.links for f in p.frames for c in f.cells) for p in plans)
    if anchored != len(tasks) or linked != len(tasks):
        raise RuntimeError("every long-horizon task must include anchors and links")

    manifest = {
        "format_version": 1,
        "name": "long-horizon-tool-reasoning-v1",
        "version": 1,
        "generator": "cid.long_horizon_training.v1",
        "seed": config.seed,
        "semantic_tasks": len(tasks),
        "accepted_plans": len(plans),
        "review_rejected": 0,
        "exact_verifier_failures": 0,
        "compiled_trajectories": trajectory_manifest.examples,
        "compiled_transitions": trajectory_manifest.transitions,
        "family_counts": dict(sorted(family_counts.items())),
        "dependency_depth_histogram": {str(k): v for k, v in sorted(depth_counts.items())},
        "evidence_count_histogram": {str(k): v for k, v in sorted(evidence_counts.items())},
        "causal_stage_histogram": {str(k): v for k, v in sorted(stage_counts.items())},
        "tool_call_target_counts": dict(sorted(source_counts.items())),
        "mode_counts": {"tool_required": len(tasks)},
        "depth_4_plus_tasks": sum(v for k, v in depth_counts.items() if k >= 4),
        "depth_6_plus_tasks": sum(v for k, v in depth_counts.items() if k >= 6),
        "thought_capacity_required": config.thought_capacity,
        "semantic_text_cap": 112,
        "max_semantic_text_chars": max_semantic,
        "tasks_with_anchor": anchored,
        "tasks_with_link": linked,
        "compiler": {
            "variants_per_task": config.variants_per_task,
            "min_delay_steps": config.min_delay_steps,
            "max_delay_steps": config.max_delay_steps,
            "seed": config.seed,
        },
        "tasks_sha256": _sha256(paths["tasks"]),
        "causal_jobs_sha256": _sha256(paths["causal_jobs"]),
        "plans_sha256": _sha256(paths["plans"]),
        "review_sha256": _sha256(paths["reviews"]),
        "compiled_sha256": trajectory_manifest.sha256,
    }
    reference_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    # Add a canonical reference pointer while keeping the normal dataset-manifest fields.
    raw_trajectory_manifest = json.loads(paths["trajectory_manifest"].read_text(encoding="utf-8"))
    raw_trajectory_manifest["name"] = "long-horizon-trajectories-v1"
    raw_trajectory_manifest["reference_manifest"] = str(reference_path)
    raw_trajectory_manifest["tag_counts"] = {
        f"family:{name}": count * config.variants_per_task
        for name, count in sorted(family_counts.items())
    }
    paths["trajectory_manifest"].write_text(
        json.dumps(raw_trajectory_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_long_horizon_cases(
    config: LongHorizonTrainingConfig | None = None,
) -> tuple[_Case, ...]:
    config = config or LongHorizonTrainingConfig()
    rng = random.Random(config.seed)
    generators = {
        "serial_directory_chain": _serial_directory_chain,
        "serial_then_calculate": _serial_then_calculate,
        "alias_policy_chain": _alias_policy_chain,
        "stale_refresh_chain": _stale_refresh_chain,
        "fork_join_calculate": _fork_join_calculate,
        "triple_branch_join": _triple_branch_join,
    }
    cases: list[_Case] = []
    for family in LONG_HORIZON_FAMILIES:
        generator = generators[family]
        for index in range(config.count_per_family):
            cases.append(generator(rng, index))
    cases.sort(key=lambda case: case.task.task_id)
    return tuple(cases)


def verify_long_horizon_task(task: TeacherTask) -> bool:
    try:
        family = str(task.metadata["family"])
        evidence = task.evidence
        answer = str(task.reference_answer)
        if family == "serial_directory_chain":
            if len(evidence) != int(task.metadata["dependency_depth"]):
                return False
            current_key = str(task.metadata["start_key"])
            for i, item in enumerate(evidence):
                if item.arguments.get("key") != current_key:
                    return False
                if i + 1 < len(evidence):
                    current_key = str(item.value["next_key"])
                elif str(item.value["result"]) != answer:
                    return False
            return True
        if family == "serial_then_calculate":
            if evidence[-1].source != "calculator":
                return False
            expression = str(evidence[-1].arguments["expression"])
            value = _eval_arithmetic(expression)
            return Decimal(str(value)) == Decimal(answer) and str(evidence[-1].value) == answer
        if family == "alias_policy_chain":
            amount = int(task.metadata["amount"])
            rate = int(task.metadata["rate"])
            expected = f"{amount * (100 + rate) / 100:.2f}"
            return expected == answer and str(evidence[-1].value) == expected
        if family == "stale_refresh_chain":
            amount = int(task.metadata["amount"])
            true_rate = int(task.metadata["true_rate"])
            stale_rate = int(task.metadata["stale_rate"])
            expected = f"{amount * (100 + true_rate) / 100:.2f}"
            return (
                stale_rate != true_rate
                and expected == answer
                and str(evidence[-1].value) == expected
            )
        if family in {"fork_join_calculate", "triple_branch_join"}:
            expression = str(evidence[-1].arguments["expression"])
            value = _eval_arithmetic(expression)
            return Decimal(str(value)) == Decimal(answer) and str(evidence[-1].value) == answer
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return False


def _serial_directory_chain(rng: random.Random, index: int) -> _Case:
    depth = 4 + index % 3
    start = f"route-{index:06d}"
    keys = [start] + [f"node-{rng.randrange(1_000_000):06d}-{i}" for i in range(depth - 1)]
    result = f"terminal-{rng.randrange(10_000_000):07d}"
    evidence: list[TeacherEvidence] = []
    summaries = ["The route is unresolved; only the initial key is known."]
    needs: list[str] = []
    for i in range(depth):
        source = _LOOKUP_SOURCES[i % len(_LOOKUP_SOURCES)]
        value: Any = {"result": result} if i == depth - 1 else {"next_key": keys[i + 1]}
        stage = _STAGE_LABELS[i]
        previous = _STAGE_LABELS[i - 1] if i else None
        evidence.append(
            _evidence(
                f"hop-{stage}",
                source,
                value,
                {"key": keys[i]},
                depends_on=(() if previous is None else (f"hop-{previous}",)),
            )
        )
        needs.append("Need the next licensed route hop before continuing the chain.")
        if i == depth - 1:
            summaries.append(f"The latest hop resolves the terminal result {result}.")
        else:
            summaries.append(
                "The latest hop resolves one next route key; downstream state remains unresolved."
            )
    task = _task(
        family="serial_directory_chain",
        index=index,
        prompt=(
            f"Start from task-local key {start}. Follow each returned `next_key` through the "
            "available read-only sources until a record returns `result`. "
            f"This instance has {depth} dependent "
            "lookups. Return only the terminal result."
        ),
        answer=result,
        evidence=tuple(evidence),
        depth=depth,
        pattern="strict_serial_cross_source_lookup_chain",
        metadata={"start_key": start},
    )
    return _Case(
        task,
        "Follow only resolved keys through the full serial source chain.",
        tuple(summaries),
        tuple(needs),
    )


def _serial_then_calculate(rng: random.Random, index: int) -> _Case:
    lookup_depth = 3 + index % 3
    start = f"calc-route-{index:06d}"
    keys = [start] + [
        f"calc-node-{rng.randrange(1_000_000):06d}-{i}" for i in range(lookup_depth - 1)
    ]
    base = rng.randint(100, 9000)
    multiplier = rng.randint(2, 17)
    offset = rng.randint(-250, 250)
    answer = str(base * multiplier + offset)
    evidence: list[TeacherEvidence] = []
    summaries = ["The numeric record is hidden behind a dependent lookup chain."]
    needs: list[str] = []
    for i in range(lookup_depth):
        last = i == lookup_depth - 1
        value: Any = (
            {"base": base, "multiplier": multiplier, "offset": offset}
            if last
            else {"next_key": keys[i + 1]}
        )
        evidence.append(
            _evidence(
                f"lookup-{_STAGE_LABELS[i]}",
                _LOOKUP_SOURCES[(i + 1) % len(_LOOKUP_SOURCES)],
                value,
                {"key": keys[i]},
                depends_on=(() if i == 0 else (f"lookup-{_STAGE_LABELS[i - 1]}",)),
            )
        )
        needs.append(
            "Need the next dependent lookup before another key or the numeric record is available."
        )
        summaries.append(
            "The terminal numeric record is now available for exact arithmetic."
            if last
            else "The latest lookup exposes one next key; downstream values remain unresolved."
        )
    evidence.append(
        _evidence(
            "calculation",
            "calculator",
            answer,
            {"expression": f"{base}*{multiplier}+({offset})"},
            depends_on=(f"lookup-{_STAGE_LABELS[lookup_depth - 1]}",),
        )
    )
    needs.append("Need exact arithmetic only after the terminal numeric record arrives.")
    summaries.append(f"Exact terminal calculation gives {answer}.")
    depth = lookup_depth + 1
    task = _task(
        family="serial_then_calculate",
        index=index,
        prompt=(
            f"Starting at {start}, follow the returned keys to the terminal numeric record. "
            "Then compute base*multiplier+offset with the calculator. Do not infer downstream keys "
            "before their parent "
            "record arrives. Return the integer only."
        ),
        answer=answer,
        evidence=tuple(evidence),
        depth=depth,
        pattern="serial_lookup_chain_then_exact_calculation",
    )
    return _Case(
        task,
        "Resolve every lookup dependency before performing the final arithmetic.",
        tuple(summaries),
        tuple(needs),
    )


def _alias_policy_chain(rng: random.Random, index: int) -> _Case:
    alias = f"alias-{index:06d}"
    entity = f"entity-{rng.randrange(1_000_000):06d}"
    class_key = f"class-{rng.randrange(100_000):05d}"
    policy_key = f"policy-{rng.randrange(100_000):05d}"
    rate_key = f"rate-{rng.randrange(100_000):05d}"
    amount = rng.randint(500, 25_000)
    rate = rng.randint(2, 24)
    answer = f"{amount * (100 + rate) / 100:.2f}"
    evidence = (
        _evidence("alias", "catalog_lookup", {"entity": entity}, {"key": alias}),
        _evidence(
            "entity",
            "entity_directory",
            {"class_key": class_key, "amount": amount},
            {"key": entity},
            depends_on=("alias",),
        ),
        _evidence(
            "class",
            "record_lookup",
            {"policy_key": policy_key},
            {"key": class_key},
            depends_on=("entity",),
        ),
        _evidence(
            "policy",
            "policy_lookup",
            {"rate_key": rate_key},
            {"key": policy_key},
            depends_on=("class",),
        ),
        _evidence(
            "rate", "release_registry", {"rate": rate}, {"key": rate_key}, depends_on=("policy",)
        ),
        _evidence(
            "calculation",
            "calculator",
            answer,
            {"expression": f"round({amount}*(100+{rate})/100,2)"},
            depends_on=("rate",),
        ),
    )
    summaries = (
        "Alias, entity class, policy, and rate are unresolved.",
        (
            "Alias is resolved to one canonical entity; class and amount still require the "
            "entity record."
        ),
        f"Entity record supplies amount {amount} and one class key; policy remains unresolved.",
        "Class record resolves the policy key; the policy-specific rate is still unknown.",
        "Policy record resolves the rate key; fetch the authoritative rate before calculating.",
        f"Authoritative rate is {rate}%; exact adjustment can now run.",
        f"Adjusted amount is {answer}.",
    )
    needs = (
        "Need canonical entity resolution for the alias.",
        "Need the canonical entity record before class selection.",
        "Need the entity class record before choosing a policy.",
        "Need the policy record before selecting its rate key.",
        "Need the authoritative rate record before arithmetic.",
        "Need exact arithmetic from the resolved amount and authoritative rate.",
    )
    task = _task(
        family="alias_policy_chain",
        index=index,
        prompt=(
            f"Resolve {alias} to its canonical entity, then follow entity class -> policy -> rate. "
            "Use the entity amount and final authoritative rate to compute "
            "amount*(1+rate/100), exactly "
            "two decimals. Return only that value."
        ),
        answer=answer,
        evidence=evidence,
        depth=6,
        pattern="alias_entity_class_policy_rate_calculate",
        metadata={"amount": amount, "rate": rate},
    )
    return _Case(
        task,
        "Resolve identity and policy dependencies before using the final rate.",
        summaries,
        needs,
    )


def _stale_refresh_chain(rng: random.Random, index: int) -> _Case:
    amount = rng.randint(800, 30_000)
    true_rate = rng.randint(2, 22)
    stale_rate = true_rate + rng.choice((-5, -4, -3, 3, 4, 5))
    if stale_rate <= 0:
        stale_rate = true_rate + 3
    cache_key = f"cache-{index:06d}"
    version_key = f"version-{rng.randrange(100_000):05d}"
    auth_key = f"authoritative-{rng.randrange(100_000):05d}"
    policy_key = f"refresh-policy-{rng.randrange(100_000):05d}"
    answer = f"{amount * (100 + true_rate) / 100:.2f}"
    evidence = (
        _evidence(
            "cache",
            "record_lookup",
            {"amount": amount, "rate": stale_rate, "version_key": version_key},
            {"key": cache_key},
            version="cached",
        ),
        _evidence(
            "version",
            "current_status",
            {"authoritative_key": auth_key},
            {"key": version_key},
            depends_on=("cache",),
            version="current",
        ),
        _evidence(
            "authoritative",
            "release_registry",
            {"rate": true_rate, "policy_key": policy_key},
            {"key": auth_key},
            depends_on=("version",),
            version="current",
        ),
        _evidence(
            "policy",
            "policy_lookup",
            {"use": "authoritative_rate"},
            {"key": policy_key},
            depends_on=("authoritative",),
        ),
        _evidence(
            "calculation",
            "calculator",
            answer,
            {"expression": f"round({amount}*(100+{true_rate})/100,2)"},
            depends_on=("policy",),
        ),
    )
    summaries = (
        "Cached state may be stale; current version and authoritative rate are unresolved.",
        f"Cache suggests rate {stale_rate}% for amount {amount}; treat it as provisional only.",
        (
            "Current-version metadata points to the authoritative record; fetch it before "
            "accepting a rate."
        ),
        (
            f"Authoritative record revises the rate to {true_rate}% and identifies the "
            "governing policy."
        ),
        "Policy confirms authoritative-rate precedence; cached rate must not influence the result.",
        f"Corrected authoritative calculation gives {answer}.",
    )
    needs = (
        "Need cached state only as a provisional starting point.",
        "Need current-version metadata before selecting the authoritative record.",
        "Need the authoritative record before any rate-dependent conclusion.",
        "Need precedence policy to validate the local correction.",
        "Need exact recomputation from the stable amount and authoritative rate.",
    )
    task = _task(
        family="stale_refresh_chain",
        index=index,
        prompt=(
            f"Read cached record {cache_key}, but treat its rate as provisional. Follow its "
            "version pointer to current metadata, fetch the authoritative rate, confirm precedence "
            "policy, and recompute "
            "amount*(1+rate/100) to two decimals. Return only the corrected value."
        ),
        answer=answer,
        evidence=evidence,
        depth=5,
        pattern="stale_cache_version_authority_policy_recompute",
        metadata={"amount": amount, "true_rate": true_rate, "stale_rate": stale_rate},
    )
    return _Case(
        task,
        "Keep stale state provisional until current authority and policy confirm a correction.",
        summaries,
        needs,
    )


def _fork_join_calculate(rng: random.Random, index: int) -> _Case:
    root = f"fork-root-{index:06d}"
    left_key = f"left-{rng.randrange(1_000_000):06d}"
    right_key = f"right-{rng.randrange(1_000_000):06d}"
    join_key = f"join-{rng.randrange(1_000_000):06d}"
    left = rng.randint(20, 500)
    right = rng.randint(20, 500)
    scale = rng.randint(2, 9)
    answer = str((left + right) * scale)
    evidence = (
        _evidence(
            "root", "catalog_lookup", {"left_key": left_key, "right_key": right_key}, {"key": root}
        ),
        _evidence(
            "left", "entity_directory", {"value": left}, {"key": left_key}, depends_on=("root",)
        ),
        _evidence(
            "right", "record_lookup", {"value": right}, {"key": right_key}, depends_on=("root",)
        ),
        _evidence(
            "join",
            "release_registry",
            {"join_key": join_key},
            {"key": f"{left_key}|{right_key}"},
            depends_on=("left", "right"),
        ),
        _evidence(
            "scale", "policy_lookup", {"scale": scale}, {"key": join_key}, depends_on=("join",)
        ),
        _evidence(
            "calculation",
            "calculator",
            answer,
            {"expression": f"({left}+{right})*{scale}"},
            depends_on=("scale",),
        ),
    )
    summaries = (
        "Root routing is unresolved; branch keys are not yet available.",
        "Root exposes two independent branch keys; both branches can now be read.",
        f"Left branch contributes {left}; right branch is still pending.",
        f"Both branch values are available ({left}, {right}); resolve their join record next.",
        "Join record supplies the policy key required for the final scale.",
        f"Scale policy supplies factor {scale}; branch merge is ready for exact arithmetic.",
        f"Merged scaled value is {answer}.",
    )
    needs = (
        "Need root routing before either branch can launch.",
        "Need the left branch value after root routing.",
        "Need the independent right branch value after root routing.",
        "Need the join record after both branch values arrive.",
        "Need the scale policy selected by the join record.",
        "Need exact arithmetic after both branches and the scale are resolved.",
    )
    task = _task(
        family="fork_join_calculate",
        index=index,
        prompt=(
            f"Read root {root}, resolve both returned branch records, then join them and follow "
            "the join "
            "policy to a scale. Return (left_value+right_value)*scale as an integer."
        ),
        answer=answer,
        evidence=evidence,
        depth=5,
        pattern="root_parallel_branches_join_policy_calculate",
    )
    return _Case(
        task,
        "Launch branches only after root routing, then wait for both before joining.",
        summaries,
        needs,
    )


def _triple_branch_join(rng: random.Random, index: int) -> _Case:
    root = f"triple-root-{index:06d}"
    keys = [f"branch-{j}-{rng.randrange(1_000_000):06d}" for j in range(3)]
    values = [rng.randint(10, 400) for _ in range(3)]
    normalize_key = f"normalize-{rng.randrange(1_000_000):06d}"
    offset = rng.randint(-100, 100)
    answer = str(sum(values) + offset)
    evidence = [
        _evidence("root", "catalog_lookup", {"keys": keys}, {"key": root}),
    ]
    for j, (key, value) in enumerate(zip(keys, values, strict=True), start=1):
        evidence.append(
            _evidence(
                f"branch-{_STAGE_LABELS[j - 1]}",
                _LOOKUP_SOURCES[j],
                {"value": value},
                {"key": key},
                depends_on=("root",),
            )
        )
    evidence.extend(
        (
            _evidence(
                "normalization",
                "policy_lookup",
                {"offset": offset, "normalization_key": normalize_key},
                {"key": "|".join(keys)},
                depends_on=("branch-alpha", "branch-beta", "branch-gamma"),
            ),
            _evidence(
                "calculation",
                "calculator",
                answer,
                {"expression": f"{values[0]}+{values[1]}+{values[2]}+({offset})"},
                depends_on=("normalization",),
            ),
        )
    )
    summaries = (
        "Root routing is unresolved; three branch keys are not yet available.",
        "Root exposes three independent branch keys; collect all branches before normalization.",
        f"The first branch contributes {values[0]}; two independent branches remain.",
        "Two branches are available; the final branch remains before the join can activate.",
        (
            f"All branch values are available ({values[0]}, {values[1]}, {values[2]}); "
            "resolve normalization."
        ),
        f"Normalization supplies offset {offset}; exact three-way merge can now run.",
        f"Normalized three-way result is {answer}.",
    )
    needs = (
        "Need root routing before launching three independent branches.",
        "Need the first branch after root routing.",
        "Need another independent branch after root routing.",
        "Need the final independent branch after root routing.",
        "Need normalization only after all three branch values arrive.",
        "Need exact three-way arithmetic after normalization is known.",
    )
    task = _task(
        family="triple_branch_join",
        index=index,
        prompt=(
            f"Read root {root}, fetch all three returned branches, wait for all of them, then "
            "apply the "
            "normalization offset and return the sum of all branches plus offset as an integer."
        ),
        answer=answer,
        evidence=tuple(evidence),
        depth=4,
        pattern="root_three_parallel_branches_barrier_normalize_calculate",
    )
    return _Case(
        task,
        "Use a three-branch barrier before normalization and final arithmetic.",
        summaries,
        needs,
    )


def _task(
    *,
    family: str,
    index: int,
    prompt: str,
    answer: str,
    evidence: tuple[TeacherEvidence, ...],
    depth: int,
    pattern: str,
    metadata: dict[str, Any] | None = None,
) -> TeacherTask:
    source_names = tuple(dict.fromkeys(item.source for item in evidence))
    descriptors = tuple(
        _calculator_descriptor() if name == "calculator" else _lookup_descriptor(name)
        for name in source_names
    )
    meta = {
        "task_kind": "long_horizon_tool_reasoning",
        "family": family,
        "interaction_pattern": pattern,
        "dependency_depth": depth,
        "training_mode": "tool_required",
        "generated_by": "cid.long_horizon_training.v1",
    }
    if metadata:
        meta.update(metadata)
    return TeacherTask(
        task_id=f"long-horizon-{family}-{index:06d}",
        prompt=prompt,
        source_descriptors=descriptors,
        evidence=evidence,
        metadata=meta,
        reference_answer=answer,
    )


def _lookup_descriptor(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Read one immutable task-local record by key.",
        "arguments": ({"name": "key", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": name in {"current_status", "release_registry"},
        "versioned": name in {"current_status", "release_registry"},
    }


def _calculator_descriptor() -> dict[str, Any]:
    return {
        "name": "calculator",
        "description": "Evaluate one exact arithmetic expression.",
        "arguments": ({"name": "expression", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def _evidence(
    evidence_id: str,
    source: str,
    value: Any,
    arguments: dict[str, Any],
    *,
    depends_on: tuple[str, ...] = (),
    version: str | None = None,
) -> TeacherEvidence:
    return TeacherEvidence(
        evidence_id=evidence_id,
        source=source,
        value=value,
        arguments=arguments,
        depends_on=depends_on,
        requires_need=True,
        version=version,
        provenance="cid.long_horizon_training.v1",
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
            cell_id=_cell_id(item.evidence_id),
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
            cells=(_goal_cell(case.goal), _state_cell(case.state_summaries[0], ())),
        ),
        TeacherFrame(
            phase="pre",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=_cells_for_phase(case, activation_phase, 0, "pre"),
        ),
    ]
    for index, item in enumerate(task.evidence):
        phase = f"after:{item.evidence_id}"
        frames.append(
            TeacherFrame(
                phase=phase,
                display=str(task.reference_answer)
                if index == len(task.evidence) - 1
                else DISPLAY_UNKNOWN_MARKER,
                cells=_cells_for_phase(case, activation_phase, index + 1, phase),
            )
        )
    final_cells = list(
        _cells_for_phase(
            case, activation_phase, len(task.evidence), f"after:{task.evidence[-1].evidence_id}"
        )
    )
    final_cells.append(
        TeacherCellPlan(
            cell_id="answer",
            semantic_text=_short(f"Final answer: {task.reference_answer}."),
            roles={CognitiveRole.CONCLUSION: 1.0},
            uncertainty=0.01,
            noise=0.0,
            lifecycle=CellLifecycle.STABLE,
            anchors=(_anchor(task.reference_answer, f"{task.task_id}|answer"),),
            links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("state"), 1.0),),
        )
    )
    frames.append(
        TeacherFrame(phase="final", display=str(task.reference_answer), cells=tuple(final_cells))
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
    arrived_count: int,
    phase: str,
) -> tuple[TeacherCellPlan, ...]:
    task = case.task
    cells: list[TeacherCellPlan] = [
        _goal_cell(case.goal),
        _state_cell(
            case.state_summaries[arrived_count],
            tuple(item.evidence_id for item in task.evidence[:arrived_count]),
        ),
    ]
    for index, item in enumerate(task.evidence):
        if index < arrived_count:
            cells.append(_percept_cell(task.task_id, item))
        elif _phase_has_reached(activation_phase[item.evidence_id], phase, task.evidence):
            cells.append(_need_cell(case.need_texts[index], item))
    return tuple(cells)


def _phase_has_reached(
    activation: str, current: str, evidence: tuple[TeacherEvidence, ...]
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
        uncertainty=0.3,
        noise=0.05,
        lifecycle=CellLifecycle.STABLE,
    )


def _state_cell(text: str, arrived_ids: tuple[str, ...]) -> TeacherCellPlan:
    links = (
        tuple(
            CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell(_cell_id(item)), 0.95)
            for item in arrived_ids[-3:]
        )
        if arrived_ids
        else (CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("goal"), 1.0),)
    )
    uncertainty = max(0.06, 0.55 - 0.07 * len(arrived_ids))
    return TeacherCellPlan(
        cell_id="state",
        semantic_text=_short(text),
        roles={CognitiveRole.HYPOTHESIS: 0.7, CognitiveRole.PLAN: 0.45},
        uncertainty=uncertainty,
        noise=max(0.03, uncertainty * 0.25),
        lifecycle=CellLifecycle.ACTIVE,
        links=links,
    )


def _need_cell(text: str, evidence: TeacherEvidence) -> TeacherCellPlan:
    links = [CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(evidence.source), 1.0)]
    links.extend(
        CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell(_cell_id(parent)), 1.0)
        for parent in evidence.depends_on
    )
    return TeacherCellPlan(
        cell_id=_cell_id(evidence.evidence_id),
        semantic_text=_short(text),
        roles={CognitiveRole.INFORMATION_NEED: 1.0, CognitiveRole.PLAN: 0.25},
        uncertainty=0.86,
        noise=0.08,
        lifecycle=CellLifecycle.WAITING,
        links=tuple(links),
    )


def _percept_cell(task_id: str, evidence: TeacherEvidence) -> TeacherCellPlan:
    links = [CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(evidence.source), 1.0)]
    links.extend(
        CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell(_cell_id(parent)), 1.0)
        for parent in evidence.depends_on
    )
    return TeacherCellPlan(
        cell_id=_cell_id(evidence.evidence_id),
        semantic_text=_short(_percept_text(evidence)),
        roles={CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.03,
        noise=0.01,
        lifecycle=CellLifecycle.STABLE,
        anchors=_anchors_for_evidence(task_id, evidence),
        links=tuple(links),
    )


def _percept_text(evidence: TeacherEvidence) -> str:
    if isinstance(evidence.value, dict):
        fields = "; ".join(f"{key}={value}" for key, value in evidence.value.items())
        return f"{evidence.evidence_id}: {fields}."
    return f"{evidence.evidence_id}: {evidence.value}."


def _anchors_for_evidence(task_id: str, evidence: TeacherEvidence) -> tuple[Anchor, ...]:
    values: list[Any] = []
    if "key" in evidence.arguments:
        values.append(evidence.arguments["key"])
    if isinstance(evidence.value, dict):
        for value in evidence.value.values():
            if isinstance(value, list):
                values.extend(value[:2])
            else:
                values.append(value)
    else:
        values.append(evidence.value)
    result: list[Anchor] = []
    seen: set[tuple[str, str]] = set()
    for idx, value in enumerate(values[:4]):
        anchor = _anchor(value, f"{task_id}|{evidence.evidence_id}|{idx}")
        key = (anchor.kind.value, str(anchor.value))
        if key not in seen:
            result.append(anchor)
            seen.add(key)
    return tuple(result)


def _anchor(value: Any, scope: str) -> Anchor:
    if isinstance(value, bool):
        value = str(value).lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Anchor(
            anchor_id=f"number:{_short_id(scope + '|' + str(value))}",
            kind=AnchorKind.NUMBER,
            value=value,
            confidence=1.0,
        )
    text = str(value)
    return Anchor(
        anchor_id=f"text:{_short_id(scope + '|' + text)}",
        kind=AnchorKind.TEXT,
        value=text[:192],
        confidence=1.0,
    )


def _cell_id(evidence_id: str) -> str:
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
}


def _eval_arithmetic(expression: str) -> int | float:
    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
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

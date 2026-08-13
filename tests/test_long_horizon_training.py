from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cid.distill import load_teacher_plans, load_teacher_tasks, review_teacher_plans
from cid.long_horizon_training import (
    LONG_HORIZON_FAMILIES,
    LongHorizonTrainingConfig,
    build_long_horizon_distillation,
    generate_long_horizon_cases,
    verify_long_horizon_task,
)


def _small_config() -> LongHorizonTrainingConfig:
    return LongHorizonTrainingConfig(
        count_per_family=6,
        variants_per_task=2,
        seed=20260813,
    )


def test_long_horizon_generation_covers_deep_tool_dependencies() -> None:
    cases = generate_long_horizon_cases(_small_config())
    tasks = tuple(case.task for case in cases)
    families = Counter(str(task.metadata["family"]) for task in tasks)
    depths = Counter(int(task.metadata["dependency_depth"]) for task in tasks)

    assert set(families) == set(LONG_HORIZON_FAMILIES)
    assert set(families.values()) == {6}
    assert min(depths) >= 4
    assert max(depths) == 6
    assert sum(depths.values()) == len(tasks)
    assert all(task.evidence for task in tasks)
    assert all(verify_long_horizon_task(task) for task in tasks)


def test_long_horizon_build_passes_review_and_emits_grounding(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    reference = tmp_path / "reference.json"
    manifest = build_long_horizon_distillation(output, reference, _small_config())

    assert manifest["semantic_tasks"] == 36
    assert manifest["compiled_trajectories"] == 72
    assert manifest["depth_4_plus_tasks"] == 36
    assert manifest["depth_6_plus_tasks"] > 0
    assert manifest["review_rejected"] == 0
    assert manifest["exact_verifier_failures"] == 0
    assert manifest["max_semantic_text_chars"] <= 112
    assert manifest["tasks_with_anchor"] == 36
    assert manifest["tasks_with_link"] == 36

    tasks = load_teacher_tasks(output / "long-horizon-teacher-tasks-v1.jsonl")
    plans = load_teacher_plans(output / "long-horizon-teacher-plans-v1.accepted.jsonl")
    assert all(review.accepted for review in review_teacher_plans(tasks, plans))

    trajectory_manifest = json.loads(
        (output / "long-horizon-trajectories-v1.manifest.json").read_text(encoding="utf-8")
    )
    assert trajectory_manifest["examples"] == 72
    assert trajectory_manifest["thought_capacity_required"] == 12
    assert trajectory_manifest["reference_manifest"] == str(reference)


def test_long_horizon_generation_is_seed_deterministic() -> None:
    left = generate_long_horizon_cases(_small_config())
    right = generate_long_horizon_cases(_small_config())
    assert tuple(case.task.to_dict() for case in left) == tuple(
        case.task.to_dict() for case in right
    )

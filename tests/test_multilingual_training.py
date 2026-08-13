from __future__ import annotations

from cid.data import load_jsonl
from cid.distill import review_teacher_plans
from cid.multilingual_training import (
    MULTILINGUAL_FAMILIES,
    MultilingualTrainingConfig,
    _plan_for,
    build_multilingual_training,
    generate_multilingual_tasks,
)


def test_multilingual_generator_is_balanced_and_cross_lingual() -> None:
    config = MultilingualTrainingConfig(
        zh_tasks=3,
        en_zh_tasks=2,
        ja_tasks=2,
        es_tasks=1,
        schedule_variants=2,
        seed=17,
    )
    tasks = generate_multilingual_tasks(config)

    assert len(tasks) == 8
    assert {str(task.metadata["family"]) for task in tasks} == set(MULTILINGUAL_FAMILIES)
    assert all(task.metadata["cross_lingual"] is True for task in tasks)
    assert all(task.metadata["canonical_tct_language"] == "en" for task in tasks)
    assert {str(task.metadata["prompt_language"]) for task in tasks} == {"en", "es", "ja", "zh"}
    assert all(len(task.evidence) == 2 for task in tasks)
    assert all(task.evidence[1].depends_on == ("bridge",) for task in tasks)


def test_generated_multilingual_plans_pass_quality_review() -> None:
    tasks = generate_multilingual_tasks(
        MultilingualTrainingConfig(
            zh_tasks=2,
            en_zh_tasks=2,
            ja_tasks=2,
            es_tasks=2,
            seed=23,
        )
    )
    plans = tuple(_plan_for(task) for task in tasks)
    reviews = review_teacher_plans(tasks, plans)

    assert all(review.accepted for review in reviews)
    assert all(plan.frames[-1].display == plan.final_answer for plan in plans)
    assert all(any(cell.anchors for frame in plan.frames for cell in frame.cells) for plan in plans)
    assert all(any(cell.links for frame in plan.frames for cell in frame.cells) for plan in plans)


def test_build_multilingual_training_materializes_two_schedule_variants(tmp_path) -> None:
    config = MultilingualTrainingConfig(
        zh_tasks=2,
        en_zh_tasks=1,
        ja_tasks=1,
        es_tasks=1,
        schedule_variants=2,
        seed=31,
    )
    manifest = build_multilingual_training(tmp_path, config)
    trajectories = load_jsonl(tmp_path / "trajectories-v1.jsonl")

    assert manifest["semantic_tasks"] == 5
    assert manifest["accepted_teacher_plans"] == 5
    assert manifest["compiled_trajectories"] == 10
    assert manifest["cross_lingual_tasks"] == 5
    assert manifest["canonical_tct_language"] == "en"
    assert len(trajectories) == 10
    assert all(example.grounding_catalog for example in trajectories)
    assert all(example.grounding_targets for example in trajectories)
    assert all(example.metadata["schedule_variant"] in {0, 1} for example in trajectories)

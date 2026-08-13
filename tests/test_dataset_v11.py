from __future__ import annotations

from collections import Counter

from cid.deep_restraint_training import (
    DeepToolRestraintConfig,
    generate_deep_tool_restraint_examples,
)
from cid.distill import TeacherCellPlan, TeacherFrame, TeacherPlan, TeacherTask
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.surface_diversity_training import (
    SurfaceDiversityConfig,
    _normalized_surface_signature,
    diversify_tasks_and_plans,
)


def test_deep_restraint_stratifies_capacity_and_never_requests_tools() -> None:
    tasks = tuple(
        _source_task(capacity=capacity, index=index)
        for capacity in (16, 32, 64, 128)
        for index in range(4)
    )
    plans = tuple(_source_plan(task) for task in tasks)

    augmented_tasks, augmented_plans = generate_deep_tool_restraint_examples(
        tasks,
        plans,
        DeepToolRestraintConfig(count_per_bucket=2, min_dependency_depth=8, seed=7),
    )

    assert len(augmented_tasks) == len(augmented_plans) == 8
    assert Counter(int(task.metadata["thought_capacity_bucket"]) for task in augmented_tasks) == {
        16: 2,
        32: 2,
        64: 2,
        128: 2,
    }
    for task, plan in zip(augmented_tasks, augmented_plans, strict=True):
        assert task.task_id.startswith("deep-restraint-source-")
        assert int(task.metadata["dependency_depth"]) >= 8
        assert task.metadata["training_mode"] == "tools_available_unnecessary"
        assert task.source_descriptors[0]["name"] == "record_lookup"
        assert not task.evidence
        assert not plan.needs
        assert plan.final_answer == task.reference_answer


def test_surface_v2_preserves_core_semantics_while_diversifying_prompts() -> None:
    tasks = tuple(
        _source_task(capacity=16, index=index, common_prompt=True) for index in range(160)
    )
    plans = tuple(_source_plan(task) for task in tasks)
    config = SurfaceDiversityConfig(
        component_name="fixture-v2",
        file_stem="fixture-v2",
        thought_capacity=16,
        seed=23,
    )

    diversified_tasks, diversified_plans = diversify_tasks_and_plans(tasks, plans, config)
    signatures = {_normalized_surface_signature(task.prompt) for task in diversified_tasks}

    assert len(diversified_tasks) == len(diversified_plans) == 160
    assert len(signatures) >= 120
    source_by_id = {task.task_id: task for task in tasks}
    source_plan_by_id = {plan.task_id: plan for plan in plans}
    plan_by_id = {plan.task_id: plan for plan in diversified_plans}
    for task in diversified_tasks:
        source_id = str(task.metadata["source_task_id"])
        source = source_by_id[source_id]
        source_plan = source_plan_by_id[source_id]
        plan = plan_by_id[task.task_id]
        assert source.prompt in task.prompt
        assert task.reference_answer == source.reference_answer
        assert task.evidence == source.evidence
        assert task.source_descriptors == source.source_descriptors
        assert task.metadata["surface_version"] == 2
        assert source_id == source.task_id
        assert plan.task_id == f"surface-v2-{source.task_id}"
        assert plan.frames == source_plan.frames
        assert plan.final_answer == source_plan.final_answer


def _source_task(*, capacity: int, index: int, common_prompt: bool = False) -> TeacherTask:
    answer = str(index % 7)
    prompt = (
        "Resolve the task-local dependency chain and return the requested answer."
        if common_prompt
        else f"Resolve task-local chain {capacity}-{index} and return {answer}."
    )
    return TeacherTask(
        task_id=f"source-{capacity}-{index}",
        prompt=prompt,
        metadata={
            "task_kind": "compositional_longtail_reasoning",
            "family": f"fixture-{index % 3}",
            "training_mode": "no_tool_required",
            "thought_capacity_bucket": capacity,
            "dependency_depth": 8 + index % 5,
        },
        reference_answer=answer,
    )


def _source_plan(task: TeacherTask) -> TeacherPlan:
    answer = str(task.reference_answer)
    premise = TeacherCellPlan(
        cell_id="premise",
        semantic_text="The task premises are self-contained.",
        roles={CognitiveRole.PLAN: 1.0},
        uncertainty=0.1,
        noise=0.02,
        lifecycle=CellLifecycle.STABLE,
    )
    conclusion = TeacherCellPlan(
        cell_id="answer",
        semantic_text=f"Conclusion: {answer}",
        roles={CognitiveRole.CONCLUSION: 1.0},
        uncertainty=0.02,
        noise=0.01,
        lifecycle=CellLifecycle.STABLE,
        anchors=(
            Anchor(
                anchor_id=f"fixture:{task.task_id}",
                kind=AnchorKind.TEXT,
                value=answer,
            ),
        ),
        links=(
            CognitiveLink(
                LinkRelation.DERIVED_FROM,
                ObjectRef.cell("premise"),
                1.0,
            ),
        ),
    )
    return TeacherPlan(
        task_id=task.task_id,
        final_answer=answer,
        frames=(TeacherFrame("initial", answer, (premise, conclusion)),),
    )

from __future__ import annotations

from collections import Counter

from cid.deep_restraint_training import (
    DeepToolRestraintConfig,
    generate_deep_tool_restraint_examples,
)
from cid.distill import (
    TeacherCellPlan,
    TeacherEvidence,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherTask,
    dump_teacher_plans,
    dump_teacher_tasks,
    review_teacher_plans,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.surface_diversity_training import (
    SurfaceDiversityConfig,
    _normalized_surface_signature,
    _rewrite_semantic_core,
    build_surface_diversified_distillation,
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


def test_surface_v3_varies_semantic_transport_without_changing_typed_plan_semantics() -> None:
    tasks = tuple(_source_task(capacity=16, index=index, common_prompt=True) for index in range(96))
    plans = tuple(_source_plan(task) for task in tasks)
    config = SurfaceDiversityConfig(
        component_name="fixture-v3",
        file_stem="fixture-v3",
        thought_capacity=16,
        seed=31,
        surface_version=3,
        diversify_semantic_text=True,
    )

    diversified_tasks, diversified_plans = diversify_tasks_and_plans(tasks, plans, config)
    changed_texts = 0
    source_plan_by_id = {plan.task_id: plan for plan in plans}
    for task, plan in zip(diversified_tasks, diversified_plans, strict=True):
        source_id = str(task.metadata["source_task_id"])
        source_plan = source_plan_by_id[source_id]
        assert task.metadata["surface_version"] == 3
        assert plan.final_answer == source_plan.final_answer
        assert len(plan.frames) == len(source_plan.frames)
        for frame, source_frame in zip(plan.frames, source_plan.frames, strict=True):
            assert frame.phase == source_frame.phase
            assert frame.display == source_frame.display
            for cell, source_cell in zip(frame.cells, source_frame.cells, strict=True):
                assert source_cell.semantic_text in cell.semantic_text
                assert cell.roles == source_cell.roles
                assert cell.lifecycle == source_cell.lifecycle
                assert cell.anchors == source_cell.anchors
                assert cell.links == source_cell.links
                assert len(cell.semantic_text) <= config.semantic_text_cap
                changed_texts += cell.semantic_text != source_cell.semantic_text
    assert changed_texts > len(tasks)


def test_surface_v3_retries_only_collided_semantic_wrapper(tmp_path) -> None:
    source = {
        "name": "docs",
        "description": "read documentation",
        "arguments": ({"name": "key", "kind": "string", "required": True},),
    }
    task = TeacherTask(
        task_id="collision-source",
        prompt="Return the documented status.",
        source_descriptors=(source,),
        evidence=(
            TeacherEvidence(
                evidence_id="e0",
                source="docs",
                value="active",
                arguments={"key": "status"},
            ),
        ),
    )
    initial = TeacherFrame(
        "initial",
        "pending",
        (TeacherCellPlan("plan", "Plan a documentation lookup.", {CognitiveRole.PLAN: 1.0}),),
    )
    pre = TeacherFrame(
        "pre",
        "pending",
        (
            TeacherCellPlan(
                "plan",
                "The answer must use documentation.",
                {CognitiveRole.CONSTRAINT: 1.0},
            ),
            TeacherCellPlan(
                "need",
                "Need the documented status.",
                {CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.9,
                noise=0.8,
            ),
        ),
    )
    after_cells = (
        TeacherCellPlan(
            "plan",
            "The documentation requirement is satisfied.",
            {CognitiveRole.CONSTRAINT: 1.0},
            uncertainty=0.1,
            noise=0.1,
            lifecycle=CellLifecycle.STABLE,
        ),
        TeacherCellPlan(
            "need",
            "The documented status is active.",
            {CognitiveRole.CONCLUSION: 1.0},
            uncertainty=0.05,
            noise=0.1,
            lifecycle=CellLifecycle.STABLE,
        ),
    )
    plan = TeacherPlan(
        task_id=task.task_id,
        final_answer="active",
        frames=(
            initial,
            pre,
            TeacherFrame("after:e0", "active", after_cells),
            TeacherFrame("final", "active", after_cells),
        ),
        needs=(
            TeacherNeed(
                "lookup",
                "need",
                "e0",
                "pre",
                "docs",
                {"key": "status"},
            ),
        ),
    )
    assert review_teacher_plans((task,), (plan,))[0].accepted

    config = SurfaceDiversityConfig(
        component_name="collision-fixture-v3",
        file_stem="collision-v3",
        thought_capacity=8,
        variants_per_task=1,
        seed=3,
        surface_version=3,
        diversify_prompt=False,
        diversify_semantic_text=True,
    )
    diversified_tasks, diversified_plans = diversify_tasks_and_plans((task,), (plan,), config)
    initial_review = review_teacher_plans(diversified_tasks, diversified_plans)[0]
    assert not initial_review.accepted
    assert any("leaks future evidence" in reason for reason in initial_review.reasons)

    tasks_path = tmp_path / "source-tasks.jsonl"
    plans_path = tmp_path / "source-plans.jsonl"
    dump_teacher_tasks((task,), tasks_path)
    dump_teacher_plans((plan,), plans_path)
    manifest = build_surface_diversified_distillation(
        tasks_path,
        plans_path,
        tmp_path / "generated",
        tmp_path / "reference-manifest.json",
        config,
    )

    assert manifest["accepted_plans"] == 1
    assert manifest["review_rejected"] == 0
    assert manifest["semantic_text_retry_plans"] == 1
    assert manifest["semantic_text_fallback_plans"] == 0


def test_surface_v4_rewrites_high_frequency_semantic_templates() -> None:
    source = "Need task-local evidence for the requested fact or relation."
    variants = {_rewrite_semantic_core(source, selector) for selector in range(12)}

    assert source not in variants
    assert len(variants) == 6
    assert all("evidence" in value.casefold() for value in variants)

    reachability = {
        _rewrite_semantic_core(
            "node-17 is reachable under the directed-edge and block constraints.", i
        )
        for i in range(12)
    }
    assert len(reachability) == 6
    assert all("node-17" in value for value in reachability)


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

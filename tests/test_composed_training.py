from __future__ import annotations

from pathlib import Path

from cid.composed_training import ComposedTrainingConfig, build_composed_distillation
from cid.distill import TeacherCellPlan, TeacherFrame, TeacherPlan, TeacherTask
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.tool_restraint_training import (
    ToolRestraintTrainingConfig,
    generate_tool_restraint_examples,
)


def test_build_composed_distillation_is_exact_reviewed_and_grounded(tmp_path: Path) -> None:
    manifest = build_composed_distillation(
        tmp_path / "generated",
        tmp_path / "reference.json",
        ComposedTrainingConfig(count_per_family=2, variants_per_task=2, seed=19),
    )

    assert manifest["semantic_tasks"] == 12
    assert manifest["accepted_plans"] == 12
    assert manifest["review_rejected"] == 0
    assert manifest["compiled_trajectories"] == 24
    assert manifest["compiled_transitions"] > 100
    assert manifest["exact_verifier_failures"] == 0
    assert manifest["tasks_with_anchor"] == 12
    assert manifest["tasks_with_link"] == 12
    assert manifest["max_semantic_text_chars"] <= 112
    assert set(manifest["dependency_depth_histogram"]) == {"2", "3"}
    assert (tmp_path / "generated" / "composed-trajectories-v1.jsonl").is_file()


def test_tool_restraint_augmentation_exposes_tools_without_requesting_them() -> None:
    tasks = tuple(_source_task(index) for index in range(3))
    plans = tuple(_source_plan(task) for task in tasks)

    augmented_tasks, augmented_plans = generate_tool_restraint_examples(
        tasks,
        plans,
        ToolRestraintTrainingConfig(count=2, seed=7),
    )

    assert len(augmented_tasks) == len(augmented_plans) == 2
    for task, plan in zip(augmented_tasks, augmented_plans, strict=True):
        assert task.task_id.startswith("restraint-source-")
        assert task.metadata["training_mode"] == "tools_available_unnecessary"
        assert task.source_descriptors
        assert not task.evidence
        assert not plan.needs
        assert plan.frames[0].display == task.reference_answer
        decision = next(cell for cell in plan.frames[0].cells if cell.cell_id == "tool-decision")
        assert "tool" in decision.semantic_text.casefold()
        assert decision.lifecycle is CellLifecycle.STABLE
        assert decision.links[0].relation is LinkRelation.DEPENDS_ON


def _source_task(index: int) -> TeacherTask:
    return TeacherTask(
        task_id=f"source-{index}",
        prompt=f"What is {index} + 1?",
        metadata={
            "task_kind": "math_word_problem",
            "training_mode": "no_tool",
            "public_dataset_id": "fixture",
        },
        reference_answer=str(index + 1),
    )


def _source_plan(task: TeacherTask) -> TeacherPlan:
    answer = str(task.reference_answer)
    solution = TeacherCellPlan(
        cell_id="solution",
        semantic_text=f"The direct result is {answer}.",
        roles={CognitiveRole.PLAN: 0.4, CognitiveRole.HYPOTHESIS: 0.4},
        uncertainty=0.05,
        noise=0.01,
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
                ObjectRef.cell("solution"),
                1.0,
            ),
        ),
    )
    return TeacherPlan(
        task_id=task.task_id,
        final_answer=answer,
        frames=(TeacherFrame("initial", answer, (solution, conclusion)),),
    )

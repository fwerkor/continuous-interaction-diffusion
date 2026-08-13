from __future__ import annotations

import json

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
from cid.natural_interaction_training import (
    NaturalInteractionConfig,
    build_natural_interaction_augmentation,
)
from cid.state import CellLifecycle, CognitiveRole


def test_natural_interaction_augmentation_adds_grounded_output_and_schema_diversity(
    tmp_path,
) -> None:
    search = {
        "name": "workspace_search",
        "description": "search task-local documents",
        "arguments": ({"name": "query", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }
    read = {
        "name": "workspace_read",
        "description": "read a task-local document",
        "arguments": ({"name": "resource_id", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }
    task = TeacherTask(
        task_id="natural-source",
        prompt="Which record gives the requested result?",
        source_descriptors=(search, read),
        evidence=(
            TeacherEvidence(
                "search-results",
                "workspace_search",
                [{"resource_id": "doc-1", "title": "Relevant record"}],
                {"query": "Which record gives the requested result?"},
            ),
            TeacherEvidence(
                "support",
                "workspace_read",
                {
                    "resource_id": "doc-1",
                    "title": "Relevant record",
                    "sentences": ["The relevant record establishes the requested result as Alpha."],
                },
                {"resource_id": "doc-1"},
                depends_on=("search-results",),
            ),
        ),
        metadata={"training_mode": "tool_required", "task_kind": "fixture"},
        reference_answer="Alpha",
    )
    initial = TeacherFrame(
        "initial",
        "Searching task-local records.",
        (
            TeacherCellPlan(
                "search",
                "Need task-local evidence for the requested fact or relation.",
                {CognitiveRole.INFORMATION_NEED: 1.0, CognitiveRole.PLAN: 0.4},
                uncertainty=0.9,
                links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("workspace_search")),),
            ),
        ),
    )
    after_search = TeacherFrame(
        "after:search-results",
        "Reading the relevant record.",
        (
            TeacherCellPlan(
                "search",
                "The search identified the relevant record.",
                {CognitiveRole.PERCEPT: 1.0},
                uncertainty=0.1,
                lifecycle=CellLifecycle.STABLE,
                links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("workspace_search")),),
            ),
            TeacherCellPlan(
                "support",
                "Need task-relevant facts about the identified record.",
                {CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.8,
                links=(
                    CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("workspace_read")),
                    CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("search")),
                ),
            ),
        ),
    )
    answer_anchor = Anchor("text:alpha", AnchorKind.TEXT, "Alpha")
    final_cells = (
        TeacherCellPlan(
            "search",
            "The search identified the relevant record.",
            {CognitiveRole.PERCEPT: 1.0},
            uncertainty=0.05,
            lifecycle=CellLifecycle.STABLE,
            links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("workspace_search")),),
        ),
        TeacherCellPlan(
            "support",
            "Relevant record states that the requested result is Alpha.",
            {CognitiveRole.PERCEPT: 1.0},
            uncertainty=0.03,
            lifecycle=CellLifecycle.STABLE,
            links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("workspace_read")),),
        ),
        TeacherCellPlan(
            "answer",
            "Conclusion: Alpha",
            {CognitiveRole.CONCLUSION: 1.0},
            uncertainty=0.02,
            lifecycle=CellLifecycle.STABLE,
            anchors=(answer_anchor,),
            links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("support")),),
        ),
    )
    plan = TeacherPlan(
        task_id=task.task_id,
        final_answer="Alpha",
        frames=(
            initial,
            after_search,
            TeacherFrame("after:support", "Alpha", final_cells),
            TeacherFrame("final", "Alpha", final_cells),
        ),
        needs=(
            TeacherNeed(
                "search",
                "search",
                "search-results",
                "initial",
                "workspace_search",
                {"query": "Which record gives the requested result?"},
            ),
            TeacherNeed(
                "read",
                "support",
                "support",
                "after:search-results",
                "workspace_read",
                {"resource_id": "doc-1"},
            ),
        ),
    )
    assert review_teacher_plans((task,), (plan,))[0].accepted

    tasks_path = tmp_path / "tasks.jsonl"
    plans_path = tmp_path / "plans.jsonl"
    dump_teacher_tasks((task,), tasks_path)
    dump_teacher_plans((plan,), plans_path)
    manifest = build_natural_interaction_augmentation(
        ((tasks_path, plans_path),),
        tmp_path / "out",
        tmp_path / "reference.json",
        NaturalInteractionConfig(variants_per_task=1, seed=17),
    )

    assert manifest["semantic_tasks"] == 1
    assert manifest["long_form_targets"] == 1
    assert manifest["tasks_with_anchor"] == 1
    assert manifest["tasks_with_link"] == 1
    augmented_task = json.loads(
        (tmp_path / "out/natural-interaction-v1-teacher-tasks.jsonl").read_text().splitlines()[0]
    )
    augmented_plan = json.loads(
        (tmp_path / "out/natural-interaction-v1-teacher-plans.accepted.jsonl")
        .read_text()
        .splitlines()[0]
    )
    source_names = {item["name"] for item in augmented_task["source_descriptors"]}
    assert "workspace_search" not in source_names
    assert "workspace_read" not in source_names
    assert len(source_names) == 3
    assert len(augmented_plan["final_answer"]) >= 80
    assert "Alpha" in augmented_plan["final_answer"]
    assert augmented_plan["final_answer"] == augmented_plan["frames"][-1]["display"]
    assert "Need task-local evidence for the requested fact or relation." not in json.dumps(
        augmented_plan
    )

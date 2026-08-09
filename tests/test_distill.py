import json
from dataclasses import replace

import pytest

from cid.data import dump_jsonl, load_jsonl
from cid.distill import (
    TeacherCellPlan,
    TeacherEvidence,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    build_teacher_request,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_requests,
    dump_teacher_reviews,
    dump_teacher_tasks,
    load_teacher_tasks,
    review_teacher_plans,
    teacher_tasks_from_trajectories,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.synthetic import SyntheticConfig, generate_synthetic


def make_teacher_task_and_plan() -> tuple[TeacherTask, TeacherPlan]:
    source = {
        "name": "docs",
        "description": "read documentation",
        "arguments": ({"name": "key", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": True,
    }
    task = TeacherTask(
        task_id="teacher-1",
        prompt="Return the documented latency.",
        protected_facts={"output_rule": "use the documented value"},
        source_descriptors=(source,),
        evidence=(
            TeacherEvidence(
                evidence_id="e0",
                source="docs",
                value=37,
                arguments={"key": "latency_ms"},
                version="v1",
            ),
        ),
    )
    key_anchor = Anchor(
        anchor_id="a:key",
        kind=AnchorKind.TEXT,
        value="latency_ms",
        object_id="metric:latency",
    )
    value_anchor = Anchor(
        anchor_id="a:value",
        kind=AnchorKind.NUMBER,
        value=37,
        unit="ms",
    )
    initial = TeacherFrame(
        phase="initial",
        display="pending",
        cells=(
            TeacherCellPlan(
                cell_id="plan",
                semantic_text="Plan a documentation lookup.",
                roles={CognitiveRole.PLAN: 1.0},
            ),
        ),
    )
    pre = TeacherFrame(
        phase="pre",
        display="pending",
        cells=(
            TeacherCellPlan(
                cell_id="plan",
                semantic_text="The answer must use the documentation source.",
                roles={CognitiveRole.CONSTRAINT: 1.0},
            ),
            TeacherCellPlan(
                cell_id="need",
                semantic_text="Need the documented latency value.",
                roles={CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.9,
                noise=0.8,
                anchors=(key_anchor,),
                links=(
                    CognitiveLink(
                        relation=LinkRelation.REQUESTS,
                        target=ObjectRef.source("docs"),
                    ),
                ),
            ),
        ),
    )
    after = TeacherFrame(
        phase="after:e0",
        display="37 ms",
        cells=(
            TeacherCellPlan(
                cell_id="plan",
                semantic_text="The documentation requirement is satisfied.",
                roles={CognitiveRole.CONSTRAINT: 1.0},
                uncertainty=0.1,
                noise=0.1,
                lifecycle=CellLifecycle.STABLE,
            ),
            TeacherCellPlan(
                cell_id="need",
                semantic_text="The documented latency is 37 ms.",
                roles={CognitiveRole.CONCLUSION: 1.0},
                uncertainty=0.05,
                noise=0.1,
                lifecycle=CellLifecycle.STABLE,
                anchors=(value_anchor,),
                links=(
                    CognitiveLink(
                        relation=LinkRelation.OBSERVES,
                        target=ObjectRef.source("docs"),
                    ),
                ),
            ),
        ),
    )
    final = TeacherFrame(
        phase="final",
        display="37 ms",
        cells=after.cells,
    )
    plan = TeacherPlan(
        task_id=task.task_id,
        final_answer="37 ms",
        frames=(initial, pre, after, final),
        needs=(
            TeacherNeed(
                need_id="lookup",
                cell_id="need",
                evidence_id="e0",
                phase="pre",
                source="docs",
                arguments={"key": "latency_ms"},
            ),
        ),
    )
    return task, plan


def test_teacher_compiler_separates_semantics_from_event_timing(tmp_path) -> None:
    task, plan = make_teacher_task_and_plan()

    (trajectory,) = compile_teacher_plans(
        (task,),
        (plan,),
        TeacherScheduleConfig(
            thought_capacity=6,
            min_delay_steps=3,
            max_delay_steps=3,
            seed=11,
        ),
    )

    assert trajectory.events[0].arrival_step == 4
    assert trajectory.binding_targets[0].first_need_step == 1
    assert trajectory.binding_targets[0].executable_step == 1
    assert trajectory.binding_targets[0].argument_steps == {"key": 1}
    steps = sorted({target.step for target in trajectory.thought_targets})
    assert steps == list(range(7))

    need_lifecycle = {
        target.step: target.lifecycle
        for target in trajectory.thought_targets
        if target.cell_id == "need"
    }
    assert need_lifecycle[1] is CellLifecycle.ACTIVE
    assert need_lifecycle[2] is CellLifecycle.WAITING
    assert need_lifecycle[3] is CellLifecycle.WAITING
    assert need_lifecycle[4] is CellLifecycle.ACTIVE
    assert need_lifecycle[5] is CellLifecycle.STABLE
    assert need_lifecycle[6] is CellLifecycle.STABLE

    slots = {
        target.cell_id: target.slot
        for target in trajectory.thought_targets
        if target.step == 1
    }
    assert len(set(slots.values())) == 2
    assert all(0 <= slot < 6 for slot in slots.values())
    assert trajectory.target_display == "37 ms"
    assert trajectory.metadata["distillation"] == "teacher-semantic-plan-v1"

    path = tmp_path / "compiled.jsonl"
    dump_jsonl((trajectory,), path)
    assert load_jsonl(path) == (trajectory,)


def test_teacher_compiler_expands_counterfactual_schedule_variants() -> None:
    task, plan = make_teacher_task_and_plan()
    trajectories = compile_teacher_plans(
        (task,),
        (plan,),
        TeacherScheduleConfig(
            thought_capacity=6,
            min_delay_steps=1,
            max_delay_steps=4,
            variants_per_task=4,
            seed=17,
        ),
    )

    assert len(trajectories) == 4
    assert [item.example_id for item in trajectories] == [
        "teacher-1::schedule-00",
        "teacher-1::schedule-01",
        "teacher-1::schedule-02",
        "teacher-1::schedule-03",
    ]
    assert {item.metadata["semantic_task_id"] for item in trajectories} == {"teacher-1"}
    assert [item.metadata["schedule_variant"] for item in trajectories] == [0, 1, 2, 3]
    assert len({item.events[0].arrival_step for item in trajectories}) > 1


def test_teacher_request_forbids_timing_and_private_cot() -> None:
    task, _ = make_teacher_task_and_plan()
    request = build_teacher_request(task)

    assert request.task_id == task.task_id
    assert "Do NOT emit numeric timesteps" in request.prompt
    assert "Do not write private chain-of-thought" in request.prompt
    assert '"evidence_id": "e0"' in request.prompt


def test_teacher_review_checks_supported_public_reference_answers() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task,
        reference_answer="37 ms",
        metadata={"task_kind": "multi_hop_qa"},
    )
    (accepted,) = review_teacher_plans((public_task,), (plan,))
    assert accepted.accepted

    wrong_task = replace(public_task, reference_answer="42 ms")
    (rejected,) = review_teacher_plans((wrong_task,), (plan,))
    assert not rejected.accepted
    assert any("public reference answer" in reason for reason in rejected.reasons)


def test_teacher_review_accepts_equivalent_gsm8k_numeric_surface_form() -> None:
    task, plan = make_teacher_task_and_plan()
    numeric_task = replace(
        task,
        reference_answer="1,250",
        metadata={"task_kind": "math_word_problem"},
    )
    numeric_plan = replace(
        plan,
        final_answer="$1,250",
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display="$1,250")),
    )
    (review,) = review_teacher_plans((numeric_task,), (numeric_plan,))
    assert review.accepted


def test_teacher_plan_parser_rejects_timing_and_physical_slots() -> None:
    raw = {
        "task_id": "bad",
        "final_answer": "x",
        "frames": [
            {
                "phase": "initial",
                "display": "x",
                "step": 0,
                "cells": [
                    {
                        "cell_id": "c0",
                        "semantic_text": "summary",
                        "roles": {"plan": 1.0},
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="must not control"):
        TeacherPlan.from_dict(raw)

    raw["frames"][0].pop("step")
    raw["frames"][0]["cells"][0]["slot"] = 3
    with pytest.raises(ValueError, match="must not control"):
        TeacherPlan.from_dict(raw)

    raw["frames"][0]["cells"][0].pop("slot")
    raw["frames"][0]["cells"][0]["reasoning"] = "hidden transcript"
    with pytest.raises(ValueError, match="unsupported fields"):
        TeacherPlan.from_dict(raw)


def test_teacher_task_request_files_are_stable_and_drop_event_schedule(tmp_path) -> None:
    source = generate_synthetic(SyntheticConfig(count_per_family=1, seed=3, thought_capacity=8))
    tasks = teacher_tasks_from_trajectories(source)
    tasks_path = tmp_path / "tasks.jsonl"
    requests_path = tmp_path / "requests.jsonl"

    dump_teacher_tasks(tasks, tasks_path)
    dump_teacher_requests(tasks, requests_path)

    assert load_teacher_tasks(tasks_path) == tasks
    request_records = [json.loads(line) for line in requests_path.read_text().splitlines()]
    assert len(request_records) == len(tasks)
    assert {item["task_id"] for item in request_records} == {item.task_id for item in tasks}
    task_payload = json.loads(tasks_path.read_text().splitlines()[0])
    assert all("arrival_step" not in item for item in task_payload["evidence"])


def test_teacher_quality_review_rejects_future_leaks_and_bad_arguments() -> None:
    task, plan = make_teacher_task_and_plan()
    pre = plan.frames[1]
    leaking = replace(
        plan,
        frames=(
            plan.frames[0],
            replace(pre, display="37 ms"),
            *plan.frames[2:],
        ),
    )
    (review,) = review_teacher_plans((task,), (leaking,))
    assert not review.accepted
    assert any("leaks future evidence" in reason for reason in review.reasons)

    bad_need = replace(plan.needs[0], arguments={"key": "wrong-key"})
    bad_arguments = replace(plan, needs=(bad_need,))
    (review,) = review_teacher_plans((task,), (bad_arguments,))
    assert not review.accepted
    assert any("does not match supplied evidence" in reason for reason in review.reasons)

    with pytest.raises(ValueError, match="failed quality review"):
        compile_teacher_plans((task,), (leaking,))


def test_teacher_quality_review_deduplicates_semantic_plans(tmp_path) -> None:
    task, plan = make_teacher_task_and_plan()
    second_task = replace(task, task_id="teacher-2")
    second_plan = replace(plan, task_id="teacher-2")

    reviews = review_teacher_plans((task, second_task), (plan, second_plan))

    assert reviews[0].accepted
    assert not reviews[1].accepted
    assert reviews[0].fingerprint == reviews[1].fingerprint
    assert reviews[1].reasons == ("semantic duplicate of teacher-1",)

    plans_path = tmp_path / "plans.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    dump_teacher_plans((plan,), plans_path)
    dump_teacher_reviews(reviews, reviews_path)
    assert json.loads(plans_path.read_text())["task_id"] == "teacher-1"
    assert len(reviews_path.read_text().splitlines()) == 2

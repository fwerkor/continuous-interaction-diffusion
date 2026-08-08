import json

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
    dump_teacher_requests,
    dump_teacher_tasks,
    load_teacher_tasks,
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


def test_teacher_request_forbids_timing_and_private_cot() -> None:
    task, _ = make_teacher_task_and_plan()
    request = build_teacher_request(task)

    assert request.task_id == task.task_id
    assert "Do NOT emit numeric timesteps" in request.prompt
    assert "Do not write private chain-of-thought" in request.prompt
    assert '"evidence_id": "e0"' in request.prompt


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

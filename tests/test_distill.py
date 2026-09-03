import json
import random
from dataclasses import replace

import pytest

from cid.contracts import FreshnessDemand
from cid.data import DISPLAY_UNKNOWN_MARKER, dump_jsonl, load_jsonl
from cid.distill import (
    TeacherCellPlan,
    TeacherEvidence,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    _allocate_teacher_slots,
    _competition_math_answer_match,
    _competition_math_choice_match,
    _future_evidence_leaks,
    _visibility_contains_marker,
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
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectKind, ObjectRef
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
        display=DISPLAY_UNKNOWN_MARKER,
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
        display=DISPLAY_UNKNOWN_MARKER,
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
        target.cell_id: target.slot for target in trajectory.thought_targets if target.step == 1
    }
    assert len(set(slots.values())) == 2
    assert all(0 <= slot < 6 for slot in slots.values())
    assert trajectory.target_display == "37 ms"
    assert trajectory.metadata["distillation"] == "teacher-semantic-plan-v1"

    path = tmp_path / "compiled.jsonl"
    dump_jsonl((trajectory,), path)
    assert load_jsonl(path) == (trajectory,)


def test_teacher_compiler_preserves_one_need_with_multiple_affected_cells() -> None:
    task, plan = make_teacher_task_and_plan()
    need = replace(plan.needs[0], affected_cell_ids=("plan",))
    plan = replace(plan, needs=(need,))

    (trajectory,) = compile_teacher_plans(
        (task,),
        (plan,),
        TeacherScheduleConfig(thought_capacity=6, min_delay_steps=1, max_delay_steps=1, seed=19),
    )

    binding = trajectory.binding_targets[0]
    assert binding.owner_cell_id == "need"
    assert binding.target_cells == (ObjectRef.cell("need"), ObjectRef.cell("plan"))


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


def test_teacher_compiler_accepts_review_filtered_plan_subset() -> None:
    task, plan = make_teacher_task_and_plan()
    omitted_task = replace(task, task_id="teacher-2")

    trajectories = compile_teacher_plans((task, omitted_task), (plan,))

    assert len(trajectories) == 1
    assert trajectories[0].example_id == task.task_id


def test_teacher_compiler_supports_no_tool_refinement_phases() -> None:
    task = TeacherTask(
        task_id="no-tool-refine",
        prompt="Determine whether the stated implication chain reaches z.",
        metadata={"task_kind": "complex_logic_reasoning", "training_mode": "no_tool"},
        reference_answer="entailed",
    )
    frames = (
        TeacherFrame(
            phase="initial",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(
                TeacherCellPlan(
                    cell_id="state",
                    semantic_text="Start from the stated seed and preserve implication direction.",
                    roles={CognitiveRole.PLAN: 1.0},
                    uncertainty=0.7,
                ),
            ),
        ),
        TeacherFrame(
            phase="refine:0",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(
                TeacherCellPlan(
                    cell_id="state",
                    semantic_text="The reachable frontier advances through the first two rules.",
                    roles={CognitiveRole.HYPOTHESIS: 0.7, CognitiveRole.PLAN: 0.4},
                    uncertainty=0.4,
                ),
            ),
        ),
        TeacherFrame(
            phase="refine:1",
            display="entailed",
            cells=(
                TeacherCellPlan(
                    cell_id="state",
                    semantic_text=(
                        "The final directed edge reaches z; disconnected rules are irrelevant."
                    ),
                    roles={CognitiveRole.PERCEPT: 0.7, CognitiveRole.HYPOTHESIS: 0.4},
                    uncertainty=0.12,
                ),
            ),
        ),
        TeacherFrame(
            phase="final",
            display="entailed",
            cells=(
                TeacherCellPlan(
                    cell_id="state",
                    semantic_text="The complete directed chain reaches z.",
                    roles={CognitiveRole.CONCLUSION: 1.0},
                    uncertainty=0.02,
                    lifecycle=CellLifecycle.STABLE,
                ),
            ),
        ),
    )
    plan = TeacherPlan(task_id=task.task_id, final_answer="entailed", frames=frames)

    (trajectory,) = compile_teacher_plans((task,), (plan,))

    assert [target.step for target in trajectory.display_targets] == [0, 1, 2, 3]
    assert [target.text for target in trajectory.display_targets] == [
        DISPLAY_UNKNOWN_MARKER,
        DISPLAY_UNKNOWN_MARKER,
        "entailed",
        "entailed",
    ]
    assert review_teacher_plans((task,), (plan,))[0].accepted


def test_teacher_review_rejects_process_status_as_display_supervision() -> None:
    task = TeacherTask(
        task_id="status-display",
        prompt="Return the result.",
        metadata={"training_mode": "no_tool"},
        reference_answer="done",
    )
    cell = TeacherCellPlan(
        cell_id="state",
        semantic_text="The task state is represented compactly.",
        roles={CognitiveRole.PLAN: 1.0},
    )
    plan = TeacherPlan(
        task_id=task.task_id,
        final_answer="done",
        frames=(
            TeacherFrame("initial", "Reasoning.", (cell,)),
            TeacherFrame(
                "final",
                "done",
                (replace(cell, roles={CognitiveRole.CONCLUSION: 1.0}),),
            ),
        ),
    )

    review = review_teacher_plans((task,), (plan,))[0]

    assert not review.accepted
    assert any("process status" in reason for reason in review.reasons)


def test_teacher_compiler_rejects_refinement_phases_with_external_evidence() -> None:
    task, plan = make_teacher_task_and_plan()
    refine = replace(plan.frames[0], phase="refine:0")
    invalid = replace(plan, frames=(plan.frames[0], refine, *plan.frames[1:]))

    review = review_teacher_plans((task,), (invalid,))[0]
    assert not review.accepted
    assert any("only supported for no-evidence" in reason for reason in review.reasons)
    with pytest.raises(ValueError, match="only supported for no-evidence"):
        compile_teacher_plans((task,), (invalid,))


def test_teacher_compiler_tolerates_entity_anchor_case_variants() -> None:
    task, plan = make_teacher_task_and_plan()
    final = plan.frames[-1]
    conclusion = final.cells[-1]
    original_anchor = conclusion.anchors[0]
    entity_anchor = Anchor(
        anchor_id="entity:person",
        kind=AnchorKind.ENTITY,
        value="Alberto De Martino",
        object_id="entity:alberto de martino",
    )
    variant_anchor = replace(entity_anchor, value="Alberto de Martino")
    after = replace(
        plan.frames[-2],
        cells=(
            *plan.frames[-2].cells[:-1],
            replace(conclusion, anchors=(entity_anchor, original_anchor)),
        ),
    )
    final = replace(
        final,
        cells=(
            *final.cells[:-1],
            replace(conclusion, anchors=(variant_anchor, original_anchor)),
        ),
    )
    case_variant_plan = replace(plan, frames=(*plan.frames[:-2], after, final))

    (trajectory,) = compile_teacher_plans((task,), (case_variant_plan,))

    catalog = {entry.anchor.anchor_id: entry.anchor for entry in trajectory.grounding_catalog}
    assert catalog["entity:person"].value == "Alberto De Martino"


def test_teacher_slot_allocator_recycles_retired_annotation_slots() -> None:
    def cell(cell_id: str, lifecycle: CellLifecycle) -> TeacherCellPlan:
        return TeacherCellPlan(
            cell_id=cell_id,
            semantic_text=cell_id,
            roles={CognitiveRole.PERCEPT: 1.0},
            lifecycle=lifecycle,
        )

    frames = [
        TeacherFrame(
            phase="initial",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(cell("a", CellLifecycle.ACTIVE),),
        ),
        TeacherFrame(
            phase="p1",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(
                cell("a", CellLifecycle.RETIRED),
                cell("b", CellLifecycle.ACTIVE),
            ),
        ),
        TeacherFrame(
            phase="p2",
            display="done",
            cells=(
                cell("a", CellLifecycle.RETIRED),
                cell("b", CellLifecycle.RETIRED),
                cell("c", CellLifecycle.ACTIVE),
            ),
        ),
    ]

    slots, visible = _allocate_teacher_slots(frames, 2, random.Random(0))

    assert visible["p1"] == frozenset({"a", "b"})
    assert visible["p2"] == frozenset({"b", "c"})
    assert slots["initial"]["a"] == slots["p1"]["a"]
    assert slots["p2"]["c"] == slots["p1"]["a"]


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


def test_teacher_review_accepts_country_nationality_surface_aliases() -> None:
    task, plan = make_teacher_task_and_plan()
    for prediction, reference in (
        ("United States", "American"),
        ("American", "America"),
        ("United States (American)", "America"),
        ("Indian", "India"),
        ("English", "British"),
        ("Cypriot", "Cyprus"),
        ("Cambodian", "Cambodia"),
        ("Australian", "Australia"),
        ("England", "British"),
    ):
        public_task = replace(
            task,
            reference_answer=reference,
            metadata={"task_kind": "multi_hop_qa"},
        )
        public_plan = replace(
            plan,
            final_answer=prediction,
            frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
        )
        (review,) = review_teacher_plans((public_task,), (public_plan,))
        assert review.accepted


def test_teacher_review_accepts_clear_place_surface_aliases() -> None:
    task, plan = make_teacher_task_and_plan()
    for prediction, reference in (
        ("Mumbai", "Bombay"),
        ("Mumbai, India", "Bombay"),
        ("Paris", "Parisian"),
        ("The Tiger’s Claw", "The Tiger's Claw"),
        ("Marienburg, German Empire", "Malbork"),
        ("Wallachia", "Principality of Wallachia"),
        ("Academy Award for Best Picture", "Oscar for Best Picture"),
    ):
        public_task = replace(
            task,
            reference_answer=reference,
            metadata={"task_kind": "multi_hop_qa"},
        )
        public_plan = replace(
            plan,
            final_answer=prediction,
            frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
        )
        (review,) = review_teacher_plans((public_task,), (public_plan,))
        assert review.accepted


def test_teacher_review_accepts_equivalent_calendar_date_order() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task,
        reference_answer="22 June 2014",
        metadata={"task_kind": "multi_hop_qa"},
    )
    public_plan = replace(
        plan,
        final_answer="June 22, 2014.",
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display="June 22, 2014.")),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert review.accepted


@pytest.mark.parametrize(
    ("prediction", "reference"),
    (
        ("2 May 2008.", "May 2"),
        ("April 17, 2008.", "17 April"),
        ("13.", "thirteen"),
        ("ABC.", "American Broadcasting Company"),
        ("Northwest.", "north-west"),
        ("About 196,000 to 600,000.", "196,000-600,000"),
        ("376 CE.", "in 376"),
        (
            "Classic Albums: Iron Maiden – The Number of the Beast.",
            "Classic Albums: Iron Maiden -- The Number of the Beast",
        ),
    ),
)
def test_teacher_review_accepts_conservative_qa_surface_equivalences(
    prediction: str,
    reference: str,
) -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task,
        reference_answer=reference,
        metadata={"task_kind": "multi_hop_qa"},
    )
    public_plan = replace(
        plan,
        final_answer=prediction,
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert review.accepted


def test_teacher_review_rejects_numeric_reference_collision_in_contradictory_text() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task,
        reference_answer="18-20",
        metadata={"task_kind": "multi_hop_qa"},
    )
    prediction = "There is no minimum age; the evidence only mentions people aged 18–20."
    public_plan = replace(
        plan,
        final_answer=prediction,
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert not review.accepted


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


def test_synthetic_teacher_tasks_reuse_persistent_bindings_for_updates() -> None:
    examples = generate_synthetic(SyntheticConfig(count_per_family=1, seed=9, thought_capacity=8))
    tasks = teacher_tasks_from_trajectories(examples)
    by_family = {str(task.metadata["family"]): task for task in tasks}

    dynamic = by_family["dynamic_state"]
    assert [item.requires_need for item in dynamic.evidence] == [True, False]
    assert dynamic.evidence[1].depends_on == ("evidence-0",)

    streaming = by_family["streaming_evidence"]
    assert [item.requires_need for item in streaming.evidence] == [True, False]
    assert streaming.evidence[1].depends_on == ("evidence-0",)

    competing = by_family["competing_sources"]
    assert [item.requires_need for item in competing.evidence] == [True, True]
    assert all(item.depends_on == () for item in competing.evidence)


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


def test_teacher_quality_review_allows_repeated_future_evidence_value() -> None:
    examples = generate_synthetic(SyntheticConfig(count_per_family=1, seed=31, thought_capacity=8))
    streaming_example = next(
        example for example in examples if example.metadata["family"] == "streaming_evidence"
    )
    task = teacher_tasks_from_trajectories((streaming_example,))[0]
    repeated = replace(
        task,
        evidence=(
            task.evidence[0],
            replace(task.evidence[1], value=task.evidence[0].value),
        ),
    )
    first_value = str(repeated.evidence[0].value)
    frames = (
        TeacherFrame(
            phase="initial",
            display=DISPLAY_UNKNOWN_MARKER,
            cells=(
                TeacherCellPlan(
                    cell_id="stream",
                    semantic_text="Need both stream chunks.",
                    roles={CognitiveRole.INFORMATION_NEED: 1.0},
                ),
            ),
        ),
        TeacherFrame(
            phase="after:evidence-0",
            display=first_value,
            cells=(
                TeacherCellPlan(
                    cell_id="stream",
                    semantic_text=f"First chunk: {first_value}.",
                    roles={CognitiveRole.PERCEPT: 1.0},
                    lifecycle=CellLifecycle.STABLE,
                ),
            ),
        ),
        TeacherFrame(
            phase="after:evidence-1",
            display=f"{first_value} {first_value}",
            cells=(
                TeacherCellPlan(
                    cell_id="stream",
                    semantic_text=f"Both chunks: {first_value}, {first_value}.",
                    roles={CognitiveRole.CONCLUSION: 1.0},
                    lifecycle=CellLifecycle.STABLE,
                ),
            ),
        ),
    )
    plan = TeacherPlan(
        task_id=repeated.task_id,
        final_answer=f"{first_value} {first_value}",
        frames=frames,
        needs=(
            TeacherNeed(
                need_id="stream",
                cell_id="stream",
                evidence_id="evidence-0",
                phase="initial",
                source="stream",
                arguments=dict(repeated.evidence[0].arguments),
                freshness=FreshnessDemand.ALWAYS,
            ),
        ),
    )

    (review,) = review_teacher_plans((repeated,), (plan,))
    assert not any("leaks future evidence" in reason for reason in review.reasons)


def test_future_leak_check_allows_future_value_already_visible_inside_record() -> None:
    task = TeacherTask(
        task_id="nested-record-value",
        prompt="Read the record, then solve the equation.",
        source_descriptors=(
            {
                "name": "record_lookup",
                "description": "read a record",
                "arguments": ({"name": "key", "kind": "string", "required": True},),
                "cacheable": True,
                "dynamic": False,
                "versioned": False,
            },
            {
                "name": "symbolic_math",
                "description": "solve an equation",
                "arguments": ({"name": "expression", "kind": "string", "required": True},),
                "cacheable": True,
                "dynamic": False,
                "versioned": False,
            },
        ),
        evidence=(
            TeacherEvidence(
                evidence_id="record",
                source="record_lookup",
                value={"a": 25, "b": -48, "target": -1248},
                arguments={"key": "equation"},
            ),
            TeacherEvidence(
                evidence_id="solution",
                source="symbolic_math",
                value="-48",
                arguments={"expression": "25*x-48=-1248"},
                depends_on=("record",),
            ),
        ),
    )
    frames = [
        TeacherFrame(
            phase="after:record",
            display="record loaded",
            cells=(
                TeacherCellPlan(
                    cell_id="record",
                    semantic_text="Resolved fields: a=25; b=-48; target=-1248.",
                    roles={CognitiveRole.PERCEPT: 1.0},
                ),
            ),
        )
    ]

    assert _future_evidence_leaks(task, frames) == ()


def test_future_leak_check_allows_future_text_already_visible_inside_search_title() -> None:
    task = TeacherTask(
        task_id="search-title-overlap",
        prompt="Find the group associated with the musical.",
        source_descriptors=(
            {
                "name": "workspace_search",
                "description": "search local records",
                "arguments": ({"name": "query", "kind": "string", "required": True},),
                "cacheable": True,
                "dynamic": False,
                "versioned": False,
            },
            {
                "name": "workspace_read",
                "description": "read one local record",
                "arguments": ({"name": "resource_id", "kind": "string", "required": True},),
                "cacheable": True,
                "dynamic": False,
                "versioned": False,
            },
        ),
        evidence=(
            TeacherEvidence(
                evidence_id="search-results",
                source="workspace_search",
                value=[{"resource_id": "doc-0", "title": "Mamma Mia! (film)"}],
                arguments={"query": "Mamma Mia"},
            ),
            TeacherEvidence(
                evidence_id="support-1",
                source="workspace_read",
                value={
                    "resource_id": "doc-0",
                    "title": "Mamma Mia! (film)",
                    "sentences": ["Mamma Mia!", "The musical uses songs by ABBA."],
                },
                arguments={"resource_id": "doc-0"},
                depends_on=("search-results",),
            ),
        ),
    )
    frames = (
        TeacherFrame(
            phase="after:search-results",
            display="search results loaded",
            cells=(
                TeacherCellPlan(
                    cell_id="search",
                    semantic_text="Relevant records: Mamma Mia! (film).",
                    roles={CognitiveRole.PERCEPT: 1.0},
                ),
            ),
        ),
    )

    assert _future_evidence_leaks(task, frames) == ()


def test_future_leak_check_ignores_terminal_punctuation_on_visible_search_title() -> None:
    task = TeacherTask(
        task_id="search-title-terminal-punctuation",
        prompt="Find the country from the task-local records.",
        source_descriptors=(
            {
                "name": "workspace_search",
                "description": "search local records",
                "arguments": ({"name": "query", "kind": "string", "required": True},),
            },
            {
                "name": "workspace_read",
                "description": "read one local record",
                "arguments": ({"name": "resource_id", "kind": "string", "required": True},),
            },
        ),
        evidence=(
            TeacherEvidence(
                evidence_id="search-results",
                source="workspace_search",
                value=[{"resource_id": "doc-0", "title": "Bolesław I the Brave"}],
                arguments={"query": "country"},
            ),
            TeacherEvidence(
                evidence_id="support",
                source="workspace_read",
                value={"resource_id": "doc-0", "title": "Bolesław I the Brave."},
                arguments={"resource_id": "doc-0"},
                depends_on=("search-results",),
            ),
        ),
    )
    frames = (
        TeacherFrame(
            phase="after:search-results",
            display="Relevant record identified.",
            cells=(
                TeacherCellPlan(
                    cell_id="search",
                    semantic_text="Relevant records: Bolesław I the Brave.",
                    roles={CognitiveRole.PERCEPT: 1.0},
                ),
            ),
        ),
    )

    assert _future_evidence_leaks(task, frames) == ()


def test_future_leak_text_markers_do_not_match_inside_unrelated_words() -> None:
    assert not _visibility_contains_marker("resolve the request before answering", "ann")
    assert not _visibility_contains_marker("use external information after retrieval", "val")
    assert _visibility_contains_marker("retrieved answer: ann", "ann")
    assert _visibility_contains_marker("candidate value is val", "val")


def test_teacher_quality_review_does_not_match_numeric_future_value_as_substring() -> None:
    source = {
        "name": "calculator",
        "description": "evaluate an expression",
        "arguments": ({"name": "expression", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }
    task = TeacherTask(
        task_id="numeric-substring",
        prompt="Compute two dependent values.",
        source_descriptors=(source,),
        evidence=(
            TeacherEvidence(
                evidence_id="first",
                source="calculator",
                value="137933",
                arguments={"expression": "137933"},
            ),
            TeacherEvidence(
                evidence_id="second",
                source="calculator",
                value="933",
                arguments={"expression": "137933-137000"},
                depends_on=("first",),
            ),
        ),
        metadata={"task_kind": "computational_reasoning"},
        reference_answer="933",
    )
    plan = TeacherPlan(
        task_id=task.task_id,
        final_answer="933",
        frames=(
            TeacherFrame(
                phase="initial",
                display=DISPLAY_UNKNOWN_MARKER,
                cells=(
                    TeacherCellPlan(
                        cell_id="work",
                        semantic_text="Need the first exact value.",
                        roles={CognitiveRole.INFORMATION_NEED: 1.0},
                    ),
                ),
            ),
            TeacherFrame(
                phase="after:first",
                display="137933",
                cells=(
                    TeacherCellPlan(
                        cell_id="first",
                        semantic_text="First result: 137933.",
                        roles={CognitiveRole.PERCEPT: 1.0},
                        lifecycle=CellLifecycle.STABLE,
                    ),
                    TeacherCellPlan(
                        cell_id="work",
                        semantic_text="Need the dependent difference.",
                        roles={CognitiveRole.INFORMATION_NEED: 1.0},
                    ),
                ),
            ),
            TeacherFrame(
                phase="after:second",
                display="933",
                cells=(
                    TeacherCellPlan(
                        cell_id="first",
                        semantic_text="First result: 137933.",
                        roles={CognitiveRole.PERCEPT: 1.0},
                        lifecycle=CellLifecycle.STABLE,
                    ),
                    TeacherCellPlan(
                        cell_id="work",
                        semantic_text="Dependent difference resolved.",
                        roles={CognitiveRole.PLAN: 1.0},
                        lifecycle=CellLifecycle.RETIRED,
                    ),
                    TeacherCellPlan(
                        cell_id="answer",
                        semantic_text="Final answer: 933.",
                        roles={CognitiveRole.CONCLUSION: 1.0},
                        lifecycle=CellLifecycle.STABLE,
                    ),
                ),
            ),
        ),
        needs=(
            TeacherNeed(
                need_id="first",
                cell_id="work",
                evidence_id="first",
                phase="initial",
                source="calculator",
                arguments={"expression": "137933"},
            ),
            TeacherNeed(
                need_id="second",
                cell_id="work",
                evidence_id="second",
                phase="after:first",
                source="calculator",
                arguments={"expression": "137933-137000"},
            ),
        ),
    )

    (review,) = review_teacher_plans((task,), (plan,))
    assert review.accepted


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


def test_teacher_review_accepts_boolean_answer_with_explanation() -> None:
    task, plan = make_teacher_task_and_plan()
    for prediction, reference in (
        ("No—one is Italian and one is Indian.", "no"),
        ("Yes. Both are American.", "yes"),
    ):
        public_task = replace(
            task, reference_answer=reference, metadata={"task_kind": "multi_hop_qa"}
        )
        public_plan = replace(
            plan,
            final_answer=prediction,
            frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
        )
        (review,) = review_teacher_plans((public_task,), (public_plan,))
        assert review.accepted


def test_teacher_review_accepts_calendar_date_with_qualifier() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task, reference_answer="May 15, 1958", metadata={"task_kind": "multi_hop_qa"}
    )
    prediction = "15 May 1958, for co-director Hampton Del Ruth."
    public_plan = replace(
        plan,
        final_answer=prediction,
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert review.accepted


def test_teacher_review_accepts_armenian_country_surface_alias() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(task, reference_answer="Armenian", metadata={"task_kind": "multi_hop_qa"})
    prediction = "Armenia"
    public_plan = replace(
        plan,
        final_answer=prediction,
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert review.accepted


def test_teacher_review_rejects_insufficiency_even_when_reference_token_appears() -> None:
    task, plan = make_teacher_task_and_plan()
    public_task = replace(
        task,
        reference_answer="India",
        metadata={"task_kind": "multi_hop_qa"},
    )
    prediction = (
        "The visible evidence says the director works in India, but does not explicitly state "
        "his nationality or country of origin."
    )
    public_plan = replace(
        plan,
        final_answer=prediction,
        frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
    )
    (review,) = review_teacher_plans((public_task,), (public_plan,))
    assert not review.accepted


def test_teacher_review_accepts_supported_answer_before_insufficiency_caveat() -> None:
    task, plan = make_teacher_task_and_plan()
    cases = (
        (
            "A heart attack, for one co-director; the visible evidence does not give the "
            "other's cause.",
            "heart attack",
        ),
        (
            "Bob Weir is American; the visible evidence does not state the co-composer's "
            "nationality.",
            "American",
        ),
    )
    for prediction, reference in cases:
        public_task = replace(
            task,
            reference_answer=reference,
            metadata={"task_kind": "multi_hop_qa"},
        )
        public_plan = replace(
            plan,
            final_answer=prediction,
            frames=(*plan.frames[:-1], replace(plan.frames[-1], display=prediction)),
        )
        (review,) = review_teacher_plans((public_task,), (public_plan,))
        assert review.accepted


def test_competition_math_answer_match_handles_latex_equivalence() -> None:
    assert _competition_math_answer_match(r"\frac{7}{3}", "7/3")
    assert _competition_math_answer_match(
        r"\frac{1033+120\sqrt{3}}{4}",
        r"\frac{1033}{4}+30\sqrt{3}",
    )
    assert _competition_math_answer_match(r"-\frac{\sqrt{2}}{2}", r"-\frac{1}{\sqrt{2}}")
    assert _competition_math_answer_match("55", r"55^\circ")
    assert _competition_math_answer_match("18", r"18\text{ degrees}")
    assert _competition_math_answer_match(r"\frac{\sqrt{7}}{3}", r"\frac{\sqrt7}{3}")
    assert _competition_math_answer_match(r"\frac{4}{3}", r"\frac43")
    assert _competition_math_answer_match("Wednesday", r"\text{Wednesday}")
    assert _competition_math_answer_match(
        r"y^2-4x-6y+9=0",
        r"-4x+y^2-6y+9=0",
    )
    assert _competition_math_answer_match("1", "k = 1")
    assert _competition_math_answer_match("1722", r"1,\!722")
    assert _competition_math_answer_match("1722", r"1{,}722")
    assert _competition_math_answer_match(
        r"\begin{pmatrix}1&-1\\1&1\end{pmatrix}",
        r"\begin{pmatrix}1&-1\\1&\phantom{-}1\end{pmatrix}",
    )
    assert _competition_math_answer_match(
        r"\begin{pmatrix}-\frac{20}{13}\\-\frac{4}{13}\end{pmatrix}",
        r"\begin{pmatrix}-20/13\\-4/13\end{pmatrix}",
    )
    assert _competition_math_answer_match(
        "(6,-17)",
        r"\begin{pmatrix}6\\-17\end{pmatrix}",
    )
    assert _competition_math_answer_match(
        r"(\frac{11}{5},\frac25,5)",
        r"\left(\frac{11}{5},\frac{2}{5},5\right)",
    )
    assert _competition_math_answer_match("-11,-1,1,11", "11,-1,-11,1")
    assert _competition_math_answer_match(r"(\frac{8}{3},3]", "(8/3,3]")
    assert _competition_math_answer_match(
        r"(-\frac{9}{2},-2)\cup(\frac{1-\sqrt5}{2},\frac{1+\sqrt5}{2})",
        r"((1-\sqrt5)/2,(1+\sqrt5)/2)\cup(-9/2,-2)",
    )
    assert _competition_math_answer_match("2400", r"2400\mbox{ cm}^2")
    assert _competition_math_answer_match("1101001", r"1101001_2")
    assert _competition_math_answer_match("306956.63", r"\$306,\!956.63")
    assert _competition_math_answer_match("7", r"7\text{ hours}.")
    assert _competition_math_answer_match("6,8,10", r"6,8,\text{ and }10")
    assert _competition_math_answer_match("6,8,10", r"6,8\text{, and }10")
    assert _competition_math_answer_match(
        r"\frac1{13}\begin{pmatrix}4&-6\\-6&9\end{pmatrix}",
        r"\begin{pmatrix}4/13&-6/13\\-6/13&9/13\end{pmatrix}",
    )
    assert _competition_math_choice_match(
        r"$\textbf{(A)}\ \tan \theta = \theta\qquad \textbf{(B)}\ \tan \theta = 2\theta$",
        "B",
        r"\tan \theta = 2\theta",
    )
    assert _competition_math_choice_match(
        r"$\text{(D) }22\quad \text{(E) }23$",
        "E",
        "23",
    )
    assert _competition_math_choice_match(
        "(A) 29 (B) 39 (C) 48 (D) 56 (E) 62",
        "A",
        "29",
    )
    assert not _competition_math_answer_match(r"\frac{7}{3}", r"\frac{8}{3}")


def _python_review_task_and_plan(source: str) -> tuple[TeacherTask, TeacherPlan]:
    task = TeacherTask(
        task_id="python-review-task",
        prompt="Write inc(x).",
        metadata={
            "task_kind": "python_programming",
            "public_tests": ["assert inc(1) == 2", "assert inc(-1) == 0"],
            "public_test_setup_code": "",
        },
        reference_answer="hidden reference code is not used by review",
    )
    frame = TeacherFrame(
        phase="initial",
        display=source,
        cells=(
            TeacherCellPlan(
                cell_id="implementation",
                semantic_text="Implement the visible Python contract.",
                roles={CognitiveRole.PLAN: 0.5},
            ),
            TeacherCellPlan(
                cell_id="answer",
                semantic_text="Public contract implementation.",
                roles={CognitiveRole.CONCLUSION: 1.0},
                links=(
                    CognitiveLink(
                        relation=LinkRelation.DERIVED_FROM,
                        target=ObjectRef(kind=ObjectKind.CELL, identifier="implementation"),
                    ),
                ),
            ),
        ),
    )
    return task, TeacherPlan(task_id=task.task_id, final_answer=source, frames=(frame,))


def test_python_programming_review_executes_public_tests_in_isolation() -> None:
    task, plan = _python_review_task_and_plan("def inc(x):\n    return x + 1")
    (review,) = review_teacher_plans((task,), (plan,))
    assert review.accepted

    _, wrong = _python_review_task_and_plan("def inc(x):\n    return x + 2")
    (review,) = review_teacher_plans((task,), (wrong,))
    assert not review.accepted
    assert any("fails public tests" in reason for reason in review.reasons)


def test_python_programming_review_rejects_unsafe_imports() -> None:
    task, plan = _python_review_task_and_plan("import os\ndef inc(x):\n    return x + 1")
    (review,) = review_teacher_plans((task,), (plan,))
    assert not review.accepted
    assert any("unsupported module 'os'" in reason for reason in review.reasons)

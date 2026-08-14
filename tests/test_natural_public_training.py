from cid.distill import TeacherScheduleConfig, compile_teacher_plans, review_teacher_plans
from cid.natural_public_training import _plan_for_task, _record_to_task


def test_natural_public_tool_task_builds_grounded_causal_plan() -> None:
    source = {
        "id": "fixture",
        "repo": "fixture/repo",
        "revision": "abc",
        "license": "Apache-2.0",
    }
    record = {
        "row_key": "row-1",
        "prompt": "Who wrote the example paper?",
        "answer": "Ada Example",
        "mode": "tool_required",
        "task_kind": "natural_paper_qa",
        "tool_source": "paper_lookup",
        "tool_arguments": {"paper": "Example", "query": "Who wrote it?"},
        "evidence": {"paper": "Example", "excerpts": ["Ada Example wrote the paper."]},
        "percept": "Paper evidence identifies Ada Example as the author.",
        "metadata": {},
    }

    task = _record_to_task(record, source)
    plan = _plan_for_task(task, record)
    (review,) = review_teacher_plans((task,), (plan,))
    assert review.accepted

    trajectories = compile_teacher_plans(
        (task,),
        (plan,),
        TeacherScheduleConfig(variants_per_task=2, seed=7),
    )
    assert len(trajectories) == 2
    assert all(example.events for example in trajectories)
    assert all(example.grounding_targets for example in trajectories)
    assert all(example.binding_targets for example in trajectories)


def test_natural_public_no_tool_task_preserves_human_target() -> None:
    source = {
        "id": "fixture-chat",
        "repo": "fixture/chat",
        "revision": "def",
        "license": "Apache-2.0",
    }
    record = {
        "row_key": "message-1",
        "prompt": "Explain recursion with a small example.",
        "answer": "Recursion is when a function calls itself on a smaller version of the problem.",
        "mode": "no_tool",
        "task_kind": "natural_instruction",
        "metadata": {},
    }

    task = _record_to_task(record, source)
    plan = _plan_for_task(task, record)
    (review,) = review_teacher_plans((task,), (plan,))
    assert review.accepted
    assert plan.final_answer == record["answer"]
    assert not plan.needs

    (trajectory,) = compile_teacher_plans(
        (task,),
        (plan,),
        TeacherScheduleConfig(variants_per_task=1, seed=11),
    )
    assert not trajectory.events
    assert trajectory.target_display == record["answer"]

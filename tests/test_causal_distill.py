from cid.causal_distill import build_causal_teacher_job
from cid.distill import TeacherEvidence, TeacherTask


def _retrieval_task() -> TeacherTask:
    descriptors = (
        {
            "name": "workspace_search",
            "description": "search",
            "arguments": ({"name": "query", "kind": "string", "required": True},),
        },
        {
            "name": "workspace_read",
            "description": "read",
            "arguments": (
                {"name": "resource_id", "kind": "string", "required": True},
            ),
        },
    )
    return TeacherTask(
        task_id="causal-1",
        prompt="Compare A and B.",
        source_descriptors=descriptors,
        evidence=(
            TeacherEvidence(
                evidence_id="search-results",
                source="workspace_search",
                value=[{"resource_id": "doc-a"}, {"resource_id": "doc-b"}],
                arguments={"query": "Compare A and B."},
            ),
            TeacherEvidence(
                evidence_id="support-a",
                source="workspace_read",
                value="A was released in 2001.",
                arguments={"resource_id": "doc-a"},
                depends_on=("search-results",),
            ),
            TeacherEvidence(
                evidence_id="support-b",
                source="workspace_read",
                value="B was released in 2005.",
                arguments={"resource_id": "doc-b"},
                depends_on=("search-results",),
            ),
        ),
        reference_answer="A",
    )


def test_causal_teacher_job_only_unlocks_dependency_ready_evidence() -> None:
    job = build_causal_teacher_job(_retrieval_task())

    assert [stage.phase for stage in job.stages] == [
        "initial",
        "after:search-results",
        "after:support-a",
        "after:support-b",
    ]
    assert [item["evidence_id"] for item in job.stages[0].available_evidence] == [
        "search-results"
    ]
    assert {item["evidence_id"] for item in job.stages[1].available_evidence} == {
        "support-a",
        "support-b",
    }
    assert job.stages[2].available_evidence == ()
    assert job.stages[-1].terminal is True


def test_causal_teacher_stage_never_contains_future_evidence_value() -> None:
    job = build_causal_teacher_job(_retrieval_task())

    initial = job.stages[0].to_dict()
    after_search = job.stages[1].to_dict()
    after_a = job.stages[2].to_dict()

    assert initial["arrived_evidence"] is None
    assert "A was released in 2001." not in str(initial)
    assert "B was released in 2005." not in str(initial)
    assert after_search["arrived_evidence"]["evidence_id"] == "search-results"
    assert "A was released in 2001." not in str(after_search)
    assert "B was released in 2005." not in str(after_search)
    assert after_a["arrived_evidence"]["value"] == "A was released in 2001."
    assert "B was released in 2005." not in str(after_a)


def test_causal_teacher_job_hides_reference_answer_from_base_task() -> None:
    job = build_causal_teacher_job(_retrieval_task())

    assert "reference_answer" not in job.task
    assert job.task["prompt"] == "Compare A and B."
    assert "evidence" not in job.task


def test_no_evidence_task_is_single_terminal_causal_stage() -> None:
    task = TeacherTask(
        task_id="plain",
        prompt="What is 2 + 3?",
        reference_answer="5",
    )
    job = build_causal_teacher_job(task)

    assert len(job.stages) == 1
    assert job.stages[0].phase == "initial"
    assert job.stages[0].terminal is True
    assert job.stages[0].available_evidence == ()

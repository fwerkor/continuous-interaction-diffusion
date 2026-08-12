import json

import pytest

from cid.causal_distill import dump_causal_teacher_jobs
from cid.distill import (
    TeacherEvidence,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    dump_teacher_tasks,
    load_teacher_plans,
    review_teacher_plans,
)
from cid.state import CellLifecycle
from cid.teacher_agent import checkout_teacher_agent_batch, commit_teacher_agent_batch
from cid.teacher_wave import (
    dump_teacher_wave_state,
    export_teacher_wave,
    finalize_teacher_wave,
    import_teacher_wave,
    load_teacher_wave_state,
    teacher_wave_status,
)


def test_dump_teacher_wave_state_is_atomic_on_write_failure(tmp_path) -> None:
    state_path = tmp_path / "state.jsonl"
    state_path.write_text("sentinel\n", encoding="utf-8")

    class Record:
        def __init__(self, task_id: str, *, fail: bool = False) -> None:
            self.task_id = task_id
            self.stage_index = 0
            self.fail = fail

        def to_dict(self):
            if self.fail:
                raise RuntimeError("synthetic serialization failure")
            return {"task_id": self.task_id, "stage_index": self.stage_index}

    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        dump_teacher_wave_state((Record("a"), Record("b", fail=True)), state_path)

    assert state_path.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.glob(".state.jsonl.*.tmp")) == []


def _task() -> TeacherTask:
    descriptors = (
        {
            "name": "workspace_search",
            "description": "Search documents.",
            "arguments": ({"name": "query", "kind": "string", "required": True},),
        },
        {
            "name": "workspace_read",
            "description": "Read a document.",
            "arguments": ({"name": "resource_id", "kind": "string", "required": True},),
        },
    )
    return TeacherTask(
        task_id="wave-task",
        prompt="Which item was released first, A or B?",
        source_descriptors=descriptors,
        evidence=(
            TeacherEvidence(
                evidence_id="search-results",
                source="workspace_search",
                value=[
                    {"resource_id": "doc-a", "title": "A"},
                    {"resource_id": "doc-b", "title": "B"},
                ],
                arguments={"query": "Which item was released first, A or B?"},
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
        metadata={"task_kind": "multi_hop_qa"},
        reference_answer="A",
    )


def _cell(cell_id, text, roles, *, uncertainty=0.5, lifecycle="active"):
    return {
        "cell_id": cell_id,
        "semantic_text": text,
        "roles": roles,
        "uncertainty": uncertainty,
        "noise": 0.3,
        "lifecycle": lifecycle,
        "anchors": [],
        "links": [],
    }


def _write_responses(path, request_path, output_by_phase):
    requests = [json.loads(line) for line in request_path.read_text().splitlines()]
    path.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "output": output_by_phase[request["phase"]],
                }
            )
            + "\n"
            for request in requests
        )
    )


def test_teacher_wave_round_trip_and_resume(tmp_path) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    jobs_path = tmp_path / "jobs.jsonl"
    state_path = tmp_path / "state.jsonl"
    dump_teacher_tasks((task,), tasks_path)
    dump_causal_teacher_jobs((task,), jobs_path)
    assert teacher_wave_status(jobs_path, state_path) == {
        "jobs": 1,
        "total_stages": 4,
        "completed_stages": 0,
        "complete_tasks": 0,
        "incomplete_tasks": 1,
        "next_phase_counts": {"initial": 1},
    }

    outputs = {
        "initial": {
            "display": "pending",
            "cells": [
                _cell(
                    "search",
                    "Need workspace evidence for the comparison.",
                    {"information_need": 1.0},
                    uncertainty=0.9,
                )
            ],
            "needs": [
                {
                    "evidence_id": "search-results",
                    "cell_id": "search",
                    "confidence": 1.0,
                    "freshness": "once",
                }
            ],
        },
        "after:search-results": {
            "display": "pending",
            "cells": [
                _cell(
                    "search",
                    "Candidate documents are available.",
                    {"percept": 1.0},
                    uncertainty=0.2,
                    lifecycle="stable",
                ),
                _cell(
                    "read-a",
                    "Need release evidence for A.",
                    {"information_need": 1.0},
                    uncertainty=0.9,
                ),
                _cell(
                    "read-b",
                    "Need release evidence for B.",
                    {"information_need": 1.0},
                    uncertainty=0.9,
                ),
            ],
            "needs": [
                {"evidence_id": "support-a", "cell_id": "read-a"},
                {"evidence_id": "support-b", "cell_id": "read-b"},
            ],
        },
        "after:support-a": {
            "display": "A: 2001; B: pending",
            "cells": [
                _cell(
                    "search",
                    "Candidate documents are available.",
                    {"percept": 1.0},
                    uncertainty=0.1,
                    lifecycle="stable",
                ),
                _cell(
                    "read-a",
                    "A was released in 2001.",
                    {"percept": 0.8, "conclusion": 0.3},
                    uncertainty=0.1,
                    lifecycle="stable",
                ),
                _cell(
                    "read-b",
                    "Still waiting for B release evidence.",
                    {"information_need": 1.0},
                    uncertainty=0.8,
                    lifecycle="waiting",
                ),
            ],
            "needs": [],
        },
        "after:support-b": {
            "display": "A",
            "cells": [
                _cell(
                    "search",
                    "Candidate documents are available.",
                    {"percept": 1.0},
                    uncertainty=0.1,
                    lifecycle="stable",
                ),
                _cell(
                    "read-a",
                    "A was released in 2001.",
                    {"percept": 1.0},
                    uncertainty=0.05,
                    lifecycle="stable",
                ),
                _cell(
                    "read-b",
                    "B was released in 2005.",
                    {"percept": 1.0},
                    uncertainty=0.05,
                    lifecycle="stable",
                ),
                _cell(
                    "answer",
                    "A was released before B.",
                    {"conclusion": 1.0},
                    uncertainty=0.02,
                    lifecycle="stable",
                ),
            ],
            "needs": [],
        },
    }

    for wave_index, phase in enumerate(outputs):
        request_path = tmp_path / f"requests-{wave_index}.jsonl"
        report = export_teacher_wave(jobs_path, state_path, request_path)
        assert report["exported_requests"] == 1
        request = json.loads(request_path.read_text().strip())
        assert request["phase"] == phase
        assert '"reference_answer"' not in request["prompt"]
        if phase == "after:search-results":
            assert "A was released in 2001." not in request["prompt"]
            assert "B was released in 2005." not in request["prompt"]
        response_path = tmp_path / f"responses-{wave_index}.jsonl"
        _write_responses(response_path, request_path, outputs)
        imported = import_teacher_wave(jobs_path, request_path, response_path, state_path)
        assert imported["imported"] == 1

    final_requests = tmp_path / "requests-done.jsonl"
    report = export_teacher_wave(jobs_path, state_path, final_requests)
    assert report == {"jobs": 1, "complete_tasks": 1, "exported_requests": 0}
    assert final_requests.read_text() == ""
    assert len(load_teacher_wave_state(state_path)) == 4
    assert teacher_wave_status(jobs_path, state_path)["complete_tasks"] == 1

    plans_path = tmp_path / "plans.jsonl"
    plans = finalize_teacher_wave((task,), jobs_path, state_path, plans_path)
    assert len(plans) == 1
    assert plans[0].final_answer == "A"
    assert [need.evidence_id for need in plans[0].needs] == [
        "search-results",
        "support-a",
        "support-b",
    ]
    assert [need.phase for need in plans[0].needs] == [
        "initial",
        "after:search-results",
        "after:search-results",
    ]
    assert load_teacher_plans(plans_path) == plans
    (review,) = review_teacher_plans((task,), plans)
    assert review.accepted, review.reasons

    (trajectory,) = compile_teacher_plans(
        (task,),
        plans,
        TeacherScheduleConfig(
            thought_capacity=8,
            min_delay_steps=3,
            max_delay_steps=3,
            seed=0,
        ),
    )
    assert trajectory.metadata["event_launch_steps"] == {
        "search-results": 0,
        "support-a": 3,
        "support-b": 3,
    }
    assert trajectory.metadata["event_arrival_steps"] == {
        "search-results": 3,
        "support-a": 6,
        "support-b": 7,
    }
    binding_steps = {
        target.need_id: target.first_need_step for target in trajectory.binding_targets
    }
    assert binding_steps["need:support-a"] == 3
    assert binding_steps["need:support-b"] == 3
    lifecycles = {
        (target.step, target.cell_id): target.lifecycle for target in trajectory.thought_targets
    }
    assert lifecycles[(5, "read-a")] is CellLifecycle.WAITING
    assert lifecycles[(5, "read-b")] is CellLifecycle.WAITING
    assert lifecycles[(6, "read-a")] is CellLifecycle.ACTIVE
    assert lifecycles[(6, "read-b")] is CellLifecycle.WAITING


def test_teacher_wave_import_is_idempotent_but_rejects_changed_output(tmp_path) -> None:
    task = _task()
    jobs_path = tmp_path / "jobs.jsonl"
    state_path = tmp_path / "state.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    dump_causal_teacher_jobs((task,), jobs_path)
    export_teacher_wave(jobs_path, state_path, requests_path)
    request = json.loads(requests_path.read_text())
    output = {
        "display": "pending",
        "cells": [
            _cell(
                "search",
                "Need workspace evidence.",
                {"information_need": 1.0},
            )
        ],
        "needs": [{"evidence_id": "search-results", "cell_id": "search"}],
    }
    responses_path.write_text(
        json.dumps({"request_id": request["request_id"], "output": output}) + "\n"
    )
    first = import_teacher_wave(jobs_path, requests_path, responses_path, state_path)
    second = import_teacher_wave(jobs_path, requests_path, responses_path, state_path)
    assert first["imported"] == 1
    assert second["unchanged"] == 1

    output["display"] = "changed"
    responses_path.write_text(
        json.dumps({"request_id": request["request_id"], "output": output}) + "\n"
    )
    with pytest.raises(ValueError, match="different output"):
        import_teacher_wave(jobs_path, requests_path, responses_path, state_path)


def test_teacher_wave_rejects_missing_or_extra_evidence_needs(tmp_path) -> None:
    task = _task()
    jobs_path = tmp_path / "jobs.jsonl"
    state_path = tmp_path / "state.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    dump_causal_teacher_jobs((task,), jobs_path)
    export_teacher_wave(jobs_path, state_path, requests_path)
    request = json.loads(requests_path.read_text())
    bad = {
        "display": "pending",
        "cells": [_cell("plan", "No need.", {"plan": 1.0})],
        "needs": [],
    }
    responses_path.write_text(
        json.dumps({"request_id": request["request_id"], "output": bad}) + "\n"
    )
    with pytest.raises(ValueError, match="every and only"):
        import_teacher_wave(jobs_path, requests_path, responses_path, state_path)

    rejects_path = tmp_path / "rejects.jsonl"
    report = import_teacher_wave(
        jobs_path,
        requests_path,
        responses_path,
        state_path,
        rejects_path=rejects_path,
    )
    assert report["rejected"] == 1
    assert report["state_records"] == 0
    rejected = json.loads(rejects_path.read_text())
    assert rejected["request_id"] == request["request_id"]
    assert "every and only" in rejected["error"]


def test_teacher_agent_workspace_resumes_commits_and_advances(tmp_path) -> None:
    task = _task()
    jobs_path = tmp_path / "jobs.jsonl"
    state_path = tmp_path / "state.jsonl"
    workspace = tmp_path / "agent"
    dump_causal_teacher_jobs((task,), jobs_path)

    first = checkout_teacher_agent_batch(jobs_path, state_path, workspace, max_requests=4)
    assert first["status"] == "checked_out"
    assert first["requests"] == 1
    resumed = checkout_teacher_agent_batch(jobs_path, state_path, workspace, max_requests=4)
    assert resumed["status"] == "resumed"
    assert resumed["pending"] == 1

    manifest = json.loads((workspace / "current" / "manifest.json").read_text())
    request_entry = manifest["requests"][0]
    request = json.loads((workspace / "current" / request_entry["request_file"]).read_text())
    assert request["phase"] == "initial"
    assert request["previous_state"] is None
    assert request["arrived_evidence"] is None
    assert "reference_answer" not in request["task"]
    assert [item["evidence_id"] for item in request["available_evidence_contracts"]] == [
        "search-results"
    ]

    bad_response = {
        "display": "pending",
        "cells": [_cell("plan", "No need yet.", {"plan": 1.0})],
        "needs": [],
    }
    response_path = workspace / "current" / request_entry["response_file"]
    response_path.write_text(json.dumps(bad_response))
    rejected = commit_teacher_agent_batch(workspace)
    assert rejected["rejected"] == 1
    assert not rejected["complete"]
    error_path = workspace / "current" / "errors" / f"{request['request_id']}.json"
    assert "every and only" in json.loads(error_path.read_text())["error"]

    response_path.write_text(
        json.dumps(
            {
                "display": "pending",
                "cells": [
                    {
                        **_cell(
                            "search",
                            "Need workspace evidence for the comparison.",
                            {"information_need": 1.0},
                        ),
                        "links": [
                            {
                                "relation": "requests",
                                "target": {
                                    "kind": "source",
                                    "identifier": "workspace_search",
                                },
                                "confidence": 1.0,
                            }
                        ],
                    }
                ],
                "needs": [{"evidence_id": "search-results", "cell_id": "search"}],
            }
        )
    )
    committed = commit_teacher_agent_batch(workspace)
    assert committed["imported"] == 1
    assert committed["complete"]
    assert not error_path.exists()

    second = checkout_teacher_agent_batch(jobs_path, state_path, workspace, max_requests=4)
    assert second["status"] == "checked_out"
    second_manifest = json.loads((workspace / "current" / "manifest.json").read_text())
    second_entry = second_manifest["requests"][0]
    second_request = json.loads((workspace / "current" / second_entry["request_file"]).read_text())
    assert second_request["phase"] == "after:search-results"
    assert second_request["previous_state"]["cells"][0]["cell_id"] == "search"
    assert second_request["arrived_evidence"]["evidence_id"] == "search-results"
    assert "A was released in 2001." not in json.dumps(second_request)
    assert "B was released in 2005." not in json.dumps(second_request)

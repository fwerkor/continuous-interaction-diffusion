import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cid.causal_distill import build_causal_teacher_job
from cid.correction_training import (
    CORRECTION_FAMILIES,
    CorrectionTrainingConfig,
    audit_correction_plans,
    build_correction_training,
    correction_teacher_response,
    generate_correction_tasks,
)
from cid.distill import (
    TeacherScheduleConfig,
    compile_teacher_plans,
    load_teacher_tasks,
    review_teacher_plans,
)
from cid.state import CognitiveRole
from cid.teacher_agent import checkout_teacher_agent_batch, commit_teacher_agent_batch
from cid.teacher_wave import finalize_teacher_wave


def test_correction_tasks_are_deterministic_and_cover_all_families() -> None:
    config = CorrectionTrainingConfig(count_per_family=3, seed=17)
    first = generate_correction_tasks(config)
    second = generate_correction_tasks(config)

    assert first == second
    assert len(first) == 3 * len(CORRECTION_FAMILIES)
    assert {str(task.metadata["family"]) for task in first} == set(CORRECTION_FAMILIES)
    assert all(len(task.evidence) == 2 for task in first)
    assert all(task.evidence[1].depends_on == ("correction",) for task in first)
    assert all(str(task.metadata["provisional_guess"]) != task.reference_answer for task in first)


def test_correction_teacher_reopens_only_dependent_state_then_stabilizes() -> None:
    task = generate_correction_tasks(CorrectionTrainingConfig(count_per_family=1, seed=23))[0]
    job = build_causal_teacher_job(task)
    previous = None
    outputs = []
    for stage in job.stages:
        request = {
            "task_id": task.task_id,
            "task": job.task,
            "previous_state": previous,
            "arrived_evidence": (
                None if stage.arrived_evidence is None else stage.arrived_evidence.to_dict()
            ),
            "available_evidence_contracts": [dict(item) for item in stage.available_evidence],
            "terminal": stage.terminal,
        }
        output = correction_teacher_response(request)
        outputs.append(output)
        previous = output

    initial = {cell["cell_id"]: cell for cell in outputs[0]["cells"]}
    corrected = {cell["cell_id"]: cell for cell in outputs[1]["cells"]}
    confirmed = {cell["cell_id"]: cell for cell in outputs[2]["cells"]}

    assert initial["context"] == corrected["context"] == confirmed["context"]
    assert initial["scope"] == corrected["scope"] == confirmed["scope"]
    assert corrected["hypothesis"]["noise"] > initial["hypothesis"]["noise"]
    assert confirmed["hypothesis"]["noise"] < corrected["hypothesis"]["noise"]
    assert corrected["answer"]["noise"] > initial["answer"]["noise"]
    assert confirmed["answer"]["noise"] < corrected["answer"]["noise"]
    assert outputs[-1]["display"] == task.reference_answer
    assert initial["hypothesis"]["anchors"]
    assert corrected["hypothesis"]["links"]


def test_small_correction_self_distillation_passes_review_and_compiles() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks_path = root / "tasks.jsonl"
        requests_path = root / "requests.jsonl"
        jobs_path = root / "jobs.jsonl"
        manifest_path = root / "manifest.json"
        state_path = root / "state.jsonl"
        workspace = root / "teacher"
        plans_path = root / "plans.jsonl"

        manifest = build_correction_training(
            tasks_path,
            requests_path,
            jobs_path,
            manifest_path,
            CorrectionTrainingConfig(count_per_family=1, seed=31),
        )
        assert manifest["tasks"] == len(CORRECTION_FAMILIES)
        assert manifest["causal_stage_histogram"] == {"3": len(CORRECTION_FAMILIES)}

        while True:
            status = checkout_teacher_agent_batch(
                jobs_path,
                state_path,
                workspace,
                max_requests=7,
            )
            if status["status"] == "complete":
                break
            current = workspace / "current"
            responses = current / "responses"
            responses.mkdir(exist_ok=True)
            for request_file in sorted((current / "requests").glob("*.json")):
                request = json.loads(request_file.read_text(encoding="utf-8"))
                output = correction_teacher_response(request)
                (responses / f"{request['request_id']}.json").write_text(
                    json.dumps(output, ensure_ascii=False),
                    encoding="utf-8",
                )
            report = commit_teacher_agent_batch(workspace)
            assert report["missing"] == 0
            assert report["rejected"] == 0

        tasks = load_teacher_tasks(tasks_path)
        plans = finalize_teacher_wave(tasks, jobs_path, state_path, plans_path)
        reviews = review_teacher_plans(tasks, plans)
        assert all(review.accepted for review in reviews)

        audit = audit_correction_plans(tasks, plans)
        assert audit["rejected"] == 0
        assert audit["accepted"] == len(tasks)
        assert audit["local_reopen_cells"] == 2 * len(tasks)
        assert audit["local_stabilization_cells"] == 2 * len(tasks)
        assert audit["anchored_hypothesis_tasks"] == len(tasks)
        assert audit["linked_hypothesis_tasks"] == len(tasks)

        trajectories = compile_teacher_plans(
            tasks,
            plans,
            TeacherScheduleConfig(
                thought_capacity=8,
                min_delay_steps=1,
                max_delay_steps=3,
                variants_per_task=2,
                seed=37,
            ),
        )
        assert len(trajectories) == 2 * len(tasks)
        assert all(
            any(
                target.roles.get(CognitiveRole.HYPOTHESIS, 0.0) > 0.0
                for target in example.thought_targets
            )
            for example in trajectories
        )
        assert all(
            example.metadata["capability"] == "speculative_local_correction"
            for example in trajectories
        )

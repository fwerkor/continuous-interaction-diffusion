import json
from pathlib import Path

from cid.causal_distill import build_causal_teacher_job
from cid.computational_training import (
    COMPUTATIONAL_FAMILIES,
    ComputationalTrainingConfig,
    build_computational_training,
    generate_computational_tasks,
)


def test_computational_task_mix_is_deterministic_and_balanced() -> None:
    config = ComputationalTrainingConfig(count_per_family=2, seed=17)
    first = generate_computational_tasks(config)
    second = generate_computational_tasks(config)

    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert len(first) == 2 * len(COMPUTATIONAL_FAMILIES)
    counts = {family: 0 for family in COMPUTATIONAL_FAMILIES}
    for task in first:
        counts[str(task.metadata["family"])] += 1
    assert set(counts.values()) == {2}


def test_computational_mix_covers_calibration_and_dependency_patterns() -> None:
    tasks = generate_computational_tasks(ComputationalTrainingConfig(count_per_family=1, seed=5))
    by_family = {str(task.metadata["family"]): task for task in tasks}

    unnecessary = by_family["calculator_unnecessary"]
    assert unnecessary.evidence == ()
    assert {item["name"] for item in unnecessary.source_descriptors} == {"calculator", "python"}
    assert unnecessary.metadata["training_mode"] == "tools_available_unnecessary"

    sequential = by_family["sequential_calculator"]
    assert sequential.evidence[1].depends_on == ("subtotal",)
    assert len(build_causal_teacher_job(sequential).stages) == 3

    parallel = by_family["parallel_calculator"]
    assert parallel.evidence[2].depends_on == ("plan-a", "plan-b")
    initial = build_causal_teacher_job(parallel).stages[0]
    assert {item["evidence_id"] for item in initial.available_evidence} == {"plan-a", "plan-b"}

    lookup = by_family["lookup_then_calculator"]
    assert [item.source for item in lookup.evidence] == ["record_lookup", "calculator"]
    assert lookup.evidence[1].depends_on == ("calibration",)


def test_build_computational_training_writes_manifest(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    requests = tmp_path / "requests.jsonl"
    jobs = tmp_path / "jobs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_computational_training(
        tasks,
        requests,
        jobs,
        manifest_path,
        ComputationalTrainingConfig(count_per_family=3, seed=9),
    )

    assert manifest["tasks"] == 30
    assert manifest["mode_counts"] == {
        "tool_required": 27,
        "tools_available_unnecessary": 3,
    }
    assert manifest["tool_call_target_counts"] == {
        "calculator": 30,
        "python": 9,
        "record_lookup": 9,
    }
    assert manifest["dependency_depth_histogram"] == {"0": 3, "1": 12, "2": 15}
    assert all(path.exists() for path in (tasks, requests, jobs, manifest_path))
    assert json.loads(manifest_path.read_text())["tasks_sha256"] == manifest["tasks_sha256"]

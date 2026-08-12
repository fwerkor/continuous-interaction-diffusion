import json
from pathlib import Path

from cid.causal_distill import build_causal_teacher_job
from cid.symbolic_training import (
    SYMBOLIC_FAMILIES,
    SymbolicTrainingConfig,
    build_symbolic_training,
    generate_symbolic_tasks,
)


def test_symbolic_task_mix_is_deterministic_and_balanced() -> None:
    config = SymbolicTrainingConfig(count_per_family=2, seed=17)
    first = generate_symbolic_tasks(config)
    second = generate_symbolic_tasks(config)

    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert len(first) == 2 * len(SYMBOLIC_FAMILIES)
    counts = {family: 0 for family in SYMBOLIC_FAMILIES}
    for task in first:
        counts[str(task.metadata["family"])] += 1
    assert set(counts.values()) == {2}


def test_symbolic_mix_covers_single_sequential_parallel_and_no_tool_patterns() -> None:
    tasks = generate_symbolic_tasks(SymbolicTrainingConfig(count_per_family=1, seed=5))
    by_family = {str(task.metadata["family"]): task for task in tasks}

    unnecessary = by_family["symbolic_unnecessary"]
    assert unnecessary.evidence == ()
    assert {item["name"] for item in unnecessary.source_descriptors} == {
        "calculator",
        "symbolic_math",
    }
    assert unnecessary.metadata["training_mode"] == "tools_available_unnecessary"

    sequential = by_family["symbolic_then_calculator"]
    assert [item.source for item in sequential.evidence] == ["symbolic_math", "calculator"]
    assert sequential.evidence[1].depends_on == ("symbolic-solution",)
    assert len(build_causal_teacher_job(sequential).stages) == 3

    lookup = by_family["lookup_then_symbolic"]
    assert [item.source for item in lookup.evidence] == ["record_lookup", "symbolic_math"]
    assert lookup.evidence[1].depends_on == ("equation-record",)

    parallel = by_family["parallel_symbolic_then_merge"]
    assert parallel.evidence[2].depends_on == ("x-solution", "y-solution")
    initial = build_causal_teacher_job(parallel).stages[0]
    assert {item["evidence_id"] for item in initial.available_evidence} == {
        "x-solution",
        "y-solution",
    }


def test_build_symbolic_training_writes_manifest(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    requests = tmp_path / "requests.jsonl"
    jobs = tmp_path / "jobs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_symbolic_training(
        tasks,
        requests,
        jobs,
        manifest_path,
        SymbolicTrainingConfig(count_per_family=3, seed=9),
    )

    assert manifest["tasks"] == 39
    assert manifest["mode_counts"] == {
        "tool_required": 36,
        "tools_available_unnecessary": 3,
    }
    assert manifest["tool_call_target_counts"] == {
        "calculator": 6,
        "record_lookup": 3,
        "symbolic_math": 39,
    }
    assert manifest["dependency_depth_histogram"] == {"0": 3, "1": 27, "2": 9}
    assert manifest["causal_stage_histogram"] == {"1": 3, "2": 27, "3": 6, "4": 3}
    assert all(path.exists() for path in (tasks, requests, jobs, manifest_path))
    assert json.loads(manifest_path.read_text())["tasks_sha256"] == manifest["tasks_sha256"]


def test_symbolic_then_calculator_parenthesizes_negative_roots() -> None:
    tasks = generate_symbolic_tasks(SymbolicTrainingConfig(count_per_family=32, seed=23))
    sequential = [
        task for task in tasks if task.metadata["family"] == "symbolic_then_calculator"
    ]
    assert any(int(task.evidence[0].value) < 0 for task in sequential)
    for task in sequential:
        expression = str(task.evidence[1].arguments["expression"])
        result = eval(expression, {"__builtins__": {}}, {"round": round})
        assert abs(float(task.evidence[1].value) - float(result)) < 1e-9

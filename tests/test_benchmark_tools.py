from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from cid.benchmark_tools import (
    BENCHMARK_ARGUMENT_CANDIDATES_METADATA_KEY,
    BENCHMARK_LIVE_TOOL_LATENCY_STEPS_METADATA_KEY,
    BENCHMARK_LIVE_TOOLS_METADATA_KEY,
    benchmark_argument_candidates,
    benchmark_tool_descriptors,
)
from cid.computational_training import calculator_descriptor as training_calculator_descriptor
from cid.computational_training import python_descriptor as training_python_descriptor
from cid.data import TrajectoryExample
from cid.evaluation import build_replay_registry
from cid.symbolic_training import symbolic_math_descriptor as training_symbolic_math_descriptor
from cid.tool_schemas import calculator_descriptor, python_descriptor, symbolic_math_descriptor


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cid_test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_values(prompt: str, task_kind: str) -> set[tuple[str, str, str]]:
    descriptors = benchmark_tool_descriptors(task_kind)
    catalog = benchmark_argument_candidates(prompt, descriptors)
    return {
        (source, argument, str(value))
        for source, by_argument in catalog.items()
        for argument, values in by_argument.items()
        for value in values
    }


def test_benchmark_schemas_are_the_training_schemas() -> None:
    assert calculator_descriptor() == training_calculator_descriptor()
    assert python_descriptor() == training_python_descriptor()
    assert symbolic_math_descriptor() == training_symbolic_math_descriptor()

    assert [item["name"] for item in benchmark_tool_descriptors("math_word_problem")] == [
        "calculator",
        "python",
    ]
    assert [item["name"] for item in benchmark_tool_descriptors("competition_math")] == [
        "symbolic_math",
        "calculator",
        "python",
    ]
    assert [
        item["name"] for item in benchmark_tool_descriptors("multiple_choice_knowledge_reasoning")
    ] == ["calculator"]
    assert benchmark_tool_descriptors("python_programming") == ()


def test_argument_candidates_are_prompt_derived_and_useful() -> None:
    prompt = "A box has 12 rows of 7 items and 5 extra. How many items are there?"
    candidates = _candidate_values(prompt, "math_word_problem")
    assert ("calculator", "expression", "12*7+5") in candidates
    assert ("python", "code", "12*7+5") in candidates
    assert all("89" not in value for _, _, value in candidates)

    symbolic_prompt = r"Solve $3x+2=20$ for x."
    symbolic = _candidate_values(symbolic_prompt, "competition_math")
    assert ("symbolic_math", "operation", "solve") in symbolic
    assert ("symbolic_math", "expression", "3*x+2=20") in symbolic
    assert ("symbolic_math", "variables", "x") in symbolic


def test_python_candidate_matches_training_enumeration_pattern() -> None:
    prompt = "How many integers from 1 through 1200 are divisible by exactly one of 3 and 19?"
    candidates = _candidate_values(prompt, "math_word_problem")
    assert (
        "python",
        "code",
        "sum(1 for x in range(1,1200+1) if (x%3==0) ^ (x%19==0))",
    ) in candidates


async def test_live_python_source_executes_training_style_computation() -> None:
    example = TrajectoryExample(
        example_id="python-benchmark",
        prompt="Compute a bounded enumeration.",
        target_display="5",
        source_descriptors=(python_descriptor(),),
        metadata={
            BENCHMARK_LIVE_TOOLS_METADATA_KEY: ["python"],
            BENCHMARK_LIVE_TOOL_LATENCY_STEPS_METADATA_KEY: 0,
        },
    )
    registry = build_replay_registry(example)
    observation = await registry.get("python").read(
        {"code": "sum(1 for x in range(1,11) if x%2==0)"}
    )
    assert observation.value == "5"
    assert observation.provenance == "deterministic-python"


async def test_live_python_source_rejects_unsafe_code() -> None:
    example = TrajectoryExample(
        example_id="python-benchmark-unsafe",
        prompt="Do not execute unsafe code.",
        target_display="done",
        source_descriptors=(python_descriptor(),),
        metadata={
            BENCHMARK_LIVE_TOOLS_METADATA_KEY: ["python"],
            BENCHMARK_LIVE_TOOL_LATENCY_STEPS_METADATA_KEY: 0,
        },
    )
    registry = build_replay_registry(example)
    observation = await registry.get("python").read({"code": "__import__('os').getcwd()"})
    assert isinstance(observation.value, dict)
    assert "error" in observation.value


def test_sweep_completion_requires_current_dataset_hash(tmp_path: Path) -> None:
    sweep = _load_script("run_public_benchmark_sweep")
    _merged_result_complete = sweep._merged_result_complete
    _result_complete = sweep._result_complete

    result = tmp_path / "results.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"dataset": {"examples": 1, "sha256": "new-hash"}}),
        encoding="utf-8",
    )

    assert _result_complete(summary, result, "new-hash")
    assert not _result_complete(summary, result, "old-hash")

    summary.write_text(
        json.dumps({"examples": 1, "dataset_sha256": "new-hash"}),
        encoding="utf-8",
    )
    assert _merged_result_complete(tmp_path, "new-hash")
    assert not _merged_result_complete(tmp_path, "old-hash")


def test_public_benchmark_preparation_enables_tools_without_gold_arguments() -> None:
    _trajectory = _load_script("prepare_public_benchmarks")._trajectory

    record = {
        "task_id": "bench-gsm-fixture",
        "semantic_id": "fixture",
        "task_kind": "math_word_problem",
        "prompt": "A box has 12 rows of 7 items and 5 extra. How many items are there?",
        "reference_answer": "89",
        "source": {
            "dataset_id": "gsm8k-test",
            "repo": "openai/gsm8k",
            "revision": "fixture",
            "upstream_split": "test",
            "use": "general_reasoning",
        },
        "resources": {},
        "metadata": {"reference_solution": "gold reasoning ending in 89"},
    }
    example = _trajectory(record)

    assert [item["name"] for item in example.source_descriptors] == ["calculator", "python"]
    assert example.metadata[BENCHMARK_LIVE_TOOLS_METADATA_KEY] == ["calculator", "python"]
    assert example.metadata[BENCHMARK_LIVE_TOOL_LATENCY_STEPS_METADATA_KEY] == 2
    candidates = example.metadata[BENCHMARK_ARGUMENT_CANDIDATES_METADATA_KEY]
    assert "12*7+5" in candidates["calculator"]["expression"]
    assert "12*7+5" in candidates["python"]["code"]
    assert "gold reasoning" not in json.dumps(candidates)

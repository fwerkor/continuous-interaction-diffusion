from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.causal_distill import build_causal_teacher_job, dump_causal_teacher_jobs
from cid.distill import TeacherEvidence, TeacherTask, dump_teacher_requests, dump_teacher_tasks

COMPUTATIONAL_FAMILIES = (
    "direct_calculator",
    "applied_formula",
    "sequential_calculator",
    "parallel_calculator",
    "calculator_unnecessary",
    "python_statistics",
    "python_enumeration",
    "lookup_then_calculator",
    "calculator_then_python",
    "parallel_lookup_merge",
)


@dataclass(frozen=True, slots=True)
class ComputationalTrainingConfig:
    count_per_family: int = 1200
    seed: int = 20260812

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")

    @property
    def total_tasks(self) -> int:
        return self.count_per_family * len(COMPUTATIONAL_FAMILIES)


def build_computational_training(
    tasks_output: str | Path,
    requests_output: str | Path,
    causal_jobs_output: str | Path,
    manifest_output: str | Path,
    config: ComputationalTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or ComputationalTrainingConfig()
    tasks = generate_computational_tasks(config)

    task_path = Path(tasks_output)
    request_path = Path(requests_output)
    jobs_path = Path(causal_jobs_output)
    manifest_path = Path(manifest_output)
    for path in (task_path, request_path, jobs_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    dump_teacher_tasks(tasks, task_path)
    dump_teacher_requests(tasks, request_path)
    dump_causal_teacher_jobs(tasks, jobs_path)

    family_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    tool_call_target_counts: Counter[str] = Counter()
    stage_counts: Counter[int] = Counter()
    dependency_depths: Counter[int] = Counter()
    for task in tasks:
        family_counts[str(task.metadata["family"])] += 1
        mode_counts[str(task.metadata["training_mode"])] += 1
        for descriptor in task.source_descriptors:
            tool_counts[str(descriptor["name"])] += 1
        for item in task.evidence:
            tool_call_target_counts[item.source] += 1
        job = build_causal_teacher_job(task)
        stage_counts[len(job.stages)] += 1
        dependency_depths[_dependency_depth(task)] += 1

    manifest = {
        "format_version": 1,
        "name": "computational-tools-v1",
        "seed": config.seed,
        "tasks": len(tasks),
        "count_per_family": config.count_per_family,
        "family_counts": dict(sorted(family_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "tool_schema_task_counts": dict(sorted(tool_counts.items())),
        "tool_call_target_counts": dict(sorted(tool_call_target_counts.items())),
        "causal_stage_histogram": {
            str(stages): count for stages, count in sorted(stage_counts.items())
        },
        "dependency_depth_histogram": {
            str(depth): count for depth, count in sorted(dependency_depths.items())
        },
        "tasks_sha256": _file_sha256(task_path),
        "requests_sha256": _file_sha256(request_path),
        "causal_jobs_sha256": _file_sha256(jobs_path),
        "tasks_path": str(task_path),
        "causal_jobs": str(jobs_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_computational_tasks(
    config: ComputationalTrainingConfig | None = None,
) -> tuple[TeacherTask, ...]:
    config = config or ComputationalTrainingConfig()
    rng = random.Random(config.seed)
    generators = {
        "direct_calculator": _direct_calculator,
        "applied_formula": _applied_formula,
        "sequential_calculator": _sequential_calculator,
        "parallel_calculator": _parallel_calculator,
        "calculator_unnecessary": _calculator_unnecessary,
        "python_statistics": _python_statistics,
        "python_enumeration": _python_enumeration,
        "lookup_then_calculator": _lookup_then_calculator,
        "calculator_then_python": _calculator_then_python,
        "parallel_lookup_merge": _parallel_lookup_merge,
    }
    tasks: list[TeacherTask] = []
    for family in COMPUTATIONAL_FAMILIES:
        generator = generators[family]
        for index in range(config.count_per_family):
            tasks.append(generator(rng, index))
    tasks.sort(key=lambda item: item.task_id)
    return tuple(tasks)


def calculator_descriptor() -> dict[str, Any]:
    return {
        "name": "calculator",
        "description": (
            "Evaluate a deterministic numeric expression exactly enough for the requested answer. "
            "Supports arithmetic, powers, abs, floor, sqrt, log, and round."
        ),
        "arguments": (
            {
                "name": "expression",
                "kind": "string",
                "description": "numeric expression to evaluate",
                "required": True,
            },
        ),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def python_descriptor() -> dict[str, Any]:
    return {
        "name": "python",
        "description": (
            "Execute a short deterministic pure-Python computation with no filesystem or network "
            "access and return its printed/result value."
        ),
        "arguments": (
            {
                "name": "code",
                "kind": "string",
                "description": "short pure computation",
                "required": True,
            },
        ),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def record_lookup_descriptor() -> dict[str, Any]:
    return {
        "name": "record_lookup",
        "description": "Read one immutable task-local record by key.",
        "arguments": ({"name": "key", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def _direct_calculator(rng: random.Random, index: int) -> TeacherTask:
    a = rng.randint(1500, 98000)
    b = rng.randint(101, 999)
    c = rng.randint(500, 9000)
    d = rng.randint(7, 97)
    expression = f"{a}*{b}+{c}*{d}"
    answer = str(a * b + c * d)
    return _task(
        family="direct_calculator",
        index=index,
        prompt=f"Compute exactly: {a} × {b} + {c} × {d}.",
        answer=answer,
        descriptors=(calculator_descriptor(),),
        evidence=(_evidence("calculation", "calculator", answer, {"expression": expression}),),
        pattern="single_tool",
        depth=1,
    )


def _applied_formula(rng: random.Random, index: int) -> TeacherTask:
    principal = rng.randint(800, 25000)
    rate_tenths = rng.randint(15, 125)
    years = rng.randint(3, 25)
    rate = rate_tenths / 10.0
    value = round(principal * (1.0 + rate / 100.0) ** years, 2)
    answer = f"{value:.2f}"
    expression = f"round({principal}*(1+{rate_tenths}/1000)**{years},2)"
    prompt = (
        f"An account starts with {principal} units and grows by {rate:.1f}% per year, compounded "
        f"annually, for {years} years. What is the final balance? Give exactly two decimal places."
    )
    return _task(
        family="applied_formula",
        index=index,
        prompt=prompt,
        answer=answer,
        descriptors=(calculator_descriptor(),),
        evidence=(_evidence("formula-value", "calculator", answer, {"expression": expression}),),
        pattern="reason_then_tool",
        depth=1,
    )


def _sequential_calculator(rng: random.Random, index: int) -> TeacherTask:
    quantity = rng.randint(20, 900)
    unit_cents = rng.randint(125, 9999)
    discount = rng.randint(4, 28)
    tax = rng.randint(3, 16)
    subtotal_cents = quantity * unit_cents
    subtotal = subtotal_cents / 100.0
    final = round(subtotal * (1.0 - discount / 100.0) * (1.0 + tax / 100.0), 2)
    subtotal_text = f"{subtotal:.2f}"
    final_text = f"{final:.2f}"
    return _task(
        family="sequential_calculator",
        index=index,
        prompt=(
            f"A buyer orders {quantity} items at {unit_cents / 100:.2f} each. Apply a {discount}% "
            f"discount, then {tax}% tax to the discounted subtotal. What is the final charge? "
            "Give exactly two decimal places."
        ),
        answer=final_text,
        descriptors=(calculator_descriptor(),),
        evidence=(
            _evidence(
                "subtotal",
                "calculator",
                subtotal_text,
                {"expression": f"round({quantity}*{unit_cents}/100,2)"},
            ),
            _evidence(
                "final-charge",
                "calculator",
                final_text,
                {"expression": (f"round({subtotal_text}*(1-{discount}/100)*(1+{tax}/100),2)")},
                depends_on=("subtotal",),
            ),
        ),
        pattern="sequential_tools",
        depth=2,
    )


def _parallel_calculator(rng: random.Random, index: int) -> TeacherTask:
    q1, q2 = rng.randint(30, 700), rng.randint(30, 700)
    p1, p2 = rng.randint(15, 800), rng.randint(15, 800)
    fee1, fee2 = rng.randint(100, 6000), rng.randint(100, 6000)
    total1 = q1 * p1 + fee1
    total2 = q2 * p2 + fee2
    difference = abs(total1 - total2)
    return _task(
        family="parallel_calculator",
        index=index,
        prompt=(
            f"Plan A costs {q1}×{p1} plus a fixed fee of {fee1}. Plan B costs {q2}×{p2} plus "
            f"a fixed fee of {fee2}. What is the absolute difference between the two total costs?"
        ),
        answer=str(difference),
        descriptors=(calculator_descriptor(),),
        evidence=(
            _evidence("plan-a", "calculator", str(total1), {"expression": f"{q1}*{p1}+{fee1}"}),
            _evidence("plan-b", "calculator", str(total2), {"expression": f"{q2}*{p2}+{fee2}"}),
            _evidence(
                "difference",
                "calculator",
                str(difference),
                {"expression": f"abs({total1}-{total2})"},
                depends_on=("plan-a", "plan-b"),
            ),
        ),
        pattern="parallel_then_merge",
        depth=2,
    )


def _calculator_unnecessary(rng: random.Random, index: int) -> TeacherTask:
    variant = index % 3
    if variant == 0:
        a, b = rng.randint(2, 35), rng.randint(2, 35)
        prompt = f"Compute {a} + {b}."
        answer = str(a + b)
    elif variant == 1:
        x = rng.randint(3, 70)
        offset = rng.randint(2, 20)
        total = x + offset
        prompt = f"Solve for x: x + {offset} = {total}."
        answer = str(x)
    else:
        groups = rng.randint(2, 12)
        each = rng.randint(2, 12)
        prompt = (
            f"There are {groups} equal groups with {each} objects in each group. "
            "How many objects are there?"
        )
        answer = str(groups * each)
    return _task(
        family="calculator_unnecessary",
        index=index,
        prompt=prompt,
        answer=answer,
        descriptors=(calculator_descriptor(), python_descriptor()),
        evidence=(),
        pattern="tools_available_unnecessary",
        depth=0,
        training_mode="tools_available_unnecessary",
    )


def _python_statistics(rng: random.Random, index: int) -> TeacherTask:
    count = 9 + 2 * (index % 4)
    values = [rng.randint(10, 999) for _ in range(count)]
    value_text = json.dumps(values, separators=(",", ":"))
    if index % 2 == 0:
        result = round(sum(values) / len(values), 2)
        answer = f"{result:.2f}"
        prompt = (
            f"For the values {values}, compute the arithmetic mean. "
            "Give exactly two decimal places."
        )
        code = f"round(sum({value_text})/len({value_text}),2)"
    else:
        ordered = sorted(values)
        result = ordered[len(ordered) // 2]
        answer = str(result)
        prompt = f"For the values {values}, compute the median."
        code = f"sorted({value_text})[len({value_text})//2]"
    return _task(
        family="python_statistics",
        index=index,
        prompt=prompt,
        answer=answer,
        descriptors=(python_descriptor(), calculator_descriptor()),
        evidence=(_evidence("statistics", "python", answer, {"code": code}),),
        pattern="python_analysis",
        depth=1,
    )


def _python_enumeration(rng: random.Random, index: int) -> TeacherTask:
    n = rng.randint(1200, 25000)
    a = rng.randint(3, 17)
    b = rng.randint(18, 37)
    count = sum(1 for value in range(1, n + 1) if (value % a == 0) ^ (value % b == 0))
    code = f"sum(1 for x in range(1,{n}+1) if (x%{a}==0) ^ (x%{b}==0))"
    return _task(
        family="python_enumeration",
        index=index,
        prompt=(
            f"How many integers from 1 through {n} are divisible by exactly one of {a} and {b}?"
        ),
        answer=str(count),
        descriptors=(python_descriptor(), calculator_descriptor()),
        evidence=(_evidence("enumeration", "python", str(count), {"code": code}),),
        pattern="python_analysis",
        depth=1,
    )


def _lookup_then_calculator(rng: random.Random, index: int) -> TeacherTask:
    key = f"sensor-{index:05d}"
    raw = rng.randint(800, 30000)
    scale_milli = rng.randint(750, 2400)
    offset_centi = rng.randint(-5000, 5000)
    scale = scale_milli / 1000.0
    offset = offset_centi / 100.0
    result = round(raw * scale + offset, 2)
    result_text = f"{result:.2f}"
    lookup_value = {"raw": raw, "scale": f"{scale:.3f}", "offset": f"{offset:.2f}"}
    expression = f"round({raw}*{scale_milli}/1000+({offset_centi}/100),2)"
    return _task(
        family="lookup_then_calculator",
        index=index,
        prompt=(
            f"Read the calibration record for {key}. Apply calibrated = raw × scale + offset and "
            "return the calibrated value with exactly two decimal places."
        ),
        answer=result_text,
        descriptors=(record_lookup_descriptor(), calculator_descriptor()),
        evidence=(
            _evidence("calibration", "record_lookup", lookup_value, {"key": key}),
            _evidence(
                "calibrated-value",
                "calculator",
                result_text,
                {"expression": expression},
                depends_on=("calibration",),
            ),
        ),
        pattern="lookup_then_calculate",
        depth=2,
    )


def _calculator_then_python(rng: random.Random, index: int) -> TeacherTask:
    base = rng.randint(1200, 9000)
    multiplier = rng.randint(12, 35)
    divisor = rng.randint(3, 11)
    threshold = (base * multiplier) // divisor
    values = [rng.randint(max(1, threshold - 4000), threshold + 4000) for _ in range(25)]
    count = sum(1 for value in values if value <= threshold)
    value_text = json.dumps(values, separators=(",", ":"))
    return _task(
        family="calculator_then_python",
        index=index,
        prompt=(
            f"First define the threshold as floor({base} × {multiplier} / {divisor}). "
            "For the values "
            f"{values}, how many are at most that threshold?"
        ),
        answer=str(count),
        descriptors=(calculator_descriptor(), python_descriptor()),
        evidence=(
            _evidence(
                "threshold",
                "calculator",
                str(threshold),
                {"expression": f"floor({base}*{multiplier}/{divisor})"},
            ),
            _evidence(
                "count",
                "python",
                str(count),
                {"code": f"sum(1 for x in {value_text} if x<={threshold})"},
                depends_on=("threshold",),
            ),
        ),
        pattern="calculator_then_python",
        depth=2,
    )


def _parallel_lookup_merge(rng: random.Random, index: int) -> TeacherTask:
    left_key = f"warehouse-a-{index:05d}"
    right_key = f"warehouse-b-{index:05d}"
    left_units, right_units = rng.randint(100, 10000), rng.randint(100, 10000)
    left_price_cents, right_price_cents = rng.randint(50, 8000), rng.randint(50, 8000)
    left_total = left_units * left_price_cents / 100.0
    right_total = right_units * right_price_cents / 100.0
    combined = round(left_total + right_total, 2)
    combined_text = f"{combined:.2f}"
    return _task(
        family="parallel_lookup_merge",
        index=index,
        prompt=(
            f"Read inventory records {left_key} and {right_key}. Each record gives units and unit "
            "price. What is the combined inventory value? Give exactly two decimal places."
        ),
        answer=combined_text,
        descriptors=(record_lookup_descriptor(), calculator_descriptor()),
        evidence=(
            _evidence(
                "left-record",
                "record_lookup",
                {"units": left_units, "unit_price": f"{left_price_cents / 100:.2f}"},
                {"key": left_key},
            ),
            _evidence(
                "right-record",
                "record_lookup",
                {"units": right_units, "unit_price": f"{right_price_cents / 100:.2f}"},
                {"key": right_key},
            ),
            _evidence(
                "combined-value",
                "calculator",
                combined_text,
                {
                    "expression": (
                        f"round({left_units}*{left_price_cents}/100+"
                        f"{right_units}*{right_price_cents}/100,2)"
                    )
                },
                depends_on=("left-record", "right-record"),
            ),
        ),
        pattern="parallel_lookup_then_merge",
        depth=2,
    )


def _task(
    *,
    family: str,
    index: int,
    prompt: str,
    answer: str,
    descriptors: tuple[dict[str, Any], ...],
    evidence: tuple[TeacherEvidence, ...],
    pattern: str,
    depth: int,
    training_mode: str = "tool_required",
) -> TeacherTask:
    return TeacherTask(
        task_id=f"compute-{family}-{index:06d}",
        prompt=prompt,
        source_descriptors=descriptors,
        evidence=evidence,
        metadata={
            "task_kind": "computational_reasoning",
            "family": family,
            "interaction_pattern": pattern,
            "dependency_depth": depth,
            "training_mode": training_mode,
            "generated_by": "cid.computational_training.v1",
        },
        reference_answer=answer,
    )


def _evidence(
    evidence_id: str,
    source: str,
    value: Any,
    arguments: dict[str, Any],
    *,
    depends_on: tuple[str, ...] = (),
) -> TeacherEvidence:
    return TeacherEvidence(
        evidence_id=evidence_id,
        source=source,
        value=value,
        arguments=arguments,
        depends_on=depends_on,
        provenance="cid.computational_training.v1",
    )


def _dependency_depth(task: TeacherTask) -> int:
    depths: dict[str, int] = {}
    for item in task.evidence:
        depths[item.evidence_id] = (
            1 if not item.depends_on else 1 + max(depths[parent] for parent in item.depends_on)
        )
    return max(depths.values(), default=0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

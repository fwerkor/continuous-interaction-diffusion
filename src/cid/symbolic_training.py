from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp

from cid.causal_distill import build_causal_teacher_job, dump_causal_teacher_jobs
from cid.computational_training import calculator_descriptor, record_lookup_descriptor
from cid.distill import TeacherEvidence, TeacherTask, dump_teacher_requests, dump_teacher_tasks

SYMBOLIC_FAMILIES = (
    "linear_equation",
    "quadratic_roots",
    "polynomial_expand",
    "polynomial_factor",
    "rational_simplify",
    "system_2x2",
    "derivative",
    "definite_integral",
    "identity_check",
    "symbolic_then_calculator",
    "lookup_then_symbolic",
    "parallel_symbolic_then_merge",
    "symbolic_unnecessary",
)


@dataclass(frozen=True, slots=True)
class SymbolicTrainingConfig:
    count_per_family: int = 1200
    seed: int = 20260812

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")

    @property
    def total_tasks(self) -> int:
        return self.count_per_family * len(SYMBOLIC_FAMILIES)


def symbolic_math_descriptor() -> dict[str, Any]:
    return {
        "name": "symbolic_math",
        "description": (
            "Perform exact symbolic algebra or calculus. Supported operations include solve, "
            "solve_system, expand, factor, simplify, differentiate, integrate, and equivalent."
        ),
        "arguments": (
            {
                "name": "operation",
                "kind": "string",
                "description": "symbolic operation to perform",
                "required": True,
            },
            {
                "name": "expression",
                "kind": "string",
                "description": "expression, equation, or semicolon-separated equation system",
                "required": True,
            },
            {
                "name": "variables",
                "kind": "string",
                "description": "comma-separated symbolic variables",
                "required": True,
            },
        ),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def build_symbolic_training(
    tasks_output: str | Path,
    requests_output: str | Path,
    causal_jobs_output: str | Path,
    manifest_output: str | Path,
    config: SymbolicTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or SymbolicTrainingConfig()
    tasks = generate_symbolic_tasks(config)

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
        "name": "symbolic-tools-v1",
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


def generate_symbolic_tasks(
    config: SymbolicTrainingConfig | None = None,
) -> tuple[TeacherTask, ...]:
    config = config or SymbolicTrainingConfig()
    rng = random.Random(config.seed)
    generators = {
        "linear_equation": _linear_equation,
        "quadratic_roots": _quadratic_roots,
        "polynomial_expand": _polynomial_expand,
        "polynomial_factor": _polynomial_factor,
        "rational_simplify": _rational_simplify,
        "system_2x2": _system_2x2,
        "derivative": _derivative,
        "definite_integral": _definite_integral,
        "identity_check": _identity_check,
        "symbolic_then_calculator": _symbolic_then_calculator,
        "lookup_then_symbolic": _lookup_then_symbolic,
        "parallel_symbolic_then_merge": _parallel_symbolic_then_merge,
        "symbolic_unnecessary": _symbolic_unnecessary,
    }
    tasks: list[TeacherTask] = []
    for family in SYMBOLIC_FAMILIES:
        generator = generators[family]
        for index in range(config.count_per_family):
            tasks.append(generator(rng, index))
    tasks.sort(key=lambda item: item.task_id)
    return tuple(tasks)


def _symbolic_args(operation: str, expression: str, variables: str = "x") -> dict[str, str]:
    return {"operation": operation, "expression": expression, "variables": variables}


def _linear_equation(rng: random.Random, index: int) -> TeacherTask:
    solution = rng.randint(-120, 120)
    while True:
        a, p = rng.randint(3, 19), rng.randint(2, 11)
        c, q = rng.randint(2, 17), rng.randint(2, 9)
        if a * p != c * q:
            break
    b, d = rng.randint(-80, 80), rng.randint(-80, 80)
    left_at_solution = a * (p * solution + b) - c * (q * solution + d)
    expression = f"{a}*({p}*x+({b}))-{c}*({q}*x+({d}))={left_at_solution}"
    prompt = (
        f"Solve exactly for x: {a}({p}x {b:+d}) - {c}({q}x {d:+d}) = "
        f"{left_at_solution}. Return x as an integer."
    )
    return _task(
        family="linear_equation",
        index=index,
        prompt=prompt,
        answer=str(solution),
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "solution",
                "symbolic_math",
                str(solution),
                _symbolic_args("solve", expression),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _quadratic_roots(rng: random.Random, index: int) -> TeacherTask:
    root_a = rng.randint(-100, 50)
    root_b = rng.randint(root_a + 1, 150)
    scale = rng.randint(2, 15)
    x = sp.Symbol("x")
    polynomial = sp.expand(scale * (x - root_a) * (x - root_b))
    expression = sp.sstr(polynomial)
    answer = f"{root_a}, {root_b}"
    return _task(
        family="quadratic_roots",
        index=index,
        prompt=(
            f"Find all real roots of {expression} = 0. Return the two roots in ascending order "
            "as comma-separated integers."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "roots",
                "symbolic_math",
                answer,
                _symbolic_args("solve", f"{expression}=0"),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _polynomial_expand(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    a, c = rng.randint(2, 8), rng.randint(2, 8)
    b, d, e = (rng.randint(-12, 12) for _ in range(3))
    source = (a * x + b) * (c * x + d) * (x + e)
    expression = f"({a}*x+({b}))*({c}*x+({d}))*(x+({e}))"
    answer = sp.sstr(sp.expand(source))
    return _task(
        family="polynomial_expand",
        index=index,
        prompt=(
            f"Expand {expression}. Return the canonical expanded polynomial in Python-style "
            "notation using * and **."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "expanded-expression",
                "symbolic_math",
                answer,
                _symbolic_args("expand", expression),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _polynomial_factor(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    scale = rng.randint(2, 17)
    roots = [rng.randint(-40, 40) for _ in range(3)]
    source = scale
    for root in roots:
        source *= x - root
    expanded = sp.expand(source)
    expression = sp.sstr(expanded)
    answer = sp.sstr(sp.factor(expanded))
    return _task(
        family="polynomial_factor",
        index=index,
        prompt=(
            f"Factor {expression} completely over the integers. Return canonical Python-style "
            "symbolic notation."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "factored-expression",
                "symbolic_math",
                answer,
                _symbolic_args("factor", expression),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _rational_simplify(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    shared = rng.randint(-12, 12)
    a, c = rng.randint(2, 11), rng.randint(2, 11)
    b, d = rng.randint(-20, 20), rng.randint(-20, 20)
    numerator = (x - shared) * (a * x + b)
    denominator = (x - shared) * (c * x + d)
    expression = f"({sp.sstr(sp.expand(numerator))})/({sp.sstr(sp.expand(denominator))})"
    answer = sp.sstr(sp.cancel(numerator / denominator))
    return _task(
        family="rational_simplify",
        index=index,
        prompt=(
            f"Simplify the rational expression {expression}. Return the exact canonical "
            "Python-style symbolic expression."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "simplified-expression",
                "symbolic_math",
                answer,
                _symbolic_args("simplify", expression),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _system_2x2(rng: random.Random, index: int) -> TeacherTask:
    x_value, y_value = rng.randint(-40, 40), rng.randint(-40, 40)
    while True:
        a, b, c, d = (rng.randint(2, 15) for _ in range(4))
        if a * d != b * c:
            break
    left = a * x_value + b * y_value
    right = c * x_value + d * y_value
    expression = f"{a}*x+{b}*y={left}; {c}*x+{d}*y={right}"
    answer = f"x={x_value}, y={y_value}"
    return _task(
        family="system_2x2",
        index=index,
        prompt=(
            f"Solve the system {a}x + {b}y = {left}; {c}x + {d}y = {right}. "
            "Return exactly `x=<integer>, y=<integer>`."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "system-solution",
                "symbolic_math",
                answer,
                _symbolic_args("solve_system", expression, "x,y"),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _derivative(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    coefficients = [rng.randint(-9, 9) for _ in range(5)]
    coefficients[0] = coefficients[0] or rng.choice((-7, 7))
    expression_obj = sum(coefficients[i] * x ** (5 - i) for i in range(5)) + rng.randint(-20, 20)
    expression = sp.sstr(expression_obj)
    answer = sp.sstr(sp.diff(expression_obj, x))
    return _task(
        family="derivative",
        index=index,
        prompt=(
            f"Differentiate {expression} with respect to x. Return the exact derivative in "
            "canonical Python-style notation."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "derivative",
                "symbolic_math",
                answer,
                _symbolic_args("differentiate", expression),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _definite_integral(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    coefficients = [rng.randint(-6, 6) for _ in range(4)]
    coefficients[0] = coefficients[0] or rng.choice((-5, 5))
    expression_obj = sum(coefficients[i] * x ** (3 - i) for i in range(4))
    lower = rng.randint(-4, 1)
    upper = rng.randint(lower + 1, 6)
    expression = sp.sstr(expression_obj)
    result = sp.integrate(expression_obj, (x, lower, upper))
    answer = sp.sstr(sp.factor(result))
    return _task(
        family="definite_integral",
        index=index,
        prompt=(
            f"Compute the exact definite integral of {expression} from x={lower} to x={upper}. "
            "Return a simplified exact number."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "integral",
                "symbolic_math",
                answer,
                _symbolic_args("integrate", f"integral({expression},x,{lower},{upper})"),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _identity_check(rng: random.Random, index: int) -> TeacherTask:
    x = sp.Symbol("x")
    a, b, c = rng.randint(2, 25), rng.randint(-50, 50), rng.randint(-50, 50)
    left_obj = (a * x + b) * (x + c)
    right_obj = sp.expand(left_obj)
    equivalent = index % 2 == 0
    if not equivalent:
        delta = rng.randint(1, 9)
        right_obj += delta if rng.random() < 0.5 else -delta
    left = f"({a}*x+({b}))*(x+({c}))"
    right = sp.sstr(right_obj)
    answer = "yes" if equivalent else "no"
    return _task(
        family="identity_check",
        index=index,
        prompt=(
            f"Are {left} and {right} identically equal for all x? Return exactly `yes` or `no`."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(),),
        evidence=(
            _evidence(
                "equivalence",
                "symbolic_math",
                answer,
                _symbolic_args("equivalent", f"{left} == {right}"),
            ),
        ),
        pattern="single_symbolic_tool",
        depth=1,
    )


def _symbolic_then_calculator(rng: random.Random, index: int) -> TeacherTask:
    solution = rng.randint(-80, 80)
    a = rng.randint(3, 17)
    c = rng.randint(2, 13)
    while c == a:
        c = rng.randint(2, 13)
    b = rng.randint(-50, 50)
    target = a * (solution + b) - c * solution
    equation = f"{a}*(x+({b}))-{c}*x={target}"
    denominator = rng.randint(3, 19)
    offset = rng.randint(10, 200)
    numeric = round((solution**2 + offset) / denominator, 2)
    answer = f"{numeric:.2f}"
    return _task(
        family="symbolic_then_calculator",
        index=index,
        prompt=(
            f"First solve {a}(x {b:+d}) - {c}x = {target}. Then compute "
            f"(x^2 + {offset}) / {denominator}. Give the final value with exactly two decimals."
        ),
        answer=answer,
        descriptors=(symbolic_math_descriptor(), calculator_descriptor()),
        evidence=(
            _evidence(
                "symbolic-solution",
                "symbolic_math",
                str(solution),
                _symbolic_args("solve", equation),
            ),
            _evidence(
                "numeric-evaluation",
                "calculator",
                answer,
                {"expression": f"round((({solution})**2+{offset})/{denominator},2)"},
                depends_on=("symbolic-solution",),
            ),
        ),
        pattern="symbolic_then_calculator",
        depth=2,
    )


def _lookup_then_symbolic(rng: random.Random, index: int) -> TeacherTask:
    key = f"equation-{index:05d}"
    solution = rng.randint(-100, 100)
    a, b = rng.randint(3, 25), rng.randint(-100, 100)
    target = a * solution + b
    record = {"a": a, "b": b, "target": target}
    equation = f"{a}*x+({b})={target}"
    return _task(
        family="lookup_then_symbolic",
        index=index,
        prompt=(
            f"Read coefficient record {key}, which defines a*x + b = target. Solve exactly for x "
            "and return the integer solution."
        ),
        answer=str(solution),
        descriptors=(record_lookup_descriptor(), symbolic_math_descriptor()),
        evidence=(
            _evidence("equation-record", "record_lookup", record, {"key": key}),
            _evidence(
                "record-solution",
                "symbolic_math",
                str(solution),
                _symbolic_args("solve", equation),
                depends_on=("equation-record",),
            ),
        ),
        pattern="lookup_then_symbolic",
        depth=2,
    )


def _parallel_symbolic_then_merge(rng: random.Random, index: int) -> TeacherTask:
    x_value, y_value = rng.randint(-120, 120), rng.randint(-120, 120)
    ax, bx = rng.randint(4, 23), rng.randint(-90, 90)
    ay, by = rng.randint(4, 23), rng.randint(-90, 90)
    tx, ty = ax * x_value + bx, ay * y_value + by
    difference = abs(x_value - y_value)
    return _task(
        family="parallel_symbolic_then_merge",
        index=index,
        prompt=(
            f"Solve {ax}x {bx:+d} = {tx} and independently solve {ay}y {by:+d} = {ty}. "
            "Then return |x-y| as an integer."
        ),
        answer=str(difference),
        descriptors=(symbolic_math_descriptor(), calculator_descriptor()),
        evidence=(
            _evidence(
                "x-solution",
                "symbolic_math",
                str(x_value),
                _symbolic_args("solve", f"{ax}*x+({bx})={tx}", "x"),
            ),
            _evidence(
                "y-solution",
                "symbolic_math",
                str(y_value),
                _symbolic_args("solve", f"{ay}*y+({by})={ty}", "y"),
            ),
            _evidence(
                "merged-difference",
                "calculator",
                str(difference),
                {"expression": f"abs({x_value}-{y_value})"},
                depends_on=("x-solution", "y-solution"),
            ),
        ),
        pattern="parallel_symbolic_then_merge",
        depth=2,
    )


def _symbolic_unnecessary(rng: random.Random, index: int) -> TeacherTask:
    variant = index % 3
    if variant == 0:
        solution = rng.randint(-200, 200)
        offset = rng.randint(2, 80)
        total = solution + offset
        prompt = f"Solve for x: x + {offset} = {total}."
        answer = str(solution)
    elif variant == 1:
        left = rng.randint(2, 40)
        right = rng.randint(2, 40)
        offset = rng.randint(2, 90)
        prompt = f"Expand ({left}*x + {offset}) + ({right}*x - {offset})."
        answer = f"{left + right}*x"
    else:
        left = rng.randint(2, 40)
        right = rng.randint(2, 40)
        first = rng.randint(-50, 50)
        second = rng.randint(-50, 50)
        prompt = (
            f"Are ({left}*x {first:+d}) + ({right}*x {second:+d}) and "
            f"({right}*x {second:+d}) + ({left}*x {first:+d}) identically equal? "
            "Return yes or no."
        )
        answer = "yes"
    return _task(
        family="symbolic_unnecessary",
        index=index,
        prompt=prompt,
        answer=answer,
        descriptors=(symbolic_math_descriptor(), calculator_descriptor()),
        evidence=(),
        pattern="tools_available_unnecessary",
        depth=0,
        training_mode="tools_available_unnecessary",
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
        task_id=f"symbolic-{family}-{index:06d}",
        prompt=prompt,
        source_descriptors=descriptors,
        evidence=evidence,
        metadata={
            "task_kind": "symbolic_reasoning",
            "family": family,
            "interaction_pattern": pattern,
            "dependency_depth": depth,
            "training_mode": training_mode,
            "generated_by": "cid.symbolic_training.v1",
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
        provenance="cid.symbolic_training.v1",
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

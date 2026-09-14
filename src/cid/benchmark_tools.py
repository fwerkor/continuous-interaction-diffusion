from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from cid.tool_schemas import calculator_descriptor, python_descriptor, symbolic_math_descriptor

BENCHMARK_LIVE_TOOLS_METADATA_KEY = "benchmark_live_tools"
BENCHMARK_ARGUMENT_CANDIDATES_METADATA_KEY = "benchmark_tool_argument_candidates"
BENCHMARK_LIVE_TOOL_LATENCY_STEPS_METADATA_KEY = "benchmark_live_tool_latency_steps"

_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_INLINE_MATH_RE = re.compile(r"\$([^$\n]{1,240})\$|\\\((.{1,240}?)\\\)|\\\[(.{1,240}?)\\\]")
_VARIABLE_RE = re.compile(r"\b([a-zA-Z])\b")
_SAFE_SYMBOLIC_RE = re.compile(r"^[0-9A-Za-z_+\-*/%().,=; <>]+$")
_CALCULATOR_FUNCTIONS = frozenset({"abs", "floor", "sqrt", "log", "round"})


def benchmark_tool_descriptors(task_kind: str) -> tuple[dict[str, Any], ...]:
    """Select benchmark tools from the exact schemas used by CID training."""

    if task_kind == "math_word_problem":
        return (calculator_descriptor(), python_descriptor())
    if task_kind == "competition_math":
        return (symbolic_math_descriptor(), calculator_descriptor(), python_descriptor())
    if task_kind in {"science_multiple_choice", "multiple_choice_knowledge_reasoning"}:
        return (calculator_descriptor(),)
    return ()


def benchmark_argument_candidates(
    prompt: str,
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    """Build a gold-free closed-world argument catalog from the visible prompt only."""

    source_names = {str(item.get("name", "")) for item in descriptors}
    result: dict[str, dict[str, list[Any]]] = {}

    def add(source: str, name: str, value: Any) -> None:
        if source not in source_names:
            return
        encoded = str(value).strip()
        if not encoded or len(encoded) > 512:
            return
        values = result.setdefault(source, {}).setdefault(name, [])
        if value not in values:
            values.append(value)

    calculator_expressions = _calculator_candidates(prompt)
    for expression in calculator_expressions:
        add("calculator", "expression", expression)

    if "python" in source_names:
        for code in _python_candidates(prompt, calculator_expressions):
            add("python", "code", code)

    if "symbolic_math" in source_names:
        for operation in _symbolic_operations(prompt):
            add("symbolic_math", "operation", operation)
        for expression in _symbolic_expressions(prompt):
            add("symbolic_math", "expression", expression)
        for variables in _symbolic_variables(prompt):
            add("symbolic_math", "variables", variables)

    return result


def benchmark_live_tool_names(
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    supported = {"calculator", "python", "symbolic_math"}
    return tuple(
        name
        for item in descriptors
        for name in (str(item.get("name", "")),)
        if name in supported
    )


def _numbers(prompt: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _NUMBER_RE.finditer(prompt):
        value = match.group(0).replace(",", "")
        if value not in values:
            values.append(value)
        if len(values) >= 10:
            break
    return tuple(values)


def _calculator_candidates(prompt: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(value: str) -> None:
        normalized = _normalize_math_fragment(value)
        if (
            normalized
            and len(normalized) <= 256
            and any(char.isdigit() for char in normalized)
            and any(operator_char in normalized for operator_char in "+-*/%")
            and _is_calculator_expression(normalized)
            and normalized not in candidates
        ):
            candidates.append(normalized)

    for fragment in _math_fragments(prompt):
        add(fragment)

    numbers = _numbers(prompt)
    if len(numbers) >= 2:
        add("+".join(numbers))
        add("*".join(numbers[:6]))
        if "average" in prompt.casefold() or "mean" in prompt.casefold():
            add(f"({'+'.join(numbers)})/{len(numbers)}")
        if len(numbers) >= 3:
            a, b, c = numbers[:3]
            for expression in (
                f"{a}*{b}+{c}",
                f"{a}*{b}-{c}",
                f"{a}+{b}*{c}",
                f"{a}-{b}*{c}",
                f"({a}+{b})*{c}",
                f"({a}-{b})*{c}",
                f"{a}*{b}/{c}",
                f"{a}/{b}*{c}",
            ):
                add(expression)
        if len(numbers) >= 4:
            a, b, c, d = numbers[:4]
            for expression in (
                f"{a}*{b}+{c}*{d}",
                f"{a}*{b}-{c}*{d}",
                f"({a}+{b})*({c}+{d})",
                f"({a}-{b})*({c}-{d})",
            ):
                add(expression)
        for left, right in zip(numbers, numbers[1:], strict=False):
            add(f"{left}+{right}")
            add(f"{left}-{right}")
            add(f"{left}*{right}")
            if float(right) != 0.0:
                add(f"{left}/{right}")
            if len(candidates) >= 28:
                break

    return tuple(candidates[:32])


def _is_calculator_expression(expression: str) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    if sum(1 for _ in ast.walk(tree)) > 128:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _CALCULATOR_FUNCTIONS
                or node.keywords
            ):
                return False
        elif isinstance(node, ast.Name):
            if node.id not in _CALCULATOR_FUNCTIONS:
                return False
        elif not isinstance(
            node,
            (
                ast.Expression,
                ast.Constant,
                ast.UnaryOp,
                ast.UAdd,
                ast.USub,
                ast.BinOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.Load,
            ),
        ):
            return False
    return True


def _python_candidates(prompt: str, calculator_expressions: Sequence[str]) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and len(value) <= 512 and value not in candidates:
            candidates.append(value)

    for expression in calculator_expressions[:16]:
        add(expression)
    numbers = _numbers(prompt)
    if numbers:
        payload = "[" + ",".join(numbers) + "]"
        add(f"sum({payload})")
        add(f"len({payload})")
        add(f"sorted({payload})")
        add(f"sum({payload})/len({payload})")
    text = prompt.casefold()
    enumeration = re.search(
        r"from\s+1\s+(?:through|to)\s+(\d+).*?"
        r"divisible\s+by\s+exactly\s+one\s+of\s+(\d+)\s+and\s+(\d+)",
        text,
    )
    if enumeration is not None:
        upper, left, right = enumeration.groups()
        add(
            f"sum(1 for x in range(1,{upper}+1) if "
            f"(x%{left}==0) ^ (x%{right}==0))"
        )
    return tuple(candidates[:24])


def _symbolic_operations(prompt: str) -> tuple[str, ...]:
    text = prompt.casefold()
    ranked: list[str] = []

    def add(value: str) -> None:
        if value not in ranked:
            ranked.append(value)

    if "differentiat" in text or "derivative" in text:
        add("differentiate")
    if "integral" in text or "integrate" in text:
        add("integrate")
    if "factor" in text:
        add("factor")
    if "expand" in text:
        add("expand")
    if "simplif" in text:
        add("simplify")
    if "system" in text:
        add("solve_system")
    if "solve" in text or "root" in text or "solution" in text:
        add("solve")
    if "identically" in text or "equivalent" in text or "equal for all" in text:
        add("equivalent")
    for fallback in ("solve", "simplify", "factor", "expand"):
        add(fallback)
    return tuple(ranked[:6])


def _symbolic_expressions(prompt: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(value: str) -> None:
        normalized = _normalize_math_fragment(value)
        if (
            normalized
            and len(normalized) <= 384
            and _SAFE_SYMBOLIC_RE.fullmatch(normalized)
            and normalized not in candidates
        ):
            candidates.append(normalized)

    for fragment in _math_fragments(prompt):
        add(fragment)
    for line in prompt.splitlines():
        if any(token in line for token in ("=", "^", "*", "/")):
            add(line)
    return tuple(candidates[:12])


def _symbolic_variables(prompt: str) -> tuple[str, ...]:
    variables: list[str] = []
    for match in _VARIABLE_RE.finditer(prompt):
        value = match.group(1)
        if value.casefold() in {"a", "i"}:
            continue
        if value not in variables:
            variables.append(value)
        if len(variables) >= 3:
            break
    if not variables:
        return ("x", "x,y")
    joined = ",".join(variables)
    return tuple(dict.fromkeys((joined, variables[0], "x", "x,y")))[:4]


def _math_fragments(prompt: str) -> tuple[str, ...]:
    fragments: list[str] = []
    for match in _INLINE_MATH_RE.finditer(prompt):
        value = next((item for item in match.groups() if item is not None), "")
        if value.strip() and value.strip() not in fragments:
            fragments.append(value.strip())
    for match in re.finditer(r"(?<!\w)([-+*/()0-9., ]{5,})(?!\w)", prompt):
        value = match.group(1).strip(" ,.")
        if value and value not in fragments:
            fragments.append(value)
    return tuple(fragments[:16])


def _normalize_math_fragment(value: str) -> str:
    text = value.strip()
    replacements = {
        "−": "-",
        "–": "-",
        "×": "*",
        "÷": "/",
        "\\times": "*",
        "\\cdot": "*",
        "\\div": "/",
        "\\left": "",
        "\\right": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", text)
    text = re.sub(r"(?<=\w)\^(?=[\w(])", "**", text)
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    return text.strip("$.,:;")

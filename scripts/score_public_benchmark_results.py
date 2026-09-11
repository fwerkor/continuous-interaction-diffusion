from __future__ import annotations

import argparse
import ast
import json
import re
import resource
import string
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SAFE_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
}
_DANGEROUS_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "help",
    "input",
    "open",
    "quit",
}
_DANGEROUS_ATTRS = {
    "chmod",
    "connect",
    "fork",
    "kill",
    "open",
    "popen",
    "rename",
    "rmdir",
    "send",
    "socket",
    "spawn",
    "system",
    "unlink",
    "write_bytes",
    "write_text",
}


def _normalize_qa(value: str) -> str:
    lowered = value.casefold()
    without_punctuation = "".join(" " if char in string.punctuation else char for char in lowered)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _qa_f1(prediction: str, reference: str) -> float:
    predicted = _normalize_qa(prediction).split()
    expected = _normalize_qa(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _qa_metrics(prediction: str, references: list[str]) -> tuple[bool, float]:
    exact = max(
        (_normalize_qa(prediction) == _normalize_qa(item) for item in references),
        default=False,
    )
    f1 = max((_qa_f1(prediction, item) for item in references), default=0.0)
    return exact, f1


def _last_number(value: str) -> Decimal | None:
    matches = _NUMBER.findall(value)
    if not matches:
        return None
    token = matches[-1].replace(",", "")
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return number.normalize()


def _numeric_equal(prediction: str, reference: str) -> bool:
    predicted = _last_number(prediction)
    expected = _last_number(reference)
    return predicted is not None and expected is not None and predicted == expected


def _normalize_math(value: str) -> str:
    compact = value.strip().replace("$", "")
    compact = compact.replace("\\left", "").replace("\\right", "")
    compact = compact.replace("\\!", "").replace("\\,", "")
    compact = compact.replace("−", "-")
    compact = re.sub(r"\s+", "", compact)
    while compact.endswith((".", ",", ";")):
        compact = compact[:-1]
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", compact)
    if boxed:
        compact = boxed.group(1)
    return compact


def _math_equal(prediction: str, reference: str) -> bool:
    if _normalize_math(prediction) == _normalize_math(reference):
        return True
    return _numeric_equal(prediction, reference)


def _extract_code(value: str) -> str:
    block = _CODE_BLOCK.search(value)
    return (block.group(1) if block else value).strip()


def _code_is_safe(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in _SAFE_IMPORTS:
                    return False, f"blocked-import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in _SAFE_IMPORTS:
                return False, f"blocked-import:{node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_CALLS:
                return False, f"blocked-call:{node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_ATTRS:
            return False, f"blocked-attribute:{node.attr}"
    return True, None


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _mbpp_passes(record: dict[str, Any]) -> tuple[bool, str | None]:
    code = _extract_code(str(record["final_text"]))
    safe, reason = _code_is_safe(code)
    if not safe:
        return False, reason
    metadata = record.get("metadata", {})
    setup = str(metadata.get("public_test_setup_code", ""))
    tests = [str(item) for item in metadata.get("public_tests", ())]
    if not tests:
        return False, "missing-tests"
    program = "\n".join((setup, code, *tests))
    with tempfile.TemporaryDirectory(prefix="cid-mbpp-") as tempdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=tempdir,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                preexec_fn=_limit_child,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if completed.returncode == 0:
        return True, None
    error = completed.stderr.strip().splitlines()
    return False, error[-1][:200] if error else f"exit:{completed.returncode}"


def _score_record(dataset: str, record: dict[str, Any]) -> dict[str, Any]:
    prediction = str(record.get("final_text", ""))
    reference = str(record.get("target_display", ""))
    metadata = record.get("metadata", {})
    scored: dict[str, Any] = {}
    if dataset == "gsm8k-test":
        scored["numeric_exact"] = _numeric_equal(prediction, reference)
    elif dataset == "math-test":
        scored["math_exact"] = _math_equal(prediction, reference)
    elif dataset == "mbpp-test":
        passed, error = _mbpp_passes(record)
        scored["pass_at_1"] = passed
        scored["execution_error"] = error
    elif dataset in {
        "hotpotqa-distractor-validation",
        "2wikimultihopqa-dev",
        "musique-validation",
    }:
        aliases = [reference]
        aliases.extend(
            str(item) for item in metadata.get("answer_aliases", ()) if str(item).strip()
        )
        exact, f1 = _qa_metrics(prediction, aliases)
        scored["qa_exact"] = exact
        scored["qa_f1"] = f1
    return scored


def score_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset = dataset_dir.name
    input_path = dataset_dir / "results.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = dataset_dir / "scored-results.jsonl"
    totals: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    examples = 0
    error_counts: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            scored = _score_record(dataset, record)
            record["standard_metrics"] = scored
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            examples += 1
            for key, value in scored.items():
                if isinstance(value, bool):
                    totals[key] += 1
                    sums[key] += int(value)
            error = scored.get("execution_error")
            if error:
                error_counts[str(error).split(":", 1)[0]] += 1

    metrics: dict[str, Any] = {"examples": examples}
    for key, count in totals.items():
        metrics[key] = sums[key] / count if count else 0.0
    if dataset in {
        "hotpotqa-distractor-validation",
        "2wikimultihopqa-dev",
        "musique-validation",
    }:
        with output_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        metrics["qa_f1"] = (
            sum(float(row["standard_metrics"].get("qa_f1", 0.0)) for row in rows) / len(rows)
            if rows
            else 0.0
        )
    if error_counts:
        metrics["execution_error_classes"] = dict(error_counts.most_common())
    summary_path = dataset_dir / "standard-metrics.json"
    summary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+")
    args = parser.parse_args()
    for raw in args.dataset_dirs:
        path = Path(raw)
        if not (path / "results.jsonl").is_file():
            print(f"SKIP {path}: results.jsonl not ready", flush=True)
            continue
        metrics = score_dataset(path)
        print(path.name, json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

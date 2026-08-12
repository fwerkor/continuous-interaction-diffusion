from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence

_ALLOWED_IMPORTS = frozenset(
    {
        "ast",
        "bisect",
        "collections",
        "datetime",
        "functools",
        "heapq",
        "itertools",
        "math",
        "re",
        "statistics",
    }
)
_FORBIDDEN_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


class _CandidateSafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORTS:
                raise ValueError(f"python answer imports unsupported module {root!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise ValueError("python answer must not use relative imports")
        root = (node.module or "").split(".", 1)[0]
        if root not in _ALLOWED_IMPORTS:
            raise ValueError(f"python answer imports unsupported module {root!r}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"python answer uses forbidden name {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("python answer must not access dunder attributes")
        self.generic_visit(node)


def validate_python_candidate_source(source: str) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"python answer has invalid syntax: {exc.msg}") from exc
    _CandidateSafetyVisitor().visit(tree)


_RUNNER = r"""
import builtins
import json
import resource
import sys

ALLOWED = {
    "ast", "bisect", "collections", "datetime", "functools", "heapq",
    "itertools", "math", "re", "statistics",
}
DENIED_BUILTINS = {
    "breakpoint", "compile", "delattr", "dir", "eval", "exec", "getattr",
    "globals", "help", "input", "locals", "open", "setattr", "vars",
}

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

payload = json.load(sys.stdin)
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level or root not in ALLOWED:
        raise ImportError(f"unsupported import: {name}")
    return real_import(name, globals, locals, fromlist, level)

safe_builtins = {
    name: value
    for name, value in vars(builtins).items()
    if name not in DENIED_BUILTINS and name != "__import__"
}
safe_builtins["__import__"] = guarded_import
namespace = {"__builtins__": safe_builtins, "__name__": "__cid_candidate__"}

try:
    exec(payload["code"], namespace, namespace)
    setup = payload.get("setup", "")
    if setup:
        exec(setup, namespace, namespace)
    for index, test in enumerate(payload.get("tests", [])):
        try:
            exec(test, namespace, namespace)
        except Exception as exc:
            raise AssertionError(f"public test {index} failed: {exc}") from exc
except BaseException as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(3)
"""


def python_public_test_reason(
    source: str,
    tests: Sequence[str],
    setup: str = "",
    *,
    timeout_seconds: float = 3.0,
) -> str | None:
    if not tests:
        return "python task has no public tests"
    try:
        validate_python_candidate_source(source)
    except ValueError as exc:
        return str(exc)

    payload = json.dumps(
        {"code": source, "setup": setup, "tests": list(tests)},
        ensure_ascii=False,
    )
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PATH": os.environ.get("PATH", ""),
    }
    with tempfile.TemporaryDirectory(prefix="cid-python-review-") as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=payload,
                text=True,
                capture_output=True,
                cwd=workdir,
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "python answer timed out on public tests"
    if result.returncode == 0:
        return None
    detail = result.stderr.strip().splitlines()
    suffix = detail[-1][:240] if detail else f"exit status {result.returncode}"
    return f"python answer fails public tests: {suffix}"

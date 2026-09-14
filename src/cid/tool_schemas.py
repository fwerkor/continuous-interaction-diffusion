from __future__ import annotations

from typing import Any


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

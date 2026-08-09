from __future__ import annotations

import json
from pathlib import Path

import pytest

from cid.public_tasks import (
    PublicTaskRowRejected,
    _adapt_row,
    _last_boxed_value,
    deterministic_split,
)


def test_deterministic_split_is_content_keyed() -> None:
    spec = {"train": 0.9, "validation": 0.05, "test": 0.05}
    first = deterministic_split("abc123", spec)
    second = deterministic_split("abc123", spec)
    assert first == second
    assert first in {"train", "validation", "test"}


def test_deterministic_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        deterministic_split("abc123", {"train": 0.8, "validation": 0.1, "test": 0.2})


def test_gsm8k_adapter_keeps_solution_but_uses_final_answer() -> None:
    prompt, answer, resources, metadata = _adapt_row(
        "gsm8k",
        {"question": "What is 6 times 7?", "answer": "Compute 6*7.\n#### 42"},
    )
    assert prompt == "What is 6 times 7?"
    assert answer == "42"
    assert resources == {}
    assert metadata["reference_solution"].endswith("#### 42")


def test_math_box_parser_handles_nested_braces() -> None:
    assert _last_boxed_value(r"Therefore the answer is \boxed{\frac{3}{7}}.") == r"\frac{3}{7}"
    assert _last_boxed_value(r"The largest value is \boxed 2$.") == "2"


def test_math_adapter_rejects_missing_final_answer() -> None:
    with pytest.raises(PublicTaskRowRejected):
        _adapt_row(
            "hendrycks_math",
            {
                "problem": "How many?",
                "solution": r"There are \boxed{} values.",
                "level": "Level 1",
                "type": "Algebra",
            },
        )


def test_mmlu_adapter_makes_choices_visible() -> None:
    prompt, answer, _, metadata = _adapt_row(
        "mmlu",
        {"question": "Pick one", "choices": ["x", "y", "z"], "answer": 1, "subject": "demo"},
    )
    assert "A. x" in prompt
    assert "B. y" in prompt
    assert answer == "y"
    assert metadata["answer_index"] == 1


def test_hotpot_adapter_keeps_evidence_out_of_prompt() -> None:
    prompt, answer, resources, metadata = _adapt_row(
        "hotpotqa",
        {
            "id": "h1",
            "question": "Which came first?",
            "answer": "A",
            "type": "comparison",
            "level": "medium",
            "context": {
                "title": ["A", "B"],
                "sentences": [["A was founded in 1900."], ["B was founded in 1950."]],
            },
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
        },
    )
    assert prompt == "Which came first?"
    assert "1900" not in prompt
    assert answer == "A"
    assert resources["evidence_bank"][0]["title"] == "A"
    assert metadata["question_type"] == "comparison"


def test_arc_adapter_resolves_labeled_answer() -> None:
    prompt, answer, _, metadata = _adapt_row(
        "arc",
        {
            "id": "arc-1",
            "question": "Which?",
            "choices": {"label": ["A", "B"], "text": ["cold", "hot"]},
            "answerKey": "B",
        },
    )
    assert prompt.endswith("B. hot")
    assert answer == "hot"
    assert metadata["answer_label"] == "B"


def test_public_dataset_registry_uses_only_training_sources() -> None:
    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "data/public-datasets.json").read_text(
            encoding="utf-8"
        )
    )
    assert sum(source["quota"] for source in registry["sources"]) == 10_000
    assert all(source["upstream_split"] != "test" for source in registry["sources"])
    assert registry["expected_sha256"] == (
        "bb92a3d6fccad2687cd5805ec5aa3eaff404644dea76cce153a088853d49f17b"
    )

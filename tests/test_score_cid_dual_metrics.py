from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_cid_dual_metrics import score_record


def row(ref: str, text: str, *, metadata: dict | None = None) -> dict:
    return {
        "target_display": ref,
        "final_text": text,
        "metadata": metadata or {},
    }


def test_known_correct_final_wrong_is_discovery_only():
    result = score_record(
        "hotpotqa-distractor-validation",
        row("Marlon Stöckinger", "Known: Marlon Stöckinger — Answer: 50"),
    )
    assert result["task_accuracy"] is False
    assert result["answer_discovery"] is True
    assert result["needs_semantic_review"] is True


def test_final_correct_without_exact_whole_string():
    result = score_record(
        "hotpotqa-distractor-validation",
        row("Marlon Stöckinger", "Known: 50 — Answer: Marlon Stöckinger"),
    )
    assert result["task_accuracy"] is True
    assert result["answer_discovery"] is True


def test_yes_or_no_with_unambiguous_answer():
    result = score_record(
        "hotpotqa-distractor-validation", row("yes", "Yes, Silene.")
    )
    assert result["task_accuracy"] is True


def test_yes_and_no_conflict_is_not_task_success():
    result = score_record(
        "hotpotqa-distractor-validation", row("yes", "Yes. No. Answer: No.")
    )
    assert result["task_accuracy"] is False
    assert result["answer_discovery"] is True


def test_answer_alias_is_discovered():
    result = score_record(
        "2wikimultihopqa-dev",
        row("New York City", "Known: NYC", metadata={"answer_aliases": ["NYC"]}),
    )
    assert result["answer_discovery"] is True
    assert result["task_accuracy"] is False


def test_choice_final_selection_overrides_known_reference():
    result = score_record(
        "arc-challenge-test",
        row(
            "green",
            "Known: green; Answer: B",
            metadata={
                "choices": ["red", "blue", "green", "yellow"],
                "answer_index": 2,
            },
        ),
    )
    assert result["task_accuracy"] is False
    assert result["answer_discovery"] is True


def test_choice_single_exact_answer():
    result = score_record(
        "arc-challenge-test",
        row("green", "green", metadata={
            "choices": ["red", "blue", "green", "yellow"], "answer_index": 2
        }),
    )
    assert result["task_accuracy"] is True


def test_choice_echo_all_options_gives_no_credit():
    result = score_record(
        "arc-challenge-test",
        row("green", "red; blue; green; yellow", metadata={
            "choices": ["red", "blue", "green", "yellow"], "answer_index": 2
        }),
    )
    assert result["task_accuracy"] is False
    assert result["answer_discovery"] is False


def test_math_sign_and_expression_boundaries():
    for ref, output in [("-2", "2"), ("1", "-1"), ("4", "4-4"),
                        ("7", r"\frac{7}{6}")]:
        result = score_record("math-test", row(ref, output))
        assert result["task_accuracy"] is False
        assert result["answer_discovery"] is False


def test_math_known_correct_final_wrong():
    result = score_record(
        "math-test", row("4", "Known: 4; Answer: 50")
    )
    assert result["task_accuracy"] is False
    assert result["answer_discovery"] is True


def test_numeric_correct_and_wrong_final():
    good = score_record("gsm8k-test", row("18", "Answer: $18.00"))
    wrong = score_record(
        "gsm8k-test", row("18", "Known: 18; Answer: 50")
    )
    assert good["task_accuracy"] is True
    assert wrong["task_accuracy"] is False
    assert wrong["answer_discovery"] is True

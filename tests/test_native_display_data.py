from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from merge_native_full_replies import assemble  # noqa: E402
from validate_native_display_data import validate_row  # noqa: E402


def test_native_reply_data_validation_and_atomic_merge(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    review = tmp_path / "review.jsonl"
    output = tmp_path / "out.jsonl"
    base = {
        "example_id": "test-example",
        "prompt": "What is the value?",
        "target_display": "37",
        "display_targets": [{"step": 1, "text": "37"}],
    }
    source.write_text(json.dumps(base) + "\n")
    review.write_text(json.dumps({
        "example_id": "test-example",
        "full_reply": "The recorded value is 37.",
        "verified": True,
        "reviewer": "human-review",
    }) + "\n")
    assert validate_row(base)
    result = assemble(source, review, output)
    assert result["rows"] == 1
    merged = json.loads(output.read_text())
    assert not validate_row(merged)
    assert merged["metadata"]["native_full_display"] == "The recorded value is 37."
    assert merged["metadata"]["native_full_display_reviewer"] == "human-review"


def test_merge_rejects_unreviewed_or_missing_without_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    review = tmp_path / "review.jsonl"
    output = tmp_path / "out.jsonl"
    source.write_text(json.dumps({
        "example_id": "test",
        "prompt": "Question",
        "target_display": "42",
    }) + "\n")
    review.write_text(json.dumps({
        "example_id": "test",
        "full_reply": "The answer is 42.",
        "verified": False,
        "reviewer": "none",
    }) + "\n")
    with pytest.raises(ValueError, match="verified review"):
        assemble(source, review, output)
    assert not output.exists()
    review.write_text("")
    with pytest.raises(ValueError, match="no verified full reply"):
        assemble(source, review, output)
    assert not output.exists()


def test_bad_full_reply_is_not_automatically_accepted() -> None:
    row = {
        "target_display": "37",
        "metadata": {"native_full_display": "The answer is 999."},
    }
    assert validate_row(row)

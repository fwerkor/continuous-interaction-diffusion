from __future__ import annotations

import json

import pytest

from cid.data import TrajectoryExample
from scripts.build_cid_dataset_v13 import annotation, build, materialize, pair_example


def test_source_annotation_requires_complete_available_answer():
    row = pair_example(0, 1, 0, 147, 284, "train")
    row["metadata"] = {}
    row["display_targets"].insert(0, {"step": -1, "text": "147"})
    item = annotation(row)
    assert item is not None
    assert item["steps"] == [1, 2]
    row["prompt"] += " Explain your reasoning."
    assert annotation(row) is None


def test_real_evidence_counterfactual():
    a = pair_example(1, 2, 0, 147, 284, "train")
    b = pair_example(1, 2, 1, 284, 147, "train")
    left, right = TrajectoryExample.from_dict(a), TrajectoryExample.from_dict(b)
    assert left.prompt == right.prompt
    assert left.thought_targets[0] == right.thought_targets[0]
    assert left.events[0].arguments == right.events[0].arguments
    assert left.events[0].value != right.events[0].value
    assert a["metadata"]["native_counterfactual_by_step"]["1"] == right.target_display
    assert b["metadata"]["native_counterfactual_by_step"]["1"] == left.target_display
    assert "147" not in left.thought_targets[0].semantic_text
    assert "284" not in right.thought_targets[0].semantic_text


def test_materialization_preserves_base_and_excludes_validation(tmp_path):
    row = pair_example(0, 1, 0, 147, 284, "base")
    row["metadata"] = {}
    base = tmp_path / "base.jsonl"
    original = json.dumps(row) + "\n"
    base.write_text(original)
    release = tmp_path / "release"
    build(base, release, 2, 1)
    out = tmp_path / "train.jsonl"
    materialize(base, release, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5
    assert base.read_text() == original
    assert rows[0]["metadata"]["native_full_display_by_step"] == {"1": "147", "2": "147"}
    assert rows[0]["metadata"]["native_full_display_contract"] == "source-complete-v1"
    assert all(r.get("metadata", {}).get("split") != "validation" for r in rows)
    with (release / "counterfactual-train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        materialize(base, release, out)

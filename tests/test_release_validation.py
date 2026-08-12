from __future__ import annotations

import json
from pathlib import Path

import pytest

from cid.release_validation import (
    assert_release_references_resolve,
    find_missing_manifest_references,
)


def test_release_validation_finds_nested_missing_reference() -> None:
    manifests = {
        "manifests/mixture.json": {
            "components": [
                {
                    "name": "interaction",
                    "reference_manifest": "data/interaction/reference.json",
                }
            ]
        }
    }

    issues = find_missing_manifest_references(
        manifests,
        {"manifests/mixture.json"},
    )

    assert len(issues) == 1
    assert issues[0].manifest_path == "manifests/mixture.json"
    assert issues[0].field_path == "components[0].reference_manifest"
    assert issues[0].target == "data/interaction/reference.json"


def test_release_validation_accepts_self_contained_release(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "data" / "interaction").mkdir(parents=True)
    (tmp_path / "data" / "interaction" / "reference.json").write_text(
        json.dumps({"name": "interaction"}) + "\n"
    )
    (tmp_path / "manifests" / "mixture.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "reference_manifest": "data/interaction/reference.json",
                    }
                ]
            }
        )
        + "\n"
    )
    paths = {
        "manifests/mixture.json",
        "data/interaction/reference.json",
    }

    assert_release_references_resolve(tmp_path, paths)


def test_release_validation_rejects_missing_path(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "mixture.json").write_text(
        json.dumps({"path": "data/missing/trajectories.jsonl"}) + "\n"
    )

    with pytest.raises(ValueError, match="missing manifest references"):
        assert_release_references_resolve(
            tmp_path,
            {"manifests/mixture.json"},
        )

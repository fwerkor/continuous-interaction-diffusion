from __future__ import annotations

import json

import pytest

from cid.data import dump_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
from cid.synthetic import SyntheticConfig, generate_synthetic
from cid.trajectory_mixture import materialize_trajectory_mixture


def _component(tmp_path, name, examples):
    data = tmp_path / f"{name}.jsonl"
    manifest_path = tmp_path / f"{name}.manifest.json"
    dump_jsonl(examples, data)
    manifest = inspect_dataset(data)
    dump_dataset_manifest(manifest, manifest_path)
    return data, manifest_path, manifest


def test_materialize_trajectory_mixture_preserves_component_bytes_and_counts(tmp_path) -> None:
    examples = generate_synthetic(SyntheticConfig(count_per_family=1, seed=11, thought_capacity=8))
    first_data, first_manifest_path, first = _component(tmp_path, "first", examples[:2])
    second_data, second_manifest_path, second = _component(tmp_path, "second", examples[2:])
    spec = {
        "name": "test-mixture",
        "version": 1,
        "components": [
            {
                "name": "first",
                "path": first_data.name,
                "manifest": first_manifest_path.name,
                "sha256": first.sha256,
                "examples": first.examples,
                "transitions": first.transitions,
                "bootstrap_transitions": first.bootstrap_transitions,
                "training_transitions": first.training_transitions,
            },
            {
                "name": "second",
                "path": second_data.name,
                "manifest": second_manifest_path.name,
                "sha256": second.sha256,
                "examples": second.examples,
                "transitions": second.transitions,
                "bootstrap_transitions": second.bootstrap_transitions,
                "training_transitions": second.training_transitions,
            },
        ],
    }
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    spec_path = manifests_dir / "mixture.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "combined.jsonl"
    manifest_output = tmp_path / "combined.manifest.json"

    manifest = materialize_trajectory_mixture(spec_path, output, manifest_output)

    assert output.read_bytes() == first_data.read_bytes() + second_data.read_bytes()
    assert manifest["examples"] == len(examples)
    assert manifest["transitions"] == first.transitions + second.transitions
    assert manifest["bootstrap_transitions"] == len(examples)
    assert manifest["training_transitions"] == (
        first.training_transitions + second.training_transitions
    )
    assert manifest["trainable_examples"] == len(examples)
    assert manifest["zero_training_transition_examples"] == 0
    assert manifest["thought_capacity_required"] <= 8
    inspected = inspect_dataset(output)
    assert inspected.sha256 == manifest["sha256"]
    assert inspected.examples == manifest["examples"]
    assert inspected.transitions == manifest["transitions"]
    assert inspected.bootstrap_transitions == manifest["bootstrap_transitions"]
    assert inspected.training_transitions == manifest["training_transitions"]
    assert inspected.tag_counts == manifest["tag_counts"]
    assert list(inspected.sources) == manifest["sources"]
    assert inspected.thought_capacity_required == manifest["thought_capacity_required"]
    assert inspected.max_trajectory_steps == manifest["max_trajectory_steps"]


def test_materialize_trajectory_mixture_rejects_duplicate_example_ids(tmp_path) -> None:
    examples = generate_synthetic(SyntheticConfig(count_per_family=1, seed=17, thought_capacity=8))
    data, manifest_path, manifest = _component(tmp_path, "same", examples[:1])
    spec = {
        "name": "duplicate-mixture",
        "version": 1,
        "components": [
            {
                "name": "first",
                "path": data.name,
                "manifest": manifest_path.name,
                "sha256": manifest.sha256,
                "examples": manifest.examples,
                "transitions": manifest.transitions,
            },
            {
                "name": "second",
                "path": data.name,
                "manifest": manifest_path.name,
                "sha256": manifest.sha256,
                "examples": manifest.examples,
                "transitions": manifest.transitions,
            },
        ],
    }
    spec_path = tmp_path / "mixture.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate trajectory example_id"):
        materialize_trajectory_mixture(
            spec_path,
            tmp_path / "combined.jsonl",
            tmp_path / "combined.manifest.json",
        )


def test_materialize_trajectory_mixture_injects_component_semantic_weight(tmp_path) -> None:
    examples = generate_synthetic(SyntheticConfig(count_per_family=1, seed=23, thought_capacity=8))
    first_data, first_manifest_path, first = _component(tmp_path, "weighted", examples[:1])
    second_data, second_manifest_path, second = _component(tmp_path, "ordinary", examples[1:2])
    spec = {
        "name": "weighted-mixture",
        "version": 1,
        "components": [
            {
                "name": "weighted",
                "path": first_data.name,
                "manifest": first_manifest_path.name,
                "sha256": first.sha256,
                "examples": first.examples,
                "transitions": first.transitions,
                "semantic_weight": 2.5,
            },
            {
                "name": "ordinary",
                "path": second_data.name,
                "manifest": second_manifest_path.name,
                "sha256": second.sha256,
                "examples": second.examples,
                "transitions": second.transitions,
            },
        ],
    }
    spec_path = tmp_path / "weighted-mixture.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "combined.jsonl"

    manifest = materialize_trajectory_mixture(
        spec_path,
        output,
        tmp_path / "combined.manifest.json",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metadata"]["training_weight"] == pytest.approx(2.5)
    assert "training_weight" not in rows[1]["metadata"]
    assert manifest["components"][0]["semantic_weight"] == pytest.approx(2.5)
    assert manifest["components"][1]["semantic_weight"] == pytest.approx(1.0)

from __future__ import annotations

import hashlib
import json

import pytest

from cid.data import dump_jsonl
from cid.dataset import (
    dump_dataset_manifest,
    inspect_dataset,
    validate_neural_training_contract,
)
from cid.synthetic import SyntheticConfig, generate_synthetic


def test_dataset_manifest_is_deterministic_and_describes_training_shape(tmp_path) -> None:
    data = tmp_path / "synthetic.jsonl"
    manifest_path = tmp_path / "manifest.json"
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=1, seed=23, thought_capacity=8)
    )
    dump_jsonl(examples, data)

    first = inspect_dataset(data)
    second = inspect_dataset(data)

    assert first == second
    assert first.examples == 5
    assert first.transitions > first.examples
    assert first.bootstrap_transitions == first.examples
    assert first.training_transitions == first.transitions + first.bootstrap_transitions
    assert first.trainable_examples == first.examples
    assert first.zero_training_transition_examples == 0
    assert first.sha256 == hashlib.sha256(data.read_bytes()).hexdigest()
    assert first.thought_capacity_required <= 8
    assert first.max_trajectory_steps >= 3
    assert first.bindings == 6
    assert first.explicit_owner_bindings == 6
    assert first.multi_cell_bindings == 6
    assert first.global_display_fallback_bindings == 6
    assert set(first.tag_counts) == {
        "family:competing_sources",
        "family:delayed_retrieval",
        "family:dynamic_state",
        "family:static_copy",
        "family:streaming_evidence",
    }
    assert first.sources

    dump_dataset_manifest(first, manifest_path)
    payload = json.loads(manifest_path.read_text())
    assert payload["sha256"] == first.sha256
    assert payload["examples"] == 5
    assert payload["training_transitions"] == first.training_transitions
    assert payload["bootstrap_transitions"] == 5
    assert payload["schema"] == "cid.TrajectoryExample.v1"


def test_neural_training_contract_rejects_unmigrated_ownerless_data(tmp_path) -> None:
    data = tmp_path / "synthetic.jsonl"
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=1, seed=29, thought_capacity=8)
    )
    dump_jsonl(examples, data)
    rows = []
    for line in data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for binding in row.get("binding_targets", ()):
            binding.pop("owner_cell_id", None)
        rows.append(row)
    data.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = inspect_dataset(data)

    assert manifest.bindings > 0
    assert manifest.explicit_owner_bindings == 0
    with pytest.raises(ValueError, match="migrate-dataset-contract-v3"):
        validate_neural_training_contract(manifest)


def test_neural_training_contract_rejects_tool_data_without_multi_cell_routes(tmp_path) -> None:
    data = tmp_path / "single-cell.jsonl"
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=1, seed=31, thought_capacity=8)
    )
    dump_jsonl(examples, data)
    rows = []
    for line in data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for binding in row.get("binding_targets", ()):
            binding["target_cells"] = binding["target_cells"][:1]
        rows.append(row)
    data.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = inspect_dataset(data)

    assert manifest.bindings > 0
    assert manifest.explicit_owner_bindings == manifest.bindings
    assert manifest.multi_cell_bindings == 0
    with pytest.raises(ValueError, match="no multi-cell"):
        validate_neural_training_contract(manifest)

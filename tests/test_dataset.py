from __future__ import annotations

import hashlib
import json

from cid.data import dump_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
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
    assert first.sha256 == hashlib.sha256(data.read_bytes()).hexdigest()
    assert first.thought_capacity_required <= 8
    assert first.max_trajectory_steps >= 3
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
    assert payload["schema"] == "cid.TrajectoryExample.v1"

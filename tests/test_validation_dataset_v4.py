from __future__ import annotations

import json

from cid.curated_v4_training import CuratedV4Config, generate_curated_v4
from cid.data import trajectory_to_dict
from cid.validation_dataset_v4 import build_contract_v4_validation


def test_build_contract_v4_validation_is_contract_clean(tmp_path) -> None:
    reasoning = tuple(
        item
        for item in generate_curated_v4(CuratedV4Config(count_per_family=2, seed=31))
        if item.metadata["family"] == "curated_v4_no_tool_reasoning"
    )
    # Duplicate with unique IDs so the stable sampler has enough rows for this tiny fixture.
    rows = []
    for index in range(8):
        base = reasoning[index % len(reasoning)]
        raw = trajectory_to_dict(base)
        raw["example_id"] = f"reasoning-{index}"
        raw["metadata"] = {**raw["metadata"], "semantic_task_id": f"reasoning-{index}"}
        rows.append(raw)
    source = tmp_path / "reasoning.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    output = tmp_path / "validation.jsonl"
    manifest_path = tmp_path / "validation.manifest.json"
    manifest = build_contract_v4_validation(
        source,
        output,
        manifest_path,
        total_examples=10,
        tool_examples=2,
        curated_examples=2,
        seed=37,
    )

    assert manifest["examples"] == 10
    assert manifest["tool_examples"] == 2
    assert manifest["curated_examples"] == 2
    assert manifest["reasoning_examples"] == 6
    assert manifest["neural_contract_version"] == 4
    assert manifest["audit"]["ok"]

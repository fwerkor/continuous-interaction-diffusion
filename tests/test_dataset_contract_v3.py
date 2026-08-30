import json
from pathlib import Path

from cid.data import dump_jsonl, trajectory_to_dict
from cid.dataset_contract_v3 import annotate_trajectory_contract_v3, migrate_dataset_contract_v3
from cid.synthetic import SyntheticConfig, generate_synthetic
from cid.validation_dataset import build_contract_v3_validation


def test_contract_v3_annotation_adds_owner_multi_cell_and_fact_policy() -> None:
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=1, seed=17, thought_capacity=8)
    )
    by_family = {example.metadata["family"]: example for example in examples}

    static_raw = trajectory_to_dict(by_family["static_copy"])
    annotated, stats = annotate_trajectory_contract_v3(static_raw)

    binding = annotated["binding_targets"][0]
    assert binding["owner_cell_id"] == "c1"
    assert [item["identifier"] for item in binding["target_cells"]] == ["c1", "c0"]
    assert binding["target_display"] == []
    assert annotated["source_descriptors"][0]["promote_results_to_fact"]
    assert stats["multi_cell_bindings"] == 1
    assert stats["display_routed_bindings"] == 0


def test_contract_v3_migration_is_streaming_and_preserves_example_count(tmp_path: Path) -> None:
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=1, seed=23, thought_capacity=8)
    )
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    dump_jsonl(examples, source)

    result = migrate_dataset_contract_v3(source, output, manifest)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["examples"] == len(examples)
    assert result["bindings"] == 6
    assert result["owner_bindings"] == 6
    assert result["multi_cell_bindings"] == 6
    assert result["display_routed_bindings"] == 0
    assert result["global_display_fallback_bindings"] == 6
    assert result["neural_contract_version"] == 3
    assert len(rows) == len(examples)
    assert all(
        binding["owner_cell_id"]
        for row in rows
        for binding in row.get("binding_targets", ())
    )


def test_validation_builder_includes_small_tool_slice_and_is_deterministic(tmp_path: Path) -> None:
    reasoning = tmp_path / "reasoning.jsonl"
    with reasoning.open("w", encoding="utf-8") as handle:
        for index in range(30):
            raw = {
                "example_id": f"reason-{index:03d}",
                "prompt": f"Reason about case {index}.",
                "target_display": str(index),
                "protected_facts": {},
                "source_descriptors": [],
                "events": [],
                "binding_targets": [],
                "grounding_catalog": [],
                "grounding_targets": [],
                "thought_targets": [
                    {
                        "step": 0,
                        "slot": 0,
                        "cell_id": "c0",
                        "semantic_text": "reason",
                        "roles": {"plan": 1.0},
                        "uncertainty": 0.2,
                        "noise": 0.2,
                        "lifecycle": "active",
                    }
                ],
                "display_targets": [{"step": 0, "text": str(index)}],
                "metadata": {"family": "reasoning"},
            }
            handle.write(json.dumps(raw) + "\n")

    first = tmp_path / "validation-a.jsonl"
    second = tmp_path / "validation-b.jsonl"
    first_manifest = tmp_path / "validation-a.json"
    second_manifest = tmp_path / "validation-b.json"
    a = build_contract_v3_validation(
        reasoning, first, first_manifest, total_examples=20, tool_examples=5, seed=101
    )
    b = build_contract_v3_validation(
        reasoning, second, second_manifest, total_examples=20, tool_examples=5, seed=101
    )

    assert a["examples"] == 20
    assert a["tool_examples"] == 5
    assert a["tool_fraction"] == 0.25
    assert a["bindings"] > 0
    assert a["multi_cell_bindings"] > 0
    assert first.read_bytes() == second.read_bytes()
    assert a["sha256"] == b["sha256"]

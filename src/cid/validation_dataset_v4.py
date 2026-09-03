from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from cid.curated_v4_training import CuratedV4Config, generate_curated_v4
from cid.data import trajectory_to_dict
from cid.dataset_contract_v4 import audit_dataset_contract_v4, rematerialize_trajectory_contract_v4
from cid.synthetic import SyntheticConfig, generate_synthetic


def build_contract_v4_validation(
    reasoning_source: str | Path,
    output_path: str | Path,
    manifest_output: str | Path,
    *,
    total_examples: int = 512,
    tool_examples: int = 96,
    curated_examples: int = 48,
    seed: int = 20260903,
) -> dict[str, Any]:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if tool_examples <= 0 or curated_examples <= 0:
        raise ValueError("tool_examples and curated_examples must be positive")
    if tool_examples + curated_examples >= total_examples:
        raise ValueError("tool plus curated examples must leave room for OOD reasoning")

    reasoning_count = total_examples - tool_examples - curated_examples
    reasoning_rows = _stable_sample_jsonl(reasoning_source, reasoning_count, seed=seed)
    reasoning: list[dict[str, Any]] = []
    for raw in reasoning_rows:
        metadata = dict(raw.get("metadata", {}))
        metadata["split"] = "validation"
        metadata["validation_kind"] = "ood_reasoning"
        raw["metadata"] = metadata
        migrated, _ = rematerialize_trajectory_contract_v4(raw)
        reasoning.append(migrated)

    count_per_family = (tool_examples + 4) // 5
    generated = generate_synthetic(
        SyntheticConfig(
            count_per_family=count_per_family,
            seed=seed ^ 0x51D4,
            thought_capacity=8,
            index_offset=200_000,
            id_prefix="validation-v4",
            split="validation",
        )
    )
    tools = sorted(
        generated,
        key=lambda example: _stable_key(example.example_id, seed ^ 0xA5A5A5A5),
    )[:tool_examples]
    tool_rows: list[dict[str, Any]] = []
    for example in tools:
        metadata = {**example.metadata, "validation_kind": "tool_required"}
        raw = trajectory_to_dict(replace(example, metadata=metadata))
        migrated, _ = rematerialize_trajectory_contract_v4(raw)
        tool_rows.append(migrated)

    curated_count_per_family = max(1, (curated_examples + 8) // 9)
    curated_pool = generate_curated_v4(
        CuratedV4Config(
            count_per_family=curated_count_per_family,
            seed=seed ^ 0xC1D4,
            training_weight=1.0,
        )
    )
    curated_selected = sorted(
        curated_pool,
        key=lambda example: _stable_key(example.example_id, seed ^ 0xC0FFEE),
    )[:curated_examples]
    curated_rows: list[dict[str, Any]] = []
    for ordinal, example in enumerate(curated_selected):
        raw = trajectory_to_dict(example)
        raw["example_id"] = f"validation-v4-curated-{ordinal:04d}-{example.example_id}"
        raw["metadata"] = {
            **raw.get("metadata", {}),
            "semantic_task_id": f"validation-v4-curated-{ordinal:04d}",
            "split": "validation",
            "validation_kind": "curated_contract",
            "training_weight": 1.0,
        }
        migrated, _ = rematerialize_trajectory_contract_v4(raw)
        curated_rows.append(migrated)

    combined = _interleave_three(reasoning, tool_rows, curated_rows)
    output = Path(output_path)
    manifest_path = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    family_counts: Counter[str] = Counter()
    with output.open("wb") as handle:
        for raw in combined:
            encoded = (json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            handle.write(encoded)
            digest.update(encoded)
            metadata = raw.get("metadata", {})
            family_counts[str(metadata.get("family", metadata.get("task_kind", "unknown")))] += 1

    audit = audit_dataset_contract_v4(output)
    if not audit["ok"]:
        raise ValueError(f"v4 validation audit failed: {audit['violations']}")
    manifest = {
        "format_version": 1,
        "name": "cid-validation-v4-512",
        "neural_contract_version": 4,
        "seed": seed,
        "examples": len(combined),
        "reasoning_examples": len(reasoning),
        "tool_examples": len(tool_rows),
        "curated_examples": len(curated_rows),
        "tool_fraction": len(tool_rows) / len(combined),
        "curated_fraction": len(curated_rows) / len(combined),
        "family_counts": dict(sorted(family_counts.items())),
        "sha256": digest.hexdigest(),
        "bytes": output.stat().st_size,
        "reasoning_source": str(reasoning_source),
        "selection": (
            "stable-hash OOD reasoning sample plus held-out v4 synthetic tool interactions "
            "and independent-seed curated contract probes"
        ),
        "curated_generation_seed": seed ^ 0xC1D4,
        "display_contract": "continuous-answer-draft-v2",
        "audit": audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _stable_sample_jsonl(path: str | Path, count: int, *, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    heap: list[tuple[int, int, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            example_id = str(raw.get("example_id", f"row-{ordinal}"))
            score = _stable_key(example_id, seed)
            item = (-score, -ordinal, line)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if len(heap) < count:
        raise ValueError(f"reasoning source contains only {len(heap)} examples; need {count}")
    selected = sorted(heap, key=lambda item: (-item[0], -item[1]))
    return [json.loads(item[2]) for item in selected]


def _stable_key(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _interleave_three(
    reasoning: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    curated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    tool_index = 0
    curated_index = 0
    for index, item in enumerate(reasoning):
        output.append(item)
        expected_tools = round((index + 1) * len(tools) / max(1, len(reasoning)))
        while tool_index < expected_tools:
            output.append(tools[tool_index])
            tool_index += 1
        expected_curated = round((index + 1) * len(curated) / max(1, len(reasoning)))
        while curated_index < expected_curated:
            output.append(curated[curated_index])
            curated_index += 1
    output.extend(tools[tool_index:])
    output.extend(curated[curated_index:])
    return output

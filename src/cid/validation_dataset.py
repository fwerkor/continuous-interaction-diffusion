from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from cid.data import trajectory_to_dict
from cid.dataset_contract_v3 import annotate_trajectory_contract_v3
from cid.synthetic import SyntheticConfig, generate_synthetic


def build_contract_v3_validation(
    reasoning_source: str | Path,
    output_path: str | Path,
    manifest_output: str | Path,
    *,
    total_examples: int = 512,
    tool_examples: int = 96,
    seed: int = 20260829,
) -> dict[str, Any]:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if tool_examples <= 0 or tool_examples >= total_examples:
        raise ValueError("tool_examples must be between zero and total_examples")

    reasoning_examples = total_examples - tool_examples
    reasoning_rows = _stable_sample_jsonl(reasoning_source, reasoning_examples, seed=seed)
    reasoning: list[dict[str, Any]] = []
    for raw in reasoning_rows:
        metadata = dict(raw.get("metadata", {}))
        metadata["split"] = "validation"
        metadata["validation_kind"] = "ood_reasoning"
        raw["metadata"] = metadata
        migrated, _ = annotate_trajectory_contract_v3(raw)
        reasoning.append(migrated)

    count_per_family = (tool_examples + 4) // 5
    generated = generate_synthetic(
        SyntheticConfig(
            count_per_family=count_per_family,
            seed=seed,
            thought_capacity=8,
            index_offset=100_000,
            id_prefix="validation-v3",
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
        migrated, _ = annotate_trajectory_contract_v3(raw)
        tool_rows.append(migrated)

    combined = _interleave(reasoning, tool_rows)
    output = Path(output_path)
    manifest_path = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    family_counts: Counter[str] = Counter()
    binding_count = 0
    multi_cell_bindings = 0
    display_routed_bindings = 0
    with output.open("wb") as handle:
        for raw in combined:
            encoded = (
                json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            metadata = raw.get("metadata", {})
            family_counts[str(metadata.get("family", metadata.get("task_kind", "unknown")))] += 1
            for binding in raw.get("binding_targets", ()):
                binding_count += 1
                multi_cell_bindings += int(len(binding.get("target_cells", ())) > 1)
                display_routed_bindings += int(bool(binding.get("target_display")))

    manifest = {
        "format_version": 1,
        "name": "cid-validation-v3-512",
        "neural_contract_version": 3,
        "seed": seed,
        "examples": len(combined),
        "reasoning_examples": len(reasoning),
        "tool_examples": len(tool_rows),
        "tool_fraction": len(tool_rows) / len(combined),
        "bindings": binding_count,
        "multi_cell_bindings": multi_cell_bindings,
        "display_routed_bindings": display_routed_bindings,
        "family_counts": dict(sorted(family_counts.items())),
        "sha256": digest.hexdigest(),
        "bytes": output.stat().st_size,
        "reasoning_source": str(reasoning_source),
        "selection": "stable-hash OOD sample plus disjoint synthetic tool validation",
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
    with Path(path).open("r", encoding="utf-8") as handle:
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
    selected = [(-neg_score, -neg_ordinal, line) for neg_score, neg_ordinal, line in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return [json.loads(line) for _, _, line in selected]


def _stable_key(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _interleave(
    reasoning: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not tools:
        return list(reasoning)
    output: list[dict[str, Any]] = []
    reasoning_index = 0
    tool_index = 0
    total = len(reasoning) + len(tools)
    for position in range(total):
        expected_tools = round((position + 1) * len(tools) / total)
        if tool_index < expected_tools and tool_index < len(tools):
            output.append(tools[tool_index])
            tool_index += 1
        elif reasoning_index < len(reasoning):
            output.append(reasoning[reasoning_index])
            reasoning_index += 1
        else:
            output.append(tools[tool_index])
            tool_index += 1
    return output

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def migrate_dataset_contract_v3(
    input_path: str | Path,
    output_path: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Migrate a CID trajectory JSONL to the neural-contract-v3 data ABI.

    The migration is semantic-preserving: prompts, answers, events, thought/display targets,
    schedules, and example counts are unchanged.  It makes the need owner explicit, derives
    conservative affected-region labels from state changes around the matching observation, and
    makes source-owned protected-result promotion explicit.
    """

    source = Path(input_path)
    output = Path(output_path)
    manifest_path = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    input_digest = hashlib.sha256()
    digest = hashlib.sha256()
    examples = 0
    adjacent_transitions = 0
    bindings = 0
    owner_bindings = 0
    multi_cell_bindings = 0
    display_routed_bindings = 0
    promoted_sources = 0
    tool_examples = 0
    max_target_cells = 0

    with source.open("r", encoding="utf-8") as src, output.open("wb") as dst:
        for line in src:
            input_digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            raw = json.loads(line)
            migrated, stats = annotate_trajectory_contract_v3(raw)
            encoded = (
                json.dumps(migrated, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            dst.write(encoded)
            digest.update(encoded)

            examples += 1
            bindings += stats["bindings"]
            owner_bindings += stats["owner_bindings"]
            multi_cell_bindings += stats["multi_cell_bindings"]
            display_routed_bindings += stats["display_routed_bindings"]
            promoted_sources += stats["promoted_sources"]
            tool_examples += int(stats["bindings"] > 0)
            max_target_cells = max(max_target_cells, stats["max_target_cells"])
            thought_steps = {int(item["step"]) for item in migrated.get("thought_targets", ())}
            adjacent_transitions += max(0, len(thought_steps) - 1)

    manifest = {
        "format_version": 1,
        "name": "cid-neural-contract-v3-migration",
        "input": str(source),
        "input_sha256": input_digest.hexdigest(),
        "output": str(output),
        "sha256": digest.hexdigest(),
        "bytes": output.stat().st_size,
        "examples": examples,
        "adjacent_transitions": adjacent_transitions,
        "training_transitions": adjacent_transitions + examples,
        "bindings": bindings,
        "owner_bindings": owner_bindings,
        "multi_cell_bindings": multi_cell_bindings,
        "display_routed_bindings": display_routed_bindings,
        "promoted_sources": promoted_sources,
        "tool_examples": tool_examples,
        "max_target_cells": max_target_cells,
        "neural_contract_version": 3,
        "routing_annotation": "observation-aligned-live-state-delta-v1",
        "display_annotation": "first-display-position-on-observation-visible-change-v1",
        "fact_policy": (
            "source-owned; explicit values preserved; absent values default false"
        ),
        "semantic_preservation": (
            "prompts, target displays, events, thought/display targets, schedules, and example "
            "multiplicity are unchanged"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def annotate_trajectory_contract_v3(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    output = dict(raw)
    metadata = dict(output.get("metadata", {}))
    bindings = [dict(item) for item in output.get("binding_targets", ())]
    sources = [dict(item) for item in output.get("source_descriptors", ())]
    events = tuple(output.get("events", ()))
    thought_targets = tuple(output.get("thought_targets", ()))
    display_targets = tuple(output.get("display_targets", ()))

    timelines: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in thought_targets:
        timelines[str(item["cell_id"])].append(item)
    for values in timelines.values():
        values.sort(key=lambda item: int(item["step"]))

    display_timeline = sorted(display_targets, key=lambda item: int(item["step"]))

    owner_bindings = 0
    multi_cell_bindings = 0
    display_routed_bindings = 0
    max_target_cells = 0
    for binding in bindings:
        targets = [dict(item) for item in binding.get("target_cells", ())]
        owner_id = binding.get("owner_cell_id")
        if owner_id is None and targets:
            owner_id = str(targets[0]["identifier"])
        if owner_id is None:
            continue
        owner_id = str(owner_id)
        binding["owner_cell_id"] = owner_id
        owner_bindings += 1

        first_need_step = int(binding.get("first_need_step", 0))
        arrival_step = _matching_arrival_step(binding, events, first_need_step)
        if arrival_step is None:
            arrival_step = first_need_step

        affected_ids: list[str] = [owner_id]
        for target in targets:
            identifier = str(target.get("identifier", ""))
            if identifier and identifier not in affected_ids:
                affected_ids.append(identifier)

        for cell_id, timeline in timelines.items():
            if cell_id in affected_ids:
                continue
            before = _state_at_or_before(timeline, first_need_step)
            after = _state_at_or_before(timeline, arrival_step)
            if before is None or after is None:
                continue
            # Only route to regions already present when the need is emitted.  This matches what
            # the runtime can materialize at that step and avoids labels for future cells.
            if int(before["step"]) != first_need_step:
                continue
            if _state_signature(before) != _state_signature(after):
                affected_ids.append(cell_id)

        binding["target_cells"] = [
            {"kind": "cell", "identifier": cell_id} for cell_id in affected_ids
        ]
        max_target_cells = max(max_target_cells, len(affected_ids))
        multi_cell_bindings += int(len(affected_ids) > 1)

        existing_display = [dict(item) for item in binding.get("target_display", ())]
        if not existing_display:
            before_display = _display_at_or_before(display_timeline, first_need_step)
            after_display = _display_at_or_before(display_timeline, arrival_step)
            if (
                before_display is not None
                and after_display is not None
                and before_display != after_display
            ):
                # A common dataset cannot know each backbone tokenizer's full answer span.  The
                # first visible position is tokenizer-independent and provides a conservative
                # positive route without pretending to know the remaining token boundaries.
                existing_display = [
                    {"kind": "display_span", "identifier": "display", "span": [0, 1]}
                ]
        binding["target_display"] = existing_display
        display_routed_bindings += int(bool(existing_display))

    promoted_sources = 0
    static_copy = str(metadata.get("family", "")) == "static_copy"
    for source in sources:
        if "promote_results_to_fact" not in source:
            source["promote_results_to_fact"] = bool(static_copy)
        promoted_sources += int(bool(source.get("promote_results_to_fact")))

    output["binding_targets"] = bindings
    output["source_descriptors"] = sources
    output["metadata"] = metadata
    return output, {
        "bindings": len(bindings),
        "owner_bindings": owner_bindings,
        "multi_cell_bindings": multi_cell_bindings,
        "display_routed_bindings": display_routed_bindings,
        "promoted_sources": promoted_sources,
        "max_target_cells": max_target_cells,
    }


def _matching_arrival_step(
    binding: Mapping[str, Any],
    events: tuple[Mapping[str, Any], ...],
    first_need_step: int,
) -> int | None:
    source = str(binding.get("source", ""))
    arguments = _canonical_json(binding.get("arguments", {}))
    exact = [
        int(event["arrival_step"])
        for event in events
        if str(event.get("source", "")) == source
        and int(event.get("arrival_step", -1)) >= first_need_step
        and _canonical_json(event.get("arguments", {})) == arguments
    ]
    if exact:
        return min(exact)
    same_source = [
        int(event["arrival_step"])
        for event in events
        if str(event.get("source", "")) == source
        and int(event.get("arrival_step", -1)) >= first_need_step
    ]
    return min(same_source) if same_source else None


def _state_at_or_before(
    timeline: list[Mapping[str, Any]], step: int
) -> Mapping[str, Any] | None:
    candidate = None
    for item in timeline:
        if int(item["step"]) > step:
            break
        candidate = item
    return candidate


def _display_at_or_before(
    timeline: list[Mapping[str, Any]], step: int
) -> str | None:
    candidate = None
    for item in timeline:
        if int(item["step"]) > step:
            break
        candidate = str(item.get("text", ""))
    return candidate


def _state_signature(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("semantic_text"),
        _canonical_json(item.get("roles", {})),
        item.get("uncertainty"),
        item.get("noise"),
        item.get("lifecycle"),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))



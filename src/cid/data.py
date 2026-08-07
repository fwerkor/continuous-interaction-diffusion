from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cid.grounding import Anchor, CognitiveLink, GroundingEntry, ObjectKind, ObjectRef


@dataclass(frozen=True, slots=True)
class ExternalEvent:
    source: str
    value: Any
    arrival_step: int
    version: str | None = None
    provenance: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("event source must be non-empty")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")


@dataclass(frozen=True, slots=True)
class BindingTarget:
    need_id: str
    source: str
    first_need_step: int
    executable_step: int | None
    target_cells: tuple[ObjectRef, ...] = ()
    target_display: tuple[ObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if self.first_need_step < 0:
            raise ValueError("first_need_step must be non-negative")
        if self.executable_step is not None and self.executable_step < self.first_need_step:
            raise ValueError("executable_step cannot precede first_need_step")
        if any(target.kind is not ObjectKind.CELL for target in self.target_cells):
            raise ValueError("binding target_cells must contain cell references")
        if any(target.kind is not ObjectKind.DISPLAY_SPAN for target in self.target_display):
            raise ValueError("binding target_display must contain display-span references")


@dataclass(frozen=True, slots=True)
class GroundingTarget:
    step: int
    cell_id: str
    anchors: tuple[Anchor, ...] = ()
    links: tuple[CognitiveLink, ...] = ()

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("grounding target step must be non-negative")
        if not self.cell_id:
            raise ValueError("grounding target cell_id must be non-empty")
        anchor_ids = tuple(anchor.anchor_id for anchor in self.anchors)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("grounding target anchor IDs must be unique")


@dataclass(frozen=True, slots=True)
class TrajectoryExample:
    example_id: str
    prompt: str
    target_display: str
    protected_facts: Mapping[str, Any] = field(default_factory=dict)
    source_descriptors: tuple[Mapping[str, Any], ...] = ()
    events: tuple[ExternalEvent, ...] = ()
    binding_targets: tuple[BindingTarget, ...] = ()
    grounding_catalog: tuple[GroundingEntry, ...] = ()
    grounding_targets: tuple[GroundingTarget, ...] = ()
    thought_targets: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrajectoryExample:
        events = tuple(ExternalEvent(**event) for event in raw.get("events", ()))
        bindings = tuple(
            BindingTarget(
                need_id=str(target["need_id"]),
                source=str(target["source"]),
                first_need_step=int(target["first_need_step"]),
                executable_step=(
                    None
                    if target.get("executable_step") is None
                    else int(target["executable_step"])
                ),
                target_cells=tuple(
                    ObjectRef.from_dict(item) for item in target.get("target_cells", ())
                ),
                target_display=tuple(
                    ObjectRef.from_dict(item) for item in target.get("target_display", ())
                ),
            )
            for target in raw.get("binding_targets", ())
        )
        grounding_catalog = tuple(
            GroundingEntry.from_dict(entry) for entry in raw.get("grounding_catalog", ())
        )
        grounding_targets = tuple(
            GroundingTarget(
                step=int(target["step"]),
                cell_id=str(target["cell_id"]),
                anchors=tuple(Anchor.from_dict(item) for item in target.get("anchors", ())),
                links=tuple(CognitiveLink.from_dict(item) for item in target.get("links", ())),
            )
            for target in raw.get("grounding_targets", ())
        )
        return cls(
            example_id=str(raw["example_id"]),
            prompt=str(raw["prompt"]),
            target_display=str(raw["target_display"]),
            protected_facts=dict(raw.get("protected_facts", {})),
            source_descriptors=tuple(dict(item) for item in raw.get("source_descriptors", ())),
            events=events,
            binding_targets=bindings,
            grounding_catalog=grounding_catalog,
            grounding_targets=grounding_targets,
            thought_targets=tuple(dict(item) for item in raw.get("thought_targets", ())),
            metadata=dict(raw.get("metadata", {})),
        )


def load_jsonl(path: str | Path) -> tuple[TrajectoryExample, ...]:
    examples: list[TrajectoryExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                examples.append(TrajectoryExample.from_dict(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CID trajectory at line {line_number}: {exc}") from exc
    return tuple(examples)


def dump_jsonl(examples: Iterable[TrajectoryExample], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for example in examples:
            raw = {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "target_display": example.target_display,
                "protected_facts": dict(example.protected_facts),
                "source_descriptors": [dict(item) for item in example.source_descriptors],
                "events": [
                    {
                        "source": event.source,
                        "value": event.value,
                        "arrival_step": event.arrival_step,
                        "version": event.version,
                        "provenance": event.provenance,
                        "arguments": dict(event.arguments),
                    }
                    for event in example.events
                ],
                "binding_targets": [
                    {
                        "need_id": target.need_id,
                        "source": target.source,
                        "first_need_step": target.first_need_step,
                        "executable_step": target.executable_step,
                        "target_cells": [item.to_dict() for item in target.target_cells],
                        "target_display": [item.to_dict() for item in target.target_display],
                    }
                    for target in example.binding_targets
                ],
                "grounding_catalog": [entry.to_dict() for entry in example.grounding_catalog],
                "grounding_targets": [
                    {
                        "step": target.step,
                        "cell_id": target.cell_id,
                        "anchors": [anchor.to_dict() for anchor in target.anchors],
                        "links": [link.to_dict() for link in target.links],
                    }
                    for target in example.grounding_targets
                ],
                "thought_targets": [dict(item) for item in example.thought_targets],
                "metadata": dict(example.metadata),
            }
            handle.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")

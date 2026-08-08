from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cid.contracts import FreshnessDemand
from cid.grounding import Anchor, CognitiveLink, GroundingEntry, ObjectKind, ObjectRef
from cid.state import CellLifecycle, CognitiveRole


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
    arguments: Mapping[str, Any] = field(default_factory=dict)
    argument_steps: Mapping[str, int] = field(default_factory=dict)
    confidence: float = 1.0
    freshness: FreshnessDemand = FreshnessDemand.ONCE
    max_age_s: float | None = None
    target_cells: tuple[ObjectRef, ...] = ()
    target_display: tuple[ObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if self.first_need_step < 0:
            raise ValueError("first_need_step must be non-negative")
        if self.executable_step is not None and self.executable_step < self.first_need_step:
            raise ValueError("executable_step cannot precede first_need_step")
        if set(self.argument_steps) - set(self.arguments):
            raise ValueError("argument_steps may only reference declared target arguments")
        if any(step < self.first_need_step for step in self.argument_steps.values()):
            raise ValueError("argument availability cannot precede first_need_step")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("binding target confidence must be in [0, 1]")
        if self.max_age_s is not None and self.max_age_s < 0:
            raise ValueError("binding target max_age_s must be non-negative")
        if self.freshness is FreshnessDemand.MAX_AGE and self.max_age_s is None:
            raise ValueError("MAX_AGE binding targets require max_age_s")
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
class ThoughtTarget:
    """One occupied TCT cell in a supervised trajectory snapshot.

    `semantic_text` is only a dataset transport representation. The training tensorizer maps it to
    a latent target; runtime cognition remains continuous and does not expose textual CoT.
    """

    step: int
    slot: int
    cell_id: str
    semantic_text: str
    roles: Mapping[CognitiveRole, float] = field(default_factory=dict)
    uncertainty: float = 1.0
    noise: float = 1.0
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE

    def __post_init__(self) -> None:
        if self.step < 0 or self.slot < 0:
            raise ValueError("thought target step and slot must be non-negative")
        if not self.cell_id:
            raise ValueError("thought target cell_id must be non-empty")
        if not self.semantic_text:
            raise ValueError("thought target semantic_text must be non-empty")
        if self.lifecycle is CellLifecycle.EMPTY:
            raise ValueError("thought targets describe occupied cells and cannot be EMPTY")
        if any(not 0.0 <= weight <= 1.0 for weight in self.roles.values()):
            raise ValueError("thought target role weights must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("thought target uncertainty must be in [0, 1]")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("thought target noise must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DisplayTarget:
    step: int
    text: str

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("display target step must be non-negative")
        if not self.text:
            raise ValueError("display target text must be non-empty")


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
    thought_targets: tuple[ThoughtTarget, ...] = ()
    display_targets: tuple[DisplayTarget, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        thought_keys = tuple((target.step, target.slot) for target in self.thought_targets)
        if len(thought_keys) != len(set(thought_keys)):
            raise ValueError("thought targets must have unique (step, slot) pairs")
        for step in {target.step for target in self.thought_targets}:
            cell_ids = tuple(
                target.cell_id for target in self.thought_targets if target.step == step
            )
            if len(cell_ids) != len(set(cell_ids)):
                raise ValueError("thought target cell IDs must be unique within a step")
        display_steps = tuple(target.step for target in self.display_targets)
        if len(display_steps) != len(set(display_steps)):
            raise ValueError("display targets must have unique steps")

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
                arguments=dict(target.get("arguments", {})),
                argument_steps={
                    str(name): int(step) for name, step in target.get("argument_steps", {}).items()
                },
                confidence=float(target.get("confidence", 1.0)),
                freshness=FreshnessDemand(str(target.get("freshness", FreshnessDemand.ONCE))),
                max_age_s=(
                    None if target.get("max_age_s") is None else float(target["max_age_s"])
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
        thought_targets = tuple(
            ThoughtTarget(
                step=int(target["step"]),
                slot=int(target["slot"]),
                cell_id=str(target["cell_id"]),
                semantic_text=str(target["semantic_text"]),
                roles={
                    CognitiveRole(str(role)): float(weight)
                    for role, weight in target.get("roles", {}).items()
                },
                uncertainty=float(target.get("uncertainty", 1.0)),
                noise=float(target.get("noise", 1.0)),
                lifecycle=CellLifecycle(str(target.get("lifecycle", CellLifecycle.ACTIVE))),
            )
            for target in raw.get("thought_targets", ())
        )
        display_targets = tuple(
            DisplayTarget(step=int(target["step"]), text=str(target["text"]))
            for target in raw.get("display_targets", ())
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
            thought_targets=thought_targets,
            display_targets=display_targets,
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
                        "arguments": dict(target.arguments),
                        "argument_steps": dict(target.argument_steps),
                        "confidence": target.confidence,
                        "freshness": target.freshness.value,
                        "max_age_s": target.max_age_s,
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
                "thought_targets": [
                    {
                        "step": target.step,
                        "slot": target.slot,
                        "cell_id": target.cell_id,
                        "semantic_text": target.semantic_text,
                        "roles": {role.value: weight for role, weight in target.roles.items()},
                        "uncertainty": target.uncertainty,
                        "noise": target.noise,
                        "lifecycle": target.lifecycle.value,
                    }
                    for target in example.thought_targets
                ],
                "display_targets": [
                    {"step": target.step, "text": target.text}
                    for target in example.display_targets
                ],
                "metadata": dict(example.metadata),
            }
            handle.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")

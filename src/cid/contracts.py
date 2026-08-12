from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from cid.grounding import Anchor, ObjectKind, ObjectRef
from cid.state import CognitiveField, DisplayCanvas, FactSnapshot


class FreshnessDemand(StrEnum):
    ONCE = "once"
    MAX_AGE = "max_age"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class ArgumentDescriptor:
    name: str
    kind: str = "any"
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("argument name must be non-empty")
        if not self.kind:
            raise ValueError("argument kind must be non-empty")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    name: str
    description: str
    arguments: tuple[ArgumentDescriptor, ...] = ()
    cacheable: bool = True
    dynamic: bool = False
    streamable: bool = False
    versioned: bool = False
    accepts_partial_arguments: bool = False

    @property
    def required_arguments(self) -> tuple[str, ...]:
        return tuple(argument.name for argument in self.arguments if argument.required)


@dataclass(frozen=True, slots=True)
class InformationNeed:
    need_id: str
    source_scores: Mapping[str, float]
    arguments: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    freshness: FreshnessDemand = FreshnessDemand.ONCE
    max_age_s: float | None = None
    target_cells: tuple[ObjectRef, ...] = ()
    target_display: tuple[ObjectRef, ...] = ()
    promote_to_fact: bool = False

    def __post_init__(self) -> None:
        if not self.need_id:
            raise ValueError("need_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("need confidence must be in [0, 1]")
        if any(not 0.0 <= score <= 1.0 for score in self.source_scores.values()):
            raise ValueError("source scores must be in [0, 1]")
        if self.max_age_s is not None and self.max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")
        if any(target.kind is not ObjectKind.CELL for target in self.target_cells):
            raise ValueError("information-need target_cells must contain cell references")
        if any(target.kind is not ObjectKind.DISPLAY_SPAN for target in self.target_display):
            raise ValueError("information-need target_display must contain display-span references")
        if len(set(self.target_cells)) != len(self.target_cells):
            raise ValueError("target cell references must be unique")

    @property
    def target_cell_ids(self) -> tuple[str, ...]:
        return tuple(target.identifier for target in self.target_cells)

    def selected_source(self) -> str | None:
        if not self.source_scores:
            return None
        return max(self.source_scores, key=self.source_scores.__getitem__)


@dataclass(frozen=True, slots=True)
class Observation:
    value: Any
    version: str | None = None
    provenance: str | None = None
    observed_at: float | None = None
    anchors: tuple[Anchor, ...] = ()


@dataclass(frozen=True, slots=True)
class Percept:
    binding_id: str
    source: str
    observation: Observation
    target_cells: tuple[ObjectRef, ...]
    target_display: tuple[ObjectRef, ...]
    projection_index: int


@dataclass(frozen=True, slots=True)
class ModelContext:
    facts: FactSnapshot
    thought: CognitiveField
    display: DisplayCanvas
    sources: tuple[SourceDescriptor, ...]
    percepts: tuple[Percept, ...]
    step: int
    prompt: str = ""


@dataclass(frozen=True, slots=True)
class ModelUpdate:
    thought: CognitiveField
    display: DisplayCanvas
    needs: tuple[InformationNeed, ...] = ()
    reopen_cells: tuple[ObjectRef, ...] = ()
    equilibrium: bool = False
    converged: bool = False

    def __post_init__(self) -> None:
        if any(target.kind is not ObjectKind.CELL for target in self.reopen_cells):
            raise ValueError("reopen_cells must contain cell references")
        if len(set(self.reopen_cells)) != len(self.reopen_cells):
            raise ValueError("reopen cell references must be unique")

    @property
    def reopen_cell_ids(self) -> tuple[str, ...]:
        return tuple(target.identifier for target in self.reopen_cells)


class CIDPolicy(Protocol):
    def step(self, context: ModelContext) -> ModelUpdate:
        ...

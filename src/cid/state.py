from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class CognitiveRole(StrEnum):
    HYPOTHESIS = "hypothesis"
    INFORMATION_NEED = "information_need"
    PERCEPT = "percept"
    PLAN = "plan"
    CONSTRAINT = "constraint"
    CONCLUSION = "conclusion"


class CellLifecycle(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    STABLE = "stable"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class Anchor:
    kind: str
    value: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("anchor confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FactItem:
    key: str
    value: Any
    source_type: str
    timestamp: float
    version: str | None = None
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class FactSnapshot:
    revision: int
    items: Mapping[str, FactItem]

    def get(self, key: str) -> FactItem | None:
        return self.items.get(key)


class FactStore:
    """Runtime-owned mutable store that exposes only immutable snapshots to model code."""

    def __init__(self) -> None:
        self._items: dict[str, FactItem] = {}
        self._revision = 0

    def publish(self, item: FactItem) -> None:
        self._items[item.key] = deepcopy(item)
        self._revision += 1

    def remove(self, key: str) -> None:
        if key in self._items:
            del self._items[key]
            self._revision += 1

    def snapshot(self) -> FactSnapshot:
        return FactSnapshot(
            revision=self._revision,
            items=MappingProxyType(deepcopy(self._items)),
        )


@dataclass(frozen=True, slots=True)
class CognitiveCell:
    semantic: tuple[float, ...]
    roles: Mapping[CognitiveRole, float] = field(default_factory=dict)
    anchors: tuple[Anchor, ...] = ()
    links: tuple[str, ...] = ()
    uncertainty: float = 1.0
    noise: float = 1.0
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("cell uncertainty must be in [0, 1]")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("cell noise must be in [0, 1]")
        if any(weight < 0.0 for weight in self.roles.values()):
            raise ValueError("role weights must be non-negative")

    def reopened(self, amount: float) -> CognitiveCell:
        return replace(self, noise=min(1.0, self.noise + max(0.0, amount)))

    def stabilized(self, amount: float) -> CognitiveCell:
        return replace(self, noise=max(0.0, self.noise - max(0.0, amount)))


@dataclass(frozen=True, slots=True)
class CognitiveField:
    cells: tuple[CognitiveCell, ...]
    step: int = 0

    @classmethod
    def empty(cls, count: int, width: int) -> CognitiveField:
        if count <= 0 or width <= 0:
            raise ValueError("count and width must be positive")
        zero = tuple(0.0 for _ in range(width))
        return cls(cells=tuple(CognitiveCell(semantic=zero) for _ in range(count)))

    def advance(self, cells: tuple[CognitiveCell, ...]) -> CognitiveField:
        if len(cells) != len(self.cells):
            raise ValueError("CID v0 uses a fixed cognitive-cell count")
        return CognitiveField(cells=cells, step=self.step + 1)


@dataclass(frozen=True, slots=True)
class DisplayCanvas:
    token_ids: tuple[int, ...]
    mask_token_id: int
    step: int = 0

    @classmethod
    def masked(cls, length: int, mask_token_id: int) -> DisplayCanvas:
        if length <= 0:
            raise ValueError("display length must be positive")
        return cls(token_ids=(mask_token_id,) * length, mask_token_id=mask_token_id)

    @property
    def unresolved(self) -> int:
        return sum(token == self.mask_token_id for token in self.token_ids)

    def advance(self, token_ids: tuple[int, ...]) -> DisplayCanvas:
        if len(token_ids) != len(self.token_ids):
            raise ValueError("display length cannot change inside a CID v0 trajectory")
        return DisplayCanvas(
            token_ids=token_ids,
            mask_token_id=self.mask_token_id,
            step=self.step + 1,
        )

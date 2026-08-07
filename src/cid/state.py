from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cid.grounding import Anchor, CognitiveLink, LinkRelation, ObjectRef


class CognitiveRole(StrEnum):
    HYPOTHESIS = "hypothesis"
    INFORMATION_NEED = "information_need"
    PERCEPT = "percept"
    PLAN = "plan"
    CONSTRAINT = "constraint"
    CONCLUSION = "conclusion"


class CellLifecycle(StrEnum):
    EMPTY = "empty"
    ACTIVE = "active"
    WAITING = "waiting"
    STABLE = "stable"
    RETIRED = "retired"


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
    cell_id: str | None = None
    roles: Mapping[CognitiveRole, float] = field(default_factory=dict)
    anchors: tuple[Anchor, ...] = ()
    links: tuple[CognitiveLink, ...] = ()
    uncertainty: float = 1.0
    noise: float = 1.0
    lifecycle: CellLifecycle = CellLifecycle.EMPTY

    def __post_init__(self) -> None:
        if not self.semantic:
            raise ValueError("cell semantic vector must be non-empty")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("cell uncertainty must be in [0, 1]")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("cell noise must be in [0, 1]")
        if any(weight < 0.0 for weight in self.roles.values()):
            raise ValueError("role weights must be non-negative")
        if self.lifecycle is CellLifecycle.EMPTY:
            if self.cell_id is not None:
                raise ValueError("empty cells cannot have a stable cell_id")
            if self.roles or self.anchors or self.links:
                raise ValueError("empty cells cannot carry cognitive metadata")
        elif not self.cell_id:
            raise ValueError("occupied cells require a stable cell_id")

    @property
    def occupied(self) -> bool:
        return self.lifecycle is not CellLifecycle.EMPTY

    @property
    def live(self) -> bool:
        return self.lifecycle not in (CellLifecycle.EMPTY, CellLifecycle.RETIRED)

    def reopened(self, amount: float) -> CognitiveCell:
        if not self.occupied:
            raise ValueError("cannot reopen an empty cognitive cell")
        return replace(self, noise=min(1.0, self.noise + max(0.0, amount)))

    def stabilized(self, amount: float) -> CognitiveCell:
        if not self.occupied:
            raise ValueError("cannot stabilize an empty cognitive cell")
        return replace(self, noise=max(0.0, self.noise - max(0.0, amount)))


@dataclass(frozen=True, slots=True)
class CognitiveField:
    """Fixed-capacity tensor layout with dynamically occupied cognitive cells."""

    cells: tuple[CognitiveCell, ...]
    step: int = 0
    next_cell_serial: int = 0

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("cognitive field capacity must be positive")
        width = len(self.cells[0].semantic)
        if any(len(cell.semantic) != width for cell in self.cells):
            raise ValueError("all cognitive cells must have the same semantic width")
        ids = [cell.cell_id for cell in self.cells if cell.cell_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("cognitive cell IDs must be unique")
        if self.next_cell_serial < 0:
            raise ValueError("next_cell_serial must be non-negative")

    @classmethod
    def empty(cls, capacity: int, width: int) -> CognitiveField:
        if capacity <= 0 or width <= 0:
            raise ValueError("capacity and width must be positive")
        zero = tuple(0.0 for _ in range(width))
        return cls(cells=tuple(CognitiveCell(semantic=zero) for _ in range(capacity)))

    @property
    def capacity(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0].semantic)

    @property
    def occupied_count(self) -> int:
        return sum(cell.occupied for cell in self.cells)

    @property
    def live_count(self) -> int:
        return sum(cell.live for cell in self.cells)

    @property
    def empty_count(self) -> int:
        return self.capacity - self.occupied_count

    @property
    def occupied_mask(self) -> tuple[bool, ...]:
        return tuple(cell.occupied for cell in self.cells)

    @property
    def live_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.cells if cell.live and cell.cell_id is not None)

    @property
    def occupied_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            cell.cell_id for cell in self.cells if cell.occupied and cell.cell_id is not None
        )

    def slot_of(self, cell_id: str) -> int:
        for slot, cell in enumerate(self.cells):
            if cell.cell_id == cell_id:
                return slot
        raise KeyError(f"unknown cognitive cell: {cell_id}")

    def get(self, cell_id: str) -> CognitiveCell:
        return self.cells[self.slot_of(cell_id)]

    def allocate(
        self,
        *,
        slot: int | None = None,
        semantic: tuple[float, ...] | None = None,
        roles: Mapping[CognitiveRole, float] | None = None,
        anchors: tuple[Anchor, ...] = (),
        links: tuple[CognitiveLink, ...] = (),
        uncertainty: float = 1.0,
        noise: float = 1.0,
        lifecycle: CellLifecycle = CellLifecycle.ACTIVE,
    ) -> tuple[CognitiveField, str]:
        if lifecycle is not CellLifecycle.ACTIVE:
            raise ValueError("new cognitive cells must start ACTIVE")
        if slot is None:
            try:
                slot = next(index for index, cell in enumerate(self.cells) if not cell.occupied)
            except StopIteration as exc:
                raise RuntimeError("cognitive field has no empty slots") from exc
        elif not 0 <= slot < self.capacity:
            raise IndexError("cognitive slot is outside field capacity")
        elif self.cells[slot].occupied:
            raise ValueError(f"cognitive slot {slot} is already occupied")

        vector = semantic if semantic is not None else (0.0,) * self.width
        if len(vector) != self.width:
            raise ValueError("allocated semantic width does not match cognitive field width")

        serial = self.next_cell_serial
        existing = set(self.occupied_cell_ids)
        while f"c{serial}" in existing:
            serial += 1
        cell_id = f"c{serial}"
        cells = list(self.cells)
        cells[slot] = CognitiveCell(
            semantic=vector,
            cell_id=cell_id,
            roles=roles or {},
            anchors=anchors,
            links=links,
            uncertainty=uncertainty,
            noise=noise,
            lifecycle=lifecycle,
        )
        return (
            CognitiveField(
                cells=tuple(cells),
                step=self.step,
                next_cell_serial=serial + 1,
            ),
            cell_id,
        )

    def retire(self, cell_id: str) -> CognitiveField:
        slot = self.slot_of(cell_id)
        cell = self.cells[slot]
        if cell.lifecycle is CellLifecycle.RETIRED:
            return self
        if cell.lifecycle is CellLifecycle.EMPTY:
            raise ValueError("cannot retire an empty cognitive cell")
        cells = list(self.cells)
        cells[slot] = replace(cell, lifecycle=CellLifecycle.RETIRED)
        return replace(self, cells=tuple(cells))

    def reclaim(self, cell_id: str) -> CognitiveField:
        slot = self.slot_of(cell_id)
        cell = self.cells[slot]
        if cell.lifecycle is not CellLifecycle.RETIRED:
            raise ValueError("only retired cognitive cells can be reclaimed")
        cells = list(self.cells)
        cells[slot] = CognitiveCell(semantic=(0.0,) * self.width)
        return replace(self, cells=tuple(cells))

    def reclaim_retired(self) -> CognitiveField:
        cells = tuple(
            CognitiveCell(semantic=(0.0,) * self.width)
            if cell.lifecycle is CellLifecycle.RETIRED
            else cell
            for cell in self.cells
        )
        return replace(self, cells=cells)

    def compact(self) -> CognitiveField:
        occupied = [cell for cell in self.cells if cell.occupied]
        empty = [CognitiveCell(semantic=(0.0,) * self.width) for _ in range(self.empty_count)]
        return replace(self, cells=tuple((*occupied, *empty)))

    def split(
        self,
        cell_id: str,
        semantics: tuple[tuple[float, ...], ...],
    ) -> tuple[CognitiveField, tuple[str, ...]]:
        if len(semantics) < 2:
            raise ValueError("split requires at least two child cells")
        source = self.get(cell_id)
        if not source.live:
            raise ValueError("only live cognitive cells can be split")
        if self.empty_count < len(semantics):
            raise RuntimeError("not enough empty slots for cognitive split")

        field = self.retire(cell_id)
        children: list[str] = []
        for semantic in semantics:
            field, child_id = field.allocate(
                semantic=semantic,
                links=(
                    CognitiveLink(
                        relation=LinkRelation.DERIVED_FROM,
                        target=ObjectRef.cell(cell_id),
                    ),
                ),
            )
            children.append(child_id)
        return field, tuple(children)

    def merge(
        self,
        cell_ids: tuple[str, ...],
        semantic: tuple[float, ...],
    ) -> tuple[CognitiveField, str]:
        if len(cell_ids) < 2:
            raise ValueError("merge requires at least two source cells")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("merge source cells must be unique")
        if self.empty_count < 1:
            raise RuntimeError("not enough empty slots for cognitive merge")
        if any(not self.get(cell_id).live for cell_id in cell_ids):
            raise ValueError("only live cognitive cells can be merged")

        field = self
        for cell_id in cell_ids:
            field = field.retire(cell_id)
        return field.allocate(
            semantic=semantic,
            links=tuple(
                CognitiveLink(
                    relation=LinkRelation.DERIVED_FROM,
                    target=ObjectRef.cell(cell_id),
                )
                for cell_id in cell_ids
            ),
        )

    def advance(self, cells: tuple[CognitiveCell, ...]) -> CognitiveField:
        if len(cells) != self.capacity:
            raise ValueError("cognitive field capacity cannot change inside a trajectory")
        return CognitiveField(
            cells=cells,
            step=self.step + 1,
            next_cell_serial=self.next_cell_serial,
        )


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

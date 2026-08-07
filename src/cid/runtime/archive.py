from __future__ import annotations

from dataclasses import dataclass

from cid.grounding import Anchor, CognitiveLink, ObjectKind, ObjectRef
from cid.state import CognitiveCell, CognitiveField, CognitiveRole


@dataclass(frozen=True, slots=True)
class CognitiveTombstone:
    cell_id: str
    roles: tuple[tuple[CognitiveRole, float], ...]
    anchors: tuple[Anchor, ...]
    links: tuple[CognitiveLink, ...]
    created_step: int
    retired_step: int
    archived_step: int
    physical_slot: int
    binding_ids: tuple[str, ...] = ()


class CognitiveArchive:
    def __init__(self) -> None:
        self._by_id: dict[str, CognitiveTombstone] = {}

    def record(
        self,
        cell: CognitiveCell,
        *,
        created_step: int,
        retired_step: int,
        archived_step: int,
        physical_slot: int,
        binding_ids: tuple[str, ...] = (),
    ) -> CognitiveTombstone:
        if cell.cell_id is None:
            raise ValueError("cannot archive a cognitive cell without cell_id")
        if cell.cell_id in self._by_id:
            raise ValueError(f"cognitive cell already archived: {cell.cell_id}")
        tombstone = CognitiveTombstone(
            cell_id=cell.cell_id,
            roles=tuple(cell.roles.items()),
            anchors=cell.anchors,
            links=cell.links,
            created_step=created_step,
            retired_step=retired_step,
            archived_step=archived_step,
            physical_slot=physical_slot,
            binding_ids=binding_ids,
        )
        self._by_id[cell.cell_id] = tombstone
        return tombstone

    def get(self, cell_id: str) -> CognitiveTombstone | None:
        return self._by_id.get(cell_id)

    def all(self) -> tuple[CognitiveTombstone, ...]:
        return tuple(self._by_id.values())

    def resolve_cell(
        self, ref: ObjectRef, field: CognitiveField
    ) -> CognitiveCell | CognitiveTombstone | None:
        if ref.kind is not ObjectKind.CELL:
            raise ValueError("cognitive archive resolves only cell references")
        try:
            return field.get(ref.identifier)
        except KeyError:
            return self.get(ref.identifier)

    def __len__(self) -> int:
        return len(self._by_id)

from __future__ import annotations

from dataclasses import dataclass, replace

from cid.state import CellLifecycle, CognitiveField

MODELED_LIFECYCLES: tuple[CellLifecycle, ...] = (
    CellLifecycle.ACTIVE,
    CellLifecycle.WAITING,
    CellLifecycle.STABLE,
    CellLifecycle.RETIRED,
)


@dataclass(frozen=True, slots=True)
class LifecycleTransitionSignals:
    """Runtime facts that gate model-proposed lifecycle transitions."""

    waiting_cells: frozenset[str] = frozenset()
    available_cells: frozenset[str] = frozenset()
    reopen_cells: frozenset[str] = frozenset()


class LifecycleTransitionController:
    """Apply model proposals without letting logits violate runtime state."""

    def apply(
        self,
        previous: CognitiveField,
        proposed: CognitiveField,
        signals: LifecycleTransitionSignals | None = None,
    ) -> CognitiveField:
        signals = signals or LifecycleTransitionSignals()
        if proposed.capacity != previous.capacity or proposed.width != previous.width:
            raise ValueError("lifecycle transition cannot change cognitive field geometry")

        previous_ids = set(previous.occupied_cell_ids)
        proposed_ids = set(proposed.occupied_cell_ids)
        removed = previous_ids - proposed_ids
        if removed:
            removed_text = ", ".join(sorted(removed))
            raise ValueError(
                "model updates cannot reclaim cognitive cells directly; "
                f"runtime reclamation is required for: {removed_text}"
            )

        cells = list(proposed.cells)
        for slot, cell in enumerate(cells):
            cell_id = cell.cell_id
            if cell_id is None:
                continue
            if cell_id not in previous_ids:
                if cell.lifecycle is not CellLifecycle.ACTIVE:
                    cells[slot] = replace(cell, lifecycle=CellLifecycle.ACTIVE)
                continue

            resolved = self.resolve(
                cell_id=cell_id,
                current=previous.get(cell_id).lifecycle,
                proposed=cell.lifecycle,
                signals=signals,
            )
            if resolved is not cell.lifecycle:
                cells[slot] = replace(cell, lifecycle=resolved)

        return CognitiveField(
            cells=tuple(cells),
            step=proposed.step,
            next_cell_serial=proposed.next_cell_serial,
        )

    @staticmethod
    def resolve(
        *,
        cell_id: str,
        current: CellLifecycle,
        proposed: CellLifecycle,
        signals: LifecycleTransitionSignals,
    ) -> CellLifecycle:
        if current is CellLifecycle.EMPTY:
            raise ValueError("EMPTY slots are handled by allocation, not lifecycle transition")
        if proposed is CellLifecycle.EMPTY:
            raise ValueError("model lifecycle proposals cannot reclaim a cognitive cell")
        if current is CellLifecycle.RETIRED:
            return CellLifecycle.RETIRED

        if current is CellLifecycle.WAITING:
            if cell_id in signals.waiting_cells:
                return CellLifecycle.WAITING
            if cell_id in signals.available_cells:
                return CellLifecycle.ACTIVE
            if proposed is CellLifecycle.WAITING:
                return CellLifecycle.ACTIVE
            return proposed

        if current is CellLifecycle.STABLE:
            if cell_id in signals.reopen_cells:
                return CellLifecycle.ACTIVE
            if proposed is CellLifecycle.ACTIVE:
                return CellLifecycle.STABLE
            if proposed is CellLifecycle.WAITING:
                return (
                    CellLifecycle.WAITING
                    if cell_id in signals.waiting_cells
                    else CellLifecycle.STABLE
                )
            return proposed

        if proposed is CellLifecycle.WAITING and cell_id not in signals.waiting_cells:
            return CellLifecycle.ACTIVE
        return proposed

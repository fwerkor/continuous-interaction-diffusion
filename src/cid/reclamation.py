from __future__ import annotations

import math
from collections.abc import Mapping, Set

from cid.state import CellLifecycle, CognitiveField

DEFAULT_RECLAMATION_GRACE_STEPS = 2
DEFAULT_RECLAMATION_LOW_WATERMARK = 0.125
DEFAULT_RECLAMATION_TARGET_WATERMARK = 0.25


def retired_reclamation_candidates(
    field: CognitiveField,
    *,
    retired_at: Mapping[str, int],
    step: int,
    grace_steps: int = DEFAULT_RECLAMATION_GRACE_STEPS,
    low_watermark: float = DEFAULT_RECLAMATION_LOW_WATERMARK,
    target_watermark: float = DEFAULT_RECLAMATION_TARGET_WATERMARK,
    pinned_cell_ids: Set[str] = frozenset(),
    force: bool = False,
) -> tuple[tuple[int, str], ...]:
    """Select RETIRED cells for runtime-controlled reclamation.

    RETIRED cells remain physically occupied until this policy selects them.  The
    caller owns archival/trace side effects and performs reclaim+compaction.
    """

    if grace_steps < 0:
        raise ValueError("reclamation grace_steps must be non-negative")
    if not 0.0 <= low_watermark <= target_watermark <= 1.0:
        raise ValueError("reclamation watermarks must satisfy 0 <= low <= target <= 1")

    low = math.ceil(field.capacity * low_watermark)
    target = math.ceil(field.capacity * target_watermark)
    if not force and field.empty_count >= low:
        return ()

    eligible: list[tuple[int, int, str]] = []
    for slot, cell in enumerate(field.cells):
        if cell.cell_id is None or cell.lifecycle is not CellLifecycle.RETIRED:
            continue
        retired_step = retired_at.get(cell.cell_id, step)
        if step - retired_step < grace_steps or cell.cell_id in pinned_cell_ids:
            continue
        eligible.append((retired_step, slot, cell.cell_id))

    eligible.sort()
    if force:
        selected = eligible
    else:
        needed = max(0, target - field.empty_count)
        selected = eligible[:needed]
    return tuple((slot, cell_id) for _, slot, cell_id in selected)

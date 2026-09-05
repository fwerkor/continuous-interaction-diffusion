from __future__ import annotations

import torch
from torch import Tensor

from cid.defaults import DEFAULT_MAX_ALLOCATIONS_PER_STEP as DEFAULT_MAX_ALLOCATIONS_PER_STEP


def prefix_allocation_mask(
    occupancy: Tensor,
    allocation_logits: Tensor,
    *,
    threshold: float,
    max_allocations: int,
) -> Tensor:
    """Select a first-free allocation prefix for each batch row.

    Empty physical slots are considered in ascending order. Once the first free
    slot falls below the allocation threshold, later free slots are not
    materialized. This makes physical slot identity a deterministic runtime
    policy rather than an arbitrary semantic label.
    """

    if occupancy.ndim == 3:
        if occupancy.shape[-1] != 1:
            raise ValueError("occupancy must have shape [batch, slots] or [batch, slots, 1]")
        occupancy = occupancy.squeeze(-1)
    if occupancy.ndim != 2 or allocation_logits.ndim != 2:
        raise ValueError("occupancy and allocation logits must have shape [batch, slots]")
    if occupancy.shape != allocation_logits.shape:
        raise ValueError("occupancy and allocation logits must have matching shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("allocation threshold must be in [0, 1]")
    if max_allocations <= 0:
        raise ValueError("max_allocations must be positive")

    occupied = occupancy.bool()
    # Allocation is a discrete runtime decision.  Evaluate its probability in
    # FP32 even when the model forward runs under BF16/FP16 autocast so small
    # rounding differences near the threshold do not change the allocation path.
    eligible = torch.sigmoid(allocation_logits.float()) >= threshold
    free = ~occupied

    # A low-confidence free slot terminates the first-free prefix. Occupied
    # positions are skipped and therefore do not terminate it.
    blocked = (free & ~eligible).cumsum(dim=1) > 0
    selected = free & eligible & ~blocked
    allocation_rank = selected.cumsum(dim=1)
    return selected & (allocation_rank <= max_allocations)

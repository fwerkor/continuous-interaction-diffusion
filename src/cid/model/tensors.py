from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from cid.grounding import ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.state import CellLifecycle


@dataclass(slots=True)
class CIDTensorBatch:
    thought_semantic: Tensor
    role_features: Tensor
    uncertainty: Tensor
    local_noise: Tensor
    slot_occupancy: Tensor
    prompt_ids: Tensor
    display_ids: Tensor
    display_noise: Tensor
    fact_memory: Tensor
    percept_memory: Tensor
    source_memory: Tensor
    lifecycle_features: Tensor | None = None
    percept_thought_mask: Tensor | None = None
    percept_display_mask: Tensor | None = None
    prompt_padding_mask: Tensor | None = None
    display_padding_mask: Tensor | None = None
    fact_padding_mask: Tensor | None = None
    percept_padding_mask: Tensor | None = None
    source_padding_mask: Tensor | None = None
    sample_mask: Tensor | None = None


@dataclass(slots=True)
class CIDTensorOutput:
    thought_semantic: Tensor
    convergence_logits: Tensor
    allocation_logits: Tensor
    role_logits: Tensor
    uncertainty: Tensor
    noise_delta: Tensor
    lifecycle_logits: Tensor
    display_logits: Tensor
    need_logits: Tensor
    source_logits: Tensor
    need_target_cell_logits: Tensor
    need_target_display_logits: Tensor
    argument_presence_logits: Tensor
    argument_query: Tensor
    anchor_query: Tensor
    anchor_presence_logits: Tensor
    anchor_kind_logits: Tensor
    link_presence_logits: Tensor
    link_relation_logits: Tensor
    link_target_kind_logits: Tensor
    link_target_query: Tensor
    revision_logits: Tensor
    refresh_logits: Tensor
    auxiliary_loss: Tensor | None = None


def live_slot_occupancy(
    slot_occupancy: Tensor, lifecycle_features: Tensor | None
) -> Tensor:
    """Return neural occupancy while keeping RETIRED slots physically reserved."""

    occupancy = slot_occupancy.clamp(0.0, 1.0)
    if lifecycle_features is None:
        return occupancy
    retired_index = MODELED_LIFECYCLES.index(CellLifecycle.RETIRED)
    retired = lifecycle_features[..., retired_index : retired_index + 1].clamp(0.0, 1.0)
    return occupancy * (1.0 - retired)


def build_percept_routing_masks(
    target_cells: tuple[tuple[ObjectRef, ...], ...],
    target_display: tuple[tuple[ObjectRef, ...], ...],
    *,
    cell_slots: Mapping[str, int],
    thought_slots: int,
    display_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if len(target_cells) != len(target_display):
        raise ValueError("percept target cell/display lists must have the same length")
    percept_count = len(target_cells)
    thought_mask = torch.zeros(
        (1, thought_slots, percept_count), dtype=torch.bool, device=device
    )
    display_mask = torch.zeros(
        (1, display_length, percept_count), dtype=torch.bool, device=device
    )
    for index, (cell_targets, display_targets) in enumerate(
        zip(target_cells, target_display, strict=True)
    ):
        if cell_targets:
            for target in cell_targets:
                slot = cell_slots.get(target.identifier)
                if slot is not None:
                    thought_mask[0, slot, index] = True
        else:
            thought_mask[0, :, index] = True

        if display_targets:
            for target in display_targets:
                if target.span is None:
                    continue
                start, end = target.span
                start = max(0, min(start, display_length))
                end = max(start, min(end, display_length))
                display_mask[0, start:end, index] = True
        else:
            display_mask[0, :, index] = True
    return thought_mask, display_mask

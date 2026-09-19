from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from cid.grounding import ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.state import CellLifecycle


class DeviceSemantic(Sequence[float]):
    """A semantic vector that stays on its source device until scalar access is required."""

    __slots__ = ("_tensor", "_sketch")

    def __init__(
        self,
        tensor: Tensor,
        *,
        sketch: tuple[float, ...] | None = None,
    ) -> None:
        if tensor.ndim != 1:
            raise ValueError("semantic tensor must be one-dimensional")
        self._tensor = tensor.detach()
        self._sketch = sketch

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def __len__(self) -> int:
        return self._tensor.numel()

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        value = self._tensor[index]
        if isinstance(index, slice):
            return tuple(float(item) for item in value.detach().float().cpu().tolist())
        return float(value)

    def __iter__(self) -> Iterator[float]:
        return iter(self._tensor.detach().float().cpu().tolist())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeviceSemantic):
            if self._tensor.shape != other._tensor.shape:
                return False
            return bool(torch.equal(self._tensor, other._tensor))
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented

    def sketch(self, samples: int = 12) -> tuple[float, ...]:
        if self._sketch is not None and samples == len(self._sketch):
            return self._sketch
        if len(self) <= samples:
            return tuple(round(value, 3) for value in self)
        last = len(self) - 1
        indexes = torch.tensor(
            [round(index * last / (samples - 1)) for index in range(samples)],
            device=self._tensor.device,
        )
        values = self._tensor.index_select(0, indexes).detach().float().cpu().tolist()
        return tuple(round(float(value), 3) for value in values)


def semantic_as_tensor(
    semantic: Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if isinstance(semantic, DeviceSemantic):
        return semantic.tensor.to(
            device=device,
            dtype=dtype or semantic.tensor.dtype,
        )
    return torch.tensor(semantic, device=device, dtype=dtype)


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

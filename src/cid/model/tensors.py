from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class CIDTensorBatch:
    thought_semantic: Tensor
    role_features: Tensor
    uncertainty: Tensor
    local_noise: Tensor
    slot_occupancy: Tensor
    display_ids: Tensor
    display_noise: Tensor
    fact_memory: Tensor
    percept_memory: Tensor
    source_memory: Tensor
    fact_padding_mask: Tensor | None = None
    percept_padding_mask: Tensor | None = None
    source_padding_mask: Tensor | None = None


@dataclass(slots=True)
class CIDTensorOutput:
    thought_semantic: Tensor
    allocation_logits: Tensor
    role_logits: Tensor
    uncertainty: Tensor
    noise_delta: Tensor
    lifecycle_logits: Tensor
    display_logits: Tensor
    need_logits: Tensor
    source_logits: Tensor
    anchor_query: Tensor
    anchor_presence_logits: Tensor
    anchor_kind_logits: Tensor
    link_presence_logits: Tensor
    link_relation_logits: Tensor
    link_target_kind_logits: Tensor
    link_target_query: Tensor
    revision_logits: Tensor
    refresh_logits: Tensor

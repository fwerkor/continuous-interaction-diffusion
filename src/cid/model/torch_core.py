from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cid.grounding import AnchorKind, LinkRelation, ObjectKind
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.components import CIDExternalFusion, CIDOutputHeads
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput


@dataclass(frozen=True, slots=True)
class TorchCIDConfig:
    vocab_size: int
    d_model: int = 512
    num_roles: int = 6
    num_lifecycles: int = len(MODELED_LIFECYCLES)
    num_anchor_kinds: int = len(AnchorKind)
    num_link_relations: int = len(LinkRelation)
    num_object_kinds: int = len(ObjectKind)
    num_refresh_actions: int = 3
    max_anchor_slots: int = 4
    max_link_slots: int = 8
    num_layers: int = 6
    num_heads: int = 8
    ff_multiplier: int = 4
    max_thought_slots: int = 128
    max_display_tokens: int = 1024
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_lifecycles != len(MODELED_LIFECYCLES):
            raise ValueError("lifecycle head predicts ACTIVE/WAITING/STABLE/RETIRED only")
        if self.num_anchor_kinds != len(AnchorKind):
            raise ValueError("num_anchor_kinds must match the typed grounding ABI")
        if self.num_link_relations != len(LinkRelation):
            raise ValueError("num_link_relations must match the typed grounding ABI")
        if self.num_object_kinds != len(ObjectKind):
            raise ValueError("num_object_kinds must match the typed grounding ABI")
        if self.max_anchor_slots <= 0 or self.max_link_slots <= 0:
            raise ValueError("grounding slot capacities must be positive")


class TorchCIDCore(nn.Module):
    """Small reference network for the CID tensor contract, not a target-scale model."""

    def __init__(self, config: TorchCIDConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, d)
        self.thought_position = nn.Embedding(config.max_thought_slots, d)
        self.display_position = nn.Embedding(config.max_display_tokens, d)
        self.channel_embedding = nn.Embedding(2, d)
        self.role_projection = nn.Linear(config.num_roles, d, bias=False)
        self.scalar_projection = nn.Linear(2, d, bias=False)
        self.occupancy_projection = nn.Linear(1, d, bias=False)
        self.display_noise_projection = nn.Linear(1, d, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=d * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.external_fusion = CIDExternalFusion(
            d,
            config.num_heads,
            dropout=config.dropout,
        )
        self.output_heads = CIDOutputHeads(
            d_model=d,
            num_roles=config.num_roles,
            num_lifecycles=config.num_lifecycles,
            num_anchor_kinds=config.num_anchor_kinds,
            num_link_relations=config.num_link_relations,
            num_object_kinds=config.num_object_kinds,
            num_refresh_actions=config.num_refresh_actions,
            max_anchor_slots=config.max_anchor_slots,
            max_link_slots=config.max_link_slots,
        )
        self.display_head = nn.Linear(d, config.vocab_size, bias=False)
        self.display_head.weight = self.token_embedding.weight

    @property
    def thought_delta(self) -> nn.Linear:
        return self.output_heads.thought_delta

    def forward(self, batch: CIDTensorBatch) -> CIDTensorOutput:
        thought = batch.thought_semantic
        batch_size, thought_slots, width = thought.shape
        if width != self.config.d_model:
            raise ValueError("thought_semantic width does not match d_model")
        display_length = batch.display_ids.shape[1]
        if thought_slots > self.config.max_thought_slots:
            raise ValueError("thought slot count exceeds configured maximum")
        if batch.slot_occupancy.shape != (batch_size, thought_slots, 1):
            raise ValueError("slot_occupancy must have shape [batch, thought_slots, 1]")
        if display_length > self.config.max_display_tokens:
            raise ValueError("display length exceeds configured maximum")

        device = thought.device
        t_pos = self.thought_position(torch.arange(thought_slots, device=device))[None, :, :]
        y_pos = self.display_position(torch.arange(display_length, device=device))[None, :, :]
        t_channel = self.channel_embedding.weight[0][None, None, :]
        y_channel = self.channel_embedding.weight[1][None, None, :]

        t_scalars = torch.cat((batch.uncertainty, batch.local_noise), dim=-1)
        thought_hidden = (
            thought
            + self.role_projection(batch.role_features)
            + self.scalar_projection(t_scalars)
            + self.occupancy_projection(batch.slot_occupancy)
            + t_pos
            + t_channel
        )
        display_hidden = (
            self.token_embedding(batch.display_ids)
            + self.display_noise_projection(batch.display_noise)
            + y_pos
            + y_channel
        )
        seed_hidden = torch.cat((thought_hidden, display_hidden), dim=1)
        thought_weight = batch.slot_occupancy.clamp(0.0, 1.0)
        display_weight = torch.ones(
            (batch_size, display_length, 1),
            dtype=thought_weight.dtype,
            device=device,
        )
        context_weight = torch.cat((thought_weight, display_weight), dim=1)

        hidden = self.backbone(seed_hidden)
        hidden = self.external_fusion(
            hidden,
            seed_hidden=seed_hidden,
            context_weight=context_weight,
            facts=batch.fact_memory,
            percepts=batch.percept_memory,
            fact_padding_mask=batch.fact_padding_mask,
            percept_padding_mask=batch.percept_padding_mask,
        )
        t_hidden = hidden[:, :thought_slots]
        y_hidden = hidden[:, thought_slots:]
        return self.output_heads(
            base_thought=thought,
            thought_hidden=t_hidden,
            display_logits=self.display_head(y_hidden),
            source_memory=batch.source_memory,
            source_padding_mask=batch.source_padding_mask,
        )

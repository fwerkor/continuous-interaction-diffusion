from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TorchCIDConfig:
    vocab_size: int
    d_model: int = 512
    num_roles: int = 6
    num_lifecycles: int = 5
    num_refresh_actions: int = 3
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
    occupancy_logits: Tensor
    role_logits: Tensor
    uncertainty: Tensor
    noise_delta: Tensor
    lifecycle_logits: Tensor
    display_logits: Tensor
    need_logits: Tensor
    source_logits: Tensor
    anchor_query: Tensor
    revision_logits: Tensor
    refresh_logits: Tensor


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
        self.external_type_embedding = nn.Embedding(2, d)
        self.role_projection = nn.Linear(config.num_roles, d, bias=False)
        self.scalar_projection = nn.Linear(2, d, bias=False)
        self.occupancy_projection = nn.Linear(1, d, bias=False)
        self.display_noise_projection = nn.Linear(1, d, bias=False)
        self.percept_projection = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.null_external = nn.Parameter(torch.zeros(1, 1, d))

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
        self.external_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.external_gate = nn.Linear(d * 2, d)
        self.final_norm = nn.LayerNorm(d)

        self.thought_delta = nn.Linear(d, d)
        self.occupancy_head = nn.Linear(d, 1)
        self.role_head = nn.Linear(d, config.num_roles)
        self.uncertainty_head = nn.Linear(d, 1)
        self.noise_head = nn.Linear(d, 1)
        self.lifecycle_head = nn.Linear(d, config.num_lifecycles)
        self.display_head = nn.Linear(d, config.vocab_size, bias=False)
        self.need_head = nn.Linear(d, 1)
        self.source_query = nn.Linear(d, d, bias=False)
        self.anchor_query = nn.Linear(d, d, bias=False)
        self.revision_head = nn.Linear(d, 3)
        self.refresh_head = nn.Linear(d, config.num_refresh_actions)

        self.display_head.weight = self.token_embedding.weight

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
        context_summary = (seed_hidden * context_weight).sum(dim=1, keepdim=True)
        context_summary = context_summary / context_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        percept_context = context_summary.expand(-1, batch.percept_memory.shape[1], -1)
        projected_percepts = self.percept_projection(
            torch.cat((batch.percept_memory, percept_context), dim=-1)
        )
        external_memory, external_padding_mask = self._external_memory(
            batch_size=batch_size,
            facts=batch.fact_memory,
            percepts=projected_percepts,
            fact_padding_mask=batch.fact_padding_mask,
            percept_padding_mask=batch.percept_padding_mask,
        )

        hidden = self.backbone(seed_hidden)
        external, _ = self.external_attention(
            hidden,
            external_memory,
            external_memory,
            key_padding_mask=external_padding_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(self.external_gate(torch.cat((hidden, external), dim=-1)))
        hidden = self.final_norm(hidden + gate * external)

        t_hidden = hidden[:, :thought_slots]
        y_hidden = hidden[:, thought_slots:]
        source_query = F.normalize(self.source_query(t_hidden), dim=-1)
        source_memory = F.normalize(batch.source_memory, dim=-1)
        source_logits = torch.einsum("bnd,bsd->bns", source_query, source_memory)
        if batch.source_padding_mask is not None:
            source_logits = source_logits.masked_fill(
                batch.source_padding_mask[:, None, :],
                torch.finfo(source_logits.dtype).min,
            )

        return CIDTensorOutput(
            thought_semantic=thought + self.thought_delta(t_hidden),
            occupancy_logits=self.occupancy_head(t_hidden).squeeze(-1),
            role_logits=self.role_head(t_hidden),
            uncertainty=torch.sigmoid(self.uncertainty_head(t_hidden)),
            noise_delta=torch.tanh(self.noise_head(t_hidden)),
            lifecycle_logits=self.lifecycle_head(t_hidden),
            display_logits=self.display_head(y_hidden),
            need_logits=self.need_head(t_hidden).squeeze(-1),
            source_logits=source_logits,
            anchor_query=self.anchor_query(t_hidden),
            revision_logits=self.revision_head(t_hidden),
            refresh_logits=self.refresh_head(t_hidden),
        )

    def _external_memory(
        self,
        *,
        batch_size: int,
        facts: Tensor,
        percepts: Tensor,
        fact_padding_mask: Tensor | None,
        percept_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        fact_count = facts.shape[1]
        percept_count = percepts.shape[1]
        if fact_count + percept_count == 0:
            return self.null_external.expand(batch_size, -1, -1), None

        fact_type = self.external_type_embedding.weight[0][None, None, :]
        percept_type = self.external_type_embedding.weight[1][None, None, :]
        memory = torch.cat((facts + fact_type, percepts + percept_type), dim=1)

        if fact_padding_mask is None and percept_padding_mask is None:
            return memory, None
        if fact_padding_mask is None:
            fact_padding_mask = torch.zeros(
                (batch_size, fact_count), dtype=torch.bool, device=facts.device
            )
        if percept_padding_mask is None:
            percept_padding_mask = torch.zeros(
                (batch_size, percept_count), dtype=torch.bool, device=percepts.device
            )
        return memory, torch.cat((fact_padding_mask, percept_padding_mask), dim=1)

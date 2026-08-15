from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cid.model.tensors import CIDTensorOutput


class CIDExternalFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        normalize_output: bool = True,
        gate_init_bias: float | None = None,
    ) -> None:
        super().__init__()
        self.external_type_embedding = nn.Embedding(2, d_model)
        self.num_heads = num_heads
        self.percept_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.null_external = nn.Parameter(torch.zeros(1, 1, d_model))
        self.external_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.external_gate = nn.Linear(d_model * 2, d_model)
        self.final_norm = nn.LayerNorm(d_model) if normalize_output else nn.Identity()
        if gate_init_bias is not None:
            nn.init.zeros_(self.external_gate.weight)
            nn.init.constant_(self.external_gate.bias, gate_init_bias)

    def forward(
        self,
        hidden: Tensor,
        *,
        seed_hidden: Tensor,
        context_weight: Tensor,
        facts: Tensor,
        percepts: Tensor,
        fact_padding_mask: Tensor | None = None,
        percept_padding_mask: Tensor | None = None,
        percept_query_mask: Tensor | None = None,
    ) -> Tensor:
        context_summary = (seed_hidden * context_weight).sum(dim=1, keepdim=True)
        context_summary = context_summary / context_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        percept_context = context_summary.expand(-1, percepts.shape[1], -1)
        projected_percepts = self.percept_projection(
            torch.cat((percepts, percept_context), dim=-1)
        )
        include_null = (
            percept_query_mask is not None or facts.shape[1] + percepts.shape[1] == 0
        )
        external_memory, external_padding_mask = self._external_memory(
            batch_size=hidden.shape[0],
            facts=facts,
            percepts=projected_percepts,
            fact_padding_mask=fact_padding_mask,
            percept_padding_mask=percept_padding_mask,
            include_null=include_null,
        )
        attention_mask = self._query_attention_mask(
            hidden=hidden,
            fact_count=facts.shape[1],
            percept_count=percepts.shape[1],
            percept_query_mask=percept_query_mask,
            fact_padding_mask=fact_padding_mask,
            percept_padding_mask=percept_padding_mask,
        )
        external, _ = self.external_attention(
            hidden,
            external_memory,
            external_memory,
            key_padding_mask=external_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(self.external_gate(torch.cat((hidden, external), dim=-1)))
        fused = self.final_norm(hidden + gate * external)
        # The null token is conditionally included in attention memory. Keep its
        # parameter connected to every graph so DDP sees a stable trainable set
        # even when ranks receive different external-memory layouts.
        return fused + self.null_external.sum() * 0.0

    def _external_memory(
        self,
        *,
        batch_size: int,
        facts: Tensor,
        percepts: Tensor,
        fact_padding_mask: Tensor | None,
        percept_padding_mask: Tensor | None,
        include_null: bool,
    ) -> tuple[Tensor, Tensor | None]:
        fact_count = facts.shape[1]
        percept_count = percepts.shape[1]
        fact_type = self.external_type_embedding.weight[0][None, None, :]
        percept_type = self.external_type_embedding.weight[1][None, None, :]
        parts = [facts + fact_type, percepts + percept_type]
        if include_null:
            parts.append(self.null_external.expand(batch_size, -1, -1))
        memory = torch.cat(parts, dim=1)

        if fact_padding_mask is None:
            fact_padding_mask = torch.zeros(
                (batch_size, fact_count), dtype=torch.bool, device=facts.device
            )
        if percept_padding_mask is None:
            percept_padding_mask = torch.zeros(
                (batch_size, percept_count), dtype=torch.bool, device=percepts.device
            )
        padding_parts = [fact_padding_mask, percept_padding_mask]
        if include_null:
            padding_parts.append(
                torch.zeros((batch_size, 1), dtype=torch.bool, device=memory.device)
            )
        padding_mask = torch.cat(padding_parts, dim=1)
        return memory, padding_mask if bool(padding_mask.any()) else None

    def _query_attention_mask(
        self,
        *,
        hidden: Tensor,
        fact_count: int,
        percept_count: int,
        percept_query_mask: Tensor | None,
        fact_padding_mask: Tensor | None,
        percept_padding_mask: Tensor | None,
    ) -> Tensor | None:
        if percept_query_mask is None:
            return None
        expected = (hidden.shape[0], hidden.shape[1], percept_count)
        if percept_query_mask.shape != expected:
            raise ValueError(
                "percept_query_mask must have shape "
                f"{expected}, got {tuple(percept_query_mask.shape)}"
            )
        fact_allowed = torch.ones(
            (hidden.shape[0], hidden.shape[1], fact_count),
            dtype=torch.bool,
            device=hidden.device,
        )
        routed_percepts = percept_query_mask.to(device=hidden.device, dtype=torch.bool)
        if fact_count == 0:
            fact_available = torch.zeros(
                hidden.shape[0], dtype=torch.bool, device=hidden.device
            )
        elif fact_padding_mask is None:
            fact_available = torch.ones(
                hidden.shape[0], dtype=torch.bool, device=hidden.device
            )
        else:
            fact_available = (~fact_padding_mask.to(hidden.device, dtype=torch.bool)).any(dim=1)

        if percept_count == 0:
            percept_available = torch.zeros(
                (hidden.shape[0], hidden.shape[1]),
                dtype=torch.bool,
                device=hidden.device,
            )
        else:
            if percept_padding_mask is None:
                percept_valid = torch.ones(
                    (hidden.shape[0], percept_count),
                    dtype=torch.bool,
                    device=hidden.device,
                )
            else:
                percept_valid = ~percept_padding_mask.to(hidden.device, dtype=torch.bool)
            percept_available = (
                routed_percepts & percept_valid[:, None, :]
            ).any(dim=-1)

        has_external = fact_available[:, None] | percept_available
        null_allowed = (~has_external).unsqueeze(-1)
        allowed = torch.cat(
            (fact_allowed, routed_percepts, null_allowed),
            dim=-1,
        )
        disallowed = ~allowed
        return (
            disallowed[:, None, :, :]
            .expand(-1, self.num_heads, -1, -1)
            .reshape(hidden.shape[0] * self.num_heads, hidden.shape[1], -1)
        )


class CIDOutputHeads(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_roles: int,
        num_lifecycles: int,
        num_anchor_kinds: int,
        num_link_relations: int,
        num_object_kinds: int,
        num_refresh_actions: int,
        max_argument_slots: int,
        max_anchor_slots: int,
        max_link_slots: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_anchor_kinds = num_anchor_kinds
        self.num_link_relations = num_link_relations
        self.num_object_kinds = num_object_kinds
        self.max_argument_slots = max_argument_slots
        self.max_anchor_slots = max_anchor_slots
        self.max_link_slots = max_link_slots

        self.thought_delta = nn.Linear(d_model, d_model)
        self.convergence_head = nn.Linear(d_model, 1)
        self.allocation_head = nn.Linear(d_model, 1)
        self.role_head = nn.Linear(d_model, num_roles)
        self.uncertainty_head = nn.Linear(d_model, 1)
        self.noise_head = nn.Linear(d_model, 1)
        self.lifecycle_head = nn.Linear(d_model, num_lifecycles)
        self.need_head = nn.Linear(d_model, 1)
        self.source_query = nn.Linear(d_model, d_model, bias=False)
        self.argument_presence_head = nn.Linear(d_model, max_argument_slots)
        self.argument_query = nn.Linear(d_model, max_argument_slots * d_model, bias=False)
        self.anchor_query = nn.Linear(d_model, max_anchor_slots * d_model, bias=False)
        self.anchor_presence_head = nn.Linear(d_model, max_anchor_slots)
        self.anchor_kind_head = nn.Linear(d_model, max_anchor_slots * num_anchor_kinds)
        self.link_presence_head = nn.Linear(d_model, max_link_slots)
        self.link_relation_head = nn.Linear(d_model, max_link_slots * num_link_relations)
        self.link_target_kind_head = nn.Linear(d_model, max_link_slots * num_object_kinds)
        self.link_target_query = nn.Linear(d_model, max_link_slots * d_model, bias=False)
        self.revision_head = nn.Linear(d_model, 3)
        self.refresh_head = nn.Linear(d_model, num_refresh_actions)

    def forward(
        self,
        *,
        base_thought: Tensor,
        thought_hidden: Tensor,
        thought_occupancy: Tensor,
        display_logits: Tensor,
        source_memory: Tensor,
        source_padding_mask: Tensor | None = None,
    ) -> CIDTensorOutput:
        batch_size, thought_slots, _ = thought_hidden.shape
        occupancy_weight = thought_occupancy.to(dtype=thought_hidden.dtype).clamp(0.0, 1.0)
        occupied_count = occupancy_weight.sum(dim=1)
        occupied_summary = (thought_hidden * occupancy_weight).sum(dim=1)
        occupied_summary = occupied_summary / occupied_count.clamp_min(1.0)
        fallback_summary = thought_hidden.mean(dim=1)
        summary = torch.where(
            (occupied_count > 0.0).expand_as(occupied_summary),
            occupied_summary,
            fallback_summary,
        )
        source_query = F.normalize(self.source_query(thought_hidden), dim=-1)
        normalized_sources = F.normalize(source_memory, dim=-1)
        source_logits = torch.einsum("bnd,bsd->bns", source_query, normalized_sources)
        if source_padding_mask is not None:
            source_logits = source_logits.masked_fill(
                source_padding_mask[:, None, :],
                torch.finfo(source_logits.dtype).min,
            )

        argument_query = self.argument_query(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_argument_slots,
            self.d_model,
        )

        anchor_query = self.anchor_query(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_anchor_slots,
            self.d_model,
        )
        anchor_kind_logits = self.anchor_kind_head(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_anchor_slots,
            self.num_anchor_kinds,
        )
        link_relation_logits = self.link_relation_head(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_link_slots,
            self.num_link_relations,
        )
        link_target_kind_logits = self.link_target_kind_head(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_link_slots,
            self.num_object_kinds,
        )
        link_target_query = self.link_target_query(thought_hidden).view(
            batch_size,
            thought_slots,
            self.max_link_slots,
            self.d_model,
        )

        return CIDTensorOutput(
            thought_semantic=base_thought + self.thought_delta(thought_hidden),
            convergence_logits=self.convergence_head(summary).squeeze(-1),
            allocation_logits=self.allocation_head(thought_hidden).squeeze(-1),
            role_logits=self.role_head(thought_hidden),
            uncertainty=torch.sigmoid(self.uncertainty_head(thought_hidden)),
            noise_delta=torch.tanh(self.noise_head(thought_hidden)),
            lifecycle_logits=self.lifecycle_head(thought_hidden),
            display_logits=display_logits,
            need_logits=self.need_head(thought_hidden).squeeze(-1),
            source_logits=source_logits,
            argument_presence_logits=self.argument_presence_head(thought_hidden),
            argument_query=argument_query,
            anchor_query=anchor_query,
            anchor_presence_logits=self.anchor_presence_head(thought_hidden),
            anchor_kind_logits=anchor_kind_logits,
            link_presence_logits=self.link_presence_head(thought_hidden),
            link_relation_logits=link_relation_logits,
            link_target_kind_logits=link_target_kind_logits,
            link_target_query=link_target_query,
            revision_logits=self.revision_head(thought_hidden),
            refresh_logits=self.refresh_head(thought_hidden),
        )

from __future__ import annotations

import torch
from torch import Tensor, nn

from cid.model.allocation import prefix_allocation_mask
from cid.model.ar import AR_CID_MODEL_TYPES, bidirectional_ar_hidden_states
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput, live_slot_occupancy


class IdentityPreservingPath(nn.Module):
    """One pretrained backbone pass; CID interaction uses small cross-attention residuals.

    If the controller neither has nor allocates any Thought cells, and there
    is no external evidence, Display logits follow the native language model.
    Internal reasoning can allocate Thought on its very first step, independent
    of external tools, and immediately condition Display on that new Thought.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_thought_slots: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.thought_queries = nn.Parameter(torch.empty(max_thought_slots, d_model))
        self.control_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.display_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.control_norm = nn.LayerNorm(d_model)
        self.display_norm = nn.LayerNorm(d_model)
        self.display_noise_condition = nn.Linear(1, d_model, bias=False)
        nn.init.zeros_(self.display_noise_condition.weight)
        self.display_gate = nn.Parameter(torch.tensor(-3.0))
        self.control_gate = nn.Parameter(torch.tensor(-2.0))
        self.num_heads = num_heads
        nn.init.normal_(self.thought_queries, std=0.02)

    @staticmethod
    def _keys(
        padding_mask: Tensor | None, reference: Tensor
    ) -> Tensor:
        if padding_mask is None:
            return torch.ones(reference.shape[:2], device=reference.device, dtype=torch.bool)
        return ~padding_mask.to(device=reference.device, dtype=torch.bool)

    @staticmethod
    def _percept_restrictions(
        mask: Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        memory_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if mask is None:
            return None
        expected = (batch_size, query_length, memory_length)
        if mask.shape != expected:
            raise ValueError(f"percept routing mask must have shape {expected}")
        return mask.to(device=device, dtype=torch.bool)

    def _native_hidden(
        self, adapter: nn.Module, ids: Tensor, keys: Tensor, batch: CIDTensorBatch
    ) -> tuple[Tensor, Tensor | None]:
        decoder = adapter.hidden_backbone()
        if adapter.backbone_family in AR_CID_MODEL_TYPES:
            hidden = bidirectional_ar_hidden_states(
                decoder,
                inputs_embeds=adapter.input_embeddings(ids),
                attention_mask=keys,
                position_ids=keys.long().cumsum(dim=1).sub(1).clamp_min(0),
            )
            return hidden, None
        for attention in adapter._llada_moe_attention_modules:
            attention._cid_key_padding_mask = keys
        router_enabled = adapter._router_aux_loss_enabled
        if router_enabled:
            adapter._captured_router_logits.clear()
            adapter._capture_router_logits = True
        try:
            kwargs: dict[str, object] = dict(
                inputs_embeds=adapter.input_embeddings(ids),
                attention_mask=keys,
                position_ids=keys.long().cumsum(dim=1).sub(1).clamp_min(0),
                return_dict=True,
            )
            if router_enabled:
                kwargs["output_router_logits"] = True
            output = decoder(**kwargs)
        finally:
            adapter._capture_router_logits = False
            for attention in adapter._llada_moe_attention_modules:
                attention._cid_key_padding_mask = None
        auxiliary = None
        if router_enabled:
            from cid.model.illada import _moe_load_balancing_loss

            router_logits = (
                tuple(adapter._captured_router_logits)
                if adapter._captured_router_logits
                else getattr(output, "router_logits", None)
            )
            router_keys = keys
            if batch.sample_mask is not None:
                router_keys = keys & batch.sample_mask[:, None].to(
                    device=keys.device, dtype=torch.bool
                )
            balance = _moe_load_balancing_loss(
                router_logits,
                num_experts=int(adapter.backbone.config.num_experts),
                top_k=int(adapter.backbone.config.num_experts_per_tok),
                attention_mask=router_keys,
            )
            if balance is not None:
                auxiliary = balance * adapter.router_aux_loss_coef
        return output.last_hidden_state, auxiliary

    def forward(self, adapter: nn.Module, batch: CIDTensorBatch) -> CIDTensorOutput:
        batch_size, slots, prompt_length, display_length = adapter._validate_batch(batch)
        dtype = adapter.input_embeddings.weight.dtype
        device = batch.display_ids.device
        prompt_keys = self._keys(batch.prompt_padding_mask, batch.prompt_ids)
        display_keys = self._keys(batch.display_padding_mask, batch.display_ids)
        keys = torch.cat((prompt_keys, display_keys), dim=1)
        token_ids = torch.cat((batch.prompt_ids, batch.display_ids), dim=1)
        hidden, auxiliary = self._native_hidden(adapter, token_ids, keys, batch)
        y_native = hidden[:, prompt_length:, :]
        thought = batch.thought_semantic.to(dtype=dtype)
        occupancy = batch.slot_occupancy.to(dtype=dtype).clamp(0.0, 1.0)
        lifecycle = (
            batch.lifecycle_features.to(dtype=dtype)
            if batch.lifecycle_features is not None
            else torch.zeros(
                batch_size, slots, adapter.config.num_lifecycles, device=device, dtype=dtype
            )
        )
        neural_occupancy = live_slot_occupancy(occupancy, lifecycle)
        thought_query = (
            self.thought_queries[None, :slots].to(dtype=dtype)
            + thought * neural_occupancy
            + adapter.role_projection(batch.role_features.to(dtype=dtype)) * neural_occupancy
            + adapter.lifecycle_projection(lifecycle) * neural_occupancy
            + adapter.scalar_projection(
                torch.cat(
                    (batch.uncertainty.to(dtype=dtype), batch.local_noise.to(dtype=dtype)),
                    dim=-1,
                )
            ) * neural_occupancy
        )

        memories: list[Tensor] = [hidden]
        memory_keys: list[Tensor] = [keys]
        for kind in ("source", "fact", "percept"):
            value = getattr(batch, f"{kind}_memory").to(dtype=dtype)
            if value.shape[1]:
                memories.append(value)
                memory_keys.append(
                    self._keys(getattr(batch, f"{kind}_padding_mask"), value)
                )
        control_memory = torch.cat(memories, dim=1)
        control_keys = torch.cat(memory_keys, dim=1)
        control_mask = torch.zeros(
            (batch_size, slots, control_memory.shape[1]),
            device=device, dtype=torch.bool,
        )
        if batch.percept_memory.shape[1]:
            percept_mask = self._percept_restrictions(
                batch.percept_thought_mask,
                batch_size=batch_size, query_length=slots,
                memory_length=batch.percept_memory.shape[1], device=device,
            )
            if percept_mask is not None:
                control_mask[:, :, -batch.percept_memory.shape[1]:] = ~percept_mask
        t_delta, _ = self.control_attention(
            thought_query,
            control_memory, control_memory,
            key_padding_mask=~control_keys,
            attn_mask=control_mask.repeat_interleave(self.num_heads, dim=0),
            need_weights=False,
        )
        t_hidden = self.control_norm(
            thought_query + torch.sigmoid(self.control_gate) * t_delta
        )
        # All CID control and lifecycle semantics continue through the original
        # heads, including allocation, need binding, revision and convergence.
        # The initial evaluation supplies the updated semantic Thought state.
        base_logits = adapter.output_embeddings(y_native)
        first = adapter.output_heads(
            base_thought=thought, thought_hidden=t_hidden,
            thought_occupancy=neural_occupancy,
            display_hidden=y_native, display_logits=base_logits,
            source_memory=batch.source_memory.to(dtype=dtype),
            source_padding_mask=batch.source_padding_mask,
        )

        # Use exactly the materializer's prefix allocation decision. Newly
        # allocated cells are eligible to influence Display in *this* update.
        # No source or tool call is required for the cognitive transition.
        newly_allocated = prefix_allocation_mask(
            occupancy.squeeze(-1).bool(),
            first.allocation_logits.detach(),
            threshold=adapter.config.identity_allocation_threshold,
            max_allocations=adapter.config.identity_max_allocations_per_step,
        )
        active_thought = neural_occupancy.squeeze(-1).bool() | newly_allocated
        fallback_thought = torch.zeros_like(active_thought)
        fallback_thought[:, 0] = True
        thought_keys = torch.where(
            active_thought.any(dim=1, keepdim=True),
            active_thought, fallback_thought,
        )
        display_memories: list[Tensor] = [
            t_hidden + first.thought_semantic.to(dtype=dtype)
        ]
        display_keys_list: list[Tensor] = [thought_keys]
        base_evidence = active_thought.any(dim=1)
        percept_evidence = torch.zeros(
            (batch_size, display_length), device=device, dtype=torch.bool,
        )
        for kind in ("source", "fact", "percept"):
            value = getattr(batch, f"{kind}_memory").to(dtype=dtype)
            if value.shape[1]:
                valid = self._keys(getattr(batch, f"{kind}_padding_mask"), value)
                display_memories.append(value)
                display_keys_list.append(valid)
                if kind == "percept":
                    targets = self._percept_restrictions(
                        batch.percept_display_mask,
                        batch_size=batch_size, query_length=display_length,
                        memory_length=value.shape[1], device=device,
                    )
                    if targets is None:
                        percept_evidence = valid.any(dim=1, keepdim=True).expand(
                            -1, display_length
                        )
                    else:
                        percept_evidence = (targets & valid[:, None, :]).any(dim=-1)
                else:
                    base_evidence = base_evidence | valid.any(dim=1)
        content = torch.cat(display_memories, dim=1)
        content_keys = torch.cat(display_keys_list, dim=1)
        display_mask = torch.zeros(
            (batch_size, display_length, content.shape[1]),
            device=device, dtype=torch.bool,
        )
        if batch.percept_memory.shape[1]:
            percept_mask = self._percept_restrictions(
                batch.percept_display_mask,
                batch_size=batch_size, query_length=display_length,
                memory_length=batch.percept_memory.shape[1], device=device,
            )
            if percept_mask is not None:
                display_mask[:, :, -batch.percept_memory.shape[1]:] = ~percept_mask
        # The diffusion timestep influences only the context correction,
        # never the native masked-LM input distribution.
        display_query = (
            y_native + self.display_noise_condition(batch.display_noise.to(dtype=dtype))
        )
        y_delta, _ = self.display_attention(
            display_query, content, content,
            key_padding_mask=~content_keys,
            attn_mask=display_mask.repeat_interleave(self.num_heads, dim=0),
            need_weights=False,
        )
        active_positions = base_evidence[:, None] | percept_evidence
        gate = active_positions.to(dtype=dtype).unsqueeze(-1)
        if self.training:
            # Straight-through gate lets language loss teach the controller
            # when to initiate reasoning while preserving exact hard routing.
            soft_reason = torch.sigmoid(first.allocation_logits.float()).amax(dim=1)
            gate = gate + (
                soft_reason - soft_reason.detach()
            ).to(dtype=dtype).view(batch_size, 1, 1)
        y_hidden = y_native + gate * torch.sigmoid(self.display_gate) * self.display_norm(y_delta)
        logits = adapter.output_embeddings(y_hidden)
        if adapter.stage_c_display is not None:
            # Stage C refines the language model's *native* expression path,
            # while the T/Tool conditional delta stays independently bounded.
            # In the neutral state CID logits equal the adapted native LM.
            logits = logits + adapter.stage_c_display(y_native)
        # The second inexpensive head pass sees the *updated* Display. Tool
        # controls and Thought are updated together within this CID step.
        result = adapter.output_heads(
            base_thought=thought, thought_hidden=t_hidden,
            thought_occupancy=neural_occupancy,
            display_hidden=y_hidden, display_logits=logits,
            source_memory=batch.source_memory.to(dtype=dtype),
            source_padding_mask=batch.source_padding_mask,
        )
        # A newly allocated Thought cell must survive at least one subsequent
        # CID step before the runtime is permitted to converge. Otherwise the
        # model may claim to "think" while finishing without any iteration.
        result.convergence_logits = result.convergence_logits.masked_fill(
            newly_allocated.any(dim=1), -20.0
        )
        if self.training:
            # The frozen native language path is an anchor, not merely an
            # initialization. Penalize excessive changes in hidden space and
            # especially a large EOS logit increase before the current EOS.
            # This regularizer does not constrain Thought cells or tool
            # routing; their task losses can still justify a correction.
            native_norm = y_native.float().norm(dim=-1).detach().clamp_min(1.0)
            shift_ratio = (y_hidden.float() - y_native.float()).norm(dim=-1) / native_norm
            available = display_keys
            eos_id = adapter.eos_token_id
            positions = torch.arange(display_length, device=device)
            boundary = torch.where(
                (batch.display_ids == eos_id).any(dim=1, keepdim=True),
                (batch.display_ids == eos_id).int().argmax(dim=1, keepdim=True),
                torch.full((batch_size, 1), display_length, device=device),
            )
            before_eos = positions[None, :] < boundary
            protected = available & (positions[None, :] <= boundary) & ~percept_evidence
            excess = (shift_ratio - 0.12).relu()
            preserve = (excess.square() * protected).sum() / protected.sum().clamp_min(1)
            eos_mask = before_eos & available & ~percept_evidence
            eos_rise = (
                logits[..., eos_id].float() - base_logits[..., eos_id].float().detach()
                - 1.5
            ).relu()
            eos_preserve = (eos_rise.square() * eos_mask).sum() / eos_mask.sum().clamp_min(1)
            auxiliary = (auxiliary if auxiliary is not None else preserve.new_zeros(()))
            auxiliary = auxiliary + 0.5 * preserve + 0.05 * eos_preserve
        result.auxiliary_loss = auxiliary
        return result

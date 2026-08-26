from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

from cid.grounding import AnchorKind, LinkRelation, ObjectKind
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.components import CIDExternalFusion, CIDOutputHeads
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput

ILLADA_8B_BASE = "GSAI-ML/iLLaDA-8B-Base"
ILLADA_8B_BASE_REVISION = "a1b5b5f8a31a3854a46205ee584178c04b45ec9a"
ILLADA_MASK_TOKEN_ID = 5
ILLADA_EOS_TOKEN_ID = 2
LLADA_MOE_7B_A1B_BASE = "inclusionAI/LLaDA-MoE-7B-A1B-Base"
LLADA_MOE_7B_A1B_BASE_REVISION = "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf"
LLADA_MOE_MASK_TOKEN_ID = 156895
LLADA_MOE_EOS_TOKEN_ID = 156892
DEFAULT_DISPLAY_CANVAS_TOKENS = 64
DEFAULT_MAX_DISPLAY_TOKENS = 1536


def _chunked_illada_mlp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    chunk_size = int(module._cid_mlp_chunk_size)
    if x.ndim != 3 or x.shape[1] <= chunk_size:
        return module.dropout(
            module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))
        )
    outputs = []
    for start in range(0, x.shape[1], chunk_size):
        chunk = x[:, start : start + chunk_size]
        outputs.append(
            module.dropout(
                module.down_proj(
                    module.act_fn(module.gate_proj(chunk)) * module.up_proj(chunk)
                )
            )
        )
    return torch.cat(outputs, dim=1)


def _chunked_illada_rms_norm_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    chunk_size = int(module._cid_norm_chunk_size)
    if hidden_states.ndim != 3 or hidden_states.shape[1] <= chunk_size:
        input_dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        values = values * torch.rsqrt(variance + module.variance_epsilon)
        return module.weight * values.to(input_dtype)
    outputs = []
    for start in range(0, hidden_states.shape[1], chunk_size):
        chunk = hidden_states[:, start : start + chunk_size]
        input_dtype = chunk.dtype
        values = chunk.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        values = values * torch.rsqrt(variance + module.variance_epsilon)
        outputs.append(module.weight * values.to(input_dtype))
    return torch.cat(outputs, dim=1)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    half = hidden_states.shape[-1] // 2
    return torch.cat((-hidden_states[..., half:], hidden_states[..., :half]), dim=-1)


def _apply_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query_states * cos + _rotate_half(query_states) * sin,
        key_states * cos + _rotate_half(key_states) * sin,
    )


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, sequence, head_dim)
        .reshape(batch, kv_heads * groups, sequence, head_dim)
    )


def _llada_moe_sdpa_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value: object | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **_: object,
) -> tuple[torch.Tensor, None, object | None]:
    """SDPA attention that restores CID key masks dropped by upstream LLaDA-MoE."""
    del position_ids, cache_position
    if output_attentions:
        raise RuntimeError("CID LLaDA-MoE attention does not expose attention weights")
    if use_cache or past_key_value is not None:
        raise RuntimeError("CID diffusion attention does not support KV caching")
    if position_embeddings is None:
        raise RuntimeError("LLaDA-MoE decoder must provide rotary position embeddings")

    batch_size, sequence_length, _ = hidden_states.shape
    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states.reshape(-1, module.head_dim)).reshape(
            batch_size, sequence_length, -1
        )
        key_states = module.k_norm(key_states.reshape(-1, module.head_dim)).reshape(
            batch_size, sequence_length, -1
        )
    value_states = module.v_proj(hidden_states)

    clip_qkv = getattr(module.config, "clip_qkv", None)
    if clip_qkv is not None:
        query_states = query_states.clamp(min=-clip_qkv, max=clip_qkv)
        key_states = key_states.clamp(min=-clip_qkv, max=clip_qkv)
        value_states = value_states.clamp(min=-clip_qkv, max=clip_qkv)

    query_states = query_states.view(
        batch_size, sequence_length, module.num_heads, module.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, sequence_length, module.num_key_value_heads, module.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, sequence_length, module.num_key_value_heads, module.head_dim
    ).transpose(1, 2)
    query_states, key_states = _apply_rotary(
        query_states, key_states, *position_embeddings
    )
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    value_states = _repeat_kv(value_states, module.num_key_value_groups)

    key_mask = getattr(module, "_cid_key_padding_mask", attention_mask)
    sdpa_mask = None
    if key_mask is not None:
        if key_mask.ndim == 2:
            sdpa_mask = key_mask[:, None, None, : key_states.shape[-2]].bool()
        elif key_mask.ndim == 4:
            sdpa_mask = key_mask[..., : key_states.shape[-2]]
        else:
            raise ValueError("CID LLaDA-MoE attention mask must be rank 2 or rank 4")
        if hidden_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

    attention_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=sdpa_mask,
        dropout_p=module.attention_dropout if module.training else 0.0,
        is_causal=False,
    )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch_size, sequence_length, module.hidden_size
    )
    return module.o_proj(attention_output), None, past_key_value


def _install_llada_moe_attention_mask_support(backbone: nn.Module) -> tuple[nn.Module, ...]:
    decoder = backbone.get_decoder()
    layers = getattr(decoder, "layers", None)
    if layers is None or not layers:
        raise RuntimeError("LLaDA-MoE decoder does not expose layers")
    attention_modules = []
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError("LLaDA-MoE decoder layer does not expose self attention")
        required = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "num_heads",
            "num_key_value_heads",
            "num_key_value_groups",
            "head_dim",
            "hidden_size",
        )
        if any(not hasattr(attention, name) for name in required):
            raise RuntimeError("unsupported LLaDA-MoE attention implementation")
        attention.forward = MethodType(_llada_moe_sdpa_forward, attention)
        attention_modules.append(attention)
    return tuple(attention_modules)


def _moe_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...] | None,
    *,
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Match the upstream LLaDA-MoE router balancing objective.

    CID calls the hidden decoder directly so it can inject TCT embeddings. That bypasses
    ``LLaDAMoEModelLM.forward()``, where the upstream implementation normally adds this
    objective. Recompute the same scalar from the decoder router logits instead of silently
    dropping router supervision during full-parameter Stage B training.
    """

    if not router_logits:
        return None
    compute_device = router_logits[0].device
    concatenated = torch.cat(
        tuple(layer_logits.to(compute_device) for layer_logits in router_logits), dim=0
    )
    routing_weights = torch.softmax(concatenated, dim=-1, dtype=torch.float32)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    )

    if attention_mask is None:
        tokens_per_expert = expert_mask.float().mean(dim=0)
        router_prob_per_expert = routing_weights.mean(dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        layer_count = concatenated.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(layer_count, batch_size, sequence_length, top_k, num_experts)
            .reshape(-1, top_k, num_experts)
            .to(device=compute_device, dtype=torch.float32)
        )
        tokens_per_expert = (
            expert_mask.float() * expert_attention_mask
        ).sum(dim=0) / expert_attention_mask.sum(dim=0).clamp_min(1.0)

        router_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(layer_count, batch_size, sequence_length, num_experts)
            .reshape(-1, num_experts)
            .to(device=compute_device, dtype=torch.float32)
        )
        router_prob_per_expert = (
            routing_weights * router_attention_mask
        ).sum(dim=0) / router_attention_mask.sum(dim=0).clamp_min(1.0)

    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts


@dataclass(frozen=True, slots=True)
class ILLaDACIDConfig:
    max_thought_slots: int = 128
    max_display_tokens: int = DEFAULT_MAX_DISPLAY_TOKENS
    display_canvas_tokens: int | None = None
    num_roles: int = 6
    num_lifecycles: int = len(MODELED_LIFECYCLES)
    num_anchor_kinds: int = len(AnchorKind)
    num_link_relations: int = len(LinkRelation)
    num_object_kinds: int = len(ObjectKind)
    num_refresh_actions: int = 3
    max_argument_slots: int = 4
    max_anchor_slots: int = 4
    max_link_slots: int = 8
    external_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.max_thought_slots <= 0 or self.max_display_tokens <= 0:
            raise ValueError("thought and display capacities must be positive")
        canvas = self.display_canvas_tokens
        if canvas is None:
            canvas = min(DEFAULT_DISPLAY_CANVAS_TOKENS, self.max_display_tokens)
            object.__setattr__(self, "display_canvas_tokens", canvas)
        if not 1 < canvas <= self.max_display_tokens:
            raise ValueError("display canvas must fit within configured display capacity")
        if self.num_lifecycles != len(MODELED_LIFECYCLES):
            raise ValueError("lifecycle head predicts ACTIVE/WAITING/STABLE/RETIRED only")
        if self.num_anchor_kinds != len(AnchorKind):
            raise ValueError("num_anchor_kinds must match the typed grounding ABI")
        if self.num_link_relations != len(LinkRelation):
            raise ValueError("num_link_relations must match the typed grounding ABI")
        if self.num_object_kinds != len(ObjectKind):
            raise ValueError("num_object_kinds must match the typed grounding ABI")
        if self.max_argument_slots <= 0:
            raise ValueError("argument slot capacity must be positive")
        if self.max_anchor_slots <= 0 or self.max_link_slots <= 0:
            raise ValueError("grounding slot capacities must be positive")
        if not 0.0 <= self.external_dropout < 1.0:
            raise ValueError("external_dropout must be in [0, 1)")


class ILLaDACIDAdapter(nn.Module):
    """CID bridge for compatible full-sequence LLaDA-family diffusion backbones."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ILLaDACIDConfig | None = None,
        *,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or ILLaDACIDConfig()

        backbone_config = backbone.config
        model_type = str(backbone_config.model_type)
        self.is_llada_moe = model_type == "llada" and int(
            getattr(backbone_config, "num_experts", 0)
        ) > 0
        if model_type != "illada" and not self.is_llada_moe:
            raise ValueError(
                "expected an iLLaDA or LLaDA-MoE backbone, "
                f"got {backbone_config.model_type!r}"
            )
        self.d_model = int(backbone_config.hidden_size)
        self.vocab_size = int(backbone_config.vocab_size)
        self.max_position_embeddings = int(backbone_config.max_position_embeddings)
        default_mask_token_id = (
            LLADA_MOE_MASK_TOKEN_ID if self.is_llada_moe else ILLADA_MASK_TOKEN_ID
        )
        self.mask_token_id = int(
            getattr(backbone_config, "mask_token_id", None) or default_mask_token_id
        )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("backbone mask token ID must be inside its vocabulary")
        self.eos_token_id = int(
            getattr(
                backbone_config,
                "eos_token_id",
                LLADA_MOE_EOS_TOKEN_ID if self.is_llada_moe else ILLADA_EOS_TOKEN_ID,
            )
        )
        self.router_aux_loss_coef = (
            float(getattr(backbone_config, "router_aux_loss_coef", 0.0))
            if self.is_llada_moe
            else 0.0
        )
        self._router_aux_loss_enabled = False
        self._llada_moe_attention_modules = (
            _install_llada_moe_attention_mask_support(backbone)
            if self.is_llada_moe
            else ()
        )
        num_heads = int(backbone_config.num_attention_heads)
        if self.d_model % num_heads:
            raise ValueError("iLLaDA hidden size must be divisible by its attention head count")

        self.channel_embedding = nn.Embedding(3, self.d_model)
        self.role_projection = nn.Linear(self.config.num_roles, self.d_model, bias=False)
        self.lifecycle_projection = nn.Linear(
            self.config.num_lifecycles, self.d_model, bias=False
        )
        self.scalar_projection = nn.Linear(2, self.d_model, bias=False)
        self.occupancy_projection = nn.Linear(1, self.d_model, bias=False)
        self.display_noise_projection = nn.Linear(1, self.d_model, bias=False)
        self.external_fusion = CIDExternalFusion(
            self.d_model,
            num_heads,
            dropout=self.config.external_dropout,
            normalize_output=False,
            gate_init_bias=-6.0,
        )
        self.output_heads = CIDOutputHeads(
            d_model=self.d_model,
            num_roles=self.config.num_roles,
            num_lifecycles=self.config.num_lifecycles,
            num_anchor_kinds=self.config.num_anchor_kinds,
            num_link_relations=self.config.num_link_relations,
            num_object_kinds=self.config.num_object_kinds,
            num_refresh_actions=self.config.num_refresh_actions,
            max_argument_slots=self.config.max_argument_slots,
            max_anchor_slots=self.config.max_anchor_slots,
            max_link_slots=self.config.max_link_slots,
        )

        nn.init.zeros_(self.channel_embedding.weight)
        nn.init.zeros_(self.display_noise_projection.weight)
        nn.init.zeros_(self.output_heads.thought_delta.weight)
        nn.init.zeros_(self.output_heads.thought_delta.bias)
        self._place_cid_modules_with_embeddings()
        self.set_backbone_trainable(not freeze_backbone)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = ILLADA_8B_BASE,
        *,
        config: ILLaDACIDConfig | None = None,
        freeze_backbone: bool = False,
        **from_pretrained_kwargs: object,
    ) -> ILLaDACIDAdapter:
        from transformers import AutoModelForCausalLM

        from_pretrained_kwargs.setdefault("trust_remote_code", True)
        if model_name_or_path == ILLADA_8B_BASE:
            from_pretrained_kwargs.setdefault("revision", ILLADA_8B_BASE_REVISION)
        elif model_name_or_path == LLADA_MOE_7B_A1B_BASE:
            from_pretrained_kwargs.setdefault("revision", LLADA_MOE_7B_A1B_BASE_REVISION)
            from_pretrained_kwargs.setdefault("attn_implementation", "sdpa")
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
        return cls(backbone, config=config, freeze_backbone=freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        self._router_aux_loss_enabled = self.is_llada_moe and trainable

    def set_gradient_checkpointing(
        self,
        enabled: bool,
        *,
        use_reentrant: bool | None = None,
    ) -> None:
        method_name = (
            "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        )
        method = getattr(self.backbone, method_name, None)
        if method is None:
            raise RuntimeError("iLLaDA backbone does not expose gradient checkpointing controls")
        if enabled and use_reentrant is not None:
            method(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        else:
            method()

    def set_mlp_chunk_size(self, chunk_size: int | None) -> None:
        """Chunk token-wise iLLaDA MLP evaluation without changing its function."""
        if chunk_size is None:
            return
        if chunk_size <= 0:
            raise ValueError("MLP chunk size must be positive")
        if self.is_llada_moe:
            # The MoE FFN receives only routed token rows and does not expose the dense
            # iLLaDA MLP contract patched below. Its sparse expert path is already token-wise.
            return
        decoder = self.backbone.get_decoder()
        layers = getattr(decoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError("iLLaDA decoder does not expose layers for MLP chunking")
        resid_pdrop = float(getattr(self.backbone.config, "resid_pdrop", 0.0))
        if resid_pdrop != 0.0:
            raise RuntimeError("exact MLP chunking requires zero residual dropout")
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            required = ("gate_proj", "up_proj", "down_proj", "act_fn", "dropout")
            if mlp is None or any(not hasattr(mlp, name) for name in required):
                raise RuntimeError("iLLaDA decoder layer does not expose the expected MLP")
            mlp._cid_mlp_chunk_size = int(chunk_size)
            mlp.forward = MethodType(_chunked_illada_mlp_forward, mlp)

    def set_norm_chunk_size(self, chunk_size: int | None) -> None:
        """Chunk token-wise iLLaDA RMSNorm evaluation without changing its function."""
        if chunk_size is None:
            return
        if chunk_size <= 0:
            raise ValueError("norm chunk size must be positive")
        decoder = self.backbone.get_decoder()
        layers = getattr(decoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError("iLLaDA decoder does not expose layers for norm chunking")
        norms = [getattr(decoder, "norm", None)]
        for layer in layers:
            norms.extend(
                (
                    getattr(layer, "input_layernorm", None),
                    getattr(layer, "post_attention_layernorm", None),
                )
            )
        for norm in norms:
            if norm is None or not hasattr(norm, "variance_epsilon") or not hasattr(norm, "weight"):
                raise RuntimeError("exact norm chunking requires iLLaDA RMSNorm modules")
            norm._cid_norm_chunk_size = int(chunk_size)
            norm.forward = MethodType(_chunked_illada_rms_norm_forward, norm)

    @property
    def input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    @property
    def output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def forward(self, batch: CIDTensorBatch) -> CIDTensorOutput:
        batch_size, thought_slots, prompt_length, display_length = self._validate_batch(batch)
        model_dtype = self.input_embeddings.weight.dtype
        thought = batch.thought_semantic.to(dtype=model_dtype)
        role_features = batch.role_features.to(dtype=model_dtype)
        uncertainty = batch.uncertainty.to(dtype=model_dtype)
        local_noise = batch.local_noise.to(dtype=model_dtype)
        slot_occupancy = batch.slot_occupancy.to(dtype=model_dtype)
        lifecycle_features = (
            torch.zeros(
                (*batch.thought_semantic.shape[:2], self.config.num_lifecycles),
                device=batch.thought_semantic.device,
                dtype=model_dtype,
            )
            if batch.lifecycle_features is None
            else batch.lifecycle_features.to(dtype=model_dtype)
        )
        display_noise = batch.display_noise.to(dtype=model_dtype)
        fact_memory = batch.fact_memory.to(dtype=model_dtype)
        percept_memory = batch.percept_memory.to(dtype=model_dtype)
        source_memory = batch.source_memory.to(dtype=model_dtype)

        t_scalars = torch.cat((uncertainty, local_noise), dim=-1)
        thought_hidden = (
            thought
            + self.role_projection(role_features)
            + self.lifecycle_projection(lifecycle_features)
            + self.scalar_projection(t_scalars)
            + self.occupancy_projection(slot_occupancy)
            + self.channel_embedding.weight[0][None, None, :]
        )
        prompt_hidden = (
            self.input_embeddings(batch.prompt_ids)
            + self.channel_embedding.weight[1][None, None, :]
        )
        display_hidden = (
            self.input_embeddings(batch.display_ids)
            + self.display_noise_projection(display_noise)
            + self.channel_embedding.weight[2][None, None, :]
        )
        seed_hidden = torch.cat((thought_hidden, prompt_hidden, display_hidden), dim=1)

        prompt_keys = self._valid_keys(
            batch.prompt_padding_mask,
            batch_size=batch_size,
            length=prompt_length,
            device=batch.display_ids.device,
        )
        display_keys = self._valid_keys(
            batch.display_padding_mask,
            batch_size=batch_size,
            length=display_length,
            device=batch.display_ids.device,
        )
        attention_mask = torch.cat(
            (slot_occupancy.squeeze(-1).bool(), prompt_keys, display_keys),
            dim=1,
        )
        position_ids = self._logical_position_ids(
            thought_slots=thought_slots,
            prompt_keys=prompt_keys,
            display_keys=display_keys,
        )
        decoder_kwargs: dict[str, object] = {
            "inputs_embeds": seed_hidden,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "return_dict": True,
        }
        if self._router_aux_loss_enabled:
            decoder_kwargs["output_router_logits"] = True
        for attention in self._llada_moe_attention_modules:
            attention._cid_key_padding_mask = attention_mask
        decoder_output = self.backbone.get_decoder()(
            **decoder_kwargs,
        )
        hidden = decoder_output.last_hidden_state

        thought_weight = slot_occupancy.clamp(0.0, 1.0)
        prompt_weight = prompt_keys.to(dtype=thought_weight.dtype).unsqueeze(-1)
        display_weight = display_keys.to(dtype=thought_weight.dtype).unsqueeze(-1)
        hidden = self.external_fusion(
            hidden,
            seed_hidden=seed_hidden,
            context_weight=torch.cat((thought_weight, prompt_weight, display_weight), dim=1),
            facts=fact_memory,
            percepts=percept_memory,
            fact_padding_mask=batch.fact_padding_mask,
            percept_padding_mask=batch.percept_padding_mask,
            percept_query_mask=self._percept_query_mask(
                batch,
                thought_slots=thought_slots,
                prompt_length=prompt_length,
                display_length=display_length,
            ),
        )

        t_hidden = hidden[:, :thought_slots]
        y_hidden = hidden[:, thought_slots + prompt_length :]
        output = self.output_heads(
            base_thought=thought,
            thought_hidden=t_hidden,
            thought_occupancy=slot_occupancy,
            display_logits=self.output_embeddings(y_hidden),
            source_memory=source_memory,
            source_padding_mask=batch.source_padding_mask,
        )
        if self._router_aux_loss_enabled:
            raw_router_loss = _moe_load_balancing_loss(
                getattr(decoder_output, "router_logits", None),
                num_experts=int(self.backbone.config.num_experts),
                top_k=int(self.backbone.config.num_experts_per_tok),
                attention_mask=attention_mask,
            )
            if raw_router_loss is not None:
                output.auxiliary_loss = raw_router_loss * self.router_aux_loss_coef
        return output

    def _percept_query_mask(
        self,
        batch: CIDTensorBatch,
        *,
        thought_slots: int,
        prompt_length: int,
        display_length: int,
    ) -> torch.Tensor | None:
        if batch.percept_thought_mask is None and batch.percept_display_mask is None:
            return None
        batch_size = batch.thought_semantic.shape[0]
        percept_count = batch.percept_memory.shape[1]
        if batch.percept_thought_mask is None:
            thought_mask = torch.ones(
                (batch_size, thought_slots, percept_count),
                dtype=torch.bool,
                device=batch.thought_semantic.device,
            )
        else:
            expected = (batch_size, thought_slots, percept_count)
            if batch.percept_thought_mask.shape != expected:
                raise ValueError(
                    f"percept_thought_mask must have shape {expected}"
                )
            thought_mask = batch.percept_thought_mask.to(
                device=batch.thought_semantic.device, dtype=torch.bool
            )

        prompt_mask = torch.zeros(
            (batch_size, prompt_length, percept_count),
            dtype=torch.bool,
            device=batch.thought_semantic.device,
        )
        if batch.percept_display_mask is None:
            display_mask = torch.ones(
                (batch_size, display_length, percept_count),
                dtype=torch.bool,
                device=batch.thought_semantic.device,
            )
        else:
            expected = (batch_size, display_length, percept_count)
            if batch.percept_display_mask.shape != expected:
                raise ValueError(
                    f"percept_display_mask must have shape {expected}"
                )
            display_mask = batch.percept_display_mask.to(
                device=batch.thought_semantic.device, dtype=torch.bool
            )
        return torch.cat((thought_mask, prompt_mask, display_mask), dim=1)

    def _validate_batch(self, batch: CIDTensorBatch) -> tuple[int, int, int, int]:
        thought = batch.thought_semantic
        if thought.ndim != 3:
            raise ValueError("thought_semantic must have shape [batch, thought_slots, hidden]")
        batch_size, thought_slots, width = thought.shape
        if width != self.d_model:
            raise ValueError(
                f"thought_semantic width {width} does not match iLLaDA hidden size {self.d_model}"
            )
        if thought_slots > self.config.max_thought_slots:
            raise ValueError("thought slot count exceeds configured maximum")
        if batch.slot_occupancy.shape != (batch_size, thought_slots, 1):
            raise ValueError("slot_occupancy must have shape [batch, thought_slots, 1]")
        if batch.role_features.shape != (batch_size, thought_slots, self.config.num_roles):
            raise ValueError("role_features shape does not match the configured cognitive roles")
        if batch.uncertainty.shape != (batch_size, thought_slots, 1):
            raise ValueError("uncertainty must have shape [batch, thought_slots, 1]")
        if batch.local_noise.shape != (batch_size, thought_slots, 1):
            raise ValueError("local_noise must have shape [batch, thought_slots, 1]")
        if batch.lifecycle_features is not None and batch.lifecycle_features.shape != (
            batch_size, thought_slots, self.config.num_lifecycles
        ):
            raise ValueError("lifecycle_features shape does not match modeled lifecycle states")

        if batch.prompt_ids.ndim != 2 or batch.prompt_ids.shape[0] != batch_size:
            raise ValueError("prompt_ids must have shape [batch, prompt_tokens]")
        prompt_length = batch.prompt_ids.shape[1]
        self._validate_padding_mask(
            batch.prompt_padding_mask,
            batch_size=batch_size,
            length=prompt_length,
            name="prompt_padding_mask",
        )
        if bool(((batch.prompt_ids < 0) | (batch.prompt_ids >= self.vocab_size)).any()):
            raise ValueError("prompt contains token IDs outside the iLLaDA vocabulary")
        if batch.display_ids.ndim != 2 or batch.display_ids.shape[0] != batch_size:
            raise ValueError("display_ids must have shape [batch, display_tokens]")
        display_length = batch.display_ids.shape[1]
        self._validate_padding_mask(
            batch.display_padding_mask,
            batch_size=batch_size,
            length=display_length,
            name="display_padding_mask",
        )
        if display_length > self.config.max_display_tokens:
            raise ValueError("display length exceeds configured maximum")
        prompt_keys = self._valid_keys(
            batch.prompt_padding_mask,
            batch_size=batch_size,
            length=prompt_length,
            device=batch.prompt_ids.device,
        )
        display_keys = self._valid_keys(
            batch.display_padding_mask,
            batch_size=batch_size,
            length=display_length,
            device=batch.display_ids.device,
        )
        logical_lengths = (
            self.config.max_thought_slots
            + prompt_keys.sum(dim=1)
            + display_keys.sum(dim=1)
        )
        if bool((logical_lengths > self.max_position_embeddings).any()):
            raise ValueError(
                "combined TCT, prompt, and display length exceeds iLLaDA context capacity"
            )
        if batch.display_noise.shape != (batch_size, display_length, 1):
            raise ValueError("display_noise must have shape [batch, display_tokens, 1]")

        for name, memory in (
            ("fact_memory", batch.fact_memory),
            ("percept_memory", batch.percept_memory),
            ("source_memory", batch.source_memory),
        ):
            if memory.ndim != 3 or memory.shape[0] != batch_size or memory.shape[2] != self.d_model:
                raise ValueError(
                    f"{name} must have shape [batch, items, {self.d_model}]"
                )
        return batch_size, thought_slots, prompt_length, display_length

    @staticmethod
    def _validate_padding_mask(
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        length: int,
        name: str,
    ) -> None:
        if mask is not None and mask.shape != (batch_size, length):
            raise ValueError(f"{name} must have shape [batch, tokens]")

    @staticmethod
    def _valid_keys(
        padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if padding_mask is None:
            return torch.ones((batch_size, length), dtype=torch.bool, device=device)
        return ~padding_mask.to(device=device, dtype=torch.bool)

    def _logical_position_ids(
        self,
        *,
        thought_slots: int,
        prompt_keys: torch.Tensor,
        display_keys: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prompt_keys.shape[0]
        device = prompt_keys.device
        thought_positions = torch.arange(thought_slots, device=device).expand(batch_size, -1)

        prompt_offsets = prompt_keys.long().cumsum(dim=1) - 1
        prompt_positions = self.config.max_thought_slots + prompt_offsets
        prompt_positions = torch.where(
            prompt_keys,
            prompt_positions,
            torch.zeros_like(prompt_positions),
        )

        prompt_lengths = prompt_keys.sum(dim=1, keepdim=True).long()
        display_offsets = display_keys.long().cumsum(dim=1) - 1
        display_positions = (
            self.config.max_thought_slots + prompt_lengths + display_offsets
        )
        display_positions = torch.where(
            display_keys,
            display_positions,
            torch.zeros_like(display_positions),
        )
        return torch.cat((thought_positions, prompt_positions, display_positions), dim=1)

    def _place_cid_modules_with_embeddings(self) -> None:
        weight = self.input_embeddings.weight
        modules = (
            self.channel_embedding,
            self.role_projection,
            self.lifecycle_projection,
            self.scalar_projection,
            self.occupancy_projection,
            self.display_noise_projection,
            self.external_fusion,
            self.output_heads,
        )
        for module in modules:
            module.to(device=weight.device, dtype=weight.dtype)

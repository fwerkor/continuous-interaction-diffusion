from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cid.grounding import AnchorKind, LinkRelation, ObjectKind
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.components import CIDExternalFusion, CIDOutputHeads
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput

ILLADA_8B_BASE = "GSAI-ML/iLLaDA-8B-Base"
ILLADA_8B_BASE_REVISION = "a1b5b5f8a31a3854a46205ee584178c04b45ec9a"
ILLADA_MASK_TOKEN_ID = 5


@dataclass(frozen=True, slots=True)
class ILLaDACIDConfig:
    max_thought_slots: int = 128
    max_display_tokens: int = 1024
    num_roles: int = 6
    num_lifecycles: int = len(MODELED_LIFECYCLES)
    num_anchor_kinds: int = len(AnchorKind)
    num_link_relations: int = len(LinkRelation)
    num_object_kinds: int = len(ObjectKind)
    num_refresh_actions: int = 3
    max_anchor_slots: int = 4
    max_link_slots: int = 8
    external_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.max_thought_slots <= 0 or self.max_display_tokens <= 0:
            raise ValueError("thought and display capacities must be positive")
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
        if not 0.0 <= self.external_dropout < 1.0:
            raise ValueError("external_dropout must be in [0, 1)")


class ILLaDACIDAdapter(nn.Module):
    """CID bridge for the native bidirectional iLLaDA masked-diffusion backbone."""

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
        if backbone_config.model_type != "illada":
            raise ValueError(f"expected an iLLaDA backbone, got {backbone_config.model_type!r}")
        self.d_model = int(backbone_config.hidden_size)
        self.vocab_size = int(backbone_config.vocab_size)
        self.max_position_embeddings = int(backbone_config.max_position_embeddings)
        num_heads = int(backbone_config.num_attention_heads)
        if self.d_model % num_heads:
            raise ValueError("iLLaDA hidden size must be divisible by its attention head count")

        self.channel_embedding = nn.Embedding(2, self.d_model)
        self.role_projection = nn.Linear(self.config.num_roles, self.d_model, bias=False)
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
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
        return cls(backbone, config=config, freeze_backbone=freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    @property
    def input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    @property
    def output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def forward(self, batch: CIDTensorBatch) -> CIDTensorOutput:
        batch_size, thought_slots, display_length = self._validate_batch(batch)
        model_dtype = self.input_embeddings.weight.dtype
        thought = batch.thought_semantic.to(dtype=model_dtype)
        role_features = batch.role_features.to(dtype=model_dtype)
        uncertainty = batch.uncertainty.to(dtype=model_dtype)
        local_noise = batch.local_noise.to(dtype=model_dtype)
        slot_occupancy = batch.slot_occupancy.to(dtype=model_dtype)
        display_noise = batch.display_noise.to(dtype=model_dtype)
        fact_memory = batch.fact_memory.to(dtype=model_dtype)
        percept_memory = batch.percept_memory.to(dtype=model_dtype)
        source_memory = batch.source_memory.to(dtype=model_dtype)

        t_scalars = torch.cat((uncertainty, local_noise), dim=-1)
        thought_hidden = (
            thought
            + self.role_projection(role_features)
            + self.scalar_projection(t_scalars)
            + self.occupancy_projection(slot_occupancy)
            + self.channel_embedding.weight[0][None, None, :]
        )
        display_hidden = (
            self.input_embeddings(batch.display_ids)
            + self.display_noise_projection(display_noise)
            + self.channel_embedding.weight[1][None, None, :]
        )
        seed_hidden = torch.cat((thought_hidden, display_hidden), dim=1)

        display_keys = torch.ones(
            (batch_size, display_length),
            dtype=torch.bool,
            device=batch.display_ids.device,
        )
        attention_mask = torch.cat(
            (slot_occupancy.squeeze(-1).bool(), display_keys),
            dim=1,
        )
        decoder_output = self.backbone.get_decoder()(
            inputs_embeds=seed_hidden,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = decoder_output.last_hidden_state

        thought_weight = slot_occupancy.clamp(0.0, 1.0)
        display_weight = torch.ones(
            (batch_size, display_length, 1),
            dtype=thought_weight.dtype,
            device=thought_weight.device,
        )
        hidden = self.external_fusion(
            hidden,
            seed_hidden=seed_hidden,
            context_weight=torch.cat((thought_weight, display_weight), dim=1),
            facts=fact_memory,
            percepts=percept_memory,
            fact_padding_mask=batch.fact_padding_mask,
            percept_padding_mask=batch.percept_padding_mask,
        )

        t_hidden = hidden[:, :thought_slots]
        y_hidden = hidden[:, thought_slots:]
        return self.output_heads(
            base_thought=thought,
            thought_hidden=t_hidden,
            display_logits=self.output_embeddings(y_hidden),
            source_memory=source_memory,
            source_padding_mask=batch.source_padding_mask,
        )

    def _validate_batch(self, batch: CIDTensorBatch) -> tuple[int, int, int]:
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

        if batch.display_ids.ndim != 2 or batch.display_ids.shape[0] != batch_size:
            raise ValueError("display_ids must have shape [batch, display_tokens]")
        display_length = batch.display_ids.shape[1]
        if display_length > self.config.max_display_tokens:
            raise ValueError("display length exceeds configured maximum")
        if thought_slots + display_length > self.max_position_embeddings:
            raise ValueError("combined TCT and display length exceeds iLLaDA context capacity")
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
        return batch_size, thought_slots, display_length

    def _place_cid_modules_with_embeddings(self) -> None:
        weight = self.input_embeddings.weight
        modules = (
            self.channel_embedding,
            self.role_projection,
            self.scalar_projection,
            self.occupancy_projection,
            self.display_noise_projection,
            self.external_fusion,
            self.output_heads,
        )
        for module in modules:
            module.to(device=weight.device, dtype=weight.dtype)

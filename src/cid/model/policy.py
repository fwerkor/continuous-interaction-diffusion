from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cid.contracts import ModelContext, ModelUpdate, Percept, SourceDescriptor
from cid.model.diffusion import CIDDiffusionScheduler
from cid.model.encoding import ILLaDATextEncoder, stable_text
from cid.model.illada import (
    ILLADA_8B_BASE,
    ILLADA_8B_BASE_REVISION,
    ILLADA_MASK_TOKEN_ID,
    ILLaDACIDAdapter,
)
from cid.model.materialize import CIDMaterializer, ClosedWorldMaterializationCatalog
from cid.model.tensors import CIDTensorBatch, build_percept_routing_masks
from cid.state import CognitiveRole, FactItem


class ILLaDAContextTensorizer:
    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        text_encoder: ILLaDATextEncoder | None = None,
    ) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder or ILLaDATextEncoder(adapter, tokenizer)
        if self.text_encoder.d_model != adapter.d_model:
            raise ValueError("runtime text encoder width must match the iLLaDA adapter")

    @classmethod
    def from_pretrained(
        cls,
        adapter: ILLaDACIDAdapter,
        model_name_or_path: str = ILLADA_8B_BASE,
        **tokenizer_kwargs: object,
    ) -> ILLaDAContextTensorizer:
        from transformers import AutoTokenizer

        tokenizer_kwargs.setdefault("trust_remote_code", True)
        if model_name_or_path == ILLADA_8B_BASE:
            tokenizer_kwargs.setdefault("revision", ILLADA_8B_BASE_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        return cls(adapter, tokenizer)

    def __call__(self, context: ModelContext) -> CIDTensorBatch:
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        thought = context.thought
        if thought.width != self.adapter.d_model:
            raise ValueError("runtime TCT width does not match iLLaDA hidden size")
        if context.display.unresolved and context.display.mask_token_id != ILLADA_MASK_TOKEN_ID:
            raise ValueError(
                f"iLLaDA display canvas must use mask token id {ILLADA_MASK_TOKEN_ID}"
            )

        role_order = tuple(CognitiveRole)
        thought_semantic = torch.tensor(
            [[cell.semantic for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        role_features = torch.tensor(
            [
                [
                    [float(cell.roles.get(role, 0.0)) for role in role_order]
                    for cell in thought.cells
                ]
            ],
            device=device,
            dtype=dtype,
        )
        uncertainty = torch.tensor(
            [[[cell.uncertainty] for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        local_noise = torch.tensor(
            [[[cell.noise] for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        slot_occupancy = torch.tensor(
            [[[float(cell.occupied)] for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        display_ids = torch.tensor(
            [context.display.token_ids],
            device=device,
            dtype=torch.long,
        )
        if bool(((display_ids < 0) | (display_ids >= self.adapter.vocab_size)).any()):
            raise ValueError("display canvas contains token IDs outside the iLLaDA vocabulary")
        display_noise = (display_ids == context.display.mask_token_id).to(dtype).unsqueeze(-1)
        display_padding_mask = torch.zeros_like(display_ids, dtype=torch.bool)
        if context.display.active_span_length < len(context.display.token_ids):
            display_padding_mask[:, context.display.active_span_length :] = True
            display_noise[:, context.display.active_span_length :] = 0.0
        prompt_ids = self.text_encoder.tokenize(context.prompt, add_special_tokens=True)

        fact_memory = self.text_encoder.encode_texts(
            tuple(self._fact_text(item) for item in context.facts.items.values()),
            detach=True,
        )
        percept_memory = self.text_encoder.encode_texts(
            tuple(self._percept_text(item) for item in context.percepts),
            detach=True,
        )
        percept_thought_mask, percept_display_mask = self._percept_target_masks(
            context, device=device
        )
        source_memory = self.text_encoder.encode_texts(
            tuple(self._source_text(item) for item in context.sources),
            detach=True,
        )
        return CIDTensorBatch(
            thought_semantic=thought_semantic,
            role_features=role_features,
            uncertainty=uncertainty,
            local_noise=local_noise,
            slot_occupancy=slot_occupancy,
            prompt_ids=prompt_ids,
            display_ids=display_ids,
            display_noise=display_noise,
            fact_memory=fact_memory,
            percept_memory=percept_memory,
            source_memory=source_memory,
            percept_thought_mask=percept_thought_mask,
            percept_display_mask=percept_display_mask,
            display_padding_mask=display_padding_mask,
        )

    @staticmethod
    def _fact_text(item: FactItem) -> str:
        return " | ".join(
            (
                f"fact={item.key}",
                f"source={item.source_type}",
                f"value={stable_text(item.value)}",
                f"version={item.version or ''}",
            )
        )

    @staticmethod
    def _percept_text(item: Percept) -> str:
        anchors = ",".join(anchor.canonical_key for anchor in item.observation.anchors)
        target_cells = ",".join(target.identifier for target in item.target_cells)
        target_display = ",".join(
            f"{target.span[0]}:{target.span[1]}"
            for target in item.target_display
            if target.span is not None
        )
        return " | ".join(
            (
                f"percept={item.binding_id}",
                f"source={item.source}",
                f"value={stable_text(item.observation.value)}",
                f"version={item.observation.version or ''}",
                f"anchors={anchors}",
                f"target_cells={target_cells}",
                f"target_display={target_display}",
            )
        )

    @staticmethod
    def _percept_target_masks(
        context: ModelContext, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_slots = {
            cell_id: context.thought.slot_of(cell_id)
            for cell_id in context.thought.occupied_cell_ids
        }
        return build_percept_routing_masks(
            tuple(percept.target_cells for percept in context.percepts),
            tuple(percept.target_display for percept in context.percepts),
            cell_slots=cell_slots,
            thought_slots=context.thought.capacity,
            display_length=len(context.display.token_ids),
            device=device,
        )

    @staticmethod
    def _source_text(item: SourceDescriptor) -> str:
        arguments = ",".join(
            f"{argument.name}:{argument.kind}:{'required' if argument.required else 'optional'}"
            for argument in item.arguments
        )
        return " | ".join(
            (
                f"source={item.name}",
                f"description={item.description}",
                f"arguments={arguments}",
                f"dynamic={item.dynamic}",
                f"versioned={item.versioned}",
                f"accepts_partial_arguments={item.accepts_partial_arguments}",
            )
        )


@dataclass(frozen=True, slots=True)
class ILLaDANeuralPolicyConfig:
    denoising_steps: int = 8
    display_revision_fraction: float = 0.125
    display_revision_margin: float = 0.15

    def __post_init__(self) -> None:
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        if not 0.0 <= self.display_revision_fraction <= 1.0:
            raise ValueError("display_revision_fraction must be in [0, 1]")
        if self.display_revision_margin < 0.0:
            raise ValueError("display_revision_margin must be non-negative")


class ILLaDANeuralPolicy:
    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tensorizer: ILLaDAContextTensorizer,
        *,
        materializer: CIDMaterializer | None = None,
        catalog: ClosedWorldMaterializationCatalog | None = None,
        scheduler: CIDDiffusionScheduler | None = None,
        config: ILLaDANeuralPolicyConfig | None = None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        if tensorizer.adapter is not adapter:
            raise ValueError("tensorizer and neural policy must share the same adapter")
        self.adapter = adapter
        self.tensorizer = tensorizer
        self.materializer = materializer or CIDMaterializer()
        self.catalog = catalog or ClosedWorldMaterializationCatalog()
        self.scheduler = scheduler or CIDDiffusionScheduler(ILLADA_MASK_TOKEN_ID)
        self.config = config or ILLaDANeuralPolicyConfig()
        self.forward_model = forward_model or adapter

    def step(self, context: ModelContext) -> ModelUpdate:
        batch = self.tensorizer(context)
        with torch.no_grad():
            output = self.forward_model(batch)
            display_ids = self.scheduler.refine_display(
                batch.display_ids,
                output.display_logits,
                reveal_fraction=self._reveal_fraction(context.step),
                revision_fraction=self.config.display_revision_fraction,
                revision_margin=self.config.display_revision_margin,
            )
        return self.materializer.materialize(
            output,
            context,
            catalog=self.catalog,
            display_token_ids=tuple(int(token) for token in display_ids[0].tolist()),
        )

    def _reveal_fraction(self, step: int) -> float:
        remaining = max(1, self.config.denoising_steps - min(step, self.config.denoising_steps - 1))
        return 1.0 / remaining

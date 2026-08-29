from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cid.contracts import ModelContext, ModelUpdate, Percept, SourceDescriptor
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.diffusion import (
    CIDDiffusionScheduler,
    denoising_noise_level,
    denoising_reveal_fraction,
)
from cid.model.encoding import (
    ILLaDATextEncoder,
    canonical_fact_text,
    canonical_percept_text,
    canonical_source_text,
)
from cid.model.illada import ILLADA_8B_BASE, ILLaDACIDAdapter
from cid.model.loading import pretrained_revision
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
        self.scheduler = CIDDiffusionScheduler(adapter.mask_token_id, adapter.eos_token_id)
        if self.text_encoder.d_model != adapter.d_model:
            raise ValueError("runtime text encoder width must match the CID adapter")

    @classmethod
    def from_pretrained(
        cls,
        adapter: ILLaDACIDAdapter,
        model_name_or_path: str = ILLADA_8B_BASE,
        **tokenizer_kwargs: object,
    ) -> ILLaDAContextTensorizer:
        from transformers import AutoTokenizer

        tokenizer_kwargs.setdefault("trust_remote_code", True)
        revision = pretrained_revision(model_name_or_path)
        if revision is not None:
            tokenizer_kwargs.setdefault("revision", revision)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        return cls(adapter, tokenizer)

    def __call__(
        self,
        context: ModelContext,
        *,
        generator: torch.Generator | None = None,
        display_noise_level: float = 1.0,
    ) -> CIDTensorBatch:
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        thought = context.thought
        if thought.width != self.adapter.d_model:
            raise ValueError("runtime TCT width does not match backbone hidden size")
        if (
            context.display.unresolved
            and context.display.mask_token_id != self.adapter.mask_token_id
        ):
            raise ValueError(
                f"display canvas must use backbone mask token id {self.adapter.mask_token_id}"
            )

        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES
        thought_semantic = torch.tensor(
            [[cell.semantic for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        role_features = torch.tensor(
            [[[float(cell.roles.get(role, 0.0)) for role in role_order] for cell in thought.cells]],
            device=device,
            dtype=dtype,
        )
        lifecycle_features = torch.zeros(
            (1, thought.capacity, len(lifecycle_order)), device=device, dtype=dtype
        )
        for slot, cell in enumerate(thought.cells):
            if cell.occupied and cell.lifecycle in lifecycle_order:
                lifecycle_features[0, slot, lifecycle_order.index(cell.lifecycle)] = 1.0
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
        thought_corruption = self.scheduler.corrupt_thought(
            thought_semantic,
            local_noise.squeeze(-1),
            slot_occupancy,
            generator=generator,
        )
        display_ids = torch.tensor(
            [context.display.token_ids],
            device=device,
            dtype=torch.long,
        )
        if bool(((display_ids < 0) | (display_ids >= self.adapter.vocab_size)).any()):
            raise ValueError("display canvas contains token IDs outside the backbone vocabulary")
        if not 0.0 <= display_noise_level <= 1.0:
            raise ValueError("display_noise_level must be in [0, 1]")
        display_noise = torch.full(
            (*display_ids.shape, 1), display_noise_level, device=device, dtype=dtype
        )
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
            thought_semantic=thought_corruption.semantic,
            role_features=role_features,
            uncertainty=uncertainty,
            local_noise=thought_corruption.noise,
            slot_occupancy=slot_occupancy,
            prompt_ids=prompt_ids,
            display_ids=display_ids,
            display_noise=display_noise,
            fact_memory=fact_memory,
            percept_memory=percept_memory,
            source_memory=source_memory,
            lifecycle_features=lifecycle_features,
            percept_thought_mask=percept_thought_mask,
            percept_display_mask=percept_display_mask,
            display_padding_mask=display_padding_mask,
        )

    @staticmethod
    def _fact_text(item: FactItem) -> str:
        return canonical_fact_text(item)

    @staticmethod
    def _percept_text(item: Percept) -> str:
        return canonical_percept_text(
            source=item.source,
            value=item.observation.value,
            version=item.observation.version,
            anchors=item.observation.anchors,
            target_cells=item.target_cells,
            target_display=item.target_display,
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
        return canonical_source_text(item)


@dataclass(frozen=True, slots=True)
class ILLaDANeuralPolicyConfig:
    denoising_steps: int = 8
    display_revision_fraction: float = 0.125
    display_revision_margin: float = 0.15
    seed: int = 0

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
        self.scheduler = scheduler or tensorizer.scheduler
        self.config = config or ILLaDANeuralPolicyConfig()
        self.forward_model = forward_model or adapter
        self.generator = torch.Generator(device=self.tensorizer.text_encoder.device)
        self.generator.manual_seed(self.config.seed)

    def step(self, context: ModelContext) -> ModelUpdate:
        diffusion_step = context.diffusion_step
        batch = self.tensorizer(
            context,
            generator=self.generator,
            display_noise_level=denoising_noise_level(diffusion_step, self.config.denoising_steps),
        )
        with torch.no_grad():
            output = self.forward_model(batch)
            display_ids = self.scheduler.refine_display(
                batch.display_ids,
                output.display_logits,
                reveal_fraction=denoising_reveal_fraction(
                    diffusion_step, self.config.denoising_steps
                ),
                revision_fraction=self.config.display_revision_fraction,
                revision_margin=self.config.display_revision_margin,
            )
        return self.materializer.materialize(
            output,
            context,
            catalog=self.catalog,
            display_token_ids=tuple(int(token) for token in display_ids[0].tolist()),
        )

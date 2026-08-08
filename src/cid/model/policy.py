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
from cid.model.tensors import CIDTensorBatch
from cid.state import CognitiveRole, FactItem


class ILLaDAContextTensorizer:
    def __init__(self, adapter: ILLaDACIDAdapter, tokenizer: Any) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.text_encoder = ILLaDATextEncoder(adapter, tokenizer)

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
        weight = self.adapter.input_embeddings.weight
        device = weight.device
        dtype = weight.dtype
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
        prompt_ids = self.text_encoder.tokenize(context.prompt, add_special_tokens=True)

        fact_memory = self.text_encoder.encode_texts(
            tuple(self._fact_text(item) for item in context.facts.items.values()),
            detach=True,
        )
        percept_memory = self.text_encoder.encode_texts(
            tuple(self._percept_text(item) for item in context.percepts),
            detach=True,
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
        return " | ".join(
            (
                f"percept={item.binding_id}",
                f"source={item.source}",
                f"value={stable_text(item.observation.value)}",
                f"version={item.observation.version or ''}",
                f"anchors={anchors}",
            )
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
            )
        )


@dataclass(frozen=True, slots=True)
class ILLaDANeuralPolicyConfig:
    denoising_steps: int = 8

    def __post_init__(self) -> None:
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")


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
    ) -> None:
        if tensorizer.adapter is not adapter:
            raise ValueError("tensorizer and neural policy must share the same adapter")
        self.adapter = adapter
        self.tensorizer = tensorizer
        self.materializer = materializer or CIDMaterializer()
        self.catalog = catalog or ClosedWorldMaterializationCatalog()
        self.scheduler = scheduler or CIDDiffusionScheduler(ILLADA_MASK_TOKEN_ID)
        self.config = config or ILLaDANeuralPolicyConfig()

    def step(self, context: ModelContext) -> ModelUpdate:
        batch = self.tensorizer(context)
        with torch.no_grad():
            output = self.adapter(batch)
            display_ids = self.scheduler.reveal_display(
                batch.display_ids,
                output.display_logits,
                reveal_fraction=self._reveal_fraction(context.step),
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

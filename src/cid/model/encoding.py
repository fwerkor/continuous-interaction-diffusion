from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from cid.contracts import SourceDescriptor
from cid.grounding import Anchor, ObjectRef
from cid.model.illada import ILLaDACIDAdapter
from cid.state import FactItem


class ILLaDATextEncoder:
    """Tokenizer/embedding bridge shared by runtime and training tensorizers.

    The encoder deliberately stays lightweight because Stage B cannot afford a second frozen
    language model on 24 GB GPUs.  It nevertheless preserves token order through a first-moment
    term instead of treating a sentence as an unordered bag of embeddings.
    """

    ENCODING_VERSION = 2
    ORDER_MOMENT_SCALE = 0.25

    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        pooling_mode: str = "mean-v1",
    ) -> None:
        if pooling_mode not in {"mean-v1", "order-aware-v2"}:
            raise ValueError(f"unsupported semantic pooling mode: {pooling_mode}")
        self._embedding = adapter.input_embeddings
        self._output_device = self._embedding.weight.device
        self._output_dtype = self._embedding.weight.dtype
        self.tokenizer = tokenizer
        self.d_model = adapter.d_model
        self.pooling_mode = pooling_mode

    @classmethod
    def from_frozen_snapshot(
        cls,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        embedding_device: torch.device | str | None = None,
        pooling_mode: str = "mean-v1",
    ) -> ILLaDATextEncoder:
        if pooling_mode not in {"mean-v1", "order-aware-v2"}:
            raise ValueError(f"unsupported semantic pooling mode: {pooling_mode}")
        snapshot = cls.__new__(cls)
        output_device = torch.device(device)
        storage_device = (
            torch.device(embedding_device) if embedding_device is not None else output_device
        )
        weight = (
            adapter.input_embeddings.weight.detach().to(device=storage_device, dtype=dtype).clone()
        )
        snapshot._embedding = nn.Embedding.from_pretrained(weight, freeze=True)
        snapshot._output_device = output_device
        snapshot._output_dtype = dtype
        snapshot.tokenizer = tokenizer
        snapshot.d_model = adapter.d_model
        snapshot.pooling_mode = pooling_mode
        return snapshot

    @property
    def device(self) -> torch.device:
        return self._output_device

    @property
    def dtype(self) -> torch.dtype:
        return self._output_dtype

    @property
    def embedding_device(self) -> torch.device:
        return self._embedding.weight.device

    def tokenize(self, text: str, *, add_special_tokens: bool) -> Tensor:
        if not text:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device=self.device)

    def encode_one(self, text: str, *, detach: bool = False) -> Tensor:
        return self.encode_texts((text,), detach=detach)[0, 0]

    def encode_texts(self, texts: tuple[str, ...], *, detach: bool = False) -> Tensor:
        if not texts:
            return torch.empty(
                (1, 0, self.d_model),
                device=self.device,
                dtype=self.dtype,
            )
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device=self.embedding_device)
        if detach:
            with torch.no_grad():
                embeddings = self._embedding(input_ids)
        else:
            embeddings = self._embedding(input_ids)
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        attention_mask = encoded["attention_mask"].to(device=self.device, dtype=self.dtype)
        weights = attention_mask.unsqueeze(-1)
        token_count = weights.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * weights).sum(dim=1) / token_count
        if self.pooling_mode == "mean-v1":
            pooled = mean
        else:
            # Mean pooling is invariant to word order. Add a centered first positional moment so
            # sequences containing the same tokens in a different order map to different semantic
            # vectors while preserving the original embedding dimensionality and scale.
            positions = torch.arange(
                embeddings.shape[1], device=self.device, dtype=self.dtype
            ).unsqueeze(0)
            lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            midpoint = (lengths - 1.0) * 0.5
            radius = midpoint.clamp_min(1.0)
            centered = ((positions - midpoint) / radius) * attention_mask
            moment_denominator = centered.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
            directional = (
                embeddings * centered.unsqueeze(-1)
            ).sum(dim=1) / moment_denominator
            pooled = mean + self.ORDER_MOMENT_SCALE * directional
        result = pooled.unsqueeze(0)
        return result.detach() if detach else result


def stable_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def canonical_fact_text(
    item: FactItem | None = None,
    *,
    key: str | None = None,
    value: Any = None,
    source_type: str = "dataset",
    version: str | None = None,
) -> str:
    if item is not None:
        key = item.key
        value = item.value
        source_type = item.source_type
        version = item.version
    if key is None:
        raise ValueError("canonical fact text requires a key")
    return " | ".join(
        (
            f"fact={key}",
            f"source={source_type}",
            f"value={stable_text(value)}",
            f"version={version or ''}",
        )
    )


def canonical_source_text(item: SourceDescriptor | Mapping[str, Any]) -> str:
    if isinstance(item, SourceDescriptor):
        name = item.name
        description = item.description
        arguments = tuple(
            (argument.name, argument.kind, argument.required) for argument in item.arguments
        )
        cacheable = item.cacheable
        dynamic = item.dynamic
        streamable = item.streamable
        versioned = item.versioned
        accepts_partial_arguments = item.accepts_partial_arguments
        promote_results_to_fact = item.promote_results_to_fact
    else:
        name = str(item.get("name", ""))
        description = str(item.get("description", ""))
        arguments = tuple(
            (
                str(argument.get("name", "")),
                str(argument.get("kind", "any")),
                bool(argument.get("required", True)),
            )
            for argument in item.get("arguments", ())
        )
        cacheable = bool(item.get("cacheable", True))
        dynamic = bool(item.get("dynamic", False))
        streamable = bool(item.get("streamable", False))
        versioned = bool(item.get("versioned", False))
        accepts_partial_arguments = bool(item.get("accepts_partial_arguments", False))
        promote_results_to_fact = bool(item.get("promote_results_to_fact", False))
    argument_text = ",".join(
        f"{argument_name}:{kind}:{'required' if required else 'optional'}"
        for argument_name, kind, required in arguments
    )
    return " | ".join(
        (
            f"source={name}",
            f"description={description}",
            f"arguments={argument_text}",
            f"cacheable={cacheable}",
            f"dynamic={dynamic}",
            f"streamable={streamable}",
            f"versioned={versioned}",
            f"accepts_partial_arguments={accepts_partial_arguments}",
            f"promote_results_to_fact={promote_results_to_fact}",
        )
    )


def canonical_percept_text(
    *,
    source: str,
    value: Any,
    version: str | None,
    target_cells: tuple[ObjectRef, ...],
    target_display: tuple[ObjectRef, ...],
    anchors: tuple[Anchor, ...] = (),
) -> str:
    anchor_text = ",".join(anchor.canonical_key for anchor in anchors)
    cell_text = ",".join(target.identifier for target in target_cells)
    display_text = ",".join(
        f"{target.span[0]}:{target.span[1]}" for target in target_display if target.span is not None
    )
    return " | ".join(
        (
            f"source={source}",
            f"value={stable_text(value)}",
            f"version={version or ''}",
            f"anchors={anchor_text}",
            f"target_cells={cell_text}",
            f"target_display={display_text}",
        )
    )

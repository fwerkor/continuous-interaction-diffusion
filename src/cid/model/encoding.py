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
    """Tokenizer/embedding bridge shared by runtime and training tensorizers."""

    def __init__(self, adapter: ILLaDACIDAdapter, tokenizer: Any) -> None:
        self._embedding = adapter.input_embeddings
        self._output_device = self._embedding.weight.device
        self._output_dtype = self._embedding.weight.dtype
        self.tokenizer = tokenizer
        self.d_model = adapter.d_model

    @classmethod
    def from_frozen_snapshot(
        cls,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        embedding_device: torch.device | str | None = None,
    ) -> ILLaDATextEncoder:
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
        pooled = (embeddings * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
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

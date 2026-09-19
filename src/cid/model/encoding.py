from __future__ import annotations

import json
from collections import OrderedDict
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
    FROZEN_SNAPSHOT_FORMAT_VERSION = 1
    ORDER_MOMENT_SCALE = 0.25
    SOURCE_IDENTITY_SCALE = 0.25
    TOKEN_CACHE_SIZE = 8192
    DETACHED_TEXT_CACHE_SIZE = 8192
    DETACHED_BATCH_CACHE_SIZE = 16
    DETACHED_BATCH_CACHE_MAX_VECTORS = 512

    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        pooling_mode: str = "order-aware-v2",
    ) -> None:
        if pooling_mode not in {"mean-v1", "order-aware-v2"}:
            raise ValueError(f"unsupported semantic pooling mode: {pooling_mode}")
        self._embedding = adapter.input_embeddings
        self._output_device = self._embedding.weight.device
        self._output_dtype = self._embedding.weight.dtype
        self.tokenizer = tokenizer
        self.d_model = adapter.d_model
        self.pooling_mode = pooling_mode
        self._is_frozen_snapshot = False
        self._token_cache: OrderedDict[tuple[str, bool], Tensor] = OrderedDict()
        self._detached_text_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._detached_batch_cache: OrderedDict[tuple[str, ...], Tensor] = OrderedDict()
        self._detached_batch_cache_vectors = 0

    @classmethod
    def from_frozen_snapshot(
        cls,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        embedding_device: torch.device | str | None = None,
        pooling_mode: str = "order-aware-v2",
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
        snapshot._is_frozen_snapshot = True
        snapshot._token_cache = OrderedDict()
        snapshot._detached_text_cache = OrderedDict()
        snapshot._detached_batch_cache = OrderedDict()
        snapshot._detached_batch_cache_vectors = 0
        return snapshot

    @classmethod
    def from_frozen_snapshot_state(
        cls,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        state: Mapping[str, Any],
        *,
        device: torch.device | str,
        embedding_device: torch.device | str | None = None,
    ) -> ILLaDATextEncoder:
        if int(state.get("format_version", 0)) != cls.FROZEN_SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported frozen semantic embedding snapshot format")
        if int(state.get("encoding_version", 0)) != cls.ENCODING_VERSION:
            raise ValueError("frozen semantic embedding snapshot encoding version is incompatible")
        pooling_mode = str(state.get("pooling_mode", ""))
        if pooling_mode not in {"mean-v1", "order-aware-v2"}:
            raise ValueError("frozen semantic embedding snapshot pooling mode is invalid")
        weight = state.get("weight")
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise ValueError("frozen semantic embedding snapshot weight is invalid")
        expected_shape = (adapter.vocab_size, adapter.d_model)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                "frozen semantic embedding snapshot geometry does not match CID adapter"
            )
        if int(state.get("d_model", -1)) != adapter.d_model:
            raise ValueError("frozen semantic embedding snapshot width does not match CID adapter")

        snapshot = cls.__new__(cls)
        output_device = torch.device(device)
        storage_device = (
            torch.device(embedding_device) if embedding_device is not None else output_device
        )
        stored_weight = weight.detach().to(device=storage_device)
        snapshot._embedding = nn.Embedding.from_pretrained(stored_weight, freeze=True)
        snapshot._output_device = output_device
        snapshot._output_dtype = stored_weight.dtype
        snapshot.tokenizer = tokenizer
        snapshot.d_model = adapter.d_model
        snapshot.pooling_mode = pooling_mode
        snapshot._is_frozen_snapshot = True
        snapshot._token_cache = OrderedDict()
        snapshot._detached_text_cache = OrderedDict()
        snapshot._detached_batch_cache = OrderedDict()
        snapshot._detached_batch_cache_vectors = 0
        return snapshot

    @property
    def is_frozen_snapshot(self) -> bool:
        return self._is_frozen_snapshot

    def frozen_snapshot_state(self) -> dict[str, Any]:
        if not self.is_frozen_snapshot:
            raise ValueError("semantic encoder is not an independent frozen snapshot")
        return {
            "format_version": self.FROZEN_SNAPSHOT_FORMAT_VERSION,
            "encoding_version": self.ENCODING_VERSION,
            "pooling_mode": self.pooling_mode,
            "d_model": self.d_model,
            "weight": self._embedding.weight.detach().cpu(),
        }

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
        cache_key = (text, add_special_tokens)
        cached = self._token_cache.get(cache_key)
        if cached is not None:
            self._token_cache.move_to_end(cache_key)
            return cached.to(device=self.device)
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        self._remember(
            self._token_cache,
            cache_key,
            input_ids.detach().cpu(),
            self.TOKEN_CACHE_SIZE,
        )
        return input_ids.to(device=self.device)

    def encode_one(self, text: str, *, detach: bool = False) -> Tensor:
        return self.encode_texts((text,), detach=detach)[0, 0]

    def encode_texts(self, texts: tuple[str, ...], *, detach: bool = False) -> Tensor:
        if not texts:
            return torch.empty(
                (1, 0, self.d_model),
                device=self.device,
                dtype=self.dtype,
            )
        if detach and self._semantic_cache_safe():
            cached_batch = self._detached_batch_cache.get(texts)
            if cached_batch is not None:
                self._detached_batch_cache.move_to_end(texts)
                return cached_batch.clone()

            vectors: list[Tensor | None] = []
            missing_texts: list[str] = []
            missing_indices: list[int] = []
            for index, text in enumerate(texts):
                cached = self._detached_text_cache.get(text)
                if cached is None:
                    vectors.append(None)
                    missing_texts.append(text)
                    missing_indices.append(index)
                else:
                    self._detached_text_cache.move_to_end(text)
                    vectors.append(cached)
            if missing_texts:
                missing_vectors = self._encode_texts_uncached(tuple(missing_texts), detach=True)[0]
                for index, text, vector in zip(
                    missing_indices,
                    missing_texts,
                    missing_vectors.unbind(0),
                    strict=True,
                ):
                    vector = vector.detach()
                    self._remember(
                        self._detached_text_cache,
                        text,
                        vector,
                        self.DETACHED_TEXT_CACHE_SIZE,
                    )
                    vectors[index] = vector
            if any(vector is None for vector in vectors):
                raise RuntimeError("detached semantic cache failed to materialize an encoded text")
            result = torch.stack(
                tuple(vector for vector in vectors if vector is not None),
                dim=0,
            ).unsqueeze(0)
            self._remember_detached_batch(texts, result)
            return result
        return self._encode_texts_uncached(texts, detach=detach)

    def _encode_texts_uncached(self, texts: tuple[str, ...], *, detach: bool) -> Tensor:
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

    def _semantic_cache_safe(self) -> bool:
        return self.is_frozen_snapshot or not self._embedding.weight.requires_grad

    @staticmethod
    def _remember(cache: OrderedDict, key: Any, value: Tensor, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _remember_detached_batch(
        self,
        key: tuple[str, ...],
        value: Tensor,
    ) -> None:
        vector_count = int(value.shape[1])
        if vector_count > self.DETACHED_BATCH_CACHE_MAX_VECTORS:
            return

        previous = self._detached_batch_cache.pop(key, None)
        if previous is not None:
            self._detached_batch_cache_vectors -= int(previous.shape[1])

        self._detached_batch_cache[key] = value.detach().clone()
        self._detached_batch_cache.move_to_end(key)
        self._detached_batch_cache_vectors += vector_count

        while (
            len(self._detached_batch_cache) > self.DETACHED_BATCH_CACHE_SIZE
            or self._detached_batch_cache_vectors > self.DETACHED_BATCH_CACHE_MAX_VECTORS
        ):
            _, evicted = self._detached_batch_cache.popitem(last=False)
            self._detached_batch_cache_vectors -= int(evicted.shape[1])

    def encode_source_descriptors(
        self,
        sources: tuple[SourceDescriptor | Mapping[str, Any], ...],
        *,
        detach: bool = False,
    ) -> Tensor:
        if not sources:
            return self.encode_texts((), detach=detach)
        descriptor_memory = self.encode_texts(
            tuple(canonical_source_text(source) for source in sources),
            detach=detach,
        )
        identity_memory = self.encode_texts(
            tuple(source_descriptor_name(source) for source in sources),
            detach=detach,
        )
        return descriptor_memory + self.SOURCE_IDENTITY_SCALE * identity_memory


def stable_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def source_descriptor_name(item: SourceDescriptor | Mapping[str, Any]) -> str:
    return item.name if isinstance(item, SourceDescriptor) else str(item.get("name", ""))


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
        name = source_descriptor_name(item)
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
        name = source_descriptor_name(item)
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

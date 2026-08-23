from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor, nn

from cid.model.illada import ILLaDACIDAdapter


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
            adapter.input_embeddings.weight.detach()
            .to(device=storage_device, dtype=dtype)
            .clone()
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

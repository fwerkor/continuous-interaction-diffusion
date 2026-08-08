from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from cid.model.illada import ILLaDACIDAdapter


class ILLaDATextEncoder:
    """Tokenizer/embedding bridge shared by runtime and training tensorizers."""

    def __init__(self, adapter: ILLaDACIDAdapter, tokenizer: Any) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer

    def tokenize(self, text: str, *, add_special_tokens: bool) -> Tensor:
        weight = self.adapter.input_embeddings.weight
        if not text:
            return torch.empty((1, 0), dtype=torch.long, device=weight.device)
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device=weight.device)

    def encode_one(self, text: str, *, detach: bool = False) -> Tensor:
        return self.encode_texts((text,), detach=detach)[0, 0]

    def encode_texts(self, texts: tuple[str, ...], *, detach: bool = False) -> Tensor:
        weight = self.adapter.input_embeddings.weight
        if not texts:
            return weight.new_empty((1, 0, self.adapter.d_model))
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device=weight.device)
        attention_mask = encoded["attention_mask"].to(device=weight.device, dtype=weight.dtype)
        if detach:
            with torch.no_grad():
                embeddings = self.adapter.input_embeddings(input_ids)
        else:
            embeddings = self.adapter.input_embeddings(input_ids)
        weights = attention_mask.unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        result = pooled.unsqueeze(0)
        return result.detach() if detach else result


def stable_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

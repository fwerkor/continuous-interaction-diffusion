from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cid.model.ar import bidirectional_ar_hidden_states, prepare_ar_backbone_for_cid


class ARDiffusionBase(nn.Module):
    """Turn a causal decoder checkpoint into a bidirectional masked-token denoiser."""

    def __init__(self, backbone: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_token_id = prepare_ar_backbone_for_cid(backbone, tokenizer)
        decoder = backbone.get_decoder()
        if getattr(decoder, "layers", None) is None:
            raise ValueError("diffusion base requires a decoder exposing transformer layers")

    def forward(self, input_ids: Tensor) -> Tensor:
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        hidden = bidirectional_ar_hidden_states(
            self.backbone.get_decoder(),
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        output_embeddings = self.backbone.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("diffusion base backbone does not expose an LM head")
        return output_embeddings(hidden)


@dataclass(frozen=True)
class PackedCorpusSource:
    name: str
    path: Path
    tokens: int
    weight: float


class PackedCorpusSampler:
    """Random fixed-length sampling over pre-tokenized flat uint32 token streams."""

    def __init__(
        self,
        sources: tuple[PackedCorpusSource, ...],
        *,
        sequence_length: int,
        seed: int,
    ) -> None:
        if not sources:
            raise ValueError("packed corpus requires at least one source")
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        self.sources = sources
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)
        weights = np.asarray([source.weight for source in sources], dtype=np.float64)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("corpus source weights must be finite and positive")
        self.weights = weights / weights.sum()
        self.arrays = tuple(np.memmap(source.path, mode="r", dtype=np.uint32) for source in sources)
        for source, array in zip(sources, self.arrays, strict=True):
            if len(array) < sequence_length:
                raise ValueError(f"corpus source {source.name!r} is shorter than one sequence")

    @classmethod
    def from_manifest(cls, path: str | Path, *, seed: int) -> PackedCorpusSampler:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        sources = tuple(
            PackedCorpusSource(
                name=str(item["name"]),
                path=(base / str(item["file"])).resolve(),
                tokens=int(item["tokens"]),
                weight=float(item["weight"]),
            )
            for item in manifest["sources"]
        )
        return cls(sources, sequence_length=int(manifest["sequence_length"]), seed=seed)

    def sample(self, batch_size: int) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        rows: list[np.ndarray] = []
        source_indices = self.rng.choice(len(self.sources), size=batch_size, p=self.weights)
        for source_index in source_indices:
            array = self.arrays[int(source_index)]
            max_start = len(array) - self.sequence_length
            start = int(self.rng.integers(0, max_start + 1))
            rows.append(np.asarray(array[start : start + self.sequence_length], dtype=np.int64))
        return torch.from_numpy(np.stack(rows, axis=0))


def masked_diffusion_loss(
    model: nn.Module,
    clean_ids: Tensor,
    *,
    mask_token_id: int,
    min_mask_ratio: float = 1e-3,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Monte-Carlo LLaDA-style masked diffusion objective."""

    if not 0.0 < min_mask_ratio <= 1.0:
        raise ValueError("min_mask_ratio must be in (0, 1]")
    batch_size, sequence_length = clean_ids.shape
    random_ratio = torch.rand(
        (batch_size, 1),
        device=clean_ids.device,
        generator=generator,
        dtype=torch.float32,
    )
    mask_ratio = min_mask_ratio + (1.0 - min_mask_ratio) * random_ratio
    masked = torch.rand(
        clean_ids.shape,
        device=clean_ids.device,
        generator=generator,
        dtype=torch.float32,
    ) < mask_ratio
    # The theoretical objective has non-zero masking probability, but a finite sequence
    # can still sample no masks at tiny t. Force one position so every row contributes.
    empty_rows = ~masked.any(dim=1)
    if bool(empty_rows.any()):
        positions = torch.randint(
            sequence_length,
            (int(empty_rows.sum().item()),),
            device=clean_ids.device,
            generator=generator,
        )
        masked[empty_rows, positions] = True

    corrupted = clean_ids.masked_fill(masked, int(mask_token_id))
    logits = model(corrupted)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        clean_ids.reshape(-1),
        reduction="none",
    ).view_as(clean_ids)
    weighted = token_loss * masked.to(token_loss.dtype) / mask_ratio
    loss = weighted.sum() / float(batch_size * sequence_length)
    masked_fraction = masked.float().mean().detach()
    mean_ratio = mask_ratio.mean().detach()
    return loss, masked_fraction, mean_ratio


def stage0_learning_rate(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmup_ratio: float,
    min_lr_ratio: float,
    schedule: str = "constant",
) -> float:
    if total_steps <= 0:
        raise ValueError("total steps must be positive")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("Stage 0 LR schedule must be 'constant' or 'cosine'")
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if schedule == "constant" or total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def fsdp_accumulation_context(model: nn.Module, *, synchronize: bool) -> Any:
    return nullcontext() if synchronize else model.no_sync()

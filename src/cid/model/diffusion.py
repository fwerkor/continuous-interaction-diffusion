from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DisplayCorruption:
    token_ids: Tensor
    labels: Tensor
    noise: Tensor
    masked: Tensor


@dataclass(frozen=True, slots=True)
class ThoughtCorruption:
    semantic: Tensor
    noise: Tensor
    epsilon: Tensor


class CIDDiffusionScheduler:
    """Training corruption and iterative reveal utilities for CID T/Y state."""

    def __init__(self, mask_token_id: int) -> None:
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        self.mask_token_id = mask_token_id

    def corrupt_display(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        *,
        eligible_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DisplayCorruption:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        probabilities = self._token_probabilities(token_ids, timesteps)
        if eligible_mask is None:
            eligible_mask = torch.ones_like(token_ids, dtype=torch.bool)
        elif eligible_mask.shape != token_ids.shape:
            raise ValueError("eligible_mask must match token_ids")

        random = torch.rand(
            token_ids.shape,
            device=token_ids.device,
            generator=generator,
        )
        masked = (random < probabilities) & eligible_mask.bool()
        masked = self._ensure_training_mask(masked, eligible_mask.bool(), timesteps)
        corrupted = token_ids.clone()
        corrupted[masked] = self.mask_token_id
        labels = token_ids.clone()
        labels[~masked] = -100
        return DisplayCorruption(
            token_ids=corrupted,
            labels=labels,
            noise=probabilities.unsqueeze(-1),
            masked=masked,
        )

    def corrupt_thought(
        self,
        semantic: Tensor,
        timesteps: Tensor,
        occupancy: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ThoughtCorruption:
        if semantic.ndim != 3:
            raise ValueError("semantic must have shape [batch, slots, hidden]")
        if occupancy.shape != (*semantic.shape[:2], 1):
            raise ValueError("occupancy must have shape [batch, slots, 1]")
        timestep = self._batch_timesteps(semantic.shape[0], timesteps, semantic.device)
        alpha = (
            torch.cos(timestep * (math.pi / 2))
            .square()
            .to(dtype=semantic.dtype)
            .view(-1, 1, 1)
        )
        epsilon = torch.randn(
            semantic.shape,
            dtype=semantic.dtype,
            device=semantic.device,
            generator=generator,
        )
        corrupted = alpha.sqrt() * semantic + (1.0 - alpha).sqrt() * epsilon
        occupied = occupancy.bool()
        corrupted = torch.where(occupied, corrupted, torch.zeros_like(corrupted))
        epsilon = torch.where(occupied, epsilon, torch.zeros_like(epsilon))
        local_noise = (
            timestep.to(dtype=semantic.dtype)
            .view(-1, 1, 1)
            .expand(-1, semantic.shape[1], -1)
        )
        local_noise = torch.where(occupied, local_noise, torch.zeros_like(local_noise))
        return ThoughtCorruption(semantic=corrupted, noise=local_noise, epsilon=epsilon)

    def reveal_display(
        self,
        token_ids: Tensor,
        logits: Tensor,
        *,
        reveal_fraction: float,
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if logits.shape[:2] != token_ids.shape:
            raise ValueError("logits must have shape [batch, tokens, vocab]")
        if not 0.0 <= reveal_fraction <= 1.0:
            raise ValueError("reveal_fraction must be in [0, 1]")
        if reveal_fraction == 0.0:
            return token_ids.clone()

        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        result = token_ids.clone()
        for batch_index in range(token_ids.shape[0]):
            masked_positions = torch.nonzero(
                token_ids[batch_index] == self.mask_token_id,
                as_tuple=False,
            ).flatten()
            if masked_positions.numel() == 0:
                continue
            reveal_count = math.ceil(masked_positions.numel() * reveal_fraction)
            ranked = masked_positions[
                confidence[batch_index, masked_positions].argsort(descending=True)
            ]
            selected = ranked[:reveal_count]
            result[batch_index, selected] = predicted[batch_index, selected]
        return result

    @staticmethod
    def _batch_timesteps(batch_size: int, timesteps: Tensor, device: torch.device) -> Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch]")
        timestep = timesteps.to(device=device, dtype=torch.float32)
        if bool(((timestep < 0.0) | (timestep > 1.0)).any()):
            raise ValueError("timesteps must be in [0, 1]")
        return timestep

    def _token_probabilities(self, token_ids: Tensor, timesteps: Tensor) -> Tensor:
        timestep = self._batch_timesteps(token_ids.shape[0], timesteps, token_ids.device)
        return timestep[:, None].expand_as(token_ids)

    @staticmethod
    def _ensure_training_mask(masked: Tensor, eligible: Tensor, timesteps: Tensor) -> Tensor:
        result = masked.clone()
        for batch_index in range(masked.shape[0]):
            if float(timesteps[batch_index]) <= 0.0 or bool(result[batch_index].any()):
                continue
            eligible_positions = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
            if eligible_positions.numel():
                result[batch_index, eligible_positions[0]] = True
        return result

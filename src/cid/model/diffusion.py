from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def denoising_reveal_fraction(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("denoising step must be non-negative")
    remaining = max(1, total_steps - min(step, total_steps - 1))
    return 1.0 / remaining


def denoising_noise_level(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("denoising step must be non-negative")
    remaining = max(1, total_steps - min(step, total_steps - 1))
    return remaining / total_steps


@dataclass(frozen=True, slots=True)
class DisplayCorruption:
    token_ids: Tensor
    labels: Tensor
    noise: Tensor
    masked: Tensor
    replaced: Tensor


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
        vocab_size: int | None = None,
        replacement_fraction: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> DisplayCorruption:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        probabilities = self._token_probabilities(token_ids, timesteps)
        if eligible_mask is None:
            eligible_mask = torch.ones_like(token_ids, dtype=torch.bool)
        elif eligible_mask.shape != token_ids.shape:
            raise ValueError("eligible_mask must match token_ids")
        if not 0.0 <= replacement_fraction <= 1.0:
            raise ValueError("replacement_fraction must be in [0, 1]")
        if replacement_fraction and (vocab_size is None or vocab_size < 3):
            raise ValueError("visible replacement corruption requires vocab_size >= 3")

        random = torch.rand(
            token_ids.shape,
            device=token_ids.device,
            generator=generator,
        )
        corrupted_positions = (random < probabilities) & eligible_mask.bool()
        corrupted_positions = self._ensure_training_mask(
            corrupted_positions,
            eligible_mask.bool(),
            timesteps,
        )
        if replacement_fraction:
            replacement_draw = torch.rand(
                token_ids.shape,
                device=token_ids.device,
                generator=generator,
            )
            replaced = corrupted_positions & (replacement_draw < replacement_fraction)
        else:
            replaced = torch.zeros_like(corrupted_positions)
        masked = corrupted_positions & ~replaced
        corrupted = token_ids.clone()
        corrupted[masked] = self.mask_token_id
        if bool(replaced.any()):
            replacements = self._replacement_tokens(
                token_ids,
                vocab_size=int(vocab_size),
                generator=generator,
            )
            corrupted[replaced] = replacements[replaced]
        labels = token_ids.clone()
        labels[~corrupted_positions] = -100
        return DisplayCorruption(
            token_ids=corrupted,
            labels=labels,
            noise=probabilities.unsqueeze(-1),
            masked=masked,
            replaced=replaced,
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
        timestep = self._thought_timesteps(
            semantic.shape[0], semantic.shape[1], timesteps, semantic.device
        )
        alpha = torch.cos(timestep * (math.pi / 2)).square().to(dtype=semantic.dtype).unsqueeze(-1)
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
        local_noise = timestep.to(dtype=semantic.dtype).unsqueeze(-1)
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

        return self.refine_display(
            token_ids,
            logits,
            reveal_fraction=reveal_fraction,
            revision_fraction=0.0,
            revision_margin=1.0,
        )

    def refine_display(
        self,
        token_ids: Tensor,
        logits: Tensor,
        *,
        reveal_fraction: float,
        revision_fraction: float,
        revision_margin: float,
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if logits.shape[:2] != token_ids.shape:
            raise ValueError("logits must have shape [batch, tokens, vocab]")
        if not 0.0 <= reveal_fraction <= 1.0:
            raise ValueError("reveal_fraction must be in [0, 1]")
        if not 0.0 <= revision_fraction <= 1.0:
            raise ValueError("revision_fraction must be in [0, 1]")
        if revision_margin < 0.0:
            raise ValueError("revision_margin must be non-negative")

        confidence = torch.empty(token_ids.shape, dtype=torch.float32, device=logits.device)
        predicted = torch.empty(token_ids.shape, dtype=torch.long, device=logits.device)
        current_confidence = torch.empty(
            token_ids.shape, dtype=torch.float32, device=logits.device
        )
        token_chunk_size = 32
        for start in range(0, token_ids.shape[1], token_chunk_size):
            stop = min(token_ids.shape[1], start + token_chunk_size)
            filtered_logits = logits[:, start:stop].float().clone()
            if self.mask_token_id < filtered_logits.shape[-1]:
                filtered_logits[..., self.mask_token_id] = torch.finfo(
                    filtered_logits.dtype
                ).min
            probabilities = torch.softmax(filtered_logits, dim=-1)
            chunk_confidence, chunk_predicted = probabilities.max(dim=-1)
            confidence[:, start:stop] = chunk_confidence
            predicted[:, start:stop] = chunk_predicted
            current_ids = token_ids[:, start:stop].unsqueeze(-1)
            current_confidence[:, start:stop] = probabilities.gather(
                dim=-1,
                index=current_ids,
            ).squeeze(-1)
        result = token_ids.clone()
        for batch_index in range(token_ids.shape[0]):
            masked_positions = torch.nonzero(
                token_ids[batch_index] == self.mask_token_id,
                as_tuple=False,
            ).flatten()
            if masked_positions.numel() and reveal_fraction:
                reveal_count = math.ceil(masked_positions.numel() * reveal_fraction)
                ranked = masked_positions[
                    confidence[batch_index, masked_positions].argsort(descending=True)
                ]
                selected = ranked[:reveal_count]
                result[batch_index, selected] = predicted[batch_index, selected]

            if revision_fraction == 0.0:
                continue
            visible_positions = torch.nonzero(
                token_ids[batch_index] != self.mask_token_id,
                as_tuple=False,
            ).flatten()
            if visible_positions.numel() == 0:
                continue
            current_ids = token_ids[batch_index, visible_positions]
            visible_confidence = current_confidence[batch_index, visible_positions]
            gains = confidence[batch_index, visible_positions] - visible_confidence
            candidates = (
                (predicted[batch_index, visible_positions] != current_ids)
                & (gains >= revision_margin)
            )
            candidate_positions = visible_positions[candidates]
            if candidate_positions.numel() == 0:
                continue
            candidate_gains = gains[candidates]
            revision_count = min(
                candidate_positions.numel(),
                math.ceil(visible_positions.numel() * revision_fraction),
            )
            ranked = candidate_positions[candidate_gains.argsort(descending=True)]
            selected = ranked[:revision_count]
            result[batch_index, selected] = predicted[batch_index, selected]
        return result

    def _replacement_tokens(
        self,
        token_ids: Tensor,
        *,
        vocab_size: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        offsets = torch.randint(
            1,
            vocab_size,
            token_ids.shape,
            device=token_ids.device,
            generator=generator,
        )
        replacements = (token_ids + offsets) % vocab_size
        mask_collision = replacements == self.mask_token_id
        replacements[mask_collision] = (replacements[mask_collision] + 1) % vocab_size
        original_collision = replacements == token_ids
        replacements[original_collision] = (replacements[original_collision] + 1) % vocab_size
        return replacements

    @staticmethod
    def _thought_timesteps(
        batch_size: int, thought_slots: int, timesteps: Tensor, device: torch.device
    ) -> Tensor:
        if timesteps.ndim == 1:
            timestep = CIDDiffusionScheduler._batch_timesteps(batch_size, timesteps, device)
            return timestep[:, None].expand(-1, thought_slots)
        if timesteps.ndim != 2 or timesteps.shape != (batch_size, thought_slots):
            raise ValueError("thought timesteps must have shape [batch] or [batch, slots]")
        timestep = timesteps.to(device=device, dtype=torch.float32)
        if bool(((timestep < 0.0) | (timestep > 1.0)).any()):
            raise ValueError("timesteps must be in [0, 1]")
        return timestep

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

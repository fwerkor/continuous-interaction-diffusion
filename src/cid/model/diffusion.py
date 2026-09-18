from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import torch
from torch import Tensor

try:
    import cid_engine as _cid_engine
except ModuleNotFoundError as exc:
    if exc.name != "cid_engine":
        raise
    _cid_engine = None

_MAX_STRUCTURAL_EDIT_TOKENS = 32


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

    def __init__(self, mask_token_id: int, eos_token_id: int | None = None) -> None:
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        if eos_token_id is not None and eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative when set")
        if eos_token_id == mask_token_id:
            raise ValueError("EOS and MASK token IDs must differ")
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id

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

        if (
            _cid_engine is not None
            and _cid_engine.CUDA_BACKEND_BUILT
            and logits.is_cuda
        ):
            confidence, predicted, current_confidence = (
                _cid_engine.display_token_statistics(token_ids, logits)
            )
        else:
            confidence = torch.empty(
                token_ids.shape, dtype=torch.float32, device=logits.device
            )
            predicted = torch.empty(token_ids.shape, dtype=torch.long, device=logits.device)
            current_confidence = torch.empty(
                token_ids.shape, dtype=torch.float32, device=logits.device
            )
            token_chunk_size = 32
            for start in range(0, token_ids.shape[1], token_chunk_size):
                stop = min(token_ids.shape[1], start + token_chunk_size)
                filtered_logits = logits[:, start:stop].float().clone()
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
            eos_position: int | None = None
            if self.eos_token_id is not None:
                current_eos = torch.nonzero(
                    token_ids[batch_index] == self.eos_token_id,
                    as_tuple=False,
                ).flatten()
                if current_eos.numel():
                    eos_position = int(current_eos[0])

            structural_edit = None
            if eos_position is not None and revision_fraction:
                structural_edit = self._structural_display_edit(
                    token_ids[batch_index],
                    predicted[batch_index],
                    confidence[batch_index] - current_confidence[batch_index],
                    eos_position=eos_position,
                    revision_fraction=revision_fraction,
                    revision_margin=revision_margin,
                )
            if structural_edit is not None:
                start, delete_count, inserted = structural_edit
                active = token_ids[batch_index, : eos_position + 1].tolist()
                revised = [
                    *active[:start],
                    *inserted,
                    *active[start + delete_count :],
                ]
                result[batch_index].fill_(self.mask_token_id)
                result[batch_index, : len(revised)] = torch.tensor(
                    revised,
                    dtype=result.dtype,
                    device=result.device,
                )
                continue

            active_content_stop = (
                eos_position if eos_position is not None else token_ids.shape[1]
            )
            masked_positions = torch.nonzero(
                token_ids[batch_index, :active_content_stop] == self.mask_token_id,
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
                token_ids[batch_index, :active_content_stop] != self.mask_token_id,
                as_tuple=False,
            ).flatten()
            if visible_positions.numel():
                current_ids = token_ids[batch_index, visible_positions]
                visible_confidence = current_confidence[batch_index, visible_positions]
                gains = confidence[batch_index, visible_positions] - visible_confidence
                candidates = (predicted[batch_index, visible_positions] != current_ids) & (
                    gains >= revision_margin
                )
                candidate_positions = visible_positions[candidates]
                if candidate_positions.numel():
                    candidate_gains = gains[candidates]
                    revision_count = min(
                        candidate_positions.numel(),
                        math.ceil(visible_positions.numel() * revision_fraction),
                    )
                    ranked = candidate_positions[candidate_gains.argsort(descending=True)]
                    selected = ranked[:revision_count]
                    result[batch_index, selected] = predicted[batch_index, selected]

            # Positions after EOS are latent capacity, not unresolved answer tokens.  The
            # boundary may still move to the right, but only when the model explicitly
            # prefers replacing the current EOS and only by a bounded denoising budget.
            # This prevents one refinement step from revealing an arbitrary canvas tail.
            if eos_position is not None and eos_position + 1 < token_ids.shape[1]:
                eos_prediction = int(predicted[batch_index, eos_position])
                eos_gain = float(
                    confidence[batch_index, eos_position]
                    - current_confidence[batch_index, eos_position]
                )
                if eos_prediction != self.eos_token_id and eos_gain >= revision_margin:
                    expansion_budget = max(
                        1,
                        math.ceil(max(1, eos_position + 1) * reveal_fraction),
                    )
                    new_eos = min(
                        token_ids.shape[1] - 1,
                        eos_position + expansion_budget,
                    )
                    predicted_tail_eos = torch.nonzero(
                        predicted[batch_index, eos_position + 1 : new_eos + 1]
                        == self.eos_token_id,
                        as_tuple=False,
                    ).flatten()
                    if predicted_tail_eos.numel():
                        new_eos = eos_position + 1 + int(predicted_tail_eos[0])
                    result[batch_index, eos_position:new_eos] = predicted[
                        batch_index, eos_position:new_eos
                    ]
                    result[batch_index, new_eos] = self.eos_token_id

        if self.eos_token_id is not None:
            for batch_index in range(result.shape[0]):
                eos_positions = torch.nonzero(
                    result[batch_index] == self.eos_token_id, as_tuple=False
                ).flatten()
                if eos_positions.numel():
                    result[batch_index, int(eos_positions[0]) + 1 :] = self.mask_token_id
        return result

    def _structural_display_edit(
        self,
        token_ids: Tensor,
        predicted: Tensor,
        gains: Tensor,
        *,
        eos_position: int,
        revision_fraction: float,
        revision_margin: float,
    ) -> tuple[int, int, tuple[int, ...]] | None:
        """Decode one local insertion/deletion from shifted suffix predictions.

        Display logits are position-wise, so a middle insertion otherwise requires the model to
        rewrite every later token one slot to the right. When the prediction already contains a
        short shifted copy of the existing suffix, that copy is enough evidence for the runtime to
        perform the splice and preserve the suffix itself. A multi-token anchor keeps an ordinary
        same-position replacement from being reinterpreted as an insertion.
        """

        if eos_position <= 0:
            return None
        active = token_ids[: eos_position + 1].tolist()
        proposal = predicted.detach().tolist()
        gain_values = gains[:eos_position].detach().tolist()
        edit_budget = min(
            _MAX_STRUCTURAL_EDIT_TOKENS,
            max(1, math.ceil(eos_position * revision_fraction)),
        )
        capacity = int(token_ids.shape[0])
        candidates: list[
            tuple[tuple[int, float, int, int], int, int, tuple[int, ...]]
        ] = []

        def matched_prefix(left: list[int], right: list[int]) -> int:
            count = 0
            for left_token, right_token in zip(left, right, strict=False):
                if left_token != right_token:
                    break
                count += 1
            return count

        def anchor_positions(sequence: list[int]) -> dict[int, dict[tuple[int, ...], list[int]]]:
            result: dict[int, dict[tuple[int, ...], list[int]]] = {2: {}, 3: {}}
            for length in (2, 3):
                for position in range(len(sequence) - length + 1):
                    key = tuple(sequence[position : position + length])
                    result[length].setdefault(key, []).append(position)
            return result

        def nearest_after(positions: list[int], start: int, maximum: int) -> int | None:
            index = bisect_right(positions, start)
            if index >= len(positions) or positions[index] > maximum:
                return None
            return positions[index]

        def add_candidate(
            *,
            start: int,
            delete_count: int,
            inserted: tuple[int, ...],
            anchor: int,
        ) -> None:
            if anchor < 2 or gain_values[start] < revision_margin:
                return
            net_growth = len(inserted) - delete_count
            if eos_position + 1 + net_growth > capacity:
                return
            score = (anchor, float(gain_values[start]), -abs(net_growth), start)
            candidates.append((score, start, delete_count, inserted))

        changed_starts = [
            start
            for start in range(eos_position)
            if proposal[start] != active[start] and gain_values[start] >= revision_margin
        ]
        if not changed_starts:
            return None

        active_anchors = anchor_positions(active)
        proposal_anchors = anchor_positions(proposal)
        for start in changed_starts:
            suffix_length = len(active) - start
            anchor_length = min(3, suffix_length)
            insertion_key = tuple(active[start : start + anchor_length])
            insertion_anchor_start = nearest_after(
                proposal_anchors[anchor_length].get(insertion_key, []),
                start,
                start + edit_budget,
            )
            if insertion_anchor_start is not None:
                inserted = tuple(proposal[start:insertion_anchor_start])
                if inserted and not any(
                    token == self.mask_token_id or token == self.eos_token_id
                    for token in inserted
                ):
                    add_candidate(
                        start=start,
                        delete_count=0,
                        inserted=inserted,
                        anchor=matched_prefix(
                            proposal[insertion_anchor_start:],
                            active[start:],
                        ),
                    )

            anchor_length = min(3, len(proposal) - start)
            if anchor_length < 2:
                continue
            deletion_key = tuple(proposal[start : start + anchor_length])
            deletion_suffix_start = nearest_after(
                active_anchors[anchor_length].get(deletion_key, []),
                start,
                min(eos_position - 1, start + edit_budget),
            )
            if deletion_suffix_start is not None:
                add_candidate(
                    start=start,
                    delete_count=deletion_suffix_start - start,
                    inserted=(),
                    anchor=matched_prefix(
                        proposal[start:],
                        active[deletion_suffix_start:],
                    ),
                )

        if not candidates:
            return None
        _, start, delete_count, inserted = max(candidates, key=lambda item: item[0])
        return start, delete_count, inserted

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
        for _ in range(3):
            forbidden = (replacements == self.mask_token_id) | (replacements == token_ids)
            if self.eos_token_id is not None:
                forbidden |= replacements == self.eos_token_id
            if not bool(forbidden.any()):
                break
            replacements[forbidden] = (replacements[forbidden] + 1) % vocab_size
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

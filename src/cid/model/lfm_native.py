from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class LFMNativeDisplayDecoder:
    """Native confidence reveal with a private CID EOS boundary.

    ``max_reveals_per_step`` defaults to 2: measured on 31 prompts it halves the
    forward-pass count (1.15 -> 0.59 forwards per output token) with unchanged
    output length and zero repetition loops, while 4+ degrades conditional
    independence between simultaneously revealed positions.

    The physical last Display cell is a CID-only sentinel, excluded from the
    native model forward. All other cells match LiquidAI's masked-LM canvas.
    Native answer markup stays internal; the external CID Display uses EOS.
    """

    def __init__(self, tokenizer: Any, *, mask_id: int, eos_id: int) -> None:
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.vocab_size = len(tokenizer)
        self.special_ids = tuple(
            int(token) for token in tokenizer.all_special_ids
            if token != eos_id and 0 <= int(token) < self.vocab_size
        )
        self.stop_ids = tuple(
            int(t) for t in tokenizer.encode("[/Answer]", add_special_tokens=False)
        )
        if not self.stop_ids:
            raise ValueError("the native LFM tokenizer must encode its answer terminator")
        if self.mask_id == self.eos_id:
            raise ValueError("MASK and EOS must be distinct")
        if any(token >= self.vocab_size for token in self.stop_ids):
            raise ValueError("native answer terminator lies outside tokenizer vocabulary")

    def _stop_at_native_marker(self, row: Tensor) -> None:
        # The marker can materialize out of order during diffusion. Commit it
        # only after the entire visible prefix is resolved.
        tokens = row.tolist()
        for pos in range(len(tokens) - len(self.stop_ids)):
            if (
                tuple(tokens[pos:pos + len(self.stop_ids)]) == self.stop_ids
                and self.mask_id not in tokens[:pos]
            ):
                row[pos:-1] = self.mask_id
                row[pos] = self.eos_id
                break

    def refine(
        self,
        display: Tensor,
        logits: Tensor,
        *,
        repetition_score_multiplier: float = 1.0,
        max_reveals_per_step: int = 2,
    ) -> Tensor:
        if not 0.0 < repetition_score_multiplier <= 1.0:
            raise ValueError("repetition_score_multiplier must be in (0, 1]")
        if max_reveals_per_step < 1:
            raise ValueError("max_reveals_per_step must be positive")
        if display.ndim != 2 or logits.ndim != 3:
            raise ValueError("expected Display [B,L] and logits [B,L,V]")
        if display.shape != logits.shape[:2] or logits.shape[-1] < self.vocab_size:
            raise ValueError("native logits and Display shapes differ")
        if not bool(display[:, -1].eq(self.eos_id).all()):
            raise ValueError("native decoder requires a terminal CID EOS sentinel")
        result = display.clone()
        for b in range(display.shape[0]):
            row = result[b]
            boundary = torch.nonzero(row[:-1].eq(self.eos_id), as_tuple=False).flatten()
            active_stop = int(boundary[0]) if boundary.numel() else row.numel() - 1
            # A reopened token before an accepted EOS remains editable. Only
            # a fully resolved prefix is complete.
            positions = torch.nonzero(row[:active_stop].eq(self.mask_id), as_tuple=False).flatten()
            if not positions.numel():
                self._stop_at_native_marker(row)
                continue
            # Match LiquidAI's reference: mask out beyond-tokenizer vocabulary,
            # allow EOS, suppress every other registered special token.
            candidates = logits[b, positions, :self.vocab_size].float().clone()
            if self.special_ids:
                candidates[:, self.special_ids] = -torch.inf
            if repetition_score_multiplier < 1.0:
                # Nonmonotonic reveal: only discourage overused token types,
                # regardless of the order in which their positions were filled.
                visible = row[:active_stop]
                visible = visible[(visible != self.mask_id) & (visible != self.eos_id)]
                if visible.numel():
                    counts = torch.bincount(visible, minlength=self.vocab_size)
                    repeated = (counts - 1).clamp_min(0).clamp_max(5)
                    candidates += repeated.to(candidates.dtype)[None] * torch.log(
                        candidates.new_tensor(repetition_score_multiplier)
                    )
            probs = candidates.softmax(dim=-1)
            confidences, guesses = probs.max(dim=-1)
            # Reveal up to ``max_reveals_per_step`` positions per model forward,
            # highest native confidence first. The reference decoder reveals
            # exactly one, which forces a forward pass per generated token. This
            # budget is a pure throughput knob: it never changes the forward
            # graph, so neutral-state parity with the frozen backbone is
            # unaffected (only the confidence ranking is consulted).
            budget = min(int(max_reveals_per_step), int(confidences.numel()))
            for slot in torch.argsort(confidences, descending=True)[:budget].tolist():
                position = int(positions[slot])
                if int(row[position]) != self.mask_id:
                    continue
                # Do not fill past a boundary accepted earlier in this call, so
                # a batched reveal can never resurrect MASKs after a real EOS.
                boundary = torch.nonzero(row[:-1].eq(self.eos_id), as_tuple=False).flatten()
                if boundary.numel() and position > int(boundary[0]):
                    continue
                guess = int(guesses[slot])
                if guess == self.eos_id:
                    # The pretrained model normally terminates with lexical
                    # [/Answer]. Treat an explicit EOS as a stop only when its
                    # preceding Display prefix has completely resolved.
                    if not bool(row[:position].eq(self.mask_id).any()):
                        row[position:-1] = self.mask_id
                        row[position] = self.eos_id
                        continue
                    # Avoid an early structural CID EOS while prefix MASKs remain.
                    candidates[slot, self.eos_id] = -torch.inf
                    guess = int(candidates[slot].argmax())
                row[position] = guess
                self._stop_at_native_marker(row)
        return result

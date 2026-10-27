"""Evidence-aware, editable confidence anchors for iterative Display decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cid.model.diffusion import CIDDiffusionScheduler


@dataclass(frozen=True, slots=True)
class DisplayRevisionConfig:
    """Runtime decoding parameters; anchors never turn Display into immutable text."""

    burst_probability: float = 0.95
    anchor_probability: float = 0.98
    correction_probability: float = 0.98
    correction_gain: float = 0.25
    correction_steps: int = 8
    max_burst_tokens: int = 16

    def __post_init__(self) -> None:
        for field in ("burst_probability", "anchor_probability", "correction_probability"):
            if not 0.0 < getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be in (0, 1]")
        if not 0.0 <= self.correction_gain <= 1.0:
            raise ValueError("correction_gain must be in [0, 1]")
        if self.correction_steps < 1 or self.max_burst_tokens < 1:
            raise ValueError("correction_steps and max_burst_tokens must be positive")


class DisplayRevisionController:
    """One controller per runtime trajectory; never share it between independent runs.

    Preserve confident proposals until a stronger alternative is consistently supported.
    A changed external evidence revision resets trust, not the Display content itself.
    """

    def __init__(self, config: DisplayRevisionConfig | None = None) -> None:
        self.config = config or DisplayRevisionConfig()
        self.reset()

    def reset(self) -> None:
        self._trusted: Tensor | None = None
        self._candidate: Tensor | None = None
        self._streak: Tensor | None = None
        self._evidence: object = None
        self._last_step: int | None = None

    def observe_context(self, context: object) -> None:
        """Drop prior trust on a new run or any new external source/fact/percept."""
        facts = context.facts
        evidence = (
            facts.revision,
            tuple(repr(item) for item in context.sources),
            tuple(repr(item) for item in context.percepts),
        )
        step = int(context.step)
        if self._last_step is not None and (step <= self._last_step or evidence != self._evidence):
            self.reset()
        self._evidence = evidence
        self._last_step = step

    def refine_display(
        self,
        scheduler: CIDDiffusionScheduler,
        token_ids: Tensor,
        logits: Tensor,
        *,
        reveal_fraction: float,
        revision_fraction: float,
        revision_margin: float,
        repetition_score_multiplier: float,
    ) -> Tensor:
        result = scheduler.refine_display(
            token_ids,
            logits,
            reveal_fraction=reveal_fraction,
            revision_fraction=revision_fraction,
            revision_margin=revision_margin,
            repetition_score_multiplier=repetition_score_multiplier,
        )
        if scheduler.eos_token_id is None:
            return result
        if (
            self._trusted is None
            or self._trusted.shape != token_ids.shape
            or (self._trusted.device != token_ids.device)
        ):
            self._trusted = torch.full_like(token_ids, -1)
            self._candidate = torch.full_like(token_ids, -1)
            self._streak = torch.zeros_like(token_ids)
        assert self._candidate is not None and self._streak is not None
        # Inference-only: no gradient graph and no mutation of caller inputs.
        with torch.no_grad():
            scores = logits.detach().float()
            proposed = scores.argmax(dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            proposed_prob = probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
            eos_id = scheduler.eos_token_id
            mask_id = scheduler.mask_token_id
            for row in range(token_ids.shape[0]):
                old_eos = torch.nonzero(token_ids[row] == eos_id, as_tuple=False).flatten()
                if not old_eos.numel():
                    continue
                eos = int(old_eos[0])
                trusted = self._trusted[row]
                candidate = self._candidate[row]
                streak = self._streak[row]

                # Only burst past the existing EOS when the entire contiguous run,
                # INCLUDING its new EOS, is confidently predicted. No guessed length.
                tail: list[int] = []
                confirmed_eos = False
                limit = min(token_ids.shape[1] - 1, eos + self.config.max_burst_tokens)
                for pos in range(eos, limit + 1):
                    if float(proposed_prob[row, pos]) < self.config.burst_probability:
                        break
                    next_token = int(proposed[row, pos])
                    if next_token == eos_id:
                        confirmed_eos = True
                        break
                    if next_token == mask_id:
                        break
                    tail.append(next_token)
                if tail and confirmed_eos:
                    end = eos + len(tail)
                    result[row, eos:end] = torch.tensor(
                        tail, device=result.device, dtype=result.dtype
                    )
                    result[row, end] = eos_id
                    result[row, end + 1 :] = mask_id

                # Previously accepted content, including EOS, is not changed by
                # a transient alternative, but may be replaced after repeated evidence.
                for pos in range(token_ids.shape[1]):
                    accepted = int(trusted[pos])
                    if accepted < 0:
                        continue
                    if accepted != int(token_ids[row, pos]):
                        trusted[pos] = -1
                        candidate[pos] = -1
                        streak[pos] = 0
                        continue
                    alternative = int(proposed[row, pos])
                    if alternative != accepted and (
                        float(proposed_prob[row, pos]) >= self.config.correction_probability
                        and float(proposed_prob[row, pos] - probabilities[row, pos, accepted])
                        >= self.config.correction_gain
                    ):
                        if alternative == int(candidate[pos]):
                            streak[pos] += 1
                        else:
                            candidate[pos] = alternative
                            streak[pos] = 1
                    else:
                        candidate[pos] = -1
                        streak[pos] = 0
                    if int(streak[pos]) >= self.config.correction_steps:
                        trusted[pos] = -1
                        candidate[pos] = -1
                        streak[pos] = 0
                    else:
                        result[row, pos] = accepted

                trusted_eos = torch.nonzero(trusted == eos_id, as_tuple=False).flatten()
                if trusted_eos.numel():
                    end = int(trusted_eos[0])
                    result[row, end] = eos_id
                    result[row, end + 1 :] = mask_id
                # Stale anchors beyond the current active span can never be reused.
                revised_eos = torch.nonzero(result[row] == eos_id, as_tuple=False).flatten()
                stop = int(revised_eos[0]) + 1 if revised_eos.numel() else token_ids.shape[1]
                trusted[stop:] = -1
                candidate[stop:] = -1
                streak[stop:] = 0
                # Trust only tokens that the model itself agrees with at high confidence.
                for pos in range(stop):
                    token = int(result[row, pos])
                    if token == mask_id or int(trusted[pos]) >= 0:
                        continue
                    if token == int(proposed[row, pos]) and (
                        float(proposed_prob[row, pos]) >= self.config.anchor_probability
                    ):
                        trusted[pos] = token
        return result

"""Native Display training corruption, independent of inference canvas length."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def corrupt_native_reference(
    reference: Tensor,
    canvas: Tensor,
    *,
    mask_id: int,
    eos_id: int,
    timestep: float,
    generator: torch.Generator | None,
    alternative: Tensor | None = None,
    denoising_steps: int = 8,
) -> tuple[Tensor, Tensor, float]:
    """Mask a complete reference, leaving the latent tail unresolved."""
    if reference.ndim != 2 or reference.shape[0] != 1:
        raise ValueError("native reference must be a single nonempty row")
    width = reference.shape[1]
    if not 0 < width < canvas.shape[1]:
        raise ValueError("native reference must fit before transport EOS")
    if denoising_steps < 1:
        raise ValueError("denoising_steps must be positive")
    if not 0 < timestep <= 1:
        raise ValueError("native corruption timestep must be in (0, 1]")
    ids = torch.full_like(canvas, mask_id)
    ids[:, -1] = eos_id
    labels = torch.full_like(canvas, -100)
    hidden = torch.rand(reference.shape, device=reference.device, generator=generator) < timestep
    if alternative is not None and alternative.shape == reference.shape:
        differs = reference.ne(alternative)
        if (
            timestep < 1.0
            and bool(differs.any())
            and bool(torch.rand((), device=reference.device, generator=generator) < 0.5)
        ):
            hidden = differs
            timestep = (
                min(
                    denoising_steps,
                    max(1, math.ceil(float(hidden.float().mean()) * denoising_steps)),
                )
                / denoising_steps
            )
    if not bool(hidden.any()):
        chosen = torch.randint(width, (), device=reference.device, generator=generator)
        hidden[0, chosen] = True
    ids[:, :width] = torch.where(hidden, mask_id, reference)
    labels[:, :width] = torch.where(hidden, reference, -100)
    return ids, labels, timestep

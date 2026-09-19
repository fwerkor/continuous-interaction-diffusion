from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
masked_diffusion_loss = import_module("cid.model.diffusion_base").masked_diffusion_loss


class _UniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        return torch.zeros(
            (*input_ids.shape, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )


def test_masked_diffusion_loss_supports_bounded_mask_ratio_range() -> None:
    model = _UniformModel(vocab_size=8)
    clean = torch.arange(64 * 16).reshape(64, 16) % 7

    loss, masked_fraction, mean_ratio = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=7,
        min_mask_ratio=0.8,
        max_mask_ratio=0.8,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.isfinite(loss)
    assert 0.7 < float(masked_fraction) < 0.9
    assert float(mean_ratio) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0.0, 1.0), (0.9, 0.8), (0.1, 1.1)],
)
def test_masked_diffusion_loss_rejects_invalid_mask_ratio_range(
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(ValueError, match="mask ratio range"):
        masked_diffusion_loss(
            _UniformModel(vocab_size=8),
            torch.tensor([[1, 2, 3]]),
            mask_token_id=7,
            min_mask_ratio=minimum,
            max_mask_ratio=maximum,
        )

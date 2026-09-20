from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
diffusion_base = import_module("cid.model.diffusion_base")
masked_diffusion_loss = diffusion_base.masked_diffusion_loss
masked_diffusion_inputs = diffusion_base._masked_diffusion_inputs


class _UniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.projected_positions = 0

    def forward(self, input_ids, *, output_mask=None):
        if output_mask is None:
            shape = (*input_ids.shape, self.vocab_size)
            self.projected_positions = input_ids.numel()
        else:
            shape = (int(output_mask.sum()), self.vocab_size)
            self.projected_positions = shape[0]
        return torch.zeros(
            shape,
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


def test_masked_diffusion_loss_projects_only_masked_positions() -> None:
    model = _UniformModel(vocab_size=16)
    clean = torch.arange(8 * 32).reshape(8, 32) % 15

    loss, masked_fraction, _ = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=15,
        min_mask_ratio=0.25,
        max_mask_ratio=0.25,
        generator=torch.Generator().manual_seed(23),
    )

    expected_projected = round(float(masked_fraction) * clean.numel())
    assert model.projected_positions == expected_projected
    assert model.projected_positions < clean.numel()
    assert torch.isfinite(loss)


class _ProjectionModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, *, output_mask=None):
        hidden = self.embedding(input_ids)
        if output_mask is not None:
            hidden = hidden[output_mask]
        return self.head(hidden)


def test_masked_only_projection_matches_full_objective_and_gradients(monkeypatch) -> None:
    torch.manual_seed(31)
    clean = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    masked = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
        dtype=torch.bool,
    )
    ratios = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    corrupted = clean.masked_fill(masked, 9)
    metrics = torch.stack((masked.float().mean(), ratios.mean()))

    def fixed_inputs(*args, **kwargs):
        del args, kwargs
        return corrupted, masked, ratios, metrics

    monkeypatch.setattr(diffusion_base, "_masked_diffusion_inputs", fixed_inputs)
    selective = _ProjectionModel(vocab_size=10, hidden_size=6)
    reference = _ProjectionModel(vocab_size=10, hidden_size=6)
    reference.load_state_dict(selective.state_dict())

    selective_loss, _, _ = masked_diffusion_loss(
        selective,
        clean,
        mask_token_id=9,
    )
    selective_loss.backward()

    logits = reference(corrupted)
    token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        clean.reshape(-1),
        reduction="none",
    ).view_as(clean)
    reference_loss = (
        token_loss * masked.to(token_loss.dtype) / ratios
    ).sum() / clean.numel()
    reference_loss.backward()

    torch.testing.assert_close(selective_loss, reference_loss, rtol=1e-6, atol=1e-7)
    for selective_parameter, reference_parameter in zip(
        selective.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            selective_parameter.grad,
            reference_parameter.grad,
            rtol=1e-6,
            atol=1e-7,
        )


def test_masked_diffusion_inputs_force_one_token_without_host_branch() -> None:
    clean = torch.arange(4 * 8).reshape(4, 8)
    corrupted, masked, ratios, metrics = masked_diffusion_inputs(
        clean,
        mask_token_id=99,
        min_mask_ratio=1.0e-9,
        max_mask_ratio=1.0e-9,
        generator=torch.Generator().manual_seed(5),
    )

    assert masked.sum(dim=1).tolist() == [1, 1, 1, 1]
    assert torch.all(corrupted[masked] == 99)
    torch.testing.assert_close(ratios, torch.full_like(ratios, 1.0e-9))
    assert float(metrics[0]) == pytest.approx(1 / 8)


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

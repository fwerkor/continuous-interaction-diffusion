from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cid.model.native_reference import corrupt_native_reference  # noqa: E402


def corrupt(reference, noise, seed, alternative=None):
    return corrupt_native_reference(
        reference,
        torch.zeros((1, 16), dtype=torch.long),
        mask_id=1,
        eos_id=2,
        timestep=noise,
        generator=torch.Generator().manual_seed(seed),
        alternative=alternative,
    )


def test_full_and_partial_native_corruption():
    reference = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    full, labels, noise = corrupt(reference, 1.0, 5)
    assert full[:, :-1].eq(1).all()
    assert full[0, -1] == 2
    assert torch.equal(labels[:, :8], reference)
    assert noise == 1.0
    ids, labels, noise = corrupt(reference, 0.25, 5)
    hidden = ids[:, :8].eq(1)
    assert hidden.any() and (~hidden).any()
    assert torch.equal(labels[:, :8][hidden], reference[hidden])
    assert labels[:, :8][~hidden].eq(-100).all()
    assert ids[:, 8:-1].eq(1).all()
    assert labels[:, 8:].eq(-100).all()


def test_counterfactual_reveals_only_common_tokens():
    a = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    b = torch.tensor([[3, 4, 15, 6, 17, 8, 9, 10]])
    differences = a.ne(b)
    successes = 0
    for seed in range(24):
        ia, ya, _ = corrupt(a, 0.5, seed, b)
        ib, yb, _ = corrupt(b, 0.5, seed, a)
        if torch.equal(ia, ib) and torch.equal(ya[:, :8].ne(-100), differences):
            successes += 1
            assert torch.equal(yb[:, :8].ne(-100), differences)
            assert ia[:, :8][differences].eq(1).all()
            assert torch.equal(ya[:, :8][differences], a[differences])
            assert torch.equal(yb[:, :8][differences], b[differences])
    assert 0 < successes < 24


def test_always_leaves_a_masked_reference_token():
    _, labels, _ = corrupt(torch.tensor([[3]]), 0.001, 2)
    assert labels[0, 0] == 3
    assert labels[:, 1:].eq(-100).all()

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cid.model.language_preservation import native_language_kl
from cid.model.losses import _position_invariant_display_loss


def test_native_kl_preserves_any_display_position_and_backpropagates() -> None:
    native = torch.randn((1, 12, 24))
    student = native.clone().requires_grad_()
    eligible = torch.ones((1, 12), dtype=torch.bool)
    eligible[:, -1] = False
    original = native_language_kl(student, native, eligible, top_k=8, chunk_size=3)
    assert abs(float(original)) < 2e-6

    shifted = native.clone()
    shifted[0, 0, 5] += 12
    shifted[0, 9, 7] += 12
    shifted[0, 11, 8] += 12
    shifted.requires_grad_()
    penalty = native_language_kl(shifted, native, eligible, top_k=8, chunk_size=3)
    assert penalty > 0.05
    penalty.backward()
    assert shifted.grad[0, 0].abs().sum() > 0
    assert shifted.grad[0, 9].abs().sum() > 0
    assert shifted.grad[0, 11].abs().sum() == 0
    assert native.grad is None


def test_answer_alignment_stays_free_with_global_language_anchor() -> None:
    answer = torch.tensor([[3, 4]])
    eligible = torch.ones((1, 10), dtype=torch.bool)
    eligible[:, -1] = False
    native = torch.zeros((1, 10, 16))
    for offset in (0, 3, 6):
        student = native.clone()
        student[0, offset, 3] = 12
        student[0, offset + 1, 4] = 12
        student.requires_grad_()
        answer_loss = _position_invariant_display_loss(
            student, answer, eligible, batch_mask=torch.ones(1, dtype=torch.bool)
        )
        language_loss = native_language_kl(student, native, eligible)
        combined = answer_loss + 0.08 * language_loss
        assert combined.isfinite()
        combined.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
        if offset == 0:
            first = float(combined)
        else:
            assert abs(float(combined) - first) < 1e-5


def test_native_kl_invalid_shape_and_empty_mask() -> None:
    logits = torch.randn((2, 4, 12), requires_grad=True)
    with pytest.raises(ValueError, match="matching"):
        native_language_kl(logits, logits[:, :2], torch.ones(2, 4, dtype=torch.bool))
    z = native_language_kl(logits, logits.detach(), torch.zeros(2, 4, dtype=torch.bool))
    z.backward()
    assert z.item() == 0.0
    assert logits.grad.abs().sum().item() == 0.0

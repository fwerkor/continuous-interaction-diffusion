from __future__ import annotations

import pytest

from cid.model.losses import _position_invariant_display_loss

torch = pytest.importorskip("torch")


def _loss(logits, target, *, eligible=None, batch_mask=None):
    batch, length, _ = logits.shape
    if eligible is None:
        eligible = torch.ones((batch, length), dtype=torch.bool)
        eligible[:, -1] = False
    if batch_mask is None:
        batch_mask = torch.ones(batch, dtype=torch.bool)
    return _position_invariant_display_loss(
        logits, target, eligible, batch_mask=batch_mask
    )


def test_short_answer_is_accepted_after_arbitrary_long_prefix() -> None:
    logits = torch.zeros((1, 9, 16), requires_grad=True)
    with torch.no_grad():
        logits[0, 6, 7] = 9.0
    answer = torch.tensor([[7]])
    loss = _loss(logits, answer)
    assert loss.item() < 0.01
    loss.backward()
    # There is no pressure to replace natural prefix positions 0..5.
    assert torch.count_nonzero(logits.grad[0, :6]) == 0
    assert torch.count_nonzero(logits.grad[0, 6]) > 0
    assert torch.count_nonzero(logits.grad[0, -1]) == 0


def test_contiguous_multitoken_answer_can_appear_at_any_position() -> None:
    answer = torch.tensor([[3, 5, 8]])
    start_losses = []
    for start in (0, 2, 5):
        logits = torch.zeros((1, 9, 16))
        for offset, token in enumerate((3, 5, 8)):
            logits[0, start + offset, token] = 9.0
        start_losses.append(_loss(logits, answer).item())
    assert max(start_losses) - min(start_losses) < 1e-6


def test_mismatched_order_and_noncontiguous_tokens_do_not_pass() -> None:
    answer = torch.tensor([[3, 5, 8]])
    logits = torch.zeros((1, 9, 16))
    logits[0, 2, 3] = 9.0
    logits[0, 3, 8] = 9.0
    logits[0, 5, 5] = 9.0
    assert _loss(logits, answer).item() > 1.0


def test_eos_padding_unknown_and_batch_mask_do_not_contribute() -> None:
    logits = torch.zeros((3, 7, 16), requires_grad=True)
    with torch.no_grad():
        logits[0, 4, 3] = 7.0
        logits[0, 5, 5] = 7.0
        logits[0, 6, 3] = 25.0
    target = torch.tensor([[3, 5], [-100, -100], [3, 5]])
    positions = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.bool)
    active = torch.tensor([True, True, False])
    loss = _loss(logits, target, eligible=positions, batch_mask=active)
    assert loss.item() < 0.05
    loss.backward()
    assert not torch.count_nonzero(logits.grad[1:])
    assert not torch.count_nonzero(logits.grad[0, 6])


def test_answer_cannot_cross_eos_or_padding() -> None:
    logits = torch.zeros((1, 5, 8))
    positions = torch.tensor([[True, True, False, False, False]])
    with pytest.raises(ValueError, match="does not fit"):
        _loss(logits, torch.tensor([[1, 2, 3]]), eligible=positions)


def test_bad_target_padding_rejected() -> None:
    logits = torch.zeros((1, 5, 8))
    with pytest.raises(ValueError, match="right-padded"):
        _loss(logits, torch.tensor([[1, -100, 2]]))


def test_native_tensorizer_keeps_answer_out_of_teacher_input_and_loss() -> None:
    from dataclasses import replace

    from test_lfm_native_protocol import NativeTokenizer, _native_adapter
    from test_training import make_trajectory

    from cid.data import DISPLAY_UNKNOWN_MARKER, DisplayTarget
    from cid.model.training import ILLaDATrajectoryTensorizer, collate_training_steps

    class TrainTokenizer(NativeTokenizer):
        pad_token_id = 1

        def __call__(self, text, *, add_special_tokens, return_tensors, padding=False):
            if (
                add_special_tokens and not padding
                and isinstance(text, str) and text.startswith("[Question]")
            ):
                return super().__call__(
                    text, add_special_tokens=add_special_tokens, return_tensors=return_tensors
                )

            def encode(value):
                return [(ord(char) % 35) + 20 for char in value] or [3]

            if isinstance(text, list):
                sequences = [encode(value) for value in text]
                width = max(map(len, sequences))
                ids = torch.full((len(sequences), width), self.pad_token_id)
                mask = torch.zeros_like(ids)
                for row, sequence in enumerate(sequences):
                    ids[row, :len(sequence)] = torch.tensor(sequence)
                    mask[row, :len(sequence)] = 1
                return {"input_ids": ids, "attention_mask": mask}
            return {"input_ids": torch.tensor([encode(text)])}

    model = _native_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(
        model, TrainTokenizer(), display_canvas_tokens=12
    )
    example = make_trajectory()
    original = tensorizer.tensorize(example, source_step=-1, timestep=0.5)
    same_answer = tensorizer.tensorize(example, source_step=0, timestep=0.5)
    for sample in (original, same_answer):
        assert sample.targets.display_ids.eq(-100).all()
        assert sample.targets.display_answer_ids.shape == (1, 2)
        assert sample.batch.display_ids[:, :-1].eq(model.mask_token_id).all()
        assert sample.batch.display_ids[0, -1] == model.eos_token_id
        assert not sample.targets.display_answer_position_mask[0, -1]
    combined = collate_training_steps(
        (original, same_answer), pad_token_id=1
    )
    assert combined.targets.display_answer_ids.shape == (2, 2)

    unknown = replace(
        example,
        display_targets=(
            DisplayTarget(step=0, text=DISPLAY_UNKNOWN_MARKER),
            DisplayTarget(step=1, text=DISPLAY_UNKNOWN_MARKER),
        ),
    )
    pending = tensorizer.tensorize(unknown, source_step=0, timestep=0.5)
    assert pending.targets.display_answer_ids.eq(-100).all()
    assert pending.targets.display_ids.eq(-100).all()

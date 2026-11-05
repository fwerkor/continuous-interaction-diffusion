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
    example = replace(
        make_trajectory(),
        metadata={
            "native_full_display": "37.",
            "native_full_display_verified": True,
            "native_full_display_reviewer": "unit-test",
        },
    )
    original = tensorizer.tensorize(example, source_step=-1, timestep=0.5)
    same_answer = tensorizer.tensorize(example, source_step=0, timestep=0.5)
    for sample in (original, same_answer):
        full = tensorizer._encode_display_text(
            "37.\n[/Answer]"
        )
        length = full.shape[1]
        assert torch.equal(sample.targets.display_ids[:, :length], full)
        assert sample.targets.display_ids[:, length:].eq(-100).all()
        assert sample.targets.display_answer_ids.shape == (1, 2)
        assert sample.batch.display_ids[:, :-1].eq(model.mask_token_id).all()
        assert sample.batch.display_ids[0, -1] == model.eos_token_id
        assert not sample.targets.display_answer_position_mask[0, -1]
    combined = collate_training_steps(
        (original, same_answer), pad_token_id=1
    )
    assert combined.targets.display_answer_ids.shape == (2, 2)

    from cid.model.losses import cid_loss

    model.train()
    result = cid_loss(model(original.batch), original.targets)
    assert torch.isfinite(result.total)
    assert result.display.item() > 0.0
    result.display.backward()

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


def test_native_completion_requires_explicit_verified_response() -> None:
    from test_training import make_trajectory

    from cid.model.training import ILLaDATrajectoryTensorizer

    target = make_trajectory()
    with pytest.raises(ValueError, match="short CID answer fragments"):
        ILLaDATrajectoryTensorizer._native_full_display_reference(
            target, 0, target.target_display
        )


def test_native_completion_rejects_unverified_answer_mismatch() -> None:
    from dataclasses import replace

    from test_training import make_trajectory

    from cid.model.training import ILLaDATrajectoryTensorizer

    target = make_trajectory()
    bad = replace(
        target,
        metadata={
            "native_full_display": "The documented latency is 300.",
            "native_full_display_verified": True,
            "native_full_display_reviewer": "unit-test",
        },
    )
    with pytest.raises(ValueError, match="lacks reference answer"):
        ILLaDATrajectoryTensorizer._native_full_display_reference(
            bad, 0, target.target_display
        )
    verified = replace(
        bad, metadata={
            "native_full_display": "The documented latency is thirty-seven.",
            "native_full_display_verified": True,
            "native_full_display_reviewer": "unit-test",
            "native_full_display_semantic_paraphrase": True,
        },
    )
    assert ILLaDATrajectoryTensorizer._native_full_display_reference(
        verified, 0, target.target_display
    ) == "The documented latency is thirty-seven."


def test_native_completion_supervises_full_reply_and_terminator() -> None:
    from cid.model.losses import _masked_cross_entropy

    targets = torch.tensor([[4, 5, 6, 7, 8, 9, -100]])
    logits = torch.zeros((1, 7, 16), requires_grad=True)
    batch_mask = torch.tensor([True])
    loss = _masked_cross_entropy(
        logits, targets, targets.ne(-100), batch_mask=batch_mask
    )
    loss.backward()
    active_positions = logits.grad.abs().sum(dim=-1).flatten().gt(0)
    assert active_positions.tolist() == [True] * 6 + [False]

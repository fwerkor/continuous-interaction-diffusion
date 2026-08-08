from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
cid_model = import_module("cid.model")
CIDDiffusionScheduler = cid_model.CIDDiffusionScheduler


def test_display_corruption_masks_only_training_targets() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    timesteps = torch.tensor([1.0, 0.0])

    corruption = scheduler.corrupt_display(tokens, timesteps)

    assert torch.equal(corruption.token_ids[0], torch.tensor([5, 5, 5, 5]))
    assert torch.equal(corruption.labels[0], tokens[0])
    assert torch.equal(corruption.token_ids[1], tokens[1])
    assert torch.equal(corruption.labels[1], torch.full((4,), -100))
    assert corruption.noise.shape == (2, 4, 1)
    assert not corruption.replaced.any()


def test_display_corruption_can_train_visible_token_revisions() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    tokens = torch.tensor([[10, 11, 12, 13]])

    corruption = scheduler.corrupt_display(
        tokens,
        torch.tensor([1.0]),
        vocab_size=32,
        replacement_fraction=1.0,
        generator=torch.Generator().manual_seed(17),
    )

    assert corruption.replaced.all()
    assert not corruption.masked.any()
    assert torch.equal(corruption.labels, tokens)
    assert torch.all(corruption.token_ids != tokens)
    assert torch.all(corruption.token_ids != 5)


def test_thought_corruption_preserves_empty_slots() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    semantic = torch.ones(1, 3, 4)
    occupancy = torch.tensor([[[1.0], [0.0], [1.0]]])

    corruption = scheduler.corrupt_thought(
        semantic,
        torch.tensor([1.0]),
        occupancy,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(corruption.semantic[0, 1], torch.zeros(4))
    assert torch.equal(corruption.epsilon[0, 1], torch.zeros(4))
    assert corruption.noise[0, 0, 0] == 1.0
    assert corruption.noise[0, 1, 0] == 0.0


def test_display_reveal_commits_highest_confidence_masked_tokens_first() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    tokens = torch.tensor([[5, 5, 9, 5]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 7] = 9.0
    logits[0, 1, 8] = 3.0
    logits[0, 3, 6] = 1.0

    half = scheduler.reveal_display(tokens, logits, reveal_fraction=0.5)
    all_tokens = scheduler.reveal_display(tokens, logits, reveal_fraction=1.0)

    assert half.tolist() == [[7, 8, 9, 5]]
    assert all_tokens.tolist() == [[7, 8, 9, 6]]


def test_display_refinement_can_revise_visible_tokens_without_emitting_mask() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    tokens = torch.tensor([[5, 9, 10, 11]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 5] = 20.0
    logits[0, 0, 7] = 10.0
    logits[0, 1, 12] = 10.0
    logits[0, 2, 10] = 10.0
    logits[0, 3, 13] = 9.0

    refined = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=1.0,
        revision_fraction=0.5,
        revision_margin=0.1,
    )

    assert refined[0, 0] == 7
    assert refined[0, 1] == 12
    assert refined[0, 2] == 10
    assert refined[0, 3] == 13
    assert not refined.eq(5).any()


def test_diffusion_scheduler_validates_timestep_range() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)

    with pytest.raises(ValueError, match="timesteps"):
        scheduler.corrupt_display(torch.tensor([[1, 2]]), torch.tensor([1.2]))

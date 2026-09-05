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


def test_thought_corruption_supports_per_slot_diffusion_levels() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    semantic = torch.ones(1, 3, 4)
    occupancy = torch.ones(1, 3, 1)

    corruption = scheduler.corrupt_thought(
        semantic,
        torch.tensor([[0.0, 0.5, 1.0]]),
        occupancy,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(corruption.semantic[0, 0], semantic[0, 0])
    assert not torch.equal(corruption.semantic[0, 1], semantic[0, 1])
    assert not torch.equal(corruption.semantic[0, 2], semantic[0, 2])
    assert corruption.noise[0, :, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])


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


def test_display_refinement_can_preserve_an_unresolved_mask() -> None:
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

    assert refined[0, 0] == 5
    assert refined[0, 1] == 12
    assert refined[0, 2] == 10
    assert refined[0, 3] == 13
    assert refined.eq(5).sum() == 1


def test_chunked_display_refinement_matches_full_softmax_reference() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)
    generator = torch.Generator().manual_seed(29)
    tokens = torch.randint(6, 64, (2, 73), generator=generator)
    tokens[:, ::5] = 5
    logits = torch.randn(2, 73, 64, generator=generator)

    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    expected = tokens.clone()
    for batch_index in range(tokens.shape[0]):
        masked_positions = torch.nonzero(tokens[batch_index] == 5, as_tuple=False).flatten()
        reveal_count = (masked_positions.numel() + 1) // 2
        ranked = masked_positions[
            confidence[batch_index, masked_positions].argsort(descending=True)
        ]
        expected[batch_index, ranked[:reveal_count]] = predicted[batch_index, ranked[:reveal_count]]

        visible_positions = torch.nonzero(tokens[batch_index] != 5, as_tuple=False).flatten()
        current_ids = tokens[batch_index, visible_positions]
        current_confidence = probabilities[
            batch_index,
            visible_positions,
            current_ids,
        ]
        gains = confidence[batch_index, visible_positions] - current_confidence
        candidates = (predicted[batch_index, visible_positions] != current_ids) & (gains >= 0.05)
        candidate_positions = visible_positions[candidates]
        candidate_gains = gains[candidates]
        revision_count = min(
            candidate_positions.numel(),
            (visible_positions.numel() + 3) // 4,
        )
        ranked = candidate_positions[candidate_gains.argsort(descending=True)]
        expected[batch_index, ranked[:revision_count]] = predicted[
            batch_index, ranked[:revision_count]
        ]

    actual = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=0.5,
        revision_fraction=0.25,
        revision_margin=0.05,
    )

    assert torch.equal(actual, expected)


def test_diffusion_scheduler_validates_timestep_range() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5)

    with pytest.raises(ValueError, match="timesteps"):
        scheduler.corrupt_display(torch.tensor([[1, 2]]), torch.tensor([1.2]))


def test_display_reveal_can_place_eos_after_parallel_mask_resolution() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[9, 5, 5, 5]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 9] = 20.0
    logits[0, 1, 7] = 20.0
    logits[0, 2, 8] = 20.0
    logits[0, 3, 2] = 20.0

    refined = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=1.0,
        revision_fraction=0.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 7, 8, 2]]


def test_display_refinement_can_move_existing_eos_and_expand_tail() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[9, 2, 5, 5]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 9] = 30.0
    logits[0, 1, 7] = 30.0
    logits[0, 2, 8] = 30.0
    logits[0, 3, 2] = 30.0

    refined = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 7, 8, 2]]


def test_display_refinement_keeps_post_eos_tail_latent_when_boundary_is_stable() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[5, 2, 5, 5, 5, 5]])
    logits = torch.zeros(1, 6, 16)
    logits[0, 0, 9] = 30.0
    logits[0, 1, 2] = 30.0
    logits[0, 2, 7] = 30.0
    logits[0, 3, 8] = 30.0
    logits[0, 4, 10] = 30.0
    logits[0, 5, 11] = 30.0

    refined = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 2, 5, 5, 5, 5]]


def test_display_refinement_bounds_eos_expansion_per_step() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[9, 2, 5, 5, 5, 5, 5, 5]])
    logits = torch.zeros(1, 8, 16)
    logits[0, 0, 9] = 30.0
    logits[0, 1, 7] = 30.0
    logits[0, 2, 8] = 30.0
    logits[0, 3, 10] = 30.0
    logits[0, 7, 2] = 30.0

    refined = scheduler.refine_display(
        tokens,
        logits,
        reveal_fraction=0.125,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 7, 2, 5, 5, 5, 5, 5]]


def test_visible_replacement_corruption_never_injects_eos() -> None:
    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[10, 11, 12, 13]])

    corruption = scheduler.corrupt_display(
        tokens,
        torch.tensor([1.0]),
        vocab_size=16,
        replacement_fraction=1.0,
        generator=torch.Generator().manual_seed(3),
    )

    assert not corruption.token_ids.eq(2).any()
    assert not corruption.token_ids.eq(5).any()
    assert torch.all(corruption.token_ids != tokens)

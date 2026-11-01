from __future__ import annotations

from dataclasses import asdict, replace
from importlib import import_module

import pytest

from cid.data import DISPLAY_UNKNOWN_MARKER, DisplayTarget
from cid.model.diffusion import CIDDiffusionScheduler, denoising_noise_level
from cid.model.illada import ILLaDACIDConfig
from cid.model.lfm import LFMCIDAdapter
from cid.model.training import ILLaDATrajectoryTensorizer
from cid.state import DisplayCanvas

torch = pytest.importorskip("torch")
lfm_fixture = import_module("test_lfm_adapter")
training_fixture = import_module("test_training")


def new_adapter() -> LFMCIDAdapter:
    return LFMCIDAdapter(
        lfm_fixture.TinyLFMBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=8,
            max_display_tokens=32,
            display_canvas_tokens=16,
            semantic_head_version=2,
            interaction_architecture="identity-preserving",
        ),
        freeze_backbone=True,
    )


def test_preallocated_canvas_default_and_legacy_compatibility() -> None:
    new = new_adapter()
    assert new.config.display_initialization == "preallocated"
    assert ILLaDACIDConfig().display_initialization == "legacy"
    assert ILLaDACIDConfig(**asdict(new.config)) == new.config
    assert new.config.identity_allocation_threshold == 0.8
    with pytest.raises(ValueError, match="display_initialization"):
        ILLaDACIDConfig(display_initialization="invalid")
    scheduler = CIDDiffusionScheduler(
        16, 7, display_initialization="preallocated",
    )
    assert scheduler.initial_display(8).token_ids == (16,) * 7 + (7,)
    assert scheduler.initial_display(8).unresolved == 7
    assert DisplayCanvas.initial_unresolved(
        8, 16, eos_token_id=7,
    ).token_ids == (16, 7) + (16,) * 6


def test_preallocated_corruption_matches_runtime_reveal_levels() -> None:
    schedule = CIDDiffusionScheduler(16, 7, display_initialization="preallocated")
    goal = torch.tensor([[11, 12, 13, 7, 16, 16, 16, 16]])
    bootstrap = schedule.corrupt_preallocated_display(
        goal, torch.tensor([1.0]), generator=torch.Generator().manual_seed(11),
    )
    assert bootstrap.token_ids.tolist() == [[16] * 7 + [7]]
    assert bootstrap.labels[0, :4].tolist() == [11, 12, 13, 7]
    assert bootstrap.labels[0, 4:].tolist() == [-100] * 4
    assert bootstrap.noise.shape == (1, 8, 1)
    assert (bootstrap.noise == 1).all()
    assert not bootstrap.replaced.any()
    for i in range(8):
        t = denoising_noise_level(i, 8)
        sample = schedule.corrupt_preallocated_display(
            goal, torch.tensor([t]), generator=torch.Generator().manual_seed(13),
        )
        assert torch.allclose(sample.noise, torch.full_like(sample.noise, t))
        if sample.token_ids[0, 3] != 7:
            assert sample.token_ids[0, -1] == 7
    complete = schedule.corrupt_preallocated_display(
        goal, torch.tensor([0.0]), generator=torch.Generator().manual_seed(11),
    )
    assert complete.token_ids.tolist() == goal.tolist()
    assert not (complete.labels != -100).any()


def test_premature_eos_waits_until_prior_mask_is_revealed() -> None:
    scheduler = CIDDiffusionScheduler(9, 7, display_initialization="preallocated")
    old = torch.tensor([[9, 9, 9, 9, 9, 9, 9, 7]])
    logits = torch.zeros((1, 8, 16))
    logits[0, 1, 7] = 30.0
    # Ensure pos0 has much lower confidence than EOS so it stays a MASK
    logits[0, 0, 2] = 0.1
    logits[0, 2, 4] = 12.0
    intermediate = scheduler.refine_display(
        old, logits, reveal_fraction=0.15, revision_fraction=0.0,
        revision_margin=1.0, repetition_score_multiplier=1.0,
    )
    assert intermediate[0, 1] == 9
    assert intermediate[0, -1] == 7
    logits[0, 0, 3] = 35.0
    final = scheduler.refine_display(
        intermediate, logits, reveal_fraction=0.5, revision_fraction=0.0,
        revision_margin=1.0, repetition_score_multiplier=1.0,
    )
    assert final[0, 0] == 3
    assert final[0, 1] == 7
    assert torch.equal(final[0, 2:], torch.full((6,), 9))


def test_bootstrap_teacher_corruption_and_noise_match_inference() -> None:
    adapter = new_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter, training_fixture.TinyTokenizer(), display_canvas_tokens=16,
    )
    example = replace(
        training_fixture.make_trajectory(),
        target_display="AB",
        display_targets=(DisplayTarget(step=1, text="AB"),),
    )
    bootstrap = tensorizer.tensorize(example, source_step=-1, timestep=1.0)
    assert bootstrap.batch.display_ids[0, 0] == adapter.mask_token_id
    assert bootstrap.batch.display_ids[0, 1] == adapter.mask_token_id
    assert bootstrap.batch.display_ids[0, -1] == adapter.eos_token_id
    assert (bootstrap.batch.display_noise == 1).all()
    assert bootstrap.targets.display_ids[0, -1] == -100
    assert (bootstrap.targets.display_ids[0] != -100).any()

    unknown_example = replace(
        example,
        target_display=DISPLAY_UNKNOWN_MARKER,
        display_targets=(
            DisplayTarget(step=1, text=DISPLAY_UNKNOWN_MARKER),
        ),
    )
    unknown = tensorizer.tensorize(
        unknown_example, source_step=0, timestep=0.52,
    )
    assert unknown.batch.display_ids[0, -1] == adapter.eos_token_id
    assert (unknown.batch.display_ids[0, :-1] == adapter.mask_token_id).all()
    assert (unknown.targets.display_ids == -100).all()

    known = replace(
        example,
        display_targets=(
            DisplayTarget(step=0, text=example.target_display),
            DisplayTarget(step=1, text=example.target_display),
        ),
    )
    state = tensorizer.tensorize(known, source_step=0, timestep=0.52)
    noise = 5 / 8
    assert (state.batch.display_noise == noise).all()
    assert state.batch.display_ids.shape == bootstrap.batch.display_ids.shape
    # When the true EOS has not been revealed, only final EOS marks the end.
    eos_positions = (state.batch.display_ids[0] == adapter.eos_token_id).nonzero().flatten()
    assert len(eos_positions) == 1
    assert int(eos_positions[0]) in (
        int(tensorizer._encode_display_text(example.target_display).shape[1]),
        state.batch.display_ids.shape[1] - 1,
    )


def test_runtime_policy_upgrades_only_vnext_legacy_bootstrap() -> None:
    from cid.contracts import ModelContext
    from cid.model.policy import ILLaDAContextTensorizer, ILLaDANeuralPolicy
    from cid.state import CognitiveField, FactStore

    adapter = new_adapter()
    tokenizer = training_fixture.TinyTokenizer()
    tensorizer = ILLaDAContextTensorizer(adapter, tokenizer)
    observed: list[torch.Tensor] = []

    def forward(batch):
        observed.append(batch.display_ids.clone())
        return adapter(batch)

    policy = ILLaDANeuralPolicy(adapter, tensorizer, forward_model=forward)
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=CognitiveField.empty(2, adapter.d_model),
        display=DisplayCanvas.initial_unresolved(
            16, adapter.mask_token_id, eos_token_id=adapter.eos_token_id,
        ),
        sources=(), percepts=(), step=0, prompt="AB",
    )
    policy.step(context)
    assert observed
    assert observed[0].tolist() == [
        [adapter.mask_token_id] * 15 + [adapter.eos_token_id]
    ]
    # The caller's state is not mutated; the converted state belongs to this
    # model proposal and is carried by the Runtime update.
    assert context.display.token_ids[1] == adapter.eos_token_id
    adapter_legacy = LFMCIDAdapter(
        lfm_fixture.TinyLFMBackbone(),
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=32),
        freeze_backbone=True,
    )
    legacy_t = ILLaDAContextTensorizer(adapter_legacy, tokenizer)
    seen: list[torch.Tensor] = []

    def legacy_forward(batch):
        seen.append(batch.display_ids.clone())
        return adapter_legacy(batch)

    legacy_policy = ILLaDANeuralPolicy(
        adapter_legacy, legacy_t, forward_model=legacy_forward,
    )
    legacy_context = replace(
        context, thought=CognitiveField.empty(2, adapter_legacy.d_model),
    )
    legacy_policy.step(legacy_context)
    assert seen[0][0, 1] == adapter_legacy.eos_token_id


def test_stage_c_language_denoising_uses_preallocated_canvas() -> None:
    from cid.stage_c import StageCExample, make_display_batch
    from cid.state import CognitiveField

    adapter = new_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter, training_fixture.TinyTokenizer(),
        display_canvas_tokens=16,
    )
    # Stage C requires the runtime tensorizer, not the training tensorizer.
    from cid.model.policy import ILLaDAContextTensorizer

    runtime_tensorizer = ILLaDAContextTensorizer(
        adapter, training_fixture.TinyTokenizer(),
        text_encoder=tensorizer.text_encoder,
    )
    thought = CognitiveField.empty(adapter.config.max_thought_slots, adapter.d_model)
    generator = torch.Generator().manual_seed(37)
    observed = set()
    for _ in range(24):
        batch, labels = make_display_batch(
            runtime_tensorizer, thought,
            StageCExample("Hello", "AB"),
            generator=generator, rollout_probability=1.0,
        )
        assert labels.shape == batch.display_ids.shape
        assert (batch.display_noise == batch.display_noise[:, :1]).all()
        t = float(batch.display_noise[0, 0, 0])
        assert any(abs(t - index / 8) < 1e-6 for index in range(1, 9))
        observed.add(t)
        eos_at = (batch.display_ids[0] == adapter.eos_token_id).nonzero().flatten()
        assert len(eos_at) == 1
        assert int(eos_at[0]) in (2, 15)
        assert (batch.display_ids[0, 3:] != adapter.eos_token_id).all() or (
            int(eos_at[0]) == 15
        )
        assert (labels[0, 3:] == -100).all()
    assert len(observed) >= 5


def test_teacher_display_revision_preserves_stable_prefix_and_moves_eos() -> None:
    # Source is a previously accepted draft, then the target is extended
    # without any tool-specific EOS relocation rule.
    adapter = new_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter, training_fixture.TinyTokenizer(), display_canvas_tokens=16,
    )
    example = replace(
        training_fixture.make_trajectory(),
        target_display="ABC",
        display_targets=(
            DisplayTarget(step=0, text="AB"),
            DisplayTarget(step=1, text="ABC"),
        ),
    )
    state = tensorizer.tensorize(example, source_step=0, timestep=.52)
    encoded = tensorizer._encode_display_text("ABC")[0]
    assert torch.equal(state.batch.display_ids[0, :2], encoded[:2])
    assert int(state.batch.display_ids[0, 2]) == adapter.eos_token_id
    assert int(state.targets.display_ids[0, 2]) == int(encoded[2])
    assert int(state.targets.display_ids[0, 3]) == adapter.eos_token_id
    assert (state.targets.display_ids[0, :2] == -100).all()
    assert (state.targets.display_ids[0, 4:] == -100).all()
    assert (state.batch.display_noise == .625).all()

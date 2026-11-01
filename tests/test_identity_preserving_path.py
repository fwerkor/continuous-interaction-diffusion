from __future__ import annotations

from dataclasses import asdict, replace
from importlib import import_module

import pytest

from cid.model.illada import ILLaDACIDConfig
from cid.model.lfm import LFMCIDAdapter

torch = pytest.importorskip("torch")
lfm_test = import_module("test_lfm_adapter")

TinyLFMBackbone = lfm_test.TinyLFMBackbone
make_batch = lfm_test.make_batch


def new_adapter() -> LFMCIDAdapter:
    return LFMCIDAdapter(
        TinyLFMBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=8,
            max_display_tokens=16,
            interaction_architecture="identity-preserving",
            semantic_head_version=2,
        ),
        freeze_backbone=True,
    )


def test_config_roundtrip_and_validation() -> None:
    config = ILLaDACIDConfig(interaction_architecture="identity-preserving")
    assert ILLaDACIDConfig(**asdict(config)) == config
    assert ILLaDACIDConfig().interaction_architecture == "joint"
    with pytest.raises(ValueError, match="interaction_architecture"):
        ILLaDACIDConfig(interaction_architecture="other")


def test_neutral_display_exact_native_and_single_decoder_call() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    batch.thought_semantic.zero_()
    native = adapter.hidden_backbone()
    counts = [0]
    def count_calls(_module, _args, _output):
        counts[0] += 1

    hook = native.register_forward_hook(count_calls)
    with torch.no_grad():
        output = adapter(batch)
        token_ids = torch.cat((batch.prompt_ids, batch.display_ids), dim=1)
        reference = native(
            inputs_embeds=adapter.input_embeddings(token_ids),
            attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
            position_ids=torch.arange(token_ids.shape[1]).unsqueeze(0),
            return_dict=True,
        ).last_hidden_state[:, -batch.display_ids.shape[1]:]
        reference_logits = adapter.output_embeddings(reference)
    hook.remove()
    assert counts[0] == 2  # one for CID, one direct native reference
    torch.testing.assert_close(output.display_logits, reference_logits, atol=0, rtol=0)
    assert output.need_logits.shape[1] == batch.thought_semantic.shape[1]


def test_thought_updates_display_and_original_control_heads() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    with torch.no_grad():
        active = adapter(batch)
        batch.slot_occupancy.zero_()
        inactive = adapter(batch)
    assert not torch.equal(active.display_logits, inactive.display_logits)
    assert active.thought_semantic.shape == batch.thought_semantic.shape
    assert active.need_logits.shape == inactive.need_logits.shape
    assert active.revision_logits.shape == inactive.revision_logits.shape


def test_external_source_and_targeted_percept_masks() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    dimension = adapter.d_model
    batch.source_memory = torch.randn(1, 1, dimension)
    batch.fact_memory = torch.randn(1, 2, dimension)
    batch.percept_memory = torch.randn(1, 2, dimension)
    batch.percept_thought_mask = torch.tensor([
        [[True, False], [False, True]]
    ])
    batch.percept_display_mask = torch.tensor([
        [[True, False]] * batch.display_ids.shape[1]
    ])
    with torch.no_grad():
        output = adapter(batch)
    assert torch.isfinite(output.display_logits).all()
    assert torch.isfinite(output.thought_semantic).all()
    assert output.need_logits.shape[1] == batch.thought_semantic.shape[1]
    bad = replace(batch, percept_display_mask=torch.ones(1, 2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="percept routing"):
        adapter(bad)


def test_trainable_control_gate_preserves_native_when_empty() -> None:
    adapter = new_adapter()
    batch = make_batch()
    batch.slot_occupancy[0, 0] = 1
    adapter(batch).display_logits.float().square().mean().backward()
    assert adapter.identity_path is not None
    for name in ("display_gate", "control_gate"):
        parameter = getattr(adapter.identity_path, name)
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    adapter.eval()
    with torch.no_grad():
        batch.slot_occupancy.zero_()
        out = adapter(batch)
        ids = torch.cat((batch.prompt_ids, batch.display_ids), dim=1)
        native = adapter.hidden_backbone()(
            inputs_embeds=adapter.input_embeddings(ids),
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
            return_dict=True,
        ).last_hidden_state
        native_logits = adapter.output_embeddings(native[:, -batch.display_ids.shape[1]:])
    torch.testing.assert_close(out.display_logits, native_logits, atol=0, rtol=0)


def test_legacy_mode_has_no_new_parameters() -> None:
    adapter = LFMCIDAdapter(
        TinyLFMBackbone(),
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16),
    )
    assert adapter.identity_path is None
    assert not any(key.startswith("identity_path.") for key in adapter.state_dict())


def test_stage_a_b_optimizers_include_identity_modules_without_legacy_injections() -> None:
    from cid.model.training import (
        stage_a_adamw_parameter_groups,
        stage_b_adamw_parameter_groups,
    )

    adapter = new_adapter()
    all_named = dict(adapter.named_parameters())
    active = {name for name, param in all_named.items() if param.requires_grad}
    assert any(name.startswith("identity_path.") for name in active)
    for prefix in (
        "channel_embedding.", "display_noise_projection.",
        "external_fusion.", "occupancy_projection.",
    ):
        assert not any(name.startswith(prefix) for name in active)
    expected_ids = {id(param) for param in all_named.values() if param.requires_grad}
    for grouping in (stage_a_adamw_parameter_groups, stage_b_adamw_parameter_groups):
        groups = grouping(adapter)
        actual_ids = [id(param) for group in groups for param in group["params"]]
        assert len(actual_ids) == len(set(actual_ids))
        assert set(actual_ids) == expected_ids


def test_identity_checkpoint_restores_config_and_weights(tmp_path) -> None:
    from cid.model.huggingface import CIDConfig, CIDModel

    config = lfm_test.import_module("test_huggingface_model")._tiny_cid_config()
    stable_adapter_config = {
        **config.adapter_config,
        "semantic_head_version": 2,
        "interaction_architecture": "identity-preserving",
    }
    config = CIDConfig(
        backbone_config=config.backbone_config,
        adapter_config=stable_adapter_config,
        semantic_embedding=config.semantic_embedding,
        neural_contract_version=6,
    )
    model = CIDModel(config)
    assert model.cid_adapter.identity_path is not None
    with torch.no_grad():
        model.semantic_embedding_weight.zero_()
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="5GB")
    restored = CIDModel.from_pretrained(tmp_path, low_cpu_mem_usage=True)
    assert restored.cid_adapter.config.interaction_architecture == "identity-preserving"
    assert restored.cid_adapter.identity_path is not None
    for name, param in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], param, rtol=0, atol=0)


def test_stage_c_trains_native_language_without_unfreezing_control() -> None:
    from cid.stage_c import attach_display_adapter, freeze_for_stage_c

    adapter = new_adapter()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    module = attach_display_adapter(adapter, rank=4, alpha=8.0)
    freeze_for_stage_c(adapter)
    output = adapter(batch)
    loss = torch.nn.functional.cross_entropy(
        output.display_logits[:, :1].float().transpose(1, 2),
        batch.display_ids[:, :1],
    )
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in module.parameters()
    )
    assert all(
        not p.requires_grad for name, p in adapter.named_parameters()
        if not name.startswith("stage_c_display.")
    )
    with torch.no_grad():
        batch.slot_occupancy[0, 0] = 1
        conditioned = adapter(batch)
    assert not torch.equal(conditioned.display_logits, output.display_logits)


def test_percept_routing_changes_only_targeted_display_positions() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    batch.source_memory = torch.empty(1, 0, adapter.d_model)
    batch.fact_memory = torch.empty(1, 0, adapter.d_model)
    batch.percept_memory = torch.randn(1, 1, adapter.d_model)
    count = batch.display_ids.shape[1]
    batch.percept_display_mask = torch.zeros(1, count, 1, dtype=torch.bool)
    batch.percept_display_mask[0, 1, 0] = True
    with torch.no_grad():
        output = adapter(batch).display_logits
        ids = torch.cat((batch.prompt_ids, batch.display_ids), dim=1)
        reference = adapter.hidden_backbone()(
            inputs_embeds=adapter.input_embeddings(ids),
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
            return_dict=True,
        ).last_hidden_state[:, -count:]
        logits_native = adapter.output_embeddings(reference)
    for index in (0, 2, 3):
        torch.testing.assert_close(
            output[:, index], logits_native[:, index], atol=0, rtol=0,
        )
    assert not torch.equal(output[:, 1], logits_native[:, 1])


def test_noise_condition_has_no_effect_on_empty_cid_display() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    with torch.no_grad():
        first = adapter(batch).display_logits
        batch.display_noise.fill_(0.0)
        second = adapter(batch).display_logits
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_tool_free_thought_allocation_changes_display_on_first_step() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    assert batch.fact_memory.shape[1] == 0
    assert batch.source_memory.shape[1] == 0
    assert batch.percept_memory.shape[1] == 0
    head = adapter.output_heads.allocation_head
    with torch.no_grad():
        # The legacy CID semantic delta starts at zero for stable Stage A.
        # Simulate a learned nonzero diffusion update before checking recurrence.
        adapter.output_heads.thought_delta.weight.fill_(0.025)
        adapter.output_heads.thought_delta.bias.fill_(0.01)
        head.weight.zero_()
        head.bias.fill_(-10.0)
        native_output = adapter(batch)
        head.bias.fill_(10.0)
        cognitive_output = adapter(batch)
    assert not torch.equal(
        native_output.display_logits, cognitive_output.display_logits
    )
    # Explicitly a cognitive transition, not a tool transition.
    assert torch.sigmoid(cognitive_output.allocation_logits[0, 0]) > 0.8
    assert cognitive_output.convergence_logits[0] <= -19.0
    assert cognitive_output.thought_semantic.shape == batch.thought_semantic.shape
    with torch.no_grad():
        batch.thought_semantic[:, 0] = cognitive_output.thought_semantic[:, 0]
        batch.slot_occupancy[:, 0] = 1.0
        next_step = adapter(batch)
    assert not torch.equal(next_step.thought_semantic, cognitive_output.thought_semantic)


def test_reasoning_gate_backpropagates_to_allocation_head_without_tools() -> None:
    adapter = new_adapter().train()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    with torch.no_grad():
        adapter.output_heads.allocation_head.bias.fill_(-10.0)
    loss = adapter(batch).display_logits.float().square().mean()
    loss.backward()
    assert adapter.output_heads.allocation_head.bias.grad is not None
    assert adapter.output_heads.allocation_head.bias.grad.float().abs().sum() > 0


def test_neural_policy_rejects_mismatched_internal_reasoning_threshold() -> None:
    from cid.model.materialize import CIDMaterializer, CIDMaterializerConfig
    from cid.model.policy import ILLaDAContextTensorizer, ILLaDANeuralPolicy

    test_policy = import_module("test_neural_policy")
    adapter = new_adapter()
    tensorizer = ILLaDAContextTensorizer(adapter, test_policy.TinyTokenizer())
    mismatched = CIDMaterializer(CIDMaterializerConfig(allocation_threshold=0.5))
    with pytest.raises(ValueError, match="Thought allocation parameters must match"):
        ILLaDANeuralPolicy(adapter, tensorizer, materializer=mismatched)
    matched = CIDMaterializer(CIDMaterializerConfig(allocation_threshold=0.8))
    policy = ILLaDANeuralPolicy(adapter, tensorizer, materializer=matched)
    assert policy.materializer is matched


def test_new_internal_thought_prevents_immediate_runtime_convergence() -> None:
    from cid.contracts import ModelContext
    from cid.model.materialize import CIDMaterializer
    from cid.state import CognitiveField, DisplayCanvas, FactStore

    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    with torch.no_grad():
        adapter.output_heads.allocation_head.weight.zero_()
        adapter.output_heads.allocation_head.bias.fill_(10.0)
        adapter.output_heads.convergence_head.weight.zero_()
        adapter.output_heads.convergence_head.bias.fill_(10.0)
        output = adapter(batch)
    assert output.convergence_logits[0] < -19.0
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=CognitiveField.empty(2, adapter.d_model),
        display=DisplayCanvas.masked(length=4, mask_token_id=adapter.mask_token_id),
        sources=(),
        percepts=(),
        step=0,
        prompt="Solve this without tools",
    )
    update = CIDMaterializer().materialize(
        output, context, display_token_ids=tuple(batch.display_ids[0].tolist())
    )
    assert update.thought.cells[0].occupied
    assert not update.converged


def test_stage_a_b_bootstrap_matches_runtime_canvas_and_noise() -> None:
    from cid.contracts import ModelContext
    from cid.model.policy import ILLaDAContextTensorizer
    from cid.model.training import ILLaDATrajectoryTensorizer
    from cid.state import CognitiveField, DisplayCanvas, FactStore

    training = import_module("test_training")
    tokenizer = training.TinyTokenizer()
    adapter = training.ILLaDACIDAdapter(
        training.TinyBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=4, max_display_tokens=16,
            interaction_architecture="identity-preserving",
            identity_initial_mask_tokens=6, semantic_head_version=2,
        ),
        freeze_backbone=True,
    )
    example = training.make_trajectory()
    trajectory = ILLaDATrajectoryTensorizer(adapter, tokenizer)
    sample = trajectory.tensorize(example, source_step=-1, timestep=1.0)
    runtime_canvas = DisplayCanvas.initial_unresolved(
        length=sample.batch.display_ids.shape[1],
        mask_token_id=adapter.mask_token_id,
        eos_token_id=adapter.eos_token_id,
        initial_mask_tokens=adapter.config.identity_initial_mask_tokens,
    )
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=CognitiveField.empty(
            capacity=sample.batch.thought_semantic.shape[1], width=adapter.d_model
        ),
        display=runtime_canvas, sources=(), percepts=(), step=0,
        prompt=example.prompt,
    )
    runtime_batch = ILLaDAContextTensorizer(adapter, tokenizer)(
        context, display_noise_level=1.0
    )
    assert sample.batch.display_ids[0].tolist() == list(runtime_canvas.token_ids)
    torch.testing.assert_close(sample.batch.display_ids, runtime_batch.display_ids)
    torch.testing.assert_close(sample.batch.display_noise, runtime_batch.display_noise)
    assert sample.batch.display_padding_mask.sum() == 0
    assert runtime_batch.display_padding_mask.sum() == 0
    assert sample.batch.display_ids[0, 6] == adapter.eos_token_id
    assert not (sample.targets.display_ids[0] == adapter.mask_token_id).any()


def test_mask_window_gradual_reveal_keeps_boundary_revision_possible() -> None:
    from cid.model.diffusion import CIDDiffusionScheduler, denoising_reveal_fraction

    scheduler = CIDDiffusionScheduler(mask_token_id=5, eos_token_id=2)
    tokens = torch.tensor([[5] * 8 + [2] + [5] * 5])
    logits = torch.full((1, 14, 20), -10.0)
    for index in range(8):
        logits[0, index, 7 + (index % 3)] = 20.0
    logits[0, 8, 2] = 20.0
    partial = scheduler.refine_display(
        tokens, logits, reveal_fraction=denoising_reveal_fraction(0, 8),
        revision_fraction=0.25, revision_margin=0.1,
    )
    assert int((partial[0, :8] != 5).sum()) == 1
    assert partial[0, 8] == 2
    assert torch.equal(partial[0, 9:], tokens[0, 9:])
    # After the initial window is resolved, the EOS location remains movable.
    logits[0, 8, 10] = 30.0
    logits[0, 9, 2] = 30.0
    expanded = scheduler.refine_display(
        partial, logits, reveal_fraction=1.0 / 8,
        revision_fraction=0.25, revision_margin=0.1,
    )
    assert expanded[0, 8] == 10
    assert expanded[0, 9] == 2
    assert torch.all(expanded[0, 10:] == 5)


def test_joint_legacy_config_unaffected_by_identity_window() -> None:
    config = ILLaDACIDConfig(
        max_thought_slots=2, max_display_tokens=4,
        identity_initial_mask_tokens=8,
    )
    assert config.interaction_architecture == "joint"
    assert config.initial_display_mask_tokens(4) == 1
    identity = replace(config, interaction_architecture="identity-preserving")
    assert identity.initial_display_mask_tokens(4) == 3
    with pytest.raises(ValueError, match="identity_initial_mask_tokens"):
        ILLaDACIDConfig(
            max_thought_slots=2, max_display_tokens=4,
            interaction_architecture="identity-preserving",
            identity_initial_mask_tokens=0,
        )



@pytest.mark.parametrize("freeze_backbone", (True, False), ids=("stage-a", "stage-b"))
def test_preallocated_identity_training_updates_active_parameters(freeze_backbone: bool) -> None:
    from cid.data import DisplayTarget
    from cid.model.training import (
        CIDTrainer,
        CIDTrainerConfig,
        ILLaDATrajectoryTensorizer,
    )

    training = import_module("test_training")
    base = training.make_trajectory()
    example = replace(
        base,
        example_id="identity-bootstrap-train",
        source_descriptors=(),
        binding_targets=(),
        thought_targets=tuple(t for t in base.thought_targets if t.step == 0),
        display_targets=(DisplayTarget(step=0, text="37"),),
        grounding_targets=(),
    )
    adapter = training.ILLaDACIDAdapter(
        training.TinyBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=4,
            max_display_tokens=16,
            interaction_architecture="identity-preserving",
            identity_initial_mask_tokens=8,
            semantic_head_version=2,
        ),
        freeze_backbone=freeze_backbone,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, training.TinyTokenizer())
    before = {
        name: param.detach().clone()
        for name, param in adapter.named_parameters()
        if param.requires_grad and name.startswith("identity_path.")
    }
    report = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(timestep_min=1.0, timestep_max=1.0),
    ).train_examples((example,), epochs=1, shuffle=False)
    assert report.transitions == 1
    assert any(
        not torch.equal(before[name], param)
        for name, param in adapter.named_parameters()
        if name in before
    )


def test_source_descriptors_can_condition_display_without_existing_thought() -> None:
    adapter = new_adapter().eval()
    batch = make_batch()
    batch.slot_occupancy.zero_()
    with torch.no_grad():
        plain = adapter(batch)
        batch.source_memory = torch.randn(1, 2, adapter.d_model)
        conditioned = adapter(batch)
    assert not torch.equal(plain.display_logits, conditioned.display_logits)
    assert conditioned.source_logits.shape[-1] == 2
    assert torch.isfinite(conditioned.need_logits).all()

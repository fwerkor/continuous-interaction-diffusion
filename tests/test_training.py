from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from cid.contracts import FreshnessDemand
from cid.data import (
    BindingTarget,
    DisplayTarget,
    ExternalEvent,
    GroundingTarget,
    ThoughtTarget,
    TrajectoryExample,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole

torch = pytest.importorskip("torch")
ILLaDATextEncoder = import_module("cid.model.encoding").ILLaDATextEncoder
nn = import_module("torch.nn")
dist = import_module("torch.distributed")
cid_model = import_module("cid.model")

ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
ILLaDACIDConfig = cid_model.ILLaDACIDConfig
ILLaDATrajectoryTensorizer = cid_model.ILLaDATrajectoryTensorizer
CIDTrainer = cid_model.CIDTrainer
CIDTrainerConfig = cid_model.CIDTrainerConfig
CIDTrainerState = cid_model.CIDTrainerState
CIDRolloutState = cid_model.CIDRolloutState
balance_rollout_windows_by_semantic_task = cid_model.balance_rollout_windows_by_semantic_task
collate_training_steps = cid_model.collate_training_steps
load_cid_adapter_checkpoint = cid_model.load_cid_adapter_checkpoint
load_stage_b_checkpoint = cid_model.load_stage_b_checkpoint
load_stage_b_model_checkpoint = cid_model.load_stage_b_model_checkpoint
save_stage_b_checkpoint = cid_model.save_stage_b_checkpoint
shard_rollout_windows = cid_model.shard_rollout_windows
shard_transitions = cid_model.shard_transitions
trajectory_rollout_windows = cid_model.trajectory_rollout_windows
trajectory_transitions = cid_model.trajectory_transitions
wrap_stage_a_ddp = cid_model.wrap_stage_a_ddp
wrap_stage_b_fsdp = cid_model.wrap_stage_b_fsdp
cid_loss = cid_model.cid_loss


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 64
    num_attention_heads = 4
    max_position_embeddings = 64


class TinyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(TinyConfig.hidden_size, TinyConfig.hidden_size, bias=False)

    def forward(self, hidden):
        return self.projection(hidden)


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer()])

    def forward(self, *, inputs_embeds, attention_mask, position_ids, return_dict):
        del position_ids
        assert return_dict
        weights = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        context = (inputs_embeds * weights).sum(dim=1, keepdim=True)
        context = context / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.layers[0](context))


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder = TinyDecoder()
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.decoder


class TinyTokenizer:
    pad_token_id = 1

    def __call__(self, text, *, add_special_tokens, return_tensors, padding=False):
        assert return_tensors == "pt"
        if isinstance(text, list):
            encoded = [self._ids(item, add_special_tokens=add_special_tokens) for item in text]
            width = max((len(item) for item in encoded), default=0)
            input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for row, item in enumerate(encoded):
                input_ids[row, : len(item)] = torch.tensor(item)
                attention_mask[row, : len(item)] = 1
            assert padding
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        assert not padding
        return {"input_ids": torch.tensor([self._ids(text, add_special_tokens=add_special_tokens)])}

    @staticmethod
    def _ids(text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [6 + (ord(char) % 50) for char in text]
        if add_special_tokens:
            ids = [2, *ids, 3]
        return ids or [4]


def make_trajectory() -> TrajectoryExample:
    latency_anchor = Anchor(
        anchor_id="a:latency",
        kind=AnchorKind.TEXT,
        value="latency",
        object_id="metric:latency",
    )
    return TrajectoryExample(
        example_id="train-1",
        prompt="Return the documented latency.",
        target_display="37",
        protected_facts={"instruction": "use the documented value"},
        source_descriptors=(
            {
                "name": "docs",
                "description": "read documentation",
                "arguments": (
                    {"name": "key", "kind": "string", "required": True},
                    {"name": "scope", "kind": "string", "required": True},
                ),
            },
        ),
        binding_targets=(
            BindingTarget(
                need_id="latency",
                source="docs",
                first_need_step=1,
                executable_step=2,
                arguments={"key": "latency_ms", "scope": "production"},
                argument_steps={"key": 1, "scope": 2},
                target_cells=(ObjectRef.cell("c1"),),
            ),
        ),
        thought_targets=(
            ThoughtTarget(
                step=0,
                slot=0,
                cell_id="c0",
                semantic_text="Understand the request.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.4,
                noise=0.5,
            ),
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="c0",
                semantic_text="The request requires documentation.",
                roles={CognitiveRole.PLAN: 0.7, CognitiveRole.INFORMATION_NEED: 0.3},
                uncertainty=0.3,
                noise=0.3,
                lifecycle=CellLifecycle.STABLE,
            ),
            ThoughtTarget(
                step=1,
                slot=1,
                cell_id="c1",
                semantic_text="Need the documented latency value.",
                roles={CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.8,
                noise=1.0,
            ),
        ),
        display_targets=(DisplayTarget(step=1, text="37"),),
        grounding_targets=(
            GroundingTarget(
                step=1,
                cell_id="c1",
                anchors=(latency_anchor,),
                links=(
                    CognitiveLink(
                        relation=LinkRelation.REQUESTS,
                        target=ObjectRef.source("docs"),
                    ),
                ),
            ),
        ),
    )


def make_adapter(*, seed: int = 123) -> ILLaDACIDAdapter:
    torch.manual_seed(seed)
    return ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
        freeze_backbone=True,
    )


def test_trajectory_tensorizer_recycles_retired_source_slot() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trajectory = replace(
        make_trajectory(),
        binding_targets=(),
        grounding_targets=(),
        source_descriptors=(),
        thought_targets=(
            ThoughtTarget(
                step=0,
                slot=0,
                cell_id="retired",
                semantic_text="The old work item is complete.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.0,
                noise=0.0,
                lifecycle=CellLifecycle.RETIRED,
            ),
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="replacement",
                semantic_text="Reuse the released slot for new work.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.2,
                noise=0.2,
            ),
        ),
    )

    sample = tensorizer.tensorize(trajectory, source_step=0, timestep=1.0)

    assert sample.targets.allocation_targets[0, 0] == 1
    assert sample.targets.thought_mask[0, 0]


def test_trajectory_tensorizer_rejects_reuse_of_live_source_slot() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trajectory = replace(
        make_trajectory(),
        binding_targets=(),
        grounding_targets=(),
        source_descriptors=(),
        thought_targets=(
            ThoughtTarget(
                step=0,
                slot=0,
                cell_id="live",
                semantic_text="The current work item is still live.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.2,
                noise=0.2,
            ),
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="replacement",
                semantic_text="This must not overwrite live work.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.2,
                noise=0.2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="occupied source slot"):
        tensorizer.tensorize(trajectory, source_step=0, timestep=1.0)


def test_trajectory_tensorizer_runs_full_optimizer_step() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=1.0)
    target_display_ids = TinyTokenizer()(
        "37",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    assert sample.batch.thought_semantic.shape == (1, 4, TinyConfig.hidden_size)
    assert sample.batch.prompt_ids.shape[1] > 0
    assert torch.equal(sample.targets.display_ids, target_display_ids)
    assert torch.all(sample.batch.display_ids != target_display_ids)
    assert sample.targets.allocation_targets[0, 1] == 1
    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.thought_mask[0, :2].all()
    assert sample.targets.source_targets[0, 1] == 0
    assert sample.targets.argument_presence_targets[0, 1, 0] == 1
    assert sample.targets.argument_presence_targets[0, 1, 1] == 0
    assert sample.targets.anchor_mask[0, 1, 0]
    assert sample.targets.link_mask[0, 1, 0]

    optimizer = torch.optim.AdamW(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    before = adapter.output_heads.allocation_head.weight.detach().clone()
    output = adapter(sample.batch)
    losses = cid_loss(output, sample.targets)

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.argument_ground)
    assert torch.isfinite(losses.anchor_ground)
    assert torch.isfinite(losses.link_ground)
    optimizer.zero_grad()
    losses.total.backward()
    optimizer.step()

    assert not torch.equal(before, adapter.output_heads.allocation_head.weight)
    assert all(parameter.grad is None for parameter in adapter.backbone.parameters())


def test_bootstrap_transition_trains_single_snapshot_from_empty_tct() -> None:
    base = make_trajectory()
    single = replace(
        base,
        example_id="single-frame",
        target_display="Planning.",
        source_descriptors=(),
        binding_targets=(),
        thought_targets=tuple(target for target in base.thought_targets if target.step == 0),
        display_targets=(DisplayTarget(step=0, text="Planning."),),
        grounding_targets=(),
    )

    transitions = trajectory_transitions((single,))
    windows = trajectory_rollout_windows((single,), max_horizon=4)
    assert [(example.example_id, step) for example, step in transitions] == [("single-frame", -1)]
    assert tuple(window.source_steps for window in windows) == ((-1,),)

    adapter = make_adapter(seed=124)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    sample = tensorizer.tensorize(single, source_step=-1, timestep=1.0)

    assert not sample.batch.slot_occupancy.any()
    assert sample.targets.allocation_targets[0, 0] == 1
    assert sample.targets.allocation_mask[0, 0]
    assert sample.targets.thought_mask[0, 0]

    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(timestep_min=1.0, timestep_max=1.0),
    )
    report = trainer.train_examples((single,), epochs=1, shuffle=False)
    assert report.transitions == 1


def test_one_shot_need_is_supervised_before_but_not_after_its_observation() -> None:
    base = make_trajectory()
    arguments = {"key": "latency_ms", "scope": "production"}
    need_step = ThoughtTarget(
        step=0,
        slot=0,
        cell_id="need",
        semantic_text="Need the documented latency.",
        roles={CognitiveRole.INFORMATION_NEED: 1.0},
        uncertainty=0.8,
        noise=0.8,
    )
    percept_step = ThoughtTarget(
        step=1,
        slot=0,
        cell_id="need",
        semantic_text="Observed latency: 37.",
        roles={CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.05,
        noise=0.02,
        lifecycle=CellLifecycle.STABLE,
    )
    example = replace(
        base,
        binding_targets=(
            BindingTarget(
                need_id="latency",
                source="docs",
                first_need_step=0,
                executable_step=0,
                arguments=arguments,
                argument_steps={"key": 0, "scope": 0},
                target_cells=(ObjectRef.cell("need"),),
            ),
        ),
        events=(
            ExternalEvent(
                source="docs",
                value="37",
                arrival_step=1,
                arguments=arguments,
            ),
        ),
        thought_targets=(need_step, percept_step),
        display_targets=(DisplayTarget(step=0, text="Waiting."), DisplayTarget(step=1, text="37")),
        grounding_targets=(),
    )

    adapter = make_adapter(seed=125)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    bootstrap = tensorizer.tensorize(example, source_step=-1, timestep=1.0)
    assimilate = tensorizer.tensorize(example, source_step=0, timestep=1.0)

    assert bootstrap.targets.need_targets[0, 0] == 1
    assert bootstrap.targets.source_targets[0, 0] == 0
    assert assimilate.targets.need_targets[0, 0] == 0
    assert assimilate.targets.source_targets[0, 0] == -100

    persistent = replace(
        example,
        binding_targets=(replace(example.binding_targets[0], freshness=FreshnessDemand.ALWAYS),),
    )
    refresh = tensorizer.tensorize(persistent, source_step=0, timestep=1.0)
    assert refresh.targets.need_targets[0, 0] == 1
    assert refresh.targets.source_targets[0, 0] == 0


def test_frozen_text_encoder_snapshot_is_independent_from_live_backbone() -> None:
    adapter = make_adapter()
    tokenizer = TinyTokenizer()
    snapshot = ILLaDATextEncoder.from_frozen_snapshot(
        adapter,
        tokenizer,
        device="cpu",
        dtype=torch.bfloat16,
    )
    before = snapshot.encode_one("stable target", detach=True).clone()

    with torch.no_grad():
        adapter.input_embeddings.weight.add_(1.0)

    after = snapshot.encode_one("stable target", detach=True)
    live = ILLaDATextEncoder(adapter, tokenizer).encode_one("stable target", detach=True)
    assert torch.equal(before, after)
    assert not torch.allclose(after.float(), live.float())
    assert snapshot.dtype is torch.bfloat16
    assert all(not parameter.requires_grad for parameter in snapshot._embedding.parameters())

    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        tokenizer,
        text_encoder=snapshot,
    )
    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=1.0)
    assert sample.batch.thought_semantic.dtype is torch.bfloat16
    assert sample.batch.fact_memory.dtype is torch.bfloat16


def test_trajectory_tensorizer_supervises_convergence_only_on_final_step() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_trajectory()
    step_one = tuple(target for target in base.thought_targets if target.step == 1)
    extended = replace(
        base,
        thought_targets=(
            *base.thought_targets,
            *(replace(target, step=2) for target in step_one),
        ),
    )

    middle = tensorizer.tensorize(extended, source_step=0, timestep=1.0)
    final = tensorizer.tensorize(extended, source_step=1, timestep=1.0)

    assert middle.targets.convergence_targets.item() == 0.0
    assert final.targets.convergence_targets.item() == 1.0


def test_trajectory_tensorizer_marks_waiting_snapshot_as_current_information_equilibrium() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_trajectory()
    step_one = tuple(target for target in base.thought_targets if target.step == 1)
    waiting_step = tuple(
        replace(
            target,
            lifecycle=(CellLifecycle.WAITING if target.cell_id == "c1" else target.lifecycle),
        )
        for target in step_one
    )
    extended = replace(
        base,
        thought_targets=(
            *(target for target in base.thought_targets if target.step == 0),
            *waiting_step,
            *(replace(target, step=2, lifecycle=CellLifecycle.STABLE) for target in step_one),
        ),
    )

    waiting = tensorizer.tensorize(extended, source_step=0, timestep=1.0)
    final = tensorizer.tensorize(extended, source_step=1, timestep=1.0)

    assert waiting.targets.convergence_targets.item() == 1.0
    assert final.targets.convergence_targets.item() == 1.0


def test_training_collator_pads_variable_sequences_and_external_memory() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    first = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=1.0)
    second_example = replace(
        make_trajectory(),
        example_id="train-2",
        prompt="Short.",
        target_display="12345",
        protected_facts={},
        source_descriptors=(),
        binding_targets=(),
        grounding_targets=(),
        display_targets=(DisplayTarget(step=1, text="12345"),),
    )
    second = tensorizer.tensorize(second_example, source_step=0, timestep=1.0)

    training_batch = collate_training_steps((first, second), pad_token_id=1)

    assert training_batch.batch.thought_semantic.shape[0] == 2
    assert training_batch.batch.prompt_padding_mask.shape[0] == 2
    assert training_batch.batch.prompt_padding_mask[1].any()
    assert training_batch.batch.display_padding_mask[0].any()
    assert training_batch.batch.fact_memory.shape[:2] == (2, 1)
    assert not training_batch.batch.fact_padding_mask[0, 0]
    assert training_batch.batch.fact_padding_mask[1, 0]
    assert training_batch.batch.source_memory.shape[:2] == (2, 1)
    assert training_batch.batch.source_padding_mask[1, 0]
    assert training_batch.targets.display_ids.shape == training_batch.batch.display_ids.shape

    output = adapter(training_batch.batch)
    losses = cid_loss(output, training_batch.targets)
    assert output.display_logits.shape[:2] == training_batch.batch.display_ids.shape
    assert torch.isfinite(losses.total)
    losses.total.backward()


def test_trajectory_tensorizer_pads_mixed_thought_capacities_per_batch() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(max_thought_slots=128, max_display_tokens=16),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        TinyTokenizer(),
        minimum_thought_slots=8,
    )
    small_example = replace(make_trajectory(), prompt="x")
    small = tensorizer.tensorize(small_example, source_step=0, timestep=1.0)
    base = replace(make_trajectory(), prompt="y")
    expanded = replace(
        base,
        example_id="train-cap32",
        thought_targets=tuple(
            replace(target, slot=31 if target.cell_id == "c1" else target.slot)
            for target in base.thought_targets
        ),
    )
    large = tensorizer.tensorize(expanded, source_step=0, timestep=1.0)

    assert small.batch.thought_semantic.shape[1] == 8
    assert large.batch.thought_semantic.shape[1] == 32

    batch = collate_training_steps((small, large), pad_token_id=1)
    assert batch.batch.thought_semantic.shape[:2] == (2, 32)
    assert not batch.batch.slot_occupancy[0, 8:].any()
    assert not batch.targets.allocation_mask[0, 8:].any()
    assert (batch.targets.lifecycle[0, 8:] == -100).all()

    output = adapter(batch.batch)
    losses = cid_loss(output, batch.targets)
    assert torch.isfinite(losses.total)


def test_trainer_uses_configured_micro_batches() -> None:
    adapter = make_adapter(seed=66)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            timestep_min=1.0,
            timestep_max=1.0,
        ),
    )
    examples = (
        make_trajectory(),
        replace(make_trajectory(), example_id="train-microbatch-2"),
    )

    report = trainer.train_examples(examples, epochs=2, shuffle=False)

    assert report.transitions == 8
    assert report.optimizer_steps == 2
    assert trainer.state.transitions_seen == 8


def test_trainer_checkpoint_restores_trainable_state_optimizer_and_progress(tmp_path) -> None:
    config = CIDTrainerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        timestep_min=1.0,
        timestep_max=1.0,
        seed=9,
    )
    adapter = make_adapter(seed=77)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        config,
    )
    report = trainer.train_examples((make_trajectory(),), epochs=2, shuffle=True)

    assert report.transitions == 4
    assert report.optimizer_steps == 2
    assert trainer.state.transitions_seen == 4
    assert trainer.state.optimizer_steps == 2
    assert trainer.state.epochs_completed == 2

    path = tmp_path / "stage-a.pt"
    trainer.save_checkpoint(path)

    restored_adapter = make_adapter(seed=77)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        config,
    )
    restored.load_checkpoint(path)

    assert restored.state == trainer.state
    assert restored.trainable_parameter_names == trainer.trainable_parameter_names
    original_parameters = dict(adapter.named_parameters())
    restored_parameters = dict(restored_adapter.named_parameters())
    for name in trainer.trainable_parameter_names:
        assert torch.equal(original_parameters[name], restored_parameters[name])
    assert all(parameter.grad is None for parameter in restored_adapter.backbone.parameters())

    original_loss = trainer.train_transition(make_trajectory(), 0)
    restored_loss = restored.train_transition(make_trajectory(), 0)
    assert torch.allclose(original_loss.total, restored_loss.total)

    inference_adapter = make_adapter(seed=77)
    inference_adapter.set_backbone_trainable(True)
    loaded_state = load_cid_adapter_checkpoint(inference_adapter, path)
    assert loaded_state == CIDTrainerState(
        transitions_seen=4,
        optimizer_steps=2,
        epochs_completed=2,
    )
    inference_parameters = dict(inference_adapter.named_parameters())
    for name in trainer.trainable_parameter_names:
        assert torch.equal(original_parameters[name], inference_parameters[name])


def test_trainer_checkpoint_restores_pending_gradient_accumulation(tmp_path) -> None:
    config = CIDTrainerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        timestep_min=1.0,
        timestep_max=1.0,
        seed=11,
    )
    example = make_rollout_trajectory()

    baseline_adapter = make_adapter(seed=78)
    baseline = CIDTrainer(
        baseline_adapter,
        ILLaDATrajectoryTensorizer(baseline_adapter, TinyTokenizer()),
        config,
    )
    baseline.train_transition(example, 0)
    baseline.train_transition(example, 1)

    adapter = make_adapter(seed=78)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        config,
    )
    trainer.train_transition(example, 0)
    path = tmp_path / "mid-accumulation.pt"
    trainer.save_checkpoint(path)

    restored_adapter = make_adapter(seed=78)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        config,
    )
    restored.load_checkpoint(path)
    restored.train_transition(example, 1)

    baseline_parameters = dict(baseline_adapter.named_parameters())
    restored_parameters = dict(restored_adapter.named_parameters())
    for name in baseline.trainable_parameter_names:
        assert torch.allclose(baseline_parameters[name], restored_parameters[name])
    assert restored.state.optimizer_steps == 1


def test_rollout_training_reports_interval_progress() -> None:
    adapter = make_adapter(seed=79)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
            rollout_horizon=2,
            timestep_min=1.0,
            timestep_max=1.0,
        ),
    )
    windows = trajectory_rollout_windows((make_rollout_trajectory(),), max_horizon=2)
    progress = []

    trainer.train_rollout_windows(
        windows,
        epochs=1,
        shuffle=False,
        progress_every_optimizer_steps=1,
        progress_callback=progress.append,
    )

    assert progress
    assert all(item.mean_loss > 0.0 for item in progress)
    assert [item.optimizer_steps for item in progress] == sorted(
        item.optimizer_steps for item in progress
    )
    assert progress[-1].rollout_windows_seen_in_epoch == len(windows)
    assert trainer.state.rollout_windows_seen_in_epoch == 0


def test_transition_sharding_is_balanced_deterministic_and_complete() -> None:
    examples = tuple(replace(make_trajectory(), example_id=f"train-{index}") for index in range(5))
    transitions = trajectory_transitions(examples)

    shards = tuple(
        shard_transitions(
            transitions,
            world_size=3,
            rank=rank,
            seed=41,
            epoch=2,
        )
        for rank in range(3)
    )
    repeated = tuple(
        shard_transitions(
            transitions,
            world_size=3,
            rank=rank,
            seed=41,
            epoch=2,
        )
        for rank in range(3)
    )

    assert shards == repeated
    assert {len(shard) for shard in shards} == {4}
    observed_ids = {example.example_id for shard in shards for example, _ in shard}
    assert observed_ids == {example.example_id for example in examples}


def test_stage_a_ddp_handles_batches_with_unused_external_modules(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    init_file = tmp_path / "ddp-init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        adapter = make_adapter(seed=88)
        forward_model = wrap_stage_a_ddp(adapter, device_ids=None)
        trainer = CIDTrainer(
            adapter,
            ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
            CIDTrainerConfig(
                learning_rate=1e-3,
                timestep_min=1.0,
                timestep_max=1.0,
            ),
            forward_model=forward_model,
        )
        no_external = replace(
            make_trajectory(),
            protected_facts={},
            source_descriptors=(),
            binding_targets=(),
            grounding_targets=(),
        )

        report = trainer.train_examples((no_external,), epochs=2, shuffle=False)

        assert report.transitions == 4
        assert report.optimizer_steps == 4
        assert torch.isfinite(torch.tensor(report.mean_loss))
    finally:
        dist.destroy_process_group()


@pytest.mark.filterwarnings("ignore:FSDP is switching to use.*NO_SHARD.*")
@pytest.mark.filterwarnings("ignore:When using .*NO_SHARD.*")
def test_stage_b_fsdp_runs_full_parameter_optimizer_step_on_cpu(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    init_file = tmp_path / "fsdp-init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        adapter = make_adapter(seed=91)
        adapter.set_backbone_trainable(True)
        snapshot = ILLaDATextEncoder.from_frozen_snapshot(
            adapter,
            TinyTokenizer(),
            device="cpu",
            dtype=torch.bfloat16,
        )
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)
        trainer = CIDTrainer(
            adapter,
            ILLaDATrajectoryTensorizer(
                adapter,
                TinyTokenizer(),
                text_encoder=snapshot,
            ),
            CIDTrainerConfig(
                learning_rate=1e-3,
                timestep_min=1.0,
                timestep_max=1.0,
            ),
            optimizer=optimizer,
            forward_model=fsdp,
            gradient_clipper=fsdp.clip_grad_norm_,
        )
        before = adapter.backbone.get_decoder().layers[0].projection.weight.detach().clone()

        report = trainer.train_examples((make_trajectory(),), epochs=1, shuffle=False)

        assert report.transitions == 2
        assert report.optimizer_steps == 2
        assert not torch.equal(
            before,
            adapter.backbone.get_decoder().layers[0].projection.weight,
        )
        assert all(parameter.requires_grad for parameter in adapter.backbone.parameters())

        checkpoint = tmp_path / "stage-b-checkpoint"
        saved_weight = adapter.backbone.get_decoder().layers[0].projection.weight.detach().clone()
        save_stage_b_checkpoint(
            fsdp,
            optimizer,
            trainer,
            checkpoint,
            dataset_sha256="dataset-v1",
        )

        restored_adapter = make_adapter(seed=91)
        restored_adapter.set_backbone_trainable(True)
        restored_snapshot = ILLaDATextEncoder.from_frozen_snapshot(
            restored_adapter,
            TinyTokenizer(),
            device="cpu",
            dtype=torch.bfloat16,
        )
        restored_fsdp = wrap_stage_b_fsdp(
            restored_adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        restored_optimizer = torch.optim.AdamW(restored_fsdp.parameters(), lr=1e-3)
        restored_trainer = CIDTrainer(
            restored_adapter,
            ILLaDATrajectoryTensorizer(
                restored_adapter,
                TinyTokenizer(),
                text_encoder=restored_snapshot,
            ),
            CIDTrainerConfig(
                learning_rate=1e-3,
                timestep_min=1.0,
                timestep_max=1.0,
            ),
            optimizer=restored_optimizer,
            forward_model=restored_fsdp,
            gradient_clipper=restored_fsdp.clip_grad_norm_,
        )
        with pytest.raises(ValueError, match="dataset SHA-256"):
            load_stage_b_checkpoint(
                restored_fsdp,
                restored_optimizer,
                restored_trainer,
                checkpoint,
                expected_dataset_sha256="different-dataset",
            )
        load_stage_b_checkpoint(
            restored_fsdp,
            restored_optimizer,
            restored_trainer,
            checkpoint,
            expected_dataset_sha256="dataset-v1",
        )

        assert restored_trainer.state == trainer.state
        assert torch.equal(
            saved_weight,
            restored_adapter.backbone.get_decoder().layers[0].projection.weight,
        )

        inference_adapter = make_adapter(seed=91)
        inference_adapter.set_backbone_trainable(True)
        inference_fsdp = wrap_stage_b_fsdp(
            inference_adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        load_stage_b_model_checkpoint(inference_fsdp, inference_adapter, checkpoint)
        assert torch.equal(
            saved_weight,
            inference_adapter.backbone.get_decoder().layers[0].projection.weight,
        )
        assert (checkpoint / "metadata.json").is_file()
        assert (checkpoint / "rank-0000.pt").is_file()
        assert (checkpoint / "optimizer-rank-0000.pt").is_file()
    finally:
        dist.destroy_process_group()


def test_stage_b_fsdp_full_shard_two_rank_smoke(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    checkpoint = tmp_path / "distributed-checkpoint"
    worker = Path(__file__).with_name("fsdp_stage_b_worker.py")
    env = os.environ.copy()
    env["CID_FSDP_SMOKE_DIR"] = str(checkpoint)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (checkpoint / "distributed" / ".metadata").is_file()
    assert (checkpoint / "metadata.json").is_file()
    assert (checkpoint / "optimizer-rank-0000.pt").is_file()
    assert (checkpoint / "optimizer-rank-0001.pt").is_file()


def make_rollout_trajectory() -> TrajectoryExample:
    base = make_trajectory()
    step_two = tuple(
        replace(
            target,
            step=2,
            semantic_text=f"{target.semantic_text} Revised with another interaction step.",
            lifecycle=CellLifecycle.STABLE,
            uncertainty=max(0.0, target.uncertainty - 0.1),
            noise=max(0.0, target.noise - 0.1),
        )
        for target in base.thought_targets
        if target.step == 1
    )
    return replace(
        base,
        example_id="rollout-1",
        thought_targets=(*base.thought_targets, *step_two),
        display_targets=(*base.display_targets, DisplayTarget(step=2, text="38")),
        target_display="38",
    )


def test_rollout_state_replaces_teacher_input_t_and_y() -> None:
    adapter = make_adapter(seed=101)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = make_rollout_trajectory()
    slots = adapter.config.max_thought_slots
    rollout = CIDRolloutState(
        thought_semantic=torch.full((1, slots, adapter.d_model), 0.25),
        role_features=torch.full((1, slots, adapter.config.num_roles), 0.75),
        uncertainty=torch.full((1, slots, 1), 0.2),
        slot_occupancy=torch.ones((1, slots, 1)),
        display_ids=torch.tensor([[17]]),
    )

    sample = tensorizer.tensorize(
        example,
        source_step=1,
        timestep=0.0,
        rollout_state=rollout,
    )

    assert torch.allclose(sample.batch.thought_semantic, rollout.thought_semantic)
    assert torch.allclose(sample.batch.role_features, rollout.role_features)
    assert torch.allclose(sample.batch.uncertainty, rollout.uncertainty)
    assert torch.equal(sample.batch.slot_occupancy, rollout.slot_occupancy)
    assert sample.batch.display_ids[0, 0] == 17
    assert sample.targets.display_ids[0, 0] != -100


def test_self_rollout_feeds_previous_prediction_into_next_transition() -> None:
    class RecordingTensorizer(ILLaDATrajectoryTensorizer):
        def __init__(self, adapter, tokenizer) -> None:
            super().__init__(adapter, tokenizer)
            self.rollout_flags: list[bool] = []

        def tensorize(self, *args, rollout_state=None, **kwargs):
            self.rollout_flags.append(rollout_state is not None)
            return super().tensorize(*args, rollout_state=rollout_state, **kwargs)

    adapter = make_adapter(seed=102)
    tensorizer = RecordingTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            learning_rate=1e-3,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
            timestep_min=0.0,
            timestep_max=0.0,
        ),
    )
    windows = trajectory_rollout_windows(
        (make_rollout_trajectory(),),
        max_horizon=2,
    )

    report = trainer.train_rollout_windows(windows, epochs=1, shuffle=False)

    assert report.transitions == 3
    assert tensorizer.rollout_flags == [False, False, True]
    assert trainer.state.epochs_completed == 1


def test_semantic_task_balancing_equalizes_total_transition_loss_mass() -> None:
    short = replace(
        make_trajectory(),
        example_id="short-schedule",
        metadata={"semantic_task_id": "short"},
    )
    long = replace(
        make_rollout_trajectory(),
        example_id="long-schedule",
        metadata={"semantic_task_id": "long"},
    )
    windows = trajectory_rollout_windows((short, long), max_horizon=8)
    balanced = balance_rollout_windows_by_semantic_task(windows)

    mass: dict[str, float] = {}
    for window in balanced:
        task_id = str(window.example.metadata["semantic_task_id"])
        mass[task_id] = mass.get(task_id, 0.0) + len(window.source_steps) * window.loss_weight

    assert mass["short"] == pytest.approx(mass["long"])
    total_transitions = sum(len(window.source_steps) for window in balanced)
    weighted_transitions = sum(len(window.source_steps) * window.loss_weight for window in balanced)
    assert weighted_transitions == pytest.approx(total_transitions)


def test_semantic_task_balancing_respects_declared_training_weight() -> None:
    ordinary = replace(
        make_trajectory(),
        example_id="ordinary-schedule",
        metadata={"semantic_task_id": "ordinary", "training_weight": 1.0},
    )
    emphasized = replace(
        make_rollout_trajectory(),
        example_id="emphasized-schedule",
        metadata={"semantic_task_id": "emphasized", "training_weight": 3.0},
    )
    windows = trajectory_rollout_windows((ordinary, emphasized), max_horizon=8)
    balanced = balance_rollout_windows_by_semantic_task(windows)

    mass: dict[str, float] = {}
    for window in balanced:
        task_id = str(window.example.metadata["semantic_task_id"])
        mass[task_id] = mass.get(task_id, 0.0) + len(window.source_steps) * window.loss_weight

    assert mass["emphasized"] == pytest.approx(3.0 * mass["ordinary"])
    total_transitions = sum(len(window.source_steps) for window in balanced)
    weighted_transitions = sum(len(window.source_steps) * window.loss_weight for window in balanced)
    assert weighted_transitions == pytest.approx(total_transitions)


def test_rollout_sharding_globally_mixes_weighted_microbatches() -> None:
    examples = tuple(replace(make_trajectory(), example_id=f"mix-{index}") for index in range(24))
    windows = list(trajectory_rollout_windows(examples, max_horizon=3))
    weighted = tuple(
        replace(window, loss_weight=0.5 if index < len(windows) // 2 else 2.0)
        for index, window in enumerate(windows)
    )
    shards = tuple(
        shard_rollout_windows(
            weighted,
            world_size=3,
            rank=rank,
            seed=11,
            epoch=4,
            micro_batch_size=2,
        )
        for rank in range(3)
    )

    chunk_weight_sequences = []
    for shard in shards:
        assert len(shard) % 2 == 0
        chunk_weights = []
        for start in range(0, len(shard), 2):
            pair = shard[start : start + 2]
            assert pair[0].loss_weight == pair[1].loss_weight
            chunk_weights.append(pair[0].loss_weight)
        chunk_weight_sequences.append(tuple(chunk_weights))

    assert len(set(chunk_weight_sequences)) == 1
    sequence = chunk_weight_sequences[0]
    assert sequence != tuple(sorted(sequence))
    assert sum(left != right for left, right in zip(sequence, sequence[1:], strict=False)) >= 2


def test_learning_rate_schedule_warms_up_then_cosine_decays() -> None:
    adapter = make_adapter(seed=107)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            warmup_steps=2,
            lr_decay_steps=6,
            min_learning_rate_ratio=0.1,
        ),
    )

    assert trainer._learning_rate_for_step(1) == pytest.approx(5e-4)
    assert trainer._learning_rate_for_step(2) == pytest.approx(1e-3)
    assert trainer._learning_rate_for_step(4) == pytest.approx(5.5e-4)
    assert trainer._learning_rate_for_step(6) == pytest.approx(1e-4)
    assert trainer._learning_rate_for_step(20) == pytest.approx(1e-4)


def test_rollout_report_keeps_raw_and_weighted_loss_separate() -> None:
    adapter = make_adapter(seed=108)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            timestep_min=0.0,
            timestep_max=0.0,
        ),
    )
    window = replace(
        trajectory_rollout_windows((make_trajectory(),), max_horizon=3)[0],
        loss_weight=2.5,
    )

    report = trainer.train_rollout_windows((window,), epochs=1, shuffle=False)

    assert report.mean_loss == pytest.approx(report.raw_mean_loss * 2.5)


def test_rollout_curriculum_and_window_sharding_are_deterministic() -> None:
    adapter = make_adapter(seed=103)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            rollout_horizon=3,
            teacher_forcing_epochs=1,
            rollout_ramp_epochs=2,
        ),
    )
    assert trainer.rollout_probability() == 0.0
    trainer.state = CIDTrainerState(epochs_completed=1)
    assert trainer.rollout_probability() == 0.5
    trainer.state = CIDTrainerState(epochs_completed=2)
    assert trainer.rollout_probability() == 1.0

    examples = tuple(
        replace(make_rollout_trajectory(), example_id=f"rollout-{index}") for index in range(5)
    )
    windows = trajectory_rollout_windows(examples, max_horizon=3)
    shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=3,
            rank=rank,
            seed=7,
            epoch=2,
        )
        for rank in range(3)
    )
    assert {tuple(len(window.source_steps) for window in shard) for shard in shards} == {(3, 3)}

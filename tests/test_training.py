from __future__ import annotations

import math
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
CIDRolloutWindow = cid_model.CIDRolloutWindow
balance_rollout_windows_by_semantic_task = cid_model.balance_rollout_windows_by_semantic_task
collate_training_steps = cid_model.collate_training_steps
load_cid_adapter_checkpoint = cid_model.load_cid_adapter_checkpoint
load_stage_b_checkpoint = cid_model.load_stage_b_checkpoint
load_stage_b_model_checkpoint = cid_model.load_stage_b_model_checkpoint
save_stage_b_checkpoint = cid_model.save_stage_b_checkpoint
shard_rollout_windows = cid_model.shard_rollout_windows
shard_transitions = cid_model.shard_transitions
stage_b_adamw_parameter_groups = cid_model.stage_b_adamw_parameter_groups
stage_b_consumed_windows_by_bucket = cid_model.stage_b_consumed_windows_by_bucket
stage_b_gradient_accumulation_steps = cid_model.stage_b_gradient_accumulation_steps
stage_b_optimizer_steps_per_epoch = cid_model.stage_b_optimizer_steps_per_epoch
trajectory_rollout_windows = cid_model.trajectory_rollout_windows
trajectory_transitions = cid_model.trajectory_transitions
wrap_stage_a_ddp = cid_model.wrap_stage_a_ddp
wrap_stage_b_fsdp = cid_model.wrap_stage_b_fsdp
cid_loss = cid_model.cid_loss
chunked_illada_mlp_forward = import_module("cid.model.illada")._chunked_illada_mlp_forward
chunked_illada_rms_norm_forward = import_module(
    "cid.model.illada"
)._chunked_illada_rms_norm_forward


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


def test_chunked_illada_mlp_matches_unchunked_forward_and_input_gradient() -> None:
    torch.manual_seed(11)
    module = SimpleNamespace(
        gate_proj=nn.Linear(16, 48, bias=False),
        up_proj=nn.Linear(16, 48, bias=False),
        down_proj=nn.Linear(48, 16, bias=False),
        act_fn=nn.SiLU(),
        dropout=nn.Dropout(0.0),
        _cid_mlp_chunk_size=7,
    )
    reference_input = torch.randn(2, 23, 16, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)

    reference = module.dropout(
        module.down_proj(
            module.act_fn(module.gate_proj(reference_input)) * module.up_proj(reference_input)
        )
    )
    chunked = chunked_illada_mlp_forward(module, chunked_input)
    reference_grad = torch.autograd.grad(reference.sum(), reference_input)[0]
    chunked_grad = torch.autograd.grad(chunked.sum(), chunked_input)[0]

    assert torch.allclose(chunked, reference, rtol=1e-6, atol=1e-6)
    assert torch.allclose(chunked_grad, reference_grad, rtol=1e-6, atol=1e-6)


def test_chunked_illada_rms_norm_matches_unchunked_forward_and_gradient() -> None:
    torch.manual_seed(12)
    weight = nn.Parameter(torch.randn(16))
    module = SimpleNamespace(
        weight=weight,
        variance_epsilon=1e-6,
        _cid_norm_chunk_size=7,
    )
    reference_input = torch.randn(2, 23, 16, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)

    values = reference_input.to(torch.float32)
    variance = values.pow(2).mean(-1, keepdim=True)
    reference = weight * (values * torch.rsqrt(variance + 1e-6)).to(reference_input.dtype)
    chunked = chunked_illada_rms_norm_forward(module, chunked_input)
    reference_grads = torch.autograd.grad(
        reference.sum(),
        (reference_input, weight),
        retain_graph=True,
    )
    chunked_grads = torch.autograd.grad(chunked.sum(), (chunked_input, weight))

    assert torch.allclose(chunked, reference, rtol=1e-6, atol=1e-6)
    assert torch.allclose(chunked_grads[0], reference_grads[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(chunked_grads[1], reference_grads[1], rtol=1e-6, atol=1e-6)


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

    assert sample.targets.allocation_targets[0, 1] == 1
    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.thought_mask[0, 1]


def test_trajectory_tensorizer_never_reuses_retired_slot_within_trajectory() -> None:
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
                semantic_text="Old state.",
                lifecycle=CellLifecycle.RETIRED,
            ),
            ThoughtTarget(step=1, slot=0, cell_id="next", semantic_text="Next state."),
            ThoughtTarget(step=2, slot=0, cell_id="next", semantic_text="Next state updated."),
            ThoughtTarget(step=2, slot=1, cell_id="later", semantic_text="Later state."),
        ),
    )

    sample = tensorizer.tensorize(trajectory, source_step=1, timestep=1.0)

    assert sample.targets.allocation_targets[0, 2] == 1.0
    assert sample.targets.allocation_targets[0, 0] == 0.0
    assert sample.targets.thought_mask[0, 2]


def test_trajectory_tensorizer_always_uses_adapter_physical_tct_capacity() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=8,
            max_display_tokens=16,
            display_canvas_tokens=8,
        ),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter, TinyTokenizer(), minimum_thought_slots=1
    )

    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=0.5)

    assert sample.batch.thought_semantic.shape[1] == 8
    assert sample.batch.slot_occupancy.shape[1] == 8
    assert sample.targets.allocation_targets.shape[1] == 8


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

    with pytest.raises(ValueError, match="removed cells without retirement"):
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
    assert torch.equal(sample.targets.display_ids[0, :2], target_display_ids[0])
    assert sample.targets.display_ids[0, 2] == 2
    assert (sample.targets.display_ids[0, 3:] == -100).all()
    assert sample.batch.display_ids.shape[1] == tensorizer.display_canvas_tokens
    assert sample.targets.allocation_targets[0, 1] == 1
    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.thought_mask[0, :2].all()
    assert sample.targets.source_targets[0, 1, 0] == 0
    assert sample.targets.argument_presence_targets[0, 1, 0, 0] == 1
    assert sample.targets.argument_presence_targets[0, 1, 0, 1] == 0
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


def test_trajectory_tensorizer_rejects_display_that_exceeds_fixed_canvas() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=4),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = replace(
        make_trajectory(),
        target_display="abcdefghij",
        display_targets=(DisplayTarget(step=1, text="abcdefghij"),),
    )

    with pytest.raises(ValueError, match="including EOS"):
        tensorizer.tensorize(example, source_step=0, timestep=1.0)


def test_trajectory_tensorizer_expands_display_to_coarse_bucket() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=4,
            max_display_tokens=32,
            display_canvas_tokens=4,
        ),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = replace(
        make_trajectory(),
        target_display="abcdefghij",
        display_targets=(DisplayTarget(step=1, text="abcdefghij"),),
    )

    sample = tensorizer.tensorize(example, source_step=0, timestep=1.0)

    # 10 target tokens + EOS are rounded to a 16-token training bucket.
    assert sample.batch.display_ids.shape[1] == 16
    assert sample.targets.display_ids[0, 10] == tensorizer.eos_token_id
    assert (sample.targets.display_ids[0, 11:] == -100).all()


def test_rollout_display_bucket_can_grow_but_never_shrinks() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=4,
            max_display_tokens=32,
            display_canvas_tokens=4,
        ),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = make_rollout_trajectory()
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=torch.full((1, 16), 5, dtype=torch.long),
    )

    sample = tensorizer.tensorize(
        example, source_step=1, timestep=0.0, rollout_state=rollout
    )

    assert sample.batch.display_ids.shape[1] == 16


def test_trajectory_tensorizer_reserves_context_for_tct_and_prompt() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = replace(
        make_trajectory(),
        prompt="p" * 54,
        target_display="abcdefghij",
        display_targets=(DisplayTarget(step=1, text="abcdefghij"),),
    )

    with pytest.raises(ValueError, match="display bucket"):
        tensorizer.tensorize(example, source_step=0, timestep=1.0)


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

    assert bootstrap.targets.need_targets[0, 0, 0] == 1
    assert bootstrap.targets.source_targets[0, 0, 0] == 0
    assert assimilate.targets.need_targets[0, 0, 0] == 0
    assert assimilate.targets.source_targets[0, 0, 0] == -100

    persistent = replace(
        example,
        binding_targets=(replace(example.binding_targets[0], freshness=FreshnessDemand.ALWAYS),),
    )
    refresh = tensorizer.tensorize(persistent, source_step=0, timestep=1.0)
    assert refresh.targets.need_targets[0, 0, 0] == 1
    assert refresh.targets.source_targets[0, 0, 0] == 0


def test_trajectory_tensorizer_keeps_multiple_bindings_on_one_cell_distinct() -> None:
    base = make_trajectory()
    first = replace(
        base.binding_targets[0],
        need_id="latency-primary",
        first_need_step=1,
        executable_step=1,
        argument_steps={"key": 1, "scope": 1},
        arguments={"key": "latency_ms", "scope": "production"},
    )
    second = replace(
        first,
        need_id="latency-secondary",
        arguments={"key": "latency_p95", "scope": "production"},
    )
    example = replace(base, binding_targets=(first, second))
    tensorizer = ILLaDATrajectoryTensorizer(make_adapter(seed=126), TinyTokenizer())

    sample = tensorizer.tensorize(example, source_step=0, timestep=1.0)

    assert sample.targets.need_targets[0, 1, :2].tolist() == [1.0, 1.0]
    assert sample.targets.source_targets[0, 1, :2].tolist() == [0, 0]
    assert sample.targets.argument_presence_targets[0, 1, 0, :2].tolist() == [1.0, 1.0]
    assert sample.targets.argument_presence_targets[0, 1, 1, :2].tolist() == [1.0, 1.0]
    assert sample.targets.argument_mask[0, 1, 0, :2].all()
    assert sample.targets.argument_mask[0, 1, 1, :2].all()


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
    assert not training_batch.batch.display_padding_mask.any()
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


def test_trajectory_tensorizer_ignores_randomized_physical_slot_placement() -> None:
    adapter = make_adapter()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_trajectory()
    randomized = replace(
        base,
        example_id="train-randomized-slots",
        thought_targets=tuple(
            replace(target, slot=(3 - target.slot))
            for target in base.thought_targets
        ),
    )

    left = tensorizer.tensorize(base, source_step=0, timestep=1.0)
    right = tensorizer.tensorize(randomized, source_step=0, timestep=1.0)

    assert left.batch.thought_semantic.shape == right.batch.thought_semantic.shape
    assert torch.equal(left.batch.slot_occupancy, right.batch.slot_occupancy)
    assert torch.equal(left.targets.allocation_targets, right.targets.allocation_targets)
    assert torch.equal(left.targets.allocation_mask, right.targets.allocation_mask)
    assert torch.equal(left.targets.thought_mask, right.targets.thought_mask)
    assert torch.equal(left.targets.source_targets, right.targets.source_targets)


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


def test_validation_loss_is_deterministic_and_preserves_training_state() -> None:
    adapter = make_adapter(seed=801)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
            rollout_horizon=2,
            timestep_min=0.1,
            timestep_max=0.9,
            seed=17,
        ),
    )
    windows = balance_rollout_windows_by_semantic_task(
        trajectory_rollout_windows((make_rollout_trajectory(),), max_horizon=2)
    )
    state_before = trainer.state
    generator_before = trainer.generator.get_state().clone()
    shuffle_before = trainer.shuffle_rng.getstate()
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in adapter.named_parameters()
    }

    first = trainer.evaluate_rollout_windows(windows, seed=12345)
    second = trainer.evaluate_rollout_windows(windows, seed=12345)

    assert first == second
    assert first.optimizer_steps == 0
    assert first.transitions > 0
    assert first.mean_loss > 0.0
    assert trainer.state == state_before
    assert torch.equal(trainer.generator.get_state(), generator_before)
    assert trainer.shuffle_rng.getstate() == shuffle_before
    assert trainer.forward_model.training
    for name, parameter in adapter.named_parameters():
        assert torch.equal(parameter.detach(), parameters_before[name])


def test_training_validation_split_excludes_validation_and_test(tmp_path) -> None:
    from cid.cli import _load_train_and_validation_examples
    from cid.data import dump_jsonl

    base = make_trajectory()
    examples = (
        replace(
            base,
            example_id="split-train",
            metadata={**base.metadata, "split": "train"},
        ),
        replace(
            base,
            example_id="split-validation",
            metadata={**base.metadata, "split": "validation"},
        ),
        replace(
            base,
            example_id="split-test",
            metadata={**base.metadata, "split": "test"},
        ),
    )
    data = tmp_path / "mixed.jsonl"
    dump_jsonl(examples, data)

    training, validation = _load_train_and_validation_examples(
        data,
        validation_data_path=None,
        max_examples=None,
        max_validation_examples=None,
    )

    assert tuple(example.example_id for example in training) == ("split-train",)
    assert tuple(example.example_id for example in validation) == ("split-validation",)


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


def test_stage_a_ddp_two_rank_mixed_external_graph_smoke() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    worker = Path(__file__).with_name("ddp_stage_a_worker.py")
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
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


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
        optimizer_groups = stage_b_adamw_parameter_groups(
            adapter, backbone_lr_scale=0.5, weight_decay=0.01
        )
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-3)
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
        restored_optimizer_groups = stage_b_adamw_parameter_groups(
            restored_adapter, backbone_lr_scale=0.5, weight_decay=0.01
        )
        restored_fsdp = wrap_stage_b_fsdp(
            restored_adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        restored_optimizer = torch.optim.AdamW(restored_optimizer_groups, lr=1e-3)
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
        assert (checkpoint / "distributed" / ".metadata").is_file()
        assert not (checkpoint / "optimizer-rank-0000.pt").exists()
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
    assert (checkpoint / "rank-0000.pt").is_file()
    assert (checkpoint / "rank-0001.pt").is_file()
    assert not (checkpoint / "optimizer-rank-0000.pt").exists()


def test_stage_b_fsdp_checkpoint_reshards_optimizer_across_world_sizes(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    checkpoint = tmp_path / "elastic-checkpoint"
    save_worker = Path(__file__).with_name("fsdp_stage_b_worker.py")
    load_worker = Path(__file__).with_name("fsdp_stage_b_elastic_load_worker.py")
    env = os.environ.copy()
    env["CID_FSDP_SMOKE_DIR"] = str(checkpoint)

    saved = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(save_worker),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert saved.returncode == 0, saved.stdout + saved.stderr

    loaded = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=3",
            str(load_worker),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr


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
        lifecycle_features=torch.nn.functional.one_hot(
            torch.zeros((1, slots), dtype=torch.long),
            num_classes=adapter.config.num_lifecycles,
        ).to(torch.float32),
        slot_occupancy=torch.ones((1, slots, 1)),
        local_noise=torch.zeros((1, slots, 1)),
        display_ids=torch.tensor([[17, *([5] * (tensorizer.display_canvas_tokens - 1))]]),
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
    assert torch.equal(sample.batch.lifecycle_features, rollout.lifecycle_features)
    assert torch.equal(sample.batch.slot_occupancy, rollout.slot_occupancy)
    assert sample.batch.display_ids[0, 0] == 17
    assert sample.targets.display_ids[0, 0] != -100



def test_rollout_missing_target_cell_is_supervised_for_recovery_allocation() -> None:
    adapter = make_adapter(seed=111)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = make_rollout_trajectory()
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
    )
    rollout.slot_occupancy[0, 1, 0] = 0.0
    rollout.thought_semantic[0, 1].zero_()
    rollout.role_features[0, 1].zero_()
    rollout.lifecycle_features[0, 1].zero_()

    sample = tensorizer.tensorize(
        example, source_step=1, timestep=0.25, rollout_state=rollout
    )

    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.allocation_targets[0, 1] == 1.0
    assert sample.targets.lifecycle[0, 1] == -100


def test_rollout_uses_predicted_local_noise_for_next_thought_corruption() -> None:
    adapter = make_adapter(seed=127)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = make_rollout_trajectory()
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    rollout = CIDRolloutState(
        thought_semantic=torch.ones_like(teacher.batch.thought_semantic),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.ones_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
        display_noise_level=0.5,
    )

    sample = tensorizer.tensorize(
        example,
        source_step=1,
        timestep=0.0,
        rollout_state=rollout,
        generator=torch.Generator().manual_seed(31),
    )

    occupied = rollout.slot_occupancy[0, :, 0].bool()
    assert not torch.equal(
        sample.batch.thought_semantic[0, occupied],
        rollout.thought_semantic[0, occupied],
    )
    assert torch.equal(
        sample.batch.local_noise[0, occupied],
        torch.ones_like(sample.batch.local_noise[0, occupied]),
    )
    assert sample.batch.display_noise[0, :, 0].tolist() == pytest.approx(
        [0.5] * sample.batch.display_ids.shape[1]
    )


def test_existing_cell_noise_delta_uses_model_visible_corruption_level() -> None:
    adapter = make_adapter(seed=112)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())

    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=0.8)

    # c0's target noise is 0.3. The model sees local_noise=0.8 after corruption,
    # so runtime-compatible delta supervision is 0.3 - 0.8, not 0.3 - 0.5.
    assert sample.batch.local_noise[0, 0, 0] == pytest.approx(0.8)
    assert sample.targets.noise_delta[0, 0, 0] == pytest.approx(-0.5)
    assert sample.targets.revision_targets[0, 0] == int(cid_model.RevisionAction.STABILIZE)


def test_rollout_extra_occupied_slot_is_supervised_to_retire() -> None:
    adapter = make_adapter(seed=113)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = make_rollout_trajectory()
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
    )
    rollout.slot_occupancy[0, 3, 0] = 1.0
    rollout.lifecycle_features[0, 3, 0] = 1.0

    sample = tensorizer.tensorize(
        example, source_step=1, timestep=0.25, rollout_state=rollout
    )

    modeled = import_module("cid.lifecycle").MODELED_LIFECYCLES
    assert sample.targets.lifecycle[0, 3] == modeled.index(CellLifecycle.RETIRED)


def test_rollout_allocation_defaults_match_runtime_contract() -> None:
    materializer = cid_model.CIDMaterializerConfig()
    trainer = CIDTrainerConfig()

    assert trainer.rollout_allocation_threshold == materializer.allocation_threshold
    assert trainer.rollout_max_allocations_per_step == materializer.max_allocations_per_step
    assert materializer.max_allocations_per_step >= 20

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


def test_self_rollout_crops_collated_padding_to_each_trajectory_capacity() -> None:
    adapter = make_adapter(seed=114)
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        TinyTokenizer(),
        minimum_thought_slots=1,
    )
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            learning_rate=1e-3,
            micro_batch_size=2,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
            timestep_min=0.0,
            timestep_max=0.0,
        ),
    )
    wide = make_rollout_trajectory()
    narrow = replace(
        wide,
        example_id="rollout-narrow",
        binding_targets=(),
        grounding_targets=(),
        thought_targets=tuple(
            target for target in wide.thought_targets if target.cell_id == "c0"
        ),
    )
    windows = (
        CIDRolloutWindow(example=narrow, source_steps=(0, 1)),
        CIDRolloutWindow(example=wide, source_steps=(0, 1)),
    )

    loss_sum, raw_loss_sum, transitions = trainer.train_rollout_microbatch(
        windows,
        rollout_probability=1.0,
    )

    assert transitions == 4
    assert math.isfinite(loss_sum)
    assert math.isfinite(raw_loss_sum)


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


def test_rollout_sharding_repeats_singleton_bucket_across_all_ranks() -> None:
    window = CIDRolloutWindow(
        example=replace(make_trajectory(), example_id="singleton"),
        source_steps=(0,),
        loss_weight=1.0,
    )

    for legacy_resume_padding in (False, True):
        shards = tuple(
            shard_rollout_windows(
                (window,),
                world_size=4,
                rank=rank,
                seed=13,
                epoch=1,
                micro_batch_size=1,
                legacy_resume_padding=legacy_resume_padding,
            )
            for rank in range(4)
        )

        assert tuple(len(shard) for shard in shards) == (1, 1, 1, 1)
        assert all(shard[0].example.example_id == "singleton" for shard in shards)


def test_stage_b_batch_resolution_is_stable_across_four_and_six_ranks() -> None:
    assert stage_b_gradient_accumulation_steps(
        world_size=4, micro_batch_size=1, target_global_batch_size=32
    ) == 8
    assert stage_b_gradient_accumulation_steps(
        world_size=6, micro_batch_size=1, target_global_batch_size=32
    ) == 5
    assert stage_b_gradient_accumulation_steps(
        world_size=6,
        micro_batch_size=1,
        target_global_batch_size=32,
        explicit_steps=8,
    ) == 8


def test_stage_b_bucket_cursor_repartitions_remaining_windows_without_replay() -> None:
    base = make_trajectory()
    windows = tuple(
        CIDRolloutWindow(
            example=replace(base, example_id=f"elastic-{index}"),
            source_steps=(0,),
            loss_weight=1.0,
        )
        for index in range(10)
    )
    old_shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=4,
            rank=rank,
            seed=17,
            epoch=1,
            micro_batch_size=1,
        )
        for rank in range(4)
    )
    cursor = stage_b_consumed_windows_by_bucket(
        windows,
        old_shards[0],
        local_windows_seen=2,
        world_size=4,
    )
    consumed_ids = {
        window.example.example_id
        for shard in old_shards
        for window in shard[:2]
    }
    new_shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=2,
            rank=rank,
            seed=17,
            epoch=1,
            micro_batch_size=1,
            consumed_windows_by_bucket=cursor,
        )
        for rank in range(2)
    )
    remaining_ids = {
        window.example.example_id
        for shard in new_shards
        for window in shard
    }

    assert len(consumed_ids) == 8
    assert len(remaining_ids) == 2
    assert consumed_ids.isdisjoint(remaining_ids)
    assert consumed_ids | remaining_ids == {
        window.example.example_id for window in windows
    }


def test_stage_b_optimizer_step_count_matches_bucket_padding() -> None:
    example = make_trajectory()
    windows = tuple(
        CIDRolloutWindow(example=example, source_steps=(0, 1, 2), loss_weight=1.0)
        for _ in range(5)
    ) + tuple(
        CIDRolloutWindow(example=example, source_steps=(0,), loss_weight=2.0)
        for _ in range(3)
    )

    assert stage_b_optimizer_steps_per_epoch(
        windows,
        world_size=4,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
    ) == 4
    assert stage_b_optimizer_steps_per_epoch(
        windows,
        world_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
    ) == 2


def test_stage_b_adamw_groups_split_backbone_cid_and_no_decay() -> None:
    adapter = make_adapter()
    adapter.set_backbone_trainable(True)

    groups = stage_b_adamw_parameter_groups(
        adapter, backbone_lr_scale=0.5, weight_decay=0.01
    )
    by_name = {str(group["group_name"]): group for group in groups}

    assert set(by_name) == {
        "backbone-decay",
        "cid-decay",
        "cid-no-decay",
    }
    assert by_name["backbone-decay"]["lr_scale"] == pytest.approx(0.5)
    assert by_name["cid-decay"]["lr_scale"] == pytest.approx(1.0)
    assert by_name["backbone-decay"]["weight_decay"] == pytest.approx(0.01)
    assert by_name["cid-no-decay"]["weight_decay"] == pytest.approx(0.0)

    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad}
    assert grouped == trainable


def test_learning_rate_schedule_preserves_stage_b_group_scales() -> None:
    adapter = make_adapter()
    adapter.set_backbone_trainable(True)
    groups = stage_b_adamw_parameter_groups(
        adapter, backbone_lr_scale=0.5, weight_decay=0.01
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            warmup_steps=2,
            lr_decay_steps=6,
            min_learning_rate_ratio=0.1,
        ),
        optimizer=optimizer,
    )

    trainer._set_learning_rate_for_step(2)
    lrs = {str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups}
    assert lrs["backbone-decay"] == pytest.approx(5e-4)
    assert lrs["cid-decay"] == pytest.approx(1e-3)

    trainer._set_learning_rate_for_step(6)
    lrs = {str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups}
    assert lrs["backbone-decay"] == pytest.approx(5e-5)
    assert lrs["cid-decay"] == pytest.approx(1e-4)


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

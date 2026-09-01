from __future__ import annotations

import json
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
    TrajectoryExampleIndex,
    dump_jsonl,
    index_training_jsonl,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.state import CellLifecycle, CognitiveField, CognitiveRole

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
CIDRolloutBindingRoute = cid_model.CIDRolloutBindingRoute
CIDRolloutWindow = cid_model.CIDRolloutWindow
CIDRolloutRecoveryError = import_module("cid.model.training").CIDRolloutRecoveryError
balance_rollout_windows_by_semantic_task = cid_model.balance_rollout_windows_by_semantic_task
collate_training_steps = cid_model.collate_training_steps
load_cid_adapter_checkpoint = cid_model.load_cid_adapter_checkpoint
materialize_indexed_rollout_windows = cid_model.materialize_indexed_rollout_windows
load_stage_b_checkpoint = cid_model.load_stage_b_checkpoint
load_stage_b_model_checkpoint = cid_model.load_stage_b_model_checkpoint
load_stage_b_semantic_encoder = cid_model.load_stage_b_semantic_encoder
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
chunked_illada_rms_norm_forward = import_module("cid.model.illada")._chunked_illada_rms_norm_forward


def _slots_by_cell(snapshot: tuple[ThoughtTarget, ...]) -> dict[str, int]:
    return {cell.cell_id: cell.slot for cell in snapshot}


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


def test_trajectory_tensorizer_retired_tombstone_blocks_slot_without_pressure() -> None:
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
                semantic_text="Continue with new work.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.2,
                noise=0.2,
            ),
        ),
    )

    sample = tensorizer.tensorize(trajectory, source_step=0, timestep=1.0)
    retired_index = MODELED_LIFECYCLES.index(CellLifecycle.RETIRED)

    assert sample.batch.slot_occupancy[0, 0]
    assert sample.batch.lifecycle_features[0, 0, retired_index] == 1
    assert sample.input_runtime_cell_ids[0] == "c0"
    assert not sample.targets.allocation_mask[0, 0]
    assert sample.targets.allocation_targets[0, 1] == 1
    assert sample.targets.thought_mask[0, 1]


def test_trajectory_tensorizer_reclaims_retired_tombstone_under_pressure_after_grace() -> None:
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
                slot=3,
                cell_id="a-retired",
                semantic_text="Old state.",
                lifecycle=CellLifecycle.RETIRED,
            ),
            ThoughtTarget(step=0, slot=2, cell_id="b-live", semantic_text="B."),
            ThoughtTarget(step=0, slot=1, cell_id="c-live", semantic_text="C."),
            ThoughtTarget(step=0, slot=0, cell_id="d-live", semantic_text="D."),
            ThoughtTarget(step=1, slot=2, cell_id="b-live", semantic_text="B1."),
            ThoughtTarget(step=1, slot=1, cell_id="c-live", semantic_text="C1."),
            ThoughtTarget(step=1, slot=0, cell_id="d-live", semantic_text="D1."),
            ThoughtTarget(step=2, slot=2, cell_id="b-live", semantic_text="B2."),
            ThoughtTarget(step=2, slot=1, cell_id="c-live", semantic_text="C2."),
            ThoughtTarget(step=2, slot=0, cell_id="d-live", semantic_text="D2."),
            ThoughtTarget(step=2, slot=3, cell_id="e-new", semantic_text="E."),
        ),
    )

    before_reclaim = tensorizer.tensorize(trajectory, source_step=0, timestep=1.0)
    assert before_reclaim.batch.slot_occupancy.all()
    assert before_reclaim.input_runtime_cell_ids == ("c0", "c1", "c2", "c3")
    assert before_reclaim.input_retired_at == (("c0", 0),)
    assert not before_reclaim.targets.allocation_targets.any()

    after_reclaim = tensorizer.tensorize(trajectory, source_step=1, timestep=1.0)
    assert after_reclaim.input_runtime_cell_ids == ("c1", "c2", "c3", None)
    assert after_reclaim.batch.slot_occupancy[0, :3].all()
    assert not after_reclaim.batch.slot_occupancy[0, 3]
    assert after_reclaim.targets.allocation_targets[0, 3] == 1
    assert after_reclaim.targets.thought_mask[0, 3]

    rollout = CIDRolloutState(
        thought_semantic=before_reclaim.batch.thought_semantic.clone(),
        role_features=before_reclaim.batch.role_features.clone(),
        uncertainty=before_reclaim.batch.uncertainty.clone(),
        lifecycle_features=before_reclaim.batch.lifecycle_features.clone(),
        slot_occupancy=before_reclaim.batch.slot_occupancy.clone(),
        local_noise=before_reclaim.batch.local_noise.clone(),
        display_ids=before_reclaim.batch.display_ids.clone(),
        runtime_cell_ids=before_reclaim.input_runtime_cell_ids,
        next_cell_serial=before_reclaim.input_next_cell_serial,
        retired_at=before_reclaim.input_retired_at,
    )
    closed_loop = tensorizer.tensorize(
        trajectory,
        source_step=1,
        timestep=1.0,
        rollout_state=rollout,
    )
    assert closed_loop.input_runtime_cell_ids == after_reclaim.input_runtime_cell_ids
    assert torch.equal(closed_loop.batch.slot_occupancy, after_reclaim.batch.slot_occupancy)
    assert torch.equal(
        closed_loop.targets.allocation_targets,
        after_reclaim.targets.allocation_targets,
    )

    grace_blocked = tensorizer.tensorize(
        trajectory,
        source_step=1,
        timestep=1.0,
        rollout_state=replace(rollout, retired_at=(("c0", 1),)),
    )
    assert grace_blocked.batch.slot_occupancy.all()

    binding_pinned = tensorizer.tensorize(
        trajectory,
        source_step=1,
        timestep=1.0,
        rollout_state=replace(
            rollout,
            binding_routes=(
                CIDRolloutBindingRoute(
                    need_id="candidate",
                    target_cells=(ObjectRef.cell("c0"),),
                    target_display=(),
                    runtime_active=False,
                ),
            ),
        ),
    )
    assert binding_pinned.batch.slot_occupancy.all()



def test_teacher_reclamation_pins_match_runtime_binding_and_strong_link_rules() -> None:
    tensorizer = ILLaDATrajectoryTensorizer(make_adapter(), TinyTokenizer())
    field = CognitiveField.empty(capacity=4, width=1)
    field, retired_runtime_id = field.allocate()
    field, live_runtime_id = field.allocate()
    field = field.retire(retired_runtime_id)
    mapping = {"retired": retired_runtime_id, "live": live_runtime_id}

    binding = BindingTarget(
        need_id="need-retired",
        source="lookup",
        first_need_step=0,
        executable_step=0,
        arguments={"q": "x"},
        target_cells=(ObjectRef.cell("retired"),),
    )
    base = replace(
        make_trajectory(),
        binding_targets=(binding,),
        events=(),
        grounding_targets=(),
    )
    assert tensorizer._teacher_reclamation_pins(
        base, source_step=0, field=field, teacher_to_runtime=mapping
    ) == frozenset((retired_runtime_id,))

    observed = replace(
        base,
        events=(ExternalEvent(source="lookup", value="ok", arrival_step=0, arguments={"q": "x"}),),
    )
    assert (
        tensorizer._teacher_reclamation_pins(
            observed, source_step=0, field=field, teacher_to_runtime=mapping
        )
        == frozenset()
    )

    linked = replace(
        observed,
        grounding_targets=(
            GroundingTarget(
                step=0,
                cell_id="live",
                links=(
                    CognitiveLink(
                        relation=LinkRelation.DEPENDS_ON,
                        target=ObjectRef.cell("retired"),
                    ),
                ),
            ),
        ),
    )
    assert tensorizer._teacher_reclamation_pins(
        linked, source_step=0, field=field, teacher_to_runtime=mapping
    ) == frozenset((retired_runtime_id,))


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
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer(), minimum_thought_slots=1)

    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=0.5)

    assert sample.batch.thought_semantic.shape[1] == 8
    assert sample.batch.slot_occupancy.shape[1] == 8
    assert sample.targets.allocation_targets.shape[1] == 8


def test_rollout_preserves_runtime_cell_identity_instead_of_teacher_ids() -> None:
    base = make_trajectory()
    teacher_ids = {"c0": "plan", "c1": "lookup-result"}
    example = replace(
        base,
        source_descriptors=(),
        binding_targets=(),
        grounding_targets=(),
        thought_targets=tuple(
            replace(target, cell_id=teacher_ids[target.cell_id]) for target in base.thought_targets
        ),
    )
    adapter = make_adapter(seed=170)
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        TinyTokenizer(),
        minimum_thought_slots=1,
    )
    trainer = CIDTrainer(adapter, tensorizer, CIDTrainerConfig(timestep_min=0.0, timestep_max=0.0))
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)

    assert sample.input_runtime_cell_ids[0] == "c0"
    assert all(cell_id is None for cell_id in sample.input_runtime_cell_ids[1:])
    assert sample.input_next_cell_serial == 1

    output = adapter(sample.batch)
    output.allocation_logits.fill_(-10.0)
    output.allocation_logits[0, 1] = 10.0
    training_batch = collate_training_steps((sample,), pad_token_id=TinyTokenizer.pad_token_id)
    state = trainer._rollout_state_from_prediction(
        sample,
        training_batch,
        output,
        example=example,
        batch_index=0,
    )

    assert state.runtime_cell_ids[:2] == ("c0", "c1")
    assert all(cell_id is None for cell_id in state.runtime_cell_ids[2:])
    assert state.next_cell_serial == 2
    assert not ({"plan", "lookup-result"} & set(filter(None, state.runtime_cell_ids)))
    next_sample = tensorizer.tensorize(example, source_step=0, timestep=0.0, rollout_state=state)
    assert next_sample.input_runtime_cell_ids[:2] == ("c0", "c1")
    assert all(cell_id is None for cell_id in next_sample.input_runtime_cell_ids[2:])


def test_rollout_prediction_tracks_retirement_age() -> None:
    base = make_trajectory()
    example = replace(
        base,
        source_descriptors=(),
        binding_targets=(),
        grounding_targets=(),
    )
    adapter = make_adapter(seed=171)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(adapter, tensorizer, CIDTrainerConfig(timestep_min=0.0, timestep_max=0.0))
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    training_batch = collate_training_steps((sample,), pad_token_id=TinyTokenizer.pad_token_id)

    active_index = MODELED_LIFECYCLES.index(CellLifecycle.ACTIVE)
    retired_output = adapter(training_batch.batch)
    retired_output.allocation_logits.fill_(-10.0)
    retired_output.lifecycle_logits.fill_(-10.0)
    retired_output.lifecycle_logits[..., active_index] = 10.0
    retired_output.lifecycle_logits[0, 0].fill_(-10.0)
    retired_output.lifecycle_logits[
        0, 0, MODELED_LIFECYCLES.index(CellLifecycle.RETIRED)
    ] = 10.0
    retired_output.need_logits.fill_(-10.0)
    retired_state = trainer._rollout_state_from_prediction(
        sample,
        training_batch,
        retired_output,
        example=example,
        batch_index=0,
    )
    assert retired_state.retired_at == (("c0", sample.target_step),)


def test_cell_link_supervision_uses_target_cell_semantic_embedding() -> None:
    base = make_trajectory()
    solution = ThoughtTarget(
        step=0,
        slot=0,
        cell_id="solution",
        semantic_text="Solve the problem from the prompt.",
    )
    answer = ThoughtTarget(
        step=0,
        slot=1,
        cell_id="answer",
        semantic_text="Return the resulting answer.",
        roles={CognitiveRole.CONCLUSION: 1.0},
    )
    example = replace(
        base,
        source_descriptors=(),
        binding_targets=(),
        thought_targets=(solution, answer),
        grounding_targets=(
            GroundingTarget(
                step=0,
                cell_id="answer",
                links=(
                    CognitiveLink(
                        relation=LinkRelation.DERIVED_FROM,
                        target=ObjectRef.cell("solution"),
                    ),
                ),
            ),
        ),
        display_targets=(DisplayTarget(step=0, text="37"),),
    )
    adapter = make_adapter(seed=171)
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        TinyTokenizer(),
        minimum_thought_slots=1,
    )
    sample = tensorizer.tensorize(example, source_step=-1, timestep=0.0)
    target = tensorizer._thought_snapshot(example, 0)
    slots = _slots_by_cell(target)
    expected = tensorizer.text_encoder.encode_one(solution.semantic_text, detach=True)

    assert torch.allclose(
        sample.targets.link_target_embeddings[0, slots["answer"], 0],
        expected,
    )


def test_cell_link_supervision_ignores_runtime_unresolvable_retired_targets() -> None:
    base = make_trajectory()
    history0 = ThoughtTarget(
        step=0,
        slot=0,
        cell_id="history",
        semantic_text="Earlier supporting state.",
    )
    answer0 = ThoughtTarget(
        step=0,
        slot=1,
        cell_id="answer",
        semantic_text="Working answer.",
    )
    history1 = replace(history0, step=1, lifecycle=CellLifecycle.RETIRED)
    answer1 = replace(answer0, step=1, semantic_text="Final answer.")
    example = replace(
        base,
        source_descriptors=(),
        binding_targets=(),
        thought_targets=(history0, answer0, history1, answer1),
        grounding_targets=(
            GroundingTarget(
                step=1,
                cell_id="answer",
                links=(
                    CognitiveLink(
                        relation=LinkRelation.DERIVED_FROM,
                        target=ObjectRef.cell("history"),
                    ),
                ),
            ),
        ),
        display_targets=(
            DisplayTarget(step=0, text="Working."),
            DisplayTarget(step=1, text="Final."),
        ),
    )
    adapter = make_adapter(seed=172)
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        TinyTokenizer(),
        minimum_thought_slots=1,
    )
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    target = tensorizer._thought_snapshot(example, 1)
    slots = _slots_by_cell(target)

    answer_slot = slots["answer"]
    assert sample.targets.link_presence_mask[0, answer_slot].all()
    assert not sample.targets.link_presence_targets[0, answer_slot].any()
    assert not sample.targets.link_mask[0, answer_slot].any()


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

    with pytest.raises(ValueError, match="without retirement"):
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


def test_optimizer_step_rejects_nonfinite_gradient_before_parameter_update() -> None:
    adapter = make_adapter(seed=166)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(learning_rate=1e-3),
    )
    parameter = next(parameter for _, parameter in trainer._trainable)
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, float("inf"))
    trainer._pending_accumulation = 1
    trainer._pending_examples = 1
    trainer._pending_global_examples = 1

    with pytest.raises(FloatingPointError, match="non-finite CID gradient norm"):
        trainer._optimizer_step()

    assert torch.equal(parameter, before)
    assert parameter.grad is None
    assert trainer.pending_accumulation_steps == 0
    assert trainer.state.optimizer_steps == 0


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

    sample = tensorizer.tensorize(example, source_step=1, timestep=0.0, rollout_state=rollout)

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


def test_rollout_windows_reject_max_age_without_wall_clock_timeline() -> None:
    base = make_trajectory()
    binding = replace(
        base.binding_targets[0],
        freshness=FreshnessDemand.MAX_AGE,
        max_age_s=5.0,
    )
    example = replace(base, binding_targets=(binding,))

    with pytest.raises(ValueError, match="MAX_AGE freshness.*wall-clock timeline"):
        trajectory_rollout_windows((example,), max_horizon=3)


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


def test_trajectory_tensorizer_supervises_one_need_owner_with_multi_region_routes() -> None:
    base = make_trajectory()
    binding = replace(
        base.binding_targets[0],
        owner_cell_id="c1",
        target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
        target_display=(ObjectRef.display_span(0, 1),),
    )
    example = replace(base, binding_targets=(binding,))
    adapter = make_adapter(seed=126)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())

    sample = tensorizer.tensorize(example, source_step=0, timestep=1.0)

    assert sample.targets.need_targets[0, 1, 0] == 1
    assert sample.targets.need_targets[0, 0].sum() == 0
    assert sample.targets.need_target_cell_targets[0, 1, 0, 0] == 1
    assert sample.targets.need_target_cell_targets[0, 1, 0, 1] == 1
    assert sample.targets.need_target_cell_mask[0, 1, 0, :2].all()
    assert sample.targets.need_target_display_targets[0, 1, 0, 0] == 1
    assert sample.targets.need_target_display_targets[0, 1, 0, 1:].sum() == 0
    assert sample.targets.need_target_display_mask[0, 1, 0, :3].all()


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


def test_order_aware_text_encoder_distinguishes_reversed_token_order() -> None:
    adapter = make_adapter(seed=127)
    tokenizer = TinyTokenizer()
    legacy = ILLaDATextEncoder(adapter, tokenizer, pooling_mode="mean-v1")
    order_aware = ILLaDATextEncoder(adapter, tokenizer, pooling_mode="order-aware-v2")

    legacy_ab = legacy.encode_one("ab", detach=True)
    legacy_ba = legacy.encode_one("ba", detach=True)
    ordered_ab = order_aware.encode_one("ab", detach=True)
    ordered_ba = order_aware.encode_one("ba", detach=True)

    assert torch.allclose(legacy_ab, legacy_ba)
    assert not torch.allclose(ordered_ab, ordered_ba)


def test_fresh_semantic_pooling_defaults_to_order_aware_v2() -> None:
    adapter = make_adapter(seed=151)
    tokenizer = TinyTokenizer()

    assert ILLaDATextEncoder(adapter, tokenizer).pooling_mode == "order-aware-v2"
    assert CIDTrainerConfig().semantic_pooling == "order-aware-v2"


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
    restored = ILLaDATextEncoder.from_frozen_snapshot_state(
        adapter,
        tokenizer,
        snapshot.frozen_snapshot_state(),
        device="cpu",
    )
    assert torch.equal(before, after)
    assert torch.equal(before, restored.encode_one("stable target", detach=True))
    assert not torch.allclose(after.float(), live.float())
    assert snapshot.is_frozen_snapshot
    assert restored.is_frozen_snapshot
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
        events=(
            ExternalEvent(
                source="docs",
                value="37",
                arrival_step=2,
                arguments={"key": "latency_ms", "scope": "production"},
            ),
        ),
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
    from cid.lifecycle import MODELED_LIFECYCLES

    c1_slot = next(target.slot for target in waiting_step if target.cell_id == "c1")
    assert final.targets.lifecycle[0, c1_slot].item() == MODELED_LIFECYCLES.index(
        CellLifecycle.ACTIVE
    )


def test_closed_loop_missing_binding_does_not_inherit_teacher_waiting_equilibrium() -> None:
    adapter = make_adapter(seed=152)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    example = replace(
        base,
        thought_targets=tuple(
            replace(
                target,
                lifecycle=(
                    CellLifecycle.WAITING
                    if target.step == 1 and target.cell_id == "c1"
                    else target.lifecycle
                ),
            )
            for target in base.thought_targets
        ),
    )
    teacher = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    assert teacher.targets.convergence_targets.item() == 1.0

    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=teacher.batch.local_noise.clone(),
        display_ids=teacher.batch.display_ids.clone(),
        active_binding_ids=(),
        executable_binding_ids=(),
    )
    closed_loop = tensorizer.tensorize(example, source_step=0, timestep=0.0, rollout_state=rollout)

    assert closed_loop.targets.convergence_targets.item() == 0.0


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
            replace(target, slot=(3 - target.slot)) for target in base.thought_targets
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


def test_rollout_physical_micro_batch_preserves_logical_update() -> None:
    config = CIDTrainerConfig(
        learning_rate=1e-3,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        rollout_horizon=2,
        teacher_forcing_epochs=0,
        rollout_ramp_epochs=0,
        timestep_min=0.0,
        timestep_max=0.0,
        seed=17,
    )
    examples = (
        make_rollout_trajectory(),
        replace(make_rollout_trajectory(), example_id="train-physical-microbatch-2"),
    )
    windows = trajectory_rollout_windows(examples, max_horizon=2)

    baseline_adapter = make_adapter(seed=178)
    baseline = CIDTrainer(
        baseline_adapter,
        ILLaDATrajectoryTensorizer(baseline_adapter, TinyTokenizer()),
        config,
    )
    baseline_report = baseline.train_rollout_windows(windows, epochs=1, shuffle=False)

    split_adapter = make_adapter(seed=178)
    split = CIDTrainer(
        split_adapter,
        ILLaDATrajectoryTensorizer(split_adapter, TinyTokenizer()),
        config,
    )
    split_report = split.train_rollout_windows(
        windows,
        epochs=1,
        shuffle=False,
        physical_micro_batch_size=1,
    )

    assert split_report.transitions == baseline_report.transitions
    assert split_report.optimizer_steps == baseline_report.optimizer_steps
    assert split.state == baseline.state
    baseline_parameters = dict(baseline_adapter.named_parameters())
    split_parameters = dict(split_adapter.named_parameters())
    for name in baseline.trainable_parameter_names:
        torch.testing.assert_close(split_parameters[name], baseline_parameters[name])


def test_trainer_preserves_reduced_gradients_across_accumulation() -> None:
    config = CIDTrainerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        timestep_min=1.0,
        timestep_max=1.0,
        seed=7,
    )
    example = make_rollout_trajectory()

    baseline_adapter = make_adapter(seed=176)
    baseline = CIDTrainer(
        baseline_adapter,
        ILLaDATrajectoryTensorizer(baseline_adapter, TinyTokenizer()),
        config,
    )
    baseline.train_transition(example, 0)
    baseline.train_transition(example, 1)

    preserved_adapter = make_adapter(seed=176)
    preserved = CIDTrainer(
        preserved_adapter,
        ILLaDATrajectoryTensorizer(preserved_adapter, TinyTokenizer()),
        config,
        preserve_reduced_gradients=True,
    )
    preserved.train_transition(example, 0)
    preserved.train_transition(example, 1)

    baseline_parameters = dict(baseline_adapter.named_parameters())
    preserved_parameters = dict(preserved_adapter.named_parameters())
    for name in baseline.trainable_parameter_names:
        assert torch.allclose(baseline_parameters[name], preserved_parameters[name])
    assert preserved.state.optimizer_steps == 1


def test_preserved_reduced_gradients_survive_mid_accumulation_checkpoint(tmp_path) -> None:
    config = CIDTrainerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=3,
        timestep_min=1.0,
        timestep_max=1.0,
        seed=8,
    )
    example = make_rollout_trajectory()

    baseline_adapter = make_adapter(seed=177)
    baseline = CIDTrainer(
        baseline_adapter,
        ILLaDATrajectoryTensorizer(baseline_adapter, TinyTokenizer()),
        config,
        preserve_reduced_gradients=True,
    )
    for source_step in (0, 1, 0):
        baseline.train_transition(example, source_step)

    adapter = make_adapter(seed=177)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        config,
        preserve_reduced_gradients=True,
    )
    trainer.train_transition(example, 0)
    trainer.train_transition(example, 1)
    path = tmp_path / "preserved-mid-accumulation.pt"
    trainer.save_checkpoint(path)

    restored_adapter = make_adapter(seed=177)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        config,
        preserve_reduced_gradients=True,
    )
    restored.load_checkpoint(path)
    restored.train_transition(example, 0)

    baseline_parameters = dict(baseline_adapter.named_parameters())
    restored_parameters = dict(restored_adapter.named_parameters())
    for name in baseline.trainable_parameter_names:
        assert torch.allclose(baseline_parameters[name], restored_parameters[name])
    assert restored.state.optimizer_steps == 1


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


def test_legacy_checkpoint_without_semantic_pooling_resumes_as_mean_v1(tmp_path) -> None:
    adapter = make_adapter(seed=146)
    config = CIDTrainerConfig(
        timestep_min=1.0,
        timestep_max=1.0,
        semantic_pooling="mean-v1",
    )
    tokenizer = TinyTokenizer()
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(
            adapter,
            tokenizer,
            text_encoder=ILLaDATextEncoder(adapter, tokenizer, pooling_mode="mean-v1"),
        ),
        config,
    )
    path = tmp_path / "current.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["trainer_config"].pop("semantic_pooling")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    restored_adapter = make_adapter(seed=146)
    restored_tokenizer = TinyTokenizer()
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(
            restored_adapter,
            restored_tokenizer,
            text_encoder=ILLaDATextEncoder(
                restored_adapter,
                restored_tokenizer,
                pooling_mode="mean-v1",
            ),
        ),
        config,
    )
    restored.load_checkpoint(legacy)
    assert restored.config.semantic_pooling == "mean-v1"


def test_stage_b_ordinary_checkpoint_persists_and_restores_frozen_semantic_snapshot(
    tmp_path,
) -> None:
    adapter = make_adapter(seed=160)
    tokenizer = TinyTokenizer()
    snapshot = ILLaDATextEncoder.from_frozen_snapshot(
        adapter,
        tokenizer,
        device="cpu",
        dtype=torch.bfloat16,
    )
    expected = snapshot.encode_one("stable target", detach=True).clone()
    adapter.set_backbone_trainable(True)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, tokenizer, text_encoder=snapshot),
        CIDTrainerConfig(timestep_min=1.0, timestep_max=1.0),
    )
    checkpoint = tmp_path / "stage-b.pt"
    trainer.save_checkpoint(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "semantic_embedding_snapshot" in payload

    restored_adapter = make_adapter(seed=161)
    restored_tokenizer = TinyTokenizer()
    loaded_encoder = load_stage_b_semantic_encoder(
        restored_adapter,
        restored_tokenizer,
        checkpoint,
        device="cpu",
        embedding_device="cpu",
    )
    assert torch.equal(expected, loaded_encoder.encode_one("stable target", detach=True))

    wrong_snapshot = ILLaDATextEncoder.from_frozen_snapshot(
        restored_adapter,
        restored_tokenizer,
        device="cpu",
        dtype=torch.bfloat16,
    )
    restored_adapter.set_backbone_trainable(True)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(
            restored_adapter,
            restored_tokenizer,
            text_encoder=wrong_snapshot,
        ),
        trainer.config,
    )
    restored.load_checkpoint(checkpoint)
    assert torch.equal(
        expected,
        restored.tensorizer.text_encoder.encode_one("stable target", detach=True),
    )


def test_stage_b_init_rejects_semantic_pooling_mismatch(tmp_path) -> None:
    adapter = make_adapter(seed=152)
    tokenizer = TinyTokenizer()
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(
            adapter,
            tokenizer,
            text_encoder=ILLaDATextEncoder(adapter, tokenizer, pooling_mode="mean-v1"),
        ),
        CIDTrainerConfig(semantic_pooling="mean-v1"),
    )
    checkpoint = tmp_path / "stage-a-mean.pt"
    trainer.save_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="semantic pooling"):
        load_cid_adapter_checkpoint(
            make_adapter(seed=152),
            checkpoint,
            expected_semantic_pooling="order-aware-v2",
        )


def test_stage_a_checkpoint_rejects_previous_neural_contract(tmp_path) -> None:
    adapter = make_adapter(seed=145)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(timestep_min=1.0, timestep_max=1.0),
    )
    path = tmp_path / "current.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    payload["neural_contract_version"] = 2
    incompatible = tmp_path / "old-contract.pt"
    torch.save(payload, incompatible)

    restored_adapter = make_adapter(seed=145)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        trainer.config,
    )
    with pytest.raises(ValueError, match="neural contract"):
        restored.load_checkpoint(incompatible)
    with pytest.raises(ValueError, match="neural contract"):
        load_cid_adapter_checkpoint(restored_adapter, incompatible)


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
    free_first = trainer.evaluate_rollout_windows(windows, seed=12345, rollout_probability=1.0)
    free_second = trainer.evaluate_rollout_windows(windows, seed=12345, rollout_probability=1.0)

    assert first == second
    assert free_first == free_second
    assert first.optimizer_steps == 0
    assert first.transitions > 0
    assert first.mean_loss > 0.0
    assert first.component_mean_losses["intent"] >= 0.0
    assert {"need_tp", "need_fp", "need_fn", "need_tn"}.issubset(first.behavior_counts)
    assert free_first.transitions == first.transitions
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


def test_training_index_path_keeps_inline_validation_without_materializing_train(tmp_path) -> None:
    from cid.cli import _index_train_and_load_validation_examples

    base = make_trajectory()
    examples = (
        replace(
            base,
            example_id="indexed-train",
            metadata={**base.metadata, "split": "train"},
        ),
        replace(
            base,
            example_id="indexed-validation",
            metadata={**base.metadata, "split": "validation"},
        ),
        replace(
            base,
            example_id="indexed-test",
            metadata={**base.metadata, "split": "test"},
        ),
    )
    data = tmp_path / "mixed-indexed.jsonl"
    dump_jsonl(examples, data)

    training, validation = _index_train_and_load_validation_examples(
        data,
        validation_data_path=None,
        max_examples=None,
        max_validation_examples=None,
    )

    assert all(isinstance(example, TrajectoryExampleIndex) for example in training)
    assert tuple(example.example_id for example in training) == ("indexed-train",)
    assert tuple(example.example_id for example in validation) == ("indexed-validation",)


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


def test_stage_a_ddp_accumulation_matches_per_microbatch_reduction() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    worker = Path(__file__).with_name("ddp_stage_a_accumulation_worker.py")
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


def test_distributed_padding_only_rank_keeps_progress_collectives_aligned() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    worker = Path(__file__).with_name("distributed_progress_worker.py")
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


def test_stage_b_fsdp_validation_keeps_padding_only_rank_in_forward_collectives() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    worker = Path(__file__).with_name("fsdp_validation_padding_worker.py")
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


def test_padding_only_shard_returns_zero_train_and_validation_reports() -> None:
    base = make_trajectory()
    windows = (
        CIDRolloutWindow(
            example=replace(base, example_id="padding-only"),
            source_steps=(0,),
            loss_weight=1.0,
        ),
    )
    local_windows = shard_rollout_windows(
        windows,
        world_size=4,
        rank=3,
        seed=1,
        epoch=1,
        shuffle=False,
        micro_batch_size=1,
        zero_gradient_padding=True,
        portable_bucket_order=True,
    )
    assert local_windows and all(window.is_padding for window in local_windows)

    adapter = make_adapter(seed=301)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(micro_batch_size=1, timestep_min=0.0, timestep_max=0.0),
    )

    train_report = trainer.train_rollout_windows(
        local_windows,
        epochs=1,
        shuffle=False,
        preserve_order=True,
    )
    assert train_report.transitions == 0
    assert train_report.mean_loss == 0.0
    assert train_report.raw_mean_loss == 0.0
    assert len(train_report.component_mean_losses) == 24
    assert all(value == 0.0 for value in train_report.component_mean_losses.values())

    validation_report = trainer.evaluate_rollout_windows(local_windows, seed=2)
    assert validation_report.transitions == 0
    assert validation_report.mean_loss == 0.0
    assert validation_report.raw_mean_loss == 0.0
    assert len(validation_report.component_mean_losses) == 24
    assert len(validation_report.behavior_counts) == 14
    assert validation_report.behavior_counts["rollout_transition_total"] == 0.0
    assert validation_report.behavior_counts["rollout_recovery_failures"] == 0.0
    assert all(value == 0.0 for value in validation_report.component_mean_losses.values())
    assert all(value == 0.0 for value in validation_report.behavior_counts.values())


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
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["format_version"] == 6
        assert metadata["neural_contract_version"] == 3
        assert metadata["semantic_embedding_snapshot"]["file"] == "semantic-embedding.pt"

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
        inference_encoder = load_stage_b_model_checkpoint(
            inference_fsdp,
            inference_adapter,
            checkpoint,
            tokenizer=TinyTokenizer(),
            semantic_device="cpu",
            semantic_embedding_device="cpu",
        )
        assert inference_encoder is not None
        assert torch.equal(
            snapshot.encode_one("stable target", detach=True),
            inference_encoder.encode_one("stable target", detach=True),
        )
        assert torch.equal(
            saved_weight,
            inference_adapter.backbone.get_decoder().layers[0].projection.weight,
        )
        assert (checkpoint / "metadata.json").is_file()
        assert (checkpoint / "semantic-embedding.pt").is_file()
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

    sample = tensorizer.tensorize(example, source_step=1, timestep=0.25, rollout_state=rollout)

    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.allocation_targets[0, 1] == 1.0
    assert sample.targets.lifecycle[0, 1] == list(
        import_module("cid.lifecycle").MODELED_LIFECYCLES
    ).index(CellLifecycle.ACTIVE)


def test_rollout_recovery_allocation_uses_runtime_first_free_prefix() -> None:
    adapter = make_adapter(seed=113)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = replace(
        make_rollout_trajectory(),
        thought_targets=(
            ThoughtTarget(step=0, slot=0, cell_id="live", semantic_text="Live0"),
            ThoughtTarget(
                step=0,
                slot=1,
                cell_id="tombstone",
                semantic_text="Done",
                lifecycle=CellLifecycle.RETIRED,
            ),
            ThoughtTarget(step=1, slot=0, cell_id="live", semantic_text="Live1"),
            ThoughtTarget(step=1, slot=2, cell_id="new", semantic_text="New1"),
        ),
        binding_targets=(),
        grounding_targets=(),
        source_descriptors=(),
        display_targets=(DisplayTarget(step=0, text="37"), DisplayTarget(step=1, text="38")),
        target_display="38",
    )

    teacher = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    # Teacher runtime still carries the retired tombstone in slot 1, so its new cell is
    # allocated in slot 2. A closed-loop rollout may already have reclaimed that tombstone.
    assert teacher.input_runtime_cell_ids[:2] == ("c0", "c1")
    assert teacher.targets.allocation_targets[0, 2] == 1.0

    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
        runtime_cell_ids=teacher.input_runtime_cell_ids,
        next_cell_serial=teacher.input_next_cell_serial,
    )
    rollout.slot_occupancy[0, 1, 0] = 0.0
    rollout.thought_semantic[0, 1].zero_()
    rollout.role_features[0, 1].zero_()
    rollout.lifecycle_features[0, 1].zero_()
    ids = list(rollout.runtime_cell_ids)
    ids[1] = None
    rollout = replace(rollout, runtime_cell_ids=tuple(ids))

    sample = tensorizer.tensorize(example, source_step=0, timestep=0.25, rollout_state=rollout)

    assert sample.targets.allocation_targets[0, 1] == 1.0
    assert sample.targets.allocation_targets[0, 2] == 0.0
    assert sample.targets.thought_mask[0, 1]


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
    padding = sample.batch.display_padding_mask[0]
    assert sample.batch.display_noise[0, ~padding, 0].tolist() == pytest.approx(
        [0.5] * int((~padding).sum())
    )
    assert sample.batch.display_noise[0, padding, 0].tolist() == pytest.approx(
        [0.0] * int(padding.sum())
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

    sample = tensorizer.tensorize(example, source_step=1, timestep=0.25, rollout_state=rollout)

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
    assert report.optimizer_steps == 3
    assert tensorizer.rollout_flags == [False, True, True]
    assert trainer.state.epochs_completed == 1


def test_self_rollout_steps_inside_window_without_resetting_closed_loop_state() -> None:
    adapter = make_adapter(seed=154)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
            timestep_min=0.0,
            timestep_max=0.0,
        ),
    )
    windows = trajectory_rollout_windows((make_rollout_trajectory(),), max_horizon=2)

    report = trainer.train_rollout_windows(windows, epochs=1, shuffle=False)

    assert report.transitions == 3
    assert report.optimizer_steps == 3
    assert trainer.state.optimizer_steps == 3


def test_rollout_observation_requires_an_executable_predicted_binding() -> None:
    adapter = make_adapter(seed=148)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    example = replace(
        make_rollout_trajectory(),
        events=(
            ExternalEvent(
                source="docs",
                value="37",
                arrival_step=2,
                arguments={"key": "latency_ms", "scope": "production"},
            ),
        ),
    )

    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    assert teacher.batch.percept_memory.shape[1] == 1
    assert teacher.targets.need_targets.sum() == 0

    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
        executable_binding_ids=(),
    )
    closed_loop = tensorizer.tensorize(
        example,
        source_step=1,
        timestep=0.0,
        rollout_state=rollout,
    )

    assert closed_loop.batch.percept_memory.shape[1] == 0
    assert closed_loop.targets.need_targets.sum() > 0


def test_partial_argument_binding_does_not_unlock_full_teacher_observation() -> None:
    adapter = make_adapter(seed=151)
    base = make_trajectory()
    descriptor = dict(base.source_descriptors[0])
    descriptor["accepts_partial_arguments"] = True
    example = replace(base, source_descriptors=(descriptor,))
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    output = adapter(sample.batch)

    binding = example.binding_targets[0]
    target = tensorizer._thought_snapshot(example, 1)
    slots = _slots_by_cell(target)
    owner_slot = slots[binding.owner_cell.identifier]
    output.need_logits.fill_(-10.0)
    output.need_logits[0, owner_slot, 0].fill_(10.0)
    output.argument_presence_logits[0, owner_slot, 0].fill_(-10.0)
    output.argument_presence_logits[0, owner_slot, 0, 0] = 10.0
    output.argument_query[0, owner_slot, 0, 0] = tensorizer.text_encoder.encode_one(
        json.dumps(binding.arguments["key"], separators=(",", ":")), detach=True
    )

    active, executable, routes = tensorizer.predicted_binding_state(
        example,
        1,
        target_output_slots=slots,
        live_slots=torch.ones(sample.batch.thought_semantic.shape[1], dtype=torch.bool),
        display_active_length=sample.batch.display_ids.shape[1],
        output=output,
        batch_index=0,
    )

    assert active == ("need:c1:0",)
    assert executable == ()
    assert routes[0].runtime_active
    assert routes[0].replay_binding_id == binding.need_id
    assert dict(routes[0].arguments) == {"key": "latency_ms"}

    progressive = replace(
        example,
        events=(
            ExternalEvent(
                source=binding.source,
                value="candidate",
                arrival_step=1,
                arguments={"key": "latency_ms"},
            ),
            ExternalEvent(
                source=binding.source,
                value="37",
                arrival_step=1,
                arguments=binding.arguments,
            ),
        ),
    )
    rollout = CIDRolloutState(
        thought_semantic=sample.batch.thought_semantic.clone(),
        role_features=sample.batch.role_features.clone(),
        uncertainty=sample.batch.uncertainty.clone(),
        lifecycle_features=sample.batch.lifecycle_features.clone(),
        slot_occupancy=sample.batch.slot_occupancy.clone(),
        local_noise=sample.batch.local_noise.clone(),
        display_ids=sample.batch.display_ids.clone(),
        active_binding_ids=active,
        executable_binding_ids=executable,
        binding_routes=routes,
        equilibrium=True,
        quiescent=True,
    )

    full_only = replace(
        example,
        events=(
            ExternalEvent(
                source=binding.source,
                value="37",
                arrival_step=1,
                arguments=binding.arguments,
            ),
        ),
    )
    assert not tensorizer.rollout_external_progress(full_only, 1, rollout)
    projections, observation_steps, _ = tensorizer._available_percept_projections(
        full_only,
        1,
        rollout_state=rollout,
    )
    assert projections == ()
    assert observation_steps == ()

    output.argument_presence_logits[0, owner_slot, 0].fill_(-10.0)
    empty_active, empty_executable, empty_routes = tensorizer.predicted_binding_state(
        example,
        1,
        target_output_slots=slots,
        live_slots=torch.ones(sample.batch.thought_semantic.shape[1], dtype=torch.bool),
        display_active_length=sample.batch.display_ids.shape[1],
        output=output,
        batch_index=0,
    )
    assert empty_active == ("need:c1:0",)
    assert empty_executable == ()
    assert empty_routes[0].runtime_active
    assert empty_routes[0].replay_binding_id == binding.need_id
    assert dict(empty_routes[0].arguments) == {}
    empty_rollout = replace(
        rollout,
        active_binding_ids=empty_active,
        executable_binding_ids=empty_executable,
        binding_routes=empty_routes,
    )
    assert not tensorizer.rollout_external_progress(full_only, 1, empty_rollout)
    projections, observation_steps, _ = tensorizer._available_percept_projections(
        full_only,
        1,
        rollout_state=empty_rollout,
    )
    assert projections == ()
    assert observation_steps == ()

    assert tensorizer.rollout_external_progress(progressive, 1, rollout)
    projections, observation_steps, _ = tensorizer._available_percept_projections(
        progressive,
        1,
        rollout_state=rollout,
    )
    assert len(projections) == 1
    assert projections[0][1].value == "candidate"
    assert dict(observation_steps)[routes[0].need_id] == 1


def test_promoted_facts_use_runtime_need_ids_in_teacher_and_closed_loop() -> None:
    adapter = make_adapter(seed=167)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    binding = replace(base.binding_targets[0], owner_cell_id="c1")
    descriptor = dict(base.source_descriptors[0])
    descriptor["promote_results_to_fact"] = True
    event = ExternalEvent(
        source=binding.source,
        value="37",
        arrival_step=2,
        version="v1",
        arguments=binding.arguments,
    )
    example = replace(
        base,
        source_descriptors=(descriptor,),
        binding_targets=(binding,),
        events=(event,),
    )

    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)

    assert any(text.startswith("fact=binding:need:c1:0 |") for text in teacher.promoted_fact_texts)
    assert not any(
        text.startswith(f"fact=binding:{binding.need_id} |") for text in teacher.promoted_fact_texts
    )

    runtime_need_id = "need:c1:3"
    rollout = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=teacher.batch.local_noise.clone(),
        display_ids=teacher.batch.display_ids.clone(),
        active_binding_ids=(runtime_need_id,),
        executable_binding_ids=(runtime_need_id,),
        binding_routes=(
            CIDRolloutBindingRoute(
                need_id=runtime_need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
                freshness=binding.freshness,
                source=binding.source,
                arguments=binding.arguments,
                replay_binding_id=binding.need_id,
            ),
        ),
    )
    closed_loop = tensorizer.tensorize(
        example,
        source_step=1,
        timestep=0.0,
        rollout_state=rollout,
    )

    assert any(
        text.startswith(f"fact=binding:{runtime_need_id} |")
        for text in closed_loop.promoted_fact_texts
    )
    assert not any(
        text.startswith(f"fact=binding:{binding.need_id} |")
        for text in closed_loop.promoted_fact_texts
    )


def test_predicted_binding_requires_live_owner_and_carries_predicted_routes() -> None:
    adapter = make_adapter(seed=153)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_trajectory()
    binding = replace(
        base.binding_targets[0],
        target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
        target_display=(ObjectRef.display_span(0, 1),),
        owner_cell_id="c1",
    )
    example = replace(base, binding_targets=(binding,))
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    output = adapter(sample.batch)
    target = tensorizer._thought_snapshot(example, 1)
    slots = _slots_by_cell(target)
    owner_slot = slots["c1"]
    other_slot = slots["c0"]
    output.need_logits.fill_(-10.0)
    output.need_logits[0, owner_slot, 0].fill_(10.0)
    output.need_target_cell_logits[0, owner_slot, 0].fill_(-10.0)
    output.need_target_cell_logits[0, owner_slot, 0, other_slot] = 10.0
    output.need_target_display_logits[0, owner_slot, 0].fill_(-10.0)
    output.need_target_display_logits[0, owner_slot, 0, 1:3] = 10.0
    output.argument_presence_logits[0, owner_slot, 0].fill_(10.0)
    output.refresh_logits[0, owner_slot, 0].fill_(-10.0)
    output.refresh_logits[
        0, owner_slot, 0, tuple(FreshnessDemand).index(FreshnessDemand.ALWAYS)
    ] = 10.0
    for argument_index, name in enumerate(("key", "scope")):
        output.argument_query[0, owner_slot, 0, argument_index] = (
            tensorizer.text_encoder.encode_one(
                json.dumps(binding.arguments[name], separators=(",", ":")), detach=True
            )
        )
    live = torch.zeros(sample.batch.thought_semantic.shape[1], dtype=torch.bool)
    live[owner_slot] = True
    live[other_slot] = True

    active, executable, routes = tensorizer.predicted_binding_state(
        example,
        1,
        target_output_slots=slots,
        live_slots=live,
        display_active_length=sample.batch.display_ids.shape[1],
        output=output,
        batch_index=0,
    )
    assert active == ("need:c1:0",)
    assert executable == ("need:c1:0",)
    assert routes[0].replay_binding_id == binding.need_id
    assert routes[0].target_cells == (ObjectRef.cell("c1"), ObjectRef.cell("c0"))
    assert routes[0].target_display == (ObjectRef.display_span(1, 3),)
    assert routes[0].freshness is FreshnessDemand.ALWAYS
    assert routes[0].runtime_active

    live[owner_slot] = False
    active, executable, routes = tensorizer.predicted_binding_state(
        example,
        1,
        target_output_slots=slots,
        live_slots=live,
        display_active_length=sample.batch.display_ids.shape[1],
        output=output,
        batch_index=0,
    )
    assert active == executable == routes == ()


def test_closed_loop_freshness_controls_which_observation_is_visible() -> None:
    adapter = make_adapter(seed=155)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    binding = base.binding_targets[0]
    example = replace(
        base,
        events=(
            ExternalEvent(
                source=binding.source,
                value="37",
                arrival_step=1,
                arguments=binding.arguments,
                version="v1",
            ),
            ExternalEvent(
                source=binding.source,
                value="38",
                arrival_step=2,
                arguments=binding.arguments,
                version="v2",
            ),
        ),
    )
    teacher = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    route = cid_model.CIDRolloutBindingRoute(
        need_id=binding.need_id,
        target_cells=binding.target_cells,
        target_display=binding.target_display,
        freshness=FreshnessDemand.ONCE,
    )
    state = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=teacher.batch.local_noise.clone(),
        display_ids=teacher.batch.display_ids.clone(),
        active_binding_ids=(binding.need_id,),
        executable_binding_ids=(binding.need_id,),
        binding_routes=(route,),
        binding_observation_steps=((binding.need_id, 1),),
    )

    once, once_steps, _ = tensorizer._available_percept_projections(example, 2, rollout_state=state)
    always, always_steps, _ = tensorizer._available_percept_projections(
        example,
        2,
        rollout_state=replace(
            state,
            binding_routes=(replace(route, freshness=FreshnessDemand.ALWAYS),),
        ),
    )

    assert once[0][1].value == "37"
    assert dict(once_steps)[binding.need_id] == 1
    assert always[0][1].value == "38"
    assert dict(always_steps)[binding.need_id] == 2


def test_closed_loop_quiescence_waits_until_external_progress() -> None:
    adapter = make_adapter(seed=156)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    binding = base.binding_targets[0]
    example = replace(
        base,
        events=(
            ExternalEvent(
                source=binding.source,
                value="38",
                arrival_step=3,
                arguments=binding.arguments,
            ),
        ),
    )
    teacher = tensorizer.tensorize(base, source_step=0, timestep=0.0)
    state = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=teacher.batch.local_noise.clone(),
        display_ids=teacher.batch.display_ids.clone(),
        active_binding_ids=(binding.need_id,),
        executable_binding_ids=(binding.need_id,),
        binding_routes=(
            cid_model.CIDRolloutBindingRoute(
                need_id=binding.need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
                freshness=FreshnessDemand.ONCE,
            ),
        ),
        equilibrium=True,
        quiescent=True,
    )
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            rollout_horizon=3,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
        ),
    )
    window = CIDRolloutWindow(example=example, source_steps=(0, 1, 2))

    before_use, before_execute, before_advance = trainer._rollout_step_plan(
        (window,), 1, [state], rollout_probability=1.0
    )
    at_use, at_execute, at_advance = trainer._rollout_step_plan(
        (window,), 2, [state], rollout_probability=1.0
    )

    assert before_use == (False,)
    assert before_execute == (True,)
    assert before_advance == (False,)
    assert at_use == (True,)
    assert at_execute == (True,)
    assert at_advance == (True,)


def test_always_freshness_requires_terminal_refresh_before_finalization() -> None:
    adapter = make_adapter(seed=158)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    binding = base.binding_targets[0]
    example = replace(
        base,
        events=(
            ExternalEvent(
                source=binding.source,
                value="37",
                arrival_step=1,
                arguments=binding.arguments,
                version="v1",
            ),
        ),
    )
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    runtime_need_id = "need:c1:0"
    prior_state = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=torch.zeros_like(teacher.batch.local_noise),
        display_ids=teacher.batch.display_ids.clone(),
        active_binding_ids=(runtime_need_id,),
        executable_binding_ids=(runtime_need_id,),
        binding_routes=(
            cid_model.CIDRolloutBindingRoute(
                need_id=runtime_need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
                freshness=FreshnessDemand.ALWAYS,
                source=binding.source,
                replay_binding_id=binding.need_id,
            ),
        ),
        binding_observation_steps=((runtime_need_id, 1),),
    )
    sample = tensorizer.tensorize(
        example,
        source_step=1,
        timestep=0.0,
        rollout_state=prior_state,
    )
    training_batch = collate_training_steps((sample,), pad_token_id=1)
    output = adapter(training_batch.batch)
    slots = _slots_by_cell(tensorizer._thought_snapshot(example, 2))
    owner_slot = slots[binding.owner_cell.identifier]
    active_index = tuple(import_module("cid.lifecycle").MODELED_LIFECYCLES).index(
        CellLifecycle.ACTIVE
    )
    output.lifecycle_logits.fill_(-10.0)
    output.lifecycle_logits[..., active_index] = 10.0
    output.allocation_logits.fill_(-10.0)
    output.need_logits.fill_(-10.0)
    output.need_logits[0, owner_slot, 0] = 10.0
    output.argument_presence_logits[0, owner_slot, 0].fill_(10.0)
    for argument_index, name in enumerate(("key", "scope")):
        output.argument_query[0, owner_slot, 0, argument_index] = (
            tensorizer.text_encoder.encode_one(
                json.dumps(binding.arguments[name], separators=(",", ":")), detach=True
            )
        )
    output.refresh_logits[0, owner_slot, 0].fill_(-10.0)
    output.refresh_logits[
        0, owner_slot, 0, tuple(FreshnessDemand).index(FreshnessDemand.ALWAYS)
    ] = 10.0
    output.convergence_logits.fill_(20.0)
    display_logits = torch.full_like(output.display_logits, -20.0)
    display_logits.scatter_(-1, training_batch.batch.display_ids.unsqueeze(-1), 20.0)
    output.display_logits = display_logits

    trainer = CIDTrainer(adapter, tensorizer, CIDTrainerConfig())
    state = trainer._rollout_state_from_prediction(
        sample,
        training_batch,
        output,
        example=example,
        batch_index=0,
    )

    assert state.converged
    assert state.quiescent
    assert not state.terminal
    assert state.binding_routes[0].freshness is FreshnessDemand.ALWAYS
    assert tensorizer.rollout_external_progress(example, 3, state)

    _, _, validated = tensorizer._available_percept_projections(example, 3, rollout_state=state)
    assert state.binding_routes[0].need_id in validated

    streaming_descriptor = dict(example.source_descriptors[0])
    streaming_descriptor.update(streamable=True, dynamic=True, versioned=True)
    streaming_example = replace(example, source_descriptors=(streaming_descriptor,))
    streaming_state = trainer._rollout_state_from_prediction(
        sample,
        training_batch,
        output,
        example=streaming_example,
        batch_index=0,
    )

    assert streaming_state.converged
    assert streaming_state.terminal
    assert not streaming_state.quiescent


def test_free_rollout_keeps_supervision_after_model_terminal_decision() -> None:
    class EarlyConvergingModel(nn.Module):
        def __init__(self, adapter) -> None:
            super().__init__()
            self.adapter = adapter

        def forward(self, batch):
            output = self.adapter(batch)
            output.convergence_logits = torch.full_like(output.convergence_logits, 20.0)
            output.need_logits = torch.full_like(output.need_logits, -20.0)
            display_logits = torch.full_like(output.display_logits, -20.0)
            display_logits.scatter_(
                -1,
                batch.display_ids.unsqueeze(-1),
                20.0,
            )
            output.display_logits = display_logits
            return output

    adapter = make_adapter(seed=157)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            timestep_min=0.0,
            timestep_max=0.0,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
        ),
        forward_model=EarlyConvergingModel(adapter),
    )
    window = CIDRolloutWindow(
        example=make_rollout_trajectory(),
        source_steps=(0, 1),
    )

    teacher_forced = trainer.evaluate_rollout_windows((window,), seed=123, rollout_probability=0.0)
    free_rollout = trainer.evaluate_rollout_windows((window,), seed=123, rollout_probability=1.0)

    assert teacher_forced.transitions == 2
    assert free_rollout.transitions == 2
    assert free_rollout.behavior_counts["convergence_total"] == 2.0
    assert free_rollout.behavior_counts["convergence_correct"] < 2.0


def test_free_rollout_records_unrecoverable_recovery_without_crashing(monkeypatch) -> None:
    class NonConvergingModel(nn.Module):
        def __init__(self, adapter) -> None:
            super().__init__()
            self.adapter = adapter

        def forward(self, batch):
            output = self.adapter(batch)
            output.convergence_logits = torch.full_like(output.convergence_logits, -20.0)
            output.need_logits = torch.full_like(output.need_logits, -20.0)
            return output

    adapter = make_adapter(seed=158)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            timestep_min=0.0,
            timestep_max=0.0,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
        ),
        forward_model=NonConvergingModel(adapter),
    )
    original_tensorize = tensorizer.tensorize

    def fail_closed_loop_recovery(example, source_step, **kwargs):
        if source_step == 1 and kwargs.get("rollout_state") is not None:
            raise CIDRolloutRecoveryError("teacher transition exceeds runtime allocation limit")
        return original_tensorize(example, source_step, **kwargs)

    monkeypatch.setattr(tensorizer, "tensorize", fail_closed_loop_recovery)
    window = CIDRolloutWindow(
        example=make_rollout_trajectory(),
        source_steps=(0, 1),
    )

    report = trainer.evaluate_rollout_windows((window,), seed=123, rollout_probability=1.0)

    assert report.transitions == 1
    assert report.behavior_counts["rollout_transition_total"] == 1.0
    assert report.behavior_counts["rollout_recovery_failures"] == 1.0
    assert report.behavior_counts["convergence_total"] == 1.0


def test_training_rollout_masks_unrecoverable_recovery_without_crashing(monkeypatch) -> None:
    adapter = make_adapter(seed=160)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            timestep_min=0.0,
            timestep_max=0.0,
            rollout_horizon=2,
            teacher_forcing_epochs=0,
            rollout_ramp_epochs=0,
        ),
    )
    original_tensorize = tensorizer.tensorize

    def fail_closed_loop_recovery(example, source_step, **kwargs):
        if (
            example.example_id == "recovery-fail"
            and source_step == 1
            and kwargs.get("rollout_state") is not None
        ):
            raise CIDRolloutRecoveryError("teacher transition exceeds runtime allocation limit")
        return original_tensorize(example, source_step, **kwargs)

    monkeypatch.setattr(tensorizer, "tensorize", fail_closed_loop_recovery)
    base = make_rollout_trajectory()
    windows = (
        CIDRolloutWindow(
            example=replace(base, example_id="recovery-fail"),
            source_steps=(0, 1),
        ),
        CIDRolloutWindow(
            example=replace(base, example_id="recovery-ok"),
            source_steps=(0, 1),
        ),
    )

    report = trainer.train_rollout_windows(
        windows,
        epochs=1,
        shuffle=False,
        physical_micro_batch_size=1,
    )

    assert report.transitions == 3
    assert trainer.state.epochs_completed == 1


def test_valid_example_accumulation_ignores_masked_rows() -> None:
    adapter = make_adapter(seed=159)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(
        adapter,
        tensorizer,
        CIDTrainerConfig(
            learning_rate=1e-3,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            timestep_min=0.0,
            timestep_max=0.0,
        ),
    )
    example = make_trajectory()
    samples = (
        tensorizer.tensorize(example, source_step=0, timestep=0.0),
        tensorizer.tensorize(
            replace(example, example_id="masked-row"),
            source_step=0,
            timestep=0.0,
        ),
    )

    masked_losses, _, _ = trainer._forward_backward(samples, sample_mask=(True, False))
    assert trainer.state.optimizer_steps == 0
    assert trainer._pending_examples == 1
    assert trainer._pending_global_examples == 1
    with torch.no_grad():
        single_batch = collate_training_steps((samples[0],), pad_token_id=1)
        single_losses = cid_loss(adapter(single_batch.batch), single_batch.targets)
    assert masked_losses.total.detach().float().item() == pytest.approx(
        single_losses.total.detach().float().item(),
        rel=1e-6,
        abs=1e-7,
    )

    trainer._forward_backward(samples, sample_mask=(True, False))
    assert trainer.state.optimizer_steps == 1
    assert trainer._pending_examples == 0
    assert trainer._pending_global_examples == 0


def test_sparse_component_losses_are_invariant_to_microbatch_packing() -> None:
    adapter = make_adapter(seed=161)
    adapter.eval()
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    rich = make_trajectory()
    plain = replace(
        make_trajectory(),
        example_id="plain-no-external-supervision",
        protected_facts={},
        source_descriptors=(),
        binding_targets=(),
        grounding_targets=(),
    )
    samples = (
        tensorizer.tensorize(rich, source_step=0, timestep=0.0),
        tensorizer.tensorize(plain, source_step=0, timestep=0.0),
    )

    with torch.no_grad():
        individual = []
        for sample in samples:
            batch = collate_training_steps((sample,), pad_token_id=1)
            individual.append(cid_loss(adapter(batch.batch), batch.targets))
        joint_batch = collate_training_steps(samples, pad_token_id=1)
        joint = cid_loss(adapter(joint_batch.batch), joint_batch.targets)

    component_names = (
        "thought",
        "convergence",
        "allocation",
        "display",
        "roles",
        "uncertainty",
        "noise",
        "lifecycle",
        "intent",
        "source",
        "need_cell_route",
        "need_display_route",
        "argument_presence",
        "argument_ground",
        "revision",
        "refresh",
        "anchor_presence",
        "anchor_kind",
        "anchor_ground",
        "link_presence",
        "link_relation",
        "link_target_kind",
        "link_ground",
    )
    for name in component_names:
        expected = 0.5 * (getattr(individual[0], name) + getattr(individual[1], name))
        assert getattr(joint, name) == pytest.approx(expected, rel=1e-5, abs=1e-6), name
    expected_total = 0.5 * (individual[0].total + individual[1].total)
    assert joint.total == pytest.approx(expected_total, rel=1e-5, abs=1e-6)


def test_stage_a_legacy_resume_repair_is_limited_to_pre_v2_checkpoints() -> None:
    needs_repair = import_module("cid.cli")._stage_a_needs_legacy_resume_repair

    assert needs_repair(data_order_version=1, windows_seen_in_epoch=3)
    assert not needs_repair(data_order_version=2, windows_seen_in_epoch=3)
    assert not needs_repair(data_order_version=3, windows_seen_in_epoch=3)
    assert not needs_repair(data_order_version=4, windows_seen_in_epoch=3)
    assert not needs_repair(data_order_version=1, windows_seen_in_epoch=0)


def test_stage_a_legacy_order_upgrades_only_at_completed_epoch_boundary() -> None:
    completed_version = import_module("cid.cli")._stage_a_completed_epoch_data_order_version

    assert completed_version(1) == 4
    assert completed_version(2) == 4
    assert completed_version(3) == 4
    assert completed_version(4) == 4
    assert completed_version(5) == 5


def test_closed_loop_diffusion_reset_requires_actual_replayed_observation() -> None:
    adapter = make_adapter(seed=160)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    binding = base.binding_targets[0]
    example = replace(
        base,
        events=(
            ExternalEvent(
                source=binding.source,
                value="38",
                arrival_step=2,
                arguments=binding.arguments,
            ),
        ),
    )
    teacher = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    runtime_need_id = "need:c1:0"
    base_state = CIDRolloutState(
        thought_semantic=teacher.batch.thought_semantic.clone(),
        role_features=teacher.batch.role_features.clone(),
        uncertainty=teacher.batch.uncertainty.clone(),
        lifecycle_features=teacher.batch.lifecycle_features.clone(),
        slot_occupancy=teacher.batch.slot_occupancy.clone(),
        local_noise=teacher.batch.local_noise.clone(),
        display_ids=teacher.batch.display_ids.clone(),
        display_noise_level=0.5,
        diffusion_step=3,
        active_binding_ids=(runtime_need_id,),
        executable_binding_ids=(runtime_need_id,),
    )
    unmatched = replace(
        base_state,
        binding_routes=(
            cid_model.CIDRolloutBindingRoute(
                need_id=runtime_need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
                source=binding.source,
                work_key='docs:{"key":"wrong"}',
                replay_binding_id=None,
            ),
        ),
    )
    matched = replace(
        base_state,
        binding_routes=(
            cid_model.CIDRolloutBindingRoute(
                need_id=runtime_need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
                source=binding.source,
                replay_binding_id=binding.need_id,
            ),
        ),
    )

    no_progress = tensorizer.tensorize(
        example, source_step=1, timestep=0.0, rollout_state=unmatched
    )
    with_progress = tensorizer.tensorize(
        example, source_step=1, timestep=0.0, rollout_state=matched
    )

    assert no_progress.diffusion_step == 3
    assert no_progress.next_diffusion_step == 4
    assert no_progress.binding_observation_steps == ()
    assert with_progress.diffusion_step == 0
    assert with_progress.next_diffusion_step == 1
    assert dict(with_progress.binding_observation_steps)[runtime_need_id] == 2


def test_wrong_source_need_has_runtime_binding_consequences() -> None:
    adapter = make_adapter(seed=161)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    base = make_rollout_trajectory()
    wrong_source = {
        "name": "search",
        "description": "wrong source",
        "arguments": (),
    }
    example = replace(
        base,
        source_descriptors=(*base.source_descriptors, wrong_source),
    )
    sample = tensorizer.tensorize(example, source_step=0, timestep=0.0)
    training_batch = collate_training_steps((sample,), pad_token_id=1)
    output = adapter(training_batch.batch)
    slots = _slots_by_cell(tensorizer._thought_snapshot(example, 1))
    owner_slot = slots[base.binding_targets[0].owner_cell.identifier]
    output.allocation_logits.fill_(-20.0)
    output.allocation_logits[0, owner_slot] = 20.0
    output.need_logits.fill_(-20.0)
    output.need_logits[0, owner_slot, 0] = 20.0
    output.source_logits.fill_(-20.0)
    output.source_logits[0, owner_slot, 0, 1] = 20.0
    output.convergence_logits.fill_(20.0)
    output.lifecycle_logits.fill_(-20.0)
    active_index = tuple(import_module("cid.lifecycle").MODELED_LIFECYCLES).index(
        CellLifecycle.ACTIVE
    )
    output.lifecycle_logits[..., active_index] = 20.0
    display_logits = torch.full_like(output.display_logits, -20.0)
    display_logits.scatter_(-1, training_batch.batch.display_ids.unsqueeze(-1), 20.0)
    output.display_logits = display_logits

    trainer = CIDTrainer(adapter, tensorizer, CIDTrainerConfig())
    state = trainer._rollout_state_from_prediction(
        sample, training_batch, output, example=example, batch_index=0
    )

    assert state.binding_routes
    assert state.binding_routes[0].source == "search"
    assert state.binding_routes[0].replay_binding_id is None
    assert state.binding_routes[0].runtime_active
    assert state.binding_routes[0].need_id in state.executable_binding_ids
    assert state.quiescent
    assert not state.terminal


def test_stage_a_fp32_cid_modules_keep_low_precision_backbone_frozen() -> None:
    adapter = make_adapter(seed=149).to(dtype=torch.bfloat16)
    adapter.set_backbone_trainable(False)
    adapter.set_cid_modules_dtype(torch.float32)

    assert {parameter.dtype for parameter in adapter.backbone.parameters()} == {torch.bfloat16}
    trainable = tuple(parameter for parameter in adapter.parameters() if parameter.requires_grad)
    assert trainable
    assert {parameter.dtype for parameter in trainable} == {torch.float32}


def test_stage_a_checkpoint_rejects_different_dataset_identity(tmp_path) -> None:
    adapter = make_adapter(seed=150)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    checkpoint = tmp_path / "stage-a.pt"
    trainer.save_checkpoint(checkpoint, dataset_sha256="dataset-a")

    restored_adapter = make_adapter(seed=150)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    with pytest.raises(ValueError, match="dataset SHA-256"):
        restored.load_checkpoint(checkpoint, expected_dataset_sha256="dataset-b")


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
        thought_targets=tuple(target for target in wide.thought_targets if target.cell_id == "c0"),
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


def test_indexed_rollout_sharding_matches_eager_examples(tmp_path) -> None:
    examples = tuple(
        replace(
            make_rollout_trajectory(),
            example_id=f"indexed-{index}",
            metadata={
                "semantic_task_id": f"task-{index // 2}",
                "training_weight": 1.0 + (index // 2),
            },
        )
        for index in range(6)
    )
    path = tmp_path / "training.jsonl"
    dump_jsonl(examples, path)

    eager = balance_rollout_windows_by_semantic_task(
        trajectory_rollout_windows(examples, max_horizon=3)
    )
    indexed_examples = index_training_jsonl(path)
    indexed = balance_rollout_windows_by_semantic_task(
        trajectory_rollout_windows(indexed_examples, max_horizon=3)
    )

    for rank in range(2):
        kwargs = dict(
            world_size=2,
            rank=rank,
            seed=77,
            epoch=2,
            shuffle=True,
            micro_batch_size=1,
            length_aware=True,
            zero_gradient_padding=True,
            portable_bucket_order=True,
        )
        eager_shard = shard_rollout_windows(eager, **kwargs)
        indexed_shard = shard_rollout_windows(indexed, **kwargs)
        eager_signature = tuple(
            (
                window.example.example_id,
                window.source_steps,
                window.loss_weight,
                window.is_padding,
            )
            for window in eager_shard
        )
        indexed_signature = tuple(
            (
                window.example.example_id,
                window.source_steps,
                window.loss_weight,
                window.is_padding,
            )
            for window in indexed_shard
        )
        assert indexed_signature == eager_signature

        materialized = materialize_indexed_rollout_windows(path, indexed_shard)
        assert tuple(window.example.example_id for window in materialized) == tuple(
            window.example.example_id for window in eager_shard
        )
        assert all(isinstance(window.example, TrajectoryExample) for window in materialized)


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
            portable_bucket_order=True,
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
        if legacy_resume_padding:
            assert not any(shard[0].is_padding for shard in shards)
        else:
            assert sum(not shard[0].is_padding for shard in shards) == 1


def test_stage_b_batch_resolution_is_stable_across_four_and_six_ranks() -> None:
    assert (
        stage_b_gradient_accumulation_steps(
            world_size=4, micro_batch_size=1, target_global_batch_size=32
        )
        == 8
    )
    assert (
        stage_b_gradient_accumulation_steps(
            world_size=6, micro_batch_size=1, target_global_batch_size=32
        )
        == 5
    )
    assert (
        stage_b_gradient_accumulation_steps(
            world_size=6,
            micro_batch_size=1,
            target_global_batch_size=32,
            explicit_steps=8,
        )
        == 8
    )


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
            portable_bucket_order=True,
        )
        for rank in range(4)
    )
    cursor = stage_b_consumed_windows_by_bucket(
        windows,
        old_shards[0],
        local_windows_seen=2,
        world_size=4,
    )
    consumed_ids = {window.example.example_id for shard in old_shards for window in shard[:2]}
    new_shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=2,
            rank=rank,
            seed=17,
            epoch=1,
            micro_batch_size=1,
            consumed_windows_by_bucket=cursor,
            portable_bucket_order=True,
        )
        for rank in range(2)
    )
    remaining_ids = {window.example.example_id for shard in new_shards for window in shard}

    assert len(consumed_ids) == 8
    assert len(remaining_ids) == 2
    assert consumed_ids.isdisjoint(remaining_ids)
    assert consumed_ids | remaining_ids == {window.example.example_id for window in windows}


def test_stage_b_portable_cursor_handles_padded_bucket_prefix_exactly() -> None:
    base = make_trajectory()
    windows = tuple(
        CIDRolloutWindow(
            example=replace(base, example_id=f"padded-elastic-{index}"),
            source_steps=(0,),
            loss_weight=1.0,
        )
        for index in range(5)
    )
    old_shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=4,
            rank=rank,
            seed=11,
            epoch=1,
            micro_batch_size=1,
            length_aware=True,
            zero_gradient_padding=True,
            portable_bucket_order=True,
        )
        for rank in range(4)
    )

    first_slice_ids = {
        shard[0].example.example_id for shard in old_shards if not shard[0].is_padding
    }
    assert len(first_slice_ids) == 4
    cursor = stage_b_consumed_windows_by_bucket(
        windows,
        old_shards[0],
        local_windows_seen=1,
        world_size=4,
    )
    assert tuple(cursor.values()) == (4,)

    new_shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=2,
            rank=rank,
            seed=11,
            epoch=1,
            micro_batch_size=1,
            consumed_windows_by_bucket=cursor,
            length_aware=True,
            zero_gradient_padding=True,
            portable_bucket_order=True,
        )
        for rank in range(2)
    )
    remaining_ids = {
        window.example.example_id
        for shard in new_shards
        for window in shard
        if not window.is_padding
    }
    assert len(remaining_ids) == 1
    assert first_slice_ids.isdisjoint(remaining_ids)
    assert first_slice_ids | remaining_ids == {window.example.example_id for window in windows}


def test_stage_b_optimizer_step_count_matches_bucket_padding() -> None:
    example = make_trajectory()
    windows = tuple(
        CIDRolloutWindow(example=example, source_steps=(0, 1, 2), loss_weight=1.0) for _ in range(5)
    ) + tuple(
        CIDRolloutWindow(example=example, source_steps=(0,), loss_weight=2.0) for _ in range(3)
    )

    assert (
        stage_b_optimizer_steps_per_epoch(
            windows,
            world_size=4,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            portable_bucket_order=True,
        )
        == 2
    )
    assert (
        stage_b_optimizer_steps_per_epoch(
            windows,
            world_size=4,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            portable_bucket_order=True,
        )
        == 1
    )


def test_stage_b_adamw_groups_split_backbone_cid_and_no_decay() -> None:
    adapter = make_adapter()
    adapter.set_backbone_trainable(True)

    groups = stage_b_adamw_parameter_groups(adapter, backbone_lr_scale=0.5, weight_decay=0.01)
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
    groups = stage_b_adamw_parameter_groups(adapter, backbone_lr_scale=0.5, weight_decay=0.01)
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


def test_revision_target_tracks_logical_state_change_not_sampled_diffusion_noise() -> None:
    adapter = make_adapter(seed=144)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())

    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=0.1)

    assert sample.targets.noise_delta[0, 0, 0] == pytest.approx(0.2)
    assert sample.targets.revision_targets[0, 0] == int(cid_model.RevisionAction.STABILIZE)


def test_self_rollout_applies_runtime_stable_reopen_gate() -> None:
    adapter = make_adapter(seed=146)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    trainer = CIDTrainer(adapter, tensorizer, CIDTrainerConfig(timestep_min=0.0, timestep_max=0.0))
    example = make_rollout_trajectory()
    sample = tensorizer.tensorize(example, source_step=1, timestep=0.0)
    training_batch = collate_training_steps((sample,), pad_token_id=1)
    output = adapter(training_batch.batch)
    stable_index = list(import_module("cid.lifecycle").MODELED_LIFECYCLES).index(
        CellLifecycle.STABLE
    )
    active_index = list(import_module("cid.lifecycle").MODELED_LIFECYCLES).index(
        CellLifecycle.ACTIVE
    )

    # c0 is STABLE at source step 1. An ACTIVE proposal without REOPEN must be blocked.
    output.lifecycle_logits[0, 0].fill_(-10.0)
    output.lifecycle_logits[0, 0, active_index] = 10.0
    output.revision_logits[0, 0].fill_(-10.0)
    output.revision_logits[0, 0, int(cid_model.RevisionAction.KEEP)] = 10.0
    blocked = trainer._rollout_state_from_prediction(
        sample, training_batch, output, example=example, batch_index=0
    )
    assert int(blocked.lifecycle_features[0, 0].argmax()) == stable_index

    output.revision_logits[0, 0].fill_(-10.0)
    output.revision_logits[0, 0, int(cid_model.RevisionAction.REOPEN)] = 10.0
    reopened = trainer._rollout_state_from_prediction(
        sample, training_batch, output, example=example, batch_index=0
    )
    assert int(reopened.lifecycle_features[0, 0].argmax()) == active_index


def test_training_display_masks_physical_tail_after_visible_eos() -> None:
    adapter = make_adapter(seed=147)
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())

    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=0.0)

    eos_positions = torch.nonzero(
        sample.batch.display_ids[0] == tensorizer.eos_token_id, as_tuple=False
    ).flatten()
    assert eos_positions.numel() == 1
    eos = int(eos_positions[0])
    assert not sample.batch.display_padding_mask[0, : eos + 1].any()
    assert sample.batch.display_padding_mask[0, eos + 1 :].all()
    assert sample.batch.display_noise[0, eos + 1 :].eq(0).all()

    collated = collate_training_steps((sample,), pad_token_id=1)
    assert torch.equal(collated.batch.display_padding_mask[0], sample.batch.display_padding_mask[0])


def test_binding_lifecycle_cells_marks_once_observation_targets_available() -> None:
    from cid.contracts import FreshnessDemand
    from cid.data import BindingTarget, ExternalEvent, TrajectoryExample
    from cid.grounding import ObjectRef
    from cid.model.training import ILLaDATrajectoryTensorizer

    binding = BindingTarget(
        need_id="need:lookup",
        source="lookup",
        first_need_step=0,
        executable_step=0,
        arguments={"key": "x"},
        argument_steps={"key": 0},
        confidence=1.0,
        freshness=FreshnessDemand.ONCE,
        max_age_s=None,
        target_cells=(ObjectRef.cell("answer"),),
        target_display=(),
        owner_cell_id="answer",
    )
    example = TrajectoryExample(
        example_id="lifecycle-available-once",
        prompt="p",
        target_display="a",
        source_descriptors=(),
        events=(ExternalEvent(source="lookup", value="v", arrival_step=2, arguments={"key": "x"}),),
        binding_targets=(binding,),
    )
    waiting, available = ILLaDATrajectoryTensorizer._binding_lifecycle_cells(example, 2)
    assert waiting == set()
    assert available == {"answer"}


def test_rollout_sharding_length_aware_limits_padding_spread() -> None:
    base = make_trajectory()
    windows = tuple(
        CIDRolloutWindow(
            example=replace(
                base,
                example_id=f"length-{index}",
                prompt="x" * (64 + index * 64),
            ),
            source_steps=(0,),
            loss_weight=1.0,
        )
        for index in range(32)
    )
    shards = tuple(
        shard_rollout_windows(
            windows,
            world_size=4,
            rank=rank,
            seed=9,
            epoch=2,
            micro_batch_size=2,
            length_aware=True,
        )
        for rank in range(4)
    )

    for shard in shards:
        assert len(shard) == 8
        for start in range(0, len(shard), 2):
            lengths = [len(window.example.prompt) for window in shard[start : start + 2]]
            assert max(lengths) - min(lengths) <= 7 * 64


def test_stage_a_checkpoint_without_data_order_marker_loads_legacy_order(tmp_path) -> None:
    adapter = make_adapter(seed=123)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("data_order_version")
    torch.save(payload, checkpoint)

    restored_adapter = make_adapter(seed=123)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    restored.load_checkpoint(checkpoint)

    assert restored.data_order_version == 1


def test_checkpoint_allows_equivalent_batch_geometry_at_clean_epoch_boundary(tmp_path) -> None:
    adapter = make_adapter(seed=321)
    config = CIDTrainerConfig(micro_batch_size=4, gradient_accumulation_steps=6)
    trainer = CIDTrainer(adapter, ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()), config)
    trainer.state = CIDTrainerState(
        transitions_seen=96, optimizer_steps=1, epochs_completed=1, rollout_windows_seen_in_epoch=0
    )
    checkpoint = tmp_path / "epoch-boundary.pt"
    trainer.save_checkpoint(checkpoint)

    restored_adapter = make_adapter(seed=321)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(micro_batch_size=8, gradient_accumulation_steps=3),
    )
    restored.load_checkpoint(checkpoint)

    assert restored.state.epochs_completed == 1
    assert restored.state.rollout_windows_seen_in_epoch == 0


def test_checkpoint_rejects_world_size_change_mid_epoch(tmp_path) -> None:
    adapter = make_adapter(seed=323)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    trainer.state = CIDTrainerState(rollout_windows_seen_in_epoch=8)
    checkpoint = tmp_path / "mid-epoch-world-size.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["world_size"] == 1
    payload["world_size"] = 4
    torch.save(payload, checkpoint)

    restored_adapter = make_adapter(seed=323)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    with pytest.raises(ValueError, match="world size does not match"):
        restored.load_checkpoint(checkpoint)


def test_checkpoint_rejects_legacy_partial_epoch_without_world_size(tmp_path) -> None:
    adapter = make_adapter(seed=324)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    trainer.state = CIDTrainerState(rollout_windows_seen_in_epoch=8)
    checkpoint = tmp_path / "legacy-mid-epoch.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("world_size")
    torch.save(payload, checkpoint)

    restored_adapter = make_adapter(seed=324)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    with pytest.raises(ValueError, match="does not record its world size"):
        restored.load_checkpoint(checkpoint)


def test_checkpoint_allows_world_size_change_at_epoch_boundary(tmp_path) -> None:
    adapter = make_adapter(seed=325)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    trainer.state = CIDTrainerState(epochs_completed=1, rollout_windows_seen_in_epoch=0)
    checkpoint = tmp_path / "epoch-boundary-world-size.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["world_size"] = 4
    torch.save(payload, checkpoint)

    restored_adapter = make_adapter(seed=325)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(),
    )
    restored.load_checkpoint(checkpoint)
    assert restored.state.epochs_completed == 1
    assert restored.state.rollout_windows_seen_in_epoch == 0


def test_checkpoint_rejects_batch_geometry_change_mid_epoch(tmp_path) -> None:
    adapter = make_adapter(seed=322)
    trainer = CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(micro_batch_size=4, gradient_accumulation_steps=6),
    )
    trainer.state = CIDTrainerState(rollout_windows_seen_in_epoch=8)
    checkpoint = tmp_path / "mid-epoch.pt"
    trainer.save_checkpoint(checkpoint)

    restored_adapter = make_adapter(seed=322)
    restored = CIDTrainer(
        restored_adapter,
        ILLaDATrajectoryTensorizer(restored_adapter, TinyTokenizer()),
        CIDTrainerConfig(micro_batch_size=8, gradient_accumulation_steps=3),
    )
    with pytest.raises(ValueError, match="trainer configuration"):
        restored.load_checkpoint(checkpoint)

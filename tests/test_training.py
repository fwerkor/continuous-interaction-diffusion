from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from cid.data import BindingTarget, DisplayTarget, GroundingTarget, ThoughtTarget, TrajectoryExample
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.model.encoding import ILLaDATextEncoder
from cid.state import CellLifecycle, CognitiveRole

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
dist = import_module("torch.distributed")
cid_model = import_module("cid.model")

ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
ILLaDACIDConfig = cid_model.ILLaDACIDConfig
ILLaDATrajectoryTensorizer = cid_model.ILLaDATrajectoryTensorizer
CIDTrainer = cid_model.CIDTrainer
CIDTrainerConfig = cid_model.CIDTrainerConfig
CIDTrainerState = cid_model.CIDTrainerState
collate_training_steps = cid_model.collate_training_steps
load_cid_adapter_checkpoint = cid_model.load_cid_adapter_checkpoint
load_stage_b_checkpoint = cid_model.load_stage_b_checkpoint
save_stage_b_checkpoint = cid_model.save_stage_b_checkpoint
shard_transitions = cid_model.shard_transitions
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

    assert report.transitions == 4
    assert report.optimizer_steps == 1
    assert trainer.state.transitions_seen == 4


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

    assert report.transitions == 2
    assert report.optimizer_steps == 1
    assert trainer.state.transitions_seen == 2
    assert trainer.state.optimizer_steps == 1
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
        transitions_seen=2,
        optimizer_steps=1,
        epochs_completed=2,
    )
    inference_parameters = dict(inference_adapter.named_parameters())
    for name in trainer.trainable_parameter_names:
        assert torch.equal(original_parameters[name], inference_parameters[name])


def test_transition_sharding_is_balanced_deterministic_and_complete() -> None:
    examples = tuple(
        replace(make_trajectory(), example_id=f"train-{index}") for index in range(5)
    )
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
    assert {len(shard) for shard in shards} == {2}
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

        assert report.transitions == 2
        assert report.optimizer_steps == 2
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

        assert report.transitions == 1
        assert report.optimizer_steps == 1
        assert not torch.equal(
            before,
            adapter.backbone.get_decoder().layers[0].projection.weight,
        )
        assert all(parameter.requires_grad for parameter in adapter.backbone.parameters())

        checkpoint = tmp_path / "stage-b-checkpoint"
        saved_weight = (
            adapter.backbone.get_decoder().layers[0].projection.weight.detach().clone()
        )
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
        assert (checkpoint / "metadata.json").is_file()
        assert (checkpoint / "rank-0000.pt").is_file()
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
    assert (checkpoint / ".metadata").is_file()

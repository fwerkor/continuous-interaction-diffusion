from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cid.contracts import FreshnessDemand
from cid.data import ThoughtTarget, TrajectoryExample
from cid.grounding import AnchorKind, LinkRelation, ObjectKind, ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.diffusion import CIDDiffusionScheduler
from cid.model.encoding import ILLaDATextEncoder, stable_text
from cid.model.illada import ILLADA_MASK_TOKEN_ID, ILLaDACIDAdapter
from cid.model.losses import CIDLoss, CIDTargets, cid_loss
from cid.model.materialize import RevisionAction
from cid.model.tensors import CIDTensorBatch
from cid.state import CognitiveRole


@dataclass(slots=True)
class CIDTrainingStep:
    example_id: str
    source_step: int
    target_step: int
    batch: CIDTensorBatch
    targets: CIDTargets


@dataclass(slots=True)
class CIDTrainingBatch:
    example_ids: tuple[str, ...]
    source_steps: tuple[int, ...]
    target_steps: tuple[int, ...]
    batch: CIDTensorBatch
    targets: CIDTargets


def collate_training_steps(
    steps: tuple[CIDTrainingStep, ...],
    *,
    pad_token_id: int,
) -> CIDTrainingBatch:
    if not steps:
        raise ValueError("cannot collate an empty CID training batch")
    if any(step.batch.thought_semantic.shape[0] != 1 for step in steps):
        raise ValueError("CIDTrainingStep inputs must each contain exactly one example")

    prompt_ids, prompt_padding_mask = _pad_2d(
        tuple(step.batch.prompt_ids for step in steps),
        pad_value=pad_token_id,
    )
    display_ids, display_padding_mask = _pad_2d(
        tuple(step.batch.display_ids for step in steps),
        pad_value=pad_token_id,
    )
    display_noise, _ = _pad_3d(tuple(step.batch.display_noise for step in steps))
    fact_memory, fact_padding_mask = _pad_3d(
        tuple(step.batch.fact_memory for step in steps)
    )
    percept_memory, percept_padding_mask = _pad_3d(
        tuple(step.batch.percept_memory for step in steps)
    )
    source_memory, source_padding_mask = _pad_3d(
        tuple(step.batch.source_memory for step in steps)
    )
    display_labels, _ = _pad_2d(
        tuple(step.targets.display_ids for step in steps),
        pad_value=-100,
    )

    batch = CIDTensorBatch(
        thought_semantic=torch.cat(tuple(step.batch.thought_semantic for step in steps)),
        role_features=torch.cat(tuple(step.batch.role_features for step in steps)),
        uncertainty=torch.cat(tuple(step.batch.uncertainty for step in steps)),
        local_noise=torch.cat(tuple(step.batch.local_noise for step in steps)),
        slot_occupancy=torch.cat(tuple(step.batch.slot_occupancy for step in steps)),
        prompt_ids=prompt_ids,
        display_ids=display_ids,
        display_noise=display_noise,
        fact_memory=fact_memory,
        percept_memory=percept_memory,
        source_memory=source_memory,
        prompt_padding_mask=prompt_padding_mask,
        display_padding_mask=display_padding_mask,
        fact_padding_mask=fact_padding_mask,
        percept_padding_mask=percept_padding_mask,
        source_padding_mask=source_padding_mask,
    )
    targets = CIDTargets(
        thought_semantic=_cat_targets(steps, "thought_semantic"),
        thought_mask=_cat_targets(steps, "thought_mask"),
        allocation_targets=_cat_targets(steps, "allocation_targets"),
        allocation_mask=_cat_targets(steps, "allocation_mask"),
        display_ids=display_labels,
        role_targets=_cat_targets(steps, "role_targets"),
        uncertainty=_cat_targets(steps, "uncertainty"),
        noise_delta=_cat_targets(steps, "noise_delta"),
        lifecycle=_cat_targets(steps, "lifecycle"),
        need_targets=_cat_targets(steps, "need_targets"),
        source_targets=_cat_targets(steps, "source_targets"),
        argument_presence_targets=_cat_targets(steps, "argument_presence_targets"),
        argument_presence_mask=_cat_targets(steps, "argument_presence_mask"),
        argument_embeddings=_cat_targets(steps, "argument_embeddings"),
        argument_mask=_cat_targets(steps, "argument_mask"),
        revision_targets=_cat_targets(steps, "revision_targets"),
        refresh_targets=_cat_targets(steps, "refresh_targets"),
        anchor_presence_targets=_cat_targets(steps, "anchor_presence_targets"),
        anchor_presence_mask=_cat_targets(steps, "anchor_presence_mask"),
        anchor_kind_targets=_cat_targets(steps, "anchor_kind_targets"),
        anchor_embeddings=_cat_targets(steps, "anchor_embeddings"),
        anchor_mask=_cat_targets(steps, "anchor_mask"),
        link_presence_targets=_cat_targets(steps, "link_presence_targets"),
        link_presence_mask=_cat_targets(steps, "link_presence_mask"),
        link_relation_targets=_cat_targets(steps, "link_relation_targets"),
        link_target_kind_targets=_cat_targets(steps, "link_target_kind_targets"),
        link_target_embeddings=_cat_targets(steps, "link_target_embeddings"),
        link_mask=_cat_targets(steps, "link_mask"),
    )
    return CIDTrainingBatch(
        example_ids=tuple(step.example_id for step in steps),
        source_steps=tuple(step.source_step for step in steps),
        target_steps=tuple(step.target_step for step in steps),
        batch=batch,
        targets=targets,
    )


@dataclass(frozen=True, slots=True)
class CIDTrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    timestep_min: float = 0.05
    timestep_max: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if not 0.0 <= self.timestep_min <= self.timestep_max <= 1.0:
            raise ValueError("timestep range must satisfy 0 <= min <= max <= 1")


@dataclass(frozen=True, slots=True)
class CIDTrainerState:
    transitions_seen: int = 0
    optimizer_steps: int = 0
    epochs_completed: int = 0


@dataclass(frozen=True, slots=True)
class CIDTrainReport:
    transitions: int
    optimizer_steps: int
    mean_loss: float


class CIDTrainer:
    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tensorizer: ILLaDATrajectoryTensorizer,
        config: CIDTrainerConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        forward_model: torch.nn.Module | None = None,
        gradient_clipper: Callable[[float], Tensor | float] | None = None,
    ) -> None:
        if tensorizer.adapter is not adapter:
            raise ValueError("trainer and trajectory tensorizer must share the same adapter")
        self.adapter = adapter
        self.forward_model = forward_model or adapter
        self.gradient_clipper = gradient_clipper
        self.tensorizer = tensorizer
        self.config = config or CIDTrainerConfig()
        self._trainable = tuple(
            (name, parameter)
            for name, parameter in adapter.named_parameters()
            if parameter.requires_grad
        )
        if not self._trainable:
            raise ValueError("trainer requires at least one trainable parameter")
        self.optimizer = optimizer or torch.optim.AdamW(
            (parameter for _, parameter in self._trainable),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        device = adapter.input_embeddings.weight.device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config.seed)
        self.shuffle_rng = random.Random(self.config.seed)
        self.state = CIDTrainerState()
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.pad_token_id = getattr(tensorizer.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("training tokenizer must define pad_token_id")
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._trainable)

    def train_transition(self, example: TrajectoryExample, source_step: int) -> CIDLoss:
        return self.train_microbatch(((example, source_step),))

    def train_microbatch(
        self,
        transitions: tuple[tuple[TrajectoryExample, int], ...],
    ) -> CIDLoss:
        if not transitions:
            raise ValueError("training micro-batch cannot be empty")
        self.forward_model.train()
        samples = tuple(
            self.tensorizer.tensorize(
                example,
                source_step,
                timestep=self._sample_timestep(),
                generator=self.generator,
            )
            for example, source_step in transitions
        )
        training_batch = collate_training_steps(
            samples,
            pad_token_id=int(self.pad_token_id),
        )
        output = self.forward_model(training_batch.batch)
        losses = cid_loss(output, training_batch.targets)
        if not bool(torch.isfinite(losses.total)):
            names = ", ".join(training_batch.example_ids)
            raise FloatingPointError(
                f"non-finite CID loss for training micro-batch: {names}"
            )
        batch_size = len(transitions)
        (losses.total * batch_size).backward()
        self._pending_accumulation += 1
        self._pending_examples += batch_size
        self.state = CIDTrainerState(
            transitions_seen=self.state.transitions_seen + batch_size,
            optimizer_steps=self.state.optimizer_steps,
            epochs_completed=self.state.epochs_completed,
        )
        if self._pending_accumulation >= self.config.gradient_accumulation_steps:
            self._optimizer_step()
        return losses

    def train_examples(
        self,
        examples: tuple[TrajectoryExample, ...],
        *,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> CIDTrainReport:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        transitions = trajectory_transitions(examples)
        if not transitions:
            raise ValueError("training data contains no adjacent thought transitions")
        return self.train_transitions(transitions, epochs=epochs, shuffle=shuffle)

    def train_transitions(
        self,
        transitions: tuple[tuple[TrajectoryExample, int], ...],
        *,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> CIDTrainReport:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not transitions:
            raise ValueError("training data contains no adjacent thought transitions")
        losses: list[float] = []
        start_optimizer_steps = self.state.optimizer_steps
        for _ in range(epochs):
            order = list(transitions)
            if shuffle:
                self.shuffle_rng.shuffle(order)
            for start in range(0, len(order), self.config.micro_batch_size):
                microbatch = tuple(order[start : start + self.config.micro_batch_size])
                loss = self.train_microbatch(microbatch)
                losses.extend([float(loss.total.detach().float())] * len(microbatch))
            self.state = CIDTrainerState(
                transitions_seen=self.state.transitions_seen,
                optimizer_steps=self.state.optimizer_steps,
                epochs_completed=self.state.epochs_completed + 1,
            )
        self.flush()
        return CIDTrainReport(
            transitions=len(losses),
            optimizer_steps=self.state.optimizer_steps - start_optimizer_steps,
            mean_loss=sum(losses) / len(losses),
        )

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed)
        self.shuffle_rng.seed(seed)

    def flush(self) -> None:
        if self._pending_accumulation:
            self._optimizer_step()

    def local_progress_state(self) -> dict[str, Any]:
        if self._pending_accumulation:
            raise RuntimeError("flush accumulated gradients before exporting trainer state")
        return {
            "trainer_config": asdict(self.config),
            "trainer_state": asdict(self.state),
            "generator_state": self.generator.get_state().cpu(),
            "shuffle_state": self.shuffle_rng.getstate(),
        }

    def restore_local_progress_state(self, state: Mapping[str, Any]) -> None:
        if state["trainer_config"] != asdict(self.config):
            raise ValueError("checkpoint trainer configuration does not match this trainer")
        trainer_state = state["trainer_state"]
        self.state = CIDTrainerState(
            transitions_seen=int(trainer_state["transitions_seen"]),
            optimizer_steps=int(trainer_state["optimizer_steps"]),
            epochs_completed=int(trainer_state.get("epochs_completed", 0)),
        )
        self.generator.set_state(state["generator_state"])
        self.shuffle_rng.setstate(state["shuffle_state"])
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, path: str | Path) -> None:
        if self._pending_accumulation:
            raise RuntimeError("flush accumulated gradients before saving a checkpoint")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trainable_state = {
            name: parameter.detach().cpu().clone() for name, parameter in self._trainable
        }
        torch.save(
            {
                "format_version": self.CHECKPOINT_VERSION,
                "trainer_config": asdict(self.config),
                "trainer_state": asdict(self.state),
                "adapter_config": asdict(self.adapter.config),
                "backbone": {
                    "model_type": str(self.adapter.backbone.config.model_type),
                    "hidden_size": self.adapter.d_model,
                    "vocab_size": self.adapter.vocab_size,
                },
                "trainable_names": self.trainable_parameter_names,
                "model_state": trainable_state,
                "optimizer_state": self.optimizer.state_dict(),
                "generator_state": self.generator.get_state().cpu(),
                "shuffle_state": self.shuffle_rng.getstate(),
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported CID trainer checkpoint version")
        if checkpoint["trainer_config"] != asdict(self.config):
            raise ValueError("checkpoint trainer configuration does not match this trainer")
        if checkpoint["adapter_config"] != asdict(self.adapter.config):
            raise ValueError("checkpoint CID adapter configuration does not match this adapter")
        if tuple(checkpoint["trainable_names"]) != self.trainable_parameter_names:
            raise ValueError("checkpoint trainable parameter set does not match this adapter")
        backbone = checkpoint["backbone"]
        if (
            int(backbone["hidden_size"]) != self.adapter.d_model
            or int(backbone["vocab_size"]) != self.adapter.vocab_size
            or str(backbone["model_type"]) != str(self.adapter.backbone.config.model_type)
        ):
            raise ValueError("checkpoint backbone geometry does not match this adapter")

        saved_state = checkpoint["model_state"]
        with torch.no_grad():
            for name, parameter in self._trainable:
                parameter.copy_(
                    saved_state[name].to(device=parameter.device, dtype=parameter.dtype)
                )
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.generator.set_state(checkpoint["generator_state"])
        self.shuffle_rng.setstate(checkpoint["shuffle_state"])
        state = checkpoint["trainer_state"]
        self.state = CIDTrainerState(
            transitions_seen=int(state["transitions_seen"]),
            optimizer_steps=int(state["optimizer_steps"]),
            epochs_completed=int(state.get("epochs_completed", 0)),
        )
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> None:
        if self._pending_examples <= 0:
            raise RuntimeError("optimizer step requires accumulated examples")
        for _, parameter in self._trainable:
            if parameter.grad is not None:
                parameter.grad.div_(self._pending_examples)
        if self.gradient_clipper is None:
            torch.nn.utils.clip_grad_norm_(
                (parameter for _, parameter in self._trainable),
                self.config.max_grad_norm,
            )
        else:
            self.gradient_clipper(self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.state = CIDTrainerState(
            transitions_seen=self.state.transitions_seen,
            optimizer_steps=self.state.optimizer_steps + 1,
            epochs_completed=self.state.epochs_completed,
        )

    def _sample_timestep(self) -> float:
        if self.config.timestep_min == self.config.timestep_max:
            return self.config.timestep_min
        value = torch.rand((), device=self.generator.device, generator=self.generator)
        span = self.config.timestep_max - self.config.timestep_min
        return self.config.timestep_min + float(value) * span


def load_cid_adapter_checkpoint(
    adapter: ILLaDACIDAdapter,
    path: str | Path,
) -> CIDTrainerState:
    """Load the CID model state from a trainer checkpoint without restoring an optimizer."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CIDTrainer.CHECKPOINT_VERSION:
        raise ValueError("unsupported CID trainer checkpoint version")
    backbone = checkpoint["backbone"]
    if (
        int(backbone["hidden_size"]) != adapter.d_model
        or int(backbone["vocab_size"]) != adapter.vocab_size
        or str(backbone["model_type"]) != str(adapter.backbone.config.model_type)
    ):
        raise ValueError("checkpoint backbone geometry does not match this adapter")
    if checkpoint["adapter_config"] != asdict(adapter.config):
        raise ValueError("checkpoint CID adapter configuration does not match this adapter")

    parameters = dict(adapter.named_parameters())
    with torch.no_grad():
        for name, saved in checkpoint["model_state"].items():
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"checkpoint parameter is missing from adapter: {name}")
            if tuple(parameter.shape) != tuple(saved.shape):
                raise ValueError(f"checkpoint parameter shape mismatch: {name}")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    state = checkpoint["trainer_state"]
    return CIDTrainerState(
        transitions_seen=int(state["transitions_seen"]),
        optimizer_steps=int(state["optimizer_steps"]),
        epochs_completed=int(state.get("epochs_completed", 0)),
    )


class ILLaDATrajectoryTensorizer:
    """Turn one supervised trajectory transition into the CID tensor/loss contract."""

    def __init__(
        self,
        adapter: ILLaDACIDAdapter,
        tokenizer: Any,
        scheduler: CIDDiffusionScheduler | None = None,
        *,
        text_encoder: ILLaDATextEncoder | None = None,
    ) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder or ILLaDATextEncoder(adapter, tokenizer)
        if self.text_encoder.d_model != adapter.d_model:
            raise ValueError("training text encoder width must match the iLLaDA adapter")
        self.scheduler = scheduler or CIDDiffusionScheduler(ILLADA_MASK_TOKEN_ID)

    def tensorize(
        self,
        example: TrajectoryExample,
        source_step: int,
        *,
        timestep: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> CIDTrainingStep:
        if not 0.0 <= timestep <= 1.0:
            raise ValueError("timestep must be in [0, 1]")
        target_step = source_step + 1
        current = self._thought_snapshot(example, source_step)
        target = self._thought_snapshot(example, target_step)
        if not target:
            raise ValueError(f"trajectory has no thought targets for step {target_step}")

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype
        capacity = self.adapter.config.max_thought_slots
        current_by_id = {cell.cell_id: cell for cell in current}
        target_by_id = {cell.cell_id: cell for cell in target}
        current_by_slot = {cell.slot: cell for cell in current}
        target_output_slots = self._target_output_slots(current, target, capacity)

        current_vectors = self._semantic_vectors(current)
        target_vectors = self._semantic_vectors(target)
        thought_semantic = torch.zeros(
            (1, capacity, self.adapter.d_model), device=device, dtype=dtype
        )
        role_features = torch.zeros(
            (1, capacity, self.adapter.config.num_roles), device=device, dtype=dtype
        )
        uncertainty = torch.ones((1, capacity, 1), device=device, dtype=dtype)
        occupancy = torch.zeros((1, capacity, 1), device=device, dtype=dtype)
        role_order = tuple(CognitiveRole)

        for cell in current:
            thought_semantic[0, cell.slot] = current_vectors[cell.cell_id]
            occupancy[0, cell.slot, 0] = 1.0
            uncertainty[0, cell.slot, 0] = cell.uncertainty
            for role_index, role in enumerate(role_order):
                role_features[0, cell.slot, role_index] = cell.roles.get(role, 0.0)

        timestep_tensor = torch.tensor([timestep], device=device)
        thought_corruption = self.scheduler.corrupt_thought(
            thought_semantic,
            timestep_tensor,
            occupancy,
            generator=generator,
        )

        target_display = self._display_text(example, target_step)
        target_display_ids = self.text_encoder.tokenize(
            target_display, add_special_tokens=False
        )
        display_corruption = self.scheduler.corrupt_display(
            target_display_ids,
            timestep_tensor,
            generator=generator,
        )

        prompt_ids = self.text_encoder.tokenize(example.prompt, add_special_tokens=True)
        fact_memory = self.text_encoder.encode_texts(
            tuple(
                f"fact={key} | value={stable_text(value)}"
                for key, value in example.protected_facts.items()
            )
        )
        percept_memory = self.text_encoder.encode_texts(
            tuple(
                " | ".join(
                    (
                        f"source={event.source}",
                        f"value={stable_text(event.value)}",
                        f"version={event.version or ''}",
                    )
                )
                for event in example.events
                if event.arrival_step <= target_step
            )
        )
        source_memory = self.text_encoder.encode_texts(
            tuple(_source_text(descriptor) for descriptor in example.source_descriptors)
        )

        batch = CIDTensorBatch(
            thought_semantic=thought_corruption.semantic,
            role_features=role_features,
            uncertainty=uncertainty,
            local_noise=thought_corruption.noise,
            slot_occupancy=occupancy,
            prompt_ids=prompt_ids,
            display_ids=display_corruption.token_ids,
            display_noise=display_corruption.noise,
            fact_memory=fact_memory,
            percept_memory=percept_memory,
            source_memory=source_memory,
        )
        targets = self._targets(
            example=example,
            source_step=source_step,
            target_step=target_step,
            current_by_id=current_by_id,
            current_by_slot=current_by_slot,
            target_by_id=target_by_id,
            target_output_slots=target_output_slots,
            target_vectors=target_vectors,
            display_labels=display_corruption.labels,
            dtype=dtype,
            device=device,
        )
        return CIDTrainingStep(
            example_id=example.example_id,
            source_step=source_step,
            target_step=target_step,
            batch=batch,
            targets=targets,
        )

    def _targets(
        self,
        *,
        example: TrajectoryExample,
        source_step: int,
        target_step: int,
        current_by_id: Mapping[str, ThoughtTarget],
        current_by_slot: Mapping[int, ThoughtTarget],
        target_by_id: Mapping[str, ThoughtTarget],
        target_output_slots: Mapping[str, int],
        target_vectors: Mapping[str, Tensor],
        display_labels: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> CIDTargets:
        del source_step
        c = self.adapter.config
        n = c.max_thought_slots
        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES
        anchor_order = tuple(AnchorKind)
        relation_order = tuple(LinkRelation)
        object_order = tuple(ObjectKind)
        freshness_order = tuple(FreshnessDemand)

        thought_target = torch.zeros((1, n, self.adapter.d_model), device=device, dtype=dtype)
        thought_mask = torch.zeros((1, n), device=device, dtype=torch.bool)
        allocation_targets = torch.zeros((1, n), device=device, dtype=dtype)
        allocation_mask = torch.tensor(
            [[slot not in current_by_slot for slot in range(n)]], device=device, dtype=torch.bool
        )
        role_targets = torch.zeros((1, n, c.num_roles), device=device, dtype=dtype)
        uncertainty = torch.ones((1, n, 1), device=device, dtype=dtype)
        noise_delta = torch.zeros((1, n, 1), device=device, dtype=dtype)
        lifecycle = torch.full((1, n), -100, device=device, dtype=torch.long)
        need_targets = torch.zeros((1, n), device=device, dtype=dtype)
        source_targets = torch.full((1, n), -100, device=device, dtype=torch.long)
        revision_targets = torch.full((1, n), -100, device=device, dtype=torch.long)
        refresh_targets = torch.full((1, n), -100, device=device, dtype=torch.long)

        argument_presence_targets = torch.zeros(
            (1, n, c.max_argument_slots), device=device, dtype=dtype
        )
        argument_presence_mask = torch.zeros(
            (1, n, c.max_argument_slots), device=device, dtype=torch.bool
        )
        argument_embeddings = torch.zeros(
            (1, n, c.max_argument_slots, self.adapter.d_model), device=device, dtype=dtype
        )
        argument_mask = torch.zeros(
            (1, n, c.max_argument_slots), device=device, dtype=torch.bool
        )
        anchor_presence_targets = torch.zeros(
            (1, n, c.max_anchor_slots), device=device, dtype=dtype
        )
        anchor_presence_mask = torch.zeros(
            (1, n, c.max_anchor_slots), device=device, dtype=torch.bool
        )
        anchor_kind_targets = torch.full(
            (1, n, c.max_anchor_slots), -100, device=device, dtype=torch.long
        )
        anchor_embeddings = torch.zeros(
            (1, n, c.max_anchor_slots, self.adapter.d_model), device=device, dtype=dtype
        )
        anchor_mask = torch.zeros((1, n, c.max_anchor_slots), device=device, dtype=torch.bool)
        link_presence_targets = torch.zeros(
            (1, n, c.max_link_slots), device=device, dtype=dtype
        )
        link_presence_mask = torch.zeros(
            (1, n, c.max_link_slots), device=device, dtype=torch.bool
        )
        link_relation_targets = torch.full(
            (1, n, c.max_link_slots), -100, device=device, dtype=torch.long
        )
        link_target_kind_targets = torch.full(
            (1, n, c.max_link_slots), -100, device=device, dtype=torch.long
        )
        link_target_embeddings = torch.zeros(
            (1, n, c.max_link_slots, self.adapter.d_model), device=device, dtype=dtype
        )
        link_mask = torch.zeros((1, n, c.max_link_slots), device=device, dtype=torch.bool)

        for cell_id, target in target_by_id.items():
            slot = target_output_slots[cell_id]
            thought_target[0, slot] = target_vectors[cell_id]
            thought_mask[0, slot] = True
            uncertainty[0, slot, 0] = target.uncertainty
            for role_index, role in enumerate(role_order):
                role_targets[0, slot, role_index] = target.roles.get(role, 0.0)
            current = current_by_id.get(cell_id)
            if current is None:
                allocation_targets[0, slot] = 1.0
                noise_delta[0, slot, 0] = target.noise - 1.0
            else:
                lifecycle[0, slot] = lifecycle_order.index(target.lifecycle)
                noise_delta[0, slot, 0] = target.noise - current.noise
                delta = target.noise - current.noise
                if delta > 1e-6:
                    revision_targets[0, slot] = int(RevisionAction.REOPEN)
                elif delta < -1e-6:
                    revision_targets[0, slot] = int(RevisionAction.STABILIZE)
                else:
                    revision_targets[0, slot] = int(RevisionAction.KEEP)

        source_names = tuple(str(item.get("name", "")) for item in example.source_descriptors)
        bindings = tuple(
            binding for binding in example.binding_targets if binding.first_need_step <= target_step
        )
        for binding in bindings:
            if binding.source not in source_names:
                raise ValueError(f"binding target references unknown source {binding.source!r}")
            source_index = source_names.index(binding.source)
            descriptor = example.source_descriptors[source_index]
            declared_arguments = tuple(descriptor.get("arguments", ()))
            for cell_ref in binding.target_cells:
                slot = target_output_slots.get(cell_ref.identifier)
                if slot is None:
                    continue
                need_targets[0, slot] = binding.confidence
                source_targets[0, slot] = source_index
                refresh_targets[0, slot] = freshness_order.index(binding.freshness)
                for argument_slot, argument in enumerate(
                    declared_arguments[: c.max_argument_slots]
                ):
                    argument_presence_mask[0, slot, argument_slot] = True
                    name = str(argument.get("name", ""))
                    available_step = binding.argument_steps.get(
                        name,
                        binding.executable_step,
                    )
                    executable = (
                        available_step is not None
                        and target_step >= available_step
                        and name in binding.arguments
                    )
                    if executable:
                        argument_presence_targets[0, slot, argument_slot] = 1.0
                        argument_embeddings[0, slot, argument_slot] = self.text_encoder.encode_one(
                            stable_text(binding.arguments[name]), detach=True
                        )
                        argument_mask[0, slot, argument_slot] = True

        grounding_by_cell = {
            item.cell_id: item for item in example.grounding_targets if item.step == target_step
        }
        for cell_id, slot in target_output_slots.items():
            if not thought_mask[0, slot]:
                continue
            anchor_presence_mask[0, slot] = True
            link_presence_mask[0, slot] = True
            grounding = grounding_by_cell.get(cell_id)
            if grounding is None:
                continue
            if len(grounding.anchors) > c.max_anchor_slots:
                raise ValueError("grounding target exceeds configured anchor slot capacity")
            if len(grounding.links) > c.max_link_slots:
                raise ValueError("grounding target exceeds configured link slot capacity")
            for index, anchor in enumerate(grounding.anchors):
                anchor_presence_targets[0, slot, index] = 1.0
                anchor_kind_targets[0, slot, index] = anchor_order.index(anchor.kind)
                anchor_embeddings[0, slot, index] = self.text_encoder.encode_one(
                    anchor.canonical_key, detach=True
                )
                anchor_mask[0, slot, index] = True
            for index, link in enumerate(grounding.links):
                link_presence_targets[0, slot, index] = 1.0
                link_relation_targets[0, slot, index] = relation_order.index(link.relation)
                link_target_kind_targets[0, slot, index] = object_order.index(link.target.kind)
                link_target_embeddings[0, slot, index] = self.text_encoder.encode_one(
                    _object_text(link.target), detach=True
                )
                link_mask[0, slot, index] = True

        return CIDTargets(
            thought_semantic=thought_target,
            thought_mask=thought_mask,
            allocation_targets=allocation_targets,
            allocation_mask=allocation_mask,
            display_ids=display_labels,
            role_targets=role_targets,
            uncertainty=uncertainty,
            noise_delta=noise_delta,
            lifecycle=lifecycle,
            need_targets=need_targets,
            source_targets=source_targets,
            argument_presence_targets=argument_presence_targets,
            argument_presence_mask=argument_presence_mask,
            argument_embeddings=argument_embeddings,
            argument_mask=argument_mask,
            revision_targets=revision_targets,
            refresh_targets=refresh_targets,
            anchor_presence_targets=anchor_presence_targets,
            anchor_presence_mask=anchor_presence_mask,
            anchor_kind_targets=anchor_kind_targets,
            anchor_embeddings=anchor_embeddings,
            anchor_mask=anchor_mask,
            link_presence_targets=link_presence_targets,
            link_presence_mask=link_presence_mask,
            link_relation_targets=link_relation_targets,
            link_target_kind_targets=link_target_kind_targets,
            link_target_embeddings=link_target_embeddings,
            link_mask=link_mask,
        )

    def _thought_snapshot(self, example: TrajectoryExample, step: int) -> tuple[ThoughtTarget, ...]:
        snapshot = tuple(target for target in example.thought_targets if target.step == step)
        capacity = self.adapter.config.max_thought_slots
        if any(target.slot >= capacity for target in snapshot):
            raise ValueError("thought target slot exceeds adapter TCT capacity")
        return snapshot

    def _semantic_vectors(self, snapshot: tuple[ThoughtTarget, ...]) -> dict[str, Tensor]:
        return {
            target.cell_id: self.text_encoder.encode_one(target.semantic_text, detach=True)
            for target in snapshot
        }

    def _target_output_slots(
        self,
        current: tuple[ThoughtTarget, ...],
        target: tuple[ThoughtTarget, ...],
        capacity: int,
    ) -> dict[str, int]:
        current_by_id = {cell.cell_id: cell for cell in current}
        current_slots = {cell.slot for cell in current}
        slots: dict[str, int] = {}
        used: set[int] = set()
        for cell in target:
            if cell.cell_id in current_by_id:
                slot = current_by_id[cell.cell_id].slot
            else:
                slot = cell.slot
                if slot in current_slots:
                    raise ValueError(
                        "new thought target cannot allocate into an occupied source slot"
                    )
            if not 0 <= slot < capacity or slot in used:
                raise ValueError(
                    "target thought transition has an invalid or colliding output slot"
                )
            slots[cell.cell_id] = slot
            used.add(slot)
        missing = set(current_by_id) - {cell.cell_id for cell in target}
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"target snapshot removed cells without RETIRED state: {names}")
        return slots

    def _display_text(self, example: TrajectoryExample, step: int) -> str:
        for target in example.display_targets:
            if target.step == step:
                return target.text
        return example.target_display


def _cat_targets(steps: tuple[CIDTrainingStep, ...], name: str) -> Tensor:
    return torch.cat(tuple(getattr(step.targets, name) for step in steps), dim=0)


def _pad_2d(
    tensors: tuple[Tensor, ...],
    *,
    pad_value: int | float,
) -> tuple[Tensor, Tensor]:
    max_length = max(tensor.shape[1] for tensor in tensors)
    output = tensors[0].new_full((len(tensors), max_length), pad_value)
    padding_mask = torch.ones(
        (len(tensors), max_length),
        dtype=torch.bool,
        device=tensors[0].device,
    )
    for row, tensor in enumerate(tensors):
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ValueError("sequence tensors must have shape [1, length]")
        length = tensor.shape[1]
        output[row, :length] = tensor[0]
        padding_mask[row, :length] = False
    return output, padding_mask


def _pad_3d(tensors: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    if any(tensor.ndim != 3 or tensor.shape[0] != 1 for tensor in tensors):
        raise ValueError("feature tensors must have shape [1, length, width]")
    width = tensors[0].shape[2]
    if any(tensor.shape[2] != width for tensor in tensors):
        raise ValueError("feature tensors in one batch must have the same width")
    max_length = max(tensor.shape[1] for tensor in tensors)
    output = tensors[0].new_zeros((len(tensors), max_length, width))
    padding_mask = torch.ones(
        (len(tensors), max_length),
        dtype=torch.bool,
        device=tensors[0].device,
    )
    for row, tensor in enumerate(tensors):
        length = tensor.shape[1]
        output[row, :length] = tensor[0]
        padding_mask[row, :length] = False
    return output, padding_mask


def _source_text(descriptor: Mapping[str, Any]) -> str:
    arguments = ",".join(
        f"{item.get('name', '')}:{item.get('kind', 'any')}"
        for item in descriptor.get("arguments", ())
    )
    return " | ".join(
        (
            f"source={descriptor.get('name', '')}",
            f"description={descriptor.get('description', '')}",
            f"arguments={arguments}",
            f"dynamic={bool(descriptor.get('dynamic', False))}",
        )
    )


def _object_text(ref: ObjectRef) -> str:
    if ref.kind is ObjectKind.DISPLAY_SPAN:
        return f"{ref.kind.value}:{ref.span[0]}:{ref.span[1]}"
    return f"{ref.kind.value}:{ref.identifier}"


def trajectory_transitions(
    examples: tuple[TrajectoryExample, ...],
) -> tuple[tuple[TrajectoryExample, int], ...]:
    transitions: list[tuple[TrajectoryExample, int]] = []
    for example in examples:
        steps = {target.step for target in example.thought_targets}
        transitions.extend(
            (example, step) for step in sorted(steps) if step + 1 in steps
        )
    return tuple(transitions)


def shard_transitions(
    transitions: tuple[tuple[TrajectoryExample, int], ...],
    *,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> tuple[tuple[TrajectoryExample, int], ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if not transitions:
        return ()
    order = list(transitions)
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    padding = (-len(order)) % world_size
    if padding:
        order.extend(order[:padding])
    return tuple(order[rank::world_size])


def wrap_stage_a_ddp(
    adapter: ILLaDACIDAdapter,
    *,
    device_ids: list[int] | None,
) -> torch.nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    ignored_state = tuple(
        name for name, parameter in adapter.named_parameters() if not parameter.requires_grad
    ) + tuple(name for name, _ in adapter.backbone.named_buffers(prefix="backbone"))
    DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
        adapter,
        ignored_state,
    )
    kwargs: dict[str, object] = {
        "device_ids": device_ids,
        "find_unused_parameters": True,
    }
    if "forward_sync_buffers" in signature(DistributedDataParallel).parameters:
        kwargs["forward_sync_buffers"] = False
    else:
        kwargs["broadcast_buffers"] = False
    return DistributedDataParallel(adapter, **kwargs)


def wrap_stage_b_fsdp(
    adapter: ILLaDACIDAdapter,
    *,
    device_id: int | torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    decoder = adapter.backbone.get_decoder()
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("iLLaDA decoder must expose non-empty layers for FSDP auto-wrap")
    layer_class = type(layers[0])
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_class},
    )
    return FullyShardedDataParallel(
        adapter,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        ),
        device_id=device_id,
        sync_module_states=False,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def save_stage_b_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: CIDTrainer,
    path: str | Path,
    *,
    dataset_sha256: str | None = None,
) -> None:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    if not dist.is_initialized():
        raise RuntimeError("Stage B checkpointing requires an initialized process group")
    destination = Path(path)
    if dist.get_rank() == 0:
        destination.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=destination / "distributed",
    )
    torch.save(
        trainer.local_progress_state(),
        destination / f"rank-{dist.get_rank():04d}.pt",
    )
    if dist.get_rank() == 0:
        metadata = {
            "format_version": 1,
            "kind": "cid-stage-b-fsdp",
            "world_size": dist.get_world_size(),
            "dataset_sha256": dataset_sha256,
            "adapter_config": asdict(trainer.adapter.config),
            "backbone": {
                "model_type": str(trainer.adapter.backbone.config.model_type),
                "hidden_size": trainer.adapter.d_model,
                "vocab_size": trainer.adapter.vocab_size,
            },
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    dist.barrier()


def load_stage_b_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: CIDTrainer,
    path: str | Path,
    *,
    expected_dataset_sha256: str | None = None,
) -> None:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    if not dist.is_initialized():
        raise RuntimeError("Stage B checkpointing requires an initialized process group")
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("kind") != "cid-stage-b-fsdp":
        raise ValueError("unsupported Stage B checkpoint format")
    if int(metadata["world_size"]) != dist.get_world_size():
        raise ValueError("Stage B resume currently requires the original world size")
    if (
        expected_dataset_sha256 is not None
        and metadata.get("dataset_sha256") != expected_dataset_sha256
    ):
        raise ValueError("Stage B checkpoint dataset SHA-256 does not match training data")
    if metadata["adapter_config"] != asdict(trainer.adapter.config):
        raise ValueError("Stage B checkpoint adapter configuration does not match")
    backbone = metadata["backbone"]
    if (
        int(backbone["hidden_size"]) != trainer.adapter.d_model
        or int(backbone["vocab_size"]) != trainer.adapter.vocab_size
        or str(backbone["model_type"]) != str(trainer.adapter.backbone.config.model_type)
    ):
        raise ValueError("Stage B checkpoint backbone geometry does not match")

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    distributed_state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(distributed_state, checkpoint_id=source / "distributed")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=distributed_state["model"],
        optim_state_dict=distributed_state["optimizer"],
        options=options,
    )
    local_state = torch.load(
        source / f"rank-{dist.get_rank():04d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    trainer.restore_local_progress_state(local_state)
    dist.barrier()

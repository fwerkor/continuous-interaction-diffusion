from __future__ import annotations

import gc
import json
import math
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cid.contracts import FreshnessDemand
from cid.data import ThoughtTarget, TrajectoryExample, training_transition_source_steps
from cid.grounding import AnchorKind, LinkRelation, ObjectKind, ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.allocation import (
    DEFAULT_MAX_ALLOCATIONS_PER_STEP,
    prefix_allocation_mask,
)
from cid.model.diffusion import (
    CIDDiffusionScheduler,
    denoising_noise_level,
    denoising_reveal_fraction,
)
from cid.model.encoding import ILLaDATextEncoder, stable_text
from cid.model.illada import ILLaDACIDAdapter
from cid.model.losses import CIDLoss, CIDTargets, cid_loss
from cid.model.materialize import RevisionAction
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput, build_percept_routing_masks
from cid.state import CellLifecycle, CognitiveRole


@dataclass(slots=True)
class CIDTrainingStep:
    example_id: str
    source_step: int
    target_step: int
    diffusion_step: int
    next_diffusion_step: int
    batch: CIDTensorBatch
    targets: CIDTargets


@dataclass(slots=True)
class CIDTrainingBatch:
    example_ids: tuple[str, ...]
    source_steps: tuple[int, ...]
    target_steps: tuple[int, ...]
    batch: CIDTensorBatch
    targets: CIDTargets


@dataclass(frozen=True, slots=True)
class CIDRolloutState:
    """Detached model state that may replace the next teacher-forced T/Y input."""

    thought_semantic: Tensor
    role_features: Tensor
    uncertainty: Tensor
    lifecycle_features: Tensor
    slot_occupancy: Tensor
    local_noise: Tensor
    display_ids: Tensor
    display_noise_level: float = 1.0


@dataclass(frozen=True, slots=True)
class CIDRolloutWindow:
    example: TrajectoryExample
    source_steps: tuple[int, ...]
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_steps:
            raise ValueError("rollout window requires at least one transition")
        if not math.isfinite(self.loss_weight) or self.loss_weight <= 0.0:
            raise ValueError("rollout window loss_weight must be finite and positive")
        if any(
            right != left + 1
            for left, right in zip(self.source_steps, self.source_steps[1:], strict=False)
        ):
            raise ValueError("rollout window source steps must be contiguous")


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
    fact_memory, fact_padding_mask = _pad_3d(tuple(step.batch.fact_memory for step in steps))
    percept_memory, percept_padding_mask = _pad_3d(
        tuple(step.batch.percept_memory for step in steps)
    )
    percept_thought_mask = _collate_percept_masks(steps, "thought")
    percept_display_mask = _collate_percept_masks(steps, "display")
    source_memory, source_padding_mask = _pad_3d(tuple(step.batch.source_memory for step in steps))
    display_labels, _ = _pad_2d(
        tuple(step.targets.display_ids for step in steps),
        pad_value=-100,
    )

    batch = CIDTensorBatch(
        thought_semantic=_pad_slot_tensors(tuple(step.batch.thought_semantic for step in steps)),
        role_features=_pad_slot_tensors(tuple(step.batch.role_features for step in steps)),
        uncertainty=_pad_slot_tensors(
            tuple(step.batch.uncertainty for step in steps), pad_value=1.0
        ),
        local_noise=_pad_slot_tensors(tuple(step.batch.local_noise for step in steps)),
        slot_occupancy=_pad_slot_tensors(tuple(step.batch.slot_occupancy for step in steps)),
        lifecycle_features=_pad_slot_tensors(
            tuple(step.batch.lifecycle_features for step in steps)
        ),
        prompt_ids=prompt_ids,
        display_ids=display_ids,
        display_noise=display_noise,
        fact_memory=fact_memory,
        percept_memory=percept_memory,
        source_memory=source_memory,
        percept_thought_mask=percept_thought_mask,
        percept_display_mask=percept_display_mask,
        prompt_padding_mask=prompt_padding_mask,
        display_padding_mask=display_padding_mask,
        fact_padding_mask=fact_padding_mask,
        percept_padding_mask=percept_padding_mask,
        source_padding_mask=source_padding_mask,
    )
    targets = CIDTargets(
        thought_semantic=_pad_slot_targets(steps, "thought_semantic"),
        thought_mask=_pad_slot_targets(steps, "thought_mask"),
        convergence_targets=_cat_targets(steps, "convergence_targets"),
        allocation_targets=_pad_slot_targets(steps, "allocation_targets"),
        allocation_mask=_pad_slot_targets(steps, "allocation_mask"),
        display_ids=display_labels,
        role_targets=_pad_slot_targets(steps, "role_targets"),
        uncertainty=_pad_slot_targets(steps, "uncertainty", pad_value=1.0),
        noise_delta=_pad_slot_targets(steps, "noise_delta"),
        lifecycle=_pad_slot_targets(steps, "lifecycle", pad_value=-100),
        need_targets=_pad_slot_targets(steps, "need_targets"),
        source_targets=_pad_slot_targets(steps, "source_targets", pad_value=-100),
        argument_presence_targets=_pad_slot_targets(steps, "argument_presence_targets"),
        argument_presence_mask=_pad_slot_targets(steps, "argument_presence_mask"),
        argument_embeddings=_pad_slot_targets(steps, "argument_embeddings"),
        argument_mask=_pad_slot_targets(steps, "argument_mask"),
        revision_targets=_pad_slot_targets(steps, "revision_targets", pad_value=-100),
        refresh_targets=_pad_slot_targets(steps, "refresh_targets", pad_value=-100),
        anchor_presence_targets=_pad_slot_targets(steps, "anchor_presence_targets"),
        anchor_presence_mask=_pad_slot_targets(steps, "anchor_presence_mask"),
        anchor_kind_targets=_pad_slot_targets(steps, "anchor_kind_targets", pad_value=-100),
        anchor_embeddings=_pad_slot_targets(steps, "anchor_embeddings"),
        anchor_mask=_pad_slot_targets(steps, "anchor_mask"),
        link_presence_targets=_pad_slot_targets(steps, "link_presence_targets"),
        link_presence_mask=_pad_slot_targets(steps, "link_presence_mask"),
        link_relation_targets=_pad_slot_targets(steps, "link_relation_targets", pad_value=-100),
        link_target_kind_targets=_pad_slot_targets(
            steps, "link_target_kind_targets", pad_value=-100
        ),
        link_target_embeddings=_pad_slot_targets(steps, "link_target_embeddings"),
        link_mask=_pad_slot_targets(steps, "link_mask"),
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
    warmup_steps: int = 0
    lr_decay_steps: int = 0
    min_learning_rate_ratio: float = 0.1
    timestep_min: float = 0.05
    timestep_max: float = 1.0
    rollout_horizon: int = 3
    teacher_forcing_epochs: int = 1
    rollout_ramp_epochs: int = 2
    rollout_allocation_threshold: float = 0.8
    rollout_max_allocations_per_step: int = DEFAULT_MAX_ALLOCATIONS_PER_STEP
    rollout_denoising_steps: int = 8
    rollout_display_revision_fraction: float = 0.125
    rollout_display_revision_margin: float = 0.15
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
        if self.warmup_steps < 0 or self.lr_decay_steps < 0:
            raise ValueError("learning-rate schedule steps must be non-negative")
        if self.lr_decay_steps and self.warmup_steps > self.lr_decay_steps:
            raise ValueError("warmup_steps cannot exceed lr_decay_steps")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if not 0.0 <= self.timestep_min <= self.timestep_max <= 1.0:
            raise ValueError("timestep range must satisfy 0 <= min <= max <= 1")
        if self.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")
        if self.teacher_forcing_epochs < 0 or self.rollout_ramp_epochs < 0:
            raise ValueError("rollout curriculum epoch counts must be non-negative")
        for name in (
            "rollout_allocation_threshold",
            "rollout_display_revision_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.rollout_max_allocations_per_step <= 0:
            raise ValueError("rollout_max_allocations_per_step must be positive")
        if self.rollout_denoising_steps <= 0:
            raise ValueError("rollout_denoising_steps must be positive")
        if self.rollout_display_revision_margin < 0.0:
            raise ValueError("rollout_display_revision_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class CIDTrainerState:
    transitions_seen: int = 0
    optimizer_steps: int = 0
    epochs_completed: int = 0
    rollout_windows_seen_in_epoch: int = 0


@dataclass(frozen=True, slots=True)
class CIDTrainReport:
    transitions: int
    optimizer_steps: int
    mean_loss: float
    raw_mean_loss: float


@dataclass(frozen=True, slots=True)
class CIDTrainProgress:
    transitions: int
    optimizer_steps: int
    mean_loss: float
    raw_mean_loss: float
    rollout_windows_seen_in_epoch: int
    learning_rate: float


class CIDTrainer:
    CHECKPOINT_VERSION = 2
    SUPPORTED_CHECKPOINT_VERSIONS = (1, 2)

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
        # FSDP CPU offload keeps the live parameter shards on host memory between
        # forwards, while tensorization and diffusion corruption still happen on the
        # compute device carried by the frozen text-encoder snapshot.
        device = tensorizer.text_encoder.device
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

    @property
    def pending_accumulation_steps(self) -> int:
        return self._pending_accumulation

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
        losses, _, _ = self._forward_backward(samples)
        return losses

    def train_rollout_microbatch(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        rollout_probability: float,
    ) -> tuple[float, float, int]:
        if not windows:
            raise ValueError("rollout micro-batch cannot be empty")
        if not 0.0 <= rollout_probability <= 1.0:
            raise ValueError("rollout_probability must be in [0, 1]")
        lengths = {len(window.source_steps) for window in windows}
        if len(lengths) != 1:
            raise ValueError("rollout micro-batch windows must have the same length")
        loss_weights = {window.loss_weight for window in windows}
        if len(loss_weights) != 1:
            raise ValueError("rollout micro-batch windows must have the same loss_weight")
        loss_weight = next(iter(loss_weights))

        rollout_states: list[CIDRolloutState | None] = [None] * len(windows)
        loss_sum = 0.0
        raw_loss_sum = 0.0
        transition_count = 0
        for offset in range(next(iter(lengths))):
            samples: list[CIDTrainingStep] = []
            for index, window in enumerate(windows):
                use_rollout = (
                    offset > 0
                    and rollout_states[index] is not None
                    and self.shuffle_rng.random() < rollout_probability
                )
                samples.append(
                    self.tensorizer.tensorize(
                        window.example,
                        window.source_steps[offset],
                        timestep=self._sample_timestep(),
                        generator=self.generator,
                        rollout_state=rollout_states[index] if use_rollout else None,
                    )
                )
            losses, output, training_batch = self._forward_backward(
                tuple(samples), loss_scale=loss_weight
            )
            batch_size = len(samples)
            raw_loss = float(losses.total.detach().float()) * batch_size
            raw_loss_sum += raw_loss
            loss_sum += raw_loss * loss_weight
            transition_count += batch_size
            # The predicted state is consumed only by a later transition in the same
            # window. Skipping the terminal materialization avoids pure accelerator work,
            # which is especially significant for compact NPU backbones.
            if offset + 1 < next(iter(lengths)) and rollout_probability > 0.0:
                rollout_states = [
                    self._rollout_state_from_prediction(
                        sample,
                        training_batch,
                        output,
                        batch_index=index,
                    )
                    for index, sample in enumerate(samples)
                ]
            del output, training_batch, losses, samples
        return loss_sum, raw_loss_sum, transition_count

    def _forward_backward(
        self,
        samples: tuple[CIDTrainingStep, ...],
        *,
        loss_scale: float = 1.0,
    ) -> tuple[CIDLoss, CIDTensorOutput, CIDTrainingBatch]:
        if not samples:
            raise ValueError("training samples cannot be empty")
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise ValueError("loss_scale must be finite and positive")
        training_batch = collate_training_steps(
            samples,
            pad_token_id=int(self.pad_token_id),
        )
        output = self.forward_model(training_batch.batch)
        losses = cid_loss(output, training_batch.targets)
        if not bool(torch.isfinite(losses.total)):
            names = ", ".join(training_batch.example_ids)
            raise FloatingPointError(f"non-finite CID loss for training micro-batch: {names}")
        batch_size = len(samples)
        (losses.total * batch_size * loss_scale).backward()
        self._pending_accumulation += 1
        self._pending_examples += batch_size
        self.state = CIDTrainerState(
            transitions_seen=self.state.transitions_seen + batch_size,
            optimizer_steps=self.state.optimizer_steps,
            epochs_completed=self.state.epochs_completed,
            rollout_windows_seen_in_epoch=self.state.rollout_windows_seen_in_epoch,
        )
        if self._pending_accumulation >= self.config.gradient_accumulation_steps:
            self._optimizer_step()
        return losses, output, training_batch

    def _rollout_state_from_prediction(
        self,
        sample: CIDTrainingStep,
        training_batch: CIDTrainingBatch,
        output: CIDTensorOutput,
        *,
        batch_index: int,
    ) -> CIDRolloutState:
        input_batch = training_batch.batch
        # ``collate_training_steps`` pads every example to the widest thought
        # capacity in the micro-batch.  A rollout state belongs to one example,
        # so carrying the padded width into its next transition can make a
        # smaller-capacity trajectory fail tensorizer geometry validation.
        thought_slots = sample.batch.thought_semantic.shape[1]
        slot_slice = slice(0, thought_slots)
        occupancy = (
            input_batch.slot_occupancy[batch_index : batch_index + 1, slot_slice]
            .detach()
            .bool()
        )
        allocation = prefix_allocation_mask(
            occupancy,
            output.allocation_logits[batch_index : batch_index + 1, slot_slice].detach(),
            threshold=self.config.rollout_allocation_threshold,
            max_allocations=self.config.rollout_max_allocations_per_step,
        ).unsqueeze(-1)
        previous_occupancy = occupancy
        occupancy = occupancy | (~occupancy & allocation)
        lifecycle_indices = output.lifecycle_logits[
            batch_index : batch_index + 1, slot_slice
        ].argmax(dim=-1)
        lifecycle_features = torch.nn.functional.one_hot(
            lifecycle_indices, num_classes=self.adapter.config.num_lifecycles
        ).to(dtype=sample.batch.role_features.dtype)
        active_index = MODELED_LIFECYCLES.index(CellLifecycle.ACTIVE)
        newly_allocated = (~previous_occupancy & allocation).squeeze(-1)
        if bool(newly_allocated.any()):
            lifecycle_features[newly_allocated] = 0.0
            lifecycle_features[..., active_index][newly_allocated] = 1.0
        lifecycle_features = lifecycle_features * occupancy.to(
            dtype=lifecycle_features.dtype
        )

        input_noise = input_batch.local_noise[batch_index : batch_index + 1, slot_slice]
        base_noise = torch.where(
            previous_occupancy,
            input_noise,
            torch.ones_like(input_noise),
        )
        local_noise = (
            base_noise
            + output.noise_delta[batch_index : batch_index + 1, slot_slice].detach()
        ).clamp(0.0, 1.0)
        local_noise = local_noise * occupancy.to(dtype=local_noise.dtype)

        display_length = sample.batch.display_ids.shape[1]
        display_ids = self.tensorizer.scheduler.refine_display(
            input_batch.display_ids[batch_index : batch_index + 1, :display_length],
            output.display_logits[batch_index : batch_index + 1, :display_length],
            reveal_fraction=denoising_reveal_fraction(
                sample.diffusion_step, self.config.rollout_denoising_steps
            ),
            revision_fraction=self.config.rollout_display_revision_fraction,
            revision_margin=self.config.rollout_display_revision_margin,
        )
        return CIDRolloutState(
            thought_semantic=output.thought_semantic[
                batch_index : batch_index + 1, slot_slice
            ].detach(),
            role_features=torch.sigmoid(
                output.role_logits[batch_index : batch_index + 1, slot_slice]
            ).detach(),
            uncertainty=output.uncertainty[
                batch_index : batch_index + 1, slot_slice
            ].detach(),
            lifecycle_features=lifecycle_features.detach(),
            slot_occupancy=occupancy.to(dtype=sample.batch.slot_occupancy.dtype).detach(),
            local_noise=local_noise.detach(),
            display_ids=display_ids.detach(),
            display_noise_level=denoising_noise_level(
                sample.next_diffusion_step, self.config.rollout_denoising_steps
            ),
        )

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

    def train_rollout_windows(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        epochs: int = 1,
        shuffle: bool = True,
        preserve_order: bool = False,
        progress_every_optimizer_steps: int | None = None,
        progress_callback: Callable[[CIDTrainProgress], None] | None = None,
    ) -> CIDTrainReport:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not windows:
            raise ValueError("training data contains no rollout windows")
        if progress_callback is not None and progress_every_optimizer_steps is None:
            raise ValueError("progress callback requires progress_every_optimizer_steps")
        if progress_every_optimizer_steps is not None and progress_every_optimizer_steps <= 0:
            raise ValueError("progress_every_optimizer_steps must be positive")
        total_loss = 0.0
        total_raw_loss = 0.0
        total_transitions = 0
        start_optimizer_steps = self.state.optimizer_steps
        progress_loss = 0.0
        progress_raw_loss = 0.0
        progress_transitions = 0
        next_progress_step = None
        if progress_callback is not None and progress_every_optimizer_steps is not None:
            next_progress_step = (
                self.state.optimizer_steps // progress_every_optimizer_steps + 1
            ) * progress_every_optimizer_steps

        def emit_progress_if_due() -> None:
            nonlocal progress_loss, progress_raw_loss, progress_transitions, next_progress_step
            if (
                progress_callback is None
                or progress_every_optimizer_steps is None
                or next_progress_step is None
                or self.state.optimizer_steps < next_progress_step
                or progress_transitions == 0
            ):
                return
            progress_callback(
                CIDTrainProgress(
                    transitions=progress_transitions,
                    optimizer_steps=self.state.optimizer_steps,
                    mean_loss=progress_loss / progress_transitions,
                    raw_mean_loss=progress_raw_loss / progress_transitions,
                    rollout_windows_seen_in_epoch=self.state.rollout_windows_seen_in_epoch,
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                )
            )
            progress_loss = 0.0
            progress_raw_loss = 0.0
            progress_transitions = 0
            while next_progress_step <= self.state.optimizer_steps:
                next_progress_step += progress_every_optimizer_steps

        for _ in range(epochs):
            rollout_probability = self.rollout_probability()
            microbatches = self._rollout_microbatches(
                windows, shuffle=shuffle, preserve_order=preserve_order
            )
            for microbatch in microbatches:
                loss_sum, raw_loss_sum, transitions = self.train_rollout_microbatch(
                    microbatch,
                    rollout_probability=rollout_probability,
                )
                total_loss += loss_sum
                total_raw_loss += raw_loss_sum
                total_transitions += transitions
                progress_loss += loss_sum
                progress_raw_loss += raw_loss_sum
                progress_transitions += transitions
                self.state = CIDTrainerState(
                    transitions_seen=self.state.transitions_seen,
                    optimizer_steps=self.state.optimizer_steps,
                    epochs_completed=self.state.epochs_completed,
                    rollout_windows_seen_in_epoch=(
                        self.state.rollout_windows_seen_in_epoch + len(microbatch)
                    ),
                )
                emit_progress_if_due()
            self.flush()
            emit_progress_if_due()
            self.state = CIDTrainerState(
                transitions_seen=self.state.transitions_seen,
                optimizer_steps=self.state.optimizer_steps,
                epochs_completed=self.state.epochs_completed + 1,
                rollout_windows_seen_in_epoch=0,
            )
        return CIDTrainReport(
            transitions=total_transitions,
            optimizer_steps=self.state.optimizer_steps - start_optimizer_steps,
            mean_loss=total_loss / total_transitions,
            raw_mean_loss=total_raw_loss / total_transitions,
        )

    def evaluate_rollout_windows(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        seed: int,
    ) -> CIDTrainReport:
        """Evaluate a stable teacher-forced diffusion objective without mutating training state.

        Validation deliberately uses a fixed RNG seed and teacher-forced inputs on every epoch.
        This makes loss values directly comparable across epochs even while Stage A changes its
        rollout curriculum. Model/optimizer state and the trainer RNG streams are restored exactly
        after evaluation.
        """

        if not windows:
            raise ValueError("validation data contains no rollout windows")
        if self._pending_accumulation:
            raise RuntimeError("flush accumulated gradients before validation")

        generator_state = self.generator.get_state().cpu()
        shuffle_state = self.shuffle_rng.getstate()
        was_training = self.forward_model.training
        total_loss = 0.0
        total_raw_loss = 0.0
        total_transitions = 0
        try:
            self.generator.manual_seed(seed)
            self.shuffle_rng.seed(seed)
            self.forward_model.eval()
            with torch.no_grad():
                microbatches = self._rollout_microbatches(
                    windows, shuffle=False, preserve_order=True
                )
                for microbatch in microbatches:
                    lengths = {len(window.source_steps) for window in microbatch}
                    if len(lengths) != 1:
                        raise ValueError(
                            "validation micro-batch windows must have the same length"
                        )
                    loss_weights = {window.loss_weight for window in microbatch}
                    if len(loss_weights) != 1:
                        raise ValueError(
                            "validation micro-batch windows must have the same loss_weight"
                        )
                    loss_weight = next(iter(loss_weights))
                    for offset in range(next(iter(lengths))):
                        samples = tuple(
                            self.tensorizer.tensorize(
                                window.example,
                                window.source_steps[offset],
                                timestep=self._sample_timestep(),
                                generator=self.generator,
                            )
                            for window in microbatch
                        )
                        training_batch = collate_training_steps(
                            samples, pad_token_id=int(self.pad_token_id)
                        )
                        output = self.forward_model(training_batch.batch)
                        losses = cid_loss(output, training_batch.targets)
                        if not bool(torch.isfinite(losses.total)):
                            names = ", ".join(training_batch.example_ids)
                            raise FloatingPointError(
                                f"non-finite CID validation loss for micro-batch: {names}"
                            )
                        batch_size = len(samples)
                        raw_loss = float(losses.total.detach().float()) * batch_size
                        total_raw_loss += raw_loss
                        total_loss += raw_loss * loss_weight
                        total_transitions += batch_size
                        del output, training_batch, losses, samples
        finally:
            self.generator.set_state(generator_state)
            self.shuffle_rng.setstate(shuffle_state)
            self.forward_model.train(was_training)

        return CIDTrainReport(
            transitions=total_transitions,
            optimizer_steps=0,
            mean_loss=total_loss / total_transitions,
            raw_mean_loss=total_raw_loss / total_transitions,
        )

    def _rollout_microbatches(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        shuffle: bool,
        preserve_order: bool,
    ) -> list[tuple[CIDRolloutWindow, ...]]:
        if preserve_order:
            microbatches: list[tuple[CIDRolloutWindow, ...]] = []
            current: list[CIDRolloutWindow] = []
            current_key: tuple[int, float] | None = None
            for window in windows:
                key = (len(window.source_steps), window.loss_weight)
                if current and (key != current_key or len(current) >= self.config.micro_batch_size):
                    microbatches.append(tuple(current))
                    current = []
                current.append(window)
                current_key = key
            if current:
                microbatches.append(tuple(current))
            return microbatches

        buckets: dict[tuple[int, float], list[CIDRolloutWindow]] = {}
        for window in windows:
            key = (len(window.source_steps), window.loss_weight)
            buckets.setdefault(key, []).append(window)
        microbatches = []
        for key in sorted(buckets):
            bucket = buckets[key]
            if shuffle:
                self.shuffle_rng.shuffle(bucket)
            for start in range(0, len(bucket), self.config.micro_batch_size):
                microbatches.append(tuple(bucket[start : start + self.config.micro_batch_size]))
        if shuffle:
            self.shuffle_rng.shuffle(microbatches)
        return microbatches

    def rollout_probability(self) -> float:
        if self.config.rollout_horizon <= 1:
            return 0.0
        completed = self.state.epochs_completed
        if completed < self.config.teacher_forcing_epochs:
            return 0.0
        if self.config.rollout_ramp_epochs == 0:
            return 1.0
        ramp_step = completed - self.config.teacher_forcing_epochs + 1
        return min(1.0, ramp_step / self.config.rollout_ramp_epochs)

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
                rollout_windows_seen_in_epoch=self.state.rollout_windows_seen_in_epoch,
            )
        self.flush()
        mean_loss = sum(losses) / len(losses)
        return CIDTrainReport(
            transitions=len(losses),
            optimizer_steps=self.state.optimizer_steps - start_optimizer_steps,
            mean_loss=mean_loss,
            raw_mean_loss=mean_loss,
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
            rollout_windows_seen_in_epoch=int(
                trainer_state.get("rollout_windows_seen_in_epoch", 0)
            ),
        )
        self.generator.set_state(state["generator_state"])
        self.shuffle_rng.setstate(state["shuffle_state"])
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.optimizer.zero_grad(set_to_none=True)

    def restore_portable_progress_state(
        self,
        state: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        """Restore clean trainer progress after changing the data-parallel world size.

        Gradient accumulation is derived from world size in Stage B, so it is the only
        trainer setting allowed to change. Rank-local RNG streams cannot be preserved
        when ranks are added or removed; reseeding gives every new rank a deterministic
        continuation while model/optimizer state and the global data cursor remain exact.
        """

        saved_config = dict(state["trainer_config"])
        current_config = asdict(self.config)
        saved_config["gradient_accumulation_steps"] = current_config[
            "gradient_accumulation_steps"
        ]
        if saved_config != current_config:
            raise ValueError("checkpoint trainer configuration does not match this trainer")
        trainer_state = state["trainer_state"]
        self.state = CIDTrainerState(
            transitions_seen=int(trainer_state["transitions_seen"]),
            optimizer_steps=int(trainer_state["optimizer_steps"]),
            epochs_completed=int(trainer_state.get("epochs_completed", 0)),
            rollout_windows_seen_in_epoch=0,
        )
        self._pending_accumulation = 0
        self._pending_examples = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.reseed(seed)

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trainable_state = {
            name: parameter.detach().cpu().clone() for name, parameter in self._trainable
        }
        gradient_state = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in self._trainable
            if parameter.grad is not None
        }
        payload = {
            "format_version": self.CHECKPOINT_VERSION,
            "trainer_config": asdict(self.config),
            "trainer_state": asdict(self.state),
            "adapter_config": asdict(self.adapter.config),
            "backbone": {
                "model_type": str(self.adapter.backbone.config.model_type),
                "hidden_size": self.adapter.d_model,
                "vocab_size": self.adapter.vocab_size,
                "mask_token_id": self.adapter.mask_token_id,
            },
            "trainable_names": self.trainable_parameter_names,
            "model_state": trainable_state,
            "optimizer_state": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state().cpu(),
            "shuffle_state": self.shuffle_rng.getstate(),
            "gradient_state": gradient_state,
            "pending_accumulation": self._pending_accumulation,
            "pending_examples": self._pending_examples,
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") not in self.SUPPORTED_CHECKPOINT_VERSIONS:
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
            or int(backbone.get("mask_token_id", self.adapter.mask_token_id))
            != self.adapter.mask_token_id
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
            rollout_windows_seen_in_epoch=int(state.get("rollout_windows_seen_in_epoch", 0)),
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_accumulation = int(checkpoint.get("pending_accumulation", 0))
        self._pending_examples = int(checkpoint.get("pending_examples", 0))
        gradient_state = checkpoint.get("gradient_state", {})
        parameters = dict(self._trainable)
        for name, saved in gradient_state.items():
            parameter = parameters[name]
            parameter.grad = saved.to(device=parameter.device, dtype=parameter.dtype)
        if self._pending_accumulation == 0 and (self._pending_examples or gradient_state):
            raise ValueError("checkpoint contains gradients without pending accumulation")
        if self._pending_accumulation > 0 and self._pending_examples <= 0:
            raise ValueError("checkpoint pending accumulation is missing example count")

    def _optimizer_step(self) -> None:
        if self._pending_examples <= 0:
            raise RuntimeError("optimizer step requires accumulated examples")
        self._set_learning_rate_for_step(self.state.optimizer_steps + 1)
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
            rollout_windows_seen_in_epoch=self.state.rollout_windows_seen_in_epoch,
        )

    def _set_learning_rate_for_step(self, step: int) -> None:
        learning_rate = self._learning_rate_for_step(step)
        for group in self.optimizer.param_groups:
            lr_scale = float(group.get("lr_scale", 1.0))
            if not math.isfinite(lr_scale) or lr_scale <= 0.0:
                raise ValueError("optimizer lr_scale must be finite and positive")
            group["lr"] = learning_rate * lr_scale

    def _learning_rate_for_step(self, step: int) -> float:
        if step <= 0:
            raise ValueError("optimizer step must be positive")
        if self.config.warmup_steps and step <= self.config.warmup_steps:
            scale = step / self.config.warmup_steps
        elif self.config.lr_decay_steps:
            decay_span = max(1, self.config.lr_decay_steps - self.config.warmup_steps)
            progress = min(
                1.0,
                max(0.0, (step - self.config.warmup_steps) / decay_span),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            floor = self.config.min_learning_rate_ratio
            scale = floor + (1.0 - floor) * cosine
        else:
            scale = 1.0
        return self.config.learning_rate * scale

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
    if checkpoint.get("format_version") not in CIDTrainer.SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError("unsupported CID trainer checkpoint version")
    backbone = checkpoint["backbone"]
    if (
        int(backbone["hidden_size"]) != adapter.d_model
        or int(backbone["vocab_size"]) != adapter.vocab_size
        or str(backbone["model_type"]) != str(adapter.backbone.config.model_type)
        or int(backbone.get("mask_token_id", adapter.mask_token_id)) != adapter.mask_token_id
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
        rollout_windows_seen_in_epoch=int(state.get("rollout_windows_seen_in_epoch", 0)),
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
        display_replacement_fraction: float = 0.25,
        minimum_thought_slots: int = 8,
        display_canvas_tokens: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder or ILLaDATextEncoder(adapter, tokenizer)
        if self.text_encoder.d_model != adapter.d_model:
            raise ValueError("training text encoder width must match the CID adapter")
        if not 0.0 <= display_replacement_fraction <= 1.0:
            raise ValueError("display_replacement_fraction must be in [0, 1]")
        if minimum_thought_slots <= 0:
            raise ValueError("minimum_thought_slots must be positive")
        self.display_replacement_fraction = display_replacement_fraction
        self.minimum_thought_slots = min(
            minimum_thought_slots,
            adapter.config.max_thought_slots,
        )
        if display_canvas_tokens is None:
            display_canvas_tokens = adapter.config.display_canvas_tokens
        if not 1 < display_canvas_tokens <= adapter.config.max_display_tokens:
            raise ValueError(
                "display_canvas_tokens must be in [2, adapter max_display_tokens]"
            )
        self.display_canvas_tokens = int(display_canvas_tokens)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = (
            adapter.eos_token_id if eos_token_id is None else int(eos_token_id)
        )
        self.scheduler = scheduler or CIDDiffusionScheduler(adapter.mask_token_id)

    def tensorize(
        self,
        example: TrajectoryExample,
        source_step: int,
        *,
        timestep: float = 0.5,
        generator: torch.Generator | None = None,
        rollout_state: CIDRolloutState | None = None,
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
        capacity = self._trajectory_thought_capacity(example)
        target_by_id = {cell.cell_id: cell for cell in target}
        target_output_slots = self._target_output_slots(current, target, capacity)

        current_vectors = self._semantic_vectors(current)
        target_vectors = self._semantic_vectors(target)
        thought_semantic = torch.zeros(
            (1, capacity, self.adapter.d_model), device=device, dtype=dtype
        )
        role_features = torch.zeros(
            (1, capacity, self.adapter.config.num_roles), device=device, dtype=dtype
        )
        lifecycle_features = torch.zeros(
            (1, capacity, self.adapter.config.num_lifecycles), device=device, dtype=dtype
        )
        uncertainty = torch.ones((1, capacity, 1), device=device, dtype=dtype)
        occupancy = torch.zeros((1, capacity, 1), device=device, dtype=dtype)
        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES

        if rollout_state is None:
            for cell in current:
                thought_semantic[0, cell.slot] = current_vectors[cell.cell_id]
                occupancy[0, cell.slot, 0] = 1.0
                uncertainty[0, cell.slot, 0] = cell.uncertainty
                lifecycle_features[0, cell.slot, lifecycle_order.index(cell.lifecycle)] = 1.0
                for role_index, role in enumerate(role_order):
                    role_features[0, cell.slot, role_index] = cell.roles.get(role, 0.0)
        else:
            self._validate_rollout_state(rollout_state, capacity)
            thought_semantic.copy_(rollout_state.thought_semantic.to(device=device, dtype=dtype))
            role_features.copy_(rollout_state.role_features.to(device=device, dtype=dtype))
            uncertainty.copy_(rollout_state.uncertainty.to(device=device, dtype=dtype))
            lifecycle_features.copy_(
                rollout_state.lifecycle_features.to(device=device, dtype=dtype)
            )
            occupancy.copy_(rollout_state.slot_occupancy.to(device=device, dtype=dtype))

        timestep_tensor = torch.tensor([timestep], device=device)
        thought_timesteps = (
            timestep_tensor
            if rollout_state is None
            else rollout_state.local_noise.to(device=device, dtype=torch.float32).squeeze(-1)
        )
        thought_corruption = self.scheduler.corrupt_thought(
            thought_semantic,
            thought_timesteps,
            occupancy,
            generator=generator,
        )

        prompt_ids = self.text_encoder.tokenize(example.prompt, add_special_tokens=True)
        target_display = self._display_text(example, target_step)
        target_text_ids = self.text_encoder.tokenize(target_display, add_special_tokens=False)
        realized_tokens = target_text_ids.shape[1]
        display_canvas_tokens = self._display_canvas_size(
            realized_tokens + 1,
            rollout_state=rollout_state,
        )
        logical_length = (
            self.adapter.config.max_thought_slots
            + prompt_ids.shape[1]
            + display_canvas_tokens
        )
        if logical_length > self.adapter.max_position_embeddings:
            raise ValueError(
                "configured TCT prefix, prompt, and display bucket exceed "
                "backbone context capacity"
            )

        target_display_ids = torch.full(
            (1, display_canvas_tokens),
            self.adapter.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        target_display_ids[:, :realized_tokens] = target_text_ids
        target_display_ids[:, realized_tokens] = self.eos_token_id
        display_supervision_mask = torch.zeros_like(target_display_ids, dtype=torch.bool)
        display_supervision_mask[:, : realized_tokens + 1] = True

        if rollout_state is None:
            display_corruption = self.scheduler.corrupt_display(
                target_display_ids,
                timestep_tensor,
                eligible_mask=display_supervision_mask,
                vocab_size=self.adapter.vocab_size,
                replacement_fraction=self.display_replacement_fraction,
                generator=generator,
            )
            display_input_ids = display_corruption.token_ids
            display_labels = display_corruption.labels
            display_noise = display_corruption.noise * display_supervision_mask.unsqueeze(-1)
        else:
            display_input_ids = torch.full_like(
                target_display_ids, self.adapter.mask_token_id
            )
            previous_display = rollout_state.display_ids.to(device=device, dtype=torch.long)
            if previous_display.shape[1] > display_input_ids.shape[1]:
                raise ValueError("rollout display exceeds configured training display capacity")
            display_input_ids[:, : previous_display.shape[1]] = previous_display
            display_labels = target_display_ids.clone()
            display_labels[~display_supervision_mask] = -100
            display_labels[display_input_ids == target_display_ids] = -100
            display_noise = torch.full(
                (*display_input_ids.shape, 1),
                rollout_state.display_noise_level,
                device=device,
                dtype=dtype,
            )

        fact_memory = self.text_encoder.encode_texts(
            tuple(
                f"fact={key} | value={stable_text(value)}"
                for key, value in example.protected_facts.items()
            )
        )
        available_events = tuple(
            event for event in example.events if event.arrival_step <= target_step
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
                for event in available_events
            )
        )
        percept_thought_mask, percept_display_mask = self._percept_target_masks(
            example,
            available_events,
            target_output_slots=target_output_slots,
            target_step=target_step,
            display_length=display_input_ids.shape[1],
            thought_slots=capacity,
            device=device,
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
            display_ids=display_input_ids,
            display_noise=display_noise,
            fact_memory=fact_memory,
            percept_memory=percept_memory,
            source_memory=source_memory,
            lifecycle_features=lifecycle_features,
            percept_thought_mask=percept_thought_mask,
            percept_display_mask=percept_display_mask,
        )
        targets = self._targets(
            example=example,
            source_step=source_step,
            target_step=target_step,
            target_by_id=target_by_id,
            target_output_slots=target_output_slots,
            target_vectors=target_vectors,
            display_labels=display_labels,
            input_occupancy=occupancy,
            input_noise_level=thought_corruption.noise,
            thought_slots=capacity,
            dtype=dtype,
            device=device,
        )
        return CIDTrainingStep(
            example_id=example.example_id,
            source_step=source_step,
            target_step=target_step,
            diffusion_step=self._runtime_diffusion_step(example, source_step),
            next_diffusion_step=self._runtime_diffusion_step(example, target_step),
            batch=batch,
            targets=targets,
        )

    def _percept_target_masks(
        self,
        example: TrajectoryExample,
        events: tuple[Any, ...],
        *,
        target_output_slots: Mapping[str, int],
        target_step: int,
        display_length: int,
        thought_slots: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        cell_targets: list[tuple[ObjectRef, ...]] = []
        display_targets: list[tuple[ObjectRef, ...]] = []
        for event in events:
            bindings = tuple(
                binding
                for binding in example.binding_targets
                if binding.first_need_step <= target_step
                and binding.source == event.source
                and dict(binding.arguments) == dict(event.arguments)
            )
            cell_targets.append(
                tuple(target for binding in bindings for target in binding.target_cells)
            )
            display_targets.append(
                tuple(target for binding in bindings for target in binding.target_display)
            )
        return build_percept_routing_masks(
            tuple(cell_targets),
            tuple(display_targets),
            cell_slots=target_output_slots,
            thought_slots=thought_slots,
            display_length=display_length,
            device=device,
        )

    def _validate_rollout_state(self, state: CIDRolloutState, thought_slots: int) -> None:
        expected_thought = (1, thought_slots, self.adapter.d_model)
        if tuple(state.thought_semantic.shape) != expected_thought:
            raise ValueError("rollout thought semantic shape does not match adapter geometry")
        expected_roles = (
            1,
            thought_slots,
            self.adapter.config.num_roles,
        )
        if tuple(state.role_features.shape) != expected_roles:
            raise ValueError("rollout role feature shape does not match adapter geometry")
        expected_scalar = (1, thought_slots, 1)
        if tuple(state.uncertainty.shape) != expected_scalar:
            raise ValueError("rollout uncertainty shape does not match adapter geometry")
        expected_lifecycle = (1, thought_slots, self.adapter.config.num_lifecycles)
        if tuple(state.lifecycle_features.shape) != expected_lifecycle:
            raise ValueError("rollout lifecycle feature shape does not match adapter geometry")
        if tuple(state.slot_occupancy.shape) != expected_scalar:
            raise ValueError("rollout occupancy shape does not match adapter geometry")
        if tuple(state.local_noise.shape) != expected_scalar:
            raise ValueError("rollout local-noise shape does not match adapter geometry")
        if state.display_ids.ndim != 2 or state.display_ids.shape[0] != 1:
            raise ValueError("rollout display IDs must have shape [1, tokens]")
        if not 0.0 <= state.display_noise_level <= 1.0:
            raise ValueError("rollout display noise level must be in [0, 1]")


    def _targets(
        self,
        *,
        example: TrajectoryExample,
        source_step: int,
        target_step: int,
        target_by_id: Mapping[str, ThoughtTarget],
        target_output_slots: Mapping[str, int],
        target_vectors: Mapping[str, Tensor],
        display_labels: Tensor,
        input_occupancy: Tensor,
        input_noise_level: Tensor,
        thought_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> CIDTargets:
        del source_step
        c = self.adapter.config
        n = thought_slots
        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES
        anchor_order = tuple(AnchorKind)
        relation_order = tuple(LinkRelation)
        object_order = tuple(ObjectKind)
        freshness_order = tuple(FreshnessDemand)
        final_step = max(target.step for target in example.thought_targets)
        waiting_equilibrium = any(
            target.lifecycle is CellLifecycle.WAITING for target in target_by_id.values()
        )

        thought_target = torch.zeros((1, n, self.adapter.d_model), device=device, dtype=dtype)
        thought_mask = torch.zeros((1, n), device=device, dtype=torch.bool)
        convergence_targets = torch.tensor(
            [float(target_step == final_step or waiting_equilibrium)],
            device=device,
            dtype=dtype,
        )
        allocation_targets = torch.zeros((1, n), device=device, dtype=dtype)
        allocation_mask = ~input_occupancy.squeeze(-1).bool()
        role_targets = torch.zeros((1, n, c.num_roles), device=device, dtype=dtype)
        uncertainty = torch.ones((1, n, 1), device=device, dtype=dtype)
        noise_delta = torch.zeros((1, n, 1), device=device, dtype=dtype)
        lifecycle = torch.full((1, n), -100, device=device, dtype=torch.long)
        need_targets = torch.zeros((1, n, c.max_need_slots), device=device, dtype=dtype)
        source_targets = torch.full(
            (1, n, c.max_need_slots), -100, device=device, dtype=torch.long
        )
        revision_targets = torch.full((1, n), -100, device=device, dtype=torch.long)
        refresh_targets = torch.full(
            (1, n, c.max_need_slots), -100, device=device, dtype=torch.long
        )

        argument_presence_targets = torch.zeros(
            (1, n, c.max_need_slots, c.max_argument_slots), device=device, dtype=dtype
        )
        argument_presence_mask = torch.zeros(
            (1, n, c.max_need_slots, c.max_argument_slots),
            device=device,
            dtype=torch.bool,
        )
        argument_embeddings = torch.zeros(
            (1, n, c.max_need_slots, c.max_argument_slots, self.adapter.d_model),
            device=device,
            dtype=dtype,
        )
        argument_mask = torch.zeros(
            (1, n, c.max_need_slots, c.max_argument_slots),
            device=device,
            dtype=torch.bool,
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
        link_presence_targets = torch.zeros((1, n, c.max_link_slots), device=device, dtype=dtype)
        link_presence_mask = torch.zeros((1, n, c.max_link_slots), device=device, dtype=torch.bool)
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
            actually_occupied = bool(input_occupancy[0, slot, 0])
            if not actually_occupied:
                # Allocation/lifecycle are trained against the state actually fed to the model.
                # A scheduled-sampling miss must therefore remain recoverable on the next step.
                allocation_targets[0, slot] = 1.0
                noise_delta[0, slot, 0] = target.noise - 1.0
            else:
                lifecycle[0, slot] = lifecycle_order.index(target.lifecycle)
                delta = target.noise - float(input_noise_level[0, slot, 0])
                noise_delta[0, slot, 0] = delta
                if delta > 1e-6:
                    revision_targets[0, slot] = int(RevisionAction.REOPEN)
                elif delta < -1e-6:
                    revision_targets[0, slot] = int(RevisionAction.STABILIZE)
                else:
                    revision_targets[0, slot] = int(RevisionAction.KEEP)

        target_slots = set(target_output_slots.values())
        retired_index = lifecycle_order.index(CellLifecycle.RETIRED)
        for slot in range(n):
            if bool(input_occupancy[0, slot, 0]) and slot not in target_slots:
                # Over-allocation is a structural rollout error. Teach the lifecycle head to
                # retire the extra occupied cell rather than silently carrying it forever.
                lifecycle[0, slot] = retired_index

        source_names = tuple(str(item.get("name", "")) for item in example.source_descriptors)
        binding_slots = self._binding_slot_schedule(example)
        bindings = tuple(
            binding for binding in example.binding_targets if binding.first_need_step <= target_step
        )
        for binding in bindings:
            if binding.source not in source_names:
                raise ValueError(f"binding target references unknown source {binding.source!r}")
            source_index = source_names.index(binding.source)
            descriptor = example.source_descriptors[source_index]
            declared_arguments = tuple(descriptor.get("arguments", ()))
            need_is_active = not (
                binding.freshness is FreshnessDemand.ONCE
                and _binding_observation_available(example, binding, target_step)
            )
            for cell_ref in binding.target_cells:
                slot = target_output_slots.get(cell_ref.identifier)
                if slot is None:
                    continue
                if not need_is_active:
                    # A one-shot request is complete once its matching observation is visible.
                    # The explicit zero need target remains supervised for this occupied slot.
                    continue
                need_slot = binding_slots[(cell_ref.identifier, binding.need_id)]
                need_targets[0, slot, need_slot] = binding.confidence
                source_targets[0, slot, need_slot] = source_index
                refresh_targets[0, slot, need_slot] = freshness_order.index(binding.freshness)
                for argument_slot, argument in enumerate(
                    declared_arguments[: c.max_argument_slots]
                ):
                    argument_presence_mask[0, slot, need_slot, argument_slot] = True
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
                        argument_presence_targets[0, slot, need_slot, argument_slot] = 1.0
                        argument_embeddings[0, slot, need_slot, argument_slot] = (
                            self.text_encoder.encode_one(
                                stable_text(binding.arguments[name]), detach=True
                            )
                        )
                        argument_mask[0, slot, need_slot, argument_slot] = True

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
            convergence_targets=convergence_targets,
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

    @staticmethod
    def _runtime_diffusion_step(example: TrajectoryExample, source_step: int) -> int:
        if source_step <= 0:
            return 0
        arrivals = [
            event.arrival_step for event in example.events if event.arrival_step <= source_step
        ]
        epoch_start = max(arrivals, default=0)
        return max(0, source_step - epoch_start)

    def _binding_slot_schedule(
        self, example: TrajectoryExample
    ) -> dict[tuple[str, str], int]:
        by_cell: dict[str, list[Any]] = {}
        for binding in example.binding_targets:
            for target in binding.target_cells:
                by_cell.setdefault(target.identifier, []).append(binding)

        schedule: dict[tuple[str, str], int] = {}
        for cell_id, bindings in by_cell.items():
            ordered = sorted(bindings, key=lambda item: (item.first_need_step, item.need_id))
            if len(ordered) > self.adapter.config.max_need_slots:
                raise ValueError(
                    f"cell {cell_id!r} requires {len(ordered)} information-need slots but "
                    f"adapter supports {self.adapter.config.max_need_slots}"
                )
            for need_slot, binding in enumerate(ordered):
                schedule[(cell_id, binding.need_id)] = need_slot
        return schedule

    def _thought_snapshot(self, example: TrajectoryExample, step: int) -> tuple[ThoughtTarget, ...]:
        canonical = self._canonical_slot_schedule(example)
        snapshot = tuple(
            replace(target, slot=canonical[(target.step, target.cell_id)])
            for target in example.thought_targets
            if target.step == step
        )
        return tuple(sorted(snapshot, key=lambda target: target.slot))

    def _trajectory_thought_capacity(self, example: TrajectoryExample) -> int:
        counts: dict[int, int] = {}
        for target in example.thought_targets:
            counts[target.step] = counts.get(target.step, 0) + 1
        required = max(counts.values(), default=1)
        maximum = self.adapter.config.max_thought_slots
        if required > maximum:
            raise ValueError(
                f"trajectory requires {required} simultaneous thought slots but adapter "
                f"supports {maximum}"
            )
        return max(self.minimum_thought_slots, required)

    def _canonical_slot_schedule(
        self, example: TrajectoryExample
    ) -> dict[tuple[int, str], int]:
        capacity = self._trajectory_thought_capacity(example)
        active_slots: dict[str, int] = {}
        release_before_next: set[str] = set()
        schedule: dict[tuple[int, str], int] = {}
        steps = sorted({target.step for target in example.thought_targets})

        for step in steps:
            reserved_reclaimed_slots = {
                active_slots[cell_id]
                for cell_id in release_before_next
                if cell_id in active_slots
            }
            for cell_id in release_before_next:
                active_slots.pop(cell_id, None)
            release_before_next = set()
            snapshot = tuple(
                sorted(
                    (target for target in example.thought_targets if target.step == step),
                    key=lambda target: target.cell_id,
                )
            )
            snapshot_ids = {target.cell_id for target in snapshot}
            stale = set(active_slots) - snapshot_ids
            if stale:
                names = ", ".join(sorted(stale))
                raise ValueError(f"thought trajectory removed cells without retirement: {names}")

            used = set(active_slots.values()) | reserved_reclaimed_slots
            for target in snapshot:
                if target.cell_id not in active_slots:
                    try:
                        slot = next(slot for slot in range(capacity) if slot not in used)
                    except StopIteration as exc:
                        raise ValueError(
                            "thought trajectory exceeds canonical slot capacity"
                        ) from exc
                    active_slots[target.cell_id] = slot
                    used.add(slot)
                schedule[(step, target.cell_id)] = active_slots[target.cell_id]
                if target.lifecycle is CellLifecycle.RETIRED:
                    release_before_next.add(target.cell_id)

        return schedule

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
        occupied_source_slots = {
            cell.slot for cell in current if cell.lifecycle is not CellLifecycle.RETIRED
        }
        slots: dict[str, int] = {}
        used: set[int] = set()
        for cell in target:
            if cell.cell_id in current_by_id:
                slot = current_by_id[cell.cell_id].slot
            else:
                slot = cell.slot
                if slot in occupied_source_slots:
                    raise ValueError(
                        "new thought target cannot allocate into an occupied source slot"
                    )
            if not 0 <= slot < capacity or slot in used:
                raise ValueError(
                    "target thought transition has an invalid or colliding output slot"
                )
            slots[cell.cell_id] = slot
            used.add(slot)
        target_ids = {cell.cell_id for cell in target}
        missing = {
            cell_id
            for cell_id, cell in current_by_id.items()
            if cell_id not in target_ids and cell.lifecycle is not CellLifecycle.RETIRED
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"target snapshot removed cells without RETIRED state: {names}")
        return slots

    def _display_canvas_size(
        self,
        required_tokens: int,
        *,
        rollout_state: CIDRolloutState | None,
    ) -> int:
        if required_tokens <= 0:
            raise ValueError("display supervision requires at least EOS")
        maximum = self.adapter.config.max_display_tokens
        if required_tokens > maximum:
            raise ValueError(
                f"display target requires {required_tokens} tokens including EOS but "
                f"adapter maximum is {maximum}"
            )
        canvas = self.display_canvas_tokens
        while canvas < required_tokens:
            canvas = min(maximum, canvas * 2)
        if rollout_state is not None:
            canvas = max(canvas, int(rollout_state.display_ids.shape[1]))
        return canvas

    def _display_text(self, example: TrajectoryExample, step: int) -> str:
        for target in example.display_targets:
            if target.step == step:
                return target.text
        return example.target_display


def _cat_targets(steps: tuple[CIDTrainingStep, ...], name: str) -> Tensor:
    return torch.cat(tuple(getattr(step.targets, name) for step in steps), dim=0)


def _pad_slot_tensors(
    tensors: tuple[Tensor, ...],
    *,
    pad_value: int | float = 0,
) -> Tensor:
    if not tensors:
        raise ValueError("cannot pad an empty slot tensor collection")
    if any(tensor.ndim < 2 or tensor.shape[0] != 1 for tensor in tensors):
        raise ValueError("slot tensors must have shape [1, slots, ...]")
    trailing = tensors[0].shape[2:]
    if any(tensor.shape[2:] != trailing for tensor in tensors):
        raise ValueError("slot tensors in one batch must share trailing dimensions")
    max_slots = max(tensor.shape[1] for tensor in tensors)
    output = tensors[0].new_full((len(tensors), max_slots, *trailing), pad_value)
    for row, tensor in enumerate(tensors):
        output[row, : tensor.shape[1]] = tensor[0]
    return output


def _pad_slot_targets(
    steps: tuple[CIDTrainingStep, ...],
    name: str,
    *,
    pad_value: int | float = 0,
) -> Tensor:
    return _pad_slot_tensors(
        tuple(getattr(step.targets, name) for step in steps),
        pad_value=pad_value,
    )


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


def _collate_percept_masks(steps: tuple[CIDTrainingStep, ...], kind: str) -> Tensor | None:
    if kind not in {"thought", "display"}:
        raise ValueError("percept mask kind must be thought or display")
    attribute = f"percept_{kind}_mask"
    masks = tuple(getattr(step.batch, attribute) for step in steps)
    if all(mask is None for mask in masks):
        return None
    query_lengths = tuple(
        step.batch.thought_semantic.shape[1]
        if kind == "thought"
        else step.batch.display_ids.shape[1]
        for step in steps
    )
    percept_lengths = tuple(step.batch.percept_memory.shape[1] for step in steps)
    output = torch.zeros(
        (len(steps), max(query_lengths), max(percept_lengths, default=0)),
        dtype=torch.bool,
        device=steps[0].batch.thought_semantic.device,
    )
    for row, (_step, mask, query_length, percept_length) in enumerate(
        zip(steps, masks, query_lengths, percept_lengths, strict=True)
    ):
        if mask is None:
            output[row, :query_length, :percept_length] = True
            continue
        expected = (1, query_length, percept_length)
        if tuple(mask.shape) != expected:
            raise ValueError(f"{attribute} must have shape {expected}, got {tuple(mask.shape)}")
        output[row, :query_length, :percept_length] = mask[0].to(dtype=torch.bool)
    return output


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
            f"streamable={bool(descriptor.get('streamable', False))}",
            f"versioned={bool(descriptor.get('versioned', False))}",
            f"accepts_partial_arguments={bool(descriptor.get('accepts_partial_arguments', False))}",
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
        steps = (target.step for target in example.thought_targets)
        transitions.extend((example, step) for step in training_transition_source_steps(steps))
    return tuple(transitions)


def trajectory_rollout_windows(
    examples: tuple[TrajectoryExample, ...],
    *,
    max_horizon: int,
) -> tuple[CIDRolloutWindow, ...]:
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    windows: list[CIDRolloutWindow] = []
    for example in examples:
        steps = (target.step for target in example.thought_targets)
        source_steps = training_transition_source_steps(steps)
        if not source_steps:
            continue
        run: list[int] = [source_steps[0]]
        runs: list[tuple[int, ...]] = []
        for step in source_steps[1:]:
            if step == run[-1] + 1:
                run.append(step)
            else:
                runs.append(tuple(run))
                run = [step]
        runs.append(tuple(run))
        for contiguous in runs:
            for start in range(0, len(contiguous), max_horizon):
                windows.append(
                    CIDRolloutWindow(
                        example=example,
                        source_steps=contiguous[start : start + max_horizon],
                    )
                )
    return tuple(windows)


def _binding_observation_available(
    example: TrajectoryExample,
    binding: Any,
    target_step: int,
) -> bool:
    """Whether this binding's own observation is visible by ``target_step``."""

    return any(
        event.arrival_step <= target_step
        and event.source == binding.source
        and dict(event.arguments) == dict(binding.arguments)
        for event in example.events
    )


def balance_rollout_windows_by_semantic_task(
    windows: tuple[CIDRolloutWindow, ...],
) -> tuple[CIDRolloutWindow, ...]:
    """Importance-weight rollout transitions by semantic task and declared task weight.

    Schedule variants and long trajectories remain fully present, while every semantic task receives
    a total loss mass proportional to ``metadata.training_weight`` (default 1.0).  The global
    transition-weight mean remains exactly one, so component reweighting changes only the mixture
    distribution and does not silently change the optimizer's overall loss scale.
    """

    if not windows:
        return ()
    transition_counts: dict[str, int] = {}
    task_weights: dict[str, float] = {}
    task_ids: list[str] = []
    total_transitions = 0
    for window in windows:
        task_id = str(window.example.metadata.get("semantic_task_id") or window.example.example_id)
        task_weight = float(window.example.metadata.get("training_weight", 1.0))
        if not math.isfinite(task_weight) or task_weight <= 0.0:
            raise ValueError("training_weight must be finite and positive")
        previous_weight = task_weights.get(task_id)
        if previous_weight is not None and not math.isclose(previous_weight, task_weight):
            raise ValueError(f"semantic task {task_id!r} has conflicting training_weight values")
        task_weights[task_id] = task_weight
        count = len(window.source_steps)
        transition_counts[task_id] = transition_counts.get(task_id, 0) + count
        task_ids.append(task_id)
        total_transitions += count
    task_count = len(transition_counts)
    if task_count == 0 or total_transitions == 0:
        return windows
    total_task_weight = sum(task_weights.values())
    weights = {
        task_id: total_transitions * task_weights[task_id] / (total_task_weight * transition_count)
        for task_id, transition_count in transition_counts.items()
    }
    return tuple(
        replace(window, loss_weight=weights[task_id])
        for window, task_id in zip(windows, task_ids, strict=True)
    )


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


def _stage_b_rollout_bucket_key(window: CIDRolloutWindow) -> str:
    return f"{len(window.source_steps)}:{float(window.loss_weight).hex()}"


def stage_b_consumed_windows_by_bucket(
    windows: tuple[CIDRolloutWindow, ...],
    local_windows: tuple[CIDRolloutWindow, ...],
    *,
    local_windows_seen: int,
    world_size: int,
    base_consumed_by_bucket: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return a world-size-portable cursor for a synchronized Stage B shard prefix."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= local_windows_seen <= len(local_windows):
        raise ValueError("local_windows_seen is outside the Stage B shard")
    totals: dict[str, int] = {}
    for window in windows:
        key = _stage_b_rollout_bucket_key(window)
        totals[key] = totals.get(key, 0) + 1
    consumed = {key: int(value) for key, value in (base_consumed_by_bucket or {}).items()}
    for key, value in consumed.items():
        if key not in totals or value < 0 or value > totals[key]:
            raise ValueError("invalid Stage B base bucket cursor")
    local_counts: dict[str, int] = {}
    for window in local_windows[:local_windows_seen]:
        key = _stage_b_rollout_bucket_key(window)
        local_counts[key] = local_counts.get(key, 0) + 1
    for key, local_count in local_counts.items():
        consumed[key] = min(
            totals[key],
            consumed.get(key, 0) + local_count * world_size,
        )
    return {key: value for key, value in consumed.items() if value}


def shard_rollout_windows(
    windows: tuple[CIDRolloutWindow, ...],
    *,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
    micro_batch_size: int = 1,
    consumed_windows_by_bucket: Mapping[str, int] | None = None,
    legacy_resume_padding: bool = False,
) -> tuple[CIDRolloutWindow, ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if not windows:
        return ()
    local_microbatches: list[tuple[CIDRolloutWindow, ...]] = []
    resume_repair_microbatches: list[tuple[CIDRolloutWindow, ...]] = []
    keys = sorted({(len(window.source_steps), window.loss_weight) for window in windows})
    for bucket_index, key in enumerate(keys):
        length, loss_weight = key
        bucket = [
            window
            for window in windows
            if len(window.source_steps) == length and window.loss_weight == loss_weight
        ]
        if shuffle:
            random.Random(seed + epoch * 1009 + length * 100_003 + bucket_index).shuffle(bucket)
        portable_key = _stage_b_rollout_bucket_key(bucket[0])
        consumed = int((consumed_windows_by_bucket or {}).get(portable_key, 0))
        if consumed < 0 or consumed > len(bucket):
            raise ValueError("invalid Stage B consumed bucket cursor")
        if consumed:
            bucket = bucket[consumed:]
        if not bucket:
            continue
        padding = (-len(bucket)) % world_size
        if padding:
            original = tuple(bucket)
            if legacy_resume_padding:
                # Older Stage-A checkpoints were produced with ``bucket[:padding]``.
                # For buckets smaller than the requested padding that under-filled
                # the global shard, so some ranks finished the epoch early.  Keep
                # the old shuffled prefix intact for an in-flight checkpoint and
                # append only the missing duplicate windows after the legacy
                # micro-batch shuffle below.
                bucket.extend(original[:padding])
            else:
                bucket.extend(original[index % len(original)] for index in range(padding))
        local_bucket = bucket[rank::world_size]
        if legacy_resume_padding:
            target_local_windows = math.ceil(len(original) / world_size)
            missing_local_windows = target_local_windows - len(local_bucket)
            if missing_local_windows > 0:
                repairs = tuple(
                    original[index % len(original)] for index in range(missing_local_windows)
                )
                for start in range(0, len(repairs), micro_batch_size):
                    resume_repair_microbatches.append(
                        repairs[start : start + micro_batch_size]
                    )
        for start in range(0, len(local_bucket), micro_batch_size):
            local_microbatches.append(tuple(local_bucket[start : start + micro_batch_size]))
    if shuffle:
        random.Random(seed + epoch * 1_000_003 + 97).shuffle(local_microbatches)
    if resume_repair_microbatches:
        local_microbatches.extend(resume_repair_microbatches)
    return tuple(window for microbatch in local_microbatches for window in microbatch)


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
        "find_unused_parameters": False,
    }
    if "forward_sync_buffers" in signature(DistributedDataParallel).parameters:
        kwargs["forward_sync_buffers"] = False
    else:
        kwargs["broadcast_buffers"] = False
    return DistributedDataParallel(adapter, **kwargs)


def stage_b_gradient_accumulation_steps(
    *,
    world_size: int,
    micro_batch_size: int,
    target_global_batch_size: int,
    explicit_steps: int | None = None,
) -> int:
    """Resolve accumulation to the closest realizable transition batch."""

    if world_size <= 0 or micro_batch_size <= 0 or target_global_batch_size <= 0:
        raise ValueError("Stage B batch dimensions must be positive")
    if explicit_steps is not None:
        if explicit_steps <= 0:
            raise ValueError("explicit Stage B gradient accumulation must be positive")
        return explicit_steps
    data_parallel_micro_batch = world_size * micro_batch_size
    return max(
        1,
        math.floor(target_global_batch_size / data_parallel_micro_batch + 0.5),
    )


def stage_b_optimizer_steps_per_epoch(
    windows: tuple[CIDRolloutWindow, ...],
    *,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return exact optimizer steps implied by Stage B bucket padding/sharding."""

    if world_size <= 0 or micro_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Stage B optimizer-step dimensions must be positive")
    if not windows:
        return 0
    bucket_counts: dict[tuple[int, float], int] = {}
    for window in windows:
        key = (len(window.source_steps), window.loss_weight)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    microbatches_per_rank = 0
    for (window_length, _), count in bucket_counts.items():
        local_windows = math.ceil(count / world_size)
        local_microbatches = math.ceil(local_windows / micro_batch_size)
        microbatches_per_rank += local_microbatches * window_length
    return math.ceil(microbatches_per_rank / gradient_accumulation_steps)


def stage_b_adamw_parameter_groups(
    adapter: ILLaDACIDAdapter,
    *,
    backbone_lr_scale: float = 0.5,
    weight_decay: float = 0.01,
) -> list[dict[str, object]]:
    """Build stable AdamW groups for Stage B before FSDP wrapping.

    ``use_orig_params=True`` keeps these original ``nn.Parameter`` objects valid after
    wrapping. Matrix/tensor weights receive AdamW decay while one-dimensional norm
    weights and biases do not. The already-trained CID modules keep the base learning
    rate; the pretrained diffusion backbone uses a conservative multiplier to reduce
    catastrophic forgetting during the one-epoch joint continuation.
    """

    if not math.isfinite(backbone_lr_scale) or backbone_lr_scale <= 0.0:
        raise ValueError("backbone_lr_scale must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")

    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        use_decay = parameter.ndim >= 2
        buckets[(is_backbone, use_decay)].append(parameter)

    groups: list[dict[str, object]] = []
    for is_backbone, use_decay in ((True, True), (True, False), (False, True), (False, False)):
        parameters = buckets[(is_backbone, use_decay)]
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "weight_decay": weight_decay if use_decay else 0.0,
                "lr_scale": backbone_lr_scale if is_backbone else 1.0,
                "group_name": (
                    f"{'backbone' if is_backbone else 'cid'}-"
                    f"{'decay' if use_decay else 'no-decay'}"
                ),
            }
        )
    if not groups:
        raise ValueError("Stage B AdamW requires trainable parameters")
    return groups


def wrap_stage_b_fsdp(
    adapter: ILLaDACIDAdapter,
    *,
    device_id: int | torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
) -> torch.nn.Module:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    decoder = adapter.hidden_backbone()
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("diffusion backbone must expose non-empty layers for FSDP auto-wrap")
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
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def _ensure_stage_b_npu_sharded_tensor_compatibility() -> None:
    """Patch the PyTorch 2.1 ShardedTensor device fallback on Ascend-only ranks."""

    npu = getattr(torch, "npu", None)
    torch_version = str(torch.__version__).split("+", 1)[0]
    if (
        npu is None
        or not npu.is_available()
        or torch.cuda.is_available()
        or not torch_version.startswith("2.1.")
    ):
        return
    try:
        from torch.distributed._shard.sharded_tensor.api import _SHARDED_OPS, ShardedTensor
    except ImportError:
        return

    key = torch.Tensor.device.__get__
    current = _SHARDED_OPS.get(key)
    if getattr(current, "_cid_npu_compatible", False):
        return

    def npu_tensor_device(types, args=(), kwargs=None, pg=None):
        del types, kwargs
        sharded = args[0]
        if not isinstance(sharded, ShardedTensor):
            raise TypeError("input needs to be a ShardedTensor")
        if sharded._local_shards:
            return sharded._local_shards[0].tensor.device
        if pg and pg._get_backend_name() == "gloo":
            return torch.device("cpu")
        return torch.device("npu", npu.current_device())

    npu_tensor_device._cid_npu_compatible = True
    _SHARDED_OPS[key] = npu_tensor_device


def _stage_b_dcp_save(state: Mapping[str, Any], destination: Path) -> None:
    import torch.distributed.checkpoint as dcp

    if hasattr(dcp, "save"):
        dcp.save(state, checkpoint_id=destination)
    else:
        dcp.save_state_dict(
            state,
            storage_writer=dcp.FileSystemWriter(destination),
        )


def _stage_b_dcp_load(state: Mapping[str, Any], source: Path) -> None:
    import torch.distributed.checkpoint as dcp

    if hasattr(dcp, "load"):
        dcp.load(state, checkpoint_id=source)
    else:
        dcp.load_state_dict(
            state,
            storage_reader=dcp.FileSystemReader(source),
        )


def _stage_b_dcp_load_optimizer_state(
    model_state: Mapping[str, Any],
    source: Path,
) -> Mapping[str, Any]:
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint.optimizer as dcp_optimizer

    loader = getattr(dcp_optimizer, "load_sharded_optimizer_state_dict", None)
    if loader is None:
        loader = dcp.load_sharded_optimizer_state_dict
    return loader(
        model_state,
        optimizer_key="optimizer",
        storage_reader=dcp.FileSystemReader(source),
    )["optimizer"]


def _stage_b_use_rank_local_optimizer_checkpoint() -> bool:
    """Use the stable same-world-size optimizer format on the Ascend PyTorch 2.1 stack."""

    npu = getattr(torch, "npu", None)
    return (
        npu is not None
        and npu.is_available()
        and not torch.cuda.is_available()
        and str(torch.__version__).split("+", 1)[0].startswith("2.1.")
    )


def _stage_b_move_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _stage_b_move_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stage_b_move_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stage_b_move_state_to_cpu(item) for item in value)
    return value


def _stage_b_save_rank_local_optimizer_state(
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> None:
    """Persist one optimizer shard with bounded host-memory residency."""

    state = _stage_b_move_state_to_cpu(optimizer.state_dict())
    try:
        with path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        del state
        gc.collect()


def _stage_b_sharded_state_dict_context(model: torch.nn.Module) -> Any:
    _ensure_stage_b_npu_sharded_tensor_compatibility()
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )

    # PyTorch 2.5's distributed-checkpoint get_state_dict() path fails when an
    # unflattened FSDP parameter has no local shard on a rank. The underlying
    # FSDP sharded state-dict path handles those zero-local-shard tensors and
    # remains compatible with Distributed Checkpoint storage.
    return FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def save_stage_b_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: CIDTrainer,
    path: str | Path,
    *,
    dataset_sha256: str | None = None,
    epoch_progress: Mapping[str, Any] | None = None,
) -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel

    if not dist.is_initialized():
        raise RuntimeError("Stage B checkpointing requires an initialized process group")
    destination = Path(path)
    if dist.get_rank() == 0:
        destination.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    rank_local_optimizer = _stage_b_use_rank_local_optimizer_checkpoint()
    with _stage_b_sharded_state_dict_context(model):
        model_state = model.state_dict()
        optimizer_state = (
            None
            if rank_local_optimizer
            else FullyShardedDataParallel.optim_state_dict(model, optimizer)
        )
    distributed_state = {"model": model_state}
    if optimizer_state is not None:
        distributed_state["optimizer"] = optimizer_state
    _stage_b_dcp_save(distributed_state, destination / "distributed")

    if rank_local_optimizer:
        # PyTorch 2.1 + torch_npu cannot reliably round-trip the sharded optimizer
        # through DCP. Materialize one rank at a time to stay within shared host memory.
        del distributed_state
        del model_state
        gc.collect()
        for checkpoint_rank in range(dist.get_world_size()):
            dist.barrier()
            if dist.get_rank() == checkpoint_rank:
                _stage_b_save_rank_local_optimizer_state(
                    optimizer,
                    destination / f"optimizer-rank-{dist.get_rank():04d}.pt",
                )
            dist.barrier()

    torch.save(
        trainer.local_progress_state(),
        destination / f"rank-{dist.get_rank():04d}.pt",
    )
    if dist.get_rank() == 0:
        metadata = {
            "format_version": 2 if rank_local_optimizer else 3,
            "kind": "cid-stage-b-fsdp",
            "model_state_layout": "fsdp-sharded-dcp",
            "optimizer_state_layout": (
                "rank-local" if rank_local_optimizer else "fsdp-sharded-dcp"
            ),
            "world_size": dist.get_world_size(),
            "dataset_sha256": dataset_sha256,
            "adapter_config": asdict(trainer.adapter.config),
            "backbone": {
                "model_type": str(trainer.adapter.backbone.config.model_type),
                "hidden_size": trainer.adapter.d_model,
                "vocab_size": trainer.adapter.vocab_size,
                "mask_token_id": trainer.adapter.mask_token_id,
            },
        }
        if epoch_progress is not None:
            metadata["epoch_progress"] = dict(epoch_progress)
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
) -> dict[str, Any]:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel

    if not dist.is_initialized():
        raise RuntimeError("Stage B checkpointing requires an initialized process group")
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    version = int(metadata.get("format_version", 0))
    if version not in (2, 3) or metadata.get("kind") != "cid-stage-b-fsdp":
        raise ValueError("unsupported Stage B checkpoint format")
    if metadata.get("model_state_layout") != "fsdp-sharded-dcp":
        raise ValueError("unsupported Stage B checkpoint state layout")
    saved_world_size = int(metadata["world_size"])
    current_world_size = dist.get_world_size()
    if version == 2 and metadata.get("optimizer_state_layout") != "rank-local":
        raise ValueError("unsupported Stage B checkpoint state layout")
    if version == 3 and metadata.get("optimizer_state_layout") != "fsdp-sharded-dcp":
        raise ValueError("unsupported Stage B checkpoint state layout")
    if version == 2 and saved_world_size != current_world_size:
        raise ValueError("Stage B v2 resume requires the original world size")
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
        or int(backbone.get("mask_token_id", trainer.adapter.mask_token_id))
        != trainer.adapter.mask_token_id
    ):
        raise ValueError("Stage B checkpoint backbone geometry does not match")

    with _stage_b_sharded_state_dict_context(model):
        model_state = model.state_dict()
    distributed_state = {"model": model_state}
    _stage_b_dcp_load(distributed_state, source / "distributed")
    with _stage_b_sharded_state_dict_context(model):
        model.load_state_dict(distributed_state["model"])
    if version == 3:
        optimizer_state = _stage_b_dcp_load_optimizer_state(
            distributed_state["model"],
            source / "distributed",
        )
        flattened_optimizer_state = FullyShardedDataParallel.optim_state_dict_to_load(
            model,
            optimizer,
            optimizer_state,
        )
        optimizer.load_state_dict(flattened_optimizer_state)
    else:
        optimizer.load_state_dict(
            torch.load(
                source / f"optimizer-rank-{dist.get_rank():04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
        )

    if saved_world_size == current_world_size:
        local_state_path = source / f"rank-{dist.get_rank():04d}.pt"
        local_state = torch.load(local_state_path, map_location="cpu", weights_only=False)
        trainer.restore_local_progress_state(local_state)
    else:
        portable_state = torch.load(
            source / "rank-0000.pt",
            map_location="cpu",
            weights_only=False,
        )
        trainer.restore_portable_progress_state(
            portable_state,
            seed=(
                trainer.config.seed
                + dist.get_rank()
                + int(portable_state["trainer_state"]["optimizer_steps"]) * 104729
            ),
        )
    dist.barrier()
    return metadata


def load_stage_b_model_checkpoint(
    model: torch.nn.Module,
    adapter: ILLaDACIDAdapter,
    path: str | Path,
) -> None:
    """Load only Stage B model shards for distributed inference/evaluation."""

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Stage B model loading requires an initialized process group")
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if (
        int(metadata.get("format_version", 0)) not in (2, 3)
        or metadata.get("kind") != "cid-stage-b-fsdp"
    ):
        raise ValueError("unsupported Stage B checkpoint format")
    if metadata.get("model_state_layout") != "fsdp-sharded-dcp":
        raise ValueError("unsupported Stage B checkpoint model state layout")
    if metadata["adapter_config"] != asdict(adapter.config):
        raise ValueError("Stage B checkpoint adapter configuration does not match")
    backbone = metadata["backbone"]
    if (
        int(backbone["hidden_size"]) != adapter.d_model
        or int(backbone["vocab_size"]) != adapter.vocab_size
        or str(backbone["model_type"]) != str(adapter.backbone.config.model_type)
        or int(backbone.get("mask_token_id", adapter.mask_token_id)) != adapter.mask_token_id
    ):
        raise ValueError("Stage B checkpoint backbone geometry does not match")

    with _stage_b_sharded_state_dict_context(model):
        model_state = model.state_dict()
    distributed_state = {"model": model_state}
    _stage_b_dcp_load(distributed_state, source / "distributed")
    with _stage_b_sharded_state_dict_context(model):
        model.load_state_dict(distributed_state["model"])
    dist.barrier()

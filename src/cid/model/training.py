from __future__ import annotations

import gc
import json
import math
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cid.contracts import FreshnessDemand
from cid.data import ThoughtTarget, TrajectoryExample, training_transition_source_steps
from cid.grounding import AnchorKind, LinkRelation, ObjectKind, ObjectRef
from cid.lifecycle import (
    MODELED_LIFECYCLES,
    LifecycleTransitionController,
    LifecycleTransitionSignals,
)
from cid.model.allocation import (
    DEFAULT_MAX_ALLOCATIONS_PER_STEP,
    prefix_allocation_mask,
)
from cid.model.diffusion import (
    CIDDiffusionScheduler,
    denoising_noise_level,
    denoising_reveal_fraction,
)
from cid.model.encoding import (
    ILLaDATextEncoder,
    canonical_fact_text,
    canonical_percept_text,
    canonical_source_text,
    stable_text,
)
from cid.model.illada import ILLaDACIDAdapter
from cid.model.losses import CIDLoss, CIDTargets, cid_loss
from cid.model.materialize import (
    CIDMaterializerConfig,
    RevisionAction,
    decode_need_target_display,
)
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput, build_percept_routing_masks
from cid.state import CellLifecycle, CognitiveRole

CID_NEURAL_CONTRACT_VERSION = 3
STAGE_B_SEMANTIC_SNAPSHOT_FILENAME = "semantic-embedding.pt"


def _trainer_frozen_semantic_snapshot(trainer: Any) -> dict[str, Any] | None:
    tensorizer = getattr(trainer, "tensorizer", None)
    encoder = getattr(tensorizer, "text_encoder", None)
    if not isinstance(encoder, ILLaDATextEncoder) or not encoder.is_frozen_snapshot:
        return None
    return encoder.frozen_snapshot_state()


def _semantic_snapshot_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    weight = state.get("weight")
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError("frozen semantic embedding snapshot weight is invalid")
    return {
        "file": STAGE_B_SEMANTIC_SNAPSHOT_FILENAME,
        "format_version": int(state["format_version"]),
        "encoding_version": int(state["encoding_version"]),
        "pooling_mode": str(state["pooling_mode"]),
        "d_model": int(state["d_model"]),
        "vocab_size": int(weight.shape[0]),
        "dtype": str(weight.dtype),
    }


@dataclass(slots=True)
class CIDTrainingStep:
    example_id: str
    source_step: int
    target_step: int
    diffusion_step: int
    next_diffusion_step: int
    batch: CIDTensorBatch
    targets: CIDTargets
    promoted_fact_texts: tuple[str, ...] = ()
    observed_binding_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class CIDTrainingBatch:
    example_ids: tuple[str, ...]
    source_steps: tuple[int, ...]
    target_steps: tuple[int, ...]
    batch: CIDTensorBatch
    targets: CIDTargets


@dataclass(frozen=True, slots=True)
class CIDRolloutBindingRoute:
    """Runtime-decoded routing carried across one detached rollout transition."""

    need_id: str
    target_cells: tuple[ObjectRef, ...]
    target_display: tuple[ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class CIDRolloutState:
    """Detached model/runtime state consumed by a later rollout transition."""

    thought_semantic: Tensor
    role_features: Tensor
    uncertainty: Tensor
    lifecycle_features: Tensor
    slot_occupancy: Tensor
    local_noise: Tensor
    display_ids: Tensor
    display_noise_level: float = 1.0
    active_binding_ids: tuple[str, ...] = ()
    executable_binding_ids: tuple[str, ...] = ()
    binding_routes: tuple[CIDRolloutBindingRoute, ...] = ()
    promoted_fact_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CIDRolloutWindow:
    example: TrajectoryExample
    source_steps: tuple[int, ...]
    loss_weight: float = 1.0
    is_padding: bool = False

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
    for row, step in enumerate(steps):
        mask = step.batch.display_padding_mask
        if mask is not None:
            width = step.batch.display_ids.shape[1]
            display_padding_mask[row, :width] |= mask[0].to(
                device=display_padding_mask.device, dtype=torch.bool
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
        need_target_cell_targets=_pad_slot_targets(steps, "need_target_cell_targets"),
        need_target_cell_mask=_pad_slot_targets(steps, "need_target_cell_mask"),
        need_target_display_targets=_pad_need_display_targets(
            steps, "need_target_display_targets"
        ),
        need_target_display_mask=_pad_need_display_targets(
            steps, "need_target_display_mask"
        ),
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
    semantic_pooling: str = "order-aware-v2"
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
        if self.semantic_pooling not in {"mean-v1", "order-aware-v2"}:
            raise ValueError("unsupported semantic_pooling mode")


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
    component_mean_losses: dict[str, float] = field(default_factory=dict)
    behavior_counts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CIDTrainProgress:
    transitions: int
    optimizer_steps: int
    mean_loss: float
    raw_mean_loss: float
    rollout_windows_seen_in_epoch: int
    learning_rate: float
    component_mean_losses: dict[str, float] = field(default_factory=dict)


CID_LOSS_COMPONENT_NAMES = (
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
    "auxiliary",
)


def _cid_loss_component_values(losses: CIDLoss) -> dict[str, Tensor]:
    return {
        name: getattr(losses, name).detach().float()
        for name in CID_LOSS_COMPONENT_NAMES
    }


def _accumulate_metric_tensors(
    destination: dict[str, Tensor],
    values: Mapping[str, Tensor],
    *,
    scale: float = 1.0,
) -> None:
    for name, value in values.items():
        contribution = value.detach().float() * scale
        destination[name] = (
            destination[name] + contribution if name in destination else contribution
        )


def _metric_tensor_means(
    values: Mapping[str, Tensor],
    denominator: int,
) -> dict[str, float]:
    if not values:
        return {}
    names = tuple(values)
    stacked = torch.stack(tuple(values[name] for name in names)) / max(1, denominator)
    host_values = stacked.detach().cpu().tolist()
    return dict(zip(names, host_values, strict=True))


def _metric_tensor_values(values: Mapping[str, Tensor]) -> dict[str, float]:
    if not values:
        return {}
    names = tuple(values)
    host_values = torch.stack(tuple(values[name] for name in names)).detach().cpu().tolist()
    return dict(zip(names, host_values, strict=True))


def _cid_behavior_counts(
    output: CIDTensorOutput,
    targets: CIDTargets,
    *,
    need_threshold: float,
    batch_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    zero = output.need_logits.new_zeros((), dtype=torch.float32)
    if batch_mask is None:
        batch_mask = torch.ones(
            output.need_logits.shape[0], dtype=torch.bool, device=output.need_logits.device
        )
    if batch_mask.shape != (output.need_logits.shape[0],):
        raise ValueError("behavior batch_mask must have shape [batch]")
    row = batch_mask.bool()

    need_mask = (
        targets.thought_mask.unsqueeze(-1).expand_as(output.need_logits).bool()
        & row[:, None, None]
    )
    need_prediction = torch.sigmoid(output.need_logits) >= need_threshold
    need_target = targets.need_targets >= 0.5
    counts = {
        "need_tp": (need_prediction & need_target & need_mask).sum().float(),
        "need_fp": (need_prediction & ~need_target & need_mask).sum().float(),
        "need_fn": (~need_prediction & need_target & need_mask).sum().float(),
        "need_tn": (~need_prediction & ~need_target & need_mask).sum().float(),
    }

    source_mask = (targets.source_targets != -100) & row[:, None, None]
    source_correct = zero
    if output.source_logits.shape[-1] > 0:
        source_prediction = output.source_logits.argmax(dim=-1)
        source_correct = ((source_prediction == targets.source_targets) & source_mask).sum().float()
    counts["source_correct"] = source_correct
    counts["source_total"] = source_mask.sum().float()

    convergence_prediction = torch.sigmoid(output.convergence_logits) >= 0.5
    convergence_target = targets.convergence_targets >= 0.5
    counts["convergence_correct"] = (
        (convergence_prediction == convergence_target) & row
    ).sum().float()
    counts["convergence_total"] = row.sum().float()

    lifecycle_mask = (targets.lifecycle != -100) & row[:, None]
    lifecycle_prediction = output.lifecycle_logits.argmax(dim=-1)
    counts["lifecycle_correct"] = (
        (lifecycle_prediction == targets.lifecycle) & lifecycle_mask
    ).sum().float()
    counts["lifecycle_total"] = lifecycle_mask.sum().float()

    display_mask = (targets.display_ids != -100) & row[:, None]
    display_prediction = output.display_logits.argmax(dim=-1)
    counts["display_token_correct"] = (
        (display_prediction == targets.display_ids) & display_mask
    ).sum().float()
    counts["display_token_total"] = display_mask.sum().float()
    return counts


class CIDTrainer:
    CHECKPOINT_VERSION = 3
    SUPPORTED_CHECKPOINT_VERSIONS = (3,)

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
        if tensorizer.text_encoder.pooling_mode != self.config.semantic_pooling:
            raise ValueError("trainer semantic pooling does not match trajectory tensorizer")
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
        self.data_order_version = 3
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
        loss_sum, raw_loss_sum, transitions, _ = self._train_rollout_microbatch_with_metrics(
            windows,
            rollout_probability=rollout_probability,
        )
        return loss_sum, raw_loss_sum, transitions

    def _train_rollout_microbatch_with_metrics(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        rollout_probability: float,
    ) -> tuple[float, float, int, dict[str, Tensor]]:
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
        sample_mask = tuple(not window.is_padding for window in windows)
        effective_batch_size = sum(sample_mask)

        rollout_states: list[CIDRolloutState | None] = [None] * len(windows)
        loss_sum = 0.0
        raw_loss_sum = 0.0
        transition_count = 0
        component_sums: dict[str, Tensor] = {}
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
                tuple(samples),
                loss_scale=loss_weight,
                sample_mask=sample_mask,
                allow_optimizer_step=True,
            )
            raw_loss = float(losses.total.detach().float()) * effective_batch_size
            raw_loss_sum += raw_loss
            loss_sum += raw_loss * loss_weight
            transition_count += effective_batch_size
            _accumulate_metric_tensors(
                component_sums,
                _cid_loss_component_values(losses),
                scale=effective_batch_size,
            )
            # The predicted state is consumed only by a later transition in the same
            # window. Skipping the terminal materialization avoids pure accelerator work,
            # which is especially significant for compact NPU backbones.
            if offset + 1 < next(iter(lengths)) and rollout_probability > 0.0:
                rollout_states = [
                    self._rollout_state_from_prediction(
                        sample,
                        training_batch,
                        output,
                        example=windows[index].example,
                        batch_index=index,
                    )
                    for index, sample in enumerate(samples)
                ]
            del output, training_batch, losses, samples
        return loss_sum, raw_loss_sum, transition_count, component_sums

    def _forward_backward(
        self,
        samples: tuple[CIDTrainingStep, ...],
        *,
        loss_scale: float = 1.0,
        sample_mask: tuple[bool, ...] | None = None,
        allow_optimizer_step: bool = True,
    ) -> tuple[CIDLoss, CIDTensorOutput, CIDTrainingBatch]:
        if not samples:
            raise ValueError("training samples cannot be empty")
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise ValueError("loss_scale must be finite and positive")
        if sample_mask is not None and len(sample_mask) != len(samples):
            raise ValueError("sample_mask length must match training samples")
        training_batch = collate_training_steps(
            samples,
            pad_token_id=int(self.pad_token_id),
        )
        valid_rows = torch.tensor(
            sample_mask if sample_mask is not None else (True,) * len(samples),
            dtype=torch.bool,
            device=training_batch.batch.thought_semantic.device,
        )
        training_batch.batch.sample_mask = valid_rows
        output = self.forward_model(training_batch.batch)
        losses = cid_loss(output, training_batch.targets, batch_mask=valid_rows)
        if not bool(torch.isfinite(losses.total)):
            names = ", ".join(training_batch.example_ids)
            raise FloatingPointError(f"non-finite CID loss for training micro-batch: {names}")
        effective_batch_size = int(valid_rows.sum())
        (losses.total * effective_batch_size * loss_scale).backward()
        self._pending_accumulation += 1
        self._pending_examples += effective_batch_size
        self.state = CIDTrainerState(
            transitions_seen=self.state.transitions_seen + effective_batch_size,
            optimizer_steps=self.state.optimizer_steps,
            epochs_completed=self.state.epochs_completed,
            rollout_windows_seen_in_epoch=self.state.rollout_windows_seen_in_epoch,
        )
        if (
            allow_optimizer_step
            and self._pending_accumulation >= self.config.gradient_accumulation_steps
        ):
            self._optimizer_step()
        return losses, output, training_batch

    def _rollout_state_from_prediction(
        self,
        sample: CIDTrainingStep,
        training_batch: CIDTrainingBatch,
        output: CIDTensorOutput,
        *,
        example: TrajectoryExample,
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
            input_batch.slot_occupancy[batch_index : batch_index + 1, slot_slice].detach().bool()
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
        revision_indices = output.revision_logits[batch_index : batch_index + 1, slot_slice].argmax(
            dim=-1
        )
        input_lifecycle = input_batch.lifecycle_features[batch_index : batch_index + 1, slot_slice]
        lifecycle_features = torch.zeros(
            (1, thought_slots, self.adapter.config.num_lifecycles),
            device=lifecycle_indices.device,
            dtype=sample.batch.role_features.dtype,
        )
        source_snapshot = self.tensorizer._thought_snapshot(example, sample.source_step)
        target_snapshot = self.tensorizer._thought_snapshot(example, sample.target_step)
        target_output_slots = self.tensorizer._target_output_slots(
            source_snapshot, target_snapshot, thought_slots
        )
        cell_id_by_slot = {slot: cell_id for cell_id, slot in target_output_slots.items()}

        retired_index = MODELED_LIFECYCLES.index(CellLifecycle.RETIRED)
        previous_retired = previous_occupancy.squeeze(-1) & (
            input_lifecycle.argmax(dim=-1) == retired_index
        )
        newly_allocated = occupancy.squeeze(-1) & ~previous_occupancy.squeeze(-1)
        predicted_retired = lifecycle_indices == retired_index
        # CIDMaterializer never revives a previously retired cell, and it coerces a
        # newly allocated cell to ACTIVE unless the model explicitly predicts WAITING.
        # Existing live cells, however, stop owning needs immediately when materialized
        # as RETIRED. Decode bindings from that same provisional live set.
        live_slots = (
            occupancy.squeeze(-1)
            & ~previous_retired
            & (~predicted_retired | newly_allocated)
        )[0]

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
        eos_positions = torch.nonzero(
            display_ids[0] == self.tensorizer.eos_token_id, as_tuple=False
        ).flatten()
        display_active_length = (
            int(eos_positions[0]) + 1 if eos_positions.numel() else display_length
        )

        (
            active_binding_ids,
            executable_binding_ids,
            binding_routes,
        ) = self.tensorizer.predicted_binding_state(
            example,
            sample.target_step,
            target_output_slots=target_output_slots,
            live_slots=live_slots,
            display_active_length=display_active_length,
            output=output,
            batch_index=batch_index,
        )
        observed_binding_ids = set(sample.observed_binding_ids)
        waiting_cells: set[str] = set()
        available_cells: set[str] = set()
        for route in binding_routes:
            destination = (
                available_cells if route.need_id in observed_binding_ids else waiting_cells
            )
            destination.update(target.identifier for target in route.target_cells)
        for slot in range(thought_slots):
            if not bool(occupancy[0, slot, 0]):
                continue
            cell_id = cell_id_by_slot.get(slot, f"__rollout_slot_{slot}")
            proposed = MODELED_LIFECYCLES[int(lifecycle_indices[0, slot])]
            if not bool(previous_occupancy[0, slot, 0]):
                resolved = (
                    CellLifecycle.WAITING
                    if proposed is CellLifecycle.WAITING and cell_id in waiting_cells
                    else CellLifecycle.ACTIVE
                )
            else:
                current_features = input_lifecycle[0, slot]
                current = (
                    MODELED_LIFECYCLES[int(current_features.argmax())]
                    if bool(current_features.abs().sum())
                    else CellLifecycle.ACTIVE
                )
                reopen = int(revision_indices[0, slot]) == int(RevisionAction.REOPEN)
                resolved = LifecycleTransitionController.resolve(
                    cell_id=cell_id,
                    current=current,
                    proposed=proposed,
                    signals=LifecycleTransitionSignals(
                        waiting_cells=frozenset(waiting_cells),
                        available_cells=frozenset(available_cells),
                        reopen_cells=frozenset((cell_id,)) if reopen else frozenset(),
                    ),
                )
            lifecycle_features[0, slot, MODELED_LIFECYCLES.index(resolved)] = 1.0

        input_noise = input_batch.local_noise[batch_index : batch_index + 1, slot_slice]
        base_noise = torch.where(
            previous_occupancy,
            input_noise,
            torch.ones_like(input_noise),
        )
        local_noise = (
            base_noise + output.noise_delta[batch_index : batch_index + 1, slot_slice].detach()
        ).clamp(0.0, 1.0)
        local_noise = local_noise * occupancy.to(dtype=local_noise.dtype)

        return CIDRolloutState(
            thought_semantic=output.thought_semantic[
                batch_index : batch_index + 1, slot_slice
            ].detach(),
            role_features=torch.sigmoid(
                output.role_logits[batch_index : batch_index + 1, slot_slice]
            ).detach(),
            uncertainty=output.uncertainty[batch_index : batch_index + 1, slot_slice].detach(),
            lifecycle_features=lifecycle_features.detach(),
            slot_occupancy=occupancy.to(dtype=sample.batch.slot_occupancy.dtype).detach(),
            local_noise=local_noise.detach(),
            display_ids=display_ids.detach(),
            display_noise_level=denoising_noise_level(
                sample.next_diffusion_step, self.config.rollout_denoising_steps
            ),
            active_binding_ids=active_binding_ids,
            executable_binding_ids=executable_binding_ids,
            binding_routes=binding_routes,
            promoted_fact_texts=sample.promoted_fact_texts,
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
        total_component_sums: dict[str, Tensor] = {}
        start_optimizer_steps = self.state.optimizer_steps
        progress_loss = 0.0
        progress_raw_loss = 0.0
        progress_transitions = 0
        progress_component_sums: dict[str, Tensor] = {}
        next_progress_step = None
        if progress_callback is not None and progress_every_optimizer_steps is not None:
            next_progress_step = (
                self.state.optimizer_steps // progress_every_optimizer_steps + 1
            ) * progress_every_optimizer_steps

        def emit_progress_if_due() -> None:
            nonlocal progress_loss, progress_raw_loss, progress_transitions, next_progress_step
            nonlocal progress_component_sums
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
                    component_mean_losses=_metric_tensor_means(
                        progress_component_sums, progress_transitions
                    ),
                )
            )
            progress_loss = 0.0
            progress_raw_loss = 0.0
            progress_transitions = 0
            progress_component_sums = {}
            while next_progress_step <= self.state.optimizer_steps:
                next_progress_step += progress_every_optimizer_steps

        for _ in range(epochs):
            rollout_probability = self.rollout_probability()
            microbatches = self._rollout_microbatches(
                windows, shuffle=shuffle, preserve_order=preserve_order
            )
            for microbatch in microbatches:
                (
                    loss_sum,
                    raw_loss_sum,
                    transitions,
                    component_sums,
                ) = self._train_rollout_microbatch_with_metrics(
                    microbatch,
                    rollout_probability=rollout_probability,
                )
                total_loss += loss_sum
                total_raw_loss += raw_loss_sum
                total_transitions += transitions
                progress_loss += loss_sum
                progress_raw_loss += raw_loss_sum
                progress_transitions += transitions
                _accumulate_metric_tensors(total_component_sums, component_sums)
                _accumulate_metric_tensors(progress_component_sums, component_sums)
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
            component_mean_losses=_metric_tensor_means(
                total_component_sums, total_transitions
            ),
        )

    def evaluate_rollout_windows(
        self,
        windows: tuple[CIDRolloutWindow, ...],
        *,
        seed: int,
        rollout_probability: float = 0.0,
        need_threshold: float = 0.6,
    ) -> CIDTrainReport:
        """Evaluate a deterministic diffusion objective without mutating training state.

        ``rollout_probability=0`` is the comparable teacher-forced objective. Setting it to 1
        feeds every non-initial step from the model's own prior prediction, exposing state drift
        that teacher-forced loss cannot reveal.
        """

        if not windows:
            raise ValueError("validation data contains no rollout windows")
        if self._pending_accumulation:
            raise RuntimeError("flush accumulated gradients before validation")
        if not 0.0 <= rollout_probability <= 1.0:
            raise ValueError("rollout_probability must be in [0, 1]")
        if not 0.0 <= need_threshold <= 1.0:
            raise ValueError("need_threshold must be in [0, 1]")

        generator_state = self.generator.get_state().cpu()
        shuffle_state = self.shuffle_rng.getstate()
        was_training = self.forward_model.training
        total_loss = 0.0
        total_raw_loss = 0.0
        total_transitions = 0
        component_sums: dict[str, Tensor] = {}
        behavior_counts: dict[str, Tensor] = {}
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
                        raise ValueError("validation micro-batch windows must have the same length")
                    loss_weights = {window.loss_weight for window in microbatch}
                    if len(loss_weights) != 1:
                        raise ValueError(
                            "validation micro-batch windows must have the same loss_weight"
                        )
                    loss_weight = next(iter(loss_weights))
                    rollout_states: list[CIDRolloutState | None] = [None] * len(microbatch)
                    for offset in range(next(iter(lengths))):
                        samples_list: list[CIDTrainingStep] = []
                        for index, window in enumerate(microbatch):
                            use_rollout = (
                                offset > 0
                                and rollout_states[index] is not None
                                and self.shuffle_rng.random() < rollout_probability
                            )
                            samples_list.append(
                                self.tensorizer.tensorize(
                                    window.example,
                                    window.source_steps[offset],
                                    timestep=self._sample_timestep(),
                                    generator=self.generator,
                                    rollout_state=(
                                        rollout_states[index] if use_rollout else None
                                    ),
                                )
                            )
                        samples = tuple(samples_list)
                        training_batch = collate_training_steps(
                            samples, pad_token_id=int(self.pad_token_id)
                        )
                        batch_mask = torch.tensor(
                            tuple(not window.is_padding for window in microbatch),
                            dtype=torch.bool,
                            device=training_batch.batch.thought_semantic.device,
                        )
                        training_batch.batch.sample_mask = batch_mask
                        output = self.forward_model(training_batch.batch)
                        losses = cid_loss(
                            output, training_batch.targets, batch_mask=batch_mask
                        )
                        if not bool(torch.isfinite(losses.total)):
                            names = ", ".join(training_batch.example_ids)
                            raise FloatingPointError(
                                f"non-finite CID validation loss for micro-batch: {names}"
                            )
                        batch_size = int(batch_mask.sum())
                        raw_loss = float(losses.total.detach().float()) * batch_size
                        total_raw_loss += raw_loss
                        total_loss += raw_loss * loss_weight
                        total_transitions += batch_size
                        _accumulate_metric_tensors(
                            component_sums,
                            _cid_loss_component_values(losses),
                            scale=batch_size,
                        )
                        _accumulate_metric_tensors(
                            behavior_counts,
                            _cid_behavior_counts(
                                output,
                                training_batch.targets,
                                need_threshold=need_threshold,
                                batch_mask=batch_mask,
                            ),
                        )
                        if offset + 1 < next(iter(lengths)) and rollout_probability > 0.0:
                            rollout_states = [
                                self._rollout_state_from_prediction(
                                    sample,
                                    training_batch,
                                    output,
                                    example=microbatch[index].example,
                                    batch_index=index,
                                )
                                for index, sample in enumerate(samples)
                            ]
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
            component_mean_losses=_metric_tensor_means(
                component_sums, total_transitions
            ),
            behavior_counts=_metric_tensor_values(behavior_counts),
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
            "data_order_version": self.data_order_version,
        }

    def restore_local_progress_state(self, state: Mapping[str, Any]) -> None:
        saved_config = dict(state["trainer_config"])
        saved_config.setdefault("semantic_pooling", "mean-v1")
        if saved_config != asdict(self.config):
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
        self.data_order_version = int(state.get("data_order_version", 1))
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
        saved_config.setdefault("semantic_pooling", "mean-v1")
        current_config = asdict(self.config)
        saved_config["gradient_accumulation_steps"] = current_config["gradient_accumulation_steps"]
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
        self.data_order_version = int(state.get("data_order_version", 1))
        self.optimizer.zero_grad(set_to_none=True)
        self.reseed(seed)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        dataset_sha256: str | None = None,
    ) -> None:
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
            "neural_contract_version": CID_NEURAL_CONTRACT_VERSION,
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
            "data_order_version": self.data_order_version,
            "dataset_sha256": dataset_sha256,
        }
        semantic_snapshot = _trainer_frozen_semantic_snapshot(self)
        if semantic_snapshot is not None:
            payload["semantic_embedding_snapshot"] = semantic_snapshot
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_dataset_sha256: str | None = None,
    ) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") not in self.SUPPORTED_CHECKPOINT_VERSIONS:
            raise ValueError("unsupported CID trainer checkpoint version")
        if checkpoint.get("neural_contract_version") != CID_NEURAL_CONTRACT_VERSION:
            raise ValueError("CID trainer checkpoint neural contract is incompatible")
        if (
            expected_dataset_sha256 is not None
            and checkpoint.get("dataset_sha256") != expected_dataset_sha256
        ):
            raise ValueError("CID trainer checkpoint dataset SHA-256 does not match training data")
        saved_trainer_config = dict(checkpoint["trainer_config"])
        saved_trainer_config.setdefault("semantic_pooling", "mean-v1")
        current_trainer_config = asdict(self.config)
        if saved_trainer_config != current_trainer_config:
            saved_geometry = (
                int(saved_trainer_config.get("micro_batch_size", 1)),
                int(saved_trainer_config.get("gradient_accumulation_steps", 1)),
            )
            current_geometry = (
                int(current_trainer_config.get("micro_batch_size", 1)),
                int(current_trainer_config.get("gradient_accumulation_steps", 1)),
            )
            saved_without_geometry = dict(saved_trainer_config)
            current_without_geometry = dict(current_trainer_config)
            for key in ("micro_batch_size", "gradient_accumulation_steps"):
                saved_without_geometry.pop(key, None)
                current_without_geometry.pop(key, None)
            trainer_state = checkpoint.get("trainer_state", {})
            clean_epoch_boundary = (
                int(trainer_state.get("rollout_windows_seen_in_epoch", 0)) == 0
                and int(checkpoint.get("pending_accumulation", 0)) == 0
                and int(checkpoint.get("pending_examples", 0)) == 0
            )
            equivalent_geometry = (
                saved_geometry[0] * saved_geometry[1]
                == current_geometry[0] * current_geometry[1]
            )
            if not (
                clean_epoch_boundary
                and equivalent_geometry
                and saved_without_geometry == current_without_geometry
            ):
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
        semantic_snapshot = checkpoint.get("semantic_embedding_snapshot")
        if semantic_snapshot is not None:
            current_encoder = self.tensorizer.text_encoder
            restored_encoder = ILLaDATextEncoder.from_frozen_snapshot_state(
                self.adapter,
                self.tensorizer.tokenizer,
                semantic_snapshot,
                device=current_encoder.device,
                embedding_device=current_encoder.embedding_device,
            )
            if restored_encoder.pooling_mode != self.config.semantic_pooling:
                raise ValueError(
                    "checkpoint frozen semantic snapshot pooling does not match "
                    "trainer configuration"
                )
            self.tensorizer.text_encoder = restored_encoder
        self.generator.set_state(checkpoint["generator_state"])
        self.shuffle_rng.setstate(checkpoint["shuffle_state"])
        self.data_order_version = int(checkpoint.get("data_order_version", 1))
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
        if self._pending_accumulation > 0 and self._pending_examples < 0:
            raise ValueError("checkpoint pending accumulation has an invalid example count")

    def _optimizer_step(self) -> None:
        normalizer = self._gradient_example_normalizer()
        self._set_learning_rate_for_step(self.state.optimizer_steps + 1)
        for _, parameter in self._trainable:
            if parameter.grad is not None:
                parameter.grad.div_(normalizer)
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

    def _gradient_example_normalizer(self) -> float:
        """Return one common per-rank divisor for the global valid-example mean gradient."""

        local_examples = float(self._pending_examples)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self.tensorizer.text_encoder.device
            total = torch.tensor(local_examples, device=device, dtype=torch.float32)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            normalizer = float(total.item()) / torch.distributed.get_world_size()
        else:
            normalizer = local_examples
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise RuntimeError("optimizer step requires at least one valid accumulated example")
        return normalizer

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
    *,
    expected_semantic_pooling: str | None = None,
) -> CIDTrainerState:
    """Load the CID model state from a trainer checkpoint without restoring an optimizer."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in CIDTrainer.SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError("unsupported CID trainer checkpoint version")
    if checkpoint.get("neural_contract_version") != CID_NEURAL_CONTRACT_VERSION:
        raise ValueError("CID trainer checkpoint neural contract is incompatible")
    saved_trainer_config = dict(checkpoint.get("trainer_config", {}))
    saved_pooling = str(saved_trainer_config.get("semantic_pooling", "mean-v1"))
    if expected_semantic_pooling is not None and saved_pooling != expected_semantic_pooling:
        raise ValueError(
            "checkpoint semantic pooling does not match requested training contract: "
            f"{saved_pooling!r} != {expected_semantic_pooling!r}"
        )
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


def load_stage_b_semantic_encoder(
    adapter: ILLaDACIDAdapter,
    tokenizer: Any,
    path: str | Path,
    *,
    device: torch.device | str,
    embedding_device: torch.device | str | None = None,
) -> ILLaDATextEncoder:
    """Restore the exact frozen semantic embedding snapshot used by Stage B training."""

    source = Path(path)
    if source.is_dir():
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        snapshot_metadata = metadata.get("semantic_embedding_snapshot")
        if not isinstance(snapshot_metadata, Mapping):
            raise ValueError("Stage B checkpoint does not contain a frozen semantic snapshot")
        filename = str(snapshot_metadata.get("file", ""))
        if not filename:
            raise ValueError("Stage B semantic snapshot metadata is missing its file")
        snapshot_path = source / filename
        state = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint.get("semantic_embedding_snapshot")
        if not isinstance(state, Mapping):
            raise ValueError("Stage B checkpoint does not contain a frozen semantic snapshot")

    encoder = ILLaDATextEncoder.from_frozen_snapshot_state(
        adapter,
        tokenizer,
        state,
        device=device,
        embedding_device=embedding_device,
    )
    if source.is_dir():
        if encoder.pooling_mode != str(snapshot_metadata.get("pooling_mode", "")):
            raise ValueError("Stage B semantic snapshot pooling metadata is inconsistent")
        if encoder.d_model != int(snapshot_metadata.get("d_model", -1)):
            raise ValueError("Stage B semantic snapshot width metadata is inconsistent")
        weight = state.get("weight")
        if not isinstance(weight, Tensor) or int(weight.shape[0]) != int(
            snapshot_metadata.get("vocab_size", -1)
        ):
            raise ValueError("Stage B semantic snapshot vocabulary metadata is inconsistent")
    return encoder


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
            raise ValueError("display_canvas_tokens must be in [2, adapter max_display_tokens]")
        self.display_canvas_tokens = int(display_canvas_tokens)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = adapter.eos_token_id if eos_token_id is None else int(eos_token_id)
        self.scheduler = scheduler or CIDDiffusionScheduler(
            adapter.mask_token_id, adapter.eos_token_id
        )

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
        state_noise = torch.ones((1, capacity, 1), device=device, dtype=dtype)
        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES

        if rollout_state is None:
            for cell in current:
                thought_semantic[0, cell.slot] = current_vectors[cell.cell_id]
                occupancy[0, cell.slot, 0] = 1.0
                uncertainty[0, cell.slot, 0] = cell.uncertainty
                state_noise[0, cell.slot, 0] = cell.noise
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
            state_noise.copy_(rollout_state.local_noise.to(device=device, dtype=dtype))

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
            self.adapter.config.max_thought_slots + prompt_ids.shape[1] + display_canvas_tokens
        )
        if logical_length > self.adapter.max_position_embeddings:
            raise ValueError(
                "configured TCT prefix, prompt, and display bucket exceed backbone context capacity"
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
            display_input_ids = torch.full_like(target_display_ids, self.adapter.mask_token_id)
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

        percept_projections = self._available_percept_projections(
            example,
            target_step,
            allowed_binding_ids=(
                None
                if rollout_state is None
                else frozenset(rollout_state.executable_binding_ids)
            ),
        )
        rollout_routes = (
            {}
            if rollout_state is None
            else {route.need_id: route for route in rollout_state.binding_routes}
        )
        percept_routes = tuple(
            self._percept_route(
                binding,
                rollout_routes=rollout_routes,
                closed_loop=rollout_state is not None,
            )
            for binding, _ in percept_projections
        )
        promoted_fact_texts = self._promoted_fact_texts(
            example,
            target_step,
            carried=() if rollout_state is None else rollout_state.promoted_fact_texts,
            allowed_binding_ids=(
                None
                if rollout_state is None
                else frozenset(rollout_state.executable_binding_ids)
            ),
        )
        fact_memory = self.text_encoder.encode_texts(
            (
                *(
                    canonical_fact_text(key=str(key), value=value, source_type="dataset")
                    for key, value in example.protected_facts.items()
                ),
                *promoted_fact_texts,
            ),
            detach=True,
        )
        percept_memory = self.text_encoder.encode_texts(
            tuple(
                canonical_percept_text(
                    source=event.source,
                    value=event.value,
                    version=event.version,
                    target_cells=route.target_cells,
                    target_display=route.target_display,
                )
                for (binding, event), route in zip(
                    percept_projections, percept_routes, strict=True
                )
            ),
            detach=True,
        )
        percept_thought_mask, percept_display_mask = self._percept_target_masks(
            percept_routes,
            target_output_slots=target_output_slots,
            display_length=display_input_ids.shape[1],
            thought_slots=capacity,
            device=device,
        )
        source_memory = self.text_encoder.encode_texts(
            tuple(canonical_source_text(descriptor) for descriptor in example.source_descriptors),
            detach=True,
        )

        display_padding_mask = torch.zeros_like(display_input_ids, dtype=torch.bool)
        eos_positions = torch.nonzero(
            display_input_ids[0] == self.eos_token_id, as_tuple=False
        ).flatten()
        if eos_positions.numel():
            display_padding_mask[:, int(eos_positions[0]) + 1 :] = True
            display_noise = display_noise.masked_fill(display_padding_mask.unsqueeze(-1), 0.0)

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
            display_padding_mask=display_padding_mask,
        )
        observed_binding_ids = frozenset(
            binding.need_id for binding, _ in percept_projections
        )
        targets = self._targets(
            example=example,
            source_step=source_step,
            target_step=target_step,
            target_by_id=target_by_id,
            target_output_slots=target_output_slots,
            target_vectors=target_vectors,
            display_labels=display_labels,
            display_supervision_mask=display_supervision_mask,
            input_occupancy=occupancy,
            input_lifecycle_features=lifecycle_features,
            input_noise_level=thought_corruption.noise,
            input_state_noise=state_noise,
            observed_binding_ids=(observed_binding_ids if rollout_state is not None else None),
            active_binding_ids=(
                None
                if rollout_state is None
                else frozenset(rollout_state.active_binding_ids)
            ),
            binding_routes=(
                None
                if rollout_state is None
                else {route.need_id: route for route in rollout_state.binding_routes}
            ),
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
            promoted_fact_texts=promoted_fact_texts,
            observed_binding_ids=tuple(observed_binding_ids),
        )

    @staticmethod
    def _available_percept_projections(
        example: TrajectoryExample,
        target_step: int,
        *,
        allowed_binding_ids: frozenset[str] | None = None,
    ) -> tuple[tuple[Any, Any], ...]:
        projections: list[tuple[Any, Any]] = []
        for binding in example.binding_targets:
            if binding.first_need_step > target_step:
                continue
            if allowed_binding_ids is not None and binding.need_id not in allowed_binding_ids:
                continue
            matching = tuple(
                event
                for event in example.events
                if event.arrival_step <= target_step
                and event.source == binding.source
                and dict(event.arguments) == dict(binding.arguments)
            )
            if matching:
                latest = max(matching, key=lambda event: event.arrival_step)
                if (
                    allowed_binding_ids is None
                    and binding.freshness is FreshnessDemand.ONCE
                    and latest.arrival_step < target_step
                ):
                    continue
                projections.append((binding, latest))
        return tuple(projections)

    @staticmethod
    def _promoted_fact_texts(
        example: TrajectoryExample,
        target_step: int,
        *,
        carried: tuple[str, ...],
        allowed_binding_ids: frozenset[str] | None,
    ) -> tuple[str, ...]:
        texts = list(carried)
        seen = set(texts)
        descriptors = {
            str(item.get("name", "")): item for item in example.source_descriptors
        }
        for binding in example.binding_targets:
            if allowed_binding_ids is not None and binding.need_id not in allowed_binding_ids:
                continue
            descriptor = descriptors.get(binding.source)
            if descriptor is None or not bool(descriptor.get("promote_results_to_fact", False)):
                continue
            matching = tuple(
                event
                for event in example.events
                if event.arrival_step <= target_step
                and event.source == binding.source
                and dict(event.arguments) == dict(binding.arguments)
            )
            if not matching:
                continue
            event = max(matching, key=lambda item: item.arrival_step)
            text = canonical_fact_text(
                key=f"binding:{binding.need_id}",
                value=event.value,
                source_type=binding.source,
                version=event.version,
            )
            if text not in seen:
                texts.append(text)
                seen.add(text)
        return tuple(texts)

    def _percept_target_masks(
        self,
        routes: tuple[CIDRolloutBindingRoute, ...],
        *,
        target_output_slots: Mapping[str, int],
        display_length: int,
        thought_slots: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        return build_percept_routing_masks(
            tuple(route.target_cells for route in routes),
            tuple(route.target_display for route in routes),
            cell_slots=target_output_slots,
            thought_slots=thought_slots,
            display_length=display_length,
            device=device,
        )

    @staticmethod
    def _percept_route(
        binding: Any,
        *,
        rollout_routes: Mapping[str, CIDRolloutBindingRoute],
        closed_loop: bool,
    ) -> CIDRolloutBindingRoute:
        if not closed_loop:
            return CIDRolloutBindingRoute(
                need_id=binding.need_id,
                target_cells=binding.target_cells,
                target_display=binding.target_display,
            )
        predicted = rollout_routes.get(binding.need_id)
        if predicted is not None:
            return predicted
        # A legacy/manually constructed rollout state may carry only executable IDs.
        # Never recover teacher multi-region routes in closed loop: the runtime always
        # routes to the live owner at minimum, while an empty display route is its
        # global-display fallback.
        owner = binding.owner_cell
        return CIDRolloutBindingRoute(
            need_id=binding.need_id,
            target_cells=() if owner is None else (owner,),
            target_display=(),
        )

    def predicted_binding_state(
        self,
        example: TrajectoryExample,
        target_step: int,
        *,
        target_output_slots: Mapping[str, int],
        live_slots: Tensor,
        display_active_length: int,
        output: CIDTensorOutput,
        batch_index: int,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[CIDRolloutBindingRoute, ...],
    ]:
        """Match runtime-decodable model needs to teacher-addressable observations."""

        thresholds = CIDMaterializerConfig()
        source_names = tuple(str(item.get("name", "")) for item in example.source_descriptors)
        schedule = self._binding_slot_schedule(example)
        active: list[str] = []
        executable: list[str] = []
        routes: list[CIDRolloutBindingRoute] = []
        if live_slots.ndim != 1 or live_slots.shape[0] != output.need_logits.shape[1]:
            raise ValueError("predicted binding live-slot mask has incompatible geometry")
        if display_active_length < 0:
            raise ValueError("display_active_length must be non-negative")
        for binding in example.binding_targets:
            if binding.first_need_step > target_step:
                continue
            owner = binding.owner_cell
            if owner is None:
                continue
            slot = target_output_slots.get(owner.identifier)
            if (
                slot is None
                or not bool(live_slots[slot])
                or binding.source not in source_names
            ):
                continue
            need_slot = schedule[(owner.identifier, binding.need_id)]
            confidence = float(
                torch.sigmoid(output.need_logits[batch_index, slot, need_slot].detach())
            )
            if confidence < thresholds.need_threshold:
                continue
            source_index = int(
                output.source_logits[batch_index, slot, need_slot].detach().argmax()
            )
            if source_index >= len(source_names) or source_names[source_index] != binding.source:
                continue
            active.append(binding.need_id)

            cell_probabilities = torch.sigmoid(
                output.need_target_cell_logits[batch_index, slot, need_slot].detach()
            )
            target_cells = [ObjectRef.cell(owner.identifier)]
            for cell_id, target_slot in sorted(
                target_output_slots.items(), key=lambda item: item[1]
            ):
                if (
                    target_slot == slot
                    or not bool(live_slots[target_slot])
                    or float(cell_probabilities[target_slot])
                    < thresholds.need_target_cell_threshold
                ):
                    continue
                target_cells.append(ObjectRef.cell(cell_id))
            display_probabilities = torch.sigmoid(
                output.need_target_display_logits[batch_index, slot, need_slot].detach()
            )
            target_display = decode_need_target_display(
                display_probabilities,
                active_length=min(display_active_length, int(display_probabilities.numel())),
                threshold=thresholds.need_target_display_threshold,
            )
            routes.append(
                CIDRolloutBindingRoute(
                    need_id=binding.need_id,
                    target_cells=tuple(target_cells),
                    target_display=target_display,
                )
            )

            descriptor = example.source_descriptors[source_index]
            required = tuple(
                (argument_index, str(argument.get("name", "")))
                for argument_index, argument in enumerate(descriptor.get("arguments", ()))
                if bool(argument.get("required", True))
            )
            arguments_ok = True
            for argument_index, name in required:
                if (
                    argument_index >= output.argument_query.shape[3]
                    or name not in binding.arguments
                ):
                    arguments_ok = False
                    break
                presence = float(
                    torch.sigmoid(
                        output.argument_presence_logits[
                            batch_index, slot, need_slot, argument_index
                        ].detach()
                    )
                )
                if presence < thresholds.argument_presence_threshold:
                    arguments_ok = False
                    break
                query = output.argument_query[
                    batch_index, slot, need_slot, argument_index
                ].detach().float()
                candidates: list[tuple[str, Tensor]] = []
                seen_values: set[str] = set()
                for candidate_binding in example.binding_targets:
                    if (
                        candidate_binding.source != binding.source
                        or name not in candidate_binding.arguments
                    ):
                        continue
                    encoded = stable_text(candidate_binding.arguments[name])
                    if encoded in seen_values:
                        continue
                    seen_values.add(encoded)
                    candidates.append(
                        (
                            encoded,
                            self.text_encoder.encode_one(encoded, detach=True).to(
                                device=query.device, dtype=torch.float32
                            ),
                        )
                    )
                if not candidates:
                    arguments_ok = False
                    break
                candidate_embeddings = torch.stack(
                    [embedding for _, embedding in candidates], dim=0
                )
                similarities = torch.nn.functional.cosine_similarity(
                    query.unsqueeze(0), candidate_embeddings, dim=-1
                )
                best_similarity, best_index = similarities.max(dim=0)
                if (
                    float(best_similarity) < thresholds.retrieval_similarity_threshold
                    or candidates[int(best_index)][0] != stable_text(binding.arguments[name])
                ):
                    arguments_ok = False
                    break
            # Partial-argument sources may start speculative I/O at runtime, but the
            # dataset event represents the binding's fully resolved observation. Do not
            # inject that teacher observation until all required arguments resolve to the
            # same closed-world values the runtime would use.
            if arguments_ok:
                executable.append(binding.need_id)
        return tuple(active), tuple(executable), tuple(routes)

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
        display_supervision_mask: Tensor,
        input_occupancy: Tensor,
        input_lifecycle_features: Tensor,
        input_noise_level: Tensor,
        input_state_noise: Tensor,
        observed_binding_ids: frozenset[str] | None,
        active_binding_ids: frozenset[str] | None,
        binding_routes: Mapping[str, CIDRolloutBindingRoute] | None,
        thought_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> CIDTargets:
        c = self.adapter.config
        n = thought_slots
        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES
        anchor_order = tuple(AnchorKind)
        relation_order = tuple(LinkRelation)
        object_order = tuple(ObjectKind)
        freshness_order = tuple(FreshnessDemand)
        final_step = max(target.step for target in example.thought_targets)
        if observed_binding_ids is None:
            waiting_cells, available_cells = self._binding_lifecycle_cells(example, target_step)
            waiting_equilibrium = any(
                target.lifecycle is CellLifecycle.WAITING for target in target_by_id.values()
            )
        else:
            waiting_cells = set()
            available_cells = set()
            active = active_binding_ids or frozenset()
            routes = binding_routes or {}
            for binding in example.binding_targets:
                if binding.need_id not in active:
                    continue
                destination = (
                    available_cells
                    if binding.need_id in observed_binding_ids
                    else waiting_cells
                )
                route = routes.get(binding.need_id)
                if route is not None:
                    destination.update(target.identifier for target in route.target_cells)
                elif binding.owner_cell is not None:
                    destination.add(binding.owner_cell.identifier)
            # Closed-loop equilibrium must reflect the binding state the model actually
            # materialized. A teacher WAITING label cannot make a failed/missing need look
            # like a valid asynchronous equilibrium.
            waiting_equilibrium = bool(waiting_cells)

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
        source_targets = torch.full((1, n, c.max_need_slots), -100, device=device, dtype=torch.long)
        need_target_cell_targets = torch.zeros(
            (1, n, c.max_need_slots, n), device=device, dtype=dtype
        )
        need_target_cell_mask = torch.zeros(
            (1, n, c.max_need_slots, n), device=device, dtype=torch.bool
        )
        display_width = display_labels.shape[1]
        need_target_display_targets = torch.zeros(
            (1, n, c.max_need_slots, display_width), device=device, dtype=dtype
        )
        need_target_display_mask = torch.zeros(
            (1, n, c.max_need_slots, display_width), device=device, dtype=torch.bool
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
                # A newly allocated cell cannot become STABLE/RETIRED in the same runtime
                # update. Train the lifecycle head on the effective hard-gated state:
                # WAITING when an unresolved binding already targets it, otherwise ACTIVE.
                allocation_targets[0, slot] = 1.0
                noise_delta[0, slot, 0] = target.noise - 1.0
                effective_lifecycle = (
                    CellLifecycle.WAITING
                    if target.lifecycle is CellLifecycle.WAITING and cell_id in waiting_cells
                    else CellLifecycle.ACTIVE
                )
                lifecycle[0, slot] = lifecycle_order.index(effective_lifecycle)
                revision_targets[0, slot] = int(RevisionAction.KEEP)
            else:
                diffusion_delta = target.noise - float(input_noise_level[0, slot, 0])
                noise_delta[0, slot, 0] = diffusion_delta
                state_delta = target.noise - float(input_state_noise[0, slot, 0])
                if state_delta > 1e-6:
                    revision_action = RevisionAction.REOPEN
                elif state_delta < -1e-6:
                    revision_action = RevisionAction.STABILIZE
                else:
                    revision_action = RevisionAction.KEEP
                revision_targets[0, slot] = int(revision_action)

                current_features = input_lifecycle_features[0, slot]
                current_lifecycle = (
                    lifecycle_order[int(current_features.argmax())]
                    if bool(current_features.abs().sum())
                    else CellLifecycle.ACTIVE
                )
                signals = LifecycleTransitionSignals(
                    waiting_cells=frozenset(waiting_cells),
                    available_cells=frozenset(available_cells),
                    reopen_cells=(
                        frozenset((cell_id,))
                        if revision_action is RevisionAction.REOPEN
                        else frozenset()
                    ),
                )
                # Resolve against the state actually presented to the model. During
                # self-rollout this may differ from the teacher snapshot, and using the
                # teacher lifecycle here creates contradictory closed-loop labels.
                effective_lifecycle = LifecycleTransitionController.resolve(
                    cell_id=cell_id,
                    current=current_lifecycle,
                    proposed=target.lifecycle,
                    signals=signals,
                )
                lifecycle[0, slot] = lifecycle_order.index(effective_lifecycle)

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
                and (
                    binding.need_id in observed_binding_ids
                    if observed_binding_ids is not None
                    else _binding_observation_available(example, binding, target_step)
                )
            )
            owner = binding.owner_cell
            if owner is None:
                continue
            slot = target_output_slots.get(owner.identifier)
            if slot is None or not need_is_active:
                continue
            need_slot = binding_slots[(owner.identifier, binding.need_id)]
            need_targets[0, slot, need_slot] = binding.confidence
            source_targets[0, slot, need_slot] = source_index
            refresh_targets[0, slot, need_slot] = freshness_order.index(binding.freshness)
            need_target_cell_mask[0, slot, need_slot] = (
                thought_mask[0] | input_occupancy[0, :, 0].bool()
            )
            need_target_cell_targets[0, slot, need_slot, slot] = 1.0
            for cell_ref in binding.target_cells:
                target_slot = target_output_slots.get(cell_ref.identifier)
                if target_slot is not None:
                    need_target_cell_targets[0, slot, need_slot, target_slot] = 1.0
            need_target_display_mask[0, slot, need_slot] = display_supervision_mask[0]
            for display_ref in binding.target_display:
                if display_ref.span is None:
                    continue
                start, end = display_ref.span
                start = max(0, min(start, display_width))
                end = max(start, min(end, display_width))
                need_target_display_targets[0, slot, need_slot, start:end] = 1.0
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
            need_target_cell_targets=need_target_cell_targets,
            need_target_cell_mask=need_target_cell_mask,
            need_target_display_targets=need_target_display_targets,
            need_target_display_mask=need_target_display_mask,
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

    @staticmethod
    def _binding_lifecycle_cells(
        example: TrajectoryExample, target_step: int
    ) -> tuple[set[str], set[str]]:
        waiting: set[str] = set()
        available: set[str] = set()
        for binding in example.binding_targets:
            if binding.first_need_step > target_step:
                continue
            observation_available = _binding_observation_available(example, binding, target_step)
            # A visible observation resolves the external need but must still mark its
            # target cells as available so the runtime lifecycle gate can release
            # WAITING -> ACTIVE on this transition.  Only unresolved needs contribute
            # waiting targets.
            if observation_available:
                available.update(target.identifier for target in binding.target_cells)
                continue
            waiting.update(target.identifier for target in binding.target_cells)
        return waiting, available

    def _binding_slot_schedule(self, example: TrajectoryExample) -> dict[tuple[str, str], int]:
        by_cell: dict[str, list[Any]] = {}
        for binding in example.binding_targets:
            owner = binding.owner_cell
            if owner is not None:
                by_cell.setdefault(owner.identifier, []).append(binding)

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
        return maximum

    def _canonical_slot_schedule(self, example: TrajectoryExample) -> dict[tuple[int, str], int]:
        capacity = self._trajectory_thought_capacity(example)
        assigned_slots: dict[str, int] = {}
        last_lifecycle: dict[str, CellLifecycle] = {}
        used_slots: set[int] = set()
        schedule: dict[tuple[int, str], int] = {}
        steps = sorted({target.step for target in example.thought_targets})

        for step in steps:
            snapshot = tuple(
                sorted(
                    (target for target in example.thought_targets if target.step == step),
                    key=lambda target: target.cell_id,
                )
            )
            snapshot_ids = {target.cell_id for target in snapshot}
            stale = {
                cell_id
                for cell_id, lifecycle in last_lifecycle.items()
                if cell_id not in snapshot_ids and lifecycle is not CellLifecycle.RETIRED
            }
            if stale:
                names = ", ".join(sorted(stale))
                raise ValueError(f"thought trajectory removed cells without retirement: {names}")

            for target in snapshot:
                if target.cell_id not in assigned_slots:
                    try:
                        slot = next(slot for slot in range(capacity) if slot not in used_slots)
                    except StopIteration as exc:
                        raise ValueError(
                            "thought trajectory exceeds canonical slot capacity"
                        ) from exc
                    assigned_slots[target.cell_id] = slot
                    used_slots.add(slot)
                schedule[(step, target.cell_id)] = assigned_slots[target.cell_id]
                last_lifecycle[target.cell_id] = target.lifecycle

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


def _pad_need_display_targets(
    steps: tuple[CIDTrainingStep, ...],
    name: str,
) -> Tensor:
    tensors = tuple(getattr(step.targets, name) for step in steps)
    if any(tensor.ndim != 4 or tensor.shape[0] != 1 for tensor in tensors):
        raise ValueError("need-display targets must have shape [1, slots, need_slots, display]")
    max_slots = max(tensor.shape[1] for tensor in tensors)
    need_slots = tensors[0].shape[2]
    if any(tensor.shape[2] != need_slots for tensor in tensors):
        raise ValueError("need-display targets in one batch must share need-slot capacity")
    max_display = max(tensor.shape[3] for tensor in tensors)
    output = tensors[0].new_zeros((len(tensors), max_slots, need_slots, max_display))
    for row, tensor in enumerate(tensors):
        output[row, : tensor.shape[1], :, : tensor.shape[3]] = tensor[0]
    return output


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
            if max_horizon == 1:
                windows.extend(
                    CIDRolloutWindow(example=example, source_steps=(step,))
                    for step in contiguous
                )
            else:
                # Rollout states are detached after every transition, so carrying them
                # through a long trajectory does not retain a BPTT graph. Splitting a
                # trajectory here only injected artificial teacher resets every N steps.
                windows.append(CIDRolloutWindow(example=example, source_steps=contiguous))
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
        if window.is_padding:
            continue
        key = _stage_b_rollout_bucket_key(window)
        local_counts[key] = local_counts.get(key, 0) + 1
    for key, local_count in local_counts.items():
        consumed[key] = min(
            totals[key],
            consumed.get(key, 0) + local_count * world_size,
        )
    return {key: value for key, value in consumed.items() if value}


def _rollout_window_length_key(window: CIDRolloutWindow) -> int:
    display_chars = max(
        (len(target.text) for target in window.example.display_targets),
        default=len(window.example.target_display),
    )
    return len(window.example.prompt) + display_chars


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
    length_aware: bool = False,
    zero_gradient_padding: bool = True,
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
            elif not zero_gradient_padding:
                bucket.extend(original[index % len(original)] for index in range(padding))
            else:
                bucket.extend(
                    replace(original[index % len(original)], is_padding=True)
                    for index in range(padding)
                )
        if length_aware and len(bucket) > 1:
            # Keep each rank-local logical microbatch close in sequence geometry.
            # Global super-batches are shuffled as units, so sample order remains
            # stochastic without letting one long trajectory pad many short ones.
            global_microbatch = world_size * micro_batch_size
            bucket.sort(key=_rollout_window_length_key)
            groups = [
                bucket[start : start + global_microbatch]
                for start in range(0, len(bucket), global_microbatch)
            ]
            if shuffle:
                for group_index, group in enumerate(groups):
                    random.Random(
                        seed + epoch * 1_000_033 + bucket_index * 10_007 + group_index
                    ).shuffle(group)
                random.Random(seed + epoch * 1_000_037 + bucket_index * 101).shuffle(groups)
            bucket = [window for group in groups for window in group]
        local_bucket = bucket[rank::world_size]
        if legacy_resume_padding:
            target_local_windows = math.ceil(len(original) / world_size)
            missing_local_windows = target_local_windows - len(local_bucket)
            if missing_local_windows > 0:
                repairs = tuple(
                    original[index % len(original)] for index in range(missing_local_windows)
                )
                for start in range(0, len(repairs), micro_batch_size):
                    resume_repair_microbatches.append(repairs[start : start + micro_batch_size])
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
    seed: int = 0,
    epoch: int = 1,
    shuffle: bool = True,
    length_aware: bool = True,
) -> int:
    """Return optimizer steps for the exact Stage B transition-level accumulation schedule."""

    if world_size <= 0 or micro_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Stage B optimizer-step dimensions must be positive")
    if epoch <= 0:
        raise ValueError("Stage B optimizer-step epoch must be positive")
    if not windows:
        return 0
    local_windows = shard_rollout_windows(
        windows,
        world_size=world_size,
        rank=0,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        micro_batch_size=micro_batch_size,
        length_aware=length_aware,
    )
    pending = 0
    optimizer_steps = 0
    current_key: tuple[int, float] | None = None
    current_size = 0

    def finish_microbatch(key: tuple[int, float] | None) -> None:
        nonlocal pending, optimizer_steps
        if key is None:
            return
        # Every transition inside a rollout window performs one backward call. Optimizer
        # boundaries are independent of window boundaries so long closed-loop trajectories
        # retain state continuity without silently inflating the effective batch size.
        pending += key[0]
        completed, pending = divmod(pending, gradient_accumulation_steps)
        optimizer_steps += completed

    for window in local_windows:
        key = (len(window.source_steps), window.loss_weight)
        if current_size and (key != current_key or current_size >= micro_batch_size):
            finish_microbatch(current_key)
            current_size = 0
        current_key = key
        current_size += 1
    if current_size:
        finish_microbatch(current_key)
    if pending:
        optimizer_steps += 1
    return optimizer_steps


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
                    f"{'backbone' if is_backbone else 'cid'}-{'decay' if use_decay else 'no-decay'}"
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

    semantic_snapshot = None
    if dist.get_rank() == 0:
        semantic_snapshot = _trainer_frozen_semantic_snapshot(trainer)
        if semantic_snapshot is None:
            raise ValueError(
                "Stage B checkpointing requires the independent frozen semantic encoder snapshot"
            )
        snapshot_path = destination / STAGE_B_SEMANTIC_SNAPSHOT_FILENAME
        temporary_snapshot = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
        torch.save(semantic_snapshot, temporary_snapshot)
        temporary_snapshot.replace(snapshot_path)
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
        assert semantic_snapshot is not None
        metadata = {
            "format_version": 6,
            "neural_contract_version": CID_NEURAL_CONTRACT_VERSION,
            "kind": "cid-stage-b-fsdp",
            "model_state_layout": "fsdp-sharded-dcp",
            "optimizer_state_layout": (
                "rank-local" if rank_local_optimizer else "fsdp-sharded-dcp"
            ),
            "world_size": dist.get_world_size(),
            "dataset_sha256": dataset_sha256,
            "semantic_pooling": getattr(trainer.config, "semantic_pooling", "mean-v1"),
            "semantic_embedding_snapshot": _semantic_snapshot_metadata(semantic_snapshot),
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
    if version not in (4, 5, 6) or metadata.get("kind") != "cid-stage-b-fsdp":
        raise ValueError("unsupported Stage B checkpoint format")
    if metadata.get("neural_contract_version") != CID_NEURAL_CONTRACT_VERSION:
        raise ValueError("Stage B checkpoint neural contract is incompatible")
    if metadata.get("model_state_layout") != "fsdp-sharded-dcp":
        raise ValueError("unsupported Stage B checkpoint state layout")
    saved_world_size = int(metadata["world_size"])
    current_world_size = dist.get_world_size()
    optimizer_layout = metadata.get("optimizer_state_layout")
    if version == 4 and optimizer_layout != "rank-local":
        raise ValueError("unsupported Stage B checkpoint state layout")
    if version == 5 and optimizer_layout != "fsdp-sharded-dcp":
        raise ValueError("unsupported Stage B checkpoint state layout")
    if version == 6 and optimizer_layout not in {"rank-local", "fsdp-sharded-dcp"}:
        raise ValueError("unsupported Stage B checkpoint state layout")
    if optimizer_layout == "rank-local" and saved_world_size != current_world_size:
        raise ValueError("Stage B rank-local resume requires the original world size")
    if version == 6 and not isinstance(metadata.get("semantic_embedding_snapshot"), Mapping):
        raise ValueError("Stage B checkpoint is missing its frozen semantic embedding snapshot")
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
    if optimizer_layout == "fsdp-sharded-dcp":
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

    if version == 6:
        current_encoder = trainer.tensorizer.text_encoder
        restored_encoder = load_stage_b_semantic_encoder(
            trainer.adapter,
            trainer.tensorizer.tokenizer,
            source,
            device=current_encoder.device,
            embedding_device=current_encoder.embedding_device,
        )
        if restored_encoder.pooling_mode != trainer.config.semantic_pooling:
            raise ValueError(
                "Stage B frozen semantic snapshot pooling does not match trainer configuration"
            )
        trainer.tensorizer.text_encoder = restored_encoder

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
    *,
    tokenizer: Any | None = None,
    semantic_device: torch.device | str | None = None,
    semantic_embedding_device: torch.device | str | None = None,
) -> ILLaDATextEncoder | None:
    """Load Stage B model shards and, for new checkpoints, their exact semantic snapshot."""

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Stage B model loading requires an initialized process group")
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    version = int(metadata.get("format_version", 0))
    if (
        version not in (4, 5, 6)
        or metadata.get("kind") != "cid-stage-b-fsdp"
    ):
        raise ValueError("unsupported Stage B checkpoint format")
    if metadata.get("neural_contract_version") != CID_NEURAL_CONTRACT_VERSION:
        raise ValueError("Stage B checkpoint neural contract is incompatible")
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

    semantic_encoder = None
    if version == 6:
        if tokenizer is None or semantic_device is None:
            raise ValueError(
                "Stage B format-6 inference requires tokenizer and semantic_device so the "
                "saved frozen semantic snapshot cannot be silently replaced by live embeddings"
            )
        semantic_encoder = load_stage_b_semantic_encoder(
            adapter,
            tokenizer,
            source,
            device=semantic_device,
            embedding_device=semantic_embedding_device,
        )
        if semantic_encoder.pooling_mode != str(metadata.get("semantic_pooling", "")):
            raise ValueError("Stage B semantic snapshot pooling does not match checkpoint metadata")

    with _stage_b_sharded_state_dict_context(model):
        model_state = model.state_dict()
    distributed_state = {"model": model_state}
    _stage_b_dcp_load(distributed_state, source / "distributed")
    with _stage_b_sharded_state_dict_context(model):
        model.load_state_dict(distributed_state["model"])
    dist.barrier()
    return semantic_encoder

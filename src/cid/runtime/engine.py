from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Any

from cid.contracts import (
    CIDPolicy,
    FreshnessDemand,
    ModelContext,
    ModelUpdate,
    Observation,
    Percept,
)
from cid.defaults import DEFAULT_BINDING_THRESHOLD
from cid.grounding import STRONG_LINK_RELATIONS, ClosedWorldGrounder, ObjectKind, ObjectRef
from cid.lifecycle import LifecycleTransitionController, LifecycleTransitionSignals
from cid.reclamation import (
    DEFAULT_RECLAMATION_GRACE_STEPS,
    DEFAULT_RECLAMATION_LOW_WATERMARK,
    DEFAULT_RECLAMATION_TARGET_WATERMARK,
    retired_reclamation_candidates,
)
from cid.runtime.archive import CognitiveArchive, CognitiveTombstone
from cid.runtime.bindings import Binding, BindingStatus, BindingTable, canonical_work_key
from cid.runtime.sources import (
    ReadOnlySource,
    RuntimeSnapshotSource,
    SourceRegistry,
    StreamingSource,
    VersionAwareSource,
)
from cid.runtime.trace import RuntimeTrace
from cid.state import CellLifecycle, CognitiveField, DisplayCanvas, FactItem, FactStore

_LOOP_MAX_PERIOD = 4
_LOOP_MIN_REPEATS = 4
_BEHAVIOR_LOOP_MIN_SPAN = 12
_BEHAVIOR_LOOP_MIN_REPEATS = 3
_LOOP_REDIFFUSION_NOISE = 0.5
_LOOP_TABOO_STEPS = 8


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_steps: int = 256
    max_total_steps: int | None = None
    max_wall_time_s: float | None = 300.0
    binding_threshold: float = DEFAULT_BINDING_THRESHOLD
    idle_yield_s: float = 0.001
    trace_display: bool = False
    trace_details: bool = False
    reclamation_grace_steps: int = DEFAULT_RECLAMATION_GRACE_STEPS
    reclamation_low_watermark: float = DEFAULT_RECLAMATION_LOW_WATERMARK
    reclamation_target_watermark: float = DEFAULT_RECLAMATION_TARGET_WATERMARK
    loop_escape_attempts: int = 2

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_total_steps is not None and self.max_total_steps <= 0:
            raise ValueError("max_total_steps must be positive when set")
        if self.max_wall_time_s is not None and self.max_wall_time_s <= 0:
            raise ValueError("max_wall_time_s must be positive when set")
        if not 0.0 <= self.binding_threshold <= 1.0:
            raise ValueError("binding_threshold must be in [0, 1]")
        if self.idle_yield_s < 0:
            raise ValueError("idle_yield_s must be non-negative")
        if self.reclamation_grace_steps < 0:
            raise ValueError("reclamation_grace_steps must be non-negative")
        if not 0.0 <= self.reclamation_low_watermark <= 1.0:
            raise ValueError("reclamation_low_watermark must be in [0, 1]")
        if not 0.0 <= self.reclamation_target_watermark <= 1.0:
            raise ValueError("reclamation_target_watermark must be in [0, 1]")
        if self.reclamation_target_watermark < self.reclamation_low_watermark:
            raise ValueError("reclamation_target_watermark cannot be below low watermark")
        if self.loop_escape_attempts < 0:
            raise ValueError("loop_escape_attempts must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    thought: CognitiveField
    display: DisplayCanvas
    facts: tuple[FactItem, ...]
    bindings: tuple[Binding, ...]
    archive: tuple[CognitiveTombstone, ...]
    trace: RuntimeTrace
    steps: int
    converged: bool


@dataclass(slots=True)
class _ExternalJob:
    task: asyncio.Task[Observation]
    owner_binding_id: str
    terminal_validation: bool = False


@dataclass(slots=True)
class _VersionJob:
    task: asyncio.Task[str | None]
    terminal_validation: bool = False


@dataclass(slots=True)
class _StreamJob:
    task: asyncio.Task[None]
    queue: asyncio.Queue[Observation]


@dataclass(frozen=True, slots=True)
class _LoopSnapshot:
    signature: tuple[Any, ...]
    behavior_signature: tuple[Any, ...]
    input_thought: CognitiveField
    input_display: DisplayCanvas
    proposed_thought: CognitiveField
    proposed_display: DisplayCanvas


class CIDRuntime:
    def __init__(
        self,
        sources: SourceRegistry,
        config: RuntimeConfig | None = None,
        grounder: ClosedWorldGrounder | None = None,
    ) -> None:
        self.sources = sources
        self.config = config or RuntimeConfig()
        self.grounder = grounder
        self.facts = FactStore()
        self.bindings = BindingTable()
        self.trace = RuntimeTrace()
        self.lifecycle = LifecycleTransitionController()
        self.archive = CognitiveArchive()
        self._jobs: dict[str, _ExternalJob] = {}
        self._version_jobs: dict[str, _VersionJob] = {}
        self._streams: dict[str, _StreamJob] = {}
        self._completed_streams: set[str] = set()
        self._detached_tasks: set[asyncio.Task[Any]] = set()
        self._external_progress = asyncio.Event()
        self._terminal_validated: dict[str, tuple[str, str | None, float | None]] = {}
        self._runtime_step = 0
        self._cache: dict[str, Observation] = {}
        self._created_at: dict[str, int] = {}
        self._retired_at: dict[str, int] = {}

    async def run(
        self,
        policy: CIDPolicy,
        *,
        thought: CognitiveField,
        display: DisplayCanvas,
        facts: Iterable[FactItem] = (),
        prompt: str = "",
    ) -> RuntimeResult:
        if self._jobs or self._version_jobs or self._streams:
            raise RuntimeError("a CIDRuntime instance cannot run concurrent trajectories")
        self.facts = FactStore()
        self.bindings = BindingTable()
        self.trace = RuntimeTrace()
        self.archive = CognitiveArchive()
        self._cache = {}
        self._completed_streams = set()
        self._detached_tasks = set()
        self._external_progress = asyncio.Event()
        self._terminal_validated = {}
        self._runtime_step = 0
        self._created_at = {cell_id: 0 for cell_id in thought.occupied_cell_ids}
        self._retired_at = {
            cell.cell_id: 0
            for cell in thought.cells
            if cell.cell_id is not None and cell.lifecycle is CellLifecycle.RETIRED
        }
        for item in facts:
            self.facts.publish(item)

        converged = False
        completed_steps = 0
        epoch_steps = 0
        started_at = time.monotonic()
        self.trace.emit("trajectory_started", 0, runtime_step=self._runtime_step)
        deadline = (
            None
            if self.config.max_wall_time_s is None
            else started_at + self.config.max_wall_time_s
        )
        descriptors = self.sources.descriptors()
        descriptor_by_name = {descriptor.name: descriptor for descriptor in descriptors}
        loop_history: list[_LoopSnapshot] = []
        taboo_signatures: dict[tuple[Any, ...], int] = {}
        loop_escape_count = 0
        last_escape_slots: tuple[int, ...] = ()
        last_escape_display_positions: tuple[int, ...] = ()
        loop_history_limit = max(
            _LOOP_MAX_PERIOD * _LOOP_MIN_REPEATS,
            _BEHAVIOR_LOOP_MIN_SPAN + _LOOP_MAX_PERIOD,
        ) + 1

        try:
            while True:
                if self._deadline_expired(deadline):
                    self.trace.emit("wall_clock_budget_exhausted", completed_steps)
                    break

                if (
                    self.config.max_total_steps is not None
                    and completed_steps >= self.config.max_total_steps
                ):
                    self.trace.emit(
                        "total_compute_budget_exhausted",
                        completed_steps,
                        runtime_step=self._runtime_step,
                    )
                    break

                if epoch_steps >= self.config.max_steps:
                    if self._has_pending_required_external_work():
                        self._launch_due_jobs(completed_steps)
                        self.trace.emit(
                            "quiescence_started",
                            completed_steps,
                            reason="compute_budget_waiting_for_evidence",
                            runtime_step=self._runtime_step,
                        )
                        advanced = await self._wait_for_external_progress(deadline)
                        if not advanced:
                            self.trace.emit("wall_clock_budget_exhausted", completed_steps)
                            break
                        epoch_steps = 0
                        self.trace.emit(
                            "quiescence_resumed",
                            completed_steps,
                            reason="external_progress",
                            runtime_step=self._runtime_step,
                        )
                        continue
                    self.trace.emit(
                        "compute_budget_exhausted",
                        completed_steps,
                        runtime_step=self._runtime_step,
                    )
                    break

                step = completed_steps
                if self.sources.advance_runtime_step(self._runtime_step):
                    await asyncio.sleep(0)
                thought = self._maybe_reclaim(thought, step)
                previous_thought = thought
                previous_display = display
                observations_before = self._observation_count()
                self._drain_completed_version_jobs(step)
                self._drain_completed_jobs(step)
                self._drain_stream_updates(step)
                if self._observation_count() > observations_before:
                    epoch_steps = 0
                    loop_history.clear()
                    taboo_signatures.clear()
                    loop_escape_count = 0
                    last_escape_slots = ()
                    last_escape_display_positions = ()
                percepts = self._project_available(step, thought)
                context = ModelContext(
                    facts=self.facts.snapshot(),
                    thought=thought,
                    display=display,
                    sources=descriptors,
                    percepts=percepts,
                    step=step,
                    prompt=prompt,
                    diffusion_step=epoch_steps,
                    trace_details=self.config.trace_details,
                )

                self.trace.emit(
                    "model_step_started",
                    step,
                    percepts=len(percepts),
                    runtime_step=self._runtime_step,
                )
                update = await asyncio.to_thread(policy.step, context)
                finished_payload: dict[str, Any] = {
                    "needs": len(update.needs),
                    "equilibrium": update.equilibrium,
                    "converged": update.converged,
                    "runtime_step": self._runtime_step,
                }
                if self.config.trace_details:
                    finished_payload["diagnostics"] = dict(update.diagnostics)
                    finished_payload["binding_threshold"] = self.config.binding_threshold
                    finished_payload["display_previous_token_ids"] = list(display.token_ids)
                    finished_payload["display_changed_positions"] = [
                        i for i, (old, new) in enumerate(
                            zip_longest(display.token_ids, update.display.token_ids)
                        ) if old != new
                    ]
                if self.config.trace_display or self.config.trace_details:
                    finished_payload.update(
                        display_token_ids=list(update.display.token_ids),
                        display_visible_token_ids=list(update.display.visible_token_ids),
                        display_unresolved=update.display.unresolved,
                        display_realized_length=update.display.realized_length,
                        display_active_span_length=update.display.active_span_length,
                    )
                self.trace.emit("model_step_finished", step, **finished_payload)
                completed_steps += 1
                epoch_steps += 1
                proposed_thought = update.thought
                proposed_display = update.display

                proposal_signature = self._proposal_signature(update)
                behavior_signature = self._behavior_signature(update)
                expired_taboo = [
                    signature
                    for signature, expires_at in taboo_signatures.items()
                    if completed_steps > expires_at
                ]
                for signature in expired_taboo:
                    taboo_signatures.pop(signature, None)

                snapshot = _LoopSnapshot(
                    signature=proposal_signature,
                    behavior_signature=behavior_signature,
                    input_thought=previous_thought,
                    input_display=previous_display,
                    proposed_thought=proposed_thought,
                    proposed_display=proposed_display,
                )
                loop_history.append(snapshot)
                if len(loop_history) > loop_history_limit:
                    del loop_history[: len(loop_history) - loop_history_limit]

                external_pending = self._has_pending_required_external_work()
                if proposal_signature in taboo_signatures and not external_pending:
                    self.trace.emit(
                        "loop_taboo_transition_blocked",
                        step,
                        runtime_step=self._runtime_step,
                    )
                    if loop_escape_count >= self.config.loop_escape_attempts:
                        self.trace.emit(
                            "loop_escape_exhausted",
                            step,
                            reason="taboo_transition_repeated",
                            attempts=loop_escape_count,
                            runtime_step=self._runtime_step,
                        )
                        break
                    loop_escape_count += 1
                    thought, display = self._rediffuse_loop_state(
                        previous_thought,
                        previous_display,
                        cell_slots=last_escape_slots,
                        display_positions=last_escape_display_positions,
                        current_thought=previous_thought,
                        current_display=previous_display,
                    )
                    self.trace.emit(
                        "loop_escape_applied",
                        step,
                        reason="taboo_transition_repeated",
                        attempt=loop_escape_count,
                        cell_slots=list(last_escape_slots),
                        display_positions=list(last_escape_display_positions),
                        runtime_step=self._runtime_step,
                    )
                    loop_history.clear()
                    epoch_steps = 0
                    self._runtime_step += 1
                    await asyncio.sleep(0)
                    continue

                loop_match = None if external_pending else self._loop_period(loop_history)
                loop_mode = "exact"
                if loop_match is None and not external_pending:
                    loop_match = self._loop_period(loop_history, behavior=True)
                    loop_mode = "behavior"
                if loop_match is not None:
                    loop_period, loop_repeats = loop_match
                    repeated_count = loop_period * loop_repeats
                    cycle = tuple(loop_history[-repeated_count:])
                    if loop_mode == "exact":
                        cycle_signatures = {item.signature for item in cycle}
                        expires_at = completed_steps + _LOOP_TABOO_STEPS
                        for signature in cycle_signatures:
                            taboo_signatures[signature] = expires_at
                    last_escape_slots, last_escape_display_positions = self._loop_targets(cycle)
                    self.trace.emit(
                        "loop_detected",
                        step,
                        period=loop_period,
                        repeats=loop_repeats,
                        mode=loop_mode,
                        cell_slots=list(last_escape_slots),
                        display_positions=list(last_escape_display_positions),
                        runtime_step=self._runtime_step,
                    )
                    if (
                        loop_period == 1
                        and self._display_is_resolved_and_stable(
                            previous_display,
                            proposed_display,
                        )
                    ):
                        display = proposed_display
                        self.trace.emit(
                            "stable_fixed_point_reached",
                            step,
                            mode=loop_mode,
                            repeats=loop_repeats,
                            runtime_step=self._runtime_step,
                        )
                        break
                    if loop_escape_count >= self.config.loop_escape_attempts:
                        self.trace.emit(
                            "loop_escape_exhausted",
                            step,
                            reason=f"{loop_mode}_cycle_detected",
                            attempts=loop_escape_count,
                            runtime_step=self._runtime_step,
                        )
                        break
                    loop_escape_count += 1
                    rollback = cycle[0]
                    thought, display = self._rediffuse_loop_state(
                        rollback.input_thought,
                        rollback.input_display,
                        cell_slots=last_escape_slots,
                        display_positions=last_escape_display_positions,
                        current_thought=previous_thought,
                        current_display=previous_display,
                    )
                    self.trace.emit(
                        "loop_escape_applied",
                        step,
                        reason=f"{loop_mode}_cycle_detected",
                        attempt=loop_escape_count,
                        period=loop_period,
                        rolled_back=rollback.input_thought.occupied_cell_ids
                        == previous_thought.occupied_cell_ids,
                        cell_slots=list(last_escape_slots),
                        display_positions=list(last_escape_display_positions),
                        runtime_step=self._runtime_step,
                    )
                    loop_history.clear()
                    epoch_steps = 0
                    self._runtime_step += 1
                    await asyncio.sleep(0)
                    continue

                display = proposed_display
                live_cell_ids = set(proposed_thought.live_cell_ids)
                unknown_reopens = set(update.reopen_cell_ids) - set(
                    previous_thought.occupied_cell_ids
                )
                if unknown_reopens:
                    unknown = ", ".join(sorted(unknown_reopens))
                    raise ValueError(f"model requested reopen for unknown cells: {unknown}")

                for need in update.needs:
                    missing_targets = set(need.target_cell_ids) - live_cell_ids
                    if missing_targets:
                        missing = ", ".join(sorted(missing_targets))
                        raise ValueError(
                            f"information need {need.need_id!r} targets non-live cells: {missing}"
                        )
                    source = need.selected_source()
                    descriptor = descriptor_by_name.get(source or "")
                    executable = descriptor is not None and all(
                        name in need.arguments for name in descriptor.required_arguments
                    )
                    self.trace.emit(
                        "information_need",
                        step,
                        need_id=need.need_id,
                        confidence=need.confidence,
                        source=source,
                        executable=executable,
                        **({
                            "arguments": deepcopy(dict(need.arguments)),
                            "source_scores": dict(need.source_scores),
                            "target_cells": [ref.identifier for ref in need.target_cells],
                            "target_display": [ref.identifier for ref in need.target_display],
                            "binding_threshold": self.config.binding_threshold,
                        } if self.config.trace_details else {}),
                    )

                touched = self.bindings.reconcile(
                    update.needs,
                    binding_threshold=self.config.binding_threshold,
                    source_descriptors=descriptor_by_name,
                )
                for binding in touched:
                    self.trace.emit(
                        "binding_active",
                        step,
                        binding_id=binding.binding_id,
                        need_id=binding.need_id,
                        source=binding.source,
                        arguments_complete=binding.arguments_complete,
                        **({
                            "arguments": deepcopy(dict(binding.arguments)),
                            "work_key": binding.work_key,
                            "target_cells": [ref.identifier for ref in binding.target_cells],
                            "target_display": [ref.identifier for ref in binding.target_display],
                        } if self.config.trace_details else {}),
                    )

                self._cancel_orphan_external_work(step)
                observations_before = self._observation_count()
                self._drain_completed_version_jobs(step)
                self._drain_completed_jobs(step)
                self._drain_stream_updates(step)
                if self._observation_count() > observations_before:
                    epoch_steps = 0
                thought = self.lifecycle.apply(
                    previous_thought,
                    proposed_thought,
                    LifecycleTransitionSignals(
                        waiting_cells=self.bindings.waiting_target_cells(),
                        available_cells=self.bindings.available_target_cells(),
                        reopen_cells=frozenset(update.reopen_cell_ids),
                    ),
                )
                final_live_cell_ids = set(thought.live_cell_ids)
                for need in update.needs:
                    missing_targets = set(need.target_cell_ids) - final_live_cell_ids
                    if missing_targets:
                        missing = ", ".join(sorted(missing_targets))
                        raise ValueError(
                            f"information need {need.need_id!r} targets cells blocked by "
                            f"lifecycle state: {missing}"
                        )
                self._trace_lifecycle_changes(previous_thought, thought, step)
                if self.config.trace_details:
                    self.trace.emit(
                        "runtime_state", step,
                        cells=[
                            {
                                "slot": slot, "cell_id": cell.cell_id,
                                "lifecycle": cell.lifecycle.value,
                                "uncertainty": cell.uncertainty, "noise": cell.noise,
                                "roles": {role.value: score for role, score in cell.roles.items()},
                            }
                            for slot, cell in enumerate(thought.cells) if cell.occupied
                        ],
                        pending_jobs=len(self._jobs),
                        unresolved_bindings=self._has_unresolved_active_binding(),
                        fact_count=len(self.facts.snapshot().items),
                    )
                self._record_lifecycle_steps(previous_thought, thought, step)
                unresolved = self._has_unresolved_active_binding()
                settled = update.equilibrium or update.converged
                if update.converged and not unresolved:
                    if self._terminal_freshness_satisfied(step):
                        converged = True
                        self.trace.emit(
                            "trajectory_finalized",
                            step,
                            runtime_step=self._runtime_step,
                        )
                        break
                    self.trace.emit(
                        "quiescence_started",
                        step,
                        reason="final_freshness_barrier",
                        runtime_step=self._runtime_step,
                    )
                    advanced = await self._wait_for_external_progress(deadline)
                    if not advanced:
                        self.trace.emit("wall_clock_budget_exhausted", completed_steps)
                        break
                    epoch_steps = 0
                    self.trace.emit(
                        "quiescence_resumed",
                        step,
                        reason="final_freshness_checked",
                        runtime_step=self._runtime_step,
                    )
                    continue

                self._launch_due_jobs(step)
                # Launching due work can resolve an active binding synchronously
                # from the runtime cache. Re-check after the launch so a stale
                # pre-launch unresolved flag cannot send an already-satisfied
                # equilibrium into an external-progress wait with no work left.
                unresolved = self._has_unresolved_active_binding()
                if settled and unresolved:
                    self.trace.emit(
                        "quiescence_started",
                        step,
                        reason="current_information_equilibrium",
                        runtime_step=self._runtime_step,
                    )
                    advanced = await self._wait_for_external_progress(deadline)
                    if not advanced:
                        self.trace.emit("wall_clock_budget_exhausted", completed_steps)
                        break
                    epoch_steps = 0
                    self.trace.emit(
                        "quiescence_resumed",
                        step,
                        reason="external_progress",
                        runtime_step=self._runtime_step,
                    )
                    continue

                if (self._jobs or self._version_jobs or self._streams) and self.config.idle_yield_s:
                    await asyncio.sleep(self.config.idle_yield_s)
                else:
                    await asyncio.sleep(0)
                self._runtime_step += 1
        finally:
            tasks = [
                *(job.task for job in self._jobs.values()),
                *(job.task for job in self._version_jobs.values()),
                *(job.task for job in self._streams.values()),
                *self._detached_tasks,
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._jobs.clear()
            self._version_jobs.clear()
            self._streams.clear()
            self._detached_tasks.clear()

        stop_events = {
            "wall_clock_budget_exhausted", "total_compute_budget_exhausted",
            "compute_budget_exhausted", "loop_escape_exhausted",
            "stable_fixed_point_reached",
        }
        stop_reason = "converged" if converged else next(
            (event.kind for event in reversed(self.trace.events) if event.kind in stop_events),
            "stopped",
        )
        self.trace.emit(
            "trajectory_finished",
            completed_steps,
            converged=converged,
            runtime_step=self._runtime_step,
            stop_reason=stop_reason,
        )
        thought = self._maybe_reclaim(thought, completed_steps, force=True)

        snapshot = self.facts.snapshot()
        return RuntimeResult(
            thought=thought,
            display=display,
            facts=tuple(snapshot.items.values()),
            bindings=self.bindings.all(),
            archive=self.archive.all(),
            trace=self.trace,
            steps=completed_steps,
            converged=converged,
        )

    def _project_available(self, step: int, thought: CognitiveField) -> tuple[Percept, ...]:
        percepts: list[Percept] = []
        for binding in self.bindings.active():
            if binding.observation is None:
                continue
            binding.cognitive_projections += 1
            target_cells = list(binding.target_cells)
            if self.grounder is not None:
                seen = set(target_cells)
                for anchor in binding.observation.anchors:
                    for cell_id in self.grounder.route(anchor, thought):
                        target = ObjectRef.cell(cell_id)
                        if target not in seen:
                            target_cells.append(target)
                            seen.add(target)
            percepts.append(
                Percept(
                    binding_id=binding.binding_id,
                    source=binding.source,
                    observation=deepcopy(binding.observation),
                    target_cells=tuple(target_cells),
                    target_display=binding.target_display,
                    projection_index=binding.cognitive_projections,
                )
            )
            self.trace.emit(
                "cognitive_projection",
                step,
                binding_id=binding.binding_id,
                index=binding.cognitive_projections,
                grounded_targets=len(target_cells) - len(binding.target_cells),
                **({
                    "source": binding.source,
                    "target_cells": [ref.identifier for ref in target_cells],
                    "target_display": [ref.identifier for ref in binding.target_display],
                } if self.config.trace_details else {}),
            )
        return tuple(percepts)

    def _launch_due_jobs(self, step: int) -> None:
        now = time.monotonic()
        for binding in self.bindings.active():
            work_key = binding.work_key
            source = self.sources.get(binding.source)
            if not binding.arguments_complete and not source.descriptor.accepts_partial_arguments:
                continue

            if (
                binding.observation is None
                and source.descriptor.cacheable
                and work_key in self._cache
            ):
                binding.observation = deepcopy(self._cache[work_key])
                binding.last_refresh_at = now
                binding.status = BindingStatus.AVAILABLE
                self.trace.emit("cache_hit", step, binding_id=binding.binding_id)

            if not self._refresh_due(binding, now):
                continue

            if (
                source.descriptor.streamable
                and binding.freshness is not FreshnessDemand.ONCE
                and isinstance(source, StreamingSource)
            ):
                self._ensure_stream(binding, source, step)
                continue

            if (
                binding.observation is not None
                and source.descriptor.versioned
                and isinstance(source, VersionAwareSource)
            ):
                self._ensure_version_probe(binding, source, step)
                continue

            self._ensure_read(binding, source, step)

    def _ensure_read(
        self,
        binding: Binding,
        source: ReadOnlySource,
        step: int,
        *,
        terminal_validation: bool = False,
    ) -> None:
        work_key = binding.work_key
        if work_key in self._jobs:
            job = self._jobs[work_key]
            if terminal_validation:
                job.terminal_validation = True
            binding.status = (
                BindingStatus.REFRESHING
                if binding.observation is not None
                else BindingStatus.WAITING
            )
            if binding.binding_id != job.owner_binding_id:
                self.trace.emit("job_deduplicated", step, binding_id=binding.binding_id)
            return

        task = asyncio.create_task(source.read(binding.arguments))
        job = _ExternalJob(
            task=task,
            owner_binding_id=binding.binding_id,
            terminal_validation=terminal_validation,
        )
        self._jobs[work_key] = job

        def mark_ready(completed: asyncio.Task[Observation]) -> None:
            if not completed.cancelled():
                self.trace.emit(
                    "external_refresh_ready",
                    step,
                    binding_id=job.owner_binding_id,
                    work_key=work_key,
                    success=completed.exception() is None,
                    runtime_step=self._runtime_step,
                )
            self._external_progress.set()

        task.add_done_callback(mark_ready)
        binding.status = (
            BindingStatus.REFRESHING if binding.observation is not None else BindingStatus.WAITING
        )
        self.trace.emit(
            "external_refresh_started",
            step,
            binding_id=binding.binding_id,
            work_key=work_key,
            arguments_complete=binding.arguments_complete,
        )

    def _ensure_version_probe(
        self,
        binding: Binding,
        source: VersionAwareSource,
        step: int,
        *,
        terminal_validation: bool = False,
    ) -> None:
        work_key = binding.work_key
        if work_key in self._jobs:
            if terminal_validation:
                self._jobs[work_key].terminal_validation = True
            return
        if work_key in self._version_jobs:
            if terminal_validation:
                self._version_jobs[work_key].terminal_validation = True
            return
        task = asyncio.create_task(source.version(binding.arguments))
        task.add_done_callback(lambda _: self._external_progress.set())
        self._version_jobs[work_key] = _VersionJob(
            task=task,
            terminal_validation=terminal_validation,
        )
        binding.status = BindingStatus.REFRESHING
        self.trace.emit(
            "version_check_started",
            step,
            binding_id=binding.binding_id,
            work_key=work_key,
        )

    def _ensure_stream(self, binding: Binding, source: StreamingSource, step: int) -> None:
        work_key = binding.work_key
        if work_key in self._streams or work_key in self._completed_streams:
            return
        queue: asyncio.Queue[Observation] = asyncio.Queue()

        async def consume() -> None:
            async for observation in source.stream(binding.arguments):
                await queue.put(deepcopy(observation))
                self._external_progress.set()

        task = asyncio.create_task(consume())
        task.add_done_callback(lambda _: self._external_progress.set())
        self._streams[work_key] = _StreamJob(task=task, queue=queue)
        binding.status = (
            BindingStatus.REFRESHING if binding.observation is not None else BindingStatus.WAITING
        )
        self.trace.emit(
            "stream_started",
            step,
            binding_id=binding.binding_id,
            work_key=work_key,
        )

    def _drain_completed_version_jobs(self, step: int) -> None:
        for work_key, job in tuple(self._version_jobs.items()):
            if not job.task.done():
                continue
            del self._version_jobs[work_key]
            version = job.task.result()
            matching = tuple(
                binding for binding in self.bindings.active() if binding.work_key == work_key
            )
            if not matching:
                continue
            changed = any(
                binding.observation is None or binding.observation.version != version
                for binding in matching
            )
            self.trace.emit(
                "version_check_finished",
                step,
                work_key=work_key,
                version=version,
                changed=changed,
            )
            if changed:
                source = self.sources.get(matching[0].source)
                self._ensure_read(
                    matching[0],
                    source,
                    step,
                    terminal_validation=job.terminal_validation,
                )
                for binding in matching[1:]:
                    binding.status = BindingStatus.REFRESHING
                continue
            checked_at = time.monotonic()
            for binding in matching:
                binding.last_refresh_at = checked_at
                binding.status = BindingStatus.AVAILABLE
                if job.terminal_validation:
                    self._mark_terminal_validated(binding, step)

    def _drain_completed_jobs(self, step: int) -> None:
        for work_key, job in tuple(self._jobs.items()):
            if not job.task.done():
                continue
            del self._jobs[work_key]
            observation = deepcopy(job.task.result())
            matching = tuple(
                binding for binding in self.bindings.active() if binding.work_key == work_key
            )
            if matching and self.sources.get(matching[0].source).descriptor.cacheable:
                self._cache[work_key] = deepcopy(observation)

            self.trace.emit(
                "external_refresh_finished",
                step,
                work_key=work_key,
                version=observation.version,
            )
            self._apply_observation(matching, observation, step)
            if job.terminal_validation:
                for binding in matching:
                    self._mark_terminal_validated(binding, step)

    def _drain_stream_updates(self, step: int) -> None:
        for work_key, stream in tuple(self._streams.items()):
            matching = tuple(
                binding for binding in self.bindings.active() if binding.work_key == work_key
            )
            try:
                observation = stream.queue.get_nowait()
            except asyncio.QueueEmpty:
                observation = None
            if observation is not None:
                self.trace.emit(
                    "stream_observation",
                    step,
                    work_key=work_key,
                    version=observation.version,
                )
                self._apply_observation(matching, observation, step)

            if not stream.task.done() or not stream.queue.empty():
                continue
            del self._streams[work_key]
            self._completed_streams.add(work_key)
            stream.task.result()
            self.trace.emit("stream_finished", step, work_key=work_key)

    def _apply_observation(
        self, matching: tuple[Binding, ...], observation: Observation, step: int
    ) -> None:
        observed_at = time.monotonic()
        for binding in matching:
            binding.observation = deepcopy(observation)
            binding.last_refresh_at = observed_at
            binding.external_refreshes += 1
            binding.status = BindingStatus.AVAILABLE
            self.trace.emit(
                "binding_observation_updated",
                step,
                binding_id=binding.binding_id,
                work_key=binding.work_key,
                version=observation.version,
                runtime_step=self._runtime_step,
                **({
                    "source": binding.source,
                    "observation": deepcopy(observation.value),
                    "provenance": observation.provenance,
                } if self.config.trace_details else {}),
            )
            if binding.promote_to_fact:
                self._promote_fact(binding, observation)

    def _cancel_orphan_external_work(self, step: int) -> None:
        active_keys = {binding.work_key for binding in self.bindings.active()}
        for jobs, kind in (
            (self._jobs, "external_refresh_cancelled"),
            (self._version_jobs, "version_check_cancelled"),
            (self._streams, "stream_cancelled"),
        ):
            for work_key, job in tuple(jobs.items()):
                if work_key in active_keys:
                    continue
                del jobs[work_key]
                job.task.cancel()
                self._detached_tasks.add(job.task)
                job.task.add_done_callback(self._detached_tasks.discard)
                self.trace.emit(kind, step, work_key=work_key)

    def _promote_fact(self, binding: Binding, observation: Observation) -> None:
        timestamp = observation.observed_at or time.monotonic()
        self.facts.publish(
            FactItem(
                key=f"binding:{binding.need_id}",
                value=observation.value,
                source_type=binding.source,
                timestamp=timestamp,
                version=observation.version,
                provenance=observation.provenance,
            )
        )

    @staticmethod
    def _refresh_due(binding: Binding, now: float) -> bool:
        if binding.observation is None:
            return True
        if binding.freshness is FreshnessDemand.ONCE:
            return False
        if binding.freshness is FreshnessDemand.ALWAYS:
            return True
        if binding.max_age_s is None:
            return False
        if binding.last_refresh_at is None:
            return True
        return now - binding.last_refresh_at >= binding.max_age_s

    def _terminal_freshness_satisfied(self, step: int) -> bool:
        now = time.monotonic()
        pending = False
        for binding in self.bindings.active():
            if binding.status is BindingStatus.CANDIDATE:
                continue
            if binding.observation is None:
                self._launch_due_jobs(step)
                pending = True
                continue

            source = self.sources.get(binding.source)
            work_key = binding.work_key
            if (
                source.descriptor.streamable
                and binding.freshness is not FreshnessDemand.ONCE
                and isinstance(source, StreamingSource)
            ):
                stream = self._streams.get(work_key)
                if stream is not None and not stream.queue.empty():
                    pending = True
                continue

            if binding.freshness is FreshnessDemand.ONCE:
                continue
            if binding.freshness is FreshnessDemand.MAX_AGE and not self._refresh_due(binding, now):
                continue

            marker = self._observation_marker(binding)
            if (
                binding.freshness is FreshnessDemand.ALWAYS
                and self._terminal_validated.get(binding.binding_id) == marker
            ):
                continue

            if (
                isinstance(source, RuntimeSnapshotSource)
                and source.current_runtime_version(binding.arguments) == binding.observation.version
            ):
                self._mark_terminal_validated(binding, step)
                continue

            self.trace.emit(
                "terminal_freshness_validation_started",
                step,
                binding_id=binding.binding_id,
                work_key=work_key,
                runtime_step=self._runtime_step,
            )
            if source.descriptor.versioned and isinstance(source, VersionAwareSource):
                self._ensure_version_probe(
                    binding,
                    source,
                    step,
                    terminal_validation=True,
                )
            else:
                self._ensure_read(
                    binding,
                    source,
                    step,
                    terminal_validation=True,
                )
            pending = True
        return not pending

    def _mark_terminal_validated(self, binding: Binding, step: int) -> None:
        if binding.observation is None:
            return
        self._terminal_validated[binding.binding_id] = self._observation_marker(binding)
        self.trace.emit(
            "terminal_freshness_validated",
            step,
            binding_id=binding.binding_id,
            work_key=binding.work_key,
            version=binding.observation.version,
            runtime_step=self._runtime_step,
        )

    @staticmethod
    def _observation_marker(binding: Binding) -> tuple[str, str | None, float | None]:
        observation = binding.observation
        return (
            binding.work_key,
            None if observation is None else observation.version,
            binding.last_refresh_at,
        )

    def _observation_count(self) -> int:
        return sum(binding.external_refreshes for binding in self.bindings.all())

    @staticmethod
    def _deadline_expired(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    @staticmethod
    def _display_is_resolved_and_stable(
        previous: DisplayCanvas,
        proposed: DisplayCanvas,
    ) -> bool:
        if proposed.token_ids != previous.token_ids:
            return False
        if proposed.unresolved or proposed.realized_length <= 0:
            return False
        return proposed.eos_token_id is None or proposed.eos_token_id in proposed.token_ids

    def _external_progress_ready(self) -> bool:
        return (
            any(job.task.done() for job in self._jobs.values())
            or any(job.task.done() for job in self._version_jobs.values())
            or any(
                not stream.queue.empty() or stream.task.done() for stream in self._streams.values()
            )
        )

    async def _wait_for_external_progress(self, deadline: float | None) -> bool:
        while True:
            if self._deadline_expired(deadline):
                return False
            self._external_progress.clear()
            # Newly-created source tasks may register runtime-step work on their first turn.
            # Let them run before deciding that no scheduled external progress exists.
            await asyncio.sleep(0)
            if self._external_progress_ready():
                return True

            next_runtime_step = self.sources.next_runtime_step()
            if next_runtime_step is not None and next_runtime_step > self._runtime_step:
                self._runtime_step = next_runtime_step
                if self.sources.advance_runtime_step(self._runtime_step):
                    await asyncio.sleep(0)
                if self._external_progress_ready() or self._external_progress.is_set():
                    return True
                continue

            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                return False
            try:
                if timeout is None:
                    await self._external_progress.wait()
                else:
                    await asyncio.wait_for(self._external_progress.wait(), timeout=timeout)
            except TimeoutError:
                return False
            return True

    def _has_unresolved_active_binding(self) -> bool:
        return any(
            binding.observation is None
            for binding in self.bindings.active()
            if binding.status is not BindingStatus.CANDIDATE
        )

    def _has_pending_required_external_work(self) -> bool:
        if any(
            binding.status in {BindingStatus.WAITING, BindingStatus.REFRESHING}
            for binding in self.bindings.active()
        ):
            return True
        return any(not stream.queue.empty() for stream in self._streams.values())

    @classmethod
    def _proposal_signature(cls, update: ModelUpdate) -> tuple[Any, ...]:
        return (
            tuple(cls._cell_loop_key(cell) for cell in update.thought.cells),
            tuple(update.display.token_ids),
            cls._need_loop_keys(update),
            tuple(update.reopen_cell_ids),
            bool(update.equilibrium),
            bool(update.converged),
        )

    @classmethod
    def _behavior_signature(cls, update: ModelUpdate) -> tuple[Any, ...]:
        return (
            tuple((cell.cell_id, cell.lifecycle.value) for cell in update.thought.cells),
            tuple(update.display.token_ids),
            cls._need_loop_keys(update),
            tuple(update.reopen_cell_ids),
            bool(update.equilibrium),
            bool(update.converged),
        )

    @staticmethod
    def _need_loop_keys(update: ModelUpdate) -> tuple[tuple[Any, ...], ...]:
        keys: list[tuple[Any, ...]] = []
        for need in update.needs:
            source = need.selected_source()
            work_key = "" if source is None else canonical_work_key(source, need.arguments)
            keys.append(
                (
                    source,
                    work_key,
                    need.freshness.value,
                    None if need.max_age_s is None else round(float(need.max_age_s), 3),
                    tuple(target.identifier for target in need.target_cells),
                    tuple(target.identifier for target in need.target_display),
                    bool(need.promote_to_fact),
                )
            )
        return tuple(keys)

    @classmethod
    def _cell_loop_key(cls, cell: Any) -> tuple[Any, ...]:
        roles = tuple(
            sorted((role.value, round(float(weight), 3)) for role, weight in cell.roles.items())
        )
        return (
            cell.cell_id,
            cell.lifecycle.value,
            round(float(cell.uncertainty), 3),
            round(float(cell.noise), 3),
            roles,
            cls._semantic_sketch(cell.semantic),
        )

    @staticmethod
    def _semantic_sketch(semantic: tuple[float, ...], samples: int = 12) -> tuple[float, ...]:
        if len(semantic) <= samples:
            return tuple(round(float(value), 3) for value in semantic)
        last = len(semantic) - 1
        indexes = tuple(round(index * last / (samples - 1)) for index in range(samples))
        return tuple(round(float(semantic[index]), 3) for index in indexes)

    def _loop_period(
        self,
        history: list[_LoopSnapshot],
        *,
        behavior: bool = False,
    ) -> tuple[int, int] | None:
        signatures = tuple(
            item.behavior_signature if behavior else item.signature for item in history
        )
        for period in range(1, _LOOP_MAX_PERIOD + 1):
            repeats = _LOOP_MIN_REPEATS
            if behavior:
                repeats = max(
                    _BEHAVIOR_LOOP_MIN_REPEATS,
                    (_BEHAVIOR_LOOP_MIN_SPAN + period - 1) // period,
                )
            required = period * repeats
            if len(signatures) < required:
                continue
            tail = signatures[-required:]
            motif = tail[:period]
            if all(signature == motif[index % period] for index, signature in enumerate(tail)):
                return period, repeats
        return None

    @classmethod
    def _loop_targets(
        cls, cycle: tuple[_LoopSnapshot, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not cycle:
            return (), ()
        capacities = {item.proposed_thought.capacity for item in cycle}
        cell_slots: list[int] = []
        if len(capacities) == 1:
            for slot in range(cycle[0].proposed_thought.capacity):
                variants = {cls._cell_loop_key(item.proposed_thought.cells[slot]) for item in cycle}
                if len(variants) > 1:
                    cell_slots.append(slot)

        display_lengths = {len(item.proposed_display.token_ids) for item in cycle}
        display_positions: list[int] = []
        if len(display_lengths) == 1:
            for position in range(len(cycle[0].proposed_display.token_ids)):
                variants = {item.proposed_display.token_ids[position] for item in cycle}
                if len(variants) > 1:
                    display_positions.append(position)

        if not cell_slots:
            latest = cycle[-1].proposed_thought
            cell_slots = [
                slot
                for slot, cell in enumerate(latest.cells)
                if cell.live and cell.lifecycle is not CellLifecycle.WAITING
            ]
        return tuple(cell_slots), tuple(display_positions)

    def _rediffuse_loop_state(
        self,
        base_thought: CognitiveField,
        base_display: DisplayCanvas,
        *,
        cell_slots: tuple[int, ...],
        display_positions: tuple[int, ...],
        current_thought: CognitiveField,
        current_display: DisplayCanvas,
    ) -> tuple[CognitiveField, DisplayCanvas]:
        compatible_thought = (
            base_thought.capacity == current_thought.capacity
            and base_thought.width == current_thought.width
            and tuple((cell.cell_id, cell.lifecycle) for cell in base_thought.cells)
            == tuple((cell.cell_id, cell.lifecycle) for cell in current_thought.cells)
        )
        thought_source = base_thought if compatible_thought else current_thought
        cells = list(thought_source.cells)
        effective_slots = cell_slots or tuple(
            slot
            for slot, cell in enumerate(cells)
            if cell.live and cell.lifecycle is not CellLifecycle.WAITING
        )
        for slot in effective_slots:
            if not 0 <= slot < len(cells):
                continue
            cell = cells[slot]
            if not cell.live or cell.lifecycle is CellLifecycle.WAITING:
                continue
            cells[slot] = cell.reopened(_LOOP_REDIFFUSION_NOISE)
        thought = replace(
            thought_source,
            cells=tuple(cells),
            step=current_thought.step,
            next_cell_serial=max(thought_source.next_cell_serial, current_thought.next_cell_serial),
        )

        compatible_display = (
            len(base_display.token_ids) == len(current_display.token_ids)
            and base_display.mask_token_id == current_display.mask_token_id
            and base_display.eos_token_id == current_display.eos_token_id
        )
        display_source = base_display if compatible_display else current_display
        token_ids = list(display_source.token_ids)
        for position in display_positions:
            if not 0 <= position < len(token_ids):
                continue
            if (
                display_source.eos_token_id is not None
                and token_ids[position] == display_source.eos_token_id
            ):
                continue
            token_ids[position] = display_source.mask_token_id
        display = replace(
            display_source,
            token_ids=tuple(token_ids),
            step=current_display.step,
        )
        return thought, display

    def _maybe_reclaim(
        self,
        thought: CognitiveField,
        step: int,
        *,
        force: bool = False,
    ) -> CognitiveField:
        pinned = self.bindings.pinned_target_cells() | self._strong_link_pins(thought)
        selected = retired_reclamation_candidates(
            thought,
            retired_at=self._retired_at,
            step=step,
            grace_steps=self.config.reclamation_grace_steps,
            low_watermark=self.config.reclamation_low_watermark,
            target_watermark=self.config.reclamation_target_watermark,
            pinned_cell_ids=pinned,
            force=force,
        )
        if not selected:
            return thought

        field = thought
        for slot, cell_id in selected:
            cell = thought.get(cell_id)
            tombstone = self.archive.record(
                cell,
                created_step=self._created_at.get(cell_id, 0),
                retired_step=self._retired_at[cell_id],
                archived_step=step,
                physical_slot=slot,
                binding_ids=self.bindings.binding_ids_targeting(cell_id),
            )
            field = field.reclaim(cell_id)
            self.trace.emit(
                "cell_reclaimed",
                step,
                cell_id=tombstone.cell_id,
                retired_step=tombstone.retired_step,
                archived_step=tombstone.archived_step,
            )

        field = field.compact()
        self.trace.emit(
            "cognitive_compaction",
            step,
            reclaimed=len(selected),
            empty_slots=field.empty_count,
        )
        return field

    @staticmethod
    def _strong_link_pins(thought: CognitiveField) -> frozenset[str]:
        return frozenset(
            link.target.identifier
            for cell in thought.cells
            if cell.live
            for link in cell.links
            if link.target.kind is ObjectKind.CELL and link.relation in STRONG_LINK_RELATIONS
        )

    def _record_lifecycle_steps(
        self, previous: CognitiveField, current: CognitiveField, step: int
    ) -> None:
        previous_ids = set(previous.occupied_cell_ids)
        for cell_id in current.occupied_cell_ids:
            if cell_id not in previous_ids:
                self._created_at.setdefault(cell_id, step)
            current_cell = current.get(cell_id)
            if current_cell.lifecycle is not CellLifecycle.RETIRED:
                continue
            if (
                cell_id not in previous_ids
                or previous.get(cell_id).lifecycle is not CellLifecycle.RETIRED
            ):
                self._retired_at.setdefault(cell_id, step)

    def _trace_lifecycle_changes(
        self, previous: CognitiveField, current: CognitiveField, step: int
    ) -> None:
        previous_ids = set(previous.occupied_cell_ids)
        for cell_id in current.occupied_cell_ids:
            current_cell = current.get(cell_id)
            if cell_id not in previous_ids:
                self.trace.emit(
                    "cell_allocated",
                    step,
                    cell_id=cell_id,
                    lifecycle=current_cell.lifecycle.value,
                )
                continue
            old = previous.get(cell_id).lifecycle
            if old is not current_cell.lifecycle:
                self.trace.emit(
                    "lifecycle_transition",
                    step,
                    cell_id=cell_id,
                    previous=old.value,
                    current=current_cell.lifecycle.value,
                )

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cid.contracts import CIDPolicy, FreshnessDemand, ModelContext, Observation, Percept
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
from cid.runtime.bindings import Binding, BindingStatus, BindingTable
from cid.runtime.sources import (
    ReadOnlySource,
    RuntimeSnapshotSource,
    SourceRegistry,
    StreamingSource,
    VersionAwareSource,
)
from cid.runtime.trace import RuntimeTrace
from cid.state import CellLifecycle, CognitiveField, DisplayCanvas, FactItem, FactStore


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_steps: int = 64
    max_wall_time_s: float | None = 300.0
    binding_threshold: float = DEFAULT_BINDING_THRESHOLD
    idle_yield_s: float = 0.001
    reclamation_grace_steps: int = DEFAULT_RECLAMATION_GRACE_STEPS
    reclamation_low_watermark: float = DEFAULT_RECLAMATION_LOW_WATERMARK
    reclamation_target_watermark: float = DEFAULT_RECLAMATION_TARGET_WATERMARK

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
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

        try:
            while True:
                if self._deadline_expired(deadline):
                    self.trace.emit("wall_clock_budget_exhausted", completed_steps)
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
                observations_before = self._observation_count()
                self._drain_completed_version_jobs(step)
                self._drain_completed_jobs(step)
                self._drain_stream_updates(step)
                if self._observation_count() > observations_before:
                    epoch_steps = 0
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
                )

                self.trace.emit(
                    "model_step_started",
                    step,
                    percepts=len(percepts),
                    runtime_step=self._runtime_step,
                )
                update = await asyncio.to_thread(policy.step, context)
                self.trace.emit(
                    "model_step_finished",
                    step,
                    needs=len(update.needs),
                    equilibrium=update.equilibrium,
                    converged=update.converged,
                    runtime_step=self._runtime_step,
                )
                completed_steps += 1
                epoch_steps += 1
                proposed_thought = update.thought
                display = update.display
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

        self.trace.emit(
            "trajectory_finished",
            completed_steps,
            converged=converged,
            runtime_step=self._runtime_step,
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
            )
        return tuple(percepts)

    def _launch_due_jobs(self, step: int) -> None:
        now = time.monotonic()
        for binding in self.bindings.active():
            work_key = binding.work_key
            source = self.sources.get(binding.source)
            if (
                not binding.arguments_complete
                and not source.descriptor.accepts_partial_arguments
            ):
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
            BindingStatus.REFRESHING
            if binding.observation is not None
            else BindingStatus.WAITING
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

    def _ensure_stream(
        self, binding: Binding, source: StreamingSource, step: int
    ) -> None:
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
            BindingStatus.REFRESHING
            if binding.observation is not None
            else BindingStatus.WAITING
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

    def _external_progress_ready(self) -> bool:
        return (
            any(job.task.done() for job in self._jobs.values())
            or any(job.task.done() for job in self._version_jobs.values())
            or any(
                not stream.queue.empty() or stream.task.done()
                for stream in self._streams.values()
            )
        )

    async def _wait_for_external_progress(self, deadline: float | None) -> bool:
        while True:
            if self._deadline_expired(deadline):
                return False
            self._external_progress.clear()
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

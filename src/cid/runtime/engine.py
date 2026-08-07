from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from cid.contracts import CIDPolicy, FreshnessDemand, ModelContext, Observation, Percept
from cid.runtime.bindings import Binding, BindingStatus, BindingTable
from cid.runtime.sources import SourceRegistry
from cid.runtime.trace import RuntimeTrace
from cid.state import CognitiveField, DisplayCanvas, FactItem, FactStore


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_steps: int = 64
    binding_threshold: float = 0.55
    idle_yield_s: float = 0.001

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.binding_threshold <= 1.0:
            raise ValueError("binding_threshold must be in [0, 1]")
        if self.idle_yield_s < 0:
            raise ValueError("idle_yield_s must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    thought: CognitiveField
    display: DisplayCanvas
    facts: tuple[FactItem, ...]
    bindings: tuple[Binding, ...]
    trace: RuntimeTrace
    steps: int
    converged: bool


@dataclass(slots=True)
class _ExternalJob:
    task: asyncio.Task[Observation]
    owner_binding_id: str


class CIDRuntime:
    def __init__(self, sources: SourceRegistry, config: RuntimeConfig | None = None) -> None:
        self.sources = sources
        self.config = config or RuntimeConfig()
        self.facts = FactStore()
        self.bindings = BindingTable()
        self.trace = RuntimeTrace()
        self._jobs: dict[str, _ExternalJob] = {}
        self._cache: dict[str, Observation] = {}

    async def run(
        self,
        policy: CIDPolicy,
        *,
        thought: CognitiveField,
        display: DisplayCanvas,
        facts: Iterable[FactItem] = (),
    ) -> RuntimeResult:
        if self._jobs:
            raise RuntimeError("a CIDRuntime instance cannot run concurrent trajectories")
        self.facts = FactStore()
        self.bindings = BindingTable()
        self.trace = RuntimeTrace()
        self._cache = {}
        for item in facts:
            self.facts.publish(item)

        converged = False
        completed_steps = 0
        descriptors = self.sources.descriptors()
        required_args = {d.name: d.required_arguments for d in descriptors}

        try:
            for step in range(self.config.max_steps):
                self._drain_completed_jobs(step)
                percepts = self._project_available(step)
                context = ModelContext(
                    facts=self.facts.snapshot(),
                    thought=thought,
                    display=display,
                    sources=descriptors,
                    percepts=percepts,
                    step=step,
                )

                self.trace.emit("model_step_started", step, percepts=len(percepts))
                update = await asyncio.to_thread(policy.step, context)
                self.trace.emit("model_step_finished", step, needs=len(update.needs))
                completed_steps = step + 1
                thought = update.thought
                display = update.display

                for need in update.needs:
                    source = need.selected_source()
                    required = required_args.get(source or "", ())
                    executable = source in required_args and all(
                        name in need.arguments for name in required
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
                    source_descriptors=required_args,
                )
                for binding in touched:
                    self.trace.emit(
                        "binding_active",
                        step,
                        binding_id=binding.binding_id,
                        need_id=binding.need_id,
                        source=binding.source,
                    )

                self._drain_completed_jobs(step)
                unresolved = self._has_unresolved_active_binding()
                if update.converged and not unresolved:
                    converged = True
                    break
                self._launch_due_jobs(step)

                if self._jobs and self.config.idle_yield_s:
                    await asyncio.sleep(self.config.idle_yield_s)
                else:
                    await asyncio.sleep(0)
        finally:
            for job in self._jobs.values():
                job.task.cancel()
            if self._jobs:
                await asyncio.gather(
                    *(job.task for job in self._jobs.values()), return_exceptions=True
                )
            self._jobs.clear()

        snapshot = self.facts.snapshot()
        return RuntimeResult(
            thought=thought,
            display=display,
            facts=tuple(snapshot.items.values()),
            bindings=self.bindings.all(),
            trace=self.trace,
            steps=completed_steps,
            converged=converged,
        )

    def _project_available(self, step: int) -> tuple[Percept, ...]:
        percepts: list[Percept] = []
        for binding in self.bindings.active():
            if binding.observation is None:
                continue
            binding.cognitive_projections += 1
            percepts.append(
                Percept(
                    binding_id=binding.binding_id,
                    source=binding.source,
                    observation=deepcopy(binding.observation),
                    target_cells=binding.target_cells,
                    target_display=binding.target_display,
                    projection_index=binding.cognitive_projections,
                )
            )
            self.trace.emit(
                "cognitive_projection",
                step,
                binding_id=binding.binding_id,
                index=binding.cognitive_projections,
            )
        return tuple(percepts)

    def _launch_due_jobs(self, step: int) -> None:
        now = time.monotonic()
        for binding in self.bindings.active():
            work_key = binding.work_key
            source = self.sources.get(binding.source)

            if (
                binding.observation is None
                and source.descriptor.cacheable
                and work_key in self._cache
            ):
                binding.observation = self._cache[work_key]
                binding.status = BindingStatus.AVAILABLE
                self.trace.emit("cache_hit", step, binding_id=binding.binding_id)

            if not self._refresh_due(binding, now):
                continue
            if work_key in self._jobs:
                job = self._jobs[work_key]
                binding.status = (
                    BindingStatus.REFRESHING
                    if binding.observation is not None
                    else BindingStatus.WAITING
                )
                if binding.binding_id != job.owner_binding_id:
                    self.trace.emit("job_deduplicated", step, binding_id=binding.binding_id)
                continue

            task = asyncio.create_task(source.read(binding.arguments))
            self._jobs[work_key] = _ExternalJob(task=task, owner_binding_id=binding.binding_id)
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
            )

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
                self._cache[work_key] = observation

            self.trace.emit(
                "external_refresh_finished",
                step,
                work_key=work_key,
                version=observation.version,
            )

            for binding in matching:
                binding.observation = observation
                binding.last_refresh_at = time.monotonic()
                binding.external_refreshes += 1
                binding.status = BindingStatus.AVAILABLE
                self.trace.emit(
                    "binding_observation_updated",
                    step,
                    binding_id=binding.binding_id,
                    version=observation.version,
                )
                if binding.promote_to_fact:
                    self._promote_fact(binding, observation)

    def _promote_fact(self, binding: Binding, observation: Observation) -> None:
        timestamp = observation.observed_at or time.monotonic()
        self.facts.publish(
            FactItem(
                key=f"binding:{binding.binding_id}",
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

    def _has_unresolved_active_binding(self) -> bool:
        return any(
            binding.observation is None
            for binding in self.bindings.active()
            if binding.status is not BindingStatus.RETIRED
        )

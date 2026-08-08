from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from cid.contracts import CIDPolicy, FreshnessDemand, ModelContext, Observation, Percept
from cid.grounding import STRONG_LINK_RELATIONS, ClosedWorldGrounder, ObjectKind, ObjectRef
from cid.lifecycle import LifecycleTransitionController, LifecycleTransitionSignals
from cid.runtime.archive import CognitiveArchive, CognitiveTombstone
from cid.runtime.bindings import Binding, BindingStatus, BindingTable
from cid.runtime.sources import SourceRegistry
from cid.runtime.trace import RuntimeTrace
from cid.state import CellLifecycle, CognitiveField, DisplayCanvas, FactItem, FactStore


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_steps: int = 64
    binding_threshold: float = 0.55
    idle_yield_s: float = 0.001
    reclamation_grace_steps: int = 2
    reclamation_low_watermark: float = 0.125
    reclamation_target_watermark: float = 0.25

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
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
        if self._jobs:
            raise RuntimeError("a CIDRuntime instance cannot run concurrent trajectories")
        self.facts = FactStore()
        self.bindings = BindingTable()
        self.trace = RuntimeTrace()
        self.archive = CognitiveArchive()
        self._cache = {}
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
        descriptors = self.sources.descriptors()
        required_args = {d.name: d.required_arguments for d in descriptors}

        try:
            for step in range(self.config.max_steps):
                if self.sources.advance_runtime_step(step):
                    await asyncio.sleep(0)
                thought = self._maybe_reclaim(thought, step)
                previous_thought = thought
                self._drain_completed_jobs(step)
                percepts = self._project_available(step, thought)
                context = ModelContext(
                    facts=self.facts.snapshot(),
                    thought=thought,
                    display=display,
                    sources=descriptors,
                    percepts=percepts,
                    step=step,
                    prompt=prompt,
                )

                self.trace.emit("model_step_started", step, percepts=len(percepts))
                update = await asyncio.to_thread(policy.step, context)
                self.trace.emit("model_step_finished", step, needs=len(update.needs))
                completed_steps = step + 1
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

    def _maybe_reclaim(
        self,
        thought: CognitiveField,
        step: int,
        *,
        force: bool = False,
    ) -> CognitiveField:
        low = math.ceil(thought.capacity * self.config.reclamation_low_watermark)
        target = math.ceil(thought.capacity * self.config.reclamation_target_watermark)
        if not force and thought.empty_count >= low:
            return thought

        binding_pins = self.bindings.pinned_target_cells()
        strong_link_pins = self._strong_link_pins(thought)
        candidates = []
        for slot, cell in enumerate(thought.cells):
            if cell.cell_id is None or cell.lifecycle is not CellLifecycle.RETIRED:
                continue
            retired_step = self._retired_at.setdefault(cell.cell_id, step)
            if step - retired_step < self.config.reclamation_grace_steps:
                continue
            if cell.cell_id in binding_pins or cell.cell_id in strong_link_pins:
                continue
            candidates.append((retired_step, slot, cell.cell_id))

        field = thought
        reclaimed = 0
        for _, _, cell_id in sorted(candidates):
            if not force and field.empty_count >= target:
                break
            slot = field.slot_of(cell_id)
            cell = field.get(cell_id)
            tombstone = self.archive.record(
                cell,
                created_step=self._created_at.get(cell_id, 0),
                retired_step=self._retired_at[cell_id],
                archived_step=step,
                physical_slot=slot,
                binding_ids=self.bindings.binding_ids_targeting(cell_id),
            )
            field = field.reclaim(cell_id)
            reclaimed += 1
            self.trace.emit(
                "cell_reclaimed",
                step,
                cell_id=tombstone.cell_id,
                retired_step=tombstone.retired_step,
                archived_step=tombstone.archived_step,
            )

        if reclaimed:
            field = field.compact()
            self.trace.emit(
                "cognitive_compaction",
                step,
                reclaimed=reclaimed,
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

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from cid.contracts import (
    ArgumentDescriptor,
    FreshnessDemand,
    InformationNeed,
    ModelContext,
    ModelUpdate,
    Observation,
    SourceDescriptor,
)
from cid.grounding import (
    Anchor,
    AnchorKind,
    ClosedWorldGrounder,
    GroundingEntry,
    ObjectRef,
)
from cid.metrics import summarize_runtime
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry
from cid.state import CellLifecycle, CognitiveField, DisplayCanvas


def seeded_thought(live_cells: int, capacity: int | None = None) -> CognitiveField:
    field = CognitiveField.empty(capacity=capacity or live_cells, width=4)
    for _ in range(live_cells):
        field, _ = field.allocate()
    return field


@dataclass
class CountingSource:
    delay_s: float = 0.0
    reads: int = 0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="source",
            description="count reads",
            arguments=(ArgumentDescriptor(name="key", kind="string"),),
            cacheable=True,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        self.reads += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return Observation(
            value=arguments["key"],
            version=str(self.reads),
            observed_at=time.monotonic(),
        )


class DuplicateNeedPolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.01)
        cell_a, cell_b = context.thought.live_cell_ids[:2]
        common = dict(
            source_scores={"source": 1.0},
            arguments={"key": "same"},
            confidence=1.0,
            freshness=FreshnessDemand.ONCE,
        )
        needs = (
            InformationNeed(need_id="need-a", target_cells=(ObjectRef.cell(cell_a),), **common),
            InformationNeed(need_id="need-b", target_cells=(ObjectRef.cell(cell_b),), **common),
        )
        converged = len(context.percepts) == 2
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=needs,
            converged=converged,
        )


class PersistentNeedPolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.005)
        target_cell = context.thought.live_cell_ids[0]
        need = InformationNeed(
            need_id="persistent",
            source_scores={"source": 1.0},
            arguments={"key": "value"},
            confidence=1.0,
            freshness=FreshnessDemand.ONCE,
            target_cells=(ObjectRef.cell(target_cell),),
        )
        projection_index = context.percepts[0].projection_index if context.percepts else 0
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=projection_index >= 3,
        )


@dataclass
class IncrementingDynamicSource:
    reads: int = 0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="dynamic",
            description="increment on every read",
            cacheable=False,
            dynamic=True,
            versioned=True,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        self.reads += 1
        await asyncio.sleep(0)
        return Observation(value=self.reads, version=str(self.reads), observed_at=time.monotonic())


@dataclass
class AnchoredSource:
    anchor: Anchor

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(name="grounded", description="return grounded evidence")

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        await asyncio.sleep(0)
        return Observation(value="evidence", anchors=(self.anchor,))


class AlwaysRefreshPolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.003)
        target_cell = context.thought.live_cell_ids[0]
        need = InformationNeed(
            need_id="dynamic-state",
            source_scores={"dynamic": 1.0},
            confidence=1.0,
            freshness=FreshnessDemand.ALWAYS,
            target_cells=(ObjectRef.cell(target_cell),),
        )
        latest = int(context.percepts[0].observation.value) if context.percepts else 0
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=latest >= 3,
        )


class RebindingPolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.003)
        seen = context.percepts[0].observation.value if context.percepts else None
        key = "b" if seen == "a" else "a"
        if seen == "b":
            key = "b"
        need = InformationNeed(
            need_id="changing-selector",
            source_scores={"source": 1.0},
            arguments={"key": key},
            confidence=1.0,
            freshness=FreshnessDemand.ONCE,
        )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=seen == "b",
        )


class StableIdCompactionPolicy:
    def __init__(self, target_cell_id: str) -> None:
        self.target_cell_id = target_cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.003)
        compacted = context.thought.compact()
        need = InformationNeed(
            need_id="stable-target",
            source_scores={"source": 1.0},
            arguments={"key": "value"},
            confidence=1.0,
            target_cells=(ObjectRef.cell(self.target_cell_id),),
        )
        return ModelUpdate(
            thought=compacted.advance(compacted.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=bool(context.percepts),
        )


class WaitingLifecyclePolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.005)
        cell_id = context.thought.live_cell_ids[0]
        cells = list(context.thought.cells)
        slot = context.thought.slot_of(cell_id)
        requested = CellLifecycle.ACTIVE if context.step > 0 else CellLifecycle.WAITING
        cells[slot] = replace(cells[slot], lifecycle=requested)
        need = InformationNeed(
            need_id="wait-for-source",
            source_scores={"source": 1.0},
            arguments={"key": "ready"},
            confidence=1.0,
            target_cells=(ObjectRef.cell(cell_id),),
        )
        return ModelUpdate(
            thought=context.thought.advance(tuple(cells)),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=bool(context.percepts),
        )


class QuiescentEvidencePolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        cell_id = context.thought.live_cell_ids[0]
        need = InformationNeed(
            need_id="quiescent-evidence",
            source_scores={"source": 1.0},
            arguments={"key": "evidence"},
            confidence=1.0,
            target_cells=(ObjectRef.cell(cell_id),),
        )
        if context.percepts:
            return ModelUpdate(
                thought=context.thought.advance(context.thought.cells),
                display=context.display.advance((7,)),
                needs=(need,),
                equilibrium=True,
                converged=True,
            )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            equilibrium=True,
        )


class GroundedRoutingPolicy:
    def __init__(self, request_cell: str, related_cell: str) -> None:
        self.request_cell = request_cell
        self.related_cell = related_cell

    def step(self, context: ModelContext) -> ModelUpdate:
        need = InformationNeed(
            need_id="grounded-evidence",
            source_scores={"grounded": 1.0},
            confidence=1.0,
            target_cells=(ObjectRef.cell(self.request_cell),),
        )
        routed = bool(context.percepts) and ObjectRef.cell(self.related_cell) in (
            context.percepts[0].target_cells
        )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=routed,
        )


async def test_duplicate_needs_share_one_external_read() -> None:
    source = CountingSource(delay_s=0.015)
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))

    result = await runtime.run(
        DuplicateNeedPolicy(),
        thought=seeded_thought(2),
        display=DisplayCanvas.masked(2, -1),
    )

    assert result.converged
    assert source.reads == 1
    assert result.trace.count("external_refresh_started") == 1
    assert result.trace.count("job_deduplicated") >= 1


async def test_static_observation_is_reprojected_without_refetch() -> None:
    source = CountingSource()
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))

    result = await runtime.run(
        PersistentNeedPolicy(),
        thought=seeded_thought(1),
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.reads == 1
    assert result.bindings[0].cognitive_projections >= 3
    assert result.bindings[0].external_refreshes == 1


async def test_model_compute_overlaps_source_wait() -> None:
    source = CountingSource(delay_s=0.04)
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))

    result = await runtime.run(
        DuplicateNeedPolicy(),
        thought=seeded_thought(2),
        display=DisplayCanvas.masked(2, -1),
    )
    metrics = summarize_runtime(result)

    assert result.converged
    assert metrics.model_steps_during_io >= 1
    assert metrics.tool_wait_overlap_s > 0


async def test_dynamic_source_refreshes_without_cache_reuse() -> None:
    source = IncrementingDynamicSource()
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))

    result = await runtime.run(
        AlwaysRefreshPolicy(),
        thought=seeded_thought(1),
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.reads == 4
    assert result.trace.count("cache_hit") == 0
    assert result.trace.count("terminal_freshness_validated") == 1


async def test_changed_arguments_invalidate_old_observation() -> None:
    source = CountingSource()
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))

    result = await runtime.run(
        RebindingPolicy(),
        thought=seeded_thought(1),
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.reads == 2
    assert result.bindings[-1].observation is not None
    assert result.bindings[-1].observation.value == "b"


async def test_binding_target_survives_physical_slot_compaction() -> None:
    source = CountingSource()
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))
    thought = CognitiveField.empty(capacity=3, width=4)
    thought, first = thought.allocate()
    thought, target = thought.allocate()
    thought = thought.retire(first).reclaim(first)
    assert thought.slot_of(target) == 1

    result = await runtime.run(
        StableIdCompactionPolicy(target),
        thought=thought,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert result.thought.slot_of(target) == 0
    assert result.bindings[-1].target_cells == (ObjectRef.cell(target),)


async def test_runtime_keeps_waiting_cell_blocked_until_observation_is_available() -> None:
    source = CountingSource(delay_s=0.02)
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20))
    thought = seeded_thought(1)
    cell_id = thought.live_cell_ids[0]

    result = await runtime.run(
        WaitingLifecyclePolicy(),
        thought=thought,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert result.thought.get(cell_id).lifecycle is CellLifecycle.ACTIVE
    transitions = tuple(
        event for event in result.trace.events if event.kind == "lifecycle_transition"
    )
    assert any(event.payload["current"] == "waiting" for event in transitions)
    assert any(
        event.payload["previous"] == "waiting" and event.payload["current"] == "active"
        for event in transitions
    )


async def test_quiescence_waits_without_consuming_the_refinement_epoch_budget() -> None:
    source = CountingSource(delay_s=0.02)
    registry = SourceRegistry()
    registry.register(source)
    runtime = CIDRuntime(
        registry,
        RuntimeConfig(max_steps=1, max_wall_time_s=1.0),
    )

    result = await runtime.run(
        QuiescentEvidencePolicy(),
        thought=seeded_thought(1),
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert result.steps == 2
    assert source.reads == 1
    starts = tuple(event for event in result.trace.events if event.kind == "quiescence_started")
    assert any(event.payload["reason"] == "current_information_equilibrium" for event in starts)
    assert result.trace.count("compute_budget_exhausted") == 0


async def test_observation_anchor_routes_percept_to_related_cognitive_cell() -> None:
    anchor = Anchor(
        anchor_id="a:model-a",
        kind=AnchorKind.ENTITY,
        value="Model A",
        object_id="model:a",
    )
    grounder = ClosedWorldGrounder((GroundingEntry(anchor=anchor),))
    registry = SourceRegistry()
    registry.register(AnchoredSource(anchor))
    runtime = CIDRuntime(registry, RuntimeConfig(max_steps=20), grounder=grounder)
    thought = CognitiveField.empty(capacity=3, width=4)
    thought, request_cell = thought.allocate()
    thought, related_cell = thought.allocate(anchors=(anchor,))

    result = await runtime.run(
        GroundedRoutingPolicy(request_cell, related_cell),
        thought=thought,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    projections = tuple(
        event for event in result.trace.events if event.kind == "cognitive_projection"
    )
    assert any(event.payload["grounded_targets"] == 1 for event in projections)


@dataclass
class ProgressiveLookupSource:
    reads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="progressive",
            description="accept a coarse selector before all fields bind",
            arguments=(
                ArgumentDescriptor(name="key", kind="string"),
                ArgumentDescriptor(name="field", kind="string"),
            ),
            cacheable=False,
            accepts_partial_arguments=True,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        self.reads.append(dict(arguments))
        await asyncio.sleep(0)
        if "field" not in arguments:
            return Observation(value="candidate", version="partial")
        return Observation(value="37", version="exact")


class ProgressiveNeedPolicy:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        percept = context.percepts[0] if context.percepts else None
        exact = percept is not None and percept.observation.version == "exact"
        arguments = {"key": "model-a"}
        if percept is not None:
            arguments["field"] = "latency"
        need = InformationNeed(
            need_id="progressive-need",
            source_scores={"progressive": 1.0},
            arguments=arguments,
            confidence=1.0,
            target_cells=(ObjectRef.cell(self.cell_id),),
        )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=exact,
        )


@dataclass
class StepVersionedSource:
    reads: int = 0
    probes: int = 0
    version_id: int = 0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="versioned",
            description="cheap version probe before full refresh",
            cacheable=False,
            dynamic=True,
            versioned=True,
        )

    def on_runtime_step(self, step: int) -> None:
        if step == 3:
            self.version_id = 1

    async def version(self, arguments: Mapping[str, Any]) -> str:
        del arguments
        self.probes += 1
        await asyncio.sleep(0)
        return str(self.version_id)

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        self.reads += 1
        await asyncio.sleep(0)
        return Observation(value=self.version_id, version=str(self.version_id))


class VersionRefreshPolicy:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        latest = context.percepts[0].observation.version if context.percepts else None
        need = InformationNeed(
            need_id="versioned-need",
            source_scores={"versioned": 1.0},
            confidence=1.0,
            freshness=FreshnessDemand.ALWAYS,
            target_cells=(ObjectRef.cell(self.cell_id),),
        )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=latest == "1",
        )


@dataclass
class StreamingCounterSource:
    read_calls: int = 0
    stream_calls: int = 0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name="stream",
            description="finite test stream",
            cacheable=False,
            dynamic=True,
            streamable=True,
            versioned=True,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        self.read_calls += 1
        return Observation(value=-1, version="read")

    async def stream(self, arguments: Mapping[str, Any]):
        del arguments
        self.stream_calls += 1
        yield Observation(value=1, version="v1")
        await asyncio.sleep(0)
        yield Observation(value=2, version="v2")


class StreamPolicy:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        latest = context.percepts[0].observation.version if context.percepts else None
        need = InformationNeed(
            need_id="stream-need",
            source_scores={"stream": 1.0},
            confidence=1.0,
            freshness=FreshnessDemand.ALWAYS,
            target_cells=(ObjectRef.cell(self.cell_id),),
        )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=latest == "v2",
        )


async def test_partial_binding_can_start_io_before_required_arguments_converge() -> None:
    source = ProgressiveLookupSource()
    registry = SourceRegistry()
    registry.register(source)
    field, cell_id = CognitiveField.empty(capacity=2, width=4).allocate()

    result = await CIDRuntime(registry, RuntimeConfig(max_steps=8)).run(
        ProgressiveNeedPolicy(cell_id),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.reads[0] == {"key": "model-a"}
    assert source.reads[-1] == {"key": "model-a", "field": "latency"}
    starts = tuple(
        event for event in result.trace.events if event.kind == "external_refresh_started"
    )
    assert starts[0].payload["arguments_complete"] is False
    first_executable = next(
        event.step
        for event in result.trace.events
        if event.kind == "information_need" and event.payload["executable"] is True
    )
    assert starts[0].step < first_executable


async def test_version_probe_avoids_full_refetch_until_source_changes() -> None:
    source = StepVersionedSource()
    registry = SourceRegistry()
    registry.register(source)
    field, cell_id = CognitiveField.empty(capacity=2, width=4).allocate()

    result = await CIDRuntime(registry, RuntimeConfig(max_steps=10)).run(
        VersionRefreshPolicy(cell_id),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.reads == 2
    assert source.probes >= 2
    assert result.trace.count("external_refresh_started") == 2
    checks = tuple(
        event for event in result.trace.events if event.kind == "version_check_finished"
    )
    assert any(event.payload["changed"] is False for event in checks)
    assert any(event.payload["changed"] is True for event in checks)


async def test_streaming_source_updates_one_persistent_binding_without_polling_reads() -> None:
    source = StreamingCounterSource()
    registry = SourceRegistry()
    registry.register(source)
    field, cell_id = CognitiveField.empty(capacity=2, width=4).allocate()

    result = await CIDRuntime(registry, RuntimeConfig(max_steps=8)).run(
        StreamPolicy(cell_id),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.converged
    assert source.read_calls == 0
    assert source.stream_calls == 1
    assert result.trace.count("stream_started") == 1
    assert result.trace.count("stream_observation") == 2
    assert result.bindings[-1].external_refreshes == 2

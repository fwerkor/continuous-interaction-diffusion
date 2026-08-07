from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
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
from cid.metrics import summarize_runtime
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry
from cid.state import CognitiveField, DisplayCanvas


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
            InformationNeed(need_id="need-a", target_cells=(cell_a,), **common),
            InformationNeed(need_id="need-b", target_cells=(cell_b,), **common),
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
            target_cells=(target_cell,),
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


class AlwaysRefreshPolicy:
    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.003)
        target_cell = context.thought.live_cell_ids[0]
        need = InformationNeed(
            need_id="dynamic-state",
            source_scores={"dynamic": 1.0},
            confidence=1.0,
            freshness=FreshnessDemand.ALWAYS,
            target_cells=(target_cell,),
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
            target_cells=(self.target_cell_id,),
        )
        return ModelUpdate(
            thought=compacted.advance(compacted.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
            converged=bool(context.percepts),
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
    assert source.reads == 3
    assert result.trace.count("cache_hit") == 0


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
    assert result.bindings[-1].target_cells == (target,)

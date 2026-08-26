from __future__ import annotations

import time

from cid.contracts import FreshnessDemand, InformationNeed, ModelContext, ModelUpdate
from cid.data import ExternalEvent, TrajectoryExample
from cid.evaluation import evaluate_runtime_result, run_replay_case, summarize_evaluations
from cid.grounding import ObjectRef
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry, StaticMappingSource
from cid.state import CognitiveField, DisplayCanvas


class EvaluationPolicy:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.004)
        arguments = {} if context.step == 0 else {"key": "latency_ms"}
        need = InformationNeed(
            need_id="latency",
            source_scores={"docs": 1.0},
            arguments=arguments,
            confidence=1.0,
            target_cells=(ObjectRef.cell(self.cell_id),),
        )
        if context.percepts:
            return ModelUpdate(
                thought=context.thought.advance(context.thought.cells),
                display=context.display.advance((37,)),
                needs=(need,),
                converged=True,
            )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
        )


class ReplayPolicy:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        need = InformationNeed(
            need_id="counter",
            source_scores={"counter": 1.0},
            arguments={"key": "value"},
            confidence=1.0,
            freshness=FreshnessDemand.ALWAYS,
            target_cells=(ObjectRef.cell(self.cell_id),),
        )
        percept = context.percepts[0] if context.percepts else None
        if percept is not None and percept.observation.version == "v2":
            return ModelUpdate(
                thought=context.thought.advance(context.thought.cells),
                display=context.display.advance((int(percept.observation.value),)),
                needs=(need,),
                converged=True,
            )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
        )


async def test_runtime_evaluation_tracks_latency_freshness_and_display() -> None:
    sources = SourceRegistry()
    sources.register(StaticMappingSource("docs", {"latency_ms": 37}, delay_s=0.006))
    field, cell_id = CognitiveField.empty(capacity=2, width=4).allocate()
    runtime = CIDRuntime(
        sources,
        RuntimeConfig(max_steps=12, idle_yield_s=0.002),
    )
    result = await runtime.run(
        EvaluationPolicy(cell_id),
        thought=field,
        display=DisplayCanvas.masked(length=1, mask_token_id=-1),
    )
    example = TrajectoryExample(
        example_id="eval-1",
        prompt="Return the documented latency.",
        target_display="37",
        source_descriptors=(
            {
                "name": "docs",
                "description": "documentation",
                "arguments": ({"name": "key", "required": True},),
            },
        ),
        events=(
            ExternalEvent(
                source="docs",
                value=37,
                arrival_step=2,
                version="static:latency_ms",
                arguments={"key": "latency_ms"},
            ),
        ),
    )

    evaluation = evaluate_runtime_result(result, example, expected_display_ids=(37,))

    assert evaluation.converged
    assert evaluation.exact_display is True
    assert evaluation.observation_coverage == 1.0
    assert evaluation.fresh_observations == 1
    assert evaluation.stale_observations == 0
    assert evaluation.missing_observations == 0
    assert evaluation.interaction.mean_latent_to_executable_steps == 1.0
    assert evaluation.interaction.mean_binding_to_observation_steps >= 0.0
    assert evaluation.interaction.mean_observation_to_projection_steps >= 0.0

    stale_example = TrajectoryExample(
        example_id="eval-stale",
        prompt=example.prompt,
        target_display="38",
        source_descriptors=example.source_descriptors,
        events=(
            ExternalEvent(
                source="docs",
                value=38,
                arrival_step=2,
                version="newer:latency_ms",
                arguments={"key": "latency_ms"},
            ),
        ),
    )
    stale = evaluate_runtime_result(result, stale_example)
    assert stale.fresh_observations == 0
    assert stale.stale_observations == 1
    assert stale.stale_observation_rate == 1.0

    summary = summarize_evaluations((evaluation, stale))
    assert summary.tasks == 2
    assert summary.convergence_rate == 1.0
    assert summary.exact_display_tasks == 1
    assert summary.exact_display_accuracy == 1.0
    assert summary.observation_coverage == 1.0
    assert summary.stale_observation_rate == 0.5
    assert summary.external_refreshes == 2
    assert summary.tool_calls_completed == 2
    assert summary.total_runtime_wall_time_s > 0
    assert 0 < summary.tool_wait_ratio <= 1
    assert 0 < summary.latency_hidden_ratio <= 1
    assert summary.tool_latency_p95_s > 0
    assert summary.peak_tool_concurrency == 1


async def test_replay_runner_delivers_dataset_events_on_exact_runtime_steps() -> None:
    example = TrajectoryExample(
        example_id="replay-dynamic",
        prompt="Return the latest counter value.",
        target_display="2",
        source_descriptors=(
            {
                "name": "counter",
                "description": "versioned counter",
                "arguments": ({"name": "key", "required": True},),
                "cacheable": False,
                "dynamic": True,
                "versioned": True,
            },
        ),
        events=(
            ExternalEvent(
                source="counter",
                value=1,
                arrival_step=2,
                version="v1",
                arguments={"key": "value"},
            ),
            ExternalEvent(
                source="counter",
                value=2,
                arrival_step=4,
                version="v2",
                arguments={"key": "value"},
            ),
        ),
    )
    field, cell_id = CognitiveField.empty(capacity=2, width=4).allocate()

    replay = await run_replay_case(
        ReplayPolicy(cell_id),
        example,
        thought=field,
        display=DisplayCanvas.masked(length=1, mask_token_id=-1),
        expected_display_ids=(2,),
        runtime_config=RuntimeConfig(max_steps=8),
    )

    observation_runtime_steps = [
        event.payload["runtime_step"]
        for event in replay.runtime.trace.events
        if event.kind == "binding_observation_updated"
    ]
    assert observation_runtime_steps == [2, 4]
    assert replay.evaluation.converged
    assert replay.evaluation.exact_display is True
    assert replay.evaluation.fresh_observations == 1
    assert replay.evaluation.stale_observations == 0
    assert replay.evaluation.interaction.mean_observation_to_projection_steps == 0.0

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cid.contracts import ArgumentDescriptor, CIDPolicy, Observation, SourceDescriptor
from cid.data import ExternalEvent, TrajectoryExample
from cid.metrics import InteractionMetrics, summarize_runtime
from cid.runtime.bindings import canonical_work_key
from cid.runtime.engine import CIDRuntime, RuntimeConfig, RuntimeResult
from cid.runtime.sources import SourceRegistry
from cid.state import CognitiveField, DisplayCanvas, FactItem


@dataclass(frozen=True, slots=True)
class RuntimeTaskEvaluation:
    example_id: str
    converged: bool
    exact_display: bool | None
    unresolved_display_tokens: int
    expected_observations: int
    observed_work_items: int
    fresh_observations: int
    stale_observations: int
    missing_observations: int
    observation_coverage: float
    stale_observation_rate: float
    interaction: InteractionMetrics


@dataclass(frozen=True, slots=True)
class RuntimeEvaluationSummary:
    tasks: int
    convergence_rate: float
    exact_display_tasks: int
    exact_display_accuracy: float
    observation_coverage: float
    stale_observation_rate: float
    mean_latent_to_executable_steps: float
    mean_binding_to_observation_steps: float
    mean_observation_to_projection_steps: float
    external_refreshes: int
    cache_hits: int
    deduplicated_jobs: int
    cognitive_projections: int
    total_runtime_wall_time_s: float
    mean_runtime_wall_time_s: float
    total_model_compute_s: float
    total_tool_wait_s: float
    tool_wait_ratio: float
    model_tool_overlap_s: float
    model_tool_overlap_ratio: float
    latency_hidden_ratio: float
    tool_calls_completed: int
    tool_latency_mean_s: float
    tool_latency_p50_s: float
    tool_latency_p95_s: float
    tool_latency_max_s: float
    mean_tool_concurrency: float
    peak_tool_concurrency: int
    mean_ready_to_bind_s: float
    ready_to_bind_p95_s: float


@dataclass(frozen=True, slots=True)
class ReplayEvaluationResult:
    runtime: RuntimeResult
    evaluation: RuntimeTaskEvaluation


class ScheduledReplaySource:
    """Read-only dataset source whose versions become visible on exact CID runtime steps."""

    def __init__(
        self,
        descriptor: SourceDescriptor,
        events: tuple[ExternalEvent, ...],
    ) -> None:
        self.descriptor = descriptor
        grouped: dict[str, list[ExternalEvent]] = defaultdict(list)
        for event in events:
            if event.source != descriptor.name:
                raise ValueError("replay source event name does not match its descriptor")
            grouped[canonical_work_key(event.source, event.arguments)].append(event)
        self._events = {
            work_key: tuple(sorted(items, key=lambda item: item.arrival_step))
            for work_key, items in grouped.items()
        }
        self._last_index: dict[str, int] = {}
        self._step = -1
        self._wake = asyncio.Event()

    def on_runtime_step(self, step: int) -> None:
        if step < self._step:
            raise ValueError("replay source runtime step cannot move backwards")
        self._step = step
        wake = self._wake
        self._wake = asyncio.Event()
        wake.set()

    def next_runtime_step(self) -> int | None:
        candidates = tuple(
            event.arrival_step
            for events in self._events.values()
            for event in events
            if event.arrival_step > self._step
        )
        return min(candidates) if candidates else None

    def current_runtime_version(self, arguments: Mapping[str, Any]) -> str | None:
        work_key = canonical_work_key(self.descriptor.name, arguments)
        due = tuple(
            event for event in self._events.get(work_key, ()) if event.arrival_step <= self._step
        )
        return due[-1].version if due else None

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        work_key = canonical_work_key(self.descriptor.name, arguments)
        events = self._events.get(work_key, ())
        while True:
            last_index = self._last_index.get(work_key, -1)
            due = tuple(
                (index, event)
                for index, event in enumerate(events)
                if index > last_index and event.arrival_step <= self._step
            )
            if due:
                index, event = due[-1]
                self._last_index[work_key] = index
                return Observation(
                    value=event.value,
                    version=event.version,
                    provenance=event.provenance or f"replay:{self.descriptor.name}",
                    observed_at=time.monotonic(),
                )
            wake = self._wake
            await wake.wait()


def build_replay_registry(example: TrajectoryExample) -> SourceRegistry:
    registry = SourceRegistry()
    events_by_source: dict[str, list[ExternalEvent]] = defaultdict(list)
    for event in example.events:
        events_by_source[event.source].append(event)
    for raw in example.source_descriptors:
        descriptor = _source_descriptor(raw)
        registry.register(
            ScheduledReplaySource(
                descriptor,
                tuple(events_by_source.get(descriptor.name, ())),
            )
        )
    return registry


async def run_replay_case(
    policy: CIDPolicy,
    example: TrajectoryExample,
    *,
    thought: CognitiveField,
    display: DisplayCanvas,
    expected_display_ids: tuple[int, ...] | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> ReplayEvaluationResult:
    max_event_step = max((event.arrival_step for event in example.events), default=0)
    config = runtime_config or RuntimeConfig(max_steps=max(16, max_event_step + 8))
    runtime = CIDRuntime(build_replay_registry(example), config)
    facts = tuple(
        FactItem(
            key=str(key),
            value=value,
            source_type="dataset",
            timestamp=0.0,
            provenance=example.example_id,
        )
        for key, value in example.protected_facts.items()
    )
    result = await runtime.run(
        policy,
        thought=thought,
        display=display,
        facts=facts,
        prompt=example.prompt,
    )
    return ReplayEvaluationResult(
        runtime=result,
        evaluation=evaluate_runtime_result(
            result,
            example,
            expected_display_ids=expected_display_ids,
        ),
    )


def summarize_evaluations(
    evaluations: tuple[RuntimeTaskEvaluation, ...],
) -> RuntimeEvaluationSummary:
    if not evaluations:
        raise ValueError("runtime evaluation summary requires at least one task")
    display_scored = tuple(item for item in evaluations if item.exact_display is not None)
    expected_observations = sum(item.expected_observations for item in evaluations)
    observed_work_items = sum(item.observed_work_items for item in evaluations)
    stale_observations = sum(item.stale_observations for item in evaluations)
    interactions = tuple(item.interaction for item in evaluations)
    total_runtime = sum(item.runtime_wall_time_s for item in interactions)
    total_tool_wait = sum(item.tool_wait_s for item in interactions)
    total_model_compute = sum(item.model_compute_s for item in interactions)
    total_overlap = sum(item.model_tool_overlap_s for item in interactions)
    tool_latencies = tuple(
        latency for item in interactions for latency in item.tool_latencies_s
    )
    ready_to_bind = tuple(
        delay for item in interactions for delay in item.ready_to_bind_delays_s
    )
    total_tool_service = sum(
        item.mean_tool_concurrency * item.runtime_wall_time_s for item in interactions
    )
    return RuntimeEvaluationSummary(
        tasks=len(evaluations),
        convergence_rate=_fraction(sum(item.converged for item in evaluations), len(evaluations)),
        exact_display_tasks=len(display_scored),
        exact_display_accuracy=(
            _fraction(
                sum(item.exact_display is True for item in display_scored),
                len(display_scored),
            )
            if display_scored
            else 0.0
        ),
        observation_coverage=(
            _fraction(observed_work_items, expected_observations)
            if expected_observations
            else 1.0
        ),
        stale_observation_rate=(
            _fraction(stale_observations, observed_work_items) if observed_work_items else 0.0
        ),
        mean_latent_to_executable_steps=_mean_metric(
            evaluations, "mean_latent_to_executable_steps"
        ),
        mean_binding_to_observation_steps=_mean_metric(
            evaluations, "mean_binding_to_observation_steps"
        ),
        mean_observation_to_projection_steps=_mean_metric(
            evaluations, "mean_observation_to_projection_steps"
        ),
        external_refreshes=sum(item.external_refreshes for item in interactions),
        cache_hits=sum(item.cache_hits for item in interactions),
        deduplicated_jobs=sum(item.deduplicated_jobs for item in interactions),
        cognitive_projections=sum(item.cognitive_projections for item in interactions),
        total_runtime_wall_time_s=total_runtime,
        mean_runtime_wall_time_s=total_runtime / len(evaluations),
        total_model_compute_s=total_model_compute,
        total_tool_wait_s=total_tool_wait,
        tool_wait_ratio=_float_fraction(total_tool_wait, total_runtime),
        model_tool_overlap_s=total_overlap,
        model_tool_overlap_ratio=_float_fraction(total_overlap, total_runtime),
        latency_hidden_ratio=_float_fraction(total_overlap, total_tool_wait),
        tool_calls_completed=len(tool_latencies),
        tool_latency_mean_s=_mean_float(tool_latencies),
        tool_latency_p50_s=_percentile(tool_latencies, 0.50),
        tool_latency_p95_s=_percentile(tool_latencies, 0.95),
        tool_latency_max_s=max(tool_latencies, default=0.0),
        mean_tool_concurrency=_float_fraction(total_tool_service, total_runtime),
        peak_tool_concurrency=max(
            (item.peak_tool_concurrency for item in interactions), default=0
        ),
        mean_ready_to_bind_s=_mean_float(ready_to_bind),
        ready_to_bind_p95_s=_percentile(ready_to_bind, 0.95),
    )


def evaluate_runtime_result(
    result: RuntimeResult,
    example: TrajectoryExample,
    *,
    expected_display_ids: tuple[int, ...] | None = None,
) -> RuntimeTaskEvaluation:
    expected = _latest_expected_observations(example)
    observed = {
        binding.work_key: binding.observation
        for binding in result.bindings
        if binding.observation is not None
    }

    fresh = 0
    stale = 0
    missing = 0
    for work_key, event in expected.items():
        observation = observed.get(work_key)
        if observation is None:
            missing += 1
        elif _observation_matches(
            observation.value,
            observation.version,
            event.value,
            event.version,
        ):
            fresh += 1
        else:
            stale += 1

    expected_count = len(expected)
    observed_count = fresh + stale
    exact_display = (
        None
        if expected_display_ids is None
        else result.display.visible_token_ids == expected_display_ids
    )
    return RuntimeTaskEvaluation(
        example_id=example.example_id,
        converged=result.converged,
        exact_display=exact_display,
        unresolved_display_tokens=result.display.unresolved,
        expected_observations=expected_count,
        observed_work_items=observed_count,
        fresh_observations=fresh,
        stale_observations=stale,
        missing_observations=missing,
        observation_coverage=observed_count / expected_count if expected_count else 1.0,
        stale_observation_rate=stale / observed_count if observed_count else 0.0,
        interaction=summarize_runtime(result),
    )


def _latest_expected_observations(example: TrajectoryExample) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for event in sorted(example.events, key=lambda item: item.arrival_step):
        latest[canonical_work_key(event.source, event.arguments)] = event
    return latest


def _observation_matches(
    observed_value: Any,
    observed_version: str | None,
    expected_value: Any,
    expected_version: str | None,
) -> bool:
    if expected_version is not None and observed_version != expected_version:
        return False
    return observed_value == expected_value


def _source_descriptor(raw: Mapping[str, Any]) -> SourceDescriptor:
    return SourceDescriptor(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        arguments=tuple(
            ArgumentDescriptor(
                name=str(argument["name"]),
                kind=str(argument.get("kind", "any")),
                description=str(argument.get("description", "")),
                required=bool(argument.get("required", True)),
            )
            for argument in raw.get("arguments", ())
        ),
        cacheable=bool(raw.get("cacheable", True)),
        dynamic=bool(raw.get("dynamic", False)),
        streamable=bool(raw.get("streamable", False)),
        versioned=bool(raw.get("versioned", False)),
        accepts_partial_arguments=bool(raw.get("accepts_partial_arguments", False)),
    )


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_metric(
    evaluations: tuple[RuntimeTaskEvaluation, ...],
    name: str,
) -> float:
    values = [float(getattr(item.interaction, name)) for item in evaluations]
    return sum(values) / len(values)


def _float_fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _mean_float(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

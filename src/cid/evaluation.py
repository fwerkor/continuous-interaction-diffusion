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
        else tuple(result.display.token_ids) == expected_display_ids
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
    )


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_metric(
    evaluations: tuple[RuntimeTaskEvaluation, ...],
    name: str,
) -> float:
    values = [float(getattr(item.interaction, name)) for item in evaluations]
    return sum(values) / len(values)

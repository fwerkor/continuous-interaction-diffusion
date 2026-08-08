from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cid.data import TrajectoryExample
from cid.metrics import InteractionMetrics, summarize_runtime
from cid.runtime.bindings import canonical_work_key
from cid.runtime.engine import RuntimeResult


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


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_metric(
    evaluations: tuple[RuntimeTaskEvaluation, ...],
    name: str,
) -> float:
    values = [float(getattr(item.interaction, name)) for item in evaluations]
    return sum(values) / len(values)

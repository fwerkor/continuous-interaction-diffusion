from __future__ import annotations

import math
from dataclasses import dataclass

from cid.runtime.engine import RuntimeResult
from cid.runtime.trace import TraceEvent


@dataclass(frozen=True, slots=True)
class InteractionMetrics:
    external_refreshes: int
    cognitive_projections: int
    cache_hits: int
    deduplicated_jobs: int
    mean_intent_lead_steps: float
    mean_latent_to_executable_steps: float
    mean_binding_to_observation_steps: float
    mean_observation_to_projection_steps: float
    model_steps_during_io: int
    tool_wait_overlap_s: float
    runtime_wall_time_s: float
    model_compute_s: float
    tool_wait_s: float
    tool_wait_ratio: float
    model_tool_overlap_s: float
    model_tool_overlap_ratio: float
    latency_hidden_ratio: float
    tool_calls_completed: int
    tool_latencies_s: tuple[float, ...]
    tool_latency_mean_s: float
    tool_latency_p50_s: float
    tool_latency_p95_s: float
    tool_latency_max_s: float
    mean_tool_concurrency: float
    peak_tool_concurrency: int
    ready_to_bind_delays_s: tuple[float, ...]
    mean_ready_to_bind_s: float
    ready_to_bind_p95_s: float
    reclaimed_cells: int
    cognitive_compactions: int


def summarize_runtime(result: RuntimeResult) -> InteractionMetrics:
    events = result.trace.events
    runtime_interval = _runtime_interval(events)
    model = _merge_intervals(_model_intervals(events))
    pending_tool = _tool_pending_intervals(events, runtime_interval)
    tool_pending = _merge_intervals(pending_tool)
    completed_tool = _completed_tool_intervals(events)
    tool_latencies = tuple(end - start for start, end in completed_tool)
    ready_to_bind = _ready_to_bind_delays(events)

    runtime_wall = _duration(runtime_interval) if runtime_interval is not None else 0.0
    model_compute = _interval_duration(model)
    tool_wait = _interval_duration(tool_pending)
    overlap = _interval_overlap_duration(model, tool_pending)
    total_tool_service = _interval_duration(pending_tool)

    return InteractionMetrics(
        external_refreshes=_count(events, "external_refresh_started"),
        cognitive_projections=_count(events, "cognitive_projection"),
        cache_hits=_count(events, "cache_hit"),
        deduplicated_jobs=_count(events, "job_deduplicated"),
        mean_intent_lead_steps=_mean_intent_lead_steps(events),
        mean_latent_to_executable_steps=_mean_latent_to_executable_steps(events),
        mean_binding_to_observation_steps=_mean_binding_to_observation_steps(events),
        mean_observation_to_projection_steps=_mean_observation_to_projection_steps(events),
        model_steps_during_io=_model_steps_during_io(events, tool_pending),
        tool_wait_overlap_s=overlap,
        runtime_wall_time_s=runtime_wall,
        model_compute_s=model_compute,
        tool_wait_s=tool_wait,
        tool_wait_ratio=_fraction(tool_wait, runtime_wall),
        model_tool_overlap_s=overlap,
        model_tool_overlap_ratio=_fraction(overlap, runtime_wall),
        latency_hidden_ratio=_fraction(overlap, tool_wait),
        tool_calls_completed=len(tool_latencies),
        tool_latencies_s=tool_latencies,
        tool_latency_mean_s=_mean_float(tool_latencies),
        tool_latency_p50_s=_percentile(tool_latencies, 0.50),
        tool_latency_p95_s=_percentile(tool_latencies, 0.95),
        tool_latency_max_s=max(tool_latencies, default=0.0),
        mean_tool_concurrency=_fraction(total_tool_service, runtime_wall),
        peak_tool_concurrency=_peak_concurrency(pending_tool),
        ready_to_bind_delays_s=ready_to_bind,
        mean_ready_to_bind_s=_mean_float(ready_to_bind),
        ready_to_bind_p95_s=_percentile(ready_to_bind, 0.95),
        reclaimed_cells=_count(events, "cell_reclaimed"),
        cognitive_compactions=_count(events, "cognitive_compaction"),
    )


def _count(events: tuple[TraceEvent, ...], kind: str) -> int:
    return sum(event.kind == kind for event in events)


def _mean_intent_lead_steps(events: tuple[TraceEvent, ...]) -> float:
    first_need: dict[str, int] = {}
    first_binding: dict[str, int] = {}
    for event in events:
        need_id = event.payload.get("need_id")
        if not isinstance(need_id, str):
            continue
        if event.kind == "information_need":
            first_need.setdefault(need_id, event.step)
        elif event.kind == "binding_active":
            first_binding.setdefault(need_id, event.step)

    leads = [
        binding_step - first_need[need_id]
        for need_id, binding_step in first_binding.items()
        if need_id in first_need
    ]
    return _mean_float(leads)


def _mean_latent_to_executable_steps(events: tuple[TraceEvent, ...]) -> float:
    first_need: dict[str, int] = {}
    first_executable: dict[str, int] = {}
    for event in events:
        if event.kind != "information_need":
            continue
        need_id = event.payload.get("need_id")
        if not isinstance(need_id, str):
            continue
        first_need.setdefault(need_id, event.step)
        if event.payload.get("executable") is True:
            first_executable.setdefault(need_id, event.step)
    delays = [
        executable_step - first_need[need_id]
        for need_id, executable_step in first_executable.items()
        if need_id in first_need
    ]
    return _mean_float(delays)


def _mean_binding_to_observation_steps(events: tuple[TraceEvent, ...]) -> float:
    first_binding: dict[str, int] = {}
    first_observation: dict[str, int] = {}
    for event in events:
        binding_id = event.payload.get("binding_id")
        if not isinstance(binding_id, str):
            continue
        if event.kind == "binding_active":
            first_binding.setdefault(binding_id, event.step)
        elif event.kind in {"binding_observation_updated", "cache_hit"}:
            first_observation.setdefault(binding_id, event.step)
    delays = [
        observation_step - first_binding[binding_id]
        for binding_id, observation_step in first_observation.items()
        if binding_id in first_binding
    ]
    return _mean_float(delays)


def _mean_observation_to_projection_steps(events: tuple[TraceEvent, ...]) -> float:
    first_observation: dict[str, int] = {}
    first_projection_after_observation: dict[str, int] = {}
    for event in events:
        binding_id = event.payload.get("binding_id")
        if not isinstance(binding_id, str):
            continue
        if event.kind in {"binding_observation_updated", "cache_hit"}:
            first_observation.setdefault(binding_id, event.step)
        elif event.kind == "cognitive_projection" and binding_id in first_observation:
            first_projection_after_observation.setdefault(binding_id, event.step)
    delays = [
        projection_step - first_observation[binding_id]
        for binding_id, projection_step in first_projection_after_observation.items()
    ]
    return _mean_float(delays)


def _runtime_interval(events: tuple[TraceEvent, ...]) -> tuple[float, float] | None:
    start = next((event.timestamp for event in events if event.kind == "trajectory_started"), None)
    end = next(
        (event.timestamp for event in reversed(events) if event.kind == "trajectory_finished"),
        None,
    )
    if start is None or end is None or end < start:
        return None
    return (start, end)


def _model_intervals(events: tuple[TraceEvent, ...]) -> list[tuple[float, float]]:
    starts: dict[int, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        if event.kind == "model_step_started":
            starts[event.step] = event.timestamp
        elif event.kind == "model_step_finished" and event.step in starts:
            intervals.append((starts.pop(event.step), event.timestamp))
    return intervals


def _completed_tool_intervals(events: tuple[TraceEvent, ...]) -> list[tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        work_key = event.payload.get("work_key")
        if not isinstance(work_key, str):
            continue
        if event.kind == "external_refresh_started":
            starts[work_key] = event.timestamp
        elif event.kind == "external_refresh_ready" and work_key in starts:
            intervals.append((starts.pop(work_key), event.timestamp))
    return intervals


def _tool_pending_intervals(
    events: tuple[TraceEvent, ...],
    runtime_interval: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        work_key = event.payload.get("work_key")
        if not isinstance(work_key, str):
            continue
        if event.kind == "external_refresh_started":
            starts[work_key] = event.timestamp
        elif event.kind in {"external_refresh_ready", "external_refresh_cancelled"}:
            start = starts.pop(work_key, None)
            if start is not None:
                intervals.append((start, event.timestamp))
    if runtime_interval is not None:
        end = runtime_interval[1]
        intervals.extend((start, end) for start in starts.values() if start <= end)
    return intervals


def _ready_to_bind_delays(events: tuple[TraceEvent, ...]) -> tuple[float, ...]:
    ready: dict[str, float] = {}
    delays: list[float] = []
    for event in events:
        work_key = event.payload.get("work_key")
        if not isinstance(work_key, str):
            continue
        if event.kind == "external_refresh_ready":
            ready[work_key] = event.timestamp
        elif event.kind == "binding_observation_updated" and work_key in ready:
            delays.append(max(0.0, event.timestamp - ready.pop(work_key)))
    return tuple(delays)


def _model_steps_during_io(
    events: tuple[TraceEvent, ...], tool_pending: list[tuple[float, float]]
) -> int:
    model = _model_intervals(events)
    return sum(
        any(_overlap(interval, pending) > 0 for pending in tool_pending)
        for interval in model
    )


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(_duration(interval) for interval in intervals)


def _interval_overlap_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    return sum(_overlap(a, b) for a in left for b in right)


def _duration(interval: tuple[float, float]) -> float:
    return max(0.0, interval[1] - interval[0])


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _mean_float(values: tuple[float, ...] | list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    points = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
        key=lambda item: (item[0], item[1]),
    )
    current = 0
    peak = 0
    for _, delta in points:
        current += delta
        peak = max(peak, current)
    return peak

from __future__ import annotations

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
    model_steps_during_io: int
    tool_wait_overlap_s: float
    reclaimed_cells: int
    cognitive_compactions: int


def summarize_runtime(result: RuntimeResult) -> InteractionMetrics:
    events = result.trace.events
    return InteractionMetrics(
        external_refreshes=_count(events, "external_refresh_started"),
        cognitive_projections=_count(events, "cognitive_projection"),
        cache_hits=_count(events, "cache_hit"),
        deduplicated_jobs=_count(events, "job_deduplicated"),
        mean_intent_lead_steps=_mean_intent_lead_steps(events),
        model_steps_during_io=_model_steps_during_io(events),
        tool_wait_overlap_s=_tool_wait_overlap(events),
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
    return sum(leads) / len(leads) if leads else 0.0


def _model_intervals(events: tuple[TraceEvent, ...]) -> list[tuple[float, float]]:
    starts: dict[int, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        if event.kind == "model_step_started":
            starts[event.step] = event.timestamp
        elif event.kind == "model_step_finished" and event.step in starts:
            intervals.append((starts.pop(event.step), event.timestamp))
    return intervals


def _io_intervals(events: tuple[TraceEvent, ...]) -> list[tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        work_key = event.payload.get("work_key")
        if not isinstance(work_key, str):
            continue
        if event.kind == "external_refresh_started":
            starts[work_key] = event.timestamp
        elif event.kind == "external_refresh_finished" and work_key in starts:
            intervals.append((starts.pop(work_key), event.timestamp))
    return intervals


def _model_steps_during_io(events: tuple[TraceEvent, ...]) -> int:
    model = _model_intervals(events)
    io = _io_intervals(events)
    return sum(any(_overlap(a, b) > 0 for b in io) for a in model)


def _tool_wait_overlap(events: tuple[TraceEvent, ...]) -> float:
    model = _model_intervals(events)
    io = _merge_intervals(_io_intervals(events))
    return sum(_overlap(a, b) for a in model for b in io)


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


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceEvent:
    kind: str
    step: int
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


class RuntimeTrace:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def emit(self, kind: str, step: int, **payload: Any) -> None:
        self._events.append(
            TraceEvent(kind=kind, step=step, timestamp=time.monotonic(), payload=payload)
        )

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def count(self, kind: str) -> int:
        return sum(event.kind == kind for event in self._events)

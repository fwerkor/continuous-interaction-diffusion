from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cid.contracts import FreshnessDemand, InformationNeed, Observation


class BindingStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    WAITING = "waiting"
    AVAILABLE = "available"
    REFRESHING = "refreshing"
    RETIRED = "retired"


def canonical_work_key(source: str, arguments: Mapping[str, Any]) -> str:
    serialized = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return f"{source}:{serialized}"


@dataclass(slots=True)
class Binding:
    binding_id: str
    need_id: str
    source: str
    arguments: dict[str, Any]
    freshness: FreshnessDemand
    max_age_s: float | None
    target_cells: tuple[int, ...]
    target_display: tuple[int, ...]
    promote_to_fact: bool
    status: BindingStatus = BindingStatus.CANDIDATE
    observation: Observation | None = None
    last_refresh_at: float | None = None
    external_refreshes: int = 0
    cognitive_projections: int = 0
    generation: int = 0

    @property
    def work_key(self) -> str:
        return canonical_work_key(self.source, self.arguments)

    def update_from_need(self, need: InformationNeed) -> None:
        arguments_changed = self.arguments != dict(need.arguments)
        self.arguments = dict(need.arguments)
        self.freshness = need.freshness
        self.max_age_s = need.max_age_s
        self.target_cells = need.target_cells
        self.target_display = need.target_display
        self.promote_to_fact = need.promote_to_fact
        self.generation += 1
        if arguments_changed:
            self.observation = None
            self.last_refresh_at = None
            self.status = BindingStatus.ACTIVE
        if self.status is BindingStatus.RETIRED:
            self.status = BindingStatus.ACTIVE


class BindingTable:
    def __init__(self) -> None:
        self._by_need: dict[str, Binding] = {}
        self._history: list[Binding] = []
        self._serial = 0

    def all(self) -> tuple[Binding, ...]:
        return (*self._history, *self._by_need.values())

    def active(self) -> tuple[Binding, ...]:
        return tuple(b for b in self._by_need.values() if b.status is not BindingStatus.RETIRED)

    def reconcile(
        self,
        needs: tuple[InformationNeed, ...],
        *,
        binding_threshold: float,
        source_descriptors: Mapping[str, tuple[str, ...]],
    ) -> tuple[Binding, ...]:
        seen: set[str] = set()
        touched: list[Binding] = []

        for need in needs:
            if need.confidence < binding_threshold:
                continue
            source = need.selected_source()
            if source is None or source not in source_descriptors:
                continue
            required = source_descriptors[source]
            if any(name not in need.arguments for name in required):
                continue

            seen.add(need.need_id)
            binding = self._by_need.get(need.need_id)
            if binding is None or binding.source != source:
                if binding is not None:
                    binding.status = BindingStatus.RETIRED
                    self._history.append(binding)
                self._serial += 1
                binding = Binding(
                    binding_id=f"b{self._serial}",
                    need_id=need.need_id,
                    source=source,
                    arguments=dict(need.arguments),
                    freshness=need.freshness,
                    max_age_s=need.max_age_s,
                    target_cells=need.target_cells,
                    target_display=need.target_display,
                    promote_to_fact=need.promote_to_fact,
                    status=BindingStatus.ACTIVE,
                )
                self._by_need[need.need_id] = binding
            else:
                binding.update_from_need(need)
                if binding.status is BindingStatus.CANDIDATE:
                    binding.status = BindingStatus.ACTIVE
            touched.append(binding)

        for need_id, binding in self._by_need.items():
            if need_id not in seen and binding.status is not BindingStatus.RETIRED:
                binding.status = BindingStatus.RETIRED

        return tuple(touched)

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from cid.contracts import ArgumentDescriptor, Observation, SourceDescriptor


class ReadOnlySource(Protocol):
    descriptor: SourceDescriptor

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        ...


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, ReadOnlySource] = {}

    def register(self, source: ReadOnlySource) -> None:
        name = source.descriptor.name
        if name in self._sources:
            raise ValueError(f"source already registered: {name}")
        self._sources[name] = source

    def get(self, name: str) -> ReadOnlySource:
        try:
            return self._sources[name]
        except KeyError as exc:
            raise KeyError(f"unknown source: {name}") from exc

    def descriptors(self) -> tuple[SourceDescriptor, ...]:
        return tuple(source.descriptor for source in self._sources.values())


@dataclass(slots=True)
class StaticMappingSource:
    name: str
    values: Mapping[str, Any]
    delay_s: float = 0.0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name=self.name,
            description="read a value from a static mapping",
            arguments=(
                ArgumentDescriptor(
                    name="key",
                    kind="string",
                    description="mapping key to read",
                ),
            ),
            cacheable=True,
            dynamic=False,
            versioned=True,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        key = str(arguments["key"])
        value = self.values[key]
        return Observation(
            value=value,
            version=f"static:{key}",
            provenance=f"{self.name}:{key}",
            observed_at=time.monotonic(),
        )


class VersionedMemorySource:
    def __init__(self, name: str, initial: Any) -> None:
        self._name = name
        self._value = initial
        self._version = 0

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name=self._name,
            description="read a mutable in-memory value",
            cacheable=False,
            dynamic=True,
            versioned=True,
        )

    def set(self, value: Any) -> None:
        self._value = value
        self._version += 1

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        return Observation(
            value=self._value,
            version=str(self._version),
            provenance=self._name,
            observed_at=time.monotonic(),
        )


@dataclass(slots=True)
class ClockSource:
    name: str = "clock"

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name=self.name,
            description="read the runtime monotonic clock",
            cacheable=False,
            dynamic=True,
            versioned=False,
        )

    async def read(self, arguments: Mapping[str, Any]) -> Observation:
        del arguments
        now = time.monotonic()
        return Observation(value=now, provenance=self.name, observed_at=now)

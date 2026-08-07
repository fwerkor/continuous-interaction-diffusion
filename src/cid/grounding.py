from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cid.state import CognitiveField


class ObjectKind(StrEnum):
    CELL = "cell"
    FACT = "fact"
    BINDING = "binding"
    SOURCE = "source"
    DISPLAY_SPAN = "display_span"
    ANCHOR = "anchor"
    SYMBOLIC_OBJECT = "symbolic_object"


class AnchorKind(StrEnum):
    ENTITY = "entity"
    NUMBER = "number"
    SYMBOL = "symbol"
    SPAN = "span"
    PATH = "path"
    URL = "url"
    TEXT = "text"


class LinkRelation(StrEnum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    REQUESTS = "requests"
    OBSERVES = "observes"
    CONSTRAINS = "constrains"
    REFERS_TO = "refers_to"


STRONG_LINK_RELATIONS = frozenset(
    {
        LinkRelation.DEPENDS_ON,
        LinkRelation.REQUESTS,
        LinkRelation.OBSERVES,
        LinkRelation.CONSTRAINS,
    }
)


@dataclass(frozen=True, slots=True)
class ObjectRef:
    kind: ObjectKind
    identifier: str
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("object reference identifier must be non-empty")
        if self.kind is ObjectKind.DISPLAY_SPAN:
            if self.span is None:
                raise ValueError("display-span references require a span")
        elif self.span is not None:
            raise ValueError("only display-span references may carry a span")
        if self.span is not None:
            start, end = self.span
            if start < 0 or end <= start:
                raise ValueError("object reference span must satisfy 0 <= start < end")

    @classmethod
    def cell(cls, cell_id: str) -> ObjectRef:
        return cls(kind=ObjectKind.CELL, identifier=cell_id)

    @classmethod
    def binding(cls, binding_id: str) -> ObjectRef:
        return cls(kind=ObjectKind.BINDING, identifier=binding_id)

    @classmethod
    def fact(cls, fact_key: str) -> ObjectRef:
        return cls(kind=ObjectKind.FACT, identifier=fact_key)

    @classmethod
    def source(cls, source_name: str) -> ObjectRef:
        return cls(kind=ObjectKind.SOURCE, identifier=source_name)

    @classmethod
    def anchor(cls, anchor_id: str) -> ObjectRef:
        return cls(kind=ObjectKind.ANCHOR, identifier=anchor_id)

    @classmethod
    def display_span(cls, start: int, end: int) -> ObjectRef:
        return cls(kind=ObjectKind.DISPLAY_SPAN, identifier="display", span=(start, end))

    @classmethod
    def symbolic(cls, object_id: str) -> ObjectRef:
        return cls(kind=ObjectKind.SYMBOLIC_OBJECT, identifier=object_id)

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {"kind": self.kind.value, "identifier": self.identifier}
        if self.span is not None:
            raw["span"] = list(self.span)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ObjectRef:
        span_raw = raw.get("span")
        return cls(
            kind=ObjectKind(str(raw["kind"])),
            identifier=str(raw["identifier"]),
            span=None if span_raw is None else (int(span_raw[0]), int(span_raw[1])),
        )


AnchorValue = str | int | float


@dataclass(frozen=True, slots=True)
class Anchor:
    anchor_id: str
    kind: AnchorKind
    value: AnchorValue
    object_id: str | None = None
    confidence: float = 1.0
    unit: str | None = None
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("anchor_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("anchor confidence must be in [0, 1]")
        if self.kind is AnchorKind.NUMBER:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("number anchors require an int or float value")
        elif not isinstance(self.value, str) or not self.value:
            raise ValueError("non-number anchors require a non-empty string value")
        if self.kind is AnchorKind.SPAN:
            if self.span is None:
                raise ValueError("span anchors require a span")
        elif self.span is not None:
            raise ValueError("only span anchors may carry a span")
        if self.span is not None:
            start, end = self.span
            if start < 0 or end <= start:
                raise ValueError("anchor span must satisfy 0 <= start < end")
        if self.unit is not None and self.kind is not AnchorKind.NUMBER:
            raise ValueError("only number anchors may carry a unit")

    @property
    def canonical_key(self) -> str:
        if self.object_id:
            return self.object_id
        if self.kind is AnchorKind.NUMBER:
            unit = self.unit or ""
            return f"number:{self.value}:{unit}"
        return f"{self.kind.value}:{self.value}"

    @property
    def ref(self) -> ObjectRef:
        return ObjectRef.anchor(self.anchor_id)

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "anchor_id": self.anchor_id,
            "kind": self.kind.value,
            "value": self.value,
            "confidence": self.confidence,
        }
        if self.object_id is not None:
            raw["object_id"] = self.object_id
        if self.unit is not None:
            raw["unit"] = self.unit
        if self.span is not None:
            raw["span"] = list(self.span)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Anchor:
        span_raw = raw.get("span")
        return cls(
            anchor_id=str(raw["anchor_id"]),
            kind=AnchorKind(str(raw["kind"])),
            value=raw["value"],
            object_id=None if raw.get("object_id") is None else str(raw["object_id"]),
            confidence=float(raw.get("confidence", 1.0)),
            unit=None if raw.get("unit") is None else str(raw["unit"]),
            span=None if span_raw is None else (int(span_raw[0]), int(span_raw[1])),
        )


@dataclass(frozen=True, slots=True)
class CognitiveLink:
    relation: LinkRelation
    target: ObjectRef
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("link confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.value,
            "target": self.target.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CognitiveLink:
        return cls(
            relation=LinkRelation(str(raw["relation"])),
            target=ObjectRef.from_dict(raw["target"]),
            confidence=float(raw.get("confidence", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class GroundingEntry:
    anchor: Anchor
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError("grounding aliases must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"anchor": self.anchor.to_dict(), "aliases": list(self.aliases)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GroundingEntry:
        return cls(
            anchor=Anchor.from_dict(raw["anchor"]),
            aliases=tuple(str(alias) for alias in raw.get("aliases", ())),
        )


class ClosedWorldGrounder:
    """Deterministic resolver for synthetic and oracle-grounded CID trajectories."""

    def __init__(self, entries: tuple[GroundingEntry, ...]) -> None:
        self._by_alias: dict[tuple[AnchorKind, str], Anchor] = {}
        for entry in entries:
            anchor = entry.anchor
            mentions = (str(anchor.value), *entry.aliases)
            for mention in mentions:
                key = (anchor.kind, self._normalize(mention))
                existing = self._by_alias.get(key)
                if existing is not None and existing.canonical_key != anchor.canonical_key:
                    raise ValueError(f"ambiguous closed-world grounding alias: {mention!r}")
                self._by_alias[key] = anchor

    def resolve(self, kind: AnchorKind, mention: str) -> Anchor | None:
        return self._by_alias.get((kind, self._normalize(mention)))

    def route(self, anchor: Anchor, field: CognitiveField) -> tuple[str, ...]:
        key = anchor.canonical_key
        return tuple(
            cell.cell_id
            for cell in field.cells
            if cell.live
            and cell.cell_id is not None
            and any(candidate.canonical_key == key for candidate in cell.anchors)
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().strip().split())

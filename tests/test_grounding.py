from __future__ import annotations

import pytest

from cid.grounding import (
    Anchor,
    AnchorKind,
    ClosedWorldGrounder,
    CognitiveLink,
    GroundingEntry,
    LinkRelation,
    ObjectKind,
    ObjectRef,
)
from cid.state import CognitiveField


def test_typed_grounding_objects_validate_their_payloads() -> None:
    entity = Anchor(
        anchor_id="a:model-a",
        kind=AnchorKind.ENTITY,
        value="Model A",
        object_id="model:a",
        confidence=0.9,
    )
    number = Anchor(
        anchor_id="a:latency",
        kind=AnchorKind.NUMBER,
        value=37.0,
        unit="ms",
    )
    span = Anchor(
        anchor_id="a:span",
        kind=AnchorKind.SPAN,
        value="Model A",
        span=(4, 11),
    )

    assert entity.canonical_key == "model:a"
    assert number.canonical_key == "number:37.0:ms"
    assert span.span == (4, 11)

    with pytest.raises(ValueError):
        ObjectRef(kind=ObjectKind.DISPLAY_SPAN, identifier="display")
    with pytest.raises(ValueError):
        Anchor(anchor_id="bad", kind=AnchorKind.NUMBER, value="37")


def test_closed_world_grounder_resolves_aliases_and_routes_to_cells() -> None:
    anchor = Anchor(
        anchor_id="a:model-a",
        kind=AnchorKind.ENTITY,
        value="Model A",
        object_id="model:a",
    )
    grounder = ClosedWorldGrounder(
        (GroundingEntry(anchor=anchor, aliases=("model a", "model-a")),)
    )
    field = CognitiveField.empty(capacity=3, width=2)
    field, cell_id = field.allocate(anchors=(anchor,))

    resolved = grounder.resolve(AnchorKind.ENTITY, "MODEL-A")

    assert resolved == anchor
    assert grounder.route(anchor, field) == (cell_id,)


def test_typed_cognitive_link_keeps_relation_and_target_kind() -> None:
    link = CognitiveLink(
        relation=LinkRelation.DEPENDS_ON,
        target=ObjectRef(kind=ObjectKind.BINDING, identifier="b7"),
        confidence=0.8,
    )

    assert link.target.kind is ObjectKind.BINDING
    assert link.target.identifier == "b7"

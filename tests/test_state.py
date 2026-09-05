import time

import pytest

from cid.grounding import LinkRelation, ObjectKind
from cid.state import (
    CellLifecycle,
    CognitiveField,
    CognitiveRole,
    DisplayCanvas,
    FactItem,
    FactStore,
)


def test_fact_snapshot_is_read_only() -> None:
    store = FactStore()
    store.publish(
        FactItem(
            key="exact",
            value="37 ms",
            source_type="docs",
            timestamp=time.monotonic(),
        )
    )
    snapshot = store.snapshot()

    with pytest.raises(TypeError):
        snapshot.items["exact"] = snapshot.items["exact"]  # type: ignore[index]

    assert store.snapshot().items["exact"].value == "37 ms"


def test_fact_snapshot_cannot_mutate_nested_runtime_value() -> None:
    store = FactStore()
    store.publish(
        FactItem(
            key="structured",
            value={"numbers": [37]},
            source_type="docs",
            timestamp=time.monotonic(),
        )
    )

    snapshot = store.snapshot()
    snapshot.items["structured"].value["numbers"].append(99)

    assert store.snapshot().items["structured"].value == {"numbers": [37]}


def test_cognitive_field_has_fixed_capacity_but_dynamic_occupancy() -> None:
    field = CognitiveField.empty(capacity=4, width=3)
    assert field.capacity == 4
    assert field.occupied_count == 0
    assert field.occupied_mask == (False, False, False, False)

    field, plan_id = field.allocate(roles={CognitiveRole.PLAN: 1.0})
    field, need_id = field.allocate(roles={CognitiveRole.INFORMATION_NEED: 1.0})

    assert field.capacity == 4
    assert field.live_cell_ids == (plan_id, need_id)
    assert field.occupied_mask == (True, True, False, False)
    assert field.get(plan_id).lifecycle is CellLifecycle.ACTIVE


def test_allocation_can_materialize_a_model_selected_physical_slot() -> None:
    field = CognitiveField.empty(capacity=4, width=2)
    field, cell_id = field.allocate(slot=3, semantic=(0.25, 0.75))

    assert field.slot_of(cell_id) == 3
    assert field.occupied_mask == (False, False, False, True)


def test_stable_cell_id_survives_physical_compaction() -> None:
    field = CognitiveField.empty(capacity=5, width=2)
    field, first = field.allocate(semantic=(1.0, 0.0))
    field, second = field.allocate(semantic=(0.0, 1.0))
    field, third = field.allocate(semantic=(1.0, 1.0))
    field = field.retire(second).reclaim(second)

    assert field.slot_of(third) == 2
    compacted = field.compact()

    assert compacted.slot_of(first) == 0
    assert compacted.slot_of(third) == 1
    assert compacted.get(third).semantic == (1.0, 1.0)


def test_retired_cell_is_distinct_from_empty_until_reclaimed() -> None:
    field = CognitiveField.empty(capacity=2, width=2)
    field, cell_id = field.allocate()
    retired = field.retire(cell_id)

    assert retired.get(cell_id).lifecycle is CellLifecycle.RETIRED
    assert retired.occupied_count == 1
    assert retired.live_count == 0
    assert retired.empty_count == 1

    reclaimed = retired.reclaim(cell_id)
    assert reclaimed.occupied_count == 0
    with pytest.raises(KeyError):
        reclaimed.get(cell_id)


def test_split_and_merge_preserve_logical_lineage() -> None:
    field = CognitiveField.empty(capacity=6, width=2)
    field, parent = field.allocate(semantic=(1.0, 0.0))
    field, children = field.split(parent, ((0.7, 0.3), (0.2, 0.8)))

    assert field.get(parent).lifecycle is CellLifecycle.RETIRED
    for child in children:
        (link,) = field.get(child).links
        assert link.relation is LinkRelation.DERIVED_FROM
        assert link.target.kind is ObjectKind.CELL
        assert link.target.identifier == parent

    field, merged = field.merge(children, semantic=(0.5, 0.5))
    assert tuple(link.target.identifier for link in field.get(merged).links) == children
    assert all(
        link.relation is LinkRelation.DERIVED_FROM for link in field.get(merged).links
    )
    assert all(field.get(child).lifecycle is CellLifecycle.RETIRED for child in children)


def test_display_canvas_uses_eos_for_visible_length_and_unresolved_span() -> None:
    canvas = DisplayCanvas.masked(length=6, mask_token_id=5, eos_token_id=2)
    updated = canvas.advance((11, 12, 2, 5, 5, 5))

    assert updated.visible_token_ids == (11, 12)
    assert updated.realized_length == 2
    assert updated.active_span_length == 3
    assert updated.unresolved == 0
    assert len(updated.token_ids) == 6


def test_unresolved_display_bootstrap_has_mask_and_eos_boundary() -> None:
    canvas = DisplayCanvas.initial_unresolved(length=6, mask_token_id=5, eos_token_id=2)

    assert canvas.token_ids == (5, 2, 5, 5, 5, 5)
    assert canvas.visible_token_ids == (5,)
    assert canvas.realized_length == 1
    assert canvas.active_span_length == 2
    assert canvas.unresolved == 1

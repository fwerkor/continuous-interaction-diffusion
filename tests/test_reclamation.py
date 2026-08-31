from __future__ import annotations

from dataclasses import replace

from cid.contracts import ModelContext, ModelUpdate
from cid.grounding import CognitiveLink, LinkRelation, ObjectKind, ObjectRef
from cid.reclamation import retired_reclamation_candidates
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry
from cid.state import CellLifecycle, CognitiveField, DisplayCanvas


class ConvergeAfterStep:
    def __init__(self, step: int) -> None:
        self.target_step = step

    def step(self, context: ModelContext) -> ModelUpdate:
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            converged=context.step >= self.target_step,
        )


def _field_with_live_and_retired(
    *, relation: LinkRelation | None = None
) -> tuple[CognitiveField, str, str]:
    field = CognitiveField.empty(capacity=2, width=2)
    field, live_id = field.allocate(semantic=(1.0, 0.0))
    field, retired_id = field.allocate(semantic=(0.0, 1.0))
    if relation is not None:
        slot = field.slot_of(live_id)
        cells = list(field.cells)
        cells[slot] = replace(
            cells[slot],
            links=(CognitiveLink(relation=relation, target=ObjectRef.cell(retired_id)),),
        )
        field = replace(field, cells=tuple(cells))
    return field.retire(retired_id), live_id, retired_id


async def test_pressure_reclaims_old_retired_cells_into_archive() -> None:
    field = CognitiveField.empty(capacity=4, width=2)
    ids: list[str] = []
    for _ in range(4):
        field, cell_id = field.allocate()
        ids.append(cell_id)
    field = field.retire(ids[2]).retire(ids[3])

    runtime = CIDRuntime(
        SourceRegistry(),
        RuntimeConfig(
            max_steps=4,
            reclamation_grace_steps=1,
            reclamation_low_watermark=0.5,
            reclamation_target_watermark=0.75,
        ),
    )
    result = await runtime.run(
        ConvergeAfterStep(1),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert {item.cell_id for item in result.archive} == {ids[2], ids[3]}
    assert {item.archived_step for item in result.archive} == {1}
    assert result.thought.empty_count == 2
    assert result.trace.count("cell_reclaimed") == 2
    assert result.trace.count("cognitive_compaction") >= 1
    assert all(len(item.roles) == 0 for item in result.archive)


async def test_strong_live_link_pins_retired_neural_state() -> None:
    field, _, retired_id = _field_with_live_and_retired(relation=LinkRelation.DEPENDS_ON)
    runtime = CIDRuntime(
        SourceRegistry(),
        RuntimeConfig(
            max_steps=1,
            reclamation_grace_steps=0,
            reclamation_low_watermark=1.0,
            reclamation_target_watermark=1.0,
        ),
    )
    result = await runtime.run(
        ConvergeAfterStep(0),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.archive == ()
    assert result.thought.get(retired_id).lifecycle is CellLifecycle.RETIRED


async def test_weak_historical_link_survives_as_archive_reference() -> None:
    field, live_id, retired_id = _field_with_live_and_retired(relation=LinkRelation.DERIVED_FROM)
    runtime = CIDRuntime(
        SourceRegistry(),
        RuntimeConfig(
            max_steps=1,
            reclamation_grace_steps=0,
            reclamation_low_watermark=1.0,
            reclamation_target_watermark=1.0,
        ),
    )
    result = await runtime.run(
        ConvergeAfterStep(0),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert len(result.archive) == 1
    tombstone = result.archive[0]
    assert tombstone.cell_id == retired_id
    assert tombstone.physical_slot == 1
    assert result.thought.empty_count == 1
    link = result.thought.get(live_id).links[0]
    assert link.relation is LinkRelation.DERIVED_FROM
    assert link.target.kind is ObjectKind.CELL
    assert link.target.identifier == retired_id
    assert runtime.archive.resolve_cell(link.target, result.thought) == tombstone


def test_archive_does_not_retain_full_semantic_vector() -> None:
    from cid.runtime.archive import CognitiveArchive

    field, _, retired_id = _field_with_live_and_retired()
    archive = CognitiveArchive()
    cell = field.get(retired_id)
    tombstone = archive.record(
        cell,
        created_step=0,
        retired_step=2,
        archived_step=5,
        physical_slot=1,
    )

    assert "semantic" not in tombstone.__slots__
    assert archive.get(retired_id) == tombstone


async def test_trajectory_finalization_archives_safe_retired_cells_without_pressure() -> None:
    field = CognitiveField.empty(capacity=4, width=2)
    field, _ = field.allocate()
    field, retired_id = field.allocate()
    field = field.retire(retired_id)
    assert field.empty_count == 2

    runtime = CIDRuntime(
        SourceRegistry(),
        RuntimeConfig(
            max_steps=1,
            reclamation_grace_steps=0,
            reclamation_low_watermark=0.25,
            reclamation_target_watermark=0.5,
        ),
    )
    result = await runtime.run(
        ConvergeAfterStep(0),
        thought=field,
        display=DisplayCanvas.masked(1, -1),
    )

    assert result.trace.count("cell_reclaimed") == 1
    assert result.archive[0].cell_id == retired_id
    assert result.thought.empty_count == 3


def _full_field_with_two_retired() -> tuple[CognitiveField, str, str]:
    field = CognitiveField.empty(capacity=4, width=2)
    field, first = field.allocate()
    field, second = field.allocate()
    field, _ = field.allocate()
    field, _ = field.allocate()
    return field.retire(first).retire(second), first, second


def test_reclamation_selector_keeps_tombstones_without_pressure() -> None:
    field = CognitiveField.empty(capacity=8, width=2)
    field, cell_id = field.allocate()
    field = field.retire(cell_id)

    assert retired_reclamation_candidates(field, retired_at={cell_id: 0}, step=10) == ()


def test_reclamation_selector_respects_grace_and_pins_under_pressure() -> None:
    field, first, second = _full_field_with_two_retired()

    assert retired_reclamation_candidates(field, retired_at={first: 0, second: 1}, step=1) == ()
    assert retired_reclamation_candidates(
        field,
        retired_at={first: 0, second: 0},
        step=2,
        pinned_cell_ids=frozenset((first,)),
    ) == ((1, second),)


def test_reclamation_selector_stops_at_target_watermark() -> None:
    field, first, second = _full_field_with_two_retired()

    selected = retired_reclamation_candidates(field, retired_at={first: 0, second: 1}, step=3)

    assert selected == ((0, first),)

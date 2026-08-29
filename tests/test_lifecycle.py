from dataclasses import replace

import pytest

from cid.lifecycle import LifecycleTransitionController, LifecycleTransitionSignals
from cid.state import CellLifecycle, CognitiveField


def proposed_lifecycle(
    field: CognitiveField, cell_id: str, lifecycle: CellLifecycle
) -> CognitiveField:
    cells = list(field.cells)
    slot = field.slot_of(cell_id)
    cells[slot] = replace(cells[slot], lifecycle=lifecycle)
    return field.advance(tuple(cells))


def test_waiting_requires_runtime_dependency_and_releases_on_observation() -> None:
    controller = LifecycleTransitionController()
    field, cell_id = CognitiveField.empty(capacity=2, width=3).allocate()

    ungrounded_wait = controller.apply(
        field, proposed_lifecycle(field, cell_id, CellLifecycle.WAITING)
    )
    assert ungrounded_wait.get(cell_id).lifecycle is CellLifecycle.ACTIVE

    waiting = controller.apply(
        field,
        proposed_lifecycle(field, cell_id, CellLifecycle.WAITING),
        LifecycleTransitionSignals(waiting_cells=frozenset({cell_id})),
    )
    assert waiting.get(cell_id).lifecycle is CellLifecycle.WAITING

    model_wants_active = proposed_lifecycle(waiting, cell_id, CellLifecycle.ACTIVE)
    still_waiting = controller.apply(
        waiting,
        model_wants_active,
        LifecycleTransitionSignals(waiting_cells=frozenset({cell_id})),
    )
    assert still_waiting.get(cell_id).lifecycle is CellLifecycle.WAITING

    partially_ready = controller.apply(
        waiting,
        model_wants_active,
        LifecycleTransitionSignals(
            waiting_cells=frozenset({cell_id}),
            available_cells=frozenset({cell_id}),
        ),
    )
    assert partially_ready.get(cell_id).lifecycle is CellLifecycle.WAITING

    released = controller.apply(
        waiting,
        model_wants_active,
        LifecycleTransitionSignals(available_cells=frozenset({cell_id})),
    )
    assert released.get(cell_id).lifecycle is CellLifecycle.ACTIVE


def test_stable_cell_requires_explicit_reopen_signal() -> None:
    controller = LifecycleTransitionController()
    field, cell_id = CognitiveField.empty(capacity=2, width=3).allocate()
    stable = controller.apply(field, proposed_lifecycle(field, cell_id, CellLifecycle.STABLE))

    blocked = controller.apply(stable, proposed_lifecycle(stable, cell_id, CellLifecycle.ACTIVE))
    assert blocked.get(cell_id).lifecycle is CellLifecycle.STABLE

    reopened = controller.apply(
        stable,
        proposed_lifecycle(stable, cell_id, CellLifecycle.ACTIVE),
        LifecycleTransitionSignals(reopen_cells=frozenset({cell_id})),
    )
    assert reopened.get(cell_id).lifecycle is CellLifecycle.ACTIVE


def test_retired_cell_cannot_be_reactivated_or_reclaimed_by_model() -> None:
    controller = LifecycleTransitionController()
    field, cell_id = CognitiveField.empty(capacity=2, width=3).allocate()
    retired = controller.apply(field, proposed_lifecycle(field, cell_id, CellLifecycle.RETIRED))

    proposed_active = proposed_lifecycle(retired, cell_id, CellLifecycle.ACTIVE)
    remains_retired = controller.apply(retired, proposed_active)
    assert remains_retired.get(cell_id).lifecycle is CellLifecycle.RETIRED

    reclaimed = retired.reclaim(cell_id)
    proposed_reclaim = reclaimed.advance(reclaimed.cells)
    with pytest.raises(ValueError, match="runtime reclamation"):
        controller.apply(retired, proposed_reclaim)


def test_new_cells_must_start_active() -> None:
    field = CognitiveField.empty(capacity=2, width=3)
    with pytest.raises(ValueError, match="start ACTIVE"):
        field.allocate(lifecycle=CellLifecycle.WAITING)


def test_new_cell_can_enter_waiting_when_runtime_dependency_is_already_pending() -> None:
    controller = LifecycleTransitionController()
    previous = CognitiveField.empty(capacity=2, width=3)
    proposed, cell_id = previous.allocate()
    cells = list(proposed.cells)
    slot = proposed.slot_of(cell_id)
    cells[slot] = replace(cells[slot], lifecycle=CellLifecycle.WAITING)
    proposed = proposed.advance(tuple(cells))

    blocked = controller.apply(previous, proposed)
    assert blocked.get(cell_id).lifecycle is CellLifecycle.ACTIVE

    waiting = controller.apply(
        previous,
        proposed,
        LifecycleTransitionSignals(waiting_cells=frozenset({cell_id})),
    )
    assert waiting.get(cell_id).lifecycle is CellLifecycle.WAITING

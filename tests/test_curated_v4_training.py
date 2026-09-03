from __future__ import annotations

from cid.curated_v4_training import CuratedV4Config, generate_curated_v4
from cid.data import DISPLAY_UNKNOWN_MARKER, is_display_process_status


def test_curated_v4_curriculum_is_deterministic_and_contract_clean() -> None:
    config = CuratedV4Config(count_per_family=2, seed=17)
    first = generate_curated_v4(config)
    second = generate_curated_v4(config)

    assert first == second
    assert len(first) == 18
    assert len({item.example_id for item in first}) == len(first)
    assert {item.metadata["family"] for item in first} == {
        "curated_v4_single_lookup",
        "curated_v4_two_hop_lookup",
        "curated_v4_parallel_compare",
        "curated_v4_streaming_evidence",
        "curated_v4_dynamic_refresh",
        "curated_v4_authoritative_correction",
        "curated_v4_no_tool_reasoning",
        "curated_v4_tool_restraint",
        "curated_v4_zh_lookup",
    }

    for example in first:
        steps = sorted({target.step for target in example.thought_targets})
        displays = {target.step: target.text for target in example.display_targets}
        assert set(displays) == set(steps)
        assert displays[steps[-1]] == example.target_display
        assert displays[steps[-2]] == example.target_display
        assert all(not is_display_process_status(text) for text in displays.values())
        grounding = {(target.step, target.cell_id) for target in example.grounding_targets}
        for binding in example.binding_targets:
            assert (binding.first_need_step, binding.owner_cell_id) in grounding
        for event in example.events:
            if not example.binding_targets:
                continue
            assert any(target.step == event.arrival_step for target in example.grounding_targets)


def test_curated_v4_cells_do_not_disappear_without_retirement() -> None:
    examples = generate_curated_v4(CuratedV4Config(count_per_family=1, seed=19))

    for example in examples:
        by_step: dict[int, dict[str, object]] = {}
        for target in example.thought_targets:
            by_step.setdefault(target.step, {})[target.cell_id] = target
        for step in range(max(by_step)):
            lost = set(by_step[step]) - set(by_step[step + 1])
            assert all(by_step[step][cell_id].lifecycle.value == "retired" for cell_id in lost)


def test_curated_v4_contains_true_partial_answer_progression() -> None:
    examples = generate_curated_v4(CuratedV4Config(count_per_family=1, seed=23))
    by_family = {str(item.metadata["family"]): item for item in examples}

    two_hop = by_family["curated_v4_two_hop_lookup"]
    displays = [target.text for target in two_hop.display_targets]
    assert displays[0].count(DISPLAY_UNKNOWN_MARKER) == 2
    assert displays[2].count(DISPLAY_UNKNOWN_MARKER) == 1
    assert DISPLAY_UNKNOWN_MARKER not in displays[-1]

    correction = by_family["curated_v4_authoritative_correction"]
    displays = [target.text for target in correction.display_targets]
    assert "provisional" in displays[0]
    assert "confirmation" in displays[2]
    assert DISPLAY_UNKNOWN_MARKER in displays[2]
    assert DISPLAY_UNKNOWN_MARKER not in displays[-1]

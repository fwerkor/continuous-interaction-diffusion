from cid.data import dump_jsonl, load_jsonl
from cid.synthetic import SyntheticConfig, SyntheticFamily, generate_synthetic


def test_synthetic_factory_is_deterministic_and_covers_all_families(tmp_path) -> None:
    config = SyntheticConfig(count_per_family=3, seed=17, thought_capacity=8)

    first = generate_synthetic(config)
    second = generate_synthetic(config)

    assert first == second
    assert len(first) == 3 * len(SyntheticFamily)
    assert {item.metadata["family"] for item in first} == {
        family.value for family in SyntheticFamily
    }
    assert len({item.example_id for item in first}) == len(first)

    path = tmp_path / "synthetic.jsonl"
    dump_jsonl(first, path)
    assert load_jsonl(path) == first


def test_synthetic_trajectories_have_contiguous_steps_and_valid_slots() -> None:
    capacity = 9
    examples = generate_synthetic(
        SyntheticConfig(count_per_family=4, seed=23, thought_capacity=capacity)
    )

    placements: dict[str, set[tuple[int, ...]]] = {}
    for example in examples:
        thought_steps = sorted({target.step for target in example.thought_targets})
        assert thought_steps == list(range(thought_steps[-1] + 1))
        assert all(0 <= target.slot < capacity for target in example.thought_targets)
        assert all(event.arrival_step >= 0 for event in example.events)
        assert all(binding.first_need_step >= 0 for binding in example.binding_targets)
        family = str(example.metadata["family"])
        step_zero_slots = tuple(
            target.slot for target in example.thought_targets if target.step == 0
        )
        placements.setdefault(family, set()).add(step_zero_slots)

    assert any(len(items) > 1 for items in placements.values())


def test_synthetic_seed_changes_physical_slot_placement() -> None:
    left = generate_synthetic(SyntheticConfig(count_per_family=2, seed=1, thought_capacity=8))
    right = generate_synthetic(SyntheticConfig(count_per_family=2, seed=2, thought_capacity=8))

    def placements(examples):
        return {
            item.example_id: tuple(
                target.slot for target in item.thought_targets if target.step == 0
            )
            for item in examples
        }

    assert placements(left) != placements(right)

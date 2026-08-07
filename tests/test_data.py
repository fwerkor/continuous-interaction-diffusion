from cid.data import BindingTarget, ExternalEvent, TrajectoryExample, dump_jsonl, load_jsonl


def test_trajectory_jsonl_round_trip(tmp_path) -> None:
    example = TrajectoryExample(
        example_id="copy-1",
        prompt="Copy the documented latency.",
        target_display="37 ms",
        protected_facts={"instruction": "quote exactly"},
        events=(ExternalEvent(source="docs", value="37 ms", arrival_step=3, version="v1"),),
        binding_targets=(
            BindingTarget(
                need_id="latency",
                source="docs",
                first_need_step=1,
                executable_step=2,
                target_cells=("c0",),
                target_display=(2, 3),
            ),
        ),
    )
    path = tmp_path / "trajectory.jsonl"
    dump_jsonl((example,), path)

    loaded = load_jsonl(path)

    assert loaded == (example,)

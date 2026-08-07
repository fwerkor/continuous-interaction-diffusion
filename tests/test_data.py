from cid.data import (
    BindingTarget,
    ExternalEvent,
    GroundingTarget,
    TrajectoryExample,
    dump_jsonl,
    load_jsonl,
)
from cid.grounding import (
    Anchor,
    AnchorKind,
    CognitiveLink,
    GroundingEntry,
    LinkRelation,
    ObjectRef,
)


def test_trajectory_jsonl_round_trip(tmp_path) -> None:
    model_anchor = Anchor(
        anchor_id="a:model-a",
        kind=AnchorKind.ENTITY,
        value="Model A",
        object_id="model:a",
    )
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
                target_cells=(ObjectRef.cell("c0"),),
                target_display=(ObjectRef.display_span(2, 4),),
            ),
        ),
        grounding_catalog=(GroundingEntry(anchor=model_anchor, aliases=("model-a",)),),
        grounding_targets=(
            GroundingTarget(
                step=2,
                cell_id="c0",
                anchors=(model_anchor,),
                links=(
                    CognitiveLink(
                        relation=LinkRelation.REQUESTS,
                        target=ObjectRef.source("docs"),
                    ),
                ),
            ),
        ),
    )
    path = tmp_path / "trajectory.jsonl"
    dump_jsonl((example,), path)

    loaded = load_jsonl(path)

    assert loaded == (example,)

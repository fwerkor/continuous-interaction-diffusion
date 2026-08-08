from cid.data import (
    BindingTarget,
    DisplayTarget,
    ExternalEvent,
    GroundingTarget,
    ThoughtTarget,
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
from cid.state import CellLifecycle, CognitiveRole


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
                arguments={"key": "latency_ms"},
                argument_steps={"key": 2},
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
        thought_targets=(
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="c0",
                semantic_text="Need the documented latency value.",
                roles={CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.8,
                noise=0.7,
                lifecycle=CellLifecycle.ACTIVE,
            ),
            ThoughtTarget(
                step=2,
                slot=0,
                cell_id="c0",
                semantic_text="The documented latency is 37 ms.",
                roles={CognitiveRole.CONCLUSION: 1.0},
                uncertainty=0.1,
                noise=0.2,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(DisplayTarget(step=2, text="37 ms"),),
    )
    path = tmp_path / "trajectory.jsonl"
    dump_jsonl((example,), path)

    loaded = load_jsonl(path)

    assert loaded == (example,)

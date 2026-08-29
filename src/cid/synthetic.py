from __future__ import annotations

import random
from dataclasses import dataclass

from cid._compat import StrEnum
from cid.contracts import FreshnessDemand
from cid.data import (
    BindingTarget,
    DisplayTarget,
    ExternalEvent,
    GroundingTarget,
    ThoughtTarget,
    TrajectoryExample,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, GroundingEntry, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole


class SyntheticFamily(StrEnum):
    STATIC_COPY = "static_copy"
    DELAYED_RETRIEVAL = "delayed_retrieval"
    DYNAMIC_STATE = "dynamic_state"
    STREAMING_EVIDENCE = "streaming_evidence"
    COMPETING_SOURCES = "competing_sources"


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    count_per_family: int = 32
    seed: int = 0
    thought_capacity: int = 8
    index_offset: int = 0
    id_prefix: str = ""
    split: str | None = None

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")
        if self.thought_capacity < 4:
            raise ValueError("thought_capacity must be at least 4")
        if self.index_offset < 0:
            raise ValueError("index_offset must be non-negative")
        if self.split is not None and self.split not in {"train", "validation", "test"}:
            raise ValueError("synthetic split must be train, validation, or test")


def generate_synthetic(config: SyntheticConfig | None = None) -> tuple[TrajectoryExample, ...]:
    config = config or SyntheticConfig()
    rng = random.Random(config.seed)
    generators = {
        SyntheticFamily.STATIC_COPY: _static_copy,
        SyntheticFamily.DELAYED_RETRIEVAL: _delayed_retrieval,
        SyntheticFamily.DYNAMIC_STATE: _dynamic_state,
        SyntheticFamily.STREAMING_EVIDENCE: _streaming_evidence,
        SyntheticFamily.COMPETING_SOURCES: _competing_sources,
    }
    examples: list[TrajectoryExample] = []
    for family, generator in generators.items():
        for relative_index in range(config.count_per_family):
            index = config.index_offset + relative_index
            examples.append(generator(rng, config, family, index))
    rng.shuffle(examples)
    return tuple(examples)


def _static_copy(
    rng: random.Random,
    config: SyntheticConfig,
    family: SyntheticFamily,
    index: int,
) -> TrajectoryExample:
    plan_slot, need_slot = rng.sample(range(config.thought_capacity), 2)
    # Keep the semantic task identity unique at large generation scales.  The
    # value remains randomized, but reusing one of only 1,000 keys caused
    # otherwise identical generated tasks to be removed by the distillation
    # review gate when count_per_family exceeded that range.
    key = f"latency_{index}"
    value = rng.randrange(10, 200)
    key_anchor = _text_anchor(f"{family}-{index}-key", key)
    value_anchor = _number_anchor(f"{family}-{index}-value", value, unit="ms")
    return TrajectoryExample(
        example_id=_example_id(config, family, index),
        prompt=f"Read {key} from the documentation and return the value exactly.",
        target_display=f"{value} ms",
        protected_facts={"output_rule": "return the documented value exactly"},
        source_descriptors=(_mapping_source("docs", promote_results_to_fact=True),),
        events=(
            ExternalEvent(
                source="docs",
                value=f"{value} ms",
                arrival_step=2,
                version="v1",
                arguments={"key": key},
            ),
        ),
        binding_targets=(
            BindingTarget(
                need_id="lookup",
                source="docs",
                first_need_step=1,
                executable_step=1,
                arguments={"key": key},
                argument_steps={"key": 1},
                target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c1",
            ),
        ),
        grounding_catalog=(
            GroundingEntry(anchor=key_anchor, aliases=(key.lower(),)),
            GroundingEntry(anchor=value_anchor),
        ),
        grounding_targets=(
            GroundingTarget(
                step=1,
                cell_id="c1",
                anchors=(key_anchor,),
                links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("docs")),),
            ),
            GroundingTarget(
                step=2,
                cell_id="c1",
                anchors=(value_anchor,),
                links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("docs")),),
            ),
        ),
        thought_targets=(
            _thought(0, plan_slot, "c0", "Plan an exact documentation lookup.", CognitiveRole.PLAN),
            _thought(
                1,
                plan_slot,
                "c0",
                "The answer must come from the documentation source.",
                CognitiveRole.PLAN,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.2,
                noise=0.2,
            ),
            _thought(
                1,
                need_slot,
                "c1",
                f"Need the documented value for {key}.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                plan_slot,
                "c0",
                "The documentation requirement is satisfied.",
                CognitiveRole.CONSTRAINT,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.1,
                noise=0.1,
            ),
            _thought(
                2,
                need_slot,
                "c1",
                f"The documented value is {value} ms.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.05,
                noise=0.1,
            ),
        ),
        display_targets=(
            DisplayTarget(step=1, text="pending"),
            DisplayTarget(step=2, text=f"{value} ms"),
        ),
        metadata=_metadata(config, family),
    )


def _delayed_retrieval(
    rng: random.Random,
    config: SyntheticConfig,
    family: SyntheticFamily,
    index: int,
) -> TrajectoryExample:
    plan_slot, need_slot = rng.sample(range(config.thought_capacity), 2)
    key = f"release_{index}"
    value = f"r{rng.randrange(10, 99)}"
    key_anchor = _text_anchor(f"{family}-{index}-key", key)
    value_anchor = _text_anchor(f"{family}-{index}-value", value)
    return TrajectoryExample(
        example_id=_example_id(config, family, index),
        prompt=f"Find the release tag for {key}. Continue reasoning while the lookup is delayed.",
        target_display=value,
        source_descriptors=(_mapping_source("registry"),),
        events=(
            ExternalEvent(
                source="registry",
                value=value,
                arrival_step=3,
                version="v1",
                arguments={"key": key},
            ),
        ),
        binding_targets=(
            BindingTarget(
                need_id="release",
                source="registry",
                first_need_step=1,
                executable_step=1,
                arguments={"key": key},
                target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c1",
            ),
        ),
        grounding_catalog=(GroundingEntry(key_anchor), GroundingEntry(value_anchor)),
        grounding_targets=(
            GroundingTarget(
                step=1,
                cell_id="c1",
                anchors=(key_anchor,),
                links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("registry")),),
            ),
            GroundingTarget(
                step=3,
                cell_id="c1",
                anchors=(value_anchor,),
                links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("registry")),),
            ),
        ),
        thought_targets=(
            _thought(0, plan_slot, "c0", "Plan the delayed lookup.", CognitiveRole.PLAN),
            _thought(
                1,
                plan_slot,
                "c0",
                "Other reasoning may continue while I/O runs.",
                CognitiveRole.PLAN,
            ),
            _thought(
                1, need_slot, "c1", f"Need release tag for {key}.", CognitiveRole.INFORMATION_NEED
            ),
            _thought(2, plan_slot, "c0", "The lookup is still outstanding.", CognitiveRole.PLAN),
            _thought(
                2,
                need_slot,
                "c1",
                f"Waiting for release tag {key}.",
                CognitiveRole.INFORMATION_NEED,
                lifecycle=CellLifecycle.WAITING,
            ),
            _thought(
                3,
                plan_slot,
                "c0",
                "The delayed lookup has completed.",
                CognitiveRole.PLAN,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.1,
                noise=0.1,
            ),
            _thought(
                3,
                need_slot,
                "c1",
                f"Release tag is {value}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.05,
                noise=0.1,
            ),
        ),
        display_targets=(
            DisplayTarget(1, "pending"),
            DisplayTarget(2, "pending"),
            DisplayTarget(3, value),
        ),
        metadata={**_metadata(config, family), "delay_steps": 2},
    )


def _dynamic_state(
    rng: random.Random,
    config: SyntheticConfig,
    family: SyntheticFamily,
    index: int,
) -> TrajectoryExample:
    plan_slot, state_slot = rng.sample(range(config.thought_capacity), 2)
    # The old 49 x 19 (first value, delta) state space contains fewer than
    # 2,000 combinations, so a 2k-family build necessarily produced many
    # semantic duplicates.  Give every generated instance a distinct first
    # observation while retaining a randomized positive refresh delta.
    first = index + 1
    second = first + rng.randrange(1, 20)
    first_anchor = _number_anchor(f"{family}-{index}-first", first)
    second_anchor = _number_anchor(f"{family}-{index}-second", second)
    return TrajectoryExample(
        example_id=_example_id(config, family, index),
        prompt="Track the live counter and return the newest observed value.",
        target_display=str(second),
        source_descriptors=(
            {
                "name": "counter",
                "description": "read the current counter",
                "arguments": (),
                "cacheable": False,
                "dynamic": True,
                "versioned": True,
                "promote_results_to_fact": False,
            },
        ),
        events=(
            ExternalEvent("counter", first, arrival_step=2, version="1"),
            ExternalEvent("counter", second, arrival_step=4, version="2"),
        ),
        binding_targets=(
            BindingTarget(
                need_id="counter",
                source="counter",
                first_need_step=1,
                executable_step=1,
                freshness=FreshnessDemand.ALWAYS,
                target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c1",
            ),
        ),
        grounding_catalog=(GroundingEntry(first_anchor), GroundingEntry(second_anchor)),
        grounding_targets=(
            GroundingTarget(
                2,
                "c1",
                anchors=(first_anchor,),
                links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("counter")),),
            ),
            GroundingTarget(
                4,
                "c1",
                anchors=(second_anchor,),
                links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source("counter")),),
            ),
        ),
        thought_targets=(
            _thought(0, plan_slot, "c0", "Plan persistent monitoring.", CognitiveRole.PLAN),
            _thought(
                1,
                plan_slot,
                "c0",
                "The answer depends on a dynamic source.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                1,
                state_slot,
                "c1",
                "Need the current counter value.",
                CognitiveRole.INFORMATION_NEED,
            ),
            _thought(
                2,
                plan_slot,
                "c0",
                "Keep monitoring after the first observation.",
                CognitiveRole.PLAN,
            ),
            _thought(
                2,
                state_slot,
                "c1",
                f"Observed counter value {first}.",
                CognitiveRole.PERCEPT,
                noise=0.5,
            ),
            _thought(3, plan_slot, "c0", "The source remains dynamic.", CognitiveRole.CONSTRAINT),
            _thought(
                3,
                state_slot,
                "c1",
                f"Current known counter is {first}.",
                CognitiveRole.CONCLUSION,
                noise=0.3,
            ),
            _thought(
                4,
                plan_slot,
                "c0",
                "A newer source version supersedes the old value.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                4,
                state_slot,
                "c1",
                f"Newest counter value is {second}.",
                CognitiveRole.CONCLUSION,
                noise=0.8,
            ),
        ),
        display_targets=(
            DisplayTarget(1, "pending"),
            DisplayTarget(2, str(first)),
            DisplayTarget(3, str(first)),
            DisplayTarget(4, str(second)),
        ),
        metadata=_metadata(config, family),
    )


def _streaming_evidence(
    rng: random.Random,
    config: SyntheticConfig,
    family: SyntheticFamily,
    index: int,
) -> TrajectoryExample:
    plan_slot, evidence_slot = rng.sample(range(config.thought_capacity), 2)
    topic = f"topic-{index}"
    first = f"evidence-{rng.randrange(100, 999)}"
    second = f"evidence-{rng.randrange(100, 999)}"
    return TrajectoryExample(
        example_id=_example_id(config, family, index),
        prompt=f"Collect the streaming evidence for {topic} and report both pieces in order.",
        target_display=f"{first} {second}",
        source_descriptors=(
            {
                "name": "stream",
                "description": "stream evidence chunks",
                "arguments": ({"name": "topic", "kind": "string", "required": True},),
                "cacheable": False,
                "dynamic": True,
                "streamable": True,
                "versioned": True,
                "promote_results_to_fact": False,
            },
        ),
        events=(
            ExternalEvent("stream", first, 2, version="1", arguments={"topic": topic}),
            ExternalEvent("stream", second, 3, version="2", arguments={"topic": topic}),
        ),
        binding_targets=(
            BindingTarget(
                need_id="stream",
                source="stream",
                first_need_step=1,
                executable_step=1,
                arguments={"topic": topic},
                freshness=FreshnessDemand.ALWAYS,
                target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c1",
            ),
        ),
        thought_targets=(
            _thought(
                0, plan_slot, "c0", "Plan incremental evidence assimilation.", CognitiveRole.PLAN
            ),
            _thought(
                1,
                plan_slot,
                "c0",
                "The answer needs multiple streaming chunks.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                1, evidence_slot, "c1", f"Need stream for {topic}.", CognitiveRole.INFORMATION_NEED
            ),
            _thought(
                2,
                plan_slot,
                "c0",
                "One chunk has arrived; more evidence is expected.",
                CognitiveRole.PLAN,
            ),
            _thought(
                2, evidence_slot, "c1", f"First chunk: {first}.", CognitiveRole.PERCEPT, noise=0.6
            ),
            _thought(
                3, plan_slot, "c0", "All expected chunks are now available.", CognitiveRole.PLAN
            ),
            _thought(
                3,
                evidence_slot,
                "c1",
                f"Combined evidence: {first} then {second}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
                uncertainty=0.1,
                noise=0.2,
            ),
        ),
        display_targets=(
            DisplayTarget(1, "pending"),
            DisplayTarget(2, f"{first} ..."),
            DisplayTarget(3, f"{first} {second}"),
        ),
        metadata=_metadata(config, family),
    )


def _competing_sources(
    rng: random.Random,
    config: SyntheticConfig,
    family: SyntheticFamily,
    index: int,
) -> TrajectoryExample:
    plan_slot, primary_slot, secondary_slot, conclusion_slot = rng.sample(
        range(config.thought_capacity), 4
    )
    key = f"score_{index}"
    primary_value = rng.randrange(50, 100)
    secondary_value = primary_value + rng.choice((-7, -5, 5, 7))
    return TrajectoryExample(
        example_id=_example_id(config, family, index),
        prompt=(
            f"Compare primary and secondary values for {key}; "
            "prefer primary when they conflict."
        ),
        target_display=str(primary_value),
        protected_facts={"source_priority": "primary > secondary"},
        source_descriptors=(_mapping_source("primary"), _mapping_source("secondary")),
        events=(
            ExternalEvent("primary", primary_value, 2, version="p1", arguments={"key": key}),
            ExternalEvent("secondary", secondary_value, 2, version="s1", arguments={"key": key}),
        ),
        binding_targets=(
            BindingTarget(
                "primary-need",
                "primary",
                1,
                1,
                arguments={"key": key},
                target_cells=(ObjectRef.cell("c1"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c1",
            ),
            BindingTarget(
                "secondary-need",
                "secondary",
                1,
                1,
                arguments={"key": key},
                target_cells=(ObjectRef.cell("c2"), ObjectRef.cell("c0")),
                target_display=(ObjectRef.display_span(0, 1),),
                owner_cell_id="c2",
            ),
        ),
        grounding_targets=(
            GroundingTarget(
                1,
                "c1",
                links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("primary")),),
            ),
            GroundingTarget(
                1,
                "c2",
                links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source("secondary")),),
            ),
            GroundingTarget(
                3,
                "c3",
                links=(
                    CognitiveLink(LinkRelation.SUPPORTS, ObjectRef.cell("c1")),
                    CognitiveLink(LinkRelation.CONFLICTS, ObjectRef.cell("c2")),
                ),
            ),
        ),
        thought_targets=(
            _thought(0, plan_slot, "c0", "Plan two independent source reads.", CognitiveRole.PLAN),
            _thought(
                1,
                plan_slot,
                "c0",
                "Primary outranks secondary on conflict.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(1, primary_slot, "c1", f"Need primary {key}.", CognitiveRole.INFORMATION_NEED),
            _thought(
                1, secondary_slot, "c2", f"Need secondary {key}.", CognitiveRole.INFORMATION_NEED
            ),
            _thought(
                2,
                plan_slot,
                "c0",
                "The sources disagree; apply source priority.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                2, primary_slot, "c1", f"Primary reports {primary_value}.", CognitiveRole.PERCEPT
            ),
            _thought(
                2,
                secondary_slot,
                "c2",
                f"Secondary reports {secondary_value}.",
                CognitiveRole.PERCEPT,
            ),
            _thought(
                3,
                plan_slot,
                "c0",
                "Primary source priority resolves the conflict.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                3,
                primary_slot,
                "c1",
                f"Primary evidence {primary_value} is authoritative.",
                CognitiveRole.PERCEPT,
            ),
            _thought(
                3,
                secondary_slot,
                "c2",
                f"Secondary evidence {secondary_value} conflicts.",
                CognitiveRole.PERCEPT,
            ),
            _thought(
                3, conclusion_slot, "c3", f"Choose {primary_value}.", CognitiveRole.CONCLUSION
            ),
        ),
        display_targets=(
            DisplayTarget(1, "pending"),
            DisplayTarget(2, "conflict"),
            DisplayTarget(3, str(primary_value)),
        ),
        metadata=_metadata(config, family),
    )


def _thought(
    step: int,
    slot: int,
    cell_id: str,
    semantic_text: str,
    role: CognitiveRole,
    *,
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE,
    uncertainty: float = 0.5,
    noise: float = 0.5,
) -> ThoughtTarget:
    return ThoughtTarget(
        step=step,
        slot=slot,
        cell_id=cell_id,
        semantic_text=semantic_text,
        roles={role: 1.0},
        uncertainty=uncertainty,
        noise=noise,
        lifecycle=lifecycle,
    )


def _mapping_source(name: str, *, promote_results_to_fact: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "description": "read a keyed value",
        "arguments": ({"name": "key", "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": True,
        "promote_results_to_fact": promote_results_to_fact,
    }


def _text_anchor(anchor_id: str, value: str) -> Anchor:
    return Anchor(anchor_id=anchor_id, kind=AnchorKind.TEXT, value=value)


def _number_anchor(anchor_id: str, value: int, *, unit: str | None = None) -> Anchor:
    return Anchor(anchor_id=anchor_id, kind=AnchorKind.NUMBER, value=value, unit=unit)


def _metadata(config: SyntheticConfig, family: SyntheticFamily) -> dict[str, object]:
    metadata: dict[str, object] = {"family": family.value}
    if config.split is not None:
        metadata["split"] = config.split
    return metadata


def _example_id(config: SyntheticConfig, family: SyntheticFamily, index: int) -> str:
    prefix = f"{config.id_prefix}-" if config.id_prefix else ""
    return f"{prefix}{family.value}-{index:06d}"

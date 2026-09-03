from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from cid.contracts import FreshnessDemand
from cid.data import (
    DISPLAY_UNKNOWN_MARKER,
    BindingTarget,
    DisplayTarget,
    ExternalEvent,
    GroundingTarget,
    ThoughtTarget,
    TrajectoryExample,
    dump_jsonl,
)
from cid.dataset import dump_dataset_manifest, inspect_dataset
from cid.grounding import CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole


@dataclass(frozen=True, slots=True)
class CuratedV4Config:
    count_per_family: int = 48
    seed: int = 20260903
    thought_capacity: int = 8
    training_weight: float = 4.0

    def __post_init__(self) -> None:
        if self.count_per_family <= 0:
            raise ValueError("count_per_family must be positive")
        if self.thought_capacity < 4:
            raise ValueError("curated v4 training requires thought_capacity >= 4")
        if self.training_weight <= 0.0:
            raise ValueError("training_weight must be positive")


_FAMILIES = (
    "single_lookup",
    "two_hop_lookup",
    "parallel_compare",
    "streaming_evidence",
    "dynamic_refresh",
    "authoritative_correction",
    "no_tool_reasoning",
    "tool_restraint",
    "zh_lookup",
)
_NAMES = ("Aster", "Beryl", "Cedar", "Delta", "Ember", "Flint", "Grove", "Harbor")
_REGIONS = ("north", "south", "east", "west", "central")


def generate_curated_v4(config: CuratedV4Config | None = None) -> tuple[TrajectoryExample, ...]:
    config = config or CuratedV4Config()
    examples: list[TrajectoryExample] = []
    builders = (
        _single_lookup,
        _two_hop_lookup,
        _parallel_compare,
        _streaming_evidence,
        _dynamic_refresh,
        _authoritative_correction,
        _no_tool_reasoning,
        _tool_restraint,
        _zh_lookup,
    )
    for family_index, (_family, builder) in enumerate(zip(_FAMILIES, builders, strict=True)):
        for index in range(config.count_per_family):
            rng = random.Random(config.seed + family_index * 100_003 + index)
            examples.append(builder(rng, config, index))
    return tuple(examples)


def build_curated_v4_training(
    output_path: str | Path,
    manifest_output: str | Path,
    config: CuratedV4Config | None = None,
) -> dict[str, object]:
    config = config or CuratedV4Config()
    examples = generate_curated_v4(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(examples, output)
    dataset_manifest = inspect_dataset(output)
    dump_dataset_manifest(dataset_manifest, manifest_output)
    family_counts = {
        family: sum(
            example.metadata.get("family") == f"curated_v4_{family}" for example in examples
        )
        for family in _FAMILIES
    }
    return {
        **dataset_manifest.to_dict(),
        "name": "curated-v4-display-curriculum",
        "neural_contract_version": 4,
        "hand_authored_archetypes": len(_FAMILIES),
        "family_counts": family_counts,
        "training_weight": config.training_weight,
    }


def _metadata(config: CuratedV4Config, family: str, index: int) -> dict[str, object]:
    task_id = f"curated-v4-{family}-{index:05d}"
    return {
        "family": f"curated_v4_{family}",
        "semantic_task_id": task_id,
        "training_weight": config.training_weight,
        "training_mode": "tool_required"
        if family not in {"no_tool_reasoning", "tool_restraint"}
        else ("no_tool" if family == "no_tool_reasoning" else "tools_available_unnecessary"),
        "curation": "hand-authored-v4-archetype",
        "neural_contract_version": 4,
        "display_contract": "continuous-answer-draft-v1",
    }


def _thought(
    step: int,
    slot: int,
    cell_id: str,
    text: str,
    role: CognitiveRole,
    *,
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE,
    uncertainty: float = 0.2,
    noise: float = 0.1,
) -> ThoughtTarget:
    return ThoughtTarget(
        step=step,
        slot=slot,
        cell_id=cell_id,
        semantic_text=text,
        roles={role: 1.0},
        uncertainty=uncertainty,
        noise=noise,
        lifecycle=lifecycle,
    )


def _mapping_source(name: str, description: str, argument: str = "key") -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "arguments": ({"name": argument, "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": True,
        "promote_results_to_fact": False,
    }


def _request_grounding(step: int, cell_id: str, source: str) -> GroundingTarget:
    return GroundingTarget(
        step=step,
        cell_id=cell_id,
        links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(source), 1.0),),
    )


def _observe_grounding(step: int, cell_id: str, source: str) -> GroundingTarget:
    return GroundingTarget(
        step=step,
        cell_id=cell_id,
        links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(source), 1.0),),
    )


def _single_lookup(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    value = rng.randrange(12, 95)
    key = f"service-{index:04d}"
    source = "service_registry"
    final = f"Documented latency: {value} ms."
    return TrajectoryExample(
        example_id=f"curated-v4-single-lookup-{index:05d}",
        prompt=(
            f"Return the documented latency for {key}. Do not guess before the registry responds."
        ),
        target_display=final,
        source_descriptors=(_mapping_source(source, "Read one authoritative service record."),),
        events=(ExternalEvent(source, value, 2, version="1", arguments={"key": key}),),
        binding_targets=(
            BindingTarget(
                "latency",
                source,
                1,
                1,
                arguments={"key": key},
                owner_cell_id="need",
                target_cells=(ObjectRef.cell("need"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "need", source),
            _observe_grounding(2, "need", source),
            _observe_grounding(3, "need", source),
        ),
        thought_targets=(
            _thought(
                0, 0, "goal", "Return only the authoritative registry latency.", CognitiveRole.PLAN
            ),
            _thought(
                1,
                0,
                "goal",
                "The registry value is required before answering.",
                CognitiveRole.CONSTRAINT,
            ),
            _thought(
                1,
                1,
                "need",
                f"Need latency for {key}.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                0,
                "goal",
                "The authoritative latency is available.",
                CognitiveRole.CONSTRAINT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                2,
                1,
                "need",
                f"Registry latency is {value} ms.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                0,
                "goal",
                "The authoritative latency remains resolved.",
                CognitiveRole.CONSTRAINT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                1,
                "need",
                f"Final latency is {value} ms.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"Documented latency: {DISPLAY_UNKNOWN_MARKER} ms."),
            DisplayTarget(1, f"Documented latency: {DISPLAY_UNKNOWN_MARKER} ms."),
            DisplayTarget(2, final),
            DisplayTarget(3, final),
        ),
        metadata=_metadata(config, "single_lookup", index),
    )


def _two_hop_lookup(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    owner = _NAMES[index % len(_NAMES)]
    region = _REGIONS[(index // len(_NAMES)) % len(_REGIONS)]
    item = f"artifact-{index:04d}"
    owner_source = "ownership_index"
    region_source = "region_catalog"
    final = f"Owner: {owner}; region: {region}."
    return TrajectoryExample(
        example_id=f"curated-v4-two-hop-{index:05d}",
        prompt=(
            f"Find the owner of {item}, then use that owner to return "
            "the owner's registered region."
        ),
        target_display=final,
        source_descriptors=(
            _mapping_source(owner_source, "Resolve an artifact to its owner.", "artifact"),
            _mapping_source(region_source, "Resolve an owner to a registered region.", "owner"),
        ),
        events=(
            ExternalEvent(owner_source, owner, 2, arguments={"artifact": item}),
            ExternalEvent(region_source, region, 4, arguments={"owner": owner}),
        ),
        binding_targets=(
            BindingTarget(
                "owner",
                owner_source,
                1,
                1,
                arguments={"artifact": item},
                owner_cell_id="owner",
                target_cells=(ObjectRef.cell("owner"),),
            ),
            BindingTarget(
                "region",
                region_source,
                3,
                3,
                arguments={"owner": owner},
                owner_cell_id="region",
                target_cells=(ObjectRef.cell("region"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "owner", owner_source),
            _observe_grounding(2, "owner", owner_source),
            _request_grounding(3, "region", region_source),
            _observe_grounding(4, "region", region_source),
        ),
        thought_targets=(
            _thought(0, 0, "goal", "Resolve owner first, then region.", CognitiveRole.PLAN),
            *(
                _thought(
                    step,
                    0,
                    "goal",
                    "Resolve owner first, then region.",
                    CognitiveRole.PLAN,
                    lifecycle=CellLifecycle.STABLE,
                )
                for step in range(1, 6)
            ),
            _thought(
                1,
                1,
                "owner",
                f"Need owner of {item}.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                1,
                "owner",
                f"Owner is {owner}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                1,
                "owner",
                f"Owner remains {owner}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                2,
                "region",
                f"Need registered region for {owner}.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                4,
                1,
                "owner",
                f"Owner is {owner}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                4,
                2,
                "region",
                f"Region is {region}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                1,
                "owner",
                f"Owner is {owner}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                2,
                "region",
                f"Final region is {region}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"Owner: {DISPLAY_UNKNOWN_MARKER}; region: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(1, f"Owner: {DISPLAY_UNKNOWN_MARKER}; region: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(2, f"Owner: {owner}; region: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(3, f"Owner: {owner}; region: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(4, final),
            DisplayTarget(5, final),
        ),
        metadata=_metadata(config, "two_hop_lookup", index),
    )


def _parallel_compare(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    primary = rng.randrange(50, 100)
    secondary = primary + rng.choice((-9, -5, 4, 8))
    key = f"score-{index:04d}"
    final = f"Primary: {primary}; secondary: {secondary}; selected: {primary}."
    return TrajectoryExample(
        example_id=f"curated-v4-parallel-{index:05d}",
        prompt=(
            f"Read primary and secondary scores for {key}. "
            "Report both and select primary on conflict."
        ),
        target_display=final,
        source_descriptors=(
            _mapping_source("primary_scores", "Authoritative primary score."),
            _mapping_source("secondary_scores", "Secondary comparison score."),
        ),
        events=(
            ExternalEvent("primary_scores", primary, 2, arguments={"key": key}),
            ExternalEvent("secondary_scores", secondary, 3, arguments={"key": key}),
        ),
        binding_targets=(
            BindingTarget(
                "primary",
                "primary_scores",
                1,
                1,
                arguments={"key": key},
                owner_cell_id="primary",
                target_cells=(ObjectRef.cell("primary"),),
            ),
            BindingTarget(
                "secondary",
                "secondary_scores",
                1,
                1,
                arguments={"key": key},
                owner_cell_id="secondary",
                target_cells=(ObjectRef.cell("secondary"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "primary", "primary_scores"),
            _request_grounding(1, "secondary", "secondary_scores"),
            _observe_grounding(2, "primary", "primary_scores"),
            _observe_grounding(3, "secondary", "secondary_scores"),
        ),
        thought_targets=(
            *(
                _thought(
                    step,
                    0,
                    "rule",
                    "Primary wins if the sources conflict.",
                    CognitiveRole.CONSTRAINT,
                    lifecycle=CellLifecycle.STABLE,
                )
                for step in range(5)
            ),
            _thought(
                1,
                1,
                "primary",
                "Need the primary score.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                1,
                2,
                "secondary",
                "Need the secondary score.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                1,
                "primary",
                f"Primary score is {primary}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                2,
                2,
                "secondary",
                "Secondary score is still unresolved.",
                CognitiveRole.INFORMATION_NEED,
                lifecycle=CellLifecycle.WAITING,
                uncertainty=0.8,
            ),
            _thought(
                3,
                1,
                "primary",
                f"Primary score is {primary}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                2,
                "secondary",
                f"Secondary score is {secondary}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                3,
                "answer",
                f"Select primary score {primary}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                4,
                1,
                "primary",
                f"Primary score is {primary}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                4,
                2,
                "secondary",
                f"Secondary score is {secondary}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                4,
                3,
                "answer",
                f"Final selection is {primary}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(
                0,
                f"Primary: {DISPLAY_UNKNOWN_MARKER}; secondary: "
                f"{DISPLAY_UNKNOWN_MARKER}; selected: {DISPLAY_UNKNOWN_MARKER}.",
            ),
            DisplayTarget(
                1,
                f"Primary: {DISPLAY_UNKNOWN_MARKER}; secondary: "
                f"{DISPLAY_UNKNOWN_MARKER}; selected: {DISPLAY_UNKNOWN_MARKER}.",
            ),
            DisplayTarget(
                2,
                f"Primary: {primary}; secondary: {DISPLAY_UNKNOWN_MARKER}; "
                f"selected: {primary} (provisional).",
            ),
            DisplayTarget(3, final),
            DisplayTarget(4, final),
        ),
        metadata=_metadata(config, "parallel_compare", index),
    )


def _streaming_evidence(
    rng: random.Random, config: CuratedV4Config, index: int
) -> TrajectoryExample:
    first = f"chunk-{rng.randrange(100, 999)}"
    second = f"chunk-{rng.randrange(100, 999)}"
    topic = f"topic-{index:04d}"
    final = f"Evidence: {first} + {second}."
    source = "evidence_stream"
    return TrajectoryExample(
        example_id=f"curated-v4-stream-{index:05d}",
        prompt=f"Collect both evidence chunks for {topic} in arrival order.",
        target_display=final,
        source_descriptors=(
            {
                **_mapping_source(source, "Stream ordered evidence chunks.", "topic"),
                "cacheable": False,
                "dynamic": True,
                "streamable": True,
            },
        ),
        events=(
            ExternalEvent(source, first, 2, version="1", arguments={"topic": topic}),
            ExternalEvent(source, second, 4, version="2", arguments={"topic": topic}),
        ),
        binding_targets=(
            BindingTarget(
                "stream",
                source,
                1,
                1,
                arguments={"topic": topic},
                freshness=FreshnessDemand.ALWAYS,
                owner_cell_id="stream",
                target_cells=(ObjectRef.cell("stream"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "stream", source),
            _observe_grounding(2, "stream", source),
            _observe_grounding(4, "stream", source),
        ),
        thought_targets=(
            *(
                _thought(
                    step,
                    0,
                    "goal",
                    "Collect two ordered stream chunks.",
                    CognitiveRole.PLAN,
                    lifecycle=(CellLifecycle.ACTIVE if step == 0 else CellLifecycle.STABLE),
                )
                for step in range(6)
            ),
            _thought(
                1,
                1,
                "stream",
                "Need the stream contents.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                1,
                "stream",
                f"First chunk is {first}; another is expected.",
                CognitiveRole.PERCEPT,
                uncertainty=0.5,
            ),
            _thought(
                3,
                1,
                "stream",
                f"First chunk remains {first}; second is outstanding.",
                CognitiveRole.INFORMATION_NEED,
                lifecycle=CellLifecycle.WAITING,
                uncertainty=0.7,
            ),
            _thought(
                4,
                1,
                "stream",
                f"Chunks are {first} then {second}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                1,
                "stream",
                f"Final ordered chunks are {first} then {second}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"Evidence: {DISPLAY_UNKNOWN_MARKER} + {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(1, f"Evidence: {DISPLAY_UNKNOWN_MARKER} + {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(2, f"Evidence: {first} + {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(3, f"Evidence: {first} + {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(4, final),
            DisplayTarget(5, final),
        ),
        metadata=_metadata(config, "streaming_evidence", index),
    )


def _dynamic_refresh(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    first = rng.randrange(10, 80)
    second = first + rng.randrange(1, 12)
    key = f"counter-{index:04d}"
    source = "live_counter"
    final = f"Current counter: {second}."
    return TrajectoryExample(
        example_id=f"curated-v4-refresh-{index:05d}",
        prompt=f"Track {key} and return the newest value after the scheduled refresh.",
        target_display=final,
        source_descriptors=(
            {
                **_mapping_source(source, "Read a versioned live counter."),
                "cacheable": False,
                "dynamic": True,
            },
        ),
        events=(
            ExternalEvent(source, first, 2, version="1", arguments={"key": key}),
            ExternalEvent(source, second, 4, version="2", arguments={"key": key}),
        ),
        binding_targets=(
            BindingTarget(
                "counter",
                source,
                1,
                1,
                arguments={"key": key},
                freshness=FreshnessDemand.ALWAYS,
                owner_cell_id="counter",
                target_cells=(ObjectRef.cell("counter"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "counter", source),
            _observe_grounding(2, "counter", source),
            _observe_grounding(4, "counter", source),
        ),
        thought_targets=(
            *(
                _thought(
                    step,
                    0,
                    "goal",
                    "Return the newest observed counter version.",
                    CognitiveRole.CONSTRAINT,
                    lifecycle=(CellLifecycle.ACTIVE if step == 0 else CellLifecycle.STABLE),
                )
                for step in range(6)
            ),
            _thought(
                1,
                1,
                "counter",
                "Need the current counter and subsequent refresh.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                1,
                "counter",
                f"Current observed value is {first}; refresh remains live.",
                CognitiveRole.PERCEPT,
                uncertainty=0.4,
            ),
            _thought(
                3,
                1,
                "counter",
                f"Known value is {first}; waiting for refresh.",
                CognitiveRole.INFORMATION_NEED,
                lifecycle=CellLifecycle.WAITING,
                uncertainty=0.7,
            ),
            _thought(
                4,
                1,
                "counter",
                f"Newest value is {second}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                1,
                "counter",
                f"Final newest value remains {second}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"Current counter: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(1, f"Current counter: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(2, f"Current counter: {first}; refresh: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(3, f"Current counter: {first}; refresh: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(4, final),
            DisplayTarget(5, final),
        ),
        metadata=_metadata(config, "dynamic_refresh", index),
    )


def _authoritative_correction(
    rng: random.Random, config: CuratedV4Config, index: int
) -> TrajectoryExample:
    correct = "enabled" if index % 2 else "disabled"
    wrong = "disabled" if correct == "enabled" else "enabled"
    key = f"feature-{index:04d}"
    final = f"Decision: {correct}."
    return TrajectoryExample(
        example_id=f"curated-v4-correction-{index:05d}",
        prompt=(
            f"A cached guess says {key} is {wrong}. "
            "Verify against the authoritative service and audit before answering."
        ),
        target_display=final,
        protected_facts={"cached_guess": wrong},
        source_descriptors=(
            _mapping_source("authority", "Read the authoritative feature state."),
            _mapping_source("audit", "Independently confirm the feature state."),
        ),
        events=(
            ExternalEvent("authority", correct, 2, arguments={"key": key}),
            ExternalEvent("audit", correct, 4, arguments={"key": key}),
        ),
        binding_targets=(
            BindingTarget(
                "authority",
                "authority",
                1,
                1,
                arguments={"key": key},
                owner_cell_id="answer",
                target_cells=(ObjectRef.cell("answer"),),
            ),
            BindingTarget(
                "audit",
                "audit",
                3,
                3,
                arguments={"key": key},
                owner_cell_id="confirm",
                target_cells=(ObjectRef.cell("confirm"), ObjectRef.cell("answer")),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "answer", "authority"),
            _observe_grounding(2, "answer", "authority"),
            _request_grounding(3, "confirm", "audit"),
            _observe_grounding(4, "confirm", "audit"),
        ),
        thought_targets=(
            _thought(
                0,
                0,
                "answer",
                f"Cached hypothesis is {wrong}; it is not authoritative.",
                CognitiveRole.HYPOTHESIS,
                uncertainty=0.75,
                noise=0.5,
            ),
            _thought(
                1,
                0,
                "answer",
                f"Verify cached hypothesis {wrong} with authority.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                0,
                "answer",
                f"Authority corrects the state to {correct}.",
                CognitiveRole.HYPOTHESIS,
                uncertainty=0.25,
                noise=0.4,
            ),
            _thought(
                3,
                0,
                "answer",
                f"Candidate is {correct}; require independent confirmation.",
                CognitiveRole.HYPOTHESIS,
                uncertainty=0.3,
            ),
            _thought(
                3,
                1,
                "confirm",
                f"Need audit confirmation for {key}.",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.8,
            ),
            _thought(
                4,
                0,
                "answer",
                f"Authority and audit agree on {correct}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                4,
                1,
                "confirm",
                f"Audit confirms {correct}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                0,
                "answer",
                f"Final decision remains {correct}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                5,
                1,
                "confirm",
                f"Confirmation remains {correct}.",
                CognitiveRole.PERCEPT,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(
                0, f"Decision: {wrong} (provisional); verification: {DISPLAY_UNKNOWN_MARKER}."
            ),
            DisplayTarget(
                1, f"Decision: {wrong} (provisional); verification: {DISPLAY_UNKNOWN_MARKER}."
            ),
            DisplayTarget(2, f"Decision: {correct}; confirmation: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(3, f"Decision: {correct}; confirmation: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(4, final),
            DisplayTarget(5, final),
        ),
        metadata=_metadata(config, "authoritative_correction", index),
    )


def _no_tool_reasoning(
    rng: random.Random, config: CuratedV4Config, index: int
) -> TrajectoryExample:
    left = rng.randrange(2, 20)
    middle = left + rng.randrange(1, 10)
    right = middle + rng.randrange(1, 10)
    answer = "yes"
    final = "Answer: yes."
    return TrajectoryExample(
        example_id=f"curated-v4-no-tool-{index:05d}",
        prompt=f"Given A={left}, B={middle}, C={right}, is A < B < C? Answer yes or no.",
        target_display=final,
        thought_targets=(
            _thought(
                0,
                0,
                "chain",
                "Compare A with B, then B with C.",
                CognitiveRole.PLAN,
                uncertainty=0.5,
            ),
            _thought(
                1,
                0,
                "chain",
                f"Both inequalities hold: {left} < {middle} and {middle} < {right}.",
                CognitiveRole.HYPOTHESIS,
                uncertainty=0.1,
            ),
            _thought(
                2,
                0,
                "chain",
                f"The ordered chain is true, so the answer is {answer}.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"Answer: {DISPLAY_UNKNOWN_MARKER}."),
            DisplayTarget(1, final),
            DisplayTarget(2, final),
        ),
        metadata=_metadata(config, "no_tool_reasoning", index),
    )


def _tool_restraint(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    value = rng.randrange(3, 50)
    final = f"Answer: {value}."
    return TrajectoryExample(
        example_id=f"curated-v4-restraint-{index:05d}",
        prompt=(
            f"The answer is explicitly given as {value}. "
            "Return it directly; the catalog tool is irrelevant."
        ),
        target_display=final,
        source_descriptors=(_mapping_source("irrelevant_catalog", "Unrelated lookup source."),),
        thought_targets=(
            _thought(
                0,
                0,
                "answer",
                f"The prompt directly supplies {value}; no lookup is needed.",
                CognitiveRole.CONSTRAINT,
                uncertainty=0.05,
            ),
            _thought(
                1,
                0,
                "answer",
                f"Return the supplied value {value} without tool use.",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(DisplayTarget(0, final), DisplayTarget(1, final)),
        metadata=_metadata(config, "tool_restraint", index),
    )


def _zh_lookup(rng: random.Random, config: CuratedV4Config, index: int) -> TrajectoryExample:
    version = f"v{rng.randrange(2, 10)}.{rng.randrange(0, 10)}"
    component = f"模块-{index:04d}"
    final = f"当前版本：{version}。"
    source = "版本注册表"
    return TrajectoryExample(
        example_id=f"curated-v4-zh-{index:05d}",
        prompt=f"查询 {component} 的当前发布版本。注册表返回前不要猜测。",
        target_display=final,
        source_descriptors=(_mapping_source(source, "查询组件的权威发布版本。", "组件"),),
        events=(ExternalEvent(source, version, 2, arguments={"组件": component}),),
        binding_targets=(
            BindingTarget(
                "version",
                source,
                1,
                1,
                arguments={"组件": component},
                owner_cell_id="version",
                target_cells=(ObjectRef.cell("version"),),
            ),
        ),
        grounding_targets=(
            _request_grounding(1, "version", source),
            _observe_grounding(2, "version", source),
        ),
        thought_targets=(
            *(
                _thought(
                    step,
                    0,
                    "goal",
                    "必须使用版本注册表中的权威值。",
                    CognitiveRole.CONSTRAINT,
                    lifecycle=(CellLifecycle.ACTIVE if step == 0 else CellLifecycle.STABLE),
                )
                for step in range(4)
            ),
            _thought(
                1,
                1,
                "version",
                f"需要查询 {component} 的当前版本。",
                CognitiveRole.INFORMATION_NEED,
                uncertainty=0.9,
            ),
            _thought(
                2,
                1,
                "version",
                f"注册表返回当前版本 {version}。",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
            _thought(
                3,
                1,
                "version",
                f"最终版本保持为 {version}。",
                CognitiveRole.CONCLUSION,
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
        display_targets=(
            DisplayTarget(0, f"当前版本：{DISPLAY_UNKNOWN_MARKER}。"),
            DisplayTarget(1, f"当前版本：{DISPLAY_UNKNOWN_MARKER}。"),
            DisplayTarget(2, final),
            DisplayTarget(3, final),
        ),
        metadata=_metadata(config, "zh_lookup", index),
    )

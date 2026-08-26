from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from cid.data import TrajectoryExample
from cid.evaluation import ReplayEvaluationResult, RuntimeTaskEvaluation, run_replay_case
from cid.grounding import ObjectKind, ObjectRef
from cid.model.encoding import ILLaDATextEncoder, stable_text
from cid.model.illada import ILLaDACIDAdapter
from cid.model.materialize import (
    AnchorCandidate,
    ArgumentCandidate,
    ClosedWorldMaterializationCatalog,
    ObjectCandidate,
)
from cid.model.policy import (
    ILLaDAContextTensorizer,
    ILLaDANeuralPolicy,
    ILLaDANeuralPolicyConfig,
)
from cid.runtime.engine import RuntimeConfig
from cid.state import CognitiveField, DisplayCanvas


@dataclass(frozen=True, slots=True)
class NeuralBenchmarkCaseResult:
    example_id: str
    final_text: str
    final_display_ids: tuple[int, ...]
    runtime_steps: int
    evaluation: RuntimeTaskEvaluation
    trace_events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "final_text": self.final_text,
            "final_display_ids": list(self.final_display_ids),
            "runtime_steps": self.runtime_steps,
            "evaluation": asdict(self.evaluation),
            "trace_events": list(self.trace_events),
        }


async def run_neural_benchmark_case(
    adapter: ILLaDACIDAdapter,
    tokenizer: Any,
    example: TrajectoryExample,
    *,
    text_encoder: ILLaDATextEncoder | None = None,
    forward_model: torch.nn.Module | None = None,
    seed_teacher_state: bool = False,
    denoising_steps: int = 8,
    max_steps: int | None = None,
) -> NeuralBenchmarkCaseResult:
    encoder = text_encoder or ILLaDATextEncoder(adapter, tokenizer)
    tensorizer = ILLaDAContextTensorizer(
        adapter,
        tokenizer,
        text_encoder=encoder,
    )
    policy = ILLaDANeuralPolicy(
        adapter,
        tensorizer,
        catalog=build_materialization_catalog(example, encoder),
        config=ILLaDANeuralPolicyConfig(denoising_steps=denoising_steps),
        forward_model=forward_model,
    )
    expected_ids_tensor = encoder.tokenize(example.target_display, add_special_tokens=False)
    expected_ids = tuple(int(token) for token in expected_ids_tensor[0].tolist())
    if not expected_ids:
        raise ValueError("benchmark target display must tokenize to at least one token")
    canvas_tokens = adapter.config.display_canvas_tokens
    if len(expected_ids) + 1 > canvas_tokens:
        raise ValueError(
            "benchmark target display plus EOS exceeds the fixed runtime display canvas"
        )

    thought = (
        teacher_seed_thought(example, adapter, encoder)
        if seed_teacher_state
        else CognitiveField.empty(
            capacity=adapter.config.max_thought_slots,
            width=adapter.d_model,
        )
    )
    display = DisplayCanvas.masked(
        length=canvas_tokens,
        mask_token_id=adapter.mask_token_id,
        eos_token_id=adapter.eos_token_id,
    )
    replay: ReplayEvaluationResult = await run_replay_case(
        policy,
        example,
        thought=thought,
        display=display,
        expected_display_ids=expected_ids,
        runtime_config=(None if max_steps is None else RuntimeConfig(max_steps=max_steps)),
    )
    final_ids = tuple(int(token) for token in replay.runtime.display.visible_token_ids)
    return NeuralBenchmarkCaseResult(
        example_id=example.example_id,
        final_text=tokenizer.decode(list(final_ids), skip_special_tokens=True),
        final_display_ids=final_ids,
        runtime_steps=replay.runtime.steps,
        evaluation=replay.evaluation,
        trace_events=replay.runtime.trace.to_dicts(),
    )


def build_materialization_catalog(
    example: TrajectoryExample,
    text_encoder: ILLaDATextEncoder,
) -> ClosedWorldMaterializationCatalog:
    arguments: list[ArgumentCandidate] = []
    seen_arguments: set[tuple[str, str, str]] = set()
    for binding in example.binding_targets:
        for name, value in binding.arguments.items():
            encoded = stable_text(value)
            key = (binding.source, str(name), encoded)
            if key in seen_arguments:
                continue
            seen_arguments.add(key)
            arguments.append(
                ArgumentCandidate(
                    source=binding.source,
                    name=str(name),
                    value=value,
                    embedding=text_encoder.encode_one(encoded, detach=True),
                )
            )

    anchors = tuple(
        AnchorCandidate(
            anchor=entry.anchor,
            embedding=text_encoder.encode_one(entry.anchor.canonical_key, detach=True),
        )
        for entry in example.grounding_catalog
    )

    object_refs: set[ObjectRef] = set()
    object_refs.update(
        ObjectRef.source(str(descriptor["name"]))
        for descriptor in example.source_descriptors
    )
    object_refs.update(ObjectRef.fact(str(key)) for key in example.protected_facts)
    object_refs.update(
        ObjectRef.cell(target.cell_id)
        for target in example.thought_targets
    )
    object_refs.update(entry.anchor.ref for entry in example.grounding_catalog)
    for binding in example.binding_targets:
        object_refs.update(binding.target_cells)
        object_refs.update(binding.target_display)
    for grounding in example.grounding_targets:
        object_refs.update(link.target for link in grounding.links)
        object_refs.update(anchor.ref for anchor in grounding.anchors)

    objects = tuple(
        ObjectCandidate(
            ref=ref,
            embedding=text_encoder.encode_one(_object_text(ref), detach=True),
        )
        for ref in sorted(object_refs, key=_object_sort_key)
    )
    return ClosedWorldMaterializationCatalog(
        arguments=tuple(arguments),
        anchors=anchors,
        objects=objects,
    )


def teacher_seed_thought(
    example: TrajectoryExample,
    adapter: ILLaDACIDAdapter,
    text_encoder: ILLaDATextEncoder,
) -> CognitiveField:
    field = CognitiveField.empty(
        capacity=adapter.config.max_thought_slots,
        width=adapter.d_model,
    )
    initial = sorted(
        (target for target in example.thought_targets if target.step == 0),
        key=lambda target: _cell_serial(target.cell_id),
    )
    for target in initial:
        semantic = tuple(
            float(value)
            for value in text_encoder.encode_one(target.semantic_text, detach=True).float().tolist()
        )
        field, cell_id = field.allocate(
            semantic=semantic,
            roles=dict(target.roles),
            uncertainty=target.uncertainty,
            noise=target.noise,
            lifecycle=target.lifecycle,
            slot=target.slot,
        )
        if cell_id != target.cell_id:
            raise ValueError(
                "teacher-seeded benchmark requires step-0 cell IDs to follow "
                "runtime allocation order"
            )
    return field


def _cell_serial(cell_id: str) -> int:
    if len(cell_id) < 2 or cell_id[0] != "c" or not cell_id[1:].isdigit():
        raise ValueError("teacher-seeded benchmark requires c<N> cell identifiers")
    return int(cell_id[1:])


def _object_sort_key(ref: ObjectRef) -> tuple[str, str, tuple[int, int]]:
    return (ref.kind.value, ref.identifier, ref.span or (-1, -1))


def _object_text(ref: ObjectRef) -> str:
    if ref.kind is ObjectKind.DISPLAY_SPAN:
        if ref.span is None:
            raise ValueError("display-span object candidate requires a span")
        return f"{ref.kind.value}:{ref.span[0]}:{ref.span[1]}"
    return f"{ref.kind.value}:{ref.identifier}"

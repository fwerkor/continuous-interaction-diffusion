from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, TypeVar

import torch
from torch import Tensor

from cid.contracts import (
    FreshnessDemand,
    InformationNeed,
    ModelContext,
    ModelUpdate,
    SourceDescriptor,
)
from cid.defaults import (
    DEFAULT_ALLOCATION_THRESHOLD,
    DEFAULT_ANCHOR_PRESENCE_THRESHOLD,
    DEFAULT_ARGUMENT_PRESENCE_THRESHOLD,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_LINK_PRESENCE_THRESHOLD,
    DEFAULT_MATERIALIZED_MAX_AGE_S,
    DEFAULT_MAX_ALLOCATIONS_PER_STEP,
    DEFAULT_NEED_TARGET_CELL_THRESHOLD,
    DEFAULT_NEED_TARGET_DISPLAY_THRESHOLD,
    DEFAULT_NEED_THRESHOLD,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectKind, ObjectRef
from cid.lifecycle import MODELED_LIFECYCLES
from cid.model.allocation import prefix_allocation_mask
from cid.model.native_engine import cuda_engine
from cid.model.tensors import CIDTensorOutput, DeviceSemantic, semantic_as_tensor
from cid.state import CellLifecycle, CognitiveField, CognitiveRole, DisplayCanvas

T = TypeVar("T")


class RevisionAction(IntEnum):
    KEEP = 0
    REOPEN = 1
    STABILIZE = 2


@dataclass(frozen=True, slots=True)
class ArgumentCandidate:
    source: str
    name: str
    value: Any
    embedding: Tensor


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    anchor: Anchor
    embedding: Tensor


@dataclass(frozen=True, slots=True)
class ObjectCandidate:
    ref: ObjectRef
    embedding: Tensor


@dataclass(frozen=True, slots=True)
class ClosedWorldMaterializationCatalog:
    arguments: tuple[ArgumentCandidate, ...] = ()
    anchors: tuple[AnchorCandidate, ...] = ()
    objects: tuple[ObjectCandidate, ...] = ()

    def resolve_argument(
        self,
        source: str,
        name: str,
        query: Tensor,
        *,
        min_similarity: float,
    ) -> Any | None:
        candidates = tuple(
            (candidate.value, candidate.embedding)
            for candidate in self.arguments
            if candidate.source == source and candidate.name == name
        )
        return _nearest_value(query, candidates, min_similarity=min_similarity)

    def resolve_anchor(
        self,
        kind: AnchorKind,
        query: Tensor,
        *,
        min_similarity: float,
    ) -> Anchor | None:
        candidates = tuple(
            (candidate.anchor, candidate.embedding)
            for candidate in self.anchors
            if candidate.anchor.kind is kind
        )
        return _nearest_value(query, candidates, min_similarity=min_similarity)

    def resolve_object(
        self,
        kind: ObjectKind,
        query: Tensor,
        *,
        min_similarity: float,
    ) -> ObjectRef | None:
        candidates = tuple(
            (candidate.ref, candidate.embedding)
            for candidate in self.objects
            if candidate.ref.kind is kind
        )
        return _nearest_value(query, candidates, min_similarity=min_similarity)


@dataclass(frozen=True, slots=True)
class CIDMaterializerConfig:
    allocation_threshold: float = DEFAULT_ALLOCATION_THRESHOLD
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    need_threshold: float = DEFAULT_NEED_THRESHOLD
    need_target_cell_threshold: float = DEFAULT_NEED_TARGET_CELL_THRESHOLD
    need_target_display_threshold: float = DEFAULT_NEED_TARGET_DISPLAY_THRESHOLD
    argument_presence_threshold: float = DEFAULT_ARGUMENT_PRESENCE_THRESHOLD
    anchor_presence_threshold: float = DEFAULT_ANCHOR_PRESENCE_THRESHOLD
    link_presence_threshold: float = DEFAULT_LINK_PRESENCE_THRESHOLD
    retrieval_similarity_threshold: float = DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD
    max_allocations_per_step: int = DEFAULT_MAX_ALLOCATIONS_PER_STEP
    max_age_s: float = DEFAULT_MATERIALIZED_MAX_AGE_S

    def __post_init__(self) -> None:
        for name in (
            "allocation_threshold",
            "convergence_threshold",
            "need_threshold",
            "need_target_cell_threshold",
            "need_target_display_threshold",
            "argument_presence_threshold",
            "anchor_presence_threshold",
            "link_presence_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.retrieval_similarity_threshold <= 1.0:
            raise ValueError("retrieval_similarity_threshold must be in [-1, 1]")
        if self.max_allocations_per_step <= 0:
            raise ValueError("max_allocations_per_step must be positive")
        if self.max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")


def decode_need_target_display(
    probabilities: Tensor,
    *,
    active_length: int,
    threshold: float,
) -> tuple[ObjectRef, ...]:
    if probabilities.ndim != 1:
        raise ValueError("need display-route probabilities must be one-dimensional")
    if active_length < 0:
        raise ValueError("need display-route active length must be non-negative")
    selected = probabilities[:active_length] >= threshold
    spans: list[ObjectRef] = []
    start: int | None = None
    for index, active in enumerate(selected.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append(ObjectRef.display_span(start, index))
            start = None
    if start is not None:
        spans.append(ObjectRef.display_span(start, int(selected.numel())))
    return tuple(spans)


class CIDMaterializer:
    def __init__(self, config: CIDMaterializerConfig | None = None) -> None:
        self.config = config or CIDMaterializerConfig()

    def materialize(
        self,
        output: CIDTensorOutput,
        context: ModelContext,
        *,
        catalog: ClosedWorldMaterializationCatalog | None = None,
        batch_index: int = 0,
        display_token_ids: tuple[int, ...] | None = None,
    ) -> ModelUpdate:
        catalog = catalog or ClosedWorldMaterializationCatalog()
        self._validate_output(output, context, batch_index)

        thought = self._materialize_cells(output, context.thought, batch_index)
        thought = self._materialize_grounding(output, thought, catalog, batch_index)
        display = self._materialize_display(context.display, display_token_ids)
        needs = self.materialize_needs(
            output,
            sources=context.sources,
            thought=thought,
            display=display,
            catalog=catalog,
            batch_index=batch_index,
        )
        reopen_cells = self._materialize_revisions(
            output,
            previous=context.thought,
            thought=thought,
            batch_index=batch_index,
        )
        convergence = float(
            torch.sigmoid(output.convergence_logits[batch_index].float()).detach()
        )
        display_stable = display.token_ids == context.display.token_ids
        display_has_boundary = (
            display.eos_token_id is None or display.eos_token_id in display.token_ids
        )
        display_nonempty = display.realized_length > 0

        return ModelUpdate(
            thought=thought,
            display=display,
            needs=needs,
            reopen_cells=reopen_cells,
            diagnostics=(
                {
                    "convergence_score": convergence,
                    "convergence_threshold": self.config.convergence_threshold,
                    "need_threshold": self.config.need_threshold,
                    "need_eligible_slots": [
                        slot for slot, cell in enumerate(thought.cells) if cell.live
                    ],
                    "allocation_eligible_slots": [
                        slot for slot, cell in enumerate(context.thought.cells)
                        if not cell.occupied
                    ],
                    "allocation_threshold": self.config.allocation_threshold,
                    "max_allocations_per_step": self.config.max_allocations_per_step,
                    "need_scores": torch.sigmoid(
                        output.need_logits[batch_index].float()
                    ).detach().cpu().tolist(),
                    "allocation_scores": torch.sigmoid(
                        output.allocation_logits[batch_index].float()
                    ).detach().cpu().tolist(),
                    "source_names": [source.name for source in context.sources],
                    "source_scores": torch.softmax(
                        output.source_logits[batch_index].float(), dim=-1
                    ).detach().cpu().tolist(),
                    "display_stable": display_stable,
                    "display_has_boundary": display_has_boundary,
                    "display_nonempty": display_nonempty,
                    "display_resolved": display.unresolved == 0,
                    "diffusion_step": context.diffusion_step,
                } if context.trace_details else {}
            ),
            equilibrium=convergence >= self.config.convergence_threshold,
            converged=(
                display.unresolved == 0
                and display_stable
                and display_has_boundary
                and display_nonempty
                and convergence >= self.config.convergence_threshold
            ),
        )

    def _materialize_cells(
        self,
        output: CIDTensorOutput,
        previous: CognitiveField,
        batch_index: int,
    ) -> CognitiveField:
        occupancy = torch.tensor(
            [[cell.occupied for cell in previous.cells]],
            dtype=torch.bool,
            device=output.allocation_logits.device,
        )
        selected = prefix_allocation_mask(
            occupancy,
            output.allocation_logits[batch_index : batch_index + 1].detach(),
            threshold=self.config.allocation_threshold,
            max_allocations=self.config.max_allocations_per_step,
        )[0]

        thought_semantic = output.thought_semantic[batch_index].detach()
        semantic_width = thought_semantic.shape[-1]
        sketch_samples = min(12, semantic_width)
        sketch_positions = _semantic_sketch_positions(semantic_width, sketch_samples)
        sketch_indexes = torch.tensor(
            sketch_positions,
            dtype=torch.long,
            device=thought_semantic.device,
        )

        role_order = tuple(CognitiveRole)
        lifecycle_order = MODELED_LIFECYCLES
        role_count = len(role_order)
        engine = cuda_engine(
            thought_semantic,
            capability="materialize_cell_snapshot",
        )
        native_snapshot = (
            engine is not None
            and output.role_logits.dtype == thought_semantic.dtype
            and output.uncertainty.dtype == thought_semantic.dtype
            and output.noise_delta.dtype == thought_semantic.dtype
            and output.lifecycle_logits.dtype == thought_semantic.dtype
        )
        if native_snapshot:
            snapshot_rows = engine.materialize_cell_snapshot(
                output.thought_semantic[batch_index : batch_index + 1].detach(),
                output.role_logits[batch_index : batch_index + 1].detach(),
                output.uncertainty[batch_index : batch_index + 1].detach(),
                output.noise_delta[batch_index : batch_index + 1].detach(),
                output.lifecycle_logits[batch_index : batch_index + 1].detach(),
                selected.unsqueeze(0),
                sketch_indexes,
            )[0].cpu().tolist()
            selected_slots = [
                slot for slot, row in enumerate(snapshot_rows) if row[0] > 0.5
            ]
            lifecycle_values = [int(row[1]) for row in snapshot_rows]
            uncertainty_values = [float(row[2]) for row in snapshot_rows]
            noise_delta_values = [float(row[3]) for row in snapshot_rows]
            role_values = [row[4 : 4 + role_count] for row in snapshot_rows]
            semantic_sketches = [row[4 + role_count :] for row in snapshot_rows]
        else:
            semantic_sketches = (
                thought_semantic.index_select(-1, sketch_indexes)
                .float()
                .cpu()
                .tolist()
            )
            selected_slots = selected.nonzero(as_tuple=False).flatten().cpu().tolist()
            role_values = (
                torch.sigmoid(output.role_logits[batch_index].float())
                .detach()
                .cpu()
                .tolist()
            )
            uncertainty_values = (
                output.uncertainty[batch_index].detach().float().cpu().flatten().tolist()
            )
            noise_delta_values = (
                output.noise_delta[batch_index].detach().float().cpu().flatten().tolist()
            )
            lifecycle_values = (
                output.lifecycle_logits[batch_index].argmax(dim=-1).detach().cpu().tolist()
            )

        newly_allocated_slots = set(selected_slots)
        field = previous
        for slot in selected_slots:
            semantic = DeviceSemantic(
                thought_semantic[slot],
                sketch=tuple(round(float(value), 3) for value in semantic_sketches[slot]),
            )
            field, _ = field.allocate(slot=slot, semantic=semantic)

        cells = list(field.cells)

        for slot, cell in enumerate(cells):
            if not cell.occupied or cell.lifecycle is CellLifecycle.RETIRED:
                continue
            predicted_lifecycle = lifecycle_order[lifecycle_values[slot]]
            if slot in newly_allocated_slots and predicted_lifecycle not in (
                CellLifecycle.ACTIVE,
                CellLifecycle.WAITING,
            ):
                predicted_lifecycle = CellLifecycle.ACTIVE
            cells[slot] = replace(
                cell,
                semantic=DeviceSemantic(
                    thought_semantic[slot],
                    sketch=tuple(
                        round(float(value), 3) for value in semantic_sketches[slot]
                    ),
                ),
                roles={
                    role: role_values[slot][index] for index, role in enumerate(role_order)
                },
                uncertainty=uncertainty_values[slot],
                noise=max(0.0, min(1.0, cell.noise + noise_delta_values[slot])),
                lifecycle=predicted_lifecycle,
            )

        return CognitiveField(
            cells=tuple(cells),
            step=previous.step + 1,
            next_cell_serial=field.next_cell_serial,
        )

    def _materialize_grounding(
        self,
        output: CIDTensorOutput,
        thought: CognitiveField,
        catalog: ClosedWorldMaterializationCatalog,
        batch_index: int,
    ) -> CognitiveField:
        cells = list(thought.cells)
        anchor_presence = (
            torch.sigmoid(output.anchor_presence_logits[batch_index].float())
            .detach()
            .cpu()
        )
        anchor_kinds = (
            output.anchor_kind_logits[batch_index].argmax(dim=-1).detach().cpu()
        )
        link_presence = (
            torch.sigmoid(output.link_presence_logits[batch_index].float())
            .detach()
            .cpu()
        )
        link_relations = (
            output.link_relation_logits[batch_index].argmax(dim=-1).detach().cpu()
        )
        link_kinds = (
            output.link_target_kind_logits[batch_index].argmax(dim=-1).detach().cpu()
        )
        anchor_order = tuple(AnchorKind)
        relation_order = tuple(LinkRelation)
        object_order = tuple(ObjectKind)

        for slot, cell in enumerate(cells):
            if not cell.live:
                continue
            anchors: list[Anchor] = []
            seen_anchor_ids: set[str] = set()
            for anchor_slot in range(anchor_presence.shape[-1]):
                if (
                    float(anchor_presence[slot, anchor_slot])
                    < self.config.anchor_presence_threshold
                ):
                    continue
                kind = anchor_order[int(anchor_kinds[slot, anchor_slot])]
                anchor = catalog.resolve_anchor(
                    kind,
                    output.anchor_query[batch_index, slot, anchor_slot],
                    min_similarity=self.config.retrieval_similarity_threshold,
                )
                if anchor is not None and anchor.anchor_id not in seen_anchor_ids:
                    anchors.append(anchor)
                    seen_anchor_ids.add(anchor.anchor_id)

            links: list[CognitiveLink] = []
            seen_links: set[tuple[LinkRelation, ObjectRef]] = set()
            for link_slot in range(link_presence.shape[-1]):
                if float(link_presence[slot, link_slot]) < self.config.link_presence_threshold:
                    continue
                relation = relation_order[int(link_relations[slot, link_slot])]
                kind = object_order[int(link_kinds[slot, link_slot])]
                target = self._resolve_object(
                    kind,
                    output.link_target_query[batch_index, slot, link_slot],
                    thought,
                    catalog,
                )
                key = (relation, target) if target is not None else None
                if target is not None and key not in seen_links:
                    links.append(CognitiveLink(relation=relation, target=target))
                    seen_links.add(key)

            cells[slot] = replace(cell, anchors=tuple(anchors), links=tuple(links))

        return replace(thought, cells=tuple(cells))

    def _resolve_object(
        self,
        kind: ObjectKind,
        query: Tensor,
        thought: CognitiveField,
        catalog: ClosedWorldMaterializationCatalog,
    ) -> ObjectRef | None:
        if kind is ObjectKind.CELL:
            cell_candidates = tuple(
                (
                    ObjectRef.cell(cell.cell_id),
                    semantic_as_tensor(cell.semantic, device=query.device),
                )
                for cell in thought.cells
                if cell.live and cell.cell_id is not None
            )
            return _nearest_value(
                query,
                cell_candidates,
                min_similarity=self.config.retrieval_similarity_threshold,
            )
        return catalog.resolve_object(
            kind,
            query,
            min_similarity=self.config.retrieval_similarity_threshold,
        )

    def materialize_needs(
        self,
        output: CIDTensorOutput,
        *,
        sources: tuple[SourceDescriptor, ...],
        thought: CognitiveField,
        display: DisplayCanvas,
        catalog: ClosedWorldMaterializationCatalog | None = None,
        batch_index: int = 0,
    ) -> tuple[InformationNeed, ...]:
        """Decode information needs with the exact runtime materialization contract.

        Training closed-loop rollout uses this public entry point so spurious needs, wrong
        sources, and wrong arguments have the same downstream binding semantics as inference.
        The closed-world catalog only resolves model queries; it does not restrict which need
        slots are allowed to exist.
        """

        if not 0 <= batch_index < output.need_logits.shape[0]:
            raise IndexError("batch_index is outside model output")
        if output.need_logits.shape[1] != thought.capacity:
            raise ValueError("model need slot count does not match runtime TCT capacity")
        if output.source_logits.shape[-1] != len(sources):
            raise ValueError("model source count does not match runtime source descriptors")
        if output.need_target_display_logits.shape[-1] != len(display.token_ids):
            raise ValueError("need-to-display routing must match runtime display capacity")
        if not sources:
            return ()
        catalog = catalog or ClosedWorldMaterializationCatalog()
        need_probs = (
            torch.sigmoid(output.need_logits[batch_index].float()).detach().cpu()
        )
        source_probs = (
            torch.softmax(output.source_logits[batch_index].float(), dim=-1)
            .detach()
            .cpu()
        )
        argument_presence = (
            torch.sigmoid(output.argument_presence_logits[batch_index].float())
            .detach()
            .cpu()
        )
        refresh_actions = (
            output.refresh_logits[batch_index].argmax(dim=-1).detach().cpu()
        )
        freshness_order = tuple(FreshnessDemand)
        needs: list[InformationNeed] = []

        for slot, cell in enumerate(thought.cells):
            if not cell.live or cell.cell_id is None:
                continue
            for need_slot in range(need_probs.shape[-1]):
                confidence = float(need_probs[slot, need_slot])
                if confidence < self.config.need_threshold:
                    continue
                scores = {
                    descriptor.name: float(source_probs[slot, need_slot, source_index])
                    for source_index, descriptor in enumerate(sources)
                }
                selected_index = int(source_probs[slot, need_slot].argmax())
                source = sources[selected_index]
                arguments: dict[str, Any] = {}
                for argument_slot, descriptor in enumerate(source.arguments):
                    if argument_slot >= output.argument_query.shape[3]:
                        break
                    if (
                        float(argument_presence[slot, need_slot, argument_slot])
                        < self.config.argument_presence_threshold
                    ):
                        continue
                    value = catalog.resolve_argument(
                        source.name,
                        descriptor.name,
                        output.argument_query[batch_index, slot, need_slot, argument_slot],
                        min_similarity=self.config.retrieval_similarity_threshold,
                    )
                    if value is not None:
                        arguments[descriptor.name] = value

                freshness = freshness_order[int(refresh_actions[slot, need_slot])]
                target_cells = self._need_target_cells(
                    output, thought, slot=slot, need_slot=need_slot, batch_index=batch_index
                )
                target_display = self._need_target_display(
                    output, display, slot=slot, need_slot=need_slot, batch_index=batch_index
                )
                needs.append(
                    InformationNeed(
                        need_id=f"need:{cell.cell_id}:{need_slot}",
                        source_scores=scores,
                        arguments=arguments,
                        confidence=confidence,
                        freshness=freshness,
                        max_age_s=self.config.max_age_s
                        if freshness is FreshnessDemand.MAX_AGE
                        else None,
                        target_cells=target_cells,
                        target_display=target_display,
                        promote_to_fact=source.promote_results_to_fact,
                    )
                )
        return tuple(needs)

    def _need_target_cells(
        self,
        output: CIDTensorOutput,
        thought: CognitiveField,
        *,
        slot: int,
        need_slot: int,
        batch_index: int,
    ) -> tuple[ObjectRef, ...]:
        owner = thought.cells[slot]
        if not owner.live or owner.cell_id is None:
            return ()
        probabilities = torch.sigmoid(
            output.need_target_cell_logits[batch_index, slot, need_slot].float()
        ).detach()
        targets = [ObjectRef.cell(owner.cell_id)]
        for target_slot, target_cell in enumerate(thought.cells):
            if (
                not target_cell.live
                or target_cell.cell_id is None
                or target_cell.cell_id == owner.cell_id
                or float(probabilities[target_slot]) < self.config.need_target_cell_threshold
            ):
                continue
            targets.append(ObjectRef.cell(target_cell.cell_id))
        return tuple(targets)

    def _need_target_display(
        self,
        output: CIDTensorOutput,
        display: DisplayCanvas,
        *,
        slot: int,
        need_slot: int,
        batch_index: int,
    ) -> tuple[ObjectRef, ...]:
        probabilities = torch.sigmoid(
            output.need_target_display_logits[batch_index, slot, need_slot].float()
        ).detach()
        return decode_need_target_display(
            probabilities,
            active_length=display.active_span_length,
            threshold=self.config.need_target_display_threshold,
        )

    @staticmethod
    def _materialize_revisions(
        output: CIDTensorOutput,
        previous: CognitiveField,
        thought: CognitiveField,
        batch_index: int,
    ) -> tuple[ObjectRef, ...]:
        actions = output.revision_logits[batch_index].argmax(dim=-1).detach().cpu()
        previous_live = set(previous.live_cell_ids)
        return tuple(
            ObjectRef.cell(cell.cell_id)
            for slot, cell in enumerate(thought.cells)
            if cell.live
            and cell.cell_id is not None
            and cell.cell_id in previous_live
            and int(actions[slot]) == int(RevisionAction.REOPEN)
        )

    @staticmethod
    def _materialize_display(
        previous: DisplayCanvas,
        token_ids: tuple[int, ...] | None,
    ) -> DisplayCanvas:
        if token_ids is None:
            token_ids = previous.token_ids
        return previous.advance(token_ids)

    @staticmethod
    def _validate_output(
        output: CIDTensorOutput,
        context: ModelContext,
        batch_index: int,
    ) -> None:
        if not 0 <= batch_index < output.thought_semantic.shape[0]:
            raise IndexError("batch_index is outside model output")
        if output.convergence_logits.shape != (output.thought_semantic.shape[0],):
            raise ValueError("model convergence logits must have shape [batch]")
        if output.thought_semantic.shape[1] != context.thought.capacity:
            raise ValueError("model thought slot count does not match runtime TCT capacity")
        if output.display_logits.shape[1] != len(context.display.token_ids):
            raise ValueError("model display length does not match runtime display canvas")
        if output.source_logits.shape[-1] != len(context.sources):
            raise ValueError("model source count does not match runtime source descriptors")
        batch_size = output.thought_semantic.shape[0]
        thought_slots = context.thought.capacity
        if output.need_target_cell_logits.shape[:2] != (batch_size, thought_slots):
            raise ValueError("need-to-cell routing must match runtime TCT geometry")
        if output.need_target_cell_logits.shape[-1] != thought_slots:
            raise ValueError("need-to-cell routing target width must match runtime TCT capacity")
        if output.need_target_display_logits.shape[:2] != (batch_size, thought_slots):
            raise ValueError("need-to-display routing must match runtime TCT geometry")
        if output.need_target_display_logits.shape[-1] != len(context.display.token_ids):
            raise ValueError("need-to-display routing must match runtime display capacity")
        if output.need_target_cell_logits.shape[2] != output.need_logits.shape[-1]:
            raise ValueError("need-to-cell routing must match information-need slot capacity")
        if output.need_target_display_logits.shape[2] != output.need_logits.shape[-1]:
            raise ValueError("need-to-display routing must match information-need slot capacity")


def _nearest_value(
    query: Tensor,
    candidates: tuple[tuple[T, Tensor], ...],
    *,
    min_similarity: float,
) -> T | None:
    if not candidates:
        return None
    query_vector = query.detach().float().reshape(-1)
    candidate_vectors = []
    for _, embedding in candidates:
        vector = embedding.detach().to(device=query.device, dtype=torch.float32).reshape(-1)
        if vector.shape != query_vector.shape:
            raise ValueError("materialization candidate embedding width does not match query width")
        candidate_vectors.append(vector)
    matrix = torch.stack(candidate_vectors)
    scores = torch.nn.functional.cosine_similarity(matrix, query_vector.unsqueeze(0), dim=-1)
    index = int(scores.argmax())
    if float(scores[index]) < min_similarity:
        return None
    return candidates[index][0]


def _semantic_sketch_positions(width: int, samples: int) -> tuple[int, ...]:
    if width <= 0 or samples <= 0:
        raise ValueError("semantic sketch dimensions must be positive")
    if samples >= width:
        return tuple(range(width))
    last = width - 1
    return tuple(round(index * last / (samples - 1)) for index in range(samples))

from __future__ import annotations

from collections.abc import Sequence
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
from cid.model.tensors import (
    CIDTensorOutput,
    DeviceSemantic,
    DeviceSemanticBatch,
    semantic_as_tensor,
)
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
    return _decode_selected_display_spans(
        selected.tolist(),
        active_length=int(selected.numel()),
    )


def _decode_selected_display_spans(
    selected: Sequence[bool | float],
    *,
    active_length: int,
) -> tuple[ObjectRef, ...]:
    spans: list[ObjectRef] = []
    start: int | None = None
    for index, active in enumerate(selected[:active_length]):
        enabled = bool(active)
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append(ObjectRef.display_span(start, index))
            start = None
    if start is not None:
        spans.append(ObjectRef.display_span(start, active_length))
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
        occupied_flags = [cell.occupied for cell in previous.cells]
        occupancy = torch.tensor(
            [occupied_flags],
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
        becomes_full = (
            sum(occupied_flags) + len(selected_slots) >= previous.capacity
        )

        if becomes_full:
            field = previous
            for slot in selected_slots:
                sketch = tuple(
                    round(float(value), 3) for value in semantic_sketches[slot]
                )
                field, _ = field.allocate(
                    slot=slot,
                    semantic=DeviceSemantic(thought_semantic[slot], sketch=sketch),
                )

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
                sketch = tuple(
                    round(float(value), 3) for value in semantic_sketches[slot]
                )
                cells[slot] = replace(
                    cell,
                    semantic=DeviceSemantic(
                        thought_semantic[slot],
                        sketch=sketch,
                    ),
                    roles={
                        role: role_values[slot][index]
                        for index, role in enumerate(role_order)
                    },
                    uncertainty=uncertainty_values[slot],
                    noise=max(
                        0.0,
                        min(1.0, cell.noise + noise_delta_values[slot]),
                    ),
                    lifecycle=predicted_lifecycle,
                )

            return CognitiveField(
                cells=tuple(cells),
                step=previous.step + 1,
                next_cell_serial=field.next_cell_serial,
            )

        semantic_owner = DeviceSemanticBatch(thought_semantic)
        field = previous
        for slot in selected_slots:
            sketch = tuple(
                round(float(value), 3) for value in semantic_sketches[slot]
            )
            field, _ = field.allocate(
                slot=slot,
                semantic=semantic_owner.row(slot, sketch=sketch),
            )

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
            sketch = tuple(
                round(float(value), 3) for value in semantic_sketches[slot]
            )
            cells[slot] = replace(
                cell,
                semantic=semantic_owner.row(slot, sketch=sketch),
                roles={
                    role: role_values[slot][index]
                    for index, role in enumerate(role_order)
                },
                uncertainty=uncertainty_values[slot],
                noise=max(
                    0.0,
                    min(1.0, cell.noise + noise_delta_values[slot]),
                ),
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
        live_mask = torch.tensor(
            [cell.live for cell in cells],
            dtype=torch.bool,
            device=output.anchor_presence_logits.device,
        ).unsqueeze(-1)

        anchor_presence = torch.sigmoid(
            output.anchor_presence_logits[batch_index].float()
        ).detach()
        link_presence = torch.sigmoid(
            output.link_presence_logits[batch_index].float()
        ).detach()
        anchor_selected = (
            anchor_presence >= self.config.anchor_presence_threshold
        ) & live_mask
        link_selected = (
            link_presence >= self.config.link_presence_threshold
        ) & live_mask

        anchor_kinds = (
            output.anchor_kind_logits[batch_index].argmax(dim=-1).detach()
        )
        link_relations = (
            output.link_relation_logits[batch_index].argmax(dim=-1).detach()
        )
        link_kinds = (
            output.link_target_kind_logits[batch_index].argmax(dim=-1).detach()
        )

        anchor_slots = anchor_selected.shape[-1]
        combined_selected = torch.cat((anchor_selected, link_selected), dim=-1)
        combined_primary = torch.cat((anchor_kinds, link_relations), dim=-1)
        combined_secondary = torch.cat(
            (
                torch.full_like(anchor_kinds, -1),
                link_kinds,
            ),
            dim=-1,
        )
        positions = combined_selected.nonzero(as_tuple=False)
        if positions.shape[0] == 0:
            for slot, cell in enumerate(cells):
                if cell.live:
                    cells[slot] = replace(cell, anchors=(), links=())
            return replace(thought, cells=tuple(cells))

        slot_indices = positions[:, 0]
        control_indices = positions[:, 1]
        decisions = torch.cat(
            (
                positions.float(),
                combined_primary[slot_indices, control_indices]
                .float()
                .unsqueeze(-1),
                combined_secondary[slot_indices, control_indices]
                .float()
                .unsqueeze(-1),
            ),
            dim=-1,
        ).cpu().tolist()

        anchor_order = tuple(AnchorKind)
        relation_order = tuple(LinkRelation)
        object_order = tuple(ObjectKind)
        anchors_by_slot: list[list[Anchor]] = [[] for _ in cells]
        links_by_slot: list[list[CognitiveLink]] = [[] for _ in cells]
        seen_anchor_ids: list[set[str]] = [set() for _ in cells]
        seen_links: list[set[tuple[LinkRelation, ObjectRef]]] = [
            set() for _ in cells
        ]

        for slot_value, control_value, primary_value, secondary_value in decisions:
            slot = int(slot_value)
            control_slot = int(control_value)
            if control_slot < anchor_slots:
                anchor_slot = control_slot
                kind = anchor_order[int(primary_value)]
                anchor = catalog.resolve_anchor(
                    kind,
                    output.anchor_query[batch_index, slot, anchor_slot],
                    min_similarity=self.config.retrieval_similarity_threshold,
                )
                if (
                    anchor is not None
                    and anchor.anchor_id not in seen_anchor_ids[slot]
                ):
                    anchors_by_slot[slot].append(anchor)
                    seen_anchor_ids[slot].add(anchor.anchor_id)
                continue

            link_slot = control_slot - anchor_slots
            relation = relation_order[int(primary_value)]
            kind = object_order[int(secondary_value)]
            target = self._resolve_object(
                kind,
                output.link_target_query[batch_index, slot, link_slot],
                thought,
                catalog,
            )
            key = (relation, target) if target is not None else None
            if target is not None and key not in seen_links[slot]:
                links_by_slot[slot].append(
                    CognitiveLink(relation=relation, target=target)
                )
                seen_links[slot].add(key)

        for slot, cell in enumerate(cells):
            if cell.live:
                cells[slot] = replace(
                    cell,
                    anchors=tuple(anchors_by_slot[slot]),
                    links=tuple(links_by_slot[slot]),
                )
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
        device = output.need_logits.device
        live_mask = torch.tensor(
            [cell.live and cell.cell_id is not None for cell in thought.cells],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(-1)
        need_probs = torch.sigmoid(
            output.need_logits[batch_index].float()
        ).detach()
        selected = (need_probs >= self.config.need_threshold) & live_mask
        positions = selected.nonzero(as_tuple=False)
        if positions.shape[0] == 0:
            return ()

        slot_indices = positions[:, 0]
        need_indices = positions[:, 1]
        source_probs = torch.softmax(
            output.source_logits[
                batch_index,
                slot_indices,
                need_indices,
            ].float(),
            dim=-1,
        ).detach()
        refresh_actions = (
            output.refresh_logits[
                batch_index,
                slot_indices,
                need_indices,
            ]
            .argmax(dim=-1)
            .detach()
        )
        argument_selected = (
            torch.sigmoid(
                output.argument_presence_logits[
                    batch_index,
                    slot_indices,
                    need_indices,
                ].float()
            )
            >= self.config.argument_presence_threshold
        )
        target_cells_selected = (
            torch.sigmoid(
                output.need_target_cell_logits[
                    batch_index,
                    slot_indices,
                    need_indices,
                ].float()
            )
            >= self.config.need_target_cell_threshold
        )
        target_display_selected = (
            torch.sigmoid(
                output.need_target_display_logits[
                    batch_index,
                    slot_indices,
                    need_indices,
                ].float()
            )
            >= self.config.need_target_display_threshold
        )

        snapshot = torch.cat(
            (
                positions.float(),
                need_probs[slot_indices, need_indices].unsqueeze(-1),
                source_probs,
                refresh_actions.float().unsqueeze(-1),
                argument_selected.float(),
                target_cells_selected.float(),
                target_display_selected.float(),
            ),
            dim=-1,
        ).cpu().tolist()

        freshness_order = tuple(FreshnessDemand)
        source_count = len(sources)
        argument_slots = output.argument_presence_logits.shape[-1]
        cell_slots = thought.capacity
        display_slots = len(display.token_ids)
        source_start = 3
        refresh_index = source_start + source_count
        argument_start = refresh_index + 1
        cell_start = argument_start + argument_slots
        display_start = cell_start + cell_slots
        display_end = display_start + display_slots

        needs: list[InformationNeed] = []
        for row in snapshot:
            slot = int(row[0])
            need_slot = int(row[1])
            confidence = float(row[2])
            cell = thought.cells[slot]
            if not cell.live or cell.cell_id is None:
                continue

            source_values = row[source_start:refresh_index]
            scores = {
                descriptor.name: float(source_values[source_index])
                for source_index, descriptor in enumerate(sources)
            }
            selected_index = max(
                range(source_count),
                key=source_values.__getitem__,
            )
            source = sources[selected_index]

            arguments: dict[str, Any] = {}
            argument_values = row[argument_start:cell_start]
            for argument_slot, descriptor in enumerate(source.arguments):
                if argument_slot >= argument_slots:
                    break
                if argument_values[argument_slot] < 0.5:
                    continue
                value = catalog.resolve_argument(
                    source.name,
                    descriptor.name,
                    output.argument_query[
                        batch_index,
                        slot,
                        need_slot,
                        argument_slot,
                    ],
                    min_similarity=self.config.retrieval_similarity_threshold,
                )
                if value is not None:
                    arguments[descriptor.name] = value

            targets = [ObjectRef.cell(cell.cell_id)]
            cell_values = row[cell_start:display_start]
            for target_slot, active in enumerate(cell_values):
                target_cell = thought.cells[target_slot]
                if (
                    active < 0.5
                    or not target_cell.live
                    or target_cell.cell_id is None
                    or target_cell.cell_id == cell.cell_id
                ):
                    continue
                targets.append(ObjectRef.cell(target_cell.cell_id))

            display_values = row[display_start:display_end]
            target_display = _decode_selected_display_spans(
                display_values,
                active_length=display.active_span_length,
            )

            freshness = freshness_order[int(row[refresh_index])]
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
                    target_cells=tuple(targets),
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
        reopen_slots = (
            output.revision_logits[batch_index]
            .argmax(dim=-1)
            .eq(int(RevisionAction.REOPEN))
            .nonzero(as_tuple=False)
            .flatten()
            .cpu()
            .tolist()
        )
        previous_live = set(previous.live_cell_ids)
        reopen: list[ObjectRef] = []
        for slot in reopen_slots:
            cell = thought.cells[slot]
            if (
                cell.live
                and cell.cell_id is not None
                and cell.cell_id in previous_live
            ):
                reopen.append(ObjectRef.cell(cell.cell_id))
        return tuple(reopen)

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

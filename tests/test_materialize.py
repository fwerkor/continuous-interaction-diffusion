from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import pytest

from cid.contracts import ArgumentDescriptor, ModelContext, SourceDescriptor
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectKind, ObjectRef
from cid.state import CognitiveField, DisplayCanvas, FactStore

torch = pytest.importorskip("torch")
cid_model = import_module("cid.model")

AnchorCandidate = cid_model.AnchorCandidate
ArgumentCandidate = cid_model.ArgumentCandidate
CIDMaterializer = cid_model.CIDMaterializer
CIDMaterializerConfig = cid_model.CIDMaterializerConfig
CIDTensorOutput = cid_model.CIDTensorOutput
ClosedWorldMaterializationCatalog = cid_model.ClosedWorldMaterializationCatalog
ObjectCandidate = cid_model.ObjectCandidate
RevisionAction = cid_model.RevisionAction


def make_output(*, d_model: int = 4) -> CIDTensorOutput:
    batch = 1
    slots = 3
    display = 3
    sources = 1
    need_slots = 2
    return CIDTensorOutput(
        thought_semantic=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
        convergence_logits=torch.tensor([10.0]),
        allocation_logits=torch.tensor([[-10.0, 10.0, -10.0]]),
        role_logits=torch.zeros(batch, slots, 6),
        uncertainty=torch.full((batch, slots, 1), 0.2),
        noise_delta=torch.zeros(batch, slots, 1),
        lifecycle_logits=torch.tensor(
            [[[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]]
        ),
        display_logits=torch.zeros(batch, display, 16),
        need_logits=torch.tensor(
            [[[10.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]]]
        ),
        source_logits=torch.zeros(batch, slots, need_slots, sources),
        need_target_cell_logits=torch.full((batch, slots, need_slots, slots), -10.0),
        need_target_display_logits=torch.full((batch, slots, need_slots, display), -10.0),
        argument_presence_logits=torch.full((batch, slots, need_slots, 4), -10.0),
        argument_query=torch.zeros(batch, slots, need_slots, 4, d_model),
        anchor_query=torch.zeros(batch, slots, 4, d_model),
        anchor_presence_logits=torch.full((batch, slots, 4), -10.0),
        anchor_kind_logits=torch.zeros(batch, slots, 4, len(AnchorKind)),
        link_presence_logits=torch.full((batch, slots, 8), -10.0),
        link_relation_logits=torch.zeros(batch, slots, 8, len(LinkRelation)),
        link_target_kind_logits=torch.zeros(batch, slots, 8, len(ObjectKind)),
        link_target_query=torch.zeros(batch, slots, 8, d_model),
        revision_logits=torch.zeros(batch, slots, 3),
        refresh_logits=torch.zeros(batch, slots, need_slots, 3),
    )


def test_materializer_creates_cells_arguments_anchors_links_and_revisions() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    field, first = field.allocate(semantic=(1.0, 0.0, 0.0, 0.0))
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(
            SourceDescriptor(
                name="lookup",
                description="lookup a key",
                arguments=(ArgumentDescriptor(name="key", kind="string"),),
            ),
        ),
        percepts=(),
        step=0,
    )
    output = make_output()
    argument_embedding = torch.tensor([0.0, 1.0, 0.0, 0.0])
    output.argument_presence_logits[0, 0, 0, 0] = 10.0
    output.argument_query[0, 0, 0, 0] = argument_embedding
    anchor = Anchor(
        anchor_id="a:model-a",
        kind=AnchorKind.ENTITY,
        value="Model A",
        object_id="model:a",
    )
    output.anchor_presence_logits[0, 0, 0] = 10.0
    output.anchor_kind_logits[0, 0, 0, list(AnchorKind).index(AnchorKind.ENTITY)] = 10.0
    output.anchor_query[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    output.link_presence_logits[0, 0, 0] = 10.0
    output.link_relation_logits[
        0, 0, 0, list(LinkRelation).index(LinkRelation.DEPENDS_ON)
    ] = 10.0
    output.link_target_kind_logits[0, 0, 0, list(ObjectKind).index(ObjectKind.CELL)] = 10.0
    output.link_target_query[0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    output.revision_logits[0, 0, int(RevisionAction.REOPEN)] = 10.0
    # A newly allocated cell may emit arbitrary revision logits before it has ever
    # existed in runtime state. Those logits must not become an illegal REOPEN request.
    output.revision_logits[0, 1, int(RevisionAction.REOPEN)] = 10.0

    catalog = ClosedWorldMaterializationCatalog(
        arguments=(
            ArgumentCandidate(
                source="lookup",
                name="key",
                value="latency_ms",
                embedding=argument_embedding,
            ),
        ),
        anchors=(
            AnchorCandidate(
                anchor=anchor,
                embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ),
        ),
    )
    materializer = CIDMaterializer(
        CIDMaterializerConfig(
            allocation_threshold=0.8,
            need_threshold=0.8,
            retrieval_similarity_threshold=0.8,
        )
    )

    update = materializer.materialize(
        output,
        context,
        catalog=catalog,
        display_token_ids=(7, 8, 9),
    )

    assert update.thought.step == 1
    assert update.thought.occupied_count == 2
    second = next(cell_id for cell_id in update.thought.occupied_cell_ids if cell_id != first)
    assert update.thought.slot_of(second) == 1
    assert update.thought.get(first).anchors == (anchor,)
    assert update.thought.get(first).links == (
        CognitiveLink(
            relation=LinkRelation.DEPENDS_ON,
            target=ObjectRef.cell(second),
        ),
    )
    assert update.needs[0].need_id == f"need:{first}:0"
    assert update.needs[0].arguments == {"key": "latency_ms"}
    assert update.needs[0].selected_source() == "lookup"
    assert update.needs[0].target_cells == (ObjectRef.cell(first),)
    assert update.reopen_cells == (ObjectRef.cell(first),)
    assert update.display.token_ids == (7, 8, 9)
    assert update.equilibrium
    assert not update.converged

    settled = materializer.materialize(
        output,
        replace(context, display=update.display),
        catalog=catalog,
        display_token_ids=(7, 8, 9),
    )
    assert settled.converged


def test_materializer_never_resolves_cell_link_from_closed_world_catalog() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    field, runtime_cell = field.allocate(semantic=(1.0, 0.0, 0.0, 0.0))
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(SourceDescriptor(name="unused", description="unused"),),
        percepts=(),
        step=0,
    )
    output = make_output()
    output.allocation_logits.fill_(-10.0)
    output.need_logits.fill_(-10.0)
    output.link_presence_logits[0, 0, 0] = 10.0
    output.link_relation_logits[0, 0, 0, list(LinkRelation).index(LinkRelation.DEPENDS_ON)] = 10.0
    output.link_target_kind_logits[0, 0, 0, list(ObjectKind).index(ObjectKind.CELL)] = 10.0
    output.link_target_query[0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    catalog = ClosedWorldMaterializationCatalog(
        objects=(
            ObjectCandidate(
                ref=ObjectRef.cell("teacher-solution"),
                embedding=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            ),
        )
    )

    update = CIDMaterializer(
        CIDMaterializerConfig(retrieval_similarity_threshold=0.8)
    ).materialize(output, context, catalog=catalog)

    assert update.thought.get(runtime_cell).links == ()


def test_materializer_leaves_need_latent_until_required_argument_resolves() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    field, first = field.allocate(semantic=(1.0, 0.0, 0.0, 0.0))
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(
            SourceDescriptor(
                name="lookup",
                description="lookup a key",
                arguments=(ArgumentDescriptor(name="key"),),
            ),
        ),
        percepts=(),
        step=0,
    )
    output = make_output()

    update = CIDMaterializer().materialize(output, context)

    assert update.needs[0].need_id == f"need:{first}:0"
    assert update.needs[0].arguments == {}
    assert not update.converged


def test_materializer_emits_multiple_bindings_from_one_cognitive_cell() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    field, first = field.allocate(semantic=(1.0, 0.0, 0.0, 0.0))
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(SourceDescriptor(name="lookup", description="lookup"),),
        percepts=(),
        step=0,
    )
    output = make_output()
    output.need_logits[0, 0, 1] = 10.0

    update = CIDMaterializer(
        CIDMaterializerConfig(need_threshold=0.8)
    ).materialize(output, context)

    assert tuple(need.need_id for need in update.needs) == (
        f"need:{first}:0",
        f"need:{first}:1",
    )
    assert all(need.target_cells == (ObjectRef.cell(first),) for need in update.needs)


def test_materializer_routes_one_need_to_multiple_cells_and_display_spans() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    field, owner = field.allocate(semantic=(1.0, 0.0, 0.0, 0.0))
    field, affected = field.allocate(semantic=(0.0, 1.0, 0.0, 0.0))
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(
            SourceDescriptor(
                name="lookup",
                description="protected lookup",
                promote_results_to_fact=True,
            ),
        ),
        percepts=(),
        step=0,
    )
    output = make_output()
    output.allocation_logits.fill_(-10.0)
    output.need_target_cell_logits[0, 0, 0, 1] = 10.0
    output.need_target_display_logits[0, 0, 0, 1:] = 10.0

    update = CIDMaterializer(
        CIDMaterializerConfig(need_threshold=0.8)
    ).materialize(output, context)

    assert update.needs[0].need_id == f"need:{owner}:0"
    assert update.needs[0].target_cells == (ObjectRef.cell(owner), ObjectRef.cell(affected))
    assert update.needs[0].target_display == (ObjectRef.display_span(1, 3),)
    assert update.needs[0].promote_to_fact


def test_materializer_rejects_runtime_model_geometry_mismatch() -> None:
    field = CognitiveField.empty(capacity=2, width=4)
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(),
        percepts=(),
        step=0,
    )

    with pytest.raises(ValueError, match="slot count"):
        CIDMaterializer().materialize(make_output(), context)


def test_materializer_requires_learned_convergence_even_when_display_is_filled() -> None:
    field = CognitiveField.empty(capacity=3, width=4)
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=field,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        sources=(),
        percepts=(),
        step=0,
    )
    output = make_output()
    output.convergence_logits[0] = -10.0
    output.source_logits = torch.empty(1, 3, 2, 0)

    update = CIDMaterializer().materialize(
        output,
        context,
        display_token_ids=(7, 8, 9),
    )

    assert update.display.unresolved == 0
    assert not update.converged


def test_prefix_allocation_stops_at_first_low_free_slot() -> None:
    from cid.model.allocation import prefix_allocation_mask

    occupancy = torch.tensor([[True, False, False, False]])
    logits = torch.tensor([[0.0, 10.0, -10.0, 20.0]])

    selected = prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.8,
        max_allocations=4,
    )

    assert selected.tolist() == [[False, True, False, False]]

from dataclasses import replace
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
cid_model = import_module("cid.model")
CIDTargets = cid_model.CIDTargets
CIDTensorBatch = cid_model.CIDTensorBatch
TorchCIDConfig = cid_model.TorchCIDConfig
TorchCIDCore = cid_model.TorchCIDCore
cid_loss = cid_model.cid_loss
balanced_allocation_loss = import_module(
    "cid.model.losses"
)._balanced_masked_binary_cross_entropy


def test_allocation_loss_balances_sparse_positive_against_many_negatives() -> None:
    logits = torch.zeros((1, 4), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)

    loss = balanced_allocation_loss(logits, targets, mask)
    loss.backward()

    positive_gradient = logits.grad[0, 0].abs()
    negative_gradient_sum = logits.grad[0, 1:].abs().sum()
    assert positive_gradient == pytest.approx(negative_gradient_sum)



def test_torch_core_shapes_and_backward() -> None:
    config = TorchCIDConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_thought_slots=8,
        max_display_tokens=16,
    )
    model = TorchCIDCore(config)
    batch_size = 2
    thought_slots = 4
    display_length = 6
    source_count = 3

    batch = CIDTensorBatch(
        thought_semantic=torch.randn(batch_size, thought_slots, config.d_model),
        role_features=torch.rand(batch_size, thought_slots, config.num_roles),
        uncertainty=torch.rand(batch_size, thought_slots, 1),
        local_noise=torch.rand(batch_size, thought_slots, 1),
        slot_occupancy=torch.tensor(
            [[[1.0], [1.0], [0.0], [0.0]], [[1.0], [0.0], [0.0], [0.0]]]
        ),
        prompt_ids=torch.randint(0, config.vocab_size, (batch_size, 5)),
        display_ids=torch.randint(0, config.vocab_size, (batch_size, display_length)),
        display_noise=torch.rand(batch_size, display_length, 1),
        fact_memory=torch.randn(batch_size, 2, config.d_model),
        percept_memory=torch.randn(batch_size, 3, config.d_model),
        source_memory=torch.randn(batch_size, source_count, config.d_model),
    )
    output = model(batch)

    assert output.thought_semantic.shape == (batch_size, thought_slots, config.d_model)
    assert output.allocation_logits.shape == (batch_size, thought_slots)
    assert output.lifecycle_logits.shape == (
        batch_size,
        thought_slots,
        config.num_lifecycles,
    )
    assert output.display_logits.shape == (batch_size, display_length, config.vocab_size)
    assert output.need_logits.shape == (batch_size, thought_slots, config.max_need_slots)
    assert output.source_logits.shape == (
        batch_size, thought_slots, config.max_need_slots, source_count
    )
    assert output.need_target_cell_logits.shape == (
        batch_size, thought_slots, config.max_need_slots, thought_slots
    )
    assert output.need_target_display_logits.shape == (
        batch_size, thought_slots, config.max_need_slots, display_length
    )
    assert output.argument_presence_logits.shape == (
        batch_size,
        thought_slots,
        config.max_need_slots,
        config.max_argument_slots,
    )
    assert output.argument_query.shape == (
        batch_size,
        thought_slots,
        config.max_need_slots,
        config.max_argument_slots,
        config.d_model,
    )
    assert output.refresh_logits.shape == (
        batch_size, thought_slots, config.max_need_slots, config.num_refresh_actions
    )
    assert output.anchor_query.shape == (
        batch_size,
        thought_slots,
        config.max_anchor_slots,
        config.d_model,
    )
    assert output.anchor_presence_logits.shape == (
        batch_size,
        thought_slots,
        config.max_anchor_slots,
    )
    assert output.anchor_kind_logits.shape == (
        batch_size,
        thought_slots,
        config.max_anchor_slots,
        config.num_anchor_kinds,
    )
    assert output.link_target_query.shape == (
        batch_size,
        thought_slots,
        config.max_link_slots,
        config.d_model,
    )
    assert output.link_relation_logits.shape == (
        batch_size,
        thought_slots,
        config.max_link_slots,
        config.num_link_relations,
    )
    assert output.link_target_kind_logits.shape == (
        batch_size,
        thought_slots,
        config.max_link_slots,
        config.num_object_kinds,
    )

    occupied = batch.slot_occupancy.squeeze(-1).bool()
    lifecycle_targets = torch.full(
        (batch_size, thought_slots), -100, dtype=torch.long
    )
    lifecycle_targets[occupied] = torch.randint(
        0, config.num_lifecycles, (int(occupied.sum()),), dtype=torch.long
    )
    anchor_presence_mask = occupied[:, :, None].expand(
        -1, -1, config.max_anchor_slots
    )
    anchor_presence_targets = torch.zeros(
        batch_size, thought_slots, config.max_anchor_slots
    )
    anchor_presence_targets[:, :, :2] = occupied[:, :, None].float()
    anchor_mask = anchor_presence_targets.bool()
    anchor_kind_targets = torch.full(
        (batch_size, thought_slots, config.max_anchor_slots), -100, dtype=torch.long
    )
    anchor_kind_targets[anchor_mask] = torch.randint(
        0, config.num_anchor_kinds, (int(anchor_mask.sum()),), dtype=torch.long
    )
    link_presence_mask = occupied[:, :, None].expand(-1, -1, config.max_link_slots)
    link_presence_targets = torch.zeros(batch_size, thought_slots, config.max_link_slots)
    link_presence_targets[:, :, :2] = occupied[:, :, None].float()
    link_mask = link_presence_targets.bool()
    link_relation_targets = torch.full(
        (batch_size, thought_slots, config.max_link_slots), -100, dtype=torch.long
    )
    link_relation_targets[link_mask] = torch.randint(
        0, config.num_link_relations, (int(link_mask.sum()),), dtype=torch.long
    )
    link_target_kind_targets = torch.full(
        (batch_size, thought_slots, config.max_link_slots), -100, dtype=torch.long
    )
    link_target_kind_targets[link_mask] = torch.randint(
        0, config.num_object_kinds, (int(link_mask.sum()),), dtype=torch.long
    )
    targets = CIDTargets(
        thought_semantic=torch.randn_like(output.thought_semantic),
        thought_mask=occupied,
        convergence_targets=torch.rand(batch_size),
        allocation_targets=torch.randint(0, 2, (batch_size, thought_slots)).float(),
        allocation_mask=~occupied,
        display_ids=torch.randint(0, config.vocab_size, (batch_size, display_length)),
        role_targets=torch.rand_like(output.role_logits),
        uncertainty=torch.rand_like(output.uncertainty),
        noise_delta=torch.rand_like(output.noise_delta) * 2.0 - 1.0,
        lifecycle=lifecycle_targets,
        need_targets=torch.rand(batch_size, thought_slots, config.max_need_slots),
        source_targets=torch.randint(
            0,
            source_count,
            (batch_size, thought_slots, config.max_need_slots),
            dtype=torch.long,
        ),
        need_target_cell_targets=torch.rand_like(output.need_target_cell_logits),
        need_target_cell_mask=occupied[:, :, None, None].expand_as(
            output.need_target_cell_logits
        ),
        need_target_display_targets=torch.rand_like(output.need_target_display_logits),
        need_target_display_mask=occupied[:, :, None, None].expand_as(
            output.need_target_display_logits
        ),
        argument_presence_targets=torch.zeros(
            batch_size, thought_slots, config.max_need_slots, config.max_argument_slots
        ),
        argument_presence_mask=occupied[:, :, None, None].expand(
            -1, -1, config.max_need_slots, config.max_argument_slots
        ),
        argument_embeddings=torch.randn_like(output.argument_query),
        argument_mask=torch.zeros(
            batch_size,
            thought_slots,
            config.max_need_slots,
            config.max_argument_slots,
            dtype=torch.bool,
        ),
        revision_targets=torch.randint(0, 3, (batch_size, thought_slots), dtype=torch.long),
        refresh_targets=torch.randint(
            0,
            config.num_refresh_actions,
            (batch_size, thought_slots, config.max_need_slots),
            dtype=torch.long,
        ),
        anchor_presence_targets=anchor_presence_targets,
        anchor_presence_mask=anchor_presence_mask,
        anchor_kind_targets=anchor_kind_targets,
        anchor_embeddings=torch.randn_like(output.anchor_query),
        anchor_mask=anchor_mask,
        link_presence_targets=link_presence_targets,
        link_presence_mask=link_presence_mask,
        link_relation_targets=link_relation_targets,
        link_target_kind_targets=link_target_kind_targets,
        link_target_embeddings=torch.randn_like(output.link_target_query),
        link_mask=link_mask,
    )
    losses = cid_loss(output, targets)

    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.thought_delta.weight.grad is not None

    altered = output.allocation_logits.clone()
    altered[occupied] = altered[occupied] + 1000.0
    altered_output = replace(output, allocation_logits=altered)
    altered_losses = cid_loss(altered_output, targets)
    assert torch.allclose(losses.allocation, altered_losses.allocation)

    altered_need = output.need_logits.clone()
    altered_need[~occupied] = altered_need[~occupied] + 1000.0
    altered_need_losses = cid_loss(replace(output, need_logits=altered_need), targets)
    assert torch.allclose(losses.intent, altered_need_losses.intent)

    ignored_targets = replace(
        targets,
        display_ids=torch.full_like(targets.display_ids, -100),
        lifecycle=torch.full_like(targets.lifecycle, -100),
        source_targets=torch.full_like(targets.source_targets, -100),
        revision_targets=torch.full_like(targets.revision_targets, -100),
        refresh_targets=torch.full_like(targets.refresh_targets, -100),
    )
    ignored_losses = cid_loss(output, ignored_targets)
    assert torch.isfinite(ignored_losses.total)
    assert ignored_losses.display == 0
    assert ignored_losses.lifecycle == 0
    assert ignored_losses.source == 0
    assert ignored_losses.revision == 0
    assert ignored_losses.refresh == 0

    anchor_order = torch.tensor([1, 0, *range(2, config.max_anchor_slots)])
    link_order = torch.tensor([1, 0, *range(2, config.max_link_slots)])
    permuted_targets = replace(
        targets,
        anchor_kind_targets=targets.anchor_kind_targets.index_select(2, anchor_order),
        anchor_embeddings=targets.anchor_embeddings.index_select(2, anchor_order),
        anchor_mask=targets.anchor_mask.index_select(2, anchor_order),
        link_relation_targets=targets.link_relation_targets.index_select(2, link_order),
        link_target_kind_targets=targets.link_target_kind_targets.index_select(2, link_order),
        link_target_embeddings=targets.link_target_embeddings.index_select(2, link_order),
        link_mask=targets.link_mask.index_select(2, link_order),
    )
    permuted_losses = cid_loss(output, permuted_targets)
    assert torch.allclose(losses.anchor_kind, permuted_losses.anchor_kind)
    assert torch.allclose(losses.anchor_ground, permuted_losses.anchor_ground)
    assert torch.allclose(losses.link_relation, permuted_losses.link_relation)
    assert torch.allclose(losses.link_target_kind, permuted_losses.link_target_kind)
    assert torch.allclose(losses.link_ground, permuted_losses.link_ground)


def test_torch_core_accepts_empty_external_memory_and_no_sources() -> None:
    config = TorchCIDConfig(vocab_size=32, d_model=16, num_layers=1, num_heads=4)
    model = TorchCIDCore(config)
    batch = CIDTensorBatch(
        thought_semantic=torch.randn(1, 2, config.d_model),
        role_features=torch.rand(1, 2, config.num_roles),
        uncertainty=torch.rand(1, 2, 1),
        local_noise=torch.rand(1, 2, 1),
        slot_occupancy=torch.zeros(1, 2, 1),
        prompt_ids=torch.empty(1, 0, dtype=torch.long),
        display_ids=torch.randint(0, config.vocab_size, (1, 3)),
        display_noise=torch.rand(1, 3, 1),
        fact_memory=torch.empty(1, 0, config.d_model),
        percept_memory=torch.empty(1, 0, config.d_model),
        source_memory=torch.empty(1, 0, config.d_model),
    )

    output = model(batch)

    assert torch.isfinite(output.thought_semantic).all()
    assert torch.isfinite(output.display_logits).all()
    assert output.source_logits.shape == (1, 2, config.max_need_slots, 0)


def test_external_fusion_keeps_mixed_empty_memory_rows_finite() -> None:
    fusion_cls = import_module("cid.model.components").CIDExternalFusion
    fusion = fusion_cls(d_model=8, num_heads=2)
    hidden = torch.randn(2, 3, 8)
    seed_hidden = torch.randn(2, 3, 8)
    context_weight = torch.ones(2, 3, 1)
    facts = torch.randn(2, 1, 8)
    percepts = torch.empty(2, 0, 8)
    fact_padding_mask = torch.tensor([[True], [False]])

    output = fusion(
        hidden,
        seed_hidden=seed_hidden,
        context_weight=context_weight,
        facts=facts,
        percepts=percepts,
        fact_padding_mask=fact_padding_mask,
    )

    assert torch.isfinite(output).all()


def test_empty_masked_cross_entropy_zero_does_not_overflow() -> None:
    masked_cross_entropy = import_module("cid.model.losses")._masked_cross_entropy
    logits = torch.full(
        (4, 8),
        torch.finfo(torch.float32).min,
        requires_grad=True,
    )
    targets = torch.full((4,), -100, dtype=torch.long)
    mask = torch.zeros(4, dtype=torch.bool)

    loss = masked_cross_entropy(logits, targets, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0

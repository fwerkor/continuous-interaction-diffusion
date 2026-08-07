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
    assert output.source_logits.shape == (batch_size, thought_slots, source_count)
    assert output.refresh_logits.shape == (batch_size, thought_slots, config.num_refresh_actions)
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
    anchor_presence_targets[:, :, 0] = occupied.float()
    anchor_mask = anchor_presence_targets.bool()
    anchor_kind_targets = torch.full(
        (batch_size, thought_slots, config.max_anchor_slots), -100, dtype=torch.long
    )
    anchor_kind_targets[anchor_mask] = torch.randint(
        0, config.num_anchor_kinds, (int(anchor_mask.sum()),), dtype=torch.long
    )
    link_presence_mask = occupied[:, :, None].expand(-1, -1, config.max_link_slots)
    link_presence_targets = torch.zeros(batch_size, thought_slots, config.max_link_slots)
    link_presence_targets[:, :, 0] = occupied.float()
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
        allocation_targets=torch.randint(0, 2, (batch_size, thought_slots)).float(),
        allocation_mask=~occupied,
        display_ids=torch.randint(0, config.vocab_size, (batch_size, display_length)),
        role_targets=torch.rand_like(output.role_logits),
        uncertainty=torch.rand_like(output.uncertainty),
        lifecycle=lifecycle_targets,
        need_targets=torch.rand(batch_size, thought_slots),
        source_targets=torch.randint(
            0, source_count, (batch_size, thought_slots), dtype=torch.long
        ),
        revision_targets=torch.randint(0, 3, (batch_size, thought_slots), dtype=torch.long),
        refresh_targets=torch.randint(
            0, config.num_refresh_actions, (batch_size, thought_slots), dtype=torch.long
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


def test_torch_core_accepts_empty_external_memory_and_no_sources() -> None:
    config = TorchCIDConfig(vocab_size=32, d_model=16, num_layers=1, num_heads=4)
    model = TorchCIDCore(config)
    batch = CIDTensorBatch(
        thought_semantic=torch.randn(1, 2, config.d_model),
        role_features=torch.rand(1, 2, config.num_roles),
        uncertainty=torch.rand(1, 2, 1),
        local_noise=torch.rand(1, 2, 1),
        slot_occupancy=torch.zeros(1, 2, 1),
        display_ids=torch.randint(0, config.vocab_size, (1, 3)),
        display_noise=torch.rand(1, 3, 1),
        fact_memory=torch.empty(1, 0, config.d_model),
        percept_memory=torch.empty(1, 0, config.d_model),
        source_memory=torch.empty(1, 0, config.d_model),
    )

    output = model(batch)

    assert torch.isfinite(output.thought_semantic).all()
    assert torch.isfinite(output.display_logits).all()
    assert output.source_logits.shape == (1, 2, 0)

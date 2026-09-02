from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from cid.model.tensors import CIDTensorOutput

NEED_INTENT_TARGET_POSITIVE_MASS = 0.20
NEED_INTENT_POSITIVE_WEIGHT_CAP = 128.0


@dataclass(frozen=True, slots=True)
class CIDLossWeights:
    thought: float = 1.0
    convergence: float = 0.2
    allocation: float = 0.3
    display: float = 1.0
    roles: float = 0.2
    uncertainty: float = 0.1
    noise: float = 0.1
    lifecycle: float = 0.1
    intent: float = 0.5
    source: float = 0.5
    need_cell_route: float = 0.2
    need_display_route: float = 0.2
    argument_presence: float = 0.1
    argument_ground: float = 0.2
    revision: float = 0.2
    refresh: float = 0.2
    anchor_presence: float = 0.1
    anchor_kind: float = 0.1
    anchor_ground: float = 0.2
    link_presence: float = 0.1
    link_relation: float = 0.1
    link_target_kind: float = 0.1
    link_ground: float = 0.2


@dataclass(slots=True)
class CIDTargets:
    thought_semantic: Tensor
    thought_mask: Tensor
    convergence_targets: Tensor
    allocation_targets: Tensor
    allocation_mask: Tensor
    display_ids: Tensor
    role_targets: Tensor
    uncertainty: Tensor
    noise_delta: Tensor
    lifecycle: Tensor
    need_targets: Tensor
    source_targets: Tensor
    need_target_cell_targets: Tensor
    need_target_cell_mask: Tensor
    need_target_display_targets: Tensor
    need_target_display_mask: Tensor
    argument_presence_targets: Tensor
    argument_presence_mask: Tensor
    argument_embeddings: Tensor
    argument_mask: Tensor
    revision_targets: Tensor
    refresh_targets: Tensor
    anchor_presence_targets: Tensor
    anchor_presence_mask: Tensor
    anchor_kind_targets: Tensor
    anchor_embeddings: Tensor
    anchor_mask: Tensor
    link_presence_targets: Tensor
    link_presence_mask: Tensor
    link_relation_targets: Tensor
    link_target_kind_targets: Tensor
    link_target_embeddings: Tensor
    link_mask: Tensor


@dataclass(frozen=True, slots=True)
class CIDLoss:
    total: Tensor
    thought: Tensor
    convergence: Tensor
    allocation: Tensor
    display: Tensor
    roles: Tensor
    uncertainty: Tensor
    noise: Tensor
    lifecycle: Tensor
    intent: Tensor
    source: Tensor
    need_cell_route: Tensor
    need_display_route: Tensor
    argument_presence: Tensor
    argument_ground: Tensor
    revision: Tensor
    refresh: Tensor
    anchor_presence: Tensor
    anchor_kind: Tensor
    anchor_ground: Tensor
    link_presence: Tensor
    link_relation: Tensor
    link_target_kind: Tensor
    link_ground: Tensor
    auxiliary: Tensor


def cid_loss(
    output: CIDTensorOutput,
    targets: CIDTargets,
    weights: CIDLossWeights | None = None,
    *,
    batch_mask: Tensor | None = None,
) -> CIDLoss:
    w = weights or CIDLossWeights()
    if batch_mask is None:
        batch_mask = torch.ones(
            output.thought_semantic.shape[0],
            dtype=torch.bool,
            device=output.thought_semantic.device,
        )
    if batch_mask.shape != (output.thought_semantic.shape[0],):
        raise ValueError("CID loss batch_mask must have shape [batch]")
    batch_mask = batch_mask.bool()

    thought_mask = _with_batch_mask(targets.thought_mask, batch_mask)
    allocation_mask = _with_batch_mask(targets.allocation_mask, batch_mask)
    display_mask = _with_batch_mask(targets.display_ids != -100, batch_mask)
    lifecycle_mask = _with_batch_mask(targets.lifecycle != -100, batch_mask)
    source_mask = _with_batch_mask(targets.source_targets != -100, batch_mask)
    need_cell_route_mask = _with_batch_mask(targets.need_target_cell_mask, batch_mask)
    need_display_route_mask = _with_batch_mask(targets.need_target_display_mask, batch_mask)
    argument_presence_mask = _with_batch_mask(targets.argument_presence_mask, batch_mask)
    argument_mask = _with_batch_mask(targets.argument_mask, batch_mask)
    revision_mask = _with_batch_mask(targets.revision_targets != -100, batch_mask)
    refresh_mask = _with_batch_mask(targets.refresh_targets != -100, batch_mask)
    anchor_presence_mask = _with_batch_mask(targets.anchor_presence_mask, batch_mask)
    link_presence_mask = _with_batch_mask(targets.link_presence_mask, batch_mask)
    (
        anchor_presence_targets,
        anchor_kind_targets,
        anchor_embeddings,
        anchor_mask,
    ) = _align_anchor_targets(output, targets)
    (
        link_presence_targets,
        link_relation_targets,
        link_target_kind_targets,
        link_target_embeddings,
        link_mask,
    ) = _align_link_targets(output, targets)
    thought = _masked_vector_mse(
        output.thought_semantic,
        targets.thought_semantic,
        thought_mask,
        batch_mask=batch_mask,
    )
    convergence = _masked_binary_cross_entropy(
        output.convergence_logits,
        targets.convergence_targets,
        batch_mask,
        batch_mask=batch_mask,
    )
    allocation = _balanced_masked_binary_cross_entropy(
        output.allocation_logits,
        targets.allocation_targets,
        allocation_mask,
        batch_mask=batch_mask,
    )
    display = _masked_cross_entropy(
        output.display_logits,
        targets.display_ids,
        display_mask,
        batch_mask=batch_mask,
    )
    roles = _masked_binary_cross_entropy(
        output.role_logits,
        targets.role_targets,
        thought_mask.unsqueeze(-1).expand_as(output.role_logits),
        batch_mask=batch_mask,
    )
    uncertainty = _masked_vector_mse(
        output.uncertainty,
        targets.uncertainty,
        thought_mask,
        batch_mask=batch_mask,
    )
    noise = _masked_vector_mse(
        output.noise_delta,
        targets.noise_delta,
        thought_mask,
        batch_mask=batch_mask,
    )
    lifecycle = _masked_cross_entropy(
        output.lifecycle_logits,
        targets.lifecycle,
        lifecycle_mask,
        batch_mask=batch_mask,
    )
    intent = _target_positive_mass_binary_cross_entropy(
        output.need_logits,
        targets.need_targets,
        thought_mask.unsqueeze(-1).expand_as(output.need_logits),
        target_positive_mass=NEED_INTENT_TARGET_POSITIVE_MASS,
        positive_weight_cap=NEED_INTENT_POSITIVE_WEIGHT_CAP,
        batch_mask=batch_mask,
    )
    source = _masked_cross_entropy(
        output.source_logits,
        targets.source_targets,
        source_mask,
        batch_mask=batch_mask,
    )
    need_cell_route = _capped_positive_weight_binary_cross_entropy(
        output.need_target_cell_logits,
        targets.need_target_cell_targets,
        need_cell_route_mask,
        positive_weight_cap=6.0,
        batch_mask=batch_mask,
    )
    need_display_route = _capped_positive_weight_binary_cross_entropy(
        output.need_target_display_logits,
        targets.need_target_display_targets,
        need_display_route_mask,
        positive_weight_cap=6.0,
        batch_mask=batch_mask,
    )
    argument_presence = _capped_positive_weight_binary_cross_entropy(
        output.argument_presence_logits,
        targets.argument_presence_targets,
        argument_presence_mask,
        positive_weight_cap=6.0,
        batch_mask=batch_mask,
    )
    argument_ground = _masked_cosine_loss(
        output.argument_query,
        targets.argument_embeddings,
        argument_mask,
        batch_mask=batch_mask,
    )
    revision = _masked_cross_entropy(
        output.revision_logits,
        targets.revision_targets,
        revision_mask,
        batch_mask=batch_mask,
    )
    refresh = _masked_cross_entropy(
        output.refresh_logits,
        targets.refresh_targets,
        refresh_mask,
        batch_mask=batch_mask,
    )
    anchor_presence = _capped_positive_weight_binary_cross_entropy(
        output.anchor_presence_logits,
        anchor_presence_targets,
        anchor_presence_mask,
        positive_weight_cap=6.0,
        batch_mask=batch_mask,
    )
    anchor_kind = _masked_cross_entropy(
        output.anchor_kind_logits,
        anchor_kind_targets,
        _with_batch_mask(anchor_mask, batch_mask),
        batch_mask=batch_mask,
    )
    anchor_ground = _masked_cosine_loss(
        output.anchor_query,
        anchor_embeddings,
        _with_batch_mask(anchor_mask, batch_mask),
        batch_mask=batch_mask,
    )
    link_presence = _capped_positive_weight_binary_cross_entropy(
        output.link_presence_logits,
        link_presence_targets,
        link_presence_mask,
        positive_weight_cap=6.0,
        batch_mask=batch_mask,
    )
    link_relation = _masked_cross_entropy(
        output.link_relation_logits,
        link_relation_targets,
        _with_batch_mask(link_mask, batch_mask),
        batch_mask=batch_mask,
    )
    link_target_kind = _masked_cross_entropy(
        output.link_target_kind_logits,
        link_target_kind_targets,
        _with_batch_mask(link_mask, batch_mask),
        batch_mask=batch_mask,
    )
    link_ground = _masked_cosine_loss(
        output.link_target_query,
        link_target_embeddings,
        _with_batch_mask(link_mask, batch_mask),
        batch_mask=batch_mask,
    )
    auxiliary = (
        output.auxiliary_loss
        if output.auxiliary_loss is not None
        else _differentiable_zero(output.thought_semantic)
    )
    total = (
        w.thought * thought
        + w.convergence * convergence
        + w.allocation * allocation
        + w.display * display
        + w.roles * roles
        + w.uncertainty * uncertainty
        + w.noise * noise
        + w.lifecycle * lifecycle
        + w.intent * intent
        + w.source * source
        + w.need_cell_route * need_cell_route
        + w.need_display_route * need_display_route
        + w.argument_presence * argument_presence
        + w.argument_ground * argument_ground
        + w.revision * revision
        + w.refresh * refresh
        + w.anchor_presence * anchor_presence
        + w.anchor_kind * anchor_kind
        + w.anchor_ground * anchor_ground
        + w.link_presence * link_presence
        + w.link_relation * link_relation
        + w.link_target_kind * link_target_kind
        + w.link_ground * link_ground
        + auxiliary
    )
    return CIDLoss(
        total=total,
        thought=thought,
        convergence=convergence,
        allocation=allocation,
        display=display,
        roles=roles,
        uncertainty=uncertainty,
        noise=noise,
        lifecycle=lifecycle,
        intent=intent,
        source=source,
        need_cell_route=need_cell_route,
        need_display_route=need_display_route,
        argument_presence=argument_presence,
        argument_ground=argument_ground,
        revision=revision,
        refresh=refresh,
        anchor_presence=anchor_presence,
        anchor_kind=anchor_kind,
        anchor_ground=anchor_ground,
        link_presence=link_presence,
        link_relation=link_relation,
        link_target_kind=link_target_kind,
        link_ground=link_ground,
        auxiliary=auxiliary,
    )


def _differentiable_zero(tensor: Tensor) -> Tensor:
    """Return graph-connected zero without reducing large finite values first."""
    return (tensor * 0.0).sum()


def _with_batch_mask(mask: Tensor, batch_mask: Tensor) -> Tensor:
    if mask.shape[0] != batch_mask.shape[0]:
        raise ValueError("mask batch dimension does not match batch_mask")
    shape = (batch_mask.shape[0],) + (1,) * (mask.ndim - 1)
    return mask.bool() & batch_mask.reshape(shape)


def _masked_element_mean(
    losses: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None,
    reference: Tensor,
) -> Tensor:
    if losses.shape != mask.shape:
        raise ValueError("masked element loss and mask shapes must match")
    selected = mask.bool()
    if batch_mask is None:
        values = losses[selected]
        if values.numel() == 0:
            return _differentiable_zero(reference)
        return values.mean()
    if batch_mask.shape != (losses.shape[0],):
        raise ValueError("component batch_mask must have shape [batch]")
    active_rows = batch_mask.bool()
    if not bool(active_rows.any()):
        return _differentiable_zero(reference)
    flat_losses = losses.reshape(losses.shape[0], -1)
    flat_mask = selected.reshape(selected.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    row_losses = (flat_losses * flat_mask.to(dtype=flat_losses.dtype)).sum(dim=1)
    row_losses = row_losses / counts.clamp_min(1).to(dtype=flat_losses.dtype)
    # A valid example with no labels for this component contributes zero. This keeps
    # CIDLoss a true per-example mean, which is required because the trainer multiplies
    # losses.total by the valid-example count before gradient accumulation.
    return row_losses[active_rows].mean()


def _masked_cosine_loss(
    query: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None = None,
) -> Tensor:
    cosine = F.cosine_similarity(query, target, dim=-1)
    return _masked_element_mean(
        1.0 - cosine,
        mask,
        batch_mask=batch_mask,
        reference=query,
    )


def _masked_vector_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None = None,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("masked MSE prediction and target shapes must match")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("masked MSE mask must have shape [batch, slots]")
    per_slot = (prediction - target).float().square().flatten(start_dim=2).mean(dim=-1)
    return _masked_element_mean(
        per_slot,
        mask,
        batch_mask=batch_mask,
        reference=prediction,
    )


def _balanced_masked_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None = None,
) -> Tensor:
    """Balance positive/negative allocation supervision within each example.

    Allocation positives are sparse because every free slot after the requested prefix is a
    negative. A plain masked mean lets those negatives dominate the gradient and can produce a
    deceptively low loss with probabilities that never cross the runtime allocation threshold.
    """
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if batch_mask is not None and batch_mask.shape != (logits.shape[0],):
        raise ValueError("component batch_mask must have shape [batch]")
    rows: list[Tensor] = []
    for row in range(logits.shape[0]):
        if batch_mask is not None and not bool(batch_mask[row]):
            continue
        valid = mask[row].bool()
        positive = losses[row][valid & (target[row] >= 0.5)]
        negative = losses[row][valid & (target[row] < 0.5)]
        if positive.numel() and negative.numel():
            rows.append(0.5 * (positive.mean() + negative.mean()))
        elif positive.numel():
            rows.append(positive.mean())
        elif negative.numel():
            rows.append(negative.mean())
        elif batch_mask is not None:
            rows.append(_differentiable_zero(logits[row : row + 1]))
    if not rows:
        return _differentiable_zero(logits)
    return torch.stack(rows).mean()


def _capped_positive_weight_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    positive_weight_cap: float,
    batch_mask: Tensor | None = None,
) -> Tensor:
    """Upweight sparse positives without letting inverse-frequency weights explode.

    Information-need labels are intentionally sparse: most thought cells do not need a tool on
    most steps.  Plain BCE therefore rewards an almost-always-negative classifier.  Full inverse
    frequency balancing can swing too far in the other direction on small batches, so we estimate
    the positive weight from each example and cap it at a conservative value.
    """
    if positive_weight_cap < 1.0:
        raise ValueError("positive_weight_cap must be at least 1")
    valid = mask.bool()
    flat_valid = valid.flatten(start_dim=1)
    flat_target = target.flatten(start_dim=1)
    positive = flat_valid & (flat_target >= 0.5)
    negative = flat_valid & ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    ratio = negative_count.to(dtype=logits.dtype) / positive_count.clamp_min(1).to(
        dtype=logits.dtype
    )
    positive_weight = ratio.clamp(min=1.0, max=positive_weight_cap)

    weights = torch.ones_like(flat_target)
    weights = torch.where(positive, positive_weight.unsqueeze(1), weights)
    weights = weights * flat_valid.to(dtype=weights.dtype)
    losses = F.binary_cross_entropy_with_logits(
        logits.flatten(start_dim=1), flat_target, reduction="none"
    )
    denominator = weights.sum(dim=1)
    row_losses = (losses * weights).sum(dim=1) / denominator.clamp_min(1.0)
    if batch_mask is None:
        valid_rows = (denominator > 0).to(dtype=losses.dtype)
        return (row_losses * valid_rows).sum() / valid_rows.sum().clamp_min(1.0)
    if batch_mask.shape != (logits.shape[0],):
        raise ValueError("component batch_mask must have shape [batch]")
    active_rows = batch_mask.bool()
    if not bool(active_rows.any()):
        return _differentiable_zero(logits)
    return row_losses[active_rows].mean()


def _target_positive_mass_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    target_positive_mass: float,
    positive_weight_cap: float,
    batch_mask: Tensor | None = None,
) -> Tensor:
    """Keep sparse positives relevant without fully balancing positive and negative classes.

    Need-intent supervision is much sparser than the other CID binary targets.  A small fixed
    positive-weight cap still lets hundreds of easy negatives dominate a row, while exact
    inverse-frequency balancing makes false-positive tool calls too cheap.  For rows that contain
    positives, choose the positive weight that would assign ``target_positive_mass`` of the
    weighted BCE denominator to the positive class, subject to a safety cap.  Rows with no
    positives retain ordinary negative BCE supervision.
    """
    if not 0.0 < target_positive_mass < 0.5:
        raise ValueError("target positive mass must be in (0, 0.5)")
    if positive_weight_cap < 1.0:
        raise ValueError("positive_weight_cap must be at least 1")
    if batch_mask is not None and batch_mask.shape != (logits.shape[0],):
        raise ValueError("component batch_mask must have shape [batch]")

    flat_logits = logits.flatten(start_dim=1)
    flat_target = target.flatten(start_dim=1)
    flat_valid = mask.bool().flatten(start_dim=1)
    positive = flat_valid & (flat_target >= 0.5)
    negative = flat_valid & ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)

    class_odds = target_positive_mass / (1.0 - target_positive_mass)
    ratio = negative_count.to(dtype=logits.dtype) / positive_count.clamp_min(1).to(
        dtype=logits.dtype
    )
    positive_weight = (ratio * class_odds).clamp(min=1.0, max=positive_weight_cap)

    weights = torch.ones_like(flat_target)
    weights = torch.where(positive, positive_weight.unsqueeze(1), weights)
    weights = weights * flat_valid.to(dtype=weights.dtype)
    losses = F.binary_cross_entropy_with_logits(flat_logits, flat_target, reduction="none")
    denominator = weights.sum(dim=1)
    row_losses = (losses * weights).sum(dim=1) / denominator.clamp_min(1.0)

    valid_rows = denominator > 0
    if batch_mask is not None:
        valid_rows = valid_rows & batch_mask.bool()
    if not bool(valid_rows.any()):
        return _differentiable_zero(logits)
    return row_losses[valid_rows].mean()


def _masked_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None = None,
) -> Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_element_mean(
        losses,
        mask,
        batch_mask=batch_mask,
        reference=logits,
    )


def _masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_mask: Tensor | None = None,
) -> Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        return _differentiable_zero(logits)
    class_count = logits.shape[-1]
    safe_target = target.masked_fill(~selected, 0)
    per_element = F.cross_entropy(
        logits.reshape(-1, class_count),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return _masked_element_mean(
        per_element,
        selected,
        batch_mask=batch_mask,
        reference=logits,
    )


def _align_anchor_targets(
    output: CIDTensorOutput,
    targets: CIDTargets,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    presence = torch.zeros_like(targets.anchor_presence_targets)
    kinds = torch.full_like(targets.anchor_kind_targets, -100)
    embeddings = torch.zeros_like(targets.anchor_embeddings)
    mask = torch.zeros_like(targets.anchor_mask)
    for batch_index, slot in _supervised_cells(targets.anchor_presence_mask):
        target_slots = torch.nonzero(
            targets.anchor_mask[batch_index, slot], as_tuple=False
        ).flatten()
        if target_slots.numel() == 0:
            continue
        query = output.anchor_query[batch_index, slot]
        target_embedding = targets.anchor_embeddings[batch_index, slot, target_slots]
        target_kind = targets.anchor_kind_targets[batch_index, slot, target_slots]
        cost = _retrieval_cost(query, target_embedding)
        cost = cost + _classification_cost(
            output.anchor_kind_logits[batch_index, slot], target_kind
        )
        for target_offset, prediction_slot in enumerate(_linear_assignment(cost)):
            source_slot = int(target_slots[target_offset])
            presence[batch_index, slot, prediction_slot] = 1.0
            kinds[batch_index, slot, prediction_slot] = targets.anchor_kind_targets[
                batch_index, slot, source_slot
            ]
            embeddings[batch_index, slot, prediction_slot] = targets.anchor_embeddings[
                batch_index, slot, source_slot
            ]
            mask[batch_index, slot, prediction_slot] = True
    return presence, kinds, embeddings, mask


def _align_link_targets(
    output: CIDTensorOutput,
    targets: CIDTargets,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    presence = torch.zeros_like(targets.link_presence_targets)
    relations = torch.full_like(targets.link_relation_targets, -100)
    kinds = torch.full_like(targets.link_target_kind_targets, -100)
    embeddings = torch.zeros_like(targets.link_target_embeddings)
    mask = torch.zeros_like(targets.link_mask)
    for batch_index, slot in _supervised_cells(targets.link_presence_mask):
        target_slots = torch.nonzero(targets.link_mask[batch_index, slot], as_tuple=False).flatten()
        if target_slots.numel() == 0:
            continue
        target_embedding = targets.link_target_embeddings[batch_index, slot, target_slots]
        target_relation = targets.link_relation_targets[batch_index, slot, target_slots]
        target_kind = targets.link_target_kind_targets[batch_index, slot, target_slots]
        cost = _retrieval_cost(output.link_target_query[batch_index, slot], target_embedding)
        cost = cost + _classification_cost(
            output.link_relation_logits[batch_index, slot], target_relation
        )
        cost = cost + _classification_cost(
            output.link_target_kind_logits[batch_index, slot], target_kind
        )
        for target_offset, prediction_slot in enumerate(_linear_assignment(cost)):
            source_slot = int(target_slots[target_offset])
            presence[batch_index, slot, prediction_slot] = 1.0
            relations[batch_index, slot, prediction_slot] = targets.link_relation_targets[
                batch_index, slot, source_slot
            ]
            kinds[batch_index, slot, prediction_slot] = targets.link_target_kind_targets[
                batch_index, slot, source_slot
            ]
            embeddings[batch_index, slot, prediction_slot] = targets.link_target_embeddings[
                batch_index, slot, source_slot
            ]
            mask[batch_index, slot, prediction_slot] = True
    return presence, relations, kinds, embeddings, mask


def _supervised_cells(presence_mask: Tensor) -> tuple[tuple[int, int], ...]:
    cell_mask = presence_mask.bool().any(dim=-1)
    return tuple(tuple(int(value) for value in item) for item in torch.nonzero(cell_mask).tolist())


def _retrieval_cost(query: Tensor, target: Tensor) -> Tensor:
    normalized_query = F.normalize(query.float(), dim=-1)
    normalized_target = F.normalize(target.float(), dim=-1)
    similarity = torch.einsum("kd,md->mk", normalized_query, normalized_target)
    return 1.0 - similarity


def _classification_cost(logits: Tensor, target: Tensor) -> Tensor:
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    return -log_probabilities[:, target.long()].transpose(0, 1)


def _linear_assignment(cost: Tensor) -> tuple[int, ...]:
    """Minimum-cost target-to-prediction assignment for small grounding sets."""

    matrix = cost.detach().float().cpu().tolist()
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if rows == 0:
        return ()
    if rows > columns:
        raise ValueError("grounding target count exceeds prediction slot capacity")

    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    if any(item < 0 for item in assignment):
        raise RuntimeError("failed to assign grounding targets")
    return tuple(assignment)

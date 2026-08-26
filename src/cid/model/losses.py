from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from cid.model.tensors import CIDTensorOutput


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
) -> CIDLoss:
    w = weights or CIDLossWeights()
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
        targets.thought_mask,
    )
    convergence = F.binary_cross_entropy_with_logits(
        output.convergence_logits,
        targets.convergence_targets,
    )
    allocation = _balanced_masked_binary_cross_entropy(
        output.allocation_logits,
        targets.allocation_targets,
        targets.allocation_mask,
    )
    display = _masked_cross_entropy(
        output.display_logits,
        targets.display_ids,
        targets.display_ids != -100,
    )
    roles = _masked_binary_cross_entropy(
        output.role_logits,
        targets.role_targets,
        targets.thought_mask.unsqueeze(-1).expand_as(output.role_logits),
    )
    uncertainty = _masked_vector_mse(
        output.uncertainty,
        targets.uncertainty,
        targets.thought_mask,
    )
    noise = _masked_vector_mse(
        output.noise_delta,
        targets.noise_delta,
        targets.thought_mask,
    )
    lifecycle = _masked_cross_entropy(
        output.lifecycle_logits,
        targets.lifecycle,
        targets.lifecycle != -100,
    )
    intent = _masked_binary_cross_entropy(
        output.need_logits,
        targets.need_targets,
        targets.thought_mask,
    )
    source = _masked_cross_entropy(
        output.source_logits,
        targets.source_targets,
        targets.source_targets != -100,
    )
    argument_presence = _masked_binary_cross_entropy(
        output.argument_presence_logits,
        targets.argument_presence_targets,
        targets.argument_presence_mask,
    )
    argument_ground = _masked_cosine_loss(
        output.argument_query,
        targets.argument_embeddings,
        targets.argument_mask,
    )
    revision = _masked_cross_entropy(
        output.revision_logits,
        targets.revision_targets,
        targets.revision_targets != -100,
    )
    refresh = _masked_cross_entropy(
        output.refresh_logits,
        targets.refresh_targets,
        targets.refresh_targets != -100,
    )
    anchor_presence = _masked_binary_cross_entropy(
        output.anchor_presence_logits,
        anchor_presence_targets,
        targets.anchor_presence_mask,
    )
    anchor_kind = _masked_cross_entropy(
        output.anchor_kind_logits,
        anchor_kind_targets,
        anchor_mask,
    )
    anchor_ground = _masked_cosine_loss(
        output.anchor_query,
        anchor_embeddings,
        anchor_mask,
    )
    link_presence = _masked_binary_cross_entropy(
        output.link_presence_logits,
        link_presence_targets,
        targets.link_presence_mask,
    )
    link_relation = _masked_cross_entropy(
        output.link_relation_logits,
        link_relation_targets,
        link_mask,
    )
    link_target_kind = _masked_cross_entropy(
        output.link_target_kind_logits,
        link_target_kind_targets,
        link_mask,
    )
    link_ground = _masked_cosine_loss(
        output.link_target_query,
        link_target_embeddings,
        link_mask,
    )
    auxiliary = (
        output.auxiliary_loss
        if output.auxiliary_loss is not None
        else output.thought_semantic.sum() * 0.0
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


def _masked_cosine_loss(query: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    cosine = F.cosine_similarity(query, target, dim=-1)
    selected = cosine[mask.bool()]
    if selected.numel() == 0:
        return query.sum() * 0.0
    return (1.0 - selected).mean()


def _masked_vector_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("masked MSE prediction and target shapes must match")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("masked MSE mask must have shape [batch, slots]")
    per_slot = (prediction - target).float().square().flatten(start_dim=2).mean(dim=-1)
    selected = per_slot[mask.bool()]
    if selected.numel() == 0:
        return prediction.sum() * 0.0
    return selected.mean()


def _balanced_masked_binary_cross_entropy(
    logits: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    """Balance positive/negative allocation supervision within each example.

    Allocation positives are sparse because every free slot after the requested prefix is a
    negative. A plain masked mean lets those negatives dominate the gradient and can produce a
    deceptively low loss with probabilities that never cross the runtime allocation threshold.
    """
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    rows: list[Tensor] = []
    for row in range(logits.shape[0]):
        valid = mask[row].bool()
        positive = losses[row][valid & (target[row] >= 0.5)]
        negative = losses[row][valid & (target[row] < 0.5)]
        if positive.numel() and negative.numel():
            rows.append(0.5 * (positive.mean() + negative.mean()))
        elif positive.numel():
            rows.append(positive.mean())
        elif negative.numel():
            rows.append(negative.mean())
    if not rows:
        return logits.sum() * 0.0
    return torch.stack(rows).mean()


def _masked_binary_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    selected = losses[mask.bool()]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    return selected.mean()


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    selected_logits = logits[mask.bool()]
    selected_target = target[mask.bool()]
    if selected_target.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(selected_logits, selected_target)


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

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from cid.model.torch_core import CIDTensorOutput


@dataclass(frozen=True, slots=True)
class CIDLossWeights:
    thought: float = 1.0
    allocation: float = 0.3
    display: float = 1.0
    roles: float = 0.2
    uncertainty: float = 0.1
    lifecycle: float = 0.1
    intent: float = 0.5
    source: float = 0.5
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
    allocation_targets: Tensor
    allocation_mask: Tensor
    display_ids: Tensor
    role_targets: Tensor
    uncertainty: Tensor
    lifecycle: Tensor
    need_targets: Tensor
    source_targets: Tensor
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
    allocation: Tensor
    display: Tensor
    roles: Tensor
    uncertainty: Tensor
    lifecycle: Tensor
    intent: Tensor
    source: Tensor
    revision: Tensor
    refresh: Tensor
    anchor_presence: Tensor
    anchor_kind: Tensor
    anchor_ground: Tensor
    link_presence: Tensor
    link_relation: Tensor
    link_target_kind: Tensor
    link_ground: Tensor


def cid_loss(
    output: CIDTensorOutput,
    targets: CIDTargets,
    weights: CIDLossWeights | None = None,
) -> CIDLoss:
    w = weights or CIDLossWeights()
    thought = F.mse_loss(output.thought_semantic, targets.thought_semantic)
    allocation = _masked_binary_cross_entropy(
        output.allocation_logits,
        targets.allocation_targets,
        targets.allocation_mask,
    )
    display = F.cross_entropy(
        output.display_logits.transpose(1, 2), targets.display_ids, ignore_index=-100
    )
    roles = F.binary_cross_entropy_with_logits(output.role_logits, targets.role_targets)
    uncertainty = F.mse_loss(output.uncertainty, targets.uncertainty)
    lifecycle = F.cross_entropy(
        output.lifecycle_logits.transpose(1, 2), targets.lifecycle, ignore_index=-100
    )
    intent = F.binary_cross_entropy_with_logits(output.need_logits, targets.need_targets)
    source = F.cross_entropy(
        output.source_logits.transpose(1, 2), targets.source_targets, ignore_index=-100
    )
    revision = F.cross_entropy(
        output.revision_logits.transpose(1, 2), targets.revision_targets, ignore_index=-100
    )
    refresh = F.cross_entropy(
        output.refresh_logits.transpose(1, 2), targets.refresh_targets, ignore_index=-100
    )
    anchor_presence = _masked_binary_cross_entropy(
        output.anchor_presence_logits,
        targets.anchor_presence_targets,
        targets.anchor_presence_mask,
    )
    anchor_kind = _masked_cross_entropy(
        output.anchor_kind_logits,
        targets.anchor_kind_targets,
        targets.anchor_mask,
    )
    anchor_ground = _masked_cosine_loss(
        output.anchor_query,
        targets.anchor_embeddings,
        targets.anchor_mask,
    )
    link_presence = _masked_binary_cross_entropy(
        output.link_presence_logits,
        targets.link_presence_targets,
        targets.link_presence_mask,
    )
    link_relation = _masked_cross_entropy(
        output.link_relation_logits,
        targets.link_relation_targets,
        targets.link_mask,
    )
    link_target_kind = _masked_cross_entropy(
        output.link_target_kind_logits,
        targets.link_target_kind_targets,
        targets.link_mask,
    )
    link_ground = _masked_cosine_loss(
        output.link_target_query,
        targets.link_target_embeddings,
        targets.link_mask,
    )
    total = (
        w.thought * thought
        + w.allocation * allocation
        + w.display * display
        + w.roles * roles
        + w.uncertainty * uncertainty
        + w.lifecycle * lifecycle
        + w.intent * intent
        + w.source * source
        + w.revision * revision
        + w.refresh * refresh
        + w.anchor_presence * anchor_presence
        + w.anchor_kind * anchor_kind
        + w.anchor_ground * anchor_ground
        + w.link_presence * link_presence
        + w.link_relation * link_relation
        + w.link_target_kind * link_target_kind
        + w.link_ground * link_ground
    )
    return CIDLoss(
        total=total,
        thought=thought,
        allocation=allocation,
        display=display,
        roles=roles,
        uncertainty=uncertainty,
        lifecycle=lifecycle,
        intent=intent,
        source=source,
        revision=revision,
        refresh=refresh,
        anchor_presence=anchor_presence,
        anchor_kind=anchor_kind,
        anchor_ground=anchor_ground,
        link_presence=link_presence,
        link_relation=link_relation,
        link_target_kind=link_target_kind,
        link_ground=link_ground,
    )


def _masked_cosine_loss(query: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    cosine = F.cosine_similarity(query, target, dim=-1)
    selected = cosine[mask.bool()]
    if selected.numel() == 0:
        return query.sum() * 0.0
    return (1.0 - selected).mean()


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

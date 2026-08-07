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
    ground: float = 0.2


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
    anchor_embeddings: Tensor
    anchor_mask: Tensor


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
    ground: Tensor


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
    ground = _masked_cosine_loss(
        output.anchor_query,
        targets.anchor_embeddings,
        targets.anchor_mask,
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
        + w.ground * ground
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
        ground=ground,
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

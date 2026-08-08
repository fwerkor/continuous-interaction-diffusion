"""Trainable CID model components.

Importing this package requires the optional `train` dependencies.
"""

from cid.model.illada import (
    ILLADA_8B_BASE,
    ILLADA_8B_BASE_REVISION,
    ILLADA_MASK_TOKEN_ID,
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
)
from cid.model.losses import CIDLoss, CIDLossWeights, CIDTargets, cid_loss
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput
from cid.model.torch_core import TorchCIDConfig, TorchCIDCore

__all__ = [
    "CIDLoss",
    "CIDLossWeights",
    "CIDTargets",
    "CIDTensorBatch",
    "CIDTensorOutput",
    "ILLADA_8B_BASE",
    "ILLADA_8B_BASE_REVISION",
    "ILLADA_MASK_TOKEN_ID",
    "ILLaDACIDAdapter",
    "ILLaDACIDConfig",
    "TorchCIDConfig",
    "TorchCIDCore",
    "cid_loss",
]

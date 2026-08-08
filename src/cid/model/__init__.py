"""Trainable CID model components.

Importing this package requires the optional `train` dependencies.
"""

from cid.model.diffusion import CIDDiffusionScheduler, DisplayCorruption, ThoughtCorruption
from cid.model.illada import (
    ILLADA_8B_BASE,
    ILLADA_8B_BASE_REVISION,
    ILLADA_MASK_TOKEN_ID,
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
)
from cid.model.losses import CIDLoss, CIDLossWeights, CIDTargets, cid_loss
from cid.model.materialize import (
    AnchorCandidate,
    ArgumentCandidate,
    CIDMaterializer,
    CIDMaterializerConfig,
    ClosedWorldMaterializationCatalog,
    ObjectCandidate,
    RevisionAction,
)
from cid.model.policy import (
    ILLaDAContextTensorizer,
    ILLaDANeuralPolicy,
    ILLaDANeuralPolicyConfig,
)
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput
from cid.model.torch_core import TorchCIDConfig, TorchCIDCore
from cid.model.training import (
    CIDTrainer,
    CIDTrainerConfig,
    CIDTrainerState,
    CIDTrainingBatch,
    CIDTrainingStep,
    CIDTrainReport,
    ILLaDATrajectoryTensorizer,
    collate_training_steps,
    load_cid_adapter_checkpoint,
    load_stage_b_checkpoint,
    save_stage_b_checkpoint,
    shard_transitions,
    trajectory_transitions,
    wrap_stage_a_ddp,
    wrap_stage_b_fsdp,
)

__all__ = [
    "CIDLoss",
    "CIDLossWeights",
    "CIDDiffusionScheduler",
    "CIDMaterializer",
    "CIDMaterializerConfig",
    "CIDTargets",
    "CIDTensorBatch",
    "CIDTensorOutput",
    "CIDTrainingBatch",
    "CIDTrainingStep",
    "CIDTrainer",
    "CIDTrainerConfig",
    "CIDTrainerState",
    "CIDTrainReport",
    "ClosedWorldMaterializationCatalog",
    "DisplayCorruption",
    "ILLADA_8B_BASE",
    "ILLADA_8B_BASE_REVISION",
    "ILLADA_MASK_TOKEN_ID",
    "ILLaDACIDAdapter",
    "ILLaDACIDConfig",
    "ILLaDAContextTensorizer",
    "ILLaDANeuralPolicy",
    "ILLaDANeuralPolicyConfig",
    "ILLaDATrajectoryTensorizer",
    "collate_training_steps",
    "load_cid_adapter_checkpoint",
    "load_stage_b_checkpoint",
    "save_stage_b_checkpoint",
    "shard_transitions",
    "trajectory_transitions",
    "wrap_stage_a_ddp",
    "wrap_stage_b_fsdp",
    "AnchorCandidate",
    "ArgumentCandidate",
    "ObjectCandidate",
    "RevisionAction",
    "ThoughtCorruption",
    "TorchCIDConfig",
    "TorchCIDCore",
    "cid_loss",
]

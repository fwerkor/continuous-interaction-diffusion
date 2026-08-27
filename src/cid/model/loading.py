from __future__ import annotations

from cid.model.illada import (
    ILLADA_8B_BASE,
    ILLADA_8B_BASE_REVISION,
    LLADA_MOE_7B_A1B_BASE,
    LLADA_MOE_7B_A1B_BASE_REVISION,
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
)
from cid.model.lfm import (
    LFM2_DIFFUSION_350M,
    LFM2_DIFFUSION_350M_REVISION,
    LFMCIDAdapter,
)


def pretrained_revision(model_name_or_path: str) -> str | None:
    if model_name_or_path == ILLADA_8B_BASE:
        return ILLADA_8B_BASE_REVISION
    if model_name_or_path == LLADA_MOE_7B_A1B_BASE:
        return LLADA_MOE_7B_A1B_BASE_REVISION
    if model_name_or_path == LFM2_DIFFUSION_350M:
        return LFM2_DIFFUSION_350M_REVISION
    return None


def backbone_model_type(model_name_or_path: str) -> str:
    from transformers import AutoConfig

    kwargs: dict[str, object] = {"trust_remote_code": True}
    revision = pretrained_revision(model_name_or_path)
    if revision is not None:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    return str(config.model_type)


def load_cid_adapter_from_pretrained(
    model_name_or_path: str,
    *,
    config: ILLaDACIDConfig | None = None,
    freeze_backbone: bool = False,
    **from_pretrained_kwargs: object,
) -> ILLaDACIDAdapter:
    if backbone_model_type(model_name_or_path) == "lfm2":
        return LFMCIDAdapter.from_pretrained(
            model_name_or_path,
            config=config,
            freeze_backbone=freeze_backbone,
            **from_pretrained_kwargs,
        )
    return ILLaDACIDAdapter.from_pretrained(
        model_name_or_path,
        config=config,
        freeze_backbone=freeze_backbone,
        **from_pretrained_kwargs,
    )

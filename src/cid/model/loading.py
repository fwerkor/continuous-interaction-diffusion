from __future__ import annotations

from cid.model.ar import AR_CID_MODEL_TYPES, prepare_ar_tokenizer
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


def load_cid_tokenizer(model_name_or_path: str, **tokenizer_kwargs: object):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer_kwargs.setdefault("trust_remote_code", True)
    revision = pretrained_revision(model_name_or_path)
    if revision is not None:
        tokenizer_kwargs.setdefault("revision", revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    config_kwargs: dict[str, object] = {
        "trust_remote_code": tokenizer_kwargs.get("trust_remote_code", True)
    }
    if tokenizer_kwargs.get("revision") is not None:
        config_kwargs["revision"] = tokenizer_kwargs["revision"]
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    model_type = str(config.model_type)
    if model_type == "cid":
        backbone_config = getattr(config, "backbone_config", {})
        model_type = str(backbone_config.get("model_type", ""))
    if model_type in AR_CID_MODEL_TYPES:
        prepare_ar_tokenizer(tokenizer)
    return tokenizer


def load_cid_adapter_from_pretrained(
    model_name_or_path: str,
    *,
    config: ILLaDACIDConfig | None = None,
    freeze_backbone: bool = False,
    **from_pretrained_kwargs: object,
) -> ILLaDACIDAdapter:
    model_type = backbone_model_type(model_name_or_path)
    if model_type == "cid":
        if config is not None:
            raise ValueError("unified CID checkpoints carry their adapter configuration internally")
        from cid.model.huggingface import load_cid_model_from_pretrained

        model = load_cid_model_from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
        model.cid_adapter.set_backbone_trainable(not freeze_backbone)
        return model.cid_adapter
    if model_type == "lfm2":
        return LFMCIDAdapter.from_pretrained(
            model_name_or_path,
            config=config,
            freeze_backbone=freeze_backbone,
            **from_pretrained_kwargs,
        )
    if model_type in AR_CID_MODEL_TYPES:
        from_pretrained_kwargs.setdefault("attn_implementation", "sdpa")
    return ILLaDACIDAdapter.from_pretrained(
        model_name_or_path,
        config=config,
        freeze_backbone=freeze_backbone,
        **from_pretrained_kwargs,
    )

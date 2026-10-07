from __future__ import annotations

import hashlib
from pathlib import Path

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


def backbone_source_identity(model_name_or_path: str) -> str:
    """Return a stable practical identity for the exact backbone source used by CID."""
    source = Path(model_name_or_path).expanduser()
    if source.exists():
        source = source.resolve()
        candidates = []
        if source.is_file():
            candidates = [source]
            root = source.parent
        else:
            root = source
            for pattern in (
                "config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "pytorch_model*.bin.index.json",
                "model*.bin",
            ):
                candidates.extend(source.glob(pattern))
        files = sorted({path.resolve() for path in candidates if path.is_file()})
        if not files:
            raise ValueError(f"local backbone source contains no model files: {source}")
        digest = hashlib.sha256()
        edge_bytes = 64 * 1024
        for file in files:
            stat = file.stat()
            digest.update(str(file.relative_to(root)).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            with file.open("rb") as handle:
                digest.update(handle.read(edge_bytes))
                if stat.st_size > edge_bytes:
                    handle.seek(max(0, stat.st_size - edge_bytes))
                    digest.update(handle.read(edge_bytes))
        return f"local-edge-sha256:{digest.hexdigest()}"
    revision = pretrained_revision(model_name_or_path)
    return f"hf:{model_name_or_path}@{revision or 'unversioned'}"


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
        adapter = model.cid_adapter
        adapter.set_backbone_trainable(not freeze_backbone)
    elif model_type == "lfm2":
        adapter = LFMCIDAdapter.from_pretrained(
            model_name_or_path,
            config=config,
            freeze_backbone=freeze_backbone,
            **from_pretrained_kwargs,
        )
    else:
        if model_type in AR_CID_MODEL_TYPES:
            from_pretrained_kwargs.setdefault("attn_implementation", "sdpa")
        adapter = ILLaDACIDAdapter.from_pretrained(
            model_name_or_path,
            config=config,
            freeze_backbone=freeze_backbone,
            **from_pretrained_kwargs,
        )
    adapter._cid_backbone_identity = backbone_source_identity(model_name_or_path)
    return adapter

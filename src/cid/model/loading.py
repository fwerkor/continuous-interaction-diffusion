from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

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


def _local_backbone_files(source: Path) -> tuple[Path, tuple[Path, ...]]:
    source = source.resolve()
    candidates: list[Path] = []
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
    files = tuple(sorted({path.resolve() for path in candidates if path.is_file()}))
    if not files:
        raise ValueError(f"local backbone source contains no model files: {source}")
    return root, files


def _file_manifest(root: Path, files: tuple[Path, ...]) -> list[dict[str, int | str]]:
    manifest: list[dict[str, int | str]] = []
    for file in files:
        stat = file.stat()
        manifest.append(
            {
                "path": str(file.relative_to(root)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "inode": int(stat.st_ino),
            }
        )
    return manifest


def _full_local_backbone_identity(root: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    chunk_bytes = 4 * 1024 * 1024
    for file in files:
        relative = str(file.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        stat = file.stat()
        digest.update(int(stat.st_size).to_bytes(8, "big"))
        with file.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


def _cached_local_backbone_identity(root: Path, files: tuple[Path, ...]) -> str:
    """Hash every backbone byte, caching by file metadata for multi-rank startup."""

    manifest = _file_manifest(root, files)
    cache_root = Path(
        os.environ.get(
            "CID_SOURCE_IDENTITY_CACHE_DIR",
            str(Path.home() / ".cache" / "cid" / "source-identities"),
        )
    ).expanduser()
    cache_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    cache_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        import fcntl

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(cached, dict)
                        and cached.get("manifest") == manifest
                        and isinstance(cached.get("identity"), str)
                    ):
                        return str(cached["identity"])
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            identity = _full_local_backbone_identity(root, files)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"manifest": manifest, "identity": identity},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(cache_path)
            return identity
    except (OSError, ImportError):
        return _full_local_backbone_identity(root, files)


def _legacy_backbone_source_identity(model_name_or_path: str) -> str | None:
    """Return only legacy identities that were already pinned strongly enough to trust."""

    source = Path(model_name_or_path).expanduser()
    if source.exists():
        root, files = _local_backbone_files(source)
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
    return None if revision is None else f"hf:{model_name_or_path}@{revision}"


def _resolve_hf_commit(model_name_or_path: str, revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name_or_path, revision=revision)
    except Exception as exc:
        if revision and len(revision) == 40 and all(
            character in "0123456789abcdefABCDEF" for character in revision
        ):
            return revision.lower()
        raise ValueError(
            "could not resolve the Hugging Face backbone revision to an exact commit SHA"
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError("Hugging Face did not return an exact backbone commit SHA")
    return sha


def backbone_source_identity(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    resolved_revision: str | None = None,
) -> str:
    """Return a content-exact local identity or resolved Hugging Face commit identity."""

    source = Path(model_name_or_path).expanduser()
    if source.exists():
        root, files = _local_backbone_files(source)
        return _cached_local_backbone_identity(root, files)
    requested_revision = revision or pretrained_revision(model_name_or_path)
    commit = resolved_revision or _resolve_hf_commit(model_name_or_path, requested_revision)
    return f"hf:{model_name_or_path}@{commit}"


def backbone_identity_matches(adapter: ILLaDACIDAdapter, saved_identity: object) -> bool:
    if saved_identity is None:
        return True
    current = getattr(adapter, "_cid_backbone_identity", None)
    legacy = getattr(adapter, "_cid_legacy_backbone_identity", None)
    return saved_identity in {current, legacy}


def apply_neural_contract_compatibility(
    adapter: ILLaDACIDAdapter,
    semantic_state: dict[str, Any],
    *,
    neural_contract_version: int,
) -> dict[str, Any]:
    """Restore legacy release semantics before constructing the runtime encoder."""

    state = dict(semantic_state)
    if neural_contract_version < 5:
        adapter.external_fusion.percept_residual = False
        state.setdefault("semantic_noise_scale", 1.0)
    elif "semantic_noise_scale" not in state:
        raise ValueError(
            "neural-contract-v5 semantic snapshot is missing semantic_noise_scale"
        )
    return state


def _canonical_tokenizer_json(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for item_key in sorted(value):
            if item_key == "name_or_path":
                continue
            result[item_key] = _canonical_tokenizer_json(value[item_key], key=item_key)
        return result
    if isinstance(value, list):
        return [_canonical_tokenizer_json(item) for item in value]
    if isinstance(value, str) and key is not None and key.endswith("_file"):
        return Path(value).name
    return value


def _serialized_tokenizer_digest(
    tokenizer: Any,
    *,
    strip_runtime_state: bool,
    tokenizer_json_runtime: dict[str, Any] | None = None,
) -> str:
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("tokenizer does not support save_pretrained")
    with tempfile.TemporaryDirectory(prefix="cid-tokenizer-identity-") as directory:
        root = Path(directory)
        save_pretrained(root)
        files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
        digest = hashlib.sha256()
        for file in files:
            relative = str(file.relative_to(root)).encode("utf-8")
            if file.suffix == ".json":
                try:
                    value = json.loads(file.read_text(encoding="utf-8"))
                    if file.name == "tokenizer.json" and isinstance(value, dict):
                        value = dict(value)
                        if strip_runtime_state:
                            value.pop("truncation", None)
                            value.pop("padding", None)
                        elif tokenizer_json_runtime is not None:
                            value.update(tokenizer_json_runtime)
                    payload = json.dumps(
                        _canonical_tokenizer_json(value),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                    payload = file.read_bytes()
            else:
                payload = file.read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


def tokenizer_source_identity(tokenizer: Any) -> str:
    """Hash the stable tokenizer contract independently from runtime padding/truncation state."""

    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if callable(save_pretrained):
        return (
            "tokenizer-v2-sha256:"
            + _serialized_tokenizer_digest(tokenizer, strip_runtime_state=True)
        )

    payload: dict[str, Any] = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
    }
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        payload["vocab"] = sorted(
            (str(token), int(index)) for token, index in get_vocab().items()
        )
    for name in (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "mask_token_id",
        "special_tokens_map",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None:
            payload[name] = value
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"tokenizer-v2-sha256:{hashlib.sha256(encoded).hexdigest()}"


def tokenizer_identity_matches(tokenizer: Any, saved_identity: object) -> bool:
    """Match current stable identities and legacy v1 fast-tokenizer runtime states."""

    if not isinstance(saved_identity, str):
        return False
    if saved_identity == tokenizer_source_identity(tokenizer):
        return True
    if not saved_identity.startswith("tokenizer-sha256:"):
        return False

    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        legacy = tokenizer_source_identity(tokenizer).replace(
            "tokenizer-v2-sha256:", "tokenizer-sha256:", 1
        )
        return saved_identity == legacy

    candidates = {
        "tokenizer-sha256:"
        + _serialized_tokenizer_digest(tokenizer, strip_runtime_state=False)
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token_id is not None and pad_token is not None:
        padding_side = str(getattr(tokenizer, "padding_side", "right")).lower()
        candidates.add(
            "tokenizer-sha256:"
            + _serialized_tokenizer_digest(
                tokenizer,
                strip_runtime_state=False,
                tokenizer_json_runtime={
                    "truncation": None,
                    "padding": {
                        "strategy": "BatchLongest",
                        "direction": "Left" if padding_side == "left" else "Right",
                        "pad_to_multiple_of": None,
                        "pad_id": int(pad_token_id),
                        "pad_type_id": int(getattr(tokenizer, "pad_token_type_id", 0)),
                        "pad_token": str(pad_token),
                    },
                },
            )
        )
    return saved_identity in candidates


def backbone_model_type(model_name_or_path: str, *, revision: str | None = None) -> str:
    from transformers import AutoConfig

    kwargs: dict[str, object] = {"trust_remote_code": True}
    selected_revision = revision or pretrained_revision(model_name_or_path)
    if selected_revision is not None:
        kwargs["revision"] = selected_revision
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
    requested_revision = from_pretrained_kwargs.get("revision")
    if requested_revision is not None:
        requested_revision = str(requested_revision)
    model_type = backbone_model_type(model_name_or_path, revision=requested_revision)
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
    resolved_revision = getattr(adapter.backbone.config, "_commit_hash", None)
    adapter._cid_backbone_identity = backbone_source_identity(
        model_name_or_path,
        revision=requested_revision,
        resolved_revision=(
            str(resolved_revision) if isinstance(resolved_revision, str) else None
        ),
    )
    adapter._cid_legacy_backbone_identity = _legacy_backbone_source_identity(
        model_name_or_path
    )
    return adapter

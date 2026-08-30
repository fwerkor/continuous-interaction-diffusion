from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any

_ACCELERATOR_TYPES = ("cuda", "npu")
_SUPPORTED_DEVICE_TYPES = ("auto", "cuda", "npu", "cpu")


def torch_device_available(torch_module: Any, device_type: str) -> bool:
    """Return whether a requested accelerator is usable without importing it on other hosts."""

    if device_type == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        return cuda is not None and bool(cuda.is_available())
    if device_type == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            return False
        npu = getattr(torch_module, "npu", None)
        return npu is not None and bool(npu.is_available())
    if device_type == "cpu":
        return True
    raise ValueError(f"unsupported torch device type {device_type!r}")


def resolve_torch_device_type(torch_module: Any, requested_device: str) -> str:
    """Resolve auto/CUDA/NPU/CPU selection, preferring CUDA then NPU for auto."""

    if requested_device not in _SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"unsupported device {requested_device!r}")
    if requested_device == "auto":
        for device_type in _ACCELERATOR_TYPES:
            if torch_device_available(torch_module, device_type):
                return device_type
        return "cpu"
    if requested_device in _ACCELERATOR_TYPES and not torch_device_available(
        torch_module, requested_device
    ):
        label = "CUDA" if requested_device == "cuda" else "Ascend NPU"
        raise RuntimeError(f"requested {label} device is unavailable")
    return requested_device


def distributed_backend(device_type: str) -> str:
    try:
        return {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}[device_type]
    except KeyError as exc:
        raise ValueError(f"unsupported distributed device type {device_type!r}") from exc


def configure_torch_accelerator(torch_module: Any, device_type: str, local_rank: int) -> None:
    """Select the local accelerator and install required runtime compatibility hooks."""

    if device_type == "cpu":
        return
    if device_type == "cuda":
        torch_module.cuda.set_device(local_rank)
        return
    if device_type != "npu":
        raise ValueError(f"unsupported accelerator type {device_type!r}")
    _configure_npu_runtime(torch_module, local_rank)


def wrap_npu_autocast(torch_module: Any, module: Any, *, dtype: Any) -> Any:
    """Wrap an unsharded model in torch_npu AMP for single-NPU full-parameter training."""

    torch_npu = importlib.import_module("torch_npu")

    class NPUAutocastForward(torch_module.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.module = wrapped

        def forward(self, *args, **kwargs):
            with torch_npu.npu.amp.autocast(dtype=dtype):
                return self.module(*args, **kwargs)

    return NPUAutocastForward(module)


def wrap_torch_autocast(
    torch_module: Any,
    module: Any,
    *,
    device_type: str,
    dtype: Any,
) -> Any:
    """Run a module under low-precision autocast while keeping FP32 trainable weights."""

    if device_type == "npu":
        return wrap_npu_autocast(torch_module, module, dtype=dtype)
    if device_type not in {"cuda", "cpu"}:
        raise ValueError(f"unsupported autocast device type {device_type!r}")

    class TorchAutocastForward(torch_module.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.module = wrapped

        def forward(self, *args, **kwargs):
            with torch_module.autocast(device_type=device_type, dtype=dtype):
                return self.module(*args, **kwargs)

    return TorchAutocastForward(module)


def _configure_npu_runtime(torch_module: Any, local_rank: int) -> None:
    torch_npu = importlib.import_module("torch_npu")
    npu = getattr(torch_module, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("Ascend NPU runtime is unavailable after importing torch_npu")

    npu.set_device(local_rank)
    if hasattr(npu, "set_compile_mode"):
        npu.set_compile_mode(jit_compile=False)

    cache_root = os.environ.get("CID_NPU_COMPILER_CACHE_DIR")
    if cache_root and hasattr(npu, "set_option"):
        rank_cache = Path(cache_root) / f"rank-{local_rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        npu.set_option(
            {
                "ACL_OP_COMPILER_CACHE_MODE": "enable",
                "ACL_OP_COMPILER_CACHE_DIR": str(rank_cache),
            }
        )

    _install_npu_fused_sdpa(torch_module, torch_npu)


def _install_npu_fused_sdpa(torch_module: Any, torch_npu: Any) -> None:
    """Use Ascend fused attention instead of the PyTorch 2.1 O(S^2) SDPA fallback."""

    functional = torch_module.nn.functional
    original = functional.scaled_dot_product_attention
    if getattr(original, "_cid_npu_fused", False):
        return

    def fused_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        **kwargs,
    ):
        if (
            query.device.type != "npu"
            or query.ndim != 4
            or key.ndim != 4
            or value.ndim != 4
            or is_causal
            or float(dropout_p) != 0.0
            or kwargs.get("enable_gqa", False)
        ):
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )

        head_dim = int(query.shape[-1])
        if head_dim not in {64, 80, 96, 120, 128, 256}:
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )

        # LFM2 rotary embeddings may leave Q/K in FP32 while V follows BF16/FP16
        # autocast. Ascend fused attention requires one shared Q/K/V dtype.
        qkv_dtypes = {query.dtype, key.dtype, value.dtype}
        if len(qkv_dtypes) > 1:
            low_precision = [
                tensor.dtype
                for tensor in (query, key, value)
                if tensor.dtype in {torch_module.bfloat16, torch_module.float16}
            ]
            if (
                not low_precision
                or len(set(low_precision)) != 1
                or not qkv_dtypes <= {torch_module.float32, low_precision[0]}
            ):
                return original(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    **kwargs,
                )
            attention_dtype = low_precision[0]
            query = query.to(dtype=attention_dtype)
            key = key.to(dtype=attention_dtype)
            value = value.to(dtype=attention_dtype)

        fusion_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch_module.bool:
                # PyTorch SDPA uses True=allowed; Ascend fused attention uses True=masked.
                fusion_mask = ~attn_mask
            elif torch_module.is_floating_point(attn_mask):
                fusion_mask = attn_mask < 0
            else:
                return original(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    **kwargs,
                )

        if fusion_mask is not None and fusion_mask.shape[-2] == 1 and query.shape[-2] != 1:
            fusion_mask = fusion_mask.expand(
                fusion_mask.shape[0],
                fusion_mask.shape[1],
                query.shape[-2],
                fusion_mask.shape[-1],
            ).contiguous()

        scale = kwargs.get("scale")
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        return torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            int(query.shape[1]),
            "BNSD",
            atten_mask=fusion_mask,
            scale=float(scale),
            keep_prob=1.0,
            sparse_mode=0,
        )[0]

    fused_sdpa._cid_npu_fused = True
    functional.scaled_dot_product_attention = fused_sdpa

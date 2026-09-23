from __future__ import annotations

from types import ModuleType

from torch import Tensor

try:
    import cid_engine as _ENGINE
except ModuleNotFoundError as exc:
    if exc.name != "cid_engine":
        raise
    _ENGINE = None


def accelerator_engine(
    tensor: Tensor,
    *,
    capability: str | None = None,
) -> ModuleType | None:
    if _ENGINE is None:
        return None
    device_type = tensor.device.type
    if device_type == "cuda":
        if not _ENGINE.CUDA_BACKEND_BUILT:
            return None
    elif device_type == "npu":
        if not getattr(_ENGINE, "CANN_BACKEND_BUILT", False):
            return None
    else:
        return None
    if capability is not None and getattr(_ENGINE, capability, None) is None:
        return None
    return _ENGINE


def cuda_engine(
    tensor: Tensor,
    *,
    capability: str | None = None,
) -> ModuleType | None:
    if _ENGINE is None or not _ENGINE.CUDA_BACKEND_BUILT or not tensor.is_cuda:
        return None
    if capability is not None and getattr(_ENGINE, capability, None) is None:
        return None
    return _ENGINE


def native_engine(*, capability: str | None = None) -> ModuleType | None:
    if _ENGINE is None:
        return None
    if capability is not None and getattr(_ENGINE, capability, None) is None:
        return None
    return _ENGINE

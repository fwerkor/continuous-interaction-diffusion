from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cid import accelerator


class _DeviceModule:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.selected: int | None = None

    def is_available(self) -> bool:
        return self._available

    def set_device(self, index: int) -> None:
        self.selected = index


class _FakeTorch:
    def __init__(self, *, cuda: bool, npu: bool = False) -> None:
        self.cuda = _DeviceModule(cuda)
        self.npu = _DeviceModule(npu)


def test_auto_device_resolution_prefers_cuda_then_npu(monkeypatch) -> None:
    monkeypatch.setattr(accelerator.importlib, "import_module", lambda name: SimpleNamespace())
    assert accelerator.resolve_torch_device_type(_FakeTorch(cuda=True, npu=True), "auto") == "cuda"
    assert accelerator.resolve_torch_device_type(_FakeTorch(cuda=False, npu=True), "auto") == "npu"
    assert accelerator.resolve_torch_device_type(_FakeTorch(cuda=False, npu=False), "auto") == "cpu"


def test_explicit_npu_requires_torch_npu(monkeypatch) -> None:
    def missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr(accelerator.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Ascend NPU"):
        accelerator.resolve_torch_device_type(_FakeTorch(cuda=False, npu=True), "npu")


def test_distributed_backend_mapping() -> None:
    assert accelerator.distributed_backend("cuda") == "nccl"
    assert accelerator.distributed_backend("npu") == "hccl"
    assert accelerator.distributed_backend("cpu") == "gloo"
    with pytest.raises(ValueError, match="unsupported distributed device"):
        accelerator.distributed_backend("mps")


def test_torch_autocast_preserves_stage_a_ddp_controls() -> None:
    class Wrapped(torch.nn.Module):
        _cid_stage_a_ddp = True

        def forward(self, value):
            return value

        def no_sync(self):
            return nullcontext("delegated")

    wrapped = accelerator.wrap_torch_autocast(
        torch,
        Wrapped(),
        device_type="cpu",
        dtype=torch.bfloat16,
    )

    assert wrapped._cid_stage_a_ddp is True
    with wrapped.no_sync() as value:
        assert value == "delegated"

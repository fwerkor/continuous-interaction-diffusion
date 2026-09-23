from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
native_engine_module = import_module("cid.model.native_engine")


class _FakeTensor:
    def __init__(self, device_type: str) -> None:
        self.device = SimpleNamespace(type=device_type)


def test_accelerator_engine_selects_cann_backend(monkeypatch) -> None:
    engine = SimpleNamespace(
        CUDA_BACKEND_BUILT=False,
        CANN_BACKEND_BUILT=True,
        display_corrupt_from_random=object(),
    )
    monkeypatch.setattr(native_engine_module, "_ENGINE", engine)

    selected = native_engine_module.accelerator_engine(
        _FakeTensor("npu"),
        capability="display_corrupt_from_random",
    )

    assert selected is engine


def test_accelerator_engine_rejects_missing_backend_or_capability(monkeypatch) -> None:
    engine = SimpleNamespace(
        CUDA_BACKEND_BUILT=True,
        CANN_BACKEND_BUILT=False,
    )
    monkeypatch.setattr(native_engine_module, "_ENGINE", engine)

    assert native_engine_module.accelerator_engine(_FakeTensor("npu")) is None
    assert (
        native_engine_module.accelerator_engine(
            _FakeTensor("cuda"),
            capability="missing",
        )
        is None
    )

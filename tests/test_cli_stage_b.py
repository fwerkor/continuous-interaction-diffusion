from __future__ import annotations

import shutil
from importlib import import_module

import pytest

cli = import_module("cid.cli")


def test_stage_b_cpu_target_uses_gloo_without_gpu_rank_minimum() -> None:
    assert cli._stage_b_execution_target(
        "cpu",
        cuda_available=True,
        npu_available=False,
        world_size=1,
        local_rank=0,
        dtype="bf16",
        cpu_offload=False,
    ) == ("cpu", None, "gloo")
    assert cli._stage_b_execution_target(
        "auto",
        cuda_available=False,
        npu_available=False,
        world_size=2,
        local_rank=1,
        dtype="bf16",
        cpu_offload=False,
    ) == ("cpu", None, "gloo")


def test_stage_b_cuda_target_keeps_four_rank_minimum() -> None:
    with pytest.raises(RuntimeError, match="at least four GPU ranks"):
        cli._stage_b_execution_target(
            "cuda",
            cuda_available=True,
            npu_available=False,
            world_size=2,
            local_rank=1,
            dtype="bf16",
            cpu_offload=False,
        )
    assert cli._stage_b_execution_target(
        "auto",
        cuda_available=True,
        npu_available=False,
        world_size=4,
        local_rank=3,
        dtype="fp16",
        cpu_offload=True,
    ) == ("cuda", 3, "nccl")


def test_stage_b_npu_target_supports_single_or_four_plus_ranks() -> None:
    assert cli._stage_b_execution_target(
        "npu",
        cuda_available=False,
        npu_available=True,
        world_size=1,
        local_rank=0,
        dtype="bf16",
        cpu_offload=False,
    ) == ("npu", 0, "hccl")
    with pytest.raises(RuntimeError, match="one NPU rank.*at least four NPU ranks"):
        cli._stage_b_execution_target(
            "npu",
            cuda_available=False,
            npu_available=True,
            world_size=2,
            local_rank=1,
            dtype="bf16",
            cpu_offload=False,
        )
    assert cli._stage_b_execution_target(
        "auto",
        cuda_available=False,
        npu_available=True,
        world_size=4,
        local_rank=2,
        dtype="bf16",
        cpu_offload=False,
    ) == ("npu", 2, "hccl")
    with pytest.raises(ValueError, match="BF16 only|--dtype bf16 only"):
        cli._stage_b_execution_target(
            "npu",
            cuda_available=False,
            npu_available=True,
            world_size=1,
            local_rank=0,
            dtype="fp16",
            cpu_offload=False,
        )
    with pytest.raises(ValueError, match="not used by NPU"):
        cli._stage_b_execution_target(
            "npu",
            cuda_available=False,
            npu_available=True,
            world_size=4,
            local_rank=0,
            dtype="bf16",
            cpu_offload=True,
        )


def test_stage_b_cpu_rejects_fp16_and_redundant_cpu_offload() -> None:
    with pytest.raises(ValueError, match="supports --dtype bf16 only"):
        cli._stage_b_execution_target(
            "cpu",
            cuda_available=False,
            npu_available=False,
            world_size=1,
            local_rank=0,
            dtype="fp16",
            cpu_offload=False,
        )
    with pytest.raises(ValueError, match="only meaningful for CUDA"):
        cli._stage_b_execution_target(
            "cpu",
            cuda_available=False,
            npu_available=False,
            world_size=1,
            local_rank=0,
            dtype="bf16",
            cpu_offload=True,
        )


def test_stage_b_single_process_cpu_initializes_local_gloo(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    dist = torch.distributed
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)

    rendezvous_dir = cli._init_stage_b_process_group(
        dist,
        backend="gloo",
        world_size=1,
    )
    try:
        assert rendezvous_dir is not None
        assert dist.is_initialized()
        assert dist.get_rank() == 0
        assert dist.get_world_size() == 1
        assert dist.get_backend() == "gloo"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous_dir is not None:
            shutil.rmtree(rendezvous_dir, ignore_errors=True)

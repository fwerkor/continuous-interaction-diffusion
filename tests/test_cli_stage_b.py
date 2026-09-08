from __future__ import annotations

import shutil
from importlib import import_module

import pytest

cli = import_module("cid.cli")


def test_stage_b_compact_high_memory_profile_prioritizes_throughput() -> None:
    profile = cli._resolve_stage_b_performance_profile(
        model_type="lfm2",
        device_type="cuda",
        accelerator_memory_bytes=48 * 1024**3,
        cpu_offload=False,
        micro_batch_size=None,
        mlp_chunk_size=None,
        norm_chunk_size=None,
        gradient_checkpointing=None,
    )
    assert profile.name == "compact-high-memory"
    assert profile.micro_batch_size == 4
    assert profile.mlp_chunk_size == 512
    assert profile.norm_chunk_size == 1024
    assert profile.gradient_checkpointing is False


def test_stage_b_illada_high_memory_profile_uses_safe_throughput_tuning() -> None:
    profile = cli._resolve_stage_b_performance_profile(
        model_type="illada",
        device_type="cuda",
        accelerator_memory_bytes=48 * 1024**3,
        cpu_offload=False,
        micro_batch_size=None,
        mlp_chunk_size=None,
        norm_chunk_size=None,
        gradient_checkpointing=None,
    )
    assert profile.name == "illada-high-memory"
    assert profile.micro_batch_size == 1
    assert profile.mlp_chunk_size == 512
    assert profile.norm_chunk_size == 1024
    assert profile.gradient_checkpointing is True


def test_stage_b_large_model_keeps_memory_safe_profile() -> None:
    profile = cli._resolve_stage_b_performance_profile(
        model_type="llada",
        device_type="cuda",
        accelerator_memory_bytes=48 * 1024**3,
        cpu_offload=False,
        micro_batch_size=None,
        mlp_chunk_size=None,
        norm_chunk_size=None,
        gradient_checkpointing=None,
    )
    assert profile.name == "memory-safe"
    assert profile.micro_batch_size == 1
    assert profile.mlp_chunk_size == 256
    assert profile.norm_chunk_size == 256
    assert profile.gradient_checkpointing is True


def test_stage_b_illada_low_memory_falls_back_to_memory_safe_profile() -> None:
    profile = cli._resolve_stage_b_performance_profile(
        model_type="illada",
        device_type="cuda",
        accelerator_memory_bytes=24 * 1024**3,
        cpu_offload=False,
        micro_batch_size=None,
        mlp_chunk_size=None,
        norm_chunk_size=None,
        gradient_checkpointing=None,
    )
    assert profile.name == "memory-safe"
    assert profile.micro_batch_size == 1
    assert profile.mlp_chunk_size == 256
    assert profile.norm_chunk_size == 256
    assert profile.gradient_checkpointing is True


def test_stage_b_performance_profile_preserves_explicit_overrides() -> None:
    profile = cli._resolve_stage_b_performance_profile(
        model_type="lfm2",
        device_type="cuda",
        accelerator_memory_bytes=48 * 1024**3,
        cpu_offload=False,
        micro_batch_size=2,
        mlp_chunk_size=384,
        norm_chunk_size=640,
        gradient_checkpointing=True,
    )
    assert profile.micro_batch_size == 2
    assert profile.mlp_chunk_size == 384
    assert profile.norm_chunk_size == 640
    assert profile.gradient_checkpointing is True


def test_stage_b_compact_profile_stays_conservative_on_low_memory_or_offload() -> None:
    for memory, cpu_offload in ((24 * 1024**3, False), (48 * 1024**3, True)):
        profile = cli._resolve_stage_b_performance_profile(
            model_type="lfm2",
            device_type="cuda",
            accelerator_memory_bytes=memory,
            cpu_offload=cpu_offload,
            micro_batch_size=None,
            mlp_chunk_size=None,
            norm_chunk_size=None,
            gradient_checkpointing=None,
        )
        assert profile.name == "memory-safe"
        assert profile.micro_batch_size == 1
        assert profile.gradient_checkpointing is True


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
        dtype="bf16",
        cpu_offload=True,
    ) == ("cuda", 3, "nccl")


def test_stage_b_cuda_rejects_fp16_without_loss_scaling() -> None:
    with pytest.raises(ValueError, match="--dtype bf16 only.*loss scaling"):
        cli._stage_b_execution_target(
            "cuda",
            cuda_available=True,
            npu_available=False,
            world_size=4,
            local_rank=0,
            dtype="fp16",
            cpu_offload=False,
        )


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

"""Run via torchrun --nproc-per-node=2 to verify language/CID bucket synchronization."""

from __future__ import annotations

import torch.distributed as dist

from cid.data import TrajectoryExample
from cid.model.training import CIDRolloutWindow, shard_rollout_windows


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        windows = tuple(
            CIDRolloutWindow(
                TrajectoryExample(
                    example_id=f"cid-{i}", prompt=f"Question {i}?", target_display="Answer",
                ),
                (-1,),
            )
            for i in range(25)
        ) + tuple(
            CIDRolloutWindow(
                TrajectoryExample(
                    example_id=f"language_sft:{i}", prompt=f"Question {i}?",
                    target_display="A complete sentence.",
                    metadata={"cid_language_sft": True},
                ),
                (-1,),
                is_language_sft=True,
            )
            for i in range(13)
        )
        local = shard_rollout_windows(
            windows, world_size=world_size, rank=rank, seed=6, epoch=1,
            micro_batch_size=2, length_aware=True,
            portable_bucket_order=True, balanced_bucket_order=True,
        )
        shape = [(len(w.source_steps), w.loss_weight, w.is_language_sft) for w in local]
        shapes = [None] * world_size
        dist.all_gather_object(shapes, shape)
        assert all(other == shape for other in shapes), "rank-local loss schedules differ"
        assert any(mode for _, _, mode in shape)
        assert any(not mode for _, _, mode in shape)
        if rank == 0:
            print(f"distributed_language_mix_ok world_size={world_size} windows={len(local)}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

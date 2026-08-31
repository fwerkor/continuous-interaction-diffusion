from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from test_training import TinyTokenizer, make_adapter, make_rollout_trajectory

from cid.model import (
    CIDRolloutWindow,
    CIDTrainer,
    CIDTrainerConfig,
    ILLaDATrajectoryTensorizer,
    shard_rollout_windows,
)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("distributed progress regression requires exactly two ranks")

        base = make_rollout_trajectory()
        windows = (
            CIDRolloutWindow(
                example=replace(base, example_id="short"),
                source_steps=(0, 1),
            ),
            CIDRolloutWindow(
                example=replace(base, example_id="long-a"),
                source_steps=(-1, 0, 1),
            ),
            CIDRolloutWindow(
                example=replace(base, example_id="long-b"),
                source_steps=(-1, 0, 1),
            ),
        )
        local_windows = shard_rollout_windows(
            windows,
            world_size=world_size,
            rank=rank,
            seed=1,
            epoch=1,
            shuffle=False,
            micro_batch_size=1,
            length_aware=False,
            zero_gradient_padding=True,
            portable_bucket_order=True,
        )
        # The first bucket contains one real window and one zero-gradient pad. Rank 1
        # therefore has no local transition loss while rank 0 advances the global
        # optimizer schedule. Both ranks must still enter the progress callback.
        assert local_windows[0].is_padding is (rank == 1)

        adapter = make_adapter(seed=300 + rank)
        trainer = CIDTrainer(
            adapter,
            ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
            CIDTrainerConfig(
                learning_rate=1e-3,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                rollout_horizon=3,
                teacher_forcing_epochs=99,
                timestep_min=0.0,
                timestep_max=0.0,
            ),
        )
        callbacks = 0

        def progress_callback(progress) -> None:
            nonlocal callbacks
            callbacks += 1
            marker = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(marker, op=dist.ReduceOp.SUM)
            assert int(marker.item()) == world_size
            assert progress.optimizer_steps > 0

        trainer.train_rollout_windows(
            local_windows,
            epochs=1,
            shuffle=False,
            preserve_order=True,
            progress_every_optimizer_steps=1,
            progress_callback=progress_callback,
        )
        assert callbacks > 0
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

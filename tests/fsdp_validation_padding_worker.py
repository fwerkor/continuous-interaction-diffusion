from __future__ import annotations

import torch
import torch.distributed as dist
from test_training import TinyTokenizer, make_adapter, make_trajectory

from cid.model import (
    CIDRolloutWindow,
    CIDTrainer,
    CIDTrainerConfig,
    ILLaDATrajectoryTensorizer,
    shard_rollout_windows,
    wrap_stage_b_fsdp,
)
from cid.model.encoding import ILLaDATextEncoder


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("FSDP validation padding regression requires exactly two ranks")

        base = make_trajectory()
        windows = (CIDRolloutWindow(example=base, source_steps=(0,)),)
        local_windows = shard_rollout_windows(
            windows,
            world_size=world_size,
            rank=rank,
            seed=1,
            epoch=1,
            shuffle=False,
            micro_batch_size=1,
            zero_gradient_padding=True,
            portable_bucket_order=True,
        )
        assert len(local_windows) == 1
        assert local_windows[0].is_padding is (rank == 1)

        adapter = make_adapter(seed=400 + rank)
        adapter.set_backbone_trainable(True)
        tokenizer = TinyTokenizer()
        snapshot = ILLaDATextEncoder.from_frozen_snapshot(
            adapter,
            tokenizer,
            device="cpu",
            dtype=torch.bfloat16,
        )
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        trainer = CIDTrainer(
            adapter,
            ILLaDATrajectoryTensorizer(adapter, tokenizer, text_encoder=snapshot),
            CIDTrainerConfig(micro_batch_size=1, timestep_min=0.0, timestep_max=0.0),
            forward_model=fsdp,
            gradient_clipper=fsdp.clip_grad_norm_,
        )

        report = trainer.evaluate_rollout_windows(local_windows, seed=2)
        assert report.transitions == (0 if rank == 1 else 1)
        assert torch.isfinite(torch.tensor(report.mean_loss))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

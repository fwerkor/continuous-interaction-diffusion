from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from test_training import TinyTokenizer, make_adapter, make_trajectory

from cid.model import CIDTrainer, CIDTrainerConfig, ILLaDATrajectoryTensorizer, wrap_stage_a_ddp


def make_trainer(*, seed: int, optimized: bool, accumulation_steps: int) -> CIDTrainer:
    adapter = make_adapter(seed=seed)
    ddp = wrap_stage_a_ddp(adapter, device_ids=None)
    if not optimized:
        ddp._cid_stage_a_ddp = False
    return CIDTrainer(
        adapter,
        ILLaDATrajectoryTensorizer(adapter, TinyTokenizer()),
        CIDTrainerConfig(
            learning_rate=1e-3,
            micro_batch_size=1,
            gradient_accumulation_steps=accumulation_steps,
            timestep_min=0.0,
            timestep_max=0.0,
            seed=19,
        ),
        forward_model=ddp,
    )


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise RuntimeError("DDP accumulation regression requires exactly two ranks")

        base = make_trajectory()
        example = replace(
            base,
            example_id=f"rank-{rank}",
            prompt=base.prompt + (" alpha" if rank == 0 else " beta"),
        )

        for accumulation_steps in (2, 3):
            torch.manual_seed(701)
            reference = make_trainer(
                seed=701,
                optimized=False,
                accumulation_steps=accumulation_steps,
            )
            torch.manual_seed(701)
            optimized = make_trainer(
                seed=701,
                optimized=True,
                accumulation_steps=accumulation_steps,
            )

            reference_report = reference.train_examples((example,), epochs=1, shuffle=False)
            optimized_report = optimized.train_examples((example,), epochs=1, shuffle=False)

            assert reference_report.optimizer_steps == 1
            assert optimized_report.optimizer_steps == 1
            reference_parameters = dict(reference.adapter.named_parameters())
            optimized_parameters = dict(optimized.adapter.named_parameters())
            assert reference_parameters.keys() == optimized_parameters.keys()
            for name, reference_parameter in reference_parameters.items():
                optimized_parameter = optimized_parameters[name]
                torch.testing.assert_close(
                    optimized_parameter,
                    reference_parameter,
                    rtol=2e-6,
                    atol=2e-7,
                    msg=lambda message, name=name: f"{name}: {message}",
                )

        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from fsdp_stage_b_worker import TinyBackbone, TinyTrainer, make_batch

from cid.model import (
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
    load_stage_b_checkpoint,
    stage_b_adamw_parameter_groups,
    wrap_stage_b_fsdp,
)
from cid.model.encoding import ILLaDATextEncoder


def main() -> None:
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(707)
        adapter = ILLaDACIDAdapter(
            TinyBackbone(),
            ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
            freeze_backbone=False,
        )
        tokenizer = object()
        semantic_encoder = ILLaDATextEncoder.from_frozen_snapshot(
            adapter,
            tokenizer,
            device="cpu",
            dtype=torch.bfloat16,
        )
        optimizer_groups = stage_b_adamw_parameter_groups(
            adapter, backbone_lr_scale=0.5, weight_decay=0.01
        )
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-3)
        trainer = TinyTrainer(
            adapter,
            tokenizer=tokenizer,
            text_encoder=semantic_encoder,
        )
        checkpoint = Path(os.environ["CID_FSDP_SMOKE_DIR"])

        metadata = load_stage_b_checkpoint(
            fsdp,
            optimizer,
            trainer,
            checkpoint,
            expected_dataset_sha256="two-rank-smoke",
        )
        assert int(metadata["world_size"]) != dist.get_world_size()
        assert trainer.marker == 10
        assert trainer.portable_seed is not None
        assert optimizer.state

        output = fsdp(make_batch())
        loss = output.thought_semantic.float().square().mean()
        loss = loss + output.display_logits.float().square().mean()
        loss.backward()
        fsdp.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

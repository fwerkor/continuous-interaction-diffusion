from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from cid.model import (
    CIDTensorBatch,
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
    load_stage_b_checkpoint,
    load_stage_b_model_checkpoint,
    save_stage_b_checkpoint,
    stage_b_adamw_parameter_groups,
    wrap_stage_b_fsdp,
)


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 32
    num_attention_heads = 4
    max_position_embeddings = 64


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(TinyConfig.hidden_size, TinyConfig.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del position_ids
        assert return_dict
        weights = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        context = (inputs_embeds * weights).sum(dim=1, keepdim=True)
        context = context / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = hidden + layer(context)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.embed_tokens = nn.Embedding(TinyConfig.vocab_size, TinyConfig.hidden_size)
        self.decoder = TinyDecoder()
        self.lm_head = nn.Linear(TinyConfig.hidden_size, TinyConfig.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_decoder(self) -> TinyDecoder:
        return self.decoder


class TinyTrainer:
    def __init__(self, adapter: ILLaDACIDAdapter, marker: int = 0) -> None:
        self.adapter = adapter
        self.marker = marker
        self.config = SimpleNamespace(seed=19)
        self.portable_seed: int | None = None

    def local_progress_state(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "trainer_config": {"gradient_accumulation_steps": dist.get_world_size()},
            "trainer_state": {"optimizer_steps": 1},
        }

    def restore_local_progress_state(self, state: dict[str, object]) -> None:
        self.marker = int(state["marker"])

    def restore_portable_progress_state(
        self,
        state: dict[str, object],
        *,
        seed: int,
    ) -> None:
        self.marker = int(state["marker"])
        self.portable_seed = seed


def make_batch() -> CIDTensorBatch:
    d = TinyConfig.hidden_size
    return CIDTensorBatch(
        thought_semantic=torch.randn(1, 3, d),
        role_features=torch.rand(1, 3, 6),
        uncertainty=torch.rand(1, 3, 1),
        local_noise=torch.rand(1, 3, 1),
        slot_occupancy=torch.tensor([[[1.0], [1.0], [0.0]]]),
        prompt_ids=torch.randint(0, TinyConfig.vocab_size, (1, 4)),
        display_ids=torch.randint(0, TinyConfig.vocab_size, (1, 3)),
        display_noise=torch.rand(1, 3, 1),
        fact_memory=torch.empty(1, 0, d),
        percept_memory=torch.empty(1, 0, d),
        source_memory=torch.empty(1, 0, d),
    )


def main() -> None:
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(101)
        adapter = ILLaDACIDAdapter(
            TinyBackbone(),
            ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
            freeze_backbone=False,
        )
        optimizer_groups = stage_b_adamw_parameter_groups(
            adapter, backbone_lr_scale=0.5, weight_decay=0.01
        )
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        assert isinstance(fsdp, FullyShardedDataParallel)
        assert fsdp.sharding_strategy is ShardingStrategy.FULL_SHARD
        optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-3)

        output = fsdp(make_batch())
        loss = output.thought_semantic.float().square().mean()
        loss = loss + output.display_logits.float().square().mean()
        loss.backward()
        fsdp.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint = Path(os.environ["CID_FSDP_SMOKE_DIR"])
        trainer = TinyTrainer(adapter, marker=dist.get_rank() + 10)
        save_stage_b_checkpoint(
            fsdp,
            optimizer,
            trainer,
            checkpoint,
            dataset_sha256="two-rank-smoke",
        )
        dist.barrier()

        torch.manual_seed(303)
        restored_adapter = ILLaDACIDAdapter(
            TinyBackbone(),
            ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
            freeze_backbone=False,
        )
        restored_optimizer_groups = stage_b_adamw_parameter_groups(
            restored_adapter, backbone_lr_scale=0.5, weight_decay=0.01
        )
        restored = wrap_stage_b_fsdp(
            restored_adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        restored_optimizer = torch.optim.AdamW(restored_optimizer_groups, lr=1e-3)
        restored_trainer = TinyTrainer(restored_adapter)
        load_stage_b_checkpoint(
            restored,
            restored_optimizer,
            restored_trainer,
            checkpoint,
            expected_dataset_sha256="two-rank-smoke",
        )
        assert restored_trainer.marker == dist.get_rank() + 10

        restored_output = restored(make_batch())
        restored_loss = restored_output.thought_semantic.float().square().mean()
        restored_loss = restored_loss + restored_output.display_logits.float().square().mean()
        restored_loss.backward()
        restored.clip_grad_norm_(1.0)
        restored_optimizer.step()
        restored_optimizer.zero_grad(set_to_none=True)

        torch.manual_seed(404)
        inference_adapter = ILLaDACIDAdapter(
            TinyBackbone(),
            ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
            freeze_backbone=False,
        )
        inference = wrap_stage_b_fsdp(
            inference_adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        load_stage_b_model_checkpoint(inference, inference_adapter, checkpoint)
        inference(make_batch())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

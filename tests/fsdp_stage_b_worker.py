from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from cid.model import CIDTensorBatch, ILLaDACIDAdapter, ILLaDACIDConfig, wrap_stage_b_fsdp


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
        fsdp = wrap_stage_b_fsdp(
            adapter,
            device_id=torch.device("cpu"),
            compute_dtype=torch.bfloat16,
        )
        assert isinstance(fsdp, FullyShardedDataParallel)
        assert fsdp.sharding_strategy is ShardingStrategy.FULL_SHARD
        optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)

        output = fsdp(make_batch())
        loss = output.thought_semantic.float().square().mean()
        loss = loss + output.display_logits.float().square().mean()
        loss.backward()
        fsdp.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(fsdp, optimizer, options=options)
        checkpoint = Path(os.environ["CID_FSDP_SMOKE_DIR"])
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=checkpoint,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            assert (checkpoint / ".metadata").is_file()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

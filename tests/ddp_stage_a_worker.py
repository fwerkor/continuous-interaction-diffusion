from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from cid.model import CIDTensorBatch, ILLaDACIDAdapter, ILLaDACIDConfig, wrap_stage_a_ddp


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 32
    num_attention_heads = 4
    max_position_embeddings = 64


class TinyDecoder(nn.Module):
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
        return SimpleNamespace(last_hidden_state=inputs_embeds + context)


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


def make_batch(*, include_external: bool) -> CIDTensorBatch:
    d_model = TinyConfig.hidden_size
    fact_memory = torch.randn(1, 1, d_model) if include_external else torch.empty(1, 0, d_model)
    return CIDTensorBatch(
        thought_semantic=torch.randn(1, 3, d_model),
        role_features=torch.rand(1, 3, 6),
        uncertainty=torch.rand(1, 3, 1),
        local_noise=torch.rand(1, 3, 1),
        slot_occupancy=torch.tensor([[[1.0], [1.0], [0.0]]]),
        prompt_ids=torch.randint(0, TinyConfig.vocab_size, (1, 4)),
        display_ids=torch.randint(0, TinyConfig.vocab_size, (1, 3)),
        display_noise=torch.rand(1, 3, 1),
        fact_memory=fact_memory,
        percept_memory=torch.empty(1, 0, d_model),
        source_memory=torch.empty(1, 0, d_model),
    )


def output_loss(output: object) -> torch.Tensor:
    names = (
        "thought_semantic",
        "convergence_logits",
        "allocation_logits",
        "role_logits",
        "uncertainty",
        "noise_delta",
        "lifecycle_logits",
        "display_logits",
        "need_logits",
        "source_logits",
        "need_target_cell_logits",
        "need_target_display_logits",
        "argument_presence_logits",
        "argument_query",
        "anchor_query",
        "anchor_presence_logits",
        "anchor_kind_logits",
        "link_presence_logits",
        "link_relation_logits",
        "link_target_kind_logits",
        "link_target_query",
        "revision_logits",
        "refresh_logits",
    )
    return sum(getattr(output, name).float().sum() for name in names)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        torch.manual_seed(211)
        adapter = ILLaDACIDAdapter(
            TinyBackbone(),
            ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
            freeze_backbone=True,
        )
        ddp = wrap_stage_a_ddp(adapter, device_ids=None)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in adapter.parameters() if parameter.requires_grad),
            lr=1e-3,
        )

        for step in range(6):
            include_external = bool((step + rank) % 2)
            optimizer.zero_grad(set_to_none=True)
            loss = output_loss(ddp(make_batch(include_external=include_external)))
            loss.backward()
            assert all(
                parameter.grad is not None
                for parameter in adapter.parameters()
                if parameter.requires_grad
            )
            optimizer.step()

            marker = torch.tensor([float(step), float(rank), float(include_external)])
            dist.all_reduce(marker)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

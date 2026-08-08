from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from cid.model import (
    ILLADA_8B_BASE,
    ILLADA_8B_BASE_REVISION,
    ILLADA_MASK_TOKEN_ID,
    CIDTensorBatch,
    ILLaDACIDAdapter,
    ILLaDACIDConfig,
)


def main() -> None:
    config = AutoConfig.from_pretrained(
        ILLADA_8B_BASE,
        revision=ILLADA_8B_BASE_REVISION,
        trust_remote_code=True,
    )
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.max_position_embeddings = 64
    config.vocab_size = 128
    config.tie_word_embeddings = True

    backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = ILLaDACIDAdapter(
        backbone,
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16),
    )

    batch = CIDTensorBatch(
        thought_semantic=torch.randn(1, 4, config.hidden_size),
        role_features=torch.rand(1, 4, 6),
        uncertainty=torch.rand(1, 4, 1),
        local_noise=torch.rand(1, 4, 1),
        slot_occupancy=torch.tensor([[[1.0], [1.0], [0.0], [0.0]]]),
        display_ids=torch.tensor([[ILLADA_MASK_TOKEN_ID, 7, ILLADA_MASK_TOKEN_ID, 9]]),
        display_noise=torch.rand(1, 4, 1),
        fact_memory=torch.randn(1, 2, config.hidden_size),
        percept_memory=torch.randn(1, 1, config.hidden_size),
        source_memory=torch.randn(1, 2, config.hidden_size),
    )
    output = model(batch)
    loss = output.display_logits.float().mean() + output.thought_semantic.float().square().mean()
    loss.backward()

    print(f"backbone={type(backbone).__name__} dtype={output.display_logits.dtype}")
    print(f"thought={tuple(output.thought_semantic.shape)}")
    print(f"display={tuple(output.display_logits.shape)}")
    print(f"anchors={tuple(output.anchor_query.shape)}")


if __name__ == "__main__":
    main()

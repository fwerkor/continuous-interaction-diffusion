from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
cid_model = import_module("cid.model")
CIDTensorBatch = cid_model.CIDTensorBatch
ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
LLADA_MOE_7B_A1B_BASE = cid_model.LLADA_MOE_7B_A1B_BASE
LLADA_MOE_7B_A1B_BASE_REVISION = cid_model.LLADA_MOE_7B_A1B_BASE_REVISION


class TinyLLaDAMoEConfig:
    model_type = "llada"
    hidden_size = 32
    vocab_size = 64
    num_attention_heads = 4
    max_position_embeddings = 256
    num_experts = 4
    num_experts_per_tok = 2
    router_aux_loss_coef = 0.01
    mask_token_id = 63
    eos_token_id = 62


class TinyLLaDAMoELayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn = TinyLLaDAMoEAttention(hidden_size)


class TinyLLaDAMoEAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = 4
        self.num_key_value_heads = 4
        self.num_key_value_groups = 1
        self.head_dim = hidden_size // self.num_heads
        self.hidden_size = hidden_size
        self.attention_dropout = 0.0
        self.config = SimpleNamespace(clip_qkv=None)


class TinyLLaDAMoEDecoder(nn.Module):
    def __init__(self, config: TinyLLaDAMoEConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLLaDAMoELayer(config.hidden_size)])
        self.projection = self.layers[0].projection
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.config = config
        self.last_output_router_logits = False

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        return_dict,
        output_router_logits=False,
    ):
        del position_ids
        assert return_dict
        self.last_output_router_logits = bool(output_router_logits)
        hidden = inputs_embeds + self.projection(inputs_embeds)
        router_logits = None
        if output_router_logits:
            router_logits = (self.gate(hidden).reshape(-1, self.config.num_experts),)
        return SimpleNamespace(last_hidden_state=hidden, router_logits=router_logits)


class TinyLLaDAMoEBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyLLaDAMoEConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder = TinyLLaDAMoEDecoder(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.decoder

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


def _batch() -> CIDTensorBatch:
    config = TinyLLaDAMoEConfig()
    return CIDTensorBatch(
        thought_semantic=torch.randn(1, 2, config.hidden_size),
        role_features=torch.rand(1, 2, 6),
        uncertainty=torch.rand(1, 2, 1),
        local_noise=torch.rand(1, 2, 1),
        slot_occupancy=torch.tensor([[[1.0], [0.0]]]),
        prompt_ids=torch.randint(0, config.vocab_size - 2, (1, 3)),
        display_ids=torch.randint(0, config.vocab_size - 2, (1, 4)),
        display_noise=torch.rand(1, 4, 1),
        fact_memory=torch.randn(1, 1, config.hidden_size),
        percept_memory=torch.randn(1, 1, config.hidden_size),
        source_memory=torch.randn(1, 1, config.hidden_size),
    )


def test_llada_moe_adapter_uses_model_tokens_and_router_auxiliary_loss() -> None:
    backbone = TinyLLaDAMoEBackbone()
    adapter = ILLaDACIDAdapter(backbone)

    output = adapter(_batch())

    assert adapter.is_llada_moe
    assert adapter.mask_token_id == 63
    assert adapter.eos_token_id == 62
    assert backbone.decoder.last_output_router_logits
    assert output.auxiliary_loss is not None
    assert torch.isfinite(output.auxiliary_loss)
    output.auxiliary_loss.backward()
    assert backbone.decoder.gate.weight.grad is not None


def test_llada_moe_stage_a_skips_router_auxiliary_loss() -> None:
    backbone = TinyLLaDAMoEBackbone()
    adapter = ILLaDACIDAdapter(backbone, freeze_backbone=True)

    output = adapter(_batch())

    assert not backbone.decoder.last_output_router_logits
    assert output.auxiliary_loss is None
    adapter.set_mlp_chunk_size(16)


def test_llada_moe_attention_keeps_masked_slots_out_of_key_context() -> None:
    backbone = TinyLLaDAMoEBackbone()
    ILLaDACIDAdapter(backbone, freeze_backbone=True)
    attention = backbone.decoder.layers[0].self_attn
    with torch.no_grad():
        identity = torch.eye(TinyLLaDAMoEConfig.hidden_size)
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        ):
            projection.weight.copy_(identity)

    first = torch.randn(1, 3, TinyLLaDAMoEConfig.hidden_size)
    second = first.clone()
    second[:, 2] += 100.0
    attention._cid_key_padding_mask = torch.tensor([[True, True, False]])
    cos = torch.ones(1, 3, attention.head_dim)
    sin = torch.zeros_like(cos)

    first_output, _, _ = attention(first, position_embeddings=(cos, sin))
    second_output, _, _ = attention(second, position_embeddings=(cos, sin))

    assert torch.allclose(first_output[:, :2], second_output[:, :2], atol=1e-5)


def test_llada_moe_from_pretrained_pins_official_revision(monkeypatch) -> None:
    calls = []

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return TinyLLaDAMoEBackbone()

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = ILLaDACIDAdapter.from_pretrained(
        LLADA_MOE_7B_A1B_BASE,
        freeze_backbone=True,
        torch_dtype="auto",
    )

    assert adapter.is_llada_moe
    assert calls == [
        (
            LLADA_MOE_7B_A1B_BASE,
            {
                "torch_dtype": "auto",
                "trust_remote_code": True,
                "revision": LLADA_MOE_7B_A1B_BASE_REVISION,
                "attn_implementation": "sdpa",
            },
        )
    ]

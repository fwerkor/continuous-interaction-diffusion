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
        self.mlp = TinySparseMoE(hidden_size, num_experts=4, top_k=2)


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, 16, bias=False)
        self.up_proj = nn.Linear(hidden_size, 16, bias=False)
        self.down_proj = nn.Linear(16, hidden_size, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TinySparseMoE(nn.Module):
    def __init__(self, hidden_size: int, *, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.expert_bias = None
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([TinyExpert(hidden_size) for _ in range(num_experts)])

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        routing_weights = torch.softmax(self.gate(flat_hidden), dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(flat_hidden.dtype)
        output = torch.zeros_like(flat_hidden)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, route_idx = torch.where(selected_experts == expert_idx)
            expert_output = expert(flat_hidden[token_idx])
            output.index_add_(
                0,
                token_idx,
                expert_output * routing_weights[token_idx, route_idx, None],
            )
        return output.reshape(batch_size, sequence_length, hidden_dim)


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
    assert output.need_logits.shape == (1, 2, adapter.config.max_need_slots)
    assert output.source_logits.shape == (1, 2, adapter.config.max_need_slots, 1)
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


def test_llada_moe_grouped_experts_match_reference_outputs_and_input_gradients() -> None:
    backbone = TinyLLaDAMoEBackbone()
    adapter = ILLaDACIDAdapter(backbone, freeze_backbone=True)
    moe = backbone.decoder.layers[0].mlp
    reference_input = torch.randn(2, 5, TinyLLaDAMoEConfig.hidden_size, requires_grad=True)
    grouped_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = moe(reference_input)
    reference_output.square().sum().backward()
    packed_layers = adapter.pack_frozen_moe_experts()
    grouped_output = moe(grouped_input)
    grouped_output.square().sum().backward()

    assert packed_layers == 1
    assert len(moe.experts) == 0
    assert torch.allclose(reference_output, grouped_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reference_input.grad, grouped_input.grad, atol=1e-5, rtol=1e-5)


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

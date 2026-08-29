from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
cid_model = import_module("cid.model")
CIDTensorBatch = cid_model.CIDTensorBatch
ILLaDACIDConfig = cid_model.ILLaDACIDConfig
LFM2_DIFFUSION_350M = cid_model.LFM2_DIFFUSION_350M
LFM2_DIFFUSION_350M_REVISION = cid_model.LFM2_DIFFUSION_350M_REVISION
LFM2_MASK_TOKEN_ID = cid_model.LFM2_MASK_TOKEN_ID
LFMCIDAdapter = cid_model.LFMCIDAdapter
load_cid_adapter_from_pretrained = cid_model.load_cid_adapter_from_pretrained


class TinyLFMConfig:
    model_type = "lfm2"
    hidden_size = 32
    vocab_size = 64
    num_attention_heads = 4
    max_position_embeddings = 256
    mask_token_id = LFM2_MASK_TOKEN_ID
    eos_token_id = 7


class TinyLFMHidden(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_attention_mask = None
        self.last_position_ids = None
        self.last_inputs_embeds = None

    def forward(self, *, inputs_embeds, attention_mask, position_ids, return_dict):
        assert return_dict
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_position_ids = position_ids.detach().clone()
        self.last_inputs_embeds = inputs_embeds.detach().clone()
        weights = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        context = (inputs_embeds * weights).sum(dim=1, keepdim=True)
        context = context / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.projection(context))


class TinyLFMBackbone(nn.Module):
    def __init__(self, *, config_mask_token_id: int | None = LFM2_MASK_TOKEN_ID) -> None:
        super().__init__()
        self.config = TinyLFMConfig()
        self.config.mask_token_id = config_mask_token_id
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.lfm2 = TinyLFMHidden(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def make_batch() -> CIDTensorBatch:
    return CIDTensorBatch(
        thought_semantic=torch.randn(1, 2, TinyLFMConfig.hidden_size),
        role_features=torch.rand(1, 2, 6),
        uncertainty=torch.rand(1, 2, 1),
        local_noise=torch.rand(1, 2, 1),
        slot_occupancy=torch.tensor([[[1.0], [0.0]]]),
        prompt_ids=torch.randint(0, TinyLFMConfig.vocab_size, (1, 3)),
        display_ids=torch.randint(0, TinyLFMConfig.vocab_size, (1, 4)),
        display_noise=torch.rand(1, 4, 1),
        fact_memory=torch.empty(1, 0, TinyLFMConfig.hidden_size),
        percept_memory=torch.empty(1, 0, TinyLFMConfig.hidden_size),
        source_memory=torch.empty(1, 0, TinyLFMConfig.hidden_size),
    )


def test_lfm_adapter_uses_native_bidirectional_hidden_backbone() -> None:
    backbone = TinyLFMBackbone()
    adapter = LFMCIDAdapter(
        backbone,
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16),
    )

    output = adapter(make_batch())

    assert adapter.backbone_family == "lfm2"
    assert adapter.hidden_backbone() is backbone.lfm2
    assert adapter.mask_token_id == LFM2_MASK_TOKEN_ID
    assert output.thought_semantic.shape == (1, 2, TinyLFMConfig.hidden_size)
    assert output.display_logits.shape == (1, 4, TinyLFMConfig.vocab_size)
    assert output.need_logits.shape == (1, 2, adapter.config.max_need_slots)
    assert output.source_logits.shape == (1, 2, adapter.config.max_need_slots, 0)
    assert output.need_target_cell_logits.shape == (1, 2, adapter.config.max_need_slots, 2)
    assert output.need_target_display_logits.shape == (1, 2, adapter.config.max_need_slots, 4)
    assert backbone.lfm2.last_attention_mask.shape == (1, 9)
    assert backbone.lfm2.last_attention_mask[0].tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert backbone.lfm2.last_position_ids[0].tolist() == [0, 1, 8, 9, 10, 11, 12, 13, 14]


def test_lfm_retired_slots_are_neurally_equivalent_to_empty_slots() -> None:
    backbone = TinyLFMBackbone()
    adapter = LFMCIDAdapter(
        backbone,
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16),
    )
    empty = make_batch()
    adapter(empty)
    empty_seed = backbone.lfm2.last_inputs_embeds[:, 1].clone()

    retired = make_batch()
    retired.slot_occupancy[0, 1, 0] = 1.0
    retired.thought_semantic[0, 1] = 100.0
    retired.role_features[0, 1] = 1.0
    retired.uncertainty[0, 1, 0] = 0.25
    retired.local_noise[0, 1, 0] = 0.75
    retired.lifecycle_features = torch.zeros(1, 2, adapter.config.num_lifecycles)
    retired.lifecycle_features[0, 1, 3] = 1.0
    adapter(retired)
    retired_seed = backbone.lfm2.last_inputs_embeds[:, 1]

    assert torch.equal(empty_seed, retired_seed)
    assert not backbone.lfm2.last_attention_mask[0, 1]


def test_lfm_adapter_skips_illada_specific_chunk_patches() -> None:
    adapter = LFMCIDAdapter(TinyLFMBackbone())

    adapter.set_mlp_chunk_size(128)
    adapter.set_norm_chunk_size(128)

    assert adapter.hidden_backbone() is adapter.backbone.lfm2


def test_lfm_from_pretrained_uses_masked_lm_and_pinned_revision(monkeypatch) -> None:
    calls = []

    class AutoModelForMaskedLM:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return TinyLFMBackbone(config_mask_token_id=None)

    transformers = ModuleType("transformers")
    transformers.AutoModelForMaskedLM = AutoModelForMaskedLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = LFMCIDAdapter.from_pretrained(freeze_backbone=True, torch_dtype="auto")

    assert adapter.mask_token_id == LFM2_MASK_TOKEN_ID
    assert all(not parameter.requires_grad for parameter in adapter.backbone.parameters())
    assert calls == [
        (
            LFM2_DIFFUSION_350M,
            {
                "torch_dtype": "auto",
                "trust_remote_code": True,
                "revision": LFM2_DIFFUSION_350M_REVISION,
                "attn_implementation": "sdpa",
            },
        )
    ]


def test_generic_loader_dispatches_lfm_by_model_type(monkeypatch) -> None:
    config_calls = []
    model_calls = []

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            config_calls.append((model_name_or_path, kwargs))
            return SimpleNamespace(model_type="lfm2")

    class AutoModelForMaskedLM:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            model_calls.append((model_name_or_path, kwargs))
            return TinyLFMBackbone(config_mask_token_id=None)

    transformers = ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    transformers.AutoModelForMaskedLM = AutoModelForMaskedLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = load_cid_adapter_from_pretrained(
        LFM2_DIFFUSION_350M,
        freeze_backbone=True,
        torch_dtype="auto",
    )

    assert isinstance(adapter, LFMCIDAdapter)
    assert config_calls == [
        (
            LFM2_DIFFUSION_350M,
            {
                "trust_remote_code": True,
                "revision": LFM2_DIFFUSION_350M_REVISION,
            },
        )
    ]
    assert model_calls[0][0] == LFM2_DIFFUSION_350M

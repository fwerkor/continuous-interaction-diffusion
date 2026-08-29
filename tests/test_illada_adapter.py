from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
cid_model = import_module("cid.model")
ILLADA_8B_BASE = cid_model.ILLADA_8B_BASE
ILLADA_8B_BASE_REVISION = cid_model.ILLADA_8B_BASE_REVISION
ILLADA_MASK_TOKEN_ID = cid_model.ILLADA_MASK_TOKEN_ID
CIDTensorBatch = cid_model.CIDTensorBatch
ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
ILLaDACIDConfig = cid_model.ILLaDACIDConfig


class TinyILLaDAConfig:
    model_type = "illada"
    hidden_size = 32
    vocab_size = 64
    num_attention_heads = 4
    max_position_embeddings = 256


class TinyILLaDADecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_attention_mask = None
        self.last_position_ids = None

    def forward(self, *, inputs_embeds, attention_mask, position_ids, return_dict):
        assert return_dict
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_position_ids = position_ids.detach().clone()
        weights = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        context = (inputs_embeds * weights).sum(dim=1, keepdim=True)
        context = context / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = inputs_embeds + self.projection(context)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyILLaDABackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyILLaDAConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder = TinyILLaDADecoder(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.decoder

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


def make_batch(*, batch_size: int = 2, thought_slots: int = 4, display_length: int = 6):
    d_model = TinyILLaDAConfig.hidden_size
    occupancy = torch.tensor(
        [[[1.0], [1.0], [0.0], [0.0]], [[1.0], [0.0], [0.0], [0.0]]]
    )
    if batch_size != 2 or thought_slots != 4:
        occupancy = torch.zeros(batch_size, thought_slots, 1)
        occupancy[:, 0] = 1.0
    return CIDTensorBatch(
        thought_semantic=torch.randn(batch_size, thought_slots, d_model),
        role_features=torch.rand(batch_size, thought_slots, 6),
        uncertainty=torch.rand(batch_size, thought_slots, 1),
        local_noise=torch.rand(batch_size, thought_slots, 1),
        slot_occupancy=occupancy,
        prompt_ids=torch.randint(0, TinyILLaDAConfig.vocab_size, (batch_size, 2)),
        display_ids=torch.randint(0, TinyILLaDAConfig.vocab_size, (batch_size, display_length)),
        display_noise=torch.rand(batch_size, display_length, 1),
        fact_memory=torch.randn(batch_size, 2, d_model),
        percept_memory=torch.randn(batch_size, 3, d_model),
        source_memory=torch.randn(batch_size, 2, d_model),
    )


def test_illada_adapter_uses_shared_bidirectional_sequence_and_native_lm_head() -> None:
    backbone = TinyILLaDABackbone()
    adapter = ILLaDACIDAdapter(
        backbone,
        ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16),
    )
    batch = make_batch()

    output = adapter(batch)

    assert output.thought_semantic.shape == (2, 4, TinyILLaDAConfig.hidden_size)
    assert output.display_logits.shape == (2, 6, TinyILLaDAConfig.vocab_size)
    assert output.anchor_query.shape == (2, 4, 4, TinyILLaDAConfig.hidden_size)
    assert output.link_target_query.shape == (2, 4, 8, TinyILLaDAConfig.hidden_size)
    assert output.need_logits.shape == (2, 4, adapter.config.max_need_slots)
    assert output.source_logits.shape == (2, 4, adapter.config.max_need_slots, 2)
    assert output.need_target_cell_logits.shape == (2, 4, adapter.config.max_need_slots, 4)
    assert output.need_target_display_logits.shape == (2, 4, adapter.config.max_need_slots, 6)
    assert adapter.output_embeddings is backbone.lm_head
    assert backbone.decoder.last_attention_mask.shape == (2, 12)
    assert torch.equal(
        backbone.decoder.last_attention_mask[:, :4],
        batch.slot_occupancy.squeeze(-1).bool(),
    )
    assert backbone.decoder.last_attention_mask[:, 4:].all()
    assert backbone.decoder.last_position_ids[0].tolist() == [
        0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15
    ]
    assert backbone.decoder.last_position_ids[1].tolist() == [
        0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15
    ]

    loss = output.display_logits.float().mean() + output.thought_semantic.float().square().mean()
    loss.backward()
    assert backbone.embed_tokens.weight.grad is not None
    assert adapter.output_heads.thought_delta.weight.grad is not None


def test_illada_adapter_can_freeze_only_the_pretrained_backbone() -> None:
    backbone = TinyILLaDABackbone()
    adapter = ILLaDACIDAdapter(backbone, freeze_backbone=True)
    output = adapter(make_batch())

    output.display_logits.float().mean().backward()

    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert adapter.external_fusion.external_attention.in_proj_weight.requires_grad
    assert adapter.output_heads.role_head.weight.requires_grad


def test_illada_adapter_accepts_empty_external_memory() -> None:
    adapter = ILLaDACIDAdapter(TinyILLaDABackbone())
    batch = make_batch(batch_size=1, thought_slots=2, display_length=3)
    batch.fact_memory = torch.empty(1, 0, TinyILLaDAConfig.hidden_size)
    batch.percept_memory = torch.empty(1, 0, TinyILLaDAConfig.hidden_size)
    batch.source_memory = torch.empty(1, 0, TinyILLaDAConfig.hidden_size)

    output = adapter(batch)

    assert torch.isfinite(output.display_logits).all()
    assert output.source_logits.shape == (1, 2, adapter.config.max_need_slots, 0)


def test_illada_adapter_position_ids_ignore_batch_padding_width() -> None:
    backbone = TinyILLaDABackbone()
    adapter = ILLaDACIDAdapter(backbone)
    batch = make_batch(batch_size=2, thought_slots=4, display_length=5)
    batch.prompt_ids = torch.randint(0, TinyILLaDAConfig.vocab_size, (2, 4))
    batch.prompt_padding_mask = torch.tensor(
        [[False, False, True, True], [False, False, False, False]]
    )
    batch.display_padding_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, False]]
    )

    adapter(batch)

    first = backbone.decoder.last_position_ids[0]
    second = backbone.decoder.last_position_ids[1]
    assert first[:4].tolist() == [0, 1, 2, 3]
    assert first[4:8].tolist() == [128, 129, 0, 0]
    assert first[8:].tolist() == [130, 131, 132, 0, 0]
    assert second[:4].tolist() == [0, 1, 2, 3]
    assert second[4:8].tolist() == [128, 129, 130, 131]
    assert second[8:].tolist() == [132, 133, 134, 135, 136]


def test_illada_adapter_controls_backbone_gradient_checkpointing() -> None:
    backbone = TinyILLaDABackbone()
    adapter = ILLaDACIDAdapter(backbone)

    adapter.set_gradient_checkpointing(True)
    assert backbone.gradient_checkpointing
    assert backbone.gradient_checkpointing_kwargs is None
    adapter.set_gradient_checkpointing(True, use_reentrant=False)
    assert backbone.gradient_checkpointing_kwargs == {"use_reentrant": False}
    adapter.set_gradient_checkpointing(False)
    assert not backbone.gradient_checkpointing


def test_illada_adapter_enforces_backbone_context_capacity() -> None:
    adapter = ILLaDACIDAdapter(
        TinyILLaDABackbone(),
        ILLaDACIDConfig(max_thought_slots=48, max_display_tokens=48),
    )

    batch = make_batch(batch_size=1, thought_slots=32, display_length=48)
    batch.prompt_ids = torch.randint(0, TinyILLaDAConfig.vocab_size, (1, 170))
    with pytest.raises(ValueError, match="context capacity"):
        adapter(batch)


def test_from_pretrained_uses_official_model_id_and_remote_code(monkeypatch) -> None:
    calls = []

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return TinyILLaDABackbone()

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = ILLaDACIDAdapter.from_pretrained(freeze_backbone=True, torch_dtype="auto")

    assert isinstance(adapter.backbone, TinyILLaDABackbone)
    assert calls == [
        (
            ILLADA_8B_BASE,
            {
                "torch_dtype": "auto",
                "trust_remote_code": True,
                "revision": ILLADA_8B_BASE_REVISION,
            },
        )
    ]
    assert ILLADA_MASK_TOKEN_ID == 5


def test_targeted_percept_attention_uses_null_only_for_unrouted_queries() -> None:
    adapter = ILLaDACIDAdapter(TinyILLaDABackbone())
    hidden = torch.zeros(1, 3, TinyILLaDAConfig.hidden_size)
    routing = torch.tensor([[[True], [False], [True]]])

    mask = adapter.external_fusion._query_attention_mask(
        hidden=hidden,
        fact_count=0,
        percept_count=1,
        percept_query_mask=routing,
        fact_padding_mask=None,
        percept_padding_mask=None,
    )

    assert mask is not None
    first_head = mask[0]
    assert first_head.tolist() == [
        [False, True],
        [True, False],
        [False, True],
    ]


def test_illada_prompt_and_display_positions_do_not_depend_on_physical_thought_width() -> None:
    config = ILLaDACIDConfig(max_thought_slots=8, max_display_tokens=16)
    narrow_backbone = TinyILLaDABackbone()
    wide_backbone = TinyILLaDABackbone()
    narrow = ILLaDACIDAdapter(narrow_backbone, config)
    wide = ILLaDACIDAdapter(wide_backbone, config)

    narrow_batch = make_batch(batch_size=1, thought_slots=4, display_length=5)
    wide_batch = make_batch(batch_size=1, thought_slots=6, display_length=5)
    wide_batch.prompt_ids = narrow_batch.prompt_ids.clone()

    narrow(narrow_batch)
    wide(wide_batch)

    prompt_length = narrow_batch.prompt_ids.shape[1]
    narrow_positions = narrow_backbone.decoder.last_position_ids[0]
    wide_positions = wide_backbone.decoder.last_position_ids[0]
    assert narrow_positions[4 : 4 + prompt_length].tolist() == wide_positions[
        6 : 6 + prompt_length
    ].tolist()
    assert narrow_positions[4 + prompt_length :].tolist() == wide_positions[
        6 + prompt_length :
    ].tolist()

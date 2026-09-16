from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest


def _optional(name: str):
    try:
        return import_module(name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(
            f"optional model dependency {name!r} is unavailable: {exc}",
            allow_module_level=True,
        )


torch = _optional("torch")
transformers = _optional("transformers")

cid_ar = import_module("cid.model.ar")
cid_huggingface = import_module("cid.model.huggingface")
cid_illada = import_module("cid.model.illada")

CID_MASK_TOKEN = cid_ar.CID_MASK_TOKEN
bidirectional_ar_hidden_states = cid_ar.bidirectional_ar_hidden_states
prepare_ar_backbone_for_cid = cid_ar.prepare_ar_backbone_for_cid
CIDConfig = cid_huggingface.CIDConfig
CIDModel = cid_huggingface.CIDModel
ILLaDACIDAdapter = cid_illada.ILLaDACIDAdapter


class _TinyTokenizer:
    def __init__(self, size: int, *, eos_token_id: int = 2) -> None:
        self._vocab = {f"tok_{index}": index for index in range(size)}
        self.eos_token_id = eos_token_id
        self.mask_token = None
        self.mask_token_id = None

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, values):
        token = values["mask_token"]
        if token in self._vocab:
            self.mask_token = token
            self.mask_token_id = self._vocab[token]
            return 0
        token_id = len(self._vocab)
        self._vocab[token] = token_id
        self.mask_token = token
        self.mask_token_id = token_id
        return 1

    def __len__(self):
        return len(self._vocab)


def _llama_config():
    return transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


def _qwen_config():
    return transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    "model_class,config_factory",
    [
        (transformers.LlamaForCausalLM, _llama_config),
        (transformers.Qwen3ForCausalLM, _qwen_config),
    ],
)
def test_ar_backbone_is_prepared_with_dedicated_mask_token(model_class, config_factory) -> None:
    model = model_class(config_factory())
    tokenizer = _TinyTokenizer(32)

    mask_token_id = prepare_ar_backbone_for_cid(model, tokenizer)

    assert tokenizer.mask_token == CID_MASK_TOKEN
    assert mask_token_id == 32
    assert model.config.mask_token_id == 32
    assert model.config.vocab_size == 33
    assert model.get_input_embeddings().weight.shape[0] == 33
    assert model.get_output_embeddings().weight.shape[0] == 33

    adapter = ILLaDACIDAdapter(model, freeze_backbone=True)
    assert adapter.mask_token_id == 32
    assert adapter.backbone_family == model.config.model_type


def test_ar_backbone_normalizes_multiple_config_eos_ids() -> None:
    config = _llama_config()
    config.eos_token_id = [2, 7]
    model = transformers.LlamaForCausalLM(config)
    tokenizer = _TinyTokenizer(32, eos_token_id=2)

    prepare_ar_backbone_for_cid(model, tokenizer)

    assert model.config.eos_token_id == 2
    adapter = ILLaDACIDAdapter(model, freeze_backbone=True)
    assert adapter.eos_token_id == 2


def test_new_mask_token_uses_mean_embedding_inside_reserved_vocab_capacity() -> None:
    model = transformers.LlamaForCausalLM(_llama_config())
    tokenizer = _TinyTokenizer(31)
    expected_mean = model.get_input_embeddings().weight.detach().float().mean(dim=0)

    mask_token_id = prepare_ar_backbone_for_cid(model, tokenizer)

    assert mask_token_id == 31
    assert model.config.vocab_size == 32
    assert torch.allclose(
        model.get_input_embeddings().weight[mask_token_id].float(),
        expected_mean,
    )


@pytest.mark.parametrize(
    "model_class,config_factory",
    [
        (transformers.LlamaForCausalLM, _llama_config),
        (transformers.Qwen3ForCausalLM, _qwen_config),
    ],
)
def test_ar_cid_hidden_path_is_bidirectional(model_class, config_factory) -> None:
    torch.manual_seed(7)
    model = model_class(config_factory()).eval()
    decoder = model.get_decoder()
    first = torch.tensor([[1, 3, 5]])
    second = torch.tensor([[1, 4, 5]])
    positions = torch.arange(3).unsqueeze(0)
    attention_mask = torch.ones_like(first)

    with torch.no_grad():
        first_hidden = bidirectional_ar_hidden_states(
            decoder,
            inputs_embeds=model.get_input_embeddings()(first),
            attention_mask=attention_mask,
            position_ids=positions,
        )
        second_hidden = bidirectional_ar_hidden_states(
            decoder,
            inputs_embeds=model.get_input_embeddings()(second),
            attention_mask=attention_mask,
            position_ids=positions,
        )

    assert not torch.allclose(first_hidden[:, 0], second_hidden[:, 0])


def _unified_config(backbone_config) -> CIDConfig:
    values = backbone_config.to_dict()
    values["mask_token_id"] = 31
    values["use_cache"] = False
    return CIDConfig(
        backbone_config=values,
        adapter_config={
            "max_thought_slots": 4,
            "max_display_tokens": 8,
            "display_canvas_tokens": 4,
            "num_roles": 6,
            "num_lifecycles": 4,
            "num_anchor_kinds": 7,
            "num_link_relations": 8,
            "num_object_kinds": 7,
            "num_refresh_actions": 3,
            "max_need_slots": 2,
            "max_argument_slots": 2,
            "max_anchor_slots": 2,
            "max_link_slots": 2,
            "external_dropout": 0.0,
        },
        semantic_embedding={
            "format_version": 1,
            "encoding_version": 2,
            "pooling_mode": "order-aware-v2",
            "d_model": 16,
            "vocab_size": 32,
            "dtype": "torch.float32",
            "tensor_key": "semantic_embedding_weight",
        },
        neural_contract_version=4,
    )


@pytest.mark.parametrize("config_factory", [_llama_config, _qwen_config])
def test_unified_ar_cid_model_round_trip(config_factory, tmp_path: Path) -> None:
    model = CIDModel(_unified_config(config_factory()))
    assert model.cid_adapter.backbone_family in {"llama", "qwen3"}

    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="5GB")
    restored = CIDModel.from_pretrained(tmp_path, low_cpu_mem_usage=True)

    assert restored.cid_adapter.backbone_family == model.cid_adapter.backbone_family
    assert restored.cid_adapter.mask_token_id == 31

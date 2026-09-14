from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest


def _optional_model_dependency(name: str):
    try:
        return import_module(name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(
            f"optional model dependency {name!r} is unavailable: {exc}",
            allow_module_level=True,
        )


torch = _optional_model_dependency("torch")
transformers = _optional_model_dependency("transformers")
safetensors_torch = _optional_model_dependency("safetensors.torch")

load_file = safetensors_torch.load_file
Lfm2Config = import_module("transformers.models.lfm2.configuration_lfm2").Lfm2Config
cid_huggingface = import_module("cid.model.huggingface")
CIDConfig = cid_huggingface.CIDConfig
CIDModel = cid_huggingface.CIDModel
unified_state_from_legacy = cid_huggingface.unified_state_from_legacy


def _tiny_cid_config() -> CIDConfig:
    backbone = Lfm2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        block_dim=32,
        block_multiple_of=8,
        conv_dim=32,
        conv_dim_out=32,
        num_heads=4,
        use_cache=False,
    ).to_dict()
    backbone["mask_token_id"] = 16
    return CIDConfig(
        backbone_config=backbone,
        adapter_config={
            "max_thought_slots": 8,
            "max_display_tokens": 16,
            "display_canvas_tokens": 8,
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
            "d_model": 32,
            "vocab_size": 64,
            "dtype": "torch.bfloat16",
            "tensor_key": "semantic_embedding_weight",
        },
        neural_contract_version=4,
    )


def test_cid_model_round_trips_as_one_safetensors_file(tmp_path: Path) -> None:
    model = CIDModel(_tiny_cid_config())
    expected_semantic = torch.arange(64 * 32, dtype=torch.bfloat16).reshape(64, 32)
    model.semantic_embedding_weight.copy_(expected_semantic)

    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="5GB")

    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("model-*-of-*.safetensors"))
    saved = load_file(tmp_path / "model.safetensors", device="cpu")
    assert "semantic_embedding_weight" in saved
    assert any(name.startswith("adapter.backbone.") for name in saved)
    assert any(name.startswith("adapter.output_heads.") for name in saved)

    restored = CIDModel.from_pretrained(tmp_path, low_cpu_mem_usage=True)
    assert restored.semantic_embedding_weight.dtype is torch.bfloat16
    assert torch.equal(restored.semantic_embedding_weight, expected_semantic)


def test_unified_state_prefixes_legacy_components() -> None:
    backbone = {"lfm2.embed_tokens.weight": torch.ones(2, 3)}
    cid = {"channel_embedding.weight": torch.full((3, 3), 2.0)}
    semantic = {
        "format_version": 1,
        "encoding_version": 2,
        "pooling_mode": "order-aware-v2",
        "d_model": 3,
        "weight": torch.full((2, 3), 3.0),
    }

    state = unified_state_from_legacy(backbone, cid, semantic)

    assert set(state) == {
        "adapter.backbone.lfm2.embed_tokens.weight",
        "adapter.channel_embedding.weight",
        "semantic_embedding_weight",
    }

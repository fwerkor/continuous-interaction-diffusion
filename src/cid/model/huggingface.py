from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from cid.model.ar import AR_CID_MODEL_TYPES
from cid.model.encoding import ILLaDATextEncoder
from cid.model.illada import ILLaDACIDAdapter, ILLaDACIDConfig
from cid.model.lfm import LFM2_EOS_TOKEN_ID, LFM2_MASK_TOKEN_ID, LFMCIDAdapter
from cid.model.lfm_bidirectional import Lfm2BidirectionalForMaskedLM
from cid.model.tensors import CIDTensorBatch, CIDTensorOutput


class CIDConfig(PretrainedConfig):
    """Hugging Face configuration for a complete CID inference checkpoint."""

    model_type = "cid"
    is_composition = True

    def __init__(
        self,
        *,
        backbone_config: Mapping[str, Any] | None = None,
        adapter_config: Mapping[str, Any] | None = None,
        semantic_embedding: Mapping[str, Any] | None = None,
        neural_contract_version: int = 4,
        base_model: str | None = None,
        release_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_config = dict(backbone_config or {})
        self.adapter_config = dict(adapter_config or {})
        self.semantic_embedding = dict(semantic_embedding or {})
        self.neural_contract_version = int(neural_contract_version)
        self.base_model = base_model
        self.release_name = release_name

        supported_backbones = {"lfm2", *AR_CID_MODEL_TYPES}
        if (
            self.backbone_config
            and self.backbone_config.get("model_type") not in supported_backbones
        ):
            raise ValueError(
                "unified CID checkpoints support LFM2, Llama, and Qwen3 backbones"
            )
        if self.semantic_embedding:
            required = {
                "format_version",
                "encoding_version",
                "pooling_mode",
                "d_model",
                "vocab_size",
                "dtype",
            }
            missing = required.difference(self.semantic_embedding)
            if missing:
                raise ValueError(
                    "CID semantic embedding metadata is incomplete: " + ", ".join(sorted(missing))
                )


class CIDModel(PreTrainedModel):
    """Complete CID model: diffusion backbone, CID modules, and semantic snapshot."""

    config_class = CIDConfig
    base_model_prefix = "adapter"
    main_input_name = "batch"
    def __init__(self, config: CIDConfig) -> None:
        super().__init__(config)
        if not config.backbone_config:
            raise ValueError("CIDConfig.backbone_config is required")
        if not config.adapter_config:
            raise ValueError("CIDConfig.adapter_config is required")
        if not config.semantic_embedding:
            raise ValueError("CIDConfig.semantic_embedding is required")

        backbone_values = dict(config.backbone_config)
        model_type = str(backbone_values.get("model_type", ""))
        backbone_values["use_cache"] = False
        backbone_values.pop("auto_map", None)
        backbone_values.pop("architectures", None)
        if model_type == "lfm2":
            backbone_values.setdefault("mask_token_id", LFM2_MASK_TOKEN_ID)
            backbone_values.setdefault("eos_token_id", LFM2_EOS_TOKEN_ID)
            backbone_config = Lfm2Config.from_dict(backbone_values)
            backbone = Lfm2BidirectionalForMaskedLM(backbone_config)
            adapter_class = LFMCIDAdapter
        elif model_type in AR_CID_MODEL_TYPES:
            if backbone_values.get("mask_token_id") is None:
                raise ValueError("unified autoregressive CID backbone must define mask_token_id")
            config_values = dict(backbone_values)
            config_values.pop("model_type", None)
            backbone_config = AutoConfig.for_model(model_type, **config_values)
            backbone = AutoModelForCausalLM.from_config(backbone_config)
            adapter_class = ILLaDACIDAdapter
        else:
            raise ValueError(f"unsupported unified CID backbone: {model_type!r}")

        adapter_config = ILLaDACIDConfig(**config.adapter_config)
        self.adapter = adapter_class(backbone, config=adapter_config, freeze_backbone=False)

        semantic = config.semantic_embedding
        expected_shape = (self.adapter.vocab_size, self.adapter.d_model)
        declared_shape = (int(semantic["vocab_size"]), int(semantic["d_model"]))
        if declared_shape != expected_shape:
            raise ValueError("semantic embedding geometry does not match the CID backbone")
        self.register_buffer(
            "semantic_embedding_weight",
            torch.empty(expected_shape, dtype=_torch_dtype(str(semantic["dtype"]))),
            persistent=True,
        )

    @property
    def cid_adapter(self) -> ILLaDACIDAdapter:
        return self.adapter

    def get_input_embeddings(self):
        return self.adapter.input_embeddings

    def set_input_embeddings(self, value):
        self.adapter.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.adapter.output_embeddings

    def set_output_embeddings(self, value):
        self.adapter.backbone.set_output_embeddings(value)

    def forward(self, batch: CIDTensorBatch, **_: Any) -> CIDTensorOutput:
        return self.adapter(batch)

    def semantic_snapshot_state(self) -> dict[str, Any]:
        semantic = self.config.semantic_embedding
        return {
            "format_version": int(semantic["format_version"]),
            "encoding_version": int(semantic["encoding_version"]),
            "pooling_mode": str(semantic["pooling_mode"]),
            "d_model": int(semantic["d_model"]),
            "weight": self.semantic_embedding_weight,
        }

    def make_text_encoder(
        self,
        tokenizer: Any,
        *,
        device: torch.device | str,
        embedding_device: torch.device | str | None = None,
    ) -> ILLaDATextEncoder:
        return ILLaDATextEncoder.from_frozen_snapshot_state(
            self.adapter,
            tokenizer,
            self.semantic_snapshot_state(),
            device=device,
            embedding_device=embedding_device,
        )


def load_cid_model_from_pretrained(
    model_name_or_path: str | Path,
    **from_pretrained_kwargs: Any,
) -> CIDModel:
    """Load a unified CID checkpoint with the source-package implementation."""

    from_pretrained_kwargs.setdefault("low_cpu_mem_usage", True)
    return CIDModel.from_pretrained(str(model_name_or_path), **from_pretrained_kwargs)


def build_unified_cid_config(
    *,
    backbone_config: Mapping[str, Any],
    adapter_config: ILLaDACIDConfig | Mapping[str, Any],
    semantic_state: Mapping[str, Any],
    neural_contract_version: int,
    base_model: str | None,
    release_name: str | None,
) -> CIDConfig:
    """Build the portable HF config used by the legacy-to-unified exporter."""

    weight = semantic_state.get("weight")
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError("frozen semantic embedding snapshot weight is invalid")
    adapter_values = (
        asdict(adapter_config)
        if isinstance(adapter_config, ILLaDACIDConfig)
        else dict(adapter_config)
    )
    backbone_values = dict(backbone_config)
    if backbone_values.get("model_type") == "lfm2":
        backbone_values.setdefault("mask_token_id", LFM2_MASK_TOKEN_ID)
        backbone_values.setdefault("eos_token_id", LFM2_EOS_TOKEN_ID)
    elif backbone_values.get("model_type") in AR_CID_MODEL_TYPES:
        if backbone_values.get("mask_token_id") is None:
            raise ValueError("autoregressive CID backbone config must define mask_token_id")
    else:
        raise ValueError(
            f"unsupported unified CID backbone: {backbone_values.get('model_type')!r}"
        )
    backbone_values["use_cache"] = False
    backbone_values.pop("auto_map", None)
    backbone_values.pop("architectures", None)
    return CIDConfig(
        backbone_config=backbone_values,
        adapter_config=adapter_values,
        semantic_embedding={
            "format_version": int(semantic_state["format_version"]),
            "encoding_version": int(semantic_state["encoding_version"]),
            "pooling_mode": str(semantic_state["pooling_mode"]),
            "d_model": int(semantic_state["d_model"]),
            "vocab_size": int(weight.shape[0]),
            "dtype": str(weight.dtype),
            "tensor_key": "semantic_embedding_weight",
        },
        neural_contract_version=neural_contract_version,
        base_model=base_model,
        release_name=release_name,
        architectures=["CIDModel"],
        auto_map={
            "AutoConfig": "configuration_cid.CIDConfig",
            "AutoModel": "modeling_cid.CIDModel",
        },
    )


def unified_state_from_legacy(
    backbone_state: Mapping[str, Tensor],
    cid_state: Mapping[str, Tensor],
    semantic_state: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Map the legacy backbone + sidecar files into one CIDModel state dict."""

    result: dict[str, Tensor] = {}
    for name, tensor in backbone_state.items():
        key = f"adapter.backbone.{name}"
        if key in result:
            raise ValueError(f"duplicate unified tensor key: {key}")
        result[key] = tensor
    for name, tensor in cid_state.items():
        if name.startswith("backbone."):
            raise ValueError("CID sidecar unexpectedly contains backbone parameters")
        key = f"adapter.{name}"
        if key in result:
            raise ValueError(f"duplicate unified tensor key: {key}")
        result[key] = tensor
    weight = semantic_state.get("weight")
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError("frozen semantic embedding snapshot weight is invalid")
    result["semantic_embedding_weight"] = weight
    return result


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported CID semantic embedding dtype: {value}")
    return dtype

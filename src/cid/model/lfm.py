from __future__ import annotations

from cid.model.illada import ILLaDACIDAdapter, ILLaDACIDConfig

LFM2_DIFFUSION_350M = "LiquidAI/LFM2.5-Encoder-350M-Diffusion"
LFM2_DIFFUSION_350M_REVISION = "5bb1c16fd5980ac2b4f7afcde762856aebd8ca0a"
LFM2_MASK_TOKEN_ID = 16
LFM2_EOS_TOKEN_ID = 7


class LFMCIDAdapter(ILLaDACIDAdapter):
    """CID bridge for the bidirectional LFM2.5 masked-diffusion backbone."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = LFM2_DIFFUSION_350M,
        *,
        config: ILLaDACIDConfig | None = None,
        freeze_backbone: bool = False,
        **from_pretrained_kwargs: object,
    ) -> LFMCIDAdapter:
        from transformers import AutoModelForMaskedLM

        from_pretrained_kwargs.setdefault("trust_remote_code", True)
        if model_name_or_path == LFM2_DIFFUSION_350M:
            from_pretrained_kwargs.setdefault("revision", LFM2_DIFFUSION_350M_REVISION)
        from_pretrained_kwargs.setdefault("attn_implementation", "sdpa")
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
        if getattr(backbone.config, "mask_token_id", None) is None:
            backbone.config.mask_token_id = LFM2_MASK_TOKEN_ID
        if getattr(backbone.config, "eos_token_id", None) is None:
            backbone.config.eos_token_id = LFM2_EOS_TOKEN_ID
        return cls(backbone, config=config, freeze_backbone=freeze_backbone)

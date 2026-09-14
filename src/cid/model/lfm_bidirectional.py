"""LFM2 backbone with bidirectional attention + non-causal short-conv.

This module is the source-package counterpart of the public checkpoint remote code.

Wired into the HF repo via `auto_map` in config.json so that

AutoModel.from_pretrained(repo, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)

both return a model with the encoder-style patches already applied.

Supports `attn_implementation` in {"eager", "sdpa", "flash_attention_2"}:

eager/sdpa consume a 4D additive pad-only mask and reproduce the exact
training-time behavior; flash_attention_2 receives the 2D padding mask (or
None) and runs the kernel non-causally via `Lfm2Attention.is_causal = False`,
yielding outputs equivalent to the unpadded forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.lfm2 import modeling_lfm2 as _lfm2_mod
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2ShortConv,
    apply_mask_to_padding_states,
)


def _bidirectional_mask(
    config,
    input_embeds: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    cache_position: torch.LongTensor | None = None,
    past_key_values=None,
    position_ids: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor | None:
    # transformers has renamed the embeds kwarg across versions
    # (input_embeds <-> inputs_embeds); accept either to stay forward-compatible.
    if input_embeds is None:
        input_embeds = kwargs.get("inputs_embeds")

    if config._attn_implementation == "flash_attention_2":
        # FA2 only uses the 2D padding mask to unpad sequences; causality is
        # controlled by `Lfm2Attention.is_causal` (set to False below).
        if attention_mask is not None and not attention_mask.all():
            return attention_mask
        return None

    device = input_embeds.device
    dtype = input_embeds.dtype
    bsz, q_len = input_embeds.shape[:2]
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    kv_len = past + q_len

    mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
    if attention_mask is not None:
        cur_len = attention_mask.size(-1)
        key_pad_flags = (attention_mask == 0).to(device=device, dtype=torch.float32)
        pad_vec = torch.zeros((bsz, kv_len), device=device, dtype=torch.float32)
        if cur_len > 0:
            pad_vec[:, past : past + cur_len] = key_pad_flags * -1e9
        mask = mask + pad_vec.to(dtype)[:, None, None, :]
    return mask


def _noncausal_shortconv_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    cache_position=None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    x = apply_mask_to_padding_states(hidden_states, attention_mask)

    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)
    Bx = B * x

    k = self.conv.weight.shape[-1]
    pad = k // 2
    conv_out = F.conv1d(
        Bx,
        weight=self.conv.weight,
        bias=self.conv.bias,
        stride=1,
        padding=pad,
        dilation=1,
        groups=Bx.shape[1],
    )
    if conv_out.shape[-1] > Bx.shape[-1]:
        conv_out = conv_out[..., : Bx.shape[-1]]
    elif conv_out.shape[-1] < Bx.shape[-1]:
        conv_out = F.pad(conv_out, (0, Bx.shape[-1] - conv_out.shape[-1]))

    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()
    return self.out_proj(y)


def _shortconv_forward(self, *args, **kwargs):
    return self.slow_forward(*args, **kwargs)


_PATCHED = False


def _install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _lfm2_mod.create_causal_mask = _bidirectional_mask
    Lfm2ShortConv.slow_forward = _noncausal_shortconv_forward
    Lfm2ShortConv.forward = _shortconv_forward
    _PATCHED = True


_install_patches()


def _set_attention_noncausal(model) -> None:
    for module in model.modules():
        if isinstance(module, Lfm2Attention):
            module.is_causal = False


class Lfm2BidirectionalModel(Lfm2Model):
    """LFM2 patched for encoder-style use:
    full bidirectional attention + non-causal short-conv."""

    def __init__(self, config):
        _install_patches()
        super().__init__(config)
        _set_attention_noncausal(self)


class Lfm2BidirectionalForMaskedLM(Lfm2PreTrainedModel):
    """LFM2 bidirectional encoder with a tied masked-LM head."""

    config_class = Lfm2Config
    base_model_prefix = "lfm2"
    _tied_weights_keys = {"lm_head.weight": "lfm2.embed_tokens.weight"}

    def __init__(self, config: Lfm2Config):
        _install_patches()
        config = type(config).from_dict({**config.to_dict(), "use_cache": False})
        super().__init__(config)
        self.lfm2 = Lfm2BidirectionalModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.lm_head.weight = self.lfm2.embed_tokens.weight

    def get_input_embeddings(self):
        return self.lfm2.embed_tokens

    def set_input_embeddings(self, value):
        self.lfm2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> MaskedLMOutput:
        return_dict = True if return_dict is None else return_dict
        outputs = self.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

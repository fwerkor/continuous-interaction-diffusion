from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MINICPM5_2B_BASE = "openbmb/MiniCPM5-2B-Base"
QWEN3_4B_BASE = "Qwen/Qwen3-4B-Base"
CID_MASK_TOKEN = "<|cid_mask|>"
AR_CID_MODEL_TYPES = frozenset({"llama", "qwen3"})


def prepare_ar_tokenizer(tokenizer: Any) -> int:
    """Install CID's dedicated mask token into an autoregressive tokenizer."""

    vocabulary = tokenizer.get_vocab()
    if CID_MASK_TOKEN in vocabulary:
        mask_token_id = int(vocabulary[CID_MASK_TOKEN])
        tokenizer.mask_token = CID_MASK_TOKEN
    else:
        added = int(tokenizer.add_special_tokens({"mask_token": CID_MASK_TOKEN}))
        if added != 1:
            raise RuntimeError("failed to add the CID mask token to the autoregressive tokenizer")
        mask_token_id = int(tokenizer.mask_token_id)
    if getattr(tokenizer, "eos_token_id", None) == mask_token_id:
        raise ValueError("CID mask token must differ from the tokenizer EOS token")
    return mask_token_id


def prepare_ar_backbone_for_cid(backbone: nn.Module, tokenizer: Any) -> int:
    """Make an AR checkpoint compatible with CID masked full-sequence training.

    Existing weights are preserved. If the tokenizer needs one extra vocabulary row,
    the new row is initialized to the mean pretrained embedding rather than random
    noise so Stage A can keep the backbone frozen safely.
    """

    model_type = str(backbone.config.model_type)
    if model_type not in AR_CID_MODEL_TYPES:
        raise ValueError(f"unsupported autoregressive CID backbone: {model_type!r}")

    mask_was_present = CID_MASK_TOKEN in tokenizer.get_vocab()
    mask_token_id = prepare_ar_tokenizer(tokenizer)
    old_vocab_size = int(backbone.config.vocab_size)
    input_embeddings = backbone.get_input_embeddings()
    output_embeddings = backbone.get_output_embeddings()
    with torch.no_grad():
        input_mean = input_embeddings.weight[:old_vocab_size].float().mean(dim=0)
        output_mean = (
            None
            if output_embeddings is None
            else output_embeddings.weight[:old_vocab_size].float().mean(dim=0)
        )

    required_vocab_size = max(int(len(tokenizer)), mask_token_id + 1)
    if required_vocab_size > old_vocab_size:
        backbone.resize_token_embeddings(required_vocab_size, mean_resizing=False)

    if not mask_was_present:
        with torch.no_grad():
            input_embeddings = backbone.get_input_embeddings()
            input_embeddings.weight[mask_token_id].copy_(
                input_mean.to(
                    device=input_embeddings.weight.device,
                    dtype=input_embeddings.weight.dtype,
                )
            )
            output_embeddings = backbone.get_output_embeddings()
            if (
                output_embeddings is not None
                and output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr()
            ):
                if output_mean is None:
                    raise RuntimeError("autoregressive output embedding mean was not captured")
                output_embeddings.weight[mask_token_id].copy_(
                    output_mean.to(
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype,
                    )
                )

    backbone.config.mask_token_id = mask_token_id
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos_token_id is None:
        raise ValueError("autoregressive CID tokenizer must define eos_token_id")
    # Some AR checkpoints (for example MiniCPM5) expose several generation stop
    # IDs in the model config while their tokenizer still has one canonical EOS.
    # CID's display diffusion uses a single structural EOS token, so keep the
    # prepared backbone aligned with the tokenizer instead of carrying the
    # generation-only list into the CID adapter.
    backbone.config.eos_token_id = int(tokenizer_eos_token_id)
    backbone.config.use_cache = False
    _set_attention_noncausal(backbone.get_decoder())
    return mask_token_id


def bidirectional_ar_hidden_states(
    decoder: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run a Llama/Qwen3 decoder as a full-sequence bidirectional denoiser.

    The decoder layers, RoPE, norms, MLPs, and all pretrained parameters are reused
    unchanged. Only the causal attention constraint and KV-cache path are removed.
    """

    _set_attention_noncausal(decoder)
    key_mask = _bidirectional_key_mask(attention_mask, inputs_embeds)
    hidden_states = inputs_embeds
    position_embeddings = decoder.rotary_emb(hidden_states, position_ids)

    use_gradient_checkpointing = bool(
        decoder.training and getattr(decoder, "gradient_checkpointing", False)
    )
    for decoder_layer in decoder.layers[: decoder.config.num_hidden_layers]:

        def layer_forward(
            states: torch.Tensor, *, layer: nn.Module = decoder_layer
        ) -> torch.Tensor:
            return layer(
                states,
                attention_mask=key_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        if use_gradient_checkpointing:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)
    return decoder.norm(hidden_states)


def _set_attention_noncausal(decoder: nn.Module) -> None:
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise RuntimeError("autoregressive CID backbone does not expose decoder layers")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError("autoregressive CID decoder layer does not expose self attention")
        attention.is_causal = False


def _bidirectional_key_mask(
    attention_mask: torch.Tensor | None,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("autoregressive CID attention mask must have shape [batch, tokens]")
    if bool(attention_mask.bool().all()):
        return None
    minimum = torch.finfo(inputs_embeds.dtype).min
    valid_keys = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    additive = torch.zeros(
        (valid_keys.shape[0], 1, 1, valid_keys.shape[1]),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    return additive.masked_fill(~valid_keys[:, None, None, :], minimum)

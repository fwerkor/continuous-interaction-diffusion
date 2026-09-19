from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class CIDDiffusionForMaskedLM(LlamaForCausalLM):
    """Bidirectional masked-diffusion language model backed by Llama weights.

    The parameter layout is intentionally identical to LlamaForCausalLM so
    AR-to-diffusion checkpoints can be published without rewriting weights.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        mask_token_id = getattr(config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("CID diffusion checkpoints require config.mask_token_id")
        self.mask_token_id = int(mask_token_id)
        self.config.use_cache = False
        self._set_bidirectional_attention()

    def _set_bidirectional_attention(self) -> None:
        for layer in self.model.layers:
            layer.self_attn.is_causal = False

    def _bidirectional_hidden_states(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
        elif attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, tokens]")

        self._set_bidirectional_attention()
        key_mask = None
        if not bool(attention_mask.bool().all()):
            minimum = torch.finfo(inputs_embeds.dtype).min
            valid_keys = attention_mask.to(
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
            key_mask = torch.zeros(
                (batch_size, 1, 1, sequence_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            key_mask = key_mask.masked_fill(
                ~valid_keys[:, None, None, :],
                minimum,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        for layer in self.model.layers[: self.config.num_hidden_layers]:
            layer_output = layer(
                hidden_states,
                attention_mask=key_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = (
                layer_output[0] if isinstance(layer_output, tuple) else layer_output
            )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states[-1] = hidden_states
            return hidden_states, tuple(all_hidden_states)
        return hidden_states, None

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast | tuple[Tensor, ...]:
        if output_attentions:
            raise ValueError(
                "CIDDiffusionForMaskedLM does not expose per-layer attention weights"
            )
        output_hidden_states = bool(output_hidden_states)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        hidden_states, all_hidden_states = self._bidirectional_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, None, all_hidden_states, None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    @staticmethod
    def _filter_top_p(logits: Tensor, top_p: float) -> Tensor:
        if top_p >= 1.0:
            return logits
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1,
        )
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        removal_mask = torch.zeros_like(remove).scatter(
            -1,
            sorted_indices,
            remove,
        )
        return logits.masked_fill(removal_mask, -torch.inf)

    def _predict_tokens(
        self,
        logits: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        logits = logits.float()
        if temperature <= 0.0:
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            return predicted, confidence

        filtered = self._filter_top_p(logits / temperature, top_p)
        probabilities = torch.softmax(filtered, dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        predicted = torch.multinomial(
            flat,
            num_samples=1,
            generator=generator,
        ).reshape(probabilities.shape[:-1])
        confidence = probabilities.gather(
            -1,
            predicted.unsqueeze(-1),
        ).squeeze(-1)
        return predicted, confidence

    @torch.inference_mode()
    def denoise(
        self,
        input_ids: Tensor,
        *,
        editable_mask: Tensor | None = None,
        steps: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Iteratively resolve MASK tokens already present in input_ids."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        result = input_ids.clone()
        unresolved = result.eq(self.mask_token_id)
        if editable_mask is not None:
            if editable_mask.shape != result.shape:
                raise ValueError("editable_mask must match input_ids")
            unresolved &= editable_mask.bool()
        else:
            editable_mask = unresolved.clone()

        initial = unresolved.sum(dim=1)
        max_masks = int(initial.max().item()) if initial.numel() else 0
        if max_masks == 0:
            return result
        if steps is None:
            steps = max_masks
        if steps <= 0:
            raise ValueError("steps must be positive")

        for step_index in range(steps):
            unresolved = result.eq(self.mask_token_id) & editable_mask
            if not bool(unresolved.any()):
                break

            logits = self(result).logits
            predicted, confidence = self._predict_tokens(
                logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            remaining_steps = steps - step_index
            for batch_index in range(result.shape[0]):
                positions = torch.nonzero(
                    unresolved[batch_index],
                    as_tuple=False,
                ).flatten()
                if positions.numel() == 0:
                    continue
                reveal_count = math.ceil(positions.numel() / remaining_steps)
                ranked = positions[
                    confidence[batch_index, positions].argsort(descending=True)
                ]
                selected = ranked[:reveal_count]
                result[batch_index, selected] = predicted[batch_index, selected]

        unresolved = result.eq(self.mask_token_id) & editable_mask
        if bool(unresolved.any()):
            logits = self(result).logits
            result[unresolved] = logits.argmax(dim=-1)[unresolved]
        return result

    @torch.inference_mode()
    def diffusion_generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 64,
        steps: int | None = None,
        block_length: int | None = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Append a masked canvas and fill it with block-wise diffusion."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if steps is None:
            steps = max_new_tokens
        if steps <= 0:
            raise ValueError("steps must be positive")
        if block_length is None:
            block_length = max_new_tokens
        if block_length <= 0:
            raise ValueError("block_length must be positive")

        batch_size, prompt_length = input_ids.shape
        result = torch.full(
            (batch_size, prompt_length + max_new_tokens),
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        result[:, :prompt_length] = input_ids

        block_count = math.ceil(max_new_tokens / block_length)
        base_steps = steps // block_count
        extra_steps = steps % block_count

        for block_index in range(block_count):
            start = prompt_length + block_index * block_length
            stop = min(start + block_length, result.shape[1])
            block_steps = base_steps + (1 if block_index < extra_steps else 0)
            block_steps = max(1, block_steps)
            editable = torch.zeros_like(result, dtype=torch.bool)
            editable[:, start:stop] = True
            result = self.denoise(
                result,
                editable_mask=editable,
                steps=block_steps,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
        return result

    def generate(self, input_ids: Tensor | None = None, **kwargs: Any) -> Tensor:
        """Diffusion-aware replacement for autoregressive GenerationMixin.generate."""

        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if input_ids is None:
            raise ValueError("input_ids are required for diffusion generation")

        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError(
                "batched padded prompts are not supported by diffusion generate; "
                "generate each prompt separately"
            )

        generation_config = kwargs.pop("generation_config", None)
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None:
            max_length = kwargs.pop("max_length", None)
            if max_length is not None:
                max_new_tokens = int(max_length) - input_ids.shape[1]
        if max_new_tokens is None:
            max_new_tokens = 64

        do_sample = kwargs.pop("do_sample", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        if generation_config is not None:
            if do_sample is None:
                do_sample = bool(getattr(generation_config, "do_sample", False))
            if temperature is None:
                temperature = float(getattr(generation_config, "temperature", 1.0))
            if top_p is None:
                top_p = float(getattr(generation_config, "top_p", 1.0))
        do_sample = bool(do_sample) if do_sample is not None else False
        temperature = float(temperature) if temperature is not None else 1.0
        top_p = float(top_p) if top_p is not None else 1.0
        if not do_sample:
            temperature = 0.0

        num_beams = int(kwargs.pop("num_beams", 1))
        num_return_sequences = int(kwargs.pop("num_return_sequences", 1))
        if num_beams != 1 or num_return_sequences != 1:
            raise ValueError(
                "beam search and multiple return sequences are not defined for this "
                "diffusion sampler"
            )

        steps = int(kwargs.pop("diffusion_steps", kwargs.pop("steps", max_new_tokens)))
        block_length = kwargs.pop("block_length", 16)
        generator = kwargs.pop("generator", None)

        for ignored in (
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "use_cache",
            "return_dict_in_generate",
            "output_scores",
        ):
            kwargs.pop(ignored, None)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"unsupported diffusion generation arguments: {unsupported}")

        return self.diffusion_generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            steps=steps,
            block_length=None if block_length is None else int(block_length),
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

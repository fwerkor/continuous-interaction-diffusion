from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
ILLaDATextEncoder = import_module("cid.model.encoding").ILLaDATextEncoder
ILLaDACIDConfig = import_module("cid.model.illada").ILLaDACIDConfig
LFMCIDAdapter = import_module("cid.model.lfm").LFMCIDAdapter
LFMNativeDisplayDecoder = import_module("cid.model.lfm_native").LFMNativeDisplayDecoder
lfm_test = import_module("test_lfm_adapter")


class NativeTokenizer:
    mask_token_id = 16
    eos_token_id = 7
    all_special_ids = [2, 7, 16]
    chat_template = "native"

    def __len__(self):
        return 64

    def encode(self, text, *, add_special_tokens=False):
        if text == "[/Answer]":
            return [31, 32, 33]
        return [3, 4]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens and return_tensors == "pt"
        assert text.startswith("[Question]\n")
        assert text.endswith("[Answer]\n")
        return {"input_ids": torch.tensor([[1, 9, 10, 11, 12, 13]])}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        assert messages[0]["role"] == "user"
        return f"[Question]\n{messages[0]['content']}\n[/Question]\n[Answer]\n"


def _native_adapter():
    return LFMCIDAdapter(
        lfm_test.TinyLFMBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=8,
            max_display_tokens=16,
            interaction_architecture="identity-preserving",
            backbone_protocol="lfm-native",
        ),
        freeze_backbone=True,
    ).eval()


def test_native_requires_identity_path_and_preallocated_display() -> None:
    with pytest.raises(ValueError, match="identity-preserving"):
        ILLaDACIDConfig(backbone_protocol="lfm-native")
    with pytest.raises(ValueError, match="preallocated"):
        ILLaDACIDConfig(
            backbone_protocol="lfm-native",
            interaction_architecture="identity-preserving",
            display_initialization="window",
        )


def test_native_prompt_is_exact_chat_template() -> None:
    adapter = _native_adapter()
    encoder = ILLaDATextEncoder(adapter, NativeTokenizer())
    assert encoder.tokenize_prompt("Hello!", protocol="lfm-native").tolist() == [
        [1, 9, 10, 11, 12, 13]
    ]


def test_native_neutral_logits_match_original_without_extra_eos_mask() -> None:
    adapter = _native_adapter()
    batch = lfm_test.make_batch()
    batch.slot_occupancy.zero_()
    batch.thought_semantic.zero_()
    batch.display_ids[0] = torch.tensor([16, 16, 16, 7])
    with torch.no_grad():
        output = adapter(batch)
        original_ids = torch.cat((batch.prompt_ids, batch.display_ids[:, :-1]), dim=1)
        native_hidden = adapter.hidden_backbone()(
            inputs_embeds=adapter.input_embeddings(original_ids),
            attention_mask=torch.ones_like(original_ids, dtype=torch.bool),
            position_ids=torch.arange(original_ids.shape[1])[None],
            return_dict=True,
        ).last_hidden_state
        original_logits = adapter.output_embeddings(native_hidden[:, -3:])
    torch.testing.assert_close(output.display_logits[:, :-1], original_logits, rtol=0, atol=0)
    assert adapter.backbone.lfm2.last_inputs_embeds.shape[1] == original_ids.shape[1]


def test_native_thought_control_remains_available() -> None:
    adapter = _native_adapter()
    batch = lfm_test.make_batch()
    batch.display_ids[0] = torch.tensor([16, 16, 16, 7])
    with torch.no_grad():
        active = adapter(batch)
        batch.slot_occupancy.zero_()
        inactive = adapter(batch)
    assert not torch.equal(active.display_logits[:, :-1], inactive.display_logits[:, :-1])
    assert active.need_logits.shape == inactive.need_logits.shape


def test_native_decoder_reveals_exactly_one_token_and_uses_confidence() -> None:
    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    display = torch.tensor([[16, 16, 16, 16, 7]])
    logits = torch.zeros((1, 5, 64))
    logits[0, 0, 20] = 6
    logits[0, 1, 21] = 12
    logits[0, 2, 22] = 8
    logits[0, 3, 23] = 7
    logits[:, :, 16] = 30
    out = decoder.refine(display, logits)
    assert out.tolist() == [[16, 21, 16, 16, 7]]
    assert display.tolist() == [[16, 16, 16, 16, 7]]


def test_native_terminator_is_private_and_idempotent() -> None:
    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    display = torch.tensor([[20, 31, 32, 16, 16, 7]])
    logits = torch.zeros(1, 6, 64)
    logits[0, 3, 33] = 10
    result = decoder.refine(display, logits)
    assert result.tolist() == [[20, 7, 16, 16, 16, 7]]
    assert decoder.refine(result, logits).tolist() == result.tolist()


def test_native_terminator_waits_for_prefix_unmasking() -> None:
    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    display = torch.tensor([[16, 31, 32, 33, 16, 7]])
    logits = torch.zeros(1, 6, 64)
    logits[0, 0, 20] = 10
    result = decoder.refine(display, logits)
    assert result.tolist() == [[20, 7, 16, 16, 16, 7]]


def test_native_rejects_nonterminal_eos_and_wrong_shapes() -> None:
    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    with pytest.raises(ValueError, match="terminal"):
        decoder.refine(torch.tensor([[16, 16]]), torch.zeros(1, 2, 64))
    with pytest.raises(ValueError, match="shapes"):
        decoder.refine(torch.tensor([[16, 7]]), torch.zeros(1, 3, 64))

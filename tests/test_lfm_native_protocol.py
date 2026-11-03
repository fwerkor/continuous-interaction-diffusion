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


def test_native_policy_materializes_one_reveal_per_cid_step() -> None:
    from cid.contracts import ModelContext
    from cid.model.materialize import CIDMaterializer, CIDMaterializerConfig
    from cid.model.policy import (
        ILLaDAContextTensorizer,
        ILLaDANeuralPolicy,
        ILLaDANeuralPolicyConfig,
    )
    from cid.state import CognitiveField, DisplayCanvas, FactStore

    adapter = _native_adapter()
    with torch.no_grad():
        adapter.output_heads.allocation_head.weight.zero_()
        adapter.output_heads.allocation_head.bias.fill_(-10)
    tokenizer = NativeTokenizer()
    tensorizer = ILLaDAContextTensorizer(adapter, tokenizer)
    materializer = CIDMaterializer(
        CIDMaterializerConfig(
            allocation_threshold=adapter.config.identity_allocation_threshold,
            max_allocations_per_step=adapter.config.identity_max_allocations_per_step,
            need_threshold=1.0, anchor_presence_threshold=1.0,
            link_presence_threshold=1.0,
        )
    )
    policy = ILLaDANeuralPolicy(
        adapter, tensorizer, materializer=materializer,
        config=ILLaDANeuralPolicyConfig(denoising_steps=10),
    )
    context = ModelContext(
        facts=FactStore().snapshot(),
        thought=CognitiveField.empty(capacity=8, width=adapter.d_model),
        display=DisplayCanvas.initial_preallocated(
            length=8, mask_token_id=16, eos_token_id=7
        ),
        sources=(),
        percepts=(),
        step=0,
        prompt="Hello!",
    )
    update = policy.step(context)
    assert update.display.eos_token_id == 7
    assert update.display.token_ids[-1] == 7
    assert update.display.unresolved == context.display.unresolved - 1
    assert update.display.token_ids != context.display.token_ids


def _revision_logits(display, *proposals):
    logits = torch.full((1, display.shape[1], 64), -12.0)
    for index, token in enumerate(proposals):
        logits[0, index, token] = 12.0
    return logits


def _native_revision(controller, decoder, display, logits):
    return controller.refine_native_display(
        decoder, display, logits, revision_fraction=0.25, revision_margin=0.1,
    )


def test_native_visible_revision_requires_stable_confident_alternative() -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([[20, 21, 22, 7, 16, 7]])
    alternative = _revision_logits(display, 20, 30, 22)
    for step in range(7):
        controller.observe_evidence(step=step, facts_revision=0)
        next_display = _native_revision(controller, decoder, display, alternative)
        assert torch.equal(next_display, display)
    controller.observe_evidence(step=7, facts_revision=0)
    reopened = _native_revision(controller, decoder, display, alternative)
    assert reopened.tolist() == [[20, 16, 22, 7, 16, 7]]
    assert display.tolist() == [[20, 21, 22, 7, 16, 7]]
    controller.observe_evidence(step=8, facts_revision=0)
    revised = _native_revision(controller, decoder, reopened, alternative)
    assert revised.tolist() == [[20, 30, 22, 7, 16, 7]]


def test_native_tool_evidence_reopens_finished_answer_and_corrects_prefix() -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([[20, 21, 7, 16, 16, 7]])
    same = _revision_logits(display, 20, 21, 7)
    controller.observe_evidence(step=0, facts_revision=0)
    assert torch.equal(_native_revision(controller, decoder, display, same), display)
    controller.observe_evidence(step=1, facts_revision=1, percepts=(("tool", 1),))
    changed = _revision_logits(display, 20, 30, 22, 23)
    reopened = _native_revision(controller, decoder, display, changed)
    assert reopened.tolist() == [[20, 16, 16, 16, 16, 7]]
    controller.observe_evidence(step=2, facts_revision=1, percepts=(("tool", 1),))
    refined = _native_revision(controller, decoder, reopened, changed)
    assert refined[0, 1] == 30
    assert refined[0, -1] == 7


def test_native_unchanged_evidence_does_not_reopen_completed_answer() -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([[20, 21, 7, 16, 16, 7]])
    logits = _revision_logits(display, 20, 21, 7)
    for step in range(4):
        controller.observe_evidence(step=step, facts_revision=0)
        assert torch.equal(_native_revision(controller, decoder, display, logits), display)


@pytest.mark.parametrize(
    ("old_tokens", "proposals", "expected"),
    [
        ([20, 21, 22, 23, 7, 16, 16, 7],
         [20, 21, 30, 22, 23, 7],
         [20, 21, 30, 22, 23, 7, 16, 7]),
        ([20, 21, 30, 22, 23, 7, 16, 7],
         [20, 21, 22, 23, 7],
         [20, 21, 22, 23, 7, 16, 16, 7]),
    ],
)
def test_native_tool_evidence_supports_gated_splice(old_tokens, proposals, expected) -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([old_tokens])
    controller.observe_evidence(step=0, facts_revision=0)
    controller.observe_evidence(step=1, facts_revision=1, percepts=("new-tool",))
    revised = _native_revision(
        controller, decoder, display, _revision_logits(display, *proposals)
    )
    assert revised.tolist() == [expected]


def test_native_revision_can_be_disabled() -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([[20, 21, 7, 16, 16, 7]])
    controller.observe_evidence(step=0, facts_revision=0)
    controller.observe_evidence(step=1, facts_revision=1)
    revised = controller.refine_native_display(
        decoder, display, _revision_logits(display, 20, 30, 22),
        revision_fraction=0, revision_margin=0.1,
    )
    assert revised.tolist() == display.tolist()


def test_native_low_confidence_alternative_does_not_force_remask() -> None:
    from cid.model.display_revision import DisplayRevisionController

    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    controller = DisplayRevisionController()
    display = torch.tensor([[20, 21, 7, 16, 16, 7]])
    low_confidence = torch.zeros((1, 6, 64))
    low_confidence[0, 1, 30] = 0.5
    for step in range(12):
        controller.observe_evidence(step=step, facts_revision=0)
        assert torch.equal(
            _native_revision(controller, decoder, display, low_confidence), display
        )


def test_native_can_fill_remasked_position_before_accepted_eos() -> None:
    decoder = LFMNativeDisplayDecoder(NativeTokenizer(), mask_id=16, eos_id=7)
    display = torch.tensor([[20, 16, 22, 7, 16, 7]])
    filled = decoder.refine(display, _revision_logits(display, 20, 30, 22))
    assert filled.tolist() == [[20, 30, 22, 7, 16, 7]]

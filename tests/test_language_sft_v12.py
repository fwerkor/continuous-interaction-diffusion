from __future__ import annotations

import json

import pytest

from cid.language_sft import (
    LanguageSFTExample,
    language_sft_token_budget,
    validate_language_sft,
)

BASE = {
    "format": "cid-language-sft-v1",
    "mode": "language_sft",
    "example_id": "test-1",
    "prompt": "User:\nExplain the unit tests.",
    "answer": "Unit tests isolate individual functions and check their outputs.",
    "category": "knowledge",
    "source": "HuggingFaceH4/ultrachat_200k",
    "license": "mit",
    "supervision": {"display": True, "thought": False, "tct": False, "tool": False},
}


def test_display_only_roundtrip_and_validation(tmp_path):
    e = LanguageSFTExample.from_dict(BASE)
    assert e.to_dict() == BASE
    path = tmp_path / "language.jsonl"
    path.write_text(json.dumps(BASE) + "\n")
    result = validate_language_sft(path)
    assert result["examples"] == 1
    assert result["thought_or_tool_targets"] == 0
    assert result["categories"] == {"knowledge": 1}


@pytest.mark.parametrize(
    "key,value",
    [
        ("thought_targets", []),
        ("tct", {}),
        ("events", []),
        ("tool_calls", []),
        ("binding_targets", []),
        ("target_display", "unexpected"),
    ],
)
def test_rejects_tct_and_tool_supervision(key, value):
    with pytest.raises(ValueError, match="must not contain"):
        LanguageSFTExample.from_dict({**BASE, key: value})


def test_rejects_hidden_thought_and_fake_loss_masks():
    with pytest.raises(ValueError, match="supervise only Display"):
        LanguageSFTExample.from_dict({**BASE, "supervision": {**BASE["supervision"], "tct": True}})
    with pytest.raises(ValueError, match="hidden-reasoning"):
        LanguageSFTExample.from_dict({**BASE, "answer": "<think>private</think> Normal answer."})


def test_token_aware_mixture():
    assert language_sft_token_budget(800) == 200
    assert language_sft_token_budget(0) == 0
    with pytest.raises(ValueError):
        language_sft_token_budget(100, language_fraction=1)


def test_language_sft_training_does_not_run_or_train_thought(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from test_stage_c import SmallTokenizer, make_small_adapter

    from cid import stage_c
    from cid.model.encoding import ILLaDATextEncoder

    adapter = make_small_adapter(display_tokens=12)
    tokenizer = SmallTokenizer()
    encoder = ILLaDATextEncoder(adapter, tokenizer)
    monkeypatch.setattr(
        stage_c,
        "load_base_for_stage_c",
        lambda *args, **kwargs: (adapter, tokenizer, encoder),
    )
    monkeypatch.setattr(
        stage_c,
        "frozen_thought_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("language_sft must not produce TCT trajectories")
        ),
    )
    dataset = tmp_path / "language.jsonl"
    dataset.write_text(
        json.dumps(
            {
                **BASE,
                "prompt": "Hi",
                "answer": "Hello",
            }
        )
        + "\n"
    )
    args = SimpleNamespace(
        device="cpu",
        dtype="fp32",
        checkpoint="mock",
        checkpoint_kind="stage-b",
        model="mock",
        data=str(dataset),
        output_dir=str(tmp_path / "out"),
        rank=4,
        alpha=None,
        learning_rate=0.05,
        epochs=1,
        thought_steps=0,
        max_answer_tokens=32,
        max_examples=1,
        seed=4,
        log_every=0,
        checkpoint_every=0,
        rollout_probability=0.75,
        continuation_tokens=8,
        max_steps=1,
        language_sft=True,
    )
    before = {k: p.detach().clone() for k, p in adapter.named_parameters()}
    stage_c.train_stage_c(args)
    assert (tmp_path / "out" / "stage-c-latest.pt").exists()
    assert set(name for name, p in adapter.named_parameters() if p.requires_grad) == {
        "stage_c_display.down.weight",
        "stage_c_display.up.weight",
    }
    for name, weight in adapter.named_parameters():
        if name in before:
            assert torch.equal(weight.detach(), before[name]), name
            assert weight.grad is None, name

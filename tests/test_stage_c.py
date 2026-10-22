from __future__ import annotations

# ruff: noqa: E402  # Optional torch dependency is checked before importing CID neural modules.
import json

import pytest

torch = pytest.importorskip("torch")

from test_illada_adapter import TinyILLaDABackbone, make_batch

from cid.model import ILLaDACIDAdapter, ILLaDACIDConfig
from cid.model.encoding import ILLaDATextEncoder
from cid.model.policy import ILLaDAContextTensorizer
from cid.stage_c import (
    StageCExample,
    attach_display_adapter,
    display_sft_loss,
    examples_from_record,
    freeze_for_stage_c,
    frozen_thought_rollout,
    load_display_adapter,
    load_stage_c_examples,
    make_display_batch,
    save_display_adapter,
)


class SmallTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_tensors="pt", padding=False):
        if isinstance(text, str):
            text = [text]
        sequences = [[6 + (ord(char) % 40) for char in item] for item in text]
        if add_special_tokens:
            sequences = [[1, *item] for item in sequences]
        max_length = max((len(item) for item in sequences), default=0)
        padded = [item + [0] * (max_length - len(item)) for item in sequences]
        masks = [[1] * len(item) + [0] * (max_length - len(item)) for item in sequences]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def make_small_adapter(*, display_tokens=16):
    adapter = ILLaDACIDAdapter(
        TinyILLaDABackbone(),
        ILLaDACIDConfig(
            max_thought_slots=4, max_display_tokens=48, display_canvas_tokens=display_tokens
        ),
        freeze_backbone=True,
    )
    return adapter


def test_display_adapter_zero_initialization_and_control_isolation():
    torch.manual_seed(24)
    adapter = make_small_adapter()
    baseline = adapter(make_batch(batch_size=1, thought_slots=4, display_length=6))
    batch = make_batch(batch_size=1, thought_slots=4, display_length=6)
    output_without = adapter(batch)
    module = attach_display_adapter(adapter, rank=4)
    assert torch.count_nonzero(module.up.weight) == 0
    output_initial = adapter(batch)
    assert torch.equal(output_without.display_logits, output_initial.display_logits)
    assert torch.equal(output_without.thought_semantic, output_initial.thought_semantic)
    assert torch.equal(output_without.convergence_logits, output_initial.convergence_logits)
    assert torch.equal(output_without.need_logits, output_initial.need_logits)
    assert baseline.display_logits.shape == output_initial.display_logits.shape

    freeze_for_stage_c(adapter)
    assert {name for name, param in adapter.named_parameters() if param.requires_grad} == {
        "stage_c_display.down.weight",
        "stage_c_display.up.weight",
    }
    fixed_before = {
        name: param.detach().clone()
        for name, param in adapter.named_parameters()
        if not name.startswith("stage_c_display.")
    }
    labels = torch.full_like(batch.display_ids, -100)
    labels[:, :2] = (batch.display_ids[:, :2] + 1) % adapter.vocab_size
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss = display_sft_loss(adapter, batch, labels)
    loss.backward()
    assert module.up.weight.grad is not None
    optimizer.step()
    changed = adapter(batch)
    assert not torch.equal(output_without.display_logits, changed.display_logits)
    for name, parameter in adapter.named_parameters():
        if not name.startswith("stage_c_display."):
            assert parameter.grad is None, name
            assert torch.equal(parameter, fixed_before[name]), name
    assert torch.equal(changed.thought_semantic, output_without.thought_semantic)
    assert torch.equal(changed.need_logits, output_without.need_logits)
    assert torch.equal(changed.convergence_logits, output_without.convergence_logits)
    module.enabled = False
    restored = adapter(batch)
    assert torch.equal(restored.display_logits, output_without.display_logits)


def test_stage_c_save_load_only_contains_low_rank_weights(tmp_path):
    adapter = make_small_adapter()
    module = attach_display_adapter(adapter, rank=3, alpha=6)
    with torch.no_grad():
        module.up.weight.fill_(0.02)
    path = tmp_path / "display.pt"
    save_display_adapter(path, adapter, base_checkpoint="base.pt", base_kind="stage-b")
    payload = torch.load(path, weights_only=True)
    assert payload["base_checkpoint"] == "base.pt"
    assert set(payload["state_dict"]) == {"down.weight", "up.weight"}
    other = make_small_adapter()
    imported = load_display_adapter(path, other)
    assert imported.rank == 3
    assert imported.alpha == 6
    batch = make_batch(batch_size=1, thought_slots=4, display_length=5)
    assert torch.allclose(
        module(batch.display_ids.new_ones((1, 5, 32)).float()),
        imported(batch.display_ids.new_ones((1, 5, 32)).float()),
    )
    with pytest.raises(ValueError, match="already attached"):
        load_display_adapter(path, other)
    with pytest.raises(ValueError, match="geometry"):
        load_display_adapter(path, type("Wrong", (), {"d_model": 2, "vocab_size": 13})())


def test_nemotron_only_uses_train_mask_and_never_reasoning():
    row = {
        "messages": [
            {"role": "system", "content": "Reply briefly"},
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "SECRET_THOUGHT"},
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "Because", "reasoning_content": "SECRET_REASON"},
        ],
        "metadata": {"train_turns_mask": [False, False, False, False, True]},
    }
    examples = examples_from_record(row)
    assert len(examples) == 1
    assert examples[0].answer == "Because"
    assert "Hello" in examples[0].prompt
    assert "Why?" in examples[0].prompt
    assert "Reply briefly" in examples[0].prompt
    assert "SECRET" not in examples[0].prompt + examples[0].answer
    with pytest.raises(ValueError, match="align"):
        examples_from_record({**row, "metadata": {"train_turns_mask": [True]}})


def test_prompt_answer_jsonl_load_and_mask(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"prompt": "Hi", "answer": "Hello"})
        + "\n"
        + json.dumps({"prompt": "Why?", "response": "Because"})
        + "\n"
    )
    assert load_stage_c_examples(path) == [
        StageCExample("Hi", "Hello"),
        StageCExample("Why?", "Because"),
    ]
    assert len(load_stage_c_examples(path, max_examples=1)) == 1


def test_real_cid_tensorizer_stage_c_denoising_with_unlabeled_t():
    torch.manual_seed(13)
    adapter = make_small_adapter(display_tokens=12)
    text_encoder = ILLaDATextEncoder(adapter, SmallTokenizer())
    tensorizer = ILLaDAContextTensorizer(adapter, text_encoder.tokenizer, text_encoder=text_encoder)
    # Run the same CID policy + runtime used in inference without any thought targets.
    attach_display_adapter(adapter, rank=4)
    freeze_for_stage_c(adapter)
    original_enabled = adapter.stage_c_display.enabled
    thought = frozen_thought_rollout(adapter, tensorizer, "Hi", steps=1)
    assert thought.capacity == 4
    assert adapter.stage_c_display.enabled == original_enabled
    batch, labels = make_display_batch(
        tensorizer,
        thought,
        StageCExample("Hi", "Hello"),
        generator=torch.Generator().manual_seed(8),
    )
    assert batch.display_ids.shape == labels.shape
    assert bool((labels != -100).any())
    assert torch.equal(labels[0, 6:], torch.full_like(labels[0, 6:], -100))
    loss = display_sft_loss(adapter, batch, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert adapter.stage_c_display.up.weight.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in adapter.named_parameters()
        if not name.startswith("stage_c_display.")
    )


def test_train_display_cli_smoke_on_tiny_cid(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cid import stage_c

    adapter = make_small_adapter(display_tokens=12)
    tokenizer = SmallTokenizer()
    encoder = ILLaDATextEncoder(adapter, tokenizer)
    monkeypatch.setattr(
        stage_c,
        "load_base_for_stage_c",
        lambda *args, **kwargs: (adapter, tokenizer, encoder),
    )
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(json.dumps({"prompt": "Hi", "answer": "Hello"}) + "\n")
    output = tmp_path / "output"
    args = SimpleNamespace(
        device="cpu",
        dtype="fp32",
        checkpoint="mock-stage-b",
        checkpoint_kind="stage-b",
        model="mock",
        data=str(dataset),
        output_dir=str(output),
        rank=4,
        alpha=None,
        learning_rate=0.05,
        epochs=1,
        thought_steps=1,
        max_answer_tokens=32,
        max_examples=1,
        seed=3,
        log_every=1,
        checkpoint_every=1,
    )
    before = {name: parameter.clone() for name, parameter in adapter.named_parameters()}
    stage_c.train_stage_c(args)
    assert (output / "stage-c-latest.pt").exists()
    assert (output / "stage-c-epoch-0001.pt").exists()
    payload = torch.load(output / "stage-c-latest.pt", weights_only=True)
    assert torch.count_nonzero(payload["state_dict"]["up.weight"]) > 0
    for name, weight in before.items():
        assert torch.equal(weight, dict(adapter.named_parameters())[name])


def test_stage_c_rejects_invalid_arguments_and_too_long_answers():
    adapter = make_small_adapter(display_tokens=12)
    text_encoder = ILLaDATextEncoder(adapter, SmallTokenizer())
    tensorizer = ILLaDAContextTensorizer(adapter, text_encoder.tokenizer, text_encoder=text_encoder)
    from cid.state import CognitiveField

    thought = CognitiveField.empty(4, 32)
    with pytest.raises(ValueError, match="exceeds"):
        make_display_batch(
            tensorizer,
            thought,
            StageCExample("Hi", "A" * 60),
            generator=torch.Generator().manual_seed(10),
            maximum_tokens=32,
        )


def test_nemotron_export_streams_only_selected_answers(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from cid.stage_c import export_nemotron_sft

    row = {
        "messages": [
            {"role": "system", "content": "Keep it short"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "private reasoning"},
        ],
        "metadata": {"train_turns_mask": [False, False, True]},
    }

    class FakeStream:
        def shuffle(self, *, seed, buffer_size):
            assert seed == 42 and buffer_size > 0
            return self

        def __iter__(self):
            return iter((row, row))

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda dataset, *, split, streaming: FakeStream()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    target = tmp_path / "nemotron.jsonl"
    counts = export_nemotron_sft(target, splits=("chat",), max_per_split=1)
    assert counts == {"chat": 1}
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["answer"] == "Hello"
    assert "private reasoning" not in json.dumps(records)

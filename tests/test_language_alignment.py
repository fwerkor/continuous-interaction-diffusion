from __future__ import annotations

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from cid.language_alignment import build_la_sample, la_objective


class SampleTokenizer:
    bos_token_id = 1
    mask_token_id = 16
    eos_token_id = 7
    chat_template = "available"

    def encode(self, text, *, add_special_tokens):
        encoded = [20 + (ord(letter) % 45) for letter in text]
        return [1, *encoded] if add_special_tokens else encoded

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "[Question]\n" + messages[0]["content"] + "\n[/Question]\n\n[Answer]\n"


def make_sample(*, seed=4, native=False, p=1.0):
    return build_la_sample(
        SampleTokenizer(),
        "hello",
        "ok",
        mask_probability=p,
        generator=torch.Generator().manual_seed(seed),
        canvas_tokens=16,
        max_answer_tokens=16,
        use_native_template=native,
        supervise_eos_probability=1.0,
    )


def test_la_masks_only_answer_and_eos_keeps_fixed_sentinel():
    sample = make_sample()
    prefix_len = len(SampleTokenizer().encode("hello", add_special_tokens=True))
    assert sample.input_ids[0] == 1
    assert sample.input_ids[prefix_len : prefix_len + 2].tolist() == [16, 16]
    assert sample.input_ids[-1].item() == 7
    assert sample.targets[:3].tolist() == [20 + (ord("o") % 45), 20 + (ord("k") % 45), 7]
    assert sample.targets[3:].eq(-100).all()
    assert sample.content_mask.sum() == 2
    assert sample.content_mask[2].item() is False


def test_la_template_changes_only_the_prompt():
    plain = make_sample(native=False)
    native = make_sample(native=True)
    torch.testing.assert_close(plain.input_ids[-16:], native.input_ids[-16:])
    torch.testing.assert_close(plain.targets, native.targets)
    assert native.input_ids.numel() > plain.input_ids.numel()


def test_la_random_mask_never_leaves_no_supervision():
    for seed in range(16):
        sample = make_sample(seed=seed, p=0.01)
        assert sample.content_mask.any()
        assert sample.targets[2] == 7


def test_la_disallows_missing_prompt_or_eos_issues():
    with pytest.raises(ValueError, match="budget"):
        build_la_sample(
            SampleTokenizer(),
            "hi",
            "hello",
            mask_probability=1.0,
            generator=torch.Generator().manual_seed(1),
            canvas_tokens=3,
            max_answer_tokens=3,
        )


def test_la_objective_distills_content_without_terminal_eos():
    torch.manual_seed(8)
    s = torch.randn(4, 64, requires_grad=True)
    t = torch.randn(4, 64, requires_grad=True)
    targets = torch.tensor([23, 24, 7, -100])
    only_content = torch.tensor([True, True, False, False])
    loss, ce, kl = la_objective(
        s,
        targets,
        teacher_logits=t,
        content_mask=only_content,
        distillation_weight=0.25,
        temperature=2.0,
    )
    assert torch.isfinite(loss)
    assert float(kl) > 0
    torch.testing.assert_close(loss.detach(), ce + 0.25 * kl)
    loss.backward()
    assert s.grad is not None
    assert t.grad is None
    assert s.grad[2].abs().sum() > 0
    with torch.no_grad():
        eos_grad = s.grad[2].clone()
    pure_ce, _, _ = la_objective(s.detach().requires_grad_(), targets)
    pure_ce.backward()
    torch.testing.assert_close(eos_grad, s.grad[2])


def test_la_objective_rejects_invalid_teacher_logits():
    s = torch.randn(2, 15)
    with pytest.raises(ValueError, match="align"):
        la_objective(
            s,
            torch.tensor([2, 4]),
            teacher_logits=torch.randn(1, 15),
            content_mask=torch.tensor([True, False]),
            distillation_weight=0.1,
        )


def test_open_perfectblend_instructions_are_extracted_without_future_messages():
    from cid.language_alignment import open_perfectblend_example

    record = {
        "conversations": [
            {"from": "human", "value": "  Explain the algorithm "},
            {"from": "gpt", "value": "  One possible method... "},
            {"from": "human", "value": "A follow-up"},
            {"from": "gpt", "value": "Another answer"},
        ],
        "source": "test",
    }
    example = open_perfectblend_example(record)
    assert example is not None
    assert example.prompt == "Explain the algorithm"
    assert example.answer == "One possible method..."
    assert open_perfectblend_example({"conversations": [{"from": "human", "value": "hi"}]}) is None
    assert open_perfectblend_example({"conversations": []}) is None

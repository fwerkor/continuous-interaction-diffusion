from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cid.la_format_kl import (  # noqa: E402
    CID_ANSWER_TOKEN,
    PromptOnly,
    TeacherDraft,
    build_pair,
    configure_cid_student,
    pure_format_kl,
    read_prompts,
)


class FakeTokenizer:
    mask_token_id = 16
    eos_token_id = 7
    unk_token_id = 0

    def __init__(self):
        self.vocab = {CID_ANSWER_TOKEN: 70}

    def encode(self, text, *, add_special_tokens):
        ids = [20 + ord(ch) % 40 for ch in text]
        return [1, *ids] if add_special_tokens else ids

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "[Question]\n" + messages[0]["content"] + "\n[/Question]\n\n[Answer]\n"


def test_prompt_only_reader_ignores_untrusted_answers(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"prompt": "Hello", "answer": {"unexpected": "object"}, "source": "S"})
        + "\n"
        + json.dumps({"prompt": "Hello", "answer": "completely different"}) + "\n",
        encoding="utf-8",
    )
    assert read_prompts(str(path)) == [PromptOnly("Hello", "S", "")]


def test_teacher_and_student_share_mask_only_canvas():
    t = FakeTokenizer()
    raw = PromptOnly("Hello", "S", "en")
    pair = build_pair(
        t, t, raw, canvas_width=16, student_answer_id=70,
        generator=torch.Generator().manual_seed(3),
        mask_probability=1.0, max_content_positions=8,
    )
    assert pair.student_ids[0] == 1
    assert pair.student_ids[-17] == 70
    assert pair.student_ids[-16:].eq(16).all()
    assert pair.teacher_ids[-16:].eq(16).all()
    assert pair.teacher_ids.numel() > pair.student_ids.numel()
    assert pair.eos_position is None
    assert pair.content_positions.tolist() == list(range(8))


def test_only_teacher_generated_draft_creates_stop_signal():
    t = FakeTokenizer()
    pair = build_pair(
        t, t, TeacherDraft("Hello", "OK", "en"),
        canvas_width=16, student_answer_id=70,
        generator=torch.Generator().manual_seed(4),
        mask_probability=1, max_content_positions=8,
    )
    assert pair.eos_position == 2
    assert pair.student_ids[-16:].eq(16).all()
    assert pair.teacher_ids[-16:].eq(16).all()


def test_pure_kl_only_backpropagates_through_student():
    s = torch.randn(8, 80, requires_grad=True)
    t = torch.randn(8, 80, requires_grad=True)
    loss, content, stop = pure_format_kl(
        s, t, torch.tensor([0, 1, 2]),
        temperature=2.0, eos_position=3, eos_id=7,
        termination_weight=0.2,
    )
    assert torch.isfinite(loss) and content > 0 and stop > 0
    loss.backward()
    assert s.grad is not None and s.grad[3].abs().sum() > 0
    assert t.grad is None
    with pytest.raises(ValueError, match="vocabulary"):
        pure_format_kl(
            s.detach(), t.detach()[:, :20], torch.tensor([0]),
            temperature=2, eos_position=None, eos_id=7, termination_weight=0,
        )


def test_reserved_token_embedding_is_initialized_without_resize():
    class MockTokenizer:
        unk_token_id = 0

        def __init__(self):
            self.vocab = {"other": 2}
            self.added = {}

        def __len__(self):
            return 64 + len(self.added)

        def get_vocab(self):
            return {**self.vocab, **self.added}

        def convert_tokens_to_ids(self, s):
            return self.added.get(s, self.vocab.get(s, 0))

        def add_special_tokens(self, spec):
            for token in spec["additional_special_tokens"]:
                self.added[token] = 64
            return 1

        def encode(self, text, *, add_special_tokens):
            return [1, 2, 3]

    tokenizer = MockTokenizer()
    emb = torch.nn.Embedding(80, 12)
    original = emb.weight.clone().detach()
    backbone = SimpleNamespace(
        get_input_embeddings=lambda: emb,
        resize_token_embeddings=lambda *a, **kw: pytest.fail("unnecessary resize"),
    )
    token = configure_cid_student(SimpleNamespace(backbone=backbone), tokenizer, resume=False)
    assert token == 64
    assert emb.weight.shape == (80, 12)
    torch.testing.assert_close(emb.weight[token], original[[1, 2, 3]].mean(0))

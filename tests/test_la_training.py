from __future__ import annotations

# ruff: noqa: E402
import json

import pytest

torch = pytest.importorskip("torch")

from cid.la_training import LARecord, canvas_bucket, dataset_sha256, read_la_records, token_ce


def test_canvas_chooses_smallest_compatible_length():
    assert canvas_bucket(1, (64, 128, 192)) == 64
    assert canvas_bucket(63, (64, 128, 192)) == 64
    assert canvas_bucket(64, (64, 128, 192)) == 128
    assert canvas_bucket(127, (64, 128, 192)) == 128
    assert canvas_bucket(190, (64, 128, 192)) == 192
    with pytest.raises(ValueError, match="too long"):
        canvas_bucket(192, (64, 128, 192))


def test_eos_weighting_reduces_eos_contribution():
    logits = torch.tensor([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 2, -100])
    direct = token_ce(logits, labels, eos_id=2, eos_weight=1.0)
    discounted = token_ce(logits, labels, eos_id=2, eos_weight=0.1)
    assert discounted < direct
    discounted.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[2], torch.zeros(3))


def test_la_dataset_reads_metadata_and_hash_stably(tmp_path):
    p = tmp_path / "data.jsonl"
    example = {"prompt": "Hello", "answer": "Hi!", "source": "source", "language": "en"}
    p.write_text(json.dumps(example) + "\n", encoding="utf-8")
    first = dataset_sha256(str(p))
    rows = read_la_records(str(p))
    assert rows == [LARecord("Hello", "Hi!", "source", "en")]
    assert dataset_sha256(str(p)) == first
    p.write_text(json.dumps({**example, "answer": "Hi!!"}) + "\n", encoding="utf-8")
    assert dataset_sha256(str(p)) != first


def test_la_dataset_rejects_empty_response(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt":"a","answer":""}\n')
    with pytest.raises(ValueError, match="line 1"):
        read_la_records(str(p))


def test_kd_full_mask_probability_bounds():
    import argparse

    from cid.la_training import train_language_alignment_v2

    base = argparse.Namespace(
        max_steps=1,
        grad_accum=1,
        learning_rate=1e-5,
        min_mask_probability=0.2,
        max_mask_probability=1.0,
        save_every=1,
        log_every=1,
        keep_last=2,
        zh_repeat=1.0,
        eos_weight=0.8,
        warmup_steps=1,
        lr_decay_steps=100,
        min_lr_ratio=0.1,
        canvas_buckets="64",
        max_answer_tokens=64,
        distillation_weight=0.0,
        native_teacher=None,
        full_mask_probability=1.01,
    )
    with pytest.raises(ValueError, match="full-mask"):
        train_language_alignment_v2(base)

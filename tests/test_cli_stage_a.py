from __future__ import annotations

import sys

import pytest

import cid.cli as cid_cli
from cid.cli import _estimate_stage_a_full_activation_bytes


def test_stage_a_activation_estimate_scales_with_geometry() -> None:
    base = _estimate_stage_a_full_activation_bytes(
        batch_size=1,
        sequence_tokens=1024,
        layer_count=32,
        hidden_size=4096,
        intermediate_size=11008,
        element_size=2,
    )
    doubled_batch = _estimate_stage_a_full_activation_bytes(
        batch_size=2,
        sequence_tokens=1024,
        layer_count=32,
        hidden_size=4096,
        intermediate_size=11008,
        element_size=2,
    )
    assert doubled_batch == 2 * base
    assert base > 3 * 1024**3


def test_stage_a_activation_estimate_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _estimate_stage_a_full_activation_bytes(
            batch_size=0,
            sequence_tokens=1024,
            layer_count=32,
            hidden_size=4096,
            intermediate_size=11008,
            element_size=2,
        )


def test_stage_a_parser_registers_training_memory_flags(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cid",
            "train",
            "--data",
            "train.jsonl",
            "--output-dir",
            "out",
            "--frozen-backbone-sharding",
            "--selective-checkpoint-fraction",
            "0.5",
            "--selective-checkpoint-prompt-tokens",
            "768",
            "--stage-a-async-activation-offload",
            "--stage-a-async-gradient-reduction",
            "--flatten-teacher-forcing-horizon",
        ],
    )
    monkeypatch.setattr(cid_cli, "_train_stage_a", lambda args: captured.update(vars(args)))

    cid_cli.main()

    assert captured["frozen_backbone_sharding"] is True
    assert captured["selective_checkpoint_fraction"] == 0.5
    assert captured["selective_checkpoint_memory_budget_gib"] is None
    assert captured["selective_checkpoint_prompt_tokens"] == 768
    assert captured["stage_a_async_activation_offload"] is True
    assert captured["stage_a_async_gradient_reduction"] is True
    assert captured["flatten_teacher_forcing_horizon"] is True


def test_stage_a_parser_supplies_selective_checkpoint_defaults(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["cid", "train", "--data", "train.jsonl", "--output-dir", "out"],
    )
    monkeypatch.setattr(cid_cli, "_train_stage_a", lambda args: captured.update(vars(args)))

    cid_cli.main()

    assert captured["frozen_backbone_sharding"] is False
    assert captured["selective_checkpoint_fraction"] is None
    assert captured["selective_checkpoint_memory_budget_gib"] is None
    assert captured["selective_checkpoint_prompt_tokens"] == 512
    assert captured["stage_a_async_activation_offload"] is False
    assert captured["stage_a_async_gradient_reduction"] is False

from __future__ import annotations

# ruff: noqa: E402  # Optional torch must be checked before neural imports.
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from test_stage_c import SmallTokenizer, make_small_adapter

from cid.data import TrajectoryExample
from cid.language_mix import (
    mix_language_rollout_windows,
    mixed_dataset_identity,
    prepare_language_mix,
)
from cid.model.encoding import ILLaDATextEncoder
from cid.model.losses import cid_loss
from cid.model.training import (
    CIDRolloutWindow,
    ILLaDATrajectoryTensorizer,
    collate_training_steps,
    shard_rollout_windows,
    stage_b_consumed_windows_by_bucket,
)


class ToyTokenizer(SmallTokenizer):
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [6 + ord(char) % 40 for char in text]


def write_language(path, n=24):
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(
                json.dumps(
                    {
                        "format": "cid-language-sft-v1",
                        "mode": "language_sft",
                        "example_id": f"lang-{i}",
                        "prompt": f"User:\nQuestion {i}?",
                        "answer": f"Explanation {i}.",
                        "category": "knowledge",
                        "source": "unit_test",
                        "license": "mit",
                        "supervision": {
                            "display": True,
                            "thought": False,
                            "tct": False,
                            "tool": False,
                        },
                    }
                )
                + "\n"
            )


def make_windows(count=30):
    return tuple(
        CIDRolloutWindow(
            TrajectoryExample(
                example_id=f"cid-{i}",
                prompt=f"CID prompt {i}",
                target_display="Answer and evidence",
                metadata={},
            ),
            (-1,),
        )
        for i in range(count)
    )


def test_mixed_schedule_is_repeatable_and_isolates_language(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path)
    cid_windows = make_windows(85)
    mixed, report = mix_language_rollout_windows(
        cid_windows,
        path,
        ToyTokenizer(),
        token_fraction=0.2,
        seed=36,
        max_prompt_tokens=60,
        max_display_tokens=48,
    )
    assert mixed[: len(cid_windows)] == cid_windows
    assert 0.19 < report["language_token_fraction"] < 0.25
    assert report["language_examples"] > 0
    assert report["nominal_language_loss_mass_fraction"] == pytest.approx(0.2)
    assert report["language_examples"] == report["unique_language_examples"]
    assert report["cid_tokenizer_tokens"] == sum(
        len(w.example.prompt) + 1 + len(w.example.target_display) + 1
        for w in cid_windows
    )
    assert sum(w.loss_weight for w in mixed[len(cid_windows) :]) == pytest.approx(
        sum(w.loss_weight for w in cid_windows) / 4
    )
    assert (
        mixed
        == mix_language_rollout_windows(
            cid_windows,
            path,
            ToyTokenizer(),
            token_fraction=0.2,
            seed=36,
            max_prompt_tokens=60,
            max_display_tokens=48,
        )[0]
    )
    assert all(w.example.metadata["cid_language_sft"] for w in mixed[len(cid_windows) :])
    assert all(
        not w.example.thought_targets and not w.example.binding_targets and not w.example.events
        for w in mixed[len(cid_windows) :]
    )
    assert all(w.is_language_sft for w in mixed[len(cid_windows) :])
    assert not any(w.is_language_sft for w in mixed[: len(cid_windows)])


def test_mixed_shards_replay_and_elastic_cursor(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path)
    windows, _ = mix_language_rollout_windows(
        make_windows(46),
        path,
        ToyTokenizer(),
        token_fraction=0.3,
        seed=77,
        max_prompt_tokens=60,
        max_display_tokens=48,
    )
    for world_size in (2, 4):
        shards = [
            shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=77,
                epoch=1,
                micro_batch_size=2,
                shuffle=True,
                length_aware=True,
                portable_bucket_order=True,
                balanced_bucket_order=True,
            )
            for rank in range(world_size)
        ]
        assert len({len(s) for s in shards}) == 1
        mode_sequences = [
            [(len(w.source_steps), w.loss_weight, w.is_language_sft) for w in shard]
            for shard in shards
        ]
        assert all(
            len({mode_sequences[r][i] for r in range(world_size)}) == 1
            for i in range(len(shards[0]))
        )
        consumed = stage_b_consumed_windows_by_bucket(
            windows,
            shards[0],
            local_windows_seen=len(shards[0]) // 2,
            world_size=world_size,
        )
        resumed = [
            shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=77,
                epoch=1,
                micro_batch_size=2,
                shuffle=True,
                consumed_windows_by_bucket=consumed,
                length_aware=True,
                portable_bucket_order=True,
                balanced_bucket_order=True,
            )
            for rank in range(world_size)
        ]
        assert len({len(s) for s in resumed}) == 1
        assert all(key.endswith(":language_sft") or ":language_sft" not in key for key in consumed)


def test_language_tensorizer_and_gradient_are_display_only():
    adapter = make_small_adapter(display_tokens=12)
    tokenizer = ToyTokenizer()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        tokenizer,
        text_encoder=ILLaDATextEncoder(adapter, tokenizer),
    )
    example = TrajectoryExample(
        example_id="language_sft:single",
        prompt="Hi",
        target_display="Hello",
        metadata={"cid_language_sft": True},
    )
    step = tensorizer.tensorize(example, -1, timestep=0.65)
    assert step.targets.display_only_mask.tolist() == [True]
    assert step.targets.display_ids.ne(-100).any()
    batch = collate_training_steps((step,), pad_token_id=tokenizer.pad_token_id)
    output = adapter(batch.batch)
    losses = cid_loss(output, batch.targets)
    assert torch.isfinite(losses.total)
    assert float(losses.display.detach()) > 0
    assert float(losses.total.detach()) == pytest.approx(float(losses.display.detach()))
    assert all(
        float(getattr(losses, name).detach()) == 0
        for name in (
            "thought",
            "convergence",
            "intent",
            "allocation",
            "need_cell_route",
            "need_display_route",
            "roles",
            "auxiliary",
        )
    )
    losses.total.backward()
    assert any(p.grad is not None for n, p in adapter.named_parameters() if "display" in n)
    assert all(
        p.grad is None
        for n, p in adapter.named_parameters()
        if "convergence_head" in n or "need_head" in n
    )


def test_language_rejects_fake_cognitive_trajectories():
    adapter = make_small_adapter(display_tokens=12)
    tokenizer = ToyTokenizer()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        tokenizer,
        text_encoder=ILLaDATextEncoder(adapter, tokenizer),
    )
    row = TrajectoryExample(
        example_id="language_sft:invalid",
        prompt="Hi",
        target_display="Hello",
        metadata={"cid_language_sft": True},
    )
    with pytest.raises(ValueError, match="no Thought/TCT"):
        tensorizer.tensorize(row, 0)


def test_mixed_fingerprint_changes_with_source_and_ratio(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path)
    baseline, manifest = mixed_dataset_identity(
        "cid-hash",
        path,
        token_fraction=0.2,
        max_prompt_tokens=128,
    )
    assert manifest["examples"] == 24
    assert baseline != "cid-hash"
    assert (
        mixed_dataset_identity(
            "cid-hash",
            path,
            token_fraction=0.3,
            max_prompt_tokens=128,
        )[0]
        != baseline
    )
    assert (
        mixed_dataset_identity(
            "cid-hash",
            path,
            token_fraction=0.2,
            max_prompt_tokens=128,
        )[0]
        == baseline
    )
    assert (
        mixed_dataset_identity(
            "cid-hash",
            None,
            token_fraction=0.2,
            max_prompt_tokens=128,
        )[0]
        == "cid-hash"
    )


def test_separate_buckets_prevent_mixed_batch():
    original = make_windows(1)[0]
    language = replace(original, is_language_sft=True)
    shards = shard_rollout_windows(
        (original, language),
        world_size=1,
        rank=0,
        seed=10,
        epoch=1,
        micro_batch_size=4,
        portable_bucket_order=True,
    )
    assert len(shards) == 2
    assert shards[0].is_language_sft != shards[1].is_language_sft


def test_stage_a_optimizer_checkpoint_roundtrip_with_language_only(tmp_path):
    from cid.model.training import CIDTrainer, CIDTrainerConfig

    tokenizer = ToyTokenizer()
    adapter = make_small_adapter(display_tokens=12)
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter,
        tokenizer,
        text_encoder=ILLaDATextEncoder(adapter, tokenizer),
    )
    config = CIDTrainerConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        rollout_horizon=1,
    )
    trainer = CIDTrainer(adapter, tensorizer, config=config)
    example = TrajectoryExample(
        example_id="language_sft:checkpoint",
        prompt="Hi",
        target_display="Hello",
        metadata={"cid_language_sft": True},
    )
    window = CIDRolloutWindow(example, (-1,), is_language_sft=True)
    report = trainer.train_rollout_windows((window,), epochs=1, shuffle=False)
    assert report.optimizer_steps == 1
    assert report.component_mean_losses["display"] > 0
    assert report.component_mean_losses["thought"] == 0
    assert report.component_mean_losses["convergence"] == 0
    assert report.component_mean_losses["intent"] == 0
    checkpoint = tmp_path / "mixed.pt"
    trainer.save_checkpoint(checkpoint, dataset_sha256="mixed-checksum")
    adapter2 = make_small_adapter(display_tokens=12)
    tensorizer2 = ILLaDATrajectoryTensorizer(
        adapter2,
        tokenizer,
        text_encoder=ILLaDATextEncoder(adapter2, tokenizer),
    )
    trainer2 = CIDTrainer(adapter2, tensorizer2, config=config)
    trainer2.load_checkpoint(checkpoint, expected_dataset_sha256="mixed-checksum")
    assert trainer2.state.optimizer_steps == 1
    with pytest.raises(ValueError, match="dataset"):
        trainer2.load_checkpoint(checkpoint, expected_dataset_sha256="different-checksum")


def test_epoch_rotation_and_resume_identity(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path, n=150)
    plan = prepare_language_mix(
        make_windows(85),
        path,
        ToyTokenizer(),
        token_fraction=0.2,
        seed=123,
        max_prompt_tokens=60,
        max_display_tokens=48,
        cache_dir=tmp_path,
    )
    first, metrics1 = plan.for_epoch(1)
    again, _ = plan.for_epoch(1)
    second, metrics2 = plan.for_epoch(2)
    assert first == again
    original = len(plan.cid_windows)
    assert first[:original] == second[:original]
    ids1 = {w.example.example_id for w in first[original:]}
    ids2 = {w.example.example_id for w in second[original:]}
    assert ids1 != ids2
    assert metrics1["epoch"] == 1 and metrics2["epoch"] == 2
    assert metrics2["nominal_language_loss_mass_fraction"] == pytest.approx(0.2)


def test_exact_token_budget_for_indexed_trajectory(tmp_path):
    from cid.data import TrajectoryExampleIndex

    raw = {
        "example_id": "indexed-case",
        "prompt": "Question",
        "target_display": "Final!",
        "display_targets": [
            {"step": 0, "text": "Draft"},
            {"step": 1, "text": "Final!"},
        ],
        "metadata": {},
    }
    row = (json.dumps(raw) + "\n").encode()
    source = tmp_path / "cid.jsonl"
    source.write_bytes(row)
    idx = TrajectoryExampleIndex(
        offset=0,
        length=len(row),
        example_id="indexed-case",
        metadata={},
        training_source_steps=(-1, 0),
        rollout_length_key=50,
    )
    windows = (CIDRolloutWindow(idx, (-1, 0)),)
    path = tmp_path / "language.jsonl"
    write_language(path, n=50)
    plan = prepare_language_mix(
        windows,
        path,
        ToyTokenizer(),
        token_fraction=0.2,
        seed=456,
        max_prompt_tokens=60,
        max_display_tokens=48,
        cid_data_path=source,
        cid_sha256="fixed-cid-sha",
        cache_dir=tmp_path,
    )
    expected_display = len("Draft") + 1 + len("Final!") + 1
    expected_total = expected_display + 2 * (len("Question") + 1)
    assert plan.cid_display_tokens == expected_display
    assert plan.cid_tokens == expected_total
    assert plan.for_epoch(1)[1]["cid_tokenizer_tokens"] == expected_total
    cache_files = list(tmp_path.glob("cid-token-budget-*.json"))
    assert len(cache_files) == 1
    assert (
        plan.cid_tokens
        == prepare_language_mix(
            windows,
            path,
            ToyTokenizer(),
            token_fraction=0.2,
            seed=456,
            max_prompt_tokens=60,
            max_display_tokens=48,
            cid_data_path=source,
            cid_sha256="fixed-cid-sha",
            cache_dir=tmp_path,
        ).cid_tokens
    )


def test_different_configurations_invalidate_mixed_checkpoint(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path)
    base, _ = mixed_dataset_identity(
        "cid",
        path,
        token_fraction=0.2,
        max_prompt_tokens=128,
        max_display_tokens=256,
        model_identity="model-A",
    )
    assert (
        base
        != mixed_dataset_identity(
            "cid",
            path,
            token_fraction=0.2,
            max_prompt_tokens=128,
            max_display_tokens=512,
            model_identity="model-A",
        )[0]
    )
    assert (
        base
        != mixed_dataset_identity(
            "cid",
            path,
            token_fraction=0.2,
            max_prompt_tokens=128,
            max_display_tokens=256,
            model_identity="model-B",
        )[0]
    )


def test_reject_repeating_small_language_pool(tmp_path):
    path = tmp_path / "language.jsonl"
    write_language(path, n=2)
    with pytest.raises(ValueError, match="pool is too small"):
        mix_language_rollout_windows(
            make_windows(180),
            path,
            ToyTokenizer(),
            token_fraction=0.5,
            max_prompt_tokens=60,
            max_display_tokens=48,
            seed=4,
        )

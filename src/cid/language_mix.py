"""Deterministic Display-only language_sft augmentation for CID Stage A/B."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from cid.data import TrajectoryExample
from cid.language_sft import iter_language_sft, validate_language_sft
from cid.model.training import CIDRolloutWindow, _rollout_window_length_key


def mixed_dataset_identity(
    cid_sha256: str,
    language_sft_path: str | Path | None,
    *,
    token_fraction: float,
    max_prompt_tokens: int,
) -> tuple[str, dict[str, Any] | None]:
    if language_sft_path is None:
        return cid_sha256, None
    if not 0 < token_fraction < 1 or max_prompt_tokens <= 0:
        raise ValueError("invalid language_sft mixture fraction or maximum prompt length")
    manifest = validate_language_sft(language_sft_path)
    contract = {
        "v": "cid-v1.2-mixed-stageab-v1",
        "cid_sha256": cid_sha256,
        "language_sft_sha256": manifest["sha256"],
        "language_token_fraction": token_fraction,
        "max_prompt_tokens": max_prompt_tokens,
    }
    sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return sha, manifest


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def mix_language_rollout_windows(
    cid_windows: tuple[CIDRolloutWindow, ...],
    language_sft_path: str | Path,
    tokenizer: Any,
    *,
    token_fraction: float = 0.20,
    max_prompt_tokens: int = 1024,
    max_display_tokens: int = 1536,
    seed: int = 0,
    backbone_protocol: str = "cid-answer",
) -> tuple[tuple[CIDRolloutWindow, ...], dict[str, float | int]]:
    if not cid_windows:
        raise ValueError("cannot mix language_sft without original CID windows")
    if not 0 < token_fraction < 1:
        raise ValueError("language_sft token fraction must be in (0, 1)")
    if max_prompt_tokens <= 0 or max_display_tokens <= 2:
        raise ValueError("invalid language_sft token limits")

    # Compact CID indexes keep prompt+Display characters but not token IDs.
    # A common chars-to-tokens conversion gives a reproducible budget without
    # loading all 8+ GB of trajectories on every FSDP rank.
    cid_budget = sum(
        max(1, math.ceil(_rollout_window_length_key(w) / 4)) * len(w.source_steps)
        for w in cid_windows if not w.is_padding
    )
    target_language_budget = round(cid_budget * token_fraction / (1.0 - token_fraction))
    eligible: list[tuple[CIDRolloutWindow, int]] = []
    skipped = 0
    for record in iter_language_sft(language_sft_path):
        prompt_tokens = _token_count(tokenizer, record.prompt)
        answer_tokens = _token_count(tokenizer, record.answer)
        terminal_tokens = (
            _token_count(tokenizer, "\n[/Answer]") if backbone_protocol == "lfm-native" else 1
        )
        if (
            prompt_tokens > max_prompt_tokens
            or answer_tokens < 1
            or answer_tokens + terminal_tokens >= max_display_tokens
        ):
            skipped += 1
            continue
        example = TrajectoryExample(
            example_id="language_sft:" + record.example_id,
            prompt=record.prompt,
            target_display=record.answer,
            # No thought_targets, events, tool calls, or TCT labels are created.
            metadata={"cid_language_sft": True},
        )
        window = CIDRolloutWindow(example, (-1,), is_language_sft=True)
        estimated_tokens = max(1, math.ceil(
            (len(record.prompt) + len(record.answer)) / 4
        ))
        eligible.append((window, estimated_tokens))
    if not eligible:
        raise ValueError("no valid language_sft records fit Stage A/B geometry")
    rng = random.Random(seed)
    order = list(range(len(eligible)))
    selected: list[CIDRolloutWindow] = []
    language_budget = 0
    passes = 0
    while language_budget < target_language_budget:
        rng.shuffle(order)
        passes += 1
        for index in order:
            window, estimated_tokens = eligible[index]
            selected.append(window)
            language_budget += estimated_tokens
            if language_budget >= target_language_budget:
                break
        if passes > 64:
            raise ValueError("language_sft token target requires excessive repetition")
    merged = cid_windows + tuple(selected)
    measured_fraction = language_budget / (cid_budget + language_budget)
    return merged, {
        "cid_estimated_tokens": cid_budget,
        "language_estimated_tokens": language_budget,
        "estimated_language_fraction": round(measured_fraction, 6),
        "language_examples": len(selected),
        "unique_eligible_language_examples": len(eligible),
        "skipped_examples": skipped,
        "sample_passes": passes,
    }

"""Tokenizer-exact, deterministic language_sft schedules for CID Stage A/B."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import DISPLAY_UNKNOWN_MARKER, TrajectoryExample, TrajectoryExampleIndex
from cid.language_sft import LanguageSFTExample, iter_language_sft, validate_language_sft
from cid.model.training import CIDRolloutWindow

MIX_VERSION = "cid-v1.2-language-token-mix-v2"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tokenizer_identity(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab() if callable(getattr(tokenizer, "get_vocab", None)) else {}
    return _sha256_json(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab": vocab,
            "chat_template": getattr(tokenizer, "chat_template", None),
            "special_tokens": getattr(tokenizer, "special_tokens_map", None),
            "name": str(getattr(tokenizer, "name_or_path", "")),
        }
    )


def mixed_dataset_identity(
    cid_sha256: str,
    language_sft_path: str | Path | None,
    *,
    token_fraction: float,
    max_prompt_tokens: int,
    max_display_tokens: int | None = None,
    model_identity: str = "",
    sampling_seed: int = 0,
    rollout_horizon: int = 1,
    max_examples: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    if language_sft_path is None:
        return cid_sha256, None
    if not 0 < token_fraction < 1 or max_prompt_tokens <= 0:
        raise ValueError("invalid language_sft mixture fraction or maximum prompt length")
    manifest = validate_language_sft(language_sft_path)
    contract = {
        "v": MIX_VERSION,
        "cid_sha256": cid_sha256,
        "language_sft_sha256": manifest["sha256"],
        "language_token_fraction": token_fraction,
        "max_prompt_tokens": max_prompt_tokens,
        "max_display_tokens": max_display_tokens,
        "model_identity": model_identity,
        "sampling_seed": sampling_seed,
        "rollout_horizon": rollout_horizon,
        "max_examples": max_examples,
    }
    return _sha256_json(contract), manifest


def _token_count(tokenizer: Any, text: str, *, special: bool = False) -> int:
    return len(tokenizer.encode(text, add_special_tokens=special))


def _prompt_tokens(tokenizer: Any, prompt: str, protocol: str) -> int:
    if protocol == "lfm-native":
        template = getattr(tokenizer, "chat_template", None)
        if template is None:
            raise ValueError("lfm-native token budgeting requires a chat template")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    count = _token_count(tokenizer, prompt, special=True)
    return count + (1 if protocol == "cid-answer" else 0)


def _visible_tokens(tokenizer: Any, answer: str) -> int:
    # CID unknown markers denote input [MASK] placeholders, not target classes.
    if DISPLAY_UNKNOWN_MARKER not in answer:
        return _token_count(tokenizer, answer)
    return sum(_token_count(tokenizer, part) for part in answer.split(DISPLAY_UNKNOWN_MARKER))


def _cid_display_tokens(
    tokenizer: Any, raw: dict[str, Any], target_step: int, protocol: str
) -> int:
    text = next(
        (str(d["text"]) for d in raw.get("display_targets", ()) if int(d["step"]) == target_step),
        str(raw["target_display"]),
    )
    if protocol == "lfm-native" and DISPLAY_UNKNOWN_MARKER not in text:
        metadata = raw.get("metadata", {})
        full = metadata.get("native_full_display_by_step", {})
        complete = full.get(str(target_step)) if isinstance(full, dict) else None
        if complete is None and text == raw["target_display"]:
            complete = metadata.get("native_full_display")
        if complete:
            return _token_count(tokenizer, complete + "\n[/Answer]")
    # A fully unknown Display has no lexical target, nor a positive EOS label.
    if text == DISPLAY_UNKNOWN_MARKER:
        return 0
    return _visible_tokens(tokenizer, text) + (0 if protocol == "lfm-native" else 1)


def _cid_window_token_budget(
    windows: tuple[CIDRolloutWindow, ...],
    tokenizer: Any,
    *,
    protocol: str,
    cid_data_path: str | Path | None,
    cid_sha256: str,
    cache_dir: str | Path | None,
) -> tuple[int, int]:
    """Count exactly tokenized Prompt and target Display across all CID transitions.

    This is a reference-token budget: stochastic MASK corruption can supervise
    fewer tokens on a particular optimizer step. It is not a claim about
    gradient magnitudes or realized loss-mask counts.
    """
    indexed: dict[tuple[int, int], set[int]] = defaultdict(set)
    inline: dict[int, tuple[TrajectoryExample, set[int]]] = {}
    for window in windows:
        if window.is_padding:
            continue
        source = window.example
        if isinstance(source, TrajectoryExampleIndex):
            indexed[(source.offset, source.length)].update(step + 1 for step in window.source_steps)
        else:
            key = id(source)
            if key not in inline:
                inline[key] = (source, set())
            inline[key][1].update(step + 1 for step in window.source_steps)

    if indexed and cid_data_path is None:
        raise ValueError("CID indexed windows require --data for exact tokenizer accounting")
    selection_digest = _sha256_json(
        (
            [
                (offset, length, sorted(steps))
                for (offset, length), steps in sorted(indexed.items())
            ],
            [(item.example_id, sorted(steps)) for item, steps in inline.values()],
        )
    )
    key = _sha256_json(
        {
            "version": MIX_VERSION,
            "cid_sha256": cid_sha256,
            "dataset_path": str(Path(cid_data_path).resolve()) if indexed else None,
            "tokenizer": tokenizer_identity(tokenizer),
            "protocol": protocol,
            "selection": selection_digest,
        }
    )
    directory = Path(
        cache_dir
        or os.environ.get("CID_LANGUAGE_MIX_CACHE_DIR", Path.home() / ".cache/cid/language-mix")
    )
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"cid-token-budget-{key}.json"
    lock = directory / f"cid-token-budget-{key}.lock"
    with lock.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if cache.is_file():
            record = json.loads(cache.read_text())
            if record.get("key") == key:
                return int(record["input_and_display_tokens"]), int(record["display_target_tokens"])

        total = 0
        display = 0

        def count_row(raw: dict[str, Any], steps: set[int]) -> None:
            nonlocal total, display
            prompt_len = _prompt_tokens(tokenizer, str(raw["prompt"]), protocol)
            token_lengths = {
                step: _cid_display_tokens(tokenizer, raw, step, protocol) for step in steps
            }
            display += sum(token_lengths.values())
            total += sum(prompt_len + tokens for tokens in token_lengths.values())

        for item, steps in inline.values():
            raw = {
                "prompt": item.prompt,
                "target_display": item.target_display,
                "display_targets": [dict(step=t.step, text=t.text) for t in item.display_targets],
                "metadata": dict(item.metadata),
            }
            count_row(raw, steps)
        if indexed:
            with Path(cid_data_path).open("rb") as data:
                for (offset, length), steps in sorted(indexed.items()):
                    data.seek(offset)
                    raw = json.loads(data.read(length))
                    count_row(raw, steps)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "key": key,
                    "input_and_display_tokens": total,
                    "display_target_tokens": display,
                    "trajectory_examples": len(indexed) + len(inline),
                },
                sort_keys=True,
            )
        )
        tmp.replace(cache)
        return total, display


@dataclass(frozen=True, slots=True)
class _Candidate:
    row: LanguageSFTExample
    input_and_display_tokens: int
    display_tokens: int


@dataclass(slots=True)
class LanguageMixPlan:
    cid_windows: tuple[CIDRolloutWindow, ...]
    candidates: tuple[_Candidate, ...]
    cid_tokens: int
    cid_display_tokens: int
    token_fraction: float
    seed: int
    skipped: int
    cid_loss_mass: float

    def for_epoch(self, epoch: int) -> tuple[tuple[CIDRolloutWindow, ...], dict[str, float | int]]:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        rng = random.Random(self.seed + epoch * 1_000_003)
        order = list(range(len(self.candidates)))
        target = round(self.cid_tokens * self.token_fraction / (1 - self.token_fraction))
        selected: list[_Candidate] = []
        language_tokens = 0
        language_display = 0
        passes = 0
        while language_tokens < target:
            passes += 1
            if passes > 2:
                raise ValueError(
                    "language_sft pool is too small for 20% token mix without "
                    "excessive repeated examples; expand the corpus"
                )
            rng.shuffle(order)
            for index in order:
                item = self.candidates[index]
                selected.append(item)
                language_tokens += item.input_and_display_tokens
                language_display += item.display_tokens
                if language_tokens >= target:
                    break

        # CID windows retain their original semantic importance weights.
        # CID task balancing defines mean loss mass 1 per transition. The
        # language stream receives 20% of *nominal total objective mass*
        # independent of the typical response length.
        language_loss_mass = self.cid_loss_mass * self.token_fraction / (1 - self.token_fraction)
        language_loss_weight = language_loss_mass / len(selected)
        language_windows = tuple(
            CIDRolloutWindow(
                example=TrajectoryExample(
                    example_id="language_sft:" + item.row.example_id,
                    prompt=item.row.prompt,
                    target_display=item.row.answer,
                    metadata={"cid_language_sft": True},
                ),
                source_steps=(-1,),
                loss_weight=language_loss_weight,
                is_language_sft=True,
            )
            for item in selected
        )
        total_tokens = self.cid_tokens + language_tokens
        report: dict[str, float | int] = {
            "cid_tokenizer_tokens": self.cid_tokens,
            "cid_display_reference_tokens": self.cid_display_tokens,
            "language_tokenizer_tokens": language_tokens,
            "language_display_reference_tokens": language_display,
            "language_token_fraction": round(language_tokens / total_tokens, 6),
            "display_reference_language_fraction": round(
                language_display / (self.cid_display_tokens + language_display), 6
            ),
            "language_loss_weight": round(language_loss_weight, 6),
            "nominal_language_loss_mass_fraction": round(
                language_loss_mass / (self.cid_loss_mass + language_loss_mass), 6
            ),
            "language_examples": len(selected),
            "unique_language_examples": len(set(item.row.example_id for item in selected)),
            "available_language_examples": len(self.candidates),
            "skipped_examples": self.skipped,
            "sample_passes": passes,
            "epoch": epoch,
        }
        return self.cid_windows + language_windows, report


def prepare_language_mix(
    cid_windows: tuple[CIDRolloutWindow, ...],
    language_sft_path: str | Path,
    tokenizer: Any,
    *,
    token_fraction: float = 0.20,
    max_prompt_tokens: int = 1024,
    max_display_tokens: int = 1536,
    seed: int = 0,
    backbone_protocol: str = "cid-answer",
    cid_data_path: str | Path | None = None,
    cid_sha256: str = "",
    cache_dir: str | Path | None = None,
) -> LanguageMixPlan:
    if not cid_windows:
        raise ValueError("cannot mix language_sft without original CID windows")
    if not 0 < token_fraction < 1:
        raise ValueError("language_sft token fraction must be in (0, 1)")
    if max_prompt_tokens <= 0 or max_display_tokens <= 2:
        raise ValueError("invalid language_sft token limits")

    cid_tokens, cid_display_tokens = _cid_window_token_budget(
        cid_windows,
        tokenizer,
        protocol=backbone_protocol,
        cid_data_path=cid_data_path,
        cid_sha256=cid_sha256,
        cache_dir=cache_dir,
    )
    candidates = []
    skipped = 0
    for record in iter_language_sft(language_sft_path):
        prompt_tokens = _prompt_tokens(tokenizer, record.prompt, backbone_protocol)
        answer_tokens = _token_count(
            tokenizer,
            record.answer + ("\n[/Answer]" if backbone_protocol == "lfm-native" else ""),
        )
        trailing = 0 if backbone_protocol == "lfm-native" else 1
        if (
            prompt_tokens > max_prompt_tokens
            or answer_tokens < 1
            or answer_tokens + trailing >= max_display_tokens
        ):
            skipped += 1
            continue
        candidates.append(
            _Candidate(record, prompt_tokens + answer_tokens + trailing, answer_tokens + trailing)
        )
    if not candidates:
        raise ValueError("no eligible language_sft records fit Stage A/B geometry")
    cid_mass = sum(
        window.loss_weight * len(window.source_steps)
        for window in cid_windows
        if not window.is_padding
    )
    return LanguageMixPlan(
        cid_windows,
        tuple(candidates),
        cid_tokens,
        cid_display_tokens,
        token_fraction,
        seed,
        skipped,
        cid_mass,
    )


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
    cid_data_path: str | Path | None = None,
    cid_sha256: str = "",
    cache_dir: str | Path | None = None,
    epoch: int = 1,
) -> tuple[tuple[CIDRolloutWindow, ...], dict[str, float | int]]:
    """One-epoch convenience API; use prepare_language_mix for multi-epoch runs."""
    plan = prepare_language_mix(
        cid_windows,
        language_sft_path,
        tokenizer,
        token_fraction=token_fraction,
        max_prompt_tokens=max_prompt_tokens,
        max_display_tokens=max_display_tokens,
        seed=seed,
        backbone_protocol=backbone_protocol,
        cid_data_path=cid_data_path,
        cid_sha256=cid_sha256,
        cache_dir=cache_dir,
    )
    return plan.for_epoch(epoch)

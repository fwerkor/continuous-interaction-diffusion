"""Display-only language examples and deterministic, source-aware mixing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LANGUAGE_SFT_FORMAT = "cid-language-sft-v1"
LANGUAGE_CATEGORIES = (
    "conversation",
    "knowledge",
    "writing",
    "summarization",
    "coding",
    "mathematics",
    "instruction_following",
    "multilingual",
)
DEFAULT_LANGUAGE_MIX = {
    "conversation": 0.20,
    "knowledge": 0.20,
    "writing": 0.15,
    "summarization": 0.10,
    "coding": 0.10,
    "mathematics": 0.10,
    "instruction_following": 0.10,
    "multilingual": 0.05,
}
DEFAULT_LANGUAGE_TOKEN_FRACTION = 0.20


@dataclass(frozen=True, slots=True)
class LanguageSFTExample:
    example_id: str
    prompt: str
    answer: str
    category: str
    source: str
    license: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> LanguageSFTExample:
        if row.get("format") != LANGUAGE_SFT_FORMAT:
            raise ValueError("language_sft format mismatch")
        if row.get("mode") != "language_sft":
            raise ValueError("language_sft mode is required")
        forbidden = {
            "thought_targets",
            "tct",
            "thought",
            "events",
            "tool_calls",
            "binding_targets",
            "grounding_targets",
            "target_display",
        }
        if forbidden & row.keys():
            raise ValueError("language_sft must not contain synthetic CID/TCT or tool targets")
        supervision = row.get("supervision")
        if supervision != {"display": True, "thought": False, "tct": False, "tool": False}:
            raise ValueError("language_sft must explicitly supervise only Display")
        values = tuple(
            row.get(k)
            for k in (
                "example_id",
                "prompt",
                "answer",
                "category",
                "source",
                "license",
            )
        )
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("language_sft fields must contain nonempty strings")
        example = cls(*(v.strip() for v in values))
        if example.category not in LANGUAGE_CATEGORIES:
            raise ValueError(f"unknown language_sft category: {example.category}")
        if "<|cid_unknown|>" in example.answer or "[/Answer]" in example.answer:
            raise ValueError("invalid Display transport token in language_sft answer")
        if any(
            marker in example.answer.lower() for marker in ("<think>", "</think>", "<|im_start|>")
        ):
            raise ValueError("language_sft contains hidden-reasoning or chat transport markers")
        return example

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_SFT_FORMAT,
            "mode": "language_sft",
            "example_id": self.example_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "category": self.category,
            "source": self.source,
            "license": self.license,
            "supervision": {"display": True, "thought": False, "tct": False, "tool": False},
        }


def iter_language_sft(path: str | Path) -> Iterator[LanguageSFTExample]:
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield LanguageSFTExample.from_dict(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid language_sft row {number}: {exc}") from exc


def validate_language_sft(path: str | Path) -> dict[str, Any]:
    import hashlib

    counter: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    digest = hashlib.sha256()
    ids: set[str] = set()
    pairs: set[bytes] = set()
    with Path(path).open("rb") as f:
        for number, line in enumerate(f, 1):
            digest.update(line)
            row = LanguageSFTExample.from_dict(json.loads(line))
            if row.example_id in ids:
                raise ValueError(f"duplicate language_sft example_id at row {number}")
            ids.add(row.example_id)
            pair_hash = hashlib.sha256((row.prompt + "\x00" + row.answer).encode()).digest()
            if pair_hash in pairs:
                raise ValueError(f"duplicate language_sft prompt/answer at row {number}")
            pairs.add(pair_hash)
            counter[row.category] += 1
            sources[row.source] += 1
    if not ids:
        raise ValueError("language_sft dataset is empty")
    return {
        "format": LANGUAGE_SFT_FORMAT,
        "examples": len(ids),
        "sha256": digest.hexdigest(),
        "categories": dict(sorted(counter.items())),
        "sources": dict(sorted(sources.items())),
        "supervision": "display_only",
        "thought_or_tool_targets": 0,
    }


def language_sft_token_budget(
    cid_tokens: int,
    *,
    language_fraction: float = DEFAULT_LANGUAGE_TOKEN_FRACTION,
) -> int:
    """Token-aware mixing, independent of the number of CID rollout windows."""
    if cid_tokens < 0 or not 0 <= language_fraction < 1:
        raise ValueError("invalid CID budget or language mixture fraction")
    return round(cid_tokens * language_fraction / (1 - language_fraction))

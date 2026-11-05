#!/usr/bin/env python3
"""Build CID-Dataset v1.2 language_sft using licensed SFT sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cid.language_sft import (
    DEFAULT_LANGUAGE_MIX,
    LANGUAGE_CATEGORIES,
    LanguageSFTExample,
    validate_language_sft,
)

SOURCES = {
    "tulu3": ("allenai/tulu-3-sft-mixture", "train", "odc-by"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft", "mit"),
    "oasst1": ("OpenAssistant/oasst1", "train", "apache-2.0"),
}
CODE = re.compile(
    r"\b(code|program|script|python|javascript|typescript|sql|function|algorithm|debug|regex|api|bash|java|c\+\+)\b",
    re.I,
)
MATH = re.compile(
    r"\b(calculate|equation|solve for|mathematical|algebra|probability|proof|"
    r"integral|derivative|fraction|percent|geometry)\b",
    re.I,
)
SUMMARY = re.compile(
    r"\b(summari[sz]e|paraphrase|rewrite|shorten|translate|rephrase|extract|tldr|bullet points)\b",
    re.I,
)
WRITING = re.compile(
    r"\b(write|draft|compose|story|poem|article|essay|email|letter|speech|blog|novel|dialogue)\b",
    re.I,
)
CONSTRAINT = re.compile(
    r"\b(exactly|format|only|json|xml|csv|words|sentences|structure|constraints|"
    r"must include|without using)\b",
    re.I,
)
KNOWLEDGE = re.compile(
    r"\b(explain|why|how does|what causes|compare|difference|history|science|"
    r"concept|describe|meaning)\b",
    re.I,
)
REJECT = (
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    "[/Answer]",
    "<|cid_unknown|>",
    "reasoning_content",
)


def classify(prompt, source, lang=""):
    if (
        lang
        and lang not in ("en", "eng")
        or re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", prompt)
    ):
        return "multilingual"
    if "code" in source.lower() or CODE.search(prompt):
        return "coding"
    if "math" in source.lower() or MATH.search(prompt):
        return "mathematics"
    if SUMMARY.search(prompt):
        return "summarization"
    if WRITING.search(prompt):
        return "writing"
    if CONSTRAINT.search(prompt):
        return "instruction_following"
    if KNOWLEDGE.search(prompt):
        return "knowledge"
    return "conversation"


def dialog_pairs(messages):
    if not isinstance(messages, list):
        return
    turns = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant", "system") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "assistant" and any(r == "user" for r, _ in turns):
            yield "\n\n".join(f"{r.capitalize()}:\n{t}" for r, t in turns[-5:]), content
        turns.append((role, content))


def eligible(prompt, answer):
    return (
        16 <= len(prompt) <= 2600
        and 65 <= len(answer) <= 3000
        and (len(answer.split()) >= 11 or re.search(r"[\u3400-\u9fff]", answer))
        and not any(x in prompt or x in answer for x in REJECT)
    )


def build(output, seed, target, scan_limit):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_TOKEN"] = ""
    quotas = {k: round(target * v) for k, v in DEFAULT_LANGUAGE_MIX.items()}
    quotas["conversation"] += target - sum(quotas.values())
    buckets = {k: [] for k in LANGUAGE_CATEGORIES}
    caps = {"tulu3": round(target * 0.53), "ultrachat": round(target * 0.40), "oasst1": target}
    counts = Counter()
    seen = set()
    audit = Counter()
    revisions = {}
    rng = random.Random(seed)
    api = HfApi(token=False)
    for key, (repo, split, lic) in SOURCES.items():
        revisions[repo] = api.dataset_info(repo).sha
        print("SOURCE", key, repo, flush=True)
        ds = load_dataset(repo, split=split, streaming=True).shuffle(seed=seed, buffer_size=2500)
        if key == "oasst1":
            rows = []
            for i, row in enumerate(ds):
                if i >= scan_limit:
                    break
                rows.append(row)
            by_id = {r["message_id"]: r for r in rows}
            pairs = []
            for row in rows:
                if row.get("role") != "assistant" or row.get("review_result") is not True:
                    continue
                parent = by_id.get(row.get("parent_id"))
                if parent and parent.get("role") == "prompter":
                    pairs.append(
                        (
                            {
                                "messages": [
                                    {"role": "user", "content": parent["text"]},
                                    {"role": "assistant", "content": row["text"]},
                                ],
                                "lang": row.get("lang", "en"),
                            },
                            str(row["message_id"]),
                        )
                    )
            stream = enumerate(pairs)
        else:
            stream = (
                (i, (row, str(row.get("id", row.get("prompt_id", i))))) for i, row in enumerate(ds)
            )
        for i, (row, source_id) in stream:
            if i >= scan_limit or counts[key] >= caps[key]:
                break
            candidates = list(dialog_pairs(row.get("messages")))
            if not candidates:
                continue
            prompt, answer = candidates[-1]
            if not eligible(prompt, answer):
                audit["filtered"] += 1
                continue
            cat = classify(prompt, str(row.get("source", key)), row.get("lang", ""))
            if len(buckets[cat]) >= quotas[cat]:
                continue
            hash_pair = hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()
            if hash_pair in seen:
                continue
            seen.add(hash_pair)
            id_value = hashlib.sha256((source_id + hash_pair).encode()).hexdigest()[:20]
            buckets[cat].append(
                LanguageSFTExample(f"{key}-{id_value}", prompt, answer, cat, repo, lic)
            )
            counts[key] += 1
        print("COLLECTED", key, counts[key], {k: len(v) for k, v in buckets.items()}, flush=True)
        del ds
    output.parent.mkdir(parents=True, exist_ok=True)
    examples = [e for bucket in buckets.values() for e in bucket]
    rng.shuffle(examples)
    temp = output.with_suffix(output.suffix + ".building")
    try:
        with temp.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    manifest = validate_language_sft(output)
    manifest.update(
        {
            "dataset_version": "1.2",
            "requested": target,
            "seed": seed,
            "category_targets": quotas,
            "upstream_revisions": revisions,
            "upstream_licenses": {v[0]: v[2] for v in SOURCES.values()},
            "source_scan_limit": scan_limit,
            "filter_audit": dict(audit),
            "recommended_language_token_fraction": 0.20,
            "recommended_cid_token_fraction": 0.80,
        }
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--target", type=int, default=48000)
    p.add_argument("--scan-limit", type=int, default=260000)
    a = p.parse_args()
    print(json.dumps(build(a.output, a.seed, a.target, a.scan_limit), indent=2))

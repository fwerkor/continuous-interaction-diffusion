from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def extract_sharegpt(record):
    for idx, turn in enumerate(record.get("conversations", ())):
        if turn.get("from") != "human" or not isinstance(turn.get("value"), str):
            continue
        for response in record["conversations"][idx + 1 :]:
            if response.get("from") == "human":
                break
            if response.get("from") == "gpt" and isinstance(response.get("value"), str):
                return turn["value"].strip(), response["value"].strip()
        break
    return None


def extract_ultrachat(record):
    messages = record.get("messages", [])
    for idx, turn in enumerate(messages):
        if turn.get("role") != "user" or not turn.get("content"):
            continue
        if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant":
            return turn["content"].strip(), messages[idx + 1]["content"].strip()
    return None


def split_of(prompt):
    value = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return "test" if value < 250 else "validation" if value < 500 else "train"


def classify_answer(answer, token_count):
    if token_count <= 32:
        return "short"
    if token_count <= 96:
        return "medium"
    return "long"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--olcc", required=True)
    ap.add_argument("--max-ultrachat", type=int, default=16000)
    ap.add_argument("--max-blend", type=int, default=18000)
    ap.add_argument("--max-olcc", type=int, default=6000)
    ap.add_argument("--max-scan", type=int, default=220000)
    ap.add_argument("--max-prompt", type=int, default=250)
    ap.add_argument("--max-answer", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    outputs = {k: [] for k in ("train", "validation", "test")}
    seen = set()
    rejected = collections.Counter()
    sources = collections.Counter()
    lengths = collections.Counter()
    rng = random.Random(args.seed)

    def accept(record, pair, source):
        if pair is None:
            rejected["schema"] += 1
            return False
        prompt, answer = pair
        if not (isinstance(prompt, str) and isinstance(answer, str)):
            rejected["schema"] += 1
            return False
        prompt = re.sub(r"[ \t]+", " ", prompt.strip())
        answer = answer.strip()
        if not prompt or not answer or len(answer) > 2500 or len(prompt) > 2000:
            rejected["text_length"] += 1
            return False
        if "\x00" in prompt or "\x00" in answer:
            rejected["invalid_text"] += 1
            return False
        key = hashlib.sha256(prompt.lower().encode("utf-8")).hexdigest()
        if key in seen:
            rejected["duplicate"] += 1
            return False
        p_count = len(tok.encode(prompt, add_special_tokens=True))
        a_count = len(tok.encode(answer, add_special_tokens=False))
        if p_count > args.max_prompt or not 4 <= a_count < args.max_answer:
            rejected["token_length"] += 1
            return False
        seen.add(key)
        split = split_of(prompt.lower())
        outputs[split].append(
            {
                "prompt": prompt,
                "answer": answer,
                "source": source,
                "language": "zh" if source == "OL-CC" else "en",
                "answer_tokens": a_count,
                "length_bucket": classify_answer(answer, a_count),
            }
        )
        sources[(split, source)] += 1
        lengths[(split, classify_answer(answer, a_count))] += 1
        return True

    from itertools import islice

    with open(args.olcc, encoding="utf-8") as f:
        for line in islice(f, args.max_olcc * 3):
            if sources[("train", "OL-CC")] >= args.max_olcc:
                break
            try:
                record = json.loads(line)
            except ValueError:
                continue
            accept(record, extract_sharegpt(record), "OL-CC")

    for dataset, split, limit, fn, source in [
        (
            "HuggingFaceH4/ultrachat_200k",
            "train_sft",
            args.max_ultrachat,
            extract_ultrachat,
            "UltraChat",
        ),
        (
            "mlabonne/open-perfectblend",
            "train",
            args.max_blend,
            extract_sharegpt,
            "OpenPerfectBlend",
        ),
    ]:
        stream = load_dataset(dataset, split=split, streaming=True)
        stream = stream.shuffle(seed=args.seed, buffer_size=3000)
        for i, item in enumerate(stream):
            if sources[("train", source)] >= limit or i >= args.max_scan:
                break
            accept(item, fn(item), source)
            if i % 15000 == 0 and i:
                print("scan", source, i, "selected", sources[("train", source)], flush=True)

    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, records in outputs.items():
        rng.shuffle(records)
        path = target / f"la-{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[split] = {
            "count": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "version": 2,
        "seed": args.seed,
        "tokenizer": args.model,
        "sources": [
            {"dataset": "lorinma/BAAI_OL-CC", "license": "apache-2.0", "revision": "main"},
            {"dataset": "HuggingFaceH4/ultrachat_200k", "license": "mit", "revision": "main"},
            {
                "dataset": "mlabonne/open-perfectblend",
                "license": "apache-2.0 (repository tag; mixed origins audit required)",
                "revision": "main",
            },
        ],
        "split": summaries,
        "by_source": {f"{s}:{n}": v for (s, n), v in sources.items()},
        "by_length": {f"{s}:{n}": v for (s, n), v in lengths.items()},
        "rejected": dict(rejected),
        "max_answer_tokens": args.max_answer,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("FINISHED", json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-examples", type=int, default=1200)
    args = parser.parse_args()

    entries = {}
    for source in args.sources:
        with Path(source).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt, answer = row.get("prompt"), row.get("answer")
                if not isinstance(prompt, str) or not isinstance(answer, str):
                    raise ValueError(f"invalid teacher draft from {source}")
                native_tags = ("[Answer]", "[/Answer]", "[Question]", "[/Question]")
                if any(tag in answer for tag in native_tags):
                    raise ValueError(f"teacher marker leaked into {source}")
                length = row.get("answer_tokens")
                if not isinstance(length, int) or length < 8 or length > 190:
                    raise ValueError("invalid teacher draft length")
                entries.setdefault(prompt, row)
    rows = list(entries.values())
    if len(rows) < args.min_examples:
        raise ValueError(f"too few native-teacher drafts: {len(rows)}")
    lengths = Counter(row["answer_tokens"] for row in rows)
    short = sum(count for length, count in lengths.items() if length < 24)
    medium = sum(count for length, count in lengths.items() if 24 <= length < 64)
    long = len(rows) - short - medium
    if min(short, medium, long) < len(rows) * 0.06:
        raise ValueError(f"teacher stop-length distribution collapsed: {short}/{medium}/{long}")
    if max(lengths.values()) > len(rows) * 0.4:
        raise ValueError("teacher stop-length distribution is concentrated on one position")

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "count": len(rows),
        "length_short": short,
        "length_medium": medium,
        "length_long": long,
        "length_exact": sorted(lengths.items()),
        "language": dict(Counter(row["language"] for row in rows)),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "sources": args.sources,
    }
    dest.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TEACHER_DATA_READY", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

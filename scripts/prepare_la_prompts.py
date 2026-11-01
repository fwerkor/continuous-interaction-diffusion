from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_prompts(source: str, output: str) -> int:
    seen: set[str] = set()
    written = 0
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(source, encoding="utf-8") as inp, dest.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"].strip()
            if not prompt or prompt in seen:
                continue
            seen.add(prompt)
            out.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "source": row.get("source", ""),
                        "language": row.get("language", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Export prompts only; discard annotated answers.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print("prompts_written", export_prompts(args.input, args.output))


if __name__ == "__main__":
    main()

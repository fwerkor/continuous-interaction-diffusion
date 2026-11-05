from __future__ import annotations

import argparse
import json
from pathlib import Path

UNKNOWN = "<|cid_unknown|>"
MARKER = "[/Answer]"


def validate_row(row: dict) -> list[str]:
    errors: list[str] = []
    metadata = row.get("metadata") or {}
    final = row["target_display"]
    complete = UNKNOWN not in final or any(
        UNKNOWN not in target["text"] for target in row.get("display_targets", ())
    )
    if not complete:
        return errors
    steps = metadata.get("native_full_display_by_step", {})
    if (
        metadata.get("native_full_display_verified") is not True
        or not isinstance(metadata.get("native_full_display_reviewer"), str)
        or not metadata["native_full_display_reviewer"].strip()
    ):
        errors.append("missing independent verification and reviewer")
    if not isinstance(steps, dict):
        return ["native_full_display_by_step must be an object"]

    def check(answer: str, response: object, label: str) -> None:
        if UNKNOWN in answer:
            return
        if not isinstance(response, str) or not response.strip():
            errors.append(f"{label}: missing native full reply")
            return
        if UNKNOWN in response or MARKER in response:
            errors.append(f"{label}: leaked internal token")
        if (answer.strip().casefold() not in response.casefold()
                and metadata.get("native_full_display_semantic_paraphrase") is not True):
            errors.append(f"{label}: reference answer absent and not verified")

    check(final, metadata.get("native_full_display"), "final")
    for target in row.get("display_targets", ()):
        step = str(target["step"])
        answer = target["text"]
        response = steps.get(step)
        if response is None and answer == final:
            response = metadata.get("native_full_display")
        check(answer, response, f"step={step}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed before training LFM-native on CID answer fragments"
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--show-errors", type=int, default=8)
    args = parser.parse_args()
    total = invalid = 0
    examples = []
    with args.data.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            errors = validate_row(row)
            if errors:
                invalid += 1
                if len(examples) < args.show_errors:
                    examples.append({"example_id": row.get("example_id"), "errors": errors[:4]})
            if args.max_examples is not None and total >= args.max_examples:
                break
    print(json.dumps({
        "examples_checked": total,
        "invalid_examples": invalid,
        "valid_examples": total - invalid,
        "sample_errors": examples,
    }, ensure_ascii=False, indent=2))
    return 0 if total and invalid == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

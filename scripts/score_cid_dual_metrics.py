"""Two complementary CID task metrics.

Task Accuracy requires an unambiguously correct answer, allowing spelling,
formatting, and reference aliases. Ambiguous outputs are NOT automatically
counted; they require semantic adjudication.

Answer Discovery Rate counts an answer mentioned anywhere in the output,
including Known: regions when the final answer is wrong.

These are CID-specific metrics, not replacements for the upstream datasets'
canonical strict EM, pass@1, or choice accuracies.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from score_public_benchmark_results import (
    _math_equal,
    _mbpp_passes,
    _normalize_qa,
    _numeric_equal,
    _qa_metrics,
)

CHOICE_DATASETS = {"arc-challenge-test", "mmlu-test"}
QA_DATASETS = {
    "hotpotqa-distractor-validation",
    "2wikimultihopqa-dev",
    "musique-validation",
}
MATH_DATASETS = {"gsm8k-test", "math-test"}
NUMERIC_RE = re.compile(
    r"(?<![\w./{}\\^+-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w./{}\\^+-])"
)
ANSWER_RE = re.compile(
    r"(?i)\b(?:final\s+answer|answer|result|conclusion)"
    r"\s*[:=]\s*([^\n;]+)"
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.replace("−", "-").replace("–", "-")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[-+]?\d+(?:\.\d+)?|[a-z]+", _fold(text)))


def _contains_phrase(text: str, ref: str) -> bool:
    wanted = _tokens(ref)
    tokens = _tokens(text)
    if not wanted or not tokens:
        return False
    return any(tokens[i:i + len(wanted)] == wanted
               for i in range(len(tokens) - len(wanted) + 1))


def _references(record: dict) -> list[str]:
    ref = str(record.get("target_display", "")).strip()
    aliases = record.get("metadata", {}).get("answer_aliases", ())
    if not isinstance(aliases, (list, tuple)):
        aliases = ()
    return list(dict.fromkeys([ref] + [
        value.strip() for value in aliases if isinstance(value, str) and value.strip()
    ]))


def _number_mentioned(output: str, reference: str) -> bool:
    target = reference.strip()
    if not re.fullmatch(r"[-+]?\d+(?:,\d+)*(?:\.\d+)?", target):
        return False
    return any(_numeric_equal(match.group(), target)
               for match in NUMERIC_RE.finditer(output))


def _math_mentioned(output: str, reference: str) -> bool:
    ref = reference.strip()
    if _math_equal(output, ref):
        return True
    if _number_mentioned(output, ref):
        return True
    if not ref or ref in ("0", "1", "-1"):
        return False
    if re.search(r"[0-9]", ref) and len(ref) < 4:
        return False
    return bool(re.search(r"(?<![a-zA-Z0-9_])" + re.escape(ref) +
                          r"(?![a-zA-Z0-9_])", output, flags=re.IGNORECASE))


def _choice_index(output: str, record: dict) -> int | None:
    metadata = record.get("metadata", {})
    choices = metadata.get("choices", ())
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    labels = metadata.get("choice_labels")
    if not isinstance(labels, (list, tuple)) or len(labels) != len(choices):
        labels = tuple(chr(ord("A") + i) for i in range(len(choices)))

    output = output.strip()
    # Prefer an explicitly selected option over a mentioned/quoted option.
    # A selection must be explicit when the output mentions multiple choices.
    selection = re.search(
        r"(?i)\b(?:final\s+answer|answer|choice|option|select|pick|choose)"
        r"\s*(?:is|:|=|-)?\s*([A-D])\b",
        output,
    )
    if selection:
        letter = selection.group(1).upper()
        found = [i for i, label in enumerate(labels)
                 if str(label).upper() == letter]
        if len(found) == 1:
            return found[0]

    for i, (choice, label) in enumerate(zip(choices, labels, strict=False)):
        if (_normalize_qa(output) == _normalize_qa(str(choice))
                or output.casefold() == str(label).casefold()):
            return i

    # An explicitly marked final answer may include an entire option's text,
    # but a Known: mention by itself does not constitute a selected answer.
    answers = ANSWER_RE.findall(output)
    if answers:
        candidate = answers[-1].strip().strip(" .")
        hits = [
            i for i, choice in enumerate(choices)
            if _normalize_qa(candidate) == _normalize_qa(str(choice))
        ]
        if len(hits) == 1:
            return hits[0]
    return None



def _qa_discovery(output: str, refs: list[str]) -> bool:
    return any(_contains_phrase(output, ref) for ref in refs)


def _qa_task_accuracy(output: str, refs: list[str]) -> tuple[bool, bool]:
    # The second return value identifies cases in need of semantic adjudication.
    exact, _ = _qa_metrics(output, refs)
    if exact:
        return True, False

    answers = ANSWER_RE.findall(output)
    if answers:
        # Explicit final-answer spans supersede facts in Known:.
        final = answers[-1].strip().strip(". ")
        good = any(_normalize_qa(final) == _normalize_qa(ref) for ref in refs)
        return good, not good

    # "Yes, ..." / "No, ..." can be a valid natural-language final answer,
    # unless it asserts both or retracts one of them.
    if any(_normalize_qa(ref) in {"yes", "no"} for ref in refs):
        words = _tokens(output)
        if words and words[0] in {"yes", "no"}:
            if any(token in {"yes", "no"} and token != words[0] for token in words[1:]):
                return False, True
            return words[0] in {_normalize_qa(ref) for ref in refs}, False

    stripped = output.strip()
    # Explicitly framed answers; no credit for "Known: X" alone.
    for prefix in ("the answer is ", "it was ", "it is ", "the correct answer is "):
        if _fold(stripped).startswith(prefix):
            candidate = stripped[len(prefix):].strip().rstrip(".")
            if any(_normalize_qa(candidate) == _normalize_qa(ref) for ref in refs):
                return True, False

    return False, _qa_discovery(output, refs)


def score_record(dataset: str, record: dict) -> dict:
    output = str(record.get("final_text", ""))
    target = str(record.get("target_display", ""))
    refs = _references(record)
    accepted = False
    discovered = False
    requires_review = False

    if dataset in CHOICE_DATASETS:
        selected = _choice_index(output, record)
        right = record.get("metadata", {}).get("answer_index")
        accepted = selected is not None and selected == right
        discovered = accepted or _qa_discovery(output, refs)
        # If all choices are copied, do not count that as discovering an answer.
        choices = record.get("metadata", {}).get("choices", [])
        if selected is None and sum(_contains_phrase(output, str(x)) for x in choices) > 1:
            discovered = False
        requires_review = not accepted and discovered
    elif dataset == "gsm8k-test":
        accepted = _numeric_equal(output, target)
        discovered = accepted or _number_mentioned(output, target)
        requires_review = discovered and not accepted
    elif dataset == "math-test":
        accepted = _math_equal(output, target)
        discovered = accepted or _math_mentioned(output, target)
        requires_review = discovered and not accepted
    elif dataset == "mbpp-test":
        standard = record.get("standard_metrics", {})
        if "pass_at_1" in standard:
            accepted = bool(standard["pass_at_1"])
        else:
            accepted, _ = _mbpp_passes(record)
        discovered = accepted
    elif dataset in QA_DATASETS:
        accepted, requires_review = _qa_task_accuracy(output, refs)
        discovered = accepted or _qa_discovery(output, refs)
        requires_review = requires_review or (discovered and not accepted)
    else:
        raise ValueError(f"Unknown public dataset: {dataset}")

    if accepted:
        discovered = True
    return {
        "task_accuracy": bool(accepted),
        "answer_discovery": bool(discovered),
        "needs_semantic_review": bool(requires_review),
    }


def score_dataset(dataset_dir: Path) -> dict:
    dataset = dataset_dir.name
    shards = sorted(dataset_dir.glob("shard-*/results.jsonl"))
    if len(shards) != 16 or len(list(dataset_dir.glob("shard-*/summary.json"))) != 16:
        raise ValueError(f"Incomplete benchmark: {dataset_dir}: {len(shards)}/16")
    counts: Counter[str] = Counter()
    audit_path = dataset_dir / "cid-dual-metrics-audit.jsonl"
    # Only ambiguous samples go to small audit JSONL, not a copy of every trajectory.
    with audit_path.open("w") as out:
        for shard in shards:
            with shard.open(encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    scored = score_record(dataset, record)
                    counts["examples"] += 1
                    for key, value in scored.items():
                        counts[key] += int(value)
                    if scored["needs_semantic_review"]:
                        out.write(json.dumps({
                            "example_id": record.get("example_id"),
                            "target": record.get("target_display"),
                            "prediction": record.get("final_text"),
                            **scored,
                        }, ensure_ascii=False) + "\n")
    examples = counts["examples"]
    if examples != json.loads((dataset_dir / "summary.json").read_text())["examples"]:
        raise ValueError(f"Dataset size mismatch: {dataset}")
    result = {
        "dataset": dataset,
        **dict(counts),
        "task_accuracy_rate": counts["task_accuracy"] / examples,
        "answer_discovery_rate": counts["answer_discovery"] / examples,
        "review_pending": counts["needs_semantic_review"],
        "rule_version": "cid-dual-v1",
        "task_accuracy_definition": (
            "Unambiguous semantically equivalent final answer; conservative deterministic "
            "matcher; ambiguous examples require review."
        ),
        "answer_discovery_definition": (
            "Correct answer generated anywhere, including Known: and even if an explicit "
            "final answer is wrong."
        ),
    }
    (dataset_dir / "cid-dual-metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.dataset_dirs:
        data = score_dataset(folder)
        print(json.dumps(data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

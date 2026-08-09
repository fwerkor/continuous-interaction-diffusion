from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class PublicTaskRowRejected(ValueError):
    """An upstream row is well-formed but unsuitable for the normalized task pool."""


@dataclass(frozen=True, slots=True)
class PublicTask:
    task_id: str
    semantic_id: str
    split: str
    task_kind: str
    prompt: str
    reference_answer: str
    source: Mapping[str, Any]
    resources: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_public_task_pool(
    registry_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Build the pinned public semantic-task pool described by ``registry_path``.

    Network/dataframe dependencies are imported lazily so the normal CID runtime stays dependency
    free. Every upstream file is downloaded at an exact dataset revision from the registry.
    """

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised by the optional data environment
        raise RuntimeError(
            "public task-pool construction requires the optional data dependencies; "
            "install with `pip install -e '.[data]'`"
        ) from exc

    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    _validate_registry(registry)
    seed = int(registry["seed"])
    split_spec = registry["internal_split"]

    tasks: list[PublicTask] = []
    seen_semantic_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    dropped_duplicates = 0
    dropped_invalid_rows = 0

    for source in registry["sources"]:
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for filename in source["files"]:
            local_path = hf_hub_download(
                repo_id=source["repo"],
                repo_type="dataset",
                filename=filename,
                revision=source["revision"],
            )
            frame = pd.read_parquet(local_path)
            for row_number, row in enumerate(frame.to_dict(orient="records")):
                row_key = f"{filename}:{row_number}"
                rows.append((row_key, _jsonable(row)))

        candidates = []
        for row_key, row in rows:
            try:
                prompt, answer, resources, metadata = _adapt_row(str(source["adapter"]), row)
            except PublicTaskRowRejected:
                dropped_invalid_rows += 1
                continue
            semantic_id = _semantic_id(prompt)
            score = _stable_hash(f"{seed}|{source['id']}|{row_key}|{semantic_id}")
            candidates.append(
                (
                    score,
                    row_key,
                    semantic_id,
                    prompt,
                    answer,
                    resources,
                    metadata,
                )
            )
        candidates.sort(key=lambda item: item[0])

        quota = int(source["quota"])
        accepted = 0
        for _, row_key, semantic_id, prompt, answer, resources, metadata in candidates:
            if semantic_id in seen_semantic_ids:
                dropped_duplicates += 1
                continue
            split = deterministic_split(semantic_id, split_spec)
            task_id = f"pub-{source['id']}-{_stable_hash(row_key + '|' + semantic_id)[:16]}"
            tasks.append(
                PublicTask(
                    task_id=task_id,
                    semantic_id=semantic_id,
                    split=split,
                    task_kind=str(source["task_kind"]),
                    prompt=prompt,
                    reference_answer=answer,
                    source={
                        "dataset_id": source["id"],
                        "repo": source["repo"],
                        "revision": source["revision"],
                        "license": source["license"],
                        "upstream_config": source["upstream_config"],
                        "upstream_split": source["upstream_split"],
                        "row_key": row_key,
                        "use": source["use"],
                    },
                    resources=resources,
                    metadata=metadata,
                )
            )
            seen_semantic_ids.add(semantic_id)
            source_counts[str(source["id"])] += 1
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise ValueError(
                f"source {source['id']!r} produced only {accepted} unique tasks; quota={quota}"
            )

    target_tasks = int(registry["target_tasks"])
    if len(tasks) != target_tasks:
        raise ValueError(f"registry target_tasks={target_tasks}, built={len(tasks)}")

    tasks.sort(key=lambda item: item.task_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    split_counts = Counter(task.split for task in tasks)
    kind_counts = Counter(task.task_kind for task in tasks)
    license_counts = Counter(str(task.source["license"]) for task in tasks)
    manifest = {
        "format_version": 1,
        "pool_name": registry["pool_name"],
        "tasks": len(tasks),
        "sha256": _file_sha256(output),
        "registry_sha256": _file_sha256(registry_file),
        "seed": seed,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "task_kind_counts": dict(sorted(kind_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "dropped_semantic_duplicates": dropped_duplicates,
        "dropped_invalid_rows": dropped_invalid_rows,
    }
    expected_sha256 = registry.get("expected_sha256")
    if expected_sha256 is not None and manifest["sha256"] != expected_sha256:
        raise ValueError(
            "public task-pool SHA-256 drifted from the pinned registry expectation: "
            f"{manifest['sha256']} != {expected_sha256}"
        )
    expected_split_counts = registry.get("expected_split_counts")
    if expected_split_counts is not None and manifest["split_counts"] != expected_split_counts:
        raise ValueError(
            "public task-pool split counts drifted from the pinned registry expectation"
        )
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def deterministic_split(semantic_id: str, split_spec: Mapping[str, Any]) -> str:
    train = float(split_spec["train"])
    validation = float(split_spec["validation"])
    test = float(split_spec["test"])
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("public task split fractions must sum to 1")
    unit = int(_stable_hash("split|" + semantic_id)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if unit < train:
        return "train"
    if unit < train + validation:
        return "validation"
    return "test"


def _adapt_row(
    adapter: str,
    row: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    if adapter == "gsm8k":
        full_answer = str(row["answer"])
        final = full_answer.rsplit("####", maxsplit=1)[-1].strip()
        return str(row["question"]).strip(), final, {}, {"reference_solution": full_answer}

    if adapter == "hendrycks_math":
        solution = str(row["solution"])
        boxed = _last_boxed_value(solution)
        if not boxed:
            raise PublicTaskRowRejected("MATH row has no recoverable boxed final answer")
        return (
            str(row["problem"]).strip(),
            boxed,
            {},
            {
                "reference_solution": solution,
                "level": str(row.get("level", "")),
                "subject": str(row.get("type", "")),
            },
        )

    if adapter == "mmlu":
        choices = [str(item) for item in row["choices"]]
        answer_index = int(row["answer"])
        prompt = _multiple_choice_prompt(str(row["question"]), choices)
        return (
            prompt,
            choices[answer_index],
            {},
            {
                "choices": choices,
                "answer_index": answer_index,
                "subject": str(row.get("subject", "")),
            },
        )

    if adapter == "arc":
        raw_choices = row["choices"]
        labels = [str(item) for item in raw_choices["label"]]
        texts = [str(item) for item in raw_choices["text"]]
        answer_label = str(row["answerKey"])
        if answer_label not in labels:
            raise ValueError(f"ARC answer label {answer_label!r} missing from choices")
        answer_index = labels.index(answer_label)
        prompt = _labeled_choice_prompt(str(row["question"]), labels, texts)
        return (
            prompt,
            texts[answer_index],
            {},
            {
                "upstream_id": str(row.get("id", "")),
                "choice_labels": labels,
                "choices": texts,
                "answer_label": answer_label,
                "answer_index": answer_index,
            },
        )

    if adapter == "mbpp":
        tests = [str(item) for item in row.get("test_list", [])]
        challenge = [str(item) for item in row.get("challenge_test_list", [])]
        return (
            str(row["text"]).strip(),
            str(row["code"]).strip(),
            {},
            {
                "upstream_task_id": int(row["task_id"]),
                "tests": tests,
                "challenge_tests": challenge,
                "test_setup_code": str(row.get("test_setup_code", "")),
            },
        )

    if adapter == "hotpotqa":
        context = row["context"]
        titles = [str(item) for item in context["title"]]
        sentence_groups = [
            [str(sentence) for sentence in group] for group in context["sentences"]
        ]
        evidence_bank = [
            {"title": title, "sentences": sentences}
            for title, sentences in zip(titles, sentence_groups, strict=True)
        ]
        supporting = row.get("supporting_facts", {})
        return (
            str(row["question"]).strip(),
            str(row["answer"]).strip(),
            {"evidence_bank": evidence_bank},
            {
                "upstream_id": str(row.get("id", "")),
                "question_type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
                "supporting_facts": supporting,
            },
        )

    if adapter == "2wikimultihopqa":
        context = _json_field(row["context"], "2Wiki context")
        supporting = _json_field(row["supporting_facts"], "2Wiki supporting_facts")
        evidences = _json_field(row.get("evidences", "[]"), "2Wiki evidences")
        evidence_bank = [
            {
                "title": str(item[0]),
                "sentences": [str(sentence) for sentence in item[1]],
            }
            for item in context
        ]
        return (
            str(row["question"]).strip(),
            str(row["answer"]).strip(),
            {"evidence_bank": evidence_bank},
            {
                "upstream_id": str(row.get("_id", "")),
                "question_type": str(row.get("type", "")),
                "supporting_facts": {
                    "title": [str(item[0]) for item in supporting],
                    "sent_id": [int(item[1]) for item in supporting],
                },
                "reasoning_evidences": evidences,
            },
        )

    if adapter == "musique":
        if not bool(row.get("answerable", True)):
            raise PublicTaskRowRejected("MuSiQue row is marked unanswerable")
        paragraphs = list(row.get("paragraphs", ()))
        evidence_bank = [
            {
                "title": str(item["title"]),
                "sentences": [str(item["paragraph_text"])],
            }
            for item in paragraphs
        ]
        supporting = [item for item in paragraphs if bool(item.get("is_supporting", False))]
        if not supporting:
            raise PublicTaskRowRejected("MuSiQue row has no supporting paragraphs")
        if len({str(item["title"]) for item in supporting}) < 2:
            raise PublicTaskRowRejected("MuSiQue row does not span multiple supporting documents")
        decomposition = [dict(item) for item in row.get("question_decomposition", ())]
        return (
            str(row["question"]).strip(),
            str(row["answer"]).strip(),
            {"evidence_bank": evidence_bank},
            {
                "upstream_id": str(row.get("id", "")),
                "supporting_facts": {
                    "title": [str(item["title"]) for item in supporting],
                    "sent_id": [0 for _ in supporting],
                },
                "hop_count": len(decomposition),
                "question_decomposition": decomposition,
                "answer_aliases": [str(item) for item in row.get("answer_aliases", ())],
            },
        )

    raise ValueError(f"unsupported public dataset adapter: {adapter}")


def _multiple_choice_prompt(question: str, choices: Sequence[str]) -> str:
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    return _labeled_choice_prompt(question, labels, choices)


def _labeled_choice_prompt(question: str, labels: Sequence[str], choices: Sequence[str]) -> str:
    lines = [question.strip()]
    lines.extend(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    return "\n".join(lines)


def _json_field(value: Any, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise PublicTaskRowRejected(f"invalid {label} JSON") from exc
    return value


def _last_boxed_value(text: str) -> str | None:
    markers = ("\\boxed", "\\fbox")
    starts = [(text.rfind(marker), marker) for marker in markers]
    start, marker = max(starts, key=lambda item: item[0])
    if start < 0:
        return None
    cursor = start + len(marker)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return None
    if text[cursor] != "{":
        end = cursor
        while end < len(text) and text[end] not in "$.,;\n":
            end += 1
        value = text[cursor:end].strip()
        return value or None
    cursor += 1
    depth = 1
    content_start = cursor
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = text[content_start:cursor].strip()
                return value or None
        cursor += 1
    return None


def _semantic_id(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    return _stable_hash(normalized)


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _validate_registry(registry: Mapping[str, Any]) -> None:
    if int(registry.get("version", 0)) != 1:
        raise ValueError("unsupported public dataset registry version")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("public dataset registry must contain sources")
    source_ids = [str(source["id"]) for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("public dataset source IDs must be unique")
    quota_sum = sum(int(source["quota"]) for source in sources)
    if quota_sum != int(registry["target_tasks"]):
        raise ValueError("public dataset quotas must sum to target_tasks")

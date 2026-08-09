from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.causal_distill import dump_causal_teacher_jobs
from cid.distill import TeacherEvidence, TeacherTask, dump_teacher_requests, dump_teacher_tasks


@dataclass(frozen=True, slots=True)
class PublicTrainingConfig:
    split: str = "train"
    seed: int = 20260809
    unnecessary_tool_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("public training split must be train, validation, or test")
        if not 0.0 <= self.unnecessary_tool_fraction <= 1.0:
            raise ValueError("unnecessary_tool_fraction must be in [0, 1]")


def prepare_public_distillation(
    pool_path: str | Path,
    tasks_output: str | Path,
    requests_output: str | Path,
    manifest_output: str | Path,
    config: PublicTrainingConfig | None = None,
    causal_jobs_output: str | Path | None = None,
) -> dict[str, Any]:
    config = config or PublicTrainingConfig()
    pool_file = Path(pool_path)
    records = _load_public_records(pool_file)
    selected = [record for record in records if str(record["split"]) == config.split]

    tasks: list[TeacherTask] = []
    mode_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    evidence_counts: Counter[int] = Counter()
    for record in selected:
        task, mode = _teacher_task_from_public(record, config)
        tasks.append(task)
        mode_counts[mode] += 1
        source_counts[str(record["source"]["dataset_id"])] += 1
        evidence_counts[len(task.evidence)] += 1

    tasks.sort(key=lambda item: item.task_id)
    task_path = Path(tasks_output)
    request_path = Path(requests_output)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    dump_teacher_tasks(tasks, task_path)
    dump_teacher_requests(tasks, request_path)
    causal_path = None if causal_jobs_output is None else Path(causal_jobs_output)
    if causal_path is not None:
        dump_causal_teacher_jobs(tuple(tasks), causal_path)

    manifest = {
        "format_version": 1,
        "split": config.split,
        "seed": config.seed,
        "unnecessary_tool_fraction": config.unnecessary_tool_fraction,
        "input_pool": str(pool_file),
        "input_pool_sha256": _file_sha256(pool_file),
        "tasks": len(tasks),
        "tasks_sha256": _file_sha256(task_path),
        "requests_sha256": _file_sha256(request_path),
        "mode_counts": dict(sorted(mode_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "evidence_count_histogram": {
            str(count): frequency for count, frequency in sorted(evidence_counts.items())
        },
    }
    if causal_path is not None:
        manifest["causal_jobs"] = str(causal_path)
        manifest["causal_jobs_sha256"] = _file_sha256(causal_path)
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _teacher_task_from_public(
    record: Mapping[str, Any],
    config: PublicTrainingConfig,
) -> tuple[TeacherTask, str]:
    source = record["source"]
    dataset_id = str(source["dataset_id"])
    metadata = {
        "semantic_id": str(record["semantic_id"]),
        "split": str(record["split"]),
        "task_kind": str(record["task_kind"]),
        "public_dataset_id": dataset_id,
        "upstream_repo": str(source["repo"]),
        "upstream_revision": str(source["revision"]),
        "upstream_config": str(source["upstream_config"]),
        "upstream_split": str(source["upstream_split"]),
        "upstream_row_key": str(source["row_key"]),
        "license": str(source["license"]),
    }
    expected_answer = str(record["reference_answer"])

    if str(source.get("use", "")) == "toolizable_retrieval":
        descriptors, evidence = _retrieval_tool_environment(record)
        return (
            TeacherTask(
                task_id=str(record["task_id"]),
                prompt=str(record["prompt"]),
                source_descriptors=descriptors,
                evidence=evidence,
                metadata={**metadata, "training_mode": "tool_required"},
                reference_answer=expected_answer,
            ),
            "tool_required",
        )

    if _use_unnecessary_tools(str(record["semantic_id"]), config):
        return (
            TeacherTask(
                task_id=str(record["task_id"]),
                prompt=str(record["prompt"]),
                source_descriptors=_workspace_descriptors(),
                metadata={**metadata, "training_mode": "tools_available_unnecessary"},
                reference_answer=expected_answer,
            ),
            "tools_available_unnecessary",
        )

    return (
        TeacherTask(
            task_id=str(record["task_id"]),
            prompt=str(record["prompt"]),
            metadata={**metadata, "training_mode": "no_tool"},
            reference_answer=expected_answer,
        ),
        "no_tool",
    )


def _retrieval_tool_environment(
    record: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[TeacherEvidence, ...]]:
    resources = record.get("resources", {})
    bank = list(resources.get("evidence_bank", ()))
    if not bank:
        raise ValueError(f"retrieval task {record['task_id']} has no evidence bank")

    metadata = record.get("metadata", {})
    supporting = metadata.get("supporting_facts", {})
    supporting_titles = [str(title) for title in supporting.get("title", ())]
    if not supporting_titles:
        raise ValueError(f"retrieval task {record['task_id']} has no supporting facts")
    unique_supporting_titles = tuple(dict.fromkeys(supporting_titles))

    resource_by_title: dict[str, tuple[str, Mapping[str, Any]]] = {}
    candidates: list[dict[str, str]] = []
    for index, resource in enumerate(bank):
        resource_id = f"doc-{index:02d}"
        title = str(resource["title"])
        resource_by_title[title] = (resource_id, resource)
        candidates.append({"resource_id": resource_id, "title": title})

    missing = [title for title in unique_supporting_titles if title not in resource_by_title]
    if missing:
        raise ValueError(
            f"retrieval task {record['task_id']} supporting titles missing from "
            f"evidence bank: {missing}"
        )

    evidence: list[TeacherEvidence] = [
        TeacherEvidence(
            evidence_id="search-results",
            source="workspace_search",
            value=candidates,
            arguments={"query": str(record["prompt"])},
            provenance=str(record["source"]["row_key"]),
        )
    ]
    for index, title in enumerate(unique_supporting_titles):
        resource_id, resource = resource_by_title[title]
        sentences = [str(sentence) for sentence in resource.get("sentences", ())]
        evidence.append(
            TeacherEvidence(
                evidence_id=f"support-{index}",
                source="workspace_read",
                value={"resource_id": resource_id, "title": title, "sentences": sentences},
                arguments={"resource_id": resource_id},
                depends_on=("search-results",),
                provenance=f"{record['source']['row_key']}#{title}",
            )
        )
    return _workspace_descriptors(), tuple(evidence)


def _workspace_descriptors() -> tuple[Mapping[str, Any], ...]:
    return (
        {
            "name": "workspace_search",
            "description": (
                "Search a task-local hidden document workspace and return candidate resource IDs "
                "and titles. The workspace may be irrelevant to the task."
            ),
            "arguments": (
                {"name": "query", "kind": "string", "required": True},
            ),
            "cacheable": True,
            "dynamic": False,
            "versioned": False,
        },
        {
            "name": "workspace_read",
            "description": "Read one task-local hidden resource by resource_id.",
            "arguments": (
                {"name": "resource_id", "kind": "string", "required": True},
            ),
            "cacheable": True,
            "dynamic": False,
            "versioned": False,
        },
    )


def _use_unnecessary_tools(semantic_id: str, config: PublicTrainingConfig) -> bool:
    digest = hashlib.sha256(
        f"{config.seed}|unnecessary-tool|{semantic_id}".encode()
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    return unit < config.unnecessary_tool_fraction


def _load_public_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid public task JSON at line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"public task line {line_number} must be a JSON object")
            required = {
                "task_id",
                "semantic_id",
                "split",
                "task_kind",
                "prompt",
                "reference_answer",
                "source",
                "resources",
                "metadata",
            }
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"public task line {line_number} missing fields: {missing}")
            records.append(raw)
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

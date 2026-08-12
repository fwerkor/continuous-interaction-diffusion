from __future__ import annotations

import hashlib
import json
import re
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
    interaction_patterns: Counter[str] = Counter()
    dependency_depths: Counter[int] = Counter()
    for record in selected:
        task, mode = _teacher_task_from_public(record, config)
        tasks.append(task)
        mode_counts[mode] += 1
        source_counts[str(record["source"]["dataset_id"])] += 1
        evidence_counts[len(task.evidence)] += 1
        pattern = str(task.metadata.get("interaction_pattern", "none"))
        interaction_patterns[pattern] += 1
        depth = int(task.metadata.get("dependency_depth", 0))
        dependency_depths[depth] += 1

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
        "python_public_test_tasks": sum(
            1
            for task in tasks
            if str(task.metadata.get("task_kind", "")) == "python_programming"
            and bool(task.metadata.get("public_tests"))
        ),
        "mode_counts": dict(sorted(mode_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "interaction_pattern_counts": dict(sorted(interaction_patterns.items())),
        "dependency_depth_histogram": {
            str(depth): frequency for depth, frequency in sorted(dependency_depths.items())
        },
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
    if str(record.get("task_kind", "")) == "python_programming":
        task_metadata = record.get("metadata", {})
        metadata["public_tests"] = [
            str(item).strip()
            for item in task_metadata.get("tests", ())
            if str(item).strip()
        ]
        metadata["public_test_setup_code"] = str(
            task_metadata.get("test_setup_code", "")
        ).strip()

    expected_answer = str(record["reference_answer"])
    prompt = _teacher_visible_prompt(record)

    if str(source.get("use", "")) == "toolizable_retrieval":
        if dataset_id == "musique-train":
            descriptors, evidence, interaction_metadata = _musique_tool_environment(record)
        else:
            descriptors, evidence = _retrieval_tool_environment(record)
            interaction_metadata = {
                "interaction_pattern": "search_then_parallel_reads",
                "dependency_depth": 2,
            }
        return (
            TeacherTask(
                task_id=str(record["task_id"]),
                prompt=prompt,
                source_descriptors=descriptors,
                evidence=evidence,
                metadata={
                    **metadata,
                    **interaction_metadata,
                    "training_mode": "tool_required",
                },
                reference_answer=expected_answer,
            ),
            "tool_required",
        )

    if _use_unnecessary_tools(str(record["semantic_id"]), config):
        return (
            TeacherTask(
                task_id=str(record["task_id"]),
                prompt=prompt,
                source_descriptors=_workspace_descriptors(),
                metadata={**metadata, "training_mode": "tools_available_unnecessary"},
                reference_answer=expected_answer,
            ),
            "tools_available_unnecessary",
        )

    return (
        TeacherTask(
            task_id=str(record["task_id"]),
            prompt=prompt,
            metadata={**metadata, "training_mode": "no_tool"},
            reference_answer=expected_answer,
        ),
        "no_tool",
    )


def _teacher_visible_prompt(record: Mapping[str, Any]) -> str:
    """Add public interface tests to programming prompts without leaking hidden answers."""

    prompt = str(record["prompt"]).strip()
    if str(record.get("task_kind", "")) != "python_programming":
        return prompt

    task_metadata = record.get("metadata", {})
    tests = [
        str(item).strip()
        for item in task_metadata.get("tests", ())
        if str(item).strip()
    ]
    setup = str(task_metadata.get("test_setup_code", "")).strip()
    if not tests and not setup:
        return prompt

    sections = [prompt, "Public evaluation contract:"]
    if setup:
        sections.extend(("Setup:", setup))
    if tests:
        sections.append("Tests:")
        sections.extend(tests)
    return "\n".join(sections)


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


_MUSIQUE_REFERENCE = re.compile(r"#(\d+)")


def _musique_tool_environment(
    record: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[TeacherEvidence, ...],
    dict[str, Any],
]:
    """Turn MuSiQue's native decomposition into a causal retrieval DAG.

    Each decomposition node gets its own search/read pair. A search is unlocked only after the
    supporting reads referenced by ``#N`` placeholders have arrived. The query argument is then
    resolved with the corresponding upstream intermediate answer, so supervision teaches the
    student to form a new tool argument from newly assimilated evidence instead of launching every
    supporting read immediately after the first workspace search.
    """

    resources = record.get("resources", {})
    bank = list(resources.get("evidence_bank", ()))
    decomposition = [dict(item) for item in record.get("metadata", {}).get(
        "question_decomposition", ()
    )]
    if not bank or not decomposition:
        raise ValueError(f"MuSiQue task {record['task_id']} lacks decomposition resources")

    candidates = [
        {"resource_id": f"doc-{index:02d}", "title": str(resource["title"])}
        for index, resource in enumerate(bank)
    ]
    evidence: list[TeacherEvidence] = []
    node_depths: list[int] = []
    root_nodes = 0
    for index, node in enumerate(decomposition):
        question = str(node.get("question", "")).strip()
        if not question:
            raise ValueError(
                f"MuSiQue task {record['task_id']} has an empty decomposition question"
            )
        # MuSiQue also contains literal strings such as "#9 Dream". Only #N values that point to
        # an already-defined decomposition node are dependency placeholders.
        references = tuple(
            dict.fromkeys(
                value - 1
                for value in (int(item) for item in _MUSIQUE_REFERENCE.findall(question))
                if 1 <= value <= index
            )
        )
        search_dependencies = tuple(f"read-hop-{reference}" for reference in references)
        if not references:
            root_nodes += 1
            node_depth = 1
        else:
            node_depth = 1 + max(node_depths[reference] for reference in references)
        node_depths.append(node_depth)

        resolved_query = question
        for reference in references:
            answer = str(decomposition[reference].get("answer", "")).strip()
            if not answer:
                raise ValueError(
                    f"MuSiQue task {record['task_id']} has an empty intermediate answer"
                )
            resolved_query = resolved_query.replace(f"#{reference + 1}", answer)
        unresolved = [
            int(item)
            for item in _MUSIQUE_REFERENCE.findall(resolved_query)
            if 1 <= int(item) <= index
        ]
        if unresolved:
            raise ValueError(
                f"MuSiQue task {record['task_id']} has unresolved decomposition references"
            )

        support_index = int(node["paragraph_support_idx"])
        if not 0 <= support_index < len(bank):
            raise ValueError(
                f"MuSiQue task {record['task_id']} support index {support_index} is out of range"
            )
        resource = bank[support_index]
        resource_id = f"doc-{support_index:02d}"
        search_id = f"search-hop-{index}"
        read_id = f"read-hop-{index}"
        provenance = str(record["source"]["row_key"])
        evidence.append(
            TeacherEvidence(
                evidence_id=search_id,
                source="workspace_search",
                value={"query": resolved_query, "candidates": candidates},
                arguments={"query": resolved_query},
                depends_on=search_dependencies,
                provenance=f"{provenance}#decomposition-{index}:search",
            )
        )
        evidence.append(
            TeacherEvidence(
                evidence_id=read_id,
                source="workspace_read",
                value={
                    "resource_id": resource_id,
                    "title": str(resource["title"]),
                    "sentences": [str(sentence) for sentence in resource.get("sentences", ())],
                },
                arguments={"resource_id": resource_id},
                depends_on=(search_id,),
                provenance=f"{provenance}#decomposition-{index}:read",
            )
        )

    depth = max(node_depths)
    return (
        _workspace_descriptors(),
        tuple(evidence),
        {
            "interaction_pattern": "decomposition_dag",
            "dependency_depth": depth,
            "decomposition_hops": len(decomposition),
            "parallel_root_needs": root_nodes,
        },
    )


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

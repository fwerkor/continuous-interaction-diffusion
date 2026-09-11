from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cid.data import BindingTarget, ExternalEvent, TrajectoryExample, dump_jsonl
from cid.public_tasks import PublicTaskRowRejected, _adapt_row, _jsonable, _semantic_id
from cid.public_training import PublicTrainingConfig, _teacher_task_from_public

SOURCES = (
    {
        "id": "gsm8k-test",
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "files": ("main/test-00000-of-00001.parquet",),
        "adapter": "gsm8k",
        "task_kind": "math_word_problem",
        "upstream_config": "main",
        "upstream_split": "test",
        "use": "general_reasoning",
        "license": "MIT",
    },
    {
        "id": "math-test",
        "repo": "EleutherAI/hendrycks_math",
        "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
        "files": tuple(
            f"{subject}/test-00000-of-00001.parquet"
            for subject in (
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            )
        ),
        "adapter": "hendrycks_math",
        "task_kind": "competition_math",
        "upstream_config": "all-seven-subjects",
        "upstream_split": "test",
        "use": "general_reasoning",
        "license": "MIT",
    },
    {
        "id": "mmlu-test",
        "repo": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "files": ("all/test-00000-of-00001.parquet",),
        "adapter": "mmlu",
        "task_kind": "multiple_choice_knowledge_reasoning",
        "upstream_config": "all",
        "upstream_split": "test",
        "use": "general_reasoning",
        "license": "MIT",
    },
    {
        "id": "arc-challenge-test",
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "files": ("ARC-Challenge/test-00000-of-00001.parquet",),
        "adapter": "arc",
        "task_kind": "science_multiple_choice",
        "upstream_config": "ARC-Challenge",
        "upstream_split": "test",
        "use": "general_reasoning",
        "license": "CC-BY-SA-4.0",
    },
    {
        "id": "mbpp-test",
        "repo": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "files": ("full/test-00000-of-00001.parquet",),
        "adapter": "mbpp",
        "task_kind": "python_programming",
        "upstream_config": "full",
        "upstream_split": "test",
        "use": "general_reasoning",
        "license": "CC-BY-4.0",
    },
    {
        "id": "hotpotqa-distractor-validation",
        "repo": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "files": ("distractor/validation-00000-of-00001.parquet",),
        "adapter": "hotpotqa",
        "task_kind": "multi_hop_qa",
        "upstream_config": "distractor",
        "upstream_split": "validation",
        "use": "toolizable_retrieval",
        "license": "CC-BY-SA-4.0",
    },
    {
        "id": "2wikimultihopqa-dev",
        "repo": "xanhho/2WikiMultihopQA",
        "revision": "612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
        "files": ("dev.parquet",),
        "adapter": "2wikimultihopqa",
        "task_kind": "multi_hop_qa",
        "upstream_config": "default",
        "upstream_split": "dev",
        "use": "toolizable_retrieval",
        "license": "Apache-2.0",
    },
    {
        "id": "musique-validation",
        "repo": "awinml/musique",
        "revision": "3ac762478df3609852f18dd33652a16820660a5e",
        "files": ("data/validation-00000-of-00001.parquet",),
        "adapter": "musique",
        "task_kind": "multi_hop_qa",
        "upstream_config": "default",
        "upstream_split": "validation",
        "use": "toolizable_retrieval",
        "license": "CC-BY-4.0",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_semantic_ids(path: Path) -> set[str]:
    semantic_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            semantic_ids.add(_semantic_id(str(json.loads(line)["prompt"])))
    return semantic_ids


def _benchmark_record(source: dict[str, Any], row_key: str, row: dict[str, Any]) -> dict[str, Any]:
    prompt, answer, resources, metadata = _adapt_row(str(source["adapter"]), row)
    return {
        "task_id": f"bench-{source['id']}-{hashlib.sha256(row_key.encode()).hexdigest()[:16]}",
        "semantic_id": _semantic_id(prompt),
        "split": "test",
        "task_kind": source["task_kind"],
        "prompt": prompt,
        "reference_answer": answer,
        "source": {
            "dataset_id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "license": source["license"],
            "upstream_config": source["upstream_config"],
            "upstream_split": source["upstream_split"],
            "row_key": row_key,
            "use": source["use"],
        },
        "resources": resources,
        "metadata": metadata,
    }


def _trajectory(record: dict[str, Any]) -> TrajectoryExample:
    metadata = {
        **dict(record["metadata"]),
        "benchmark_dataset_id": record["source"]["dataset_id"],
        "benchmark_repo": record["source"]["repo"],
        "benchmark_revision": record["source"]["revision"],
        "benchmark_upstream_split": record["source"]["upstream_split"],
        "task_kind": record["task_kind"],
        "semantic_id": record["semantic_id"],
    }
    if record["task_kind"] == "python_programming":
        metadata["public_tests"] = list(record["metadata"].get("tests", ()))
        metadata["challenge_tests"] = list(record["metadata"].get("challenge_tests", ()))
        metadata["public_test_setup_code"] = str(record["metadata"].get("test_setup_code", ""))

    if record["source"]["use"] != "toolizable_retrieval":
        return TrajectoryExample(
            example_id=record["task_id"],
            prompt=record["prompt"],
            target_display=record["reference_answer"],
            metadata=metadata,
        )

    task, _ = _teacher_task_from_public(
        record,
        PublicTrainingConfig(split="test", seed=20260911, unnecessary_tool_fraction=0.0),
    )
    evidence_bank = list(record.get("resources", {}).get("evidence_bank", ()))
    metadata["benchmark_workspace_documents"] = [
        {
            "resource_id": f"doc-{index:02d}",
            "title": str(resource["title"]),
            "sentences": [str(sentence) for sentence in resource.get("sentences", ())],
        }
        for index, resource in enumerate(evidence_bank)
    ]
    metadata["benchmark_workspace_search_top_k"] = 5
    metadata["benchmark_workspace_latency_steps"] = 2
    metadata["benchmark_supporting_resource_ids"] = list(
        dict.fromkeys(
            str(evidence.arguments["resource_id"])
            for evidence in task.evidence
            if evidence.source == "workspace_read" and "resource_id" in evidence.arguments
        )
    )
    events: list[ExternalEvent] = []
    bindings: list[BindingTarget] = []
    for index, evidence in enumerate(task.evidence):
        arrival_step = min(2 * (index + 1), 56)
        events.append(
            ExternalEvent(
                source=evidence.source,
                value=evidence.value,
                arrival_step=arrival_step,
                version=f"static:{evidence.evidence_id}",
                provenance=evidence.provenance,
                arguments=dict(evidence.arguments),
            )
        )
        bindings.append(
            BindingTarget(
                need_id=evidence.evidence_id,
                source=evidence.source,
                first_need_step=0,
                executable_step=0,
                arguments=dict(evidence.arguments),
                argument_steps={name: 0 for name in evidence.arguments},
            )
        )
    metadata.update(dict(task.metadata))
    metadata["benchmark_tool_events"] = len(events)
    return TrajectoryExample(
        example_id=record["task_id"],
        prompt=task.prompt,
        target_display=record["reference_answer"],
        protected_facts=dict(task.protected_facts),
        source_descriptors=task.source_descriptors,
        events=tuple(events),
        binding_targets=tuple(bindings),
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-data", required=True)
    args = parser.parse_args()

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("requires pandas and huggingface_hub") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_ids = _training_semantic_ids(Path(args.training_data))
    overall: dict[str, Any] = {"format_version": 2, "datasets": {}}

    for source in SOURCES:
        examples: list[TrajectoryExample] = []
        raw_rows = 0
        rejected = 0
        overlap = 0
        upstream_files: list[dict[str, Any]] = []
        for filename in source["files"]:
            local_path = Path(
                hf_hub_download(
                    repo_id=source["repo"],
                    repo_type="dataset",
                    filename=filename,
                    revision=source["revision"],
                )
            )
            upstream_files.append(
                {
                    "filename": filename,
                    "sha256": _sha256(local_path),
                    "bytes": local_path.stat().st_size,
                }
            )
            frame = pd.read_parquet(local_path)
            for row_number, row in enumerate(frame.to_dict(orient="records")):
                raw_rows += 1
                row_key = f"{filename}:{row_number}"
                try:
                    record = _benchmark_record(source, row_key, _jsonable(row))
                except PublicTaskRowRejected:
                    rejected += 1
                    continue
                if record["semantic_id"] in training_ids:
                    overlap += 1
                    continue
                examples.append(_trajectory(record))

        path = output_dir / f"{source['id']}.jsonl"
        dump_jsonl(examples, path)
        entry = {
            "repo": source["repo"],
            "revision": source["revision"],
            "upstream_config": source["upstream_config"],
            "upstream_split": source["upstream_split"],
            "task_kind": source["task_kind"],
            "raw_rows": raw_rows,
            "rejected_rows": rejected,
            "training_semantic_overlap_excluded": overlap,
            "examples": len(examples),
            "jsonl": path.name,
            "jsonl_sha256": _sha256(path),
            "upstream_files": upstream_files,
        }
        overall["datasets"][source["id"]] = entry
        print(source["id"], json.dumps(entry, ensure_ascii=False, sort_keys=True), flush=True)

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(overall, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest} sha256={_sha256(manifest)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

DATASETS = (
    "arc-challenge-test",
    "mbpp-test",
    "gsm8k-test",
    "musique-validation",
    "math-test",
    "hotpotqa-distractor-validation",
    "2wikimultihopqa-dev",
    "mmlu-test",
)


def _result_complete(summary_path: Path, result_path: Path) -> bool:
    if not summary_path.is_file() or not result_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = int(summary["dataset"]["examples"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    with result_path.open("r", encoding="utf-8") as handle:
        actual = sum(1 for line in handle if line.strip())
    return actual == expected


def _merge_dataset(dataset: str, output_root: Path, shard_count: int) -> None:
    dataset_dir = output_root / dataset
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard in range(shard_count):
        shard_dir = dataset_dir / f"shard-{shard:02d}"
        with (shard_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
        summaries.append(json.loads((shard_dir / "summary.json").read_text(encoding="utf-8")))

    records.sort(key=lambda item: str(item.get("example_id", "")))
    merged_results = dataset_dir / "results.jsonl"
    with merged_results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    task_metrics = [record.get("task_metrics", {}) for record in records]
    choice_rows = [item for item in task_metrics if item.get("choice_scored")]
    interactions = [record["evaluation"]["interaction"] for record in records]
    summary = {
        "dataset": dataset,
        "examples": len(records),
        "shards": shard_count,
        "code_commits": sorted(
            {
                str(item.get("code", {}).get("git_commit"))
                for item in summaries
                if item.get("code", {}).get("git_commit")
            }
        ),
        "dataset_sha256": summaries[0]["dataset"]["sha256"] if summaries else None,
        "task_metrics": {
            "answer_exact_accuracy": (
                sum(bool(item.get("answer_exact")) for item in task_metrics) / len(task_metrics)
                if task_metrics
                else 0.0
            ),
            "mean_qa_f1": (
                sum(float(item.get("qa_f1", 0.0)) for item in task_metrics) / len(task_metrics)
                if task_metrics
                else 0.0
            ),
            "choice_tasks": len(choice_rows),
            "choice_accuracy": (
                sum(bool(item.get("choice_correct")) for item in choice_rows) / len(choice_rows)
                if choice_rows
                else None
            ),
            "choice_unmapped": sum(
                item.get("choice_prediction_index") is None for item in choice_rows
            ),
        },
        "runtime": {
            "convergence_rate": (
                sum(bool(record["evaluation"]["converged"]) for record in records) / len(records)
                if records
                else 0.0
            ),
            "total_runtime_wall_time_s": sum(
                float(item.get("runtime_wall_time_s", 0.0)) for item in interactions
            ),
            "total_model_compute_s": sum(
                float(item.get("model_compute_s", 0.0)) for item in interactions
            ),
            "total_tool_wait_s": sum(float(item.get("tool_wait_s", 0.0)) for item in interactions),
            "tool_calls_completed": sum(
                int(item.get("tool_calls_completed", 0)) for item in interactions
            ),
            "external_refreshes": sum(
                int(item.get("external_refreshes", 0)) for item in interactions
            ),
        },
        "outer_wall_time_s_sum_across_shards": sum(
            float(item.get("timing", {}).get("outer_wall_time_s", 0.0)) for item in summaries
        ),
        "shard_summaries": [
            str((dataset_dir / f"shard-{index:02d}" / "summary.json").relative_to(output_root))
            for index in range(shard_count)
        ],
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    data_dir = Path(args.data_dir).resolve()
    release = Path(args.release).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus:
        raise SystemExit("at least one GPU is required")
    if args.workers_per_gpu <= 0 or args.shard_count <= 0:
        raise SystemExit("workers-per-gpu and shard-count must be positive")

    queue: deque[tuple[str, int]] = deque()
    for dataset in DATASETS:
        data_path = data_dir / f"{dataset}.jsonl"
        if not data_path.is_file():
            raise SystemExit(f"missing benchmark dataset: {data_path}")
        for shard in range(args.shard_count):
            shard_dir = output_root / dataset / f"shard-{shard:02d}"
            if not _result_complete(shard_dir / "summary.json", shard_dir / "results.jsonl"):
                queue.append((dataset, shard))

    running: dict[int, tuple[subprocess.Popen[Any], str, int, int, Any]] = {}
    gpu_load: dict[int, int] = defaultdict(int)
    failures: list[dict[str, Any]] = []
    started = time.time()

    def launch(dataset: str, shard: int, gpu: int) -> None:
        shard_dir = output_root / dataset / f"shard-{shard:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (shard_dir / "run.log").open("a", encoding="utf-8")
        command = [
            args.python,
            "-m",
            "cid.cli",
            "benchmark",
            "--data",
            str(data_dir / f"{dataset}.jsonl"),
            "--checkpoint",
            str(release),
            "--checkpoint-kind",
            "release",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--seed",
            "0",
            "--max-steps",
            str(args.max_steps),
            "--shard-count",
            str(args.shard_count),
            "--shard-index",
            str(shard),
            "--progress-every",
            "100",
            "--output",
            str(shard_dir / "results.jsonl"),
            "--summary-output",
            str(shard_dir / "summary.json"),
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(repo / "src"),
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running[process.pid] = (process, dataset, shard, gpu, log_handle)
        gpu_load[gpu] += 1
        print(
            f"START dataset={dataset} shard={shard}/{args.shard_count} gpu={gpu} pid={process.pid}",
            flush=True,
        )

    while queue or running:
        launched = True
        while queue and launched:
            launched = False
            for gpu in gpus:
                if gpu_load[gpu] >= args.workers_per_gpu or not queue:
                    continue
                dataset, shard = queue.popleft()
                launch(dataset, shard, gpu)
                launched = True
        time.sleep(2.0)
        for pid, item in list(running.items()):
            process, dataset, shard, gpu, log_handle = item
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            del running[pid]
            gpu_load[gpu] -= 1
            if code != 0:
                failures.append({"dataset": dataset, "shard": shard, "gpu": gpu, "exit_code": code})
                print(
                    f"FAIL dataset={dataset} shard={shard} gpu={gpu} exit={code}",
                    flush=True,
                )
            else:
                print(f"DONE dataset={dataset} shard={shard} gpu={gpu}", flush=True)

        for dataset in DATASETS:
            dataset_dir = output_root / dataset
            if (dataset_dir / "summary.json").is_file():
                continue
            complete = all(
                _result_complete(
                    dataset_dir / f"shard-{shard:02d}" / "summary.json",
                    dataset_dir / f"shard-{shard:02d}" / "results.jsonl",
                )
                for shard in range(args.shard_count)
            )
            if complete:
                _merge_dataset(dataset, output_root, args.shard_count)
                print(f"MERGED dataset={dataset}", flush=True)

    report = {
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "datasets": list(DATASETS),
        "gpus": list(gpus),
        "workers_per_gpu": args.workers_per_gpu,
        "shard_count": args.shard_count,
        "failures": failures,
    }
    (output_root / "sweep.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

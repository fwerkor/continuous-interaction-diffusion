from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cid.computational_training import record_lookup_descriptor
from cid.data import dump_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
from cid.distill import (
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_reviews,
    dump_teacher_tasks,
    load_teacher_tasks,
    review_teacher_plans,
)

DEFAULT_CAPACITY_BUCKETS = (16, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class DeepToolRestraintConfig:
    count_per_bucket: int = 1000
    min_dependency_depth: int = 8
    capacity_buckets: tuple[int, ...] = DEFAULT_CAPACITY_BUCKETS
    seed: int = 20260813

    def __post_init__(self) -> None:
        if self.count_per_bucket <= 0:
            raise ValueError("count_per_bucket must be positive")
        if self.min_dependency_depth <= 0:
            raise ValueError("min_dependency_depth must be positive")
        if not self.capacity_buckets or any(value <= 0 for value in self.capacity_buckets):
            raise ValueError("capacity_buckets must contain positive capacities")
        if len(set(self.capacity_buckets)) != len(self.capacity_buckets):
            raise ValueError("capacity_buckets must be unique")

    @property
    def total_tasks(self) -> int:
        return self.count_per_bucket * len(self.capacity_buckets)


def build_deep_tool_restraint_distillation(
    source_tasks_path: str | Path,
    source_plans_path: str | Path,
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: DeepToolRestraintConfig | None = None,
) -> dict[str, Any]:
    config = config or DeepToolRestraintConfig()
    source_tasks = load_teacher_tasks(source_tasks_path)
    selected_tasks = _select_source_tasks(source_tasks, config)
    selected_ids = {task.task_id for task in selected_tasks}
    source_plans = _load_selected_plans(source_plans_path, selected_ids)
    tasks, plans = _augment_selected_tasks(selected_tasks, source_plans)

    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        raise RuntimeError(
            f"deep tool-restraint review rejected {len(rejected)} plans: "
            f"{[item.to_dict() for item in rejected[:8]]}"
        )
    if any(task.evidence for task in tasks):
        raise RuntimeError("deep tool-restraint tasks must not contain external evidence")
    if any(plan.needs for plan in plans):
        raise RuntimeError("deep tool-restraint plans must not request tools")

    plan_by_id = {plan.task_id: plan for plan in plans}
    trajectories = []
    for capacity in config.capacity_buckets:
        bucket_tasks = tuple(
            task for task in tasks if int(task.metadata["thought_capacity_bucket"]) == capacity
        )
        bucket_plans = tuple(plan_by_id[task.task_id] for task in bucket_tasks)
        trajectories.extend(
            compile_teacher_plans(
                bucket_tasks,
                bucket_plans,
                TeacherScheduleConfig(
                    thought_capacity=capacity,
                    variants_per_task=1,
                    seed=config.seed + capacity,
                ),
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output / "deep-tool-restraint-teacher-tasks-v1.jsonl",
        "plans": output / "deep-tool-restraint-teacher-plans-v1.accepted.jsonl",
        "reviews": output / "deep-tool-restraint-teacher-review-v1.jsonl",
        "trajectories": output / "deep-tool-restraint-trajectories-v1.jsonl",
        "trajectory_manifest": output / "deep-tool-restraint-trajectories-v1.manifest.json",
    }
    dump_teacher_tasks(tasks, paths["tasks"])
    dump_teacher_plans(plans, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])
    trajectory_manifest = inspect_dataset(paths["trajectories"])
    dump_dataset_manifest(trajectory_manifest, paths["trajectory_manifest"])

    capacity_counts = Counter(int(task.metadata["thought_capacity_bucket"]) for task in tasks)
    family_counts = Counter(str(task.metadata["family"]) for task in tasks)
    depth_counts = Counter(int(task.metadata["dependency_depth"]) for task in tasks)
    family_by_capacity: dict[str, dict[str, int]] = defaultdict(dict)
    for capacity in config.capacity_buckets:
        counts = Counter(
            str(task.metadata["family"])
            for task in tasks
            if int(task.metadata["thought_capacity_bucket"]) == capacity
        )
        family_by_capacity[str(capacity)] = dict(sorted(counts.items()))

    max_semantic = max(
        len(cell.semantic_text) for plan in plans for frame in plan.frames for cell in frame.cells
    )
    if max_semantic > 112:
        raise RuntimeError(f"deep tool-restraint semantic text cap exceeded: {max_semantic}")

    manifest = {
        "format_version": 1,
        "name": "deep-tool-restraint-v1",
        "version": 1,
        "generator": "cid.deep_restraint_training.v1",
        "seed": config.seed,
        "semantic_tasks": len(tasks),
        "accepted_plans": len(plans),
        "review_rejected": 0,
        "compiled_trajectories": trajectory_manifest.examples,
        "compiled_transitions": trajectory_manifest.transitions,
        "mode_counts": {"tools_available_unnecessary": len(tasks)},
        "intentional_semantic_overlap_with_compositional_longtail": len(tasks),
        "minimum_dependency_depth": config.min_dependency_depth,
        "dependency_depth_histogram": {str(k): v for k, v in sorted(depth_counts.items())},
        "capacity_bucket_counts": {str(k): v for k, v in sorted(capacity_counts.items())},
        "family_counts": dict(sorted(family_counts.items())),
        "family_counts_by_capacity": dict(family_by_capacity),
        "tool_schema_task_counts": {"record_lookup": len(tasks)},
        "thought_capacity_required": max(config.capacity_buckets),
        "max_semantic_text_chars": max_semantic,
        "tasks_without_evidence": sum(not task.evidence for task in tasks),
        "plans_without_needs": sum(not plan.needs for plan in plans),
        "tasks_sha256": _sha256(paths["tasks"]),
        "plans_sha256": _sha256(paths["plans"]),
        "review_sha256": _sha256(paths["reviews"]),
        "compiled_sha256": trajectory_manifest.sha256,
        "source_tasks_sha256": _sha256(Path(source_tasks_path)),
        "source_plans_sha256": _sha256(Path(source_plans_path)),
    }
    reference_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trajectory_raw = json.loads(paths["trajectory_manifest"].read_text(encoding="utf-8"))
    trajectory_raw.update(
        {
            "name": "deep-tool-restraint-trajectories-v1",
            "reference_manifest": str(reference_path),
            "thought_capacity_required": max(config.capacity_buckets),
            "capacity_bucket_counts": {str(k): v for k, v in sorted(capacity_counts.items())},
            "tag_counts": {"mode:tools_available_unnecessary": len(tasks)},
        }
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(trajectory_raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_deep_tool_restraint_examples(
    source_tasks: tuple[TeacherTask, ...],
    source_plans: tuple[TeacherPlan, ...],
    config: DeepToolRestraintConfig | None = None,
) -> tuple[tuple[TeacherTask, ...], tuple[TeacherPlan, ...]]:
    config = config or DeepToolRestraintConfig()
    plan_by_id = {plan.task_id: plan for plan in source_plans}
    selected = _select_source_tasks(source_tasks, config, plan_ids=set(plan_by_id))
    return _augment_selected_tasks(selected, source_plans)


def _select_source_tasks(
    source_tasks: tuple[TeacherTask, ...],
    config: DeepToolRestraintConfig,
    *,
    plan_ids: set[str] | None = None,
) -> tuple[TeacherTask, ...]:
    selected: list[TeacherTask] = []
    for capacity in config.capacity_buckets:
        candidates = [
            task
            for task in source_tasks
            if (plan_ids is None or task.task_id in plan_ids)
            and not task.evidence
            and str(task.metadata.get("training_mode", "")) == "no_tool_required"
            and int(task.metadata.get("thought_capacity_bucket", 0)) == capacity
            and int(task.metadata.get("dependency_depth", 0)) >= config.min_dependency_depth
        ]
        if len(candidates) < config.count_per_bucket:
            raise ValueError(
                f"capacity {capacity} has only {len(candidates)} eligible deep restraint tasks; "
                f"need {config.count_per_bucket}"
            )
        ranked = sorted(
            candidates,
            key=lambda task: hashlib.sha256(
                f"{config.seed}|{capacity}|{task.task_id}".encode()
            ).digest(),
        )
        selected.extend(ranked[: config.count_per_bucket])

    return tuple(selected)


def _augment_selected_tasks(
    selected: tuple[TeacherTask, ...],
    source_plans: tuple[TeacherPlan, ...],
) -> tuple[tuple[TeacherTask, ...], tuple[TeacherPlan, ...]]:
    plan_by_id = {plan.task_id: plan for plan in source_plans}
    missing = sorted(task.task_id for task in selected if task.task_id not in plan_by_id)
    if missing:
        raise ValueError(f"selected deep-restraint tasks are missing plans: {missing[:8]}")

    tasks: list[TeacherTask] = []
    plans: list[TeacherPlan] = []
    descriptor = record_lookup_descriptor()
    for source_task in selected:
        task_id = f"deep-restraint-{source_task.task_id}"
        task = replace(
            source_task,
            task_id=task_id,
            source_descriptors=(descriptor,),
            metadata={
                **dict(source_task.metadata),
                "training_mode": "tools_available_unnecessary",
                "source_task_id": source_task.task_id,
                "augmentation": "deep_tool_restraint",
                "generated_by": "cid.deep_restraint_training.v1",
            },
        )
        plan = replace(plan_by_id[source_task.task_id], task_id=task_id)
        tasks.append(task)
        plans.append(plan)

    paired = sorted(zip(tasks, plans, strict=True), key=lambda pair: pair[0].task_id)
    return tuple(item[0] for item in paired), tuple(item[1] for item in paired)


def _load_selected_plans(
    path: str | Path,
    selected_ids: set[str],
) -> tuple[TeacherPlan, ...]:
    plans: list[TeacherPlan] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            task_id = str(raw.get("task_id", ""))
            if task_id in selected_ids:
                plans.append(TeacherPlan.from_dict(raw))
    found = {plan.task_id for plan in plans}
    missing = sorted(selected_ids - found)
    if missing:
        raise ValueError(f"source plan file is missing selected task IDs: {missing[:8]}")
    return tuple(plans)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

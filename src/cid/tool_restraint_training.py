from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cid.computational_training import calculator_descriptor, record_lookup_descriptor
from cid.data import dump_jsonl
from cid.distill import (
    TeacherCellPlan,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_reviews,
    dump_teacher_tasks,
    load_teacher_plans,
    load_teacher_tasks,
    review_teacher_plans,
)
from cid.grounding import CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole
from cid.symbolic_training import symbolic_math_descriptor


@dataclass(frozen=True, slots=True)
class ToolRestraintTrainingConfig:
    count: int = 6500
    seed: int = 20260813
    thought_capacity: int = 8

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.thought_capacity <= 0:
            raise ValueError("thought_capacity must be positive")


def build_tool_restraint_distillation(
    source_tasks_path: str | Path,
    source_plans_path: str | Path,
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: ToolRestraintTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or ToolRestraintTrainingConfig()
    tasks = load_teacher_tasks(source_tasks_path)
    plans = load_teacher_plans(source_plans_path)
    selected_tasks, selected_plans = generate_tool_restraint_examples(tasks, plans, config)

    reviews = review_teacher_plans(selected_tasks, selected_plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        raise RuntimeError(
            f"tool-restraint review rejected {len(rejected)} plans: "
            f"{[item.to_dict() for item in rejected[:10]]}"
        )

    trajectories = compile_teacher_plans(
        selected_tasks,
        selected_plans,
        TeacherScheduleConfig(
            thought_capacity=config.thought_capacity,
            variants_per_task=1,
            seed=config.seed,
        ),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output / "tool-restraint-teacher-tasks-v1.jsonl",
        "plans": output / "tool-restraint-teacher-plans-v1.accepted.jsonl",
        "reviews": output / "tool-restraint-teacher-review-v1.jsonl",
        "trajectories": output / "tool-restraint-trajectories-v1.jsonl",
        "trajectory_manifest": output / "tool-restraint-trajectories-v1.manifest.json",
    }
    dump_teacher_tasks(selected_tasks, paths["tasks"])
    dump_teacher_plans(selected_plans, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])

    dataset_counts = Counter(
        str(task.metadata.get("public_dataset_id", "unknown")) for task in selected_tasks
    )
    kind_counts = Counter(str(task.metadata.get("task_kind", "unknown")) for task in selected_tasks)
    descriptor_counts: Counter[str] = Counter()
    for task in selected_tasks:
        for descriptor in task.source_descriptors:
            descriptor_counts[str(descriptor["name"])] += 1

    max_cells = max(len(frame.cells) for plan in selected_plans for frame in plan.frames)
    max_semantic_text_chars = max(
        len(cell.semantic_text)
        for plan in selected_plans
        for frame in plan.frames
        for cell in frame.cells
    )
    if max_cells > config.thought_capacity:
        raise RuntimeError(
            f"tool-restraint plan requires {max_cells} cells but thought capacity is "
            f"{config.thought_capacity}"
        )
    if max_semantic_text_chars > 144:
        raise RuntimeError(f"tool-restraint semantic text cap exceeded: {max_semantic_text_chars}")
    if any(task.evidence for task in selected_tasks):
        raise RuntimeError("tool-restraint tasks must not contain external evidence")
    if any(plan.needs for plan in selected_plans):
        raise RuntimeError("tool-restraint plans must not request tools")

    compiled_transitions = sum(
        max(
            [target.step for target in trajectory.thought_targets]
            + [target.step for target in trajectory.display_targets]
            + [0]
        )
        for trajectory in trajectories
    )
    manifest = {
        "format_version": 1,
        "name": "natural-tool-restraint-v1",
        "version": 1,
        "seed": config.seed,
        "semantic_tasks": len(selected_tasks),
        "accepted_plans": len(selected_plans),
        "review_rejected": 0,
        "compiled_trajectories": len(trajectories),
        "compiled_transitions": compiled_transitions,
        "source_dataset_counts": dict(sorted(dataset_counts.items())),
        "task_kind_counts": dict(sorted(kind_counts.items())),
        "tool_schema_task_counts": dict(sorted(descriptor_counts.items())),
        "mode_counts": {"tools_available_unnecessary": len(selected_tasks)},
        "intentional_semantic_overlap_with_public_base": len(selected_tasks),
        "augmentation": "accepted natural no-tool task with irrelevant tools exposed",
        "thought_capacity_required": config.thought_capacity,
        "max_cells_per_frame": max_cells,
        "max_semantic_text_chars": max_semantic_text_chars,
        "tasks_sha256": _sha256(paths["tasks"]),
        "plans_sha256": _sha256(paths["plans"]),
        "review_sha256": _sha256(paths["reviews"]),
        "compiled_sha256": _sha256(paths["trajectories"]),
        "source_tasks_sha256": _sha256(Path(source_tasks_path)),
        "source_plans_sha256": _sha256(Path(source_plans_path)),
    }
    reference_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "tool-restraint-trajectories-v1",
                "schema": "cid.TrajectoryExample.v1",
                "examples": len(trajectories),
                "transitions": compiled_transitions,
                "thought_capacity_required": config.thought_capacity,
                "sha256": manifest["compiled_sha256"],
                "reference_manifest": str(reference_path),
                "tag_counts": {"mode:tools_available_unnecessary": len(trajectories)},
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_tool_restraint_examples(
    tasks: tuple[TeacherTask, ...],
    plans: tuple[TeacherPlan, ...],
    config: ToolRestraintTrainingConfig | None = None,
) -> tuple[tuple[TeacherTask, ...], tuple[TeacherPlan, ...]]:
    config = config or ToolRestraintTrainingConfig()
    plan_by_id = {plan.task_id: plan for plan in plans}
    candidates = [
        task
        for task in tasks
        if task.task_id in plan_by_id
        and not task.evidence
        and str(task.metadata.get("training_mode", "")) == "no_tool"
        and len(plan_by_id[task.task_id].frames) == 1
        and plan_by_id[task.task_id].frames[0].phase == "initial"
    ]
    if config.count > len(candidates):
        raise ValueError(
            f"requested {config.count} tool-restraint examples but only {len(candidates)} "
            "accepted no-tool candidates are available"
        )

    ranked = sorted(
        candidates,
        key=lambda task: hashlib.sha256(f"{config.seed}|{task.task_id}".encode()).digest(),
    )[: config.count]

    augmented_tasks: list[TeacherTask] = []
    augmented_plans: list[TeacherPlan] = []
    for source_task in ranked:
        source_plan = plan_by_id[source_task.task_id]
        task_id = f"restraint-{source_task.task_id}"
        descriptors = _irrelevant_tool_descriptors(source_task)
        task = replace(
            source_task,
            task_id=task_id,
            source_descriptors=descriptors,
            metadata={
                **dict(source_task.metadata),
                "training_mode": "tools_available_unnecessary",
                "source_task_id": source_task.task_id,
                "augmentation": "tool_restraint",
                "generated_by": "cid.tool_restraint_training.v1",
            },
        )
        frame = source_plan.frames[0]
        cells = list(frame.cells)
        target_cell_id = _reasoning_cell_id(cells)
        cells.append(
            TeacherCellPlan(
                cell_id="tool-decision",
                semantic_text=_decision_text(source_task),
                roles={CognitiveRole.CONSTRAINT: 0.7, CognitiveRole.PLAN: 0.8},
                uncertainty=0.04,
                noise=0.02,
                lifecycle=CellLifecycle.STABLE,
                links=(
                    CognitiveLink(
                        LinkRelation.DEPENDS_ON,
                        ObjectRef.cell(target_cell_id),
                        1.0,
                    ),
                ),
            )
        )
        plan = replace(
            source_plan,
            task_id=task_id,
            frames=(replace(frame, cells=tuple(cells)),),
        )
        augmented_tasks.append(task)
        augmented_plans.append(plan)

    paired = sorted(
        zip(augmented_tasks, augmented_plans, strict=True),
        key=lambda pair: pair[0].task_id,
    )
    return tuple(pair[0] for pair in paired), tuple(pair[1] for pair in paired)


def _irrelevant_tool_descriptors(task: TeacherTask) -> tuple[dict[str, Any], ...]:
    kind = str(task.metadata.get("task_kind", ""))
    if kind in {"math_word_problem", "competition_math"}:
        return (
            calculator_descriptor(),
            symbolic_math_descriptor(),
            record_lookup_descriptor(),
        )
    # Do not expose Python on programming tasks: the supervision should teach restraint from
    # irrelevant tools, not a blanket aversion to executing code when execution is useful.
    return (record_lookup_descriptor(), calculator_descriptor())


def _reasoning_cell_id(cells: list[TeacherCellPlan]) -> str:
    for cell in cells:
        if cell.roles.get(CognitiveRole.CONCLUSION, 0.0) <= 0.0:
            return cell.cell_id
    return cells[0].cell_id


def _decision_text(task: TeacherTask) -> str:
    kind = str(task.metadata.get("task_kind", ""))
    variants = {
        "math_word_problem": (
            (
                "The prompt supplies every quantity needed; available tools do not resolve "
                "missing evidence."
            ),
            (
                "No external fact is missing, so tool calls would add latency without changing "
                "the solution basis."
            ),
        ),
        "competition_math": (
            (
                "This problem is self-contained; solve from the stated mathematics without "
                "external retrieval."
            ),
            "The mathematical premises are complete, so available external tools are unnecessary.",
        ),
        "python_programming": (
            (
                "The required function can be derived from the specification; unrelated lookup "
                "tools are unnecessary."
            ),
            (
                "No external record is needed to implement the requested behavior from the "
                "supplied specification."
            ),
        ),
        "science_multiple_choice": (
            (
                "The choices and stated scenario are sufficient; no external record is required "
                "for this decision."
            ),
            (
                "The prompt contains the evidence needed to select an option, so available tools "
                "should remain idle."
            ),
        ),
        "multiple_choice_knowledge_reasoning": (
            (
                "The supplied question is answerable from internal knowledge; no task-specific "
                "external evidence is missing."
            ),
            (
                "No external evidence contract is required here; answer directly rather than "
                "calling an available tool."
            ),
        ),
    }
    options = variants.get(
        kind,
        (
            "The prompt is self-contained; available tools are unnecessary for this answer.",
            "No missing external evidence justifies a tool call for this task.",
        ),
    )
    index = int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16) % len(options)
    return options[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

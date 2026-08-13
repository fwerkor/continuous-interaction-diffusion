from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cid.causal_distill import dump_causal_teacher_jobs
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

_PREAMBLES = (
    "Work only within this task-local instance.",
    "Treat this as an isolated task-local case.",
    "Use the declared sources for this instance only.",
    "Resolve this instance from its task-local records.",
    "Keep all reasoning scoped to the records named below.",
    "Handle this case using only its declared external interfaces.",
    "Process the following task with task-local source semantics.",
    "For this isolated instance, follow the declared evidence path.",
    "Use the available task-local sources without importing outside assumptions.",
    "Treat record names as opaque identifiers local to this task.",
    "Solve the case under the source and dependency rules stated here.",
    "Operate only on evidence licensed by this task's declared sources.",
)

_DEPENDENCY_CLAUSES = (
    "Do not infer a dependent value before its prerequisite has been resolved.",
    "Keep unresolved downstream values unknown until their parent evidence arrives.",
    "Respect the dependency order instead of guessing later records early.",
    "A downstream read becomes usable only after the evidence it depends on is available.",
    "Preserve unresolved state across dependency boundaries until the required evidence arrives.",
    "Do not collapse the dependency chain by predicting task-local record contents.",
    "Use each resolved prerequisite to license only the downstream work it actually enables.",
    "Keep dependent branches pending until their required parent state is known.",
    "Treat missing downstream evidence as unknown rather than filling it from pattern matching.",
    "Advance dependent work only from evidence already exposed in the current task state.",
    "Do not substitute likely values for records that have not yet been observed.",
    "Honor evidence availability when moving from one task stage to the next.",
)

_SOURCE_CLAUSES = (
    "Task-local identifiers do not reveal their record contents.",
    "Source names describe interfaces, not hidden return values.",
    "Only returned evidence may establish external facts.",
    "Keep interface metadata separate from the values eventually returned by those interfaces.",
    "An available source is not itself evidence for any particular answer.",
    "Do not treat a source schema as if it already contained the requested record.",
    "External values become usable only when their observations are present.",
    "Preserve the distinction between knowing which source to query and knowing its result.",
    "Do not turn source descriptions into guessed task facts.",
    "Use source metadata for routing while keeping unseen record values unresolved.",
)

_OUTPUT_CLAUSES = (
    "Return the final answer in exactly the format requested by the core task.",
    "Preserve the requested final-answer format and omit unrelated commentary.",
    "Use the core task's output contract for the final response.",
    "Once the required evidence is resolved, answer using the requested compact format.",
    "The final response should contain only what the task's answer format requires.",
    "Keep the final output aligned with the explicit formatting instruction in the task.",
    "Do not change the requested answer representation when presenting the final result.",
    "Follow the task's stated output shape after the evidence-dependent reasoning is complete.",
    "Render the conclusion using the exact response form requested below.",
    "Use the requested answer format after completing the dependency-aware computation.",
)

_ID_RE = re.compile(r"\b[a-z][a-z0-9-]*-\d{4,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

_SEMANTIC_ROLE_PREFIXES = {
    "information_need": (
        "Open need: ",
        "Required evidence: ",
        "Pending input: ",
        "Unresolved requirement: ",
        "Needed observation: ",
        "External requirement: ",
    ),
    "percept": (
        "Observed state: ",
        "Available evidence: ",
        "Current observation: ",
        "Evidence summary: ",
        "Observed value: ",
        "Percept state: ",
    ),
    "conclusion": (
        "Resolved conclusion: ",
        "Current conclusion: ",
        "Answer state: ",
        "Conclusion summary: ",
        "Resolved answer: ",
        "Final-state conclusion: ",
    ),
    "constraint": (
        "Active constraint: ",
        "Constraint state: ",
        "Current restriction: ",
        "Applicable constraint: ",
        "Scope constraint: ",
        "Constraint summary: ",
    ),
    "hypothesis": (
        "Working hypothesis: ",
        "Current hypothesis: ",
        "Provisional state: ",
        "Candidate state: ",
        "Hypothesis summary: ",
        "Current candidate: ",
    ),
    "plan": (
        "Current plan: ",
        "Plan state: ",
        "Next-step plan: ",
        "Working plan: ",
        "Plan summary: ",
        "Active plan: ",
    ),
    "default": (
        "Current state: ",
        "State summary: ",
        "Working state: ",
        "Current note: ",
        "State note: ",
        "Snapshot state: ",
    ),
}

_SEMANTIC_STAGE_SUFFIXES = (
    "",
    " [current]",
    " [tracked]",
    " [task-local]",
    " [state]",
    " [active context]",
)


@dataclass(frozen=True, slots=True)
class SurfaceDiversityConfig:
    component_name: str
    file_stem: str
    thought_capacity: int
    variants_per_task: int = 2
    min_delay_steps: int = 1
    max_delay_steps: int = 4
    seed: int = 20260813
    max_tasks: int | None = None
    surface_version: int = 2
    diversify_prompt: bool = True
    diversify_semantic_text: bool = False
    semantic_text_cap: int = 144

    def __post_init__(self) -> None:
        if not self.component_name or not self.file_stem:
            raise ValueError("component_name and file_stem must be non-empty")
        if self.thought_capacity <= 0:
            raise ValueError("thought_capacity must be positive")
        if self.variants_per_task <= 0:
            raise ValueError("variants_per_task must be positive")
        if self.min_delay_steps <= 0 or self.max_delay_steps < self.min_delay_steps:
            raise ValueError("invalid delay range")
        if self.max_tasks is not None and self.max_tasks <= 0:
            raise ValueError("max_tasks must be positive when provided")
        if self.surface_version <= 0:
            raise ValueError("surface_version must be positive")
        if self.semantic_text_cap <= 0:
            raise ValueError("semantic_text_cap must be positive")


def build_surface_diversified_distillation(
    source_tasks_path: str | Path,
    source_plans_path: str | Path,
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: SurfaceDiversityConfig,
) -> dict[str, Any]:
    source_tasks = load_teacher_tasks(source_tasks_path)
    selected_tasks = _select_tasks(source_tasks, config)
    selected_ids = {task.task_id for task in selected_tasks}
    source_plans = _load_selected_plans(source_plans_path, selected_ids)
    tasks, plans = diversify_tasks_and_plans(selected_tasks, source_plans, config)

    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    semantic_text_retry_plans = 0
    semantic_text_fallback_plans = 0
    if rejected and config.diversify_semantic_text:
        # Surface wrappers are intentionally semantically weak, but external
        # evidence values are arbitrary. A wrapper token such as "active"
        # can therefore accidentally equal a future evidence value and turn
        # an otherwise causal source plan into a leaking one. Retry alternate
        # deterministic wrappers for only the collided task; preserve the
        # source semantic text only when every safe wrapper candidate fails.
        # The normal causal reviewer remains the acceptance gate throughout.
        rejected_ids = {review.task_id for review in rejected}
        source_plan_by_id = {plan.task_id: plan for plan in source_plans}
        task_by_id = {task.task_id: task for task in tasks}
        repaired: list[TeacherPlan] = []
        for plan in plans:
            if plan.task_id not in rejected_ids:
                repaired.append(plan)
                continue
            task = task_by_id[plan.task_id]
            source_task_id = str(task.metadata["source_task_id"])
            source_plan = source_plan_by_id[source_task_id]
            replacement: TeacherPlan | None = None
            # Most collisions are accidental lexical matches between a
            # wrapper (for example "Active plan") and an unseen evidence
            # value (for example "active").  Re-sample only the wrappers for
            # this task and keep the first variant that satisfies the same
            # causal reviewer.  This preserves semantic-surface diversity
            # without weakening evidence visibility.
            for retry in range(1, 32):
                candidate = _surface_semantic_plan(
                    replace(source_plan, task_id=plan.task_id),
                    source_task_id,
                    config,
                    variant_offset=retry,
                )
                if review_teacher_plans((task,), (candidate,))[0].accepted:
                    replacement = candidate
                    semantic_text_retry_plans += 1
                    break
            if replacement is None:
                replacement = replace(source_plan, task_id=plan.task_id)
                semantic_text_fallback_plans += 1
            repaired.append(replacement)
        plans = tuple(repaired)
        reviews = review_teacher_plans(tasks, plans)
        rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        raise RuntimeError(
            f"surface-diversified review rejected {len(rejected)} plans: "
            f"{[item.to_dict() for item in rejected[:8]]}"
        )

    schedule = TeacherScheduleConfig(
        thought_capacity=config.thought_capacity,
        min_delay_steps=config.min_delay_steps,
        max_delay_steps=config.max_delay_steps,
        variants_per_task=config.variants_per_task,
        seed=config.seed,
    )
    trajectories = compile_teacher_plans(tasks, plans, schedule)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output / f"{config.file_stem}-teacher-tasks.jsonl",
        "causal_jobs": output / f"{config.file_stem}-causal-jobs.jsonl",
        "plans": output / f"{config.file_stem}-teacher-plans.accepted.jsonl",
        "reviews": output / f"{config.file_stem}-teacher-review.jsonl",
        "trajectories": output / f"{config.file_stem}-trajectories.jsonl",
        "trajectory_manifest": output / f"{config.file_stem}-trajectories.manifest.json",
    }
    dump_teacher_tasks(tasks, paths["tasks"])
    dump_causal_teacher_jobs(tasks, paths["causal_jobs"])
    dump_teacher_plans(plans, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])
    trajectory_manifest = inspect_dataset(paths["trajectories"])
    dump_dataset_manifest(trajectory_manifest, paths["trajectory_manifest"])

    signatures = Counter(_normalized_surface_signature(task.prompt) for task in tasks)
    semantic_signatures = Counter(
        _normalized_surface_signature(cell.semantic_text)
        for plan in plans
        for frame in plan.frames
        for cell in frame.cells
    )
    family_counts = Counter(str(task.metadata.get("family", "unknown")) for task in tasks)
    max_semantic = max(
        len(cell.semantic_text) for plan in plans for frame in plan.frames for cell in frame.cells
    )
    tasks_with_anchor = sum(
        any(cell.anchors for frame in plan.frames for cell in frame.cells) for plan in plans
    )
    tasks_with_link = sum(
        any(cell.links for frame in plan.frames for cell in frame.cells) for plan in plans
    )

    manifest = {
        "format_version": 1,
        "name": config.component_name,
        "version": config.surface_version,
        "generator": f"cid.surface_diversity_training.v{config.surface_version}",
        "seed": config.seed,
        "semantic_tasks": len(tasks),
        "accepted_plans": len(plans),
        "review_rejected": 0,
        "compiled_trajectories": trajectory_manifest.examples,
        "compiled_transitions": trajectory_manifest.transitions,
        "compiled_bootstrap_transitions": trajectory_manifest.bootstrap_transitions,
        "compiled_training_transitions": trajectory_manifest.training_transitions,
        "thought_capacity_required": config.thought_capacity,
        "family_counts": dict(sorted(family_counts.items())),
        "normalized_prompt_signatures": len(signatures),
        "largest_normalized_prompt_group": max(signatures.values()),
        "normalized_prompt_signature_ratio": round(len(signatures) / len(tasks), 6),
        "normalized_semantic_text_signatures": len(semantic_signatures),
        "largest_normalized_semantic_text_group": max(semantic_signatures.values()),
        "normalized_semantic_text_signature_ratio": round(
            len(semantic_signatures) / sum(semantic_signatures.values()), 6
        ),
        "max_semantic_text_chars": max_semantic,
        "tasks_with_anchor": tasks_with_anchor,
        "tasks_with_link": tasks_with_link,
        "surface_payload_policy": (
            "preserve core task/evidence/typed plan semantics; vary prompt and semantic transport "
            "surfaces according to config"
        ),
        "diversify_prompt": config.diversify_prompt,
        "diversify_semantic_text": config.diversify_semantic_text,
        "semantic_text_retry_plans": semantic_text_retry_plans,
        "semantic_text_fallback_plans": semantic_text_fallback_plans,
        "compiler": {
            "variants_per_task": config.variants_per_task,
            "min_delay_steps": config.min_delay_steps,
            "max_delay_steps": config.max_delay_steps,
            "seed": config.seed,
        },
        "tasks_sha256": _sha256(paths["tasks"]),
        "causal_jobs_sha256": _sha256(paths["causal_jobs"]),
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
            "name": f"{config.file_stem}-trajectories",
            "reference_manifest": str(reference_path),
            "thought_capacity_required": config.thought_capacity,
        }
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(trajectory_raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def diversify_tasks_and_plans(
    source_tasks: tuple[TeacherTask, ...],
    source_plans: tuple[TeacherPlan, ...],
    config: SurfaceDiversityConfig,
) -> tuple[tuple[TeacherTask, ...], tuple[TeacherPlan, ...]]:
    plan_by_id = {plan.task_id: plan for plan in source_plans}
    tasks: list[TeacherTask] = []
    plans: list[TeacherPlan] = []
    for source_task in source_tasks:
        if source_task.task_id not in plan_by_id:
            raise ValueError(f"source task {source_task.task_id!r} has no accepted plan")
        task_id = f"surface-v{config.surface_version}-{source_task.task_id}"
        task = replace(
            source_task,
            task_id=task_id,
            prompt=(
                _surface_prompt(source_task, config.seed)
                if config.diversify_prompt
                else source_task.prompt
            ),
            metadata={
                **dict(source_task.metadata),
                "surface_version": config.surface_version,
                "source_task_id": source_task.task_id,
                "augmentation": "surface_diversification",
                "generated_by": f"cid.surface_diversity_training.v{config.surface_version}",
            },
        )
        source_plan = plan_by_id[source_task.task_id]
        plan = replace(source_plan, task_id=task_id)
        if config.diversify_semantic_text:
            plan = _surface_semantic_plan(plan, source_task.task_id, config)
        tasks.append(task)
        plans.append(plan)
    paired = sorted(zip(tasks, plans, strict=True), key=lambda pair: pair[0].task_id)
    return tuple(item[0] for item in paired), tuple(item[1] for item in paired)


def _surface_semantic_plan(
    plan: TeacherPlan,
    source_task_id: str,
    config: SurfaceDiversityConfig,
    *,
    variant_offset: int = 0,
) -> TeacherPlan:
    frames = []
    for frame in plan.frames:
        cells = []
        for cell in frame.cells:
            role = max(cell.roles, key=cell.roles.__getitem__, default=None)
            role_name = getattr(role, "value", str(role)) if role is not None else "default"
            prefixes = _SEMANTIC_ROLE_PREFIXES.get(role_name, _SEMANTIC_ROLE_PREFIXES["default"])
            digest = hashlib.sha256(
                (
                    f"{config.seed}|{variant_offset}|{source_task_id}|"
                    f"{frame.phase}|{cell.cell_id}"
                ).encode()
            ).digest()
            prefix = prefixes[digest[0] % len(prefixes)]
            suffix = _SEMANTIC_STAGE_SUFFIXES[digest[1] % len(_SEMANTIC_STAGE_SUFFIXES)]
            candidate = f"{prefix}{cell.semantic_text}{suffix}"
            if len(candidate) > config.semantic_text_cap:
                candidate = f"{prefix}{cell.semantic_text}"
            if len(candidate) > config.semantic_text_cap:
                candidate = cell.semantic_text
            cells.append(replace(cell, semantic_text=candidate))
        frames.append(replace(frame, cells=tuple(cells)))
    return replace(plan, frames=tuple(frames))


def _select_tasks(
    source_tasks: tuple[TeacherTask, ...],
    config: SurfaceDiversityConfig,
) -> tuple[TeacherTask, ...]:
    if config.max_tasks is None or config.max_tasks >= len(source_tasks):
        return source_tasks
    ranked = sorted(
        source_tasks,
        key=lambda task: hashlib.sha256(f"{config.seed}|{task.task_id}".encode()).digest(),
    )
    return tuple(ranked[: config.max_tasks])


def _load_selected_plans(path: str | Path, selected_ids: set[str]) -> tuple[TeacherPlan, ...]:
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


def _surface_prompt(task: TeacherTask, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{task.task_id}".encode()).digest()
    preamble = _PREAMBLES[digest[0] % len(_PREAMBLES)]
    dependency = _DEPENDENCY_CLAUSES[digest[1] % len(_DEPENDENCY_CLAUSES)]
    source = _SOURCE_CLAUSES[digest[2] % len(_SOURCE_CLAUSES)]
    output = _OUTPUT_CLAUSES[digest[3] % len(_OUTPUT_CLAUSES)]
    core = " ".join(task.prompt.split())
    arrangements = (
        (preamble, core, dependency),
        (preamble, core, source),
        (source, core, output),
        (dependency, core, output),
        (core, source, dependency),
        (core, dependency, output),
    )
    return " ".join(arrangements[digest[4] % len(arrangements)])


def _normalized_surface_signature(prompt: str) -> str:
    value = prompt.casefold()
    value = _ID_RE.sub("<id>", value)
    value = _NUMBER_RE.sub("<n>", value)
    return _SPACE_RE.sub(" ", value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

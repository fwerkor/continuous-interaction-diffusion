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
    TeacherEvidence,
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
from cid.grounding import ObjectKind, ObjectRef
from cid.state import CognitiveRole
from cid.surface_diversity_training import SurfaceDiversityConfig, _surface_semantic_plan


@dataclass(frozen=True, slots=True)
class NaturalInteractionConfig:
    thought_capacity: int = 8
    variants_per_task: int = 2
    min_delay_steps: int = 1
    max_delay_steps: int = 4
    seed: int = 20260813
    semantic_text_cap: int = 144

    def __post_init__(self) -> None:
        if self.thought_capacity <= 0:
            raise ValueError("thought_capacity must be positive")
        if self.variants_per_task <= 0:
            raise ValueError("variants_per_task must be positive")
        if self.min_delay_steps <= 0 or self.max_delay_steps < self.min_delay_steps:
            raise ValueError("invalid delay range")
        if self.semantic_text_cap <= 0:
            raise ValueError("semantic_text_cap must be positive")


@dataclass(frozen=True, slots=True)
class _ToolProfile:
    name: str
    search_name: str
    search_argument: str
    read_name: str
    read_argument: str
    distractor_name: str


_TOOL_PROFILES = (
    _ToolProfile(
        "documents", "document_search", "query", "document_read", "document_id", "web_lookup"
    ),
    _ToolProfile(
        "knowledge", "knowledge_search", "text", "knowledge_fetch", "record_id", "catalog_search"
    ),
    _ToolProfile("corpus", "corpus_query", "terms", "corpus_open", "item_id", "archive_lookup"),
    _ToolProfile(
        "index", "index_lookup", "request", "resource_fetch", "resource_key", "general_search"
    ),
    _ToolProfile(
        "archive", "archive_search", "question", "archive_read", "entry_id", "directory_search"
    ),
    _ToolProfile(
        "evidence", "evidence_search", "query", "evidence_read", "evidence_id", "broad_search"
    ),
)

_OUTPUT_INSTRUCTIONS = (
    "Give the answer first, then briefly explain which retrieved facts support it.",
    "Respond with the conclusion followed by a concise evidence-grounded explanation.",
    "State the answer, then summarize the task-local records that justify it.",
    "Return a short grounded answer: conclusion first, supporting evidence second.",
    (
        "After resolving the records, provide the answer and a compact explanation of the "
        "evidence chain."
    ),
    (
        "Answer directly and include a brief summary of the retrieved facts used to reach the "
        "conclusion."
    ),
)

_PREFIX_RE = re.compile(
    r"^(?:Conclusion|Current observation|Observed state|Available evidence|Evidence summary|"
    r"Observed value|Percept state|Resolved conclusion|Current conclusion|Answer state|"
    r"Conclusion summary|Resolved answer|Final-state conclusion):\s*",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


def build_natural_interaction_augmentation(
    source_pairs: tuple[tuple[str | Path, str | Path], ...],
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: NaturalInteractionConfig | None = None,
) -> dict[str, Any]:
    config = config or NaturalInteractionConfig()
    selected: list[tuple[TeacherTask, TeacherPlan]] = []
    source_hashes: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()

    for tasks_path, plans_path in source_pairs:
        task_path = Path(tasks_path)
        plan_path = Path(plans_path)
        tasks = {task.task_id: task for task in load_teacher_tasks(task_path)}
        plans = _load_plans(plan_path)
        for plan in plans:
            task = tasks.get(plan.task_id)
            if task is None:
                raise ValueError(f"accepted plan {plan.task_id!r} has no source task")
            if str(task.metadata.get("training_mode", "")) != "tool_required":
                continue
            if task.task_id in seen_source_ids:
                raise ValueError(f"duplicate natural source task ID: {task.task_id}")
            seen_source_ids.add(task.task_id)
            selected.append((task, plan))
        source_hashes.append(
            {
                "tasks": str(task_path),
                "tasks_sha256": _sha256(task_path),
                "plans": str(plan_path),
                "plans_sha256": _sha256(plan_path),
            }
        )

    tasks: list[TeacherTask] = []
    plans: list[TeacherPlan] = []
    source_by_augmented_id: dict[str, tuple[TeacherTask, TeacherPlan]] = {}
    profile_counts: Counter[str] = Counter()
    target_lengths: list[int] = []
    for source_task, source_plan in selected:
        task, plan, profile = _augment_pair(source_task, source_plan, config)
        tasks.append(task)
        plans.append(plan)
        source_by_augmented_id[task.task_id] = (source_task, source_plan)
        profile_counts[profile.name] += 1
        target_lengths.append(len(plan.final_answer))

    paired = sorted(zip(tasks, plans, strict=True), key=lambda pair: pair[0].task_id)
    tasks_tuple = tuple(pair[0] for pair in paired)
    plans_tuple = tuple(pair[1] for pair in paired)
    reviews = review_teacher_plans(tasks_tuple, plans_tuple)
    rejected = tuple(review for review in reviews if not review.accepted)
    semantic_text_retry_plans = 0
    semantic_text_fallback_plans = 0
    if rejected:
        rejected_ids = {review.task_id for review in rejected}
        repaired_plans: list[TeacherPlan] = []
        for plan in plans_tuple:
            if plan.task_id not in rejected_ids:
                repaired_plans.append(plan)
                continue
            source_task, source_plan = source_by_augmented_id[plan.task_id]
            replacement: TeacherPlan | None = None
            for retry in range(1, 32):
                candidate_task, candidate_plan, _ = _augment_pair(
                    source_task,
                    source_plan,
                    config,
                    semantic_variant_offset=retry,
                )
                if review_teacher_plans((candidate_task,), (candidate_plan,))[0].accepted:
                    replacement = candidate_plan
                    semantic_text_retry_plans += 1
                    break
            if replacement is None:
                fallback_task, fallback_plan, _ = _augment_pair(
                    source_task,
                    source_plan,
                    config,
                    diversify_semantic_text=False,
                )
                if not review_teacher_plans((fallback_task,), (fallback_plan,))[0].accepted:
                    repaired_plans.append(plan)
                    continue
                replacement = fallback_plan
                semantic_text_fallback_plans += 1
            repaired_plans.append(replacement)
        plans_tuple = tuple(repaired_plans)
        reviews = review_teacher_plans(tasks_tuple, plans_tuple)
        rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        raise RuntimeError(
            f"natural interaction review rejected {len(rejected)} plans: "
            f"{[item.to_dict() for item in rejected[:8]]}"
        )

    trajectories = compile_teacher_plans(
        tasks_tuple,
        plans_tuple,
        TeacherScheduleConfig(
            thought_capacity=config.thought_capacity,
            min_delay_steps=config.min_delay_steps,
            max_delay_steps=config.max_delay_steps,
            variants_per_task=config.variants_per_task,
            seed=config.seed,
        ),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output / "natural-interaction-v1-teacher-tasks.jsonl",
        "causal_jobs": output / "natural-interaction-v1-causal-jobs.jsonl",
        "plans": output / "natural-interaction-v1-teacher-plans.accepted.jsonl",
        "reviews": output / "natural-interaction-v1-teacher-review.jsonl",
        "trajectories": output / "natural-interaction-v1-trajectories.jsonl",
        "trajectory_manifest": output / "natural-interaction-v1-trajectories.manifest.json",
    }
    dump_teacher_tasks(tasks_tuple, paths["tasks"])
    dump_causal_teacher_jobs(tasks_tuple, paths["causal_jobs"])
    dump_teacher_plans(plans_tuple, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])
    trajectory_manifest = inspect_dataset(paths["trajectories"])
    dump_dataset_manifest(trajectory_manifest, paths["trajectory_manifest"])

    sorted_lengths = sorted(target_lengths)
    reference_path = Path(reference_manifest_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "name": "natural-grounded-interaction-v1",
        "version": 1,
        "generator": "cid.natural_interaction_training.v1",
        "seed": config.seed,
        "source_semantic_tasks": len(selected),
        "semantic_tasks": len(tasks_tuple),
        "accepted_plans": len(plans_tuple),
        "review_rejected": 0,
        "compiled_trajectories": trajectory_manifest.examples,
        "compiled_transitions": trajectory_manifest.transitions,
        "compiled_bootstrap_transitions": trajectory_manifest.bootstrap_transitions,
        "compiled_training_transitions": trajectory_manifest.training_transitions,
        "thought_capacity_required": config.thought_capacity,
        "tool_schema_profiles": dict(sorted(profile_counts.items())),
        "semantic_text_retry_plans": semantic_text_retry_plans,
        "semantic_text_fallback_plans": semantic_text_fallback_plans,
        "long_form_targets": sum(length >= 80 for length in target_lengths),
        "target_chars_p50": _quantile(sorted_lengths, 0.50),
        "target_chars_p95": _quantile(sorted_lengths, 0.95),
        "target_chars_max": max(sorted_lengths, default=0),
        "tasks_with_anchor": sum(
            any(cell.anchors for frame in plan.frames for cell in frame.cells)
            for plan in plans_tuple
        ),
        "tasks_with_link": sum(
            any(cell.links for frame in plan.frames for cell in frame.cells)
            for plan in plans_tuple
        ),
        "source_pairs": source_hashes,
        "tasks_sha256": _sha256(paths["tasks"]),
        "causal_jobs_sha256": _sha256(paths["causal_jobs"]),
        "plans_sha256": _sha256(paths["plans"]),
        "review_sha256": _sha256(paths["reviews"]),
        "compiled_sha256": trajectory_manifest.sha256,
        "compiler": {
            "variants_per_task": config.variants_per_task,
            "min_delay_steps": config.min_delay_steps,
            "max_delay_steps": config.max_delay_steps,
            "seed": config.seed,
        },
    }
    reference_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trajectory_raw = json.loads(paths["trajectory_manifest"].read_text(encoding="utf-8"))
    trajectory_raw.update(
        {
            "name": "natural-interaction-v1-trajectories",
            "reference_manifest": str(reference_path),
            "thought_capacity_required": config.thought_capacity,
        }
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(trajectory_raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _augment_pair(
    source_task: TeacherTask,
    source_plan: TeacherPlan,
    config: NaturalInteractionConfig,
    *,
    semantic_variant_offset: int = 0,
    diversify_semantic_text: bool = True,
) -> tuple[TeacherTask, TeacherPlan, _ToolProfile]:
    digest = hashlib.sha256(f"{config.seed}|{source_task.task_id}".encode()).digest()
    profile = _TOOL_PROFILES[digest[0] % len(_TOOL_PROFILES)]
    task_id = f"natural-interaction-v1-{source_task.task_id}"
    response = _grounded_response(source_task, source_plan, digest[1])
    source_map = {
        "workspace_search": (profile.search_name, {"query": profile.search_argument}),
        "workspace_read": (profile.read_name, {"resource_id": profile.read_argument}),
    }

    descriptors = tuple(
        _remap_descriptor(item, source_map) for item in source_task.source_descriptors
    )
    descriptors += (
        {
            "name": profile.distractor_name,
            "description": (
                "Search a broad external index that may not contain this task-local corpus."
            ),
            "arguments": ({"name": "query", "kind": "string", "required": True},),
            "cacheable": True,
            "dynamic": False,
            "versioned": False,
        },
    )
    evidence = tuple(_remap_evidence(item, source_map) for item in source_task.evidence)
    prompt = (
        f"{source_task.prompt.rstrip()}\n\n"
        f"{_OUTPUT_INSTRUCTIONS[digest[2] % len(_OUTPUT_INSTRUCTIONS)]}"
    )
    task = replace(
        source_task,
        task_id=task_id,
        prompt=prompt,
        source_descriptors=descriptors,
        evidence=evidence,
        metadata={
            **dict(source_task.metadata),
            "source_task_kind": str(source_task.metadata.get("task_kind", "")),
            "task_kind": "natural_grounded_interaction",
            "augmentation": "natural_grounded_interaction",
            "source_task_id": source_task.task_id,
            "tool_schema_profile": profile.name,
            "output_policy": "answer_then_grounded_evidence_summary",
            "canonical_reference_answer": source_plan.final_answer,
            "generated_by": "cid.natural_interaction_training.v1",
        },
        reference_answer=response,
    )

    surface_config = SurfaceDiversityConfig(
        component_name="natural-grounded-interaction-v1",
        file_stem="natural-interaction-v1",
        thought_capacity=config.thought_capacity,
        seed=config.seed,
        surface_version=4,
        diversify_prompt=False,
        diversify_semantic_text=True,
        rewrite_semantic_text=True,
        semantic_text_cap=config.semantic_text_cap,
    )
    plan = replace(source_plan, task_id=task_id)
    if diversify_semantic_text:
        plan = _surface_semantic_plan(
            plan,
            source_task.task_id,
            surface_config,
            variant_offset=semantic_variant_offset,
        )
    plan = _remap_plan_sources(plan, source_map)
    frames = list(plan.frames)
    frames[-1] = replace(frames[-1], display=response)
    plan = replace(plan, final_answer=response, frames=tuple(frames))
    return task, plan, profile


def _grounded_response(task: TeacherTask, plan: TeacherPlan, selector: int) -> str:
    canonical = _SPACE_RE.sub(" ", plan.final_answer).strip()
    final_frame = plan.frames[-1]
    evidence_notes: list[str] = []
    for cell in final_frame.cells:
        if cell.roles.get(CognitiveRole.CONCLUSION, 0.0) > 0.0:
            continue
        if "search" in cell.cell_id.casefold():
            continue
        if cell.roles.get(CognitiveRole.INFORMATION_NEED, 0.0) > 0.5:
            continue
        if max(
            cell.roles.get(CognitiveRole.PERCEPT, 0.0),
            cell.roles.get(CognitiveRole.HYPOTHESIS, 0.0),
        ) <= 0.0:
            continue
        note = _clean_note(cell.semantic_text)
        if note.casefold().startswith("relevant records:"):
            continue
        if note and note.casefold() != canonical.casefold() and note not in evidence_notes:
            evidence_notes.append(note)
        if len(evidence_notes) >= 3:
            break
    if not evidence_notes:
        evidence_notes.extend(_evidence_fallback_notes(task))

    joined = "; ".join(evidence_notes[:3])
    if selector % 3 == 0:
        response = f"Answer: {canonical}\nEvidence: {joined}"
    elif selector % 3 == 1:
        response = f"{canonical.rstrip('.!?')}. The supporting task-local records show: {joined}"
    else:
        response = f"Conclusion: {canonical}\nSupporting evidence: {joined}"
    response = _SPACE_RE.sub(" ", response.replace("\n", " \n ")).replace(" \n ", "\n").strip()
    if len(response) < 80:
        response += (
            " This conclusion follows from the retrieved task-local records rather than an "
            "unsupported guess."
        )
    if len(response) > 480:
        response = response[:477].rstrip(" ,;:-") + "..."
    return response


def _evidence_fallback_notes(task: TeacherTask) -> list[str]:
    notes: list[str] = []
    for evidence in task.evidence:
        value = evidence.value
        if not isinstance(value, dict):
            continue
        title = str(value.get("title", "")).strip()
        sentences = [str(item).strip() for item in value.get("sentences", ()) if str(item).strip()]
        if not sentences:
            continue
        note = f"{title}: {sentences[0]}" if title else sentences[0]
        notes.append(_SPACE_RE.sub(" ", note)[:180].rstrip())
        if len(notes) >= 3:
            break
    return notes or [
        "The required task-local records were retrieved and integrated before answering."
    ]


def _clean_note(text: str) -> str:
    value = _PREFIX_RE.sub("", text).strip()
    value = re.sub(r"\s*\[(?:current|tracked|task-local|state|active context)\]\s*$", "", value)
    return _SPACE_RE.sub(" ", value).strip(" ;")[:180]


def _remap_descriptor(
    descriptor: dict[str, Any] | Any,
    source_map: dict[str, tuple[str, dict[str, str]]],
) -> dict[str, Any]:
    raw = dict(descriptor)
    old_source = str(raw.get("name", ""))
    new_source, argument_map = source_map.get(old_source, (old_source, {}))
    raw["name"] = new_source
    raw["arguments"] = tuple(
        {
            **dict(argument),
            "name": argument_map.get(
                str(argument.get("name", "")), str(argument.get("name", ""))
            ),
        }
        for argument in raw.get("arguments", ())
    )
    if old_source == "workspace_search":
        raw["description"] = (
            "Search the task-local document collection and return candidate records."
        )
    elif old_source == "workspace_read":
        raw["description"] = "Read one task-local record selected from the document collection."
    return raw


def _remap_evidence(
    evidence: TeacherEvidence,
    source_map: dict[str, tuple[str, dict[str, str]]],
) -> TeacherEvidence:
    new_source, argument_map = source_map.get(evidence.source, (evidence.source, {}))
    return replace(
        evidence,
        source=new_source,
        arguments={argument_map.get(key, key): value for key, value in evidence.arguments.items()},
    )


def _remap_plan_sources(
    plan: TeacherPlan,
    source_map: dict[str, tuple[str, dict[str, str]]],
) -> TeacherPlan:
    needs = []
    for need in plan.needs:
        new_source, argument_map = source_map.get(need.source, (need.source, {}))
        needs.append(
            replace(
                need,
                source=new_source,
                arguments={
                    argument_map.get(key, key): value for key, value in need.arguments.items()
                },
            )
        )
    frames = []
    for frame in plan.frames:
        cells = []
        for cell in frame.cells:
            links = []
            for link in cell.links:
                target = link.target
                if target.kind is ObjectKind.SOURCE and target.identifier in source_map:
                    target = ObjectRef.source(source_map[target.identifier][0])
                links.append(replace(link, target=target))
            cells.append(replace(cell, links=tuple(links)))
        frames.append(replace(frame, cells=tuple(cells)))
    return replace(plan, needs=tuple(needs), frames=tuple(frames))


def _load_plans(path: Path) -> tuple[TeacherPlan, ...]:
    plans: list[TeacherPlan] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                plans.append(TeacherPlan.from_dict(json.loads(line)))
    return tuple(plans)


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

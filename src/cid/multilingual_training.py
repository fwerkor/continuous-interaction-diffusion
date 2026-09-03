from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import DISPLAY_UNKNOWN_MARKER, dump_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
from cid.distill import (
    TeacherCellPlan,
    TeacherEvidence,
    TeacherFrame,
    TeacherNeed,
    TeacherPlan,
    TeacherScheduleConfig,
    TeacherTask,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_tasks,
    review_teacher_plans,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole

MULTILINGUAL_FAMILIES = (
    "zh_prompt_en_catalog_zh_roster",
    "en_prompt_zh_index_en_dispatch",
    "ja_prompt_en_registry_ja_support",
    "es_prompt_fr_directory_es_roster",
)


@dataclass(frozen=True, slots=True)
class MultilingualTrainingConfig:
    zh_tasks: int = 450
    en_zh_tasks: int = 300
    ja_tasks: int = 225
    es_tasks: int = 225
    schedule_variants: int = 2
    seed: int = 20260813

    def __post_init__(self) -> None:
        counts = (self.zh_tasks, self.en_zh_tasks, self.ja_tasks, self.es_tasks)
        if any(count <= 0 for count in counts):
            raise ValueError("multilingual family counts must be positive")
        if self.schedule_variants <= 0:
            raise ValueError("schedule_variants must be positive")

    @property
    def total_tasks(self) -> int:
        return self.zh_tasks + self.en_zh_tasks + self.ja_tasks + self.es_tasks


def build_multilingual_training(
    output_dir: str | Path,
    config: MultilingualTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or MultilingualTrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks = generate_multilingual_tasks(config)
    plans = tuple(_plan_for(task) for task in tasks)
    reviews = review_teacher_plans(tasks, plans)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = "; ".join(
            f"{review.task_id}: {', '.join(review.reasons)}" for review in rejected[:8]
        )
        raise ValueError(f"generated multilingual plans failed review: {detail}")

    trajectories = compile_teacher_plans(
        tasks,
        plans,
        TeacherScheduleConfig(
            thought_capacity=8,
            min_delay_steps=1,
            max_delay_steps=4,
            variants_per_task=config.schedule_variants,
            seed=config.seed + 17,
        ),
    )

    tasks_path = output / "teacher-tasks-v1.jsonl"
    plans_path = output / "teacher-plans-v1.accepted.jsonl"
    trajectories_path = output / "trajectories-v1.jsonl"
    trajectory_manifest_path = output / "trajectories-v1.manifest.json"
    manifest_path = output / "reference-manifest-v1.json"

    dump_teacher_tasks(tasks, tasks_path)
    dump_teacher_plans(plans, plans_path)
    dump_jsonl(trajectories, trajectories_path)
    trajectory_manifest = inspect_dataset(trajectories_path)
    dump_dataset_manifest(trajectory_manifest, trajectory_manifest_path)

    family_counts = Counter(str(task.metadata["family"]) for task in tasks)
    prompt_languages = Counter(str(task.metadata["prompt_language"]) for task in tasks)
    answer_languages = Counter(str(task.metadata["answer_language"]) for task in tasks)
    source_language_pairs = Counter(
        "->".join(str(item) for item in task.metadata["source_languages"]) for task in tasks
    )
    manifest = {
        "format_version": 1,
        "name": "multilingual-v1",
        "seed": config.seed,
        "semantic_tasks": len(tasks),
        "accepted_teacher_plans": len(plans),
        "compiled_trajectories": len(trajectories),
        "schedule_variants": config.schedule_variants,
        "family_counts": dict(sorted(family_counts.items())),
        "prompt_language_counts": dict(sorted(prompt_languages.items())),
        "answer_language_counts": dict(sorted(answer_languages.items())),
        "source_language_pair_counts": dict(sorted(source_language_pairs.items())),
        "cross_lingual_tasks": sum(bool(task.metadata.get("cross_lingual")) for task in tasks),
        "canonical_tct_language": "en",
        "tasks_sha256": _file_sha256(tasks_path),
        "plans_sha256": _file_sha256(plans_path),
        "trajectories_sha256": trajectory_manifest.sha256,
        "transitions": trajectory_manifest.transitions,
        "thought_capacity_required": trajectory_manifest.thought_capacity_required,
        "max_trajectory_steps": trajectory_manifest.max_trajectory_steps,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_multilingual_tasks(
    config: MultilingualTrainingConfig | None = None,
) -> tuple[TeacherTask, ...]:
    config = config or MultilingualTrainingConfig()
    rng = random.Random(config.seed)
    tasks: list[TeacherTask] = []
    for index in range(config.zh_tasks):
        tasks.append(_zh_en_zh_task(rng, index))
    for index in range(config.en_zh_tasks):
        tasks.append(_en_zh_en_task(rng, index))
    for index in range(config.ja_tasks):
        tasks.append(_ja_en_ja_task(rng, index))
    for index in range(config.es_tasks):
        tasks.append(_es_fr_es_task(rng, index))
    tasks.sort(key=lambda task: task.task_id)
    return tuple(tasks)


def _zh_en_zh_task(rng: random.Random, index: int) -> TeacherTask:
    project = f"PRJ-ZH-{index:04d}"
    team = f"Team-{rng.choice(('Aurora', 'Birch', 'Cedar', 'Delta', 'Ember'))}-{index:04d}"
    engineer = rng.choice(("林澈", "周宁", "沈岚", "顾言", "唐禾", "许川", "苏遥", "陆清"))
    prompt = (
        f"请先查询英文项目目录中 {project} 的负责团队，再用得到的团队代号查询中文值班表。"
        "回答当前值班工程师姓名，只给出姓名。"
    )
    return _two_hop_task(
        family="zh_prompt_en_catalog_zh_roster",
        index=index,
        prompt=prompt,
        answer=engineer,
        prompt_language="zh",
        answer_language="zh",
        first_source="catalog_en",
        first_source_language="en",
        first_argument_name="project",
        first_argument_value=project,
        first_value=team,
        second_source="roster_zh",
        second_source_language="zh",
        second_argument_name="team",
        second_value=engineer,
        first_need="Need the English catalog entry for the requested project.",
        first_percept=f"The catalog resolves the project to team {team}.",
        second_need="Need the Chinese roster entry for the resolved team.",
        second_percept=f"The Chinese roster returns engineer {engineer}.",
        goal="Resolve the project through two sources and answer in Chinese.",
    )


def _en_zh_en_task(rng: random.Random, index: int) -> TeacherTask:
    sku = f"SKU-CN-{index:04d}"
    warehouse = f"华东仓-{index:04d}"
    city = rng.choice(("Singapore", "Osaka", "Rotterdam", "Vancouver", "Auckland", "Dublin"))
    prompt = (
        f"Look up {sku} in the Chinese product index to obtain its warehouse code, then query "
        "the English dispatch table with that code. Return the dispatch city only."
    )
    return _two_hop_task(
        family="en_prompt_zh_index_en_dispatch",
        index=index,
        prompt=prompt,
        answer=city,
        prompt_language="en",
        answer_language="en",
        first_source="product_index_zh",
        first_source_language="zh",
        first_argument_name="sku",
        first_argument_value=sku,
        first_value=warehouse,
        second_source="dispatch_en",
        second_source_language="en",
        second_argument_name="warehouse",
        second_value=city,
        first_need="Need the Chinese product-index record for the requested SKU.",
        first_percept=f"The Chinese index resolves the SKU to warehouse {warehouse}.",
        second_need="Need the English dispatch entry for the resolved warehouse.",
        second_percept=f"The dispatch table returns city {city}.",
        goal="Bridge Chinese source evidence into an English answer.",
    )


def _ja_en_ja_task(rng: random.Random, index: int) -> TeacherTask:
    package = f"PKG-JA-{index:04d}"
    channel = f"channel-{rng.choice(('stable', 'rapid', 'lts', 'edge'))}-{index:04d}"
    desk = rng.choice(("東京窓口", "大阪窓口", "札幌窓口", "福岡窓口", "名古屋窓口"))
    prompt = (
        f"{package} について、まず英語の release registry で配布チャネルを確認し、そのチャネルを"
        "日本語のサポート表で検索してください。担当窓口だけを日本語で答えてください。"
    )
    return _two_hop_task(
        family="ja_prompt_en_registry_ja_support",
        index=index,
        prompt=prompt,
        answer=desk,
        prompt_language="ja",
        answer_language="ja",
        first_source="release_registry_en",
        first_source_language="en",
        first_argument_name="package",
        first_argument_value=package,
        first_value=channel,
        second_source="support_ja",
        second_source_language="ja",
        second_argument_name="channel",
        second_value=desk,
        first_need="Need the English release-registry entry for the requested package.",
        first_percept=f"The registry resolves the package to channel {channel}.",
        second_need="Need the Japanese support entry for the resolved channel.",
        second_percept=f"The Japanese support table returns desk {desk}.",
        goal="Resolve English registry evidence and answer in Japanese.",
    )


def _es_fr_es_task(rng: random.Random, index: int) -> TeacherTask:
    service = f"SRV-ES-{index:04d}"
    department = f"département-{rng.choice(('alpha', 'bleu', 'cèdre', 'delta'))}-{index:04d}"
    office = rng.choice(("Madrid", "Sevilla", "Valencia", "Bilbao", "Málaga", "Zaragoza"))
    prompt = (
        f"Consulta {service} en el directorio francés para obtener el departamento responsable. "
        "Después usa ese departamento en la tabla española de guardias. "
        "Responde solo con la ciudad."
    )
    return _two_hop_task(
        family="es_prompt_fr_directory_es_roster",
        index=index,
        prompt=prompt,
        answer=office,
        prompt_language="es",
        answer_language="es",
        first_source="directory_fr",
        first_source_language="fr",
        first_argument_name="service",
        first_argument_value=service,
        first_value=department,
        second_source="roster_es",
        second_source_language="es",
        second_argument_name="department",
        second_value=office,
        first_need="Need the French directory entry for the requested service.",
        first_percept=f"The French directory resolves the service to {department}.",
        second_need="Need the Spanish roster entry for the resolved department.",
        second_percept=f"The Spanish roster returns city {office}.",
        goal="Bridge French source evidence into a Spanish answer.",
    )


def _two_hop_task(
    *,
    family: str,
    index: int,
    prompt: str,
    answer: str,
    prompt_language: str,
    answer_language: str,
    first_source: str,
    first_source_language: str,
    first_argument_name: str,
    first_argument_value: str,
    first_value: str,
    second_source: str,
    second_source_language: str,
    second_argument_name: str,
    second_value: str,
    first_need: str,
    first_percept: str,
    second_need: str,
    second_percept: str,
    goal: str,
) -> TeacherTask:
    task_id = f"multilingual-{family}-{index:05d}"
    return TeacherTask(
        task_id=task_id,
        prompt=prompt,
        source_descriptors=(
            _lookup_descriptor(first_source, first_source_language, first_argument_name),
            _lookup_descriptor(second_source, second_source_language, second_argument_name),
        ),
        evidence=(
            TeacherEvidence(
                evidence_id="bridge",
                source=first_source,
                value=first_value,
                arguments={first_argument_name: first_argument_value},
                provenance="cid.multilingual_training.v1",
            ),
            TeacherEvidence(
                evidence_id="answer",
                source=second_source,
                value=second_value,
                arguments={second_argument_name: first_value},
                depends_on=("bridge",),
                provenance="cid.multilingual_training.v1",
            ),
        ),
        metadata={
            "task_kind": "cross_lingual_tool_reasoning",
            "family": family,
            "interaction_pattern": "cross_lingual_sequential_two_hop",
            "dependency_depth": 2,
            "training_mode": "tool_required",
            "prompt_language": prompt_language,
            "answer_language": answer_language,
            "source_languages": [first_source_language, second_source_language],
            "cross_lingual": True,
            "canonical_tct_language": "en",
            "generated_by": "cid.multilingual_training.v1",
            "goal": goal,
            "first_need": first_need,
            "first_percept": first_percept,
            "second_need": second_need,
            "second_percept": second_percept,
        },
        reference_answer=answer,
    )


def _lookup_descriptor(name: str, language: str, argument_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Read one immutable task-local record. Source language: {language}.",
        "arguments": ({"name": argument_name, "kind": "string", "required": True},),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }


def _plan_for(task: TeacherTask) -> TeacherPlan:
    first, second = task.evidence
    meta = task.metadata
    first_arg_name, first_arg_value = next(iter(first.arguments.items()))
    answer = str(task.reference_answer)

    goal = _cell(
        "goal",
        str(meta["goal"]),
        {CognitiveRole.PLAN: 1.0, CognitiveRole.CONSTRAINT: 0.35},
        uncertainty=0.25,
        noise=0.05,
        anchors=(_anchor(str(first_arg_value), f"{task.task_id}:query"),),
    )
    first_need = _cell(
        "bridge",
        str(meta["first_need"]),
        {CognitiveRole.INFORMATION_NEED: 1.0},
        uncertainty=0.85,
        noise=0.25,
        links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(first.source), 1.0),),
    )
    first_percept = _cell(
        "bridge",
        str(meta["first_percept"]),
        {CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.08,
        noise=0.04,
        lifecycle=CellLifecycle.STABLE,
        anchors=(_anchor(str(first.value), f"{task.task_id}:bridge"),),
        links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(first.source), 1.0),),
    )
    second_need = _cell(
        "answer-source",
        str(meta["second_need"]),
        {CognitiveRole.INFORMATION_NEED: 1.0},
        uncertainty=0.8,
        noise=0.22,
        anchors=(_anchor(str(first.value), f"{task.task_id}:bridge"),),
        links=(
            CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("bridge"), 1.0),
            CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(second.source), 1.0),
        ),
    )
    second_percept = _cell(
        "answer-source",
        str(meta["second_percept"]),
        {CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.06,
        noise=0.03,
        lifecycle=CellLifecycle.STABLE,
        anchors=(_anchor(answer, f"{task.task_id}:answer-value"),),
        links=(
            CognitiveLink(LinkRelation.DEPENDS_ON, ObjectRef.cell("bridge"), 1.0),
            CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(second.source), 1.0),
        ),
    )
    conclusion = _cell(
        "conclusion",
        "The requested answer is supported by the completed cross-language evidence chain.",
        {CognitiveRole.CONCLUSION: 1.0},
        uncertainty=0.02,
        noise=0.01,
        lifecycle=CellLifecycle.STABLE,
        anchors=(_anchor(answer, f"{task.task_id}:answer-value"),),
        links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("answer-source"), 1.0),),
    )

    return TeacherPlan(
        task_id=task.task_id,
        final_answer=answer,
        frames=(
            TeacherFrame(phase="initial", display=DISPLAY_UNKNOWN_MARKER, cells=(goal,)),
            TeacherFrame(
                phase="pre",
                display=DISPLAY_UNKNOWN_MARKER,
                cells=(goal, first_need),
            ),
            TeacherFrame(
                phase="after:bridge",
                display=DISPLAY_UNKNOWN_MARKER,
                cells=(goal, first_percept, second_need),
            ),
            TeacherFrame(
                phase="after:answer",
                display=answer,
                cells=(goal, first_percept, second_percept),
            ),
            TeacherFrame(
                phase="final",
                display=answer,
                cells=(goal, first_percept, second_percept, conclusion),
            ),
        ),
        needs=(
            TeacherNeed(
                need_id="need:bridge",
                cell_id="bridge",
                evidence_id="bridge",
                phase="pre",
                source=first.source,
                arguments=dict(first.arguments),
            ),
            TeacherNeed(
                need_id="need:answer",
                cell_id="answer-source",
                evidence_id="answer",
                phase="after:bridge",
                source=second.source,
                arguments=dict(second.arguments),
            ),
        ),
    )


def _cell(
    cell_id: str,
    semantic_text: str,
    roles: dict[CognitiveRole, float],
    *,
    uncertainty: float,
    noise: float,
    lifecycle: CellLifecycle = CellLifecycle.ACTIVE,
    anchors: tuple[Anchor, ...] = (),
    links: tuple[CognitiveLink, ...] = (),
) -> TeacherCellPlan:
    return TeacherCellPlan(
        cell_id=cell_id,
        semantic_text=semantic_text,
        roles=roles,
        uncertainty=uncertainty,
        noise=noise,
        lifecycle=lifecycle,
        anchors=anchors,
        links=links,
    )


def _anchor(value: str, anchor_id: str) -> Anchor:
    return Anchor(
        anchor_id=anchor_id,
        kind=AnchorKind.ENTITY,
        value=value,
        object_id=f"synthetic:{value.casefold()}",
        confidence=1.0,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

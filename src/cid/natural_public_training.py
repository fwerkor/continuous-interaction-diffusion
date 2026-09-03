from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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
    dump_teacher_reviews,
    dump_teacher_tasks,
    review_teacher_plans,
)
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole

_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class NaturalPublicBuildConfig:
    source: str
    quota: int
    seed: int = 20260814
    thought_capacity: int = 8
    variants_per_task: int = 2
    min_delay_steps: int = 1
    max_delay_steps: int = 4

    def __post_init__(self) -> None:
        if self.source not in {"nq-open", "oasst1", "multidoc2dial", "qasper"}:
            raise ValueError(f"unsupported natural public source: {self.source}")
        if self.quota <= 0:
            raise ValueError("natural public quota must be positive")
        if self.source == "oasst1" and self.variants_per_task != 1:
            raise ValueError("oasst1 has no external timing; use one trajectory per semantic task")


def build_natural_public_component(
    *,
    registry_path: str | Path,
    output_dir: str | Path,
    reference_manifest_output: str | Path,
    config: NaturalPublicBuildConfig,
    exclude_prompts: Iterable[str] = (),
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    source_spec = next((item for item in registry["sources"] if item["id"] == config.source), None)
    if source_spec is None:
        raise ValueError(f"source {config.source!r} is not present in {registry_file}")
    expected_quota = int(source_spec["quota"])
    if config.quota != expected_quota:
        raise ValueError(f"source quota drifted: config={config.quota}, registry={expected_quota}")

    cache = Path(cache_dir or Path.home() / ".cache" / "cid-natural-public")
    cache.mkdir(parents=True, exist_ok=True)
    excluded = {_semantic_id(prompt) for prompt in exclude_prompts if str(prompt).strip()}
    records = _load_source_records(source_spec, config, cache)
    selected, dropped_duplicates, dropped_excluded = _select_records(
        records, config.quota, config.seed, excluded
    )

    tasks: list[TeacherTask] = []
    plans: list[TeacherPlan] = []
    for record in selected:
        task = _record_to_task(record, source_spec)
        plan = _plan_for_task(task, record)
        tasks.append(task)
        plans.append(plan)

    tasks_tuple = tuple(sorted(tasks, key=lambda item: item.task_id))
    plan_by_id = {plan.task_id: plan for plan in plans}
    plans_tuple = tuple(plan_by_id[task.task_id] for task in tasks_tuple)
    reviews = review_teacher_plans(tasks_tuple, plans_tuple)
    rejected = tuple(review for review in reviews if not review.accepted)
    if rejected:
        detail = "; ".join(f"{item.task_id}: {', '.join(item.reasons)}" for item in rejected[:12])
        raise ValueError(f"natural public plans failed review ({len(rejected)}): {detail}")

    trajectories = compile_teacher_plans(
        tasks_tuple,
        plans_tuple,
        TeacherScheduleConfig(
            thought_capacity=config.thought_capacity,
            variants_per_task=config.variants_per_task,
            min_delay_steps=config.min_delay_steps,
            max_delay_steps=config.max_delay_steps,
            seed=config.seed,
        ),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = config.source.replace("-", "_")
    paths = {
        "tasks": output / f"{stem}-teacher-tasks.jsonl",
        "plans": output / f"{stem}-teacher-plans.accepted.jsonl",
        "reviews": output / f"{stem}-teacher-review.jsonl",
        "trajectories": output / f"{stem}-trajectories.jsonl",
        "trajectory_manifest": output / f"{stem}-trajectories.manifest.json",
    }
    dump_teacher_tasks(tasks_tuple, paths["tasks"])
    dump_teacher_plans(plans_tuple, paths["plans"])
    dump_teacher_reviews(reviews, paths["reviews"])
    dump_jsonl(trajectories, paths["trajectories"])
    trajectory_manifest = inspect_dataset(paths["trajectories"])
    dump_dataset_manifest(trajectory_manifest, paths["trajectory_manifest"])

    mode_counts = Counter(str(task.metadata["training_mode"]) for task in tasks_tuple)
    prompt_lengths = sorted(len(task.prompt) for task in tasks_tuple)
    target_lengths = sorted(len(str(task.reference_answer or "")) for task in tasks_tuple)
    manifest = {
        "format_version": 1,
        "name": f"natural-public-{config.source}-v1",
        "version": 1,
        "source": config.source,
        "generator": "cid.natural_public_training.v1",
        "registry": str(registry_file),
        "registry_sha256": _sha256(registry_file),
        "upstream": {key: value for key, value in source_spec.items() if key not in {"quota"}},
        "semantic_tasks": len(tasks_tuple),
        "accepted_plans": len(plans_tuple),
        "review_rejected": 0,
        "dropped_semantic_duplicates": dropped_duplicates,
        "dropped_existing_prompt_overlap": dropped_excluded,
        "mode_counts": dict(sorted(mode_counts.items())),
        "prompt_chars_p50": _quantile(prompt_lengths, 0.5),
        "prompt_chars_p95": _quantile(prompt_lengths, 0.95),
        "target_chars_p50": _quantile(target_lengths, 0.5),
        "target_chars_p95": _quantile(target_lengths, 0.95),
        "tasks_with_anchor": sum(
            any(cell.anchors for frame in plan.frames for cell in frame.cells)
            for plan in plans_tuple
        ),
        "tasks_with_link": sum(
            any(cell.links for frame in plan.frames for cell in frame.cells) for plan in plans_tuple
        ),
        "compiled_trajectories": trajectory_manifest.examples,
        "compiled_transitions": trajectory_manifest.transitions,
        "compiled_bootstrap_transitions": trajectory_manifest.bootstrap_transitions,
        "compiled_training_transitions": trajectory_manifest.training_transitions,
        "thought_capacity_required": config.thought_capacity,
        "compiler": {
            "variants_per_task": config.variants_per_task,
            "min_delay_steps": config.min_delay_steps,
            "max_delay_steps": config.max_delay_steps,
            "seed": config.seed,
        },
        "tasks_path": str(paths["tasks"]),
        "tasks_sha256": _sha256(paths["tasks"]),
        "plans_path": str(paths["plans"]),
        "plans_sha256": _sha256(paths["plans"]),
        "review_path": str(paths["reviews"]),
        "review_sha256": _sha256(paths["reviews"]),
        "trajectory_path": str(paths["trajectories"]),
        "compiled_sha256": trajectory_manifest.sha256,
        "trajectory_manifest": str(paths["trajectory_manifest"]),
    }
    reference = Path(reference_manifest_output)
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trajectory_payload = json.loads(paths["trajectory_manifest"].read_text(encoding="utf-8"))
    trajectory_payload.update(
        {
            "name": f"natural-public-{config.source}-v1-trajectories",
            "reference_manifest": str(reference),
            "thought_capacity_required": config.thought_capacity,
        }
    )
    paths["trajectory_manifest"].write_text(
        json.dumps(trajectory_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def collect_unique_prompts_from_trajectories(path: str | Path) -> tuple[str, ...]:
    seen: set[str] = set()
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            metadata = raw.get("metadata", {})
            semantic_id = str(metadata.get("semantic_task_id") or raw.get("example_id", ""))
            if semantic_id in seen:
                continue
            seen.add(semantic_id)
            prompts.append(str(raw["prompt"]))
    return tuple(prompts)


def _load_source_records(
    source: Mapping[str, Any],
    config: NaturalPublicBuildConfig,
    cache: Path,
) -> list[dict[str, Any]]:
    if config.source == "nq-open":
        return _load_nq_open(source)
    if config.source == "oasst1":
        return _load_oasst1(source)
    if config.source == "multidoc2dial":
        return _load_multidoc2dial(source, cache)
    if config.source == "qasper":
        return _load_qasper(source, cache)
    raise AssertionError(config.source)


def _load_nq_open(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=str(source["repo"]),
        repo_type="dataset",
        revision=str(source["revision"]),
        filename=str(source["file"]),
    )
    frame = pd.read_parquet(path)
    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(frame.to_dict(orient="records")):
        prompt = _clean(str(row["question"]))
        answers = [_clean(str(item)) for item in list(row["answer"]) if _clean(str(item))]
        answers = list(dict.fromkeys(answers))
        if not prompt or not answers:
            continue
        records.append(
            {
                "row_key": f"{source['file']}:{row_number}",
                "prompt": prompt,
                "answer": answers[0],
                "aliases": answers,
                "mode": "tool_required",
                "task_kind": "natural_open_qa",
                "tool_source": "knowledge_search",
                "tool_arguments": {"query": prompt},
                "evidence": {"query": prompt, "answers": answers[:8]},
                "percept": f"Retrieved answer: {answers[0]}",
                "metadata": {},
            }
        )
    return records


def _load_oasst1(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=str(source["repo"]),
        repo_type="dataset",
        revision=str(source["revision"]),
        filename=str(source["file"]),
    )
    frame = pd.read_parquet(path)
    rows = {str(row.message_id): row for row in frame.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        if row.role != "assistant" or row.lang != "en" or bool(row.deleted) or bool(row.synthetic):
            continue
        if not bool(row.review_result) or not _rank_zero(row.rank):
            continue
        parent = rows.get(str(row.parent_id))
        if parent is None or parent.role != "prompter" or parent.lang != "en":
            continue
        if bool(parent.deleted) or not bool(parent.review_result):
            continue
        if not _oasst_quality_ok(row):
            continue
        target = _clean_multiline(str(row.text))
        if not 20 <= len(target) <= 4000:
            continue
        chain = _oasst_chain(rows, str(parent.message_id), max_messages=7)
        if not chain or chain[-1][0] != "prompter":
            continue
        prompt = _conversation_prompt(chain)
        if not 20 <= len(prompt) <= 7000:
            continue
        records.append(
            {
                "row_key": str(row.message_id),
                "prompt": prompt,
                "answer": target,
                "mode": "no_tool",
                "task_kind": "natural_instruction",
                "metadata": {
                    "message_tree_id": str(row.message_tree_id),
                    "conversation_messages": len(chain),
                },
            }
        )
    return records


def _load_multidoc2dial(source: Mapping[str, Any], cache: Path) -> list[dict[str, Any]]:
    archive = _download_verified(source, cache)
    extract = cache / "multidoc2dial-extracted"
    if not extract.exists():
        extract.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract)
    root = extract / "multidoc2dial"
    dialogues = json.loads((root / "multidoc2dial_dial_train.json").read_text(encoding="utf-8"))[
        "dial_data"
    ]
    documents = json.loads((root / "multidoc2dial_doc.json").read_text(encoding="utf-8"))[
        "doc_data"
    ]
    records: list[dict[str, Any]] = []
    for domain, domain_dialogues in dialogues.items():
        doc_map = documents[domain]
        for dialogue in domain_dialogues:
            turns = dialogue["turns"]
            for index, turn in enumerate(turns):
                if (
                    turn.get("role") != "agent"
                    or index == 0
                    or turns[index - 1].get("role") != "user"
                ):
                    continue
                refs = list(turn.get("references") or ())
                target = _clean_multiline(str(turn.get("utterance", "")))
                if not refs or not target:
                    continue
                history = [
                    (str(item["role"]), _clean_multiline(str(item["utterance"])))
                    for item in turns[max(0, index - 6) : index]
                    if _clean_multiline(str(item.get("utterance", "")))
                ]
                prompt = _conversation_prompt(history)
                if not prompt:
                    continue
                excerpts: list[dict[str, str]] = []
                seen: set[tuple[str, str]] = set()
                for ref in refs:
                    doc_id = str(ref["doc_id"])
                    span_id = str(ref["id_sp"])
                    doc = doc_map.get(doc_id)
                    if doc is None:
                        continue
                    span = doc.get("spans", {}).get(span_id)
                    if span is None:
                        continue
                    text = _clean(str(span.get("text_sec") or span.get("text_sp") or ""))
                    if len(text) < 20:
                        continue
                    key = (doc_id, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    excerpts.append({"title": _clean(str(doc["title"])), "text": text[:1600]})
                if not excerpts:
                    continue
                query = _clean(str(turns[index - 1]["utterance"]))
                records.append(
                    {
                        "row_key": f"{dialogue['dial_id']}:{turn['turn_id']}",
                        "prompt": prompt,
                        "answer": target,
                        "mode": "tool_required",
                        "task_kind": "natural_document_dialogue",
                        "tool_source": "document_lookup",
                        "tool_arguments": {"query": query},
                        "evidence": {"documents": excerpts[:4]},
                        "percept": _evidence_summary(excerpts, target),
                        "metadata": {
                            "domain": domain,
                            "dialogue_id": str(dialogue["dial_id"]),
                            "turn_id": int(turn["turn_id"]),
                            "reference_spans": len(refs),
                        },
                    }
                )
    return records


def _load_qasper(source: Mapping[str, Any], cache: Path) -> list[dict[str, Any]]:
    archive = _download_verified(source, cache)
    extract = cache / "qasper-extracted"
    if not extract.exists():
        extract.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(extract, filter="data")
    payload = json.loads((extract / "qasper-train-v0.3.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for paper_id, paper in payload.items():
        title = _clean(str(paper["title"]))
        for item in paper["qas"]:
            answer, evidence = _qasper_consensus(item.get("answers", ()))
            if not answer or not evidence:
                continue
            question = _clean(str(item["question"]))
            prompt = f"Paper: {title}\nQuestion: {question}"
            records.append(
                {
                    "row_key": str(item["question_id"]),
                    "prompt": prompt,
                    "answer": answer,
                    "mode": "tool_required",
                    "task_kind": "natural_paper_qa",
                    "tool_source": "paper_lookup",
                    "tool_arguments": {"paper": title, "query": question},
                    "evidence": {"paper": title, "excerpts": evidence[:4]},
                    "percept": f"Paper evidence supports: {_short(answer, 112)}",
                    "metadata": {"paper_id": str(paper_id), "paper_title": title},
                }
            )
    return records


def _select_records(
    records: Sequence[Mapping[str, Any]],
    quota: int,
    seed: int,
    excluded: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    ranked = sorted(
        records,
        key=lambda record: _stable_hash(
            f"{seed}|{record['row_key']}|{_semantic_id(str(record['prompt']))}"
        ),
    )
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    dropped_duplicates = 0
    dropped_excluded = 0
    for record in ranked:
        semantic = _semantic_id(str(record["prompt"]))
        if semantic in excluded:
            dropped_excluded += 1
            continue
        if semantic in seen:
            dropped_duplicates += 1
            continue
        seen.add(semantic)
        selected.append(dict(record))
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise ValueError(f"natural source produced {len(selected)} tasks; quota={quota}")
    return selected, dropped_duplicates, dropped_excluded


def _record_to_task(record: Mapping[str, Any], source: Mapping[str, Any]) -> TeacherTask:
    task_hash = _stable_hash(str(record["row_key"]) + "|" + str(record["prompt"]))[:18]
    task_id = f"natural-{source['id']}-{task_hash}"
    metadata = {
        "task_kind": str(record["task_kind"]),
        "training_mode": str(record["mode"]),
        "natural_public_source": str(source["id"]),
        "upstream_repo": source.get("repo"),
        "upstream_revision": source.get("revision"),
        "upstream_split": "train",
        "upstream_row_key": str(record["row_key"]),
        "license": str(source["license"]),
        "semantic_id": _semantic_id(str(record["prompt"])),
        **dict(record.get("metadata", {})),
    }
    if record["mode"] == "no_tool":
        return TeacherTask(
            task_id=task_id,
            prompt=str(record["prompt"]),
            metadata=metadata,
            reference_answer=str(record["answer"]),
        )
    tool_source = str(record["tool_source"])
    arguments = dict(record["tool_arguments"])
    descriptor = {
        "name": tool_source,
        "description": "Retrieve task-relevant evidence for the user's natural request.",
        "arguments": tuple(
            {"name": name, "kind": "string", "required": True} for name in arguments
        ),
        "cacheable": True,
        "dynamic": False,
        "versioned": False,
    }
    evidence = TeacherEvidence(
        evidence_id="evidence",
        source=tool_source,
        value=record["evidence"],
        arguments=arguments,
        provenance=f"{source['id']}:{record['row_key']}",
    )
    return TeacherTask(
        task_id=task_id,
        prompt=str(record["prompt"]),
        source_descriptors=(descriptor,),
        evidence=(evidence,),
        metadata=metadata,
        reference_answer=str(record["answer"]),
    )


def _plan_for_task(task: TeacherTask, record: Mapping[str, Any]) -> TeacherPlan:
    if not task.evidence:
        goal = TeacherCellPlan(
            cell_id="goal",
            semantic_text=(
                "Answer the user's request directly while preserving its stated constraints."
            ),
            roles={CognitiveRole.PLAN: 1.0},
            uncertainty=0.25,
            noise=0.04,
            lifecycle=CellLifecycle.STABLE,
        )
        answer = TeacherCellPlan(
            cell_id="answer",
            semantic_text="A complete response satisfying the request is ready.",
            roles={CognitiveRole.CONCLUSION: 1.0},
            uncertainty=0.03,
            noise=0.01,
            lifecycle=CellLifecycle.STABLE,
            anchors=(_text_anchor(str(task.reference_answer), f"{task.task_id}|answer"),),
            links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("goal"), 0.8),),
        )
        return TeacherPlan(
            task_id=task.task_id,
            final_answer=str(task.reference_answer),
            frames=(
                TeacherFrame(phase="initial", display=DISPLAY_UNKNOWN_MARKER, cells=(goal,)),
                TeacherFrame(
                    phase="final", display=str(task.reference_answer), cells=(goal, answer)
                ),
            ),
        )

    evidence = task.evidence[0]
    goal = TeacherCellPlan(
        cell_id="goal",
        semantic_text="Use external information to resolve the user's request.",
        roles={CognitiveRole.PLAN: 1.0, CognitiveRole.CONSTRAINT: 0.3},
        uncertainty=0.4,
        noise=0.04,
        lifecycle=CellLifecycle.STABLE,
    )
    need = TeacherCellPlan(
        cell_id="evidence",
        semantic_text="Task-relevant external evidence is still required.",
        roles={CognitiveRole.INFORMATION_NEED: 1.0},
        uncertainty=0.85,
        noise=0.05,
        lifecycle=CellLifecycle.WAITING,
        links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(evidence.source), 1.0),),
    )
    retired_need = TeacherCellPlan(
        cell_id="evidence",
        semantic_text="The requested evidence has arrived.",
        roles={CognitiveRole.INFORMATION_NEED: 0.2},
        uncertainty=0.05,
        noise=0.01,
        lifecycle=CellLifecycle.RETIRED,
        links=(CognitiveLink(LinkRelation.REQUESTS, ObjectRef.source(evidence.source), 1.0),),
    )
    percept = TeacherCellPlan(
        cell_id="percept",
        semantic_text=_short(str(record.get("percept") or "Relevant evidence was retrieved."), 144),
        roles={CognitiveRole.PERCEPT: 1.0},
        uncertainty=0.04,
        noise=0.01,
        lifecycle=CellLifecycle.STABLE,
        anchors=(_text_anchor(str(task.reference_answer), f"{task.task_id}|evidence"),),
        links=(CognitiveLink(LinkRelation.OBSERVES, ObjectRef.source(evidence.source), 1.0),),
    )
    answer = TeacherCellPlan(
        cell_id="answer",
        semantic_text="The retrieved evidence supports the requested response.",
        roles={CognitiveRole.CONCLUSION: 1.0},
        uncertainty=0.03,
        noise=0.01,
        lifecycle=CellLifecycle.STABLE,
        anchors=(_text_anchor(str(task.reference_answer), f"{task.task_id}|answer"),),
        links=(CognitiveLink(LinkRelation.DERIVED_FROM, ObjectRef.cell("percept"), 1.0),),
    )
    need_target = TeacherNeed(
        need_id="need:evidence",
        cell_id="evidence",
        evidence_id="evidence",
        phase="pre",
        source=evidence.source,
        arguments=dict(evidence.arguments),
    )
    return TeacherPlan(
        task_id=task.task_id,
        final_answer=str(task.reference_answer),
        frames=(
            TeacherFrame(phase="initial", display=DISPLAY_UNKNOWN_MARKER, cells=(goal,)),
            TeacherFrame(phase="pre", display=DISPLAY_UNKNOWN_MARKER, cells=(goal, need)),
            TeacherFrame(
                phase="after:evidence",
                display=str(task.reference_answer),
                cells=(goal, retired_need, percept),
            ),
            TeacherFrame(
                phase="final",
                display=str(task.reference_answer),
                cells=(goal, retired_need, percept, answer),
            ),
        ),
        needs=(need_target,),
    )


def _download_verified(source: Mapping[str, Any], cache: Path) -> Path:
    url = str(source["url"])
    suffix = ".zip" if url.endswith(".zip") else ".tgz"
    path = cache / f"{source['id']}{suffix}"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    actual = _sha256(path)
    expected = str(source["sha256"])
    if actual != expected:
        raise ValueError(
            f"upstream archive SHA mismatch for {source['id']}: {actual} != {expected}"
        )
    return path


def _oasst_chain(
    rows: Mapping[str, Any], message_id: str, max_messages: int
) -> list[tuple[str, str]]:
    chain: list[tuple[str, str]] = []
    current = rows.get(message_id)
    seen: set[str] = set()
    while current is not None and len(chain) < max_messages:
        current_id = str(current.message_id)
        if current_id in seen:
            break
        seen.add(current_id)
        if current.lang != "en" or bool(current.deleted) or not bool(current.review_result):
            break
        chain.append((str(current.role), _clean_multiline(str(current.text))))
        parent_id = str(current.parent_id)
        current = rows.get(parent_id)
    chain.reverse()
    return [(role, text) for role, text in chain if text]


def _conversation_prompt(turns: Sequence[tuple[str, str]]) -> str:
    if not turns:
        return ""
    if len(turns) == 1 and turns[0][0] in {"user", "prompter"}:
        return turns[0][1]
    labels = {"user": "User", "prompter": "User", "agent": "Assistant", "assistant": "Assistant"}
    return "\n".join(f"{labels.get(role, role.title())}: {text}" for role, text in turns)


def _oasst_quality_ok(row: Any) -> bool:
    labels = row.labels
    if not isinstance(labels, dict):
        return True
    names = [str(item) for item in labels.get("name", ())]
    values = list(labels.get("value", ()))
    lookup = {name: float(values[index]) for index, name in enumerate(names) if index < len(values)}
    if lookup.get("quality", 1.0) < 0.65:
        return False
    if lookup.get("helpfulness", 1.0) < 0.5:
        return False
    return all(
        lookup.get(name, 0.0) <= limit
        for name, limit in {
            "fails_task": 0.25,
            "pii": 0.25,
            "not_appropriate": 0.25,
            "toxicity": 0.75,
        }.items()
    )


def _rank_zero(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and number == 0.0


def _qasper_consensus(answers: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    candidates: list[tuple[str, list[str]]] = []
    for annotation in answers:
        answer = annotation.get("answer", {})
        if bool(answer.get("unanswerable", False)):
            continue
        text = _clean(str(answer.get("free_form_answer") or ""))
        if not text:
            spans = [_clean(str(item)) for item in answer.get("extractive_spans", ())]
            spans = [item for item in spans if item]
            if spans:
                text = "; ".join(spans)
        if not text and answer.get("yes_no") is not None:
            text = "yes" if bool(answer["yes_no"]) else "no"
        evidence = [_clean(str(item)) for item in answer.get("evidence", ())]
        evidence = [item for item in evidence if item]
        if text and evidence:
            candidates.append((text, evidence))
    if not candidates:
        return "", []
    counts = Counter(_answer_key(text) for text, _ in candidates)
    best_key, _ = counts.most_common(1)[0]
    matching = [(text, evidence) for text, evidence in candidates if _answer_key(text) == best_key]
    text = min((item[0] for item in matching), key=lambda value: (len(value), value))
    evidence: list[str] = []
    for _, items in matching:
        for item in items:
            if item not in evidence:
                evidence.append(item)
    return text, evidence


def _evidence_summary(excerpts: Sequence[Mapping[str, str]], target: str) -> str:
    if excerpts:
        title = _short(str(excerpts[0].get("title", "document")), 56)
        return _short(f"Retrieved guidance from {title} supports the response: {target}", 144)
    return _short(f"Retrieved guidance supports the response: {target}", 144)


def _text_anchor(value: str, scope: str) -> Anchor:
    text = _clean(value)
    return Anchor(
        anchor_id=f"text:{_stable_hash(scope + '|' + text)[:16]}",
        kind=AnchorKind.TEXT,
        value=text[:192] or "response",
        confidence=1.0,
    )


def _semantic_id(prompt: str) -> str:
    return _stable_hash(_clean(prompt).casefold())


def _answer_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(text).casefold()).strip()


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _clean_multiline(text: str) -> str:
    lines = [_clean(line) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _short(text: str, limit: int) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip(" ,;:-") + "…"


def _quantile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    index = int(round((len(values) - 1) * fraction))
    return int(values[index])

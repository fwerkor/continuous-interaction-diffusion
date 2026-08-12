from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cid.teacher_wave import (
    TEACHER_SOFT_QUALITY_GUIDANCE,
    TeacherStageOutput,
    _import_one_teacher_response,
    _load_jobs,
    _request_id,
    dump_teacher_wave_state,
    load_teacher_wave_state,
    teacher_stage_soft_warning_codes,
    validate_teacher_stage_tct_quality,
)

PROTOCOL = "cid.teacher-agent.v1"

INSTRUCTIONS = """# CID interactive teacher workspace

This directory is a compact interface for a strong interactive teacher (including an agent using
local-shell-mcp). The normal teacher-wave JSONL format remains the canonical production format;
this workspace only removes repeated prompt boilerplate and response-file bookkeeping.

Read every JSON file under `current/requests/`. For each request, write exactly one JSON object to
`current/responses/<request_id>.json`. The response file contains the stage output directly; do not
wrap it in `request_id` or `output`.

Rules for one stage:

- Use only `task`, `previous_state`, and `arrived_evidence` in that request. Future evidence is not
  visible and must not be inferred from dataset metadata or other requests.
- Do not write private chain-of-thought. `semantic_text` is a concise cognitive-state summary.
- Preserve every cell ID in `previous_state.cells`. Retire an obsolete cell with
  `lifecycle="retired"` instead of deleting it.
- Allowed roles are `hypothesis`, `information_need`, `percept`, `plan`, `constraint`, and
  `conclusion`. Role weights are in [0, 1].
- Emit exactly one `needs` entry for every item in `available_evidence_contracts`, and no others.
  Attach each need to a current cell with a positive `information_need` role. Source and arguments
  are owned by the contract and are deliberately omitted from the response.
- If a contract has `freshness_hint`, copy it to that need's `freshness`. `always` means the
  binding remains live for later refreshes or stream chunks.
- At a terminal stage, emit no new needs, make `display` the concise final answer, and include at
  least one cell with a positive `conclusion` role.

{soft_quality_guidance}

Minimal response schema (optional cell/need fields use parser defaults):

```json
{
  "display": "non-empty current display state",
  "cells": [
    {
      "cell_id": "stable-logical-id",
      "semantic_text": "short state summary",
      "roles": {"plan": 1.0},
      "uncertainty": 0.5,
      "noise": 0.5,
      "lifecycle": "active",
      "anchors": [],
      "links": []
    }
  ],
  "needs": [
    {"evidence_id": "...", "cell_id": "...", "confidence": 1.0, "freshness": "once"}
  ]
}
```

After writing some or all responses, run `cid teacher-agent-commit --workspace <this-directory>`.
Valid responses are persisted independently. Missing or rejected responses stay in the current
batch and can be fixed in place. Run `cid teacher-agent-checkout ...` again after a fully successful
commit to advance to the next causal stages. Calling checkout before completion simply resumes the
same batch, which makes the workflow safe across interrupted ChatGPT/LSM sessions.
""".replace("{soft_quality_guidance}", TEACHER_SOFT_QUALITY_GUIDANCE)


def checkout_teacher_agent_batch(
    jobs_path: str | Path,
    state_path: str | Path,
    workspace_path: str | Path,
    *,
    max_requests: int = 8,
) -> dict[str, Any]:
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")

    jobs_source = Path(jobs_path).resolve()
    state_source = Path(state_path).resolve()
    workspace = Path(workspace_path)
    current = workspace / "current"
    manifest_path = current / "manifest.json"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")

    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        pending = _manifest_pending_requests(manifest)
        if pending:
            return {
                "status": "resumed",
                "requests": len(manifest["requests"]),
                "pending": len(pending),
                "workspace": str(workspace),
                "manifest": str(manifest_path),
            }
        shutil.rmtree(current)
    elif current.exists():
        raise ValueError(
            "teacher agent workspace has an incomplete current directory without a manifest: "
            f"{current}"
        )

    jobs = _load_jobs(jobs_source)
    state = load_teacher_wave_state(state_source)
    pending_contexts, complete_tasks = _pending_contexts(jobs, state, max_requests=max_requests)
    if not pending_contexts:
        return {
            "status": "complete",
            "requests": 0,
            "pending": 0,
            "jobs": len(jobs),
            "complete_tasks": complete_tasks,
            "workspace": str(workspace),
        }

    requests_dir = current / "requests"
    responses_dir = current / "responses"
    errors_dir = current / "errors"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    manifest_requests: list[dict[str, Any]] = []
    for index, context in enumerate(pending_contexts):
        request_id = str(context["request_id"])
        filename = f"{index:03d}-{request_id}.json"
        request_file = requests_dir / filename
        request_file.write_text(
            json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_requests.append(
            {
                "request_id": request_id,
                "task_id": context["task_id"],
                "stage_index": context["stage_index"],
                "phase": context["phase"],
                "request_file": str(request_file.relative_to(current)),
                "response_file": f"responses/{request_id}.json",
            }
        )

    manifest = {
        "protocol": PROTOCOL,
        "jobs": str(jobs_source),
        "state": str(state_source),
        "requests": manifest_requests,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "checked_out",
        "requests": len(manifest_requests),
        "pending": len(manifest_requests),
        "jobs": len(jobs),
        "complete_tasks": complete_tasks,
        "workspace": str(workspace),
        "manifest": str(manifest_path),
    }


def commit_teacher_agent_batch(workspace_path: str | Path) -> dict[str, Any]:
    workspace = Path(workspace_path)
    current = workspace / "current"
    manifest_path = current / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"teacher agent workspace has no current batch: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    jobs = {str(job["task_id"]): job for job in _load_jobs(manifest["jobs"])}
    state = load_teacher_wave_state(manifest["state"])
    requests = {
        str(item["request_id"]): {
            "request_id": str(item["request_id"]),
            "task_id": str(item["task_id"]),
            "stage_index": int(item["stage_index"]),
            "phase": str(item["phase"]),
        }
        for item in manifest["requests"]
    }

    imported = 0
    unchanged = 0
    missing = 0
    rejected = 0
    soft_warning_counts: Counter[str] = Counter()
    errors_dir = current / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    for item in manifest["requests"]:
        request_id = str(item["request_id"])
        key = (str(item["task_id"]), int(item["stage_index"]))
        response_path = current / str(item["response_file"])
        if not response_path.exists():
            existing = state.get(key)
            if existing is not None and existing.request_id == request_id:
                unchanged += 1
            else:
                missing += 1
            continue

        try:
            output = _load_response_output(response_path, request_id)
            task_id = str(item["task_id"])
            stage_index = int(item["stage_index"])
            stage = list(jobs[task_id]["stages"])[stage_index]
            validate_teacher_stage_tct_quality(stage, TeacherStageOutput.from_dict(output))
            status = _import_one_teacher_response(
                {"request_id": request_id, "output": output},
                requests,
                jobs,
                state,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected += 1
            (errors_dir / f"{request_id}.json").write_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "response_file": str(response_path.relative_to(current)),
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            continue

        if status == "imported":
            imported += 1
        else:
            unchanged += 1
        task_id = str(item["task_id"])
        stage_index = int(item["stage_index"])
        stage = list(jobs[task_id]["stages"])[stage_index]
        parsed_output = state[(task_id, stage_index)].output
        soft_warning_counts.update(teacher_stage_soft_warning_codes(stage, parsed_output))
        error_path = errors_dir / f"{request_id}.json"
        if error_path.exists():
            error_path.unlink()

    dump_teacher_wave_state(state.values(), manifest["state"])
    remaining = missing + rejected
    report = {
        "imported": imported,
        "unchanged": unchanged,
        "missing": missing,
        "rejected": rejected,
        "remaining": remaining,
        "complete": remaining == 0,
        "state_records": len(state),
        "soft_warning_counts": dict(sorted(soft_warning_counts.items())),
        "workspace": str(workspace),
    }
    (current / "commit.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _pending_contexts(
    jobs: tuple[Mapping[str, Any], ...],
    state: Mapping[tuple[str, int], Any],
    *,
    max_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    contexts: list[dict[str, Any]] = []
    complete_tasks = 0
    for job in jobs:
        task_id = str(job["task_id"])
        stages = list(job["stages"])
        stage_index = 0
        while (task_id, stage_index) in state:
            stage_index += 1
        extras = [
            record_stage
            for record_task, record_stage in state
            if record_task == task_id and record_stage >= stage_index
        ]
        if extras:
            raise ValueError(f"teacher state for {task_id} is not a contiguous stage prefix")
        if stage_index == len(stages):
            complete_tasks += 1
            continue

        stage = stages[stage_index]
        previous = None if stage_index == 0 else state[(task_id, stage_index - 1)].output
        phase = str(stage["phase"])
        if len(contexts) < max_requests:
            contexts.append(
                {
                    "protocol": PROTOCOL,
                    "request_id": _request_id(task_id, stage_index, phase),
                    "task_id": task_id,
                    "stage_index": stage_index,
                    "phase": phase,
                    "terminal": bool(stage.get("terminal", False)),
                    "task": dict(job["task"]),
                    "previous_state": None if previous is None else previous.to_dict(),
                    "arrived_evidence": stage.get("arrived_evidence"),
                    "available_evidence_contracts": [
                        dict(contract) for contract in stage.get("available_evidence", ())
                    ],
                }
            )
    return contexts, complete_tasks


def _load_manifest(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported teacher agent manifest: {path}")
    if not isinstance(raw.get("requests"), list):
        raise ValueError("teacher agent manifest requires a requests list")
    return raw


def _manifest_pending_requests(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    state = load_teacher_wave_state(str(manifest["state"]))
    pending: list[Mapping[str, Any]] = []
    for item in manifest["requests"]:
        key = (str(item["task_id"]), int(item["stage_index"]))
        record = state.get(key)
        if record is None or record.request_id != str(item["request_id"]):
            pending.append(item)
    return pending


def _load_response_output(path: Path, request_id: str) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("teacher agent response must be one JSON object")
    if set(raw) == {"request_id", "output"}:
        if str(raw["request_id"]) != request_id:
            raise ValueError("teacher agent response request_id does not match its filename")
        output = raw["output"]
        if not isinstance(output, dict):
            raise ValueError("teacher agent wrapped response output must be an object")
        return output
    return raw

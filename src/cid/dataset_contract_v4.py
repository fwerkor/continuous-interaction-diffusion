from __future__ import annotations

import hashlib
import json
import re as _re
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cid.data import DISPLAY_UNKNOWN_MARKER, is_display_process_status
from cid.dataset_contract_v3 import annotate_trajectory_contract_v3

_PROVISIONAL = _re.compile(r"^Provisional:\s*(.+?);\s*verifying[.! ]*$", _re.IGNORECASE)
_CORRECTION = _re.compile(
    r"^Evidence contradicts\s+(.+?);\s*revising to\s+(.+?)\s+and confirming[.! ]*$",
    _re.IGNORECASE,
)
_PRIMARY_PARTIAL = _re.compile(
    r"^Primary value observed:\s*(.+?);\s*secondary comparison is still pending[.! ]*$",
    _re.IGNORECASE,
)
_DYNAMIC_PARTIAL = _re.compile(
    r"^Observed\s+(.+?);\s*keeping the live binding open for a newer value[.! ]*$",
    _re.IGNORECASE,
)
_STREAM_PARTIAL = _re.compile(
    r"^Received first stream chunk:\s*(.+?);\s*awaiting continuation[.! ]*$",
    _re.IGNORECASE,
)
_PENDING = _re.compile(r"\bpending\b", _re.IGNORECASE)
_ELLIPSIS_PLACEHOLDER = _re.compile(r"(?<!\.)\.\.\.(?!\.)")


def rematerialize_display_text_v4(text: str, *, final_answer: str) -> str:
    """Map one legacy display snapshot to the v4 answer-draft contract.

    The transformation is deliberately conservative: process-only narration becomes an unresolved
    slot, existing answer-bearing text is preserved, and common explicit placeholders become CID's
    tokenizer-independent unresolved marker.
    """

    stripped = text.strip()
    if stripped == final_answer.strip():
        return final_answer
    if is_display_process_status(stripped):
        return DISPLAY_UNKNOWN_MARKER

    provisional = _PROVISIONAL.fullmatch(stripped)
    if provisional:
        candidate = provisional.group(1).strip()
        return f"{candidate} (provisional; verification: {DISPLAY_UNKNOWN_MARKER})"

    correction = _CORRECTION.fullmatch(stripped)
    if correction:
        corrected = correction.group(2).strip()
        return f"{corrected} (confirmation: {DISPLAY_UNKNOWN_MARKER})"

    primary = _PRIMARY_PARTIAL.fullmatch(stripped)
    if primary:
        value = primary.group(1).strip()
        return (
            f"Primary: {value}; secondary: {DISPLAY_UNKNOWN_MARKER}; "
            f"selected: {value} (provisional)."
        )

    dynamic = _DYNAMIC_PARTIAL.fullmatch(stripped)
    if dynamic:
        value = dynamic.group(1).strip()
        return f"Current value: {value}; refresh: {DISPLAY_UNKNOWN_MARKER}."

    stream = _STREAM_PARTIAL.fullmatch(stripped)
    if stream:
        first = stream.group(1).strip()
        return f"{first}, {DISPLAY_UNKNOWN_MARKER}"

    rewritten = _PENDING.sub(DISPLAY_UNKNOWN_MARKER, stripped)
    rewritten = _ELLIPSIS_PLACEHOLDER.sub(DISPLAY_UNKNOWN_MARKER, rewritten)
    return rewritten or DISPLAY_UNKNOWN_MARKER


def rematerialize_trajectory_contract_v4(
    raw: Mapping[str, Any],
    *,
    apply_v3_routing: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Rematerialize a trajectory for neural contract v4 without changing task semantics."""

    source = dict(raw)
    if apply_v3_routing:
        source, routing_stats = annotate_trajectory_contract_v3(source)
    else:
        routing_stats = {
            "bindings": len(source.get("binding_targets", ())),
            "owner_bindings": sum(
                item.get("owner_cell_id") is not None for item in source.get("binding_targets", ())
            ),
            "multi_cell_bindings": sum(
                len(item.get("target_cells", ())) > 1 for item in source.get("binding_targets", ())
            ),
            "display_routed_bindings": sum(
                bool(item.get("target_display")) for item in source.get("binding_targets", ())
            ),
            "global_display_fallback_bindings": sum(
                not item.get("target_display") for item in source.get("binding_targets", ())
            ),
            "promoted_sources": sum(
                bool(item.get("promote_results_to_fact"))
                for item in source.get("source_descriptors", ())
            ),
            "max_target_cells": max(
                (len(item.get("target_cells", ())) for item in source.get("binding_targets", ())),
                default=0,
            ),
        }

    thought_targets = [dict(item) for item in source.get("thought_targets", ())]
    if not thought_targets:
        raise ValueError("neural contract v4 requires thought targets")
    steps = sorted({int(item["step"]) for item in thought_targets})
    if steps != list(range(steps[-1] + 1)):
        raise ValueError("neural contract v4 requires contiguous thought steps starting at zero")

    final_answer = str(source.get("target_display", ""))
    explicit_display = {
        int(item["step"]): str(item["text"]) for item in source.get("display_targets", ())
    }
    current = DISPLAY_UNKNOWN_MARKER
    display_targets: list[dict[str, Any]] = []
    status_rewrites = 0
    placeholder_rewrites = 0
    preserved_partial = 0
    for step in steps:
        original = explicit_display.get(step)
        if original is not None:
            rewritten = rematerialize_display_text_v4(original, final_answer=final_answer)
            status_rewrites += int(
                original.strip() != final_answer.strip() and is_display_process_status(original)
            )
            placeholder_rewrites += int(
                rewritten != original.strip() and not is_display_process_status(original)
            )
            preserved_partial += int(
                rewritten not in {DISPLAY_UNKNOWN_MARKER, final_answer}
                and original.strip() != final_answer.strip()
            )
            current = rewritten
        if step == steps[-1]:
            current = final_answer
        display_targets.append({"step": step, "text": current})

    appended_settle_step = 0
    if len(display_targets) < 2 or display_targets[-2]["text"] != final_answer:
        old_final = steps[-1]
        new_final = old_final + 1
        last_thought = [item for item in thought_targets if int(item["step"]) == old_final]
        thought_targets.extend({**item, "step": new_final} for item in last_thought)
        grounding_targets = [dict(item) for item in source.get("grounding_targets", ())]
        grounding_targets.extend(
            {**item, "step": new_final}
            for item in grounding_targets
            if int(item.get("step", -1)) == old_final
        )
        source["grounding_targets"] = grounding_targets
        display_targets.append({"step": new_final, "text": final_answer})
        appended_settle_step = 1

    metadata = dict(source.get("metadata", {}))
    metadata["neural_contract_version"] = 4
    metadata["display_contract"] = "continuous-answer-draft-v1"
    source["metadata"] = metadata
    source["thought_targets"] = thought_targets
    source["display_targets"] = display_targets

    return source, {
        **routing_stats,
        "status_rewrites": status_rewrites,
        "placeholder_rewrites": placeholder_rewrites,
        "preserved_partial_targets": preserved_partial,
        "appended_settle_steps": appended_settle_step,
    }


def migrate_dataset_contract_v4(
    input_path: str | Path,
    output_path: str | Path,
    manifest_output: str | Path,
    *,
    curated_path: str | Path | None = None,
) -> dict[str, Any]:
    """Stream a trajectory dataset into neural contract v4 and optionally append curated data."""

    source = Path(input_path)
    output = Path(output_path)
    manifest_path = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    counters: Counter[str] = Counter()
    seen_ids: set[str] = set()

    with tempfile.NamedTemporaryFile(
        "wb",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as src:
                for line_number, raw_line in enumerate(src, start=1):
                    input_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        raw = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid input trajectory at line {line_number}: {exc}"
                        ) from exc
                    rematerialized, stats = rematerialize_trajectory_contract_v4(raw)
                    example_id = str(rematerialized.get("example_id", ""))
                    if not example_id or example_id in seen_ids:
                        raise ValueError(
                            f"duplicate or empty example_id during v4 migration: {example_id!r}"
                        )
                    seen_ids.add(example_id)
                    encoded = (
                        json.dumps(rematerialized, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    handle.write(encoded)
                    output_digest.update(encoded)
                    counters.update(stats)
                    counters["base_examples"] += 1

            if curated_path is not None:
                with Path(curated_path).open("rb") as curated:
                    for line_number, raw_line in enumerate(curated, start=1):
                        if not raw_line.strip():
                            continue
                        raw = json.loads(raw_line)
                        rematerialized, stats = rematerialize_trajectory_contract_v4(
                            raw, apply_v3_routing=True
                        )
                        example_id = str(rematerialized.get("example_id", ""))
                        if not example_id or example_id in seen_ids:
                            raise ValueError(
                                f"duplicate or empty curated example_id at line {line_number}: "
                                f"{example_id!r}"
                            )
                        seen_ids.add(example_id)
                        encoded = (
                            json.dumps(rematerialized, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode("utf-8")
                        handle.write(encoded)
                        output_digest.update(encoded)
                        counters.update(stats)
                        counters["curated_examples"] += 1
            handle.flush()
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    audit = audit_dataset_contract_v4(output)
    if not audit["ok"]:
        raise ValueError(f"v4 dataset audit failed: {audit['violations']}")
    manifest = {
        "format_version": 1,
        "name": "cid-dataset-v16",
        "neural_contract_version": 4,
        "input": str(source),
        "input_sha256": input_digest.hexdigest(),
        "output": str(output),
        "sha256": output_digest.hexdigest(),
        "bytes": output.stat().st_size,
        "examples": audit["examples"],
        "base_examples": counters["base_examples"],
        "curated_examples": counters["curated_examples"],
        "status_rewrites": counters["status_rewrites"],
        "placeholder_rewrites": counters["placeholder_rewrites"],
        "preserved_partial_targets": counters["preserved_partial_targets"],
        "appended_settle_steps": counters["appended_settle_steps"],
        "display_contract": "continuous-answer-draft-v1",
        "audit": audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_dataset_contract_v4(path: str | Path) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    top_partial: Counter[str] = Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for _line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            counters["examples"] += 1
            final_answer = str(raw.get("target_display", ""))
            thought_steps = sorted({int(item["step"]) for item in raw.get("thought_targets", ())})
            display = {
                int(item["step"]): str(item["text"]) for item in raw.get("display_targets", ())
            }
            if not thought_steps or set(display) != set(thought_steps):
                violations["display_step_coverage"] += 1
                continue
            ordered = [display[step] for step in thought_steps]
            if ordered[-1] != final_answer:
                violations["terminal_not_final"] += 1
            if len(ordered) < 2 or ordered[-2] != final_answer:
                violations["terminal_not_stable"] += 1
            if DISPLAY_UNKNOWN_MARKER in ordered[-1]:
                violations["terminal_unknown"] += 1
            for text in ordered[:-1]:
                counters["display_targets"] += 1
                if DISPLAY_UNKNOWN_MARKER in text:
                    counters["unknown_targets"] += 1
                if text == final_answer:
                    counters["preterminal_final_targets"] += 1
                elif text != DISPLAY_UNKNOWN_MARKER:
                    counters["partial_answer_targets"] += 1
                    top_partial[text] += 1
                if is_display_process_status(text):
                    violations["process_status_targets"] += 1
            metadata = raw.get("metadata", {})
            if int(metadata.get("neural_contract_version", -1)) != 4:
                violations["missing_v4_metadata"] += 1
    return {
        "ok": not violations,
        "examples": counters["examples"],
        "display_targets": counters["display_targets"],
        "unknown_targets": counters["unknown_targets"],
        "partial_answer_targets": counters["partial_answer_targets"],
        "preterminal_final_targets": counters["preterminal_final_targets"],
        "top_partial_targets": top_partial.most_common(20),
        "violations": dict(sorted(violations.items())),
    }

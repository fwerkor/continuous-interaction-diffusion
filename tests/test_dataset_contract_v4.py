from __future__ import annotations

import json

from cid.data import DISPLAY_UNKNOWN_MARKER
from cid.dataset_contract_v4 import (
    audit_dataset_contract_v4,
    is_display_process_status,
    rematerialize_display_text_v4,
    rematerialize_trajectory_contract_v4,
)


def _raw_trajectory(*, stable_tail: bool = False) -> dict:
    thought = [
        {
            "step": 0,
            "slot": 0,
            "cell_id": "state",
            "semantic_text": "Reason about the requested answer.",
            "roles": {"plan": 1.0},
            "uncertainty": 0.5,
            "noise": 0.2,
            "lifecycle": "active",
        },
        {
            "step": 1,
            "slot": 0,
            "cell_id": "state",
            "semantic_text": "The requested answer is true.",
            "roles": {"conclusion": 1.0},
            "uncertainty": 0.05,
            "noise": 0.05,
            "lifecycle": "stable",
        },
    ]
    displays = [
        {"step": 0, "text": "Reasoning."},
        {"step": 1, "text": "true"},
    ]
    if stable_tail:
        thought.append({**thought[-1], "step": 2})
        displays.append({"step": 2, "text": "true"})
    return {
        "example_id": "legacy-v4-fixture",
        "prompt": "Is the stated proposition true?",
        "target_display": "true",
        "protected_facts": {},
        "source_descriptors": [],
        "events": [],
        "binding_targets": [],
        "grounding_catalog": [],
        "grounding_targets": [],
        "thought_targets": thought,
        "display_targets": displays,
        "metadata": {"family": "fixture"},
    }


def test_v4_display_rewrite_distinguishes_status_from_answer_bearing_partial() -> None:
    assert is_display_process_status("Relevant documents identified; reading supporting evidence.")
    assert is_display_process_status(
        "Evidence from Giant Steps incorporated; continuing with remaining evidence."
    )
    assert (
        rematerialize_display_text_v4(
            "Relevant documents identified; reading supporting evidence.",
            final_answer="37 ms",
        )
        == DISPLAY_UNKNOWN_MARKER
    )
    assert (
        rematerialize_display_text_v4(
            "George is identified as the husband; the exact date remains pending.",
            final_answer="November 4, 2014",
        )
        == f"George is identified as the husband; the exact date remains {DISPLAY_UNKNOWN_MARKER}."
    )
    assert (
        rematerialize_display_text_v4(
            "Provisional: eligible; verifying.",
            final_answer="ineligible",
        )
        == f"eligible (provisional; verification: {DISPLAY_UNKNOWN_MARKER})"
    )
    assert (
        rematerialize_display_text_v4(
            "Evidence contradicts eligible; revising to ineligible and confirming.",
            final_answer="ineligible",
        )
        == f"ineligible (confirmation: {DISPLAY_UNKNOWN_MARKER})"
    )


def test_v4_rematerialization_adds_one_stable_final_step() -> None:
    migrated, stats = rematerialize_trajectory_contract_v4(_raw_trajectory())

    assert [item["text"] for item in migrated["display_targets"]] == [
        DISPLAY_UNKNOWN_MARKER,
        "true",
        "true",
    ]
    assert sorted({item["step"] for item in migrated["thought_targets"]}) == [0, 1, 2]
    assert stats["status_rewrites"] == 1
    assert stats["appended_settle_steps"] == 1
    assert migrated["metadata"]["neural_contract_version"] == 4


def test_v4_rematerialization_does_not_duplicate_existing_stable_tail() -> None:
    migrated, stats = rematerialize_trajectory_contract_v4(_raw_trajectory(stable_tail=True))

    assert [item["text"] for item in migrated["display_targets"]] == [
        DISPLAY_UNKNOWN_MARKER,
        "true",
        "true",
    ]
    assert stats["appended_settle_steps"] == 0



def test_v4_rematerialization_derives_supported_multi_hop_partial_answer() -> None:
    raw = _raw_trajectory()
    raw["metadata"]["task_kind"] = "multi_hop_qa"
    raw["thought_targets"] = [
        {
            "step": 0,
            "slot": 0,
            "cell_id": "search",
            "semantic_text": "Need the answer-bearing records.",
            "roles": {"information_need": 1.0},
            "uncertainty": 0.8,
            "noise": 0.5,
            "lifecycle": "waiting",
        },
        {
            "step": 1,
            "slot": 0,
            "cell_id": "support-0",
            "semantic_text": "Coast Guard (film) — director: Edward Ludwig.",
            "roles": {"evidence": 1.0},
            "uncertainty": 0.1,
            "noise": 0.1,
            "lifecycle": "stable",
        },
        {
            "step": 1,
            "slot": 1,
            "cell_id": "support-1",
            "semantic_text": "Need task-relevant facts about Edward Ludwig.",
            "roles": {"information_need": 1.0},
            "uncertainty": 0.8,
            "noise": 0.5,
            "lifecycle": "waiting",
        },
        {
            "step": 2,
            "slot": 0,
            "cell_id": "answer",
            "semantic_text": "The answer is Santa Monica, California.",
            "roles": {"conclusion": 1.0},
            "uncertainty": 0.05,
            "noise": 0.05,
            "lifecycle": "stable",
        },
    ]
    raw["display_targets"] = [
        {"step": 0, "text": "Reasoning."},
        {
            "step": 1,
            "text": "Evidence from Coast Guard incorporated; continuing with remaining evidence.",
        },
        {"step": 2, "text": "Santa Monica, California."},
    ]
    raw["target_display"] = "Santa Monica, California."

    migrated, stats = rematerialize_trajectory_contract_v4(raw)
    displays = {item["step"]: item["text"] for item in migrated["display_targets"]}

    assert displays[0] == DISPLAY_UNKNOWN_MARKER
    assert displays[1] == (
        "Known: Coast Guard (film) — director: Edward Ludwig. "
        f"Answer: {DISPLAY_UNKNOWN_MARKER}"
    )
    assert displays[2] == "Santa Monica, California."
    assert displays[3] == "Santa Monica, California."
    assert stats["derived_partial_targets"] == 1
    assert migrated["metadata"]["display_contract"] == "continuous-answer-draft-v2"


def test_v4_rematerialization_does_not_expose_internal_dependency_as_partial_answer() -> None:
    raw = _raw_trajectory()
    raw["metadata"]["task_kind"] = "applied_formula"
    raw["thought_targets"][0]["cell_id"] = "support-0"
    raw["thought_targets"][0]["semantic_text"] = "entity-1 — amount: 11260."

    migrated, stats = rematerialize_trajectory_contract_v4(raw)

    assert migrated["display_targets"][0]["text"] == DISPLAY_UNKNOWN_MARKER
    assert stats["derived_partial_targets"] == 0

def test_v4_audit_rejects_unmigrated_process_display(tmp_path) -> None:
    raw = _raw_trajectory(stable_tail=True)
    raw["metadata"]["neural_contract_version"] = 4
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    audit = audit_dataset_contract_v4(path)

    assert not audit["ok"]
    assert audit["violations"]["process_status_targets"] == 1

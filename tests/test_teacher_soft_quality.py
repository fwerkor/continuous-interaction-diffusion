from cid.teacher_wave import (
    TEACHER_SOFT_QUALITY_GUIDANCE,
    TeacherStageOutput,
    _build_stage_prompt,
    teacher_stage_soft_warning_codes,
)


def _cell(cell_id: str, text: str, *, role: str = "percept") -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "semantic_text": text,
        "roles": {role: 1.0},
        "uncertainty": 0.3,
        "noise": 0.2,
        "lifecycle": "active",
        "anchors": [],
        "links": [],
    }


def test_teacher_stage_prompt_contains_soft_compactness_and_grounding_guidance() -> None:
    job = {
        "task": {"task_id": "soft-quality", "prompt": "Answer from evidence."},
        "stages": [
            {
                "phase": "initial",
                "arrived_evidence": None,
                "available_evidence": [],
                "terminal": False,
            }
        ],
    }

    prompt = _build_stage_prompt(job, 0, None)

    assert TEACHER_SOFT_QUALITY_GUIDANCE in prompt
    assert "target <=160 characters" in prompt
    assert "normally <=240" in prompt
    assert "3-6 live cells" in prompt
    assert "do not copy whole evidence sentences or paragraphs" in prompt
    assert "entity:alice-example" in prompt
    assert '"relation":"observes"' in prompt


def test_teacher_soft_quality_warnings_are_non_blocking_observability() -> None:
    output = TeacherStageOutput.from_dict(
        {
            "display": "evidence received",
            "cells": [
                _cell(
                    f"cell-{index}",
                    "x" * (250 if index == 0 else 20),
                )
                for index in range(7)
            ],
            "needs": [],
        }
    )

    warnings = teacher_stage_soft_warning_codes(
        {"arrived_evidence": {"evidence_id": "support-a"}},
        output,
    )

    assert set(warnings) == {
        "semantic_text_over_target",
        "semantic_text_over_preferred_max",
        "live_cells_over_preferred_count",
        "arrived_evidence_without_anchor",
        "arrived_evidence_without_link",
    }

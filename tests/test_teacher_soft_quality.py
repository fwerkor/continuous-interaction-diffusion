import pytest

from cid.teacher_wave import (
    TEACHER_SOFT_QUALITY_GUIDANCE,
    TeacherStageOutput,
    _build_stage_prompt,
    teacher_stage_soft_warning_codes,
    validate_teacher_stage_tct_quality,
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
    assert "Target <=144 characters" in prompt
    assert "rejected above 192 characters" in prompt
    assert "3-6 live cells" in prompt
    assert "Never paste a whole source sentence or paragraph" in prompt
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


def _source_link(relation: str, source: str) -> dict[str, object]:
    return {
        "relation": relation,
        "target": {"kind": "source", "identifier": source},
        "confidence": 1.0,
    }


def _cell_link(relation: str, cell_id: str) -> dict[str, object]:
    return {
        "relation": relation,
        "target": {"kind": "cell", "identifier": cell_id},
        "confidence": 1.0,
    }


def _anchor(value: str) -> dict[str, object]:
    return {
        "anchor_id": f"entity:{value.casefold().replace(' ', '-')}",
        "kind": "entity",
        "value": value,
        "confidence": 1.0,
    }


def test_interactive_tct_quality_requires_requests_link_for_need() -> None:
    stage = {
        "arrived_evidence": None,
        "available_evidence": [{"evidence_id": "search-results", "source": "workspace_search"}],
        "terminal": False,
    }
    output = TeacherStageOutput.from_dict(
        {
            "display": "pending",
            "cells": [_cell("search", "Need evidence.", role="information_need")],
            "needs": [{"evidence_id": "search-results", "cell_id": "search"}],
        }
    )

    with pytest.raises(ValueError, match="requests link"):
        validate_teacher_stage_tct_quality(stage, output)


def test_interactive_tct_quality_requires_grounded_observed_percept() -> None:
    stage = {
        "arrived_evidence": {
            "evidence_id": "support-a",
            "source": "workspace_read",
            "value": {"title": "A", "sentences": ["A was released in 2001."]},
        },
        "available_evidence": [],
        "terminal": False,
    }
    no_anchor = _cell("support-a", "A — release: 2001.")
    no_anchor["links"] = [_source_link("observes", "workspace_read")]
    output = TeacherStageOutput.from_dict(
        {"display": "A observed", "cells": [no_anchor], "needs": []}
    )
    with pytest.raises(ValueError, match="grounding anchor"):
        validate_teacher_stage_tct_quality(stage, output)

    no_observes = _cell("support-a", "A — release: 2001.")
    no_observes["anchors"] = [_anchor("A")]
    output = TeacherStageOutput.from_dict(
        {"display": "A observed", "cells": [no_observes], "needs": []}
    )
    with pytest.raises(ValueError, match="observes link"):
        validate_teacher_stage_tct_quality(stage, output)


def test_interactive_tct_quality_rejects_long_or_verbatim_semantics() -> None:
    long_output = TeacherStageOutput.from_dict(
        {
            "display": "pending",
            "cells": [_cell("state", "x" * 193, role="plan")],
            "needs": [],
        }
    )
    with pytest.raises(ValueError, match="exceeds interactive TCT limit"):
        validate_teacher_stage_tct_quality(
            {"arrived_evidence": None, "available_evidence": [], "terminal": False},
            long_output,
        )

    copied = (
        "A was released in 2001 and this sentence deliberately remains long enough "
        "to look like copied source prose rather than cognitive state."
    )
    cell = _cell("support-a", copied)
    cell["anchors"] = [_anchor("A")]
    cell["links"] = [_source_link("observes", "workspace_read")]
    output = TeacherStageOutput.from_dict({"display": "observed", "cells": [cell], "needs": []})
    with pytest.raises(ValueError, match="copy arrived evidence verbatim"):
        validate_teacher_stage_tct_quality(
            {
                "arrived_evidence": {
                    "evidence_id": "support-a",
                    "source": "workspace_read",
                    "value": {"title": "A", "sentences": [copied]},
                },
                "available_evidence": [],
                "terminal": False,
            },
            output,
        )


def test_interactive_terminal_conclusion_links_to_percept() -> None:
    percept = _cell("support-a", "A — release: 2001.")
    percept["anchors"] = [_anchor("A")]
    percept["links"] = [_source_link("observes", "workspace_read")]
    conclusion = _cell("answer", "Conclusion: A.", role="conclusion")
    conclusion["links"] = [_cell_link("derived_from", "support-a")]
    output = TeacherStageOutput.from_dict(
        {"display": "A.", "cells": [percept, conclusion], "needs": []}
    )
    validate_teacher_stage_tct_quality(
        {
            "arrived_evidence": {
                "evidence_id": "support-a",
                "source": "workspace_read",
                "value": {"title": "A", "sentences": ["A was released in 2001."]},
            },
            "available_evidence": [],
            "terminal": True,
        },
        output,
    )

    conclusion["links"] = []
    output = TeacherStageOutput.from_dict(
        {"display": "A.", "cells": [percept, conclusion], "needs": []}
    )
    with pytest.raises(ValueError, match="derived_from"):
        validate_teacher_stage_tct_quality(
            {
                "arrived_evidence": {
                    "evidence_id": "support-a",
                    "source": "workspace_read",
                    "value": {"title": "A", "sentences": ["A was released in 2001."]},
                },
                "available_evidence": [],
                "terminal": True,
            },
            output,
        )

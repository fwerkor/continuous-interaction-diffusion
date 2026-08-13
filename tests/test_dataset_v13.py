from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cid.release_validation import find_surface_summary_mismatches

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_v13_semantic_mixture_prioritizes_natural_grounded_interaction() -> None:
    mixture = _load("data/training-semantic-mixture-v13.json")
    components = {item["name"]: item for item in mixture["components"]}
    natural = _load("data/natural-interaction-v1.reference-manifest.json")

    assert mixture["version"] == 13
    assert mixture["semantic_tasks"] == sum(item["tasks"] for item in mixture["components"])
    assert components["natural-grounded-interaction-v1"]["tasks"] == natural["semantic_tasks"]
    assert components["natural-grounded-interaction-v1"]["training_weight"] == 4.0
    assert components["public-base-v1"]["training_weight"] == 2.0
    assert components["public-interaction-v1"]["training_weight"] == 2.5
    assert natural["long_form_targets"] == natural["semantic_tasks"]
    assert natural["target_chars_p50"] >= 160
    assert natural["tasks_with_anchor"] == natural["semantic_tasks"]
    assert natural["tasks_with_link"] == natural["semantic_tasks"]
    assert len(natural["tool_schema_profiles"]) == 6

    policy = mixture["training_mass_policy"]
    assert policy["natural_source_and_augmentation_fraction"] >= 0.35
    assert policy["natural_tool_fraction"] >= 0.27
    assert policy["long_form_grounded_fraction"] >= 0.18
    assert policy["tool_restraint_fraction"] >= 0.08


def test_v13_replaces_template_heavy_v3_slices_and_has_strict_ood_probe() -> None:
    mixture = _load("data/training-semantic-mixture-v13.json")
    names = {item["name"] for item in mixture["components"]}
    probe = _load("data/generalization-probe-v1.reference-manifest.json")

    assert "complex-logic-reasoning-v4" in names
    assert "compositional-longtail-reasoning-v4" in names
    assert "deep-tool-restraint-v4" in names
    assert "complex-logic-reasoning-v3" not in names
    assert "compositional-longtail-reasoning-v3" not in names
    assert "deep-tool-restraint-v3" not in names
    assert probe["exact_logic_spec_overlap_with_training"] == 0
    assert set(probe["strict_holdout_axes"]) == {"domain", "exact_logic_spec"}
    assert not find_surface_summary_mismatches(
        ROOT,
        "data/training-semantic-mixture-v13.json",
    )


def test_v13_trajectory_mixture_declares_semantic_weights_and_v4_components() -> None:
    mixture = _load("data/training-trajectory-mixture-v13.json")
    components = {item["name"]: item for item in mixture["components"]}

    assert mixture["version"] == 13
    assert mixture["semantic_mixture"] == "data/training-semantic-mixture-v13.json"
    assert mixture["max_trajectory_steps"] == 44
    assert mixture["examples"] == sum(item["examples"] for item in mixture["components"])
    assert mixture["training_transitions"] == sum(
        item["training_transitions"] for item in mixture["components"]
    )
    assert components["public-base"]["semantic_weight"] == 2.0
    assert components["public-interaction"]["semantic_weight"] == 2.5
    assert components["natural-interaction-v1"]["semantic_weight"] == 4.0
    assert components["tool-restraint"]["semantic_weight"] == 1.5
    assert components["deep-tool-restraint-v4"]["semantic_weight"] == 1.5
    assert "complex-logic-v4" in components
    assert "compositional-v4" in components
    assert "deep-tool-restraint-v4" in components
    assert "complex-logic-v3" not in components
    assert "compositional-v3" not in components
    assert "deep-tool-restraint-v3" not in components

    pinned = _load("data/training-trajectories-v13.reference-manifest.json")
    spec_bytes = (ROOT / "data/training-trajectory-mixture-v13.json").read_bytes()
    assert pinned["trajectory_mixture_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert pinned["examples"] == mixture["examples"]
    assert pinned["transitions"] == mixture["transitions"]
    assert pinned["training_transitions"] == mixture["training_transitions"]
    assert pinned["max_trajectory_steps"] == mixture["max_trajectory_steps"]

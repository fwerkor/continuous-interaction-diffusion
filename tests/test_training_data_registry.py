import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_training_semantic_mixture_v2_matches_component_reference_manifests() -> None:
    mixture = _load("data/training-semantic-mixture-v2.json")
    references = {
        "public-base-v1": _load("data/public-teacher-v1.train.reference-manifest.json"),
        "public-interaction-v1": _load(
            "data/public-interaction-teacher-v1.train.reference-manifest.json"
        ),
        "mechanism-teacher-v1": _load("data/mechanism-teacher-v1.reference-manifest.json"),
    }

    assert mixture["semantic_tasks"] == 28_055
    assert mixture["sequential_dependency_tasks"] == 4_510
    assert mixture["dependency_depth_3_plus_tasks"] == 1_114
    assert sum(mixture["mode_counts"].values()) == 28_055
    assert sum(component["tasks"] for component in mixture["components"]) == 28_055
    for component in mixture["components"]:
        reference = references[component["name"]]
        task_count = reference.get("tasks", reference.get("semantic_tasks"))
        assert component["tasks"] == task_count
        assert component["tasks_sha256"] == reference["tasks_sha256"]
        assert component["causal_jobs_sha256"] == reference["causal_jobs_sha256"]

    interaction = references["public-interaction-v1"]
    assert interaction["interaction_pattern_counts"] == {
        "decomposition_dag": 4_510,
        "search_then_parallel_reads": 4_515,
    }
    assert interaction["dependency_depth_histogram"] == {
        "2": 7_911,
        "3": 968,
        "4": 146,
    }


def test_training_semantic_mixture_v3_adds_computational_tool_distillation() -> None:
    mixture = _load("data/training-semantic-mixture-v3.json")
    references = {
        "public-base-v1": _load("data/public-teacher-v1.train.reference-manifest.json"),
        "public-interaction-v1": _load(
            "data/public-interaction-teacher-v1.train.reference-manifest.json"
        ),
        "mechanism-teacher-v1": _load("data/mechanism-teacher-v1.reference-manifest.json"),
        "computational-teacher-v1": _load("data/computational-teacher-v1.reference-manifest.json"),
    }

    assert mixture["semantic_tasks"] == 40_055
    assert mixture["mode_counts"] == {
        "no_tool": 6_921,
        "tool_required": 31_191,
        "tools_available_unnecessary": 1_943,
    }
    assert sum(mixture["mode_counts"].values()) == 40_055
    assert sum(component["tasks"] for component in mixture["components"]) == 40_055
    for component in mixture["components"]:
        reference = references[component["name"]]
        task_count = reference.get("tasks", reference.get("semantic_tasks"))
        assert component["tasks"] == task_count
        assert component["tasks_sha256"] == reference["tasks_sha256"]
        assert component["causal_jobs_sha256"] == reference["causal_jobs_sha256"]

    computational = references["computational-teacher-v1"]
    assert computational["semantic_tasks"] == 12_000
    assert computational["causal_stages"] == 31_200
    assert computational["accepted_plans"] == 12_000
    assert computational["review_rejected"] == 0
    assert computational["compiled_trajectories"] == 24_000
    assert computational["compiled_transitions"] == 111_609
    assert computational["tool_schema_task_counts"] == {
        "calculator": 12_000,
        "python": 4_800,
        "record_lookup": 2_400,
    }


def test_training_semantic_mixture_v4_adds_symbolic_tool_distillation() -> None:
    mixture = _load("data/training-semantic-mixture-v4.json")
    references = {
        "public-base-v1": _load("data/public-teacher-v1.train.reference-manifest.json"),
        "public-interaction-v1": _load(
            "data/public-interaction-teacher-v1.train.reference-manifest.json"
        ),
        "mechanism-teacher-v1": _load("data/mechanism-teacher-v1.reference-manifest.json"),
        "computational-teacher-v1": _load("data/computational-teacher-v1.reference-manifest.json"),
        "symbolic-teacher-v1": _load("data/symbolic-teacher-v1.reference-manifest.json"),
    }

    assert mixture["semantic_tasks"] == 55_655
    assert mixture["sequential_dependency_tasks"] == 10_510
    assert mixture["mode_counts"] == {
        "no_tool": 6_921,
        "tool_required": 45_591,
        "tools_available_unnecessary": 3_143,
    }
    assert sum(mixture["mode_counts"].values()) == 55_655
    assert sum(component["tasks"] for component in mixture["components"]) == 55_655
    for component in mixture["components"]:
        reference = references[component["name"]]
        task_count = reference.get("tasks", reference.get("semantic_tasks"))
        assert component["tasks"] == task_count
        assert component["tasks_sha256"] == reference["tasks_sha256"]
        assert component["causal_jobs_sha256"] == reference["causal_jobs_sha256"]

    symbolic = references["symbolic-teacher-v1"]
    assert symbolic["semantic_tasks"] == 15_600
    assert symbolic["causal_stages"] == 34_800
    assert symbolic["review_accepted"] == 15_600
    assert symbolic["review_rejected"] == 0
    assert symbolic["compiled_examples"] == 31_200
    assert symbolic["compiled_transitions"] == 121_748
    assert symbolic["tool_replay_failures"] == 0
    assert symbolic["tool_call_target_counts"] == {
        "calculator": 2_400,
        "record_lookup": 1_200,
        "symbolic_math": 15_600,
    }


def test_training_semantic_mixture_v5_adds_speculative_local_correction() -> None:
    mixture = _load("data/training-semantic-mixture-v5.json")
    references = {
        "public-base-v1": _load("data/public-teacher-v1.train.reference-manifest.json"),
        "public-interaction-v1": _load(
            "data/public-interaction-teacher-v1.train.reference-manifest.json"
        ),
        "mechanism-teacher-v1": _load("data/mechanism-teacher-v1.reference-manifest.json"),
        "computational-teacher-v1": _load("data/computational-teacher-v1.reference-manifest.json"),
        "symbolic-teacher-v1": _load("data/symbolic-teacher-v1.reference-manifest.json"),
        "speculative-local-correction-v1": _load(
            "data/correction-teacher-v1.reference-manifest.json"
        ),
    }

    assert mixture["semantic_tasks"] == 65_655
    assert mixture["sequential_dependency_tasks"] == 20_510
    assert mixture["speculative_local_correction_tasks"] == 10_000
    assert mixture["mode_counts"] == {
        "no_tool": 6_921,
        "tool_required": 55_591,
        "tools_available_unnecessary": 3_143,
    }
    assert sum(mixture["mode_counts"].values()) == 65_655
    assert sum(component["tasks"] for component in mixture["components"]) == 65_655
    for component in mixture["components"]:
        reference = references[component["name"]]
        task_count = reference.get("tasks", reference.get("semantic_tasks"))
        assert component["tasks"] == task_count
        assert component["tasks_sha256"] == reference["tasks_sha256"]
        assert component["causal_jobs_sha256"] == reference["causal_jobs_sha256"]

    correction = references["speculative-local-correction-v1"]
    assert correction["semantic_tasks"] == 10_000
    assert correction["causal_stages"] == 30_000
    assert correction["review_accepted"] == 10_000
    assert correction["review_rejected"] == 0
    assert correction["compiled_trajectories"] == 20_000
    assert correction["compiled_transitions"] == 120_021
    assert correction["semantic_correction_audit"]["local_reopen_cells"] == 20_000
    assert correction["semantic_correction_audit"]["local_stabilization_cells"] == 20_000
    assert correction["trajectory_audit"]["stable_unrelated_context_examples"] == 20_000
    assert correction["teacher_stage_validation"]["rejected"] == 0
    assert correction["teacher_stage_validation"]["missing"] == 0
    assert correction["teacher_stage_validation"]["soft_warning_counts"] == {}


def test_mechanism_reference_manifest_covers_all_cid_specific_families() -> None:
    manifest = _load("data/mechanism-teacher-v1.reference-manifest.json")

    assert manifest["semantic_tasks"] == 10_000
    assert manifest["causal_stages"] == 26_000
    assert manifest["persistent_binding_tasks"] == 4_000
    assert manifest["parallel_root_need_tasks"] == 2_000
    assert manifest["family_counts"] == {
        "competing_sources": 2_000,
        "delayed_retrieval": 2_000,
        "dynamic_state": 2_000,
        "static_copy": 2_000,
        "streaming_evidence": 2_000,
    }


def test_public_mixture_v1_tracks_current_teacher_evidence_abi() -> None:
    mixture = _load("data/training-semantic-mixture-v1.json")
    base = _load("data/public-teacher-v1.train.reference-manifest.json")
    interaction = _load("data/public-interaction-teacher-v1.train.reference-manifest.json")
    by_name = {component["name"]: component for component in mixture["components"]}

    assert by_name["public-base-v1"]["tasks_sha256"] == base["tasks_sha256"]
    assert by_name["public-base-v1"]["causal_jobs_sha256"] == base["causal_jobs_sha256"]
    assert by_name["public-interaction-v1"]["tasks_sha256"] == interaction["tasks_sha256"]
    assert (
        by_name["public-interaction-v1"]["causal_jobs_sha256"] == interaction["causal_jobs_sha256"]
    )


def test_training_trajectory_mixture_v5_matches_compiled_components() -> None:
    mixture = _load("data/training-trajectory-mixture-v5.json")
    assert mixture["name"] == "training-trajectory-mixture-v5"
    assert mixture["version"] == 5
    assert mixture["examples"] == 179_608
    assert mixture["transitions"] == 1_027_548
    assert mixture["thought_capacity_required"] == 8
    assert [component["name"] for component in mixture["components"]] == [
        "public-base",
        "public-interaction",
        "mechanism",
        "computational",
        "symbolic",
        "local-correction",
    ]

    assert sum(component["examples"] for component in mixture["components"]) == mixture["examples"]
    assert (
        sum(component["transitions"] for component in mixture["components"])
        == mixture["transitions"]
    )

    materialized = _load("data/training-trajectories-v5.reference-manifest.json")
    materialized_components = {
        component["name"]: component for component in materialized["components"]
    }
    for component in mixture["components"]:
        reference = materialized_components[component["name"]]
        assert component["path"] == reference["path"]
        assert component["sha256"] == reference["sha256"]
        assert component["examples"] == reference["examples"]
        assert component["transitions"] == reference["transitions"]

    assert materialized["mixture"] == mixture["name"]
    assert materialized["mixture_version"] == mixture["version"]
    assert materialized["examples"] == mixture["examples"]
    assert materialized["transitions"] == mixture["transitions"]
    assert materialized["thought_capacity_required"] == mixture["thought_capacity_required"]
    assert (
        materialized["sha256"] == "d771d5ddcf94c1b8b7ae9a1b7df38944fc3c5974d34867ec4c0ae392b7c9120b"
    )
    assert materialized["bytes"] == 2_391_099_272


def test_training_semantic_mixture_v7_adds_cid_self_identity() -> None:
    mixture = _load("data/training-semantic-mixture-v7.json")
    identity = _load("data/self-identity-teacher-v1.reference-manifest.json")
    component = next(
        item for item in mixture["components"] if item["name"] == "cid-self-identity-v1"
    )

    assert mixture["version"] == 7
    assert mixture["semantic_tasks"] == 78_375
    assert mixture["self_identity_tasks"] == 720
    assert mixture["self_identity_language_counts"] == {"en": 540, "zh": 180}
    assert mixture["mode_counts"] == {
        "no_tool": 19_641,
        "tool_required": 55_591,
        "tools_available_unnecessary": 3_143,
    }
    assert component["tasks"] == identity["semantic_tasks"] == 720
    assert component["tasks_sha256"] == identity["tasks_sha256"]
    assert component["teacher_protocol"] == "deterministic-no-tool-self-model"
    assert identity["accepted_plans"] == 720
    assert identity["review_rejected"] == 0
    assert identity["compiled_trajectories"] == 1_440
    assert identity["compiled_transitions"] == 1_440
    assert identity["model_name"] == "CID-v1"
    assert identity["method_name"] == "Continuous Interaction Diffusion"
    assert identity["family_counts"] == {
        "acronym": 80,
        "architecture_summary": 80,
        "async_interaction": 80,
        "channels": 80,
        "method_overview": 80,
        "model_class": 80,
        "name": 80,
        "runtime_boundary": 80,
        "tct": 80,
    }


def test_training_trajectory_mixture_v7_appends_self_identity() -> None:
    mixture = _load("data/training-trajectory-mixture-v7.json")
    identity = _load("data/self-identity-teacher-v1.reference-manifest.json")
    component = next(item for item in mixture["components"] if item["name"] == "self-identity")

    assert mixture["version"] == 7
    assert mixture["examples"] == 205_048
    assert mixture["transitions"] == 1_076_988
    assert mixture["thought_capacity_required"] == 8
    assert component["examples"] == identity["compiled_trajectories"]
    assert component["transitions"] == identity["compiled_transitions"]
    assert component["sha256"] == identity["compiled_sha256"]


def test_training_semantic_mixture_v10_adds_longtail_and_long_horizon_without_probe_leakage() -> (
    None
):
    mixture = _load("data/training-semantic-mixture-v10.json")
    longtail = _load("data/compositional-teacher-v1.reference-manifest.json")
    probe = _load("data/generalization-probe-v1.reference-manifest.json")
    long_horizon = _load("data/long-horizon-teacher-v1.reference-manifest.json")
    by_name = {component["name"]: component for component in mixture["components"]}

    assert mixture["name"] == "training-semantic-mixture-v10"
    assert mixture["version"] == 10
    assert mixture["semantic_tasks"] == 130_075
    assert mixture["thought_capacity_required"] == 128
    assert mixture["mode_counts"] == {
        "no_tool": 39_641,
        "tool_required": 80_791,
        "tools_available_unnecessary": 9_643,
    }
    assert mixture["sequential_dependency_tasks"] == 45_710
    assert mixture["dependency_depth_3_plus_tasks"] == 49_048
    assert mixture["dependency_depth_4_plus_tasks"] == 40_330
    assert mixture["dependency_depth_8_plus_tasks"] == 13_054
    assert mixture["dependency_depth_16_plus_tasks"] == 7_640
    assert mixture["tool_required_dependency_depth_4_plus_tasks"] == 12_146
    assert mixture["compositional_capacity_bucket_counts"] == {
        "8": 6_000,
        "16": 5_000,
        "32": 4_000,
        "64": 3_000,
        "128": 2_000,
    }
    assert mixture["super_complex_64_plus_tasks"] == 5_000
    assert mixture["ultra_complex_128_tasks"] == 2_000
    assert mixture["generalization_probe_tasks_excluded"] == 4_000

    compositional = by_name["compositional-longtail-reasoning-v1"]
    assert compositional["tasks"] == longtail["semantic_tasks"] == 20_000
    assert compositional["tasks_sha256"] == longtail["tasks_sha256"]
    assert probe["semantic_tasks"] == 4_000
    assert probe["training_eligible"] is False
    assert probe["strict_holdout_axes"] == ["domain", "exact_logic_spec"]
    assert probe["exact_logic_spec_overlap_with_training"] == 0
    assert all(component["name"] != probe["name"] for component in mixture["components"])

    deep_tools = by_name["long-horizon-tool-reasoning-v1"]
    assert deep_tools["tasks"] == long_horizon["semantic_tasks"] == 12_000
    assert deep_tools["tasks_sha256"] == long_horizon["tasks_sha256"]
    assert deep_tools["causal_jobs_sha256"] == long_horizon["causal_jobs_sha256"]
    assert long_horizon["depth_4_plus_tasks"] == 12_000
    assert long_horizon["depth_6_plus_tasks"] == 3_332
    assert long_horizon["review_rejected"] == 0
    assert long_horizon["exact_verifier_failures"] == 0


def test_training_trajectory_mixture_v10_combines_128_slots_and_long_horizon_tools() -> None:
    mixture = _load("data/training-trajectory-mixture-v10.json")
    by_name = {component["name"]: component for component in mixture["components"]}
    longtail = _load("data/compositional-teacher-v1.reference-manifest.json")
    long_horizon = _load("data/long-horizon-teacher-v1.reference-manifest.json")

    assert mixture["name"] == "training-trajectory-mixture-v10"
    assert mixture["version"] == 10
    assert mixture["examples"] == 301_948
    assert mixture["transitions"] == 2_032_831
    assert mixture["thought_capacity_required"] == 128
    assert mixture["max_trajectory_steps"] == 33
    assert sum(component["examples"] for component in mixture["components"]) == mixture["examples"]
    assert (
        sum(component["transitions"] for component in mixture["components"])
        == mixture["transitions"]
    )

    compositional = by_name["compositional-longtail"]
    assert compositional["examples"] == longtail["compiled_trajectories"] == 40_000
    assert compositional["transitions"] == longtail["compiled_transitions"] == 180_000
    assert compositional["sha256"] == longtail["compiled_sha256"]

    deep_tools = by_name["long-horizon-tools"]
    assert deep_tools["examples"] == long_horizon["compiled_trajectories"] == 24_000
    assert deep_tools["transitions"] == long_horizon["compiled_transitions"] == 447_626
    assert deep_tools["sha256"] == long_horizon["compiled_sha256"]


def test_training_semantic_mixture_v11_adds_surface_v2_and_deep_restraint() -> None:
    mixture = _load("data/training-semantic-mixture-v11.json")
    composed = _load("data/composed-teacher-v2.reference-manifest.json")
    long_horizon = _load("data/long-horizon-teacher-v2.reference-manifest.json")
    deep = _load("data/deep-tool-restraint-v1.reference-manifest.json")
    by_name = {component["name"]: component for component in mixture["components"]}

    assert mixture["name"] == "training-semantic-mixture-v11"
    assert mixture["version"] == 11
    assert mixture["semantic_tasks"] == 134_075
    assert mixture["mode_counts"] == {
        "no_tool": 39_641,
        "tool_required": 80_791,
        "tools_available_unnecessary": 13_643,
    }
    assert mixture["mode_fractions"] == {
        "no_tool": 0.29566,
        "tool_required": 0.60258,
        "tools_available_unnecessary": 0.10176,
    }
    assert mixture["dependency_depth_3_plus_tasks"] == 53_048
    assert mixture["dependency_depth_4_plus_tasks"] == 44_330
    assert mixture["dependency_depth_8_plus_tasks"] == 17_054
    assert mixture["dependency_depth_16_plus_tasks"] == 10_251
    assert mixture["deep_tool_restraint_tasks"] == 4_000
    assert mixture["deep_tool_restraint_depth_16_plus_tasks"] == 2_611
    assert sum(component["tasks"] for component in mixture["components"]) == 134_075

    assert "composed-tool-reasoning-v1" not in by_name
    assert "long-horizon-tool-reasoning-v1" not in by_name
    assert by_name["composed-tool-reasoning-v2"]["tasks_sha256"] == composed["tasks_sha256"]
    assert by_name["long-horizon-tool-reasoning-v2"]["tasks_sha256"] == long_horizon["tasks_sha256"]
    assert by_name["deep-tool-restraint-v1"]["tasks_sha256"] == deep["tasks_sha256"]

    assert composed["semantic_tasks"] == 12_000
    assert composed["review_rejected"] == 0
    assert composed["normalized_prompt_signatures"] == 4_672
    assert composed["largest_normalized_prompt_group"] == 11
    assert long_horizon["semantic_tasks"] == 12_000
    assert long_horizon["review_rejected"] == 0
    assert long_horizon["normalized_prompt_signatures"] == 4_062
    assert long_horizon["largest_normalized_prompt_group"] == 11
    assert deep["semantic_tasks"] == 4_000
    assert deep["review_rejected"] == 0
    assert deep["tasks_without_evidence"] == 4_000
    assert deep["plans_without_needs"] == 4_000
    assert deep["capacity_bucket_counts"] == {
        "16": 1_000,
        "32": 1_000,
        "64": 1_000,
        "128": 1_000,
    }


def test_training_trajectory_mixture_v11_uses_v2_replacements_and_deep_restraint() -> None:
    mixture = _load("data/training-trajectory-mixture-v11.json")
    composed = _load("data/composed-teacher-v2.reference-manifest.json")
    long_horizon = _load("data/long-horizon-teacher-v2.reference-manifest.json")
    deep = _load("data/deep-tool-restraint-v1.reference-manifest.json")
    by_name = {component["name"]: component for component in mixture["components"]}

    assert mixture["name"] == "training-trajectory-mixture-v11"
    assert mixture["version"] == 11
    assert mixture["examples"] == 305_948
    assert mixture["transitions"] == 2_054_831
    assert mixture["thought_capacity_required"] == 128
    assert mixture["max_trajectory_steps"] == 33
    assert sum(component["examples"] for component in mixture["components"]) == mixture["examples"]
    assert (
        sum(component["transitions"] for component in mixture["components"])
        == mixture["transitions"]
    )

    assert "composed" not in by_name
    assert "long-horizon-tools" not in by_name
    assert by_name["composed-v2"]["examples"] == composed["compiled_trajectories"] == 24_000
    assert by_name["composed-v2"]["transitions"] == composed["compiled_transitions"] == 237_058
    assert by_name["composed-v2"]["sha256"] == composed["compiled_sha256"]
    assert (
        by_name["long-horizon-tools-v2"]["examples"]
        == long_horizon["compiled_trajectories"]
        == 24_000
    )
    assert (
        by_name["long-horizon-tools-v2"]["transitions"]
        == long_horizon["compiled_transitions"]
        == 447_626
    )
    assert by_name["long-horizon-tools-v2"]["sha256"] == long_horizon["compiled_sha256"]
    assert by_name["deep-tool-restraint"]["examples"] == deep["compiled_trajectories"] == 4_000
    assert by_name["deep-tool-restraint"]["transitions"] == deep["compiled_transitions"] == 22_000
    assert by_name["deep-tool-restraint"]["sha256"] == deep["compiled_sha256"]

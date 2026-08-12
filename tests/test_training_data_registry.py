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

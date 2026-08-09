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
    assert sum(mixture["mode_counts"].values()) == 28_055
    assert sum(component["tasks"] for component in mixture["components"]) == 28_055
    for component in mixture["components"]:
        reference = references[component["name"]]
        task_count = reference.get("tasks", reference.get("semantic_tasks"))
        assert component["tasks"] == task_count
        assert component["tasks_sha256"] == reference["tasks_sha256"]
        assert component["causal_jobs_sha256"] == reference["causal_jobs_sha256"]


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
        by_name["public-interaction-v1"]["causal_jobs_sha256"]
        == interaction["causal_jobs_sha256"]
    )

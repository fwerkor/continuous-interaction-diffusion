from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cid.compositional_training import (
    COMPOSITIONAL_FAMILIES,
    PROBE_DOMAINS,
    TRAIN_DOMAINS,
    CompositionalTrainingConfig,
    build_compositional_training_streaming,
)
from cid.data import load_jsonl
from cid.distill import load_teacher_tasks


def _small_config() -> CompositionalTrainingConfig:
    return CompositionalTrainingConfig(
        train_capacity_counts=((8, 10), (16, 10), (32, 10), (64, 10), (128, 10)),
        probe_capacity_counts=((16, 10), (32, 10), (64, 10), (128, 10)),
        variants_per_task=1,
        probe_variants_per_task=1,
    )


def _build(tmp_path: Path):
    generated = tmp_path / "generated"
    result = build_compositional_training_streaming(generated, _small_config())
    train_tasks = load_teacher_tasks(generated / "compositional-teacher-tasks-v1.jsonl")
    probe_tasks = load_teacher_tasks(generated / "generalization-probe-tasks-v1.jsonl")
    train_trajectories = load_jsonl(generated / "compositional-trajectories-v1.jsonl")
    probe_manifest = json.loads((generated / "generalization-probe-v1.manifest.json").read_text())
    return result, train_tasks, probe_tasks, train_trajectories, probe_manifest


def test_compositional_curriculum_spans_families_and_super_complex_capacity(tmp_path) -> None:
    result, train_tasks, _, train_trajectories, _ = _build(tmp_path)
    manifest = result["train_manifest"]

    assert len(train_tasks) == 50
    assert len(train_trajectories) == 50
    assert manifest["audit"]["exact_verifier_failures"] == 0
    assert manifest["audit"]["max_semantic_text_chars"] <= 144
    assert manifest["audit"]["max_live_cells"] >= 100
    assert manifest["thought_capacity_required"] == 128
    assert manifest["thought_capacity_curriculum"] == [8, 16, 32, 64, 128]

    family_counts = Counter(str(task.metadata["family"]) for task in train_tasks)
    assert set(family_counts) == set(COMPOSITIONAL_FAMILIES)
    assert set(family_counts.values()) == {5}

    cap128 = [task for task in train_tasks if int(task.metadata["thought_capacity_bucket"]) == 128]
    assert cap128
    assert min(int(task.metadata["target_live_cells"]) for task in cap128) >= 76
    assert max(int(task.metadata["target_live_cells"]) for task in cap128) <= 120


def test_compositional_probe_is_domain_disjoint_and_not_training_eligible(tmp_path) -> None:
    result, train_tasks, probe_tasks, _, probe_manifest = _build(tmp_path)

    train_domains = {str(task.metadata["domain"]) for task in train_tasks}
    probe_domains = {str(task.metadata["domain"]) for task in probe_tasks}
    assert train_domains <= set(TRAIN_DOMAINS)
    assert probe_domains <= set(PROBE_DOMAINS)
    assert train_domains.isdisjoint(probe_domains)
    assert probe_manifest["training_eligible"] is False
    assert result["train_manifest"]["ood_probe_excluded_from_training"] is True


def test_compositional_trajectories_stay_within_declared_capacity_bucket(tmp_path) -> None:
    _, train_tasks, _, train_trajectories, _ = _build(tmp_path)
    task_by_id = {task.task_id: task for task in train_tasks}

    for trajectory in train_trajectories:
        task = task_by_id[trajectory.example_id]
        capacity = int(task.metadata["thought_capacity_bucket"])
        assert all(target.slot < capacity for target in trajectory.thought_targets)
        assert int(trajectory.metadata["thought_capacity_bucket"]) == capacity

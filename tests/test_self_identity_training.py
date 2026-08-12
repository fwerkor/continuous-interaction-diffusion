import json
from pathlib import Path

from cid.distill import review_teacher_plans
from cid.self_identity_training import (
    FAMILIES,
    SelfIdentityTrainingConfig,
    build_self_identity_training,
    generate_self_identity_tasks_and_plans,
    load_self_identity_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data/cid-self-identity-v1.contract.json"


def test_self_identity_generation_is_balanced_unique_and_bilingual() -> None:
    contract = load_self_identity_contract(CONTRACT)
    tasks, plans = generate_self_identity_tasks_and_plans(
        contract,
        SelfIdentityTrainingConfig(count_per_family=8, seed=7, variants_per_task=1),
    )

    assert len(tasks) == len(FAMILIES) * 8
    assert len({task.prompt for task in tasks}) == len(tasks)
    assert {task.metadata["family"] for task in tasks} == set(FAMILIES)
    assert sum(task.metadata["language"] == "zh" for task in tasks) == len(FAMILIES) * 2
    assert all(not task.evidence and not task.source_descriptors for task in tasks)
    assert all(
        "Continuous Interaction Diffusion" in plan.frames[0].cells[0].semantic_text
        for plan in plans
    )
    assert all(review.accepted for review in review_teacher_plans(tasks, plans))


def test_self_identity_plans_ground_identity_and_architecture() -> None:
    contract = load_self_identity_contract(CONTRACT)
    tasks, plans = generate_self_identity_tasks_and_plans(
        contract,
        SelfIdentityTrainingConfig(count_per_family=1, seed=3, variants_per_task=1),
    )

    assert len(tasks) == len(FAMILIES)
    for plan in plans:
        initial, final = plan.frames
        identity = next(cell for cell in initial.cells if cell.cell_id == "self_identity")
        architecture = next(
            cell for cell in initial.cells if cell.cell_id == "architecture_contract"
        )
        conclusion = next(cell for cell in final.cells if cell.cell_id == "answer")
        assert identity.anchors and identity.anchors[0].object_id == "cid:self"
        assert architecture.links
        assert conclusion.links
        assert "CID" in initial.display


def test_build_self_identity_training_writes_reviewed_trajectories(tmp_path: Path) -> None:
    outputs = {
        "tasks_output": tmp_path / "tasks.jsonl",
        "plans_output": tmp_path / "plans.jsonl",
        "reviews_output": tmp_path / "reviews.jsonl",
        "trajectories_output": tmp_path / "trajectories.jsonl",
        "trajectory_manifest_output": tmp_path / "trajectories.manifest.json",
        "reference_manifest_output": tmp_path / "reference.json",
    }
    manifest = build_self_identity_training(
        contract_path=CONTRACT,
        config=SelfIdentityTrainingConfig(count_per_family=2, seed=11, variants_per_task=2),
        **outputs,
    )

    assert manifest["semantic_tasks"] == len(FAMILIES) * 2
    assert manifest["accepted_plans"] == len(FAMILIES) * 2
    assert manifest["review_rejected"] == 0
    assert manifest["compiled_trajectories"] == len(FAMILIES) * 4
    assert manifest["compiled_transitions"] == len(FAMILIES) * 4
    trajectory_manifest = json.loads(outputs["trajectory_manifest_output"].read_text())
    assert trajectory_manifest["examples"] == len(FAMILIES) * 4
    assert trajectory_manifest["sources"] == []

import json

from cid.distill import build_teacher_request, load_teacher_tasks
from cid.public_training import PublicTrainingConfig, prepare_public_distillation


def _base_record(**overrides):
    raw = {
        "task_id": "pub-task-1",
        "semantic_id": "semantic-1",
        "split": "train",
        "task_kind": "math_word_problem",
        "prompt": "What is 2 + 3?",
        "reference_answer": "5",
        "source": {
            "dataset_id": "gsm8k-main",
            "repo": "openai/gsm8k",
            "revision": "rev",
            "license": "MIT",
            "upstream_config": "main",
            "upstream_split": "train",
            "row_key": "train.parquet:1",
            "use": "general_reasoning",
        },
        "resources": {},
        "metadata": {"reference_solution": "2 + 3 = 5"},
    }
    raw.update(overrides)
    return raw


def _write_pool(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_teacher_reference_answer_is_persisted_but_hidden_from_request(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, [_base_record()])
    tasks_path = tmp_path / "tasks.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    manifest_path = tmp_path / "manifest.json"

    prepare_public_distillation(
        pool,
        tasks_path,
        requests_path,
        manifest_path,
        PublicTrainingConfig(unnecessary_tool_fraction=0.0),
    )

    (task,) = load_teacher_tasks(tasks_path)
    assert task.reference_answer == "5"
    assert task.metadata["task_kind"] == "math_word_problem"
    assert "reference_solution" not in task.metadata
    request = build_teacher_request(task).prompt
    assert '"reference_answer"' not in request
    assert "2 + 3 = 5" not in request


def test_public_distillation_respects_owned_split_and_deterministic_modes(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    records = [
        _base_record(task_id="train-a", semantic_id="a"),
        _base_record(task_id="train-b", semantic_id="b"),
        _base_record(task_id="val-a", semantic_id="c", split="validation"),
    ]
    _write_pool(pool, records)

    first_tasks = tmp_path / "first-tasks.jsonl"
    first_requests = tmp_path / "first-requests.jsonl"
    first_manifest = tmp_path / "first-manifest.json"
    second_tasks = tmp_path / "second-tasks.jsonl"
    second_requests = tmp_path / "second-requests.jsonl"
    second_manifest = tmp_path / "second-manifest.json"
    config = PublicTrainingConfig(seed=7, unnecessary_tool_fraction=0.5)

    first = prepare_public_distillation(
        pool, first_tasks, first_requests, first_manifest, config
    )
    second = prepare_public_distillation(
        pool, second_tasks, second_requests, second_manifest, config
    )

    assert first["tasks"] == 2
    assert first["tasks_sha256"] == second["tasks_sha256"]
    assert first["requests_sha256"] == second["requests_sha256"]
    assert sum(first["mode_counts"].values()) == 2
    assert {task.task_id for task in load_teacher_tasks(first_tasks)} == {"train-a", "train-b"}


def test_hotpot_public_task_becomes_search_then_supporting_reads(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    hotpot = _base_record(
        task_id="hotpot-1",
        semantic_id="hotpot-semantic",
        task_kind="multi_hop_qa",
        prompt="Which item was released first?",
        reference_answer="Item A",
        source={
            "dataset_id": "hotpotqa-distractor",
            "repo": "hotpotqa/hotpot_qa",
            "revision": "rev",
            "license": "CC-BY-SA-4.0",
            "upstream_config": "distractor",
            "upstream_split": "train",
            "row_key": "train.parquet:4",
            "use": "toolizable_retrieval",
        },
        resources={
            "evidence_bank": [
                {"title": "Item A", "sentences": ["Item A was released in 2001."]},
                {"title": "Noise", "sentences": ["This is unrelated."]},
                {"title": "Item B", "sentences": ["Item B was released in 2005."]},
            ]
        },
        metadata={
            "upstream_id": "up-1",
            "supporting_facts": {
                "title": ["Item A", "Item B"],
                "sent_id": [0, 0],
            },
            "question_type": "comparison",
            "level": "medium",
        },
    )
    _write_pool(pool, [hotpot])
    tasks_path = tmp_path / "tasks.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_public_distillation(
        pool,
        tasks_path,
        requests_path,
        manifest_path,
        PublicTrainingConfig(unnecessary_tool_fraction=0.0),
    )

    (task,) = load_teacher_tasks(tasks_path)
    assert manifest["mode_counts"] == {"tool_required": 1}
    assert [descriptor["name"] for descriptor in task.source_descriptors] == [
        "workspace_search",
        "workspace_read",
    ]
    assert [evidence.evidence_id for evidence in task.evidence] == [
        "search-results",
        "support-0",
        "support-1",
    ]
    assert task.evidence[0].arguments == {"query": "Which item was released first?"}
    assert task.evidence[1].arguments == {"resource_id": "doc-00"}
    assert task.evidence[2].arguments == {"resource_id": "doc-02"}
    assert task.evidence[0].depends_on == ()
    assert task.evidence[1].depends_on == ("search-results",)
    assert task.evidence[2].depends_on == ("search-results",)
    assert "supporting_facts" not in task.metadata
    request = build_teacher_request(task).prompt
    assert "Item A was released in 2001." in request
    assert '"reference_answer"' not in request


def test_unnecessary_tool_mode_exposes_schema_without_fake_evidence(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, [_base_record()])
    tasks_path = tmp_path / "tasks.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    manifest_path = tmp_path / "manifest.json"

    prepare_public_distillation(
        pool,
        tasks_path,
        requests_path,
        manifest_path,
        PublicTrainingConfig(unnecessary_tool_fraction=1.0),
    )

    (task,) = load_teacher_tasks(tasks_path)
    assert task.metadata["training_mode"] == "tools_available_unnecessary"
    assert len(task.source_descriptors) == 2
    assert task.evidence == ()

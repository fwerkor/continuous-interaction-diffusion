import json

from cid.causal_distill import build_causal_teacher_job
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


def test_mbpp_teacher_prompt_exposes_public_tests_but_not_hidden_code(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    record = _base_record(
        task_id="mbpp-1",
        semantic_id="mbpp-semantic",
        task_kind="python_programming",
        prompt="Write a function to increment a number.",
        reference_answer="def inc(x):\n    return x + 1",
        source={
            "dataset_id": "mbpp-full-train",
            "repo": "google-research-datasets/mbpp",
            "revision": "rev",
            "license": "CC-BY-4.0",
            "upstream_config": "full",
            "upstream_split": "train",
            "row_key": "train.parquet:1",
            "use": "general_reasoning",
        },
        metadata={
            "upstream_task_id": 1,
            "tests": ["assert inc(1) == 2", "assert inc(-1) == 0"],
            "challenge_tests": ["assert inc(100) == 101"],
            "test_setup_code": "import math",
        },
    )
    _write_pool(pool, [record])
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
    assert task.task_id == "mbpp-1"
    assert "Public evaluation contract:" in task.prompt
    assert "import math" in task.prompt
    assert "assert inc(1) == 2" in task.prompt
    assert "assert inc(-1) == 0" in task.prompt
    assert "assert inc(100) == 101" not in task.prompt
    assert "def inc(x):" not in task.prompt
    assert task.reference_answer.startswith("def inc(x):")
    assert json.loads(manifest_path.read_text())["python_public_test_tasks"] == 1
    assert task.metadata["public_tests"] == ["assert inc(1) == 2", "assert inc(-1) == 0"]
    assert task.metadata["public_test_setup_code"] == "import math"
    assert "challenge_tests" not in task.metadata
    request = build_teacher_request(task).prompt
    assert "assert inc(1) == 2" in request
    assert "assert inc(100) == 101" not in request
    assert "def inc(x):" not in request


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


def test_musique_public_task_uses_decomposition_dependency_dag(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    musique = _base_record(
        task_id="musique-1",
        semantic_id="musique-semantic",
        task_kind="multi_hop_qa",
        prompt="Where was the author of Work born?",
        reference_answer="Paris",
        source={
            "dataset_id": "musique-train",
            "repo": "awinml/musique",
            "revision": "rev",
            "license": "CC-BY-4.0",
            "upstream_config": "default",
            "upstream_split": "train",
            "row_key": "train.parquet:7",
            "use": "toolizable_retrieval",
        },
        resources={
            "evidence_bank": [
                {"title": "Work", "sentences": ["Work was written by Alice Example."]},
                {"title": "Noise", "sentences": ["Unrelated text."]},
                {"title": "Alice Example", "sentences": ["Alice Example was born in Paris."]},
            ]
        },
        metadata={
            "supporting_facts": {"title": ["Work", "Alice Example"], "sent_id": [0, 0]},
            "hop_count": 2,
            "question_decomposition": [
                {
                    "id": 10,
                    "question": "Work >> author",
                    "answer": "Alice Example",
                    "paragraph_support_idx": 0,
                },
                {
                    "id": 11,
                    "question": "Where was #1 born?",
                    "answer": "Paris",
                    "paragraph_support_idx": 2,
                },
            ],
        },
    )
    _write_pool(pool, [musique])
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
    assert task.metadata["interaction_pattern"] == "decomposition_dag"
    assert task.metadata["dependency_depth"] == 2
    assert manifest["interaction_pattern_counts"] == {"decomposition_dag": 1}
    assert manifest["dependency_depth_histogram"] == {"2": 1}
    assert [item.evidence_id for item in task.evidence] == [
        "search-hop-0",
        "read-hop-0",
        "search-hop-1",
        "read-hop-1",
    ]
    assert task.evidence[0].depends_on == ()
    assert task.evidence[1].depends_on == ("search-hop-0",)
    assert task.evidence[2].depends_on == ("read-hop-0",)
    assert task.evidence[2].arguments == {"query": "Where was Alice Example born?"}
    assert task.evidence[3].depends_on == ("search-hop-1",)

    job = build_causal_teacher_job(task)
    assert [item["evidence_id"] for item in job.stages[0].available_evidence] == [
        "search-hop-0"
    ]
    assert [item["evidence_id"] for item in job.stages[2].available_evidence] == [
        "search-hop-1"
    ]


def test_musique_independent_decomposition_roots_can_launch_in_parallel(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    musique = _base_record(
        task_id="musique-parallel",
        semantic_id="musique-parallel-semantic",
        task_kind="multi_hop_qa",
        prompt="Are A and B from the same country?",
        reference_answer="no",
        source={
            "dataset_id": "musique-train",
            "repo": "awinml/musique",
            "revision": "rev",
            "license": "CC-BY-4.0",
            "upstream_config": "default",
            "upstream_split": "train",
            "row_key": "train.parquet:8",
            "use": "toolizable_retrieval",
        },
        resources={
            "evidence_bank": [
                {"title": "A", "sentences": ["A is French."]},
                {"title": "B", "sentences": ["B is Canadian."]},
            ]
        },
        metadata={
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "hop_count": 2,
            "question_decomposition": [
                {
                    "id": 20,
                    "question": "What country is A from?",
                    "answer": "France",
                    "paragraph_support_idx": 0,
                },
                {
                    "id": 21,
                    "question": "What country is B from?",
                    "answer": "Canada",
                    "paragraph_support_idx": 1,
                },
            ],
        },
    )
    _write_pool(pool, [musique])
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
    assert task.metadata["parallel_root_needs"] == 2
    assert task.metadata["dependency_depth"] == 1
    job = build_causal_teacher_job(task)
    assert [item["evidence_id"] for item in job.stages[0].available_evidence] == [
        "search-hop-0",
        "search-hop-1",
    ]


def test_musique_literal_hash_number_is_not_a_decomposition_reference(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    musique = _base_record(
        task_id="musique-hash-title",
        semantic_id="musique-hash-title-semantic",
        task_kind="multi_hop_qa",
        prompt="What song did the lyricist of #9 Dream write for David Bowie?",
        reference_answer="Fame",
        source={
            "dataset_id": "musique-train",
            "repo": "awinml/musique",
            "revision": "rev",
            "license": "CC-BY-4.0",
            "upstream_config": "default",
            "upstream_split": "train",
            "row_key": "train.parquet:9",
            "use": "toolizable_retrieval",
        },
        resources={
            "evidence_bank": [
                {"title": "Number 9 Dream", "sentences": ["The lyricist was John Lennon."]},
                {"title": "Fame", "sentences": ["John Lennon co-wrote Fame for David Bowie."]},
            ]
        },
        metadata={
            "supporting_facts": {"title": ["Number 9 Dream", "Fame"], "sent_id": [0, 0]},
            "hop_count": 2,
            "question_decomposition": [
                {
                    "id": 30,
                    "question": "#9 Dream >> lyrics by",
                    "answer": "John Lennon",
                    "paragraph_support_idx": 0,
                },
                {
                    "id": 31,
                    "question": "what song did #1 write for David Bowie",
                    "answer": "Fame",
                    "paragraph_support_idx": 1,
                },
            ],
        },
    )
    _write_pool(pool, [musique])
    tasks_path = tmp_path / "tasks.jsonl"
    prepare_public_distillation(
        pool,
        tasks_path,
        tmp_path / "requests.jsonl",
        tmp_path / "manifest.json",
        PublicTrainingConfig(unnecessary_tool_fraction=0.0),
    )

    (task,) = load_teacher_tasks(tasks_path)
    assert task.evidence[0].arguments == {"query": "#9 Dream >> lyrics by"}
    assert task.evidence[0].depends_on == ()
    assert task.evidence[2].arguments == {
        "query": "what song did John Lennon write for David Bowie"
    }
    assert task.evidence[2].depends_on == ("read-hop-0",)

# Public dataset registry

CID keeps every externally sourced training-task dataset in pinned registries. The general-purpose
pool is defined by `configs/public-datasets.json`; the interaction-heavy multi-hop pool is defined by
`configs/public-interaction-datasets.json`. These registries are the source of truth for repository,
config/split, revision, license, intended use, and sampling quota. Generated task-pool JSONL files
are intentionally ignored by Git; they can be reproduced with `cid build-public-task-pool`.

## Rules

- Only sources listed in one of the committed public-dataset registries may enter a public task
  pool.
- Every source is pinned to an exact upstream revision.
- Training-task construction uses upstream **training** data only. Public benchmark test splits are
  kept out of this pool so they remain available for later evaluation.
- The semantic task is assigned to CID train/validation/test before any toolization, timing
  randomization, or counterfactual expansion. All descendants of one `semantic_id` must remain in
  the same split.
- Every generated record retains repository, revision, license, upstream config/split, and source
  row provenance.
- Licenses shown here describe the upstream datasets. Distribution or publication of derived data
  must continue to satisfy the corresponding upstream terms.

## Public task pool v1

| Registry ID | Upstream dataset | Config / split | Pinned revision | License | v1 quota | Intended use |
|---|---|---|---|---|---:|---|
| `gsm8k-main` | `openai/gsm8k` | `main / train` | `740312add88f781978c0658806c59bc2815b9866` | MIT | 2,500 | General mathematical reasoning |
| `hendrycks-math` | `EleutherAI/hendrycks_math` | seven subjects / train | `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT | 2,500 | Competition mathematics |
| `mmlu-auxiliary-train` | `cais/mmlu` | `all / auxiliary_train` | `c30699e8356da336a370243923dbaf21066bb9fe` | MIT | 2,000 | Broad knowledge and multiple-choice reasoning |
| `hotpotqa-distractor` | `hotpotqa/hotpot_qa` | `distractor / train` | `1908d6afbbead072334abe2965f91bd2709910ab` | CC-BY-SA-4.0 | 1,500 | Multi-hop retrieval/toolization; context becomes an evidence bank |
| `arc-challenge` | `allenai/ai2_arc` | `ARC-Challenge / train` | `210d026faf9955653af8916fad021475a3f00453` | CC-BY-SA-4.0 | 600 | Science reasoning |
| `arc-easy` | `allenai/ai2_arc` | `ARC-Easy / train` | `210d026faf9955653af8916fad021475a3f00453` | CC-BY-SA-4.0 | 526 | Science reasoning |
| `mbpp-full-train` | `google-research-datasets/mbpp` | `full / train` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC-BY-4.0 | 374 | Python programming with executable tests |

Total: **10,000 semantic tasks**.

The initial CID-owned split is deterministic and content-keyed: 90% train, 5% validation, and 5%
test in expectation. It does not reuse the upstream benchmark test split. The pinned v1 build has
9,030 train, 493 validation, and 477 test tasks, with output SHA-256
`bb92a3d6fccad2687cd5805ec5aa3eaff404644dea76cce153a088853d49f17b`.

## Record shape

The normalized JSONL record contains:

- `task_id`: stable source-row-derived ID;
- `semantic_id`: normalized-prompt hash used for deduplication and split assignment;
- `split`: CID-owned `train`, `validation`, or `test`;
- `task_kind`, `prompt`, and `reference_answer`;
- `source`: exact upstream provenance and license;
- `resources`: material available for later toolization but not necessarily visible in the prompt;
- `metadata`: solutions, tests, choices, supporting-fact annotations, and other task-specific fields.

HotpotQA is intentionally normalized as `question -> prompt` plus `context -> resources.evidence_bank`.
This lets later CID dataset construction expose the evidence through retrieval/file/database sources
without rewriting the underlying semantic question.

## Reproducing v1

```bash
pip install -e '.[data]'
cid build-public-task-pool
```

The default outputs are:

```text
data/generated/public-task-pool-v1.jsonl
data/generated/public-task-pool-v1.manifest.json
```

The manifest records the generated file SHA-256, registry SHA-256, source/task-kind/license counts,
CID split counts, and semantic duplicates removed during construction.
`data/public-task-pool-v1.reference-manifest.json` is the committed reference for the pinned v1
build; generated data should match it without committing the 23 MB JSONL itself.

## Public interaction task pool v1

The interaction pool adds public tasks that already contain explicit multi-document evidence. It
is kept separate from the general pool so the original 10k build remains immutable and auditable.
Every accepted interaction task must span at least two distinct supporting documents.

| Registry ID | Upstream dataset | Config / split | Pinned revision | License | v1 quota | Intended use |
|---|---|---|---|---|---:|---|
| `2wikimultihopqa-train` | `xanhho/2WikiMultihopQA` (official mirror of 2WikiMultiHopQA) | default / train | `612bc5039a457880d9e7d84c3b0a4cf154b70e4f` | Apache-2.0 | 5,000 | Multi-document retrieval with explicit reasoning evidence |
| `musique-train` | `awinml/musique` (mirror of MuSiQue) | default / train | `3ac762478df3609852f18dd33652a16820660a5e` | CC-BY-4.0 upstream dataset | 5,000 | 2–4 hop multi-document retrieval |

The pinned build contains **10,000 semantic tasks** with 9,025 train, 482 validation, and 493 test
examples. It has output SHA-256
`44cc895235a0536a8a75b5ee312f17a0537f7824055009b5ddc74134a985d920`.
The accepted support-document distribution is 7,537 tasks with two supporting documents, 1,121
with three, and 1,342 with four. Rows whose supporting annotations collapse to only one distinct
document are rejected and deterministically replaced by later candidates.

Reproduce it with:

```bash
cid build-public-task-pool \
  --registry configs/public-interaction-datasets.json \
  --output data/generated/public-interaction-task-pool-v1.jsonl \
  --manifest-output data/generated/public-interaction-task-pool-v1.manifest.json
```

`data/public-interaction-task-pool-v1.reference-manifest.json` pins the committed reference build.

## Teacher-ready semantic mixture v1

`cid prepare-public-distillation` converts a normalized public task pool into timing-free
`TeacherTask` records. Public answer/solution annotations remain in the private task record for
later auditing but are excluded from the teacher-visible payload. Likewise, MATH solutions, MMLU
answer indices, Hotpot supporting annotations, and executable MBPP labels are not copied into
teacher-visible metadata.

For retrieval tasks the tool environment is normalized to two read-only source schemas:

```text
workspace_search(query) -> candidate resource IDs and titles
workspace_read(resource_id) -> one hidden document
```

The search evidence is the dependency root. All supporting-document reads depend only on that
search result, so the 2–4 reads become simultaneously eligible after search completes. Causal
teacher jobs expose only evidence that has arrived at the current stage; future evidence values
remain hidden even though the offline orchestrator stores the complete task.

The current train mixture is pinned by `data/training-semantic-mixture-v1.json`:

- 18,055 semantic tasks total;
- 10,391 tool-required tasks (57.55%);
- 6,921 no-tool tasks (38.33%);
- 743 tasks with tools available but unnecessary (4.12%);
- zero semantic-prompt overlap between the base and interaction pools.

Reference manifests for the teacher-ready components are
`data/public-teacher-v1.train.reference-manifest.json` and
`data/public-interaction-teacher-v1.train.reference-manifest.json`.

When adding another public dataset, update its registry and this document in the same commit.
Prefer upstream training splits with clear licensing and preserve evaluation splits for benchmarking
whenever possible.

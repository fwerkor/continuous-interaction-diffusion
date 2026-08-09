# Public dataset registry

CID keeps every externally sourced training-task dataset in a pinned registry at
`data/public-datasets.json`. The registry is the source of truth for repository/config/split,
revision, license, intended use, and sampling quota. Generated task-pool JSONL files are intentionally
ignored by Git; they can be reproduced with `cid build-public-task-pool`.

## Rules

- Only sources listed in `data/public-datasets.json` may enter the public task pool.
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

When adding another public dataset, update both `data/public-datasets.json` and this document in the
same commit. Prefer upstream training splits with clear licensing and preserve evaluation splits for
benchmarking whenever possible.

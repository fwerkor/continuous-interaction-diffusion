# Training plan

The implementation is organized for staged conversion rather than immediate from-scratch 4B
pretraining.

## Dataset artifact location

The GitHub repository contains implementation, tests, documentation, and the small source registries
under `configs/`. Released training data, component manifests, and materialized datasets are published
only in the Hugging Face dataset repository `fwerkor/CID-Dataset`. The local `data/` directory is a
gitignored workspace used by builders and training commands. Historical `data/...` paths mentioned
below refer to dataset-release artifacts, not files tracked by GitHub.

## Stage 0 — runtime oracle

Use scripted/oracle policies to validate binding lifecycle, event timing, cache reuse, dynamic
refresh, fact protection, and metrics independently of neural quality.

## Stage 1 — supervised CID adapters

Start from an existing masked-diffusion language model. Freeze most of the backbone initially and
train:

The supported full-sequence diffusion backbones are `GSAI-ML/iLLaDA-8B-Base`,
`inclusionAI/LLaDA-MoE-7B-A1B-Base`, and `LiquidAI/LFM2.5-Encoder-350M-Diffusion`. The latter is the
base for CID-v1-0.4B. The common CID adapter reuses each model's native token embedding,
bidirectional hidden-state stack, and LM head. LFM2 keeps its pretrained mixture of full-attention
and non-causal short-convolution layers; no architecture morphing is performed. With
`freeze_backbone=True`, only the CID projections, external-perception fusion, and prediction heads
are trainable; later stages can unfreeze the same backbone without changing the tensor/runtime ABI.
For LLaDA-MoE, Stage B also restores the upstream router load-balancing objective while Stage A
skips router-logit materialization because the router is frozen. The LFM2.5 checkpoint is distributed
under the upstream LFM Open License v1.0.

- TCT slot projection and role heads;
- empty-slot allocation and occupied-cell lifecycle heads;
- source/need confidence heads;
- argument and typed grounding heads;
- percept encoder/cross-attention adapters;
- local support/conflict and lifecycle heads.

The first trainable data path is `ILLaDATrajectoryTensorizer`. It turns adjacent supervised
trajectory snapshots into a model input and `CIDTargets`: current TCT occupancy is retained,
continuous cognition is corrupted, the next display target is masked with the diffusion schedule,
and allocation/lifecycle targets are derived from stable cell identity across the transition.
New cells train allocation on previously empty slots; existing cells train lifecycle on their
logical identity even if later compaction moves their physical storage.

Display corruption also includes visible replacement noise. Some selected target tokens are
replaced with wrong non-mask vocabulary tokens while the clean token remains the supervision
target. This teaches the display head to correct already-visible stale text after new evidence
arrives, rather than learning only MASK-to-token completion. At inference, masked positions are
revealed by confidence and a bounded fraction of visible positions may be rewritten when the best
alternative exceeds the current token probability by a configured margin. The mask token itself is
excluded from output candidates.

Teacher trajectories should contain pre-arrival and post-arrival states, not only final answers.
They should also supervise cell creation, retirement, optional split/merge lineage, and stable cell
identity across physical compaction. Arrival time, source freshness, cache availability, and
physical slot placement are randomized so a model cannot assign permanent semantics to slot index.
Allocation loss is masked to slots that are `EMPTY` at the current step. Lifecycle cross-entropy is
masked to existing cells and has four classes: `ACTIVE`, `WAITING`, `STABLE`, and `RETIRED`.
Runtime-gated transitions remain hard constraints during both training rollouts and inference.
Every transition also has a trajectory-level equilibrium target for the learned convergence head.
Ordinary intermediate states supervise zero; the final supervised state and snapshots containing
required `WAITING` cognition supervise one. At inference this signal means that no further useful
refinement is expected from the currently available information. The runtime interprets it as
quiescence when required evidence is pending and as a terminal candidate only when the display is
resolved. It therefore does not treat "no MASK tokens remain" as sufficient evidence that the task
is finished.
Training rollouts should also randomize slot pressure. Retired cells are archived and reclaimed by
runtime policy rather than model logits; supervision should not teach the model to encode garbage
collection decisions in TCT. Reclamation traces can be used to measure whether a trained model
creates excessive short-lived cognition or retains cells unnecessarily.

Grounding supervision is multi-valued per cell. Anchor slots learn presence, anchor kind, and a
retrieval embedding for the canonical object. Link slots learn presence, relation type, target
`ObjectKind`, and a target retrieval embedding. Presence masks let the model use fewer anchors or
links than the fixed per-cell grounding capacity. Anchor/link target order is treated as a set:
loss computation performs minimum-cost assignment between target objects and neural grounding
slots, so the model cannot exploit an arbitrary teacher-side ordering convention. The first
training stage uses trajectory-local closed-world catalogs so grounding quality can be measured
without requiring open-world entity linking.

Source arguments use a separate fixed-capacity slot set keyed by the selected source schema.
Argument slot `k` corresponds to the `k`th declared argument of that source and predicts both
presence and a retrieval query. This permits an information need to emerge before every required
argument is executable, while keeping argument names/types under the runtime-owned source schema.

### Stage A launcher and checkpoints

`CIDTrainer` is the first executable Stage A trainer. It supports random diffusion timesteps,
gradient accumulation, gradient clipping, deterministic transition shuffling, optimizer resume, and
CID-only checkpoints. When the pretrained backbone is frozen, checkpoints contain only trainable
CID parameters plus optimizer/progress/RNG state; the pinned pretrained backbone is reloaded
separately.
`load_cid_adapter_checkpoint()` restores those CID parameters directly for runtime evaluation.

`cid train` runs this path on one device. Under `torchrun`, it switches to DDP automatically. Each
rank receives the same shuffled transition order and a padded equal-length shard so every rank
executes the same number of backward passes. Frozen backbone parameters/buffers are excluded from
DDP initialization synchronization, while trainable CID parameters are synchronized. Model loading
is serialized across local ranks to avoid staging six simultaneous 8B CPU copies before transfer to
the GPUs.

`CIDTrainingStep` remains a single-example representation. `collate_training_steps()` pads prompt,
display, fact, percept, and source dimensions into a variable-length micro-batch and supplies the
corresponding attention/padding masks. `CIDTrainerConfig.micro_batch_size` controls this local batch;
gradient accumulation then scales the effective batch independently. Accumulated gradients are
normalized by the number of examples rather than by the number of micro-batches, so a smaller final
micro-batch is not overweighted. Native backbone gradient checkpointing is enabled by the launcher
by default to reduce activation
memory while retaining gradients to CID inputs through the frozen backbone. The adapter also
constructs per-sample position IDs from valid prompt and
display lengths. Padding introduced by another sample therefore does not alter the logical
positions of a trajectory's display tokens.

Adjacent transitions are also grouped into bounded `CIDRolloutWindow` sequences. Training starts
with teacher-forced T/Y inputs, then scheduled sampling linearly increases the chance that the
previous detached model prediction replaces the next transition's source state, and finally reaches
self-rollout. The teacher trajectory continues to define the correct next-state targets and external
event schedule. Predicted cognition therefore changes the model's input distribution without being
reused as its own label. Allocation masking follows the actually fed occupancy state, including
rollout errors, while runtime lifecycle legality remains outside the learned transition head.

This DDP path is for the frozen-backbone Stage A phase.

### Stage B FSDP full-parameter launcher

`cid train-full` is the quality-first full-parameter continuation path after Stage A. It uses FSDP
`FULL_SHARD` with `use_orig_params=True`, FP32 master parameters, BF16 forward/reduction, and AdamW.
The CUDA production path requires at least four GPU ranks; six A6000s are preferred for the 8B
model. CPU offload stays disabled by default, preserving the existing GPU path.
`--fsdp-cpu-offload` explicitly moves FSDP parameter/gradient shards and optimizer state to host
memory while keeping standard AdamW; this is the supported low-memory path for the 7B-A1B variant
on four 24 GB GPUs.

Stage B can also run entirely on CPU with `--device cpu`. The CPU path uses Gloo and BF16, keeps
parameters and optimizer state on host memory, and does not require four ranks. A direct one-process
launch is supported; for large multi-socket servers, `torchrun` can use multiple CPU ranks and real
FSDP sharding. `--fsdp-cpu-offload` must remain disabled because there is no separate accelerator
to offload from. For example:

```bash
OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 \
  torchrun --standalone --nproc-per-node=2 -m cid.cli train-full \
  --device cpu --dtype bf16 \
  --model LiquidAI/LFM2.5-Encoder-350M-Diffusion \
  --data /path/to/training-trajectories.jsonl \
  --output-dir /path/to/stage-b \
  --init-cid-checkpoint /path/to/stage-a-latest.pt
```

The thread count and CPU-rank count are deployment tuning parameters rather than model semantics.
On a single-process CPU launch, PyTorch may reduce `FULL_SHARD` to `NO_SHARD`; multi-rank CPU
launches retain FSDP sharding.

Stage B separates optimization policy by parameter role. CID modules use the configured peak
learning rate (`1e-5` by default), while iLLaDA backbone groups use `backbone_lr_scale=0.5`. Matrix
and embedding weights receive `weight_decay=0.01`; one-dimensional norm weights and biases receive
zero decay. The trainer preserves these group multipliers through a 3% linear warmup and cosine
decay to 10% of peak. The default target transition batch is 32 and gradient accumulation is derived
from the actual world size unless explicitly overridden. At micro-batch 1 this resolves to 8
accumulation steps on four ranks and 5 on six ranks.

Stage B starts at rollout probability 1.0 by default. Stage A has already performed the
teacher-forcing-to-rollout curriculum, so restarting that curriculum after unfreezing the backbone
would train the full model on an easier distribution than the one used at inference. `--epochs` is a
target-total count: resuming a partially completed one-epoch run with `--epochs 1` finishes that
same epoch instead of scheduling a second one.

The adapter is initially loaded as FP32 on CPU. A frozen BF16 snapshot of the input embedding is
copied to the local GPU for dataset semantic transport targets and external-memory text, then FSDP
moves and shards the trainable wrap units via `device_id`. This avoids a transient full-model FP32
GPU copy before sharding. Prompt and display token IDs still use the live trainable embedding inside
the FSDP forward, while the fixed snapshot keeps target embeddings stationary as the backbone
changes.

Stage B checkpoints use FSDP sharded model state with `torch.distributed.checkpoint`; optimizer
state remains rank-local. Rank-local trainer state preserves the LR schedule, diffusion/shuffle RNG,
transition and optimizer counts, completed epochs, and the per-rank rollout-window cursor. The
default launcher logs every 100 optimizer steps and writes a periodic checkpoint every 2,500 steps
at a clean gradient-accumulation boundary. `stage-b-latest` is updated only after a checkpoint is
complete, and the previous periodic checkpoint is removed only after its successor is safe. Resume
requires the same world size and exact training JSONL SHA-256, and the stored trainer configuration
prevents silently changing LR, accumulation, or rollout policy midway through a run.

A fresh Stage B run requires `--init-cid-checkpoint`; `--resume` is mutually exclusive and restores
the complete Stage B model/optimizer state. `scripts/train_cid_v1_7b_a1b_4x3090.sh` provides a
separate, resumable 4×3090 launcher whose default root is `/workspace/cid-v1-7b-a1b`; it never
uses the dense 8B output directories. The repository tests the optimizer grouping and LR
multipliers, a CPU FSDP checkpoint round trip, and a true two-rank Gloo `FULL_SHARD`
forward/backward/distributed-checkpoint smoke. The two-rank smoke is a correctness test only; the
8B production launcher itself enforces four or more GPU ranks.

### Neural replay benchmark

`cid benchmark` loads either a Stage A CID-only checkpoint or a Stage B sharded model checkpoint and
runs the neural policy against the step-exact replay sources in `cid.evaluation`. The default starts
from an empty TCT and a masked display. `--seed-teacher-state` supplies only the dataset's step-0 TCT
and is intended as a diagnostic for separating initial allocation failures from downstream binding,
assimilation, revision, and convergence behavior.

Stage A evaluation is single-process. Stage B evaluation runs under the checkpoint's original FSDP
world size and restores model shards only; no optimizer is constructed or loaded. Per-case JSONL
records final text/token IDs, runtime steps, and the complete task evaluation. The summary JSON
aggregates convergence, exact display accuracy, observation coverage/staleness, latent-to-executable
delay, binding-to-observation delay, and observation-to-projection lag.

### Teacher distillation compiler

`cid.distill` keeps semantic teacher supervision separate from runtime scheduling. A `TeacherTask`
contains the immutable prompt, protected facts, source schemas, and evidence values, but deliberately
omits their original arrival steps. `build_teacher_request()` asks a strong teacher for a compact
semantic plan made of typed cognitive frames and information needs. It explicitly rejects private
reasoning transcripts as a target representation: `semantic_text` is a short state summary used only
as dataset transport for a latent target.

Teacher plans cannot choose physical TCT slots, diffusion steps, evidence arrival times, or cache
schedules. The parser rejects those fields and any unknown control fields rather than silently
discarding them. `compile_teacher_plans()` then independently samples physical slot placement and
event delays. While a required observation is outstanding, the compiler inserts `WAITING` frames;
at arrival it forces the affected cell through `ACTIVE` for assimilation before applying a teacher
state that may become `STABLE`. The result is the same `TrajectoryExample` ABI consumed by the
synthetic generator and Stage A trainer.

The offline flow is:

```text
TrajectoryExample tasks
    -> prepare-distillation
    -> timing-free teacher request JSONL
    -> strong teacher semantic plan JSONL
    -> review-distillation
    -> compile-distillation
    -> randomized supervised TrajectoryExample JSONL
```

This separation is intentional. Distillation should improve semantic supervision quality without
allowing teacher-specific call timing or a fixed physical slot convention to become part of the
student's learned policy.

`review-distillation` is a deterministic pre-compiler gate. It rejects semantic plans that expose
future evidence before the corresponding `after:<evidence>` phase, bind required source arguments
to values inconsistent with the supplied evidence, omit a conclusion-bearing final state, or
duplicate an already accepted task/plan payload. The review report carries a stable content
fingerprint for every plan, so filtering decisions can be audited independently of the training
run.

For public training data, `prepare-public-distillation` additionally writes a causal teacher-job
specification. Evidence records carry an explicit `depends_on` relation. The orchestrator first
shows only the immutable task and dependency-ready evidence contracts; each later teacher call sees
the accepted previous semantic state plus the evidence that has just arrived. Future evidence values
are never placed in that call's visible payload. In retrieval tasks the search result is the root and
all supporting-document reads depend on it, allowing multiple reads to become eligible together.
The older single-request JSONL is retained as an inspection/debug artifact; production teacher
generation should consume the causal job specification.

The currently pinned public semantic mixture is `data/training-semantic-mixture-v1.json`. It combines
the general 10k public task pool with a separate 10k interaction-heavy pool derived from the training
splits of 2WikiMultiHopQA and MuSiQue. After the CID-owned train split, it contains 18,055 semantic
tasks: 10,391 tool-required, 6,921 no-tool, and 743 tools-available-but-unnecessary tasks. Every task
in the interaction component spans at least two distinct supporting documents.

The overall semantic mixture is `data/training-semantic-mixture-v2.json`. It adds 10,000 generated
mechanism tasks, with 2,000 examples from each of the five deterministic families. These seeds are
converted back to timing-free `TeacherTask` records and relabeled by the same strong semantic
teacher; the programmatic TCT text in the seed trajectories is not treated as final high-quality
supervision. `data/mechanism-seed-v1.reference-manifest.json` pins the seed trajectory build and
`data/mechanism-teacher-v1.reference-manifest.json` pins the teacher-task/causal-job ABI.

The resulting v2 mixture contains 28,055 semantic tasks: 20,391 tool-required, 6,921 no-tool, and
743 tools-available-but-unnecessary. The generated mechanism component contributes:

- 2,000 static exact reads;
- 2,000 delayed reads;
- 2,000 dynamic-state tasks where one `freshness=always` binding receives a later version;
- 2,000 streaming tasks where later chunks reuse the original persistent binding;
- 2,000 competing-source tasks where two independent reads become executable together.

The current overall mixture is `data/training-semantic-mixture-v3.json`. It adds 12,000
computational-tool semantic tasks, bringing the total to 40,055: 31,191 tool-required, 6,921
no-tool, and 1,943 tools-available-but-unnecessary. The computational component contains 1,200
examples from each of ten balanced families: direct calculator use, applied formula evaluation,
sequential calculator calls, parallel calculator fan-out/merge, calculator-available negative
examples, Python statistics, Python enumeration, record lookup followed by calculation, calculator
followed by Python, and parallel record lookup followed by a merged calculation. Six thousand of
these tasks have dependency depth two.

`cid build-computational-training` deterministically produces the teacher tasks and causal jobs.
The pinned self-distillation contains 31,200 causal teacher stages. All 12,000 semantic plans pass
`review-distillation`; compilation uses two independently sampled timing/slot schedules per semantic
task, yielding 24,000 `TrajectoryExample` records and 111,609 adjacent supervised transitions.
`data/computational-teacher-v1.reference-manifest.json` records the exact hashes. Teacher TCT text is
kept as short semantic state rather than a reasoning transcript; arrived calculator/Python/lookup
percepts carry anchors and `observes` links, executable needs carry `requests` links, and terminal
conclusions carry `derived_from` links.

The current overall mixture is `data/training-semantic-mixture-v4.json`. It adds 15,600 generated
symbolic-tool semantic tasks and brings the total to 55,655: 45,591 tool-required, 6,921 no-tool,
and 3,143 tools-available-but-unnecessary. Thirteen balanced symbolic families cover exact linear
and quadratic solving, polynomial expansion and factorization, rational simplification, two-variable
linear systems, differentiation, definite integration, identity checking, symbolic-to-calculator
chains, record-to-symbolic chains, parallel symbolic fan-out/merge, and trivial manipulations where
the symbolic tool should not be called. The component contributes 3,600 dependency-depth-two tasks.

`cid build-symbolic-training` deterministically materializes the timing-free teacher tasks and causal
jobs. The final self-distillation contains 34,800 causal teacher stages and 15,600/15,600 plans pass
`review-distillation`. Every serialized symbolic/calculator call was independently re-executed before
release (18,000 calls, zero mismatches). Two timing/slot schedules per semantic task compile to
31,200 `TrajectoryExample` records and 121,748 adjacent supervised transitions. Exact hashes and
compiler parameters are pinned by `data/symbolic-teacher-v1.reference-manifest.json`.

The v5 semantic mixture is `data/training-semantic-mixture-v5.json`. It adds 10,000
speculative local-correction semantic tasks, bringing the total to 65,655. Each correction task
starts with a plausible but explicitly uncertain wrong hypothesis, exposes contradictory
authoritative evidence, then exposes an independent confirmation. The correction stage increases
noise only for the hypothesis and its dependent answer so training emits local `REOPEN` targets;
unrelated context/scope cells are copied unchanged. Confirmation lowers those cells' noise and emits
`STABILIZE` targets. Two timing/slot schedules per semantic task yield 20,000 compiled trajectories
and 120,021 adjacent supervised transitions. All 30,000 causal teacher stages pass validation with
zero rejects, missing responses, or soft warnings, and all 10,000 plans pass both general review and
the local-correction audit. Exact hashes are pinned by
`data/correction-teacher-v1.reference-manifest.json`.

The latest semantic mixture, published as `manifests/training-semantic-mixture-v14.json` on Hugging
Face, contains **192,297 semantic tasks** across 19 components. Relative to v13 it adds 51,500
independent train-only natural public tasks from Natural Questions Open, OASST1, MultiDoc2Dial, and
QASPER. Their original user-facing prompts are preserved rather than wrapped in CID-specific
causality instructions. The existing 4,000-task compositional OOD probe remains excluded from
training.

Schedule variants and trajectory length are first balanced at semantic-task granularity; explicit
component `training_weight` is then applied. v14 assigns about **49.0%** of effective semantic loss
mass to natural source/augmentation supervision and **39.0%** to natural tool interaction while
retaining the existing mechanism, symbolic, correction, long-horizon, compositional, and restraint
curricula. These are weights, not claims of independent source examples. The corresponding Hugging
Face trajectory specification is `manifests/training-trajectory-mixture-v14.json`:
**421,798 trajectories and 3,011,462 training transitions**, with maximum TCT capacity 128.

The four v14 natural sources contribute 51,500 new independent semantic tasks; together with the
16,102 existing public-source tasks, the release contains **67,602 public-source semantic tasks**
before counting behavioral augmentations such as tool restraint or grounded response rewriting.
OASST1 is the main long-form no-tool addition, while MultiDoc2Dial and QASPER add naturally worded
document-grounded responses. Natural Questions Open primarily broadens the distribution of real
open-domain query wording.

### Compositional long-tail curriculum

The long-tail curriculum extends no-tool reasoning beyond the original eight-slot training
distribution. The CID adapter supports up to 128 physical thought slots, so this curriculum samples
five capacity buckets: **8, 16, 32, 64, and 128**. The 128-slot bucket is not padding-only: its tasks
require roughly 76--120 simultaneously represented cognitive objects, including multi-branch DAGs,
competing candidates, interventions, constraint meshes, and local hypothesis repair. Smaller buckets
remain the majority so ordinary reasoning does not become unnecessarily expensive.

`cid build-compositional-training` deterministically generates 20,000 training tasks and a separate
4,000-task generalization probe. The training capacity mixture is 6,000/5,000/4,000/3,000/2,000 for
8/16/32/64/128 slots respectively. Ten balanced families cover boolean DAG composition, blocked
reachability, spatial and causal interventions, ordering meshes, open-world quantifier DAGs,
candidate elimination, numeric DAGs, iterative conditional policies, and internal hypothesis
repair. Every final answer is independently recomputed from a compact machine-readable logic spec;
the normal teacher-plan review remains enabled, including duplicate rejection, semantic-text limits,
anchors, and typed cognitive links.

No-tool plans may contain `refine:0`, `refine:1`, ... semantic frames. These frames supervise compact
TCT state refinement rather than hidden chain-of-thought text, and they are forbidden on plans with
external evidence. The generalization probe is excluded from training and uses a strictly disjoint
domain vocabulary plus a heavier depth/capacity tail and denser long-range dependencies. Surface
rephrasings are measured as a generalization axis rather than claimed as a strict lexical holdout.

The trainer treats 128 as a **maximum capacity**, not a mandatory tensor width for every example.
Each trajectory is tensorized at its actual slot footprint (with an eight-slot minimum), and mixed
micro-batches pad only to the largest footprint in that batch. This keeps ordinary 8/16-slot examples
close to their previous memory cost while preserving a real 128-slot ability ceiling.

```bash
cid build-compositional-training \
  --output-dir data/generated \
  --variants-per-task 2 \
  --probe-variants-per-task 1
```

The resulting component is included in `data/training-semantic-mixture-v10.json` and
`data/training-trajectory-mixture-v10.json`; the 4,000-task probe is explicitly excluded from both.

### Long-horizon tool curriculum

`cid build-long-horizon-training` adds 12,000 tool-required semantic tasks whose dependency depth is
always 4--6. Six balanced families cover strict serial cross-source lookup, serial lookup followed by
exact calculation, alias/entity/class/policy/rate resolution, stale-cache correction through current
authority, two-branch fork/join, and a three-branch barrier before normalization. Every downstream
need is activated only after the evidence that licenses its arguments has arrived.

All 12,000 deterministic plans pass the standard semantic review and an independent exact verifier;
no future-evidence leakage is accepted. Two randomized timing/slot schedules per task yield 24,000
trajectories and 447,626 adjacent supervised transitions, with a maximum trajectory length of 33
steps. The component contributes 12,000 depth-4+ tool tasks, including 3,332 at depth 6, lifting the
overall v10 depth-4+ count to 40,330.

```bash
cid build-long-horizon-training
```

### v11 surface diversity and deep tool restraint

Two v10 tool-reasoning components had high semantic quality but overly regular task wording: the
12,000-task composed slice collapsed to eight normalized prompt signatures, and the 12,000-task
long-horizon slice to six. v11 rebuilds both from their accepted v1 tasks/plans while preserving the
core prompt, source descriptors, evidence contracts, reference answers, and TCT semantic frames.
Only a deterministic short wrapper and the task ID change. `composed-tool-reasoning-v2` yields 4,672
normalized signatures and `long-horizon-tool-reasoning-v2` yields 4,062; the largest normalized
prompt group in either component is 11. All 24,000 replacement plans pass the normal semantic review.

The new `deep-tool-restraint-v1` component samples 4,000 accepted compositional long-tail tasks with
dependency depth at least eight. It selects exactly 1,000 tasks from each 16/32/64/128-slot bucket,
exposes an irrelevant read-only `record_lookup` interface, but provides no external evidence and
preserves plans with zero tool needs. These hard negatives train tool restraint during genuinely deep
internal reasoning instead of only on short self-contained questions. They increase the v11
`tools_available_unnecessary` fraction to 10.176%; 2,611 of the added examples have dependency depth
at least 16.

```bash
cid build-surface-diverse-training --component composed
cid build-surface-diverse-training --component long-horizon
cid build-deep-tool-restraint-training
```

The three release manifests are `data/composed-teacher-v2.reference-manifest.json`,
`data/long-horizon-teacher-v2.reference-manifest.json`, and
`data/deep-tool-restraint-v1.reference-manifest.json`. v11 references the v2 replacements instead of
the corresponding v1 components, so surface augmentation does not double-count the same semantic
supervision.

The historical v5 six-component training input is pinned separately by
`data/training-trajectory-mixture-v5.json`. It preserves every reviewed schedule variant from
`public-base`, `public-interaction`, `mechanism`, `computational`, `symbolic`, and
`local-correction`: **179,608 trajectories and 1,027,548 adjacent supervised transitions** in total.
The materializer verifies every component SHA-256 and example count, rejects duplicate
`example_id`s across components, and concatenates the original JSONL bytes deterministically. Epoch
shuffling still happens at the rollout-window layer in the trainer, so file concatenation order is
not a learned curriculum unless `--no-shuffle` is explicitly requested.

```bash
cid materialize-trajectory-mixture \
  --spec data/training-trajectory-mixture-v5.json \
  --output data/generated/training-trajectories-v5.jsonl \
  --manifest-output data/generated/training-trajectories-v5.manifest.json

cid train \
  --data data/generated/training-trajectories-v5.jsonl \
  --output-dir runs/cid-stage-a
```

The verified materialized dataset identity is pinned by
`data/training-trajectories-v5.reference-manifest.json`: 179,608 examples, 1,027,548 transitions,
2,391,099,272 bytes, SHA-256
`d771d5ddcf94c1b8b7ae9a1b7df38944fc3c5974d34867ec4c0ae392b7c9120b`.

`inspect_dataset` is streaming, so Stage B can verify the multi-gigabyte dataset identity and TCT
capacity without first duplicating the whole input in memory. The actual trainer still loads the
trajectory examples for rollout-window construction after that verification step.

`TeacherEvidence.requires_need` distinguishes explicit model-launched work from later arrivals on
an existing binding. A causal action contract that must stay live carries
`freshness_hint="always"`; the teacher-output validator enforces that hint without exposing future
evidence values.

### Causal teacher production

Production teacher labeling is a resumable sequence of causal waves. Start from the generated job
file with an empty state file:

```bash
cid teacher-wave-export \
  --jobs data/generated/public-interaction-teacher-causal-v1.train.jsonl \
  --state data/generated/public-interaction-teacher-wave.state.jsonl \
  --output data/generated/wave-000.requests.jsonl
```

Each request contains exactly one task stage. The worker returns JSONL records of the form:

```json
{"request_id":"tw-...","output":{"display":"...","cells":[...],"needs":[...]}}
```

The output must preserve all previously introduced logical cell IDs. For every currently available
evidence contract it emits exactly one new need attached to an `information_need` cell. Source names
and arguments come from the gold contract and are not re-generated by the teacher. Terminal stages
emit no new needs and must contain a conclusion cell.

Import a completed worker batch with:

```bash
cid teacher-wave-import \
  --jobs data/generated/public-interaction-teacher-causal-v1.train.jsonl \
  --requests data/generated/wave-000.requests.jsonl \
  --responses data/generated/wave-000.responses.jsonl \
  --state data/generated/public-interaction-teacher-wave.state.jsonl \
  --rejects-output data/generated/wave-000.rejects.jsonl
```

Valid responses are persisted even when other records in the same batch are malformed. Rejected
records remain incomplete, so the next export emits the same stable request IDs again for retry.
`teacher-wave-import` without `--rejects-output` is strict and aborts on the first invalid record.
Progress is inspectable at any point:

```bash
cid teacher-wave-status \
  --jobs data/generated/public-interaction-teacher-causal-v1.train.jsonl \
  --state data/generated/public-interaction-teacher-wave.state.jsonl
```

For an interactive strong teacher operating through local-shell-mcp, the equivalent agent adapter
avoids materializing a large request/response JSONL pair on every wave. It checks out a small batch
as separate pretty-printed request files and writes the stable protocol instructions only once:

```bash
cid teacher-agent-checkout \
  --jobs data/generated/public-interaction-teacher-causal-v1.train.jsonl \
  --state data/generated/public-interaction-teacher-wave.state.jsonl \
  --workspace .cid/teacher-agent \
  --max-requests 8
```

The agent reads `.cid/teacher-agent/INSTRUCTIONS.md` and `current/requests/*.json`, then writes the
stage output directly to `current/responses/<request_id>.json`. Request files contain structured
`task`, `previous_state`, `arrived_evidence`, and `available_evidence_contracts` fields instead of
repeating the long worker prompt for every item. They still exclude the reference answer and all
future evidence values.

The interactive adapter treats TCT structure as supervised data, not optional metadata. Its
`semantic_text` is a compact fact/state representation (target <=144 characters, hard limit 192),
and full source sentences or paragraphs are rejected when they appear to be copied into percept
cells. Evidence-bearing percepts carry grounding `anchors` plus an `observes` link to the source;
information-need cells carry `requests` links to their contracted source; terminal conclusions link
back to supporting percept cells with `derived_from`. This keeps TCT semantics separate from document
storage while preserving the typed cognitive graph required by the model heads.

Commit whatever has been completed:

```bash
cid teacher-agent-commit --workspace .cid/teacher-agent
```

Valid records are persisted immediately while malformed or missing files remain in the current
batch. Validation errors are written to `current/errors/`, so an agent can correct only those
responses and commit again. Checkout is interruption-safe: before the current batch is fully
committed it simply resumes the same files; afterward the next checkout advances those tasks to
their next causally visible stages. The state and final `TeacherPlan` format are identical to the
normal teacher-wave pipeline, so this adapter does not create a second training-data ABI.

Repeat export/worker/import until `complete_tasks == jobs`, then assemble normal `TeacherPlan`
records and run the deterministic review gate:

```bash
cid teacher-wave-finalize \
  --tasks data/generated/public-interaction-teacher-tasks-v1.train.jsonl \
  --jobs data/generated/public-interaction-teacher-causal-v1.train.jsonl \
  --state data/generated/public-interaction-teacher-wave.state.jsonl \
  --output data/generated/public-interaction-teacher-plans-v1.train.jsonl

cid review-distillation \
  --tasks data/generated/public-interaction-teacher-tasks-v1.train.jsonl \
  --plans data/generated/public-interaction-teacher-plans-v1.train.jsonl \
  --accepted-plans-output data/generated/public-interaction-teacher-plans-v1.accepted.jsonl \
  --report-output data/generated/public-interaction-teacher-review-v1.jsonl
```

The review gate also checks public reference answers for multi-hop QA, multiple-choice tasks, and
GSM8K-style numeric tasks. MATH and executable code are intentionally left to task-specific
equivalence/execution validators rather than unsafe string equality.

After semantic plans pass review, runtime counterfactuals require no additional teacher calls.
Multiple physical-slot and asynchronous-latency variants can be compiled directly:

```bash
cid compile-distillation \
  --tasks data/generated/public-interaction-teacher-tasks-v1.train.jsonl \
  --plans data/generated/public-interaction-teacher-plans-v1.accepted.jsonl \
  --output data/generated/public-interaction-trajectories-v1.jsonl \
  --thought-capacity 8 \
  --min-delay-steps 1 \
  --max-delay-steps 6 \
  --variants-per-task 4 \
  --seed 0
```

Needs activated by the same evidence phase launch at the same runtime step. Their I/O remains
simultaneously in flight; arrivals are then ordered consistently with the teacher's semantic frame
sequence. This preserves causal semantic supervision while training actual asynchronous overlap.

After compilation, `dataset-manifest` records the exact JSONL SHA-256 together with example and
transition counts, scenario/distillation tags, source names, maximum trajectory depth, and required
TCT capacity. This gives every training run a deterministic dataset identity instead of relying on
mutable filenames.

## Stage 2 — joint T/Y refinement

Unfreeze selected backbone blocks and optimize coupled thought/display denoising. The main training
families are static copying, delayed retrieval, dynamic state tracking, streaming evidence, and
conflicting sources. Distillation can provide high-quality plans and revision targets, but the
event schedule must be synthesized independently so the student cannot reduce TCT to a textual
chain-of-thought imitation task.

## Stage 3 — end-to-end model

Only after the runtime and adapter path show measurable intent lead time and useful post-arrival
revision should we spend compute on a dedicated small model. A 4B-class model can then be trained
with the same state/data contract rather than changing the system architecture during scaling.

## Dataset record

`cid.data.TrajectoryExample` stores:

- user/task input and protected facts;
- source descriptors;
- target final display;
- per-step `ThoughtTarget` snapshots with stable cell ID, physical slot, semantic transport text,
  roles, uncertainty, editability/noise, and lifecycle;
- optional per-step `DisplayTarget` text for pre-/post-arrival revision supervision;
- external events with arrival step/time and source version;
- binding targets and affected regions;
- a closed-world `grounding_catalog` of canonical anchors and aliases;
- per-step `grounding_targets` containing typed anchors and cognitive links for each supervised
  cell.

Binding targets use typed `ObjectRef` values. Cell references carry stable `cell_id` values and
display targets use explicit `DISPLAY_SPAN` references, so neither depends on physical TCT layout.
Binding targets also record target arguments, executable timing, confidence, and freshness demand,
which directly supervise need/source/argument/refresh heads. Optional per-argument availability
steps let a trajectory supervise partially bound calls before the whole source invocation becomes
executable.

`ThoughtTarget.semantic_text` is a dataset transport format rather than runtime chain-of-thought.
The tensorizer embeds it into the latent TCT target; the deployed model only carries continuous
cell semantics, typed anchors/links, and runtime-visible needs.

The schema deliberately records *when* evidence becomes available. Flattening events into the
initial prompt destroys the central CID training signal.

# Continuous Interaction Diffusion

Reference implementation of **Continuous Interaction Diffusion (CID)**, a model--runtime
co-design for diffusion-native reasoning with persistent asynchronous perception.

CID keeps three coupled channels:

- **F / facts** — externally controlled values that the model can read but cannot overwrite;
- **T / thought** — a revisable Typed Cognitive Tensor (TCT);
- **Y / display** — a revisable token canvas that converges to user-visible output.

Tool use is represented as persistent **perceptual bindings**, not one-shot serialized calls.
The runtime can keep denoising while read-only I/O is in flight, reuse static observations
without repeating I/O, refresh dynamic sources, and re-project the same observation as
cognition changes. Sources may opt into progressive selectors so a binding can begin external work
before every required argument converges. Version-aware sources use lightweight freshness probes,
streamable sources feed incremental observations into successive denoising steps, and percept
cell/display targets become query-specific neural routing masks rather than metadata only.

TCT uses a fixed physical capacity for efficient tensor execution but dynamic logical occupancy.
Cognitive objects receive stable `cell_id` values, so they can be allocated, retired, reclaimed,
split, merged, or physically compacted without binding tool/fact links to tensor position.
Allocation is predicted separately from lifecycle: empty slots may allocate new `ACTIVE` cells,
while existing cells transition among `ACTIVE / WAITING / STABLE / RETIRED` under runtime gates.
Unresolved bindings hold `WAITING`, explicit revision signals reopen `STABLE`, and only runtime
reclamation can turn `RETIRED` storage back into `EMPTY`.

Retired cells are reclaimed under slot pressure only after a grace period and safety checks.
Bindings and strong live cognitive dependencies pin neural state. Before reclamation, lightweight
tombstones preserve identity, typed anchors/links, lifecycle timing, and provenance without keeping
the full semantic vector; weak historical links remain resolvable through this archive.

Grounding is typed rather than encoded as ad-hoc strings. Anchors attach canonical symbolic
objects to cognitive cells, `ObjectRef` distinguishes cells/facts/bindings/sources/display spans,
and `CognitiveLink` records typed relations between them. The reference neural core predicts
multiple anchor/link slots per cognitive cell, while a closed-world oracle grounder provides a
deterministic training and runtime path before open-world entity resolution is introduced.

> This repository is the implementation workspace. The current code establishes the model
> contract, asynchronous runtime, reference neural core, training data schema, and evaluation
> instrumentation. It does **not** yet claim the empirical performance proposed by the paper.

## Design goals

1. Make the paper's state machine executable rather than simulate CID with prompt strings.
2. Keep model-specific tensor geometry behind a stable CID contract.
3. Enforce the fact-channel write boundary in the runtime.
4. Treat repeated information needs as persistent perception while deduplicating external I/O.
5. Make event arrival, source freshness, cache state, and local reopening first-class training data.
6. Support an adapter path from existing masked-diffusion language models before considering
   from-scratch pretraining.

## Repository layout

```text
src/cid/state.py            Three-channel state and immutable fact snapshots
src/cid/grounding.py        Typed anchors, object references, links, oracle grounding
src/cid/lifecycle.py        Event-aware cognitive lifecycle transition controller
src/cid/runtime/archive.py  Lightweight tombstones for reclaimed cognitive cells
src/cid/contracts.py        Model/runtime information-need and percept contracts
src/cid/runtime/            Async scheduler, bindings, source registry, traces
src/cid/model/torch_core.py Small trainable PyTorch reference core
src/cid/model/illada.py     Real iLLaDA masked-diffusion backbone adapter
src/cid/model/components.py Shared CID fusion and prediction heads
src/cid/model/materialize.py Neural-output to runtime-contract materializer
src/cid/model/diffusion.py  T/Y corruption and display reveal schedule
src/cid/model/policy.py     iLLaDA context tensorizer and runtime neural policy
src/cid/model/training.py   Distilled trajectory to training tensors/targets
src/cid/synthetic.py        Reproducible five-family mechanism trajectory factory
src/cid/data.py             Trajectory JSONL schema and validation
src/cid/teacher_agent.py    Resumable interactive/LSM teacher workspace adapter
src/cid/metrics.py          CID-specific interaction metrics
docs/architecture.md        Concrete v0 architecture decisions
docs/training.md            Staged training plan
docs/evaluation.md          Runtime/task evaluation metric contract
docs/public-datasets.md     Pinned public-task sources, licenses, and quotas
examples/                   Small executable runtime examples
tests/                      Runtime invariants and concurrency tests
```

## Install

Runtime and data tooling have no mandatory third-party dependencies:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

For the neural reference core:

```bash
python -m pip install -e '.[train]'
```

For reproducible public task-pool construction:

```bash
python -m pip install -e '.[data]'
```

## Diffusion backbone adapters

CID supports three production backbone sizes: dense `GSAI-ML/iLLaDA-8B-Base`, sparse
`inclusionAI/LLaDA-MoE-7B-A1B-Base`, and the compact
`LiquidAI/LFM2.5-Encoder-350M-Diffusion` used by CID-v1-0.4B. All three keep the pretrained
masked-token embedding and LM head, run `[TCT | prompt | display]` through their native
bidirectional sequence model, and attach the same CID external-perception fusion and prediction
heads. The LFM2 path preserves its original full-attention/short-convolution stack rather than
converting its weights into an iLLaDA-shaped model. The MoE path keeps its 64-expert/Top-8 routing
and adds the upstream load-balancing auxiliary loss only when the backbone is unfrozen in Stage B.
The immutable prompt remains token-level conditioning rather than being flattened into protected
Facts. Empty TCT slots are masked as attention keys so they can query context for allocation without
contaminating the current display.

The same model loader and CID training/runtime ABI are used across CPU, NVIDIA CUDA, and Ascend NPU
backends; backend support is not maintained as separate model forks:

| CID variant | Backbone | CPU / Gloo | NVIDIA CUDA / NCCL | Ascend NPU / HCCL |
| --- | --- | --- | --- | --- |
| CID-v1-8B | iLLaDA-8B-Base | supported | supported | supported |
| CID-v1-7B-A1B | LLaDA-MoE-7B-A1B-Base | supported | supported | supported |
| CID-v1-0.4B | LFM2.5-Encoder-350M-Diffusion | supported | supported | supported |

Stage A uses the same DDP path on CUDA and NPU. Stage B uses FSDP `FULL_SHARD` for multi-rank
accelerator training; the compact 0.4B path additionally supports one-NPU BF16 full-parameter
training without FSDP. See `docs/training.md` for device-specific launch and checkpoint details.

```python
import torch

from cid.model import ILLaDACIDAdapter

model = ILLaDACIDAdapter.from_pretrained(
    freeze_backbone=True,
    dtype=torch.bfloat16,
)
```

The loader pins the official checkpoint revision used by this repository and enables the model's
required Hugging Face remote code. Use `load_cid_adapter_from_pretrained()` when the backbone may
be any supported family; it dispatches LFM2 to `AutoModelForMaskedLM` and LLaDA-family checkpoints
to their existing causal-LM wrappers. The 8B checkpoint is about 16.5 GB. For an interface smoke
test using the official iLLaDA implementation without downloading the 8B weights:

```bash
python examples/illada_tiny_smoke.py
```

Special tokens are backbone-specific: iLLaDA uses mask id `5`, LLaDA-MoE uses mask id `156895`
and EOS id `156892`, and LFM2.5 diffusion uses mask id `16` and EOS id `7`. The adapter carries
these IDs into training, runtime, checkpoints, and benchmarks so artifacts from different backbones
cannot be mixed silently. The LFM2.5 checkpoint remains subject to its upstream LFM Open License
v1.0; this repository's Apache-2.0 license does not replace the model-weight license.
`CIDDiffusionScheduler` supplies
masked-display corruption, continuous TCT corruption, confidence-ranked iterative reveal, and
bounded visible-token revision. Training
mixes ordinary mask corruption with visible wrong-token replacement so the display head learns to
correct stale text after new evidence arrives instead of treating every revealed token as final.

The neural path is executable end to end: `ILLaDAContextTensorizer` converts a runtime
`ModelContext`, `ILLaDANeuralPolicy` performs a denoising step, and `CIDMaterializer` converts
allocation/lifecycle/need/source/argument/grounding/revision/refresh predictions back into the
typed runtime contract. A separate learned convergence logit estimates whether cognition has reached
equilibrium under the information currently available. If required external work is still pending,
the runtime enters quiescence without consuming additional model steps and resumes when an event
arrives. A fully denoised display can terminate only after required bindings are resolved and the
final freshness barrier is satisfied; filling the last MASK alone is insufficient.
Closed-world candidate retrieval is used for the first training stage.

Training data uses typed per-step `ThoughtTarget` and `DisplayTarget` records rather than free-form
dictionaries. `ILLaDATrajectoryTensorizer` constructs adjacent-step supervision and the CID loss
performs permutation-invariant assignment for multi-anchor and multi-link targets. The test suite
includes a complete tiny-backbone optimizer step with the pretrained backbone frozen.

The raw semantic-task pool is built separately from CID trajectories. Public sources, exact
revisions, licenses, upstream splits, and sampling quotas are registered in
`configs/public-datasets.json`, `configs/public-interaction-datasets.json`, and
`configs/natural-public-datasets-v1.json`, with the complete registry documented in
`docs/public-datasets.md`. Build the pinned 10,000-task general pool with:

```bash
cid build-public-task-pool
```

Build the separate 10,000-task interaction-heavy pool from 2WikiMultiHopQA and MuSiQue with:

```bash
cid build-public-task-pool \
  --registry configs/public-interaction-datasets.json \
  --output data/generated/public-interaction-task-pool-v1.jsonl \
  --manifest-output data/generated/public-interaction-task-pool-v1.manifest.json
```

Convert a pool into teacher-ready CID tasks and causal evidence-exposure jobs with:

```bash
cid prepare-public-distillation
```

Released CID datasets are published on Hugging Face at `fwerkor/CID-Dataset`; the GitHub
repository intentionally does not track release data or release manifests. The local `data/` tree is
a gitignored build/training workspace. **v12 through v17 are preserved as complete releases** so
training artifacts do not skip a version.

- **v12:** 132,122 semantic tasks, 305,948 trajectories, 2,303,169 training transitions;
  materialized SHA-256 `2f2da01e3963b4ac758e023dcb8659afc2d81999c88fd9df361ec058112f3478`.
- **v13:** 140,797 semantic tasks, 323,298 trajectories, 2,485,228 training transitions;
  materialized SHA-256 `fcda158c66f911b9521e37ffcbdba038710bde607a4762f498b0e70bd99f5de2`.
- **v14:** 192,297 semantic tasks, 421,798 trajectories, 3,011,462 training transitions. It is the
  last release using the pre-v3 need-routing data ABI.
- **v15:** keeps the exact v14 semantic tasks, schedules, trajectories, prompts, answers, evidence,
  and transition count while rematerializing them for neural contract v3. Every binding has an
  explicit stable owner; affected-cell supervision is derived conservatively from matching
  observations, and protected-result promotion is explicit source-owned metadata.
- **v16:** 192,729 semantic tasks, 422,230 trajectories, 2,724,556 adjacent transitions, and
  3,146,786 total training transitions. It introduced the neural-contract-v4 materialization, 432
  high-control curriculum trajectories, and stable terminal tails. Canonical SHA-256:
  `76980590fb21d75d3dfe466e8b39716362bc249bad73416bcb690f7a59155e6b`.
- **v17 (current):** keeps exactly the same 192,729 semantic tasks, 422,230 trajectories, and
  3,146,786 training transitions while tightening Display contract v2. It removes 31,260 previously
  under-detected process-status target occurrences and derives 40,158 grounded partial-answer targets
  only from already-supported multi-hop QA facts. Canonical SHA-256:
  `07662203cc23f5ee628623090ad029740e51b3d6efb13466a6dcad23a2a3b143`.

The current source tree defines **neural contract v4** for new training runs. Display state is now a
continuously revisable answer draft: unresolved answer content is represented by the model MASK
state, EOS may move across the fixed physical canvas, and semantic equilibrium does not terminate a
trajectory until the materialized display is resolved and stable for a subsequent step. Generic
process-status supervision such as `Reasoning.` or `Retrieving evidence.` is rejected. Consequently,
v3 checkpoints are intentionally incompatible with v4. Dataset release **v16** introduced that
rematerialization and the high-weight v4 curriculum. **v17** tightens the same semantic corpus without
adding or dropping tasks: residual process narration missed by the original detector is replaced by
`<|cid_unknown|>`, and multi-hop QA states expose a conservative `Known: ... Answer:
<|cid_unknown|>` draft only when the corresponding `support-*` facts are already present in that
step's TCT. Internal dependency curricula do not receive this derived user-visible partial.

v14 introduced **51,500 independent train-only natural public tasks**: 30,000 Natural Questions Open
queries, 15,000 MultiDoc2Dial document-grounded dialogue turns, 4,500 high-quality human OASST1
instruction/response examples, and 2,000 QASPER paper-grounded questions. Together with the existing
public components this raises independent public-source supervision to 67,602 semantic tasks. The
new tasks preserve the original user-facing wording; the builder does not prepend CID-specific
"wait for evidence" or dependency-order scaffolding.

The trainer first equalizes schedule/trajectory loss mass at semantic-task granularity and then
applies explicit component `training_weight`. In v14, natural source/augmentation supervision receives
about 49.0% of effective semantic loss mass and natural tool interaction about 39.0%. Existing
mechanism, symbolic, correction, long-horizon, compositional, and restraint curricula remain in the
mixture rather than being replaced by the new natural data.

Download the current pinned release into the local gitignored workspace before training, for
example:

```bash
hf download fwerkor/CID-Dataset \
  release/training-trajectories.jsonl \
  release/materialized-manifest.json \
  evaluation/validation-v4/validation-512.jsonl \
  evaluation/validation-v4/validation-512.manifest.json \
  --repo-type dataset --local-dir .cid/hf-v17
```

v15 is reproducible from the verified v14 materialization with `cid migrate-dataset-contract-v3`.
This migration does not relabel task answers or regenerate semantic teacher plans. The separate
512-example validation set keeps 416 held-out compositional/OOD reasoning trajectories and adds 96
held-out synthetic tool interactions (18.75%) so per-epoch validation exercises source selection,
binding, observation assimilation, and the v3 affected-region ABI instead of measuring only no-tool
reasoning. It is a training/runtime validation set, not a replacement for the formal benchmark.

The materializer verifies every component SHA/count and global `example_id` uniqueness before
writing the combined file. Training shuffles rollout windows per epoch unless `--no-shuffle` is
requested.

Build the self-identity component reproducibly with:

```bash
cid build-self-identity-training
```

Build the deterministic computational teacher jobs with:

```bash
cid build-computational-training
```

The pinned build and self-distillation hashes are published with the component in the Hugging Face dataset repository.

Build the deterministic symbolic teacher jobs with:

```bash
cid build-symbolic-training
```

The symbolic component contains 34,800 causal teacher stages; all 15,600 semantic plans pass the
quality gate and compile to 31,200 independently randomized runtime trajectories. Its exact build, review, tool-replay, and compilation hashes are published with the component in the Hugging Face dataset repository.

Build the deterministic speculative local-correction teacher jobs with:

```bash
cid build-correction-training
```

The released component contains 30,000 validated causal stages and 20,000 compiled trajectories;
its exact generation, review, correction-audit, and compilation hashes are published with the component in the Hugging Face dataset repository.

Build the current data-quality additions reproducibly with:

```bash
cid build-natural-interaction-training
cid build-natural-public-training --source nq-open
cid build-natural-public-training --source oasst1
cid build-natural-public-training --source multidoc2dial
cid build-natural-public-training --source qasper
cid build-compositional-training
cid build-deep-tool-restraint-training
cid build-surface-diverse-training --component logic-v4
cid build-surface-diverse-training --component compositional-v4
cid build-surface-diverse-training --component deep-restraint-v4
```

The natural-interaction builder preserves accepted public evidence contracts while adding grounded
long-form display targets and deterministic tool-schema variation. The v4 surface builders preserve
typed plan semantics, anchors, links, lifecycle, and causal needs while paraphrasing only selected
high-frequency TCT state templates. The deep-restraint builder reuses accepted compositional plans,
exposes an irrelevant read-only source, and keeps every selected task free of evidence arrivals and
tool needs.

Generated task data lives under the gitignored local `data/` workspace and is never committed to
GitHub. Every public record retains exact upstream provenance. The original public-pool builders
assign content-keyed CID splits before toolization; the v14 natural-public registry is explicitly
train-only and is deduplicated before timing augmentation.

For deterministic mechanism-training data before teacher distillation is available:

```bash
cid generate-synthetic \
  --output data/synthetic.jsonl \
  --count-per-family 1000 \
  --seed 0 \
  --thought-capacity 8
```

The generator covers static copying, delayed retrieval, dynamic state tracking, streaming evidence,
and competing sources. Physical TCT slot placement is randomized independently of logical cell ID.

For high-quality distillation, first strip runtime timing and physical-slot choices out of the
teacher input:

```bash
cid prepare-distillation \
  --data data/synthetic.jsonl \
  --tasks-output data/teacher-tasks.jsonl \
  --requests-output data/teacher-requests.jsonl
```

Each request asks the teacher for concise typed cognitive-state summaries, needs, anchors, and
links. The teacher is explicitly forbidden from choosing diffusion steps, evidence arrival times,
or physical TCT slots. Save one returned plan JSON object per line, then compile it back into the
normal training ABI:

```bash
cid review-distillation \
  --tasks data/teacher-tasks.jsonl \
  --plans data/teacher-plans.jsonl \
  --accepted-plans-output data/teacher-plans.accepted.jsonl \
  --report-output data/teacher-review.jsonl
```

The review rejects future-evidence leakage, required-argument mismatches, missing final conclusion
state, legacy process-status Display targets, and exact semantic duplicates before the plans reach
training. Intermediate Display frames describe the current user-visible answer draft and use
`<|cid_unknown|>` wherever answer content is not yet resolved; the trainer maps that marker to the
backbone MASK token rather than tokenizing it as literal text.

```bash
cid compile-distillation \
  --tasks data/teacher-tasks.jsonl \
  --plans data/teacher-plans.accepted.jsonl \
  --output data/distilled.jsonl \
  --thought-capacity 8 \
  --min-delay-steps 1 \
  --max-delay-steps 4 \
  --seed 0
```

The compiler independently randomizes event latency and physical slot placement, inserts
`WAITING` states while external evidence is outstanding, forces arrival-time assimilation through
`ACTIVE`, and only then permits the teacher's stable post-evidence state. Teacher plan parsing is
strict: unknown fields and any attempt to control timing or slot placement are rejected.

Pin the exact JSONL used for a run with a deterministic manifest:

```bash
cid dataset-manifest \
  --data data/distilled.jsonl \
  --output data/distilled.manifest.json
```

The manifest records the raw-file SHA-256, example/transition counts, source set, scenario tags,
maximum trajectory depth, and minimum TCT capacity required by the data.

### Stage A training

Stage A freezes the pretrained iLLaDA backbone and trains the CID projections, external-perception
fusion, and prediction heads. A single A6000 can run the launcher directly:

```bash
cid train \
  --data data/synthetic.jsonl \
  --validation-data data/validation.jsonl \
  --output-dir runs/stage-a \
  --thought-capacity 128 \
  --display-canvas-tokens 64 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --dtype bf16
```

With six GPUs, use `torchrun`; the command detects the distributed environment automatically:

```bash
torchrun --standalone --nproc-per-node=6 -m cid.cli train \
  --data data/synthetic.jsonl \
  --output-dir runs/stage-a-6gpu \
  --thought-capacity 128 \
  --display-canvas-tokens 64 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --dtype bf16
```

Each rank loads the pinned 8B backbone serially before moving it to its GPU, avoiding six concurrent
CPU copies during startup. DDP synchronizes only trainable CID state; frozen backbone parameters and
buffers are excluded from initialization sync. Consecutive trajectory transitions are grouped into full contiguous rollout windows, bucketed by
window length, padded to equal rank counts, and sharded before training. Rollout state is detached
after every transition, so preserving a long window does not retain a long BPTT graph.

Training uses scheduled sampling instead of remaining permanently teacher-forced. By default the
first epoch uses teacher inputs, the next two epochs linearly ramp the probability of feeding the
model's own detached state into the following transition, and later epochs use self-rollout. A
`--rollout-horizon` value of 1 disables multi-step self-rollout; values greater than 1 preserve the
full contiguous trajectory rather than injecting periodic teacher resets. The carried rollout state
includes TCT/display state plus runtime-materialized tool bindings, executable-argument state,
observations, diffusion-epoch position, and promoted facts. Teacher trajectories still define
next-step supervision and replayable external events, but an event can enter a self-rollout only
after the model has materialized a matching source+argument work item. Spurious or wrong bindings
therefore retain their runtime waiting/termination consequences. If a rollout terminates or becomes
quiescent before a later supervised transition, that transition still receives a teacher-input
correction loss without overwriting the blocked detached runtime state.

The trainer pads variable-length prompt and external-memory sequences inside each micro-batch.
CID v1 uses a fixed physical TCT width of 128 slots in both training and runtime inference. Dataset
schedule slots are canonicalized to a deterministic first-free layout inside that fixed field; unused
slots remain empty, and retired slots stay reserved within one supervised trajectory because
reclamation is runtime-owned rather than a learned transition. The v1 training commands therefore
require `--thought-capacity 128`. `--display-canvas-tokens` is the minimum display bucket (`64` by
default); training expands it only when needed through coarse buckets (64, 128, 256, 512, 1024,
then the configured maximum, 1536 by default). The realized text is terminated by EOS and positions
after EOS receive no token loss. This avoids paying for a 1536-token canvas on short examples while
keeping every released target representable without truncation. Within self-rollout a display bucket
may grow but never shrink. Gradient checkpointing is enabled by default for the native iLLaDA stack
and can be disabled with `--no-gradient-checkpointing`. With the six-GPU example above, the effective
transition batch is `2 × 8 × 6 = 96`; start at micro-batch 1 if the real trajectory lengths are
substantially larger.

Stage A keeps the frozen backbone in the requested BF16/FP16 storage precision but keeps trainable
CID modules and AdamW state in FP32; autocast supplies low-precision forward compute without losing
sub-ULP optimizer updates. Stage A checkpoints contain trainable CID parameters, optimizer state,
progress, RNG state, and the training-dataset SHA-256; they
do not duplicate the frozen backbone. At every completed epoch, the trainer writes a permanent
`stage-a-epoch-XXXX.pt` snapshot; `stage-a-latest.pt` and the corresponding step name are compatibility
symlinks to that epoch snapshot. Resume with `--resume <checkpoint>`. When held-out trajectories are
provided with `--validation-data` (or are present in `--data` with `metadata.split=validation`), the
trainer computes a deterministic teacher-forced validation loss after every epoch and appends it to
`validation_metrics.jsonl`. The fixed validation RNG seed makes the values comparable across epochs;
validation does not update parameters or trigger automatic early stopping. For inference,
`load_cid_adapter_checkpoint()` loads the CID-only state without constructing a trainer.

### Stage B full-parameter training

After Stage A has learned the CID runtime contract, `train-full` performs one joint full-parameter
continuation of iLLaDA and the CID modules. The production path is deliberately **AdamW-only** and
requires at least four GPU ranks for the 8B model. Six 48 GiB A6000s are preferred; four are the
supported minimum. Two-rank training is rejected rather than silently switching optimizer or CPU
offload semantics. With the current 8.25B iLLaDA checkpoint plus roughly 0.45B CID parameters, the
FP32 parameter/gradient/Adam state alone is about 32 GiB per rank at world size 4 and 22 GiB at
world size 6 before activations and FSDP all-gathers.

The default profile is quality-first:

- FSDP `FULL_SHARD`, FP32 master parameters, BF16 forward/reduction, gradient checkpointing enabled;
- AdamW with `weight_decay=0.01` on matrix/tensor weights and zero decay on one-dimensional
  norm/bias parameters;
- peak CID learning rate `1e-5`; the pretrained iLLaDA backbone uses a conservative `0.5` multiplier
  (`5e-6` by default);
- 3% linear warmup followed by cosine decay to 10% of the peak rate over the target Stage B epochs;
- full rollout from the first Stage B batch (`teacher_forcing_epochs=0`, `rollout_ramp_epochs=0`),
  because the curriculum has already been completed in Stage A;
- target effective transition batch 32. Gradient accumulation is resolved automatically from the
  world size: with micro-batch 1 this is 8 on four GPUs and 5 on six GPUs (effective batch 30);
- one **target total** epoch by default. On resume, `--epochs 1` means finish epoch 1 rather than add
  another epoch.

A completed Stage A CID checkpoint is required for a fresh Stage B launch:

```bash
WORLD_SIZE=6
torchrun --standalone --nproc-per-node=${WORLD_SIZE} -m cid.cli train-full \
  --data data/distilled.jsonl \
  --validation-data data/validation.jsonl \
  --output-dir runs/stage-b \
  --init-cid-checkpoint runs/stage-a/stage-a-epoch-0003.pt \
  --thought-capacity 128 \
  --max-display-tokens 1536 \
  --display-canvas-tokens 64 \
  --dtype bf16
```

The initial FP32 iLLaDA copy stays on host memory while the frozen BF16 text-embedding snapshot is
created. FSDP then moves and shards wrap units onto each GPU via `device_id`, avoiding a transient
full FP32 8B allocation on every A6000 before sharding is active. The frozen embedding snapshot is
used only for dataset transport targets and external-memory text; prompt/display IDs still use the
live trainable embedding through the FSDP forward.

Full-parameter checkpoints are directories written with `torch.distributed.checkpoint`; model and
optimizer states stay sharded and are never gathered as a full checkpoint on rank 0. The launcher
logs every 100 optimizer steps and writes a resumable periodic checkpoint every 2,500 steps at the
next clean gradient-accumulation boundary. Every completed epoch is additionally retained permanently
as `stage-b-epoch-XXXX` (or `.pt` on the single-NPU compact path); `stage-b-latest` and the epoch-end
step name are symlinks to that snapshot. Periodic cleanup never deletes epoch snapshots. Stage B uses
the same deterministic per-epoch validation objective and `validation_metrics.jsonl` as Stage A.
Resume requires the same dataset SHA-256 and resolved trainer configuration. Same-world-size resume
is always supported. New data-order-v4 checkpoints keep each rollout bucket in a world-size-independent
canonical order and may also resume mid-epoch with a different FSDP world size; legacy partial-epoch
checkpoints and the Ascend rank-local optimizer checkpoint format must resume with their original
world size:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} -m cid.cli train-full \
  --data data/distilled.jsonl \
  --output-dir runs/stage-b \
  --resume runs/stage-b/stage-b-step-00002500 \
  --dtype bf16
```

The defaults can be overridden for controlled ablations with `--learning-rate`,
`--backbone-lr-scale`, `--target-global-batch-size`, `--gradient-accumulation-steps`,
`--warmup-ratio`, and `--min-learning-rate-ratio`, but the release run should keep one frozen
configuration once it starts.

### Neural replay benchmark

`cid benchmark` runs a trained CID checkpoint through the same step-exact replay sources used by the
runtime evaluator and writes both per-case JSONL and an aggregate JSON summary. The benchmark uses
the same coarse display buckets as training, expanding from the checkpoint's minimum canvas up to
`max_display_tokens` when a target requires it. The default starts from an empty TCT and the
canonical unresolved Display state (`MASK`, `EOS`, then latent capacity), matching the neural
training representation of `<|cid_unknown|>` while leaving room for EOS to move as the answer grows;
`--seed-teacher-state` is available only as a diagnostic that isolates downstream interaction
behavior from initial cognitive allocation.

For a Stage A CID-only checkpoint:

```bash
cid benchmark \
  --data data/validation.jsonl \
  --checkpoint runs/stage-a/stage-a-step-00001000.pt \
  --checkpoint-kind stage-a \
  --output runs/eval/cases.jsonl \
  --summary-output runs/eval/summary.json \
  --dtype bf16
```

Stage B checkpoints remain sharded, so evaluation uses the original FSDP world size instead of
gathering the 8B model onto one rank:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} -m cid.cli benchmark \
  --data data/validation.jsonl \
  --checkpoint runs/stage-b/stage-b-step-00002500 \
  --checkpoint-kind stage-b \
  --output runs/eval-stage-b/cases.jsonl \
  --summary-output runs/eval-stage-b/summary.json \
  --dtype bf16
```

The summary reports convergence, exact display accuracy, observation coverage/staleness,
latent-to-executable delay, binding-to-observation delay, and observation-to-projection lag. For
Stage A, `--dtype bf16` means BF16 backbone/compute under autocast while CID-specific parameters stay
FP32, matching training; thresholded runtime probabilities are evaluated in FP32. Stage B benchmark
loading restores model shards only; it does not construct or load optimizer state.

The teacher-forced/free-rollout metrics written during training are per-transition head diagnostics.
Stage A also runs a bounded family-diverse subset through the actual replay runtime after each epoch,
recording end-to-end task results separately in `runtime_validation_metrics.jsonl`.

Runtime decision thresholds are policy knobs rather than fixed checkpoint contracts. `cid benchmark`
exposes the recommended defaults through CLI options such as `--need-threshold`,
`--convergence-threshold`, `--allocation-threshold`, `--binding-threshold`, routing/presence
thresholds, retrieval similarity, reclamation watermarks, and display-revision controls. Override them
for calibration or deployment-specific latency/recall trade-offs; omitting them uses the recommended
defaults shared by the Python runtime. Run `cid benchmark --help` for the complete set. Stage A
and Stage B training also expose `--rollout-allocation-threshold` and
`--rollout-max-allocations-per-step` so closed-loop training can record the same runtime policy
explicitly.

## Quick demo

```bash
cid demo
```

The demo runs a delayed read-only source. CID starts another denoising step while the source
request is outstanding, then assimilates the observation through the existing binding.

## Citation

Preprint: [arXiv:2608.10438](https://arxiv.org/abs/2608.10438) · DOI: [10.48550/arXiv.2608.10438](https://doi.org/10.48550/arXiv.2608.10438)

```bibtex
@article{cao2026continuous,
  title   = {Continuous Interaction Diffusion: A Diffusion-Native Runtime for Asynchronous Tool-Augmented Reasoning},
  author  = {Cao, Yuhang},
  journal = {arXiv preprint arXiv:2608.10438},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.10438},
  url     = {https://arxiv.org/abs/2608.10438}
}
```

## Current implementation boundary

The v0 runtime intentionally accepts only read-only sources. Side-effecting tools need a
separate commitment/authorization protocol and are outside this repository's first milestone.

The paper source is maintained separately in
[`fwerkor/continuous-interaction-diffusion-paper`](https://github.com/fwerkor/continuous-interaction-diffusion-paper).

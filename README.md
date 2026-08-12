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

## iLLaDA backbone adapter

CID now includes a real adapter for `GSAI-ML/iLLaDA-8B-Base`. The adapter keeps iLLaDA's native
masked-token input embedding and LM head for the display channel, runs `[TCT | prompt | display]`
through the same bidirectional decoder, and attaches CID-specific external-perception fusion and
prediction heads. The immutable prompt remains token-level conditioning rather than being flattened
into protected Facts. Empty TCT slots are masked as attention keys so they can query context for
allocation without contaminating the current display.

```python
import torch

from cid.model import ILLaDACIDAdapter

model = ILLaDACIDAdapter.from_pretrained(
    freeze_backbone=True,
    dtype=torch.bfloat16,
)
```

The loader pins the official checkpoint revision used by this repository and enables the model's
required Hugging Face remote code. The full checkpoint is about 16.5 GB. For an interface smoke
test using the official iLLaDA implementation without downloading the 8B weights:

```bash
python examples/illada_tiny_smoke.py
```

iLLaDA's tokenizer uses token id `5` for `<[MASK]>`; the same value is exported as
`ILLADA_MASK_TOKEN_ID`. `CIDDiffusionScheduler` now supplies masked-display corruption, continuous
TCT corruption, confidence-ranked iterative reveal, and bounded visible-token revision. Training
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
`data/public-datasets.json` and `data/public-interaction-datasets.json`, with the complete registry
documented in `docs/public-datasets.md`. Build the pinned 10,000-task general pool with:

```bash
cid build-public-task-pool
```

Build the separate 10,000-task interaction-heavy pool from 2WikiMultiHopQA and MuSiQue with:

```bash
cid build-public-task-pool \
  --registry data/public-interaction-datasets.json \
  --output data/generated/public-interaction-task-pool-v1.jsonl \
  --manifest-output data/generated/public-interaction-task-pool-v1.manifest.json
```

Convert a pool into teacher-ready CID tasks and causal evidence-exposure jobs with:

```bash
cid prepare-public-distillation
```

The current overall semantic mixture is pinned by `data/training-semantic-mixture-v5.json` and
contains 65,655 tasks. It combines the 18,055-task public mixture, 10,000 generated mechanism tasks,
12,000 computational-tool tasks, 15,600 symbolic-tool tasks, and 10,000 speculative local-correction
tasks. The computational component covers
calculator use, deterministic Python execution, immutable record lookup followed by calculation,
serial dependencies, parallel fan-out/merge, and negative examples where tools are deliberately
unnecessary. The symbolic component adds exact equation solving, systems, expansion, factorization,
rational simplification, differentiation, integration, identity checking, symbolic-to-numeric
chains, record-to-symbolic chains, parallel symbolic merge, and another 1,200 no-tool calibration
cases. The local-correction component explicitly supervises a plausible but wrong hypothesis,
contradictory evidence that reopens only that hypothesis and its dependent answer, and independent
confirmation that stabilizes the corrected state while unrelated cells remain unchanged. Retrieval
tasks still use task-local `workspace_search`/`workspace_read`; dynamic and streaming mechanism tasks
reuse persistent bindings. Causal teacher jobs reveal evidence values only after their corresponding
arrival stage.

The corresponding compiled training mixture is pinned by
`data/training-trajectory-mixture-v5.json`: **179,608 runtime trajectories and 1,027,548 adjacent
supervised transitions** across the same six semantic components. Materialize the single JSONL
expected by Stage A/B with:

```bash
cid materialize-trajectory-mixture \
  --spec data/training-trajectory-mixture-v5.json \
  --output data/generated/training-trajectories-v5.jsonl \
  --manifest-output data/generated/training-trajectories-v5.manifest.json
```

The materializer verifies every component SHA/count and global `example_id` uniqueness before
writing the combined file. Its output is deterministic; training shuffles rollout windows per epoch
unless `--no-shuffle` is requested. The verified materialized identity is pinned by
`data/training-trajectories-v5.reference-manifest.json` (179,608 examples, 1,027,548 transitions,
SHA-256 `d771d5ddcf94c1b8b7ae9a1b7df38944fc3c5974d34867ec4c0ae392b7c9120b`).

Build the deterministic computational teacher jobs with:

```bash
cid build-computational-training
```

The pinned build and self-distillation hashes are recorded in
`data/computational-teacher-v1.reference-manifest.json`.

Build the deterministic symbolic teacher jobs with:

```bash
cid build-symbolic-training
```

The symbolic component contains 34,800 causal teacher stages; all 15,600 semantic plans pass the
quality gate and compile to 31,200 independently randomized runtime trajectories. Its exact build,
review, tool-replay, and compilation hashes are pinned by
`data/symbolic-teacher-v1.reference-manifest.json`.

Build the deterministic speculative local-correction teacher jobs with:

```bash
cid build-correction-training
```

The released component contains 30,000 validated causal stages and 20,000 compiled trajectories;
its exact generation, review, correction-audit, and compilation hashes are pinned by
`data/correction-teacher-v1.reference-manifest.json`.

Generated task data lives under `data/generated/` and is not committed. Every record retains exact
upstream provenance; semantic IDs are deduplicated and assigned to the CID train/validation/test
split before later toolization or timing augmentation.

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
state, and exact semantic duplicates before the plans reach training.

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
  --output-dir runs/stage-a \
  --thought-capacity 8 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --dtype bf16
```

With six GPUs, use `torchrun`; the command detects the distributed environment automatically:

```bash
torchrun --standalone --nproc-per-node=6 -m cid.cli train \
  --data data/synthetic.jsonl \
  --output-dir runs/stage-a-6gpu \
  --thought-capacity 8 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --dtype bf16
```

Each rank loads the pinned 8B backbone serially before moving it to its GPU, avoiding six concurrent
CPU copies during startup. DDP synchronizes only trainable CID state; frozen backbone parameters and
buffers are excluded from initialization sync. Consecutive trajectory transitions are grouped into
rollout windows, bucketed by window length, padded to equal rank counts, and sharded before training.

Training uses scheduled sampling instead of remaining permanently teacher-forced. By default the
first epoch uses teacher T/Y inputs, the next two epochs linearly ramp the probability of feeding the
model's own detached prediction into the following transition, and later epochs use self-rollout.
The default rollout horizon is three transitions. Configure this with
`--rollout-horizon`, `--teacher-forcing-epochs`, and `--rollout-ramp-epochs`. Teacher trajectories
continue to provide the next-step supervision and event schedule; predicted T/Y replaces only the
input state, preventing incorrect rollouts from becoming self-generated labels.

The trainer pads variable-length prompt/display/external-memory sequences inside each micro-batch.
Gradient checkpointing is enabled by default for the native iLLaDA stack and can be disabled with
`--no-gradient-checkpointing`. With the six-GPU example above, the effective transition batch is
`2 × 8 × 6 = 96`; start at micro-batch 1 if the real trajectory lengths are substantially larger.
Per-sample RoPE position IDs are computed from valid token counts, so padding a short prompt beside
a longer sample cannot shift the logical display positions seen by iLLaDA.

Stage A checkpoints contain trainable CID parameters, optimizer state, progress, and RNG state; they
do not duplicate the frozen 8B backbone. Resume with `--resume <checkpoint>`. For inference,
`load_cid_adapter_checkpoint()` loads the CID-only state into an iLLaDA adapter without constructing
a trainer.

### Stage B full-parameter training

After the CID heads have learned the basic runtime contract, `train-full` unfreezes the whole 8B
model and uses FSDP `FULL_SHARD` across the six GPUs. Parameters remain FP32 master weights while
forward/reduction compute uses BF16 by default. Each native iLLaDA decoder layer is an auto-wrap
unit, so parameter, gradient, and Adam state are sharded instead of replicated on every A6000.

```bash
torchrun --standalone --nproc-per-node=6 -m cid.cli train-full \
  --data data/distilled.jsonl \
  --output-dir runs/stage-b-6gpu \
  --init-cid-checkpoint runs/stage-a/stage-a-step-00001000.pt \
  --thought-capacity 8 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --dtype bf16
```

Stage B creates a frozen BF16 snapshot of the pretrained input embedding before FSDP wrapping. The
snapshot is used only to encode dataset transport targets and external-memory text. Prompt/display
token IDs still use the live trainable iLLaDA embedding inside the FSDP forward. This avoids calling
a sharded embedding outside FSDP and keeps the target retrieval space stable while the backbone
updates.

Full-parameter checkpoints are directories written with `torch.distributed.checkpoint`; model and
optimizer states stay sharded and are never gathered as an 8B checkpoint on rank 0. Each rank also
saves its own diffusion/shuffle RNG state and progress. Resume with:

```bash
torchrun --standalone --nproc-per-node=6 -m cid.cli train-full \
  --data data/distilled.jsonl \
  --output-dir runs/stage-b-6gpu \
  --resume runs/stage-b-6gpu/stage-b-step-00002000 \
  --thought-capacity 8 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --dtype bf16
```

Resume currently requires the same world size and the exact same training JSONL SHA-256. Global
epoch numbering and per-rank RNG state continue from the checkpoint. The launcher serializes the
initial full-model load across ranks to limit host-RAM pressure; the final A6000 memory envelope must
still be measured on the real cards before increasing micro-batch size.

### Neural replay benchmark

`cid benchmark` runs a trained CID checkpoint through the same step-exact replay sources used by the
runtime evaluator and writes both per-case JSONL and an aggregate JSON summary. The default starts
from an empty TCT; `--seed-teacher-state` is available only as a diagnostic that isolates downstream
interaction behavior from initial cognitive allocation.

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
torchrun --standalone --nproc-per-node=6 -m cid.cli benchmark \
  --data data/validation.jsonl \
  --checkpoint runs/stage-b-6gpu/stage-b-step-00002000 \
  --checkpoint-kind stage-b \
  --output runs/eval-stage-b/cases.jsonl \
  --summary-output runs/eval-stage-b/summary.json \
  --dtype bf16
```

The summary reports convergence, exact display accuracy, observation coverage/staleness,
latent-to-executable delay, binding-to-observation delay, and observation-to-projection lag. Stage B
benchmark loading restores model shards only; it does not construct or load optimizer state.

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

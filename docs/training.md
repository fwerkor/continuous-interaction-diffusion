# Training plan

The implementation is organized for staged conversion rather than immediate from-scratch 4B
pretraining.

## Stage 0 — runtime oracle

Use scripted/oracle policies to validate binding lifecycle, event timing, cache reuse, dynamic
refresh, fact protection, and metrics independently of neural quality.

## Stage 1 — supervised CID adapters

Start from an existing masked-diffusion language model. Freeze most of the backbone initially and
train:

The first supported backbone is `GSAI-ML/iLLaDA-8B-Base`. `ILLaDACIDAdapter` reuses its native
token embedding, bidirectional decoder, and LM head. With `freeze_backbone=True`, only the CID
projections, external-perception fusion, and prediction heads are trainable; later stages can
unfreeze the same backbone without changing the tensor/runtime ABI.

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

Teacher trajectories should contain pre-arrival and post-arrival states, not only final answers.
They should also supervise cell creation, retirement, optional split/merge lineage, and stable cell
identity across physical compaction. Arrival time, source freshness, cache availability, and
physical slot placement are randomized so a model cannot assign permanent semantics to slot index.
Allocation loss is masked to slots that are `EMPTY` at the current step. Lifecycle cross-entropy is
masked to existing cells and has four classes: `ACTIVE`, `WAITING`, `STABLE`, and `RETIRED`.
Runtime-gated transitions remain hard constraints during both training rollouts and inference.
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
CID-only checkpoints. When the iLLaDA backbone is frozen, checkpoints contain only trainable CID
parameters plus optimizer/progress/RNG state; the pinned pretrained backbone is reloaded separately.
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
micro-batch is not overweighted. Native iLLaDA gradient checkpointing is enabled by the launcher by
default to reduce activation memory while retaining gradients to CID inputs through the frozen
backbone. The iLLaDA adapter also constructs per-sample RoPE `position_ids` from valid prompt and
display lengths. Padding introduced by another sample therefore does not alter the logical
positions of a trajectory's display tokens.

This DDP path is for the frozen-backbone Stage A phase.

### Stage B FSDP full-parameter launcher

`cid train-full` is the full-parameter continuation path for the six-A6000 setup. It requires a
multi-GPU `torchrun` job and wraps each native iLLaDA decoder layer with FSDP `FULL_SHARD` using
`use_orig_params=True`. The underlying trainable parameters remain FP32 master weights; BF16 is the
default forward and gradient-reduction dtype. `CIDTrainer` delegates gradient clipping to FSDP so
the norm is computed across shards instead of clipping each rank's local fragment independently.

The trajectory tensorizer cannot call a sharded token embedding outside FSDP forward. Before the
model is wrapped, Stage B therefore copies the pretrained input embedding into a frozen BF16 text
encoder. Dataset semantic transport text, external-memory records, and grounding/argument retrieval
targets are encoded through that fixed snapshot. Prompt and display IDs remain ordinary model
inputs and use the live, trainable embedding inside the FSDP forward. Besides avoiding illegal
out-of-forward access to sharded parameters, this keeps target embeddings stationary while the
language model changes.

Stage B checkpoints use `torch.distributed.checkpoint` for model and optimizer state. No rank
materializes a full 8B state dict. Rank-local trainer files preserve diffusion RNG, shuffle RNG,
transition/optimizer counts, and completed epoch count. Metadata pins backbone geometry, CID
adapter config, original world size, and the training JSONL SHA-256. Resume rejects a different
dataset or world size and continues the next global epoch rather than restarting the shuffle seed.

An optional Stage A CID-only checkpoint may initialize the CID heads before full unfreezing. It is
mutually exclusive with Stage B `--resume`, which already restores the entire model and optimizer.
The repository tests the FSDP path both with a CPU world-size-1 checkpoint round trip and a true
two-rank Gloo `FULL_SHARD` forward/backward/distributed-checkpoint smoke. The real 8B A6000 memory
ceiling still needs to be measured before raising local micro-batch size above one.

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

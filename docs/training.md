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
backbone.

This DDP path is for the frozen-backbone Stage A phase. Joint/full-parameter training in Stage 2
requires sharded model/optimizer state (FSDP or equivalent) rather than replicating Adam state on
every GPU.

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

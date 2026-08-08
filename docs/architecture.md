# CID v0 architecture

This document turns the paper's open design space into concrete implementation choices for the
first trainable system. The choices are deliberately narrow enough to implement and ablate.

## 1. Stable boundary: runtime objects, model tensors inside

The runtime owns facts, bindings, external jobs, cache state, provenance, and refresh policy. It
never reaches into arbitrary backbone activations. The model receives a frozen `FactSnapshot`, a
TCT, a display canvas, source descriptors, and current percepts; it returns updated T/Y state plus
typed information needs.

This is the most important boundary in the repository. It lets us replace the neural backbone
without changing asynchronous semantics, and it prevents a model adapter from silently mutating
protected external facts.

## 2. TCT v0: fixed capacity, dynamic occupancy

The neural interface uses a fixed physical capacity `N_max`, but the number of cognitive objects
is dynamic. At diffusion step `s`, only a subset of the physical slots is occupied. Each occupied
slot contains:

- a stable logical `cell_id` independent of physical position;
- semantic vector `h`;
- soft role distribution;
- uncertainty scalar;
- local diffusion/editability scalar;
- lifecycle state;
- sparse anchors and links carried by the runtime/data layer.

`EMPTY` is a physical-storage state, not a neural lifecycle class. The allocation head is evaluated
for empty slots and creates a new logical cell in `ACTIVE`. Existing cells use a separate lifecycle
head over `ACTIVE / WAITING / STABLE / RETIRED`. `RETIRED` preserves identity and lineage until the
runtime explicitly reclaims the storage back to `EMPTY`.

This gives the model a fixed tensor shape

```text
[batch, N_max, d_model]
```

for efficient batching and distributed training while allowing actual cognitive occupancy to
grow and shrink with task complexity. The PyTorch reference core receives an occupancy feature for
every physical slot. Its allocation logits are trained only on currently empty slots; lifecycle
logits are trained only for existing cells.

Bindings, fact links, and cognitive edges never store physical slot indices. They reference stable
`cell_id` values, so compaction can move a cell from one physical slot to another without changing
its external identity. The state layer supports allocation, retirement, reclamation, compaction,
split, and merge while keeping `N_max` constant.

### 2.1 Event-aware lifecycle transitions

Lifecycle logits are proposals rather than direct state writes. `LifecycleTransitionController`
combines the proposal with binding and revision state before committing a transition:

- `ACTIVE -> WAITING` is valid only when an unresolved binding targets that cell;
- a `WAITING` cell cannot leave while any targeted binding remains unresolved;
- once all targeted bindings are resolved, `WAITING -> ACTIVE` is runtime-enabled;
- `STABLE -> ACTIVE` requires an explicit local `reopen_cells` signal;
- `RETIRED` cannot be reactivated or converted to `EMPTY` by model output;
- `RETIRED -> EMPTY` is runtime garbage collection through explicit reclamation.

This keeps differentiable lifecycle prediction in the model while making external-event and memory
safety invariants non-negotiable at runtime.

### 2.2 Retired-cell archive and reclamation

`RETIRED` cells are not immediately erased. The runtime tracks their retirement step and keeps the
full cell in TCT until it is safe and useful to reclaim the physical slot. Reclamation is triggered
when the fraction of empty physical slots falls below a configurable low watermark, and a final
safe sweep also runs when a trajectory ends. Eligible cells are reclaimed oldest first until the
target watermark is reached.

A retired cell is reclaimable only after a configurable grace period, when no non-retired binding
targets it, and when no live cognitive cell has a strong relation requiring its neural state.
`DEPENDS_ON`, `REQUESTS`, `OBSERVES`, and `CONSTRAINS` are strong relations. Historical relations
such as `DERIVED_FROM`, `SUPPORTS`, `CONFLICTS`, and `REFERS_TO` may continue to point to an archived
cell.

Before physical reclamation, runtime writes a `CognitiveTombstone` containing stable identity,
roles, anchors, typed links, creation/retirement/archive steps, former physical slot, and binding
provenance. The full semantic vector is deliberately omitted. Weak links can therefore remain
resolvable through the archive without keeping an expensive neural slot alive. After one or more
reclaims, the field is compacted; stable `cell_id` references make this position-independent.

### 2.3 Typed grounding layer

TCT grounding is an explicit ABI rather than an untyped string convention. `ObjectRef` identifies
the runtime object being referenced and distinguishes `CELL`, `FACT`, `BINDING`, `SOURCE`,
`DISPLAY_SPAN`, `ANCHOR`, and external `SYMBOLIC_OBJECT` identities. Information-need targets,
percept targets, reopen requests, and cognitive links use these references rather than physical
slot indices or ambiguous strings.

An `Anchor` gives a continuous cognitive cell a typed symbolic attachment. The initial schema
supports entity, number, symbol, span, path, URL, and text anchors, with canonical `object_id`,
confidence, numeric units, and token spans where applicable. A `CognitiveLink` is a typed edge with
relations such as `SUPPORTS`, `CONFLICTS`, `DEPENDS_ON`, `DERIVED_FROM`, `REQUESTS`, `OBSERVES`,
`CONSTRAINS`, and `REFERS_TO`. Split/merge lineage is represented with `DERIVED_FROM` cell edges.

The neural interface allows multiple anchors and links per cognitive cell. It therefore uses small
fixed grounding capacities per physical TCT slot and predicts dynamic presence, anchor type,
canonical-object retrieval queries, link relation, link-target type, and link-target retrieval
queries. This preserves fixed tensor shapes without assuming that one cognitive object can mention
only one symbolic object or relation.

For pre-training runtime tests, `ClosedWorldGrounder` provides deterministic resolution over a
trajectory-local `GroundingEntry` catalog. Tool observations may carry typed anchors; when such an
anchor matches an anchor already attached to a live cognitive cell, the runtime can add that cell
to the percept's routing targets even when the original binding did not explicitly name it. This
tests symbolic-to-continuous routing independently of learned entity linking.

Open-world retrieval and canonicalization are intentionally not part of v0. The training/runtime
ABI is fixed now so a learned resolver can replace the oracle later without changing TCT state,
dataset, or checkpoint geometry.

## 3. Shared refinement backbone

T and Y are refined by one backbone with channel/type embeddings, followed by channel-specific
heads. External facts/percepts are supplied as cross-attention memory. This makes information able
to move between cognition and display during the same update while preserving separate output
semantics.

The reference `TorchCIDCore` is intentionally small and generic. It is not the final 4B model. Its
API is the target for an adapter around an existing masked-diffusion LM: map model hidden states to
the fixed-capacity TCT, retain the model's masked-token denoising for Y, and attach CID
allocation/lifecycle/role/intent/revision plus typed grounding heads.

The first real bridge is `ILLaDACIDAdapter` for `GSAI-ML/iLLaDA-8B-Base`. iLLaDA accepts
`inputs_embeds`, so CID runs `[TCT | prompt | display]` through the native bidirectional decoder.
The prompt is immutable token-level conditioning, not a fourth mutable cognitive channel and not a
`FactItem`. This preserves its language structure while reserving F for externally protected facts.
Display logits still come from the checkpoint's original LM head. The adapter does not replace or
emulate iLLaDA's language-model stack.

Training micro-batches may pad prompt and display regions to different widths. The adapter therefore
passes explicit per-sample RoPE `position_ids`: TCT positions remain fixed, valid prompt tokens are
contiguous after TCT, and valid display tokens begin immediately after that sample's real prompt
length. Padding positions are attention-masked and cannot shift the logical location of another
sample's display tokens.

Empty TCT slots remain query positions because allocation must be predicted for them, but they are
masked as attention keys. Occupied TCT cells and display tokens are visible keys. This prevents
unused fixed-capacity storage from perturbing the display while allowing an empty slot to infer
whether it should allocate from the current context. Facts and percepts remain separate external
memory channels and enter through the shared CID cross-attention fusion after native backbone
refinement. The external residual gate is initialized near zero so the untrained CID path starts
close to the pretrained display behavior.

The runtime-facing `ILLaDANeuralPolicy` tensorizes `ModelContext`, executes the adapter, reveals a
subset of masked display tokens, and passes neural outputs through `CIDMaterializer`. Materialization
turns allocation/lifecycle logits into TCT proposals, retrieves typed anchors and link targets from
a trajectory-local catalog, decodes schema-positioned argument slots, emits persistent
`InformationNeed` objects, and converts revision predictions into typed reopen references. Runtime
lifecycle gates remain authoritative after materialization.

## 4. Information need before executable call

A runtime-visible need has a stable `need_id`, source probabilities, partially bound arguments,
confidence, freshness demand, and typed cell/display targets. Runtime activation has two gates:

1. need confidence must cross the binding threshold;
2. a source must be selected and its required arguments must be executable.

The training model exposes argument presence and one retrieval query for each source-schema argument
position. An information need may therefore become visible before all required argument slots are
grounded. That lead time is directly measurable and corresponds to RQ1.

## 5. One need, persistent binding

Bindings are keyed by `need_id`; external work is keyed by `(source, canonical arguments)`. This
means two cognitive needs may keep distinct bindings and target links while sharing a single
in-flight or cached external observation.

Every denoising step with an available active binding emits a fresh `Percept` object to the model.
No external call is required for this cognitive refresh. Dynamic refresh is controlled separately
by binding policy.

## 6. Async execution

Model steps run in a worker thread from the asyncio runtime. Read-only source jobs run as asyncio
tasks. Therefore source latency can overlap actual model compute rather than merely alternating
between `model.step()` and `await tool()`.

The runtime does not seal a model-declared converged display while an active required binding is
still unresolved. A hard step/wall-clock budget can still terminate the trajectory.

## 7. Local reopening

The neural core predicts support/conflict evidence per cognitive slot. Training converts those
signals into local editability targets. Runtime-facing policies may also increase local noise for
cells linked to a newly conflicting percept and emit typed cell references through `reopen_cells`.
The transition controller is the gate that permits `STABLE -> ACTIVE`. We keep this mechanism
explicit instead of hiding it inside a global remasking schedule so RQ4/RQ5 ablations are possible.

## 8. Source scope

v0 sources are read-only. A source descriptor declares cacheability, dynamism, streaming/version
properties, and required arguments. Mutation, authorization, rollback, and irreversible actions
are intentionally excluded until perception/revision works on its own.

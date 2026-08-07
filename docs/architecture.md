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
- lifecycle logits;
- sparse anchors and links carried by the runtime/data layer.

The lifecycle is `EMPTY -> ACTIVE/WAITING/STABLE -> RETIRED -> EMPTY`. `EMPTY` means no cognitive
object exists in that physical slot. `RETIRED` preserves the identity and lineage of a cognitive
object that has stopped participating in current reasoning until the runtime reclaims its storage.

This gives the model a fixed tensor shape

```text
[batch, N_max, d_model]
```

for efficient batching and distributed training while allowing actual cognitive occupancy to
grow and shrink with task complexity. The PyTorch reference core receives an occupancy feature for
every physical slot and predicts next-step occupancy in addition to lifecycle. Empty slots remain
available as allocation candidates.

Bindings, fact links, and cognitive edges never store physical slot indices. They reference stable
`cell_id` values, so compaction can move a cell from one physical slot to another without changing
its external identity. The state layer supports allocation, retirement, reclamation, compaction,
split, and merge while keeping `N_max` constant.

## 3. Shared refinement backbone

T and Y are refined by one backbone with channel/type embeddings, followed by channel-specific
heads. External facts/percepts are supplied as cross-attention memory. This makes information able
to move between cognition and display during the same update while preserving separate output
semantics.

The reference `TorchCIDCore` is intentionally small and generic. It is not the final 4B model. Its
API is the target for an adapter around an existing masked-diffusion LM: map model hidden states to
the fixed-capacity TCT, retain the model's masked-token denoising for Y, and attach CID
occupancy/role/intent/revision heads.

## 4. Information need before executable call

A runtime-visible need has a stable `need_id`, source probabilities, partially bound arguments,
confidence, freshness demand, and target cell/display links. Runtime activation has two gates:

1. need confidence must cross the binding threshold;
2. a source must be selected and its required arguments must be executable.

The training model can expose useful source/need confidence before the second gate is satisfied.
That lead time is directly measurable and corresponds to RQ1.

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
cells linked to a newly conflicting percept. We keep this mechanism explicit instead of hiding it
inside a global remasking schedule so RQ4/RQ5 ablations are possible.

## 8. Source scope

v0 sources are read-only. A source descriptor declares cacheability, dynamism, streaming/version
properties, and required arguments. Mutation, authorization, rollback, and irreversible actions
are intentionally excluded until perception/revision works on its own.

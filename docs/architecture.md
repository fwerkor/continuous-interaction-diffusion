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

## 2. TCT v0: fixed slots, typed side channels

The first neural model uses a fixed number `N` of cognitive slots. Each slot contains:

- semantic vector `h`;
- soft role distribution;
- uncertainty scalar;
- local diffusion/editability scalar;
- lifecycle logits;
- sparse anchors and links carried by the runtime/data layer.

Fixed slots are chosen for v0 because dynamic allocation would entangle the core hypothesis with
memory-management policy. Slot creation/retirement can be tested later as a separate extension.

## 3. Shared refinement backbone

T and Y are refined by one backbone with channel/type embeddings, followed by channel-specific
heads. External facts/percepts are supplied as cross-attention memory. This makes information able
to move between cognition and display during the same update while preserving separate output
semantics.

The reference `TorchCIDCore` is intentionally small and generic. It is not the final 4B model. Its
API is the target for an adapter around an existing masked-diffusion LM: map model hidden states to
T slots, retain the model's masked-token denoising for Y, and attach CID role/intent/revision heads.

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

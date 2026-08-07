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
cognition changes.

TCT uses a fixed physical capacity for efficient tensor execution but dynamic logical occupancy.
Cognitive objects receive stable `cell_id` values, so they can be allocated, retired, reclaimed,
split, merged, or physically compacted without binding tool/fact links to tensor position.
Allocation is predicted separately from lifecycle: empty slots may allocate new `ACTIVE` cells,
while existing cells transition among `ACTIVE / WAITING / STABLE / RETIRED` under runtime gates.
Unresolved bindings hold `WAITING`, explicit revision signals reopen `STABLE`, and only runtime
reclamation can turn `RETIRED` storage back into `EMPTY`.

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
src/cid/lifecycle.py        Event-aware cognitive lifecycle transition controller
src/cid/contracts.py        Model/runtime information-need and percept contracts
src/cid/runtime/            Async scheduler, bindings, source registry, traces
src/cid/model/torch_core.py Trainable PyTorch reference core (optional dependency)
src/cid/data.py             Trajectory JSONL schema and validation
src/cid/metrics.py          CID-specific interaction metrics
docs/architecture.md        Concrete v0 architecture decisions
docs/training.md            Staged training plan
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

## Quick demo

```bash
cid demo
```

The demo runs a delayed read-only source. CID starts another denoising step while the source
request is outstanding, then assimilates the observation through the existing binding.

## Current implementation boundary

The v0 runtime intentionally accepts only read-only sources. Side-effecting tools need a
separate commitment/authorization protocol and are outside this repository's first milestone.

The paper source is maintained separately in
[`fwerkor/continuous-interaction-diffusion-paper`](https://github.com/fwerkor/continuous-interaction-diffusion-paper).

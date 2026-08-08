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
TCT corruption, and confidence-ranked iterative reveal.

The neural path is executable end to end: `ILLaDAContextTensorizer` converts a runtime
`ModelContext`, `ILLaDANeuralPolicy` performs a denoising step, and `CIDMaterializer` converts
allocation/lifecycle/need/source/argument/grounding/revision/refresh predictions back into the
typed runtime contract. Closed-world candidate retrieval is used for the first training stage.

Training data uses typed per-step `ThoughtTarget` and `DisplayTarget` records rather than free-form
dictionaries. `ILLaDATrajectoryTensorizer` constructs adjacent-step supervision and the CID loss
performs permutation-invariant assignment for multi-anchor and multi-link targets. The test suite
includes a complete tiny-backbone optimizer step with the pretrained backbone frozen.

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

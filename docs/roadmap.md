# Implementation roadmap

CID should be developed in layers so a failure in one hypothesis does not invalidate the rest of
the system.

## M0 — executable semantics

Status: implemented in the initial repository scaffold.

- protected fact snapshots;
- fixed-capacity, dynamically occupied TCT with stable cell identity;
- separate empty-slot allocation and event-gated cell lifecycle semantics;
- pressure-aware retired-cell archival, reclamation, and physical compaction;
- typed anchors, object references, cognitive links, and closed-world oracle grounding;
- typed latent information needs;
- persistent bindings with external-I/O deduplication;
- asynchronous model/source overlap;
- static cognitive re-projection and dynamic refresh policies;
- trajectory schema and CID-specific runtime metrics;
- small PyTorch reference core and multi-head training objective.

Exit criterion: runtime invariants are covered by deterministic tests and do not depend on model
quality.

## M1 — masked-diffusion backbone adapter

Status: backbone bridge implemented for iLLaDA; runtime materialization and argument decoding remain
before the M1 exit criterion is complete.

Implement the first real model bridge around an existing masked-diffusion LM. The adapter should:

1. reserve a fixed TCT capacity and learn allocation decisions for empty physical slots;
2. preserve the backbone's masked-token denoising path for Y;
3. encode protected facts separately from transient percepts;
4. materialize need/source/anchor/link/refresh predictions into the runtime contract;
5. map runtime percepts back into context-conditioned percept embeddings;
6. expose local revision signals without serializing cognition into text.

The adapter must randomize or compact physical slot placement during training so learned cognitive
roles attach to cell content and type rather than hard-coded positions.

Exit criterion: an untrained or lightly trained adapter can run complete CID trajectories with the
same runtime used by the oracle policies.

## M2 — synthetic trajectory factory

Build data generators for the five evaluation families: static copying, delayed retrieval, dynamic
state tracking, streaming evidence, and competing sources. Every generated sample must include
arrival timing, closed-world symbolic catalogs, typed grounding targets, and pre-/post-arrival
states rather than flattening evidence into the prompt.

Exit criterion: millions of reproducible trajectories can be generated with controlled event
latency, freshness, cache state, and counterfactual arrival schedules.

## M3 — adapter training

Train CID-specific heads and adapters first, with most backbone weights frozen. Measure source
selection, argument binding, intent lead time, assimilation lag, exact copying, and stale-value
rate before unfreezing the backbone.

Exit criterion: latent needs reliably precede executable calls and new observations produce local,
beneficial revisions rather than global corruption.

## M4 — joint T/Y training

Unfreeze selected backbone blocks and train coupled thought/display denoising with randomized event
schedules. Add the paper's ablations: untyped latent state, no anchors, no persistent re-projection,
no dynamic refresh, global rather than local reopening, and asynchronous autoregressive baselines.

Exit criterion: CID demonstrates a measurable quality/latency advantage attributable to revision
and persistent perception, not only to asynchronous I/O.

## M5 — dedicated small model

Only after M1--M4 establish the mechanism should we train a dedicated ~4B-class CID model. The
runtime/data contract remains unchanged; scaling should improve the model rather than redefine the
system around a new checkpoint.

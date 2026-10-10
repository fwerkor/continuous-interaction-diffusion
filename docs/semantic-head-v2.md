# Semantic head v2 (neural contract 6)

Stage A revealed a scale conflict in neural contract 5. The target Thought vectors
are pooled backbone embeddings, but the original cell-routing cosine operated
directly on the predicted Thought. At an empty bootstrap cell the residual head
initially emits zero. Normalizing that zero vector with an epsilon of 1e-12
produces enormous routing gradients, which can distort the semantic predictor.

Contract 6 makes three numerical changes:

- The semantic delta is predicted from normalized backbone features, scaled to
  the frozen semantic embedding RMS and capped at four times that RMS after
  combining it with the previous Thought.
- Cell matching has independent, 128-dimensional query/key projections over
  backbone features. Source-descriptor matching retains the separate
  embedding-space query. These routes cannot send gradients into the semantic
  residual head.
- Low-norm route vectors and cosine grounding losses use bounded
  normalization floors. Non-finite Thought states still fail validation.

Stage A and Stage B training always create semantic head version 2. This is an
incompatible training contract; contract-5 checkpoints must not be resumed as
contract 6. Legacy portable inference models retain semantic head version 1
when their adapter configuration lacks a version field, preserving their
original behavior. An existing contract-5 checkpoint can still be evaluated
with its original code, but producing a contract-6 model requires a new training
run.

The RMS cap prevents runaway closed-loop magnitude, but does not establish that
semantic predictions, tool routing or display decoding are accurate. Verify
those behaviors with teacher-forced and closed-loop validation before release.

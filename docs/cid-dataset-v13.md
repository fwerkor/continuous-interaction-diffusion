# CID-Dataset v1.3 — complete conditional Display supervision

This release is a checksummed additive overlay of CID-Dataset v1.2. It leaves
the v1.1 trajectory archive, all original Thought/tool targets, and the v1.2
Language SFT component unchanged. It adds complete-reply contracts and
tool-grounded counterfactual trajectories.

## Data contract

* `source-complete-v1` annotations preserve the original reference verbatim.
  The builder accepts explicit short-output contracts and source tasks whose
  original target is a complete natural response. Only matching Display states
  at or after the last observed event are annotated. This is deterministic
  format/provenance verification, not a claim of independent human fact review.
  Unclassified targets remain answer fragments.
* The additional counterfactual pairs have the same prompt and lookup key,
  identical pre-observation Thought, different recorded tool results, and
  correspondingly different complete replies. Each variant is a separate
  task-local world. The validation split uses disjoint record IDs. These are
  verified synthetic fixtures, not real-world measurements.
* Unknown and provisional outputs never receive positive completion labels.
  Existing independently verified full replies retain their existing contract.

## Materialize

Obtain the existing `release/training-trajectories.jsonl` and all four files
under `components/display-v13/` in CID-Dataset main (or tag v1.3).
The base file and overlay file digests are checked before the output is
committed:

```sh
python scripts/build_cid_dataset_v13.py materialize \
  --base /path/to/CID-Dataset/release/training-trajectories.jsonl \
  --release-dir /path/to/CID-Dataset/components/display-v13 \
  --output /path/to/training-trajectories-v1.3.jsonl
```

The materialized file contains 422,230 preserved base trajectories plus
6,000 additional training variants. The independent 256 validation variants
are not appended. Keep using v1.2's separate language_sft training split.

## Training behavior

For native LFM-style Display, complete references use randomly sampled masks
at the sampled discrete denoising level, with at least one supervised token.
The lexical completion marker follows the original native model protocol;
logical EOS remains the hidden transport sentinel. No reference text or length
is supplied to inference. Bootstrap without a language-only plan uses a
fully masked canvas. Genuine prior drafts and closed-loop predictions stay
editable; correct visible tokens are excluded from CE, and changed/masked
reference tokens are supervised.

For equal-token-length matched references, a fraction of noninitial noise
samples reveal only the shared tokens and supervise the differing tokens.
For other samples ordinary random masks are used. This is teacher-forced
training augmentation only. Full-reference examples have no duplicate span
loss; legacy fragment loss has coefficient 0.1, and no prescribed answer
offset. Language-only examples use task-compatible input-only plans and
remain Display-only, without TCT labels.

No new architecture, extra distillation stage or decoding heuristic is
introduced. Use a clean initial base for a formal A/B retrain; do not resume
an old optimizer state on the changed objective and dataset. Evaluate
independently held-out English, matched counterfactual facts, real tool
interaction, revision, repetition, and completion quality before committing
to long training. The implementation smoke tests establish execution
correctness, not recovered language ability or generalization.

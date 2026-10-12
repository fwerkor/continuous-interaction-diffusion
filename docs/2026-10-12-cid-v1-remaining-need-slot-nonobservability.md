# CID v1 — Remaining non-observable Need-slot labels after the canonical-order patch

**Scope:** research-only worktree `audit/cid-future-need-invariance-20261012`. No production training, checkpoints, runtime defaults, or main-branch code changed. The CID v1 architecture, including fully revisable Display tokens and EOS, is preserved.

## Newly confirmed deterministic counterexamples

The repair in `62300f6` and `8cc21bc` removes hash-randomized slot assignment and sorts by `(first_need_step, source_rank, json.dumps(arguments), freshness, need_id)`. This makes typical per-owner slot labels more consistent but does **not** establish input observability.

Using the actual `ILLaDATrajectoryTensorizer.tensorize`, the same adapter, step, timestep, and identical CPU random generator seed:

1. Two simultaneous same-owner/same-source Needs have unavailable arguments until step 2. Changing only one **future argument value** reverses their sorted Need slots at target step 1. All fields of `CIDTensorBatch` are bitwise identical; the two supervised Need logits change from `[1.0, 0.35]` to `[0.35, 1.0]`.
2. Two simultaneous same-owner/same-source Needs have otherwise identical sort keys but different confidence and Display routes. Swapping only their hidden `need_id` names reverses both Need supervision and Display routing supervision. Every `CIDTensorBatch` field is bitwise identical.

Thus the target-slot convention remains non-identifiable in such cases. A deterministic sort is not equivalent to a model-observable sort.

## Measured prevalence and limitations

A complete source scan of the first 250,000 lines of the released 422,230-trajectory `training-trajectories.jsonl` found **744** examples with repeated owner Cells, all in the first 100,000 lines examined. In those cases all **744** repeated groups shared the same Source and first Need step, with different arguments; none of these 744 groups had `argument_steps` later than `first_need_step`. **The remaining 172,230 lines were not successfully scanned**; no corpus-level hidden-future prevalence claim is justified. Even arguments marked available by step metadata are not by themselves evidence that the exact values appear in the prompt/TCT tensor. The counterexamples establish a general contract defect but do not prove that this second defect causes the existing production 0.4B failure.

In the actual 0.4B heldout assessment documented separately, allocation and Need materialization were already failing at the operational thresholds, and corrected Need-head-only transfer still yielded 0/6 exact full answers in a six-task 128-step test. This new finding does not supersede those observations.

## Repair requirements (not yet implemented)

For simultaneous Needs without a fully **model-observable** stable distinguishing key, the loss should be permutation-invariant over their action-slot *sets*, matching Need confidence, Source, argument presence/content, freshness and routing consistently. Matching must not silently swap only one head while leaving the others mismatched. Ensure that the match applies identically in Stage A and B and does not change the runtime's freedom to select ordered action slots. Evaluate performance overhead only on ambiguous groups. Alternatively, introduce observable per-action query identity in the input **before** teaching any per-slot target identity; that is an architectural/interface change requiring full compatibility tests.

Do not reintroduce example-ID hashes, use unavailable future arguments or hidden need IDs for a neural target, lower action thresholds in isolation, or claim that a Need-only checkpoint swap repairs executable action selection. Do not retrain production until a unified label/loss contract and a small full-model Stage A/B 128-step validation are demonstrated.

## Reproduction

```sh
PYTHONPATH=src:tests /workspace/.cid-cpu-test-venv/bin/python experiments/need-future-argument-and-hidden-id-counterfactual.py
```

Runs on CPU in seconds, independently of any active model training.

## Implemented repair and CPU evidence (later continuation, 2026-10-12)

The isolated branch now carries a real production-path remedy:

- Tensorization marks only cells owning 2+ trajectory Need identities as candidate action-permutation groups, even if only one remains active at a particular step.
- The CID multi-head loss computes a detached **joint minimum-cost assignment** between predicted action slots and full supervised actions, then applies the same slot permutation to Need confidence, Source, Cell/Display routing, argument presence, argument embeddings/masks, and freshness. The exact existing Hungarian matcher is reused; single-Need rows retain the original fast path and loss.
- Candidate matching includes the true positive-mass weighting of Need BCE and per-example normalization for the participating action heads. Assignment is nondifferentiable, while all selected normal losses preserve gradients into the relevant heads.
- The loss does not mutate teacher targets. Expensive full-dimension target copies are avoided when the assignment is identity; no CPU/GPU transfer is introduced for ordinary one-Need batches. This adds no restrictions on Display token editing or EOS.

CPU regression tests confirm that changing **only** hidden example/Need IDs or unavailable future arguments leaves the input identical and the *full* CID loss invariant; seven participating action-head gradients are equal and finite. Additional tests cover joint assignment minimizing the actual multi-head objective under alternate weights, three concurrent Needs, collated masked rows, and a real Tiny CIDTrainer Stage A update. A first sweep of training, torch-core, materializer, Stage A/B CLI tests passed with one skipped (see final logs for exact status).

A paired, fixed-features 150-update head-only experiment through the real tensorizer and CID objective yielded:

| Heldout condition (same input, contradictory historical labels) | Old fixed slot supervision | New joint permutation loss |
| --- | --- | --- |
| Need probabilities for two candidate slots | 0.7636, 0.7637 | 0.9910, 0.3500 |
| Candidate actions over 0.6 threshold | 2 | 1 |
| Final optimization loss | 0.30066 | 0.25263 |

This is a controlled **supervision proof**, not full 0.4B model quality: only two Need logits were optimized against fixed, actual CID model features and a tiny model. No improvements in real 128-step exact answers are claimed.

As checked on unist2, all A6000 GPUs 0–7 were already occupied by CUDA processes: GPUs 0–3 on the existing 0.4B Stage B job and GPUs 4–7 by a separate root-owned job. Consequently no new actual 0.4B A/B run was started or any GPU job interrupted; this blocks the full-model acceptance gate, not the correctness of the CPU code tests.

## Checkpoint migration guard

The clean integration branch records \`supervision_contract_version=2\` in new Stage A and Stage B training checkpoints. A resumed optimizer is rejected if that marker is missing or mismatched; this prevents continuing old randomly supervised action heads under the new set-matched objective. Likewise, \`train-full --init-cid-checkpoint\` requires a corrected Stage A checkpoint by default. The inference-only \`load_cid_adapter_checkpoint\` interface remains compatible with old checkpoints to preserve historical baselines and released artifacts. No existing running GPU job is stopped or migrated implicitly.

This *training-only* compatibility marker is separate from \`CID_NEURAL_CONTRACT_VERSION\` and the public CID v1 architecture, so no neural geometry/schema modification is claimed.

# CID Stage C only: algorithm-selection results — 2026-10-26

Stage C optimization was scoped *only* to: no retraining or modifications to CID Stage A/B, T/F reasoning, tool control, or the original runtime. All model backbone weights remained frozen. All experimental checkpoints are optional adapter weights, not replacements of the original CID release. Current main commit: `046f5bc467faac42940b7ed65f8195783b658283`; `main` was not modified or pushed.

## Controlled Stage C adaptation algorithms on frozen published 0.4B

Each trained from the same CID-v1-0.4B-1018 base, Nemotron-v3 pilot 4,000 JSONL rows, same seed 20261023, 2,000 optimizer steps, fp16/bf16 backbone+fp32 residual, lr1e-4, 80% correct-prefix/EOS boundary continuation denoising and 20% full denoising, 2 frozen thought steps. Compare identical 60 prompts/seeds, 37 with reference substrings and 23 without, under Canvas64 and Canvas256. A reference substring is an intentionally weak directional metric, not semantic correctness; longer response tokens do not necessarily mean more fluent.

| Head | 64 fact reference/37 | 64 median output tokens | 64 converged/60 | 64 >=32 tokens/60 | 256 fact reference/37 | 256 median | 256 converged/60 | 256 >=32 tokens/60 | CID17 evaluation correct/17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rank32 linear output LoRA (current balanced C) | 18 | 6 | 56 | 5 | 12 | 7 | 53 | 10 | 3 |
| Rank128 nonlinear vocab-logit residual (fp32 RMSNorm + SiLU projection, zero-init last layer) | 18 | 6 | 57 | 3 | 6 | 6 | 55 | 5 | 3 |
| Rank512 nonlinear Display hidden residual then frozen LM head | 13 | 5 | 56 | 8 | 2 | 4 | 53 | 8 | 0 |

Evaluation CID17: native tool/fact/control benchmark on same 17 exact cases (`data/cid-validation-17.jsonl`), same seed/config; convergence 8/17 for all three head types. Samples too few to establish release-level improvements.

On a separate heldout 64-example *teacher-forced* Nemotron pilot subset: original frozen model NLL 5.9473; linear Rank32 5.6426, linear Rank128 5.5252, nonlinear vocab Rank128 5.5850, hidden Rank512 5.3391. Importantly, the lowest teacher-forced loss (hidden Rank512) has the worst free-run benchmark performance (facts 13/37 and 2/37; CID 0/17). Thus optimizing SFT token NLL is not sufficient for full CID free-run quality.

## Larger diverse Stage C-only SFT trial on free ise2 RTX3090

Main `046f5bc` source tar copied verbatim from original main. Dataset `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` 20k JSONL, random seed 6314 split into 19k training and initially 1000 holdout; 330 holdout rows were removed due to exact prompt overlap, leaving 670 prompt-disjoint holdout rows. From the 19k training pool, run from same original frozen 0.4B release with linear Rank32, lr1e-4, 80/20 continuation/full, max 5k steps, checkpoint every 1k, seed20261026. Five checkpoints independently evaluated on GPU0 while training on GPU1.

| Step | 64 fact reference/37 | 64 median | 64 converged/60 | 64 >=32 tokens | Mean teacher forced loss n=64 holdout | 256 fact reference/37 | 256 median | CID17 evaluation exact |
|---|---:|---:|---:|---:|---:|---:|---|
| 1000 | 18 | 4.5 | 58 | 1 | 5.3119 | — | — | — |
| 2000 | 18 | 5 | 59 | 0 | 5.2451 | — | — | — |
| 3000 | 20 | 6 | 58 | 2 | 5.1969 | 11 | 7 | 2/17, convergence11/17 |
| 4000 | 18 | 5 | 59 | 2 | 5.1629 | — | — | — |
| 5000 | 19 | 4 | 59 | 1 | 5.1124 | 12 | 7 | 2/17, convergence9/17 |

5000 canvas256 convergence55/60 and >=32 tokens6/60. 3000 canvas256 convergence56/60 and >=32 tokens2/60. The original linear Rank32 2000 pilot on canvas256 fact hits12/37 and CID17 evaluation3/17. More steps and broader SFT improve some particular scores and reduce token NLL monotonically but do not dominate across answer correctness, control-task exactness and both Display capacities. Training loss had no reliable monotonic relationship with free-run quality. The same formal17 test file was checksum-verified on mypc and ise2: `0004a2672802ea254812db4d8be430306fb840438bdae8b40098fa79d69cdc37`.

## Decision

**Keep Stage C as original frozen-backbone Rank32 Display-only LoRA with balanced EOS-boundary + full-sequence denoising.** It is the simplest reversible candidate with the most robust tradeoff among tested alternatives. It improves answer length relative to the frozen base but is not a verified full language-restoration solution. Do not switch to hidden-residual adapters just because their SFT NLL is lower; they cause significant native CID regressions. Do not replace a previously used experimental Stage C adapter with the larger-data checkpoint; it does not beat the old baseline across all required regression metrics. Do not change A/B, CID core runtime, or publish a new release based on these limited samples.

## Next Stage C only (no A/B retraining)

When stable ready-to-evaluate unified 2B/4B Stage B model snapshots are available, keep A/B frozen, reuse existing Stage C algorithm and train solely Display rank32 using a balanced corpus (short exact answers + long fluent instructions), choose checkpoint using paired end-to-end multilingual instruction benchmarks, no-regression tool tests and independent heldout data rather than token CE alone. A/B remains the cognitive runtime; Stage C stays optional/reversible. Do not assume a generic public SFT teacher answer is necessarily semantically aligned with CID-produced T states. For 0.4B this is an open research limitation rather than a solved config issue.

## Paths

On mypc-debian under `<experiment-root>/`:
- `code/` original untouched 046f main, `code-nonlinear-exp/` nonlinear vocab head v2, `code-hidden-exp/` frozen-vocab hidden residual head v3.
- trained adapters: `train-eos-balanced-2000/stage-c-latest.pt` (original baseline), `train-nonlinear-r128-2000/stage-c-latest.pt`, `train-hidden-r512-2000/stage-c-latest.pt`.
- `eval-60-*-c{64,256}.json`; `eval-heldout-*.json`; `benchmark-preserve-20261026/`.

On ise2 under `/data/cid4b/stage-c-scale-20261026/`:
- `code/` version046f source, `train.jsonl`, `heldout.jsonl`, `CODE_COMMIT.txt`, `run-linear32-5000/` with snapshots1k through5k, full train log, `eval/probe60-*-c64.json`, `eval/probe60-{3000,5000}-c256.json`, `eval/heldout-{1000,2000,3000,4000,5000}.json`, `eval/formal17-{3000,5000}.jsonl`, summary and logs.

All GPU jobs completed. Original `main` and official training processes untouched.

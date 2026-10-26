# CID Stage C follow-up experiments — 2026-10-26

Base: published `fwerkor/CID-v1-0.4B-1018` (release), code `046f5bc`, RTX 5060 Laptop GPU, bf16. No changes were pushed to `main`. The experiment-only code lives in `code-preserve-exp/`. Original frozen CID backbone/control/T weights remain unchanged.

## Hypotheses

1. Unconstrained Display output residual shifts correct base-model predictions; selective teacher distribution preservation might protect them.
2. Training on teacher-forced clean answer prefixes does not match inference on imperfect self-produced Display states. Real runtime trace measures this exposure mismatch; a hybrid rollout-repair objective might help.

## Training controls

- Original balanced Stage C (control): 2000 optimizer steps, Nemotron v3 4000-row sample, rank32 Display-only low-rank residual, lr 1e-4, T rollout 2 steps unsupervised, 80% EOS boundary samples, 16-token lookahead, seed20261023.
- Preserve (candidate A): same configuration, plus teacher/student KL preservation (weight 1.0) on teacher-confident >=0.1 positions where the frozen original predicts the gold label or correctly reproduces an already visible clean prefix token. Teacher receives identical batch with Stage C disabled; both teacher and student use no T/control supervision. 2000 optimizer steps.
- Runtime repair (candidate B): same configuration, plus 40% samples from the frozen model's own Display after 2 runtime steps. Supervise changes to incorrect visible prefix and short continuation toward the reference response rather than assuming every training prefix is already correct. Preserve weight=0.0. 2000 optimizer steps.

## Hand-built 20-prompt paired evaluation

Each arm has identical 20 prompts/seeds; inference max_steps16, denoising_steps8, Display canvas64 or256. Nine prompts have short reference strings: substring inclusion is only a weak directional metric, not correctness validation.

| Canvas | Arm | Median output tokens | Converged /20 | Contains reference /9 | Responses >=32 tokens /20 |
|---|---|---:|---:|---:|---:|
| 64 | original CID | 3 | 19 | 3 | 0 |
| 64 | Stage C 2000 | 7 | 19 | 3 | 1 |
| 64 | selective KL | 5.5 | 18 | 5 | 1 |
| 64 | self-state repair | 6.5 | 18 | 4 | 4 |
| 256 | original CID | 3 | 20 | 2 | 0 |
| 256 | Stage C 2000 | 8 | 20 | 3 | 1 |
| 256 | selective KL | 6 | 18 | 2 | 0 |
| 256 | self-state repair | 7 | 17 | 3 | 4 |

## Formal CID validation subset

17 fixed examples, one each from 17 families, derived from 34 sampled validation-v4 cases from ise2. Original dataset task replay, 8 denoising steps, 20 model steps maximum, 25 s per task cap, canvas64, same hardware/software/data. Results from `benchmark-preserve-20261026/{baseline,balanced,preserve-w1,runtime-repair}-{summary.json,.jsonl}`.

| Arm | Evaluation exact-display /17 | Raw strict string exact /17 | Convergence /17 | Completed tool calls | Observation coverage |
|---|---:|---:|---:|---:|---:|
| original CID | 2 | 1 | 9 | 20 | 100% |
| Stage C 2000 | 3 | 2 | 8 | 20 | 100% |
| selective KL | 3 | 2 | 9 | 20 | 100% |
| self-state repair | 2 | 1 | 9 | 20 | 100% |

Important distinction: `evaluation.exact_display` ignores some innocuous answer-format differences, while `strict_exact_display` is a raw strict check; e.g. `Answer: 35` is accepted by evaluation against `Answer: 35.` but not by the raw check. Thus previous numbers 2/3/3/2 are the **evaluation display-correct counts**, not strict raw string exact counts. All four arms completed the same 20 tool calls across 17 cases; this does not prove semantic tool trajectories are identical or preservation holds in all tasks. 17 tasks is too small for release-level conclusions.

## Runtime trace diagnostic

10 base-model no-tool trajectories on ise2, canvas64, 16-step cap; trace saved in `ise2:/data/cid4b/stage-c-1018/preserve-20261026/rollout-states-base10.json`.

- 62 model states recorded.
- Visible answer span median 2 during first two model steps, vs median 4 for later steps.
- 38 of 146 changed token positions (~26%) were revisions to positions that were already visible in the prior state.
- Actual trajectories revise and retract answer content, whereas Stage C's clean-prefix objective assumes mostly correct historical content. The percentages are from 10 sampled base trajectories, not an estimate for all prompts/models.

## Interpretation

- Improvement in language expansion is real but separable from reliable factual accuracy.
- The selective KL regularizer can help weakly on some examples (5/9 vs3/9 reference substring at canvas64; restore 9/17 convergence on formal tasks) but it is not consistent across canvas sizes and partly undoes response expansion.
- Training with real self-generated Display states is an important exposure-mismatch experiment but the naive positionwise repair targets caused more long repeated outputs and did not improve formal CID task accuracy; reject this version.
- This is not enough to claim generic knowledge destruction or a single broken implementation. Baseline 0.4B formal validation remains weak; output-only LoRA cannot necessarily recover representations not present in frozen hidden states.
- Runtime Display revision/trajectory sampling is a more precise research target than simply increasing SFT steps, increasing residual strength, or blindly applying KL.
- Avoid mainline changes and release updates without larger independent paired benchmarks, explicit semantic correctness assessment, and staged regressions across tool behaviors.

## Artifacts

- local candidate code: `code-preserve-exp` (modified stage_c.py and cli.py; no push)
- trained adapters: `train-preserve-w1-2000/stage-c-latest.pt`, `train-runtime-repair-2000/stage-c-latest.pt`
- training logs: matching .log files
- E2E traces: `eval-{preserve-w1,runtime-repair}-c{64,256}-20.json`
- formal task results: `benchmark-preserve-20261026`
- validated existing tests: `11 passed` in experimental clone; separate real-GPU 3-step runtime-repair smoke succeeded.

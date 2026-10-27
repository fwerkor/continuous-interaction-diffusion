# CID Stage C research — 2026-10-24

## Environment and reproducibility

- Model: `fwerkor/CID-v1-0.4B-1018` (same frozen release for every arm)
- Code: `046f5bc467faac42940b7ed65f8195783b658283` on `main`
- GPU: RTX 5060 Laptop 8GB on mypc-debian
- Main Stage C settings: output-only residual rank 32, lr=1e-4, 2 frozen T rollout steps, EOS-boundary rollout probability=0.80, continuation window=16, seed=20261023, max-answer-tokens=384, bf16.
- Evaluation: `probe_extended.py`, 20 fixed prompts with per-prompt policy seed=1729+i, runtime max_steps=16, denoising_steps=8, no external tools. 9 prompts have a reference answer; the metric below is case-insensitive reference-substring inclusion (NOT a robust semantic accuracy score).
- Re-running the original 4,000-row experiment reproduced the previous balanced 2,000-step adapter checkpoint byte-for-byte (SHA-256 `e5bdc4db0fea88f36231b1e21d24b933ae52c10d2879d1409afbd47808e65da9`).

## Controlled arms

- Baseline CID: no Stage C residual.
- Balanced-2000: original Nemotron SFT v3 pilot data, first 2000 accepted optimizer steps, EOS-balanced Stage C, original prompts.
- Shortanswer-2000: 1052 data records with 40–420 answer characters and prompt ≤1100 chars; 2 epochs / 2000 accepted steps; other hyperparameters unchanged.
- Alignedprompt-2000: full source data but the fixed `System: You are a helpful and harmless assistant; You are not allowed to use any tools` header was stripped from exactly 2000/4000 records; other hyperparameters unchanged.
- Original-3461: original 4000 records, one full epoch; 3461 accepted optimizer steps (the rest filtered by token/context budget), same training recipe.
- Scale05: at inference only, multiply Stage C residual by 0.5; no retraining.

## 20-prompt evaluations; Display canvas = 64

| Arm | Median generated tokens | Converged/20 | Reference substring / 9 | <=5 tokens / 20 |
|---|---:|---:|---:|---:|
| Baseline CID | 3 | 19 | 3 | 14 |
| Balanced-2000 | 7 | 19 | 3 | 7 |
| Balanced-2000 ×0.5 | 4.5 | 20 | 4 | 11 |
| Shortanswer-2000 | 5 | 19 | 4 | 11 |
| Alignedprompt-2000 | 7 | 20 | 4 | 8 |
| Original-3461 | 5 | 20 | 5 | 12 |
| Original-3461 ×0.5 | 4 | 19 | 3 | 12 |

## Display canvas = 256 control

| Arm | Median generated tokens | Converged/20 | Reference substring / 9 |
|---|---:|---:|---:|
| Baseline | 3 | 20 | 2 |
| Balanced-2000 | 8 | 20 | 3 |
| Shortanswer-2000 | 4.5 | 19 | 3 |
| Alignedprompt-2000 | 7 | 19 | 3 |
| Original-3461 | 6 | 17 | 3 |

## Illustrative cases (canvas=64)

- Days in week: baseline `7`; balanced `There are 7 days in a week.` — improved linguistic form.
- RAM vs SSD: baseline `IAM`; balanced `RAM is a type of memory that is a` — topical expansion, still truncated.
- HTTP acronym: baseline `HTTP is the the HTTP protocol protocol.`; balanced `Hypertext Transfer Protocol (HTTP) stands for the HTTP Transfer Protocol.` — closer to correct.
- 12+19: baseline `21`; balanced `19, 26 is 12.`; balanced ×0.5 `31`; 3461-step `Here, 12 plus 19 is 30.`; 3461-step ×0.5 `Conclusion: 31` — factual outputs are sensitive and non-monotonic.
- Linux open source: 3461-step ×0.5 answers `no`, so residual scaling is not uniformly safe.
- Python/list-generation and short-story prompts still show repetition or very short output despite Stage C.

## Conclusions and limitations

1. Stage C increases natural-language continuation and can produce recognizably more complete sentences without altering original CID parameters.
2. The change is **not** yet a reliable gain in factual correctness or instruction following; the above 9-answer substring criterion is intentionally weak and 20 prompts are insufficient for release decisions.
3. Simply selecting short SFT answers does not solve short / degenerate CID outputs. Fixed-prompt normalization alone has no clear sizable benefit.
4. More optimization is not monotonic: from 2000 to 3461 steps, at 64-token canvas median answer length fell 7 → 5 while reference substring hits went 3/9 → 5/9; at 256-token canvas the convergence rate also fell.
5. Logit-residual scale trades off generation length and individual factual outputs. 0.5x is **not** a universal improvement.
6. Frozen control/convergence heads mean these tests do not establish preservation of end-to-end CID tool use, reasoning or all deployment benchmarks.
7. These are hypotheses for further controlled experiments, not evidence to change defaults or publish a new released Stage C checkpoint.

Recommended next validation: evaluate early-stopped checkpoints on a larger independent held-out conversational set plus formal CID validation/tool tasks; compare 2B/4B when training checkpoints and GPU availability permit, rather than over-tuning 0.4B using 20 hand-written examples.

Raw traces, data, checkpoints and training logs remain in this directory. No source-code changes were pushed.

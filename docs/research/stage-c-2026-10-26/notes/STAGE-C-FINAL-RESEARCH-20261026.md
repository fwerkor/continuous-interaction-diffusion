# CID Stage C: final research decision and proposed next-generation design
Date: 2026-10-26
Base model: CID-v1-0.4B-1018 (published frozen release), code main `046f5bc`. All new experimental code is in *isolated clones*, not merged. Original 2B/4B/other model trainings were not interrupted.

## Empirical question
Can CID v1 acquire natural-language fluency by ordinary SFT on a rank-32 additive Display logits residual, without damaging original tool/decision capabilities? What is the limiting mechanism, and which changes are justified?

## Experimental facts

Matched Stage C training arms each use 2000 optimization steps, same 4000-row Nemotron-SFT v3 pilot dataset, seed20261023, frozen CID release, 2 frozen unsupervised Thought steps, rank32 unless stated, lr1e-4, ~80% runtime-shaped prefix/EOS cases with 16-token lookahead, 20% full denoising. Prior arms were also evaluated as documented in `STAGE-C-RESEARCH-20261024.md` and `STAGE-C-RESEARCH-20261026.md`.

| Candidate (changes from balanced Stage C) | Median response tokens, canvas64 | Median response tokens, canvas256 | 64 converged/20 | 256 converged/20 | 256 responses >=32 tokens/20 | Official 17-task eval exact / convergence |
|---|---:|---:|---:|---:|---:|---|
| Baseline CID without Stage C | 3 | 3 | 19 | 20 | 0 | 2/17, 9/17 |
| Existing balanced Stage C, rank32 | 7 | 8 | 19 | 20 | 1 | 3/17, 8/17 |
| Selective confidence-gated KL preservation | 5.5 | 6 | 18 | 18 | 0 | 3/17, 9/17 |
| 40% actual runtime-state gold position-wise repair | 6.5 | 7 | 18 | 17 | 4 | 2/17, 9/17 |
| Local near-gold one-token prefix repair, 40% | 6 | 6.5 | 20 | 17 | 2 | 2/17, 9/17 |
| Single-token EOS lookahead instead of 16-token parallel future | 2 | 2 | 19 | 18 | 0 | not run, because language restoration failed |
| Higher-capacity rank128, other settings unchanged | 5 | 6 | 19 | 20 | 1 | 3/17, 9/17 |
| Live nonterminal EOS frontier 4x CE weight | 5.5 | 8 | 20 | 17 | 4 | 2/17, 8/17 |

Warning: 20 conversation prompts and 17 formal tasks are *small pilot sets*. Task `evaluation.exact_display` accepts some formatting variants and is not identical to raw strict string matching. All four earlier variants and the latest localrepair/rank128/frontier preserved 20 completed tool calls over these 17 formal tasks, with 100% observation coverage, but this does not establish comprehensive preservation of CID control/semantic decision ability.

Held-out SFT teacher-forced evaluation on 64 **unseen** training-pool examples (`Python random.Random(20261023).shuffle`, source examples indices >=3000 after shuffle; only first 2000 accepted for stage C training) gives:

| Arm | Mean token NLL (nats) | Mean supervised token argmax accuracy |
|---|---:|---:|
| Frozen baseline | 5.9473 | 0.2319 |
| Rank32 balanced | 5.6426 | 0.2415 |
| Rank128 balanced | 5.5252 | 0.2509 |
| Rank32 frontier weight4 | 5.6496 | 0.2400 |

The adapted head learns generalizable local predictions; it is not just a decoding trick. However, reducing teacher-forced token loss and scaling rank do *not* imply better full-sentence free-run generation. Longer answers are also not equivalent to greater factual correctness.

Additional causal probe: forcing at least 8 runtime steps increases median answer length for balanced Stage C on canvas64 from 7 to 8.5, but worsens long repetition (>=32 tokens 1/20 -> 3/20), while correct reference inclusions stay at 3/9. The early convergence head contributes to brevity, but simply lowering the stop gate is not an adequate fix.

## Root cause assessment

The evidence **rules out a simple EOS implementation defect, simply insufficient LoRA rank, or a single easily tunable residual scale** as sufficient explanations. It does *not* uniquely identify a root cause. The dominant architectural mismatch is:

1. CID A/B solves information gathering and decision making with dynamically evolving T/F and a revisable Display canvas; in ordinary generative SFT, Stage C is instead forced to map a prompt plus a frozen, often semantically unaligned T snapshot to an unrelated full natural-language completion.
2. The output-only low-rank residual can readily bias a few token logits and EOS growth, but cannot reconstruct task semantics missing in its frozen hidden states. When supervised against a new answer, it can fight the information already represented by CID instead of merely verbalizing it.
3. Clean gold prefixes vs actual self-generated, repeatedly revised Display trajectories are mismatched. 10 real base trajectories exhibit 38 previously visible token changes among 146 token edits (~26%). Naively repairing every self-generated position fails because many prefixes are *semantically incompatible* with teacher answers, resulting in ungrounded repetition.
4. Model control and exactness constraints remain important: precise tool/exact-output tasks should not be treated like conversational paraphrase tasks.

This is a diagnosis supported by several negative controls, **not a demonstrated theorem**. The published 0.4B checkpoint may also be intrinsically too weak; it is essential to evaluate 2B and 4B before concluding CID architecture is at fault.

## Recommended final design: grounded, post-decision Display rendering

Keep CID A/B and current runtime unchanged for cognitive/tool work. Enable Stage C only in a *separate final Display rendering pass* after the runtime has satisfied tool/freshness barriers and determined the answer. Avoid modifying frozen T/F, retrieval heads, convergence heads, tool actions or the decision-bearing original Display while tool/decision reasoning is ongoing.

- **Semantic anchor**: capture the original CID answer or authoritative structured facts/tool evidence (and exact-output requirements) in immutable output metadata. Treat semantic anchors as the source of truth rather than expecting a public SFT response to be the answer to an unrelated CID state.
- **Render-conditioned training**: create paired examples `(prompt, frozen CID final T/F and evidence, original answer anchor, factual-equivalent fluent rendering)`, with fluently rewritten targets checked for exact anchor preservation. On exact-format tasks, bypass Stage C; never let a language renderer change authoritative facts or a required exact value.
- **Reuse the existing Display-only rank32 residual**, activated during final rendering only, and existing diffusion decoder; no extra thought head, new cognitive structure, or separately supervised T required. Train with parallel future-token denoising and mixed proper EOS terminal/nonterminal examples. Prefer paired *CID-specific factual trajectories* over unrestricted ungrounded answer-only SFT.
- **Preserve by construction**: protect/copy source answer spans or provide a verified immutable semantic anchor; reject the rendered candidate if its anchor/evidence validation fails, falling back to original CID answer. Use an extraction/equality check appropriate to the task, never a mere fluency score. Optional local correction supervision only for suffix-compatible, aligned edits; never force non-matching self-generated prefixes to match a different reference index-by-index.
- **A/B task exactness independent of C**: core CID output/control remains the formal v1 target. Stage C is a presentation option for free-form conversational tasks, disabled for strict exact-output benchmarks unless format-preserving rendering is provably correct.
- **Training datasets**: source from CID's own trajectories plus verified grounded rewrites (a teacher model may propose rewrites, but reject factual divergence). A public chat SFT corpus may support style diversity but is not sufficient alone as the central objective.
- **Release gate**: paired 512-task CID validation with no changes to tool calls/observations, no loss of baseline semantic correctness, robust answer preservation, plus a large *independent* free-form chat test set judged for actual factual fluency, nonconvergence, repetition, and EOS accuracy. Validate both 2B and 4B, and fixed native 64/extended 256 Display capacities. The 0.4B pilot is too weak to justify production promotion alone.

This is the **recommended final solution to implement and validate next**, not a completed or verified renderer. The experiments completed here support the need for grounded answer conditioning and decision/rendering separation, but no result yet proves that this proposed two-pass scheme will restore high-quality language. Avoid presenting it as already solved.

## Current decision

- Preserve code on `main` unchanged. Its balanced Stage C remains optional and the best tested *language-length* baseline, not a production language restoration solution.
- Reject immediate merge of the KL, real-state correction, one-step EOS, local corruption, frontier weighting, and rank128 experimental branches: none wins consistently across both display capacities, correctness and convergence.
- Do not change the published model or official card on the basis of this pilot.
- Existing experimental code/checkpoints and logs in `code-{preserve,stepwise,localrepair,frontier}-exp`, `train-*-2000/`, `eval-*.json`, `benchmark-preserve-20261026/`. Nothing was pushed.

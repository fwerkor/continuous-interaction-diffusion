# Evidence package layout

The full raw evaluation JSON/JSONL, scripts, six experimental patches, logs and the study manifest are provided in the six independent tar.gz files under archive/ rather than as unpacked GitHub files. They can be extracted individually with tar -xzf. Local probes, 20k-scale trials, and the native-17 task traces are separate bundles. The four study notes and the 60-prompt summary CSV can be viewed directly on GitHub.

---

# Frozen 0.4B CID Stage C: experimental evidence archive

Recorded 24–26 October 2026. This is a **research archive, not a release or a production model update**. Original CID Stage A/B, tool-control heads and the runtime remained unchanged throughout these isolated Stage C trials.

## Experimental basis

- **Frozen released base model:** fwerkor/CID-v1-0.4B-1018.
- **Experiment code base:** 046f5bc467faac42940b7ed65f8195783b658283 (the contemporaneous source snapshot, not current main).
- **Matched pilot:** Nemotron-SFT-v3 4,000-row source sample, seed 20261023, rank-32 Display-only residual LoRA unless varied, learning rate 1e-4, two unsupervised frozen Thought steps, 80% continuation/EOS-boundary plus 20% full denoising. Most arms used 2,000 optimizer steps.
- **Evaluation:** fixed 20- and 60-prompt free-run comparisons at Display canvases 64 and 256; 64-case teacher-forced heldout; a *small* 17-task native tool/exact-output benchmark. Answers containing a reference substring are NOT equivalent to semantically correct answers.
- **Separate larger-data trial:** on ise2, 20k sampled instruction examples (19k candidate training rows, 670 prompt-disjoint heldout after overlap filtering) and linear rank-32 adapters at 1k–5k steps. Results are isolated under results/scale-20k; they are not matched GPU controls for the local experiments.

## What the experiments actually support

1. **Length is not correctness.** On 20 fixed prompts at canvas 64, balanced Stage C increased median response tokens from 3 to 7, while weak reference-substring inclusions stayed at 3/9. Outputs sometimes improved in wording but still failed arithmetic or other factual checks.
2. **Teacher-forced loss does not rank actual CID generation quality.** On 64 heldout examples, the rank-512 nonlinear hidden residual obtained NLL 5.3391 versus 5.6426 for linear rank-32. But 60-prompt free-run reference inclusion was *worse* for rank-512: 13/37 vs 18/37 at canvas 64 and 2/37 vs 12/37 at canvas 256, with native CID-task regressions.
3. **More steps or broader SFT did not monotonically help.** In the separate 20k-scale trial, teacher-forced heldout loss fell from 5.3119 at step 1k to 5.1124 at 5k. Free-run reference metrics, length, and the 17-task validation did not uniformly improve. No new checkpoint consistently dominates.
4. **Runtime-state mismatch is a plausible mechanism, not a proved root cause.** In a 10-trajectory trace, 38 of 146 token changes (~26%) revised previously visible positions. Naive positionwise repair against unrelated answers increased repetition and did not clearly improve native task exactness.
5. **Keep Stage C optional and reversible.** The existing balanced frozen-backbone linear rank-32 Display residual is the least disruptive exploratory baseline; this is *not* a verified restoration of fluent, semantically grounded generation. Separating final answer rendering from tool/decision cognition with verified semantic anchors is a proposal **not yet implemented or validated**.

### Interpreting the numbers

These tests are pilots: 9 reference-substring examples in the 20-prompt set; 37 in the 60-prompt set; only 17 formal native tasks; and 64 heldout teacher-forced examples. Native task exact-display scoring is more permissive than byte-exact matching. Convergence and successful tool observation counts do not prove semantic preservation. Many variants show non-monotonic tradeoffs across canvas capacities; do not select a new production checkpoint using NLL or median length alone.

### Directory guide

| Relative path | Contents |
| --- | --- |
| notes/ | Four contemporaneous experimental notes, including limitations and alternative algorithms |
| results/local-probes/ | All 40 original local free-run / heldout JSON comparisons |
| results/summary-60.csv | Recomputed, directional 60-prompt metrics derived from original JSON |
| results/formal-17/ | Nine experimental arms, each with native per-task traces and compact summary |
| results/scale-20k/eval/ | Original heldout, 60-prompt, and native 17-task evaluations from ise2 |
| data/cid-validation-{17,34}.jsonl | Curated formal evaluation inputs, for independent review |
| logs/local/ | Local training logs; work-machine path prefixes have been anonymized |
| scripts/local/ | Ad hoc probe scripts used to produce the local metrics |
| patches/ | Six isolated, *unmerged* experimental code diffs with the original base commit SHA |
| MANIFEST.json | Per-file SHA-256 and exact byte length after archiving |

The six patches are historical experiments (frontier, hidden, localrepair, nonlinear, preserve, stepwise), **not** a request to apply them to current main. The notes were written before this archive; their statements about what was pushed reflect the experiment date, not the current state.

### Exclusions and integrity

No raw Nemotron training/holdout text is redistributed. No model weights, Stage A checkpoint, adapter optimizer tensors, Python virtualenv, downloaded base-model snapshot, temporary cache, token or credentials are included. Therefore **this Git repository does not yet provide a durable backup of the excluded 11 GB Stage A checkpoint or Stage C weight files**; keep separate verified copies.

Original structured evaluations are retained, except workstation-specific absolute home paths are replaced by the marker <local-home>/ in 15 text files. MANIFEST.json hashes refer to the archived versions.

To reproduce, use the published frozen 0.4B checkpoint, check out CID commit 046f5bc..., reconstruct the stated dataset split using the described seeds, apply any patch only to a disposable matching-source clone, train its adapter, then run the included probe scripts. Some original hardware-dependent runs require Ascend/NVIDIA devices and the matching optional Python libraries. Full regeneration has **not** been repeated as part of this archive.

**Release gate:** extensive independent free-run semantic benchmarks, tool and exact-output no-regression suites, controlled testing on larger 2B/4B checkpoints, and successful validation of any proposed grounded post-decision renderer.

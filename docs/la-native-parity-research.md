# LA native-parity research (2026-11-01)

## Reproducible observation

The LFM2.5-Encoder-350M-Diffusion model card documents one-mask-per-forward,
highest-confidence unmasking. It bans all special tokens except EOS, uses
[Question]/[Answer] prompt formatting, and truncates text at [/Answer].
Its output protocol differs from CID preallocated EOS+revision decoding.

Ten English prompts were evaluated with identical original base weights
under both official sampling and CID preallocated sampling:

* Official (template): 10/10 nonempty; all 10 included [/Answer] boundary.
* Official (bare prompt): 10/10 nonempty; 9/10 included [/Answer].
* CID fixed 64-token EOS canvas, about eight tokens revealed per step:
  significant repetition, broken JSON/code/formatting even on untouched base.
* CID fixed 64-token EOS canvas, one token revealed per step, no revisions,
  special-token filtering: substantially more coherent English.
* Increasing reveals to 2, 4 or 8 degraded several quality examples.
  Revision fraction 0.25 was not the primary driver at one-token reveal.

The exact task logs were stored under
/workspace/cid-la-350m-20261101/native-decode-*.json and decoder-*.json.

## Teacher-parity distillation

The production v2 teacher path originally fed the **student**'s structural EOS
canvas to a native 350M teacher. This was inconsistent with the pretrained
teacher's original all-MASK answer canvas. The new opt-in
--teacher-native-mask-only path retains the template-encoded teacher prompt
but replaces any visible CID EOS in the teacher answer canvas with MASK.
The student still uses bare Prompt and supervised CID structural EOS.

The --full-mask-probability option samples fully masked answers more often,
addressing the mismatch between sparse teacher forcing and fully masked
runtime bootstrapping. Native teacher content-KL only supervises masked
answer positions; the structural EOS is supervised by student CE alone.

In the early experiment, teacher-native KL=1 with a 40% full-MASK curriculum
generated nonempty coherent English for all 10 decoding probes at step 1600,
where the previous 3000-step SFT had four empty answers under identical slow
CID decoder. This is an early result, not a clean causal claim: use the matched
no-KL control with exactly the same data/seed/mask schedule.

The teacher baseline itself returned 341 for 17*23 and invalid factorial
code under official greedy sampling. Neither should be counted as an
LA-specific regression.

## Deployment boundary

CID quality mode is **opt in**:
--display-max-reveals-per-step 1
--display-ban-non-eos-specials
--display-revision-fraction 0

Defaults are unchanged to preserve existing CID inference behavior and
throughput. One-token reveal can require far more forward passes than the
normal eight-step denoiser. Measure end-to-end tool calls, convergence, input
language compatibility, and inference latency before recommending it as a
production default.

A user-level native Prompt adapter for instruction-tuned bases is still a
viable zero-training control. It must be made consistent in Stage A/B training
and inference if pursued. For a Base model with no chat template, retain the
bare Prompt path.

No model has yet passed production-quality English, executable code, tool and
EOS acceptance gates; KL/no-KL matching requires additional checks.

## Matched English-format parity experiment (19 held-out prompts)

Same dataset (35,913 training records), random seed, exactly 2,400 optimizer
steps, 4x gradient accumulation, LR schedule, 40% all-MASK examples, CID EOS
target and official LFM2.5 confidence-greedy one-token sampler; only the
teacher native-format KL term differs:

| model | nonempty | native [/Answer] close |
| --- | ---: | ---: |
| unchanged 350M, native prompt | 19/19 | 19/19 |
| bare Prompt SFT, no KL | 19/19 | 4/19 |
| bare Prompt + native-format teacher KL (weight 1) | 19/19 | 17/19 |

The original 350M in the test also makes arithmetic mistakes, so exact task
accuracy cannot be judged from the presence of fluent paragraphs. These
observations establish a strong answer-format retention effect, not equality
of capabilities. The KD student is still verbose/repetitive on some questions.
A separate step 3200 KD snapshot also obtained 17/19 native closings, showing
the end-marker metric is not monotonically improved by more steps.

Follow-up ablations of a naive minimum-distance constraint between simultaneous
reveals did not reliably restore original generation at 4 or 8 tokens per pass.
That experimental code was discarded rather than promoted. More sophisticated
dependency-aware approaches remain a research direction.

Large teacher parity should be assessed on genuine hold-out instructions and
the full CID tool interaction suite, not on answer-boundary restoration alone.

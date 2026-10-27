# CID public benchmark scoring: task success vs. answer discovery

CID reports **two separate end-to-end task metrics** rather than treating
the appearance of a reference answer in the model's intermediate output as
successful task completion.

## Definitions

Let $D$ denote a fixed benchmark, $y_i$ its gold answer, and
$o_i$ the generated Display for example $i$.

**Task Accuracy (TA)** is the fraction of examples for which the model
provides a semantically correct, unambiguous answer to the task.
Equivalent surface realizations, accepted aliases, numeric forms,
mathematically equivalent values, and distinct code implementations are
acceptable when they genuinely answer the question. An explicit final
answer that contradicts an earlier correct mention takes precedence: that
trajectory is **not** successful under TA.

**Answer Discovery Rate (ADR)** is the fraction of examples for which
the generated Display contains an identifiable correct answer somewhere,
including CID's Known: region. A later incorrect final answer does not
erase successful discovery. ADR describes information retrieval or
intermediate answer identification, **not** successful user-facing
answering. In particular, ADR is not an alternate name for EM.

With fully semantically adjudicated examples and the definitions above,
TA should not exceed ADR. Both metrics count the same fixed set of
examples and use the same denominator. Report uncertainty and the number
of cases pending semantic adjudication.

## Dataset-specific procedure

| Dataset | Task Accuracy | Answer Discovery Rate |
| --- | --- | --- |
| ARC-Challenge, MMLU | Unambiguous correct option selection or semantically equivalent final option text | Correct option appears in output; reproducing every option does not suffice |
| GSM8K | Final numeric answer equality after normalization | Correct numeric answer appears as a standalone numeric value |
| MATH | Final mathematical equivalence; normalization is a conservative automatic subset | Correct mathematical answer is identified anywhere, respecting sign and expression boundaries |
| MBPP | Execution pass@1 using the published test harness | Same pass@1 unless independently validated as a correct implementation |
| HotpotQA, 2WikiMultiHopQA, MuSiQue | Exact normalized answer, accepted aliases, or semantically validated paraphrase provided as an unambiguous answer | Gold answer or accepted alias is found in the Display, including Known: |

Simple substrings are not sufficient to establish mathematical correctness:
`-2` is not `2`, `4` in `4-4` does not imply the expression equals
four, and the numerator in `\\frac{7}{6}` does not establish an answer
of seven. Likewise, an option list copied from the question does not
establish which option the model chose.

## Example

For a question whose reference answer is **Marlon Stöckinger**:

| Output | TA | ADR |
| --- | --- | --- |
| `Answer: Marlon Stöckinger` | 1 | 1 |
| `Known: Marlon Stöckinger — Answer: 50` | 0 | 1 |
| `Answer: 50` | 0 | 0 |

The middle output is evidence of answer *discovery*, not correct
completion. TA is not awarded simply because text overlaps a reference.

## Implementation and validation

`scripts/score_cid_dual_metrics.py` consumes completed shard outputs
from the existing public sweep and writes, under each dataset directory:

- `cid-dual-metrics.json`: counts, denominators, TA, ADR and review
  candidates.
- `cid-dual-metrics-audit.jsonl`: examples requiring semantic
  adjudication, preserving model output, reference and example ID.

The automatic TA number is a **conservative, deterministic preliminary
score**, not a completed semantic-judge evaluation. Paraphrases that are
not recognized automatically must be adjudicated against the *question*
and reference, ideally blinded to which model produced them, using a
documented rubric, manual review or a validated LLM judge with an audit
sample. No unreviewed case is silently promoted to correct. Numeric
equivalence and answer-string/alias detection do not alone establish
general semantic equivalence.

For reproducible comparisons, apply **the same scoring definitions,
benchmark examples, model runtime, token/step budget, judge prompt or
rubric and adjudication procedure** to every checkpoint. A new CID
metric must not be misreported as a dataset publisher's official score;
publish the canonical metrics separately when comparing to external
leaderboards.

### Reproduction

```bash
python scripts/score_cid_dual_metrics.py \
  results-1018-fp16/arc-challenge-test \
  results-1018-fp16/mmlu-test \
  results-1018-fp16/gsm8k-test \
  results-1018-fp16/math-test \
  results-1018-fp16/mbpp-test \
  results-1018-fp16/hotpotqa-distractor-validation \
  results-1018-fp16/2wikimultihopqa-dev \
  results-1018-fp16/musique-validation
```

### Paper-ready wording

> We report Task Accuracy (TA) and Answer Discovery Rate (ADR) as
> complementary end-to-end measures. TA credits a correctly resolved task
> only when the output provides an unambiguous semantically correct
> answer, whereas ADR credits successful identification of a gold answer
> anywhere in the generated Display, including intermediate cognitive
> regions. An answer identified during reasoning but contradicted by the
> final response is counted toward ADR, not TA. We evaluate both measures
> on identical examples and runtime budgets, retain each dataset's
> canonical scores for external comparison, and audit uncertain semantic
> matches separately.

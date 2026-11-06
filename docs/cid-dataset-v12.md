# CID-Dataset v1.2 — Language SFT

CID-Dataset v1.2 preserves the 422,230 original v1.1 tool and TCT trajectories
byte-for-byte and introduces a separately versioned `language_sft` split to
restore natural-language fluency during Stage A/B optimization.

## Corpus and splits

The release targets approximately 300,000 unique instruction-response examples.
The release manifest records exact train/validation counts, SHA256, source
revisions, category distribution, and rejected examples. The categories cover
conversation, knowledge, writing, summarization, coding, mathematics,
instruction following, and multilingual interactions. Category ceilings
prevent one domain from overwhelming the corpus while allowing the actual
distribution to follow the supply of eligible licensed data.

Sources include the pinned Tülu 3 SFT mixture (dataset-level ODC-BY; review
upstream notices), UltraChat 200k (MIT), and previously accepted v1.2
OASST1 (Apache-2.0) and GSM8K (MIT) examples. An additional
Apache-2.0 Aya Dataset supplement supplies multilingual coverage. Answers containing internal
Thought, TCT, or runtime transport markers are rejected.

All records use `cid-language-sft-v1` and supervise **Display only**.
There are no Thought/TCT labels, tool calls, or fabricated cognitive traces.
The expanded corpus deduplicates by normalized prompt, including across
training/validation. Validation examples take precedence when a previous
v1.2 training prompt collides with a held-out prompt.

## Building the expanded release

```sh
python scripts/expand_language_sft_v12.py \
  --existing /path/to/CID-Dataset-v1.2/components/language-sft \
  --output-dir /path/to/CID-Dataset-v1.2-expanded/components/language-sft \
  --target-total 300000 --seed 20261106

python scripts/complete_cid_dataset_v12_release.py \
  --parent-dir /path/to/CID-Dataset-v1.2 \
  --expanded-dir /path/to/CID-Dataset-v1.2-expanded
```

Building requires the `datasets`, `pyarrow`, and `huggingface_hub`
packages. Release metadata specifies source revisions for reproducibility.

## Stage A/B mixture

Add the same flags to a normal `cid train` or `cid train-full` command:

```sh
--language-sft-data /path/to/language-sft-train.jsonl \
--language-sft-token-fraction 0.20 \
--language-sft-max-prompt-tokens 1024
```

Each model uses its own tokenizer. The primary 80/20 ratio counts the exact
tokenized Prompt plus target Display (including the protocol's delimiter
tokens), summed over CID supervised transitions. For CID indexed trajectories,
the token accounting is cached by dataset/checkpoint identity, tokenizer
vocabulary, protocol, and window selection under
`~/.cache/cid/language-mix`. All distributed ranks reuse the same cache.

The reported ratio is exact for these reference token sequences. The
stochastic number of tokens masked and used by an individual diffusion
optimizer step can differ. For this reason the trainer also reports the
Display-reference token fraction separately.

A deterministic per-epoch sampler draws language examples from the eligible
corpus without replacement until the requested token budget is reached.
Epoch-specific seeds broaden coverage while preserving reproducible
checkpoint replay. At most two passes through the corpus are allowed; if the
dataset cannot supply the requested budget, training fails with a
descriptive error instead of silently repeating the same small set many times.

CID windows retain their original semantic-task importance weights. The
language-only windows get a common scalar weight calibrated so they receive
20% of the **nominal objective weight mass**. This stabilizes the relative
contribution when language responses are much longer than CID answers.
Nominal weight mass does not guarantee 20% of actual gradient magnitude.

Training uses the existing DDP/FSDP gradient accumulation and independent
mode buckets. Language loss is Display cross-entropy only; CID trajectories
continue to train all original losses. The Stage B learning-rate scheduler
accounts for each epoch's separately sampled language windows.

The mixed checkpoint fingerprint includes dataset SHA256, sampling version,
model, seed, rollout horizon, example cap, Prompt limit, Display limit, and
mixture ratio. A changed corpus or sampler configuration cannot silently
resume a previous mixed optimizer state.

The optional `cid train-display --language-sft` remains a separate
Display-only adaptation on a frozen checkpoint. It is not required for
Stage A/B mixing.

## Evaluation

Evaluate on the held-out language_sft validation set for fluency, instruction
following, long-form responses, and repeated output. Independently measure
CID tool correctness, TCT behavior, and runtime latency against v1.1.
A 20% token budget alone does not establish improved model quality; model
weights require independent training and evaluation.

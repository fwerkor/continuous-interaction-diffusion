# CID-Dataset v1.2 — Language SFT

The language_sft component extends CID-Dataset v1.1 without rewriting the
original 422,230 CID training trajectories or altering tool and TCT semantics.

## Corpus

48,000 sampled instruction/response pairs across conversation (20%),
knowledge (20%), writing (15%), summarization (10%), coding (10%),
mathematics (10%), instruction following (10%), and multilingual (5%).
Train: 46,560. Validation: 1,440. Samples are source-aware and deduplicated.

Upstream sources, pinned revisions, counts, licenses and data SHA256 hashes
are listed in components/language-sft/language-sft-v1.2.manifest.json.
Sources: Tülu 3 SFT mixture (dataset-level ODC-BY, observe upstream notices),
UltraChat 200k (MIT), OASST1 (Apache-2.0), GSM8K (MIT).

Each example explicitly enables Display supervision and disables
Thought, TCT, and tools. No TCT labels, tool calls, or event traces are
invented for language-only data. Schema validation rejects them.

## Building

Use scripts/prepare_language_sft_v12.py to sample source data, then
scripts/finalize_language_sft_v12.py to add natural-language mathematical
solutions and construct independent validation split.

Python dependency requirements: datasets, pyarrow, huggingface_hub.

## Training

CID supports an isolated Display-only pass on a frozen checkpoint:

    cid train-display --language-sft \
      --data path/to/language-sft-train.jsonl \
      --checkpoint path/to/cid-release \
      --checkpoint-kind release \
      --output-dir path/to/display-adapter

This updates the Display output adapter only and leaves TCT, Thought, tools,
convergence, and the backbone frozen. It does not supervise or train TCT.

Stage A and B now support training the same CID adapter and backbone with
an 80/20 language mixture. Both commands accept:

    --language-sft-data path/to/language-sft-train.jsonl
    --language-sft-token-fraction 0.20
    --language-sft-max-prompt-tokens 1024

For example, append these flags to an existing `cid train` or
`cid train-full` command. Other CID v1.1 flags and rollout logic remain
unchanged. The language-only examples create empty TCT inputs (no teacher
cognitive state) and supervise only Display CE. They do not train
Thought/TCT/tool/convergence losses; the normal CID windows continue to train
all their objectives.

The mixture uses the shared, deterministic v5 sharding and optimizer
accumulation logic. Language and CID examples have distinct buckets and never
share a physical micro-batch, so each rank runs the same loss path in each
distributed step. Checkpoints include the CID manifest checksum, language data
checksum, and mixture configuration in their combined dataset identity.
The original CID-only checkpoint identity remains backward compatible.
Changing the language data or its sampling fraction requires a new optimizer
run, while resuming with the same contract replays the same windows.

The current mixture target is **estimated tokens**: original CID rollout
windows have compact prompt-plus-Display character-length indexes (estimated
at four characters/token, multiplied by supervised rollout offsets);
language_sft records are validated with the model tokenizer but budgeted using
the same character conversion. The reported ratio is therefore a reproducible
approximation, not an exact measured tokenizer-token fraction. This avoids
re-tokenizing the full multi-gigabyte CID dataset on every FSDP rank.
The Stage B learning-rate schedule includes the additional language windows.
Use the held-out language_sft validation JSONL for separate fluency metrics;
existing CID runtime validation remains separate and comparable to v1.1.

For native LFM protocols, full language responses train the native Display
reply sequence. The loss does not fabricate the short-answer auxiliary target
used for ordinary CID v1.1 tool trajectories.

For legacy CID v1.1 factual short answers, the native model falls back
to factual answer-span supervision; it does not invent full language
targets or refuse to train without metadata.native_full_display.

No model-quality improvements are claimed without independent end-to-end
tool correctness, Display fluency, and repetition benchmarks.

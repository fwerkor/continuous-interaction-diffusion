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

The recommended future Stage A/B mixing target is 80% original CID tokens
and 20% language_sft tokens. This mixing is NOT automatically integrated
into Stage A/B in the current code; train-display remains a separate pass.

For legacy CID v1.1 factual short answers, the native model falls back
to factual answer-span supervision; it does not invent full language
targets or refuse to train without metadata.native_full_display.

No model-quality improvements are claimed without independent end-to-end
tool correctness, Display fluency, and repetition benchmarks.

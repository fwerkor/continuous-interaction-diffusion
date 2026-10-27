# Stage C: Display-only language adaptation

Stage C is an **optional** low-rank output residual trained using public conversational
answers. It starts from a completed CID Stage B checkpoint and **does not train**
the backbone, TCT representations, tool-use/control heads, need routing, or
convergence heads. There are **no thought labels, artificial T targets, or
reasoning-trace losses**. The original CID runtime and all original model
parameters remain unchanged.

## Setup and public data

Install the optional dependencies:

```bash
pip install -e '.[stage-c]'
```

The default data source is
[`nvidia/Nemotron-SFT-Instruction-Following-Chat-v4`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4)
(2026). Review its OpenMDW-1.1 terms and source attribution requirements before
redistribution or production use. The two selected dataset splits target chat
quality and response-length control. Download only a small streamed subset:

```bash
cid prepare-stage-c-data --output data/stage-c-sft.jsonl --max-per-split 20000
```

The exporter observes the dataset's `metadata.train_turns_mask` and discards
`reasoning_content`; prior turns are retained as conversation context, never
as thought supervision. Existing JSONL with `{"prompt": "...", "answer": "..."}`
or `messages` plus `metadata.train_turns_mask` is also accepted.

## Train an adapter

For an **unsharded** Stage B `.pt` checkpoint:

```bash
cid train-display \
  --checkpoint /path/to/stage-b-epoch-0003.pt --checkpoint-kind stage-b \
  --model /path/to/base-backbone \
  --data data/stage-c-sft.jsonl --output-dir runs/stage-c \
  --rank 32 --learning-rate 1e-4 --thought-steps 2 --dtype bf16
```

For a **unified HF CID release**, use `--checkpoint /path/to/cid-release
--checkpoint-kind release` instead (and omit `--model`). Sharded FSDP Stage B
checkpoints must first be consolidated/exported to a supported unified release.
The original checkpoint is only read, never overwritten.

The frozen CID policy executes up to `--thought-steps` updates per prompt through
the standard CID runtime to obtain an unsupervised T state (no tool sources for
the public chat data). That T state conditions a standard masked Display
Diffusion cross-entropy step. Only the two fp32 low-rank matrices receive
gradients. You may set `--thought-steps 0` to ablate T conditioning.

**EOS-boundary training (revised):** Unlike the original full-answer-only
denoising, the default `--rollout-probability 0.75` trains on genuine
runtime-shaped Display states: a partly masked prefix, a current EOS, and a
masked latent tail. The current EOS must predict the next answer token;
future tokens are supervised in a bounded `--continuation-tokens 16` window.
A substantial share of prefix samples land at, or near, the **true answer
ending** and explicitly supervise keeping EOS there; this counters unlimited
expansion without inventing EOS labels at arbitrary truncation boundaries. Tiny prefixes,
including the exact `[MASK, EOS, MASK...]` initial condition, are oversampled.
The remaining 25% of batches retain full-answer diffusion denoising. This
addresses a specific training/inference mismatch but does not change CID's
convergence control or guarantee fluent replies.

A short experimental run can use `--max-steps 1000` with
`--checkpoint-every 100`. The dataset still needs a held-out validation
split; Stage C loss alone is not evidence of improved natural language.

The trainer skips samples whose answer (including EOS) or combined prompt,
TCT and Display exceed model capacity. `--max-answer-tokens` defaults to 512;
the Display canvas expands to fit targets up to the configured maximum. For
answers longer than the checkpoint's usual runtime canvas, increase the
**benchmark/inference** `--display-canvas-tokens` accordingly; the training
script does not modify runtime canvas defaults. The implementation trains one
example per optimizer step to keep the initial Stage C pathway simple rather
than claiming high-throughput distributed training.

Output files:

- `runs/stage-c/stage-c-epoch-0001.pt` — standalone Display residual checkpoint.
- `runs/stage-c/stage-c-latest.pt` — last completed / periodic checkpoint.

Stage C **does not** currently resume optimizer/data-cursor state from these
files. Periodic saves preserve adapter weights only. Use a new output directory
for each independent run.

## Inference and ablation

Use the **same base checkpoint** as Stage C training:

```bash
cid benchmark --data data/validation.jsonl \
  --checkpoint /path/to/cid-release --checkpoint-kind release \
  --stage-c-adapter runs/stage-c/stage-c-latest.pt \
  --display-canvas-tokens 512 \
  --output runs/with-stage-c.jsonl --summary-output runs/with-stage-c-summary.json
```

For a control, repeat the command without `--stage-c-adapter`. The output-only
residual adds to the original LM-head logits, without replacing/merging the
original LM head (important if input/output embeddings are tied). Unloading or
disabling the residual restores the **exact original model output** for the
same input. CID control heads are unchanged for identical hidden states, but
**end-to-end** traces may change once different Display tokens feed subsequent
steps. Validate ordinary conversation quality, length control, task accuracy,
tool behavior and convergence before enabling this adapter by default.

Tests: `pytest tests/test_stage_c.py` and `pytest`.

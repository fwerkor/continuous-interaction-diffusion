#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/cid-v1-7b-a1b}"
MODEL="${MODEL:-/workspace/models/LLaDA-MoE-7B-A1B-Base}"
DATA="${DATA:-/workspace/cid-moe-data/release/training-trajectories.jsonl}"
VALIDATION_DATA="${VALIDATION_DATA:-}"
PYTHON_ENV="${PYTHON_ENV:-/workspace/cid-small-work/.venv}"
REPO="${REPO:-/workspace/cid-moe-work}"

STAGE_A="$ROOT/stage-a"
STAGE_B="$ROOT/stage-b"
LOG_DIR="$ROOT/logs"
mkdir -p "$STAGE_A" "$STAGE_B" "$LOG_DIR"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "missing model checkpoint: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "missing training data: $DATA" >&2
  exit 2
fi
if [[ -n "$VALIDATION_DATA" && ! -f "$VALIDATION_DATA" ]]; then
  echo "missing validation data: $VALIDATION_DATA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

TORCHRUN="$PYTHON_ENV/bin/torchrun"

stage_a_args=(
  --model "$MODEL"
  --data "$DATA"
  --output-dir "$STAGE_A"
  --device auto
  --dtype bf16
  --epochs 3
  --learning-rate 1e-4
  --weight-decay 0.01
  --micro-batch-size 1
  --gradient-accumulation-steps 24
  --max-grad-norm 1.0
  --rollout-horizon 3
  --semantic-pooling order-aware-v2
  --teacher-forcing-epochs 1
  --rollout-ramp-epochs 2
  --thought-capacity 128
  --max-display-tokens 1536
  --display-canvas-tokens 64
  --log-every-steps 25
  --checkpoint-every-steps 1000
)
if [[ -n "$VALIDATION_DATA" ]]; then
  stage_a_args+=(--validation-data "$VALIDATION_DATA")
fi

if [[ -f "$STAGE_A/stage-a-latest.pt" ]]; then
  stage_a_args+=(--resume "$STAGE_A/stage-a-latest.pt")
fi

printf 'Starting CID-v1-7B-A1B Stage A\n' | tee -a "$LOG_DIR/pipeline.log"
"$TORCHRUN" --standalone --nproc-per-node=4 -m cid.cli train "${stage_a_args[@]}" \
  2>&1 | tee -a "$LOG_DIR/stage-a.log"

if [[ ! -f "$STAGE_A/stage-a-latest.pt" ]]; then
  echo "Stage A completed without stage-a-latest.pt" >&2
  exit 3
fi

printf 'Starting CID-v1-7B-A1B Stage B\n' | tee -a "$LOG_DIR/pipeline.log"
stage_b_args=(
  --model "$MODEL"
  --data "$DATA"
  --output-dir "$STAGE_B"
  --dtype bf16
  --epochs 1
  --learning-rate 1e-5
  --backbone-lr-scale 0.5
  --weight-decay 0.01
  --micro-batch-size 1
  --target-global-batch-size 32
  --max-grad-norm 1.0
  --warmup-ratio 0.03
  --min-learning-rate-ratio 0.1
  --rollout-horizon 3
  --semantic-pooling order-aware-v2
  --teacher-forcing-epochs 0
  --rollout-ramp-epochs 0
  --thought-capacity 128
  --max-display-tokens 1536
  --display-canvas-tokens 64
  --log-every-steps 25
  --checkpoint-every-steps 1000
  --fsdp-cpu-offload
)
if [[ -n "$VALIDATION_DATA" ]]; then
  stage_b_args+=(--validation-data "$VALIDATION_DATA")
fi

if [[ -d "$STAGE_B/stage-b-latest" ]]; then
  stage_b_args+=(--resume "$STAGE_B/stage-b-latest")
else
  stage_b_args+=(--init-cid-checkpoint "$STAGE_A/stage-a-latest.pt")
fi

"$TORCHRUN" --standalone --nproc-per-node=4 -m cid.cli train-full "${stage_b_args[@]}" \
  2>&1 | tee -a "$LOG_DIR/stage-b.log"

printf 'CID-v1-7B-A1B two-stage training completed\n' | tee -a "$LOG_DIR/pipeline.log"

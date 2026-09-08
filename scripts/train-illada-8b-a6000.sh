#!/usr/bin/env bash
# Throughput-oriented, memory-safe production launcher for CID iLLaDA-8B on
# one 8x RTX A6000 48 GiB node.
#
# Required environment:
#   DATA=/path/to/training-trajectories.jsonl
#   VALIDATION_DATA=/path/to/validation.jsonl
#   MODEL=/path/to/iLLaDA-8B-Base
#   RUN_ROOT=/path/to/run
#
# Usage:
#   scripts/train-illada-8b-a6000.sh stage-a
#   scripts/train-illada-8b-a6000.sh stage-b1
#   scripts/train-illada-8b-a6000.sh stage-b2
#   scripts/train-illada-8b-a6000.sh all
#
# Stage B2 intentionally resumes a one-epoch B1 checkpoint. The saved one-epoch
# scheduler is therefore already at its LR floor, making B2 a low-LR on-policy
# stabilization pass rather than stretching the original decay across two epochs.
set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in
  stage-a|stage-b1|stage-b2|all) ;;
  *)
    echo "usage: $0 {stage-a|stage-b1|stage-b2|all}" >&2
    exit 2
    ;;
esac

: "${DATA:?set DATA to the CID training JSONL}"
: "${VALIDATION_DATA:?set VALIDATION_DATA to the held-out validation JSONL}"
: "${MODEL:?set MODEL to the local iLLaDA-8B-Base directory}"
: "${RUN_ROOT:?set RUN_ROOT to the output run directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "$NPROC_PER_NODE" != "8" ]]; then
  echo "this production preset requires NPROC_PER_NODE=8" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Reduces fragmentation when FSDP repeatedly materializes full wrapped layers.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Fail collectives instead of silently hanging if a rank dies.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

STAGE_A_DIR="${STAGE_A_DIR:-$RUN_ROOT/stage-a}"
STAGE_B_DIR="${STAGE_B_DIR:-$RUN_ROOT/stage-b}"
STAGE_A_CHECKPOINT="${STAGE_A_CHECKPOINT:-$STAGE_A_DIR/stage-a-epoch-0003.pt}"
STAGE_B1_CHECKPOINT="${STAGE_B1_CHECKPOINT:-$STAGE_B_DIR/stage-b-epoch-0001}"

run_stage_a() {
  mkdir -p "$STAGE_A_DIR"
  local resume_args=()
  if [[ -n "${STAGE_A_RESUME:-}" ]]; then
    resume_args=(--resume "$STAGE_A_RESUME")
  fi
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m cid.cli train \
    --data "$DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$STAGE_A_DIR" \
    --model "$MODEL" \
    "${resume_args[@]}" \
    --device cuda \
    --dtype bf16 \
    --epochs 3 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --micro-batch-size 1 \
    --physical-micro-batch-size 1 \
    --target-global-batch-size 96 \
    --gradient-checkpointing \
    --max-grad-norm 1.0 \
    --rollout-horizon 3 \
    --teacher-forcing-epochs 1 \
    --rollout-ramp-epochs 2 \
    --semantic-pooling order-aware-v2 \
    --thought-capacity 128 \
    --max-display-tokens 1536 \
    --display-canvas-tokens 64 \
    --log-every-steps 100 \
    --checkpoint-every-steps 5000
}

run_stage_b1() {
  [[ -f "$STAGE_A_CHECKPOINT" ]] || {
    echo "missing Stage A checkpoint: $STAGE_A_CHECKPOINT" >&2
    exit 2
  }
  mkdir -p "$STAGE_B_DIR"
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m cid.cli train-full \
    --data "$DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$STAGE_B_DIR" \
    --model "$MODEL" \
    --init-cid-checkpoint "$STAGE_A_CHECKPOINT" \
    --device cuda \
    --dtype bf16 \
    --epochs 1 \
    --learning-rate 1e-5 \
    --backbone-lr-scale 0.5 \
    --weight-decay 0.01 \
    --micro-batch-size 1 \
    --target-global-batch-size 32 \
    --mlp-chunk-size 512 \
    --norm-chunk-size 1024 \
    --gradient-checkpointing \
    --warmup-ratio 0.03 \
    --min-learning-rate-ratio 0.1 \
    --max-grad-norm 1.0 \
    --rollout-horizon 3 \
    --teacher-forcing-epochs 0 \
    --rollout-ramp-epochs 0 \
    --semantic-pooling order-aware-v2 \
    --thought-capacity 128 \
    --max-display-tokens 1536 \
    --display-canvas-tokens 64 \
    --log-every-steps 100 \
    --checkpoint-every-steps 2500
}

run_stage_b2() {
  [[ -d "$STAGE_B1_CHECKPOINT" ]] || {
    echo "missing Stage B1 checkpoint: $STAGE_B1_CHECKPOINT" >&2
    exit 2
  }
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m cid.cli train-full \
    --data "$DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$STAGE_B_DIR" \
    --model "$MODEL" \
    --resume "$STAGE_B1_CHECKPOINT" \
    --device cuda \
    --dtype bf16 \
    --epochs 2 \
    --learning-rate 1e-5 \
    --backbone-lr-scale 0.5 \
    --weight-decay 0.01 \
    --micro-batch-size 1 \
    --target-global-batch-size 32 \
    --mlp-chunk-size 512 \
    --norm-chunk-size 1024 \
    --gradient-checkpointing \
    --warmup-ratio 0.03 \
    --min-learning-rate-ratio 0.1 \
    --max-grad-norm 1.0 \
    --rollout-horizon 3 \
    --teacher-forcing-epochs 0 \
    --rollout-ramp-epochs 0 \
    --semantic-pooling order-aware-v2 \
    --thought-capacity 128 \
    --max-display-tokens 1536 \
    --display-canvas-tokens 64 \
    --log-every-steps 100 \
    --checkpoint-every-steps 2500
}

case "$MODE" in
  stage-a) run_stage_a ;;
  stage-b1) run_stage_b1 ;;
  stage-b2) run_stage_b2 ;;
  all)
    run_stage_a
    run_stage_b1
    run_stage_b2
    ;;
esac

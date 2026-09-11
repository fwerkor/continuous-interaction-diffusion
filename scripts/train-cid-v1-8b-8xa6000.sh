#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in
  stage-a|stage-b|all) ;;
  *)
    echo "usage: $0 {stage-a|stage-b|all}" >&2
    exit 2
    ;;
esac

: "${DATA:?set DATA to the CID training JSONL}"
: "${VALIDATION_DATA:?set VALIDATION_DATA to the held-out validation JSONL}"
: "${MODEL:?set MODEL to the local GSAI-ML/iLLaDA-8B-Base directory}"
: "${RUN_ROOT:?set RUN_ROOT to the output run directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$NPROC_PER_NODE" != "8" ]]; then
  echo "CID 8B production preset requires NPROC_PER_NODE=8" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#_cuda_devices[@]}" -ne 8 ]]; then
  echo "CID 8B production preset requires exactly 8 CUDA devices; got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

STAGE_A_DIR="${STAGE_A_DIR:-$RUN_ROOT/stage-a}"
STAGE_B_DIR="${STAGE_B_DIR:-$RUN_ROOT/stage-b}"
STAGE_A_FINAL="${STAGE_A_FINAL:-$STAGE_A_DIR/stage-a-epoch-0003.pt}"
STAGE_B_FINAL="${STAGE_B_FINAL:-$STAGE_B_DIR/stage-b-epoch-0001}"
STAGE_A_LATEST="${STAGE_A_LATEST:-$STAGE_A_DIR/stage-a-latest.pt}"
STAGE_B_LATEST="${STAGE_B_LATEST:-$STAGE_B_DIR/stage-b-latest}"

for path in "$DATA" "$VALIDATION_DATA"; do
  [[ -f "$path" ]] || { echo "missing file: $path" >&2; exit 2; }
done
[[ -d "$MODEL" && -f "$MODEL/config.json" ]] || {
  echo "invalid local iLLaDA model directory: $MODEL" >&2
  exit 2
}
mkdir -p "$RUN_ROOT" "$STAGE_A_DIR" "$STAGE_B_DIR"

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_stage_a() {
  if [[ -f "$STAGE_A_FINAL" ]]; then
    echo "Stage A already complete: $STAGE_A_FINAL"
    return 0
  fi

  local resume_args=()
  if [[ -n "${STAGE_A_RESUME:-}" ]]; then
    resume_args=(--resume "$STAGE_A_RESUME")
  elif [[ -e "$STAGE_A_LATEST" ]]; then
    resume_args=(--resume "$STAGE_A_LATEST")
    echo "Stage A auto-resume: $STAGE_A_LATEST"
  fi

  run_cmd "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m cid.cli train \
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
    --checkpoint-every-steps 1000
}

run_stage_b() {
  if [[ -d "$STAGE_B_FINAL" ]]; then
    echo "Stage B already complete: $STAGE_B_FINAL"
    return 0
  fi

  local start_args=()
  if [[ -n "${STAGE_B_RESUME:-}" ]]; then
    start_args=(--resume "$STAGE_B_RESUME")
  elif [[ -e "$STAGE_B_LATEST" ]]; then
    start_args=(--resume "$STAGE_B_LATEST")
    echo "Stage B auto-resume: $STAGE_B_LATEST"
  else
    if [[ ! -f "$STAGE_A_FINAL" && "$DRY_RUN" != "1" ]]; then
      echo "missing completed Stage A checkpoint: $STAGE_A_FINAL" >&2
      exit 2
    fi
    start_args=(--init-cid-checkpoint "$STAGE_A_FINAL")
  fi

  run_cmd "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m cid.cli train-full \
    --data "$DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$STAGE_B_DIR" \
    --model "$MODEL" \
    "${start_args[@]}" \
    --device cuda \
    --dtype bf16 \
    --epochs 1 \
    --learning-rate 2e-5 \
    --backbone-lr-scale 0.25 \
    --weight-decay 0.01 \
    --micro-batch-size 1 \
    --target-global-batch-size 32 \
    --mlp-chunk-size 512 \
    --norm-chunk-size 1024 \
    --gradient-checkpointing \
    --no-fsdp-cpu-offload \
    --lr-schedule wsd-linear \
    --warmup-ratio 0.01 \
    --wsd-decay-ratio 0.10 \
    --min-learning-rate-ratio 0.10 \
    --max-grad-norm 1.0 \
    --rollout-horizon 3 \
    --teacher-forcing-epochs 0 \
    --rollout-ramp-epochs 0 \
    --semantic-pooling order-aware-v2 \
    --thought-capacity 128 \
    --max-display-tokens 1536 \
    --display-canvas-tokens 64 \
    --log-every-steps 100 \
    --checkpoint-every-steps 1000
}

case "$MODE" in
  stage-a) run_stage_a ;;
  stage-b) run_stage_b ;;
  all)
    run_stage_a
    run_stage_b
    ;;
esac

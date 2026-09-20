#!/usr/bin/env bash
set -Eeuo pipefail

: "${DATA:?set DATA to the CID training JSONL}"
: "${VALIDATION_DATA:?set VALIDATION_DATA to the held-out validation JSONL}"
: "${MODEL:?set MODEL to the local GSAI-ML/iLLaDA-8B-Base directory}"
: "${RUN_ROOT:?set RUN_ROOT to the Stage A output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_MAIN="${REQUIRE_MAIN:-1}"

if [[ "$NPROC_PER_NODE" != "4" ]]; then
  echo "CID 8B A800 preset requires NPROC_PER_NODE=4" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#_cuda_devices[@]}" -ne 4 ]]; then
  echo "CID 8B A800 preset requires exactly 4 CUDA devices; got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

if [[ "$REQUIRE_MAIN" == "1" && "$(git -C "$REPO_ROOT" branch --show-current)" != "main" ]]; then
  echo "CID 8B production preset requires the main branch" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
from importlib.metadata import version

import cid_engine

parts = tuple(int(part) for part in version("cid-engine").split(".")[:3])
if parts < (0, 6, 0):
    raise SystemExit(f"cid-engine>=0.6.0 required, found {cid_engine.__version__}")
if not cid_engine.CUDA_BACKEND_BUILT:
    raise SystemExit("cid-engine CUDA backend is required")
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

for path in "$MODEL/config.json" "$DATA" "$VALIDATION_DATA"; do
  [[ -e "$path" ]] || { echo "missing $path" >&2; exit 2; }
done
mkdir -p "$RUN_ROOT"

LATEST="$RUN_ROOT/stage-a-latest.pt"
FINAL="$RUN_ROOT/stage-a-epoch-0003.pt"
resume_args=()
if [[ -f "$LATEST" ]]; then
  shards_ok=1
  for rank in 0 1 2 3; do
    shard="$RUN_ROOT/stage-a-latest.optimizer-rank-$(printf '%04d' "$rank").pt"
    [[ -f "$shard" ]] || shards_ok=0
  done
  if (( shards_ok )); then
    resume_args=(--resume "$LATEST")
  else
    echo "latest checkpoint exists but optimizer shards are incomplete; refusing partial resume" >&2
    exit 2
  fi
fi

if [[ -f "$FINAL" ]]; then
  echo "Stage A already complete: $FINAL"
  exit 0
fi

cmd=(
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 -m cid.cli train
  --data "$DATA"
  --validation-data "$VALIDATION_DATA"
  --output-dir "$RUN_ROOT"
  --model "$MODEL"
  "${resume_args[@]}"
  --device cuda
  --dtype bf16
  --epochs 3
  --learning-rate 1e-4
  --weight-decay 0.01
  --micro-batch-size 1
  --physical-micro-batch-size 1
  --target-global-batch-size 96
  --zero-redundancy-optimizer
  --cpu-gradient-stash-threshold-tokens 512
  --mlp-chunk-size 64
  --norm-chunk-size 128
  --gradient-checkpointing
  --max-grad-norm 1.0
  --rollout-horizon 3
  --teacher-forcing-epochs 1
  --rollout-ramp-epochs 2
  --semantic-pooling order-aware-v2
  --thought-capacity 128
  --max-display-tokens 1536
  --display-canvas-tokens 64
  --log-every-steps 100
  --checkpoint-every-steps 200
)

printf '+ '
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then
  exec "${cmd[@]}"
fi

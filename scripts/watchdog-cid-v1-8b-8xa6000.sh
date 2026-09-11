#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${LAUNCHER:-$SCRIPT_DIR/train-cid-v1-8b-8xa6000.sh}"
: "${RUN_ROOT:?set RUN_ROOT to the CID 8B run directory}"

RESTART_DELAY_S="${RESTART_DELAY_S:-60}"
GPU_WAIT_S="${GPU_WAIT_S:-60}"
DISK_WAIT_S="${DISK_WAIT_S:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-350}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
WATCHDOG_LOG="${WATCHDOG_LOG:-$RUN_ROOT/watchdog.log}"
STATUS_FILE="${STATUS_FILE:-$RUN_ROOT/watchdog-status.json}"
LOCK_FILE="${LOCK_FILE:-$RUN_ROOT/.watchdog.lock}"

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another CID 8B watchdog already holds $LOCK_FILE" >&2
  exit 3
fi

write_status() {
  local state="$1"
  local detail="${2:-}"
  local restarts="${3:-0}"
  python - "$STATUS_FILE" "$state" "$detail" "$restarts" <<'PY'
import json, os, sys, tempfile, time
path, state, detail, restarts = sys.argv[1:]
payload = {
    "state": state,
    "detail": detail,
    "restarts": int(restarts),
    "pid": os.getppid(),
    "updated_unix_s": time.time(),
}
dirname = os.path.dirname(path) or "."
fd, tmp = tempfile.mkstemp(prefix=".watchdog-status-", dir=dirname, text=True)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCHDOG_LOG"
}

available_gpu_count() {
  nvidia-smi -L 2>/dev/null | wc -l
}

compute_pid_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 ~ /^[0-9]+$/ {count++} END {print count+0}'
}

wait_for_disk_space() {
  while true; do
    local free_kb free_gb
    free_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
    free_gb=$((free_kb / 1024 / 1024))
    if [[ "$free_gb" -ge "$MIN_FREE_GB" ]]; then
      return 0
    fi
    write_status "waiting_for_disk" "${free_gb}GB free; need ${MIN_FREE_GB}GB" "$restarts"
    log "only ${free_gb}GB free at $RUN_ROOT; waiting for at least ${MIN_FREE_GB}GB"
    sleep "$DISK_WAIT_S"
  done
}

wait_for_eight_idle_gpus() {
  while true; do
    local gpu_count active
    gpu_count="$(available_gpu_count)"
    if [[ "$gpu_count" -lt 8 ]]; then
      write_status "blocked" "only $gpu_count GPUs visible; need 8" "$restarts"
      log "only $gpu_count GPUs visible; waiting for 8"
      sleep "$GPU_WAIT_S"
      continue
    fi
    active="$(compute_pid_count)"
    if [[ "$active" -eq 0 ]]; then
      return 0
    fi
    write_status "waiting_for_gpus" "$active compute processes still active" "$restarts"
    log "$active GPU compute processes are active; waiting for all 8 GPUs to become idle"
    sleep "$GPU_WAIT_S"
  done
}

stop_requested=0
child_pid=""
terminate_child() {
  stop_requested=1
  write_status "stopping" "signal received" "$restarts"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    log "forwarding termination to training process group $child_pid"
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  write_status "stopped" "signal received" "$restarts"
  exit 130
}
trap terminate_child INT TERM

[[ -x "$LAUNCHER" ]] || { echo "launcher is not executable: $LAUNCHER" >&2; exit 2; }

restarts=0
write_status "ready" "watchdog initialized" "$restarts"
log "CID 8B watchdog initialized; launcher=$LAUNCHER run_root=$RUN_ROOT"

while true; do
  wait_for_disk_space
  if [[ "$WAIT_FOR_GPUS" == "1" ]]; then
    wait_for_eight_idle_gpus
  fi

  write_status "running" "Stage A/B launcher active" "$restarts"
  log "starting/resuming CID 8B Stage A/B attempt $((restarts + 1))"
  set +e
  setsid "$LAUNCHER" all >>"$WATCHDOG_LOG" 2>&1 &
  child_pid=$!
  wait "$child_pid"
  rc=$?
  child_pid=""
  set -e

  if [[ "$stop_requested" == "1" ]]; then
    exit 130
  fi
  if [[ "$rc" -eq 0 ]]; then
    write_status "completed" "Stage A and Stage B completed" "$restarts"
    log "CID 8B Stage A/B completed successfully"
    exit 0
  fi

  restarts=$((restarts + 1))
  write_status "restarting" "launcher exited rc=$rc" "$restarts"
  log "launcher exited rc=$rc; latest clean checkpoint will be used on restart"
  if [[ "$MAX_RESTARTS" -gt 0 && "$restarts" -ge "$MAX_RESTARTS" ]]; then
    write_status "failed" "restart limit reached after rc=$rc" "$restarts"
    log "restart limit $MAX_RESTARTS reached; watchdog exiting"
    exit "$rc"
  fi
  sleep "$RESTART_DELAY_S"
done

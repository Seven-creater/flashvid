#!/usr/bin/env bash
set -euo pipefail

# Ownership-aware launcher for the final untrained-4B and SFT-4B text agents.
# One vLLM process serves both the frozen base name and the selected LoRA name.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/eval/flashvid_budget_v1}"
VLLM_BIN="${VLLM_BIN:-${PROJECT_DIR}/.venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_DIR}/models/Qwen3.5-4B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RESULT_ROOT}/checkpoints/qwen35_4b_agent_lora}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-${RESULT_ROOT}/checkpoints/validation/checkpoint_selection.json}"
SELECTED_CHECKPOINT="${SELECTED_CHECKPOINT:-}"
HOST="${SFT_CONTROLLER_HOST:-127.0.0.1}"
PORT="${SFT_CONTROLLER_PORT:-8300}"
GPUS="${SFT_CONTROLLER_GPUS:-4,5,6,7}"
BASE_NAME="${SFT_BASE_MODEL_NAME:-Qwen3.5-4B-Agent-SFT-base}"
SFT_NAME="${SFT_MODEL_NAME:-Qwen3.5-4B-Agent-SFT}"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/.runtime/selected_sft_controller}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/flashvid_budget_v1/selected_sft_controller}"
PIDFILE="${STATE_DIR}/controller.pid"
CHECKPOINT_STATE="${STATE_DIR}/selected_checkpoint.path"
start_in_progress=0

usage() {
  echo "usage: $0 start|stop|status" >&2
}

[[ $# -eq 1 ]] || { usage; exit 2; }
action="$1"
case "$action" in
  start|stop|status) ;;
  *) usage; exit 2 ;;
esac

infer_checkpoint() {
  "$PYTHON_BIN" - "$CHECKPOINT_SELECTION" "$CHECKPOINT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

selection = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
payload = json.loads(selection.read_text(encoding="utf-8"))
name = str(payload.get("selection", {}).get("selected_checkpoint") or "")
if not name:
    raise SystemExit("checkpoint selection has no selected checkpoint")
candidate = (root / name).resolve()
candidate.relative_to(root)
if not candidate.is_dir():
    raise SystemExit(f"selected checkpoint directory not found: {candidate}")
print(candidate)
PY
}

listener_pid() {
  local pids
  pids="$(
    ss -H -ltnp "sport = :${PORT}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u
  )"
  [[ -n "$pids" ]] || return 1
  [[ "$(wc -w <<<"$pids")" -eq 1 ]] || return 2
  echo "$pids"
}

read_pid() {
  local pid
  [[ -f "$PIDFILE" ]] || return 1
  pid="$(tr -d '[:space:]' <"$PIDFILE")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  echo "$pid"
}

read_checkpoint_state() {
  local checkpoint
  [[ -f "$CHECKPOINT_STATE" ]] || return 1
  checkpoint="$(head -n 1 "$CHECKPOINT_STATE")"
  [[ -n "$checkpoint" ]] || return 1
  printf '%s\n' "$checkpoint"
}

write_checkpoint_state() {
  local temporary="${CHECKPOINT_STATE}.partial.$$"
  printf '%s\n' "$SELECTED_CHECKPOINT" >"$temporary"
  mv -f "$temporary" "$CHECKPOINT_STATE"
}

owned_process() {
  local pid="$1"
  local recorded
  local command_line
  recorded="$(read_pid 2>/dev/null)" || return 1
  [[ "$pid" == "$recorded" && -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "$command_line" == *"vllm"*"serve"* ]] || return 1
  [[ "$command_line" == *"${BASE_MODEL}"* ]] || return 1
  [[ "$command_line" == *"--port ${PORT}"* ]] || return 1
  [[ "$command_line" == *"--served-model-name ${BASE_NAME}"* ]] || return 1
  [[ "$command_line" == *"--lora-modules ${SFT_NAME}="* ]] || return 1
  if [[ -n "$SELECTED_CHECKPOINT" ]]; then
    [[ "$command_line" == *"${SFT_NAME}=${SELECTED_CHECKPOINT}"* ]] || return 1
  fi
  [[ "$command_line" == *"--data-parallel-size 4"* ]] || return 1
}

model_health() {
  local response
  response="$(curl --silent --show-error --fail --connect-timeout 2 --max-time 5 \
    "http://${HOST}:${PORT}/v1/models" 2>/dev/null)" || return 1
  [[ -n "$response" ]] || return 1
  printf '%s' "$response" | "$PYTHON_BIN" -c \
    'import json,sys
try:
    payload=json.load(sys.stdin)
except (json.JSONDecodeError, TypeError, ValueError):
    raise SystemExit(1)
models={str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
expected=set(sys.argv[1:])
raise SystemExit(0 if expected <= models else 1)' \
    "$BASE_NAME" "$SFT_NAME"
}

wait_healthy() {
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT_SECONDS:-900}))
  while ((SECONDS < deadline)); do
    if model_health; then
      return 0
    fi
    sleep 5
  done
  echo "selected SFT controller did not become healthy on port ${PORT}" >&2
  return 1
}

stop_owned() {
  local pid
  local deadline
  local pgid
  if ! pid="$(read_pid 2>/dev/null)"; then
    rm -f "$PIDFILE" "$CHECKPOINT_STATE"
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PIDFILE" "$CHECKPOINT_STATE"
    return 0
  fi
  owned_process "$pid" || {
    echo "PID ${pid} is live but not the selected SFT controller; refusing" >&2
    return 2
  }
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ || "$pgid" != "$pid" ]]; then
    echo "selected SFT controller PID ${pid} is not its isolated process-group leader; refusing" >&2
    return 2
  fi
  kill -TERM -- "-${pgid}"
  deadline=$((SECONDS + ${STOP_TIMEOUT_SECONDS:-180}))
  while kill -0 -- "-${pgid}" 2>/dev/null; do
    if ((SECONDS >= deadline)); then
      echo "selected SFT controller process group ${pgid} did not stop; refusing SIGKILL" >&2
      return 1
    fi
    sleep 2
  done
  rm -f "$PIDFILE" "$CHECKPOINT_STATE"
  deadline=$((SECONDS + ${STOP_TIMEOUT_SECONDS:-180}))
  while listener_pid >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "port ${PORT} did not drain after selected controller exit" >&2
      return 1
    fi
    sleep 2
  done
}

cleanup_start_failure() {
  local exit_status=$?
  if [[ "$start_in_progress" -eq 1 ]]; then
    stop_owned >/dev/null 2>&1 || true
  fi
  exit "$exit_status"
}

trap cleanup_start_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$STATE_DIR" "$LOG_DIR"

if [[ "$action" == "start" ]]; then
  [[ -x "$PYTHON_BIN" && -x "$VLLM_BIN" ]] || {
    echo "project Python/vLLM is unavailable" >&2
    exit 2
  }
  [[ -d "$BASE_MODEL" ]] || {
    echo "base model not found: ${BASE_MODEL}" >&2
    exit 2
  }
  if [[ -z "$SELECTED_CHECKPOINT" ]]; then
    SELECTED_CHECKPOINT="$(infer_checkpoint)"
  else
    SELECTED_CHECKPOINT="$(realpath "$SELECTED_CHECKPOINT")"
  fi
  [[ -d "$SELECTED_CHECKPOINT" ]] || {
    echo "selected checkpoint not found: ${SELECTED_CHECKPOINT}" >&2
    exit 2
  }
elif [[ "$action" == "status" && ! -x "$PYTHON_BIN" ]]; then
  echo "project Python is unavailable" >&2
  exit 2
elif [[ -z "$SELECTED_CHECKPOINT" ]]; then
  # Stopping an owned process must not depend on a selection report that may
  # have been moved or become unreadable after the service was launched.
  SELECTED_CHECKPOINT="$(read_checkpoint_state 2>/dev/null || true)"
  if [[ -z "$SELECTED_CHECKPOINT" && -f "$CHECKPOINT_SELECTION" ]]; then
    SELECTED_CHECKPOINT="$(infer_checkpoint 2>/dev/null || true)"
  fi
elif [[ -e "$SELECTED_CHECKPOINT" ]]; then
  SELECTED_CHECKPOINT="$(realpath "$SELECTED_CHECKPOINT")"
fi

case "$action" in
  status)
    pid="$(listener_pid 2>/dev/null)" || {
      echo "port ${PORT}: no listener"
      exit 1
    }
    if owned_process "$pid" && model_health; then
      echo "selected SFT controller healthy: pid=${pid} port=${PORT}"
    else
      echo "port ${PORT}: listener is unowned or unhealthy" >&2
      exit 2
    fi
    ;;
  stop)
    stop_owned
    ;;
  start)
    if pid="$(listener_pid 2>/dev/null)"; then
      if owned_process "$pid" && model_health; then
        write_checkpoint_state
        echo "selected SFT controller already healthy: pid=${pid}"
        exit 0
      fi
      echo "port ${PORT} has an unverified listener; refusing to replace it" >&2
      exit 2
    fi
    rm -f "$PIDFILE"
    write_checkpoint_state
    start_in_progress=1
    export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
    export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/transformers"
    export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
    export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    if [[ -d "${PROJECT_DIR}/.venv/cuda-compat" ]]; then
      export LD_LIBRARY_PATH="${PROJECT_DIR}/.venv/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    nohup setsid env -u VLLM_PLUGINS \
      CUDA_VISIBLE_DEVICES="$GPUS" \
      VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm_selected_sft_text_controller" \
      TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor_selected_sft_text_controller" \
      TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton_selected_sft_text_controller" \
      "$VLLM_BIN" serve "$BASE_MODEL" \
        --served-model-name "$BASE_NAME" \
        --enable-lora \
        --lora-modules "${SFT_NAME}=${SELECTED_CHECKPOINT}" \
        --max-lora-rank 16 \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 4 \
        --dtype bfloat16 \
        --max-model-len "${CONTROLLER_MAX_MODEL_LEN:-8192}" \
        --max-num-seqs "${CONTROLLER_MAX_NUM_SEQS:-64}" \
        --max-num-batched-tokens "${CONTROLLER_MAX_BATCHED_TOKENS:-32768}" \
        --gpu-memory-utilization "${CONTROLLER_GPU_MEMORY_UTILIZATION:-0.90}" \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        >"${LOG_DIR}/controller.log" 2>&1 </dev/null &
    echo "$!" >"$PIDFILE"
    wait_healthy
    pid="$(listener_pid)"
    owned_process "$pid" || {
      echo "healthy selected SFT endpoint failed ownership verification" >&2
      exit 2
    }
    start_in_progress=0
    echo "selected SFT controller started: pid=${pid} port=${PORT}"
    ;;
esac

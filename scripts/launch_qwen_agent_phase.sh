#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
QWEN_STALL_TIMEOUT_S=${QWEN_STALL_TIMEOUT_S:-90}
QWEN_WATCH_ROOT=${QWEN_WATCH_ROOT:-$PROJECT_DIR/results/eval/qwen_agent_search}

usage() {
  echo "usage: $0 CONFIG PHASE MODEL_GROUP [run_qwen_agent_search.py args ...]" >&2
}

[[ $# -ge 3 ]] || { usage; exit 2; }
CONFIG=$1
PHASE=$2
MODEL_KEY=$3
shift 3
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "experiment config not found: $CONFIG" >&2; exit 2; }
[[ "$MODEL_KEY" == "q4" || "$MODEL_KEY" == "q9" || "$MODEL_KEY" == "sft9" ]] || {
  echo "MODEL_GROUP must be q4, q9, or sft9" >&2
  exit 2
}
if [[ "$MODEL_KEY" == "sft9" && "$PHASE" != "final_matrix" ]]; then
  echo "sft9 model group is only valid for final_matrix" >&2
  exit 2
fi

mkdir -p "$PROJECT_DIR/logs"
safe_phase=${PHASE//[^A-Za-z0-9_.-]/_}
safe_model=${MODEL_KEY//[^A-Za-z0-9_.-]/_}
PID_FILE="$PROJECT_DIR/logs/qwen_${safe_phase}_${safe_model}.pid"
LOG_FILE="$PROJECT_DIR/logs/qwen_${safe_phase}_${safe_model}.log"
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$command_line" == *"run_qwen_agent_search.py"* ]]; then
      echo "owned phase is already running: pid=$pid log=$LOG_FILE" >&2
    else
      echo "pid file points to a different process; refusing to overwrite: $pid" >&2
    fi
    exit 1
  fi
  rm -f "$PID_FILE"
fi

args=(
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_qwen_agent_search.py"
  --config "$CONFIG"
  --phase "$PHASE"
  --resume
)
if [[ "$PHASE" == "final_matrix" ]]; then
  args+=(--final-model-group "$MODEL_KEY")
else
  args+=(--model-key "$MODEL_KEY")
fi
args+=("$@")

launch_args=("${args[@]}")
if [[ "$QWEN_STALL_TIMEOUT_S" != "0" ]]; then
  launch_args=(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_with_progress_watchdog.py"
    --watch-root "$QWEN_WATCH_ROOT"
    --stall-seconds "$QWEN_STALL_TIMEOUT_S"
    --
    "${args[@]}"
  )
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
setsid nohup "${launch_args[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started phase=$PHASE model=$MODEL_KEY pid=$pid stall_timeout_s=$QWEN_STALL_TIMEOUT_S log=$LOG_FILE"

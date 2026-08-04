#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
STALL_TIMEOUT_S=${QWEN_STALL_TIMEOUT_S:-90}
REQUEST_TIMEOUT_S=${QWEN_REQUEST_TIMEOUT_S:-80}
[[ $# -gt 0 ]] || {
  echo "usage: $0 run_qwen_counterfactuals.py arguments" >&2
  exit 2
}
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }

args=("$@")
dataset=""
output=""
has_resume=0
has_timeout=0
for ((index=0; index<${#args[@]}; index++)); do
  case "${args[$index]}" in
    --dataset)
      dataset=${args[$((index + 1))]:-}
      ;;
    --output)
      output=${args[$((index + 1))]:-}
      ;;
    --resume)
      has_resume=1
      ;;
    --timeout)
      args[$((index + 1))]=$REQUEST_TIMEOUT_S
      has_timeout=1
      ;;
  esac
done
[[ "$dataset" == "lvbench" || "$dataset" == "lsdbench" || "$dataset" == "cgbench" ]] || {
  echo "a valid --dataset is required" >&2
  exit 2
}
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
(( has_resume )) || args+=(--resume)
(( has_timeout )) || args+=(--timeout "$REQUEST_TIMEOUT_S")

mkdir -p "$PROJECT_DIR/logs" "$(dirname "$output")"
PID_FILE="$PROJECT_DIR/logs/qwen_counterfactual_${dataset}.pid"
LOG_FILE="$PROJECT_DIR/logs/qwen_counterfactual_${dataset}.log"
HEARTBEAT_FILE="$PROJECT_DIR/logs/qwen_counterfactual_${dataset}.heartbeat.json"
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [[ "$command_line" == *"run_qwen_counterfactuals.py"* ]] || {
      echo "pid file points to an unowned process: $pid" >&2
      exit 2
    }
    echo "owned counterfactual run is already active: pid=$pid log=$LOG_FILE" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
rm -f "$HEARTBEAT_FILE"
export QWEN_PROGRESS_HEARTBEAT="$HEARTBEAT_FILE"
setsid nohup "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_with_progress_watchdog.py" \
  --watch-root "$(dirname "$output")" --heartbeat-file "$HEARTBEAT_FILE" \
  --stall-seconds "$STALL_TIMEOUT_S" -- \
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_qwen_counterfactuals.py" "${args[@]}" \
  >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started dataset=$dataset pid=$pid request_timeout_s=$REQUEST_TIMEOUT_S stall_timeout_s=$STALL_TIMEOUT_S heartbeat=$HEARTBEAT_FILE log=$LOG_FILE"

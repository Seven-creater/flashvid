#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
STALL_TIMEOUT_S=${QWEN_STALL_TIMEOUT_S:-90}
[[ $# -eq 1 ]] || { echo "usage: $0 RESCUE_INDEX" >&2; exit 2; }
RESCUE_INDEX=$1
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$RESCUE_INDEX" ]] || { echo "rescue index not found: $RESCUE_INDEX" >&2; exit 2; }

mkdir -p "$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/logs/qwen_trajectory_rescue.pid"
LOG_FILE="$PROJECT_DIR/logs/qwen_trajectory_rescue.log"
HEARTBEAT_FILE="$PROJECT_DIR/logs/qwen_trajectory_rescue.heartbeat.json"
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [[ "$command_line" == *"run_qwen_rescue_trajectories.py"* ]] || {
      echo "pid file points to an unowned process: $pid" >&2
      exit 2
    }
    echo "owned rescue is already running: pid=$pid log=$LOG_FILE" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
rm -f "$HEARTBEAT_FILE"
export QWEN_PROGRESS_HEARTBEAT="$HEARTBEAT_FILE"
setsid nohup "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_with_progress_watchdog.py" \
  --watch-root "$PROJECT_DIR/results/eval/qwen_agent_search/trajectories/rescue" \
  --heartbeat-file "$HEARTBEAT_FILE" \
  --stall-seconds "$STALL_TIMEOUT_S" -- \
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_qwen_rescue_trajectories.py" \
  --rescue-index "$RESCUE_INDEX" --resume \
  >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started qwen trajectory rescue pid=$pid stall_timeout_s=$STALL_TIMEOUT_S heartbeat=$HEARTBEAT_FILE log=$LOG_FILE"

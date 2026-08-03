#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  echo "usage: $0 MODEL_PATH SERVED_NAME [PORT] [DP]" >&2
}

[[ $# -ge 2 && $# -le 4 ]] || { usage; exit 2; }
MODEL_PATH=$1
SERVED_NAME=$2
PORT=${3:-8200}
DP=${4:-4}
[[ -e "$MODEL_PATH" ]] || { echo "model path not found: $MODEL_PATH" >&2; exit 2; }

mkdir -p "$PROJECT_DIR/logs"
safe_name=${SERVED_NAME//[^A-Za-z0-9_.-]/_}
LAUNCH_PID_FILE="$PROJECT_DIR/logs/qwen_service_${PORT}.launch.pid"
LOG_FILE="$PROJECT_DIR/logs/qwen_service_${safe_name}_${PORT}.log"

if [[ -f "$LAUNCH_PID_FILE" ]]; then
  old_pid=$(cat "$LAUNCH_PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    command_line=$(tr '\0' ' ' < "/proc/$old_pid/cmdline")
    if [[ "$command_line" == *"serve_qwen_agent.sh"* ]]; then
      echo "owned service launcher is already running: pid=$old_pid log=$LOG_FILE" >&2
    else
      echo "launch pid file points to an unowned process: $old_pid" >&2
    fi
    exit 1
  fi
  rm -f "$LAUNCH_PID_FILE"
fi

cd "$PROJECT_DIR"
setsid nohup "$PROJECT_DIR/scripts/serve_qwen_agent.sh" \
  "$MODEL_PATH" "$SERVED_NAME" "$PORT" "$DP" \
  >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$LAUNCH_PID_FILE"
echo "started model=$SERVED_NAME pid=$pid port=$PORT log=$LOG_FILE"

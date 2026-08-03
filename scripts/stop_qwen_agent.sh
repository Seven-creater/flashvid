#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${1:-8200}
PID_FILE="$PROJECT_DIR/logs/qwen_agent_${PORT}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "no owned Qwen Agent pid file for port $PORT"
  exit 0
fi

pid=$(cat "$PID_FILE")
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "removed stale pid file for port $PORT"
  exit 0
fi

command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
if [[ "$command_line" != *"vllm"* ]] || [[ "$command_line" != *"--port $PORT"* ]]; then
  echo "pid $pid is not the owned vLLM service for port $PORT; refusing to stop it" >&2
  exit 1
fi

kill "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "stopped owned Qwen Agent service pid=$pid port=$PORT"
    exit 0
  fi
  sleep 1
done

echo "owned service pid=$pid did not stop within 30 seconds" >&2
exit 1

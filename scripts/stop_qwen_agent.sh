#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${1:-8200}
PID_FILE="$PROJECT_DIR/logs/qwen_agent_${PORT}.pid"

case "$PORT" in
  8200|8201) ;;
  *)
    echo "refusing to stop non-project port $PORT; allowed ports are 8200 and 8201" >&2
    exit 2
    ;;
esac

if [[ ! -f "$PID_FILE" ]]; then
  echo "no owned Qwen Agent pid file for port $PORT"
  exit 0
fi

pid=$(cat "$PID_FILE")
[[ "$pid" =~ ^[0-9]+$ ]] || {
  echo "invalid owned pid file for port $PORT" >&2
  exit 1
}
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "removed stale pid file for port $PORT"
  exit 0
fi

process_uid=$(awk '/^Uid:/ {print $2}' "/proc/$pid/status")
if [[ "$process_uid" != "$(id -u)" ]]; then
  echo "pid $pid belongs to uid $process_uid; refusing to stop it" >&2
  exit 1
fi
command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
if [[ "$command_line" != *"vllm"* ]] || [[ "$command_line" != *"--port $PORT"* ]]; then
  echo "pid $pid is not the owned vLLM service for port $PORT; refusing to stop it" >&2
  exit 1
fi
owner_marker=$(tr '\0' '\n' < "/proc/$pid/environ" | grep -Fx "FLASHVID_QWEN_OWNER_DIR=$PROJECT_DIR" || true)
if [[ -z "$owner_marker" ]]; then
  echo "pid $pid lacks this project's ownership marker; refusing to stop it" >&2
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

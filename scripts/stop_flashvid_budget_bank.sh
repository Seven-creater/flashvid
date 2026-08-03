#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_flashvid_budget_bank.sh
source "${PROJECT_DIR}/scripts/lib_flashvid_budget_bank.sh"

stop_mode="${1:---all}"
case "$stop_mode" in
  --all)
    stop_ids=(controller r010 r025 r050 r100)
    ;;
  --controller-only)
    stop_ids=(controller)
    ;;
  --perception-only)
    stop_ids=(r010 r025 r050 r100)
    ;;
  *)
    echo "usage: $0 [--all|--controller-only|--perception-only]" >&2
    exit 2
    ;;
esac

failures=0

stop_one() {
  local id="$1"
  local pid
  local pidfile
  local pgid
  local remaining
  pidfile="$(service_pidfile "$id")"

  if ! pid="$(read_service_pid "$id" 2>/dev/null)"; then
    if [[ -e "$pidfile" ]]; then
      echo "$id: invalid pidfile left untouched: $pidfile" >&2
      return 1
    fi
    echo "$id: not running"
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pidfile"
    echo "$id: removed stale pidfile for pid $pid"
    return 0
  fi
  if ! owned_service_process "$id" "$pid"; then
    echo "$id: pid $pid is not an owned budget-bank process; refusing to signal it" >&2
    return 1
  fi

  pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
  if [[ "$pgid" != "$pid" ]]; then
    echo "$id: pid $pid is not its process-group leader; refusing an unsafe group signal" >&2
    return 1
  fi

  kill -TERM -- "-$pgid"
  remaining=30
  while kill -0 "$pid" 2>/dev/null && (( remaining > 0 )); do
    sleep 1
    remaining=$((remaining - 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "$id: graceful stop timed out; sending KILL to owned process group $pgid" >&2
    kill -KILL -- "-$pgid"
  fi
  rm -f "$pidfile"
  echo "$id: stopped pid=$pid"
}

# In all-services mode, stop the controller first so no new work is scheduled.
for id in "${stop_ids[@]}"; do
  if ! stop_one "$id"; then
    failures=1
  fi
done

exit "$failures"

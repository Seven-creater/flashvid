#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_flashvid_budget_bank.sh
source "${PROJECT_DIR}/scripts/lib_flashvid_budget_bank.sh"

wait_seconds=0
check_ids=("${BUDGET_SERVICE_IDS[@]}")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      check_ids=("${BUDGET_SERVICE_IDS[@]}")
      shift
      ;;
    --perception-only)
      check_ids=("${BUDGET_PERCEPTION_IDS[@]}")
      shift
      ;;
    --controller-only)
      check_ids=(controller)
      shift
      ;;
    --wait)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        echo "usage: $0 [--all|--perception-only|--controller-only] [--wait SECONDS]" >&2
        exit 2
      }
      wait_seconds="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--all|--perception-only|--controller-only] [--wait SECONDS]" >&2
      exit 2
      ;;
  esac
done

model_is_served() {
  local response="$1"
  local expected="$2"
  EXPECTED_MODEL="$expected" "$HEALTH_PYTHON" -c \
    'import json, os, sys
payload = json.load(sys.stdin)
expected = os.environ["EXPECTED_MODEL"]
models = {str(item.get("id")) for item in payload.get("data", [])}
raise SystemExit(0 if expected in models else 1)' <<<"$response"
}

check_one() {
  local id="$1"
  local pid
  local port
  local response
  local model

  if ! pid="$(read_service_pid "$id" 2>/dev/null)"; then
    echo "$id: missing or invalid pidfile" >&2
    return 1
  fi
  if ! owned_service_process "$id" "$pid"; then
    echo "$id: pid $pid is dead or its command signature does not match" >&2
    return 1
  fi

  port="$(service_port "$id")"
  model="$(service_model "$id")"
  curl --silent --fail --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${port}/health" >/dev/null || return 1
  response="$(curl --silent --fail --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${port}/v1/models")" || return 1
  model_is_served "$response" "$model" || return 1
  echo "$id: healthy pid=$pid port=$port model=$model"
}

deadline=$((SECONDS + wait_seconds))
while true; do
  failed=0
  for id in "${check_ids[@]}"; do
    if ! check_one "$id"; then
      failed=1
    fi
  done
  if [[ "$failed" -eq 0 ]]; then
    exit 0
  fi
  if (( SECONDS >= deadline )); then
    echo "budget-bank health check failed; inspect ${BUDGET_BANK_LOG_DIR}" >&2
    exit 1
  fi
  sleep 5
done

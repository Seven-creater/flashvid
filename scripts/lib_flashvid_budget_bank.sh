#!/usr/bin/env bash

# Shared constants and ownership checks for the isolated FlashVID budget bank.
# This file is sourced by the launch, health-check, and stop scripts.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGET_BANK_STATE_DIR="${BUDGET_BANK_STATE_DIR:-${PROJECT_DIR}/.runtime/flashvid_budget_bank}"
BUDGET_BANK_LOG_DIR="${BUDGET_BANK_LOG_DIR:-${PROJECT_DIR}/logs/flashvid_budget_bank}"
FLASHVID_MEDIA_ROOT="${FLASHVID_MEDIA_ROOT:-/dev/shm/flashvid_perception}"
SERVICE_HOST="${SERVICE_HOST:-127.0.0.1}"

FLASHVID_4B_MODEL="${FLASHVID_4B_MODEL:-${PROJECT_DIR}/models/Qwen3.5-4B}"
QWEN_9B_MODEL="${QWEN_9B_MODEL:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"
FLASHVID_BIN="${FLASHVID_BIN:-${PROJECT_DIR}/.venv/bin/flashvid-serve}"
VLLM_BIN="${VLLM_BIN:-${PROJECT_DIR}/.venv/bin/vllm}"
HEALTH_PYTHON="${HEALTH_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"

BUDGET_PERCEPTION_IDS=(r010 r025 r050 r100)
BUDGET_SERVICE_IDS=("${BUDGET_PERCEPTION_IDS[@]}" controller)

service_port() {
  case "$1" in
    r010) echo 8101 ;;
    r025) echo 8102 ;;
    r050) echo 8103 ;;
    r100) echo 8104 ;;
    controller) echo 8200 ;;
    *) return 2 ;;
  esac
}

service_model() {
  case "$1" in
    r010) echo "Qwen3.5-4B-FlashVID-r010" ;;
    r025) echo "Qwen3.5-4B-FlashVID-r025" ;;
    r050) echo "Qwen3.5-4B-FlashVID-r050" ;;
    r100) echo "Qwen3.5-4B-FlashVID-r100" ;;
    controller) echo "Qwen3.5-9B" ;;
    *) return 2 ;;
  esac
}

service_ratio() {
  case "$1" in
    r010) echo "0.10" ;;
    r025) echo "0.25" ;;
    r050) echo "0.50" ;;
    r100) echo "1.00" ;;
    *) return 2 ;;
  esac
}

service_backend() {
  case "$1" in
    r010|r025|r050) echo "flashvid" ;;
    # A 100% retention budget is the native, no-compression control.  Loading
    # the out-of-tree architecture even with pruning disabled changes the
    # numerical path, so r100 deliberately uses the native vLLM model class.
    r100) echo "native_bypass" ;;
    controller) echo "text_controller" ;;
    *) return 2 ;;
  esac
}

service_gpu() {
  case "$1" in
    r010) echo "${BUDGET_GPU_R010:-0}" ;;
    r025) echo "${BUDGET_GPU_R025:-1}" ;;
    r050) echo "${BUDGET_GPU_R050:-2}" ;;
    r100) echo "${BUDGET_GPU_R100:-3}" ;;
    controller) echo "${BUDGET_GPU_CONTROLLER:-4,5,6,7}" ;;
    *) return 2 ;;
  esac
}

service_pidfile() {
  echo "${BUDGET_BANK_STATE_DIR}/$1.pid"
}

service_logfile() {
  echo "${BUDGET_BANK_LOG_DIR}/$1.log"
}

read_service_pid() {
  local pidfile
  local pid
  pidfile="$(service_pidfile "$1")"
  [[ -f "$pidfile" ]] || return 1
  pid="$(tr -d '[:space:]' < "$pidfile")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  echo "$pid"
}

owned_service_process() {
  local id="$1"
  local pid="$2"
  local command_line
  local expected_model
  local expected_port

  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  expected_model="$(service_model "$id")"
  expected_port="$(service_port "$id")"

  [[ "$command_line" == *"--served-model-name ${expected_model}"* ]] || return 1
  [[ "$command_line" == *"--port ${expected_port}"* ]] || return 1
  if [[ "$id" == "controller" ]]; then
    [[ "$command_line" == *"vllm"*"serve"* ]] || return 1
    [[ "$command_line" == *"${QWEN_9B_MODEL}"* ]] || return 1
    [[ "$command_line" == *"--data-parallel-size "* ]] || return 1
  elif [[ "$(service_backend "$id")" == "native_bypass" ]]; then
    [[ "$command_line" == *"vllm"*"serve"* ]] || return 1
    [[ "$command_line" != *"flashvid-serve"* ]] || return 1
    [[ "$command_line" != *"--vision-retention-ratio"* ]] || return 1
    [[ "$command_line" == *"${FLASHVID_4B_MODEL}"* ]] || return 1
    [[ "$command_line" == *"--data-parallel-size 1"* ]] || return 1
    [[ "$command_line" == *"--allowed-local-media-path ${FLASHVID_MEDIA_ROOT}"* ]] || return 1
  else
    [[ "$command_line" == *"flashvid-serve"* ]] || return 1
    [[ "$command_line" == *"${FLASHVID_4B_MODEL}"* ]] || return 1
    [[ "$command_line" == *"--vision-retention-ratio $(service_ratio "$id")"* ]] || return 1
    [[ "$command_line" == *"--allowed-local-media-path ${FLASHVID_MEDIA_ROOT}"* ]] || return 1
  fi
}

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib_flashvid_budget_bank.sh
source "${PROJECT_DIR}/scripts/lib_flashvid_budget_bank.sh"

launch_mode="${1:---all}"
case "$launch_mode" in
  --all)
    launch_ids=("${BUDGET_SERVICE_IDS[@]}")
    ;;
  --perception-only)
    launch_ids=("${BUDGET_PERCEPTION_IDS[@]}")
    ;;
  --controller-only)
    launch_ids=(controller)
    ;;
  *)
    echo "usage: $0 [--all|--perception-only|--controller-only]" >&2
    exit 2
    ;;
esac

require_file_or_directory() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 2
  fi
}

require_file_or_directory "FlashVID launcher" "$FLASHVID_BIN"
require_file_or_directory "vLLM launcher" "$VLLM_BIN"
require_file_or_directory "health-check Python" "$HEALTH_PYTHON"
require_file_or_directory "Qwen3.5-4B model" "$FLASHVID_4B_MODEL"
require_file_or_directory "Qwen3.5-9B model" "$QWEN_9B_MODEL"

mkdir -p "$BUDGET_BANK_STATE_DIR" "$BUDGET_BANK_LOG_DIR" "$FLASHVID_MEDIA_ROOT"
chmod 700 "$FLASHVID_MEDIA_ROOT"

export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/transformers"
export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export FLASHINFER_WORKSPACE_BASE="${PROJECT_DIR}"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.cache/torch_extensions"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -d "${PROJECT_DIR}/.venv/cuda-compat" ]]; then
  export LD_LIBRARY_PATH="${PROJECT_DIR}/.venv/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

assert_startable() {
  local id="$1"
  local pid
  local url
  local pidfile
  pidfile="$(service_pidfile "$id")"

  if pid="$(read_service_pid "$id" 2>/dev/null)"; then
    if owned_service_process "$id" "$pid"; then
      echo "$id is already running with owned pid $pid" >&2
      return 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
      echo "$id pidfile points to an unowned live process ($pid); refusing to overwrite it" >&2
      return 1
    fi
    rm -f "$pidfile"
  elif [[ -e "$pidfile" ]]; then
    echo "$id has an invalid pidfile: $pidfile" >&2
    return 1
  fi

  url="http://127.0.0.1:$(service_port "$id")/health"
  if curl --silent --fail --connect-timeout 1 --max-time 2 "$url" >/dev/null 2>&1; then
    echo "$id port $(service_port "$id") already has a healthy unowned service; refusing to replace it" >&2
    return 1
  fi
}

for id in "${launch_ids[@]}"; do
  assert_startable "$id"
done

start_perception() {
  local id="$1"
  local gpu
  local model_name
  local port
  local ratio
  local logfile
  local pidfile
  local backend
  local -a kv_cache_args=()
  gpu="$(service_gpu "$id")"
  model_name="$(service_model "$id")"
  port="$(service_port "$id")"
  ratio="$(service_ratio "$id")"
  backend="$(service_backend "$id")"
  logfile="$(service_logfile "$id")"
  pidfile="$(service_pidfile "$id")"

  if [[ -n "${PERCEPTION_KV_CACHE_MEMORY_BYTES:-}" ]]; then
    kv_cache_args=(--kv-cache-memory-bytes "$PERCEPTION_KV_CACHE_MEMORY_BYTES")
  fi

  if [[ "$backend" == "native_bypass" ]]; then
    nohup setsid env -u VLLM_PLUGINS \
      CUDA_VISIBLE_DEVICES="$gpu" \
      VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm_native_${id}" \
      TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor_native_${id}" \
      TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton_native_${id}" \
      "$VLLM_BIN" serve "$FLASHVID_4B_MODEL" \
        --served-model-name "$model_name" Qwen3.5-4B-native \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        --host "$SERVICE_HOST" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len "${PERCEPTION_MAX_MODEL_LEN:-32768}" \
        --max-num-seqs "${PERCEPTION_MAX_NUM_SEQS:-8}" \
        --max-num-batched-tokens "${PERCEPTION_MAX_BATCHED_TOKENS:-32768}" \
        --gpu-memory-utilization "${PERCEPTION_GPU_MEMORY_UTILIZATION:-0.90}" \
        "${kv_cache_args[@]}" \
        --enable-prompt-tokens-details \
        --limit-mm-per-prompt '{"image":0,"video":1}' \
        --allowed-local-media-path "$FLASHVID_MEDIA_ROOT" \
        >"$logfile" 2>&1 < /dev/null &
  else
    nohup setsid env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      VLLM_PLUGINS="flashvid_qwen3_5" \
      VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm_flashvid_${id}" \
      TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor_flashvid_${id}" \
      TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton_flashvid_${id}" \
      "$FLASHVID_BIN" "$FLASHVID_4B_MODEL" \
        --vision-retention-ratio "$ratio" \
        --served-model-name "$model_name" \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        --host "$SERVICE_HOST" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len "${PERCEPTION_MAX_MODEL_LEN:-32768}" \
        --max-num-seqs "${PERCEPTION_MAX_NUM_SEQS:-8}" \
        --max-num-batched-tokens "${PERCEPTION_MAX_BATCHED_TOKENS:-32768}" \
        --gpu-memory-utilization "${PERCEPTION_GPU_MEMORY_UTILIZATION:-0.90}" \
        "${kv_cache_args[@]}" \
        --enable-prompt-tokens-details \
        --limit-mm-per-prompt '{"image":0,"video":1}' \
        --allowed-local-media-path "$FLASHVID_MEDIA_ROOT" \
        >"$logfile" 2>&1 < /dev/null &
  fi
  echo "$!" > "$pidfile"
  echo "started $id pid=$(cat "$pidfile") gpu=$gpu ratio=$ratio backend=$backend port=$port"
}

start_controller() {
  local logfile
  local pidfile
  logfile="$(service_logfile controller)"
  pidfile="$(service_pidfile controller)"

  nohup setsid env -u VLLM_PLUGINS \
    CUDA_VISIBLE_DEVICES="$(service_gpu controller)" \
    VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm_qwen9b_text_controller" \
    TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor_qwen9b_text_controller" \
    TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton_qwen9b_text_controller" \
    "$VLLM_BIN" serve "$QWEN_9B_MODEL" \
      --served-model-name "$(service_model controller)" \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --host "$SERVICE_HOST" \
      --port "$(service_port controller)" \
      --tensor-parallel-size 1 \
      --data-parallel-size "${CONTROLLER_DATA_PARALLEL_SIZE:-4}" \
      --dtype bfloat16 \
      --max-model-len "${CONTROLLER_MAX_MODEL_LEN:-32768}" \
      --max-num-seqs "${CONTROLLER_MAX_NUM_SEQS:-64}" \
      --max-num-batched-tokens "${CONTROLLER_MAX_BATCHED_TOKENS:-32768}" \
      --gpu-memory-utilization "${CONTROLLER_GPU_MEMORY_UTILIZATION:-0.90}" \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      >"$logfile" 2>&1 < /dev/null &
  echo "$!" > "$pidfile"
  echo "started controller pid=$(cat "$pidfile") gpu=$(service_gpu controller) port=$(service_port controller)"
}

if [[ "$launch_mode" != "--controller-only" ]]; then
  start_perception r010
  start_perception r025
  start_perception r050
  start_perception r100
fi
if [[ "$launch_mode" != "--perception-only" ]]; then
  start_controller
fi

bash "${PROJECT_DIR}/scripts/healthcheck_flashvid_budget_bank.sh" \
  "$launch_mode" --wait "${HEALTH_TIMEOUT_SECONDS:-900}"

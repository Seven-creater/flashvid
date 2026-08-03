#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${1:-"$PROJECT_DIR/models/Qwen3.5-9B"}
DATA_PARALLEL_SIZE=${2:-8}
PORT=${3:-8001}
ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-/data02/pretrained_model/cvr_learn}

# This launcher never searches for or stops another service. Call the
# ownership-aware stop helper for this project explicitly before changing GPU
# layouts; a busy GPU or port is treated as an external scheduling constraint.

mkdir -p "$PROJECT_DIR/.cache" "$PROJECT_DIR/logs"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/transformers"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
# The host provides CUDA 11.8, while the bundled FlashInfer sampler emits
# CUDA-12-only headers.  Use vLLM's compatible sampler fallback; model
# execution remains on all data-parallel GPUs.
export VLLM_USE_FLASHINFER_SAMPLER=0
# The host driver is older than the CUDA runtime bundled in the shared venv.
# Reuse the compatibility library path used by the healthy legacy service;
# without it, vLLM fails during torch._C._cuda_init before binding the port.
if [[ -d "$PROJECT_DIR/.venv/cuda-compat" ]]; then
  export LD_LIBRARY_PATH="$PROJECT_DIR/.venv/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PROJECT_DIR/.venv/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name Qwen3.5-9B \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --limit-mm-per-prompt '{"image":9999,"video":1}' \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH"

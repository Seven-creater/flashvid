#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RATIO="${1:-0.10}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/Qwen3.5-4B}"
PORT="${PORT:-8000}"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR must be an existing local model directory with config.json; Hugging Face IDs are forbidden: $MODEL_DIR" >&2
  exit 2
fi
MODEL_DIR=$(cd "$MODEL_DIR" && pwd -P)

export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export FLASHINFER_WORKSPACE_BASE="${PROJECT_DIR}"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.cache/torch_extensions"
export VLLM_PLUGINS="flashvid_qwen3_5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LD_LIBRARY_PATH="${PROJECT_DIR}/.venv/cuda-compat:${LD_LIBRARY_PATH:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"
exec "${PROJECT_DIR}/.venv/bin/flashvid-serve" "${MODEL_DIR}" \
  --vision-retention-ratio "${RATIO}" \
  --served-model-name qwen3.5-4b-flashvid \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  2>&1 | tee "${PROJECT_DIR}/logs/server-dp8-r${RATIO}.log"

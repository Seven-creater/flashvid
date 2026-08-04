#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${1:?usage: serve_qwen_agent.sh MODEL_PATH SERVED_NAME [PORT] [DP]}
SERVED_NAME=${2:?usage: serve_qwen_agent.sh MODEL_PATH SERVED_NAME [PORT] [DP]}
PORT=${3:-8200}
DATA_PARALLEL_SIZE=${4:-4}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
# Direct-video inputs live under the shared dataset tree, while EVA-selected
# frame caches live under this project's /data02 workspace. vLLM accepts one
# local-media root, so /data02 is the narrowest common ancestor of both.
ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-/data02}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
ALLOW_SHARED_GPUS=${ALLOW_SHARED_GPUS:-0}
VLLM_BIN=${VLLM_BIN:-$PROJECT_DIR/.venv/bin/vllm}
PID_FILE="$PROJECT_DIR/logs/qwen_agent_${PORT}.pid"
LORA_PATH=${LORA_PATH:-}
LORA_SERVED_NAME=${LORA_SERVED_NAME:-}
MAX_LORA_RANK=${MAX_LORA_RANK:-16}

mkdir -p "$PROJECT_DIR/.cache" "$PROJECT_DIR/logs"

if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH must be an existing local model directory with config.json; Hugging Face IDs are forbidden: $MODEL_PATH" >&2
  exit 2
fi
MODEL_PATH=$(cd "$MODEL_PATH" && pwd -P)

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable not found: $VLLM_BIN" >&2
  exit 2
fi
VLLM_ENV_DIR=$(cd "$(dirname "$VLLM_BIN")/.." && pwd)
CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-$VLLM_ENV_DIR/cuda-compat}
if [[ -n "$LORA_PATH" || -n "$LORA_SERVED_NAME" ]]; then
  [[ -n "$LORA_PATH" && -n "$LORA_SERVED_NAME" ]] || {
    echo "LORA_PATH and LORA_SERVED_NAME must be set together" >&2
    exit 2
  }
  [[ -d "$LORA_PATH" ]] || { echo "LoRA checkpoint not found: $LORA_PATH" >&2; exit 2; }
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(cat "$PID_FILE")
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "owned Qwen Agent service is already running: pid=$existing_pid port=$PORT" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q .; then
  echo "port $PORT is already in use; refusing to touch an unowned service" >&2
  exit 1
fi

active_gpu_pids=$(
  nvidia-smi -i "$CUDA_DEVICES" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -u
)
if [[ -n "$active_gpu_pids" ]]; then
  if [[ "$ALLOW_SHARED_GPUS" != "1" ]]; then
    echo "selected GPUs are already in use; refusing to touch these PIDs:" >&2
    echo "$active_gpu_pids" >&2
    echo "set ALLOW_SHARED_GPUS=1 only after confirming coexistence is safe" >&2
    exit 2
  fi
  echo "warning: sharing selected GPUs with existing PIDs; they will not be touched:" >&2
  echo "$active_gpu_pids" >&2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/transformers"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
# stop_qwen_agent.sh requires this exact marker before it will signal a PID.
# It survives exec and prevents a stale pid file from targeting another user's
# vLLM process that happens to reuse the same PID and port.
export FLASHVID_QWEN_OWNER_DIR="$PROJECT_DIR"
if [[ -d "$CUDA_COMPAT_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo $$ > "$PID_FILE"
cleanup() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

lora_args=()
if [[ -n "$LORA_PATH" ]]; then
  lora_args=(
    --enable-lora
    --max-lora-rank "$MAX_LORA_RANK"
    --lora-modules "$LORA_SERVED_NAME=$LORA_PATH"
  )
fi

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --limit-mm-per-prompt '{"image":9999,"video":1}' \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  "${lora_args[@]}"

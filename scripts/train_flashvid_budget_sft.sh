#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-swift}"
SWIFT_BIN="${SWIFT_BIN:-${SFT_ENV_DIR}/bin/swift}"
SWIFT_PYTHON="${SWIFT_PYTHON:-${SFT_ENV_DIR}/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/eval/flashvid_budget_v1/checkpoints/qwen35_4b_agent_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

train_data=""
resume=0

usage() {
  echo "usage: $0 --train-data FILE [--output-dir DIR] [--resume]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      train_data="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -f "$train_data" ]] || { echo "training JSONL not found: $train_data" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || { echo "Qwen3.5-4B model not found: $MODEL_PATH" >&2; exit 2; }
[[ -x "$SWIFT_BIN" && -x "$SWIFT_PYTHON" ]] || {
  echo "ms-swift environment not found at $SFT_ENV_DIR" >&2
  echo "Create it separately and install exactly ms-swift==4.4.2." >&2
  exit 2
}

"$SWIFT_PYTHON" -c \
  'import importlib.metadata as m
version = m.version("ms-swift")
assert version == "4.4.2", f"expected ms-swift 4.4.2, found {version}"'
"$SWIFT_PYTHON" -c \
  'import flash_attn
from flash_attn import flash_attn_func
assert callable(flash_attn_func)' || {
  echo "flash_attn is required by --attn_impl flash_attn but is unavailable" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR"

active_gpu_pids="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -u
)"
if [[ -n "$active_gpu_pids" ]]; then
  echo "all inference services must be stopped before 8-GPU SFT; active GPU PIDs:" >&2
  echo "$active_gpu_pids" >&2
  exit 2
fi

resume_args=()
if [[ "$resume" -eq 1 ]]; then
  latest_checkpoint="$(
    find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
      | sort -t- -k2,2n \
      | tail -n 1
  )"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "--resume requested but no checkpoint-* directory exists in $OUTPUT_DIR" >&2
    exit 2
  fi
  resume_args=(--resume_from_checkpoint "${OUTPUT_DIR}/${latest_checkpoint}")
  echo "resuming from ${OUTPUT_DIR}/${latest_checkpoint}"
fi

# ms-swift's default loss scale trains assistant messages while masking user
# messages and tool responses. This preserves tool calls and final answers
# without training on observations or hidden reasoning.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE=8 \
"$SWIFT_BIN" sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$train_data" \
  --split_dataset_ratio 0 \
  --add_version false \
  --check_model false \
  --torch_dtype bfloat16 \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_length 8192 \
  --gradient_checkpointing true \
  --packing false \
  --attn_impl flash_attn \
  --loss_scale default \
  --save_strategy epoch \
  --save_total_limit 3 \
  --seed 42 \
  --data_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  "${resume_args[@]}"

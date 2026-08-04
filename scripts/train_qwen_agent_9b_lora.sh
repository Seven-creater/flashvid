#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-swift}"
SWIFT_BIN="${SWIFT_BIN:-${SFT_ENV_DIR}/bin/swift}"
SWIFT_PYTHON="${SWIFT_PYTHON:-${SFT_ENV_DIR}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"
EXPECTED_MODEL_ARTIFACT_SHA256="${EXPECTED_MODEL_ARTIFACT_SHA256:-5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/eval/qwen_agent_search/sft_checkpoints/qwen35_9b_lora}"
FORMAL_OUTPUT_DIR="$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

train_data=""
resume=0
smoke=0
output_dir_explicit=0

usage() {
  echo "usage: $0 --train-data FILE [--output-dir DIR] [--resume] [--smoke]" >&2
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
      output_dir_explicit=1
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ "$smoke" -eq 1 ]]; then
  [[ "$output_dir_explicit" -eq 1 ]] || {
    echo "--smoke requires an explicit independent --output-dir" >&2
    exit 2
  }
  [[ "$resume" -eq 0 ]] || {
    echo "--smoke cannot be combined with --resume" >&2
    exit 2
  }
  [[ "$OUTPUT_DIR" != "$FORMAL_OUTPUT_DIR" ]] || {
    echo "--smoke output directory must differ from the formal training output directory" >&2
    exit 2
  }
fi

[[ -f "$train_data" ]] || { echo "training JSONL not found: $train_data" >&2; exit 2; }
[[ -e "$MODEL_PATH" ]] || { echo "Qwen3.5-9B model not found: $MODEL_PATH" >&2; exit 2; }
[[ -x "$SWIFT_BIN" && -x "$SWIFT_PYTHON" ]] || {
  echo "ms-swift environment not found at $SFT_ENV_DIR" >&2
  exit 2
}
if [[ "$smoke" -eq 1 ]]; then
  resolved_output_dir="$($SWIFT_PYTHON -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT_DIR")"
  resolved_formal_output_dir="$($SWIFT_PYTHON -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$FORMAL_OUTPUT_DIR")"
  case "${resolved_output_dir}/" in
    "${resolved_formal_output_dir}/"*)
      echo "--smoke output directory must resolve outside the formal training output directory" >&2
      exit 2
      ;;
  esac
  case "${resolved_formal_output_dir}/" in
    "${resolved_output_dir}/"*)
      echo "--smoke output directory cannot contain the formal training output directory" >&2
      exit 2
      ;;
  esac
  if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "--smoke requires a new or empty output directory: $OUTPUT_DIR" >&2
    exit 2
  fi
fi
[[ "$EXPECTED_MODEL_ARTIFACT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "EXPECTED_MODEL_ARTIFACT_SHA256 must be a SHA-256" >&2
  exit 2
}
gpu_count="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
[[ "$gpu_count" -ge 1 && $((32 % gpu_count)) -eq 0 ]] || {
  echo "effective batch 32 must be divisible by selected GPU count: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
gradient_accumulation_steps=$((32 / gpu_count))

"$SWIFT_PYTHON" -c \
  'import importlib.metadata as m
version = m.version("ms-swift")
assert version == "4.4.2", f"expected ms-swift 4.4.2, found {version}"'
"$SWIFT_PYTHON" - "$train_data" <<'PY'
import json
import sys

rows = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"row {line_number}: missing messages")
        trainable = 0
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                if not isinstance(message.get("loss"), bool):
                    raise SystemExit(f"row {line_number}: assistant loss mask missing")
                trainable += int(message["loss"])
            elif "loss" in message:
                raise SystemExit(f"row {line_number}: loss set on non-assistant role")
        if trainable < 1:
            raise SystemExit(f"row {line_number}: no trainable assistant target")
        rows += 1
if rows == 0:
    raise SystemExit("SFT data is empty")
print(f"validated {rows} SFT rows with explicit assistant loss masks")
PY

# Bind training to the exact frozen 9B artifact and prove that ms-swift's real
# Qwen3.5 template honors each request-level assistant loss flag.  Both gates
# run before the script reserves any GPU process.
mkdir -p "$OUTPUT_DIR/preflight"
"$SWIFT_PYTHON" "$PROJECT_DIR/scripts/fingerprint_model_artifact.py" \
  --model-root "$MODEL_PATH" \
  --expected-sha256 "$EXPECTED_MODEL_ARTIFACT_SHA256" \
  --output "$OUTPUT_DIR/preflight/model_artifact.json"
"$SWIFT_PYTHON" "$PROJECT_DIR/scripts/verify_swift_loss_mask.py" \
  --sft-data "$train_data" \
  --model "$MODEL_PATH" \
  --samples 3 \
  --output "$OUTPUT_DIR/preflight/swift_loss_mask.json"

active_gpu_pids="$(
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -u
)"
if [[ -n "$active_gpu_pids" ]]; then
  echo "selected GPUs must be idle before Qwen3.5-9B SFT; active PIDs:" >&2
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
    echo "--resume requested but no checkpoint-* exists in $OUTPUT_DIR" >&2
    exit 2
  fi
  resume_args=(--resume_from_checkpoint "${OUTPUT_DIR}/${latest_checkpoint}")
fi

training_length_args=(
  --num_train_epochs 3
  --save_strategy epoch
  --save_total_limit 3
)
if [[ "$smoke" -eq 1 ]]; then
  training_length_args=(
    --max_steps 1
    --save_strategy no
  )
fi

# gpu_count x batch 1 x accumulation (32/gpu_count) = effective batch 32.
# Per-message `loss` masks in the JSONL restrict supervision to approved
# plan/tool/memory/stop/final assistant turns.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE="$gpu_count" \
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
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_length 16384 \
  --gradient_checkpointing true \
  --packing false \
  --template_backend swift \
  --loss_scale default \
  --strict true \
  --seed 42 \
  --data_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  "${training_length_args[@]}" \
  "${resume_args[@]}"

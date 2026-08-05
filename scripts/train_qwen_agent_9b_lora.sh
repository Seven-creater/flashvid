#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-qwen35-sft-cu124}"
SWIFT_BIN="${SWIFT_BIN:-${SFT_ENV_DIR}/bin/swift}"
SWIFT_PYTHON="${SWIFT_PYTHON:-${SFT_ENV_DIR}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"
EXPECTED_MODEL_ARTIFACT_SHA256="${EXPECTED_MODEL_ARTIFACT_SHA256:-5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/eval/qwen_agent_search/sft_checkpoints/qwen35_9b_lora}"
FORMAL_OUTPUT_DIR=""
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-43008}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

train_data=""
resume=0
smoke=0
output_dir_explicit=0
formal_output_dir_explicit=0
smoke_report=""
release_project_services=0
load_weights_preflight=0

usage() {
  echo "usage: $0 --train-data FILE [--output-dir DIR] [--resume] [--smoke --formal-output-dir DIR | --smoke-report FILE] [--release-project-services] [--load-weights-preflight]" >&2
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
    --formal-output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      FORMAL_OUTPUT_DIR="$2"
      formal_output_dir_explicit=1
      shift 2
      ;;
    --smoke-report)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      smoke_report="$2"
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
    --release-project-services)
      release_project_services=1
      shift
      ;;
    --load-weights-preflight)
      load_weights_preflight=1
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
  [[ "$formal_output_dir_explicit" -eq 1 ]] || {
    echo "--smoke requires --formal-output-dir to bind the later formal run" >&2
    exit 2
  }
  [[ -z "$smoke_report" ]] || {
    echo "--smoke cannot be combined with --smoke-report" >&2
    exit 2
  }
  [[ "$OUTPUT_DIR" != "$FORMAL_OUTPUT_DIR" ]] || {
    echo "--smoke output directory must differ from the formal training output directory" >&2
    exit 2
  }
fi
if [[ "$smoke" -eq 0 ]]; then
  [[ "$formal_output_dir_explicit" -eq 0 ]] || {
    echo "--formal-output-dir is only valid with --smoke" >&2
    exit 2
  }
  [[ -n "$smoke_report" ]] || {
    echo "formal training requires --smoke-report from a passed bound smoke run" >&2
    exit 2
  }
fi
if [[ "$load_weights_preflight" -eq 1 && "$smoke" -ne 1 ]]; then
  echo "--load-weights-preflight is only valid with --smoke" >&2
  exit 2
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
if [[ "$smoke" -eq 0 ]]; then
  "$SWIFT_PYTHON" "$PROJECT_DIR/scripts/qwen_sft_smoke_gate.py" check \
    --report "$smoke_report" \
    --formal-output-dir "$OUTPUT_DIR" \
    --train-data "$train_data" \
    --base-model-artifact-sha256 "$EXPECTED_MODEL_ARTIFACT_SHA256"
fi

# The server cannot reach huggingface.co. Training uses the frozen local model;
# the mirror remains available for harmless metadata lookups by dependencies.
export HF_ENDPOINT
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ "$release_project_services" -eq 1 ]]; then
  # The helper has a strict port allowlist, same-UID check, vLLM command check,
  # and an environment ownership marker.  A foreign or ambiguous PID aborts.
  bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" 8200
  bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" 8201
fi

gpu_free_memory_ok() {
  local devices=$1
  local device memory
  IFS=',' read -ra requested_devices <<<"$devices"
  for device in "${requested_devices[@]}"; do
    memory="$(
      nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d '[:space:]'
    )"
    [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_GPU_MEMORY_MIB" ]] || return 1
  done
}

eight_gpu_set="0,1,2,3,4,5,6,7"
four_gpu_set="4,5,6,7"
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  if gpu_free_memory_ok "$eight_gpu_set"; then
    CUDA_VISIBLE_DEVICES="$eight_gpu_set"
  elif gpu_free_memory_ok "$four_gpu_set"; then
    CUDA_VISIBLE_DEVICES="$four_gpu_set"
  else
    echo "no 4/8-GPU SFT layout with at least 42 GiB free per GPU" >&2
    exit 2
  fi
fi
case "$CUDA_VISIBLE_DEVICES" in
  "$eight_gpu_set") gpu_count=8; gradient_accumulation_steps=4 ;;
  "$four_gpu_set") gpu_count=4; gradient_accumulation_steps=8 ;;
  *)
    echo "CUDA_VISIBLE_DEVICES must be $eight_gpu_set or $four_gpu_set" >&2
    exit 2
    ;;
esac
gpu_free_memory_ok "$CUDA_VISIBLE_DEVICES" || {
  echo "each selected GPU must provide at least 42 GiB free: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
# A small foreign process is allowed only when the measured free-memory gate
# still passes. The launcher never signals or otherwise manages foreign PIDs.

"$SWIFT_PYTHON" -c \
  'import importlib.metadata as m
version = m.version("ms-swift")
assert version == "4.4.2", f"expected ms-swift 4.4.2, found {version}"'
mkdir -p "$OUTPUT_DIR/preflight"
training_data="$train_data"
loss_mask_samples=3
if [[ "$smoke" -eq 1 ]]; then
  training_data="$OUTPUT_DIR/preflight/smoke_one_sample.jsonl"
  smoke_rank_padding=$((gpu_count * gradient_accumulation_steps))
  "$SWIFT_PYTHON" - "$train_data" "$training_data" "$smoke_rank_padding" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
copies = int(sys.argv[3])
if copies < 1:
    raise SystemExit("smoke rank padding must be positive")
with source.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"row {line_number}: expected an object")
        # One unique logical sample is repeated only so every data-parallel rank
        # receives a complete gradient-accumulation window for the 1-step smoke.
        encoded = json.dumps(value, ensure_ascii=False) + "\n"
        output.write_text(encoded * copies, encoding="utf-8")
        break
    else:
        raise SystemExit("SFT data is empty")
PY
  loss_mask_samples=1
fi

"$SWIFT_PYTHON" - "$training_data" <<'PY'
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
"$SWIFT_PYTHON" "$PROJECT_DIR/scripts/fingerprint_model_artifact.py" \
  --model-root "$MODEL_PATH" \
  --expected-sha256 "$EXPECTED_MODEL_ARTIFACT_SHA256" \
  --output "$OUTPUT_DIR/preflight/model_artifact.json"
"$SWIFT_PYTHON" "$PROJECT_DIR/scripts/verify_swift_loss_mask.py" \
  --sft-data "$training_data" \
  --model "$MODEL_PATH" \
  --samples "$loss_mask_samples" \
  --output "$OUTPUT_DIR/preflight/swift_loss_mask.json"
model_preflight_args=()
if [[ "$load_weights_preflight" -eq 1 ]]; then
  model_preflight_args=(--load-weights)
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$SWIFT_PYTHON" "$PROJECT_DIR/scripts/preflight_qwen35_9b_lora.py" \
  --model "$MODEL_PATH" \
  --expected-gpu-count "$gpu_count" \
  --output "$OUTPUT_DIR/preflight/qwen35_9b_runtime.json" \
  "${model_preflight_args[@]}"

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
training_warmup_ratio=0.05
if [[ "$smoke" -eq 1 ]]; then
  # A one-step smoke cannot spend its only optimizer step at zero learning
  # rate. Formal three-epoch training keeps the registered 0.05 warmup.
  training_warmup_ratio=0
  training_length_args=(
    --max_steps 1
    --logging_strategy steps
    --logging_steps 1
    --logging_first_step true
    --save_strategy steps
    --save_steps 1
    --save_total_limit 1
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
  --dataset "$training_data" \
  --split_dataset_ratio 0 \
  --add_version false \
  --check_model false \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
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
  --warmup_ratio "$training_warmup_ratio" \
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

if [[ "$smoke" -eq 1 ]]; then
  "$SWIFT_PYTHON" "$PROJECT_DIR/scripts/verify_qwen35_lora_smoke.py" \
    --output-dir "$OUTPUT_DIR" \
    --report "$OUTPUT_DIR/preflight/training_update.json"
  "$SWIFT_PYTHON" "$PROJECT_DIR/scripts/qwen_sft_smoke_gate.py" bind \
    --report "$OUTPUT_DIR/preflight/training_update.json" \
    --smoke-output-dir "$OUTPUT_DIR" \
    --formal-output-dir "$FORMAL_OUTPUT_DIR" \
    --train-data "$train_data" \
    --base-model-artifact-sha256 "$EXPECTED_MODEL_ARTIFACT_SHA256"
fi

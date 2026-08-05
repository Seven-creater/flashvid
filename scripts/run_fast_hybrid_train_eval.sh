#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"
SERVICE_OWNER_PROJECT_DIR="${SERVICE_OWNER_PROJECT_DIR:-$PROJECT_DIR}"

PYTHON="${PYTHON:-/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PYTHON")/vllm}"
SFT_PYTHON="${SFT_PYTHON:-$PROJECT_DIR/.venv-qwen35-sft-cu124/bin/python}"
SFT_ENV_DIR="${SFT_ENV_DIR:-$(cd "$(dirname "$SFT_PYTHON")/.." && pwd)}"
SWIFT_PYTHON="${SWIFT_PYTHON:-$SFT_PYTHON}"
SWIFT_BIN="${SWIFT_BIN:-$SFT_ENV_DIR/bin/swift}"
export SFT_ENV_DIR SWIFT_PYTHON SWIFT_BIN
export VLLM_BIN
[[ -x "$VLLM_BIN" ]] || {
  echo "vLLM executable not found: $VLLM_BIN" >&2
  exit 2
}
MODEL_PATH="${MODEL_PATH:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"
MODEL_SHA="${MODEL_SHA:-5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60}"
CONFIG="${CONFIG:-configs/experiments/fast_hybrid_eva_sft.json}"
CONFIG_SHA="${CONFIG_SHA:-74c2e3a7ecf0dbae30d00037bc748fe0861fdba2501fcf94efeb8aa58856e830}"
ROOT="${ROOT:-results/eval/fast_hybrid_eva_sft}"
DIRECT_TEST_AFTER_TRAIN="${DIRECT_TEST_AFTER_TRAIN:-0}"
SFT_SOURCE="$ROOT/sft_data/fast_hybrid_sft.jsonl"
SFT_DATA="$ROOT/sft_data/fast_hybrid_sft_max16384.jsonl"
SFT_LENGTH_AUDIT="$ROOT/sft_data/fast_hybrid_sft_max16384_audit.json"
FORMAL_DIR="$ROOT/checkpoints/qwen35_9b_lora"
CHECKPOINT_AUDIT="$FORMAL_DIR/checkpoint_audit.json"
PROTOCOL="$ROOT/frozen/evaluation_protocol.json"
WINNER="$ROOT/dev_eval/winner.json"
if [[ "$DIRECT_TEST_AFTER_TRAIN" == "1" ]]; then
  WINNER="$ROOT/final_test/final_epoch_winner.json"
elif [[ "$DIRECT_TEST_AFTER_TRAIN" != "0" ]]; then
  echo "DIRECT_TEST_AFTER_TRAIN must be 0 or 1" >&2
  exit 2
fi
STATE="$ROOT/autopilot/train_eval_state.jsonl"
TEST_CANDIDATE_ROOT="${TEST_CANDIDATE_ROOT:-results/eval/fast_hybrid_eva/frozen_direct}"
SERVICES_STARTED=0
CURRENT_STAGE=initializing

mkdir -p "$ROOT/frozen" "$ROOT/checkpoints/frozen" "$ROOT/dev_eval" \
  "$ROOT/final_test" "$ROOT/autopilot" "$ROOT/logs"

mark_stage() {
  "$PYTHON" - "$STATE" "$1" "$2" <<'PY'
import datetime, json, os, sys
path, stage, status = sys.argv[1:]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": datetime.datetime.now().astimezone().isoformat(), "stage": stage, "status": status}) + "\n")
    handle.flush(); os.fsync(handle.fileno())
PY
}

stop_owned_services() {
  if [[ "$SERVICE_OWNER_PROJECT_DIR" != "$PROJECT_DIR" ]]; then
    bash "$SERVICE_OWNER_PROJECT_DIR/scripts/stop_qwen_agent.sh" 8200
    bash "$SERVICE_OWNER_PROJECT_DIR/scripts/stop_qwen_agent.sh" 8201
  fi
  bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" 8200
  bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" 8201
  SERVICES_STARTED=0
}

cleanup() {
  local code=$?
  if (( SERVICES_STARTED == 1 )); then
    stop_owned_services || true
  fi
  if (( code != 0 )); then
    mark_stage "$CURRENT_STAGE" failed || true
  fi
  exit "$code"
}
trap cleanup EXIT

json_field() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value=value[key]
print(value)
PY
}

evaluation_passed() {
  local path=$1 phase=$2 mode=$3 expected_served=${4:-}
  [[ -f "$path/evaluation_run.json" ]] || return 1
  "$PYTHON" - "$path/evaluation_run.json" "$phase" "$mode" "$PROTOCOL_SHA" "$expected_served" <<'PY' >/dev/null
import hashlib, json, sys
from pathlib import Path
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value.get("status") == "passed"
assert value.get("phase") == sys.argv[2]
assert value.get("mode") == sys.argv[3]
assert value.get("evaluation_protocol_sha256") == sys.argv[4]
assert not sys.argv[5] or value.get("served_model_sha256") == sys.argv[5]
files=value.get("files")
assert isinstance(files, dict) and set(files) == {"lvbench", "lsdbench", "cgbench"}
for reference in files.values():
    path=Path(reference["path"])
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
PY
}

model_healthy() {
  local port=$1 expected=$2 response
  response=$(curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null) || return 1
  "$PYTHON" -c 'import json,sys; expected=sys.argv[1]; value=json.load(sys.stdin); ids={str(x.get("id")) for x in value.get("data",[]) if isinstance(x,dict)}; raise SystemExit(0 if expected in ids else 1)' "$expected" <<<"$response"
}

wait_model() {
  local port=$1 expected=$2 deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if model_healthy "$port" "$expected"; then return 0; fi
    sleep 5
  done
  echo "model did not become healthy: port=$port model=$expected" >&2
  tail -n 120 "$PROJECT_DIR"/logs/*"_${port}.log" 2>/dev/null || true
  return 1
}

start_base_pair() {
  stop_owned_services
  CUDA_DEVICES=0,1,2,3 ALLOW_SHARED_GPUS=1 \
    bash "$PROJECT_DIR/scripts/launch_qwen_agent_service.sh" "$MODEL_PATH" Qwen3.5-9B 8200 4
  SERVICES_STARTED=1
  CUDA_DEVICES=4,5,6,7 ALLOW_SHARED_GPUS=0 \
    bash "$PROJECT_DIR/scripts/launch_qwen_agent_service.sh" "$MODEL_PATH" Qwen3.5-9B 8201 4
  wait_model 8200 Qwen3.5-9B
  wait_model 8201 Qwen3.5-9B
}

ensure_base_pair() {
  if model_healthy 8200 Qwen3.5-9B && model_healthy 8201 Qwen3.5-9B; then
    return 0
  fi
  start_base_pair
}

start_checkpoint_pair() {
  local checkpoint_config=$1 adapter served
  adapter=$(json_field "$checkpoint_config" adapter.path)
  served=$(json_field "$checkpoint_config" served_name)
  stop_owned_services
  CUDA_DEVICES=0,1,2,3 ALLOW_SHARED_GPUS=1 LORA_PATH="$adapter" LORA_SERVED_NAME="$served" \
    bash "$PROJECT_DIR/scripts/launch_qwen_agent_service.sh" "$MODEL_PATH" Qwen3.5-9B 8200 4
  SERVICES_STARTED=1
  CUDA_DEVICES=4,5,6,7 ALLOW_SHARED_GPUS=0 LORA_PATH="$adapter" LORA_SERVED_NAME="$served" \
    bash "$PROJECT_DIR/scripts/launch_qwen_agent_service.sh" "$MODEL_PATH" Qwen3.5-9B 8201 4
  wait_model 8200 "$served"
  wait_model 8201 "$served"
}

unique_match() {
  local label=$1; shift
  local matches=( "$@" )
  if ((${#matches[@]} != 1)); then
    echo "$label must resolve to exactly one file, found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

CURRENT_STAGE=freeze_eval_protocol
mark_stage "$CURRENT_STAGE" started
bash scripts/recover_lvbench_sft_inputs.sh
DEV_LV=$(unique_match dev-lv "$ROOT"/candidates/dev/lvbench/*uniform32.jsonl)
DEV_LSD=$(unique_match dev-lsd "$ROOT"/candidates/dev/lsdbench/*uniform32.jsonl)
DEV_CG=$(unique_match dev-cg "$ROOT"/candidates/dev/cgbench/*uniform32.jsonl)
TEST_LV=$(unique_match test-lv-repaired "$ROOT"/candidates/test/lvbench/*uniform32_repaired.jsonl)
TEST_LSD=$(unique_match test-lsd "$TEST_CANDIDATE_ROOT"/lsdbench/*uniform32.jsonl)
TEST_CG=$(unique_match test-cg "$TEST_CANDIDATE_ROOT"/cgbench/*uniform32.jsonl)
"$PYTHON" scripts/freeze_fast_hybrid_eval_protocol.py \
  --experiment-config "$CONFIG" --expected-experiment-config-sha256 "$CONFIG_SHA" \
  --candidate "dev:lvbench=$DEV_LV" --candidate "dev:lsdbench=$DEV_LSD" \
  --candidate "dev:cgbench=$DEV_CG" --candidate "test:lvbench=$TEST_LV" \
  --candidate "test:lsdbench=$TEST_LSD" --candidate "test:cgbench=$TEST_CG" \
  --output "$PROTOCOL"
PROTOCOL_SHA=$(sha256sum "$PROTOCOL" | awk '{print $1}')
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=teacher_dev
mark_stage "$CURRENT_STAGE" started
if ! evaluation_passed "$ROOT/dev_eval/teacher" dev teacher "$MODEL_SHA"; then
  ensure_base_pair
  "$PYTHON" scripts/run_fast_hybrid_sft_eval.py \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
    --phase dev --mode teacher --run-id teacher_dev_v1 \
    --output-root "$ROOT/dev_eval/teacher" --python "$PYTHON" \
    --repo-root "$PROJECT_DIR" --resume
fi
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=training_smoke
mark_stage "$CURRENT_STAGE" started
stop_owned_services
[[ -f "$SFT_SOURCE" ]] || { echo "approved SFT data is missing: $SFT_SOURCE" >&2; exit 1; }
"$SFT_PYTHON" scripts/filter_swift_sft_length.py \
  --input "$SFT_SOURCE" --output "$SFT_DATA" --audit "$SFT_LENGTH_AUDIT" \
  --model "$MODEL_PATH" --model-artifact-sha256 "$MODEL_SHA" --max-length 16384
SMOKE_POINTER="$ROOT/checkpoints/active_smoke_report.txt"
SMOKE_REPORT=""
if [[ -f "$SMOKE_POINTER" ]]; then
  SMOKE_REPORT=$(<"$SMOKE_POINTER")
  if ! "$SFT_PYTHON" scripts/qwen_sft_smoke_gate.py check \
      --report "$SMOKE_REPORT" --formal-output-dir "$FORMAL_DIR" \
      --train-data "$SFT_DATA" --base-model-artifact-sha256 "$MODEL_SHA" >/dev/null 2>&1; then
    SMOKE_REPORT=""
  fi
fi
if [[ -z "$SMOKE_REPORT" ]]; then
  SMOKE_DIR="$ROOT/checkpoints/smoke_qwen35_9b_lora"
  if [[ -d "$SMOKE_DIR" && -n "$(find "$SMOKE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    SMOKE_DIR="${SMOKE_DIR}_retry_$(date +%Y%m%d_%H%M%S)"
  fi
  bash scripts/train_qwen_agent_9b_lora.sh \
    --train-data "$SFT_DATA" --output-dir "$SMOKE_DIR" \
    --smoke --formal-output-dir "$FORMAL_DIR" \
    --release-project-services --load-weights-preflight
  SMOKE_REPORT="$SMOKE_DIR/preflight/training_update.json"
  printf '%s\n' "$SMOKE_REPORT" > "$SMOKE_POINTER"
fi
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=formal_training
mark_stage "$CURRENT_STAGE" started
if ! "$SFT_PYTHON" scripts/verify_sft_checkpoints.py \
    --checkpoint-root "$FORMAL_DIR" --output "$CHECKPOINT_AUDIT" >/dev/null 2>&1; then
  resume_args=()
  partial=( "$FORMAL_DIR"/checkpoint-* )
  if ((${#partial[@]})); then resume_args=(--resume); fi
  bash scripts/train_qwen_agent_9b_lora.sh \
    --train-data "$SFT_DATA" --output-dir "$FORMAL_DIR" \
    --smoke-report "$SMOKE_REPORT" "${resume_args[@]}"
fi
"$SFT_PYTHON" scripts/verify_sft_checkpoints.py \
  --checkpoint-root "$FORMAL_DIR" --output "$CHECKPOINT_AUDIT"
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=freeze_checkpoints
mark_stage "$CURRENT_STAGE" started
CHECKPOINT_CONFIGS=()
while IFS=$'\t' read -r checkpoint_name epoch global_step; do
  checkpoint_path="$FORMAL_DIR/$checkpoint_name"
  checkpoint_config="$ROOT/checkpoints/frozen/$checkpoint_name.json"
  served_name="Qwen3.5-9B-FastHybrid-SFT-E${epoch}"
  "$PYTHON" scripts/freeze_fast_hybrid_sft_checkpoint.py \
    --checkpoint-id "$checkpoint_name" --epoch "$epoch" --global-step "$global_step" \
    --adapter-root "$checkpoint_path" --base-artifact-sha256 "$MODEL_SHA" \
    --train-data "$SFT_DATA" --experiment-config "$CONFIG" \
    --expected-experiment-config-sha256 "$CONFIG_SHA" --served-name "$served_name" \
    --base-url http://127.0.0.1:8200/v1 --base-url http://127.0.0.1:8201/v1 \
    --output "$checkpoint_config"
  CHECKPOINT_CONFIGS+=( "$checkpoint_config" )
done < <("$PYTHON" - "$CHECKPOINT_AUDIT" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value["status"] == "complete" and len(value["checkpoints"]) == 3
for item in value["checkpoints"]:
    print(f'{item["name"]}\t{round(float(item["epoch"]))}\t{item["global_step"]}')
PY
)
test "${#CHECKPOINT_CONFIGS[@]}" -eq 3
mark_stage "$CURRENT_STAGE" passed

if [[ "$DIRECT_TEST_AFTER_TRAIN" == "1" ]]; then
  CURRENT_STAGE=select_final_epoch_for_direct_test
  mark_stage "$CURRENT_STAGE" started
  FINAL_CHECKPOINT_CONFIG="${CHECKPOINT_CONFIGS[-1]}"
  "$PYTHON" - "$PROTOCOL" "$PROTOCOL_SHA" "$FINAL_CHECKPOINT_CONFIG" \
    "$CHECKPOINT_AUDIT" "$ROOT/dev_eval/teacher" "$WINNER" <<'PY'
import json, sys
from pathlib import Path

from flashvid_eval.fast_hybrid_eval_protocol import freeze_json, load_protocol, sha256_file
from scripts.run_fast_hybrid_sft_eval import _load_checkpoint

protocol_path, protocol_sha, config_path, audit_path, teacher_root, output_path = map(Path, sys.argv[1:])
protocol_sha = str(protocol_sha)
protocol = load_protocol(protocol_path, protocol_sha)
checkpoint = _load_checkpoint(config_path, protocol)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
expected_epoch = max(int(round(float(item["epoch"]))) for item in audit["checkpoints"])
if int(checkpoint["epoch"]) != expected_epoch:
    raise RuntimeError("final checkpoint config is not the maximum trained epoch")
payload = {
    "schema_version": 1,
    "kind": "fast_hybrid_sft_winner",
    "status": "passed",
    "evaluation_protocol": {"path": str(protocol_path.resolve()), "sha256": protocol_sha},
    "evaluation_protocol_sha256": protocol_sha,
    "teacher_run": {
        "path": str(teacher_root.resolve()),
        "metadata_sha256": sha256_file(teacher_root / "evaluation_run.json"),
    },
    "selection": {
        "policy": "final_epoch_direct_test_no_dev_selection",
        "dev_evaluated": False,
        "selected": {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
        },
    },
    "checkpoint_config": {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "served_stack_sha256": checkpoint["served_stack_sha256"],
    },
    "checkpoint_dev_run": None,
}
freeze_json(output_path, payload)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  mark_stage "$CURRENT_STAGE" passed
else
CURRENT_STAGE=checkpoint_dev
mark_stage "$CURRENT_STAGE" started
CHECKPOINT_SELECTION_ARGS=()
for checkpoint_config in "${CHECKPOINT_CONFIGS[@]}"; do
  checkpoint_id=$(json_field "$checkpoint_config" checkpoint_id)
  checkpoint_served_sha=$(json_field "$checkpoint_config" served_stack_sha256)
  dev_root="$ROOT/dev_eval/$checkpoint_id"
  if ! evaluation_passed "$dev_root" dev checkpoint "$checkpoint_served_sha"; then
    start_checkpoint_pair "$checkpoint_config"
    "$PYTHON" scripts/run_fast_hybrid_sft_eval.py \
      --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
      --phase dev --mode checkpoint --checkpoint-config "$checkpoint_config" \
      --run-id "${checkpoint_id}_dev_v1" --output-root "$dev_root" \
      --python "$PYTHON" --repo-root "$PROJECT_DIR" --resume
    stop_owned_services
  fi
  CHECKPOINT_SELECTION_ARGS+=( --checkpoint "$checkpoint_config=$dev_root" )
done
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=select_winner
mark_stage "$CURRENT_STAGE" started
set +e
"$PYTHON" scripts/select_fast_hybrid_sft_winner.py \
  --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --teacher-run-root "$ROOT/dev_eval/teacher" \
  "${CHECKPOINT_SELECTION_ARGS[@]}" --output "$WINNER"
selection_code=$?
set -e
if (( selection_code != 0 && selection_code != 2 )); then exit "$selection_code"; fi
mark_stage "$CURRENT_STAGE" passed

if (( selection_code == 2 )); then
  CURRENT_STAGE=blocked_no_winner
  "$PYTHON" scripts/summarize_fast_hybrid_sft.py \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
    --winner "$WINNER" --output-root "$ROOT"
  mark_stage "$CURRENT_STAGE" passed
  CURRENT_STAGE=complete
  mark_stage "$CURRENT_STAGE" passed
  trap - EXIT
  exit 0
fi
fi

CURRENT_STAGE=winner_test
mark_stage "$CURRENT_STAGE" started
WINNER_CONFIG=$(json_field "$WINNER" checkpoint_config.path)
WINNER_SERVED_SHA=$(json_field "$WINNER_CONFIG" served_stack_sha256)
if ! evaluation_passed "$ROOT/final_test/winner" test checkpoint "$WINNER_SERVED_SHA"; then
  start_checkpoint_pair "$WINNER_CONFIG"
  "$PYTHON" scripts/run_fast_hybrid_sft_eval.py \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
    --phase test --mode checkpoint --winner "$WINNER" --run-id winner_test_v1 \
    --output-root "$ROOT/final_test/winner" --python "$PYTHON" \
    --repo-root "$PROJECT_DIR" --resume
  stop_owned_services
fi
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=final_summary
mark_stage "$CURRENT_STAGE" started
"$PYTHON" scripts/summarize_fast_hybrid_sft.py \
  --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --winner "$WINNER" --test-root "$ROOT/final_test/winner" --output-root "$ROOT"
mark_stage "$CURRENT_STAGE" passed
CURRENT_STAGE=complete
mark_stage "$CURRENT_STAGE" passed
trap - EXIT

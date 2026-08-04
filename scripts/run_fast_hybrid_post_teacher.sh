#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"
SERVICE_OWNER_PROJECT_DIR="${SERVICE_OWNER_PROJECT_DIR:-$PROJECT_DIR}"

PYTHON="${PYTHON:-/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python}"
CONFIG="${CONFIG:-configs/experiments/fast_hybrid_eva_sft.json}"
CONFIG_SHA="${CONFIG_SHA:-74c2e3a7ecf0dbae30d00037bc748fe0861fdba2501fcf94efeb8aa58856e830}"
TRAIN600="${TRAIN600:-results/eval/qwen_agent_search/frozen/train600.jsonl}"
TRAIN600_SHA="${TRAIN600_SHA:-3995454d973aeb6efe6887e821cd5197e0f17a5b9cc32d7c719e6583b487e6c0}"
ROOT="${ROOT:-results/eval/fast_hybrid_eva_sft}"
CTRL="$ROOT/trajectories/control_v2"
BASE_SPECS="$CTRL/base_plan.jsonl"
PRE_BASE="$ROOT/trajectories/prejudge/base"
PRE_RESCUE="$ROOT/trajectories/prejudge/rescue"
COMP="$ROOT/trajectories/compression_v2"
SFT_DIR="$ROOT/sft_data"
MODEL_SHA="${MODEL_SHA:-5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60}"
MODEL_PATH="${MODEL_PATH:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"

LV_MANIFEST=/data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/splits/lvbench_train.jsonl
LV_SHA=5e0ec526ba3645a0c230943fbed50e0c302a7a519dfb29371f54777875b29cc2
LSD_MANIFEST=/data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/splits/lsdbench_train.jsonl
LSD_SHA=5bd1d9c6be5bdc44c7a9bd796e2110863b4df8e1b3435d84bcff51ca2d491cd1
CG_MANIFEST=/data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/splits/cgbench_train.jsonl
CG_SHA=1c7d631c17cfbd578056a2616539bd4e20d24acf1c22ead86ada69197dcaf6c8
LV_VIDEO=/data02/pretrained_model/cvr_learn/cvr_data/06_lvbench
LSD_VIDEO=/data02/pretrained_model/cvr_learn/cvr_data/07_lsdbench/videos
CG_VIDEO=/data02/pretrained_model/cvr_learn/cvr_data/04_cg-bench/videos_partial

STATE="$ROOT/autopilot/post_teacher_state.jsonl"
mkdir -p "$(dirname "$STATE")" "$PRE_BASE" "$COMP/replays" "$COMP/ready" "$SFT_DIR"

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

model_healthy() {
  local port=$1 expected=$2 response
  response=$(curl --fail --silent --show-error --max-time 5 \
    "http://127.0.0.1:${port}/v1/models" 2>/dev/null) || return 1
  "$PYTHON" -c 'import json,sys; expected=sys.argv[1]; value=json.load(sys.stdin); ids={str(x.get("id")) for x in value.get("data",[]) if isinstance(x,dict)}; raise SystemExit(0 if expected in ids else 1)' \
    "$expected" <<<"$response"
}

wait_model() {
  local port=$1 expected=$2 deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if model_healthy "$port" "$expected"; then return 0; fi
    sleep 5
  done
  echo "model did not become healthy: port=$port model=$expected" >&2
  return 1
}

ensure_base_service() {
  local port=$1 devices=$2 allow_shared=$3
  if model_healthy "$port" Qwen3.5-9B; then return 0; fi
  echo "base service health check failed; safely restarting owned port $port" >&2
  if [[ "$SERVICE_OWNER_PROJECT_DIR" != "$PROJECT_DIR" ]]; then
    bash "$SERVICE_OWNER_PROJECT_DIR/scripts/stop_qwen_agent.sh" "$port"
  fi
  bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" "$port"
  CUDA_DEVICES="$devices" ALLOW_SHARED_GPUS="$allow_shared" \
    bash "$PROJECT_DIR/scripts/launch_qwen_agent_service.sh" \
      "$MODEL_PATH" Qwen3.5-9B "$port" 4
  if ! wait_model "$port" Qwen3.5-9B; then
    bash "$PROJECT_DIR/scripts/stop_qwen_agent.sh" "$port" || true
    return 1
  fi
}

ensure_base_services() {
  ensure_base_service 8200 0,1,2,3 1
  ensure_base_service 8201 4,5,6,7 0
}
CURRENT_STAGE=initializing
trap 'code=$?; if ((code != 0)); then mark_stage "$CURRENT_STAGE" failed || true; fi; exit $code' EXIT

verify_hashes() {
  printf '%s  %s\n' \
    "$CONFIG_SHA" "$CONFIG" \
    "$TRAIN600_SHA" "$TRAIN600" \
    "$LV_SHA" "$LV_MANIFEST" \
    "$LSD_SHA" "$LSD_MANIFEST" \
    "$CG_SHA" "$CG_MANIFEST" | sha256sum -c -
}

candidate_from_base_teacher() {
  "$PYTHON" - "$ROOT" "$1" <<'PY'
import glob, hashlib, json, sys
from pathlib import Path
root, dataset = sys.argv[1:]
files = sorted(glob.glob(f"{root}/trajectories/raw/base/*/{dataset}/frozen_inputs_{dataset}.json"))
if not files:
    raise SystemExit(f"no Base Teacher frozen inputs for {dataset}")
items = [json.load(open(path, encoding="utf-8"))["candidate_results"] for path in files]
if len({item["sha256"] for item in items}) != 1:
    raise SystemExit(f"{dataset} candidate SHA drift")
for item in items:
    path = Path(item["path"])
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
        print(path); break
else:
    raise SystemExit(f"no unchanged candidate file for {dataset}")
PY
}

CURRENT_STAGE=base_teacher_audit
mark_stage "$CURRENT_STAGE" started
verify_hashes
test -f "$BASE_SPECS"
BASE_RAW=( "$ROOT"/trajectories/raw/base/*/*/*_fast_hybrid_eva.jsonl )
test "${#BASE_RAW[@]}" -eq 36
test "$(wc -l < "$BASE_SPECS")" -eq 7200
test "$(wc -l "${BASE_RAW[@]}" | tail -n 1 | awk '{print $1}')" -eq 7200
BASE_SPEC_SHA=$(sha256sum "$BASE_SPECS" | awk '{print $1}')
TEACHER_AUDIT="$ROOT/run_plans/teacher_${BASE_SPEC_SHA:0:12}_audit.json"
REPAIRED_TEACHER_AUDIT="$ROOT/run_plans/teacher_${BASE_SPEC_SHA:0:12}_repaired_audit.json"
if [[ -f "$REPAIRED_TEACHER_AUDIT" ]]; then
  TEACHER_AUDIT="$REPAIRED_TEACHER_AUDIT"
  "$PYTHON" scripts/repair_fast_hybrid_teacher_provenance.py \
    --plan "$ROOT/run_plans/teacher_${BASE_SPEC_SHA:0:12}.json" \
    --failed-audit "$ROOT/run_plans/teacher_${BASE_SPEC_SHA:0:12}_audit.json" \
    --output-audit "$TEACHER_AUDIT" >/dev/null
fi
"$PYTHON" - "$TEACHER_AUDIT" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value["status"] == "passed" and value["jobs"] == 36 and value["rows"] == 7200 and not value["issues"], value
PY
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=prepare_base
mark_stage "$CURRENT_STAGE" started
"$PYTHON" scripts/prepare_fast_hybrid_judges.py \
  --train600 "$TRAIN600" --expected-train600-sha256 "$TRAIN600_SHA" \
  --specs "$BASE_SPECS" --raw-trajectories "${BASE_RAW[@]}" --output-dir "$PRE_BASE"
BASE_ELIGIBLE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible"])' "$PRE_BASE/summary.json")
test "$(wc -l < "$PRE_BASE/prejudge_completion_index.jsonl")" -eq 7200
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=judge_base
mark_stage "$CURRENT_STAGE" started
if (( BASE_ELIGIBLE > 0 )); then
  ensure_base_services
  "$PYTHON" scripts/launch_fast_hybrid_judge_matrix.py \
    --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
    --specs "$PRE_BASE/judge_specs.jsonl" \
    --trajectories "$PRE_BASE/judge_trajectories.jsonl" \
    --python "$PYTHON" --repo-root "$PROJECT_DIR" \
    --concurrency-per-endpoint 16 --timeout 80 --max-tokens 512 \
    --resume --retry-failed-processes
fi
BASE_JUDGED=( "$ROOT"/trajectories/judged/base/*.jsonl )
if (( BASE_ELIGIBLE == 0 )); then BASE_JUDGED=( "$PRE_BASE/judge_trajectories.jsonl" ); fi
test "${#BASE_JUDGED[@]}" -gt 0
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=plan_rescue
mark_stage "$CURRENT_STAGE" started
"$PYTHON" scripts/control_fast_hybrid_trajectories.py \
  --phase plan-rescue --train600 "$TRAIN600" \
  --expected-train600-sha256 "$TRAIN600_SHA" --config-sha256 "$CONFIG_SHA" \
  --dataset-manifest-sha256 "lvbench=$LV_SHA" \
  --dataset-manifest-sha256 "lsdbench=$LSD_SHA" \
  --dataset-manifest-sha256 "cgbench=$CG_SHA" \
  --trajectories "${BASE_JUDGED[@]}" \
  --prejudge-index "$PRE_BASE/prejudge_completion_index.jsonl" --output-dir "$CTRL"
RESCUE_PLANNED=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["planned"])' "$CTRL/plan-rescue_summary.json")
mark_stage "$CURRENT_STAGE" passed

CAND_LV=$(candidate_from_base_teacher lvbench)
CAND_LSD=$(candidate_from_base_teacher lsdbench)
CAND_CG=$(candidate_from_base_teacher cgbench)
RESCUE_ELIGIBLE=0
if (( RESCUE_PLANNED > 0 )); then
  CURRENT_STAGE=rescue_teacher
  mark_stage "$CURRENT_STAGE" started
  ensure_base_services
  "$PYTHON" scripts/launch_fast_hybrid_teacher_matrix.py \
    --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
    --specs "$CTRL/rescue_plan.jsonl" \
    --candidate-results "lvbench=$CAND_LV" \
    --candidate-results "lsdbench=$CAND_LSD" \
    --candidate-results "cgbench=$CAND_CG" \
    --python "$PYTHON" --repo-root "$PROJECT_DIR" \
    --concurrency-per-endpoint 16 --timeout 80 --resume
  mark_stage "$CURRENT_STAGE" passed

  CURRENT_STAGE=prepare_rescue
  mark_stage "$CURRENT_STAGE" started
  RESCUE_RAW=( "$ROOT"/trajectories/raw/rescue/*/*/*_fast_hybrid_eva.jsonl )
  test "${#RESCUE_RAW[@]}" -gt 0
  mkdir -p "$PRE_RESCUE"
  "$PYTHON" scripts/prepare_fast_hybrid_judges.py \
    --train600 "$TRAIN600" --expected-train600-sha256 "$TRAIN600_SHA" \
    --specs "$CTRL/rescue_plan.jsonl" --raw-trajectories "${RESCUE_RAW[@]}" \
    --output-dir "$PRE_RESCUE"
  RESCUE_ELIGIBLE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible"])' "$PRE_RESCUE/summary.json")
  mark_stage "$CURRENT_STAGE" passed

  CURRENT_STAGE=judge_rescue
  mark_stage "$CURRENT_STAGE" started
  if (( RESCUE_ELIGIBLE > 0 )); then
    ensure_base_services
    "$PYTHON" scripts/launch_fast_hybrid_judge_matrix.py \
      --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
      --specs "$PRE_RESCUE/judge_specs.jsonl" \
      --trajectories "$PRE_RESCUE/judge_trajectories.jsonl" \
      --python "$PYTHON" --repo-root "$PROJECT_DIR" \
      --concurrency-per-endpoint 16 --timeout 80 --max-tokens 512 \
      --resume --retry-failed-processes
  fi
  mark_stage "$CURRENT_STAGE" passed
fi

CURRENT_STAGE=select_stable
mark_stage "$CURRENT_STAGE" started
SELECT_TRAJECTORIES=( "${BASE_JUDGED[@]}" )
SELECT_INDEXES=( "$PRE_BASE/prejudge_completion_index.jsonl" )
if (( RESCUE_PLANNED > 0 )); then
  SELECT_INDEXES+=( "$PRE_RESCUE/prejudge_completion_index.jsonl" )
  RESCUE_JUDGED=( "$ROOT"/trajectories/judged/rescue/*.jsonl )
  if ((${#RESCUE_JUDGED[@]})); then SELECT_TRAJECTORIES+=( "${RESCUE_JUDGED[@]}" ); fi
fi
"$PYTHON" scripts/control_fast_hybrid_trajectories.py \
  --phase select --train600 "$TRAIN600" --expected-train600-sha256 "$TRAIN600_SHA" \
  --config-sha256 "$CONFIG_SHA" \
  --dataset-manifest-sha256 "lvbench=$LV_SHA" \
  --dataset-manifest-sha256 "lsdbench=$LSD_SHA" \
  --dataset-manifest-sha256 "cgbench=$CG_SHA" \
  --trajectories "${SELECT_TRAJECTORIES[@]}" \
  --prejudge-index "${SELECT_INDEXES[@]}" --output-dir "$CTRL"
"$PYTHON" - "$CTRL/select_summary.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
assert value["sft_start_gate"]["passed"] is True, value["sft_start_gate"]
PY
mark_stage "$CURRENT_STAGE" passed

finalize_compression() {
  local markers=( "$COMP"/replays/round_*.jsonl.done ) results=() marker replay_args=()
  for marker in "${markers[@]}"; do results+=( "${marker%.done}" ); done
  if ((${#results[@]})); then replay_args=( --replay-results "${results[@]}" ); fi
  "$PYTHON" scripts/finalize_fast_hybrid_compression.py \
    --train600 "$TRAIN600" --expected-train600-sha256 "$TRAIN600_SHA" \
    --base-selected "$CTRL/selected.jsonl" \
    --specs "$CTRL/compression_replay_specs.jsonl" \
    "${replay_args[@]}" --output-dir "$COMP"
}

run_ready_dataset() {
  local dataset=$1 round=$2 ready=$3 count manifest manifest_sha candidate video_root base_url output marker
  count=$("$PYTHON" -c 'import json,sys; print(sum(json.loads(x)["dataset"]==sys.argv[2] for x in open(sys.argv[1]) if x.strip()))' "$ready" "$dataset")
  if (( count == 0 )); then return 0; fi
  case "$dataset" in
    lvbench) manifest=$LV_MANIFEST; manifest_sha=$LV_SHA; candidate=$CAND_LV; video_root=$LV_VIDEO; base_url=http://127.0.0.1:8200/v1 ;;
    lsdbench) manifest=$LSD_MANIFEST; manifest_sha=$LSD_SHA; candidate=$CAND_LSD; video_root=$LSD_VIDEO; base_url=http://127.0.0.1:8201/v1 ;;
    cgbench) manifest=$CG_MANIFEST; manifest_sha=$CG_SHA; candidate=$CAND_CG; video_root=$CG_VIDEO; base_url=http://127.0.0.1:8200/v1 ;;
  esac
  output="$COMP/replays/round_${round}_${dataset}.jsonl"; marker="$output.done"
  if [[ ! -f "$marker" ]]; then
    "$PYTHON" scripts/run_fast_hybrid_replays.py \
      --dataset "$dataset" --manifest "$manifest" --expected-manifest-sha256 "$manifest_sha" \
      --train600-manifest-sha256 "$TRAIN600_SHA" --candidate-results "$candidate" \
      --specs "$ready" --config-sha256 "$CONFIG_SHA" --model-artifact-sha256 "$MODEL_SHA" \
      --video-root "$video_root" --frame-root "$ROOT/frames/compression/round_${round}/${dataset}" \
      --base-url "$base_url" --api-key no --model Qwen3.5-9B --temperature 0.2 \
      --max-turns 8 --max-call-visual-tokens 12000 --max-total-visual-tokens 48000 \
      --timeout 80 --concurrency 16 --output "$output" --resume
    "$PYTHON" - "$ready" "$output" "$dataset" <<'PY'
import json, sys
ready_path, output_path, dataset = sys.argv[1:]
ready=[json.loads(x) for x in open(ready_path) if x.strip() and json.loads(x)["dataset"]==dataset]
rows=[json.loads(x) for x in open(output_path) if x.strip()]
expected=[str(i) for spec in ready for i in spec["replica_trajectory_ids"]]
actual=[str(row.get("trajectory_id") or "") for row in rows]
assert len(expected)==3*len(ready) and len(actual)==len(set(actual)) and set(actual)==set(expected)
PY
    : > "$marker"
  fi
}

CURRENT_STAGE=compression
mark_stage "$CURRENT_STAGE" started
mkdir -p "$COMP/replays" "$COMP/ready" "$ROOT/frames/compression"
while :; do
  finalize_compression
  COMPLETE=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["compression_complete"]).lower())' "$COMP/compression_summary.json")
  if [[ "$COMPLETE" == true ]]; then break; fi
  READY_NODES=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ready_nodes"])' "$COMP/compression_summary.json")
  (( READY_NODES > 0 )) || { echo "compression incomplete but no node ready" >&2; exit 1; }
  ROUND=""; MAX_ROUND=0
  SNAPSHOTS=( "$COMP"/ready/round_*.jsonl )
  for snapshot in "${SNAPSHOTS[@]}"; do
    stem=${snapshot##*/}; stem=${stem#round_}; stem=${stem%.jsonl}; number=$((10#$stem))
    (( number > MAX_ROUND )) && MAX_ROUND=$number
    cmp -s "$COMP/compression_ready.jsonl" "$snapshot" && ROUND=$stem
  done
  if [[ -z "$ROUND" ]]; then MAX_ROUND=$((MAX_ROUND+1)); ROUND=$(printf '%04d' "$MAX_ROUND"); cp "$COMP/compression_ready.jsonl" "$COMP/ready/round_${ROUND}.jsonl"; fi
  READY="$COMP/ready/round_${ROUND}.jsonl"
  ensure_base_services
  (run_ready_dataset lvbench "$ROUND" "$READY"; run_ready_dataset cgbench "$ROUND" "$READY") & p0=$!
  (run_ready_dataset lsdbench "$ROUND" "$READY") & p1=$!
  batch_status=0
  wait "$p0" || batch_status=$?
  wait "$p1" || batch_status=$?
  (( batch_status == 0 )) || exit "$batch_status"
done
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=build_sft
mark_stage "$CURRENT_STAGE" started
"$PYTHON" scripts/build_fast_hybrid_sft.py \
  --selected "$COMP/selected_pruned.jsonl" \
  --output "$SFT_DIR/fast_hybrid_sft.jsonl" \
  --summary "$SFT_DIR/fast_hybrid_sft_summary.json"
"$PYTHON" scripts/check_fast_hybrid_sft.py \
  --train-manifest "$TRAIN600" --selected "$COMP/selected_pruned.jsonl" \
  --sft-data "$SFT_DIR/fast_hybrid_sft.jsonl" \
  --summary "$SFT_DIR/fast_hybrid_sft_summary.json" --config-sha256 "$CONFIG_SHA" \
  --output "$SFT_DIR/fast_hybrid_sft_audit.json"
mark_stage "$CURRENT_STAGE" passed
CURRENT_STAGE=complete
mark_stage "$CURRENT_STAGE" passed
trap - EXIT

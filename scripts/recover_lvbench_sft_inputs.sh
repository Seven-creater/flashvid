#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"
SERVICE_OWNER_PROJECT_DIR="${SERVICE_OWNER_PROJECT_DIR:-$PROJECT_DIR}"

PYTHON="${PYTHON:-/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python}"
ROOT="${ROOT:-results/eval/fast_hybrid_eva_sft}"
VIDEO_DIR="${LV_VIDEO_DIR:-/data02/pretrained_model/cvr_learn/cvr_data/06_lvbench/videos}"
MANIFEST=/data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/frozen/manifests/lvbench_manifest_42_100.jsonl
MANIFEST_SHA=2e71b4ae1fb1fe88c5eeb8d93d099f4a149eff31626b9f60a74a8b5e05743bbb
ORIGINAL=results/eval/fast_hybrid_eva/frozen_direct/lvbench/lvbench_qwen_qwen3-5-9b-3e3c0499_direct_no_think_greedy_uniform32.jsonl
ORIGINAL_SHA=3cabe8e342c0b9c821dd4d2bdf636a25a651d07405a2741f82897626a7d20280
RECOVERY="$ROOT/candidates/test/lvbench_recovery"
SUBSET="$RECOVERY/lvbench_missing2_manifest.jsonl"
PATCH_DIR="$RECOVERY/direct_patch"
PATCH_NAME=lvbench_qwen_qwen3-5-9b-3e3c0499_direct_no_think_greedy_uniform32.jsonl
REPAIRED_DIR="$ROOT/candidates/test/lvbench"
REPAIRED="$REPAIRED_DIR/lvbench_qwen_qwen3-5-9b-3e3c0499_direct_no_think_greedy_uniform32_repaired.jsonl"
MODEL_PATH=/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B
MODEL_SHA=5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60
QWEN_CONFIG_SHA=9d54cf66790845757755a63fd7b20f4c9e84199f01462cdbf8980bb851a62351
STARTED_SERVICE=0

mkdir -p "$RECOVERY" "$REPAIRED_DIR" "$VIDEO_DIR" "$ROOT/logs" "$ROOT/autopilot"
exec 9>"$ROOT/autopilot/lvbench_recovery.lock"
flock 9

cleanup() {
  local code=$?
  if (( STARTED_SERVICE == 1 )); then
    bash scripts/stop_qwen_agent.sh 8200 || true
  fi
  exit "$code"
}
trap cleanup EXIT

video_valid() {
  [[ -f "$1" ]] && ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null \
    | "$PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.stdin.read().strip()) > 0 else 1)'
}

extract_one() {
  local chunk=$1 member=$2
  local output="$VIDEO_DIR/$member"
  if video_valid "$output"; then return 0; fi
  "$PYTHON" scripts/extract_hf_mirror_zip_member.py \
    --url "https://hf-mirror.com/datasets/lmms-lab/LVBench/resolve/main/video_chunks/videos_chunk_${chunk}.zip" \
    --member "$member" --output "$output" \
    --report "$RECOVERY/${member%.mp4}_recovery.json"
}

extract_one 005 idZkam9zqAs.mp4 & p0=$!
extract_one 013 gXnhqF0TqqI.mp4 & p1=$!
extract_status=0
wait "$p0" || extract_status=$?
wait "$p1" || extract_status=$?
(( extract_status == 0 )) || exit "$extract_status"
video_valid "$VIDEO_DIR/idZkam9zqAs.mp4"
video_valid "$VIDEO_DIR/gXnhqF0TqqI.mp4"

"$PYTHON" scripts/repair_direct_candidate_file.py prepare \
  --manifest "$MANIFEST" --original "$ORIGINAL" --output "$SUBSET" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-original-sha256 "$ORIGINAL_SHA" --expected-count 2 \
  --report "$RECOVERY/prepare_report.json"
SUBSET_SHA=$(sha256sum "$SUBSET" | awk '{print $1}')

model_healthy() {
  local response
  response=$(curl --fail --silent --show-error --max-time 5 \
    http://127.0.0.1:8200/v1/models 2>/dev/null) || return 1
  "$PYTHON" -c 'import json,sys; ids={str(x.get("id")) for x in json.load(sys.stdin).get("data",[]) if isinstance(x,dict)}; raise SystemExit(0 if "Qwen3.5-9B" in ids else 1)' <<<"$response"
}

for _ in $(seq 1 24); do
  if model_healthy; then break; fi
  sleep 5
done
if ! model_healthy; then
  if [[ "$SERVICE_OWNER_PROJECT_DIR" != "$PROJECT_DIR" ]]; then
    bash "$SERVICE_OWNER_PROJECT_DIR/scripts/stop_qwen_agent.sh" 8200
  fi
  bash scripts/stop_qwen_agent.sh 8200
  CUDA_DEVICES=0,1,2,3 ALLOW_SHARED_GPUS=1 \
    bash scripts/launch_qwen_agent_service.sh "$MODEL_PATH" Qwen3.5-9B 8200 4
  STARTED_SERVICE=1
  for _ in $(seq 1 180); do
    if model_healthy; then break; fi
    sleep 5
  done
  model_healthy || { echo "Qwen3.5-9B recovery endpoint failed health check" >&2; exit 1; }
fi

"$PYTHON" scripts/evaluate_mcq.py \
  --dataset lvbench --backend qwen_baseline \
  --annotations /data02/usr/wangqihao/Demo/test/flashvid/data/LVBench_raw_full_root.jsonl \
  --video-root /data02/pretrained_model/cvr_learn/cvr_data/06_lvbench \
  --base-url http://127.0.0.1:8200/v1 --api-key no --model Qwen3.5-9B \
  --model-artifact-sha256 "$MODEL_SHA" --manifest "$SUBSET" \
  --expected-manifest-sha256 "$SUBSET_SHA" --sample 2 --seed 42 \
  --output-dir "$PATCH_DIR" --concurrency 2 --timeout 90 \
  --experiment-config-sha256 "$QWEN_CONFIG_SHA" --baseline-mode direct \
  --qwen-protocol no_think_greedy --direct-sampling uniform32 --resume

PATCH="$PATCH_DIR/$PATCH_NAME"
test -f "$PATCH"
"$PYTHON" scripts/repair_direct_candidate_file.py merge \
  --manifest "$MANIFEST" --original "$ORIGINAL" --patch "$PATCH" \
  --output "$REPAIRED" --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-original-sha256 "$ORIGINAL_SHA" --expected-count 2 \
  --report "$RECOVERY/merge_report.json"
trap - EXIT
if (( STARTED_SERVICE == 1 )); then bash scripts/stop_qwen_agent.sh 8200; fi

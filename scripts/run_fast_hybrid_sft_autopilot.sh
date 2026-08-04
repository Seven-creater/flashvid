#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"

PYTHON="${PYTHON:-/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python}"
CONFIG="${CONFIG:-configs/experiments/fast_hybrid_eva_sft.json}"
CONFIG_SHA="${CONFIG_SHA:-74c2e3a7ecf0dbae30d00037bc748fe0861fdba2501fcf94efeb8aa58856e830}"
ROOT="${ROOT:-results/eval/fast_hybrid_eva_sft}"
TEACHER_CODE_PROJECT_DIR="${TEACHER_CODE_PROJECT_DIR:-$PROJECT_DIR}"
TEACHER_CONFIG="${TEACHER_CONFIG:-$TEACHER_CODE_PROJECT_DIR/configs/experiments/fast_hybrid_eva_sft.json}"
BASE_SPECS="$ROOT/trajectories/control_v2/base_plan.jsonl"
STATE="$ROOT/autopilot/master_state.jsonl"
CURRENT_STAGE=initializing
mkdir -p "$ROOT/autopilot" "$ROOT/logs"

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

cleanup() {
  local code=$?
  if (( code != 0 )); then mark_stage "$CURRENT_STAGE" failed || true; fi
  exit "$code"
}
trap cleanup EXIT

teacher_audit_path() {
  local specs_sha
  specs_sha=$(sha256sum "$BASE_SPECS" | awk '{print $1}')
  printf '%s/run_plans/teacher_%s_audit.json\n' "$ROOT" "${specs_sha:0:12}"
}

teacher_audit_passed() {
  local audit=$1
  [[ -f "$audit" ]] || return 1
  "$PYTHON" - "$audit" <<'PY' >/dev/null
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value.get("status") == "passed"
assert value.get("jobs") == 36 and value.get("rows") == 7200
assert not value.get("issues")
PY
}

teacher_running() {
  pgrep -af 'launch_fast_hybrid_teacher_matrix.py.*base_plan.jsonl' \
    | grep -v "$$" >/dev/null 2>&1
}

candidate_from_teacher_inputs() {
  "$PYTHON" - "$ROOT" "$1" <<'PY'
import glob, hashlib, json, sys
from pathlib import Path
root, dataset = sys.argv[1:]
files=sorted(glob.glob(f"{root}/trajectories/raw/base/*/{dataset}/frozen_inputs_{dataset}.json"))
if not files:
    raise SystemExit(f"no frozen Teacher input found for {dataset}")
refs=[json.load(open(path, encoding="utf-8"))["candidate_results"] for path in files]
if len({ref["sha256"] for ref in refs}) != 1:
    raise SystemExit(f"candidate SHA drift for {dataset}")
for ref in refs:
    path=Path(ref["path"])
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]:
        print(path)
        break
else:
    raise SystemExit(f"unchanged candidate file not found for {dataset}")
PY
}

resume_base_teacher_once() {
  local lv lsd cg
  lv=$(candidate_from_teacher_inputs lvbench)
  lsd=$(candidate_from_teacher_inputs lsdbench)
  cg=$(candidate_from_teacher_inputs cgbench)
  "$PYTHON" "$TEACHER_CODE_PROJECT_DIR/scripts/launch_fast_hybrid_teacher_matrix.py" \
    --config "$TEACHER_CONFIG" --expected-config-sha256 "$CONFIG_SHA" \
    --specs "$BASE_SPECS" --candidate-results "lvbench=$lv" \
    --candidate-results "lsdbench=$lsd" --candidate-results "cgbench=$cg" \
    --python "$PYTHON" --repo-root "$TEACHER_CODE_PROJECT_DIR" \
    --concurrency-per-endpoint 16 --timeout 80 --resume
}

CURRENT_STAGE=wait_base_teacher
mark_stage "$CURRENT_STAGE" started
[[ -f "$BASE_SPECS" ]] || { echo "base Teacher plan is missing: $BASE_SPECS" >&2; exit 1; }
TEACHER_AUDIT=$(teacher_audit_path)
while ! teacher_audit_passed "$TEACHER_AUDIT"; do
  if teacher_running; then
    sleep 60
    continue
  fi
  echo "base Teacher exited without a passed audit; resuming missing work once" >&2
  resume_base_teacher_once
  teacher_audit_passed "$TEACHER_AUDIT" || {
    echo "base Teacher audit is still incomplete after the bounded resume" >&2
    exit 1
  }
done
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=post_teacher
mark_stage "$CURRENT_STAGE" started
bash scripts/run_fast_hybrid_post_teacher.sh
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=train_and_evaluate
mark_stage "$CURRENT_STAGE" started
bash scripts/run_fast_hybrid_train_eval.sh
mark_stage "$CURRENT_STAGE" passed

CURRENT_STAGE=complete
mark_stage "$CURRENT_STAGE" passed
trap - EXIT

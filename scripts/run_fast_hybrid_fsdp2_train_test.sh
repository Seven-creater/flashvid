#!/usr/bin/env bash
set -euo pipefail

# One durable entry point for the approved fast path:
#   1) prove one optimizer step with 8-GPU FSDP2 FULL_SHARD;
#   2) train all three LoRA epochs on GPUs 0-7;
#   3) serve the final epoch as two independent 4-GPU DP=4 endpoints;
#   4) evaluate the frozen Test300 once with 32 total concurrent requests.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ROOT="${ROOT:-/data02/usr/wangqihao/Demo/test/qwen_agent_search/results/eval/fast_hybrid_eva_sft}"
SFT_ENV_DIR="${SFT_ENV_DIR:-/data02/usr/wangqihao/Demo/test/qwen_agent_search/.venv-qwen35-sft-cu124}"
MODEL_PATH="${MODEL_PATH:-/data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B}"
MODEL_SHA="${MODEL_SHA:-5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60}"
SFT_DATA="$ROOT/sft_data/fast_hybrid_sft_max16384.jsonl"
FORMAL_DIR="$ROOT/checkpoints/qwen35_9b_lora"
ACTIVE_POINTER="$ROOT/checkpoints/active_smoke_report.txt"
FSDP2_METADATA="$ROOT/checkpoints/fsdp2_smoke_metadata.json"
SWIFT_PYTHON="$SFT_ENV_DIR/bin/python"

[[ -x "$SWIFT_PYTHON" ]] || { echo "missing SFT Python: $SWIFT_PYTHON" >&2; exit 2; }
[[ -f "$SFT_DATA" ]] || { echo "missing filtered SFT data: $SFT_DATA" >&2; exit 2; }

export ROOT SFT_ENV_DIR SWIFT_PYTHON
export SWIFT_BIN="$SFT_ENV_DIR/bin/swift"
export MODEL_PATH MODEL_SHA
export USE_FSDP2=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DIRECT_TEST_AFTER_TRAIN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

launcher_sha=$(sha256sum scripts/train_qwen_agent_9b_lora.sh | awk '{print $1}')
plugin_sha=$(sha256sum scripts/swift_fsdp2_bf16_lora_plugin.py | awk '{print $1}')
smoke_report=""
if [[ -f "$FSDP2_METADATA" ]]; then
  smoke_report=$(
    "$SWIFT_PYTHON" - "$FSDP2_METADATA" "$launcher_sha" "$plugin_sha" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if (
    value.get("launcher_sha256") == sys.argv[2]
    and value.get("plugin_sha256") == sys.argv[3]
):
    report = Path(str(value.get("report") or ""))
    if report.is_file():
        print(report)
PY
  )
fi
if [[ -n "$smoke_report" ]]; then
  if ! "$SWIFT_PYTHON" scripts/qwen_sft_smoke_gate.py check \
      --report "$smoke_report" --formal-output-dir "$FORMAL_DIR" \
      --train-data "$SFT_DATA" --base-model-artifact-sha256 "$MODEL_SHA"; then
    smoke_report=""
  fi
fi

if [[ -z "$smoke_report" ]]; then
  smoke_dir="$ROOT/checkpoints/smoke_qwen35_9b_lora_fsdp2_bf16_$(date +%Y%m%d_%H%M%S)"
  bash scripts/train_qwen_agent_9b_lora.sh \
    --train-data "$SFT_DATA" --output-dir "$smoke_dir" \
    --smoke --formal-output-dir "$FORMAL_DIR" \
    --release-project-services --load-weights-preflight
  smoke_report="$smoke_dir/preflight/training_update.json"
  "$SWIFT_PYTHON" scripts/qwen_sft_smoke_gate.py check \
    --report "$smoke_report" --formal-output-dir "$FORMAL_DIR" \
    --train-data "$SFT_DATA" --base-model-artifact-sha256 "$MODEL_SHA"
  "$SWIFT_PYTHON" - "$FSDP2_METADATA" "$smoke_report" "$launcher_sha" "$plugin_sha" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "fast_hybrid_fsdp2_smoke",
    "report": str(Path(sys.argv[2]).resolve()),
    "launcher_sha256": sys.argv[3],
    "plugin_sha256": sys.argv[4],
    "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "fsdp": "fsdp2",
}
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
PY
fi

# The existing pipeline validates this report again before formal training.
printf '%s\n' "$smoke_report" > "$ACTIVE_POINTER"
exec bash scripts/run_fast_hybrid_train_eval.sh

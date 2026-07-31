#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
MODEL_DIR="${PROJECT_DIR}/models/Qwen3.5-4B"
CACHE_DIR="${PROJECT_DIR}/.cache/modelscope"

mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"
"${PYTHON}" -m pip install "modelscope>=1.28"

MODELSCOPE_CACHE="${CACHE_DIR}" "${PYTHON}" - <<PY
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen3.5-4B",
    local_dir="${MODEL_DIR}",
)
PY

test -f "${MODEL_DIR}/config.json"
echo "Model downloaded to ${MODEL_DIR}"


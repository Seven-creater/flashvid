#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
MODEL_DIR="${PROJECT_DIR}/models/Qwen3.5-4B"
CACHE_DIR="${PROJECT_DIR}/.cache/modelscope"
PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
# ModelScope prepends the URL scheme internally; this value must be a host.
MODELSCOPE_DOMAIN="www.modelscope.cn"

mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL
export MODELSCOPE_DOMAIN
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
"${PYTHON}" -m pip install --index-url "${PYPI_INDEX_URL}" "modelscope>=1.28"

MODELSCOPE_CACHE="${CACHE_DIR}" "${PYTHON}" - <<PY
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen3.5-4B",
    local_dir="${MODEL_DIR}",
)
PY

test -f "${MODEL_DIR}/config.json"
echo "Model downloaded to ${MODEL_DIR}"

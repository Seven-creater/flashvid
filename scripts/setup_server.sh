#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/data02/usr/wangqihao/miniconda3/bin/conda}"
ENV_DIR="${PROJECT_DIR}/.venv"
CACHE_DIR="${PROJECT_DIR}/.cache"

mkdir -p "${CACHE_DIR}/pip" "${CACHE_DIR}/torch" "${PROJECT_DIR}/logs" \
  "${PROJECT_DIR}/results" "${PROJECT_DIR}/models"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_DIR}" python=3.12 pip
fi

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TORCH_HOME="${CACHE_DIR}/torch"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install "vllm==0.25.1"
"${ENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[test,bench]"

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import vllm

print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY


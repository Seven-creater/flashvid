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
"${ENV_DIR}/bin/python" -m pip install "nvidia-cuda-runtime==13.2.86"
"${ENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[test,bench]"

TORCH_CUDA_MAJOR="$("${ENV_DIR}/bin/python" -c 'import torch; print(torch.version.cuda.split(".")[0])')"
if [[ "${TORCH_CUDA_MAJOR}" -ge 13 ]]; then
  "${CONDA_BIN}" install -y -p "${ENV_DIR}" "anaconda::cuda-compat=13.0.2"
  CUDA_TOOLKIT="${ENV_DIR}/lib/python3.12/site-packages/nvidia/cu13"
  if [[ -d "${CUDA_TOOLKIT}/lib" && ! -e "${CUDA_TOOLKIT}/lib64" ]]; then
    ln -s lib "${CUDA_TOOLKIT}/lib64"
  fi
  if [[ ! -e "${CUDA_TOOLKIT}/lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "${CUDA_TOOLKIT}/lib/libcudart.so"
  fi
  if [[ ! -e "${CUDA_TOOLKIT}/lib/libcuda.so" ]]; then
    ln -s "${ENV_DIR}/cuda-compat/libcuda.so" "${CUDA_TOOLKIT}/lib/libcuda.so"
  fi
  export LD_LIBRARY_PATH="${ENV_DIR}/cuda-compat:${LD_LIBRARY_PATH:-}"
fi

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import vllm

print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

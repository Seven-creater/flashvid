#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_PYTHON="${SWIFT_PYTHON:-${PROJECT_DIR}/.venv-swift/bin/python}"
CUDA_HOME="${CUDA_HOME:-${PROJECT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13}"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/.runtime/flash_attn_install}"
REPORT="${REPORT:-${STATE_DIR}/installed.json}"
VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"

mkdir -p "$STATE_DIR" "${PROJECT_DIR}/.cache/pip"
exec 9>"${STATE_DIR}/install.lock"
flock -n 9 || { echo "another flash-attn installation is active" >&2; exit 2; }

[[ -x "$SWIFT_PYTHON" ]] || { echo "training Python not found: $SWIFT_PYTHON" >&2; exit 2; }
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "CUDA nvcc not found under $CUDA_HOME" >&2; exit 2; }
[[ -x /usr/bin/gcc && -x /usr/bin/g++ ]] || { echo "system C/C++ compiler is missing" >&2; exit 2; }

export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export PATH="${CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MAX_JOBS="${MAX_JOBS:-8}"
export PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"

"$SWIFT_PYTHON" -m pip install --no-build-isolation "flash-attn==${VERSION}"
"$SWIFT_PYTHON" - "$REPORT" "$VERSION" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

from flash_attn import flash_attn_func

installed = importlib.metadata.version("flash-attn")
expected = sys.argv[2]
if installed != expected or not callable(flash_attn_func):
    raise SystemExit(f"flash-attn verification failed: {installed=}, {expected=}")
output = Path(sys.argv[1])
payload = {
    "status": "installed",
    "flash_attn_version": installed,
    "python": sys.executable,
}
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=output.parent,
    prefix=f".{output.name}.", suffix=".tmp", delete=False,
) as handle:
    temporary = Path(handle.name)
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
temporary.replace(output)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

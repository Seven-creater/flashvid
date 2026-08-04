#!/usr/bin/env bash
set -euo pipefail

# Build the training-only environment without mutating or inheriting from the
# inference venv.  Qwen3.5's linear-attention stack requires Python 3.12; the
# CUDA runtime and every top-level training dependency are pinned separately.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON312="${PYTHON312:-python3.12}"
SWIFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-qwen35-sft-cu121}"
SWIFT_PYTHON="${SWIFT_ENV_DIR}/bin/python"
LOCK_FILE="${LOCK_FILE:-${PROJECT_DIR}/configs/training/qwen35_9b_sft_cuda121.lock.txt}"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/.runtime/ms_swift_442}"
REPORT="${REPORT:-${STATE_DIR}/installed.json}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"

mkdir -p "$STATE_DIR" "${PROJECT_DIR}/.cache/pip"
exec 9>"${STATE_DIR}/install.lock"
flock -n 9 || {
  echo "another ms-swift installation owns ${STATE_DIR}/install.lock" >&2
  exit 2
}

command -v "$PYTHON312" >/dev/null 2>&1 || {
  echo "Python 3.12 executable not found: ${PYTHON312}" >&2
  exit 2
}
[[ -f "$LOCK_FILE" ]] || { echo "training dependency lock not found: $LOCK_FILE" >&2; exit 2; }
"$PYTHON312" -c \
  'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

if [[ ! -x "$SWIFT_PYTHON" ]]; then
  "$PYTHON312" -m venv "$SWIFT_ENV_DIR"
fi
"$SWIFT_PYTHON" -c \
  'import sys; assert sys.version_info[:2] == (3, 12), sys.version; assert sys.prefix != sys.base_prefix'
grep -Eq '^include-system-site-packages = false$' "$SWIFT_ENV_DIR/pyvenv.cfg" || {
  echo "training venv must not inherit system site-packages: $SWIFT_ENV_DIR" >&2
  exit 2
}

export PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"
export HF_ENDPOINT
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
"$SWIFT_PYTHON" -m pip install --upgrade pip setuptools wheel
# This is the last PyTorch release for which the official project publishes a
# CUDA 12.1 wheel set.  Install it before packages with CUDA build extensions.
"$SWIFT_PYTHON" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
# CUDA extensions use the active environment because build isolation would
# hide the torch wheel they compile against.
"$SWIFT_PYTHON" -m pip install "ninja==1.13.0" "packaging==25.0"
"$SWIFT_PYTHON" -m pip install --no-build-isolation -r "$LOCK_FILE"
"$SWIFT_PYTHON" -m pip check

"$SWIFT_PYTHON" - "$REPORT" "$LOCK_FILE" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

expected = {
    "ms-swift": "4.4.2",
    "transformers": "5.9.0",
    "qwen-vl-utils": "0.0.14",
    "peft": "0.19.1",
    "flash-linear-attention": "0.4.2",
    "causal-conv1d": "1.6.2.post1",
    "flash-attn": "2.8.3",
}
versions = {name: importlib.metadata.version(name) for name in expected}
if versions != expected:
    raise SystemExit(f"training dependency lock mismatch: {versions!r}")

import torch
if not str(torch.__version__).startswith("2.5.1+cu121") or torch.version.cuda != "12.1":
    raise SystemExit(
        f"expected torch 2.5.1+cu121 / CUDA 12.1, found {torch.__version__} / {torch.version.cuda}"
    )

from swift import get_processor, get_template  # noqa: F401

output = Path(sys.argv[1])
lock = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "installed",
    "versions": versions,
    "torch_version": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "lock_file": str(lock.resolve()),
    "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    "python": sys.executable,
    "base_prefix": sys.base_prefix,
    "prefix": sys.prefix,
}
with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    dir=output.parent,
    prefix=f".{output.name}.",
    suffix=".tmp",
    delete=False,
) as handle:
    temporary = Path(handle.name)
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
temporary.replace(output)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

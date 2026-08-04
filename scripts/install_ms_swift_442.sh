#!/usr/bin/env bash
set -euo pipefail

# Build the training-only environment without mutating or inheriting from the
# inference venv.  Qwen3.5's linear-attention stack requires Python 3.12; the
# CUDA runtime and every top-level training dependency are pinned separately.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audit_only=0
if [[ $# -eq 1 && "$1" == "--audit-only" ]]; then
  audit_only=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--audit-only]" >&2
  exit 2
fi
PYTHON312="${PYTHON312:-python3.12}"
SWIFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-qwen35-sft-cu121}"
SWIFT_PYTHON="${SWIFT_ENV_DIR}/bin/python"
LOCK_FILE="${LOCK_FILE:-${PROJECT_DIR}/configs/training/qwen35_9b_sft_cuda121.lock.txt}"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/.runtime/ms_swift_442}"
REPORT="${REPORT:-${STATE_DIR}/installed.json}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_WHEEL_BASE="${TORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu121}"
TORCH_WHEEL_URL="${TORCH_WHEEL_BASE}/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL_URL="${TORCH_WHEEL_BASE}/torchvision-0.20.1%2Bcu121-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_WHEEL_URL="${TORCH_WHEEL_BASE}/torchaudio-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl"
TORCH_WHEEL_SHA256="222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76"
TORCHVISION_WHEEL_SHA256="48cf3a716f70370ed5dcb656e7497415ef37860b07e67ea4b1ef8598efe28445"
TORCHAUDIO_WHEEL_SHA256="5648a01f23033f15d60dc638f91c2d4c66c0a01621162471e806064acda63b70"
FLA_WHEEL_URL="${FLA_WHEEL_URL:-https://pypi.tuna.tsinghua.edu.cn/packages/60/ee/a3cba17965482b35c4990af90bad108e82c32edcb59911c37f318b5f4198/flash_linear_attention-0.4.2-py3-none-any.whl}"
FLA_CORE_WHEEL_URL="${FLA_CORE_WHEEL_URL:-https://pypi.tuna.tsinghua.edu.cn/packages/ee/36/3c303f92bafea7c3f97d68bbb83d18cc42e30cd0bfb1b7cfe589360f11d6/fla_core-0.4.2-py3-none-any.whl}"
FLA_WHEEL_SHA256="c08be006ce4dbe1be81f54938ee8e6fc7968cfba397c8d06c7669e97b8c44c0d"
FLA_CORE_WHEEL_SHA256="cba3db29380002da3cbfc0db94d6efac19aaf528900d19c05c2765e8f3cc485b"

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
[[ "$HF_ENDPOINT" == "https://hf-mirror.com" ]] || {
  echo "HF_ENDPOINT must remain https://hf-mirror.com on this server" >&2
  exit 2
}
[[ "$PYPI_INDEX_URL" == "https://pypi.tuna.tsinghua.edu.cn/simple" ]] || {
  echo "PYPI_INDEX_URL must remain the Tsinghua domestic mirror" >&2
  exit 2
}
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
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX

# Refuse mirror redirects before starting any large download.  The hashes are
# the SHA-256 values published in PyTorch's official cu121 wheel index; pip
# verifies that the domestic mirror serves byte-identical artifacts.
"$SWIFT_PYTHON" - \
  "$TORCH_WHEEL_URL" \
  "$TORCHVISION_WHEEL_URL" \
  "$TORCHAUDIO_WHEEL_URL" \
  "$FLA_WHEEL_URL" \
  "$FLA_CORE_WHEEL_URL" <<'PY'
from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"domestic wheel URL redirected to {newurl!r}")


opener = urllib.request.build_opener(NoRedirect)
for value in sys.argv[1:]:
    parsed = urllib.parse.urlparse(value)
    allowed_hosts = {"mirrors.aliyun.com", "pypi.tuna.tsinghua.edu.cn"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise SystemExit(f"unapproved PyTorch wheel host: {value}")
    request = urllib.request.Request(value, method="HEAD")
    try:
        with opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise SystemExit(f"wheel probe returned HTTP {response.status}: {value}")
            if int(response.headers.get("Content-Length", "0")) <= 0:
                raise SystemExit(f"wheel probe lacks Content-Length: {value}")
    except (urllib.error.URLError, RuntimeError) as error:
        raise SystemExit(f"domestic wheel probe failed: {value}: {error}") from error
PY

if [[ "$audit_only" -eq 1 ]]; then
  echo "domestic mirror audit passed; no packages were downloaded"
  exit 0
fi

"$SWIFT_PYTHON" -m pip install --index-url "$PYPI_INDEX_URL" --upgrade pip setuptools wheel
# This is the last PyTorch release for which the official project publishes a
# CUDA 12.1 wheel set. Install byte-identical domestic mirror copies directly;
# an index page may otherwise redirect pip back to download.pytorch.org.
"$SWIFT_PYTHON" -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  "${TORCH_WHEEL_URL}#sha256=${TORCH_WHEEL_SHA256}" \
  "${TORCHVISION_WHEEL_URL}#sha256=${TORCHVISION_WHEEL_SHA256}" \
  "${TORCHAUDIO_WHEEL_URL}#sha256=${TORCHAUDIO_WHEEL_SHA256}"

# FLA and fla-core 0.4.2 both publish platform-independent wheels.  Reject an
# sdist rather than attempting a local CUDA build.  transformers 5.9.0 supplies
# the official torch Conv1d fallback, while training explicitly uses SDPA for
# full-attention layers, so causal-conv1d and flash-attn are intentionally absent.
"$SWIFT_PYTHON" -m pip install \
  --index-url "$PYPI_INDEX_URL" --no-deps \
  "${FLA_WHEEL_URL}#sha256=${FLA_WHEEL_SHA256}" \
  "${FLA_CORE_WHEEL_URL}#sha256=${FLA_CORE_WHEEL_SHA256}"
"$SWIFT_PYTHON" -m pip install --index-url "$PYPI_INDEX_URL" -r "$LOCK_FILE"
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
    "fla-core": "0.4.2",
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
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
if not callable(chunk_gated_delta_rule):
    raise SystemExit("flash-linear-attention gated-delta kernel is unavailable")

optional_extensions = {}
for distribution in ("causal-conv1d", "flash-attn"):
    try:
        optional_extensions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        optional_extensions[distribution] = None

output = Path(sys.argv[1])
lock = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "installed",
    "versions": versions,
    "torch_version": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "full_attention_backend": "sdpa",
    "linear_attention_backend": "flash-linear-attention",
    "optional_extensions": optional_extensions,
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

#!/usr/bin/env bash
set -euo pipefail

# Build the training-only environment without mutating the inference venv.
# The child venv can read the already-installed CUDA/PyTorch stack from the
# project interpreter, while any dependency overrides stay inside .venv-swift.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
SWIFT_ENV_DIR="${SFT_ENV_DIR:-${PROJECT_DIR}/.venv-swift}"
SWIFT_PYTHON="${SWIFT_ENV_DIR}/bin/python"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/.runtime/ms_swift_442}"
REPORT="${REPORT:-${STATE_DIR}/installed.json}"

mkdir -p "$STATE_DIR" "${PROJECT_DIR}/.cache/pip"
exec 9>"${STATE_DIR}/install.lock"
flock -n 9 || {
  echo "another ms-swift installation owns ${STATE_DIR}/install.lock" >&2
  exit 2
}

[[ -x "$BASE_PYTHON" ]] || {
  echo "base project Python not found: ${BASE_PYTHON}" >&2
  exit 2
}

if [[ ! -x "$SWIFT_PYTHON" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$SWIFT_ENV_DIR"
fi

export PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"
"$SWIFT_PYTHON" -m pip install \
  --upgrade \
  --upgrade-strategy only-if-needed \
  "ms-swift==4.4.2"

"$SWIFT_PYTHON" - "$REPORT" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

version = importlib.metadata.version("ms-swift")
if version != "4.4.2":
    raise SystemExit(f"expected ms-swift 4.4.2, found {version}")

from swift import get_processor, get_template  # noqa: F401

output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "installed",
    "ms_swift_version": version,
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

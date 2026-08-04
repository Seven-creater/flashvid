#!/usr/bin/env bash
set -euo pipefail

# Destructive actions remain opt-in at the lower-level launcher.  This explicit
# preflight entry point is the only command that releases this project's 8200
# and 8201 services, then loads the model and performs one sample / one step.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$PROJECT_DIR/scripts/train_qwen_agent_9b_lora.sh" \
  --release-project-services \
  --load-weights-preflight \
  --smoke \
  "$@"

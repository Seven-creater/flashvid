from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import ARCHITECTURE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashvid-serve",
        description="Serve Qwen3.5 with FlashVID vision-side compression.",
    )
    parser.add_argument("model", help="Qwen3.5 model path or Hugging Face ID")
    parser.add_argument(
        "--vision-retention-ratio",
        type=float,
        default=0.1,
        help="Fraction of vision-encoder tokens to retain, in (0, 1].",
    )
    return parser


def build_vllm_command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    args, passthrough = _parser().parse_known_args(argv)
    ratio = args.vision_retention_ratio
    if not 0.0 < ratio <= 1.0:
        raise SystemExit("--vision-retention-ratio must be in (0, 1]")
    forbidden = {"--hf-overrides", "--video-pruning-rate"}
    conflicts = [item for item in passthrough if item.split("=", 1)[0] in forbidden]
    if conflicts:
        raise SystemExit(
            "Do not pass --hf-overrides or --video-pruning-rate; "
            "flashvid-serve manages them."
        )

    executable = Path(sys.executable).with_name(
        "vllm.exe" if os.name == "nt" else "vllm"
    )
    command = [
        str(executable),
        "serve",
        args.model,
        "--hf-overrides",
        json.dumps({"architectures": [ARCHITECTURE]}),
        "--video-pruning-rate",
        str(1.0 - ratio),
        *passthrough,
    ]
    environment = os.environ.copy()
    environment["FLASHVID_VISION_RETENTION_RATIO"] = str(ratio)
    environment.setdefault("VLLM_PLUGINS", "flashvid_qwen3_5")
    return command, environment


def main() -> None:
    command, environment = build_vllm_command(sys.argv[1:])
    raise SystemExit(subprocess.call(command, env=environment))

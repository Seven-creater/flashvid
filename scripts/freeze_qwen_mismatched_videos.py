from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.qwen_mismatch_maps import freeze_mismatch_maps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze same-duration wrong-video controls for Qwen diagnostics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_mismatch_maps(args.config, output_root=args.output_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

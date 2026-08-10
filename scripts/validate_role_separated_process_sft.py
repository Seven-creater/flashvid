#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.role_separated_orchestration import (
    load_config,
    require_valid_config,
    training_quantity_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen role-separated process SFT plan."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_valid_config(config)
    print(
        json.dumps(
            {
                "status": "passed",
                "config": str(args.config.resolve()),
                "training_quantity": training_quantity_report(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

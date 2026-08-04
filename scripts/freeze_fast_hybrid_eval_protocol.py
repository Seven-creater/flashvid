#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.fast_hybrid_eval_protocol import (
    DATASETS,
    SPLITS,
    build_protocol,
    freeze_json,
    sha256_file,
)


def _candidate_mapping(values: list[str]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise ValueError("--candidate must use SPLIT:DATASET=PATH")
        identity, raw_path = value.split("=", 1)
        split, dataset = (part.strip().lower() for part in identity.split(":", 1))
        key = (split, dataset)
        if split not in SPLITS or dataset not in DATASETS or key in result:
            raise ValueError(f"invalid or duplicate candidate mapping: {value}")
        result[key] = Path(raw_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-experiment-config-sha256", required=True)
    parser.add_argument("--candidate", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = build_protocol(
            experiment_config_path=args.experiment_config,
            expected_experiment_config_sha256=args.expected_experiment_config_sha256,
            candidate_paths=_candidate_mapping(args.candidate),
        )
        freeze_json(args.output, payload)
        report = {
            "status": "passed",
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "protocol_fingerprint": payload["protocol_fingerprint"],
        }
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from flashvid_eval.perception_memory_gate import (
    DATASETS,
    PerceptionMemoryGateError,
    evaluate_dev_gate,
    evaluate_test_gate,
    method_from_config,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def evaluate_config(payload: dict[str, Any]) -> dict[str, Any]:
    phase = str(payload.get("phase") or "").strip().lower()
    raw_counts = payload.get("expected_counts")
    counts = None
    if raw_counts is not None:
        if not isinstance(raw_counts, dict) or set(raw_counts) != set(DATASETS):
            raise PerceptionMemoryGateError("expected_counts must cover all datasets")
        counts = {str(key): int(value) for key, value in raw_counts.items()}
    baseline_raw = payload.get("baseline")
    if not isinstance(baseline_raw, dict):
        raise PerceptionMemoryGateError("baseline must be an object")
    raw_manifests = payload.get("expected_manifest_sha256")
    if not isinstance(raw_manifests, dict) or set(raw_manifests) != set(DATASETS):
        raise PerceptionMemoryGateError(
            "expected_manifest_sha256 must cover all datasets"
        )
    manifests = {str(key): str(value) for key, value in raw_manifests.items()}
    baseline = method_from_config(baseline_raw)
    if phase == "dev":
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise PerceptionMemoryGateError("dev candidates must be an array")
        return evaluate_dev_gate(
            baseline=baseline,
            candidates=[method_from_config(item) for item in raw_candidates],
            expected_manifest_sha256=manifests,
            expected_counts=counts,
        )
    if phase == "test":
        raw_candidate = payload.get("candidate")
        if not isinstance(raw_candidate, dict):
            raise PerceptionMemoryGateError("test candidate must be an object")
        raw_dev_gate = payload.get("dev_gate_report")
        if not isinstance(raw_dev_gate, dict):
            raise PerceptionMemoryGateError(
                "test dev_gate_report must be an object"
            )
        dev_gate_path = raw_dev_gate.get("path")
        dev_gate_sha256 = raw_dev_gate.get("sha256")
        if not dev_gate_path or not dev_gate_sha256:
            raise PerceptionMemoryGateError(
                "test dev_gate_report requires path and sha256"
            )
        return evaluate_test_gate(
            baseline=baseline,
            candidate=method_from_config(raw_candidate),
            expected_manifest_sha256=manifests,
            dev_gate_report_path=Path(str(dev_gate_path)),
            dev_gate_report_sha256=str(dev_gate_sha256),
            expected_counts=counts,
        )
    raise PerceptionMemoryGateError("phase must be dev or test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Dev/Test gate for Perception-Memory EVA SFT."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise PerceptionMemoryGateError("config root must be an object")
        report = evaluate_config(config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "passed": False,
            "error": str(exc),
        }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

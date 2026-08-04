#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("smoke report must be a JSON object")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def bind_smoke_report(
    report_path: Path,
    *,
    smoke_output_dir: Path,
    formal_output_dir: Path,
    train_data: Path,
    base_model_artifact_sha256: str,
) -> dict[str, Any]:
    report = _load_report(report_path)
    if report.get("status") != "passed" or report.get("global_step") != 1:
        raise ValueError("formal training requires a passed one-step smoke report")
    if not train_data.is_file():
        raise FileNotFoundError(train_data)
    smoke_root = smoke_output_dir.resolve()
    if report_path.resolve().parent.parent != smoke_root:
        raise ValueError("smoke report is not inside the bound smoke output directory")
    gate = {
        "schema_version": 1,
        "smoke_output_dir": str(smoke_root),
        "formal_output_dir": str(formal_output_dir.resolve()),
        "train_data": {
            "path": str(train_data.resolve()),
            "sha256": _sha256_file(train_data),
        },
        "base_model_artifact_sha256": _sha256(
            base_model_artifact_sha256,
            "base_model_artifact_sha256",
        ),
    }
    report["formal_training_gate"] = gate
    _atomic_write(report_path, report)
    return report


def validate_smoke_report(
    report_path: Path,
    *,
    formal_output_dir: Path,
    train_data: Path,
    base_model_artifact_sha256: str,
) -> dict[str, Any]:
    report = _load_report(report_path)
    if report.get("status") != "passed" or report.get("global_step") != 1:
        raise ValueError("formal training requires a passed one-step smoke report")
    gate = report.get("formal_training_gate")
    if not isinstance(gate, Mapping) or gate.get("schema_version") != 1:
        raise ValueError("smoke report has no formal-training binding")
    if gate.get("smoke_output_dir") != str(report_path.resolve().parent.parent):
        raise ValueError("smoke report moved outside its bound smoke output directory")
    if gate.get("formal_output_dir") != str(formal_output_dir.resolve()):
        raise ValueError("smoke report is bound to a different formal output directory")
    if not train_data.is_file():
        raise FileNotFoundError(train_data)
    train_reference = gate.get("train_data")
    if not isinstance(train_reference, Mapping):
        raise ValueError("smoke report has no bound training data")
    if train_reference.get("path") != str(train_data.resolve()):
        raise ValueError("smoke report is bound to a different training-data path")
    if train_reference.get("sha256") != _sha256_file(train_data):
        raise ValueError("training data changed after the smoke run")
    expected_model = _sha256(
        base_model_artifact_sha256,
        "base_model_artifact_sha256",
    )
    if gate.get("base_model_artifact_sha256") != expected_model:
        raise ValueError("smoke report is bound to a different base-model artifact")
    return dict(gate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind and enforce the Qwen LoRA smoke gate.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("bind", "check"):
        child = subparsers.add_parser(command)
        child.add_argument("--report", type=Path, required=True)
        child.add_argument("--formal-output-dir", type=Path, required=True)
        child.add_argument("--train-data", type=Path, required=True)
        child.add_argument("--base-model-artifact-sha256", required=True)
        if command == "bind":
            child.add_argument("--smoke-output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bind":
        report = bind_smoke_report(
            args.report,
            smoke_output_dir=args.smoke_output_dir,
            formal_output_dir=args.formal_output_dir,
            train_data=args.train_data,
            base_model_artifact_sha256=args.base_model_artifact_sha256,
        )
        payload: Mapping[str, Any] = report["formal_training_gate"]
    else:
        payload = validate_smoke_report(
            args.report,
            formal_output_dir=args.formal_output_dir,
            train_data=args.train_data,
            base_model_artifact_sha256=args.base_model_artifact_sha256,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare and merge exact-sample repairs for unavailable frozen Direct candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def index_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise ValueError(f"{path} has a missing or duplicate sample_id")
        result[sample_id] = row
    return result


def freeze_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def unavailable_ids(candidate_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in candidate_rows
        if not row.get("prediction")
        and row.get("data_unavailable") is True
        and row.get("failure_class") == "data_unavailable"
    }


def prepare_subset(
    manifest: Path,
    original: Path,
    output: Path,
    *,
    expected_manifest_sha256: str,
    expected_original_sha256: str,
    expected_count: int,
) -> dict[str, Any]:
    if sha256_file(manifest) != expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 changed")
    if sha256_file(original) != expected_original_sha256:
        raise RuntimeError("original candidate SHA-256 changed")
    manifest_rows = read_jsonl(manifest)
    original_rows = read_jsonl(original)
    manifest_index = index_rows(manifest_rows, manifest)
    original_index = index_rows(original_rows, original)
    if set(manifest_index) != set(original_index):
        raise ValueError("original candidate identities differ from manifest")
    targets = unavailable_ids(original_rows)
    if len(targets) != expected_count:
        raise ValueError(
            f"expected {expected_count} unavailable candidates, found {len(targets)}"
        )
    subset = [row for row in manifest_rows if str(row["sample_id"]) in targets]
    freeze_jsonl(output, subset)
    return {
        "status": "passed",
        "mode": "prepare",
        "target_ids": sorted(targets),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }


def _validate_patch(row: dict[str, Any], sample_id: str) -> None:
    request = row.get("protocol_request") or {}
    prediction = str(row.get("prediction") or "").upper()
    if len(prediction) != 1 or not "A" <= prediction <= "H":
        raise ValueError(f"patch has no valid prediction: {sample_id}")
    if (
        row.get("error")
        or row.get("data_unavailable") is True
        or row.get("failure_class")
        or row.get("model_parse_failure") is True
        or row.get("annotation_leak_check") != "passed"
        or int(row.get("candidate_rerun") or 0) != 0
    ):
        raise ValueError(f"patch did not complete cleanly: {sample_id}")
    if (
        row.get("baseline_mode") != "direct"
        or row.get("sampling_id") != "uniform32"
        or row.get("enable_thinking") is not False
        or request.get("max_tokens") != 512
        or float(request.get("temperature", -1.0)) != 0.0
    ):
        raise ValueError(f"patch protocol differs from clean Direct: {sample_id}")
    if (
        row.get("visual_usage_complete") is not True
        or not isinstance(row.get("visual_tokens"), (int, float))
        or not isinstance(row.get("total_tokens"), (int, float))
        or float(row["total_tokens"]) <= 0
    ):
        raise ValueError(f"patch Token accounting is incomplete: {sample_id}")


def merge_candidates(
    manifest: Path,
    original: Path,
    patch: Path,
    output: Path,
    *,
    expected_manifest_sha256: str,
    expected_original_sha256: str,
    expected_count: int,
) -> dict[str, Any]:
    if sha256_file(manifest) != expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 changed")
    if sha256_file(original) != expected_original_sha256:
        raise RuntimeError("original candidate SHA-256 changed")
    manifest_rows = read_jsonl(manifest)
    original_rows = read_jsonl(original)
    patch_rows = read_jsonl(patch)
    manifest_index = index_rows(manifest_rows, manifest)
    original_index = index_rows(original_rows, original)
    patch_index = index_rows(patch_rows, patch)
    if set(manifest_index) != set(original_index):
        raise ValueError("original candidate identities differ from manifest")
    targets = unavailable_ids(original_rows)
    if len(targets) != expected_count or set(patch_index) != targets:
        raise ValueError("patch identities differ from the unavailable candidate set")
    for sample_id, row in patch_index.items():
        _validate_patch(row, sample_id)
        if str(row.get("video") or "") != str(manifest_index[sample_id].get("video") or ""):
            raise ValueError(f"patch video differs from manifest: {sample_id}")
    merged = [
        patch_index.get(str(manifest_row["sample_id"]), original_index[str(manifest_row["sample_id"])])
        for manifest_row in manifest_rows
    ]
    freeze_jsonl(output, merged)
    merged_rows = read_jsonl(output)
    if unavailable_ids(merged_rows):
        raise RuntimeError("merged candidate still contains unavailable targets")
    return {
        "status": "passed",
        "mode": "merge",
        "target_ids": sorted(targets),
        "original": {"path": str(original.resolve()), "sha256": expected_original_sha256},
        "patch": {"path": str(patch.resolve()), "sha256": sha256_file(patch)},
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "rows": len(merged_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "merge"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        common = {
            "manifest": args.manifest,
            "original": args.original,
            "output": args.output,
            "expected_manifest_sha256": args.expected_manifest_sha256,
            "expected_original_sha256": args.expected_original_sha256,
            "expected_count": args.expected_count,
        }
        if args.mode == "prepare":
            if args.patch is not None:
                raise ValueError("prepare does not accept --patch")
            payload = prepare_subset(**common)
        else:
            if args.patch is None:
                raise ValueError("merge requires --patch")
            payload = merge_candidates(patch=args.patch, **common)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

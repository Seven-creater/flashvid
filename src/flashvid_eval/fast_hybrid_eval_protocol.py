"""Immutable Fast Hybrid EVA SFT evaluation protocol and result auditing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


DATASETS = ("lvbench", "lsdbench", "cgbench")
SPLITS = ("dev", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def freeze_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
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
    os.replace(temporary, path)


def _index(rows: list[Mapping[str, Any]], path: Path) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id in indexed:
            raise ValueError(f"{path} has a missing or duplicate sample_id")
        indexed[sample_id] = row
    return indexed


def manifest_sample_ids(path: Path) -> frozenset[str]:
    """Return the exact non-empty sample identity set from a frozen manifest."""

    return frozenset(_index(read_jsonl(path), path))


def _validate_candidate(path: Path, manifest_ids: set[str]) -> tuple[str, int]:
    rows = read_jsonl(path)
    indexed = _index(rows, path)
    if set(indexed) != manifest_ids:
        raise ValueError(f"candidate and manifest sample identities differ: {path}")
    for sample_id, row in indexed.items():
        request = row.get("protocol_request") or {}
        prediction = str(row.get("prediction") or "").strip().upper()
        unavailable = (
            row.get("data_unavailable") is True
            and row.get("failure_class") == "data_unavailable"
        )
        if not prediction and not unavailable:
            raise ValueError(f"candidate has unresolved non-data error: {sample_id}")
        if prediction and (len(prediction) != 1 or not "A" <= prediction <= "H"):
            raise ValueError(f"candidate prediction is invalid: {sample_id}")
        if (
            row.get("baseline_mode") != "direct"
            or row.get("sampling_id") != "uniform32"
            or row.get("enable_thinking") is not False
            or request.get("max_tokens") != 512
            or float(request.get("temperature", -1.0)) != 0.0
        ):
            raise ValueError(f"candidate protocol is not clean Direct: {sample_id}")
    return sha256_file(path), len(rows)


def build_protocol(
    *,
    experiment_config_path: Path,
    expected_experiment_config_sha256: str,
    candidate_paths: Mapping[tuple[str, str], Path],
) -> dict[str, Any]:
    expected_config = require_sha256(
        expected_experiment_config_sha256, "expected_experiment_config_sha256"
    )
    if sha256_file(experiment_config_path) != expected_config:
        raise RuntimeError("Fast Hybrid experiment config SHA-256 changed")
    config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Fast Hybrid experiment config must be schema_version=1")
    if set(candidate_paths) != {
        (split, dataset) for split in SPLITS for dataset in DATASETS
    }:
        raise ValueError("candidate paths must cover dev/test x all three datasets")

    datasets: dict[str, Any] = {}
    for split in SPLITS:
        split_datasets: dict[str, Any] = {}
        for dataset in DATASETS:
            source = config["datasets"][dataset]
            manifest = Path(str(source[f"{split}_manifest"]))
            manifest_sha = require_sha256(
                source[f"{split}_manifest_sha256"],
                f"{dataset}.{split}_manifest_sha256",
            )
            if not manifest.is_file() or sha256_file(manifest) != manifest_sha:
                raise RuntimeError(f"{dataset} {split} manifest is missing or changed")
            manifest_rows = read_jsonl(manifest)
            manifest_ids = set(_index(manifest_rows, manifest))
            candidate = candidate_paths[(split, dataset)].resolve()
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            candidate_sha, candidate_count = _validate_candidate(candidate, manifest_ids)
            if candidate_count != len(manifest_rows):
                raise ValueError(f"{dataset} {split} candidate count differs from manifest")
            split_datasets[dataset] = {
                "annotations": str(Path(str(source["annotations"])).resolve()),
                "video_root": str(Path(str(source["video_root"])).resolve()),
                "manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": manifest_sha,
                    "count": len(manifest_rows),
                },
                "candidate": {
                    "path": str(candidate),
                    "sha256": candidate_sha,
                    "count": candidate_count,
                },
            }
        datasets[split] = split_datasets

    teacher = config["teacher"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fast_hybrid_sft_evaluation_protocol",
        "experiment_config": {
            "path": str(experiment_config_path.resolve()),
            "sha256": expected_config,
        },
        "base_model_artifact_sha256": require_sha256(
            teacher["model_artifact_sha256"], "teacher.model_artifact_sha256"
        ),
        "model_path": str(Path(str(teacher["model_path"])).resolve()),
        "agent_version": "fast_hybrid_v2",
        "official_eva_commit": str(teacher["official_eva_commit"]),
        "parameters": {
            "max_turns": 6,
            "max_call_visual_tokens": 12000,
            "max_total_visual_tokens": 24000,
            "temperature": 0.0,
            "seed": 42,
            "timeout": 80.0,
            "concurrency_per_endpoint": 16,
            "endpoints": ["http://127.0.0.1:8200/v1", "http://127.0.0.1:8201/v1"],
        },
        "splits": datasets,
    }
    payload["protocol_fingerprint"] = canonical_sha256(payload)
    return payload


def load_protocol(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = require_sha256(expected_sha256, "expected_protocol_sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError("evaluation protocol file is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "fast_hybrid_sft_evaluation_protocol":
        raise ValueError("invalid Fast Hybrid evaluation protocol")
    fingerprint = value.get("protocol_fingerprint")
    payload = {key: item for key, item in value.items() if key != "protocol_fingerprint"}
    if fingerprint != canonical_sha256(payload):
        raise RuntimeError("evaluation protocol fingerprint mismatch")
    for split in SPLITS:
        for dataset in DATASETS:
            entry = value["splits"][split][dataset]
            for label in ("manifest", "candidate"):
                reference = entry[label]
                target = Path(reference["path"])
                if not target.is_file() or sha256_file(target) != reference["sha256"]:
                    raise RuntimeError(f"{split}/{dataset} {label} is missing or changed")
    return value


def audit_result_file(
    path: Path,
    *,
    dataset: str,
    expected_count: int,
    expected_sample_ids: frozenset[str] | set[str],
    manifest_sha256: str,
    candidate_sha256: str,
    experiment_config_sha256: str,
    served_model_sha256: str,
    teacher_model_sha256: str,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    indexed = _index(rows, path)
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, found {len(rows)}")
    expected_ids = frozenset(str(value) for value in expected_sample_ids)
    if "" in expected_ids or len(expected_ids) != expected_count:
        raise ValueError("expected sample IDs do not match expected_count")
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - set(indexed))
        extras = sorted(set(indexed) - expected_ids)
        raise ValueError(
            f"{path}: sample IDs differ from the frozen manifest "
            f"(missing={missing[:3]}, extras={extras[:3]})"
        )
    failures = leaks = reruns = 0
    for sample_id, row in indexed.items():
        if str(row.get("dataset") or "").lower() != dataset:
            raise ValueError(f"{path}: wrong dataset at {sample_id}")
        expected_fields = {
            "manifest_sha256": manifest_sha256,
            "candidate_results_sha256": candidate_sha256,
            "experiment_config_sha256": experiment_config_sha256,
            "model_artifact_sha256": served_model_sha256,
            "teacher_model_sha256": teacher_model_sha256,
            "agent_version": "fast_hybrid_v2",
        }
        for key, expected in expected_fields.items():
            if row.get(key) != expected:
                raise ValueError(f"{path}: {sample_id} frozen field changed: {key}")
        if row.get("annotation_leak_check") != "passed":
            leaks += 1
        reruns += int(row.get("candidate_rerun") or 0)
        failures += int(
            bool(row.get("error") or row.get("api_error") or row.get("frame_error") or row.get("parse_error"))
        )
        for key in ("end_to_end_total_tokens", "end_to_end_visual_tokens"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{path}: {sample_id} has invalid {key}")
        if row.get("candidate_cost_complete") is not True:
            raise ValueError(f"{path}: {sample_id} frozen-candidate cost is incomplete")
        if row.get("end_to_end_total_tokens_complete") is not True:
            raise ValueError(f"{path}: {sample_id} total-token cost is incomplete")
        if row.get("end_to_end_visual_tokens_complete") is not True:
            raise ValueError(f"{path}: {sample_id} visual-token cost is incomplete")
    if leaks or reruns or failures > expected_count * 0.01:
        raise ValueError(
            f"{path}: audit failed (leaks={leaks}, reruns={reruns}, failures={failures})"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "engineering_failures": failures,
        "annotation_leaks": leaks,
        "candidate_reruns": reruns,
    }


__all__ = [
    "DATASETS",
    "SPLITS",
    "audit_result_file",
    "build_protocol",
    "canonical_sha256",
    "freeze_json",
    "load_protocol",
    "manifest_sample_ids",
    "read_jsonl",
    "require_sha256",
    "sha256_file",
]

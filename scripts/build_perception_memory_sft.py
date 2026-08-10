#!/usr/bin/env python3
"""Export selected Perception-Memory EVA trajectories as process-SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from flashvid_eval.perception_memory_sft import (
    build_perception_memory_sft_records,
    enforce_perception_memory_selection_gate,
)
from flashvid_eval.qwen_sft import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = (
    "config_sha256",
    "experiment_config_sha256",
    "diagnostics_gate_sha256",
    "manifest_sha256",
    "train600_manifest_sha256",
    "dataset_manifest_sha256",
    "candidate_results_sha256",
    "model_artifact_sha256",
    "teacher_model_sha256",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl_once(path: Path) -> tuple[list[dict[str, Any]], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"selected trajectory JSONL does not exist: {resolved}")
    payload = resolved.read_bytes()
    digest = _sha256_bytes(payload)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"selected trajectory JSONL is not UTF-8: {resolved}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON at {resolved}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(row, Mapping):
            raise ValueError(f"trajectory at {resolved}:{line_number} must be an object")
        rows.append(dict(row))
    return rows, digest


def _validate_digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def _hash_field_coverage(
    trajectories: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    values: list[str] = []
    missing = 0
    for index, row in enumerate(trajectories):
        value = row.get(field)
        if value in (None, ""):
            missing += 1
            continue
        values.append(_validate_digest(value, f"trajectory[{index}].{field}"))
    return {
        "covered_rows": len(trajectories) - missing,
        "missing_rows": missing,
        "values": sorted(set(values)),
    }


def _top_level_prompt_hashes(row: Mapping[str, Any], row_index: int) -> list[str]:
    values: list[str] = []
    for field in ("prompt_sha256", "request_prompt_sha256"):
        value = row.get(field)
        if value not in (None, ""):
            values.append(_validate_digest(value, f"trajectory[{row_index}].{field}"))
    bundle = row.get("prompt_hashes")
    if isinstance(bundle, Mapping):
        iterable = bundle.values()
    elif isinstance(bundle, list):
        iterable = bundle
    elif bundle in (None, ""):
        iterable = ()
    else:
        raise ValueError(f"trajectory[{row_index}].prompt_hashes has invalid type")
    for value in iterable:
        values.append(
            _validate_digest(value, f"trajectory[{row_index}].prompt_hashes")
        )
    return values


def _provenance_coverage(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models: list[str] = []
    missing_models = 0
    top_level_prompts: list[str] = []
    request_prompts: list[str] = []
    missing_request_prompts = 0
    request_count = 0
    for row_index, row in enumerate(trajectories):
        model = str(row.get("model") or "").strip()
        if model:
            models.append(model)
        else:
            missing_models += 1
        top_level_prompts.extend(_top_level_prompt_hashes(row, row_index))
        trace = row.get("request_trace")
        if not isinstance(trace, list):
            continue
        for request_index, raw_request in enumerate(trace):
            request_count += 1
            if not isinstance(raw_request, Mapping):
                raise ValueError(
                    f"trajectory[{row_index}].request_trace[{request_index}] must be an object"
                )
            explicit = raw_request.get(
                "prompt_hash", raw_request.get("request_prompt_sha256")
            )
            if explicit not in (None, ""):
                request_prompts.append(
                    _validate_digest(
                        explicit,
                        (
                            f"trajectory[{row_index}].request_trace"
                            f"[{request_index}].prompt_hash"
                        ),
                    )
                )
                continue
            messages = raw_request.get("messages", raw_request.get("request_messages"))
            if isinstance(messages, list) and messages:
                request_prompts.append(canonical_sha256(messages))
            else:
                missing_request_prompts += 1
    return {
        "trajectory_rows": len(trajectories),
        "models": {
            "covered_rows": len(trajectories) - missing_models,
            "missing_rows": missing_models,
            "values": sorted(set(models)),
        },
        "hash_fields": {
            field: _hash_field_coverage(trajectories, field)
            for field in _HASH_FIELDS
        },
        "top_level_prompt_hashes": {
            "count": len(top_level_prompts),
            "values": sorted(set(top_level_prompts)),
        },
        "request_prompt_hashes": {
            "covered_requests": request_count - missing_request_prompts,
            "missing_requests": missing_request_prompts,
            "values": sorted(set(request_prompts)),
        },
    }


def _validate_provenance(coverage: Mapping[str, Any], row_count: int) -> None:
    models = coverage["models"]
    if models["covered_rows"] != row_count or models["values"] != ["Qwen3.5-9B"]:
        raise ValueError("selected trajectories must all use the frozen Qwen3.5-9B")
    hash_fields = coverage["hash_fields"]
    singleton_fields = (
        "experiment_config_sha256",
        "diagnostics_gate_sha256",
        "train600_manifest_sha256",
        "model_artifact_sha256",
    )
    for field in singleton_fields:
        item = hash_fields[field]
        if item["covered_rows"] != row_count or len(item["values"]) != 1:
            raise ValueError(f"selected trajectories have incomplete or mixed {field}")
    for field in ("dataset_manifest_sha256", "candidate_results_sha256"):
        item = hash_fields[field]
        if item["covered_rows"] != row_count or not item["values"]:
            raise ValueError(f"selected trajectories have incomplete {field}")
    prompts = coverage["request_prompt_hashes"]
    if prompts["missing_requests"] or not prompts["covered_requests"]:
        raise ValueError("selected trajectories have incomplete request prompt hashes")


def _unique_sorted_trajectories(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen_samples: set[tuple[str, str]] = set()
    seen_trajectories: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        dataset = str(row.get("dataset") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not dataset or not sample_id or not trajectory_id:
            raise ValueError(
                f"trajectory[{index}] requires dataset, sample_id, and trajectory_id"
            )
        sample_key = (dataset, sample_id)
        if sample_key in seen_samples:
            raise ValueError(f"duplicate selected sample: {dataset}/{sample_id}")
        if trajectory_id in seen_trajectories:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen_samples.add(sample_key)
        seen_trajectories.add(trajectory_id)
        result.append(row)
    if not result:
        raise ValueError("no selected trajectories were provided")
    return sorted(
        result,
        key=lambda row: (
            str(row["trajectory_id"]),
            str(row["dataset"]),
            str(row["sample_id"]),
        ),
    )


def _filter_experiment_config(
    rows: Sequence[Mapping[str, Any]], expected_sha256: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_sha256 is None:
        return [dict(row) for row in rows], {
            "enabled": False,
            "expected_experiment_config_sha256": None,
            "input_rows": len(rows),
            "included_rows": len(rows),
            "excluded_rows": 0,
            "excluded_trajectories": [],
        }
    expected = _validate_digest(
        expected_sha256, "expected_experiment_config_sha256"
    )
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        actual = _validate_digest(
            row.get("experiment_config_sha256"),
            f"trajectory[{index}].experiment_config_sha256",
        )
        if actual == expected:
            included.append(row)
            continue
        excluded.append(
            {
                "dataset": str(row.get("dataset") or ""),
                "sample_id": str(row.get("sample_id") or ""),
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "experiment_config_sha256": actual,
            }
        )
    if not included:
        raise ValueError("experiment-config filter removed every selected trajectory")
    return included, {
        "enabled": True,
        "expected_experiment_config_sha256": expected,
        "input_rows": len(rows),
        "included_rows": len(included),
        "excluded_rows": len(excluded),
        "excluded_trajectories": sorted(
            excluded,
            key=lambda item: (
                item["dataset"], item["sample_id"], item["trajectory_id"]
            ),
        ),
    }


def _selection_threshold_failures(
    summary: Mapping[str, Any],
    *,
    minimum_total: int,
    minimum_per_dataset: int,
    minimum_candidate_fixes: int,
    minimum_candidate_fixes_per_dataset: int,
) -> list[str]:
    failures: list[str] = []
    if int(summary.get("selected_trajectories") or 0) < minimum_total:
        failures.append(f"selected<{minimum_total}")
    selected_by_dataset = summary.get("selected_by_dataset")
    fixes_by_dataset = summary.get("candidate_fixes_by_dataset")
    if not isinstance(selected_by_dataset, Mapping) or not isinstance(
        fixes_by_dataset, Mapping
    ):
        raise ValueError("selection gate summary lacks per-dataset counts")
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        if int(selected_by_dataset.get(dataset, 0)) < minimum_per_dataset:
            failures.append(f"{dataset}<{minimum_per_dataset}")
        if (
            int(fixes_by_dataset.get(dataset, 0))
            < minimum_candidate_fixes_per_dataset
        ):
            failures.append(
                f"{dataset}_candidate_fixes<{minimum_candidate_fixes_per_dataset}"
            )
    if int(summary.get("candidate_fixes") or 0) < minimum_candidate_fixes:
        failures.append(f"candidate_fixes<{minimum_candidate_fixes}")
    return failures


def _record_identity(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("process-SFT record has no metadata object")
    trajectory_id = str(metadata.get("trajectory_id") or "").strip()
    prefix_index = metadata.get("prefix_index")
    target = str(metadata.get("episode_target_type") or "").strip()
    if not trajectory_id or isinstance(prefix_index, bool) or not isinstance(prefix_index, int):
        raise ValueError("process-SFT record requires trajectory_id and integer prefix_index")
    if not target:
        raise ValueError("process-SFT record requires episode_target_type")
    record_id = f"{trajectory_id}#prefix-{prefix_index:+05d}#target-{target}"
    return trajectory_id, prefix_index, target, record_id


def _unique_sorted_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    materialized: list[tuple[tuple[str, int, str, str], dict[str, Any]]] = []
    for raw in records:
        record = deepcopy(dict(raw))
        identity = _record_identity(record)
        record_id = identity[-1]
        if record_id in seen:
            raise ValueError(f"duplicate process-SFT record_id: {record_id}")
        seen.add(record_id)
        metadata = record["metadata"]
        existing = metadata.get("record_id")
        if existing not in (None, record_id):
            raise ValueError("process-SFT record_id does not match stable identity")
        metadata["record_id"] = record_id
        materialized.append((identity, record))
    return [record for _identity, record in sorted(materialized, key=lambda item: item[0])]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _commit_pair(
    output: Path,
    output_payload: bytes,
    summary: Path,
    summary_payload: bytes,
) -> None:
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: set[Path] = set()
    try:
        staged[output] = _stage_bytes(output, output_payload)
        staged[summary] = _stage_bytes(summary, summary_payload)
        for destination in (output, summary):
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.bak"
                )
                os.replace(destination, backup)
                backups[destination] = backup
        for destination in (output, summary):
            os.replace(staged[destination], destination)
            committed.add(destination)
    except BaseException:
        for destination in committed:
            if destination.exists():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def build(
    *,
    selected_paths: Sequence[Path],
    output: Path,
    summary_path: Path,
    minimum_total: int = 360,
    minimum_per_dataset: int = 100,
    minimum_candidate_fixes: int = 90,
    minimum_candidate_fixes_per_dataset: int = 20,
    allow_underfilled_training_set: bool = False,
    underfilled_authorization_reason: str | None = None,
    include_experiment_config_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not selected_paths:
        raise ValueError("at least one --selected JSONL is required")
    output = output.resolve()
    summary_path = summary_path.resolve()
    if output == summary_path:
        raise ValueError("SFT output and summary paths must differ")
    resolved_inputs = [path.resolve() for path in selected_paths]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("the same selected trajectory file was provided more than once")
    if output in resolved_inputs or summary_path in resolved_inputs:
        raise ValueError("outputs may not overwrite a selected trajectory input")
    existing = [path for path in (output, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite to replace the complete output pair: "
            + ", ".join(str(path) for path in existing)
        )
    authorization_reason = str(underfilled_authorization_reason or "").strip()
    if allow_underfilled_training_set and not authorization_reason:
        raise ValueError("underfilled training requires an authorization reason")
    if not allow_underfilled_training_set and authorization_reason:
        raise ValueError(
            "underfilled authorization reason requires allow_underfilled_training_set"
        )

    all_rows: list[dict[str, Any]] = []
    input_entries: list[dict[str, Any]] = []
    for path in sorted(resolved_inputs):
        rows, digest = _read_jsonl_once(path)
        all_rows.extend(rows)
        input_entries.append({"path": str(path), "sha256": digest, "rows": len(rows)})
    filtered_rows, filter_audit = _filter_experiment_config(
        all_rows, include_experiment_config_sha256
    )
    trajectories = _unique_sorted_trajectories(filtered_rows)
    raw_records = [
        record
        for trajectory in trajectories
        for record in build_perception_memory_sft_records(trajectory)
    ]
    records = _unique_sorted_records(raw_records)
    gate_kwargs = {
        "minimum_total": minimum_total,
        "minimum_per_dataset": minimum_per_dataset,
        "minimum_candidate_fixes": minimum_candidate_fixes,
        "minimum_candidate_fixes_per_dataset": minimum_candidate_fixes_per_dataset,
    }
    if allow_underfilled_training_set:
        gate = enforce_perception_memory_selection_gate(
            trajectories,
            records,
            minimum_total=0,
            minimum_per_dataset=0,
            minimum_candidate_fixes=0,
            minimum_candidate_fixes_per_dataset=0,
        )
    else:
        gate = enforce_perception_memory_selection_gate(
            trajectories, records, **gate_kwargs
        )
    unmet_conditions = _selection_threshold_failures(gate, **gate_kwargs)
    output_payload = _jsonl_bytes(records)
    output_sha256 = _sha256_bytes(output_payload)
    provenance = _provenance_coverage(trajectories)
    _validate_provenance(provenance, len(trajectories))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "selected_inputs": input_entries,
        "selected_input_set_sha256": canonical_sha256(input_entries),
        "selected_filter": filter_audit,
        "selection_gate": {
            "thresholds": {
                "minimum_total": minimum_total,
                "minimum_per_dataset": minimum_per_dataset,
                "minimum_candidate_fixes": minimum_candidate_fixes,
                "minimum_candidate_fixes_per_dataset": (
                    minimum_candidate_fixes_per_dataset
                ),
            },
            "result": gate,
            "passed_frozen_gate": not unmet_conditions,
            "unmet_conditions": unmet_conditions,
            "underfilled_override": {
                "enabled": allow_underfilled_training_set,
                "applied": bool(allow_underfilled_training_set and unmet_conditions),
                "authorization_reason": authorization_reason or None,
            },
        },
        "provenance_coverage": provenance,
        "sorting": ["trajectory_id", "prefix_index", "episode_target_type"],
        "outputs": {
            "sft_jsonl": {
                "path": str(output),
                "rows": len(records),
                "sha256": output_sha256,
            },
            "summary_json": {"path": str(summary_path)},
        },
    }
    summary_payload = _json_bytes(summary)
    _commit_pair(output, output_payload, summary_path, summary_payload)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Perception-Memory EVA process-SFT data from "
            "stable selected trajectories."
        )
    )
    parser.add_argument("--selected", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-total", type=int, default=360)
    parser.add_argument("--minimum-per-dataset", type=int, default=100)
    parser.add_argument("--minimum-candidate-fixes", type=int, default=90)
    parser.add_argument("--minimum-candidate-fixes-per-dataset", type=int, default=20)
    parser.add_argument(
        "--allow-underfilled-training-set",
        action="store_true",
        help=(
            "Explicitly authorize training below the frozen quantity thresholds; "
            "all stability, leakage, provenance, frame, and hash gates remain enforced."
        ),
    )
    parser.add_argument("--underfilled-authorization-reason")
    parser.add_argument("--include-experiment-config-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    frozen_minima = (360, 100, 90, 20)
    requested_minima = (
        args.minimum_total,
        args.minimum_per_dataset,
        args.minimum_candidate_fixes,
        args.minimum_candidate_fixes_per_dataset,
    )
    if any(requested < frozen for requested, frozen in zip(requested_minima, frozen_minima)):
        parser.error("process-SFT selection thresholds may not weaken the frozen plan")
    if args.allow_underfilled_training_set and not str(
        args.underfilled_authorization_reason or ""
    ).strip():
        parser.error("--allow-underfilled-training-set requires an authorization reason")
    if (
        args.underfilled_authorization_reason
        and not args.allow_underfilled_training_set
    ):
        parser.error(
            "--underfilled-authorization-reason requires "
            "--allow-underfilled-training-set"
        )
    try:
        result = build(
            selected_paths=args.selected,
            output=args.output,
            summary_path=args.summary,
            minimum_total=args.minimum_total,
            minimum_per_dataset=args.minimum_per_dataset,
            minimum_candidate_fixes=args.minimum_candidate_fixes,
            minimum_candidate_fixes_per_dataset=(
                args.minimum_candidate_fixes_per_dataset
            ),
            allow_underfilled_training_set=args.allow_underfilled_training_set,
            underfilled_authorization_reason=(
                args.underfilled_authorization_reason
            ),
            include_experiment_config_sha256=(
                args.include_experiment_config_sha256
            ),
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    envelope = {
        "status": "passed",
        **result,
        "summary_sha256": _sha256_bytes(_json_bytes(result)),
    }
    print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

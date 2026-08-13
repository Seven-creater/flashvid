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
    QUALITY_CONTRACT_VERSION,
    VISUAL_PATH_CLASSIFIER_VERSION,
    build_perception_memory_sft_records,
    classify_visual_path,
    enforce_perception_memory_selection_gate,
)
from flashvid_eval.qwen_sft import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = (
    "config_sha256",
    "experiment_config_sha256",
    "diagnostics_gate_sha256",
    "training_source_lock_sha256",
    "role_prompt_schema_bundle_sha256",
    "manifest_sha256",
    "train600_manifest_sha256",
    "dataset_manifest_sha256",
    "candidate_results_sha256",
    "model_artifact_sha256",
    "teacher_model_sha256",
)
_ROLE_RUNTIME_FIELDS = (
    "role_separated_runtime_version",
    "controller_output_constraint_version",
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


def _string_field_coverage(
    trajectories: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    values: list[str] = []
    missing = 0
    for row in trajectories:
        value = str(row.get(field) or "").strip()
        if value:
            values.append(value)
        else:
            missing += 1
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
        "role_runtime_fields": {
            field: _string_field_coverage(trajectories, field)
            for field in _ROLE_RUNTIME_FIELDS
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


def _validate_provenance(
    coverage: Mapping[str, Any],
    row_count: int,
    *,
    expected_training_source_lock_sha256: str | None = None,
    expected_role_separated_runtime_version: str | None = None,
    expected_role_prompt_schema_bundle_sha256: str | None = None,
    expected_controller_output_constraint_version: str | None = None,
) -> None:
    models = coverage["models"]
    if models["covered_rows"] != row_count or models["values"] != ["Qwen3.5-9B"]:
        raise ValueError("selected trajectories must all use the frozen Qwen3.5-9B")
    hash_fields = coverage["hash_fields"]
    singleton_fields = (
        "experiment_config_sha256",
        "train600_manifest_sha256",
        "model_artifact_sha256",
    )
    for field in singleton_fields:
        item = hash_fields[field]
        if item["covered_rows"] != row_count or len(item["values"]) != 1:
            raise ValueError(f"selected trajectories have incomplete or mixed {field}")
    diagnostics = hash_fields["diagnostics_gate_sha256"]
    training_lock = hash_fields["training_source_lock_sha256"]
    diagnostics_complete = (
        diagnostics["covered_rows"] == row_count
        and len(diagnostics["values"]) == 1
    )
    training_lock_complete = (
        training_lock["covered_rows"] == row_count
        and len(training_lock["values"]) == 1
    )
    if diagnostics_complete == training_lock_complete:
        raise ValueError(
            "selected trajectories must use exactly one complete source gate: "
            "diagnostics_gate_sha256 or training_source_lock_sha256"
        )
    if diagnostics["covered_rows"] not in {0, row_count}:
        raise ValueError("selected trajectories have partial diagnostics_gate_sha256")
    if training_lock["covered_rows"] not in {0, row_count}:
        raise ValueError("selected trajectories have partial training_source_lock_sha256")
    if expected_training_source_lock_sha256 is not None:
        expected = _validate_digest(
            expected_training_source_lock_sha256,
            "expected_training_source_lock_sha256",
        )
        if not training_lock_complete or training_lock["values"] != [expected]:
            raise ValueError(
                "selected trajectories are not bound to the expected training source lock"
            )
    expected_role_contract = {
        "role_separated_runtime_version": expected_role_separated_runtime_version,
        "role_prompt_schema_bundle_sha256": (
            expected_role_prompt_schema_bundle_sha256
        ),
        "controller_output_constraint_version": (
            expected_controller_output_constraint_version
        ),
    }
    supplied_contract_fields = {
        field for field, value in expected_role_contract.items() if value is not None
    }
    if training_lock_complete:
        if supplied_contract_fields != set(expected_role_contract):
            raise ValueError(
                "role-separated trajectories require the complete expected runtime contract"
            )
        expected_role_contract["role_prompt_schema_bundle_sha256"] = _validate_digest(
            expected_role_contract["role_prompt_schema_bundle_sha256"],
            "expected_role_prompt_schema_bundle_sha256",
        )
        for field in _ROLE_RUNTIME_FIELDS:
            expected = str(expected_role_contract[field] or "").strip()
            if not expected:
                raise ValueError(f"expected_{field} must be non-empty")
            item = coverage["role_runtime_fields"][field]
            if item["covered_rows"] != row_count or item["values"] != [expected]:
                raise ValueError(
                    f"selected trajectories are not bound to the expected {field}"
                )
        prompt_bundle = hash_fields["role_prompt_schema_bundle_sha256"]
        if (
            prompt_bundle["covered_rows"] != row_count
            or prompt_bundle["values"]
            != [expected_role_contract["role_prompt_schema_bundle_sha256"]]
        ):
            raise ValueError(
                "selected trajectories are not bound to the expected "
                "role_prompt_schema_bundle_sha256"
            )
    elif supplied_contract_fields:
        raise ValueError(
            "expected role runtime contract is only valid with a training source lock"
        )
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


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, int, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("process-SFT record has no metadata object")
    trajectory_id = str(metadata.get("trajectory_id") or "").strip()
    role = str(metadata.get("process_role") or "").strip().casefold()
    if not trajectory_id or role not in {"planner", "observer"}:
        raise ValueError("process-SFT record requires trajectory_id and role episode")
    if role == "planner":
        episode_index = -1
    else:
        raw_index = metadata.get("terminal_prefix_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError("Observer process-SFT record requires a prefix index")
        episode_index = raw_index
    record_id = f"{trajectory_id}#role-{role}#episode-{episode_index:+05d}"
    return trajectory_id, role, episode_index, record_id


def _unique_sorted_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    materialized: list[tuple[tuple[str, str, int, str], dict[str, Any]]] = []
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
    include_observer: bool = False,
    completion_gate_kind: str = "visual_csv",
    include_experiment_config_sha256: str | None = None,
    expected_training_source_lock_sha256: str | None = None,
    expected_role_separated_runtime_version: str | None = None,
    expected_role_prompt_schema_bundle_sha256: str | None = None,
    expected_controller_output_constraint_version: str | None = None,
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
        for record in build_perception_memory_sft_records(
            trajectory,
            include_observer=include_observer,
            completion_gate_kind=completion_gate_kind,
        )
    ]
    records = _unique_sorted_records(raw_records)
    if completion_gate_kind == "visual_csv":
        for trajectory in trajectories:
            trajectory["quality_contract_version"] = QUALITY_CONTRACT_VERSION
            trajectory["visual_path_classifier_version"] = (
                VISUAL_PATH_CLASSIFIER_VERSION
            )
            trajectory["visual_path_family"] = classify_visual_path(trajectory)
    gate = enforce_perception_memory_selection_gate(
        trajectories,
        records,
        completion_gate_kind=completion_gate_kind,
    )
    output_payload = _jsonl_bytes(records)
    output_sha256 = _sha256_bytes(output_payload)
    provenance = _provenance_coverage(trajectories)
    _validate_provenance(
        provenance,
        len(trajectories),
        expected_training_source_lock_sha256=(
            expected_training_source_lock_sha256
        ),
        expected_role_separated_runtime_version=(
            expected_role_separated_runtime_version
        ),
        expected_role_prompt_schema_bundle_sha256=(
            expected_role_prompt_schema_bundle_sha256
        ),
        expected_controller_output_constraint_version=(
            expected_controller_output_constraint_version
        ),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "selected_inputs": input_entries,
        "selected_input_set_sha256": canonical_sha256(input_entries),
        "selected_filter": filter_audit,
        "selection_quality_gate": {
            "result": gate,
            "passed": True,
            "quantity_is_advisory": True,
        },
        "quantity_distribution": {
            key: deepcopy(gate[key])
            for key in (
                "selected_trajectories",
                "selected_by_dataset",
                "candidate_fixes",
                "candidate_fixes_by_dataset",
                "candidate_training_strata",
                "visual_path_distribution",
                "prefixes",
                "planner_decisions",
                "role_episodes",
                "assistant_targets",
            )
        },
        "export_policy": {
            "planner_required": True,
            "observer_included": include_observer,
            "completion_gate_kind": completion_gate_kind,
            "quality_contract_version": QUALITY_CONTRACT_VERSION,
            "visual_path_classifier_version": VISUAL_PATH_CLASSIFIER_VERSION,
            "training_source_lock_sha256": (
                expected_training_source_lock_sha256
            ),
            "role_runtime_contract": {
                "role_separated_runtime_version": (
                    expected_role_separated_runtime_version
                ),
                "role_prompt_schema_bundle_sha256": (
                    expected_role_prompt_schema_bundle_sha256
                ),
                "controller_output_constraint_version": (
                    expected_controller_output_constraint_version
                ),
            },
        },
        "provenance_coverage": provenance,
        "sorting": ["trajectory_id", "process_role", "terminal_prefix_index"],
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
    parser.add_argument(
        "--include-observer",
        action="store_true",
        help="Also export a separate full Observer episode for an independent adapter.",
    )
    parser.add_argument(
        "--completion-gate-kind",
        choices=("visual_csv", "legacy_prefix_judge"),
        default="visual_csv",
    )
    parser.add_argument("--include-experiment-config-sha256")
    parser.add_argument("--expected-training-source-lock-sha256")
    parser.add_argument("--expected-role-separated-runtime-version")
    parser.add_argument("--expected-role-prompt-schema-bundle-sha256")
    parser.add_argument("--expected-controller-output-constraint-version")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build(
            selected_paths=args.selected,
            output=args.output,
            summary_path=args.summary,
            include_observer=args.include_observer,
            completion_gate_kind=args.completion_gate_kind,
            include_experiment_config_sha256=(
                args.include_experiment_config_sha256
            ),
            expected_training_source_lock_sha256=(
                args.expected_training_source_lock_sha256
            ),
            expected_role_separated_runtime_version=(
                args.expected_role_separated_runtime_version
            ),
            expected_role_prompt_schema_bundle_sha256=(
                args.expected_role_prompt_schema_bundle_sha256
            ),
            expected_controller_output_constraint_version=(
                args.expected_controller_output_constraint_version
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

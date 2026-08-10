"""Immutable repair-scope preparation and replay-result reconciliation.

The base replay output is never edited.  ``prepare_repair_scope`` freezes the
exact failed subset of a complete base run.  ``merge_replay_results`` later
accepts one replacement result per failed source trajectory, validates its
lineage, and materializes a new prefix-bindable success-only artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .perception_memory_prefix_judge import bind_prefix_jobs
from .perception_memory_replay import (
    canonical_sha256,
    file_sha256,
    public_model_sample,
)
from .runner import parse_question_time_range


REPAIR_SCOPE_VERSION = "perception_memory_repair_scope_v2"
_CACHED_INPUT = "cached_repair_input.jsonl"
_EXPLICIT_TIME_INPUT = "explicit_time_mismatch.jsonl"
_FROZEN_SCOPE = "frozen_scope.json"
_MERGED_SUCCESS = "merged_success.jsonl"
_DOUBLE_FAILURES = "double_failures.jsonl"
_MERGE_SUMMARY = "merge_summary.json"
_MM_SS_TOKEN = re.compile(r"(?<![\d:])(?P<minutes>\d+):(?P<seconds>[0-5]\d)(?![:\d])")
_TIME_RANGE_CONNECTOR = re.compile(
    r"(?:-|\u2013|\u2014|~|to|through|until|from)", re.IGNORECASE
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(dict(value))
    return rows


def _read_shards(
    paths: Sequence[Path], *, label: str
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    if not paths:
        raise ValueError(f"at least one {label} shard is required")
    rows: list[dict[str, Any]] = []
    file_sha_by_id: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    seen_file_hashes: set[str] = set()
    identity = _source_identity if label == "source" else _base_identity
    for path in paths:
        shard = _read_jsonl(path)
        digest = file_sha256(path)
        if digest in seen_file_hashes:
            raise ValueError(f"duplicate {label} shard content: {path}")
        seen_file_hashes.add(digest)
        files.append(
            {"path": str(path.resolve()), "sha256": digest, "rows": len(shard)}
        )
        for row in shard:
            source_id = identity(row)[2]
            if source_id in file_sha_by_id:
                raise ValueError(
                    f"{label} shards contain duplicate trajectory ID: {source_id}"
                )
            file_sha_by_id[source_id] = digest
            rows.append(row)
    return rows, file_sha_by_id, files


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows
    ).encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_directory_atomic(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    """Create a whole result directory without overwriting any prior artifact."""

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name, content in artifacts.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _text(row.get("dataset"), "dataset"),
        _text(row.get("sample_id"), "sample_id"),
        _text(row.get("trajectory_id"), "trajectory_id"),
    )


def _base_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _text(row.get("dataset"), "dataset"),
        _text(row.get("sample_id"), "sample_id"),
        _text(row.get("source_trajectory_id"), "source_trajectory_id"),
    )


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    identity,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset, sample_id, source_id = identity(row)
        if source_id in indexed:
            raise ValueError(f"{label} contains duplicate trajectory ID: {source_id}")
        indexed[source_id] = dict(row)
    return indexed


def _result_failed(row: Mapping[str, Any], *, label: str) -> bool:
    error = row.get("error")
    error_type = row.get("error_type")
    if error is None and error_type in (None, ""):
        return False
    if not isinstance(error, str) or not error.strip():
        raise ValueError(f"{label} has inconsistent error/error_type fields")
    if not isinstance(error_type, str) or not error_type.strip():
        raise ValueError(f"{label} has inconsistent error/error_type fields")
    return True


def _validate_result_lineage(
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    original_source_sha256: str,
    require_original_source_file: bool,
) -> None:
    dataset, sample_id, source_id = _source_identity(source)
    if _base_identity(result) != (dataset, sample_id, source_id):
        raise ValueError(f"result identity differs from source: {source_id}")
    expected_row_sha = canonical_sha256(source)
    if result.get("source_row_sha256") != expected_row_sha:
        raise ValueError(f"result source_row_sha256 mismatch: {source_id}")
    if require_original_source_file and result.get("source_file_sha256") != (
        original_source_sha256
    ):
        raise ValueError(f"result source_file_sha256 mismatch: {source_id}")


def _validate_prefix_success(row: Mapping[str, Any], *, label: str) -> None:
    if _result_failed(row, label=label):
        raise ValueError(f"{label} is not a successful trajectory")
    try:
        jobs = bind_prefix_jobs([row])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} cannot bind to prefix jobs: {error}") from error
    if not jobs:
        raise ValueError(f"{label} produced no prefix jobs")


def _prefix_failure_reason(
    row: Mapping[str, Any], *, label: str
) -> dict[str, str] | None:
    """Return a frozen reason when an otherwise error-free row is unbindable."""

    if _result_failed(row, label=label):
        return {
            "kind": "reported_error",
            "error_type": _text(row.get("error_type"), f"{label} error_type"),
            "detail": _text(row.get("error"), f"{label} error"),
        }
    try:
        _validate_prefix_success(row, label=label)
    except ValueError as error:
        return {
            "kind": "prefix_unbindable",
            "error_type": type(error).__name__,
            "detail": str(error),
        }
    return None


def _validate_success_public_lineage(
    result: Mapping[str, Any], source: Mapping[str, Any], *, label: str
) -> None:
    """Prevent a hash-carrying result from swapping question/candidate content."""

    sample = public_model_sample(source)
    expected_public = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "video": sample.video,
        "question": sample.question,
        "choices": dict(sample.choices),
    }
    if result.get("public_sample") != expected_public:
        raise ValueError(f"{label} public_sample differs from immutable source")
    if result.get("candidate_answer") != sample.candidate_answer:
        raise ValueError(f"{label} candidate differs from immutable source")


def _requested_intervals(source: Mapping[str, Any]) -> list[list[float]]:
    raw_steps = source.get("tool_steps", source.get("tool_calls"))
    if not isinstance(raw_steps, list):
        return []
    intervals: list[list[float]] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping):
            continue
        arguments = raw.get("arguments")
        values = arguments if isinstance(arguments, Mapping) else raw
        try:
            start = float(values["start_time"])
            end = float(values["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            intervals.append([start, end])
    return intervals


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(float(left[1]), float(right[1]))


def _legacy_decimalized_question_interval(question: str) -> list[float] | None:
    """Reconstruct the retired MM:SS-as-decimal parser from public text only."""

    matches = list(_MM_SS_TOKEN.finditer(question))
    if not matches:
        return None

    def decimalized(match: re.Match[str]) -> float:
        return float(match.group("minutes")) + float(match.group("seconds")) / 100.0

    first = decimalized(matches[0])
    if len(matches) == 1:
        return [max(0.0, first - 1.0), first + 1.0]
    between = question[matches[0].end() : matches[1].start()]
    if not _TIME_RANGE_CONNECTOR.search(between):
        return None
    second = decimalized(matches[1])
    return [min(first, second), max(first, second)]


def _same_interval(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
        for a, b in zip(left, right, strict=True)
    )


def _explicit_time_mismatch(
    source: Mapping[str, Any], base_failure: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Classify bad cached time parsing without consulting private annotations."""

    if "invalid_frame_reference" not in str(base_failure.get("error") or ""):
        return None
    sample = public_model_sample(source)
    parsed = parse_question_time_range(sample.question)
    legacy_decimalized = _legacy_decimalized_question_interval(sample.question)
    requested = _requested_intervals(source)
    if parsed is None or legacy_decimalized is None or not requested:
        return None
    parsed_list = [float(parsed[0]), float(parsed[1])]
    match = re.search(r"perception step\s+(\d+)", str(base_failure.get("error") or ""))
    if match is None:
        return None
    failed_step_index = int(match.group(1))
    raw_steps = source.get("tool_steps", source.get("tool_calls"))
    if not isinstance(raw_steps, list) or failed_step_index >= len(raw_steps):
        return None
    raw_failed_step = raw_steps[failed_step_index]
    if not isinstance(raw_failed_step, Mapping):
        return None
    raw_arguments = raw_failed_step.get("arguments")
    failed_values = (
        raw_arguments if isinstance(raw_arguments, Mapping) else raw_failed_step
    )
    try:
        failed_interval = [
            float(failed_values["start_time"]),
            float(failed_values["end_time"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        failed_interval[1] <= failed_interval[0]
        or _intervals_overlap(parsed_list, failed_interval)
        or not _same_interval(failed_interval, legacy_decimalized)
    ):
        return None
    public_sample = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "video": sample.video,
        "question": sample.question,
        "choices": dict(sample.choices),
    }
    dataset, sample_id, source_id = _source_identity(source)
    return {
        "schema_version": 1,
        "rescue_mode": "rescue_explicit_time",
        "classification": "source_explicit_time_parse_mismatch",
        "reason": "explicit_time_source_interval_mismatch",
        "parsed_time_source": "public_question",
        "dataset": dataset,
        "sample_id": sample_id,
        "source_trajectory_id": source_id,
        "source_row_sha256": canonical_sha256(source),
        "base_row_sha256": canonical_sha256(base_failure),
        "parsed_time_range": parsed_list,
        "parsed_question_interval": parsed_list,
        "failed_step_index": failed_step_index,
        "failed_step_requested_interval": failed_interval,
        "legacy_decimalized_interval": legacy_decimalized,
        "source_requested_intervals": requested,
        "public_sample": public_sample,
        "source_row": dict(source),
    }


def _scope_entries(
    source_rows: Sequence[Mapping[str, Any]],
    base_by_id: Mapping[str, Mapping[str, Any]],
    failure_reasons: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_scope: list[dict[str, Any]] = []
    base_scope: list[dict[str, Any]] = []
    for source in source_rows:
        dataset, sample_id, source_id = _source_identity(source)
        base = base_by_id[source_id]
        source_scope.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_trajectory_id": source_id,
                "source_row_sha256": canonical_sha256(source),
            }
        )
        base_scope.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_trajectory_id": source_id,
                "base_row_sha256": canonical_sha256(base),
                "status": "failed" if source_id in failure_reasons else "success",
                "failure_reason": failure_reasons.get(source_id),
            }
        )
    return source_scope, base_scope


def prepare_repair_scope(
    *,
    source_paths: Sequence[Path],
    base_results_paths: Sequence[Path],
    output_dir: Path,
    expected_rows: int = 2917,
    expected_explicit_time_rows: int | None = 37,
    expected_explicit_time_samples: int | None = 6,
) -> dict[str, Any]:
    """Freeze the complete failed subset without mutating the base run."""

    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    source_rows, source_file_by_id, source_files = _read_shards(
        source_paths, label="source"
    )
    base_rows, base_file_by_id, base_files = _read_shards(
        base_results_paths, label="base"
    )
    if len(source_rows) != expected_rows or len(base_rows) != expected_rows:
        raise ValueError(
            "formal repair preparation requires complete source/base scope: "
            f"expected {expected_rows}, found source={len(source_rows)}, "
            f"base={len(base_rows)}"
        )
    source_by_id = _index_unique(source_rows, _source_identity, label="source JSONL")
    base_by_id = _index_unique(base_rows, _base_identity, label="base JSONL")
    source_ids = set(source_by_id)
    base_ids = set(base_by_id)
    if source_ids != base_ids:
        missing = sorted(source_ids - base_ids)
        extras = sorted(base_ids - source_ids)
        raise ValueError(
            "base scope differs from immutable source: "
            f"missing={missing[:3]}, extras={extras[:3]}"
        )

    ordered_sources: list[dict[str, Any]] = []
    failure_reasons: dict[str, dict[str, str]] = {}
    for source in source_rows:
        copied = dict(source)
        ordered_sources.append(copied)
        _, _, source_id = _source_identity(source)
        base = dict(base_by_id[source_id])
        _validate_result_lineage(
            base,
            source,
            original_source_sha256=source_file_by_id[source_id],
            require_original_source_file=True,
        )
        if not _result_failed(base, label=f"base result {source_id}"):
            _validate_success_public_lineage(
                base, source, label=f"base result {source_id}"
            )
        failure_reason = _prefix_failure_reason(base, label=f"base result {source_id}")
        if failure_reason is not None:
            failure_reasons[source_id] = failure_reason

    cached_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    base_success_ids: list[str] = []
    failure_ids: list[str] = []
    for source in ordered_sources:
        _, _, source_id = _source_identity(source)
        base = dict(base_by_id[source_id])
        if source_id not in failure_reasons:
            base_success_ids.append(source_id)
            continue
        failure_ids.append(source_id)
        mismatch = _explicit_time_mismatch(source, base)
        if mismatch is None:
            cached_rows.append(source)
        else:
            mismatch.update(
                {
                    "source_file_sha256": source_file_by_id[source_id],
                    "base_file_sha256": base_file_by_id[source_id],
                }
            )
            mismatch_rows.append(mismatch)

    cached_bytes = _jsonl_bytes(cached_rows)
    mismatch_bytes = _jsonl_bytes(mismatch_rows)
    explicit_samples = {
        (str(row["dataset"]), str(row["sample_id"])) for row in mismatch_rows
    }
    if expected_explicit_time_rows is not None and len(mismatch_rows) != (
        expected_explicit_time_rows
    ):
        raise ValueError(
            "explicit-time mismatch row count differs from the registered scope: "
            f"expected {expected_explicit_time_rows}, found {len(mismatch_rows)}"
        )
    if expected_explicit_time_samples is not None and len(explicit_samples) != (
        expected_explicit_time_samples
    ):
        raise ValueError(
            "explicit-time mismatch sample count differs from the registered scope: "
            f"expected {expected_explicit_time_samples}, found {len(explicit_samples)}"
        )
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    mismatch_sha = hashlib.sha256(mismatch_bytes).hexdigest()
    source_scope, base_scope = _scope_entries(
        ordered_sources, base_by_id, failure_reasons
    )
    frozen_scope = {
        "schema_version": 1,
        "repair_scope_version": REPAIR_SCOPE_VERSION,
        "expected_rows": expected_rows,
        "expected_explicit_time_rows": expected_explicit_time_rows,
        "expected_explicit_time_samples": expected_explicit_time_samples,
        "source": {
            "files": source_files,
            "rows": len(source_rows),
            "scope_sha256": canonical_sha256(source_scope),
        },
        "base": {
            "files": base_files,
            "rows": len(base_rows),
            "scope_sha256": canonical_sha256(base_scope),
        },
        "base_success_count": len(base_success_ids),
        "base_failure_count": len(failure_ids),
        "base_success_ids": base_success_ids,
        "base_failure_ids": failure_ids,
        "base_failure_reasons": failure_reasons,
        "cached_repair_ids": [row["trajectory_id"] for row in cached_rows],
        "explicit_time_mismatch_ids": [
            row["source_trajectory_id"] for row in mismatch_rows
        ],
        "artifacts": {
            _CACHED_INPUT: {"rows": len(cached_rows), "sha256": cached_sha},
            _EXPLICIT_TIME_INPUT: {
                "rows": len(mismatch_rows),
                "sha256": mismatch_sha,
            },
        },
    }
    scope_bytes = _json_bytes(frozen_scope)
    _write_directory_atomic(
        output_dir,
        {
            _CACHED_INPUT: cached_bytes,
            _EXPLICIT_TIME_INPUT: mismatch_bytes,
            _FROZEN_SCOPE: scope_bytes,
        },
    )
    return {
        "status": "passed",
        "output_dir": str(output_dir.resolve()),
        "source_rows": len(source_rows),
        "base_success": len(base_success_ids),
        "base_failed": len(failure_ids),
        "cached_repair": len(cached_rows),
        "explicit_time_rescue": len(mismatch_rows),
        "source_files": source_files,
        "base_files": base_files,
        "frozen_scope_sha256": file_sha256(output_dir / _FROZEN_SCOPE),
    }


def _load_scope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("frozen scope is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("frozen scope must be an object")
    scope = dict(value)
    if (
        scope.get("schema_version") != 1
        or scope.get("repair_scope_version") != REPAIR_SCOPE_VERSION
    ):
        raise ValueError("unsupported frozen repair scope")
    return scope


def _verify_prepared_artifacts(
    scope_path: Path, scope: Mapping[str, Any]
) -> tuple[Path, Path]:
    artifacts = scope.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("frozen scope has no artifact lineage")
    cached_path = scope_path.parent / _CACHED_INPUT
    mismatch_path = scope_path.parent / _EXPLICIT_TIME_INPUT
    for path in (cached_path, mismatch_path):
        record = artifacts.get(path.name)
        if not isinstance(record, Mapping):
            raise ValueError(f"frozen scope lacks artifact: {path.name}")
        rows = _read_jsonl(path)
        if len(rows) != int(record.get("rows", -1)):
            raise ValueError(f"prepared artifact row count changed: {path.name}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"prepared artifact hash changed: {path.name}")
    return cached_path, mismatch_path


def merge_replay_results(
    *,
    source_paths: Sequence[Path],
    base_results_paths: Sequence[Path],
    frozen_scope_path: Path,
    replacement_result_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Reconcile immutable base successes with one strictly bound repair result."""

    if not replacement_result_paths:
        raise ValueError("at least one replacement result JSONL is required")
    scope = _load_scope(frozen_scope_path)
    scope_sha = file_sha256(frozen_scope_path)
    source_record = scope.get("source")
    base_record = scope.get("base")
    if not isinstance(source_record, Mapping) or not isinstance(base_record, Mapping):
        raise ValueError("frozen scope lacks source/base lineage")
    source_rows, source_file_by_id, source_files = _read_shards(
        source_paths, label="source"
    )
    base_rows, _base_file_by_id, base_files = _read_shards(
        base_results_paths, label="base"
    )
    if source_files != source_record.get("files"):
        raise ValueError("immutable source shards differ from frozen scope")
    if base_files != base_record.get("files"):
        raise ValueError("base result shards differ from frozen scope")
    if len(source_rows) != int(source_record.get("rows", -1)) or len(base_rows) != int(
        base_record.get("rows", -1)
    ):
        raise ValueError("source/base row count differs from frozen scope")
    source_by_id = _index_unique(source_rows, _source_identity, label="source JSONL")
    base_by_id = _index_unique(base_rows, _base_identity, label="base JSONL")
    if set(source_by_id) != set(base_by_id):
        raise ValueError("source/base scope differs during merge")
    frozen_failure_reasons = scope.get("base_failure_reasons")
    if not isinstance(frozen_failure_reasons, Mapping):
        raise ValueError("frozen scope lacks base failure reasons")
    current_failure_reasons: dict[str, dict[str, str]] = {}
    for source in source_rows:
        _, _, source_id = _source_identity(source)
        base = base_by_id[source_id]
        if not _result_failed(base, label=f"base result {source_id}"):
            _validate_success_public_lineage(
                base, source, label=f"base result {source_id}"
            )
        failure_reason = _prefix_failure_reason(base, label=f"base result {source_id}")
        if failure_reason is not None:
            current_failure_reasons[source_id] = failure_reason
    if current_failure_reasons != frozen_failure_reasons:
        raise ValueError("base failure classification differs from frozen scope")
    source_scope, base_scope = _scope_entries(
        source_rows, base_by_id, current_failure_reasons
    )
    if canonical_sha256(source_scope) != source_record.get("scope_sha256"):
        raise ValueError("source row scope differs from frozen scope")
    if canonical_sha256(base_scope) != base_record.get("scope_sha256"):
        raise ValueError("base row scope differs from frozen scope")

    cached_path, mismatch_path = _verify_prepared_artifacts(frozen_scope_path, scope)
    cached_rows = _read_jsonl(cached_path)
    mismatch_rows = _read_jsonl(mismatch_path)
    cached_ids = list(scope.get("cached_repair_ids") or [])
    rescue_ids = list(scope.get("explicit_time_mismatch_ids") or [])
    failure_ids = list(scope.get("base_failure_ids") or [])
    if set(frozen_failure_reasons) != set(failure_ids):
        raise ValueError("frozen base failure reasons differ from failed scope")
    if len(failure_ids) != len(set(failure_ids)) or set(failure_ids) != (
        set(cached_ids) | set(rescue_ids)
    ):
        raise ValueError("frozen failed repair lanes are inconsistent")
    if set(cached_ids) & set(rescue_ids):
        raise ValueError("cached and explicit-time repair scopes overlap")
    if [_source_identity(row)[2] for row in cached_rows] != cached_ids:
        raise ValueError("cached repair artifact IDs differ from frozen scope")
    if [row.get("source_trajectory_id") for row in mismatch_rows] != rescue_ids:
        raise ValueError("explicit-time artifact IDs differ from frozen scope")
    for row in mismatch_rows:
        source_id = _text(row.get("source_trajectory_id"), "source_trajectory_id")
        embedded = row.get("source_row")
        if not isinstance(embedded, Mapping) or canonical_sha256(embedded) != (
            canonical_sha256(source_by_id[source_id])
        ):
            raise ValueError(f"explicit-time source row changed: {source_id}")

    replacements: dict[str, dict[str, Any]] = {}
    replacement_files: list[dict[str, Any]] = []
    for path in replacement_result_paths:
        rows = _read_jsonl(path)
        replacement_files.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
        )
        for row in rows:
            source_id = _text(row.get("source_trajectory_id"), "source_trajectory_id")
            if source_id in replacements:
                raise ValueError(f"duplicate replacement result: {source_id}")
            if source_id not in failure_ids:
                raise ValueError(f"replacement is outside failed scope: {source_id}")
            replacements[source_id] = dict(row)
    missing = sorted(set(failure_ids) - set(replacements))
    if missing:
        raise ValueError(f"replacement results are incomplete: {missing[:3]}")

    cached_input_sha = file_sha256(cached_path)
    rescue_manifest_sha = file_sha256(mismatch_path)
    merged: list[dict[str, Any]] = []
    double_failures: list[dict[str, Any]] = []
    replaced_success_ids: list[str] = []
    base_success_ids: list[str] = []
    for source in source_rows:
        dataset, sample_id, source_id = _source_identity(source)
        base = dict(base_by_id[source_id])
        _validate_result_lineage(
            base,
            source,
            original_source_sha256=source_file_by_id[source_id],
            require_original_source_file=True,
        )
        if source_id not in failure_ids:
            _validate_success_public_lineage(
                base, source, label=f"base result {source_id}"
            )
            _validate_prefix_success(base, label=f"base result {source_id}")
            merged.append(base)
            base_success_ids.append(source_id)
            continue

        replacement = replacements[source_id]
        _validate_result_lineage(
            replacement,
            source,
            original_source_sha256=source_file_by_id[source_id],
            require_original_source_file=False,
        )
        lane = "cached_repair" if source_id in cached_ids else "explicit_time_rescue"
        if lane == "cached_repair":
            if replacement.get("source_file_sha256") != cached_input_sha:
                raise ValueError(f"cached repair input hash mismatch: {source_id}")
        else:
            if replacement.get("rescue_manifest_sha256") != rescue_manifest_sha:
                raise ValueError(
                    f"explicit-time rescue manifest hash mismatch: {source_id}"
                )
            if replacement.get("frozen_scope_sha256") != scope_sha:
                raise ValueError(
                    f"explicit-time frozen scope hash mismatch: {source_id}"
                )
        if not _result_failed(replacement, label=f"replacement result {source_id}"):
            _validate_success_public_lineage(
                replacement, source, label=f"replacement result {source_id}"
            )
        replacement_failure_reason = _prefix_failure_reason(
            replacement, label=f"replacement result {source_id}"
        )
        if replacement_failure_reason is not None:
            double_failures.append(
                {
                    "schema_version": 1,
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "source_trajectory_id": source_id,
                    "source_row_sha256": canonical_sha256(source),
                    "repair_lane": lane,
                    "base_failure_reason": frozen_failure_reasons[source_id],
                    "replacement_failure_reason": replacement_failure_reason,
                    "base_failure": base,
                    "replacement_failure": replacement,
                }
            )
            continue
        merged.append(replacement)
        replaced_success_ids.append(source_id)

    expected_success_ids = set(source_by_id) - {
        row["source_trajectory_id"] for row in double_failures
    }
    actual_success_ids = {
        _text(row.get("source_trajectory_id"), "source_trajectory_id") for row in merged
    }
    if actual_success_ids != expected_success_ids or len(actual_success_ids) != len(
        merged
    ):
        raise AssertionError("merged success scope is inconsistent")
    if merged:
        try:
            jobs = bind_prefix_jobs(merged)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"merged success cannot bind to prefix jobs: {error}"
            ) from error
        if not jobs:
            raise ValueError("merged success produced no prefix jobs")

    merged_bytes = _jsonl_bytes(merged)
    double_bytes = _jsonl_bytes(double_failures)
    summary = {
        "schema_version": 1,
        "repair_scope_version": REPAIR_SCOPE_VERSION,
        "status": "passed" if not double_failures else "completed_with_failures",
        "source_rows": len(source_rows),
        "base_success": len(base_success_ids),
        "replacement_success": len(replaced_success_ids),
        "double_failures": len(double_failures),
        "merged_success": len(merged),
        "base_success_ids": base_success_ids,
        "replacement_success_ids": replaced_success_ids,
        "double_failure_ids": [row["source_trajectory_id"] for row in double_failures],
        "source_files": source_files,
        "base_files": base_files,
        "frozen_scope_sha256": scope_sha,
        "replacement_files": replacement_files,
        "artifacts": {
            _MERGED_SUCCESS: {
                "rows": len(merged),
                "sha256": hashlib.sha256(merged_bytes).hexdigest(),
            },
            _DOUBLE_FAILURES: {
                "rows": len(double_failures),
                "sha256": hashlib.sha256(double_bytes).hexdigest(),
            },
        },
        "prefix_bind_passed": True,
    }
    _write_directory_atomic(
        output_dir,
        {
            _MERGED_SUCCESS: merged_bytes,
            _DOUBLE_FAILURES: double_bytes,
            _MERGE_SUMMARY: _json_bytes(summary),
        },
    )
    return {**summary, "output_dir": str(output_dir.resolve())}


__all__ = [
    "REPAIR_SCOPE_VERSION",
    "merge_replay_results",
    "prepare_repair_scope",
]

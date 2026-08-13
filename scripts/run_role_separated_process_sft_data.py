#!/usr/bin/env python3
"""Prepare and run the Train600 role-separated Process-SFT data stages.

This pipeline is deliberately training-free.  It freezes a Train600-only source
lock, replays immutable cached frame shards, partitions failures, runs the
candidate-blind Visual-CSV checks, exports balanced Planner episodes plus an
image-bearing smoke probe, and applies the exact ms-swift length gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from flashvid_eval.perception_memory_eva import (
    ROLE_SEPARATED_CONTROLLER_OUTPUT_CONSTRAINT_VERSION,
    ROLE_SEPARATED_RUNTIME_VERSION,
    role_prompt_schema_bundle_sha256,
)
from flashvid_eval.perception_memory_replay import file_sha256
from flashvid_eval.perception_memory_visual_csv import (
    VISUAL_CSV_SEEDS,
    bind_visual_csv_jobs,
)
from flashvid_eval.qwen_sft import read_jsonl
from flashvid_eval.role_separated_orchestration import (
    TEST_MANIFEST_SHA256,
    load_config,
    require_valid_config,
)


PIPELINE_VERSION = "role_separated_process_sft_data_v1"
DATASETS = ("cgbench", "lsdbench", "lvbench")
STOP_STAGES = (
    "prepare",
    "replay",
    "partition",
    "visual_csv",
    "selection",
    "build",
    "length",
)
IMPLEMENTATION_FILES = (
    "scripts/run_role_separated_process_sft_data.py",
    "scripts/replay_perception_memory_trajectories.py",
    "scripts/judge_perception_memory_visual_csv.py",
    "scripts/select_perception_memory_visual_csv_trajectories.py",
    "scripts/build_perception_memory_sft.py",
    "scripts/filter_swift_sft_length.py",
    "src/flashvid_eval/perception_memory_replay.py",
    "src/flashvid_eval/perception_memory_eva.py",
    "src/flashvid_eval/perception_memory_visual_csv.py",
    "src/flashvid_eval/perception_memory_selection.py",
    "src/flashvid_eval/perception_memory_sft.py",
    "src/flashvid_eval/qwen_sft.py",
    "src/flashvid_eval/privacy.py",
    "src/flashvid_eval/qwen_agents/core.py",
    "src/flashvid_eval/runner.py",
    "src/flashvid_eval/client.py",
    "src/flashvid_eval/role_separated_orchestration.py",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_TEST_KEY_RE = re.compile(r"(?:^|_)(?:test(?:300)?|final_test|test_manifest)(?:_|$)")
_TEST_MANIFEST_BASENAME_RE = re.compile(r"manifest_42_100\.jsonl$", re.IGNORECASE)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _file_record(path: Path, *, jsonl: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if jsonl:
        record["rows"] = len(read_jsonl(path))
    return record


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ).encode("utf-8"),
    )


def _sample_key(row: Mapping[str, Any], label: str) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "").strip().lower()
    sample_id = str(row.get("sample_id") or "").strip()
    if dataset not in DATASETS or not sample_id:
        raise ValueError(f"{label} has an invalid dataset/sample_id")
    return dataset, sample_id


def _trajectory_id(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("trajectory_id") or "").strip()
    if not value:
        raise ValueError(f"{label} has no trajectory_id")
    return value


def _reject_test_identity(value: Any, path: str = "$") -> None:
    forbidden_shas = set(TEST_MANIFEST_SHA256.values())
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().casefold().replace("-", "_").replace(" ", "_")
            if _TEST_KEY_RE.search(key):
                raise ValueError(f"training source lock contains a Test field at {path}.{raw_key}")
            if key in {"split", "phase"} and str(item).strip().casefold() == "test":
                raise ValueError(f"training source lock contains a Test phase at {path}.{raw_key}")
            _reject_test_identity(item, f"{path}.{raw_key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_test_identity(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    normalized = value.replace("\\", "/")
    if value.lower() in forbidden_shas:
        raise ValueError(f"training source lock contains a frozen Test SHA at {path}")
    if any(component.casefold() == "final_test" for component in normalized.split("/")):
        raise ValueError(f"training source lock contains a final_test path at {path}")
    if _TEST_MANIFEST_BASENAME_RE.search(Path(normalized).name):
        raise ValueError(f"training source lock contains a Test manifest at {path}")


def _implementation_records(
    repo_root: Path, implementation_files: Mapping[str, Path]
) -> tuple[dict[str, str], str]:
    root = repo_root.resolve()
    records: dict[str, str] = {}
    for relative, raw_path in sorted(implementation_files.items()):
        path = raw_path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"implementation file escapes repository: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        records[str(relative).replace("\\", "/")] = file_sha256(path)
    if not records:
        raise ValueError("implementation file allowlist cannot be empty")
    return records, _canonical_sha256(records)


def _validate_experiment_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _validate_sha256(expected_sha256, "experiment config SHA-256")
    if file_sha256(path) != expected:
        raise ValueError("experiment config SHA-256 changed")
    config = load_config(path)
    require_valid_config(config)
    if config.get("experiment_id") != "role-separated-perception-memory-process-sft-v1":
        raise ValueError("unexpected role-separated experiment_id")
    roles = config.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("experiment roles are missing")
    trainable = sorted(
        name for name, role in roles.items() if isinstance(role, Mapping) and role.get("trainable") is True
    )
    if trainable != ["planner"]:
        raise ValueError("only Planner may be trainable")
    for name in ("observer", "verifier", "answerer"):
        role = roles.get(name)
        if not isinstance(role, Mapping) or role.get("adapter") != "frozen_base" or role.get("trainable") is not False:
            raise ValueError(f"{name} must remain frozen base")
    return config


def _train600_record(path: Path, expected_sha256: str) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    expected = _validate_sha256(expected_sha256, "Train600 SHA-256")
    if file_sha256(path) != expected:
        raise ValueError("Train600 SHA-256 changed")
    rows = read_jsonl(path)
    counts: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        key = _sample_key(row, f"Train600 row {index}")
        if key in keys:
            raise ValueError(f"duplicate Train600 sample key: {key[0]}/{key[1]}")
        keys.add(key)
        counts[key[0]] += 1
    if len(rows) != 600 or dict(counts) != {dataset: 200 for dataset in DATASETS}:
        raise ValueError(f"Train600 must contain 600 rows/200 per dataset, found {dict(counts)}")
    return (
        {
            "path": str(path.resolve()),
            "sha256": expected,
            "rows": len(rows),
            "dataset_rows": {dataset: counts[dataset] for dataset in DATASETS},
            "sample_keys_sha256": _canonical_sha256(sorted(keys)),
        },
        keys,
    )


def _source_shard_record(
    paths: Sequence[Path],
    *,
    train600_sha256: str,
    train600_keys: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    resolved = [path.resolve() for path in paths]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError("source shard paths must be non-empty and unique")
    items: list[dict[str, Any]] = []
    source_by_id: dict[str, dict[str, Any]] = {}
    all_keys: list[tuple[str, str]] = []
    for path in resolved:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"source shard is empty: {path}")
        ids: list[str] = []
        keys: list[tuple[str, str]] = []
        for index, raw in enumerate(rows):
            row = dict(raw)
            label = f"{path}:{index + 1}"
            identity = _trajectory_id(row, label)
            if identity in source_by_id:
                raise ValueError(f"duplicate source trajectory_id: {identity}")
            key = _sample_key(row, label)
            if key not in train600_keys:
                raise ValueError(f"source trajectory is outside Train600: {key[0]}/{key[1]}")
            if row.get("train600_manifest_sha256") != train600_sha256:
                raise ValueError(f"source trajectory Train600 SHA drifted: {identity}")
            source_by_id[identity] = row
            ids.append(identity)
            keys.append(key)
            all_keys.append(key)
        items.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "trajectory_ids_sha256": _canonical_sha256(sorted(ids)),
                "sample_keys_sha256": _canonical_sha256(sorted(keys)),
            }
        )
    all_ids = sorted(source_by_id)
    unique_keys = sorted(set(all_keys))
    return (
        {
            "count": len(items),
            "rows": len(source_by_id),
            "unique_trajectory_ids": len(source_by_id),
            "unique_sample_keys": len(unique_keys),
            "trajectory_ids_sha256": _canonical_sha256(all_ids),
            "sample_keys_with_repetitions_sha256": _canonical_sha256(sorted(all_keys)),
            "unique_sample_keys_sha256": _canonical_sha256(unique_keys),
            "items": items,
        },
        source_by_id,
    )


def _attribution_record(
    summary_path: Path,
    expected_summary_sha256: str,
    frozen_input_path: Path,
) -> dict[str, Any]:
    expected = _validate_sha256(expected_summary_sha256, "role attribution SHA-256")
    if file_sha256(summary_path) != expected:
        raise ValueError("role attribution summary SHA-256 changed")
    summary = _read_json(summary_path)
    decision = summary.get("decision")
    if not isinstance(decision, Mapping) or {
        "train_new_planner_lora": decision.get("train_new_planner_lora"),
        "train_observer_lora": decision.get("train_observer_lora"),
        "keep_verifier_base": decision.get("keep_verifier_base"),
        "keep_answerer_base": decision.get("keep_answerer_base"),
    } != {
        "train_new_planner_lora": True,
        "train_observer_lora": False,
        "keep_verifier_base": True,
        "keep_answerer_base": True,
    }:
        raise ValueError("role attribution does not select Planner-only SFT")
    runtime_commit = str(summary.get("runtime_commit") or "").strip().lower()
    if not _GIT_HEAD_RE.fullmatch(runtime_commit):
        raise ValueError("role attribution runtime_commit is invalid")
    frozen = summary.get("frozen_input")
    if not isinstance(frozen, Mapping):
        raise ValueError("role attribution has no frozen input")
    frozen_path = frozen_input_path.resolve()
    frozen_sha = file_sha256(frozen_path)
    frozen_rows = read_jsonl(frozen_path)
    if (
        frozen.get("sha256") != frozen_sha
        or int(frozen.get("samples", -1)) != len(frozen_rows)
        or len(frozen_rows) != 30
    ):
        raise ValueError("role attribution frozen input changed")
    return {
        "path": str(summary_path.resolve()),
        "sha256": expected,
        "runtime_commit": runtime_commit,
        "frozen_input": {
            "path": str(frozen_path),
            "sha256": frozen_sha,
            "rows": len(frozen_rows),
        },
        "trainable_roles": ["planner"],
        "frozen_base_roles": ["observer", "verifier", "answerer"],
    }


def build_training_source_lock(
    *,
    repo_root: Path,
    experiment_config: Path,
    expected_experiment_config_sha256: str,
    train600: Path,
    expected_train600_sha256: str,
    source_shards: Sequence[Path],
    role_attribution_summary: Path,
    expected_role_attribution_sha256: str,
    frozen_role_input: Path,
    observer_model: str,
    observer_artifact_sha256: str,
    verifier_model: str,
    verifier_artifact_sha256: str,
    base_model_artifact_sha256: str,
    git_head: str,
    git_branch: str,
    implementation_files: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = repo_root.resolve()
    config = _validate_experiment_config(
        experiment_config, expected_experiment_config_sha256
    )
    train_record, train_keys = _train600_record(train600, expected_train600_sha256)
    shards_record, source_by_id = _source_shard_record(
        source_shards,
        train600_sha256=train_record["sha256"],
        train600_keys=train_keys,
    )
    attribution = _attribution_record(
        role_attribution_summary,
        expected_role_attribution_sha256,
        frozen_role_input,
    )
    artifacts = config.get("role_ablation", {}).get("artifacts", {})
    base = artifacts.get("base") if isinstance(artifacts, Mapping) else None
    observer_sha = _validate_sha256(observer_artifact_sha256, "Observer artifact")
    verifier_sha = _validate_sha256(verifier_artifact_sha256, "Verifier artifact")
    base_model_sha = _validate_sha256(
        base_model_artifact_sha256, "base model artifact"
    )
    if (
        not isinstance(base, Mapping)
        or base.get("model") != observer_model
        or base.get("artifact_sha256") != observer_sha
    ):
        raise ValueError("Observer must use the frozen Base attribution artifact")
    if base.get("model") != verifier_model or base.get("artifact_sha256") != verifier_sha:
        raise ValueError("Verifier must use the frozen Base attribution artifact")
    if base.get("artifact_sha256") != base_model_sha:
        raise ValueError("training base model must use the frozen Base attribution artifact")
    head = str(git_head).strip().lower()
    if not _GIT_HEAD_RE.fullmatch(head) or not str(git_branch).strip():
        raise ValueError("implementation Git identity is invalid")
    implementation, implementation_bundle = _implementation_records(
        root, implementation_files
    )
    prompt_schema_bundle_sha256 = role_prompt_schema_bundle_sha256()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "role_separated_runtime_version": ROLE_SEPARATED_RUNTIME_VERSION,
        "role_prompt_schema_bundle_sha256": prompt_schema_bundle_sha256,
        "controller_output_constraint_version": (
            ROLE_SEPARATED_CONTROLLER_OUTPUT_CONSTRAINT_VERSION
        ),
        "experiment_config": {
            "path": str(experiment_config.resolve()),
            "sha256": _validate_sha256(
                expected_experiment_config_sha256, "experiment config SHA-256"
            ),
        },
        "train600": train_record,
        "source_shards": shards_record,
        "role_attribution": attribution,
        "observer": {
            "model": observer_model,
            "adapter": "frozen_base",
            "trainable": False,
            "artifact_sha256": observer_sha,
            "prompt_schema_bundle_sha256": prompt_schema_bundle_sha256,
        },
        "verifier": {
            "model": verifier_model,
            "adapter": "frozen_base",
            "trainable": False,
            "artifact_sha256": verifier_sha,
            "prompt_schema_bundle_sha256": prompt_schema_bundle_sha256,
        },
        "training_base_model": {
            "model": observer_model,
            "artifact_sha256": base_model_sha,
        },
        "implementation": {
            "git_head": head,
            "git_branch": str(git_branch).strip(),
            "files": implementation,
            "bundle_sha256": implementation_bundle,
        },
    }
    _reject_test_identity(payload)
    return payload, source_by_id


def validate_training_source_lock(
    payload: Mapping[str, Any], **build_kwargs: Any
) -> dict[str, dict[str, Any]]:
    expected, source_by_id = build_training_source_lock(**build_kwargs)
    if dict(payload) != expected:
        raise ValueError("training source lock differs from current frozen inputs")
    _reject_test_identity(payload)
    return source_by_id


def _git_identity(
    repo_root: Path,
    *,
    expected_head: str,
    expected_branch: str,
    implementation_files: Sequence[str],
) -> tuple[str, str]:
    def output(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    head = output("rev-parse", "HEAD")
    branch = output("branch", "--show-current")
    if head != expected_head or branch != expected_branch:
        raise RuntimeError("Git HEAD/branch differs from the registered data run")
    dirty = output("status", "--porcelain", "--", *implementation_files)
    if dirty:
        raise RuntimeError("a locked implementation file differs from Git HEAD")
    return head, branch


def _validate_endpoints(base_urls: Sequence[str], gpu_ids: Sequence[int]) -> None:
    if not 1 <= len(base_urls) <= 4 or len(set(base_urls)) != len(base_urls):
        raise ValueError("data pipeline requires 1-4 unique model endpoints")
    if len(gpu_ids) != len(base_urls) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-id must uniquely bind each of the 1-4 endpoints")
    for value in base_urls:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid model endpoint: {value}")
    if any(value < 0 for value in gpu_ids):
        raise ValueError("GPU IDs must be non-negative")


def _scope_inputs(
    paths: Sequence[Path], source_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[Path], dict[str, Any]]:
    resolved = [path.resolve() for path in paths]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError("replay scope paths must be non-empty and unique")
    selected: set[str] = set()
    records: list[dict[str, Any]] = []
    for path in resolved:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"replay scope is empty: {path}")
        for index, row in enumerate(rows):
            identity = _trajectory_id(row, f"{path}:{index + 1}")
            source = source_by_id.get(identity)
            if source is None or _canonical_sha256(row) != _canonical_sha256(source):
                raise ValueError(f"replay scope row is not an exact locked source: {identity}")
            if identity in selected:
                raise ValueError(f"duplicate replay scope trajectory: {identity}")
            selected.add(identity)
        records.append(_file_record(path, jsonl=True))
    return resolved, {
        "mode": "full" if len(selected) == len(source_by_id) else "explicit_scope",
        "rows": len(selected),
        "locked_source_rows": len(source_by_id),
        "coverage_ratio": len(selected) / len(source_by_id),
        "trajectory_ids_sha256": _canonical_sha256(sorted(selected)),
        "files": records,
    }


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(list(command), ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"EXIT {result.returncode}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return int(result.returncode)


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    values = list(command)
    return {"argv": values, "sha256": _canonical_sha256(values)}


def _stage_passed(state: Mapping[str, Any], stage: str) -> bool:
    value = (state.get("stages") or {}).get(stage)
    return isinstance(value, Mapping) and value.get("status") == "passed"


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    _atomic_write_json(path, state)


def _stop_after(
    state: dict[str, Any], status_path: Path, stage: str, requested: str
) -> bool:
    if stage != requested:
        return False
    state["status"] = "stopped"
    state["current_stage"] = stage
    state["stopped_after"] = stage
    _save_state(status_path, state)
    return True


def _replay_command(
    args: argparse.Namespace,
    *,
    source: Path,
    output: Path,
    source_lock: Path,
    source_lock_sha256: str,
    experiment_config_sha256: str,
) -> list[str]:
    command = [
        args.python,
        str(args.repo_root / "scripts/replay_perception_memory_trajectories.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--training-source-lock",
        str(source_lock),
        "--training-source-lock-sha256",
        source_lock_sha256,
        "--experiment-config-sha256",
        experiment_config_sha256,
        "--model",
        args.model,
        "--served-model-artifact-sha256",
        args.observer_artifact_sha256,
        "--role-separated-observer",
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--max-frames-per-call",
        str(args.max_frames_per_call),
        "--timeout",
        str(args.request_timeout),
        "--concurrency",
        str(args.replay_concurrency),
        "--local-media-paths",
    ]
    for endpoint in args.base_url:
        command.extend(["--base-url", endpoint])
    if output.exists():
        command.append("--resume")
    return command


def _audit_replay_output(
    source: Path,
    output: Path,
    *,
    source_lock_sha256: str,
    experiment_config_sha256: str,
    observer_artifact_sha256: str,
    role_separated_runtime_version: str,
    role_prompt_schema_bundle_sha256: str,
    controller_output_constraint_version: str,
) -> dict[str, Any]:
    sources = read_jsonl(source)
    results = read_jsonl(output)
    source_by_id = {
        _trajectory_id(row, "replay source"): row for row in sources
    }
    result_by_id: dict[str, dict[str, Any]] = {}
    source_sha = file_sha256(source)
    for index, raw in enumerate(results):
        row = dict(raw)
        identity = str(row.get("source_trajectory_id") or "").strip()
        if identity not in source_by_id or identity in result_by_id:
            raise RuntimeError(f"replay output has invalid/duplicate source identity: {identity}")
        if row.get("source_row_sha256") != _canonical_sha256(source_by_id[identity]):
            raise RuntimeError(f"replay source-row binding changed: {identity}")
        if row.get("source_file_sha256") != source_sha:
            raise RuntimeError(f"replay source-file binding changed: {identity}")
        if row.get("training_source_lock_sha256") != source_lock_sha256:
            raise RuntimeError(f"replay training-source-lock binding changed: {identity}")
        if row.get("experiment_config_sha256") != experiment_config_sha256:
            raise RuntimeError(f"replay experiment binding changed: {identity}")
        for field, expected in (
            ("role_separated_runtime_version", role_separated_runtime_version),
            ("role_prompt_schema_bundle_sha256", role_prompt_schema_bundle_sha256),
            (
                "controller_output_constraint_version",
                controller_output_constraint_version,
            ),
        ):
            if row.get(field) != expected:
                raise RuntimeError(f"replay {field} binding changed: {identity}")
        if any(key in row for key in ("diagnostics_gate_sha256", "audit_summary_sha256")):
            raise RuntimeError("Train600 replay may not contain a Test diagnostics gate")
        if row.get("role_separated_observer") is not True:
            raise RuntimeError(f"replay did not use role-separated Observer: {identity}")
        has_error = bool(row.get("error") or row.get("error_type"))
        if row.get("status") not in (None, "failure" if has_error else "success"):
            raise RuntimeError(f"replay status/error fields disagree: {identity}")
        if not has_error:
            if row.get("served_model_artifact_sha256") != observer_artifact_sha256:
                raise RuntimeError(f"replay Observer artifact changed: {identity}")
            if row.get("candidate_rerun") != 0 or row.get("annotation_leak_check") != "passed":
                raise RuntimeError(f"replay integrity gate failed: {identity}")
        result_by_id[identity] = row
    if set(result_by_id) != set(source_by_id):
        raise RuntimeError("replay output does not exactly cover its source input")
    failures = sum(bool(row.get("error") or row.get("error_type")) for row in results)
    return {
        "input": _file_record(source, jsonl=True),
        "output": _file_record(output, jsonl=True),
        "success": len(results) - failures,
        "failure": failures,
    }


def _partition_replays(
    replay_outputs: Sequence[Path],
    *,
    success_path: Path,
    failure_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in replay_outputs:
        for row in read_jsonl(path):
            identity = str(row.get("source_trajectory_id") or "").strip()
            if not identity or identity in seen:
                raise RuntimeError("replay partition contains an invalid duplicate identity")
            seen.add(identity)
            has_error = bool(row.get("error") or row.get("error_type"))
            (failures if has_error else successes).append(dict(row))
    successes.sort(key=lambda row: str(row["source_trajectory_id"]))
    failures.sort(key=lambda row: str(row["source_trajectory_id"]))
    if successes:
        jobs = bind_visual_csv_jobs(successes)
        if not jobs:
            raise RuntimeError("successful replay rows produced no Visual-CSV jobs")
    _atomic_write_jsonl(success_path, successes)
    _atomic_write_jsonl(failure_path, failures)
    summary = {
        "status": "passed",
        "source_rows": len(seen),
        "success": len(successes),
        "failure": len(failures),
        "failure_is_quantity_advisory": True,
        "artifacts": {
            "success": _file_record(success_path, jsonl=True),
            "failure": _file_record(failure_path, jsonl=True),
        },
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _audit_visual_csv(
    trajectories: Path,
    output: Path,
    *,
    verifier_artifact_sha256: str,
) -> dict[str, Any]:
    jobs = bind_visual_csv_jobs(read_jsonl(trajectories))
    expected = {job.prefix_id for job in jobs}
    rows = read_jsonl(output)
    actual = [str(row.get("prefix_id") or "") for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Visual-CSV output does not exactly cover replay prefixes")
    failures = 0
    for row in rows:
        if row.get("verifier_artifact_sha256") != verifier_artifact_sha256:
            raise RuntimeError("Visual-CSV verifier artifact changed")
        if row.get("candidate_blind") is not True or row.get("tools_disabled") is not True:
            raise RuntimeError("Visual-CSV violated candidate/tool isolation")
        if row.get("annotation_leak_check") != "passed":
            raise RuntimeError("Visual-CSV annotation leak check failed")
        confirmations = row.get("visual_csv_confirmations")
        if not isinstance(confirmations, list) or len(confirmations) != 3:
            raise RuntimeError("Visual-CSV row lacks exactly three confirmations")
        if {item.get("judge_seed") for item in confirmations if isinstance(item, Mapping)} != set(VISUAL_CSV_SEEDS):
            raise RuntimeError("Visual-CSV seeds changed")
        failures += sum(bool(item.get("error")) for item in confirmations if isinstance(item, Mapping))
    return {
        "prefixes": len(rows),
        "confirmation_failures": failures,
        "output": _file_record(output, jsonl=True),
    }


def _selection_command(
    args: argparse.Namespace, trajectories: Path, judgments: Path
) -> list[str]:
    root = args.run_root / "selection"
    command = [
        args.python,
        str(args.repo_root / "scripts/select_perception_memory_visual_csv_trajectories.py"),
        "--trajectories",
        str(trajectories),
        "--visual-csv-results",
        str(judgments),
        "--verifier-artifact-sha256",
        args.verifier_artifact_sha256,
        "--answers",
        str(args.train600),
        "--expected-answers-sha256",
        args.expected_train600_sha256,
        "--labeled-output",
        str(root / "labeled.jsonl"),
        "--selected-output",
        str(root / "selected.jsonl"),
        "--summary",
        str(root / "summary.json"),
    ]
    if any((root / name).exists() for name in ("labeled.jsonl", "selected.jsonl", "summary.json")):
        command.append("--overwrite")
    return command


def _build_command(
    args: argparse.Namespace,
    *,
    selected: Path,
    output: Path,
    summary: Path,
    source_lock_sha256: str,
    role_separated_runtime_version: str,
    role_prompt_schema_bundle_sha256: str,
    controller_output_constraint_version: str,
    include_observer: bool,
) -> list[str]:
    command = [
        args.python,
        str(args.repo_root / "scripts/build_perception_memory_sft.py"),
        "--selected",
        str(selected),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--completion-gate-kind",
        "visual_csv",
        "--include-experiment-config-sha256",
        args.expected_experiment_config_sha256,
        "--expected-training-source-lock-sha256",
        source_lock_sha256,
        "--expected-role-separated-runtime-version",
        role_separated_runtime_version,
        "--expected-role-prompt-schema-bundle-sha256",
        role_prompt_schema_bundle_sha256,
        "--expected-controller-output-constraint-version",
        controller_output_constraint_version,
    ]
    if include_observer:
        command.append("--include-observer")
    if output.exists() or summary.exists():
        command.append("--overwrite")
    return command


def _length_command(
    args: argparse.Namespace, *, source: Path, output: Path, audit: Path
) -> list[str]:
    return [
        args.sft_python,
        str(args.repo_root / "scripts/filter_swift_sft_length.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--audit",
        str(audit),
        "--model",
        str(args.model_path),
        "--model-artifact-sha256",
        args.base_model_artifact_sha256,
        "--max-length",
        "16384",
    ]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.repo_root = args.repo_root.resolve()
    args.run_root = args.run_root.resolve()
    args.train600 = args.train600.resolve()
    args.observer_artifact_sha256 = _validate_sha256(
        args.observer_artifact_sha256, "Observer artifact"
    )
    args.verifier_artifact_sha256 = _validate_sha256(
        args.verifier_artifact_sha256, "Verifier artifact"
    )
    args.base_model_artifact_sha256 = _validate_sha256(
        args.base_model_artifact_sha256, "base model artifact"
    )
    args.expected_train600_sha256 = _validate_sha256(
        args.expected_train600_sha256, "Train600 SHA-256"
    )
    args.expected_experiment_config_sha256 = _validate_sha256(
        args.expected_experiment_config_sha256, "experiment config SHA-256"
    )
    args.expected_role_attribution_sha256 = _validate_sha256(
        args.expected_role_attribution_sha256, "role attribution SHA-256"
    )
    if args.prepare_only:
        if args.stop_after != "length":
            raise ValueError("--prepare-only cannot be combined with --stop-after")
        args.stop_after = "prepare"
    if args.run_root.exists() and not args.resume:
        raise FileExistsError("run root exists; pass --resume")
    args.run_root.mkdir(parents=True, exist_ok=True)
    implementation_paths = {
        relative: args.repo_root / relative for relative in IMPLEMENTATION_FILES
    }
    head, branch = _git_identity(
        args.repo_root,
        expected_head=args.expected_git_head,
        expected_branch=args.expected_git_branch,
        implementation_files=IMPLEMENTATION_FILES,
    )
    build_kwargs = {
        "repo_root": args.repo_root,
        "experiment_config": args.experiment_config.resolve(),
        "expected_experiment_config_sha256": args.expected_experiment_config_sha256,
        "train600": args.train600,
        "expected_train600_sha256": args.expected_train600_sha256,
        "source_shards": [path.resolve() for path in args.source_shard],
        "role_attribution_summary": args.role_attribution_summary.resolve(),
        "expected_role_attribution_sha256": args.expected_role_attribution_sha256,
        "frozen_role_input": args.frozen_role_input.resolve(),
        "observer_model": args.model,
        "observer_artifact_sha256": args.observer_artifact_sha256,
        "verifier_model": args.model,
        "verifier_artifact_sha256": args.verifier_artifact_sha256,
        "base_model_artifact_sha256": args.base_model_artifact_sha256,
        "git_head": head,
        "git_branch": branch,
        "implementation_files": implementation_paths,
    }
    source_lock_payload, source_by_id = build_training_source_lock(**build_kwargs)
    source_lock_path = args.run_root / "source_lock/training_source_lock.json"
    source_lock_bytes = (
        json.dumps(source_lock_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if source_lock_path.exists():
        if source_lock_path.read_bytes() != source_lock_bytes:
            raise RuntimeError("existing training source lock drifted")
    else:
        _atomic_write_bytes(source_lock_path, source_lock_bytes)
    validate_training_source_lock(
        _read_json(source_lock_path), **build_kwargs
    )
    source_lock_sha = file_sha256(source_lock_path)
    role_runtime_version = str(
        source_lock_payload["role_separated_runtime_version"]
    )
    role_prompt_bundle = str(
        source_lock_payload["role_prompt_schema_bundle_sha256"]
    )
    controller_constraint_version = str(
        source_lock_payload["controller_output_constraint_version"]
    )
    replay_inputs, scope = _scope_inputs(
        args.scope_source or args.source_shard,
        source_by_id,
    )
    if args.stop_after != "prepare":
        _validate_endpoints(args.base_url, args.gpu_id)
    fingerprint = _canonical_sha256(
        {
            "pipeline_version": PIPELINE_VERSION,
            "source_lock_sha256": source_lock_sha,
            "role_separated_runtime_version": role_runtime_version,
            "role_prompt_schema_bundle_sha256": role_prompt_bundle,
            "controller_output_constraint_version": controller_constraint_version,
            "scope": scope,
            "base_urls": list(args.base_url),
            "gpu_ids": list(args.gpu_id),
            "model": args.model,
            "observer_artifact_sha256": args.observer_artifact_sha256,
            "verifier_artifact_sha256": args.verifier_artifact_sha256,
            "base_model_artifact_sha256": args.base_model_artifact_sha256,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "max_frames_per_call": args.max_frames_per_call,
            "request_timeout": args.request_timeout,
            "replay_concurrency": args.replay_concurrency,
            "judge_concurrency": args.judge_concurrency,
            "stop_after": args.stop_after,
        }
    )
    status_path = args.run_root / "status.json"
    if status_path.exists():
        state = _read_json(status_path)
        if state.get("config_sha256") != fingerprint:
            raise RuntimeError("pipeline resume fingerprint changed")
        state["status"] = "running"
        state["error"] = None
    else:
        state = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "config_sha256": fingerprint,
            "status": "running",
            "current_stage": "prepare",
            "stages": {},
            "training_quantity_policy": "advisory_only",
            "quality_contract_policy": "blocking",
            "error": None,
        }
    try:
        if not _stage_passed(state, "prepare"):
            state["stages"]["prepare"] = {
                "status": "passed",
                "source_lock": _file_record(source_lock_path),
                "scope": scope,
                "endpoint_count": len(args.base_url),
                "gpu_ids": list(args.gpu_id),
            }
            _save_state(status_path, state)
        elif state["stages"]["prepare"]["source_lock"] != _file_record(source_lock_path):
            raise RuntimeError("passed training source lock changed")
        if _stop_after(state, status_path, "prepare", args.stop_after):
            return state

        replay_outputs: list[Path] = []
        replay_audits: list[dict[str, Any]] = []
        state["current_stage"] = "replay"
        for index, source in enumerate(replay_inputs):
            output = args.run_root / f"trajectories/replay_{index:03d}_{source.stem}.jsonl"
            replay_outputs.append(output)
            command = _replay_command(
                args,
                source=source,
                output=output,
                source_lock=source_lock_path,
                source_lock_sha256=source_lock_sha,
                experiment_config_sha256=args.expected_experiment_config_sha256,
            )
            if not _stage_passed(state, "replay"):
                code = _run_command(
                    command,
                    cwd=args.repo_root,
                    log_path=args.run_root / f"logs/replay_{index:03d}.log",
                )
                if code not in {0, 1}:
                    raise RuntimeError(f"replay command failed with exit code {code}")
            replay_audits.append(
                _audit_replay_output(
                    source,
                    output,
                    source_lock_sha256=source_lock_sha,
                    experiment_config_sha256=args.expected_experiment_config_sha256,
                    observer_artifact_sha256=args.observer_artifact_sha256,
                    role_separated_runtime_version=role_runtime_version,
                    role_prompt_schema_bundle_sha256=role_prompt_bundle,
                    controller_output_constraint_version=(
                        controller_constraint_version
                    ),
                )
            )
        if _stage_passed(state, "replay"):
            if replay_audits != state["stages"]["replay"].get("shards"):
                raise RuntimeError("passed replay outputs changed")
        else:
            state["stages"]["replay"] = {
                "status": "passed",
                "commands": [
                    _command_record(
                        _replay_command(
                            args,
                            source=source,
                            output=output,
                            source_lock=source_lock_path,
                            source_lock_sha256=source_lock_sha,
                            experiment_config_sha256=args.expected_experiment_config_sha256,
                        )
                    )
                    for source, output in zip(replay_inputs, replay_outputs)
                ],
                "shards": replay_audits,
                "source_rows": sum(item["input"]["rows"] for item in replay_audits),
                "success": sum(item["success"] for item in replay_audits),
                "failure": sum(item["failure"] for item in replay_audits),
            }
            _save_state(status_path, state)
        if _stop_after(state, status_path, "replay", args.stop_after):
            return state

        state["current_stage"] = "partition"
        success_path = args.run_root / "trajectories/success.jsonl"
        failure_path = args.run_root / "trajectories/failures.jsonl"
        partition_summary_path = args.run_root / "trajectories/partition_summary.json"
        if not _stage_passed(state, "partition"):
            partition = _partition_replays(
                replay_outputs,
                success_path=success_path,
                failure_path=failure_path,
                summary_path=partition_summary_path,
            )
            state["stages"]["partition"] = {
                "status": "passed",
                "summary": partition,
                "summary_artifact": _file_record(partition_summary_path),
            }
            _save_state(status_path, state)
        else:
            frozen = state["stages"]["partition"]
            current_summary = _read_json(partition_summary_path)
            if (
                frozen.get("summary_artifact") != _file_record(partition_summary_path)
                or frozen.get("summary") != current_summary
                or current_summary.get("artifacts", {}).get("success")
                != _file_record(success_path, jsonl=True)
                or current_summary.get("artifacts", {}).get("failure")
                != _file_record(failure_path, jsonl=True)
            ):
                raise RuntimeError("passed replay partition changed")
        if _stop_after(state, status_path, "partition", args.stop_after):
            return state

        state["current_stage"] = "visual_csv"
        judgments = args.run_root / "csv_labels/visual_csv.jsonl"
        judge_command = [
            args.python,
            str(args.repo_root / "scripts/judge_perception_memory_visual_csv.py"),
            "--trajectories",
            str(success_path),
            "--output",
            str(judgments),
            "--model",
            args.model,
            "--verifier-artifact-sha256",
            args.verifier_artifact_sha256,
            "--concurrency",
            str(args.judge_concurrency),
            "--timeout",
            str(args.request_timeout),
            "--local-media-paths",
        ]
        for endpoint in args.base_url:
            judge_command.extend(["--base-url", endpoint])
        if judgments.exists() or judgments.with_suffix(judgments.suffix + ".progress.jsonl").exists():
            judge_command.append("--resume")
        if not _stage_passed(state, "visual_csv"):
            if _run_command(
                judge_command,
                cwd=args.repo_root,
                log_path=args.run_root / "logs/visual_csv.log",
            ):
                raise RuntimeError("Visual-CSV command failed")
            audit = _audit_visual_csv(
                success_path,
                judgments,
                verifier_artifact_sha256=args.verifier_artifact_sha256,
            )
            state["stages"]["visual_csv"] = {
                "status": "passed",
                "command": _command_record(judge_command),
                "input": _file_record(success_path, jsonl=True),
                "audit": audit,
            }
            _save_state(status_path, state)
        else:
            frozen_visual = state["stages"]["visual_csv"]
            if (
                frozen_visual.get("input") != _file_record(success_path, jsonl=True)
                or frozen_visual.get("audit")
                != _audit_visual_csv(
                    success_path,
                    judgments,
                    verifier_artifact_sha256=args.verifier_artifact_sha256,
                )
            ):
                raise RuntimeError("passed Visual-CSV input/output changed")
        if _stop_after(state, status_path, "visual_csv", args.stop_after):
            return state

        state["current_stage"] = "selection"
        selection_command = _selection_command(args, success_path, judgments)
        selected = args.run_root / "selection/selected.jsonl"
        selection_summary_path = args.run_root / "selection/summary.json"
        if not _stage_passed(state, "selection"):
            if _run_command(
                selection_command,
                cwd=args.repo_root,
                log_path=args.run_root / "logs/selection.log",
            ):
                raise RuntimeError("offline selection command failed")
            selection_summary = _read_json(selection_summary_path)
            balance = selection_summary.get("quality_balancing")
            if (
                not isinstance(balance, Mapping)
                or balance.get("status") != "applied"
                or int(selection_summary.get("selected") or 0) <= 0
            ):
                raise RuntimeError("stable trajectories cannot satisfy the frozen quality balance")
            state["stages"]["selection"] = {
                "status": "passed",
                "command": _command_record(selection_command),
                "labeled": _file_record(
                    args.run_root / "selection/labeled.jsonl", jsonl=True
                ),
                "selected": _file_record(selected, jsonl=True),
                "summary": _file_record(selection_summary_path),
                "training_quantity": {
                    "selected": selection_summary["selected"],
                    "candidate_fixes": selection_summary.get("candidate_fixes", 0),
                    "policy": "advisory_only",
                },
            }
            _save_state(status_path, state)
        else:
            frozen_selection = state["stages"]["selection"]
            if (
                frozen_selection.get("labeled")
                != _file_record(
                    args.run_root / "selection/labeled.jsonl", jsonl=True
                )
                or frozen_selection.get("selected")
                != _file_record(selected, jsonl=True)
                or frozen_selection.get("summary")
                != _file_record(selection_summary_path)
            ):
                raise RuntimeError("passed selection artifacts changed")
        if _stop_after(state, status_path, "selection", args.stop_after):
            return state

        state["current_stage"] = "build"
        planner_raw = args.run_root / "sft_data/planner/raw.jsonl"
        planner_summary = args.run_root / "sft_data/planner/summary.json"
        probe_raw = args.run_root / "sft_data/observer/smoke_probe_raw.jsonl"
        probe_summary = args.run_root / "sft_data/observer/smoke_probe_summary.json"
        build_commands = [
            _build_command(
                args,
                selected=selected,
                output=planner_raw,
                summary=planner_summary,
                source_lock_sha256=source_lock_sha,
                role_separated_runtime_version=role_runtime_version,
                role_prompt_schema_bundle_sha256=role_prompt_bundle,
                controller_output_constraint_version=controller_constraint_version,
                include_observer=False,
            ),
            _build_command(
                args,
                selected=selected,
                output=probe_raw,
                summary=probe_summary,
                source_lock_sha256=source_lock_sha,
                role_separated_runtime_version=role_runtime_version,
                role_prompt_schema_bundle_sha256=role_prompt_bundle,
                controller_output_constraint_version=controller_constraint_version,
                include_observer=True,
            ),
        ]
        if not _stage_passed(state, "build"):
            for index, command in enumerate(build_commands):
                if _run_command(
                    command,
                    cwd=args.repo_root,
                    log_path=args.run_root / f"logs/build_{index}.log",
                ):
                    raise RuntimeError("Process-SFT build command failed")
            planner_rows = read_jsonl(planner_raw)
            if any(row.get("images") for row in planner_rows):
                raise RuntimeError("Planner training data must be text-only")
            probe_rows = read_jsonl(probe_raw)
            if not any(row.get("images") for row in probe_rows):
                raise RuntimeError("smoke probe build has no real image-bearing episode")
            state["stages"]["build"] = {
                "status": "passed",
                "commands": [_command_record(command) for command in build_commands],
                "planner": _file_record(planner_raw, jsonl=True),
                "planner_summary": _file_record(planner_summary),
                "image_probe": _file_record(probe_raw, jsonl=True),
                "image_probe_summary": _file_record(probe_summary),
            }
            _save_state(status_path, state)
        elif (
            state["stages"]["build"].get("planner") != _file_record(planner_raw, jsonl=True)
            or state["stages"]["build"].get("planner_summary")
            != _file_record(planner_summary)
            or state["stages"]["build"].get("image_probe") != _file_record(probe_raw, jsonl=True)
            or state["stages"]["build"].get("image_probe_summary")
            != _file_record(probe_summary)
        ):
            raise RuntimeError("passed Process-SFT build artifacts changed")
        if _stop_after(state, status_path, "build", args.stop_after):
            return state

        state["current_stage"] = "length"
        if not args.sft_python or args.model_path is None:
            raise ValueError("length stage requires --sft-python and --model-path")
        planner_train = args.run_root / "sft_data/planner/train.jsonl"
        planner_audit = args.run_root / "sft_data/planner/length_audit.json"
        probe_train = args.run_root / "sft_data/observer/smoke_probe.jsonl"
        probe_audit = args.run_root / "sft_data/observer/smoke_probe_length_audit.json"
        length_commands = [
            _length_command(args, source=planner_raw, output=planner_train, audit=planner_audit),
            _length_command(args, source=probe_raw, output=probe_train, audit=probe_audit),
        ]
        if not _stage_passed(state, "length"):
            for index, command in enumerate(length_commands):
                if _run_command(
                    command,
                    cwd=args.repo_root,
                    log_path=args.run_root / f"logs/length_{index}.log",
                ):
                    raise RuntimeError("ms-swift length filter failed")
            planner_length = _read_json(planner_audit)
            probe_length = _read_json(probe_audit)
            if int(planner_length.get("rejected_trajectories", -1)) != 0:
                raise RuntimeError("Planner length filtering would break the frozen quality balance")
            if planner_length.get("input_rows") != planner_length.get("retained_rows"):
                raise RuntimeError("Planner length gate silently changed the training corpus")
            if not any(row.get("images") for row in read_jsonl(probe_train)):
                raise RuntimeError("no length-safe image-bearing smoke probe remains")
            state["stages"]["length"] = {
                "status": "passed",
                "commands": [_command_record(command) for command in length_commands],
                "planner_train": _file_record(planner_train, jsonl=True),
                "planner_audit": _file_record(planner_audit),
                "smoke_probe": _file_record(probe_train, jsonl=True),
                "smoke_probe_audit": _file_record(probe_audit),
                "quality_balance_preserved": True,
            }
            _save_state(status_path, state)
        else:
            frozen_length = state["stages"]["length"]
            if (
                frozen_length.get("planner_train")
                != _file_record(planner_train, jsonl=True)
                or frozen_length.get("planner_audit") != _file_record(planner_audit)
                or frozen_length.get("smoke_probe")
                != _file_record(probe_train, jsonl=True)
                or frozen_length.get("smoke_probe_audit")
                != _file_record(probe_audit)
            ):
                raise RuntimeError("passed length-gate artifacts changed")
        state["status"] = "passed"
        state["current_stage"] = "complete"
        state["error"] = None
        _save_state(status_path, state)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-experiment-config-sha256", required=True)
    parser.add_argument("--train600", type=Path, required=True)
    parser.add_argument("--expected-train600-sha256", required=True)
    parser.add_argument("--source-shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--scope-source",
        type=Path,
        action="append",
        help="Optional exact-row subset JSONL for a bounded rare-path preflight.",
    )
    parser.add_argument("--role-attribution-summary", type=Path, required=True)
    parser.add_argument("--expected-role-attribution-sha256", required=True)
    parser.add_argument("--frozen-role-input", type=Path, required=True)
    parser.add_argument("--observer-artifact-sha256", required=True)
    parser.add_argument("--verifier-artifact-sha256", required=True)
    parser.add_argument("--base-model-artifact-sha256", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--expected-git-branch", required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--gpu-id", type=int, action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sft-python")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-call", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--replay-concurrency", type=int, default=16)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--stop-after", choices=STOP_STAGES, default="length")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = run_pipeline(args)
    except BaseException as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fail-closed helpers for Fast Hybrid bulk launchers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.fast_hybrid_trajectory_control import (
    composite_trajectory_id,
    controller_fingerprint,
)


DATASETS = ("lvbench", "lsdbench", "cgbench")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def parse_dataset_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} entries use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset not in DATASETS or dataset in result or not raw_path.strip():
            raise ValueError(f"invalid or duplicate {label} entry: {value}")
        result[dataset] = Path(raw_path.strip())
    if set(result) != set(DATASETS):
        raise ValueError(f"{label} must cover exactly {DATASETS}")
    return result


def load_frozen_config(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = file_sha256(path)
    expected = expected_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("expected config SHA-256 must be 64 lowercase hexadecimal characters")
    if actual != expected:
        raise RuntimeError(f"config SHA-256 mismatch: expected {expected}, got {actual}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("experiment config must be a schema_version=1 JSON object")
    if set(config.get("datasets") or {}) != set(DATASETS):
        raise ValueError(f"config datasets must cover exactly {DATASETS}")
    return config, actual


def _require_sha(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _rows_by_sample(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in indexed:
            raise ValueError(f"{path} has a missing or duplicate sample_id")
        indexed[sample_id] = row
    return rows, indexed


def validate_inputs(
    config: Mapping[str, Any],
    config_sha256: str,
    specs_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    """Validate every spec against immutable config and source manifests."""

    specs = read_jsonl(specs_path)
    if not specs:
        return [], {}, file_sha256(specs_path)
    train = config.get("train600") or {}
    train_path = Path(str(train.get("path") or ""))
    train_sha = _require_sha(train.get("sha256"), "train600.sha256")
    if not train_path.is_file() or file_sha256(train_path) != train_sha:
        raise RuntimeError("Train600 is missing or its SHA-256 changed")

    source_rows: dict[str, dict[str, Any]] = {}
    dataset_hashes: dict[str, str] = {}
    source_ids: dict[str, set[str]] = {}
    for dataset in DATASETS:
        dataset_config = config["datasets"][dataset]
        manifest = Path(str(dataset_config["train_manifest"]))
        expected = _require_sha(
            dataset_config["train_manifest_sha256"],
            f"datasets.{dataset}.train_manifest_sha256",
        )
        if not manifest.is_file() or file_sha256(manifest) != expected:
            raise RuntimeError(f"{dataset} Train200 manifest is missing or changed")
        rows, indexed = _rows_by_sample(manifest)
        source_rows[dataset] = {
            "path": manifest,
            "rows": rows,
            "index": indexed,
            "sha256": expected,
        }
        dataset_hashes[dataset] = expected
        source_ids[dataset] = set(indexed)

    generation = config.get("trajectory_generation") or {}
    base_budgets = {int(value) for value in generation.get("visual_budgets") or []}
    base_seeds = {int(value) for value in generation.get("generation_seeds") or []}
    rescue_budgets = {
        int(value) for value in generation.get("rescue_visual_budgets") or []
    }
    rescue_seeds = {int(value) for value in generation.get("rescue_seeds") or []}
    judge_seeds = [int(value) for value in generation.get("judge_seeds") or []]
    expected_controller = controller_fingerprint(
        manifest_sha256=train_sha,
        config_sha256=config_sha256,
        dataset_manifest_sha256s=dataset_hashes,
    )
    seen: set[str] = set()
    phases: set[str] = set()
    for spec in specs:
        dataset = str(spec.get("dataset") or "").lower()
        sample_id = str(spec.get("sample_id") or "")
        phase = str(spec.get("phase") or "")
        schedule = str(spec.get("schedule_id") or "")
        trajectory_id = str(spec.get("trajectory_id") or "")
        if dataset not in DATASETS or sample_id not in source_ids.get(dataset, set()):
            raise ValueError(f"spec is outside a frozen Train200 manifest: {dataset}/{sample_id}")
        if phase not in {"base", "rescue"} or not schedule:
            raise ValueError("spec phase/schedule_id is invalid")
        if trajectory_id in seen or not trajectory_id:
            raise ValueError(f"duplicate or missing trajectory_id: {trajectory_id}")
        seen.add(trajectory_id)
        phases.add(phase)
        budget = int(spec.get("max_total_visual_tokens", 0))
        seed = int(spec.get("planner_seed", -1))
        if phase == "base":
            valid = (
                budget in base_budgets
                and seed in base_seeds
                and int(spec.get("max_turns", 0)) == int(generation.get("max_turns", 0))
                and schedule == f"budget_{budget:06d}_seed_{seed}"
            )
        else:
            valid = (
                budget in rescue_budgets
                and seed in rescue_seeds
                and int(spec.get("max_turns", 0))
                == int(generation.get("rescue_max_turns", 0))
                and schedule.endswith(f"budget_{budget:06d}_seed_{seed}")
            )
        if not valid or int(spec.get("max_call_visual_tokens", 0)) != min(12_000, budget):
            raise ValueError(f"spec schedule parameters do not match config: {trajectory_id}")
        replica = int(spec.get("replica_id", -1))
        if replica != 0 or trajectory_id != composite_trajectory_id(
            dataset, sample_id, schedule, replica
        ):
            raise ValueError(f"spec trajectory identity is invalid: {trajectory_id}")
        if list(spec.get("required_judge_seeds") or []) != judge_seeds:
            raise ValueError(f"spec Judge seeds differ from config: {trajectory_id}")
        expected_fields = {
            "config_sha256": config_sha256,
            "manifest_sha256": train_sha,
            "train600_manifest_sha256": train_sha,
            "dataset_manifest_sha256": dataset_hashes[dataset],
            "controller_fingerprint": expected_controller,
        }
        for key, expected in expected_fields.items():
            if str(spec.get(key) or "").lower() != expected:
                raise RuntimeError(f"spec {key} mismatch: {trajectory_id}")
        claimed = str(spec.get("run_spec_fingerprint") or "")
        payload = {key: value for key, value in spec.items() if key != "run_spec_fingerprint"}
        if claimed != canonical_sha256(payload):
            raise RuntimeError(f"spec run fingerprint mismatch: {trajectory_id}")
    if len(phases) != 1:
        raise ValueError("one launcher invocation must contain exactly one phase")
    return specs, source_rows, file_sha256(specs_path)


def freeze_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def freeze_json(path: Path, payload: Mapping[str, Any]) -> None:
    freeze_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def freeze_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    freeze_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "job"


@dataclass(frozen=True)
class BulkJob:
    job_id: str
    endpoint: str
    command: tuple[str, ...]
    log_path: Path
    output_path: Path
    expected_ids: tuple[str, ...]


def shell_line(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def detached_shell_line(
    *, repo_root: Path, python: str, script: Path, argv: Sequence[str], log: Path
) -> str:
    command = [python, str(script), *argv]
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"setsid nohup env PYTHONPATH={shlex.quote(str(repo_root / 'src'))} "
        f"{shell_line(command)} > {shlex.quote(str(log))} 2>&1 < /dev/null &"
    )


def execute_jobs(
    jobs: Sequence[BulkJob],
    *,
    repo_root: Path,
    retry_failed_processes: bool,
) -> list[dict[str, Any]]:
    """Run one child at a time per endpoint, while endpoints run in parallel."""

    queues: dict[str, list[BulkJob]] = {}
    for job in jobs:
        queues.setdefault(job.endpoint, []).append(job)

    def worker(endpoint_jobs: Sequence[BulkJob]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in endpoint_jobs:
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            attempts = 2 if retry_failed_processes else 1
            return_code = -1
            for attempt in range(1, attempts + 1):
                with job.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[bulk-launch attempt={attempt}] {shell_line(job.command)}\n")
                    log.flush()
                    completed = subprocess.run(
                        list(job.command),
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                return_code = completed.returncode
                if return_code == 0:
                    break
            results.append(
                {
                    "job_id": job.job_id,
                    "endpoint": job.endpoint,
                    "return_code": return_code,
                    "log": str(job.log_path),
                    "output": str(job.output_path),
                }
            )
            if return_code != 0:
                break
        return results

    collected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = [pool.submit(worker, queue) for queue in queues.values()]
        for future in as_completed(futures):
            collected.extend(future.result())
    failed = [row for row in collected if row["return_code"] != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} bulk child process(es) failed: {failed[:2]}")
    return sorted(collected, key=lambda row: row["job_id"])

#!/usr/bin/env python3
"""Fail-closed post-repair Perception-Memory process-SFT pipeline.

This wrapper does not create new training logic.  It only binds and resumes the
registered stages after the repair lane: prefix Judges, offline selection, SFT
export, owned-service shutdown, the required one-step smoke, and formal SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.perception_memory_prefix_judge import bind_prefix_jobs
from flashvid_eval.qwen_sft import read_jsonl


AUTOPILOT_VERSION = "perception_memory_post_repair_sft_v1"
EXPECTED_REPAIR_ROWS = 2917
EXPECTED_PORTS = tuple(range(8200, 8208))
TRAIN600_SHA256 = "3995454d973aeb6efe6887e821cd5197e0f17a5b9cc32d7c719e6583b487e6c0"
SELECTION_MINIMA = (360, 100, 90, 20)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _jsonl_record(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {"path": str(path.resolve()), "rows": len(rows), "sha256": _file_sha256(path)}


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _write_json_atomic(path, state)


def _stage_passed(state: Mapping[str, Any], stage: str) -> bool:
    value = (state.get("stages") or {}).get(stage)
    return isinstance(value, Mapping) and value.get("status") == "passed"


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _verify_git(repo_root: Path, branch: str, head: str) -> None:
    if _git_output(repo_root, "rev-parse", "HEAD") != head:
        raise RuntimeError("autopilot Git HEAD changed")
    if _git_output(repo_root, "branch", "--show-current") != branch:
        raise RuntimeError("autopilot Git branch changed")
    if _git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("autopilot checkout is dirty")


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    values = list(command)
    return {"argv": values, "sha256": _canonical_sha256(values)}


def _run_command(
    command: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str] | None = None
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} COMMAND {json.dumps(list(command))}\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"{_now()} EXIT {result.returncode}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return int(result.returncode)


def _parse_service_bindings(values: Sequence[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        try:
            raw_port, raw_pid = value.split("=", 1)
            port, pid = int(raw_port), int(raw_pid)
        except (TypeError, ValueError) as error:
            raise ValueError("--owned-service must be PORT=PID") from error
        if port in result or pid <= 0:
            raise ValueError("owned service ports must be unique and PIDs positive")
        result[port] = pid
    if tuple(sorted(result)) != EXPECTED_PORTS or len(set(result.values())) != 8:
        raise ValueError("owned services must bind unique PIDs for ports 8200-8207")
    return result


def _proc_start_ticks(stat_text: str) -> int:
    close = stat_text.rfind(")")
    if close < 0:
        raise ValueError("invalid /proc stat")
    fields_after_comm = stat_text[close + 2 :].split()
    if len(fields_after_comm) < 20:
        raise ValueError("invalid /proc stat field count")
    return int(fields_after_comm[19])


def _current_uid() -> int:
    if not hasattr(os, "getuid"):
        raise RuntimeError("owned-service verification requires Linux")
    return int(os.getuid())


def _capture_service(
    *, pid: int, port: int, log_root: Path, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    root = proc_root / str(pid)
    if not root.is_dir():
        raise RuntimeError(f"owned service PID is not live: {pid}")
    status = (root / "status").read_text(encoding="utf-8")
    uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
    uid_fields = uid_line.split()
    current_uid = _current_uid()
    if len(uid_fields) < 2 or int(uid_fields[1]) != current_uid:
        raise RuntimeError(f"service PID {pid} is not owned by the current UID")
    argv = [
        item.decode("utf-8", errors="replace")
        for item in (root / "cmdline").read_bytes().split(b"\0")
        if item
    ]
    if not any(Path(item).name == "transformers" for item in argv):
        raise RuntimeError(f"service PID {pid} is not a Transformers CLI")
    if "serve" not in argv or "Qwen3.5-9B" not in argv:
        raise RuntimeError(f"service PID {pid} is not the bound Qwen3.5-9B service")
    try:
        actual_port = int(argv[argv.index("--port") + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"service PID {pid} has no valid --port") from error
    if actual_port != port:
        raise RuntimeError(f"service PID {pid} serves port {actual_port}, not {port}")
    expected_log = (log_root / f"transformers_{port}.log").resolve()
    stdout_link = Path(os.readlink(root / "fd" / "1")).resolve()
    stderr_link = Path(os.readlink(root / "fd" / "2")).resolve()
    if stdout_link != expected_log or stderr_link != expected_log:
        raise RuntimeError(f"service PID {pid} is not bound to the experiment log")
    return {
        "pid": pid,
        "port": port,
        "uid": current_uid,
        "start_ticks": _proc_start_ticks((root / "stat").read_text(encoding="utf-8")),
        "argv_sha256": _canonical_sha256(argv),
        "log_path": str(expected_log),
    }


def _same_service(snapshot: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    fields = ("pid", "port", "uid", "start_ticks", "argv_sha256", "log_path")
    return all(snapshot.get(field) == current.get(field) for field in fields)


def _stop_services(
    inventory: Sequence[Mapping[str, Any]],
    *,
    log_root: Path,
    proc_root: Path = Path("/proc"),
    timeout: float = 30.0,
) -> None:
    live: list[int] = []
    for snapshot in inventory:
        pid, port = int(snapshot["pid"]), int(snapshot["port"])
        if not (proc_root / str(pid)).is_dir():
            continue
        current = _capture_service(pid=pid, port=port, log_root=log_root, proc_root=proc_root)
        if not _same_service(snapshot, current):
            raise RuntimeError(f"PID {pid} was reused or its service identity changed")
        live.append(pid)
    for pid in live:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while any((proc_root / str(pid)).is_dir() for pid in live):
        if time.monotonic() >= deadline:
            raise RuntimeError("owned services did not exit after SIGTERM")
        time.sleep(0.5)


def _judge_command(
    args: argparse.Namespace,
    trajectories: Path,
    output: Path,
    *,
    resume: bool,
    retry_errors: bool = False,
) -> list[str]:
    command = [
        args.python,
        str(args.repo_root / "scripts/judge_perception_memory_prefixes.py"),
        "--trajectories",
        str(trajectories),
        "--output",
        str(output),
        "--model",
        args.model,
        "--concurrency",
        str(args.judge_concurrency),
        "--timeout",
        str(args.request_timeout),
        "--judge-seed",
        "17",
        "--judge-seed",
        "42",
        "--judge-seed",
        "73",
    ]
    for base_url in args.base_url:
        command.extend(["--base-url", base_url])
    if resume:
        command.append("--resume")
    if retry_errors:
        command.append("--retry-errors")
    return command


def _selection_command(
    args: argparse.Namespace, trajectories: Path, judgments: Path
) -> list[str]:
    directory = args.run_root / "selection"
    command = [
        args.python,
        str(args.repo_root / "scripts/select_perception_memory_trajectories.py"),
        "--trajectories",
        str(trajectories),
        "--prefix-judgments",
        str(judgments),
        "--answers",
        str(args.answers),
        "--expected-answers-sha256",
        args.expected_answers_sha256,
        "--labeled-output",
        str(directory / "labeled.jsonl"),
        "--selected-output",
        str(directory / "selected.jsonl"),
        "--summary",
        str(directory / "summary.json"),
    ]
    if any((directory / name).exists() for name in ("labeled.jsonl", "selected.jsonl", "summary.json")):
        command.append("--overwrite")
    return command


def _build_command(args: argparse.Namespace) -> list[str]:
    output = args.run_root / "sft_data/perception_memory_sft.jsonl"
    summary = args.run_root / "sft_data/summary.json"
    command = [
        args.python,
        str(args.repo_root / "scripts/build_perception_memory_sft.py"),
        "--selected",
        str(args.run_root / "selection/selected.jsonl"),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--minimum-total",
        str(SELECTION_MINIMA[0]),
        "--minimum-per-dataset",
        str(SELECTION_MINIMA[1]),
        "--minimum-candidate-fixes",
        str(SELECTION_MINIMA[2]),
        "--minimum-candidate-fixes-per-dataset",
        str(SELECTION_MINIMA[3]),
    ]
    if output.exists() or summary.exists():
        command.append("--overwrite")
    return command


def _training_command(
    args: argparse.Namespace, *, smoke: bool, resume: bool = False
) -> list[str]:
    train_data = args.run_root / "sft_data/perception_memory_sft.jsonl"
    smoke_dir = args.run_root / "checkpoints/smoke"
    formal_dir = args.run_root / "checkpoints/formal"
    command = [
        "bash",
        str(args.repo_root / "scripts/train_qwen_agent_9b_lora.sh"),
        "--train-data",
        str(train_data),
        "--output-dir",
        str(smoke_dir if smoke else formal_dir),
    ]
    if smoke:
        command.extend(
            ["--formal-output-dir", str(formal_dir), "--smoke", "--load-weights-preflight"]
        )
    else:
        command.extend(
            ["--smoke-report", str(smoke_dir / "preflight/training_update.json")]
        )
        if resume:
            command.append("--resume")
    return command


def _training_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "SFT_ENV_DIR": str(args.sft_env_dir),
            "MODEL_PATH": str(args.model_path),
            "EXPECTED_MODEL_ARTIFACT_SHA256": args.expected_model_artifact_sha256,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "USE_FSDP2": "1",
            "HF_ENDPOINT": args.hf_endpoint,
        }
    )
    return env


def _audit_judgments(trajectories: Path, output: Path) -> dict[str, Any]:
    expected_ids = {job.prefix_id for job in bind_prefix_jobs(read_jsonl(trajectories))}
    rows = read_jsonl(output)
    actual_ids = [str(row.get("prefix_id") or "") for row in rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise RuntimeError("prefix Judge output does not exactly cover the merged input")
    infrastructure_errors = 0
    evidence_insufficient = 0
    required_seeds = {17, 42, 73}
    for row in rows:
        if row.get("annotation_leak_check") != "passed":
            raise RuntimeError("prefix Judge row failed annotation leak audit")
        confirmations = row.get("judge_confirmations")
        if not isinstance(confirmations, list) or len(confirmations) != len(
            required_seeds
        ):
            raise RuntimeError("prefix Judge row lacks exactly three confirmations")
        if any(not isinstance(item, Mapping) for item in confirmations):
            raise RuntimeError("prefix Judge confirmation must be an object")
        seeds = [int(item.get("judge_seed")) for item in confirmations]
        if len(seeds) != len(set(seeds)) or set(seeds) != required_seeds:
            raise RuntimeError("prefix Judge seeds must be exactly 17/42/73")
        if any(
            item.get("annotation_leak_check") != "passed"
            or item.get("failure_class") == "annotation_leak"
            for item in confirmations
        ):
            raise RuntimeError("prefix Judge confirmation failed annotation leak audit")
        errors = [
            item
            for item in confirmations
            if item.get("error") is not None
        ]
        infrastructure_errors += len(errors)
        if errors:
            continue
        if any(
            item.get("parsed_valid") is not True
            for item in confirmations
        ):
            evidence_insufficient += 1
    return {
        "expected_prefixes": len(expected_ids),
        "rows": len(rows),
        "failures": infrastructure_errors,
        "infrastructure_errors": infrastructure_errors,
        "persistent_infrastructure_failures": infrastructure_errors,
        "evidence_insufficient": evidence_insufficient,
        "output_sha256": _file_sha256(output),
    }


def _unique_trajectory_ids(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> set[str]:
    values = [str(row.get("source_trajectory_id") or "").strip() for row in rows]
    if any(not value for value in values):
        raise RuntimeError(f"{label} contains an empty source_trajectory_id")
    if len(values) != len(set(values)):
        raise RuntimeError(f"{label} contains duplicate source_trajectory_id values")
    return set(values)


def _audit_repair_partition(repair_run_root: Path) -> tuple[Path, dict[str, Any]]:
    merged_dir = repair_run_root / "merged"
    success_path = merged_dir / "merged_success.jsonl"
    failure_path = merged_dir / "double_failures.jsonl"
    summary_path = merged_dir / "merge_summary.json"
    scope_path = repair_run_root / "repair_scope/frozen_scope.json"
    summary = _read_json(summary_path)
    scope = _read_json(scope_path)
    successes = read_jsonl(success_path)
    failures = read_jsonl(failure_path)
    success_ids = _unique_trajectory_ids(successes, label="merged success")
    failure_ids = _unique_trajectory_ids(failures, label="double failures")
    if success_ids & failure_ids:
        raise RuntimeError("repair success/failure partitions overlap")

    frozen_success = scope.get("base_success_ids")
    frozen_failure = scope.get("base_failure_ids")
    if not isinstance(frozen_success, list) or not isinstance(frozen_failure, list):
        raise RuntimeError("frozen repair scope has no complete trajectory ID lists")
    frozen_values = [str(value).strip() for value in [*frozen_success, *frozen_failure]]
    if any(not value for value in frozen_values) or len(frozen_values) != len(
        set(frozen_values)
    ):
        raise RuntimeError("frozen repair scope contains empty or duplicate IDs")
    frozen_ids = set(frozen_values)
    if len(frozen_ids) != EXPECTED_REPAIR_ROWS:
        raise RuntimeError("frozen repair scope is not the registered 2917 rows")
    if success_ids | failure_ids != frozen_ids:
        raise RuntimeError("repair partitions do not exactly cover the frozen scope")

    success_record = _jsonl_record(success_path)
    failure_record = _jsonl_record(failure_path)
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("repair merge summary has no artifact records")
    for name, record in (
        ("merged_success.jsonl", success_record),
        ("double_failures.jsonl", failure_record),
    ):
        expected = artifacts.get(name)
        if not isinstance(expected, Mapping) or expected.get("rows") != record["rows"]:
            raise RuntimeError(f"repair merge summary row count changed: {name}")
        if expected.get("sha256") != record["sha256"]:
            raise RuntimeError(f"repair merge summary SHA-256 changed: {name}")
    scope_sha256 = _file_sha256(scope_path)
    if summary.get("frozen_scope_sha256") != scope_sha256:
        raise RuntimeError("repair merge is bound to a different frozen scope")
    if summary.get("prefix_bind_passed") is not True:
        raise RuntimeError("repair success partition did not pass prefix binding")
    if summary.get("status") not in {"passed", "completed_with_failures"}:
        raise RuntimeError("repair merge has an unsupported status")
    if (
        summary.get("source_rows") != EXPECTED_REPAIR_ROWS
        or summary.get("merged_success") != success_record["rows"]
        or summary.get("double_failures") != failure_record["rows"]
        or success_record["rows"] + failure_record["rows"] != EXPECTED_REPAIR_ROWS
    ):
        raise RuntimeError("repair merge summary does not match the complete partition")
    if set(summary.get("double_failure_ids") or []) != failure_ids:
        raise RuntimeError("repair merge summary double-failure IDs changed")

    return success_path, {
        "merged_success": success_record,
        "excluded_double_failures": failure_record,
        "excluded_trajectory_ids_sha256": _canonical_sha256(sorted(failure_ids)),
        "frozen_scope": {
            "path": str(scope_path.resolve()),
            "rows": len(frozen_ids),
            "sha256": scope_sha256,
        },
        "partition_complete": True,
        "partition_disjoint": True,
        "prefix_bind_passed": True,
    }


def _wait_for_repair(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    status_path = args.repair_run_root / "status.json"
    deadline = time.monotonic() + args.wait_timeout
    while True:
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("status") == "failed":
                raise RuntimeError(f"repair pipeline failed: {status.get('error')}")
            if status.get("status") == "passed" and status.get("current_stage") == "complete":
                break
        if time.monotonic() >= deadline:
            raise TimeoutError("repair pipeline did not complete before the timeout")
        time.sleep(args.poll_seconds)
    merged, partition = _audit_repair_partition(args.repair_run_root)
    record = partition["merged_success"]
    expected_sha = ((status.get("stages") or {}).get("merge") or {}).get(
        "merged_success_sha256"
    )
    if expected_sha != record["sha256"]:
        raise RuntimeError("repair status and merged output SHA-256 differ")
    return merged, {
        "repair_status": {"path": str(status_path.resolve()), "sha256": _file_sha256(status_path)},
        "merged": record,
        "repair_partition": partition,
        "merge_summary_sha256": _file_sha256(
            args.repair_run_root / "merged/merge_summary.json"
        ),
    }


def _fingerprint(args: argparse.Namespace) -> str:
    return _canonical_sha256(
        {
            "version": AUTOPILOT_VERSION,
            "repo_root": str(args.repo_root.resolve()),
            "branch": args.expected_git_branch,
            "head": args.expected_git_head,
            "repair_run_root": str(args.repair_run_root.resolve()),
            "answers": str(args.answers.resolve()),
            "answers_sha256": args.expected_answers_sha256,
            "base_urls": args.base_url,
            "model": args.model,
            "judge_concurrency": args.judge_concurrency,
            "owned_services": args.owned_service,
            "service_log_root": str(args.service_log_root.resolve()),
            "sft_env_dir": str(args.sft_env_dir.resolve()),
            "model_path": str(args.model_path.resolve()),
            "model_artifact_sha256": args.expected_model_artifact_sha256,
            "selection_minima": SELECTION_MINIMA,
        }
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.repo_root = args.repo_root.resolve()
    args.repair_run_root = args.repair_run_root.resolve()
    args.run_root = args.run_root.resolve()
    if len(args.base_url) != 8 or len(set(args.base_url)) != 8:
        raise ValueError("autopilot requires eight unique Judge endpoints")
    if args.expected_answers_sha256 != TRAIN600_SHA256:
        raise ValueError("autopilot requires the frozen Train600 SHA-256")
    if _file_sha256(args.answers) != args.expected_answers_sha256:
        raise ValueError("frozen Train600 answer artifact changed")
    bindings = _parse_service_bindings(args.owned_service)
    fingerprint = _fingerprint(args)
    status_path = args.run_root / "status.json"
    if args.run_root.exists():
        if not args.resume:
            raise FileExistsError("autopilot run root exists; pass --resume")
        state = _read_json(status_path)
        if state.get("config_sha256") != fingerprint:
            raise RuntimeError("autopilot resume fingerprint mismatch")
        state["status"] = "running"
        state["error"] = None
    else:
        args.run_root.mkdir(parents=True)
        state = {
            "schema_version": 1,
            "autopilot_version": AUTOPILOT_VERSION,
            "config_sha256": fingerprint,
            "status": "running",
            "current_stage": "inventory_services",
            "created_at": _now(),
            "updated_at": _now(),
            "stages": {},
            "error": None,
        }
    _save_state(status_path, state)

    try:
        _verify_git(args.repo_root, args.expected_git_branch, args.expected_git_head)
        if not _stage_passed(state, "inventory_services"):
            inventory = [
                _capture_service(
                    pid=bindings[port], port=port, log_root=args.service_log_root
                )
                for port in EXPECTED_PORTS
            ]
            state["stages"]["inventory_services"] = {
                "status": "passed",
                "ended_at": _now(),
                "services": inventory,
            }
            _save_state(status_path, state)
        inventory = state["stages"]["inventory_services"]["services"]

        if not _stage_passed(state, "wait_repair"):
            state["current_stage"] = "wait_repair"
            _save_state(status_path, state)
            merged, audit = _wait_for_repair(args)
            state["stages"]["wait_repair"] = {
                "status": "passed",
                "ended_at": _now(),
                **audit,
            }
            _save_state(status_path, state)
        else:
            merged, audit = _wait_for_repair(args)
            frozen_audit = state["stages"]["wait_repair"]
            if (
                audit["merged"]["sha256"] != frozen_audit["merged"]["sha256"]
                or audit["repair_partition"]
                != frozen_audit.get("repair_partition")
            ):
                raise RuntimeError("passed repair partition changed")

        judgments = args.run_root / "prefix_judgments.jsonl"
        progress = judgments.with_suffix(judgments.suffix + ".progress.jsonl")
        if not _stage_passed(state, "prefix_judge"):
            state["current_stage"] = "prefix_judge"
            judge_stage = state["stages"].setdefault("prefix_judge", {})
            command = _judge_command(
                args,
                merged,
                judgments,
                resume=judgments.exists() or progress.exists(),
            )
            judge_stage["status"] = "running"
            judge_stage["input"] = _jsonl_record(merged)
            judge_stage.setdefault("commands", []).append(_command_record(command))
            _save_state(status_path, state)
            if _run_command(command, cwd=args.repo_root, log_path=args.run_root / "logs/prefix_judge.log"):
                raise RuntimeError("prefix Judge command failed")
            judge_audit = _audit_judgments(merged, judgments)
            if judge_audit.get("infrastructure_errors", judge_audit["failures"]):
                if judge_stage.get("retry_started"):
                    judge_stage["persistent_infrastructure_failures"] = judge_audit.get(
                        "infrastructure_errors", judge_audit["failures"]
                    )
                else:
                    retry = _judge_command(
                        args, merged, judgments, resume=True, retry_errors=True
                    )
                    judge_stage["retry_started"] = True
                    judge_stage["retry_command"] = _command_record(retry)
                    _save_state(status_path, state)
                    if _run_command(
                        retry,
                        cwd=args.repo_root,
                        log_path=args.run_root / "logs/prefix_judge_retry.log",
                    ):
                        raise RuntimeError("prefix Judge retry command failed")
                    judge_audit = _audit_judgments(merged, judgments)
                    judge_stage["persistent_infrastructure_failures"] = (
                        judge_audit.get(
                            "infrastructure_errors", judge_audit["failures"]
                        )
                    )
            judge_stage.update(
                {"status": "passed", "ended_at": _now(), "audit": judge_audit}
            )
            _save_state(status_path, state)
        elif _audit_judgments(merged, judgments)["output_sha256"] != state["stages"]["prefix_judge"]["audit"]["output_sha256"]:
            raise RuntimeError("passed prefix Judge output changed")

        for stage, command in (
            ("selection", _selection_command(args, merged, judgments)),
            ("build_sft", _build_command(args)),
        ):
            output_paths = (
                {
                    "labeled": args.run_root / "selection/labeled.jsonl",
                    "selected": args.run_root / "selection/selected.jsonl",
                    "summary": args.run_root / "selection/summary.json",
                }
                if stage == "selection"
                else {
                    "sft_data": args.run_root
                    / "sft_data/perception_memory_sft.jsonl",
                    "summary": args.run_root / "sft_data/summary.json",
                }
            )
            if _stage_passed(state, stage):
                output = (
                    args.run_root / "selection/selected.jsonl"
                    if stage == "selection"
                    else args.run_root / "sft_data/perception_memory_sft.jsonl"
                )
                if _jsonl_record(output) != state["stages"][stage]["output"]:
                    raise RuntimeError(f"passed {stage} output changed")
                current_artifacts = {
                    name: _file_record(path) for name, path in output_paths.items()
                }
                if current_artifacts != state["stages"][stage].get("artifacts"):
                    raise RuntimeError(f"passed {stage} artifacts changed")
                continue
            state["current_stage"] = stage
            inputs = (
                {
                    "trajectories": _jsonl_record(merged),
                    "judgments": _jsonl_record(judgments),
                    "answers": _file_record(args.answers),
                }
                if stage == "selection"
                else {
                    "selected": _jsonl_record(
                        args.run_root / "selection/selected.jsonl"
                    )
                }
            )
            state["stages"][stage] = {
                "status": "running",
                "command": _command_record(command),
                "inputs": inputs,
            }
            _save_state(status_path, state)
            if _run_command(command, cwd=args.repo_root, log_path=args.run_root / f"logs/{stage}.log"):
                raise RuntimeError(f"{stage} command failed")
            output = (
                args.run_root / "selection/selected.jsonl"
                if stage == "selection"
                else args.run_root / "sft_data/perception_memory_sft.jsonl"
            )
            state["stages"][stage].update(
                {
                    "status": "passed",
                    "ended_at": _now(),
                    "output": _jsonl_record(output),
                    "artifacts": {
                        name: _file_record(path) for name, path in output_paths.items()
                    },
                }
            )
            _save_state(status_path, state)

        if not _stage_passed(state, "stop_services"):
            state["current_stage"] = "stop_services"
            _save_state(status_path, state)
            _stop_services(inventory, log_root=args.service_log_root)
            state["stages"]["stop_services"] = {
                "status": "passed",
                "ended_at": _now(),
                "stopped_pids": [int(item["pid"]) for item in inventory],
            }
            _save_state(status_path, state)

        training_env = _training_env(args)
        smoke_report = args.run_root / "checkpoints/smoke/preflight/training_update.json"
        if not _stage_passed(state, "smoke"):
            state["current_stage"] = "smoke"
            if smoke_report.is_file():
                command = [
                    args.python,
                    str(args.repo_root / "scripts/qwen_sft_smoke_gate.py"),
                    "check",
                    "--report",
                    str(smoke_report),
                    "--formal-output-dir",
                    str(args.run_root / "checkpoints/formal"),
                    "--train-data",
                    str(args.run_root / "sft_data/perception_memory_sft.jsonl"),
                    "--base-model-artifact-sha256",
                    args.expected_model_artifact_sha256,
                ]
            else:
                smoke_dir = args.run_root / "checkpoints/smoke"
                if smoke_dir.exists() and any(smoke_dir.iterdir()):
                    raise RuntimeError("incomplete smoke output cannot be resumed safely")
                command = _training_command(args, smoke=True)
            state["stages"]["smoke"] = {
                "status": "running",
                "command": _command_record(command),
                "training_data_sha256": _file_sha256(
                    args.run_root / "sft_data/perception_memory_sft.jsonl"
                ),
            }
            _save_state(status_path, state)
            if _run_command(command, cwd=args.repo_root, log_path=args.run_root / "logs/smoke.log", env=training_env):
                raise RuntimeError("one-step SFT smoke failed")
            if not smoke_report.is_file():
                raise RuntimeError("smoke completed without a bound report")
            state["stages"]["smoke"].update(
                {"status": "passed", "ended_at": _now(), "report_sha256": _file_sha256(smoke_report)}
            )
            _save_state(status_path, state)
        else:
            if not smoke_report.is_file() or _file_sha256(smoke_report) != state[
                "stages"
            ]["smoke"].get("report_sha256"):
                raise RuntimeError("passed smoke report changed")
            if _file_sha256(
                args.run_root / "sft_data/perception_memory_sft.jsonl"
            ) != state["stages"]["smoke"].get("training_data_sha256"):
                raise RuntimeError("training data changed after the smoke gate")

        if not _stage_passed(state, "formal_sft"):
            state["current_stage"] = "formal_sft"
            formal_dir = args.run_root / "checkpoints/formal"
            checkpoints = sorted(formal_dir.glob("checkpoint-*")) if formal_dir.exists() else []
            command = _training_command(args, smoke=False, resume=bool(checkpoints))
            state["stages"]["formal_sft"] = {
                "status": "running",
                "command": _command_record(command),
                "resumed_from_checkpoint": bool(checkpoints),
            }
            _save_state(status_path, state)
            if _run_command(command, cwd=args.repo_root, log_path=args.run_root / "logs/formal_sft.log", env=training_env):
                raise RuntimeError("formal SFT command failed")
            checkpoints = sorted(formal_dir.glob("checkpoint-*"))
            if not checkpoints:
                raise RuntimeError("formal SFT completed without checkpoints")
            state["stages"]["formal_sft"].update(
                {
                    "status": "passed",
                    "ended_at": _now(),
                    "checkpoints": [path.name for path in checkpoints],
                }
            )
            _save_state(status_path, state)
        else:
            formal_dir = args.run_root / "checkpoints/formal"
            expected = state["stages"]["formal_sft"].get("checkpoints")
            actual = [path.name for path in sorted(formal_dir.glob("checkpoint-*"))]
            if not expected or actual != expected:
                raise RuntimeError("passed formal SFT checkpoints changed")

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
    parser.add_argument("--expected-git-branch", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--repair-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--expected-answers-sha256", default=TRAIN600_SHA256)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--owned-service", action="append", required=True)
    parser.add_argument("--service-log-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=80.0)
    parser.add_argument("--sft-env-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-model-artifact-sha256", required=True)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout", type=float, default=43200.0)
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

#!/usr/bin/env python3
"""Run and audit one frozen Fast Hybrid Teacher or SFT Dev/Test matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashvid_eval.fast_hybrid_eval_protocol import (
    DATASETS,
    audit_result_file,
    canonical_sha256,
    freeze_json,
    load_protocol,
    manifest_sample_ids,
    require_sha256,
    sha256_file,
)
try:
    from scripts.fast_hybrid_bulk_common import BulkJob, execute_jobs, shell_line
    from scripts.fingerprint_qwen_lora_stack import fingerprint_stack
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from fast_hybrid_bulk_common import (  # type: ignore[no-redef]
        BulkJob,
        execute_jobs,
        shell_line,
    )
    from fingerprint_qwen_lora_stack import (  # type: ignore[no-redef]
        fingerprint_stack,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_checkpoint(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("kind") != "fast_hybrid_sft_checkpoint":
        raise ValueError("not a frozen Fast Hybrid SFT checkpoint")
    claimed = value.get("checkpoint_fingerprint")
    body = {key: item for key, item in value.items() if key != "checkpoint_fingerprint"}
    if claimed != canonical_sha256(body):
        raise RuntimeError("checkpoint fingerprint mismatch")
    if value["experiment_config"]["sha256"] != protocol["experiment_config"]["sha256"]:
        raise RuntimeError("checkpoint belongs to another Fast Hybrid experiment")
    base_hash = require_sha256(
        value["base_model_artifact_sha256"], "base_model_artifact_sha256"
    )
    if base_hash != protocol["base_model_artifact_sha256"]:
        raise RuntimeError("checkpoint base model differs from evaluation protocol")
    adapter = Path(value["adapter"]["path"])
    stack = fingerprint_stack(base_hash, adapter)
    for key in ("adapter_artifact_sha256", "served_stack_sha256"):
        if stack[key] != (
            value["adapter"]["artifact_sha256"]
            if key == "adapter_artifact_sha256"
            else value[key]
        ):
            raise RuntimeError(f"checkpoint {key} changed")
    return value


def _load_winner(path: Path, protocol_sha256: str) -> Path:
    value = _load_json(path)
    if value.get("kind") != "fast_hybrid_sft_winner" or value.get("status") != "passed":
        raise ValueError("Test evaluation requires a passed Fast Hybrid SFT winner")
    if value.get("evaluation_protocol_sha256") != protocol_sha256:
        raise RuntimeError("winner belongs to another evaluation protocol")
    reference = value.get("checkpoint_config") or {}
    checkpoint = Path(str(reference.get("path") or ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != reference.get("sha256"):
        raise RuntimeError("winner checkpoint config is missing or changed")
    return checkpoint


def build_jobs(
    *,
    protocol: Mapping[str, Any],
    phase: str,
    run_id: str,
    served_name: str,
    served_model_sha256: str,
    teacher_model_sha256: str,
    output_root: Path,
    python: str,
    repo_root: Path,
    resume: bool,
) -> list[BulkJob]:
    parameters = protocol["parameters"]
    endpoints = [str(value).rstrip("/") for value in parameters["endpoints"]]
    jobs: list[BulkJob] = []
    for index, dataset in enumerate(DATASETS):
        entry = protocol["splits"][phase][dataset]
        output_dir = output_root / dataset
        command = [
            python,
            str(repo_root / "scripts" / "evaluate_mcq.py"),
            "--dataset",
            dataset,
            "--backend",
            "fast_hybrid_eva",
            "--annotations",
            str(entry["annotations"]),
            "--video-root",
            str(entry["video_root"]),
            "--base-url",
            endpoints[index % len(endpoints)],
            "--api-key",
            "no",
            "--model",
            served_name,
            "--sample",
            str(entry["manifest"]["count"]),
            "--seed",
            str(parameters["seed"]),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(entry["manifest"]["path"]),
            "--expected-manifest-sha256",
            str(entry["manifest"]["sha256"]),
            "--frame-root",
            str(output_root / "frames" / dataset),
            "--concurrency",
            str(parameters["concurrency_per_endpoint"]),
            "--max-turns",
            str(parameters["max_turns"]),
            "--max-call-visual-tokens",
            str(parameters["max_call_visual_tokens"]),
            "--max-total-visual-tokens",
            str(parameters["max_total_visual_tokens"]),
            "--agent-version",
            str(protocol["agent_version"]),
            "--timeout",
            str(parameters["timeout"]),
            "--candidate-results",
            str(entry["candidate"]["path"]),
            "--controller-temperature",
            str(parameters["temperature"]),
            "--experiment-config-sha256",
            str(protocol["experiment_config"]["sha256"]),
            "--model-artifact-sha256",
            served_model_sha256,
            "--teacher-model-artifact-sha256",
            teacher_model_sha256,
        ]
        if resume:
            # A child can exit successfully while still writing per-sample
            # timeout/error rows.  Resume those rows once before the frozen
            # matrix audit instead of treating their sparse error schema as a
            # protocol-field drift.
            command.extend(("--resume", "--retry-errors"))
        jobs.append(
            BulkJob(
                job_id=f"{phase}:{run_id}:{dataset}",
                endpoint=endpoints[index % len(endpoints)],
                command=tuple(command),
                log_path=output_root / "logs" / f"{dataset}.log",
                output_path=output_dir / f"{dataset}_fast_hybrid_eva.jsonl",
                expected_ids=(),
            )
        )
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--phase", choices=("dev", "test"), required=True)
    parser.add_argument("--mode", choices=("teacher", "checkpoint"), required=True)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--winner", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        protocol_sha = require_sha256(
            args.expected_protocol_sha256, "expected_protocol_sha256"
        )
        protocol = load_protocol(args.protocol, protocol_sha)
        if args.mode == "teacher":
            if args.checkpoint_config is not None or args.winner is not None:
                raise ValueError("Teacher mode does not accept checkpoint/winner")
            served_name = "Qwen3.5-9B"
            served_hash = protocol["base_model_artifact_sha256"]
            checkpoint_reference = None
        else:
            if args.phase == "test":
                if args.winner is None or args.checkpoint_config is not None:
                    raise ValueError("Test checkpoint mode requires only --winner")
                checkpoint_path = _load_winner(args.winner, protocol_sha)
            else:
                if args.checkpoint_config is None or args.winner is not None:
                    raise ValueError("Dev checkpoint mode requires only --checkpoint-config")
                checkpoint_path = args.checkpoint_config
            checkpoint = _load_checkpoint(checkpoint_path, protocol)
            served_name = str(checkpoint["served_name"])
            served_hash = str(checkpoint["served_stack_sha256"])
            checkpoint_reference = {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "checkpoint_id": checkpoint["checkpoint_id"],
            }
        teacher_hash = str(protocol["base_model_artifact_sha256"])
        jobs = build_jobs(
            protocol=protocol,
            phase=args.phase,
            run_id=args.run_id,
            served_name=served_name,
            served_model_sha256=served_hash,
            teacher_model_sha256=teacher_hash,
            output_root=args.output_root,
            python=args.python,
            repo_root=args.repo_root.resolve(),
            resume=args.resume,
        )
        plan = {
            "schema_version": 1,
            "kind": "fast_hybrid_sft_eval_plan",
            "phase": args.phase,
            "mode": args.mode,
            "run_id": args.run_id,
            "evaluation_protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": protocol_sha,
            },
            "checkpoint_config": checkpoint_reference,
            "served_name": served_name,
            "served_model_sha256": served_hash,
            "teacher_model_sha256": teacher_hash,
            "jobs": [
                {
                    **asdict(job),
                    "command": list(job.command),
                    "log_path": str(job.log_path),
                    "output_path": str(job.output_path),
                }
                for job in jobs
            ],
        }
        plan["plan_fingerprint"] = canonical_sha256(plan)
        plan_path = args.output_root / "evaluation_plan.json"
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
        for job in jobs:
            print(shell_line(job.command))
        if args.dry_run:
            return 0
        freeze_json(plan_path, plan)
        # Each child is resume-safe. Retry the failed process once so a transient
        # service/API interruption does not strand an otherwise complete matrix.
        execute_jobs(jobs, repo_root=args.repo_root.resolve(), retry_failed_processes=True)
        files: dict[str, Any] = {}
        for job, dataset in zip(jobs, DATASETS):
            entry = protocol["splits"][args.phase][dataset]
            files[dataset] = audit_result_file(
                job.output_path,
                dataset=dataset,
                expected_count=int(entry["manifest"]["count"]),
                expected_sample_ids=manifest_sample_ids(
                    Path(str(entry["manifest"]["path"]))
                ),
                manifest_sha256=str(entry["manifest"]["sha256"]),
                candidate_sha256=str(entry["candidate"]["sha256"]),
                experiment_config_sha256=str(protocol["experiment_config"]["sha256"]),
                served_model_sha256=served_hash,
                teacher_model_sha256=teacher_hash,
            )
        report = {
            "schema_version": 1,
            "kind": "fast_hybrid_sft_eval_run",
            "status": "passed",
            "phase": args.phase,
            "mode": args.mode,
            "run_id": args.run_id,
            "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
            "evaluation_protocol_sha256": protocol_sha,
            "checkpoint_config": checkpoint_reference,
            "served_model_sha256": served_hash,
            "teacher_model_sha256": teacher_hash,
            "files": files,
        }
        freeze_json(args.output_root / "evaluation_run.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

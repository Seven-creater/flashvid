from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .qwen_reporting import promotion_decision


DATASETS = ("lvbench", "lsdbench", "cgbench")
REQUIRED_AGENT_SEEDS = (17, 42, 73)
SUPPORTED_STAGES = (
    "a0_eva_clean",
    "a1_storyboard_zoom",
    "a2_multi_clue_memory",
    "a3_hierarchical_search",
    "a4_independent_arbitration",
)
DEV_PHASES = frozenset(
    {
        "protocol_smoke",
        "protocol_audit",
        "direct_dev",
        "agent_dev",
        "teacher_dev",
        "sft_dev",
        "final_matrix",
    }
)

SMOKE_PROTOCOL_REJECTION_REASON = "smoke_engineering_failure_rate_above_1_percent"


def canonical_sha256(payload: Any) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _command_value(command: Sequence[str], flag: str, *, required: bool = True) -> str | None:
    matches = [index for index, value in enumerate(command) if value == flag]
    if not matches:
        if required:
            raise ValueError(f"task command is missing {flag}")
        return None
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"task command has an invalid {flag}")
    return str(command[matches[0] + 1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"JSONL row {line_number} has no sample_id: {path}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id} in {path}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def load_frozen_run_plan(path: Path, experiment_config_sha256: str) -> dict[str, Any]:
    payload = _load_json(path, "run plan")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported run plan schema: {path}")
    claimed = str(payload.get("plan_sha256") or "")
    computed = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    if claimed != computed:
        raise RuntimeError(f"run plan self-hash mismatch: {path}")
    if str(payload.get("config_sha256") or "") != experiment_config_sha256:
        raise RuntimeError(f"run plan belongs to a different experiment config: {path}")
    if payload.get("phase") not in DEV_PHASES:
        raise ValueError(f"selection cannot consume phase {payload.get('phase')!r}: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or payload.get("task_count") != len(tasks):
        raise ValueError(f"run plan task_count/tasks mismatch: {path}")
    payload["_source_path"] = str(path.resolve())
    payload["_source_sha256"] = file_sha256(path)
    return payload


@dataclass(frozen=True)
class DevRun:
    phase: str
    plan_path: str
    plan_sha256: str
    task_id: str
    dataset: str
    model_key: str
    model: str
    protocol: str
    seed: int
    mode: str | None
    sampling: str | None
    strategy: str | None
    variant_id: str | None
    agent_config_path: str | None
    agent_config_sha256: str | None
    result_path: str
    result_sha256: str
    rows: tuple[dict[str, Any], ...]


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {label} matching {pattern} in {directory}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_task_result(plan: Mapping[str, Any], task: Mapping[str, Any]) -> DevRun:
    phase = str(plan["phase"])
    command_raw = task.get("command")
    if not isinstance(command_raw, list) or not all(isinstance(item, str) for item in command_raw):
        raise ValueError("run plan task command must be a string list")
    command = list(command_raw)
    dataset = str(task.get("dataset") or "")
    model_key = str(task.get("model_key") or "")
    model = str(task.get("model") or "")
    protocol = str(_command_value(command, "--qwen-protocol"))
    seed = int(str(_command_value(command, "--seed")))
    output_dir = Path(str(task.get("output_dir") or ""))
    if not output_dir.is_dir():
        raise FileNotFoundError(f"task output directory is missing: {output_dir}")

    result_path = _single_file(output_dir, f"{dataset}_*.jsonl", "result JSONL")
    frozen_path = _single_file(
        output_dir,
        f"frozen_inputs_{dataset}_*.json",
        "frozen input record",
    )
    frozen = _load_json(frozen_path, "frozen input record")
    manifest_path = Path(str(task.get("manifest") or ""))
    expected_manifest_hash = str(task.get("manifest_sha256") or "").lower()
    if not manifest_path.is_file() or file_sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError(f"task manifest changed or is unavailable: {manifest_path}")
    manifest_rows = _read_jsonl(manifest_path)
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    manifest_answers = {
        str(row["sample_id"]): str(row.get("answer") or "") for row in manifest_rows
    }
    rows = _read_jsonl(result_path)
    result_ids = {str(row["sample_id"]) for row in rows}
    if result_ids != manifest_ids:
        missing = sorted(manifest_ids - result_ids)
        extra = sorted(result_ids - manifest_ids)
        raise RuntimeError(
            f"incomplete task result {result_path}: missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(str(row.get("dataset")) != dataset for row in rows):
        raise RuntimeError(f"result dataset differs from run plan: {result_path}")
    for row in rows:
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError(f"scored Dev result has no answer: {result_path}")
        if answer != manifest_answers[str(row["sample_id"])]:
            raise RuntimeError(f"Dev result answer differs from frozen manifest: {result_path}")
        expected_correct = row.get("prediction") == answer
        if not isinstance(row.get("correct"), bool) or row["correct"] != expected_correct:
            raise RuntimeError(f"Dev result correctness field is inconsistent: {result_path}")
    if any(
        row.get("model") not in {None, "", model} or (not row.get("model") and not _row_failed(row))
        for row in rows
    ):
        raise RuntimeError(f"result model differs from run plan: {result_path}")
    fingerprints = {str(row.get("run_fingerprint") or "") for row in rows}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise RuntimeError(f"result run fingerprint is missing or mixed: {result_path}")
    if fingerprints != {str(frozen.get("run_fingerprint") or "")}:
        raise RuntimeError(f"result/frozen run fingerprint mismatch: {result_path}")
    if str(frozen.get("model") or "") != model:
        raise RuntimeError(f"frozen model differs from run plan: {frozen_path}")
    if str(frozen.get("model_artifact_sha256") or "") != str(
        task.get("model_artifact_sha256") or ""
    ):
        raise RuntimeError(f"frozen model artifact differs from run plan: {frozen_path}")
    if str(frozen.get("experiment_config_sha256") or "") != str(
        plan.get("config_sha256") or ""
    ):
        raise RuntimeError(f"frozen experiment config differs from run plan: {frozen_path}")
    frozen_manifest = frozen.get("manifest")
    if not isinstance(frozen_manifest, dict) or str(
        frozen_manifest.get("sha256") or ""
    ).lower() != expected_manifest_hash:
        raise RuntimeError(f"frozen manifest differs from run plan: {frozen_path}")
    if str(frozen.get("qwen_protocol") or "") != protocol:
        raise RuntimeError(f"frozen protocol differs from run plan: {frozen_path}")
    if int(frozen.get("seed", frozen.get("generation_seed", -1))) != seed:
        raise RuntimeError(f"frozen seed differs from run plan: {frozen_path}")

    mode = _command_value(command, "--baseline-mode", required=False)
    sampling = _command_value(command, "--direct-sampling", required=False)
    strategy: str | None = None
    variant_id: str | None = None
    agent_path_value = _command_value(command, "--agent-config", required=False)
    agent_config_path: str | None = None
    agent_config_hash: str | None = None
    if phase in {"agent_dev", "teacher_dev", "sft_dev"} and agent_path_value is None:
        raise ValueError(f"{phase} task has no --agent-config")
    if agent_path_value is not None and phase in {
        "agent_dev",
        "teacher_dev",
        "sft_dev",
        "final_matrix",
    }:
        agent_path = Path(agent_path_value)
        if not agent_path.is_file():
            raise FileNotFoundError(agent_path)
        agent_config_hash = file_sha256(agent_path)
        expected_agent_hash = str(task.get("agent_config_sha256") or "")
        if agent_config_hash != expected_agent_hash:
            raise RuntimeError(f"agent config changed after run plan freeze: {agent_path}")
        frozen_agent = frozen.get("agent_config")
        if not isinstance(frozen_agent, dict) or str(
            frozen_agent.get("sha256") or ""
        ) != agent_config_hash:
            raise RuntimeError(f"frozen agent config differs from run plan: {frozen_path}")
        agent_payload = _load_json(agent_path, "agent config")
        settings = agent_payload.get("agent", agent_payload)
        if not isinstance(settings, dict):
            raise ValueError(f"agent settings must be an object: {agent_path}")
        strategy = str(settings.get("strategy") or "")
        variant = agent_payload.get("search_variant")
        if not isinstance(variant, dict) or not variant.get("variant_id"):
            raise ValueError(f"materialized Dev agent config has no search variant: {agent_path}")
        variant_id = str(variant["variant_id"])
        if phase == "agent_dev":
            if plan.get("framework_filter") != strategy:
                raise RuntimeError(f"agent strategy differs from run plan: {agent_path}")
            planned_variant = plan.get("search_variant_id")
            if planned_variant != "all" and planned_variant != variant_id:
                raise RuntimeError(f"agent variant differs from run plan: {agent_path}")
        elif phase in {"teacher_dev", "sft_dev"}:
            frozen_winner = plan.get("frozen_winner")
            if not isinstance(frozen_winner, Mapping) or frozen_winner.get(
                "agent_config_sha256"
            ) != agent_config_hash:
                raise RuntimeError(
                    f"{phase} agent config differs from frozen winner: {agent_path}"
                )
        if any(
            row.get("strategy") not in {None, "", strategy}
            or (not row.get("strategy") and not _row_failed(row))
            for row in rows
        ):
            raise RuntimeError(f"result strategy differs from agent config: {result_path}")
        agent_config_path = str(agent_path.resolve())

    return DevRun(
        phase=phase,
        plan_path=str(plan["_source_path"]),
        plan_sha256=str(plan["_source_sha256"]),
        task_id=str(task.get("task_id") or ""),
        dataset=dataset,
        model_key=model_key,
        model=model,
        protocol=protocol,
        seed=seed,
        mode=str(mode) if mode is not None else None,
        sampling=str(sampling) if sampling is not None else None,
        strategy=strategy,
        variant_id=variant_id,
        agent_config_path=agent_config_path,
        agent_config_sha256=agent_config_hash,
        result_path=str(result_path.resolve()),
        result_sha256=file_sha256(result_path),
        rows=tuple(rows),
    )


def load_dev_runs(
    plan_paths: Iterable[Path],
    experiment_config_sha256: str,
) -> tuple[DevRun, ...]:
    runs: list[DevRun] = []
    seen_plans: set[str] = set()
    seen_tasks: set[tuple[str, str, str]] = set()
    for path in plan_paths:
        plan = load_frozen_run_plan(path, experiment_config_sha256)
        plan_hash = str(plan["plan_sha256"])
        if plan_hash in seen_plans:
            raise ValueError(f"duplicate run plan supplied: {path}")
        seen_plans.add(plan_hash)
        for task in plan["tasks"]:
            if not isinstance(task, dict):
                raise ValueError(f"run plan task is not an object: {path}")
            identity = (
                str(plan["phase"]),
                str(task.get("task_id") or ""),
                str(task.get("dataset") or ""),
            )
            if identity in seen_tasks:
                raise ValueError(f"duplicate logical Dev task across plans: {identity}")
            seen_tasks.add(identity)
            runs.append(load_task_result(plan, task))
    return tuple(runs)


def load_protocol_smoke_rejection(
    plan_path: Path,
    experiment_config_sha256: str,
    *,
    model_key: str,
    protocol: str,
    failure_rate_threshold: float = 0.01,
) -> dict[str, Any]:
    """Validate explicit smoke failures that justify skipping a full protocol audit."""
    plan = load_frozen_run_plan(plan_path, experiment_config_sha256)
    if plan.get("phase") != "protocol_smoke":
        raise ValueError("protocol smoke rejection requires a protocol_smoke plan")

    matching_tasks: dict[str, Mapping[str, Any]] = {}
    for task in plan["tasks"]:
        if not isinstance(task, Mapping):
            raise ValueError(f"run plan task is not an object: {plan_path}")
        command_raw = task.get("command")
        if not isinstance(command_raw, list) or not all(
            isinstance(item, str) for item in command_raw
        ):
            raise ValueError("run plan task command must be a string list")
        command = list(command_raw)
        if str(task.get("model_key") or "") != model_key:
            continue
        if _command_value(command, "--qwen-protocol") != protocol:
            continue
        if _command_value(command, "--baseline-mode") != "direct":
            raise ValueError("protocol smoke rejection requires Direct smoke results")
        if _command_value(command, "--direct-sampling") != "uniform64":
            raise ValueError("protocol smoke rejection requires uniform64 smoke results")
        dataset = str(task.get("dataset") or "")
        if dataset in matching_tasks:
            raise ValueError(f"duplicate smoke task for {model_key}:{protocol}:{dataset}")
        matching_tasks[dataset] = task

    if set(matching_tasks) != set(DATASETS):
        raise RuntimeError(
            f"smoke plan must declare all datasets for {model_key}:{protocol}"
        )

    expected_rows = 0
    observed_rows = 0
    failures = 0
    sources: list[dict[str, Any]] = []
    for dataset in DATASETS:
        task = matching_tasks[dataset]
        manifest_path = Path(str(task.get("manifest") or ""))
        expected_manifest_hash = str(task.get("manifest_sha256") or "").lower()
        if not manifest_path.is_file() or file_sha256(manifest_path) != expected_manifest_hash:
            raise RuntimeError(f"smoke manifest changed or is unavailable: {manifest_path}")
        manifest_rows = _read_jsonl(manifest_path)
        manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
        expected_rows += len(manifest_rows)

        output_dir = Path(str(task.get("output_dir") or ""))
        result_paths = sorted(output_dir.glob(f"{dataset}_*.jsonl")) if output_dir.is_dir() else []
        if not result_paths:
            continue
        if len(result_paths) != 1:
            raise ValueError(
                f"expected at most one smoke result for {dataset}, found {len(result_paths)}"
            )
        result_path = result_paths[0]
        rows = _read_jsonl(result_path)
        result_ids = {str(row["sample_id"]) for row in rows}
        if not result_ids <= manifest_ids:
            raise RuntimeError(f"smoke result contains samples outside its manifest: {result_path}")
        if any(str(row.get("dataset") or "") != dataset for row in rows):
            raise RuntimeError(f"smoke result dataset differs from run plan: {result_path}")
        expected_model = str(task.get("model") or "")
        if any(
            row.get("model") not in {None, "", expected_model}
            or (not row.get("model") and not _row_failed(row))
            for row in rows
        ):
            raise RuntimeError(f"smoke result model differs from run plan: {result_path}")
        frozen_paths = sorted(output_dir.glob(f"frozen_inputs_{dataset}_*.json"))
        if len(frozen_paths) != 1:
            raise ValueError(
                f"expected one frozen smoke input for {dataset}, found {len(frozen_paths)}"
            )
        frozen = _load_json(frozen_paths[0], "frozen smoke input")
        fingerprints = {str(row.get("run_fingerprint") or "") for row in rows}
        if len(fingerprints) != 1 or "" in fingerprints:
            raise RuntimeError(f"smoke result run fingerprint is missing or mixed: {result_path}")
        if fingerprints != {str(frozen.get("run_fingerprint") or "")}:
            raise RuntimeError(f"smoke result/frozen fingerprint mismatch: {result_path}")
        if str(frozen.get("experiment_config_sha256") or "") != experiment_config_sha256:
            raise RuntimeError(f"frozen smoke config differs from run plan: {frozen_paths[0]}")
        if str(frozen.get("model") or "") != expected_model:
            raise RuntimeError(f"frozen smoke model differs from run plan: {frozen_paths[0]}")
        if str(frozen.get("qwen_protocol") or "") != protocol:
            raise RuntimeError(f"frozen smoke protocol differs from run plan: {frozen_paths[0]}")
        frozen_manifest = frozen.get("manifest")
        if not isinstance(frozen_manifest, Mapping) or str(
            frozen_manifest.get("sha256") or ""
        ).lower() != expected_manifest_hash:
            raise RuntimeError(f"frozen smoke manifest differs from run plan: {frozen_paths[0]}")
        observed_rows += len(rows)
        failures += sum(_row_failed(row) for row in rows)
        sources.append(
            {
                "dataset": dataset,
                "result": str(result_path.resolve()),
                "sha256": file_sha256(result_path),
                "rows": len(rows),
            }
        )

    failure_rate = failures / expected_rows if expected_rows else 0.0
    if failure_rate <= failure_rate_threshold:
        raise RuntimeError(
            f"{model_key}:{protocol} explicit smoke failure rate {failure_rate:.6f} "
            f"does not exceed {failure_rate_threshold:.6f}"
        )
    return {
        "model_key": model_key,
        "protocol": protocol,
        "reason": SMOKE_PROTOCOL_REJECTION_REASON,
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "missing_rows": expected_rows - observed_rows,
        "failures": failures,
        "failure_rate": failure_rate,
        "threshold": failure_rate_threshold,
        "source_run_plan": str(plan_path.resolve()),
        "source_run_plan_sha256": file_sha256(plan_path),
        "sources": sources,
    }


def _row_failed(row: Mapping[str, Any]) -> bool:
    direct_media_accounting_failed = False
    if row.get("baseline_mode") == "direct":
        estimated_frames = row.get("sampled_frames_estimated")
        actual_frames = row.get("sampled_frames_actual")
        direct_media_accounting_failed = bool(
            not isinstance(estimated_frames, int)
            or isinstance(estimated_frames, bool)
            or not isinstance(actual_frames, int)
            or isinstance(actual_frames, bool)
            or actual_frames != estimated_frames
            or row.get("visual_usage_complete") is not True
        )
    request_trace = row.get("request_trace")
    agent_length_truncation = isinstance(request_trace, list) and any(
        isinstance(item, Mapping) and item.get("finish_reason") == "length"
        for item in request_trace
    )
    is_agent = bool(row.get("strategy"))
    visual_tokens = row.get("visual_tokens")
    agent_visual_accounting_failed = is_agent and bool(
        row.get("visual_token_accounting_complete") is not True
        or not isinstance(visual_tokens, (int, float))
        or isinstance(visual_tokens, bool)
        or not math.isfinite(float(visual_tokens))
        or float(visual_tokens) < 0
    )
    return bool(
        row.get("error")
        or row.get("error_type")
        or row.get("parse_error")
        or row.get("model_parse_failure")
        or row.get("data_unavailable")
        or row.get("control_unavailable")
        or row.get("prediction") is None
        or row.get("finish_reason") == "length"
        or agent_length_truncation
        or agent_visual_accounting_failed
        or row.get("branch_failures")
        or direct_media_accounting_failed
    )


def _row_had_length_truncation(row: Mapping[str, Any]) -> bool:
    if bool(row.get("length_retry_used")):
        return True
    attempts = row.get("request_attempts")
    baseline_length = isinstance(attempts, list) and any(
        isinstance(attempt, Mapping) and attempt.get("finish_reason") == "length"
        for attempt in attempts
    )
    request_trace = row.get("request_trace")
    agent_length = isinstance(request_trace, list) and any(
        isinstance(attempt, Mapping) and attempt.get("finish_reason") == "length"
        for attempt in request_trace
    )
    return baseline_length or agent_length


@dataclass
class Point:
    point_id: str
    summary: dict[str, Any]
    rows: dict[str, Mapping[str, Any]]


def _make_point(point_id: str, runs: Sequence[DevRun]) -> Point:
    by_dataset: dict[str, DevRun] = {}
    for run in runs:
        if run.dataset in by_dataset:
            raise ValueError(f"duplicate dataset run in point {point_id}: {run.dataset}")
        by_dataset[run.dataset] = run
    missing_datasets = sorted(set(DATASETS) - set(by_dataset))
    rows: dict[str, Mapping[str, Any]] = {}
    dataset_accuracies: list[float] = []
    for dataset in DATASETS:
        run = by_dataset.get(dataset)
        if run is None:
            continue
        correct = 0
        for row in run.rows:
            key = f"{dataset}:{row['sample_id']}"
            if key in rows:
                raise ValueError(f"duplicate composite sample in point {point_id}: {key}")
            rows[key] = row
            correct += bool(row.get("correct"))
        dataset_accuracies.append(correct / len(run.rows) if run.rows else 0.0)

    failures = sum(_row_failed(row) for row in rows.values())
    leak_failures = sum(
        row.get("annotation_leak_check") == "failed"
        or row.get("failure_class") == "annotation_leak"
        for row in rows.values()
    )
    leak_audit_unknown = sum(
        row.get("annotation_leak_check") not in {"passed", "not_run"}
        for row in rows.values()
    )
    candidate_reruns = sum(int(row.get("candidate_rerun") or 0) for row in rows.values())
    token_values = [row.get("total_tokens") for row in rows.values()]
    token_accounting_complete = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
        for value in token_values
    )
    failure_rate = failures / len(rows) if rows else 1.0
    agent_rows = [row for row in rows.values() if row.get("strategy")]
    visual_token_accounting_complete = all(
        row.get("visual_token_accounting_complete") is True
        and isinstance(row.get("visual_tokens"), (int, float))
        and not isinstance(row.get("visual_tokens"), bool)
        and math.isfinite(float(row["visual_tokens"]))
        and float(row["visual_tokens"]) >= 0
        for row in agent_rows
    )
    rejection_reasons: list[str] = []
    if missing_datasets:
        rejection_reasons.append("missing_datasets")
    if failure_rate > 0.01:
        rejection_reasons.append("failure_rate_above_1_percent")
    if leak_failures:
        rejection_reasons.append("annotation_leak")
    if leak_audit_unknown:
        rejection_reasons.append("annotation_leak_audit_unknown")
    if candidate_reruns:
        rejection_reasons.append("candidate_rerun_nonzero")
    if not token_accounting_complete:
        rejection_reasons.append("total_token_accounting_incomplete")
    if agent_rows and not visual_token_accounting_complete:
        rejection_reasons.append("visual_token_accounting_incomplete")
    correct = sum(bool(row.get("correct")) for row in rows.values())
    summary = {
        "point_id": point_id,
        "datasets": sorted(by_dataset),
        "missing_datasets": missing_datasets,
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "accuracy_stdev_across_datasets": (
            statistics.pstdev(dataset_accuracies)
            if len(dataset_accuracies) > 1
            else 0.0
        ),
        "failures": failures,
        "failure_rate": failure_rate,
        "annotation_leak": leak_failures,
        "annotation_leak_audit_unknown": leak_audit_unknown,
        "candidate_rerun": candidate_reruns,
        "mean_total_tokens": (
            statistics.fmean(float(value) for value in token_values)
            if token_accounting_complete and token_values
            else None
        ),
        "token_accounting_complete": token_accounting_complete,
        "visual_token_accounting_complete": visual_token_accounting_complete,
        "eligible": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "sources": [
            {
                "dataset": run.dataset,
                "task_id": run.task_id,
                "run_plan": run.plan_path,
                "run_plan_sha256": run.plan_sha256,
                "result": run.result_path,
                "result_sha256": run.result_sha256,
            }
            for run in sorted(runs, key=lambda item: item.dataset)
        ],
    }
    return Point(point_id, summary, rows)


def _regressed(baseline: Point, candidate: Point) -> int:
    if set(baseline.rows) != set(candidate.rows):
        raise RuntimeError("paired points do not contain the same Dev samples")
    return sum(
        bool(baseline.rows[key].get("correct"))
        and not bool(candidate.rows[key].get("correct"))
        for key in baseline.rows
    )


def _point_rank(point: Point) -> tuple[float, float, int, float, str]:
    summary = point.summary
    mean_total_tokens = summary.get("mean_total_tokens")
    return (
        -float(summary.get("accuracy") or 0.0),
        float(summary.get("accuracy_stdev") or summary.get("accuracy_stdev_across_datasets") or 0.0),
        int(summary.get("regressed") or 0),
        float(mean_total_tokens) if mean_total_tokens is not None else math.inf,
        point.point_id,
    )


def _variant_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    search = config["agent_search"]
    declared = search.get("variants")
    if isinstance(declared, list):
        values = tuple(
            str(item.get("id") or "")
            for item in declared
            if isinstance(item, Mapping)
        )
        if len(values) != len(declared) or any(not value for value in values):
            raise ValueError("agent_search.variants contains an invalid id")
        if len(values) != len(set(values)):
            raise ValueError("agent_search.variants contains duplicate ids")
        return values
    values: list[str] = []
    for overview, fps, intervals, turns in product(
        search["overview_frames"],
        search["local_fps"],
        search["max_intervals"],
        search["max_turns"],
    ):
        fps_id = str(float(fps)).replace(".", "p")
        values.append(
            f"ov{int(overview):03d}_fps{fps_id}_int{int(intervals):02d}_turn{int(turns):02d}"
        )
    return tuple(values)


def _select_protocols(
    config: Mapping[str, Any],
    runs: Sequence[DevRun],
    blocking: list[str],
    protocol_rejections: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Point]]:
    selections: dict[str, Any] = {}
    selected_points: dict[str, Point] = {}
    rejections = protocol_rejections or {}
    protocol_runs = [run for run in runs if run.phase == "protocol_audit"]
    for model_key in config["models"]:
        points: dict[str, Point] = {}
        rejected_points: dict[str, dict[str, Any]] = {}
        for protocol in config["protocols"]:
            rejection = (rejections.get(model_key) or {}).get(protocol)
            group = [
                run
                for run in protocol_runs
                if run.model_key == model_key
                and run.protocol == protocol
                and run.mode == "direct"
                and run.sampling == "uniform64"
            ]
            if not group:
                if rejection is not None:
                    rejected_points[protocol] = {
                        "point_id": f"{model_key}:{protocol}:smoke_rejected",
                        "eligible": False,
                        "rejection_reasons": [str(rejection["reason"])],
                        "rejection_evidence": dict(rejection),
                    }
                    continue
                blocking.append(f"missing_protocol_audit:{model_key}:{protocol}")
                continue
            points[protocol] = _make_point(f"{model_key}:{protocol}:uniform64", group)
            if rejection is not None:
                points[protocol].summary["eligible"] = False
                reason = str(rejection["reason"])
                if reason not in points[protocol].summary["rejection_reasons"]:
                    points[protocol].summary["rejection_reasons"].append(reason)
                points[protocol].summary["rejection_evidence"] = dict(rejection)
            length_truncations = sum(
                _row_had_length_truncation(row)
                for row in points[protocol].rows.values()
            )
            points[protocol].summary["initial_length_truncations"] = length_truncations
            if protocol == "think" and length_truncations:
                reason = "frozen_32768_length_truncation"
                points[protocol].summary["eligible"] = False
                if reason not in points[protocol].summary["rejection_reasons"]:
                    points[protocol].summary["rejection_reasons"].append(reason)
                blocking.append(
                    f"protocol_frozen_32768_length_truncation:{model_key}"
                )
            if points[protocol].summary["missing_datasets"]:
                blocking.append(f"incomplete_protocol_audit:{model_key}:{protocol}")
        reference = points.get("no_think")
        if reference is not None:
            for point in points.values():
                point.summary["regressed"] = _regressed(reference, point)
        eligible = [point for point in points.values() if point.summary["eligible"]]
        if not eligible:
            blocking.append(f"no_eligible_protocol:{model_key}")
            selections[model_key] = {
                "selected": None,
                "points": [
                    *(points[key].summary for key in sorted(points)),
                    *(rejected_points[key] for key in sorted(rejected_points)),
                ],
            }
            continue
        selected = min(eligible, key=_point_rank)
        selected_points[model_key] = selected
        selections[model_key] = {
            "selected": selected.point_id,
            "protocol": selected.point_id.split(":")[1],
            "selection_order": "accuracy,stability,regressions,total_tokens",
            "points": [
                *(points[key].summary for key in sorted(points)),
                *(rejected_points[key] for key in sorted(rejected_points)),
            ],
        }
    return selections, selected_points


def _select_direct(
    config: Mapping[str, Any],
    runs: Sequence[DevRun],
    protocol_selection: Mapping[str, Any],
    blocking: list[str],
) -> tuple[dict[str, Any], dict[str, Point]]:
    selections: dict[str, Any] = {}
    selected_points: dict[str, Point] = {}
    direct_runs = [run for run in runs if run.phase == "direct_dev"]
    expected_sampling = [str(item["id"]) for item in config["direct_sampling"]]
    for model_key in config["models"]:
        protocol = (protocol_selection.get(model_key) or {}).get("protocol")
        if not protocol:
            continue
        points: dict[str, Point] = {}
        for sampling in expected_sampling:
            group = [
                run
                for run in direct_runs
                if run.model_key == model_key
                and run.protocol == protocol
                and run.mode == "direct"
                and run.sampling == sampling
            ]
            if not group:
                blocking.append(f"missing_direct:{model_key}:{protocol}:{sampling}")
                continue
            points[sampling] = _make_point(f"{model_key}:{protocol}:{sampling}", group)
            if points[sampling].summary["missing_datasets"]:
                blocking.append(f"incomplete_direct:{model_key}:{protocol}:{sampling}")
        reference = points.get("uniform64")
        if reference is not None:
            for point in points.values():
                point.summary["regressed"] = _regressed(reference, point)
        eligible = [point for point in points.values() if point.summary["eligible"]]
        if not eligible:
            blocking.append(f"no_eligible_direct:{model_key}:{protocol}")
            selections[model_key] = {"selected": None, "points": [p.summary for p in points.values()]}
            continue
        selected = min(eligible, key=_point_rank)
        selected_points[model_key] = selected
        selections[model_key] = {
            "selected": selected.point_id,
            "protocol": protocol,
            "sampling": selected.point_id.split(":")[2],
            "selection_order": "accuracy,stability,regressions,total_tokens",
            "points": [points[key].summary for key in sorted(points)],
        }
    return selections, selected_points


def _agent_point(
    point_id: str,
    runs_by_seed: Mapping[int, Sequence[DevRun]],
    incumbent_by_seed: Mapping[int, Point],
) -> Point:
    per_seed: dict[int, Point] = {
        seed: _make_point(f"{point_id}:seed{seed}", runs)
        for seed, runs in sorted(runs_by_seed.items())
    }
    missing_seeds = sorted(set(REQUIRED_AGENT_SEEDS) - set(per_seed))
    rows: dict[str, Mapping[str, Any]] = {}
    correct_by_seed: dict[int, int] = {}
    regressions = 0
    all_eligible = not missing_seeds
    for seed in REQUIRED_AGENT_SEEDS:
        point = per_seed.get(seed)
        if point is None:
            continue
        correct_by_seed[seed] = int(point.summary["correct"])
        regressions += _regressed(incumbent_by_seed[seed], point)
        all_eligible = all_eligible and bool(point.summary["eligible"])
        rows.update({f"seed{seed}:{key}": value for key, value in point.rows.items()})
    accuracies = [float(per_seed[seed].summary["accuracy"]) for seed in per_seed]
    totals = [float(per_seed[seed].summary["mean_total_tokens"]) for seed in per_seed if per_seed[seed].summary["mean_total_tokens"] is not None]
    failures = sum(int(point.summary["failures"]) for point in per_seed.values())
    total = sum(int(point.summary["total"]) for point in per_seed.values())
    rejection_reasons = [
        reason
        for point in per_seed.values()
        for reason in point.summary["rejection_reasons"]
    ]
    if missing_seeds:
        rejection_reasons.append("missing_seeds")
    summary = {
        "point_id": point_id,
        "seeds": sorted(per_seed),
        "missing_seeds": missing_seeds,
        "correct_by_seed": correct_by_seed,
        "mean_correct": statistics.fmean(correct_by_seed.values()) if correct_by_seed else None,
        "mean_accuracy": statistics.fmean(accuracies) if accuracies else None,
        "accuracy": statistics.fmean(accuracies) if accuracies else None,
        "accuracy_stdev": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "regressed": regressions,
        "mean_total_tokens": statistics.fmean(totals) if len(totals) == len(per_seed) and totals else None,
        "total": total,
        "failures": failures,
        "failure_rate": failures / total if total else 1.0,
        "annotation_leak": sum(int(point.summary["annotation_leak"]) for point in per_seed.values()),
        "candidate_rerun": sum(int(point.summary["candidate_rerun"]) for point in per_seed.values()),
        "eligible": all_eligible,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "per_seed": {str(seed): point.summary for seed, point in per_seed.items()},
    }
    return Point(point_id, summary, rows)


def _select_agents(
    config: Mapping[str, Any],
    runs: Sequence[DevRun],
    teacher_model_key: str,
    protocol: str,
    direct: Point,
    blocking: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    configured_stages = [
        value for value in config["agent_search"]["framework_order"] if value in SUPPORTED_STAGES
    ]
    if tuple(configured_stages) != SUPPORTED_STAGES:
        blocking.append("agent_framework_order_is_not_strict_A0_through_A4")
    configured_seeds = tuple(int(value) for value in config["agent_search"]["seeds"])
    if configured_seeds != REQUIRED_AGENT_SEEDS:
        blocking.append("agent_seeds_are_not_exactly_17_42_73")
    expected_variants = _variant_ids(config)
    agent_runs = [
        run
        for run in runs
        if run.phase == "agent_dev"
        and run.model_key == teacher_model_key
        and run.protocol == protocol
    ]
    unexpected_stages = sorted(
        {str(run.strategy) for run in agent_runs if run.strategy not in SUPPORTED_STAGES}
    )
    if unexpected_stages:
        blocking.append(f"unexpected_agent_stages:{','.join(unexpected_stages)}")
    incumbent_by_seed = {seed: direct for seed in REQUIRED_AGENT_SEEDS}
    incumbent_descriptor: dict[str, Any] = {
        "kind": "direct",
        "point_id": direct.point_id,
        "correct_by_seed": {
            seed: int(direct.summary["correct"]) for seed in REQUIRED_AGENT_SEEDS
        },
    }
    stages: list[dict[str, Any]] = []

    for stage in SUPPORTED_STAGES:
        stage_runs = [run for run in agent_runs if run.strategy == stage]
        by_variant_and_hash: dict[tuple[str, str], list[DevRun]] = {}
        for run in stage_runs:
            by_variant_and_hash.setdefault(
                (str(run.variant_id), str(run.agent_config_sha256)), []
            ).append(run)
        seen_variants = {variant for variant, _digest in by_variant_and_hash}
        missing_variants = sorted(set(expected_variants) - seen_variants)
        extra_variants = sorted(seen_variants - set(expected_variants))
        if not stage_runs:
            blocking.append(f"missing_agent_stage:{stage}")
        if missing_variants:
            blocking.append(f"missing_agent_variants:{stage}:{len(missing_variants)}")
        if extra_variants:
            blocking.append(f"unexpected_agent_variants:{stage}:{len(extra_variants)}")
        variant_hash_counts: dict[str, int] = {}
        for variant, _digest in by_variant_and_hash:
            variant_hash_counts[variant] = variant_hash_counts.get(variant, 0) + 1
        ambiguous = sorted(
            variant for variant, count in variant_hash_counts.items() if count != 1
        )
        if ambiguous:
            blocking.append(f"ambiguous_agent_config_hash:{stage}:{ambiguous[0]}")

        candidates: list[Point] = []
        candidate_metadata: dict[str, tuple[str, str, str]] = {}
        baseline_correct = {
            seed: int(incumbent_by_seed[seed].summary["correct"])
            for seed in REQUIRED_AGENT_SEEDS
        }
        for (variant, digest), group in sorted(by_variant_and_hash.items()):
            if variant not in expected_variants or variant in ambiguous:
                continue
            by_seed: dict[int, list[DevRun]] = {}
            for run in group:
                by_seed.setdefault(run.seed, []).append(run)
            observed_seeds = set(by_seed)
            if observed_seeds != set(REQUIRED_AGENT_SEEDS):
                blocking.append(
                    f"incomplete_agent_seeds:{stage}:{variant}:"
                    f"{sorted(observed_seeds)}"
                )
            for seed, seed_runs in by_seed.items():
                observed_datasets = {run.dataset for run in seed_runs}
                if observed_datasets != set(DATASETS):
                    blocking.append(
                        f"incomplete_agent_datasets:{stage}:{variant}:seed{seed}"
                    )
            point_id = f"{teacher_model_key}:{protocol}:{stage}:{variant}:{digest[:12]}"
            point = _agent_point(point_id, by_seed, incumbent_by_seed)
            first = group[0]
            point.summary.update(
                {
                    "agent_config_path": str(first.agent_config_path),
                    "agent_config_sha256": str(first.agent_config_sha256),
                    "strategy": stage,
                    "variant_id": variant,
                }
            )
            if set(point.summary["correct_by_seed"]) == set(REQUIRED_AGENT_SEEDS):
                decision = promotion_decision(
                    baseline_correct,
                    point.summary["correct_by_seed"],
                    minimum_mean_gain=2.0,
                    minimum_seed_wins=2,
                )
                point.summary["promotion_gate"] = {
                    "accepted": decision.accepted,
                    "seed_gains": decision.seed_gains,
                    "mean_gain": decision.mean_gain,
                    "seed_wins": decision.seed_wins,
                    "reason": decision.reason,
                }
            else:
                point.summary["promotion_gate"] = {
                    "accepted": False,
                    "reason": "incomplete_seed_set",
                }
            candidates.append(point)
            candidate_metadata[point_id] = (
                str(first.agent_config_path),
                str(first.agent_config_sha256),
                variant,
            )

        promoted = [
            point
            for point in candidates
            if point.summary["eligible"]
            and point.summary["promotion_gate"].get("accepted") is True
        ]
        selected = min(promoted, key=_point_rank) if promoted else None
        stage_report = {
            "stage": stage,
            "incumbent_before": incumbent_descriptor,
            "expected_variant_count": len(expected_variants),
            "observed_variant_count": len(seen_variants),
            "missing_variants": missing_variants,
            "extra_variants": extra_variants,
            "selected": selected.point_id if selected is not None else None,
            "accepted": selected is not None,
            "candidates": [point.summary for point in sorted(candidates, key=lambda value: value.point_id)],
        }
        if selected is not None:
            config_path, config_hash, variant = candidate_metadata[selected.point_id]
            selected.summary["agent_config_path"] = config_path
            selected.summary["agent_config_sha256"] = config_hash
            selected.summary["strategy"] = stage
            selected.summary["variant_id"] = variant
            incumbent_by_seed = {
                seed: _make_point(
                    f"{selected.point_id}:incumbent:seed{seed}",
                    [run for run in by_variant_and_hash[(variant, config_hash)] if run.seed == seed],
                )
                for seed in REQUIRED_AGENT_SEEDS
            }
            incumbent_descriptor = {
                "kind": "agent",
                "point_id": selected.point_id,
                "strategy": stage,
                "variant_id": variant,
                "agent_config": {"path": config_path, "sha256": config_hash},
                "correct_by_seed": selected.summary["correct_by_seed"],
            }
        stage_report["incumbent_after"] = incumbent_descriptor
        stages.append(stage_report)

    winner = incumbent_descriptor if incumbent_descriptor["kind"] == "agent" else None
    if winner is None:
        blocking.append("no_agent_cleared_the_direct_promotion_gate")
    return {
        "teacher_model_key": teacher_model_key,
        "protocol": protocol,
        "direct_incumbent": direct.summary,
        "required_seeds": list(REQUIRED_AGENT_SEEDS),
        "strict_stage_order": list(SUPPORTED_STAGES),
        "stages": stages,
        "final_incumbent": incumbent_descriptor,
    }, winner


def build_dev_selection_report(
    config: Mapping[str, Any],
    runs: Sequence[DevRun],
    *,
    teacher_model_key: str = "q9",
    protocol_rejections: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    experiment_hash = canonical_sha256(config)
    blocking: list[str] = []
    protocol_selection, _protocol_points = _select_protocols(
        config, runs, blocking, protocol_rejections
    )
    direct_selection, direct_points = _select_direct(
        config, runs, protocol_selection, blocking
    )
    if teacher_model_key not in config.get("models", {}):
        raise ValueError(f"unknown teacher model key: {teacher_model_key}")
    teacher_protocol = (protocol_selection.get(teacher_model_key) or {}).get("protocol")
    teacher_direct = direct_points.get(teacher_model_key)
    agent_selection: dict[str, Any] | None = None
    winner: dict[str, Any] | None = None
    if teacher_protocol and teacher_direct is not None:
        agent_selection, winner = _select_agents(
            config,
            runs,
            teacher_model_key,
            teacher_protocol,
            teacher_direct,
            blocking,
        )
    else:
        blocking.append("teacher_protocol_or_direct_not_selected")

    source_plans = sorted(
        {
            (run.plan_path, run.plan_sha256)
            for run in runs
        }
    )
    winner_summary = None
    if winner is not None:
        winner_summary = {
            "winner_id": (
                f"{winner['strategy']}_{winner['variant_id']}_{teacher_model_key}_{teacher_protocol}"
            ),
            "model_key": teacher_model_key,
            "protocol": teacher_protocol,
            "seed": int(config["execution"]["sample_seed"]),
            "strategy": winner["strategy"],
            "variant_id": winner["variant_id"],
            "agent_config": winner["agent_config"],
        }
        if winner_summary["seed"] not in REQUIRED_AGENT_SEEDS:
            blocking.append("frozen_winner_seed_is_not_one_of_17_42_73")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if not blocking and winner_summary is not None else "blocked",
        "experiment_config_sha256": experiment_hash,
        "policy": {
            "protocol_and_direct_selected_per_model": True,
            "agent_teacher_model_key": teacher_model_key,
            "promotion_minimum_mean_correct_gain": 2.0,
            "promotion_minimum_seed_wins": 2,
            "failure_rate_max": 0.01,
            "annotation_leak_max": 0,
            "tie_break_order": [
                "accuracy",
                "accuracy_stdev",
                "regressed",
                "mean_total_tokens",
            ],
        },
        "source_run_plans": [
            {"path": path, "sha256": digest} for path, digest in source_plans
        ],
        "protocol_selection": protocol_selection,
        "direct_selection": direct_selection,
        "agent_selection": agent_selection,
        "winner": winner_summary,
        "blocking_errors": sorted(set(blocking)),
    }
    if protocol_rejections:
        report["protocol_rejections"] = protocol_rejections
    report["selection_state_sha256"] = canonical_sha256(report)
    return report


def write_frozen_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return file_sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return file_sha256(path)


def build_frozen_winner(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    report_sha256: str,
) -> dict[str, Any]:
    if report.get("status") != "passed" or not isinstance(report.get("winner"), dict):
        raise ValueError("cannot freeze a winner from a blocked selection report")
    winner = dict(report["winner"])
    return {
        "schema_version": 2,
        **winner,
        "experiment_config_sha256": report["experiment_config_sha256"],
        "selection_state_sha256": report["selection_state_sha256"],
        "selection_report": {
            "path": str(report_path.resolve()),
            "sha256": report_sha256,
        },
        "source_run_plans": report["source_run_plans"],
        "strict_stage_order": list(SUPPORTED_STAGES),
    }

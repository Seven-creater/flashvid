#!/usr/bin/env python3
"""Run the repair-only public explicit-time Perception-Memory rescue lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.perception_memory_eva import PerceptionMemoryEvaEvaluator
from flashvid_eval.privacy import assert_annotation_free_request
from flashvid_eval.runner import parse_question_time_range
from flashvid_eval.schemas import ModelSample


_PUBLIC_SAMPLE_FIELDS = {"dataset", "sample_id", "video", "question", "choices"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _validate_frozen_scope(
    scope_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> str:
    try:
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("frozen scope is not valid JSON") from exc
    if not isinstance(scope, Mapping):
        raise ValueError("frozen scope must be an object")
    artifacts = scope.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("frozen scope requires artifacts")
    artifact = artifacts.get(manifest_path.name)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            f"frozen scope does not bind artifact {manifest_path.name}"
        )
    if str(artifact.get("sha256") or "").lower() != manifest_sha256:
        raise ValueError("frozen scope explicit-time manifest SHA-256 mismatch")
    return _file_sha256(scope_path)


def _validate_manifest_row(
    row: Mapping[str, Any],
) -> tuple[ModelSample, tuple[tuple[float, float], ...]]:
    public = row.get("public_sample")
    source = row.get("source_row")
    if not isinstance(public, Mapping) or set(public) != _PUBLIC_SAMPLE_FIELDS:
        raise ValueError("repair row requires exact annotation-free public_sample")
    if not isinstance(source, Mapping):
        raise ValueError("repair row requires source_row")
    assert_annotation_free_request({"public_sample": public})
    for field in ("dataset", "sample_id"):
        if str(row.get(field)) != str(public.get(field)):
            raise ValueError(f"repair identity differs from public_sample.{field}")
        if str(source.get(field)) != str(public.get(field)):
            raise ValueError(f"source identity differs from public_sample.{field}")
    source_trajectory_id = str(row.get("source_trajectory_id") or "")
    if not source_trajectory_id or source_trajectory_id != str(
        source.get("trajectory_id") or ""
    ):
        raise ValueError("repair source_trajectory_id mismatch")
    expected_source_sha = str(row.get("source_row_sha256") or "").lower()
    if expected_source_sha != _canonical_sha256(source):
        raise ValueError("repair source_row_sha256 mismatch")
    if source.get("scoring_deferred") is not True:
        raise ValueError("repair source must keep scoring deferred")

    question = str(public.get("question") or "")
    parsed = parse_question_time_range(question)
    recorded_parsed = row.get("parsed_time_range")
    if parsed is None or recorded_parsed != list(parsed):
        raise ValueError("repair parsed_time_range differs from public question")
    raw_intervals = row.get("source_requested_intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError("repair row requires source_requested_intervals")
    intervals: list[tuple[float, float]] = []
    for raw in raw_intervals:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or not isinstance(raw[0], (int, float))
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], (int, float))
        ):
            raise ValueError("source_requested_intervals must be numeric pairs")
        intervals.append((float(raw[0]), float(raw[1])))

    choices = public.get("choices")
    if not isinstance(choices, Mapping) or not choices:
        raise ValueError("public_sample.choices must be a non-empty object")
    normalized_choices = {str(key): str(value) for key, value in choices.items()}
    candidate = source.get("candidate_answer")
    candidate_answer = (
        str(candidate) if str(candidate) in normalized_choices else None
    )
    return (
        ModelSample(
            dataset=str(public["dataset"]),
            sample_id=str(public["sample_id"]),
            video=str(public["video"]),
            question=question,
            choices=normalized_choices,
            candidate_answer=candidate_answer,
        ),
        tuple(intervals),
    )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_one(
    row: Mapping[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    video_root: Path,
    frame_root: Path,
    timeout: float,
    local_media_paths: bool,
    max_turns: int,
    seed: int,
    manifest_sha256: str,
    frozen_scope_sha256: str,
) -> dict[str, Any]:
    sample, source_intervals = _validate_manifest_row(row)
    source = row["source_row"]
    assert isinstance(source, Mapping)
    evaluator = PerceptionMemoryEvaEvaluator(
        OpenAICompatibleClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            local_file_urls_as_paths=local_media_paths,
        ),
        model,
        video_root,
        frame_root,
        max_turns=max_turns,
        max_frames_per_call=96,
        seed=seed,
        candidate_results_sha256=source.get("candidate_results_sha256"),
        model_artifact_sha256=source.get("model_artifact_sha256"),
        manifest_sha256=source.get("manifest_sha256"),
        experiment_config_sha256=source.get("experiment_config_sha256"),
        diagnostics_gate_sha256=source.get("diagnostics_gate_sha256"),
        scoring_deferred=True,
        train600_manifest_sha256=source.get("train600_manifest_sha256"),
        trajectory_schedule_id="repair-explicit-time-v1",
        trajectory_variant_id="rescue_explicit_time",
        trajectory_replica_id=0,
    )
    result = evaluator.run(
        sample,
        rescue_source_requested_intervals=source_intervals,
    )
    result.update(
        {
            "source_trajectory_id": row["source_trajectory_id"],
            "source_row_sha256": row["source_row_sha256"],
            "source_file_sha256": row.get("source_file_sha256"),
            "base_file_sha256": row.get("base_file_sha256"),
            "rescue_manifest_sha256": manifest_sha256,
            "frozen_scope_sha256": frozen_scope_sha256,
            "repair_reason": row.get("reason"),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--frozen-scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--base-url", action="append")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--expected-count", type=int, default=6)
    parser.add_argument("--local-media-paths", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = _load_jsonl(args.repair_manifest)
        if len(rows) != args.expected_count:
            raise ValueError(
                f"repair manifest must contain {args.expected_count} rows, got {len(rows)}"
            )
        manifest_sha256 = _file_sha256(args.repair_manifest)
        frozen_scope_sha256 = _validate_frozen_scope(
            args.frozen_scope, args.repair_manifest, manifest_sha256
        )
        identities = [str(row.get("source_trajectory_id") or "") for row in rows]
        if not all(identities) or len(identities) != len(set(identities)):
            raise ValueError("repair manifest requires unique source_trajectory_id values")
        for row in rows:
            _validate_manifest_row(row)

        existing: dict[str, dict[str, Any]] = {}
        if args.output.exists() and args.resume:
            for row in _load_jsonl(args.output):
                identity = str(row.get("source_trajectory_id") or "")
                if identity in existing:
                    raise ValueError("resume output contains duplicate source trajectory")
                if row.get("rescue_manifest_sha256") != manifest_sha256:
                    raise ValueError("resume output uses a different repair manifest")
                if row.get("frozen_scope_sha256") != frozen_scope_sha256:
                    raise ValueError("resume output uses a different frozen scope")
                existing[identity] = row
        elif args.output.exists():
            raise ValueError("output exists; use --resume to preserve completed rows")

        base_urls = args.base_url or ["http://127.0.0.1:8200/v1"]
        pending = [row for row in rows if row["source_trajectory_id"] not in existing]
        with ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(pending) or 1)
        ) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    row,
                    base_url=base_urls[index % len(base_urls)],
                    api_key=args.api_key,
                    model=args.model,
                    video_root=args.video_root,
                    frame_root=args.frame_root,
                    timeout=args.timeout,
                    local_media_paths=args.local_media_paths,
                    max_turns=args.max_turns,
                    seed=args.seed,
                    manifest_sha256=manifest_sha256,
                    frozen_scope_sha256=frozen_scope_sha256,
                ): row
                for index, row in enumerate(pending)
            }
            for future in as_completed(futures):
                result = future.result()
                existing[str(result["source_trajectory_id"])] = result
                ordered = [existing[item] for item in identities if item in existing]
                _write_jsonl_atomic(args.output, ordered)

        output_rows = [existing[item] for item in identities]
        errors = sum(row.get("error") is not None for row in output_rows)
        summary = {
            "status": "passed" if errors == 0 else "failed",
            "rows": len(output_rows),
            "errors": errors,
            "rescue_manifest_sha256": manifest_sha256,
            "frozen_scope_sha256": frozen_scope_sha256,
            "output": str(args.output.resolve()),
        }
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        summary = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

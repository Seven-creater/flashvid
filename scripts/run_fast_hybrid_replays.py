#!/usr/bin/env python3
"""Replay dependency-ready Fast Hybrid compression specs with official EVA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.fast_hybrid_eva import (
    OFFICIAL_EVA_COMMIT,
    FastHybridEvaEvaluator,
)
from flashvid_eval.privacy import assert_deferred_result_public
from flashvid_eval.qwen_sft import canonical_sha256, read_jsonl, sha256_file
from flashvid_eval.runner import frozen_candidate_costs
from flashvid_eval.schemas import ModelSample


REPLAY_SEEDS = (17, 42, 73)


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a SHA-256")
    return normalized


def load_public_samples(path: Path, dataset: str) -> tuple[dict[str, ModelSample], str]:
    samples: dict[str, ModelSample] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        if str(row.get("dataset") or "").lower() != dataset:
            raise ValueError(f"manifest row {line_number} has a different dataset")
        sample_id = str(row.get("sample_id") or "").strip()
        choices = row.get("choices")
        if not sample_id or sample_id in samples or not isinstance(choices, dict):
            raise ValueError(f"manifest row {line_number} is invalid or duplicated")
        samples[sample_id] = ModelSample(
            dataset=dataset,
            sample_id=sample_id,
            video=str(row.get("video") or ""),
            question=str(row.get("question") or ""),
            choices={str(key): str(value) for key, value in choices.items()},
            candidate_answer=None,
        )
    if not samples:
        raise ValueError("manifest is empty")
    return samples, sha256_file(path)


def load_candidates(
    path: Path,
    samples: Mapping[str, ModelSample],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], str]:
    answers: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in samples or sample_id in records:
            raise ValueError(f"candidate row {line_number} is outside/duplicates the manifest")
        request = row.get("protocol_request") or {}
        if (
            row.get("baseline_mode") != "direct"
            or row.get("sampling_id") != "uniform32"
            or row.get("enable_thinking") is not False
            or request.get("max_tokens") != 512
            or float(request.get("temperature", -1.0)) != 0.0
        ):
            raise ValueError(f"candidate row {line_number} violates the frozen protocol")
        prediction = str(row.get("prediction") or "").strip().upper()
        if prediction not in samples[sample_id].option_letters:
            raise ValueError(f"candidate row {line_number} has no valid prediction")
        if row.get("error"):
            raise ValueError(f"candidate row {line_number} contains an unresolved error")
        answers[sample_id] = prediction
        records[sample_id] = dict(row)
    if set(records) != set(samples):
        missing = sorted(set(samples) - set(records))
        raise ValueError(f"candidate file is incomplete: {missing[:3]}")
    return answers, records, sha256_file(path)


def load_ready_specs(
    path: Path,
    *,
    dataset: str,
    train600_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    fingerprints: set[str] = set()
    for line_number, row in enumerate(read_jsonl(path), 1):
        if str(row.get("dataset") or "").lower() != dataset:
            continue
        if row.get("train600_manifest_sha256") != train600_sha256:
            raise ValueError(f"spec row {line_number} Train600 hash mismatch")
        if row.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError(f"spec row {line_number} dataset hash mismatch")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"spec row {line_number} config hash mismatch")
        fingerprint = str(row.get("counterfactual_fingerprint") or "")
        expected = canonical_sha256(
            {key: value for key, value in row.items() if key != "counterfactual_fingerprint"}
        )
        if fingerprint != expected or fingerprint in fingerprints:
            raise ValueError(f"spec row {line_number} has an invalid/duplicate fingerprint")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError("ready replay batch may contain only one DAG node per sample")
        planned_calls = row.get("planned_calls")
        replica_ids = row.get("replica_trajectory_ids")
        if not isinstance(planned_calls, list) or not planned_calls:
            raise ValueError(f"spec row {line_number} has no planned calls")
        if not isinstance(replica_ids, list) or len(replica_ids) != 3:
            raise ValueError(f"spec row {line_number} must freeze exactly three replicas")
        sample_ids.add(sample_id)
        fingerprints.add(fingerprint)
        specs.append(dict(row))
    if not specs:
        raise ValueError(f"no {dataset} ready replay specs")
    return specs


def _cost_fields(
    result: dict[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_cost = frozen_candidate_costs(candidate)
    candidate_usage = candidate_cost["usage"]
    agent_usage = {
        key: int((result.get("usage") or {}).get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    candidate_visual = candidate_cost["visual_tokens"]
    agent_visual_value = result.get("visual_tokens")
    agent_visual_complete = result.get("visual_usage_complete") is True
    if (
        isinstance(agent_visual_value, bool)
        or not isinstance(agent_visual_value, (int, float))
        or not math.isfinite(float(agent_visual_value))
        or float(agent_visual_value) < 0
    ):
        agent_visual = None
        agent_visual_complete = False
    else:
        agent_visual = int(agent_visual_value)
    end_to_end_visual_complete = bool(
        candidate_cost["visual_complete"] and agent_visual_complete
    )
    end_to_end_visual = (
        int(candidate_visual) + int(agent_visual)
        if end_to_end_visual_complete
        else None
    )
    candidate_latency = float(
        candidate.get("latency_s", candidate.get("elapsed_s", 0.0)) or 0.0
    )
    return {
        "candidate_usage": candidate_usage,
        "candidate_visual_tokens": candidate_visual,
        "candidate_latency_s": candidate_latency,
        "candidate_usage_complete": candidate_cost["usage_complete"],
        "candidate_visual_tokens_complete": candidate_cost["visual_complete"],
        "candidate_cost_complete": candidate_cost["complete"],
        "agent_usage": agent_usage,
        "agent_visual_tokens": agent_visual,
        "agent_total_tokens_complete": False,
        "agent_visual_tokens_complete": agent_visual_complete,
        "end_to_end_prompt_tokens": candidate_usage["prompt_tokens"]
        + agent_usage["prompt_tokens"],
        "end_to_end_completion_tokens": candidate_usage["completion_tokens"]
        + agent_usage["completion_tokens"],
        "end_to_end_total_tokens": candidate_usage["total_tokens"]
        + agent_usage["total_tokens"],
        "end_to_end_total_tokens_complete": False,
        "end_to_end_visual_tokens": end_to_end_visual,
        "end_to_end_visual_tokens_complete": end_to_end_visual_complete,
        "end_to_end_latency_s": candidate_latency
        + float(result.get("latency_s", 0.0) or 0.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_sha256 = _require_sha256(args.config_sha256, "config-sha256")
    model_sha256 = _require_sha256(args.model_artifact_sha256, "model-artifact-sha256")
    train600_sha256 = _require_sha256(args.train600_manifest_sha256, "Train600 SHA-256")
    expected_manifest = _require_sha256(
        args.expected_manifest_sha256, "expected-manifest-sha256"
    )
    samples, manifest_sha256 = load_public_samples(args.manifest, args.dataset)
    if manifest_sha256 != expected_manifest:
        raise RuntimeError("dataset manifest SHA-256 mismatch")
    candidates, candidate_records, candidate_sha256 = load_candidates(
        args.candidate_results, samples
    )
    specs = load_ready_specs(
        args.specs,
        dataset=args.dataset,
        train600_sha256=train600_sha256,
        dataset_manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    unknown = sorted({str(spec["sample_id"]) for spec in specs} - set(samples))
    if unknown:
        raise ValueError(f"replay specs contain unknown samples: {unknown[:3]}")

    source = {
        "dataset": args.dataset,
        "manifest_sha256": manifest_sha256,
        "train600_manifest_sha256": train600_sha256,
        "candidate_results_sha256": candidate_sha256,
        "specs_sha256": sha256_file(args.specs),
        "config_sha256": config_sha256,
        "model": args.model,
        "model_artifact_sha256": model_sha256,
        "official_eva_commit": OFFICIAL_EVA_COMMIT,
        "judge_seeds": list(REPLAY_SEEDS),
        "judge_temperature": args.temperature,
    }
    run_fingerprint = canonical_sha256(
        {
            **source,
            "implementation_sha256": sha256_file(Path(__file__)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output.with_suffix(args.output.suffix + ".frozen.json")
    frozen = {"schema_version": 1, **source, "run_fingerprint": run_fingerprint}
    if frozen_path.is_file():
        if json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("frozen replay inputs changed")
    else:
        frozen_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    jobs = [
        (spec, replica_id, seed)
        for spec in specs
        for replica_id, seed in enumerate(REPLAY_SEEDS)
    ]
    cached: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        for row in read_jsonl(args.output):
            if row.get("counterfactual_run_fingerprint") != run_fingerprint:
                raise RuntimeError("replay resume fingerprint mismatch")
            trajectory_id = str(row.get("trajectory_id") or "")
            if not trajectory_id or trajectory_id in cached:
                raise RuntimeError("replay resume has a missing/duplicate trajectory_id")
            cached[trajectory_id] = row

    client = OpenAICompatibleClient(args.base_url, args.api_key, args.timeout)

    def execute(spec: Mapping[str, Any], replica_id: int, seed: int) -> dict[str, Any]:
        trajectory_id = str(spec["replica_trajectory_ids"][replica_id])
        sample_id = str(spec["sample_id"])
        evaluator = FastHybridEvaEvaluator(
            client,
            args.model,
            args.video_root,
            args.frame_root,
            version="fast_hybrid_v2",
            max_turns=max(args.max_turns, len(spec["planned_calls"]) + 1),
            max_call_visual_tokens=args.max_call_visual_tokens,
            max_total_visual_tokens=args.max_total_visual_tokens,
            candidate_results_sha256=candidate_sha256,
            teacher_model_sha256=model_sha256,
            experiment_config_sha256=config_sha256,
            scoring_deferred=True,
            teacher_temperature=args.temperature,
            generation_seed=seed,
            trajectory_context={
                "experiment_config_sha256": config_sha256,
                "model_artifact_sha256": model_sha256,
                "manifest_sha256": manifest_sha256,
                "train600_manifest_sha256": train600_sha256,
                "trajectory_schedule_id": str(spec["schedule_id"]),
                "trajectory_variant_id": str(spec["variant_id"]),
                "trajectory_replica_id": replica_id,
            },
        )
        try:
            result = evaluator.replay_fixed_evidence(
                samples[sample_id], candidates[sample_id], list(spec["planned_calls"])
            )
        except Exception as error:
            result = {
                "backend": "fast_hybrid_eva_replay",
                "prediction": None,
                "final_prediction": None,
                "candidate_answer": candidates[sample_id],
                "candidate_rerun": 0,
                "fallback_to_candidate": False,
                "usage": {},
                "visual_tokens": 0,
                "latency_s": 0.0,
                "tool_calls": [],
                "request_trace": [],
                "annotation_leak_check": "passed",
                "error": f"{type(error).__name__}: {error}",
            }
        result.update(
            {
                "dataset": args.dataset,
                "sample_id": sample_id,
                "video": samples[sample_id].video,
                "scoring_deferred": True,
                "trajectory_id": trajectory_id,
                "phase": "compression",
                "schedule_id": spec["schedule_id"],
                "variant_id": spec["variant_id"],
                "family_id": spec["family_id"],
                "replica_id": str(replica_id),
                "judge_seed": seed,
                "base_trajectory_id": spec["base_trajectory_id"],
                "counterfactual_fingerprint": spec["counterfactual_fingerprint"],
                "counterfactual_run_fingerprint": run_fingerprint,
                "controller_fingerprint": spec.get("controller_fingerprint"),
                "manifest_sha256": train600_sha256,
                "train600_manifest_sha256": train600_sha256,
                "dataset_manifest_sha256": manifest_sha256,
                "config_sha256": config_sha256,
                **_cost_fields(result, candidate_records[sample_id]),
            }
        )
        assert_deferred_result_public(result)
        return result

    expected_ids = {
        str(spec["replica_trajectory_ids"][replica]): (spec, replica, seed)
        for spec, replica, seed in jobs
    }
    extras = sorted(set(cached) - set(expected_ids))
    if extras:
        raise RuntimeError(f"replay resume contains unexpected trajectories: {extras[:3]}")
    pending = [job for identity, job in expected_ids.items() if identity not in cached]
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(execute, *job) for job in pending]
            for future in as_completed(futures):
                row = future.result()
                cached[str(row["trajectory_id"])] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
    ordered = [cached[identity] for identity in sorted(expected_ids)]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    errors = sum(bool(row.get("error")) for row in ordered)
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "specs": len(specs),
        "replicas": 3,
        "completed": len(ordered),
        "errors": errors,
        "failure_rate": errors / len(ordered),
        "output": str(args.output.resolve()),
        "run_fingerprint": run_fingerprint,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lvbench", "lsdbench", "cgbench"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train600-manifest-sha256", required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--api-key", default="no")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-call-visual-tokens", type=int, default=24000)
    parser.add_argument("--max-total-visual-tokens", type=int, default=48000)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

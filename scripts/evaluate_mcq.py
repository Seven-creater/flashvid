from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.baseline_diagnostics import DEFAULT_DURATION_BUCKET_EDGES_S
from flashvid_eval.datasets import VideoIndex, load_samples
from flashvid_eval.flashvid_budget import BudgetEndpointPool
from flashvid_eval.flashvid_hybrid import (
    FlashVIDHybridConfig,
    FlashVIDHybridEvaluator,
    evaluate_flashvid_trajectories,
)
from flashvid_eval.fast_hybrid_eva import FastHybridEvaEvaluator, OFFICIAL_EVA_COMMIT
from flashvid_eval.perception_memory_eva import (
    RESCUE_TRAJECTORY_VARIANTS,
    PerceptionMemoryEvaEvaluator,
)
from flashvid_eval.offline_budget import normalize_candidate
from flashvid_eval.answers import extract_strict_answer_letter
from flashvid_eval.qwen_evaluation import (
    QwenBaselineConfig,
    QwenBaselineRunner,
    direct_sampling_spec,
    evaluate_qwen_runner,
)
from flashvid_eval.qwen_protocol import QWEN_PROTOCOLS
from flashvid_eval.qwen_agents import InferenceProtocol as AgentInferenceProtocol
from flashvid_eval.qwen_agents import build_strategy as build_qwen_agent_strategy
from flashvid_eval.qwen_trajectories import (
    QwenTrajectoryRunner,
    TrajectoryGenerationConfig,
)
from flashvid_eval.runner import Evaluator, available_samples, evaluate, select_manifest
from flashvid_eval.schemas import Sample


_CANDIDATE_NORMALIZER_INSTRUCTION = (
    "Normalize the answer from a frozen model response. Do not solve the "
    "question and do not infer missing visual evidence. Match only an answer "
    "explicitly stated in the response to one of the option letters or exact "
    "option texts. Output exactly Answer: X, or NONE if it is not recoverable."
)


def _read_manifest(path: Path) -> list[Sample]:
    return [
        Sample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_candidate_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id"))
        if sample_id in answers:
            raise ValueError(f"duplicate frozen candidate sample_id: {sample_id}")
        prediction = str(record.get("prediction") or record.get("final_prediction") or "").strip().upper()
        if prediction:
            answers[sample_id] = prediction
    return answers


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validated_sha256(value: str | None, label: str) -> str:
    if value is None or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be 64 hexadecimal characters")
    return value.lower()


def _validate_perception_memory_diagnostics_gate(path: Path) -> str:
    """Fail closed unless the current Fast-Hybrid Test300 audit is complete."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Perception-Memory diagnostics summary must be an object")
    if payload.get("status") != "passed" or payload.get("scope_passed") is not True:
        raise ValueError("Perception-Memory paired badcase audit did not pass")
    if payload.get("paired_samples", payload.get("samples")) != 300:
        raise ValueError("Perception-Memory paired badcase audit must contain 300 samples")
    datasets = payload.get("datasets")
    expected_datasets = {"lvbench", "lsdbench", "cgbench"}
    if not isinstance(datasets, dict) or set(datasets) != expected_datasets:
        raise ValueError("Perception-Memory paired badcase audit has the wrong dataset scope")
    if any(
        not isinstance(datasets[name], dict) or datasets[name].get("samples") != 100
        for name in expected_datasets
    ):
        raise ValueError("Perception-Memory paired badcase audit requires 100 samples per dataset")
    required_flips = {
        "untrained_correct_sft_wrong",
        "untrained_wrong_sft_correct",
        "both_correct",
        "both_wrong",
    }
    required_failures = {
        "localization",
        "visual_fact_extraction",
        "cross_interval_memory",
        "incomplete_evidence_early_stop",
        "judging",
        "candidate_gate",
        "engineering",
    }
    if set(payload.get("flip_totals") or {}) != required_flips:
        raise ValueError("Perception-Memory paired badcase audit has incomplete flip groups")
    if set(payload.get("failure_mode_totals") or {}) != required_failures:
        raise ValueError("Perception-Memory paired badcase audit has incomplete failure taxonomy")
    if (
        payload.get("taxonomy_coverage_passed") is not True
        or payload.get("taxonomy_classified") != payload.get("taxonomy_required")
    ):
        raise ValueError("Perception-Memory paired badcase audit has unclassified failures")
    required_funnel = {
        "target_hit",
        "evidence_state_valid",
        "evidence_complete",
        "judge_correct",
    }
    if any(
        set((datasets[name].get("funnel") or {})) != required_funnel
        for name in expected_datasets
    ):
        raise ValueError("Perception-Memory paired badcase audit has an incomplete funnel")
    return _file_sha256(path)


def _validate_perception_memory_variant(value: str) -> str:
    """Accept only the base schedule or a pre-registered duration-only rescue."""

    variant = str(value).strip()
    allowed = {"base", *RESCUE_TRAJECTORY_VARIANTS}
    if variant not in allowed:
        raise ValueError(
            "perception_memory_eva trajectory-variant-id must be one of: "
            + ", ".join(sorted(allowed))
        )
    return variant


def _candidate_superset_allowed(
    backend: str, defer_scoring: bool, trajectory_variant_id: str
) -> bool:
    """Allow a frozen Train200 candidate file for a label-free rescue subset."""

    return bool(
        backend == "perception_memory_eva"
        and defer_scoring
        and trajectory_variant_id in RESCUE_TRAJECTORY_VARIANTS
    )


def _local_media_transport(backend: str, enabled: bool) -> str:
    """Resolve the explicit Transformers local-media compatibility mode."""

    if enabled and backend != "perception_memory_eva":
        raise ValueError(
            "--local-media-paths is only valid with perception_memory_eva"
        )
    return "path" if enabled else "file_url"


def _stable_model_slug(model: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"
    identity = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:48]}-{identity}"


def _json_duration_bucket_edges() -> list[float | str]:
    return [
        "inf" if value == float("inf") else float(value)
        for value in DEFAULT_DURATION_BUCKET_EDGES_S
    ]


def _load_validated_mismatch_map(
    path: Path,
    *,
    samples: list[Sample],
    manifest_hash: str,
    video_root: Path,
) -> tuple[dict[str, str], str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("mapping"), dict):
        raise ValueError(
            "mismatched-video-map must be a structured object with a mapping field"
        )
    if str(payload.get("manifest_sha256", "")).lower() != manifest_hash.lower():
        raise ValueError("mismatched-video-map manifest SHA-256 does not match this run")
    if payload.get("duration_bucket_edges_s") != _json_duration_bucket_edges():
        raise ValueError("mismatched-video-map duration bucket edges do not match")
    mapped_root = payload.get("video_root")
    if mapped_root is not None and Path(str(mapped_root)).resolve() != video_root.resolve():
        raise ValueError("mismatched-video-map video_root does not match this run")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    raw_mapping = payload["mapping"]
    mapping = {str(sample_id): str(video) for sample_id, video in raw_mapping.items()}
    unknown = sorted(set(mapping) - set(sample_by_id))
    if unknown:
        raise ValueError(
            "mismatched-video-map contains IDs outside the manifest: "
            + ", ".join(unknown[:5])
        )
    if any(not video for video in mapping.values()):
        raise ValueError("mismatched-video-map contains an empty target video")

    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("mismatched-video-map must include auditable assignments")
    assignment_mapping: dict[str, str] = {}
    assignment_rows: list[tuple[str, str, str, int]] = []
    bucket_count = len(DEFAULT_DURATION_BUCKET_EDGES_S) - 1
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("mismatched-video-map assignment must be an object")
        sample_id = str(assignment.get("sample_id", ""))
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"invalid mismatch assignment sample_id: {sample_id}")
        if sample_id in assignment_mapping:
            raise ValueError(f"duplicate mismatch assignment sample_id: {sample_id}")
        source = str(assignment.get("source_video", ""))
        target = str(assignment.get("target_video", ""))
        bucket = assignment.get("duration_bucket")
        if str(assignment.get("dataset", "")) != sample.dataset:
            raise ValueError(f"mismatch assignment dataset differs for {sample_id}")
        if source != sample.video:
            raise ValueError(f"mismatch assignment source differs for {sample_id}")
        if not target or target == source:
            raise ValueError(f"mismatch assignment is not a wrong video for {sample_id}")
        if isinstance(bucket, bool) or not isinstance(bucket, int) or not 0 <= bucket < bucket_count:
            raise ValueError(f"invalid mismatch duration bucket for {sample_id}")
        assignment_mapping[sample_id] = target
        assignment_rows.append((sample.dataset, source, target, bucket))
    if assignment_mapping != mapping:
        raise ValueError("mismatched-video-map mapping and assignments disagree")
    source_buckets = {
        (dataset, source): bucket
        for dataset, source, _target, bucket in assignment_rows
    }
    for dataset, _source, target, bucket in assignment_rows:
        target_bucket = source_buckets.get((dataset, target))
        if target_bucket is None:
            raise ValueError(
                "mismatch target is not an audited source video in the same dataset"
            )
        if target_bucket != bucket:
            raise ValueError("mismatch target crosses an explicit duration bucket")
    return mapping, _file_sha256(path), payload


def _write_frozen_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(f"frozen input changed during resume: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _default_candidate_normalization_cache(
    candidate_results: Path,
    dataset: str,
    model: str,
) -> Path:
    """Return a content-addressed cache shared by smoke, fixed, and trajectory runs."""

    source_hash = _file_sha256(candidate_results)
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return (
        candidate_results.parent
        / ".candidate_normalization"
        / f"{dataset}_{source_hash[:16]}_{model_hash}.jsonl"
    )


def _normalize_frozen_candidates_unlocked(
    path: Path,
    samples: list[Sample],
    client: OpenAICompatibleClient,
    model: str,
    cache_path: Path,
    *,
    read_only: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int, str]:
    """Recover candidate letters without rerunning Direct video inference."""

    source_hash = _file_sha256(path)
    direct_records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id"))
        if sample_id in direct_records:
            raise ValueError(f"duplicate frozen candidate sample_id: {sample_id}")
        direct_records[sample_id] = record
    missing = [sample.sample_id for sample in samples if sample.sample_id not in direct_records]
    if missing:
        raise ValueError(
            f"frozen candidate file is missing {len(missing)} manifest samples; "
            f"first missing sample_id={missing[0]}"
        )

    cached: dict[str, dict] = {}
    legacy_cached_ids: set[str] = set()
    normalizer_prompt_hash = hashlib.sha256(
        _CANDIDATE_NORMALIZER_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("candidate_results_sha256") != source_hash:
                raise RuntimeError(
                    f"candidate normalization cache source changed: {cache_path}"
                )
            sample_id = str(record.get("sample_id"))
            if sample_id in cached:
                raise RuntimeError(
                    f"duplicate candidate normalization cache sample_id: {sample_id}"
                )
            legacy_record = all(
                record.get(field) is None
                for field in (
                    "choices_sha256",
                    "normalizer_model",
                    "normalizer_prompt_sha256",
                )
            )
            if legacy_record:
                if not read_only or int(record.get("direct_rerun", -1)) != 0:
                    raise RuntimeError(
                        f"legacy candidate normalization cache is not safe for reuse: "
                        f"{cache_path}"
                    )
                legacy_cached_ids.add(sample_id)
            elif not read_only and record.get("normalizer_model") != model:
                raise RuntimeError(
                    f"candidate normalization model changed: {cache_path}"
                )
            if (
                not legacy_record
                and record.get("normalizer_prompt_sha256") != normalizer_prompt_hash
            ):
                raise RuntimeError(
                    f"candidate normalization prompt changed: {cache_path}"
                )
            cached[sample_id] = record

    answers: dict[str, str] = {}
    sources: dict[str, str] = {}
    reasons: dict[str, str] = {}
    model_calls = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        direct = direct_records[sample.sample_id]
        record_dataset = direct.get("dataset")
        if record_dataset is not None and str(record_dataset) != sample.dataset:
            raise ValueError(
                f"candidate dataset mismatch for {sample.sample_id}: {record_dataset}"
            )
        record_video = direct.get("video")
        if record_video is not None:
            normalized_record_video = str(record_video).replace("\\", "/")
            normalized_sample_video = str(sample.video).replace("\\", "/")
            if normalized_record_video != normalized_sample_video:
                raise ValueError(
                    f"candidate video mismatch for {sample.sample_id}: "
                    f"{record_video} != {sample.video}"
                )
        choices_hash = hashlib.sha256(
            json.dumps(
                sample.choices,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cached_record = cached.get(sample.sample_id)
        cache_matches = cached_record is not None and (
            cached_record.get("choices_sha256") == choices_hash
            or sample.sample_id in legacy_cached_ids
        )
        if cache_matches:
            cached_dataset = cached_record.get("dataset")
            if cached_dataset is not None and str(cached_dataset) != sample.dataset:
                raise RuntimeError(
                    f"candidate normalization cache dataset changed for "
                    f"{sample.sample_id}: {cache_path}"
                )
            answer = cached_record.get("candidate_answer")
            source = str(cached_record.get("candidate_source") or "none")
            reason = str(cached_record.get("normalization_reason") or "")
        else:
            if read_only:
                state = "missing" if cached_record is None else "choice hash mismatch"
                raise RuntimeError(
                    f"read-only candidate normalization cache is {state} for "
                    f"{sample.sample_id}: {cache_path}"
                )
            prediction = (
                direct.get("candidate_answer")
                or direct.get("prediction")
                or direct.get("final_prediction")
            )
            raw = direct.get("raw_response") or direct.get("candidate_raw_response") or ""
            normalized = normalize_candidate(prediction, raw, sample.choices)
            answer = normalized.answer
            source = normalized.source
            reason = normalized.reason
            if answer is None and str(raw).strip():
                prompt = (
                    _CANDIDATE_NORMALIZER_INSTRUCTION
                    + "\n\n"
                    "Options:\n"
                    + "\n".join(
                        f"{letter}: {text}" for letter, text in sample.choices.items()
                    )
                    + f"\n\nFrozen response:\n{str(raw)[:4000]}"
                )
                result = client.chat(
                    model,
                    [{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=0.0,
                )
                model_calls += 1
                parsed = extract_strict_answer_letter(
                    result.content,
                    sample.option_letters,
                )
                if parsed is not None:
                    answer = parsed
                    source = "normalized"
                    reason = "text_normalizer"
                else:
                    reason = f"{reason};text_normalizer_failed".strip(";")
            row = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "candidate_answer": answer,
                "candidate_source": source,
                "normalization_reason": reason,
                "candidate_results_sha256": source_hash,
                "choices_sha256": choices_hash,
                "normalizer_model": model,
                "normalizer_prompt_sha256": normalizer_prompt_hash,
                "direct_rerun": 0,
            }
            cached[sample.sample_id] = row
            with cache_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
        if answer in sample.option_letters:
            answers[sample.sample_id] = str(answer)
        else:
            answer = None
            source = "none"
        sources[sample.sample_id] = source
        reasons[sample.sample_id] = reason
    if not read_only:
        temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
        temporary.write_text(
            "".join(
                json.dumps(cached[sample_id], ensure_ascii=False) + "\n"
                for sample_id in sorted(cached)
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    scope_payload = [
        {
            "sample_id": sample.sample_id,
            "answer": answers.get(sample.sample_id),
            "source": sources[sample.sample_id],
        }
        for sample in samples
    ]
    scope_hash = hashlib.sha256(
        json.dumps(
            scope_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return answers, sources, reasons, model_calls, scope_hash


def _normalize_frozen_candidates(
    path: Path,
    samples: list[Sample],
    client: OpenAICompatibleClient,
    model: str,
    cache_path: Path,
    *,
    read_only: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int, str]:
    if read_only:
        return _normalize_frozen_candidates_unlocked(
            path,
            samples,
            client,
            model,
            cache_path,
            read_only=True,
        )
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            return _normalize_frozen_candidates_unlocked(
                path,
                samples,
                client,
                model,
                cache_path,
                read_only=read_only,
            )
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified MCQ evaluator for long-video datasets.")
    parser.add_argument("--dataset", choices=("cgbench", "lvbench", "lsdbench"), required=True)
    parser.add_argument(
        "--backend",
        choices=(
            "direct",
            "text_only",
            "agent",
            "hybrid",
            "hybrid_frozen",
            "fast_hybrid_eva",
            "perception_memory_eva",
            "flashvid_hybrid",
            "qwen_baseline",
            "qwen_agent",
        ),
        required=True,
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="no")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--manifest", type=Path, help="Use this already-frozen manifest verbatim.")
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-frames-per-call", type=int, default=128)
    parser.add_argument("--max-call-visual-tokens", type=int, default=4000)
    parser.add_argument("--max-total-visual-tokens", type=int, default=8000)
    parser.add_argument(
        "--agent-version",
        choices=(
            "v2a",
            "v2b",
            "v2c",
            "v2d",
            "hybrid_v1",
            "hybrid_v2",
            "hybrid_v3a",
            "hybrid_v3b",
            "hybrid_v3c",
            "hybrid_v3d",
            "hybrid_v3e",
            "hybrid_v3f",
            "hybrid_v3g",
            "fast_hybrid_v1",
            "fast_hybrid_v2",
            "perception_memory_v1",
            "flashvid_budget_v1",
        ),
        default="v2a",
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--local-media-paths",
        action="store_true",
        help=(
            "Send local file:// frame URLs as absolute paths on the wire for "
            "Transformers serve. Valid only for perception_memory_eva; the "
            "default keeps file:// URLs for other OpenAI-compatible servers."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--qwen-protocol",
        choices=tuple(QWEN_PROTOCOLS),
        default="no_think",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=(
            "question_choices",
            "choices_only",
            "permuted_choices",
            "direct",
            "mismatched_video",
        ),
    )
    parser.add_argument(
        "--direct-sampling",
        choices=("uniform32", "uniform64", "uniform128", "fps2"),
    )
    parser.add_argument(
        "--option-permutation-seed",
        type=int,
        choices=(17, 42, 73),
    )
    parser.add_argument(
        "--mismatched-video-map",
        type=Path,
        help="Frozen JSON object mapping sample_id to a wrong video in the same duration bucket.",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        help="Frozen Qwen-only Agent strategy configuration.",
    )
    parser.add_argument(
        "--expected-agent-config-sha256",
        help="Expected SHA-256 of --agent-config from the frozen run plan.",
    )
    parser.add_argument(
        "--defer-scoring",
        action="store_true",
        help=(
            "Do not join or serialize labels. Valid only for Qwen-only Train600 "
            "trajectory generation."
        ),
    )
    parser.add_argument(
        "--diagnostics-gate-summary",
        type=Path,
        help=(
            "Passed 300-sample paired badcase audit. Required by "
            "perception_memory_eva before any model request."
        ),
    )
    parser.add_argument(
        "--trajectory-schedule-id",
        help="Stable schedule identity for a label-free Qwen trajectory run.",
    )
    parser.add_argument(
        "--train600-manifest-sha256",
        help=(
            "SHA-256 of the frozen merged Train600 artifact. Required with "
            "--defer-scoring and recorded separately from the active Train200 hash."
        ),
    )
    parser.add_argument("--trajectory-replica-id", type=int, default=0)
    parser.add_argument(
        "--trajectory-variant-id",
        default="base",
        help=(
            "Trajectory family variant. Perception-Memory accepts base plus its four "
            "pre-registered duration-only rescue coverage variants."
        ),
    )
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument(
        "--candidate-normalization-cache",
        type=Path,
        help=(
            "Content-addressed, append-safe normalized candidate cache shared "
            "across smoke, fixed-budget, trajectory, and checkpoint runs."
        ),
    )
    parser.add_argument(
        "--candidate-normalization-read-only",
        action="store_true",
        help=(
            "Require every manifest candidate to exist in the supplied cache; "
            "never invoke the current controller to normalize missing entries."
        ),
    )
    parser.add_argument("--controller-base-url")
    parser.add_argument("--controller-model")
    parser.add_argument(
        "--controller-config-override",
        help=(
            "Required audit label when a perception-bank config is reused "
            "with a different text-only controller (for example an SFT checkpoint)."
        ),
    )
    parser.add_argument("--perception-endpoints", type=Path)
    parser.add_argument("--expected-endpoint-config-sha256")
    parser.add_argument(
        "--budget-policy",
        choices=("fixed", "random", "model"),
        default="model",
    )
    parser.add_argument(
        "--budget-strategy",
        choices=(
            "fixed_r010",
            "fixed_r025",
            "fixed_r050",
            "fixed_r100",
            "model_requested",
            "random_uniform",
            "random_matched",
            "route_rule",
            "uncertainty_escalation",
        ),
        help=(
            "Training-free execution policy. When supplied it overrides only "
            "the retention ratio, leaving the controller's interval plan intact."
        ),
    )
    parser.add_argument(
        "--budget-random-seed",
        type=int,
        choices=(17, 42, 73),
    )
    parser.add_argument(
        "--budget-match-distribution",
        help=(
            "Canonical JSON object with exact keys 0.10/0.25/0.50/1.00. "
            "Used only by random_matched."
        ),
    )
    parser.add_argument(
        "--controller-prompt-id",
        choices=("legacy_v1", "budget_rubric_v1", "budget_escalation_v1"),
        default="legacy_v1",
    )
    parser.add_argument(
        "--experiment-config-sha256",
        help="SHA-256 of the immutable outer sweep configuration.",
    )
    parser.add_argument(
        "--model-artifact-sha256",
        help=(
            "SHA-256 identity of the served model artifact (for example the "
            "frozen model index/checksum manifest). Required by qwen_baseline."
        ),
    )
    parser.add_argument(
        "--teacher-model-artifact-sha256",
        help=(
            "SHA-256 of the frozen base/Teacher model. For an SFT LoRA run, "
            "--model-artifact-sha256 identifies the served base+adapter stack "
            "while this value continues to identify the unchanged base model."
        ),
    )
    parser.add_argument(
        "--server-max-model-len",
        type=int,
        default=131072,
        help="Served context limit used to cap a length-retry request.",
    )
    parser.add_argument(
        "--fixed-retention-ratio",
        type=float,
        choices=(0.10, 0.25, 0.50, 1.00),
        default=0.50,
    )
    parser.add_argument(
        "--perception-cache-root",
        type=Path,
        default=Path("cache/flashvid_perception"),
    )
    parser.add_argument(
        "--perception-media-root",
        type=Path,
        default=Path("/dev/shm/flashvid_perception"),
    )
    parser.add_argument("--max-perception-calls", type=int, default=5)
    parser.add_argument(
        "--controller-temperature",
        "--teacher-temperature",
        dest="controller_temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument("--trajectories-per-sample", type=int, default=1)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--perception-temperature", type=float, default=0.0)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Build a new manifest only from videos currently present under --video-root.",
    )
    parser.add_argument(
        "--wait-for-available",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Poll partial downloads until --sample accessible items exist.",
    )
    args = parser.parse_args()
    local_media_transport = _local_media_transport(
        args.backend, args.local_media_paths
    )
    if args.defer_scoring and args.backend not in {
        "qwen_agent",
        "fast_hybrid_eva",
        "perception_memory_eva",
    }:
        raise ValueError(
            "--defer-scoring is supported only by qwen_agent, fast_hybrid_eva, "
            "and perception_memory_eva"
        )
    if args.perception_temperature != 0.0:
        raise ValueError("FlashVID perception temperature is frozen at 0")
    if args.experiment_config_sha256 and (
        len(args.experiment_config_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in args.experiment_config_sha256
        )
    ):
        raise ValueError("experiment-config-sha256 must be 64 hexadecimal characters")
    budget_match_distribution = None
    if args.budget_match_distribution is not None:
        try:
            budget_match_distribution = json.loads(
                args.budget_match_distribution
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "budget-match-distribution must be valid JSON"
            ) from exc
        if not isinstance(budget_match_distribution, dict):
            raise ValueError("budget-match-distribution must be a JSON object")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_root = args.frame_root or args.video_root / ".flashvid_eval_frames"
    manifest = args.manifest or args.output_dir / f"{args.dataset}_manifest_{args.seed}_{args.sample}.jsonl"
    if args.manifest:
        samples = _read_manifest(manifest)
    elif manifest.exists():
        samples = _read_manifest(manifest)
    else:
        all_samples = load_samples(args.dataset, args.annotations)
        while True:
            candidates = available_samples(all_samples, args.video_root) if args.available_only else all_samples
            if len(candidates) >= args.sample or not args.wait_for_available:
                break
            print(
                f"waiting for accessible {args.dataset} videos: "
                f"{len(candidates)}/{args.sample}",
                flush=True,
            )
            time.sleep(args.wait_for_available)
        samples = select_manifest(candidates, args.sample, args.seed)
        if len(samples) < args.sample:
            raise RuntimeError(
                f"requested {args.sample} samples but only {len(samples)} accessible items are available"
            )
        temporary = manifest.with_suffix(manifest.suffix + ".partial")
        temporary.write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in samples),
            encoding="utf-8",
        )
        temporary.replace(manifest)
    manifest_hash = _file_sha256(manifest)
    if (
        args.expected_manifest_sha256
        and manifest_hash.lower() != args.expected_manifest_sha256.lower()
    ):
        raise RuntimeError(
            f"manifest SHA-256 mismatch: expected {args.expected_manifest_sha256}, "
            f"got {manifest_hash}"
        )
    client = OpenAICompatibleClient(
        args.base_url,
        args.api_key,
        args.timeout,
        local_file_urls_as_paths=args.local_media_paths,
    )
    if args.backend == "qwen_baseline":
        if args.baseline_mode is None:
            raise ValueError("qwen_baseline requires --baseline-mode")
        experiment_config_sha256 = (
            _validated_sha256(
                args.experiment_config_sha256,
                "experiment-config-sha256",
            )
            if args.defer_scoring or args.experiment_config_sha256 is not None
            else None
        )
        model_artifact_sha256 = (
            _validated_sha256(
                args.model_artifact_sha256,
                "model-artifact-sha256",
            )
            if args.defer_scoring or args.model_artifact_sha256 is not None
            else None
        )
        if args.server_max_model_len <= 0:
            raise ValueError("server-max-model-len must be positive")
        if args.baseline_mode == "mismatched_video" and args.mismatched_video_map is None:
            raise ValueError("mismatched_video requires --mismatched-video-map")
        if args.baseline_mode != "mismatched_video" and args.mismatched_video_map is not None:
            raise ValueError(
                "--mismatched-video-map is only valid with mismatched_video mode"
            )
        sampling = (
            direct_sampling_spec(args.direct_sampling)
            if args.direct_sampling is not None
            else None
        )
        mismatched_videos = None
        mismatch_hash = None
        mismatch_payload = None
        if args.mismatched_video_map is not None:
            mismatched_videos, mismatch_hash, mismatch_payload = (
                _load_validated_mismatch_map(
                    args.mismatched_video_map,
                    samples=samples,
                    manifest_hash=manifest_hash,
                    video_root=args.video_root,
                )
            )
        run_context = {
            "manifest_sha256": manifest_hash,
            "experiment_config_sha256": experiment_config_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "annotations_sha256": _file_sha256(args.annotations),
        }
        config = QwenBaselineConfig(
            mode=args.baseline_mode,
            protocol=QWEN_PROTOCOLS[args.qwen_protocol],
            direct_sampling=sampling,
            option_permutation_seed=args.option_permutation_seed,
            mismatched_videos=mismatched_videos,
            generation_seed=args.seed,
            run_context=run_context,
            max_model_len=args.server_max_model_len,
        )
        runner = QwenBaselineRunner(
            client,
            args.model,
            args.video_root,
            config,
        )
        method_parts = [
            "qwen",
            _stable_model_slug(args.model),
            args.baseline_mode,
            args.qwen_protocol,
        ]
        if args.direct_sampling:
            method_parts.append(args.direct_sampling)
        if args.option_permutation_seed is not None:
            method_parts.append(f"seed{args.option_permutation_seed}")
        method_id = "_".join(method_parts)
        _write_frozen_json(
            args.output_dir / f"frozen_inputs_{args.dataset}_{method_id}.json",
            {
                "dataset": args.dataset,
                "manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": manifest_hash,
                },
                "annotations": {
                    "path": str(args.annotations.resolve()),
                    "sha256": _file_sha256(args.annotations),
                    "model_access": False,
                },
                "model": args.model,
                "model_slug": _stable_model_slug(args.model),
                "model_artifact_sha256": model_artifact_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "generation_seed": args.seed,
                "server_max_model_len": args.server_max_model_len,
                "baseline_mode": args.baseline_mode,
                "qwen_protocol": args.qwen_protocol,
                "direct_sampling": args.direct_sampling,
                "option_permutation_seed": args.option_permutation_seed,
                "mismatched_video_map": (
                    {
                        "path": str(args.mismatched_video_map.resolve()),
                        "sha256": mismatch_hash,
                        "mapped": len(mismatched_videos or {}),
                        "control_unavailable": (
                            len(samples) - len(mismatched_videos or {})
                        ),
                        "duration_bucket_edges_s": _json_duration_bucket_edges(),
                        "schema_version": (
                            mismatch_payload.get("schema_version")
                            if mismatch_payload is not None
                            else None
                        ),
                    }
                    if args.mismatched_video_map is not None
                    else None
                ),
                "run_fingerprint": runner.run_fingerprint(),
            },
        )
        summary = evaluate_qwen_runner(
            samples,
            runner,
            method_id,
            args.output_dir,
            concurrency=args.concurrency,
            resume=args.resume,
            retry_errors=args.retry_errors,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.backend == "qwen_agent":
        if args.agent_config is None or not args.agent_config.is_file():
            raise ValueError("qwen_agent requires --agent-config JSON")
        experiment_config_sha256 = _validated_sha256(
            args.experiment_config_sha256,
            "experiment-config-sha256",
        )
        model_artifact_sha256 = _validated_sha256(
            args.model_artifact_sha256,
            "model-artifact-sha256",
        )
        if args.server_max_model_len <= 0:
            raise ValueError("server-max-model-len must be positive")
        agent_payload = json.loads(args.agent_config.read_text(encoding="utf-8"))
        if not isinstance(agent_payload, dict):
            raise ValueError("agent-config must contain a JSON object")
        agent_settings = agent_payload.get("agent", agent_payload)
        if not isinstance(agent_settings, dict):
            raise ValueError("agent-config agent field must be a JSON object")
        agent_config_hash = _file_sha256(args.agent_config)
        if args.expected_agent_config_sha256 is not None:
            expected_agent_hash = _validated_sha256(
                args.expected_agent_config_sha256,
                "expected-agent-config-sha256",
            )
            if agent_config_hash != expected_agent_hash:
                raise RuntimeError(
                    "agent-config SHA-256 mismatch: "
                    f"expected {expected_agent_hash}, got {agent_config_hash}"
                )
        if args.defer_scoring and args.trajectory_schedule_id is None:
            raise ValueError("--defer-scoring requires --trajectory-schedule-id")
        if not args.defer_scoring and args.trajectory_schedule_id is not None:
            raise ValueError("--trajectory-schedule-id requires --defer-scoring")
        if args.defer_scoring and args.trajectory_replica_id < 0:
            raise ValueError("trajectory-replica-id must be non-negative")
        if args.defer_scoring and not args.trajectory_variant_id.strip():
            raise ValueError("trajectory-variant-id cannot be empty")
        if not args.defer_scoring and args.trajectory_variant_id != "base":
            raise ValueError("non-base trajectory-variant-id requires --defer-scoring")
        if args.defer_scoring and args.expected_agent_config_sha256 is None:
            raise ValueError("--defer-scoring requires --expected-agent-config-sha256")
        if args.defer_scoring and args.train600_manifest_sha256 is None:
            raise ValueError("--defer-scoring requires --train600-manifest-sha256")
        if args.defer_scoring and args.model != "Qwen3.5-9B":
            raise ValueError("deferred trajectory generation requires Qwen3.5-9B")
        train600_manifest_sha256 = (
            _validated_sha256(
                args.train600_manifest_sha256,
                "train600-manifest-sha256",
            )
            if args.defer_scoring
            else None
        )
        qwen_protocol = QWEN_PROTOCOLS[args.qwen_protocol]
        run_context = {
            "manifest_sha256": manifest_hash,
            "train600_manifest_sha256": train600_manifest_sha256,
            "experiment_config_sha256": experiment_config_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "agent_config_sha256": agent_config_hash,
            "annotations_sha256": _file_sha256(args.annotations),
        }
        agent_protocol = AgentInferenceProtocol(
            enable_thinking=qwen_protocol.enable_thinking,
            temperature=qwen_protocol.temperature,
            top_p=qwen_protocol.top_p,
            top_k=qwen_protocol.top_k,
            min_p=qwen_protocol.min_p,
            presence_penalty=qwen_protocol.presence_penalty,
            repetition_penalty=qwen_protocol.repetition_penalty,
            seed=args.seed,
            # Direct MCQ output is deliberately short, but Agent planner and
            # observer turns contain structured tool/evidence payloads.  A
            # shared 512-token cap truncated otherwise valid Agent turns in
            # smoke.  Keep the final judge at the frozen Direct budget while
            # giving non-thinking planning turns enough room to terminate.
            planner_max_tokens=(
                max(2048, qwen_protocol.max_tokens)
                if not qwen_protocol.enable_thinking
                else qwen_protocol.max_tokens
            ),
            observer_max_tokens=(
                max(2048, qwen_protocol.max_tokens)
                if not qwen_protocol.enable_thinking
                else qwen_protocol.max_tokens
            ),
            judge_max_tokens=qwen_protocol.max_tokens,
            direct_max_tokens=qwen_protocol.max_tokens,
            length_retry_max_tokens=qwen_protocol.length_retry_max_tokens,
            server_max_model_len=args.server_max_model_len,
            run_context_sha256=_canonical_sha256(run_context),
        )
        base_runner = build_qwen_agent_strategy(
            agent_settings,
            client=client,
            model=args.model,
            video_root=args.video_root,
            frame_root=frame_root,
            protocol=agent_protocol,
        )
        runner = base_runner

        def result_adapter(trace):
            return trace.to_result_dict()

        schedule_id = None
        if args.defer_scoring:
            schedule_id = str(args.trajectory_schedule_id)
            runner = QwenTrajectoryRunner(
                strategy=base_runner,
                client=client,
                model=args.model,
                protocol=agent_protocol,
                config=TrajectoryGenerationConfig(
                    schedule_id=schedule_id,
                    dataset_manifest_sha256=manifest_hash,
                    train600_manifest_sha256=str(train600_manifest_sha256),
                    experiment_config_sha256=experiment_config_sha256,
                    agent_config_sha256=agent_config_hash,
                    model_artifact_sha256=model_artifact_sha256,
                    replica_id=args.trajectory_replica_id,
                    variant_id=args.trajectory_variant_id,
                ),
            )
            result_adapter = None
        method_parts = [
            "qwen_agent",
            _stable_model_slug(args.model),
            base_runner.strategy_id,
            args.qwen_protocol,
            f"seed{args.seed}",
        ]
        if schedule_id is not None:
            method_parts.extend(
                [
                    re.sub(r"[^A-Za-z0-9_.-]+", "-", schedule_id).strip("-")[:64],
                    f"replica{args.trajectory_replica_id}",
                ]
            )
        method_id = "_".join(method_parts)
        _write_frozen_json(
            args.output_dir / f"frozen_inputs_{args.dataset}_{method_id}.json",
            {
                "dataset": args.dataset,
                "manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": manifest_hash,
                },
                "annotations": {
                    "path": str(args.annotations.resolve()),
                    "sha256": _file_sha256(args.annotations),
                    "model_access": False,
                },
                "model": args.model,
                "model_slug": _stable_model_slug(args.model),
                "model_artifact_sha256": model_artifact_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "qwen_protocol": args.qwen_protocol,
                "seed": args.seed,
                "server_max_model_len": args.server_max_model_len,
                "scoring_deferred": args.defer_scoring,
                "train600_manifest_sha256": train600_manifest_sha256,
                "trajectory_schedule_id": schedule_id,
                "trajectory_replica_id": args.trajectory_replica_id,
                "trajectory_variant_id": args.trajectory_variant_id,
                "agent_config": {
                    "path": str(args.agent_config.resolve()),
                    "sha256": agent_config_hash,
                },
                "run_fingerprint": runner.run_fingerprint(),
            },
        )
        summary = evaluate_qwen_runner(
            samples,
            runner,
            method_id,
            args.output_dir,
            concurrency=args.concurrency,
            resume=args.resume,
            retry_errors=args.retry_errors,
            result_adapter=result_adapter,
            defer_scoring=args.defer_scoring,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    candidate_answers: dict[str, str] | None = None
    candidate_sources: dict[str, str] | None = None
    if args.backend == "hybrid_frozen":
        if args.agent_version != "hybrid_v3c":
            raise ValueError(
                "hybrid_frozen is the frozen-candidate hybrid_v3c baseline"
            )
        if args.candidate_results is None or not args.candidate_results.is_file():
            raise ValueError(
                "hybrid_frozen requires --candidate-results with frozen Direct output"
            )
        normalization_cache = (
            args.candidate_normalization_cache
            or _default_candidate_normalization_cache(
                args.candidate_results,
                args.dataset,
                args.model,
            )
        )
        (
            candidate_answers,
            candidate_sources,
            normalization_reasons,
            normalizer_calls,
            normalized_candidate_hash,
        ) = (
            _normalize_frozen_candidates(
                args.candidate_results,
                samples,
                client,
                args.model,
                normalization_cache,
                read_only=args.candidate_normalization_read_only,
            )
        )
        evaluator = Evaluator(
            client,
            args.model,
            args.video_root,
            frame_root,
            max_turns=args.max_turns or 6,
            max_call_visual_tokens=args.max_call_visual_tokens,
            max_total_visual_tokens=args.max_total_visual_tokens,
            agent_version=args.agent_version,
        )
        candidate_hash = _file_sha256(args.candidate_results)
        _write_frozen_json(
            args.output_dir / f"frozen_inputs_{args.dataset}.json",
            {
                    "dataset": args.dataset,
                    "manifest": {
                        "path": str(manifest.resolve()),
                        "sha256": manifest_hash,
                    },
                    "candidate_results": {
                        "path": str(args.candidate_results.resolve()),
                        "sha256": candidate_hash,
                        "direct_rerun": 0,
                    },
                    "candidate_normalization": {
                        "cache_path": str(normalization_cache.resolve()),
                        "scope_sha256": normalized_candidate_hash,
                        "valid": len(candidate_answers),
                        "text_normalizer_calls": normalizer_calls,
                        "reasons": normalization_reasons,
                    },
            },
        )
    elif args.backend in {"fast_hybrid_eva", "perception_memory_eva"}:
        diagnostics_gate_sha256 = None
        if (
            args.backend == "fast_hybrid_eva"
            and args.agent_version not in {"fast_hybrid_v1", "fast_hybrid_v2"}
        ):
            raise ValueError(
                "fast_hybrid_eva requires --agent-version fast_hybrid_v1 or fast_hybrid_v2"
            )
        if (
            args.backend == "perception_memory_eva"
            and args.agent_version != "perception_memory_v1"
        ):
            raise ValueError(
                "perception_memory_eva requires --agent-version perception_memory_v1"
            )
        if args.backend == "perception_memory_eva":
            args.trajectory_variant_id = _validate_perception_memory_variant(
                args.trajectory_variant_id
            )
            if args.diagnostics_gate_summary is None:
                raise ValueError(
                    "perception_memory_eva requires --diagnostics-gate-summary"
                )
            diagnostics_gate_sha256 = _validate_perception_memory_diagnostics_gate(
                args.diagnostics_gate_summary
            )
        if args.candidate_results is None or not args.candidate_results.is_file():
            raise ValueError(
                f"{args.backend} requires --candidate-results with frozen clean Direct output"
            )
        if args.defer_scoring:
            if args.trajectory_schedule_id is None:
                raise ValueError(
                    f"{args.backend} --defer-scoring requires --trajectory-schedule-id"
                )
            if args.expected_agent_config_sha256 is not None:
                raise ValueError(
                    f"{args.backend} uses its pinned implementation, not "
                    "--expected-agent-config-sha256"
                )
            if args.train600_manifest_sha256 is None:
                raise ValueError(
                    f"{args.backend} --defer-scoring requires --train600-manifest-sha256"
                )
            if args.trajectory_replica_id < 0:
                raise ValueError("trajectory-replica-id must be non-negative")
            if not args.trajectory_variant_id.strip():
                raise ValueError("trajectory-variant-id cannot be empty")
            if args.model != "Qwen3.5-9B":
                raise ValueError("Fast Hybrid SFT trajectories require Qwen3.5-9B")
        elif args.trajectory_schedule_id is not None:
            raise ValueError("--trajectory-schedule-id requires --defer-scoring")
        experiment_config_sha256 = _validated_sha256(
            args.experiment_config_sha256,
            "experiment-config-sha256",
        )
        model_artifact_sha256 = _validated_sha256(
            args.model_artifact_sha256,
            "model-artifact-sha256",
        )
        teacher_model_artifact_sha256 = (
            _validated_sha256(
                args.teacher_model_artifact_sha256,
                "teacher-model-artifact-sha256",
            )
            if args.teacher_model_artifact_sha256 is not None
            else model_artifact_sha256
        )
        train600_manifest_sha256 = (
            _validated_sha256(
                args.train600_manifest_sha256,
                "train600-manifest-sha256",
            )
            if args.defer_scoring
            else None
        )
        candidate_answers = {}
        candidate_records: dict[str, dict] = {}
        unavailable_ids: set[str] = set()
        protocol_violations: list[str] = []
        for line_number, line in enumerate(
            args.candidate_results.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id"))
            if sample_id in candidate_records:
                raise ValueError(f"duplicate frozen candidate sample_id: {sample_id}")
            candidate_records[sample_id] = record
            prediction = str(record.get("prediction") or "").strip().upper()
            if (
                not prediction
                and record.get("data_unavailable") is True
                and record.get("failure_class") == "data_unavailable"
            ):
                unavailable_ids.add(sample_id)
                continue
            protocol_request = record.get("protocol_request") or {}
            if (
                record.get("baseline_mode") != "direct"
                or record.get("sampling_id") != "uniform32"
                or record.get("enable_thinking") is not False
                or protocol_request.get("max_tokens") != 512
                or float(protocol_request.get("temperature", -1.0)) != 0.0
            ):
                protocol_violations.append(f"line {line_number} ({sample_id})")
            if prediction:
                candidate_answers[sample_id] = prediction
        if protocol_violations:
            raise ValueError(
                "candidate file is not clean no_think/uniform32/max_tokens=512 Direct: "
                + ", ".join(protocol_violations[:10])
            )
        sample_by_id = {sample.sample_id: sample for sample in samples}
        unexpected = sorted(set(candidate_records).difference(sample_by_id))
        if unexpected and not _candidate_superset_allowed(
            args.backend, args.defer_scoring, args.trajectory_variant_id
        ):
            raise ValueError(
                "frozen clean Direct contains samples outside the active manifest: "
                + ", ".join(unexpected[:10])
            )
        missing_records = sorted(set(sample_by_id).difference(candidate_records))
        if missing_records:
            raise ValueError(
                "frozen clean Direct is missing active-manifest records: "
                + ", ".join(missing_records[:10])
            )
        unresolved = sorted(
            set(sample_by_id).difference(candidate_answers).difference(unavailable_ids)
        )
        if unresolved:
            raise ValueError(
                "frozen clean Direct has unresolved non-data errors: "
                + ", ".join(unresolved[:10])
            )
        video_index = VideoIndex(args.video_root)
        falsely_unavailable: list[str] = []
        for sample_id in sorted(unavailable_ids):
            try:
                video_index.resolve(sample_by_id[sample_id].video)
            except FileNotFoundError:
                continue
            falsely_unavailable.append(sample_id)
        if falsely_unavailable:
            raise ValueError(
                "frozen clean Direct marks accessible videos unavailable: "
                + ", ".join(falsely_unavailable[:10])
            )
        invalid = sorted(
            sample_id
            for sample_id, prediction in candidate_answers.items()
            if sample_id in sample_by_id
            and prediction not in sample_by_id[sample_id].option_letters
        )
        if invalid:
            raise ValueError(
                "frozen clean Direct contains invalid predictions: "
                + ", ".join(invalid[:10])
            )
        candidate_answers = {
            sample_id: candidate_answers[sample_id]
            for sample_id in sample_by_id
            if sample_id in candidate_answers
        }
        candidate_sources = {
            sample_id: "parsed" if sample_id in candidate_answers else "none"
            for sample_id in sample_by_id
        }
        candidate_hash = _file_sha256(args.candidate_results)
        if args.backend == "fast_hybrid_eva":
            evaluator = FastHybridEvaEvaluator(
                client,
                args.model,
                args.video_root,
                frame_root,
                version=args.agent_version,
                max_turns=args.max_turns or 6,
                max_call_visual_tokens=args.max_call_visual_tokens,
                max_total_visual_tokens=args.max_total_visual_tokens,
                candidate_results_sha256=candidate_hash,
                teacher_model_sha256=teacher_model_artifact_sha256,
                served_model_sha256=model_artifact_sha256,
                manifest_sha256=manifest_hash,
                experiment_config_sha256=experiment_config_sha256,
                scoring_deferred=args.defer_scoring,
                teacher_temperature=args.controller_temperature,
                generation_seed=args.seed,
                trajectory_context=(
                    {
                        "experiment_config_sha256": experiment_config_sha256,
                        "model_artifact_sha256": model_artifact_sha256,
                        "manifest_sha256": manifest_hash,
                        "train600_manifest_sha256": train600_manifest_sha256,
                        "trajectory_schedule_id": str(args.trajectory_schedule_id),
                        "trajectory_variant_id": args.trajectory_variant_id,
                        "trajectory_replica_id": args.trajectory_replica_id,
                    }
                    if args.defer_scoring
                    else None
                ),
            )
        else:
            evaluator = PerceptionMemoryEvaEvaluator(
                client,
                args.model,
                args.video_root,
                frame_root,
                max_turns=args.max_turns or 6,
                max_frames_per_call=args.max_frames_per_call,
                seed=args.seed,
                candidate_results_sha256=candidate_hash,
                model_artifact_sha256=model_artifact_sha256,
                manifest_sha256=manifest_hash,
                experiment_config_sha256=experiment_config_sha256,
                diagnostics_gate_sha256=diagnostics_gate_sha256,
                scoring_deferred=args.defer_scoring,
                train600_manifest_sha256=train600_manifest_sha256,
                trajectory_schedule_id=args.trajectory_schedule_id,
                trajectory_variant_id=args.trajectory_variant_id,
                trajectory_replica_id=args.trajectory_replica_id,
            )
        _write_frozen_json(
            args.output_dir / f"frozen_inputs_{args.dataset}.json",
            {
                "dataset": args.dataset,
                "manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": manifest_hash,
                },
                "candidate_results": {
                    "path": str(args.candidate_results.resolve()),
                    "sha256": candidate_hash,
                    "direct_rerun": 0,
                    "parsed": len(candidate_answers),
                    "source_data_unavailable": len(unavailable_ids),
                },
                "agent_version": args.agent_version,
                "scoring_deferred": args.defer_scoring,
                "teacher_temperature": args.controller_temperature,
                "generation_seed": args.seed,
                "experiment_config_sha256": experiment_config_sha256,
                "model_artifact_sha256": model_artifact_sha256,
                "teacher_model_artifact_sha256": teacher_model_artifact_sha256,
                "train600_manifest_sha256": train600_manifest_sha256,
                "diagnostics_gate_sha256": diagnostics_gate_sha256,
                "local_media_transport": local_media_transport,
                "trajectory_schedule_id": args.trajectory_schedule_id,
                "trajectory_variant_id": args.trajectory_variant_id,
                "trajectory_replica_id": args.trajectory_replica_id,
                "official_eva_commit": OFFICIAL_EVA_COMMIT,
                "run_fingerprint": evaluator.run_fingerprint(),
            },
        )
    elif args.backend == "flashvid_hybrid":
        if args.agent_version != "flashvid_budget_v1":
            raise ValueError(
                "flashvid_hybrid requires --agent-version flashvid_budget_v1"
            )
        if args.candidate_results is None or not args.candidate_results.is_file():
            raise ValueError(
                "flashvid_hybrid requires --candidate-results with frozen Direct output"
            )
        if args.perception_endpoints is None or not args.perception_endpoints.is_file():
            raise ValueError(
                "flashvid_hybrid requires --perception-endpoints JSON"
            )
        endpoint_config_hash = _file_sha256(args.perception_endpoints)
        if (
            args.expected_endpoint_config_sha256
            and endpoint_config_hash.lower()
            != args.expected_endpoint_config_sha256.lower()
        ):
            raise RuntimeError(
                "perception endpoint config SHA-256 mismatch: expected "
                f"{args.expected_endpoint_config_sha256}, got {endpoint_config_hash}"
            )
        endpoint_payload = json.loads(
            args.perception_endpoints.read_text(encoding="utf-8")
        )
        configured_media_root = endpoint_payload.get("allowed_media_root")
        if (
            configured_media_root is not None
            and Path(configured_media_root).resolve()
            != args.perception_media_root.resolve()
        ):
            raise RuntimeError(
                "perception media root differs from endpoint service config: "
                f"{args.perception_media_root} != {configured_media_root}"
            )
        controller_client = OpenAICompatibleClient(
            args.controller_base_url or args.base_url,
            args.api_key,
            args.timeout,
        )
        controller_model = args.controller_model or args.model
        configured_controller = endpoint_payload.get("controller")
        if isinstance(configured_controller, dict):
            configured_url = str(configured_controller.get("base_url") or "").rstrip("/")
            requested_url = str(args.controller_base_url or args.base_url).rstrip("/")
            configured_model = str(configured_controller.get("model") or "")
            if (
                configured_url
                and configured_url != requested_url
                and not args.controller_config_override
            ):
                raise RuntimeError(
                    f"controller base URL differs from endpoint config: "
                    f"{requested_url} != {configured_url}"
                )
            if (
                configured_model
                and configured_model != controller_model
                and not args.controller_config_override
            ):
                raise RuntimeError(
                    f"controller model differs from endpoint config: "
                    f"{controller_model} != {configured_model}"
                )
        normalization_cache = (
            args.candidate_normalization_cache
            or _default_candidate_normalization_cache(
                args.candidate_results,
                args.dataset,
                controller_model,
            )
        )
        (
            candidate_answers,
            candidate_sources,
            normalization_reasons,
            normalizer_calls,
            normalized_candidate_hash,
        ) = (
            _normalize_frozen_candidates(
                args.candidate_results,
                samples,
                controller_client,
                controller_model,
                normalization_cache,
                read_only=args.candidate_normalization_read_only,
            )
        )
        candidate_hash = _file_sha256(args.candidate_results)
        evaluator = FlashVIDHybridEvaluator(
            controller_client,
            controller_model,
            BudgetEndpointPool.load(args.perception_endpoints),
            args.video_root,
            frame_root,
            args.perception_cache_root,
            args.perception_media_root,
            FlashVIDHybridConfig(
                budget_policy=args.budget_policy,
                fixed_retention_ratio=args.fixed_retention_ratio,
                max_turns=args.max_turns or 6,
                max_perception_calls=args.max_perception_calls,
                controller_temperature=args.controller_temperature,
                trajectory_index=args.trajectory_index,
                budget_strategy=args.budget_strategy,
                budget_random_seed=args.budget_random_seed,
                budget_match_distribution=budget_match_distribution,
                controller_prompt_id=args.controller_prompt_id,
            ),
            candidate_sources=candidate_sources,
            candidate_file_hash=candidate_hash,
            normalized_candidate_hash=normalized_candidate_hash,
            manifest_hash=manifest_hash,
            experiment_config_hash=args.experiment_config_sha256,
        )
        evaluator.freeze_artifacts(args.output_dir)
        frozen_inputs = {
            "dataset": args.dataset,
            "manifest": {"path": str(manifest.resolve()), "sha256": manifest_hash},
            "candidate_results": {
                "path": str(args.candidate_results.resolve()),
                "sha256": candidate_hash,
                "direct_rerun": 0,
            },
            "annotations": {
                "path": str(args.annotations.resolve()),
                "sha256": _file_sha256(args.annotations),
                "model_access": False,
            },
            "perception_endpoint_config": {
                "path": str(args.perception_endpoints.resolve()),
                "sha256": endpoint_config_hash,
            },
            "controller_config_override": args.controller_config_override,
            "implementation_sha256": evaluator.implementation_sha256,
            "candidate_normalization": {
                "cache_path": str(normalization_cache.resolve()),
                "scope_sha256": normalized_candidate_hash,
                "valid": len(candidate_answers),
                "recovered": sum(
                    source == "normalized" for source in candidate_sources.values()
                ),
                "text_normalizer_calls": normalizer_calls,
                "reasons": normalization_reasons,
            },
        }
        _write_frozen_json(
            args.output_dir / f"frozen_inputs_{args.dataset}.json",
            frozen_inputs,
        )
    else:
        evaluator = Evaluator(
            client,
            args.model,
            args.video_root,
            frame_root,
            max_turns=args.max_turns or 3,
            max_call_visual_tokens=args.max_call_visual_tokens,
            max_total_visual_tokens=args.max_total_visual_tokens,
            agent_version=args.agent_version,
        )
    if args.backend == "flashvid_hybrid" and args.trajectories_per_sample > 1:
        summary = evaluate_flashvid_trajectories(
            samples,
            evaluator,
            candidate_answers or {},
            args.output_dir,
            trajectories_per_sample=args.trajectories_per_sample,
            concurrency=args.concurrency,
            resume=args.resume,
            retry_errors=args.retry_errors,
        )
    else:
        summary = evaluate(
            samples,
            evaluator,
            args.backend,
            args.output_dir,
            concurrency=args.concurrency,
            resume=args.resume,
            retry_errors=args.retry_errors,
            candidate_answers=candidate_answers,
            candidate_sources=candidate_sources,
            candidate_records=(
                candidate_records
                if args.backend in {"fast_hybrid_eva", "perception_memory_eva"}
                else None
            ),
            defer_scoring=args.defer_scoring,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

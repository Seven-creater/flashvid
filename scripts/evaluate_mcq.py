from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.datasets import load_samples
from flashvid_eval.flashvid_budget import BudgetEndpointPool
from flashvid_eval.flashvid_hybrid import (
    FlashVIDHybridConfig,
    FlashVIDHybridEvaluator,
    evaluate_flashvid_trajectories,
)
from flashvid_eval.offline_budget import normalize_candidate
from flashvid_eval.answers import extract_strict_answer_letter
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
            "flashvid_hybrid",
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
            "flashvid_budget_v1",
        ),
        default="v2a",
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
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
    client = OpenAICompatibleClient(args.base_url, args.api_key, args.timeout)
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
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

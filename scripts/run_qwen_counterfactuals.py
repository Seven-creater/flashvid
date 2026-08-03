#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.qwen_agents import AgentConfig, FrameRequest, FrameTool, InferenceProtocol
from flashvid_eval.qwen_protocol import QWEN_PROTOCOLS
from flashvid_eval.qwen_sft import materialize_trajectory_identity
from flashvid_eval.qwen_trajectories import FixedEvidenceReplayStrategy
from flashvid_eval.schemas import ModelSample


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validated_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return normalized


def load_model_samples(path: Path, dataset: str) -> tuple[dict[str, ModelSample], str]:
    rows: dict[str, ModelSample] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if str(payload.get("dataset") or "").lower() != dataset:
            raise ValueError(f"manifest row {line_number} has wrong dataset")
        sample_id = str(payload.get("sample_id") or "")
        if not sample_id or sample_id in rows:
            raise ValueError(f"manifest row {line_number} has duplicate/empty sample_id")
        choices = payload.get("choices")
        if not isinstance(choices, dict) or not choices:
            raise ValueError(f"manifest row {line_number} has invalid choices")
        rows[sample_id] = ModelSample(
            dataset=dataset,
            sample_id=sample_id,
            video=str(payload.get("video") or ""),
            question=str(payload.get("question") or ""),
            choices={str(key): str(value) for key, value in choices.items()},
            candidate_answer=None,
        )
    return rows, file_sha256(path)


def load_specs(
    path: Path,
    *,
    dataset: str,
    train600_manifest_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if str(payload.get("dataset") or "").lower() != dataset:
            continue
        if payload.get("manifest_sha256") != train600_manifest_sha256:
            raise ValueError(f"spec row {line_number} Train600 hash mismatch")
        if payload.get("train600_manifest_sha256") != train600_manifest_sha256:
            raise ValueError(f"spec row {line_number} explicit Train600 hash mismatch")
        if payload.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError(f"spec row {line_number} dataset manifest hash mismatch")
        if payload.get("config_sha256") != config_sha256:
            raise ValueError(f"spec row {line_number} config hash mismatch")
        fingerprint = str(payload.get("counterfactual_fingerprint") or "")
        expected = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "counterfactual_fingerprint"
            }
        )
        if fingerprint != expected:
            raise ValueError(f"spec row {line_number} fingerprint mismatch")
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate counterfactual fingerprint at row {line_number}")
        calls = payload.get("planned_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"spec row {line_number} has no planned_calls")
        fingerprints.add(fingerprint)
        specs.append(payload)
    if not specs:
        raise ValueError(f"no {dataset} counterfactual specs in {path}")
    return specs


def frame_requests(spec: Mapping[str, Any]) -> tuple[FrameRequest, ...]:
    requests: list[FrameRequest] = []
    for index, raw in enumerate(spec["planned_calls"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"planned_calls[{index}] must be an object")
        requests.append(
            FrameRequest(
                start_time=float(raw["start_time"]),
                end_time=float(raw["end_time"]),
                nframes=int(raw["nframes"]),
                resize=float(raw["resize"]),
                evidence_request=str(raw.get("evidence_request") or ""),
            )
        )
    return tuple(requests)


def replica_seed(base_seed: int, spec: Mapping[str, Any], replica_id: int) -> int:
    digest = hashlib.sha256(
        (
            f"{base_seed}\0{spec['counterfactual_fingerprint']}\0{replica_id}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    config_sha256 = validated_sha256(args.config_sha256, "config-sha256")
    model_artifact_sha256 = validated_sha256(
        args.model_artifact_sha256, "model-artifact-sha256"
    )
    expected_manifest_sha256 = validated_sha256(
        args.expected_manifest_sha256, "expected-manifest-sha256"
    )
    train600_manifest_sha256 = validated_sha256(
        args.train600_manifest_sha256, "train600-manifest-sha256"
    )
    expected_agent_sha256 = validated_sha256(
        args.expected_agent_config_sha256, "expected-agent-config-sha256"
    )
    samples, manifest_sha256 = load_model_samples(args.manifest, args.dataset)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 mismatch")
    agent_config_sha256 = file_sha256(args.agent_config)
    if agent_config_sha256 != expected_agent_sha256:
        raise RuntimeError("agent config SHA-256 mismatch")
    agent_payload = json.loads(args.agent_config.read_text(encoding="utf-8"))
    agent_settings = agent_payload.get("agent", agent_payload)
    if not isinstance(agent_settings, dict):
        raise ValueError("agent config must contain an object")
    base_agent_config = AgentConfig.from_mapping(agent_settings)
    replay_agent_config = replace(
        base_agent_config,
        strategy="counterfactual_fixed_evidence",
    )
    specs = load_specs(
        args.specs,
        dataset=args.dataset,
        train600_manifest_sha256=train600_manifest_sha256,
        dataset_manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    unknown = sorted(
        {str(spec.get("sample_id") or "") for spec in specs} - set(samples)
    )
    if unknown:
        raise ValueError(f"counterfactual specs contain unknown samples: {unknown[:3]}")
    if args.replicas != 3:
        raise ValueError("counterfactual stability is frozen at exactly 3 replicas")

    protocol_spec = QWEN_PROTOCOLS[args.qwen_protocol]
    source_context = {
        "dataset": args.dataset,
        "manifest_sha256": manifest_sha256,
        "train600_manifest_sha256": train600_manifest_sha256,
        "config_sha256": config_sha256,
        "agent_config_sha256": agent_config_sha256,
        "model": args.model,
        "model_artifact_sha256": model_artifact_sha256,
        "specs_sha256": file_sha256(args.specs),
        "protocol": args.qwen_protocol,
        "replicas": args.replicas,
        "base_seed": args.seed,
    }
    run_fingerprint = canonical_sha256(
        {
            **source_context,
            "implementation_sha256": file_sha256(Path(__file__)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output.with_suffix(args.output.suffix + ".frozen.json")
    frozen_payload = {"schema_version": 1, **source_context, "run_fingerprint": run_fingerprint}
    if frozen_path.is_file():
        if json.loads(frozen_path.read_text(encoding="utf-8")) != frozen_payload:
            raise RuntimeError("frozen counterfactual inputs changed")
    else:
        temporary = frozen_path.with_suffix(frozen_path.suffix + ".partial")
        temporary.write_text(
            json.dumps(frozen_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(frozen_path)

    expected_jobs = [
        (spec, replica_id)
        for spec in specs
        for replica_id in range(args.replicas)
    ]
    cached: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        for line_number, line in enumerate(
            args.output.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(args.output.read_text(encoding="utf-8").splitlines()):
                    continue
                raise
            if row.get("counterfactual_run_fingerprint") != run_fingerprint:
                raise RuntimeError("counterfactual resume fingerprint mismatch")
            trajectory_id = str(row.get("trajectory_id") or "")
            if trajectory_id in cached:
                raise RuntimeError(f"duplicate resume trajectory_id: {trajectory_id}")
            cached[trajectory_id] = row

    client = OpenAICompatibleClient(args.base_url, args.api_key, args.timeout)
    frame_tool = FrameTool(
        args.frame_root,
        max_frames_per_call=base_agent_config.max_frames_per_call,
    )

    def expected_id(spec: Mapping[str, Any], replica_id: int) -> str:
        family = f"{spec['schedule_id']}~{spec['variant_id']}"
        from flashvid_eval.qwen_sft import composite_trajectory_id

        return composite_trajectory_id(
            args.dataset,
            str(spec["sample_id"]),
            family,
            replica_id,
        )

    pending = [
        (spec, replica_id)
        for spec, replica_id in expected_jobs
        if expected_id(spec, replica_id) not in cached
    ]

    def run_one(spec: Mapping[str, Any], replica_id: int) -> dict[str, Any]:
        seed = replica_seed(args.seed, spec, replica_id)
        protocol = InferenceProtocol(
            enable_thinking=protocol_spec.enable_thinking,
            temperature=protocol_spec.temperature,
            top_p=protocol_spec.top_p,
            top_k=protocol_spec.top_k,
            min_p=protocol_spec.min_p,
            presence_penalty=protocol_spec.presence_penalty,
            repetition_penalty=protocol_spec.repetition_penalty,
            seed=seed,
            planner_max_tokens=protocol_spec.max_tokens,
            observer_max_tokens=protocol_spec.max_tokens,
            judge_max_tokens=protocol_spec.max_tokens,
            direct_max_tokens=protocol_spec.max_tokens,
            length_retry_max_tokens=protocol_spec.length_retry_max_tokens,
            server_max_model_len=args.server_max_model_len,
            run_context_sha256=run_fingerprint,
        )
        strategy = FixedEvidenceReplayStrategy(
            client=client,
            model=args.model,
            video_root=args.video_root,
            frame_tool=frame_tool,
            config=replay_agent_config,
            protocol=protocol,
            planned_calls=frame_requests(spec),
            source_fingerprint=str(spec["counterfactual_fingerprint"]),
        )
        trace = strategy.run(samples[str(spec["sample_id"])])
        row = materialize_trajectory_identity(
            trace.to_result_dict(),
            schedule_id=str(spec["schedule_id"]),
            variant_id=str(spec["variant_id"]),
            replica_id=replica_id,
            judge_seed=seed,
            manifest_sha256=train600_manifest_sha256,
            dataset_manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
        )
        row.update(
            {
                "base_trajectory_id": spec["base_trajectory_id"],
                "counterfactual_fingerprint": spec["counterfactual_fingerprint"],
                "counterfactual_run_fingerprint": run_fingerprint,
                "model_artifact_sha256": model_artifact_sha256,
                "agent_config_sha256": agent_config_sha256,
                "scoring_deferred": True,
            }
        )
        return row

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(run_one, spec, replica_id): (spec, replica_id)
                for spec, replica_id in pending
            }
            for future in as_completed(futures):
                row = future.result()
                trajectory_id = str(row["trajectory_id"])
                cached[trajectory_id] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())

    ordered = [cached[expected_id(spec, replica)] for spec, replica in expected_jobs]
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    errors = sum(bool(row.get("error")) for row in ordered)
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "specs": len(specs),
        "replicas": args.replicas,
        "completed": len(ordered),
        "errors": errors,
        "failure_rate": errors / len(ordered) if ordered else None,
        "output": str(args.output),
        "run_fingerprint": run_fingerprint,
        "scoring_deferred": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen Qwen-only evidence intervals at lower frame budgets."
    )
    parser.add_argument("--dataset", choices=("lvbench", "lsdbench", "cgbench"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train600-manifest-sha256", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--expected-agent-config-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--api-key", default="no")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--qwen-protocol", choices=tuple(QWEN_PROTOCOLS), required=True)
    parser.add_argument("--server-max-model-len", type=int, default=131072)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_matrix(args)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

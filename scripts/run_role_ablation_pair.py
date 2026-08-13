#!/usr/bin/env python3
"""Run one frozen-input Observer, Verifier, or Answerer paired ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.qwen_sft import read_jsonl
from flashvid_eval.role_ablation_replay import (
    PAIRED_ROLES,
    run_paired_role_ablation,
    validate_role_ablation_dev30_scope,
)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
        for row in rows
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("materialized role-ablation run SHA-256 changed")
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError("materialized role-ablation run must be an object")
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("materialized role-ablation run lacks execution contract")
    role = str(execution.get("paired_role") or "")
    if role not in PAIRED_ROLES or not str(execution.get("mode") or "").startswith(
        f"fixed_{role}_pair"
    ):
        raise ValueError("materialized run is not a fixed paired-role cell")
    bindings = execution.get("paired_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "control",
        "treatment",
    }:
        raise ValueError("materialized run lacks paired control/treatment bindings")
    normalized: dict[str, dict[str, str]] = {}
    for arm in ("control", "treatment"):
        binding = bindings[arm]
        if not isinstance(binding, Mapping) or set(binding) != {
            "base_url",
            "model",
            "artifact_sha256",
        }:
            raise ValueError(f"materialized {arm} binding schema changed")
        base_url = str(binding["base_url"] or "").strip()
        model = str(binding["model"] or "").strip()
        artifact = str(binding["artifact_sha256"] or "").strip()
        if (
            not base_url.startswith("http://127.0.0.1:")
            or not model
            or len(artifact) != 64
            or any(character not in "0123456789abcdef" for character in artifact)
        ):
            raise ValueError(f"materialized {arm} binding is invalid")
        normalized[arm] = {
            "base_url": base_url,
            "model": model,
            "artifact_sha256": artifact,
        }
    return {
        "run_id": str(value.get("id") or ""),
        "role": role,
        "bindings": normalized,
        "run_config_sha256": actual_sha256,
    }


def _frozen_inputs(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("frozen role input SHA-256 changed")
    rows = read_jsonl(path)
    validate_role_ablation_dev30_scope(rows)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}")
    contract = _run_contract(args.run_config, args.expected_run_sha256)
    frozen_rows = _frozen_inputs(args.input, args.expected_input_sha256)
    role = contract["role"]
    control_binding = contract["bindings"]["control"]
    treatment_binding = contract["bindings"]["treatment"]
    control = OpenAICompatibleClient(
        control_binding["base_url"],
        api_key=args.api_key,
        timeout=args.timeout,
        local_file_urls_as_paths=args.local_media_paths,
    )
    treatment = OpenAICompatibleClient(
        treatment_binding["base_url"],
        api_key=args.api_key,
        timeout=args.timeout,
        local_file_urls_as_paths=args.local_media_paths,
    )
    downstream_verifier = (
        OpenAICompatibleClient(
            control_binding["base_url"],
            api_key=args.api_key,
            timeout=args.timeout,
            local_file_urls_as_paths=args.local_media_paths,
        )
        if role == "observer"
        else None
    )
    downstream_answerer = (
        OpenAICompatibleClient(
            control_binding["base_url"],
            api_key=args.api_key,
            timeout=args.timeout,
            local_file_urls_as_paths=args.local_media_paths,
        )
        if role == "observer"
        else None
    )
    rows = [
        run_paired_role_ablation(
            frozen,
            role=role,
            control_client=control,
            control_model=control_binding["model"],
            control_artifact_sha256=control_binding["artifact_sha256"],
            treatment_client=treatment,
            treatment_model=treatment_binding["model"],
            treatment_artifact_sha256=treatment_binding["artifact_sha256"],
            materialized_run_id=contract["run_id"],
            materialized_run_sha256=contract["run_config_sha256"],
            downstream_verifier_client=downstream_verifier,
            downstream_verifier_model=(control_binding["model"] if role == "observer" else None),
            downstream_verifier_artifact_sha256=(
                control_binding["artifact_sha256"] if role == "observer" else None
            ),
            downstream_answerer_client=downstream_answerer,
            downstream_answerer_model=(control_binding["model"] if role == "observer" else None),
            downstream_answerer_artifact_sha256=(
                control_binding["artifact_sha256"] if role == "observer" else None
            ),
        )
        for frozen in frozen_rows
    ]
    payload = _jsonl_bytes(rows)
    _write(args.output, payload)
    return {
        "run_id": contract["run_id"],
        "run_config": str(args.run_config.resolve()),
        "run_config_sha256": contract["run_config_sha256"],
        "role": role,
        "samples": len(rows),
        "input": str(args.input.resolve()),
        "input_sha256": args.expected_input_sha256,
        "datasets": validate_role_ablation_dev30_scope(rows),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "offline_scoring_required": True,
        "labels_serialized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--local-media-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "passed", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

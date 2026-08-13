#!/usr/bin/env python3
"""Regenerate Perception-Memory states from immutable Fast Hybrid frame caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.perception_memory_replay import (
    PerceptionMemoryReplay,
    ReplayConfig,
    replay_jsonl,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit-summary",
        type=Path,
        help="Passed 300-sample paired badcase summary; replay is blocked without it.",
    )
    parser.add_argument(
        "--training-source-lock",
        type=Path,
        help="Frozen Train600/source provenance lock for role-separated replay.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        help=(
            "OpenAI-compatible endpoint; repeat to distribute trajectories "
            "deterministically across endpoints. Defaults to 8200."
        ),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument(
        "--served-model-artifact-sha256",
        help="SHA-256 of the model stack serving the replayed Observer.",
    )
    parser.add_argument("--experiment-config-sha256")
    parser.add_argument("--training-source-lock-sha256")
    parser.add_argument(
        "--role-separated-observer",
        action="store_true",
        help=(
            "Use the frozen five-field Observer protocol with strict frame-index "
            "binding; requires --served-model-artifact-sha256."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-call", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--local-media-paths",
        action="store_true",
        help="Send local file:// media as absolute paths for Transformers serving.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ReplayConfig(
            model=args.model,
            seed=args.seed,
            perception_max_tokens=args.max_tokens,
            max_frames_per_call=args.max_frames_per_call,
            request_timeout_s=args.timeout,
            local_media_transport="path" if args.local_media_paths else "file_url",
            role_separated_observer=args.role_separated_observer,
            served_model_artifact_sha256=args.served_model_artifact_sha256,
            experiment_config_sha256=args.experiment_config_sha256,
            training_source_lock_sha256=args.training_source_lock_sha256,
        )
        base_urls = args.base_urls or ["http://127.0.0.1:8200/v1"]
        clients = [
            OpenAICompatibleClient(
                base_url,
                api_key=args.api_key,
                timeout=args.timeout,
                local_file_urls_as_paths=args.local_media_paths,
            )
            for base_url in base_urls
        ]
        replayer = PerceptionMemoryReplay(clients, config)
        summary = replay_jsonl(
            input_path=args.input,
            output_path=args.output,
            audit_summary_path=args.audit_summary,
            training_source_lock_path=args.training_source_lock,
            replayer=replayer,
            concurrency=args.concurrency,
            resume=args.resume,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

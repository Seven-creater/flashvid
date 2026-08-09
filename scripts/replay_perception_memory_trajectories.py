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
        required=True,
        help="Passed 300-sample paired badcase summary; replay is blocked without it.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=80.0)
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
            local_media_transport="path" if args.local_media_paths else "file_url",
        )
        replayer = PerceptionMemoryReplay(
            OpenAICompatibleClient(
                args.base_url,
                api_key=args.api_key,
                timeout=args.timeout,
                local_file_urls_as_paths=args.local_media_paths,
            ),
            config,
        )
        summary = replay_jsonl(
            input_path=args.input,
            output_path=args.output,
            audit_summary_path=args.audit_summary,
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

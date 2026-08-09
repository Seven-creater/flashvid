from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.run_perception_memory_repair_pipeline import (
    _cached_command,
    _rescue_command,
    run_pipeline,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        expected_git_branch="codex/perception-memory-eva",
        expected_git_head="a" * 40,
        source=[tmp_path / f"source-{index}.jsonl" for index in range(8)],
        base_results=[tmp_path / f"base-{index}.jsonl" for index in range(8)],
        audit_summary=tmp_path / "audit.json",
        video_root=tmp_path / "videos",
        frame_root=tmp_path / "frames",
        base_url=[f"http://127.0.0.1:{8200 + index}/v1" for index in range(8)],
        base_worker_match=["replay_perception_memory_trajectories.py", "base_v11"],
        model="Qwen3.5-9B",
        seed=42,
        max_tokens=4096,
        max_frames_per_call=128,
        max_turns=6,
        request_timeout=300.0,
        poll_seconds=30.0,
        wait_timeout=43200.0,
        resume=False,
    )


def test_commands_use_one_combined_cached_input_and_all_eight_endpoints(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    prepared = tmp_path / "repair_scope"
    cached = _cached_command(args, prepared, tmp_path / "cached.jsonl")
    rescue = _rescue_command(args, prepared, tmp_path / "rescue.jsonl")

    assert cached[cached.index("--input") + 1] == str(
        prepared / "cached_repair_input.jsonl"
    )
    assert cached.count("--base-url") == rescue.count("--base-url") == 8
    assert cached[cached.index("--concurrency") + 1] == "8"
    assert rescue[rescue.index("--expected-rows") + 1] == "37"
    assert rescue[rescue.index("--expected-samples") + 1] == "6"
    assert "--resume" in cached and "--resume" in rescue


def test_pipeline_rejects_non_eight_shard_or_endpoint_scope(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.source = args.source[:-1]
    with pytest.raises(ValueError, match="exactly 8 source"):
        run_pipeline(args)

    args = _args(tmp_path)
    args.base_url = args.base_url[:-1]
    with pytest.raises(ValueError, match="exactly 8 unique"):
        run_pipeline(args)

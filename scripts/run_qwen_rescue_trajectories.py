#!/usr/bin/env python3
"""Run the frozen dense A4 rescue schedule on only no-stable Train600 items."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.qwen_sft import canonical_sha256, sha256_file


def _load_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("rescue index must be schema v1")
    claimed = str(payload.get("rescue_index_sha256") or "")
    computed = canonical_sha256(
        {key: value for key, value in payload.items() if key != "rescue_index_sha256"}
    )
    if claimed != computed:
        raise RuntimeError("rescue index self-hash mismatch")
    for field in ("experiment_config", "source_base_bundle", "frozen_winner", "agent_config"):
        reference = payload.get(field)
        if not isinstance(reference, Mapping):
            raise ValueError(f"rescue index has no {field}")
        source = Path(str(reference.get("path") or ""))
        expected = reference.get("file_sha256", reference.get("sha256"))
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"frozen rescue reference changed: {source}")
    return payload


def build_commands(index: Mapping[str, Any], *, resume: bool, retry_errors: bool) -> list[list[str]]:
    config_ref = index["experiment_config"]
    config = json.loads(Path(config_ref["path"]).read_text(encoding="utf-8"))
    if canonical_sha256(config) != config_ref["canonical_sha256"]:
        raise RuntimeError("experiment config canonical hash changed")
    model = config["models"]["q9"]
    commands: list[list[str]] = []
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        manifest = index["manifests"][dataset]
        if int(manifest["count"]) == 0:
            continue
        manifest_path = Path(str(manifest["path"]))
        if not manifest_path.is_file() or sha256_file(manifest_path) != manifest["sha256"]:
            raise RuntimeError(f"frozen rescue manifest changed: {manifest_path}")
        data = config["datasets"][dataset]
        command = [
            sys.executable,
            "scripts/evaluate_mcq.py",
            "--dataset",
            dataset,
            "--backend",
            "qwen_agent",
            "--annotations",
            str(data["annotations"]),
            "--video-root",
            str(data["video_root"]),
            "--base-url",
            str(model["base_url"]),
            "--model",
            str(model["served_name"]),
            "--model-artifact-sha256",
            str(model["artifact_sha256"]),
            "--manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            str(manifest["sha256"]),
            "--sample",
            str(manifest["count"]),
            "--seed",
            str(int(index["frozen_winner"]["seed"]) + 99991),
            "--output-dir",
            str(manifest["result_dir"]),
            "--concurrency",
            str(config["execution"]["concurrency"]),
            "--timeout",
            "80",
            "--experiment-config-sha256",
            str(config_ref["canonical_sha256"]),
            "--frame-root",
            str(Path(str(config["result_root"])) / "frame_cache/q9"),
            "--expected-agent-config-sha256",
            str(index["agent_config"]["sha256"]),
            "--qwen-protocol",
            str(index["frozen_winner"]["protocol"]),
            "--agent-config",
            str(index["agent_config"]["path"]),
            "--defer-scoring",
            "--train600-manifest-sha256",
            str(index["train600_manifest_sha256"]),
            "--trajectory-schedule-id",
            str(index["schedule_id"]),
            "--trajectory-variant-id",
            str(index["variant_id"]),
        ]
        if resume:
            command.append("--resume")
        if retry_errors:
            command.append("--retry-errors")
        commands.append(command)
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-index", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        index = _load_index(args.rescue_index)
        commands = build_commands(
            index, resume=args.resume, retry_errors=args.retry_errors
        )
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "commands": commands}, indent=2))
            return 0
        for command in commands:
            subprocess.run(command, check=True)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "passed", "datasets_run": len(commands)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize the frozen Fast Hybrid SFT Dev selection and optional Test300 run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashvid_eval.fast_hybrid_eval_protocol import (
    DATASETS,
    audit_result_file,
    load_protocol,
    manifest_sample_ids,
    read_jsonl,
    require_sha256,
    sha256_file,
)
try:
    from scripts.run_fast_hybrid_sft_eval import _load_checkpoint
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from run_fast_hybrid_sft_eval import _load_checkpoint  # type: ignore[no-redef]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{key} must be a finite non-negative number")
    return float(value)


def _summarize_rows(paths: Mapping[str, Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct_by_dataset: dict[str, int] = {}
    for dataset in DATASETS:
        dataset_rows = read_jsonl(paths[dataset])
        correct_by_dataset[dataset] = sum(
            str(row.get("prediction") or "").upper()
            == str(row.get("answer") or "").upper()
            for row in dataset_rows
        )
        rows.extend(dataset_rows)
    if not rows:
        raise ValueError("no evaluation rows")
    failures = sum(
        bool(
            row.get("error")
            or row.get("api_error")
            or row.get("frame_error")
            or row.get("parse_error")
        )
        for row in rows
    )
    return {
        "samples": len(rows),
        "correct": sum(correct_by_dataset.values()),
        "correct_by_dataset": correct_by_dataset,
        "mean_total_tokens": fmean(
            _number(row, "end_to_end_total_tokens") for row in rows
        ),
        "mean_visual_tokens": fmean(
            _number(row, "end_to_end_visual_tokens") for row in rows
        ),
        "mean_latency_s": fmean(
            _number(row, "end_to_end_latency_s") for row in rows
        ),
        "mean_tool_calls": fmean(len(row.get("tool_calls") or []) for row in rows),
        "engineering_failures": failures,
        "failure_rate": failures / len(rows),
        "annotation_leaks": sum(
            row.get("annotation_leak_check") != "passed" for row in rows
        ),
        "candidate_reruns": sum(int(row.get("candidate_rerun") or 0) for row in rows),
    }


def _audit_test(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    metadata_path = root / "evaluation_run.json"
    metadata = _load_object(metadata_path)
    if (
        metadata.get("kind") != "fast_hybrid_sft_eval_run"
        or metadata.get("status") != "passed"
        or metadata.get("phase") != "test"
        or metadata.get("mode") != "checkpoint"
        or metadata.get("evaluation_protocol_sha256") != protocol_sha
        or metadata.get("served_model_sha256") != checkpoint["served_stack_sha256"]
        or metadata.get("teacher_model_sha256")
        != protocol["base_model_artifact_sha256"]
    ):
        raise RuntimeError("final Test300 metadata is incomplete or mismatched")
    paths: dict[str, Path] = {}
    for dataset in DATASETS:
        entry = protocol["splits"]["test"][dataset]
        path = Path(metadata["files"][dataset]["path"])
        audit = audit_result_file(
            path,
            dataset=dataset,
            expected_count=int(entry["manifest"]["count"]),
            expected_sample_ids=manifest_sample_ids(
                Path(str(entry["manifest"]["path"]))
            ),
            manifest_sha256=str(entry["manifest"]["sha256"]),
            candidate_sha256=str(entry["candidate"]["sha256"]),
            experiment_config_sha256=str(protocol["experiment_config"]["sha256"]),
            served_model_sha256=str(checkpoint["served_stack_sha256"]),
            teacher_model_sha256=str(protocol["base_model_artifact_sha256"]),
        )
        if metadata["files"][dataset].get("sha256") != audit["sha256"]:
            raise RuntimeError(f"final result changed after audit: {path}")
        paths[dataset] = path
    return metadata, paths


def _write_outputs(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline = payload["frozen_untrained_test"]
    lines = [
        "# Fast Hybrid EVA 9B SFT summary",
        "",
        f"状态：`{payload['status']}`。",
        "",
        "| 系统 | 样本 | 正确 | LVBench | LSDBench | CG-Bench | 平均总Token | 平均视觉Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| 冻结未训练Agent | 300 | "
            f"{baseline['overall_correct']} | "
            f"{baseline['per_dataset_correct']['lvbench']} | "
            f"{baseline['per_dataset_correct']['lsdbench']} | "
            f"{baseline['per_dataset_correct']['cgbench']} | "
            f"{baseline['mean_total_tokens']:.1f} | "
            f"{baseline['mean_visual_tokens']:.1f} |"
        ),
    ]
    final = payload.get("final_test")
    if isinstance(final, Mapping):
        metrics = final["summary"]
        lines.append(
            "| SFT-9B Agent | "
            f"{metrics['samples']} | {metrics['correct']} | "
            f"{metrics['correct_by_dataset']['lvbench']} | "
            f"{metrics['correct_by_dataset']['lsdbench']} | "
            f"{metrics['correct_by_dataset']['cgbench']} | "
            f"{metrics['mean_total_tokens']:.1f} | "
            f"{metrics['mean_visual_tokens']:.1f} |"
        )
        lines.extend(
            [
                "",
                f"最终验收：`{'通过' if final['passed'] else '未通过'}`。",
                (
                    "只有总Token和视觉Token均下降至少30%时才称为‘明显降低’："
                    f"`{'是' if final['clearly_reduced_30pct'] else '否'}`。"
                ),
            ]
        )
    else:
        lines.extend(["", "没有Dev checkpoint通过预注册门槛，因此未运行Test300。"])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        protocol_sha = require_sha256(
            args.expected_protocol_sha256, "expected_protocol_sha256"
        )
        protocol = load_protocol(args.protocol, protocol_sha)
        winner = _load_object(args.winner)
        if (
            winner.get("kind") != "fast_hybrid_sft_winner"
            or winner.get("evaluation_protocol_sha256") != protocol_sha
        ):
            raise ValueError("winner does not belong to the frozen evaluation protocol")
        experiment = _load_object(Path(protocol["experiment_config"]["path"]))
        baseline = dict(experiment["frozen_test_teacher"])
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "fast_hybrid_sft_summary",
            "status": "blocked" if winner.get("status") == "blocked" else "pending",
            "evaluation_protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": protocol_sha,
            },
            "winner": {
                "path": str(args.winner.resolve()),
                "sha256": sha256_file(args.winner),
                "status": winner.get("status"),
                "selection": winner.get("selection"),
            },
            "frozen_untrained_test": baseline,
            "final_test": None,
        }
        if winner.get("status") == "passed":
            if args.test_root is None:
                raise ValueError("passed winner requires --test-root")
            reference = winner.get("checkpoint_config") or {}
            checkpoint_path = Path(str(reference.get("path") or ""))
            if (
                not checkpoint_path.is_file()
                or sha256_file(checkpoint_path) != reference.get("sha256")
            ):
                raise RuntimeError("winner checkpoint config is missing or changed")
            checkpoint = _load_checkpoint(checkpoint_path, protocol)
            metadata, paths = _audit_test(
                args.test_root,
                protocol=protocol,
                protocol_sha=protocol_sha,
                checkpoint=checkpoint,
            )
            summary = _summarize_rows(paths)
            conditions = {
                "overall_correct_strictly_higher": summary["correct"]
                > int(baseline["overall_correct"]),
                "each_dataset_not_lower": all(
                    int(summary["correct_by_dataset"][dataset])
                    >= int(baseline["per_dataset_correct"][dataset])
                    for dataset in DATASETS
                ),
                "total_tokens_strictly_lower": summary["mean_total_tokens"]
                < float(baseline["mean_total_tokens"]),
                "visual_tokens_strictly_lower": summary["mean_visual_tokens"]
                < float(baseline["mean_visual_tokens"]),
                "failure_rate_at_most_1pct": summary["failure_rate"] <= 0.01,
                "annotation_leak_zero": summary["annotation_leaks"] == 0,
                "candidate_rerun_zero": summary["candidate_reruns"] == 0,
            }
            total_ratio = summary["mean_total_tokens"] / float(
                baseline["mean_total_tokens"]
            )
            visual_ratio = summary["mean_visual_tokens"] / float(
                baseline["mean_visual_tokens"]
            )
            payload["status"] = "passed" if all(conditions.values()) else "completed_not_passed"
            payload["final_test"] = {
                "metadata_sha256": sha256_file(args.test_root / "evaluation_run.json"),
                "files": {dataset: str(path.resolve()) for dataset, path in paths.items()},
                "summary": summary,
                "conditions": conditions,
                "passed": all(conditions.values()),
                "total_token_ratio": total_ratio,
                "visual_token_ratio": visual_ratio,
                "clearly_reduced_30pct": total_ratio <= 0.70 and visual_ratio <= 0.70,
                "run_id": metadata.get("run_id"),
            }
        elif args.test_root is not None:
            raise ValueError("blocked winner must not have a Test300 root")
        _write_outputs(args.output_root, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

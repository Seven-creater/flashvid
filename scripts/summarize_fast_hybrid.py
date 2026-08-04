from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        rows[sample_id] = row
    return rows


def _metric(dataset: str, direct_path: Path, agent_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direct = _rows(direct_path)
    agent = _rows(agent_path)
    if set(direct) != set(agent):
        raise ValueError(
            f"{dataset} ID mismatch: direct={len(direct)}, agent={len(agent)}, "
            f"direct_only={len(set(direct) - set(agent))}, agent_only={len(set(agent) - set(direct))}"
        )
    ids = sorted(direct)
    direct_correct = sum(bool(direct[item].get("correct")) for item in ids)
    agent_correct = sum(bool(agent[item].get("correct")) for item in ids)
    fixed = [item for item in ids if not direct[item].get("correct") and agent[item].get("correct")]
    harmed = [item for item in ids if direct[item].get("correct") and not agent[item].get("correct")]
    fallback = sum(bool(agent[item].get("fallback_to_candidate")) for item in ids)
    errors = sum(bool(agent[item].get("error")) for item in ids)
    source_unavailable = sum(
        bool(agent[item].get("error")) and bool(agent[item].get("data_unavailable"))
        for item in ids
    )
    engineering_errors = errors - source_unavailable
    leaks = sum(agent[item].get("annotation_leak_check") != "passed" for item in ids)
    reruns = sum(int(agent[item].get("candidate_rerun", 0) or 0) for item in ids)
    visual = [float(agent[item].get("visual_tokens") or 0) for item in ids]
    total = [float(agent[item].get("total_tokens") or 0) for item in ids]
    latency = [float(agent[item].get("latency_s") or 0) for item in ids]
    common_ids = [
        item
        for item in ids
        if not direct[item].get("error") and not agent[item].get("error")
    ]
    badcases = [
        {
            "dataset": dataset,
            "sample_id": item,
            "kind": "direct_right_agent_wrong",
            "direct_prediction": direct[item].get("prediction"),
            "agent_prediction": agent[item].get("prediction"),
            "answer": direct[item].get("answer"),
            "tool_calls": agent[item].get("tool_calls", []),
        }
        for item in harmed
    ]
    return (
        {
            "dataset": dataset,
            "samples": len(ids),
            "direct_correct": direct_correct,
            "agent_correct": agent_correct,
            "gain": agent_correct - direct_correct,
            "common_valid_samples": len(common_ids),
            "direct_common_valid_correct": sum(
                bool(direct[item].get("correct")) for item in common_ids
            ),
            "agent_common_valid_correct": sum(
                bool(agent[item].get("correct")) for item in common_ids
            ),
            "direct_wrong_agent_right": len(fixed),
            "direct_right_agent_wrong": len(harmed),
            "fallback_to_candidate": fallback,
            "errors": errors,
            "source_data_unavailable": source_unavailable,
            "engineering_errors": engineering_errors,
            "annotation_leaks": leaks,
            "candidate_reruns": reruns,
            "mean_visual_tokens": mean(visual) if visual else 0.0,
            "mean_total_tokens": mean(total) if total else 0.0,
            "mean_latency_s": mean(latency) if latency else 0.0,
            "passed": (
                agent_correct > direct_correct
                and engineering_errors <= len(ids) * 0.01
                and leaks == 0
                and reruns == 0
            ),
            "direct_path": str(direct_path.resolve()),
            "agent_path": str(agent_path.resolve()),
        },
        badcases,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("DATASET", "DIRECT_JSONL", "AGENT_JSONL"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries: list[dict[str, Any]] = []
    badcases: list[dict[str, Any]] = []
    for dataset, direct, agent in args.pair:
        summary, cases = _metric(dataset, Path(direct), Path(agent))
        summaries.append(summary)
        badcases.extend(cases)
    result = {
        "datasets": summaries,
        "all_three_passed": len(summaries) == 3 and all(item["passed"] for item in summaries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    header = "| Dataset | Direct | Agent | Gain | 改对 | 改错 | Fallback | 工程错误 | 源视频缺失 | Visual Token | Total Token | Latency(s) | Pass |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    lines = [
        f"| {item['dataset']} | {item['direct_correct']}/{item['samples']} | "
        f"{item['agent_correct']}/{item['samples']} | {item['gain']:+d} | "
        f"{item['direct_wrong_agent_right']} | {item['direct_right_agent_wrong']} | "
        f"{item['fallback_to_candidate']} | {item['engineering_errors']} | "
        f"{item['source_data_unavailable']} | "
        f"{item['mean_visual_tokens']:.1f} | {item['mean_total_tokens']:.1f} | "
        f"{item['mean_latency_s']:.2f} | {'yes' if item['passed'] else 'no'} |"
        for item in summaries
    ]
    (args.output_dir / "summary.md").write_text(
        "# Fast Hybrid EVA\n\n" + header + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    badcase_lines = ["# Direct 正确但 Agent 改错\n"]
    for case in badcases:
        badcase_lines.append(
            f"- `{case['dataset']}:{case['sample_id']}`: Direct={case['direct_prediction']}, "
            f"Agent={case['agent_prediction']}, GT={case['answer']}"
        )
    (args.output_dir / "badcases.md").write_text(
        "\n".join(badcase_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

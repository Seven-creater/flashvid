from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct/agent evaluation JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows = []
    result_paths = (
        sorted(args.input_dir.glob("*_direct.jsonl"))
        + sorted(args.input_dir.glob("*_agent.jsonl"))
        + sorted(args.input_dir.glob("*_hybrid.jsonl"))
    )
    for path in result_paths:
        latest = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[str(record.get("sample_id"))] = record
        records = list(latest.values())
        if not records:
            continue
        correct = sum(bool(item.get("correct")) for item in records)
        visual_values = [float(item["visual_tokens"]) for item in records if item.get("visual_tokens") is not None]
        total_values = [float(item["total_tokens"]) for item in records if item.get("total_tokens") is not None]
        candidate_changed = sum(bool(item.get("candidate_changed")) for item in records)
        fallback_to_candidate = sum(bool(item.get("fallback_to_candidate")) for item in records)
        gate_triggered = sum(bool(item.get("change_gate_triggered")) for item in records)
        rows.append(
            {
                "dataset": records[0].get("dataset"),
                "backend": path.stem.rsplit("_", 1)[-1],
                "total": len(records),
                "correct": correct,
                "accuracy": correct / len(records),
                "errors": sum(bool(item.get("error")) for item in records),
                "mean_rounds": sum(float(item.get("rounds", 0) or 0) for item in records) / len(records),
                "mean_visual_tokens": sum(visual_values) / len(visual_values) if visual_values else None,
                "mean_total_tokens": sum(total_values) / len(total_values) if total_values else None,
                "candidate_changed": candidate_changed,
                "fallback_to_candidate": fallback_to_candidate,
                "change_gate_triggered": gate_triggered,
            }
        )
    if not rows:
        raise SystemExit(f"No result JSONL files found in {args.input_dir}")
    by_backend = {}
    for row in rows:
        by_backend.setdefault(row["backend"], []).append(row)
    if "direct" in by_backend:
        direct_mean = sum(row["accuracy"] for row in by_backend["direct"]) / len(by_backend["direct"])
        comparison = {
            f"{backend}_mean_delta_vs_direct": (
                sum(row["accuracy"] for row in backend_rows) / len(backend_rows)
            )
            - direct_mean
            for backend, backend_rows in by_backend.items()
            if backend != "direct"
        }
    else:
        comparison = None
    result = {"rows": rows, "comparison": comparison}
    output_json = args.output_json or args.input_dir / "summary.json"
    output_md = args.output_md or args.input_dir / "summary.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Evaluation summary",
        "",
        "| Dataset | Backend | Accuracy | Errors | Mean rounds | Mean visual tokens | Mean total tokens | Changed | Candidate fallback | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        visual = "n/a" if row["mean_visual_tokens"] is None else f"{row['mean_visual_tokens']:.0f}"
        total_tokens = "n/a" if row["mean_total_tokens"] is None else f"{row['mean_total_tokens']:.0f}"
        lines.append(
            f"| {row['dataset']} | {row['backend']} | {row['accuracy']:.2%} | "
            f"{row['errors']} | {row['mean_rounds']:.2f} | {visual} | {total_tokens} | "
            f"{row['candidate_changed']} | {row['fallback_to_candidate']} | {row['change_gate_triggered']} |"
        )
    if comparison:
        lines.extend([""])
        lines.extend(f"{key}: **{value:+.2%}**" for key, value in sorted(comparison.items()))
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

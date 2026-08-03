#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from flashvid_eval.qwen_dev_selection import canonical_sha256
from flashvid_eval.qwen_final_reporting import build_final_report


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content.encode("utf-8"))
    temporary.replace(path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen-only 长视频 Agent 固定工程测试集结果",
        "",
        "> 这 300 条样本已被用于多轮工程分析，因此不称为统计盲测集。",
        "",
        "| 方法 | 正确/300 | Accuracy | Visual Token | Total Token | Latency(s) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in report["methods"].items():
        nominal = item["nominal"]
        lines.append(
            f"| {method} | {nominal['correct']}/{nominal['denominator']} | "
            f"{_fmt(nominal['accuracy'])} | {_fmt(item['mean_visual_tokens'], 1)} | "
            f"{_fmt(item['mean_total_tokens'], 1)} | {_fmt(item['mean_latency_s'], 2)} | "
            f"{item['failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| 比较 | 共同有效 | 增益 | 改对 | 改错 | McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["comparisons"].items():
        lines.append(
            f"| {name} | {item['common_valid']} | {item['gain']:+d} | "
            f"{item['corrected']} | {item['regressed']} | "
            f"{_fmt(item['mcnemar_exact_p'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## 严格验收",
            "",
        ]
    )
    for key, passed in report["success_conditions"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{key}`")
    lines.append("")
    lines.append(f"总体结论：{'通过' if report['overall_success'] else '未通过'}。")
    lines.append("")
    return "\n".join(lines)


def _badcases_markdown(
    report: dict[str, Any], methods: dict[str, list[dict[str, Any]]]
) -> str:
    def one(prefix: str) -> str:
        return next(name for name in methods if name.startswith(prefix))

    pairs = [
        (one("q9_direct_"), "q9_best_untrained"),
        ("q9_best_untrained", one("sft9_")),
        ("q4_no_video", one("q4_direct_")),
    ]
    lines = ["# 固定工程测试集配对坏例", ""]
    for baseline_name, candidate_name in pairs:
        left = {
            (row["_dataset_identity"], row["sample_id"]): row
            for row in methods[baseline_name]
        }
        right = {
            (row["_dataset_identity"], row["sample_id"]): row
            for row in methods[candidate_name]
        }
        lines.extend([f"## {baseline_name} → {candidate_name}", ""])
        changed = []
        for identity in sorted(left.keys() & right.keys()):
            before, after = left[identity], right[identity]
            if bool(before.get("correct")) == bool(after.get("correct")):
                continue
            changed.append(
                (
                    identity,
                    "改对" if after.get("correct") else "改错",
                    before.get("prediction"),
                    after.get("prediction"),
                    before.get("answer"),
                )
            )
        lines.extend(
            [
                "| 数据集 | sample_id | 变化 | 原预测 | 新预测 | 答案 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for (dataset, sample_id), change, before, after, answer in changed:
            lines.append(
                f"| {dataset} | {sample_id} | {change} | {before} | {after} | {answer} |"
            )
        if not changed:
            lines.append("| - | - | 无变化 | - | - | - |")
        lines.append("")
    lines.append(f"报告哈希：`{report['report_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen Qwen final300 matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    report, methods = build_final_report(
        args.run_plan,
        canonical_sha256(config),
    )
    _atomic_write(
        args.output_dir / "summary.json",
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write(args.output_dir / "summary.md", _summary_markdown(report))
    _atomic_write(
        args.output_dir / "badcases.md",
        _badcases_markdown(report, methods),
    )
    print(
        json.dumps(
            {
                "overall_success": report["overall_success"],
                "report_sha256": report["report_sha256"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

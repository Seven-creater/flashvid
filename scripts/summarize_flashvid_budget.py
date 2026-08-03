from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flashvid_eval.budget_reporting import (
    DATASETS,
    MethodSpec,
    aggregate_dataset_metrics,
    exact_mcnemar,
    fixed_budget_curve,
    load_method_specs,
    pareto_acceptance,
    read_jsonl_all,
    read_jsonl_latest,
    resolve_result_path,
    summarize_records,
)

def _load_method(
    root: Path,
    spec: MethodSpec,
    expected_by_stage: dict[str, int],
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], ...]]]:
    datasets: dict[str, Any] = {}
    records_by_dataset: dict[str, tuple[dict[str, Any], ...]] = {}
    sources: dict[str, str | None] = {}
    expected = expected_by_stage.get(spec.stage)
    for dataset in DATASETS:
        path = resolve_result_path(root, spec.paths.get(dataset, ()))
        sources[dataset] = str(path.resolve()) if path else None
        if path is None:
            datasets[dataset] = summarize_records((), expected=expected)
            records_by_dataset[dataset] = ()
            continue
        loaded = read_jsonl_latest(path)
        datasets[dataset] = summarize_records(
            loaded.records,
            expected=expected,
            duplicates=loaded.duplicate_sample_ids,
            invalid_lines=loaded.invalid_lines,
        )
        records_by_dataset[dataset] = loaded.records
    aggregate = aggregate_dataset_metrics(datasets)
    return (
        {
            "label": spec.label or spec.name,
            "stage": spec.stage,
            "role": spec.role,
            "fixed_ratio": spec.fixed_ratio,
            "sources": sources,
            "datasets": datasets,
            "aggregate": aggregate,
            "mean_raw_visual_tokens": aggregate.get("mean_raw_visual_tokens"),
        },
        records_by_dataset,
    )


def _pairwise(
    methods: dict[str, dict[str, Any]],
    records: dict[str, dict[str, tuple[dict[str, Any], ...]]],
    specs: dict[str, MethodSpec],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    pairs: set[tuple[str, str]] = set()
    for name, spec in specs.items():
        reference = spec.reference
        if reference and reference != name:
            pairs.add((name, reference))
    for stage in ("validation", "final"):
        role_names = {
            spec.role: name
            for name, spec in specs.items()
            if spec.stage == stage and spec.role
        }
        sft = role_names.get("sft_dynamic")
        if sft:
            for reference_role in ("untrained_dynamic", "fixed_100"):
                reference = role_names.get(reference_role)
                if reference and reference != sft:
                    pairs.add((sft, reference))

    for name, reference in sorted(pairs):
        comparison_key = f"{name}__vs__{reference}"
        if reference not in methods:
            comparisons[comparison_key] = {
                "status": "pending",
                "method": name,
                "reference": reference,
                "reason": "reference_method_not_configured",
            }
            continue
        per_dataset = {
            dataset: exact_mcnemar(
                records[reference].get(dataset, ()),
                records[name].get(dataset, ()),
            )
            for dataset in DATASETS
        }
        baseline_all = [
            {**record, "sample_id": f"{dataset}:{record.get('sample_id')}"}
            for dataset in DATASETS
            for record in records[reference].get(dataset, ())
        ]
        method_all = [
            {**record, "sample_id": f"{dataset}:{record.get('sample_id')}"}
            for dataset in DATASETS
            for record in records[name].get(dataset, ())
        ]
        comparisons[comparison_key] = {
            "status": (
                "complete"
                if all(value["status"] == "complete" for value in per_dataset.values())
                else "partial"
                if any(value["status"] == "complete" for value in per_dataset.values())
                else "pending"
            ),
            "method": name,
            "reference": reference,
            "datasets": per_dataset,
            "aggregate": exact_mcnemar(baseline_all, method_all),
        }
    return comparisons


def _trajectory_summary(root: Path) -> dict[str, Any]:
    selected_root = root / "trajectories" / "selected"
    if not selected_root.is_dir():
        return {"status": "pending"}
    rows: list[dict[str, Any]] = []
    no_positive = 0
    expected_answers = 0
    route_counts: dict[str, int] = {}
    for dataset in DATASETS:
        path = selected_root / f"{dataset}.jsonl"
        if path.is_file():
            rows.extend(read_jsonl_all(path))
        summary_path = selected_root / f"{dataset}_summary.json"
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            selection = payload.get("selection") or {}
            inputs = payload.get("inputs") or {}
            no_positive += int(selection.get("no_positive_count") or 0)
            expected_answers += int(inputs.get("answer_count") or 0)
            for route, count in (selection.get("no_positive_route_counts") or {}).items():
                route_counts[route] = route_counts.get(route, 0) + int(count)
    if not rows and not expected_answers:
        return {"status": "pending"}
    metrics = summarize_records(rows)
    primary_ids = {
        (str(row.get("dataset")), str(row.get("sample_id")))
        for row in rows
        if row.get("_selection_role") == "primary"
    }
    largest_share = metrics["budget"].get("largest_ratio_share")
    conditions = {
        "at_least_300_positive_training_samples": len(primary_ids) >= 300,
        "at_least_three_budget_levels_selected": (
            metrics["budget"].get("distinct_ratios", 0) >= 3
        ),
        "no_budget_level_above_70pct": (
            largest_share is not None and largest_share <= 0.70
        ),
    }
    return {
        "status": "complete" if rows else "partial",
        "selected_records": len(rows),
        "positive_training_samples": len(primary_ids),
        "expected_training_samples": expected_answers or None,
        "no_positive_trajectory": no_positive,
        "no_positive_rate": (
            no_positive / expected_answers if expected_answers else None
        ),
        "no_positive_route_counts": route_counts,
        "budget": metrics["budget"],
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def _engineering_acceptance(method: dict[str, Any]) -> dict[str, Any]:
    aggregate = method["aggregate"]
    if aggregate.get("status") != "complete":
        return {"status": "pending", "passed": None}
    harms = {
        dataset: int(method["datasets"][dataset]["candidate"]["changes_harmed"])
        for dataset in DATASETS
    }
    failure_rate = aggregate.get(
        "engineering_failure_rate", aggregate.get("failure_rate")
    )
    conditions = {
        "annotation_leak_zero_and_checked_for_every_record": (
            aggregate.get("annotation_leak_failures") == 0
            and aggregate.get("annotation_leak_passed") == aggregate.get("records")
        ),
        "duplicate_sample_ids_zero": aggregate.get("duplicate_sample_id_count") == 0,
        "candidate_rerun_zero": aggregate.get("candidate_rerun_total") == 0,
        "candidate_harmed_at_most_one_per_dataset": all(
            count <= 1 for count in harms.values()
        ),
        "failure_rate_at_most_one_percent": (
            failure_rate is not None and failure_rate <= 0.01
        ),
    }
    return {
        "status": "complete",
        "passed": all(conditions.values()),
        "candidate_changes_harmed": harms,
        "conditions": conditions,
    }


def summarize(
    root: Path,
    specs: tuple[MethodSpec, ...],
    expected_by_stage: dict[str, int],
) -> dict[str, Any]:
    spec_map = {spec.name: spec for spec in specs}
    methods: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for spec in specs:
        methods[spec.name], records[spec.name] = _load_method(
            root, spec, expected_by_stage
        )
    acceptance = {
        stage: pareto_acceptance(methods, spec_map, stage)
        for stage in ("validation", "final")
    }
    sft_methods = [
        name for name, spec in spec_map.items()
        if spec.role == "sft_dynamic"
    ]
    return {
        "schema_version": 1,
        "root": str(root.resolve()),
        "expected_samples_per_dataset": expected_by_stage,
        "methods": methods,
        "paired_vs_reference": _pairwise(methods, records, spec_map),
        "fixed_budget_curves": {
            stage: fixed_budget_curve(methods, spec_map, stage)
            for stage in ("validation", "final")
        },
        "trajectory_data": _trajectory_summary(root),
        "pareto_acceptance": acceptance,
        "engineering_acceptance": {
            name: _engineering_acceptance(methods[name]) for name in sft_methods
        },
        "checkpoint_selection": (
            json.loads(
                (root / "checkpoints" / "validation" / "checkpoint_selection.json")
                .read_text(encoding="utf-8")
            )
            if (root / "checkpoints" / "validation" / "checkpoint_selection.json").is_file()
            else {"status": "pending", "selected_checkpoint": None}
        ),
        "non_result_acceptance": {
            "ratio_1_plugin_native_parity": {
                "status": "pending",
                "note": "requires the dedicated observation-parity artifact",
            },
            "long_run_setsids_nohup_resume": {
                "status": "pending",
                "note": "requires launcher/run metadata rather than inference JSONL",
            },
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# FlashVID Dynamic-Budget SFT Summary",
        "",
        "缺失或未完成的方法统一标记为 `pending`，不会按 0 分伪装成已完成实验。",
        "",
        "## Evaluation matrix",
        "",
        "| Stage | Method | Status | Raw correct / N | Raw accuracy | Available correct / N | Available accuracy | Source unavailable | Retained visual tokens | Engineering failures | Fixed candidate | Harmed candidate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in summary["methods"].items():
        agg = method["aggregate"]
        lines.append(
            f"| {method['stage']} | {method['label']} | {agg.get('status', 'pending')} | "
            f"{agg.get('correct', 0)}/{agg.get('records', 0)} | "
            f"{_fmt(agg.get('accuracy'))} | "
            f"{agg.get('available_correct', agg.get('correct', 0))}/"
            f"{agg.get('available_records', agg.get('records', 0))} | "
            f"{_fmt(agg.get('accuracy_available', agg.get('accuracy')))} | "
            f"{agg.get('data_unavailable', 0)} | "
            f"{_fmt(agg.get('mean_retained_visual_tokens'))} | "
            f"{agg.get('engineering_failures', agg.get('failures', 0))} | "
            f"{agg.get('candidate_changes_fixed', 0)} | "
            f"{agg.get('candidate_changes_harmed', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Runtime and token cost",
            "",
            "| Stage | Method | Rounds | Tool calls | Raw visual | Retained visual | Agent tokens | Perception tokens | Agent latency (s) | Perception latency (s) | Total latency (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in summary["methods"].values():
        aggregate = method["aggregate"]
        runtime = aggregate.get("runtime") or {}
        tokens = aggregate.get("tokens") or {}
        lines.append(
            f"| {method['stage']} | {method['label']} | "
            f"{_fmt(runtime.get('mean_rounds'))} | "
            f"{_fmt(runtime.get('mean_tool_calls'))} | "
            f"{_fmt(tokens.get('mean_raw_visual_tokens'))} | "
            f"{_fmt(tokens.get('mean_retained_visual_tokens'))} | "
            f"{_fmt(tokens.get('mean_controller_tokens'))} | "
            f"{_fmt(tokens.get('mean_perception_tokens'))} | "
            f"{_fmt(runtime.get('mean_controller_latency_s'))} | "
            f"{_fmt(runtime.get('mean_perception_latency_s'))} | "
            f"{_fmt(runtime.get('mean_latency_s'))} |"
        )

    lines.extend(["", "## Candidate audit", ""])
    lines.extend(
        [
            "| Stage | Method | Valid candidate | Normalized recovery | Changed | Changed right | Changed wrong | Wrong-to-wrong | Fallback |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in summary["methods"].values():
        totals = {
            key: sum(
                int(dataset["candidate"].get(key) or 0)
                for dataset in method["datasets"].values()
            )
            for key in (
                "valid",
                "normalized_recovered",
                "changed",
                "changes_fixed",
                "changes_harmed",
                "changed_wrong_to_wrong",
                "fallback_to_candidate",
            )
        }
        lines.append(
            f"| {method['stage']} | {method['label']} | {totals['valid']} | "
            f"{totals['normalized_recovered']} | {totals['changed']} | "
            f"{totals['changes_fixed']} | {totals['changes_harmed']} | "
            f"{totals['changed_wrong_to_wrong']} | {totals['fallback_to_candidate']} |"
        )

    lines.extend(["", "## Fixed-budget curves", ""])
    for stage, curve in summary["fixed_budget_curves"].items():
        lines.extend(
            [
                f"### {stage}",
                "",
                "| Ratio | Method | Status | Correct | Accuracy | Retained tokens | Pareto-efficient |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for point in curve["points"]:
            lines.append(
                f"| {point['retention_ratio']:.0%} | {point['method']} | "
                f"{point['status']} | {_fmt(point['correct'])} | "
                f"{_fmt(point['accuracy'])} | "
                f"{_fmt(point['mean_retained_visual_tokens'])} | "
                f"{_fmt(point['pareto_efficient'])} |"
            )
        lines.append("")

    lines.extend(["## Paired McNemar comparisons", ""])
    for _, comparison in summary["paired_vs_reference"].items():
        aggregate = comparison.get("aggregate") or {}
        lines.append(
            f"- `{comparison.get('method')}` vs `{comparison.get('reference')}`: "
            f"status={comparison.get('status')}, common-valid="
            f"{aggregate.get('paired_common_valid', 0)}, "
            f"wins/losses={aggregate.get('wins_baseline_wrong_method_correct', 0)}/"
            f"{aggregate.get('losses_baseline_correct_method_wrong', 0)}, "
            f"exact p={_fmt(aggregate.get('mcnemar_exact_two_sided_p'), 4)}."
        )

    lines.extend(["", "## SFT Pareto acceptance", ""])
    for stage, result in summary["pareto_acceptance"].items():
        lines.append(
            f"- {stage}: status=`{result.get('status')}`, passed=`{result.get('passed')}`."
        )
        for key, value in (result.get("conditions") or {}).items():
            lines.append(f"  - {key}: `{value}`")

    trajectory = summary["trajectory_data"]
    lines.extend(
        [
            "",
            "## Trajectory selection",
            "",
            f"- Status: `{trajectory.get('status')}`",
            f"- Positive training samples: `{trajectory.get('positive_training_samples', 'pending')}`",
            f"- No-positive trajectories: `{trajectory.get('no_positive_trajectory', 'pending')}`",
            f"- Budget distribution: `{json.dumps(trajectory.get('budget', {}).get('step_ratio_counts', {}), ensure_ascii=False)}`",
            "",
            "## Checkpoint selection",
            "",
        ]
    )
    selection = summary["checkpoint_selection"].get(
        "selection", summary["checkpoint_selection"]
    )
    lines.append(
        f"- Status: `{selection.get('status', 'pending')}`; selected: "
        f"`{selection.get('selected_checkpoint')}`."
    )
    lines.extend(
        [
            "",
            "## Non-result checks",
            "",
            "这些项目不能从推理 JSONL 可靠推断，缺少专用证据时保持 pending：",
        ]
    )
    for key, value in summary["non_result_acceptance"].items():
        lines.append(f"- {key}: `{value['status']}` — {value['note']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered FlashVID budget report. Missing files are "
            "reported as pending instead of silently omitted."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval/flashvid_budget_v1"),
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        help="Optional JSON method matrix; relative paths resolve under --root.",
    )
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--final-count", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    expected = {
        "validation": args.validation_count,
        "final": args.final_count,
    }
    result = summarize(args.root, load_method_specs(args.method_config), expected)
    output_json = args.output_json or args.root / "summary.json"
    output_md = args.output_md or args.root / "summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".partial")
    temporary_md = output_md.with_suffix(output_md.suffix + ".partial")
    temporary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_md.write_text(markdown(result), encoding="utf-8")
    temporary_json.replace(output_json)
    temporary_md.replace(output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

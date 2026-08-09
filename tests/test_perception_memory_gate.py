from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashvid_eval.perception_memory_gate import (
    DATASETS,
    MethodRuns,
    PerceptionMemoryGateError,
    SeedRun,
    evaluate_dev_gate,
    evaluate_test_gate,
)
from scripts.gate_perception_memory import evaluate_config


EXPECTED_MANIFESTS = {
    dataset: f"{DATASETS.index(dataset) + 2}" * 64 for dataset in DATASETS
}
DEV_SEEDS = (17, 42, 73)


def _artifact(method_id: str) -> str:
    return hashlib.sha256(method_id.encode("utf-8")).hexdigest()


def _row(
    dataset: str,
    index: int,
    *,
    correct: bool,
    total_tokens: int,
    visual_tokens: int,
    incomplete_stop: bool,
    seed: int = 42,
    model_artifact_sha256: str = "a" * 64,
) -> dict:
    answer = "A"
    prediction = "A" if correct else "B"
    return {
        "dataset": dataset,
        "sample_id": f"{dataset}-{index:03d}",
        "answer": answer,
        "prediction": prediction,
        "candidate_answer": "B",
        "candidate_changed": prediction != "B",
        "fallback_to_candidate": False,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "backend": "perception_memory_eva",
        "agent_version": "perception_memory_eva_v1",
        "implementation_sha256": "1" * 64,
        "implementation_bundle_sha256": "6" * 64,
        "frame_tool_identity": {"kind": "official_eva", "commit": "7" * 40},
        "manifest_sha256": f"{DATASETS.index(dataset) + 2}" * 64,
        "candidate_results_sha256": f"{DATASETS.index(dataset) + 5}" * 64,
        "experiment_config_sha256": "8" * 64,
        "diagnostics_gate_sha256": "9" * 64,
        "model_artifact_sha256": model_artifact_sha256,
        "max_turns": 6,
        "max_frames_per_call": 128,
        "controller_max_tokens": 512,
        "perception_max_tokens": 1024,
        "judge_max_tokens": 512,
        "seed": seed,
        "stop_reason": (
            "max_turns_incomplete" if incomplete_stop else "evidence_complete"
        ),
        "evidence_complete": not incomplete_stop,
        "candidate_cost_complete": True,
        "end_to_end_total_tokens": total_tokens,
        "end_to_end_total_tokens_complete": True,
        "end_to_end_visual_tokens": visual_tokens,
        "end_to_end_visual_tokens_complete": True,
        "error": None,
    }


def _write_method(
    tmp_path: Path,
    method_id: str,
    seeds: tuple[int, ...],
    counts: dict[str, int],
    correct_counts: dict[str, int],
    *,
    total_tokens: int,
    visual_tokens: int,
    incomplete_stops: int,
    model_artifact_sha256: str | None = None,
) -> MethodRuns:
    artifact_sha256 = model_artifact_sha256 or _artifact(method_id)
    runs: list[SeedRun] = []
    for seed in seeds:
        paths: dict[str, Path] = {}
        remaining_incomplete = incomplete_stops
        for dataset in DATASETS:
            path = tmp_path / method_id / str(seed) / f"{dataset}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows: list[dict] = []
            for index in range(counts[dataset]):
                incomplete = remaining_incomplete > 0
                remaining_incomplete -= int(incomplete)
                rows.append(
                    _row(
                        dataset,
                        index,
                        correct=index < correct_counts[dataset],
                        total_tokens=total_tokens,
                        visual_tokens=visual_tokens,
                        incomplete_stop=incomplete,
                        seed=seed,
                        model_artifact_sha256=artifact_sha256,
                    )
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            paths[dataset] = path
        runs.append(SeedRun(seed, paths))
    return MethodRuns(method_id, tuple(runs))


def _config_method(method: MethodRuns) -> dict:
    return {
        "method_id": method.method_id,
        "runs": [
            {
                "seed": run.seed,
                "paths": {key: str(value) for key, value in run.result_paths.items()},
            }
            for run in method.runs
        ],
    }


def _write_dev_gate_report(
    tmp_path: Path,
    baseline: MethodRuns,
    candidate: MethodRuns,
) -> tuple[Path, str]:
    payload = {
        "schema_version": 1,
        "phase": "dev",
        "status": "passed",
        "passed": True,
        "selected_method_id": candidate.method_id,
        "baseline": {
            "method_id": baseline.method_id,
            "model_artifact_sha256": _artifact(baseline.method_id),
        },
        "candidates": [
            {
                "method_id": candidate.method_id,
                "model_artifact_sha256": _artifact(candidate.method_id),
                "passed": True,
            }
        ],
    }
    path = tmp_path / "dev_gate_report.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_dev_gate_uses_all_seeds_and_selects_unique_deterministic_winner(
    tmp_path: Path,
) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=3,
    )
    higher_cost = _write_method(
        tmp_path,
        "checkpoint-1",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=70,
        visual_tokens=56,
        incomplete_stops=1,
    )
    winner = _write_method(
        tmp_path,
        "checkpoint-2",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )

    report = evaluate_dev_gate(
        baseline=baseline,
        candidates=[higher_cost, winner],
        expected_manifest_sha256=EXPECTED_MANIFESTS,
        expected_counts=counts,
    )

    assert report["passed"] is True
    assert report["selected_method_id"] == "checkpoint-2"
    assert report["baseline"]["summary"]["seed_count"] == 3
    selected = next(
        item for item in report["candidates"] if item["method_id"] == "checkpoint-2"
    )
    assert selected["total_token_ratio"] == pytest.approx(0.6)
    assert selected["visual_token_ratio"] == pytest.approx(0.5)
    assert all(selected["conditions"].values())
    json.dumps(report)


def test_dev_gate_blocks_failures_and_incomplete_stops_not_decreasing(
    tmp_path: Path,
) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=1,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=1,
    )
    first_path = candidate.runs[0].result_paths["lvbench"]
    rows = [json.loads(line) for line in first_path.read_text().splitlines()]
    rows[0]["error"] = "API failed"
    first_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = evaluate_dev_gate(
        baseline=baseline,
        candidates=[candidate],
        expected_manifest_sha256=EXPECTED_MANIFESTS,
        expected_counts=counts,
    )
    point = report["candidates"][0]
    assert report["passed"] is False
    assert point["conditions"]["failure_rate_at_most_1pct"] is False
    assert point["conditions"]["incomplete_stops_decrease"] is False
    assert point["paired_costs"]["joint_complete_samples"] == 17


@pytest.mark.parametrize(
    ("field", "value", "condition"),
    [
        ("annotation_leak_check", "failed", "annotation_leak_zero"),
        ("candidate_rerun", 1, "candidate_rerun_zero"),
    ],
)
def test_dev_gate_rejects_leak_and_candidate_rerun(
    tmp_path: Path, field: str, value: object, condition: str
) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["lvbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = evaluate_dev_gate(
        baseline=baseline,
        candidates=[candidate],
        expected_manifest_sha256=EXPECTED_MANIFESTS,
        expected_counts=counts,
    )
    assert report["passed"] is False
    assert report["candidates"][0]["conditions"][condition] is False


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=1,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["cgbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["sample_id"] = rows[0]["sample_id"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(PerceptionMemoryGateError, match="duplicate sample_id"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_mismatched_seed_or_sample_scope_is_rejected(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=2,
    )
    wrong_seeds = _write_method(
        tmp_path,
        "wrong-seeds",
        (17, 42),
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    with pytest.raises(PerceptionMemoryGateError, match="requires seeds"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[wrong_seeds],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )

    wrong_scope = _write_method(
        tmp_path,
        "wrong-scope",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    path = wrong_scope.runs[0].result_paths["lvbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_id"] = "different-sample"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(PerceptionMemoryGateError, match="sample scope mismatch"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[wrong_scope],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent_version", "different-runtime"),
        ("implementation_bundle_sha256", "f" * 64),
    ],
)
def test_same_runtime_audit_rejects_changed_runtime_bundle(
    tmp_path: Path, field: str, value: str
) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["lsdbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(PerceptionMemoryGateError, match="runtime field mismatch"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_final_test_gate_enforces_accuracy_floors_and_30_percent_reduction(
    tmp_path: Path,
) -> None:
    counts = {dataset: 100 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {"lvbench": 45, "lsdbench": 63, "cgbench": 43},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=20,
    )
    candidate = _write_method(
        tmp_path,
        "selected-sft",
        (42,),
        counts,
        {"lvbench": 47, "lsdbench": 65, "cgbench": 45},
        total_tokens=70,
        visual_tokens=56,
        incomplete_stops=5,
    )

    dev_report, dev_report_sha256 = _write_dev_gate_report(
        tmp_path, baseline, candidate
    )
    report = evaluate_test_gate(
        baseline=baseline,
        candidate=candidate,
        expected_manifest_sha256=EXPECTED_MANIFESTS,
        dev_gate_report_path=dev_report,
        dev_gate_report_sha256=dev_report_sha256,
    )
    assert report["passed"] is True
    assert report["candidate"]["summary"]["mean_correct"] == 157
    assert report["total_token_ratio"] == pytest.approx(0.7)
    assert report["visual_token_ratio"] == pytest.approx(0.7)
    assert all(report["conditions"].values())


def test_config_interface_produces_machine_readable_dev_report(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=100,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=60,
        incomplete_stops=0,
    )
    report = evaluate_config(
        {
            "phase": "dev",
            "expected_counts": counts,
            "expected_manifest_sha256": EXPECTED_MANIFESTS,
            "baseline": _config_method(baseline),
            "candidates": [_config_method(candidate)],
        }
    )
    assert report["passed"] is True
    assert json.loads(json.dumps(report))["selected_method_id"] == "checkpoint"


@pytest.mark.parametrize(
    "field",
    [
        "candidate_cost_complete",
        "end_to_end_total_tokens_complete",
        "end_to_end_visual_tokens_complete",
    ],
)
def test_gate_rejects_incomplete_end_to_end_cost_fields(
    tmp_path: Path, field: str
) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["lvbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(PerceptionMemoryGateError, match=f"{field} must be true"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_gate_never_falls_back_to_agent_only_token_fields(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["lvbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].pop("end_to_end_total_tokens")
    rows[0].pop("end_to_end_total_tokens_complete")
    rows[0]["total_tokens"] = 1
    rows[0]["total_token_accounting_complete"] = True
    rows[0]["agent_total_tokens_complete"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(
        PerceptionMemoryGateError,
        match="end_to_end_total_tokens_complete must be true",
    ):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_dev_gate_requires_exact_preregistered_seed_set(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=1,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        (42,),
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )
    with pytest.raises(
        PerceptionMemoryGateError, match=r"requires seeds \[17, 42, 73\]"
    ):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_gate_rejects_mixed_model_artifacts_within_method(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["cgbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["model_artifact_sha256"] = "f" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(PerceptionMemoryGateError, match="exactly one model artifact"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_gate_rejects_manifest_not_pinned_by_frozen_config(tmp_path: Path) -> None:
    counts = {dataset: 2 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        DEV_SEEDS,
        counts,
        {dataset: 1 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=3,
    )
    candidate = _write_method(
        tmp_path,
        "checkpoint",
        DEV_SEEDS,
        counts,
        {dataset: 2 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=0,
    )
    path = candidate.runs[0].result_paths["lsdbench"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["manifest_sha256"] = "f" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(PerceptionMemoryGateError, match="frozen config"):
        evaluate_dev_gate(
            baseline=baseline,
            candidates=[candidate],
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            expected_counts=counts,
        )


def test_test_gate_requires_seed42_and_pinned_dev_selection(tmp_path: Path) -> None:
    counts = {dataset: 100 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {dataset: 50 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=10,
    )
    candidate = _write_method(
        tmp_path,
        "selected-sft",
        (17,),
        counts,
        {dataset: 52 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=1,
    )
    dev_report, dev_sha256 = _write_dev_gate_report(tmp_path, baseline, candidate)
    with pytest.raises(PerceptionMemoryGateError, match=r"requires seeds \[42\]"):
        evaluate_test_gate(
            baseline=baseline,
            candidate=candidate,
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            dev_gate_report_path=dev_report,
            dev_gate_report_sha256=dev_sha256,
        )


@pytest.mark.parametrize("failure", ["sha256", "selected_method"])
def test_test_gate_rejects_unbound_dev_report(
    tmp_path: Path, failure: str
) -> None:
    counts = {dataset: 100 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {dataset: 50 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=10,
    )
    candidate = _write_method(
        tmp_path,
        "selected-sft",
        (42,),
        counts,
        {dataset: 52 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=1,
    )
    dev_report, dev_sha256 = _write_dev_gate_report(tmp_path, baseline, candidate)
    expected_error = "SHA-256"
    if failure == "sha256":
        dev_sha256 = "0" * 64
    else:
        payload = json.loads(dev_report.read_text(encoding="utf-8"))
        payload["selected_method_id"] = "different-checkpoint"
        dev_report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        dev_sha256 = hashlib.sha256(dev_report.read_bytes()).hexdigest()
        expected_error = "selected_method_id"
    with pytest.raises(PerceptionMemoryGateError, match=expected_error):
        evaluate_test_gate(
            baseline=baseline,
            candidate=candidate,
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            dev_gate_report_path=dev_report,
            dev_gate_report_sha256=dev_sha256,
        )


def test_test_gate_rejects_model_artifact_not_selected_on_dev(tmp_path: Path) -> None:
    counts = {dataset: 100 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {dataset: 50 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=10,
    )
    candidate = _write_method(
        tmp_path,
        "selected-sft",
        (42,),
        counts,
        {dataset: 52 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=1,
        model_artifact_sha256="e" * 64,
    )
    dev_report, dev_sha256 = _write_dev_gate_report(tmp_path, baseline, candidate)
    with pytest.raises(PerceptionMemoryGateError, match="candidate model artifact"):
        evaluate_test_gate(
            baseline=baseline,
            candidate=candidate,
            expected_manifest_sha256=EXPECTED_MANIFESTS,
            dev_gate_report_path=dev_report,
            dev_gate_report_sha256=dev_sha256,
        )


def test_config_test_phase_requires_and_records_dev_gate_binding(tmp_path: Path) -> None:
    counts = {dataset: 100 for dataset in DATASETS}
    baseline = _write_method(
        tmp_path,
        "untrained",
        (42,),
        counts,
        {dataset: 50 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
        incomplete_stops=10,
    )
    candidate = _write_method(
        tmp_path,
        "selected-sft",
        (42,),
        counts,
        {dataset: 52 for dataset in DATASETS},
        total_tokens=60,
        visual_tokens=40,
        incomplete_stops=1,
    )
    dev_report, dev_sha256 = _write_dev_gate_report(tmp_path, baseline, candidate)
    report = evaluate_config(
        {
            "phase": "test",
            "expected_manifest_sha256": EXPECTED_MANIFESTS,
            "baseline": _config_method(baseline),
            "candidate": _config_method(candidate),
            "dev_gate_report": {
                "path": str(dev_report),
                "sha256": dev_sha256,
            },
        }
    )
    assert report["dev_gate_binding"] == {
        "path": str(dev_report.resolve()),
        "sha256": dev_sha256,
        "selected_method_id": "selected-sft",
    }

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import run_qwen_agent_search as search_runner

from flashvid_eval.qwen_dev_selection import (
    DATASETS,
    REQUIRED_AGENT_SEEDS,
    SUPPORTED_STAGES,
    build_dev_selection_report,
    build_frozen_winner,
    canonical_sha256,
    file_sha256,
    load_dev_runs,
    load_protocol_smoke_rejection,
    _row_failed,
    write_frozen_json,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_qwen_agent_dev_winner.py"
PROTOCOL_SCRIPT = Path(__file__).parents[1] / "scripts" / "select_qwen_protocol_dev.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(tmp_path: Path) -> dict:
    datasets = {}
    for dataset in DATASETS:
        manifest = tmp_path / "manifests" / f"{dataset}_dev.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "dataset": dataset,
                        "sample_id": f"{dataset}-{index}",
                        "video": f"{dataset}-{index}.mp4",
                        "question": "Q",
                        "choices": {"A": "a", "B": "b"},
                        "answer": "A",
                    }
                )
                + "\n"
                for index in range(3)
            ),
            encoding="utf-8",
        )
        datasets[dataset] = {
            "dev": {"path": str(manifest), "sha256": file_sha256(manifest)}
        }
    return {
        "schema_version": 1,
        "experiment_id": "selection-test",
        "result_root": str(tmp_path / "results"),
        "source_workspace": str(tmp_path),
        "datasets": datasets,
        "models": {
            "q4": {"served_name": "Qwen3.5-4B"},
            "q9": {"served_name": "Qwen3.5-9B"},
        },
        "protocols": {"no_think": {}, "think": {}},
        "direct_sampling": [
            {"id": "uniform32"},
            {"id": "uniform64"},
            {"id": "uniform128"},
            {"id": "fps2"},
        ],
        "agent_search": {
            "framework_order": list(SUPPORTED_STAGES),
            "overview_frames": [32],
            "local_fps": [1.0],
            "max_intervals": [2],
            "max_turns": [4],
            "seeds": list(REQUIRED_AGENT_SEEDS),
        },
        "execution": {"sample_seed": 42},
    }


def _scores(counts: tuple[int, int, int], dataset: str) -> int:
    return counts[DATASETS.index(dataset)]


def _task(
    tmp_path: Path,
    config: dict,
    *,
    phase: str,
    dataset: str,
    model_key: str,
    protocol: str,
    seed: int,
    correct_count: int,
    tokens: int,
    suffix: str,
    mode: str | None = None,
    sampling: str | None = None,
    strategy: str | None = None,
    variant_id: str | None = None,
    leak: bool = False,
) -> dict:
    model = config["models"][model_key]["served_name"]
    manifest = Path(config["datasets"][dataset]["dev"]["path"])
    output_dir = tmp_path / "outputs" / phase / suffix / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "scripts/evaluate_mcq.py",
        "--qwen-protocol",
        protocol,
        "--seed",
        str(seed),
    ]
    agent_path = None
    agent_hash = None
    if mode is not None:
        command.extend(["--baseline-mode", mode])
    if sampling is not None:
        command.extend(["--direct-sampling", sampling])
    if strategy is not None:
        assert variant_id is not None
        agent_path = tmp_path / "agent_configs" / strategy / f"{variant_id}.json"
        if not agent_path.exists():
            _write_json(
                agent_path,
                {
                    "schema_version": 1,
                    "search_variant": {"variant_id": variant_id},
                    "agent": {"strategy": strategy},
                },
            )
        agent_hash = file_sha256(agent_path)
        command.extend(["--agent-config", str(agent_path)])

    fingerprint = canonical_sha256(
        {"phase": phase, "suffix": suffix, "dataset": dataset, "seed": seed}
    )
    rows = []
    for index in range(3):
        correct = index < correct_count
        row = {
            "dataset": dataset,
            "sample_id": f"{dataset}-{index}",
            "model": model,
            "strategy": strategy,
            "prediction": "A" if correct else "B",
            "answer": "A",
            "correct": correct,
            "total_tokens": tokens,
            "annotation_leak_check": "failed" if leak and index == 0 else "passed",
            "candidate_rerun": 0,
            "run_fingerprint": fingerprint,
        }
        if mode is not None:
            row["baseline_mode"] = mode
        if mode == "direct":
            assert sampling is not None
            sampled_frames = {
                "uniform32": 32,
                "uniform64": 64,
                "uniform128": 128,
                "fps2": 768,
            }[sampling]
            row.update(
                {
                    "sampled_frames_estimated": sampled_frames,
                    "sampled_frames_actual": sampled_frames,
                    "visual_usage_complete": True,
                }
            )
        rows.append(row)
    result = output_dir / f"{dataset}_result.jsonl"
    result.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    frozen = {
        "dataset": dataset,
        "manifest": {
            "path": str(manifest),
            "sha256": file_sha256(manifest),
        },
        "model": model,
        "model_artifact_sha256": model_key[1] * 64,
        "experiment_config_sha256": canonical_sha256(config),
        "qwen_protocol": protocol,
        "seed": seed,
        "run_fingerprint": fingerprint,
    }
    if agent_path is not None:
        frozen["agent_config"] = {"path": str(agent_path), "sha256": agent_hash}
    _write_json(output_dir / f"frozen_inputs_{dataset}_result.json", frozen)
    return {
        "phase": phase,
        "task_id": f"{suffix}-{dataset}",
        "dataset": dataset,
        "split": "dev",
        "model_key": model_key,
        "model": model,
        "model_artifact_sha256": model_key[1] * 64,
        "base_url": "http://127.0.0.1:1/v1",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "output_dir": str(output_dir),
        "resume": True,
        "concurrency": 1,
        "agent_config_sha256": agent_hash,
        "command": command,
    }


def _plan(
    tmp_path: Path,
    config: dict,
    phase: str,
    tasks: list[dict],
    *,
    framework: str | None = None,
    variant: str | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": phase,
        "model_filter": None,
        "protocol_filter": None,
        "framework_filter": framework,
        "search_variant_id": variant,
        "config_path": str(tmp_path / "config.json"),
        "config_sha256": canonical_sha256(config),
        "frozen_winner": None,
        "disabled_frameworks": [],
        "task_count": len(tasks),
        "tasks": tasks,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    path = tmp_path / "plans" / f"{phase}_{framework or 'all'}_{variant or 'all'}.json"
    _write_json(path, payload)
    return path


@pytest.mark.parametrize(
    "invalid_fields",
    (
        {"sampled_frames_actual": 63},
        {"sampled_frames_actual": None},
        {"sampled_frames_estimated": None},
        {"visual_usage_complete": False},
        {"visual_usage_complete": None},
    ),
)
def test_direct_row_requires_executed_frame_match_and_complete_visual_usage(
    invalid_fields: dict[str, object],
) -> None:
    row = {
        "baseline_mode": "direct",
        "prediction": "A",
        "sampled_frames_estimated": 64,
        "sampled_frames_actual": 64,
        "visual_usage_complete": True,
    }
    assert _row_failed(row) is False
    assert _row_failed({**row, **invalid_fields}) is True


def test_non_direct_row_does_not_require_direct_media_accounting() -> None:
    assert (
        _row_failed(
            {
                "strategy": "a1_storyboard_zoom",
                "prediction": "A",
                "sampled_frames_estimated": 64,
                "sampled_frames_actual": None,
                "visual_usage_complete": False,
            }
        )
        is False
    )


def _complete_matrix(tmp_path: Path, *, leak_stage: str | None = None) -> tuple[dict, list[Path]]:
    config = _config(tmp_path)
    protocol_tasks = []
    for model_key in ("q4", "q9"):
        for protocol, counts, tokens in (
            ("no_think", (1, 1, 1), 50),
            ("think", (2, 1, 1), 80),
        ):
            for dataset in DATASETS:
                protocol_tasks.append(
                    _task(
                        tmp_path,
                        config,
                        phase="protocol_audit",
                        dataset=dataset,
                        model_key=model_key,
                        protocol=protocol,
                        seed=42,
                        correct_count=_scores(counts, dataset),
                        tokens=tokens,
                        suffix=f"{model_key}-{protocol}",
                        mode="direct",
                        sampling="uniform64",
                    )
                )
    plans = [_plan(tmp_path, config, "protocol_audit", protocol_tasks)]

    direct_tasks = []
    direct_profiles = {
        "uniform32": ((1, 1, 1), 80),
        "uniform64": ((1, 1, 1), 100),
        "uniform128": ((1, 1, 0), 150),
        "fps2": ((1, 0, 0), 300),
    }
    for model_key in ("q4", "q9"):
        for sampling, (counts, tokens) in direct_profiles.items():
            for dataset in DATASETS:
                direct_tasks.append(
                    _task(
                        tmp_path,
                        config,
                        phase="direct_dev",
                        dataset=dataset,
                        model_key=model_key,
                        protocol="think",
                        seed=42,
                        correct_count=_scores(counts, dataset),
                        tokens=tokens,
                        suffix=f"{model_key}-{sampling}",
                        mode="direct",
                        sampling=sampling,
                    )
                )
    plans.append(_plan(tmp_path, config, "direct_dev", direct_tasks))

    variant = "ov032_fps1p0_int02_turn04"
    stage_profiles = {
        "a0_eva_clean": (2, 2, 1),
        "a1_storyboard_zoom": (2, 2, 2),
        "a2_multi_clue_memory": (3, 2, 2),
        "a3_hierarchical_search": (3, 3, 1),
        "a4_independent_arbitration": (3, 3, 2),
    }
    for stage, counts in stage_profiles.items():
        tasks = []
        for seed in REQUIRED_AGENT_SEEDS:
            for dataset in DATASETS:
                tasks.append(
                    _task(
                        tmp_path,
                        config,
                        phase="agent_dev",
                        dataset=dataset,
                        model_key="q9",
                        protocol="think",
                        seed=seed,
                        correct_count=_scores(counts, dataset),
                        tokens=200,
                        suffix=f"{stage}-{variant}-seed{seed}",
                        strategy=stage,
                        variant_id=variant,
                        leak=stage == leak_stage,
                    )
                )
        plans.append(
            _plan(
                tmp_path,
                config,
                "agent_dev",
                tasks,
                framework=stage,
                variant=variant,
            )
        )
    return config, plans


def test_complete_dev_matrix_selects_per_model_and_strictly_promotes(tmp_path: Path) -> None:
    config, plans = _complete_matrix(tmp_path)
    runs = load_dev_runs(plans, canonical_sha256(config))
    report = build_dev_selection_report(config, runs)
    assert report["status"] == "passed"
    assert "protocol_rejections" not in report
    assert report["protocol_selection"]["q4"]["protocol"] == "think"
    assert report["protocol_selection"]["q9"]["protocol"] == "think"
    assert report["direct_selection"]["q4"]["sampling"] == "uniform32"
    assert report["direct_selection"]["q9"]["sampling"] == "uniform32"
    stages = report["agent_selection"]["stages"]
    assert [stage["stage"] for stage in stages] == list(SUPPORTED_STAGES)
    assert [stage["accepted"] for stage in stages] == [True, False, True, False, False]
    assert report["winner"]["strategy"] == "a2_multi_clue_memory"
    assert report["winner"]["seed"] == 42

    summary_path = tmp_path / "summary.json"
    summary_sha = write_frozen_json(summary_path, report)
    winner = build_frozen_winner(
        report, report_path=summary_path, report_sha256=summary_sha
    )
    assert winner["schema_version"] == 2
    assert winner["agent_config"]["sha256"] == report["winner"]["agent_config"]["sha256"]
    assert winner["strict_stage_order"] == list(SUPPORTED_STAGES)
    winner_path = tmp_path / "winner.json"
    write_frozen_json(winner_path, winner)
    loaded = search_runner.load_frozen_winner(
        winner_path,
        config,
        canonical_sha256(config),
        check_files=True,
    )
    assert loaded.strategy == "a2_multi_clue_memory"
    assert loaded.selection_report.sha256 == summary_sha


def test_protocol_selection_rejects_mixed_thinking_output_budgets(
    tmp_path: Path,
) -> None:
    config, plans = _complete_matrix(tmp_path)
    protocol_plan = json.loads(plans[0].read_text(encoding="utf-8"))
    task = next(
        item
        for item in protocol_plan["tasks"]
        if item["dataset"] == "lvbench"
        and item["model_key"] == "q9"
        and item["command"][item["command"].index("--qwen-protocol") + 1]
        == "think"
    )
    result_path = next(Path(task["output_dir"]).glob("lvbench_*.jsonl"))
    rows = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["length_retry_used"] = True
    rows[0]["request_attempts"] = [
        {"max_tokens": 8192, "finish_reason": "length"},
        {"max_tokens": 32768, "finish_reason": "stop"},
    ]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_dev_selection_report(
        config, load_dev_runs(plans, canonical_sha256(config))
    )
    assert report["status"] == "blocked"
    assert "protocol_frozen_32768_length_truncation:q9" in report["blocking_errors"]
    points = report["protocol_selection"]["q9"]["points"]
    think = next(point for point in points if ":think:" in point["point_id"])
    assert think["initial_length_truncations"] == 1
    assert think["eligible"] is False
    assert "frozen_32768_length_truncation" in think["rejection_reasons"]


def test_q4_think_smoke_failure_can_replace_its_full_dev_audit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    audit_tasks = []
    for model_key, protocols in (("q4", ("no_think",)), ("q9", ("no_think", "think"))):
        for protocol in protocols:
            for dataset in DATASETS:
                audit_tasks.append(
                    _task(
                        tmp_path,
                        config,
                        phase="protocol_audit",
                        dataset=dataset,
                        model_key=model_key,
                        protocol=protocol,
                        seed=42,
                        correct_count=1 if protocol == "no_think" else 2,
                        tokens=50 if protocol == "no_think" else 80,
                        suffix=f"{model_key}-{protocol}",
                        mode="direct",
                        sampling="uniform64",
                    )
                )
    audit_plan = _plan(tmp_path, config, "protocol_audit", audit_tasks)

    smoke_tasks = [
        _task(
            tmp_path,
            config,
            phase="protocol_smoke",
            dataset=dataset,
            model_key="q4",
            protocol="think",
            seed=42,
            correct_count=1,
            tokens=80,
            suffix="q4-think-smoke",
            mode="direct",
            sampling="uniform64",
        )
        for dataset in DATASETS
    ]
    smoke_plan = _plan(tmp_path, config, "protocol_smoke", smoke_tasks)
    first_result = next(Path(smoke_tasks[0]["output_dir"]).glob("*.jsonl"))
    rows = [json.loads(line) for line in first_result.read_text(encoding="utf-8").splitlines()]
    rows[0].update(
        {
            "prediction": None,
            "correct": False,
            "error_type": "request_timeout",
        }
    )
    first_result.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for task in smoke_tasks[1:]:
        next(Path(task["output_dir"]).glob("*.jsonl")).unlink()

    evidence = load_protocol_smoke_rejection(
        smoke_plan,
        canonical_sha256(config),
        model_key="q4",
        protocol="think",
    )
    assert evidence["failures"] == 1
    assert evidence["observed_rows"] == 3
    assert evidence["expected_rows"] == 9
    report = build_dev_selection_report(
        config,
        load_dev_runs([audit_plan], canonical_sha256(config)),
        protocol_rejections={"q4": {"think": evidence}},
    )
    assert report["protocol_selection"]["q4"]["protocol"] == "no_think"
    assert report["protocol_selection"]["q9"]["protocol"] == "think"
    assert "missing_protocol_audit:q4:think" not in report["blocking_errors"]
    rejected = next(
        point
        for point in report["protocol_selection"]["q4"]["points"]
        if point["point_id"] == "q4:think:smoke_rejected"
    )
    assert rejected["eligible"] is False
    assert rejected["rejection_evidence"]["failures"] == 1

    config_path = tmp_path / "config.json"
    output_path = tmp_path / "protocol_selection.json"
    _write_json(config_path, config)
    completed = subprocess.run(
        [
            sys.executable,
            str(PROTOCOL_SCRIPT),
            "--config",
            str(config_path),
            "--run-plan",
            str(audit_plan),
            "--reject-q4-think-from-smoke",
            str(smoke_plan),
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["q4_protocol"] == "no_think"
    frozen = json.loads(output_path.read_text(encoding="utf-8"))
    assert frozen["protocol_rejections"]["q4"]["think"]["failures"] == 1


def test_smoke_rejection_requires_explicit_failure_rate_above_one_percent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    smoke_tasks = [
        _task(
            tmp_path,
            config,
            phase="protocol_smoke",
            dataset=dataset,
            model_key="q4",
            protocol="think",
            seed=42,
            correct_count=1,
            tokens=80,
            suffix="q4-think-smoke-clean",
            mode="direct",
            sampling="uniform64",
        )
        for dataset in DATASETS
    ]
    smoke_plan = _plan(tmp_path, config, "protocol_smoke", smoke_tasks)
    with pytest.raises(RuntimeError, match="does not exceed"):
        load_protocol_smoke_rejection(
            smoke_plan,
            canonical_sha256(config),
            model_key="q4",
            protocol="think",
        )


def test_leaking_candidate_is_rejected_without_corrupting_incumbent(tmp_path: Path) -> None:
    config, plans = _complete_matrix(tmp_path, leak_stage="a2_multi_clue_memory")
    report = build_dev_selection_report(
        config, load_dev_runs(plans, canonical_sha256(config))
    )
    assert report["status"] == "passed"
    stage = report["agent_selection"]["stages"][2]
    assert stage["accepted"] is False
    assert stage["candidates"][0]["annotation_leak"] > 0
    assert report["winner"]["strategy"] == "a3_hierarchical_search"
    assert report["winner"]["strategy"] != "a2_multi_clue_memory"


def test_missing_later_stage_blocks_freeze_even_if_an_earlier_agent_won(
    tmp_path: Path,
) -> None:
    config, plans = _complete_matrix(tmp_path)
    plans = plans[:-1]
    report = build_dev_selection_report(
        config, load_dev_runs(plans, canonical_sha256(config))
    )
    assert report["status"] == "blocked"
    assert any("missing_agent_stage:a4" in value for value in report["blocking_errors"])
    with pytest.raises(ValueError, match="blocked"):
        build_frozen_winner(
            report,
            report_path=tmp_path / "summary.json",
            report_sha256="0" * 64,
        )


def test_loader_rejects_result_not_bound_to_frozen_run_fingerprint(tmp_path: Path) -> None:
    config, plans = _complete_matrix(tmp_path)
    plan = json.loads(plans[0].read_text(encoding="utf-8"))
    result_dir = Path(plan["tasks"][0]["output_dir"])
    result = next(result_dir.glob("*.jsonl"))
    rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    rows[0]["run_fingerprint"] = "tampered"
    result.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_dev_runs(plans, canonical_sha256(config))


def test_cli_writes_machine_readable_report_and_schema_v2_winner(tmp_path: Path) -> None:
    config, plans = _complete_matrix(tmp_path)
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    summary = tmp_path / "selection_summary.json"
    winner = tmp_path / "winner.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(config_path),
        "--summary-output",
        str(summary),
        "--winner-output",
        str(winner),
    ]
    for plan in plans:
        command.extend(["--run-plan", str(plan)])
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    output = json.loads(completed.stdout)
    assert output["status"] == "passed"
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "passed"
    frozen = json.loads(winner.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == 2
    assert frozen["selection_report"]["sha256"] == file_sha256(summary)

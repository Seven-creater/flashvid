from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flashvid_eval.fast_hybrid_trajectory_control import build_base_run_specs


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_script("fast_hybrid_bulk_common")
teacher = _load_script("launch_fast_hybrid_teacher_matrix")
judge = _load_script("launch_fast_hybrid_judge_matrix")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    result_root = tmp_path / "results"
    train_rows: list[dict] = []
    datasets: dict[str, dict] = {}
    candidates: dict[str, Path] = {}
    for dataset in common.DATASETS:
        row = {
            "dataset": dataset,
            "sample_id": f"{dataset}-one",
            "video": f"{dataset}.mp4",
            "question": "What happens?",
            "choices": {"A": "first", "B": "second"},
            "answer": "B",
        }
        train_rows.append(row)
        manifest = tmp_path / f"{dataset}_train.jsonl"
        _write_jsonl(manifest, [row])
        candidate = tmp_path / f"{dataset}_candidate.jsonl"
        _write_jsonl(
            candidate,
            [
                {
                    "sample_id": row["sample_id"],
                    "prediction": "B",
                    "baseline_mode": "direct",
                    "sampling_id": "uniform32",
                    "enable_thinking": False,
                    "protocol_request": {"max_tokens": 512, "temperature": 0.0},
                }
            ],
        )
        candidates[dataset] = candidate
        annotation = tmp_path / f"{dataset}.json"
        annotation.write_text("[]\n", encoding="utf-8")
        video_root = tmp_path / f"{dataset}_videos"
        video_root.mkdir()
        datasets[dataset] = {
            "annotations": str(annotation),
            "video_root": str(video_root),
            "train_manifest": str(manifest),
            "train_manifest_sha256": common.file_sha256(manifest),
        }
    train600 = tmp_path / "train600.jsonl"
    _write_jsonl(train600, train_rows)
    train_sha = common.file_sha256(train600)
    config = {
        "schema_version": 1,
        "experiment_id": "test",
        "result_root": str(result_root),
        "teacher": {
            "model": "Qwen3.5-9B",
            "model_artifact_sha256": "f" * 64,
            "agent_version": "fast_hybrid_v2",
            "temperature": 0.2,
            "base_urls": ["http://127.0.0.1:8200/v1", "http://127.0.0.1:8201/v1"],
        },
        "datasets": datasets,
        "train600": {"path": str(train600), "sha256": train_sha},
        "trajectory_generation": {
            "visual_budgets": [6000, 12000, 18000, 24000],
            "generation_seeds": [17, 42, 73],
            "max_turns": 6,
            "judge_seeds": [17, 42, 73],
            "judge_temperature": 0.2,
            "rescue_visual_budgets": [32000, 48000],
            "rescue_seeds": [101, 211],
            "rescue_max_turns": 8,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_sha = common.file_sha256(config_path)
    dataset_hashes = {
        dataset: item["train_manifest_sha256"] for dataset, item in datasets.items()
    }
    all_specs = build_base_run_specs(
        train_rows,
        manifest_sha256=train_sha,
        config_sha256=config_sha,
        dataset_manifest_sha256s=dataset_hashes,
    )
    selected = [
        dict(spec)
        for spec in all_specs
        if spec["schedule_id"] == "budget_006000_seed_17"
    ]
    specs_path = tmp_path / "specs.jsonl"
    _write_jsonl(specs_path, selected)
    return config, config_path, config_sha, candidates, selected, specs_path


def test_teacher_groups_complete_schedule_and_splits_endpoints(tmp_path: Path) -> None:
    config, _, config_sha, candidates, specs, specs_path = _fixture(tmp_path)
    validated, sources, _ = common.validate_inputs(config, config_sha, specs_path)
    jobs = teacher.build_jobs(
        config=config,
        config_sha256=config_sha,
        specs=validated,
        source_rows=sources,
        candidate_paths=candidates,
        python="python",
        repo_root=ROOT,
        concurrency=16,
        timeout=80,
        resume=True,
        retry_errors=True,
        write_artifacts=False,
    )
    assert len(jobs) == 3
    assert [job.endpoint for job in jobs] == [
        "http://127.0.0.1:8200/v1",
        "http://127.0.0.1:8201/v1",
        "http://127.0.0.1:8200/v1",
    ]
    assert len({job.output_path for job in jobs}) == 3
    for job in jobs:
        command = list(job.command)
        assert command[command.index("--candidate-results") + 1] == str(
            candidates[job.dataset]
        )
        assert command[command.index("--trajectory-schedule-id") + 1] == job.schedule_id
        assert "--defer-scoring" in command
        assert "--resume" in command
        assert "--retry-errors" in command
        assert command[command.index("--base-url") + 1] == job.endpoint


def test_teacher_fails_closed_when_candidate_is_missing(tmp_path: Path) -> None:
    config, _, config_sha, candidates, _, specs_path = _fixture(tmp_path)
    _write_jsonl(candidates["cgbench"], [])
    specs, sources, _ = common.validate_inputs(config, config_sha, specs_path)
    with pytest.raises(RuntimeError, match="candidate file is missing"):
        teacher.build_jobs(
            config=config,
            config_sha256=config_sha,
            specs=specs,
            source_rows=sources,
            candidate_paths=candidates,
            python="python",
            repo_root=ROOT,
            concurrency=1,
            timeout=80,
            resume=False,
            retry_errors=False,
            write_artifacts=False,
        )


def test_spec_or_config_hash_change_is_rejected(tmp_path: Path) -> None:
    config, config_path, config_sha, _, _, specs_path = _fixture(tmp_path)
    loaded, actual = common.load_frozen_config(config_path, config_sha)
    assert loaded == config and actual == config_sha
    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="config SHA-256 mismatch"):
        common.load_frozen_config(config_path, config_sha)

    specs = common.read_jsonl(specs_path)
    specs[0]["max_turns"] = 7
    _write_jsonl(specs_path, specs)
    with pytest.raises(ValueError, match="schedule parameters"):
        common.validate_inputs(config, config_sha, specs_path)


def test_judge_groups_nonempty_schedules_and_keeps_seed_matrix(tmp_path: Path) -> None:
    config, _, config_sha, _, specs, specs_path = _fixture(tmp_path)
    public_rows = []
    for spec in specs:
        public_rows.append(
            {
                "trajectory_id": spec["trajectory_id"],
                "dataset": spec["dataset"],
                "sample_id": spec["sample_id"],
                "scoring_deferred": True,
                "trajectory_schedule_id": spec["schedule_id"],
                "generation_seed": spec["planner_seed"],
            }
        )
    trajectories = tmp_path / "judge_trajectories.jsonl"
    _write_jsonl(trajectories, public_rows)
    validated, _, _ = common.validate_inputs(config, config_sha, specs_path)
    jobs = judge.build_jobs(
        config=config,
        specs=validated,
        specs_path=specs_path,
        trajectory_paths=[trajectories],
        python="python",
        repo_root=ROOT,
        concurrency=16,
        timeout=80,
        max_tokens=512,
        resume=True,
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert job.expected_ids == tuple(sorted(row["trajectory_id"] for row in specs))
    command = list(job.command)
    assert command.count("--judge-seed") == 3
    assert command[command.index("--temperature") + 1] == "0.2"
    assert command[command.index("--base-url") + 1] == job.endpoint
    assert "--resume" in command


def test_judge_empty_prefilter_creates_no_jobs(tmp_path: Path) -> None:
    config, _, _, _, _, specs_path = _fixture(tmp_path)
    specs_path.write_text("", encoding="utf-8")
    trajectories = tmp_path / "empty.jsonl"
    trajectories.write_text("", encoding="utf-8")
    assert (
        judge.build_jobs(
            config=config,
            specs=[],
            specs_path=specs_path,
            trajectory_paths=[trajectories],
            python="python",
            repo_root=ROOT,
            concurrency=1,
            timeout=80,
            max_tokens=512,
            resume=False,
        )
        == []
    )


def test_detached_command_is_server_ready(tmp_path: Path) -> None:
    line = common.detached_shell_line(
        repo_root=ROOT,
        python="/tmp/venv/bin/python",
        script=ROOT / "scripts" / "launch_fast_hybrid_teacher_matrix.py",
        argv=["--resume"],
        log=tmp_path / "launcher.log",
    )
    assert "setsid nohup" in line
    assert "PYTHONPATH=" in line
    assert line.endswith("2>&1 < /dev/null &")

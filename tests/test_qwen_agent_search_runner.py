from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.freeze_qwen_sft_checkpoint import freeze_checkpoint
from flashvid_eval.qwen_checkpoint_selection import freeze_sft_winner
from flashvid_eval.qwen_dev_selection import write_frozen_json


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen_agent_search.py"
SERVICE_LAUNCHER = (
    Path(__file__).parents[1] / "scripts" / "launch_qwen_agent_service.sh"
)
SPEC = importlib.util.spec_from_file_location("run_qwen_agent_search", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_service_launcher_does_not_depend_on_a_git_executable_bit() -> None:
    content = SERVICE_LAUNCHER.read_text(encoding="utf-8")
    assert 'setsid nohup bash "$PROJECT_DIR/scripts/serve_qwen_agent.sh"' in content


def _write(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "sha256": runner.file_sha256(path)}


def _manifest(tmp_path: Path, dataset: str, split: str, count: int) -> dict[str, str]:
    lines = []
    for index in range(count):
        lines.append(
            json.dumps(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset}-{split}-{index}",
                    "video": f"{dataset}-{split}-{index}.mp4",
                    "question": "What happens?",
                    "choices": {"A": "one", "B": "two"},
                    "answer": "A",
                    "metadata": {},
                }
            )
            + "\n"
        )
    return _write(tmp_path / "manifests" / f"{dataset}_{split}.jsonl", "".join(lines))


def _config(tmp_path: Path, *, shared_endpoint: bool = False) -> dict:
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "evaluate_mcq.py").write_text("# fake\n", encoding="utf-8")
    for framework in runner.SUPPORTED_AGENTS:
        _write(
            workspace / "configs" / "agents" / f"{framework}.json",
            json.dumps({"schema_version": 1, "agent": {"strategy": framework}}),
        )
    datasets = {}
    for dataset in runner.DATASETS:
        annotation = _write(tmp_path / "annotations" / f"{dataset}.json", "[]\n")
        video_root = tmp_path / "videos" / dataset
        video_root.mkdir(parents=True)
        datasets[dataset] = {
            "annotations": annotation["path"],
            "video_root": str(video_root),
            "train": _manifest(tmp_path, dataset, "train", 200),
            "dev": _manifest(tmp_path, dataset, "dev", 50),
            "final": _manifest(tmp_path, dataset, "final", 100),
        }
    train600_content = ""
    for dataset in runner.DATASETS:
        train_path = Path(datasets[dataset]["train"]["path"])
        train600_content += "".join(
            runner.canonical_json(json.loads(line)) + "\n"
            for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    train600_path = tmp_path / "manifests" / "train600.jsonl"
    train600_path.write_bytes(train600_content.encode("utf-8"))
    train600 = {
        "path": str(train600_path),
        "sha256": runner.file_sha256(train600_path),
    }
    result_root = tmp_path / "results"
    for dataset in runner.DATASETS:
        for seed in (17, 42, 73):
            _write(
                result_root
                / "frozen"
                / "mismatched_video_maps"
                / f"{dataset}_dev_seed{seed}.json",
                json.dumps({"mapping": {}}),
            )
    q4_url = "http://127.0.0.1:8200/v1" if shared_endpoint else "http://127.0.0.1:8400/v1"
    return {
        "schema_version": 1,
        "experiment_id": "qwen-agent-unit",
        "result_root": str(result_root),
        "source_workspace": str(workspace),
        "constraints": {},
        "datasets": datasets,
        "models": {
            "q9": {
                "path": "/models/q9",
                "served_name": "Qwen3.5-9B",
                "artifact_sha256": "9" * 64,
                "base_url": "http://127.0.0.1:8200/v1",
            },
            "q4": {
                "path": "/models/q4",
                "served_name": "Qwen3.5-4B",
                "artifact_sha256": "4" * 64,
                "base_url": q4_url,
            },
        },
        "protocols": {
            "no_think": dict(runner.FROZEN_PROTOCOL_SPECS["no_think"]),
            "think": {
                **runner.FROZEN_PROTOCOL_SPECS["think"],
            },
        },
        "direct_sampling": [
            {"id": "uniform32", "num_frames": 32},
            {"id": "uniform64", "num_frames": 64},
            {"id": "uniform128", "num_frames": 128},
            {"id": "fps2", "fps": 2.0, "max_frames": 768},
        ],
        "diagnostics": {
            "modes": [
                "question_choices",
                "choices_only",
                "permuted_choices",
                "mismatched_video",
            ],
            "seeds": [17, 42, 73],
        },
        "agent_search": {
            "framework_order": [*runner.SUPPORTED_AGENTS, "a5_specialist_composition"],
            "overview_frames": [32, 64, 128],
            "local_fps": [1.0, 2.0],
            "max_intervals": [2, 4, 6],
            "max_turns": [4, 6, 8],
            "seeds": [17, 42, 73],
            "variants": [
                {
                    "id": "fast",
                    "overview_frames": 32,
                    "local_fps": 1.0,
                    "max_intervals": 2,
                    "max_turns": 4,
                    "hierarchy_nodes": 4,
                    "hierarchy_depth": 1,
                    "local_window_s": 60.0,
                },
                {
                    "id": "coverage",
                    "overview_frames": 128,
                    "local_fps": 2.0,
                    "max_intervals": 6,
                    "max_turns": 8,
                    "hierarchy_nodes": 12,
                    "hierarchy_depth": 3,
                    "local_window_s": 120.0,
                },
            ],
        },
        "sft": {"train600": train600, "trajectories_per_sample": 12},
        "execution": {"concurrency": 24, "sample_seed": 42, "resume": True},
    }


def _config_file(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _winner(tmp_path: Path, config: dict, strategy: str = "a3_hierarchical_search") -> Path:
    config_hash = runner.canonical_sha256(config)
    agent_path = Path(config["source_workspace"]) / "configs" / "agents" / f"{strategy}.json"
    agent_config = {
        "path": str(agent_path),
        "sha256": runner.file_sha256(agent_path),
    }
    source_plans = [{"path": str(tmp_path / "agent_dev_plan.json"), "sha256": "f" * 64}]
    winner_summary = {
        "winner_id": "dev-winner",
        "model_key": "q9",
        "protocol": "think",
        "seed": 73,
        "strategy": strategy,
        "variant_id": "ov064_fps1p0_int03_turn06",
        "agent_config": agent_config,
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_config_sha256": config_hash,
        "source_run_plans": source_plans,
        "protocol_selection": {
            "q9": {"protocol": "think"},
            "q4": {"protocol": "think"},
        },
        "direct_selection": {
            "q9": {"sampling": "uniform32"},
            "q4": {"sampling": "uniform32"},
        },
        "policy": {
            "protocol_and_direct_selected_per_model": True,
            "agent_teacher_model_key": "q9",
            "promotion_minimum_mean_correct_gain": 2.0,
            "promotion_minimum_seed_wins": 2,
            "failure_rate_max": 0.01,
            "annotation_leak_max": 0,
            "tie_break_order": [
                "accuracy",
                "accuracy_stdev",
                "regressed",
                "mean_total_tokens",
            ],
        },
        "agent_selection": {
            "strict_stage_order": list(runner.SUPPORTED_AGENTS),
            "stages": [{"stage": value} for value in runner.SUPPORTED_AGENTS],
            "final_incumbent": {
                "kind": "agent",
                "strategy": strategy,
                "variant_id": "ov064_fps1p0_int03_turn06",
                "agent_config": agent_config,
            },
        },
        "winner": winner_summary,
        "blocking_errors": [],
    }
    report["selection_state_sha256"] = runner.canonical_sha256(report)
    report_path = tmp_path / "selection_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payload = {
        "schema_version": 2,
        "winner_id": "dev-winner",
        "experiment_config_sha256": config_hash,
        "model_key": "q9",
        "protocol": "think",
        "seed": 73,
        "strategy": strategy,
        "variant_id": "ov064_fps1p0_int03_turn06",
        "agent_config": agent_config,
        "selection_state_sha256": report["selection_state_sha256"],
        "selection_report": {
            "path": str(report_path),
            "sha256": runner.file_sha256(report_path),
        },
        "source_run_plans": source_plans,
        "strict_stage_order": list(runner.SUPPORTED_AGENTS),
    }
    path = tmp_path / "winner.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_protocol_audit_enumerates_models_protocols_and_datasets(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    runner.validate_inputs(config)
    tasks = runner.build_tasks(config, "protocol_audit", check_files=True)
    assert len(tasks) == 12
    assert {task.model_key for task in tasks} == {"q4", "q9"}
    assert all("--baseline-mode" in task.command for task in tasks)
    assert all(task.command[task.command.index("--baseline-mode") + 1] == "direct" for task in tasks)
    assert all(task.command[task.command.index("--direct-sampling") + 1] == "uniform64" for task in tasks)
    assert all("--expected-manifest-sha256" in task.command for task in tasks)
    assert all(task.resume and task.concurrency == 24 for task in tasks)
    assert all(task.command[0] == sys.executable for task in tasks)
    assert all("--model-artifact-sha256" in task.command for task in tasks)


def test_post_audit_dev_phases_require_one_explicit_protocol(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    for phase in ("blind_diagnostics", "direct_dev", "agent_smoke", "agent_dev"):
        with pytest.raises(ValueError, match="protocol"):
            runner.build_tasks(config, phase, check_files=True)


def test_blind_diagnostics_expands_permutations_and_wrong_video_seeds(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    tasks = runner.build_tasks(
        config,
        "blind_diagnostics",
        protocol_filter="no_think",
        check_files=True,
    )
    # Per dataset/model/protocol: 2 simple modes + 3 permutations + 3 wrong-video controls.
    assert len(tasks) == 3 * 2 * 1 * 8
    permutations = [task for task in tasks if "--option-permutation-seed" in task.command]
    mismatched = [task for task in tasks if "--mismatched-video-map" in task.command]
    assert len(permutations) == len(mismatched) == 3 * 2 * 1 * 3
    assert {
        int(task.command[task.command.index("--option-permutation-seed") + 1])
        for task in permutations
    } == {17, 42, 73}
    assert all("--direct-sampling" in task.command for task in mismatched)


def test_direct_dev_is_the_full_sampling_matrix(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    tasks = runner.build_tasks(
        config, "direct_dev", protocol_filter="think", check_files=True
    )
    assert len(tasks) == 3 * 2 * 1 * 4
    assert {
        task.command[task.command.index("--direct-sampling") + 1] for task in tasks
    } == runner.DIRECT_SAMPLING_IDS


def test_agent_dev_enumerates_three_seeds_and_never_runs_a5(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    variant = runner.search_variants(config)[0]
    tasks = runner.build_tasks(
        config,
        "agent_dev",
        protocol_filter="think",
        framework_filter="a0_eva_clean",
        search_variant_id=variant.variant_id,
        check_files=True,
        write_variants=True,
    )
    assert len(tasks) == 3 * 2 * 1 * 1 * 3
    commands = [" ".join(task.command) for task in tasks]
    assert not any("a5_specialist" in command for command in commands)
    assert all("--backend qwen_agent" in command for command in commands)
    assert {
        int(task.command[task.command.index("--seed") + 1]) for task in tasks
    } == {17, 42, 73}
    assert all(task.agent_config_sha256 and task.agent_config_sha256 != "0" * 64 for task in tasks)
    variant_path = Path(tasks[0].command[tasks[0].command.index("--agent-config") + 1])
    variant_payload = json.loads(variant_path.read_text(encoding="utf-8"))
    assert variant_payload["search_variant"]["variant_id"] == variant.variant_id
    assert variant_payload["agent"]["overview_frames"] == variant.overview_frames
    assert all(variant.variant_id in task.output_dir for task in tasks)


def test_agent_dev_all_expands_only_the_preregistered_variants(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    tasks = runner.build_tasks(
        config,
        "agent_dev",
        protocol_filter="think",
        framework_filter="a1_storyboard_zoom",
        search_variant_id="all",
        model_filter="q9",
        check_files=True,
        write_variants=True,
    )
    assert len(tasks) == 3 * 1 * 1 * 1 * 3 * 2
    paths = {
        task.command[task.command.index("--agent-config") + 1] for task in tasks
    }
    assert len(paths) == 2


def test_agent_smoke_hashes_first_ten_without_mutating_dev(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    source = Path(config["datasets"]["lvbench"]["dev"]["path"])
    before = source.read_bytes()
    tasks = runner.build_tasks(
        config,
        "agent_smoke",
        protocol_filter="no_think",
        check_files=True,
        write_smoke=False,
        model_filter="q9",
    )
    assert source.read_bytes() == before
    assert len(tasks) == 3 * 1 * 1 * 5
    assert all(task.split == "smoke" for task in tasks)
    assert all(task.command[task.command.index("--sample") + 1] == "10" for task in tasks)
    assert not Path(tasks[0].manifest).exists()


def test_final_requires_and_validates_a_frozen_winner(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    config_hash = runner.canonical_sha256(config)
    with pytest.raises(ValueError, match="frozen-winner-config"):
        runner.build_tasks(config, "final_test", check_files=True)
    winner = runner.load_frozen_winner(
        _winner(tmp_path, config), config, config_hash, check_files=True
    )
    tasks = runner.build_tasks(
        config,
        "final_test",
        frozen_winner=winner,
        check_files=True,
    )
    assert len(tasks) == 3
    assert all(task.split == "final" and task.model_key == "q9" for task in tasks)
    assert all("a3_hierarchical_search.json" in " ".join(task.command) for task in tasks)

    changed = {**config, "experiment_id": "different"}
    with pytest.raises(RuntimeError, match="different experiment"):
        runner.load_frozen_winner(
            _winner(tmp_path, config),
            changed,
            runner.canonical_sha256(changed),
            check_files=True,
        )


def test_final_rejects_legacy_or_tampered_selection_winner(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_hash = runner.canonical_sha256(config)
    path = _winner(tmp_path, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Dev selector"):
        runner.load_frozen_winner(path, config, config_hash, check_files=True)

    path = _winner(tmp_path, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    report_path = Path(payload["selection_report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["blocking_errors"] = ["tampered"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner.load_frozen_winner(path, config, config_hash, check_files=True)


def test_a5_cannot_be_loaded_as_a_frozen_winner(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = Path(config["source_workspace"])
    a5 = workspace / "configs" / "agents" / "a5_specialist_composition.json"
    _write(a5, json.dumps({"agent": {"strategy": "a5_specialist_composition"}}))
    path = _winner(tmp_path, config, strategy="a5_specialist_composition")
    with pytest.raises(ValueError, match="A5 is disabled"):
        runner.load_frozen_winner(
            path,
            config,
            runner.canonical_sha256(config),
            check_files=True,
        )


def test_trajectory_uses_train_and_twelve_deterministic_runs(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    winner = runner.load_frozen_winner(
        _winner(tmp_path, config),
        config,
        runner.canonical_sha256(config),
        check_files=True,
    )
    tasks = runner.build_tasks(
        config,
        "trajectory",
        frozen_winner=winner,
        check_files=True,
        write_variants=True,
    )
    assert len(tasks) == 3 * 12
    assert all(task.split == "train" for task in tasks)
    lvbench = [task for task in tasks if task.dataset == "lvbench"]
    assert len({task.output_dir for task in lvbench}) == 12
    assert len({task.command[task.command.index("--seed") + 1] for task in lvbench}) == 12
    assert all("--defer-scoring" in task.command for task in tasks)
    assert all("--train600-manifest-sha256" in task.command for task in tasks)
    config_paths = {
        task.command[task.command.index("--agent-config") + 1] for task in lvbench
    }
    assert len(config_paths) == 12
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in config_paths]
    settings = [item["agent"] for item in payloads]
    assert len({item["local_fps"] for item in settings}) > 1
    assert len({item["hierarchy_nodes"] for item in settings}) > 1
    assert len({item["hierarchy_depth"] for item in settings}) > 1
    assert len({item["local_window_s"] for item in settings}) > 1
    assert len(
        {item["search_variant"]["effective_schedule_fingerprint"] for item in payloads}
    ) == 12


def test_trajectory_rejects_a_frozen_q4_winner(tmp_path: Path) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    q9 = runner.load_frozen_winner(
        _winner(tmp_path, config),
        config,
        runner.canonical_sha256(config),
        check_files=True,
    )
    q4 = runner.FrozenWinner(
        winner_id=q9.winner_id,
        model_key="q4",
        protocol=q9.protocol,
        seed=q9.seed,
        strategy=q9.strategy,
        agent_config=q9.agent_config,
        source_path=q9.source_path,
        source_sha256=q9.source_sha256,
        selection_report=q9.selection_report,
    )
    with pytest.raises(ValueError, match="Qwen3.5-9B"):
        runner.build_tasks(
            config,
            "trajectory",
            frozen_winner=q4,
            check_files=True,
            write_variants=True,
        )


def test_teacher_and_sft_dev_are_bound_to_frozen_winner_and_adapter(
    tmp_path: Path,
) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    config_hash = runner.canonical_sha256(config)
    winner_path = _winner(tmp_path, config)
    winner = runner.load_frozen_winner(
        winner_path, config, config_hash, check_files=True
    )
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text('{"messages":[]}\n', encoding="utf-8")
    adapter = tmp_path / "checkpoint-epoch1"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    freeze_checkpoint(
        checkpoint_id="epoch1",
        epoch=1,
        adapter_root=adapter,
        base_artifact_sha256=config["models"]["q9"]["artifact_sha256"],
        train_data=train_data,
        frozen_winner=winner_path,
        experiment_config_sha256=config_hash,
        served_name="Qwen3.5-9B-sft-epoch1",
        base_url="http://127.0.0.1:8301/v1",
        output=checkpoint_path,
    )
    checkpoint = runner.load_frozen_checkpoint(
        checkpoint_path, config, config_hash, winner, check_files=True
    )
    teacher_tasks = runner.build_tasks(
        config,
        "teacher_dev",
        frozen_winner=winner,
        check_files=True,
    )
    checkpoint_tasks = runner.build_tasks(
        config,
        "sft_dev",
        frozen_winner=winner,
        frozen_checkpoint=checkpoint,
        check_files=True,
    )
    assert len(teacher_tasks) == len(checkpoint_tasks) == 3
    assert {task.model for task in teacher_tasks} == {"Qwen3.5-9B"}
    assert {task.model for task in checkpoint_tasks} == {
        "Qwen3.5-9B-sft-epoch1"
    }
    assert all(task.manifest == other.manifest for task, other in zip(teacher_tasks, checkpoint_tasks))
    selection_report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_config_sha256": config_hash,
        "selected": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "epoch": checkpoint.epoch,
            "checkpoint_config": {
                "path": str(checkpoint_path.resolve()),
                "sha256": runner.file_sha256(checkpoint_path),
            },
        },
        "blocking_errors": [],
    }
    selection_report["selection_state_sha256"] = runner.canonical_sha256(
        selection_report
    )
    selection_path = tmp_path / "sft_selection.json"
    selection_sha = write_frozen_json(selection_path, selection_report)
    sft_winner_payload = freeze_sft_winner(
        selection_report,
        report_path=selection_path,
        report_sha256=selection_sha,
    )
    sft_winner_path = tmp_path / "sft_winner.json"
    write_frozen_json(sft_winner_path, sft_winner_payload)
    sft_winner = runner.load_frozen_sft_winner(
        sft_winner_path, config, config_hash, winner, check_files=True
    )
    for group, expected in (("q9", 12), ("q4", 12), ("sft9", 3)):
        final_tasks = runner.build_tasks(
            config,
            "final_matrix",
            frozen_winner=winner,
            frozen_sft_winner=sft_winner,
            final_model_group=group,
            check_files=True,
        )
        assert len(final_tasks) == expected
        runner.assert_safe_execution(final_tasks)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="adapter files changed"):
        runner.load_frozen_checkpoint(
            checkpoint_path, config, config_hash, winner, check_files=True
        )


def test_run_plan_resume_refuses_any_hash_change(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path, _config(tmp_path))
    config = runner.load_config(config_path)
    tasks = runner.build_tasks(
        config, "protocol_audit", model_filter="q9", check_files=True
    )
    plan = runner.build_run_plan(
        config_path,
        config,
        "protocol_audit",
        tasks,
        frozen_winner=None,
        frozen_checkpoint=None,
        frozen_sft_winner=None,
        model_filter="q9",
        protocol_filter=None,
        framework_filter=None,
        search_variant_id=None,
        final_model_group=None,
    )
    path = tmp_path / "plan.json"
    runner.freeze_run_plan(path, plan, resume=False)
    assert runner.freeze_run_plan(path, plan, resume=True) == plan["plan_sha256"]
    with pytest.raises(FileExistsError):
        runner.freeze_run_plan(path, plan, resume=False)
    changed = dict(plan)
    changed["config_sha256"] = "f" * 64
    changed["plan_sha256"] = runner.canonical_sha256(
        {key: value for key, value in changed.items() if key != "plan_sha256"}
    )
    with pytest.raises(RuntimeError, match="hash conflict"):
        runner.freeze_run_plan(path, changed, resume=True)


def test_execution_is_sequential_and_rejects_shared_endpoint_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = runner.load_config(_config_file(tmp_path, _config(tmp_path, shared_endpoint=True)))
    shared_tasks = runner.build_tasks(shared, "protocol_audit", check_files=True)
    with pytest.raises(RuntimeError, match="one --model-key phase"):
        runner.assert_safe_execution(shared_tasks)

    tasks = runner.build_tasks(
        shared,
        "protocol_audit",
        check_files=True,
        model_filter="q9",
    )
    seen: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        assert check
        seen.append((command, cwd))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.execute_tasks(
        tasks,
        Path(shared["source_workspace"]),
        endpoint_preflight=False,
    )
    assert [item[0] for item in seen] == [list(task.command) for task in tasks]
    assert all(Path(command[1]).name == "evaluate_mcq.py" for command, _ in seen)


def test_endpoint_preflight_requires_exact_served_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.load_config(_config_file(tmp_path, _config(tmp_path)))
    tasks = runner.build_tasks(
        config,
        "protocol_audit",
        model_filter="q9",
        protocol_filter="no_think",
        check_files=True,
    )
    requests: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    def good_open(request: object, *, timeout: float):
        assert timeout == 10.0
        requests.append(request.full_url)
        return Response(json.dumps({"data": [{"id": "Qwen3.5-9B"}]}).encode())

    monkeypatch.setattr(runner.urllib.request, "urlopen", good_open)
    runner.preflight_endpoints(tasks)
    assert requests == ["http://127.0.0.1:8200/v1/models"]

    def wrong_open(request: object, *, timeout: float):
        return Response(json.dumps({"data": [{"id": "different-model"}]}).encode())

    monkeypatch.setattr(runner.urllib.request, "urlopen", wrong_open)
    with pytest.raises(RuntimeError, match="does not serve exact model"):
        runner.preflight_endpoints(tasks)


def test_cli_dry_run_writes_no_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = _config_file(tmp_path, config)
    plan_path = Path(config["result_root"]) / "run_plans" / "protocol_audit_q9.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--phase",
            "protocol_audit",
            "--model-key",
            "q9",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"task_count": 6' in completed.stdout
    assert not plan_path.exists()

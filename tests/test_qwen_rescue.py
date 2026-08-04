from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from flashvid_eval.qwen_sft import (
    ExpectedTrajectoryProvenance,
    materialize_trajectory_identity,
    sha256_file,
    validate_trajectory_provenance,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen_rescue_trajectories.py"
SPEC = importlib.util.spec_from_file_location("run_qwen_rescue_trajectories", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
FREEZE_SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_qwen_rescue_inputs.py"
FREEZE_SPEC = importlib.util.spec_from_file_location(
    "freeze_qwen_rescue_inputs", FREEZE_SCRIPT
)
assert FREEZE_SPEC and FREEZE_SPEC.loader
freeze_module = importlib.util.module_from_spec(FREEZE_SPEC)
FREEZE_SPEC.loader.exec_module(freeze_module)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_rescue_variant_uses_the_trajectory_runner_provenance() -> None:
    record = materialize_trajectory_identity(
        {
            "dataset": "lvbench",
            "sample_id": "sample",
            "tool_steps": [],
            "scoring_deferred": True,
            "model": "Qwen3.5-9B",
            "model_artifact_sha256": "b" * 64,
            "agent_config_sha256": "a" * 64,
            "trajectory_runner_fingerprint": "f" * 64,
        },
        schedule_id="rescue_a4_dense_v1",
        variant_id="rescue",
        replica_id=0,
        judge_seed="nested",
        manifest_sha256="e" * 64,
        dataset_manifest_sha256="d" * 64,
        config_sha256="c" * 64,
    )
    expected = ExpectedTrajectoryProvenance(
        model="Qwen3.5-9B",
        model_artifact_sha256="b" * 64,
        dataset_manifest_sha256s=frozenset({"d" * 64}),
        agent_config_sha256s=frozenset({"a" * 64}),
        runner_fingerprints=frozenset({"f" * 64}),
    )
    validate_trajectory_provenance(record, expected)
    assert record["family_id"] == "rescue_a4_dense_v1~rescue"


def test_rescue_winner_is_bound_to_the_passed_selection_report(tmp_path: Path) -> None:
    config_hash = "c" * 64
    agent = tmp_path / "agent.json"
    _write_json(agent, {"agent": {"strategy": "a3_hierarchical_search"}})
    source_plans = [{"path": "plan.json", "sha256": "f" * 64}]
    winner_fields = {
        "winner_id": "winner",
        "model_key": "q9",
        "protocol": "no_think",
        "seed": 42,
        "strategy": "a3_hierarchical_search",
        "variant_id": "variant",
        "agent_config": {"path": str(agent), "sha256": sha256_file(agent)},
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_config_sha256": config_hash,
        "source_run_plans": source_plans,
        "winner": winner_fields,
        "blocking_errors": [],
    }
    report["selection_state_sha256"] = freeze_module.canonical_sha256(report)
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    winner = {
        "schema_version": 2,
        "experiment_config_sha256": config_hash,
        **winner_fields,
        "selection_state_sha256": report["selection_state_sha256"],
        "selection_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "source_run_plans": source_plans,
    }
    winner_path = tmp_path / "winner.json"
    _write_json(winner_path, winner)
    assert freeze_module._load_winner(winner_path, config_hash)["winner_id"] == "winner"

    winner["protocol"] = "think"
    _write_json(winner_path, winner)
    try:
        freeze_module._load_winner(winner_path, config_hash)
    except RuntimeError as error:
        assert "selection report" in str(error)
    else:
        raise AssertionError("tampered frozen winner was accepted")


def test_rescue_command_is_label_blind_dense_a4_and_skips_empty_dataset(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config = {
        "result_root": str(tmp_path / "results"),
        "execution": {"concurrency": 24},
        "models": {
            "q9": {
                "base_url": "http://127.0.0.1:8200/v1",
                "served_name": "Qwen3.5-9B",
                "artifact_sha256": "b" * 64,
            }
        },
        "datasets": {
            dataset: {
                "annotations": str(tmp_path / f"{dataset}.annotations"),
                "video_root": str(tmp_path / f"{dataset}.videos"),
            }
            for dataset in ("lvbench", "lsdbench", "cgbench")
        },
    }
    _write_json(config_path, config)
    manifest = tmp_path / "lvbench.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    index = {
        "experiment_config": {
            "path": str(config_path),
            "canonical_sha256": module.canonical_sha256(config),
        },
        "frozen_winner": {"seed": 42, "protocol": "no_think"},
        "agent_config": {"path": str(tmp_path / "agent.json"), "sha256": "a" * 64},
        "train600_manifest_sha256": "e" * 64,
        "schedule_id": "rescue_a4_dense_v1",
        "variant_id": "rescue",
        "manifests": {
            "lvbench": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "count": 1,
                "result_dir": str(tmp_path / "rescue/lvbench"),
            },
            "lsdbench": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "count": 0,
                "result_dir": str(tmp_path / "rescue/lsdbench"),
            },
            "cgbench": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "count": 0,
                "result_dir": str(tmp_path / "rescue/cgbench"),
            },
        },
    }
    commands = module.build_commands(index, resume=True, retry_errors=False)
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--trajectory-variant-id") + 1] == "rescue"
    assert command[command.index("--trajectory-schedule-id") + 1] == "rescue_a4_dense_v1"
    assert command[command.index("--timeout") + 1] == "80"
    assert "--defer-scoring" in command and "--resume" in command
    assert all(
        forbidden not in " ".join(command)
        for forbidden in ("correct_answer", "time_range", "clue_intervals", "question_type")
    )

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_budget_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_budget_sweep", SCRIPT)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def _write(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "sha256": sweep.file_sha256(path)}


def _config(tmp_path: Path) -> dict:
    annotations = _write(tmp_path / "annotations.json", "[]\n")
    manifest = _write(
        tmp_path / "manifest.jsonl",
        "".join(json.dumps({"sample_id": f"s{index}"}) + "\n" for index in range(12)),
    )
    candidates = _write(tmp_path / "candidates.jsonl", "{}\n")
    normalization = _write(tmp_path / "normalization.jsonl", "{}\n")
    candidates["normalization_cache"] = normalization
    endpoint = _write(tmp_path / "endpoints.json", "{}\n")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    datasets = {}
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        datasets[dataset] = {
            "annotations": dict(annotations),
            "video_root": str(video_root),
            "manifests": {"dev": dict(manifest), "final": dict(manifest)},
            "candidates": {"dev": dict(candidates), "final": dict(candidates)},
        }
    return {
        "schema_version": 1,
        "experiment_id": "unit-sweep",
        "result_root": str(tmp_path / "results"),
        "datasets": datasets,
        "controllers": {
            "q9": {"base_url": "http://127.0.0.1:8200/v1", "model": "q9"},
            "q4base": {"base_url": "http://127.0.0.1:8300/v1", "model": "q4"},
            "ck39": {"base_url": "http://127.0.0.1:8300/v1", "model": "ck39"},
        },
        "endpoint_config": endpoint,
        "prompt_ids": ["legacy_v1", "budget_rubric_v1", "budget_escalation_v1"],
        "concurrency": 24,
        "seeds": {"sample": 42, "random_uniform": [17, 42, 73]},
        "policies": [
            {
                "id": "candidate_only",
                "execution": "candidate_only",
                "stages": ["smoke", "dev", "final"],
            },
            {
                "id": "q9_fixed",
                "controller": "q9",
                "strategy": "fixed_r010",
                "prompt_id": "legacy_v1",
                "stages": ["smoke", "dev", "final"],
            },
            {
                "id": "q9_random",
                "controller": "q9",
                "strategy": "random_uniform",
                "prompt_id": "budget_rubric_v1",
                "stages": ["smoke", "dev", "final"],
            },
        ],
        "final_selection": {
            "mode": "explicit",
            "policy_ids": ["candidate_only", "q9_fixed"],
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_dev_expands_random_seeds_and_builds_evaluator_only_commands(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _config(tmp_path))
    config = sweep.load_config(config_path)
    sweep.validate_inputs(config)
    specs = sweep.build_run_specs(config, "dev", resume=True, write_smoke=False)

    assert len(specs) == 15  # 3 datasets * (candidate + fixed + 3 random seeds)
    assert sum(spec.is_offline for spec in specs) == 3
    commands = [spec.command for spec in specs if spec.command]
    assert commands
    assert all(command[1] == "scripts/evaluate_mcq.py" for command in commands)
    assert all("--resume" in command for command in commands)
    assert all("--budget-strategy" in command for command in commands)
    assert all("--controller-prompt-id" in command for command in commands)
    random = [command for command in commands if "--budget-random-seed" in command]
    assert {int(command[command.index("--budget-random-seed") + 1]) for command in random} == {17, 42, 73}
    assert all("--candidate-normalization-read-only" in command for command in commands)


def test_smoke_manifest_is_first_ten_without_modifying_dev(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = Path(config["datasets"]["lvbench"]["manifests"]["dev"]["path"])
    before = source.read_bytes()
    specs = sweep.build_run_specs(config, "smoke", resume=False, write_smoke=True)
    assert source.read_bytes() == before
    smoke = Path(config["result_root"]) / "smoke" / "manifests" / "lvbench_dev10.jsonl"
    assert len(smoke.read_text(encoding="utf-8").splitlines()) == 10
    lv_command = next(
        spec.command
        for spec in specs
        if spec.dataset == "lvbench" and spec.policy_id == "q9_fixed"
    )
    assert lv_command is not None
    assert lv_command[lv_command.index("--manifest") + 1] == str(smoke)
    assert lv_command[lv_command.index("--expected-manifest-sha256") + 1] == sweep.file_sha256(smoke)


def test_frozen_config_refuses_changes_and_requires_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = Path(config["result_root"])
    digest = sweep.freeze_config(config, root, resume=False)
    assert digest == sweep.canonical_sha256(config)
    with pytest.raises(FileExistsError):
        sweep.freeze_config(config, root, resume=False)
    assert sweep.freeze_config(config, root, resume=True) == digest
    changed = {**config, "concurrency": 8}
    with pytest.raises(RuntimeError, match="changed"):
        sweep.freeze_config(changed, root, resume=True)


def test_final_uses_only_dev_non_dominated_and_always_include(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "pareto": {
                    "system": {
                        "non_dominated_methods": [
                            "q9_direct",
                            "q9_v3c_frozen",
                        ]
                    },
                    "q9_controller": {
                        "non_dominated_methods": ["q9_random_seed42"]
                    },
                    "q4_controller": {"non_dominated_methods": []},
                }
            }
        ),
        encoding="utf-8",
    )
    config["final_selection"] = {
        "mode": "dev_non_dominated",
        "dev_summary": str(summary),
        "always_include": ["candidate_only"],
    }
    specs = sweep.build_run_specs(config, "final", resume=False, write_smoke=False)
    assert {spec.policy_id for spec in specs} == {"candidate_only", "q9_random_seed42"}


def test_multiple_controller_stage_requires_an_explicit_phase(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["policies"].append(
        {
            "id": "q4_fixed",
            "controller": "q4base",
            "strategy": "fixed_r010",
            "prompt_id": "legacy_v1",
            "stages": ["dev"],
        }
    )
    with pytest.raises(ValueError, match="one phase at a time with --controller"):
        sweep.build_run_specs(config, "dev", resume=False, write_smoke=False)

    q9 = sweep.build_run_specs(
        config,
        "dev",
        resume=False,
        write_smoke=False,
        controller_filter="q9",
    )
    q4 = sweep.build_run_specs(
        config,
        "dev",
        resume=True,
        write_smoke=False,
        controller_filter="q4base",
    )
    assert all(
        spec.is_offline or "q9" in (spec.command or ())
        for spec in q9
    )
    assert {spec.policy_id for spec in q4 if not spec.is_offline} == {"q4_fixed"}
    assert all("--resume" in (spec.command or ()) for spec in q4 if not spec.is_offline)


def test_dry_run_writes_no_frozen_or_generated_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = _write_config(tmp_path, config)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path), "--stage", "dev", "--dry-run"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"runnable": 12' in completed.stdout
    assert not Path(config["result_root"]).exists()


def test_server_templates_load_but_placeholders_cannot_execute() -> None:
    dev_path = Path(__file__).parents[1] / "configs" / "experiments" / "budget_sweep_dev.json"
    final_path = dev_path.with_name("budget_sweep_final.json")
    matched_path = dev_path.with_name("budget_sweep_matched.json")
    dev = sweep.load_config(dev_path)
    final = sweep.load_config(final_path)
    matched = sweep.load_config(matched_path)
    sweep.validate_inputs(dev, check_files=False)
    sweep.validate_inputs(final, check_files=False)
    sweep.validate_inputs(matched, check_files=False)
    assert final["experiment_id"].endswith("final-v1")
    # The production server may have every frozen Dev input, while a local
    # checkout normally does not. Template safety must therefore be asserted
    # from the explicit frozen-summary gates rather than filesystem absence.
    final_pending = sweep.pending_frozen_summaries(final, "final")
    matched_pending = sweep.pending_frozen_summaries(matched, "dev")
    assert any("sha256_not_frozen" in item["reasons"] for item in final_pending)
    assert matched_pending
    assert all("sha256_not_frozen" not in item["reasons"] for item in matched_pending)
    assert any("file_missing" in item["reasons"] for item in matched_pending)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(final_path),
            "--stage",
            "final",
            "--dry-run",
            "--allow-missing-inputs",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "valid_template_waiting_for_frozen_dev_inputs" in completed.stdout
    assert "final_selection.dev_summary" in completed.stdout
    matched_preview = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(matched_path),
            "--stage",
            "dev",
            "--dry-run",
            "--allow-missing-inputs",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "valid_template_waiting_for_frozen_dev_inputs" in matched_preview.stdout
    assert "budget_distribution_from_dev.summary" in matched_preview.stdout


def test_training_and_budget_service_keys_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["policies"][1]["sft"] = True
    with pytest.raises(ValueError, match="forbidden"):
        sweep.load_config(_write_config(tmp_path, config))


def test_random_matched_freezes_distribution_and_expands_three_seeds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["policies"].append(
        {
            "id": "q9_matched",
            "controller": "q9",
            "strategy": "random_matched",
            "prompt_id": "legacy_v1",
            "stages": ["dev"],
            "budget_distribution": {
                "0.10": 0.5,
                "0.25": 0.25,
                "0.50": 0.15,
                "1.00": 0.1,
            },
        }
    )
    config = sweep.load_config(_write_config(tmp_path, config))
    specs = [
        spec
        for spec in sweep.build_run_specs(config, "dev", resume=False, write_smoke=False)
        if spec.policy_id.startswith("q9_matched")
    ]
    assert len(specs) == 9
    assert {spec.policy_id.rsplit("seed", 1)[-1] for spec in specs} == {"17", "42", "73"}
    expected = '{"0.10":0.5,"0.25":0.25,"0.50":0.15,"1.00":0.1}'
    assert all(
        spec.command[spec.command.index("--budget-match-distribution") + 1] == expected
        for spec in specs
        if spec.command
    )


def test_random_matched_distribution_can_only_come_from_hashed_dev_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dev_summary = _write(
        tmp_path / "dev_summary.json",
        json.dumps(
            {
                "matched_random": {
                    "source_policy_id": "q9_adaptive",
                    "budget_distribution": {
                        "0.10": 0.6,
                        "0.25": 0.2,
                        "0.50": 0.1,
                        "1.00": 0.1,
                    },
                }
            }
        ),
    )
    config["policies"].append(
        {
            "id": "q9_matched",
            "controller": "q9",
            "strategy": "random_matched",
            "prompt_id": "legacy_v1",
            "stages": ["dev"],
            "random_seed": 17,
            "budget_distribution_from_dev": {"summary": dev_summary},
        }
    )
    loaded = sweep.load_config(_write_config(tmp_path, config))
    sweep.validate_inputs(loaded)
    specs = sweep.build_run_specs(loaded, "dev", resume=False, write_smoke=False)
    matched = [spec for spec in specs if spec.policy_id == "q9_matched"]
    assert len(matched) == 3
    assert all("final_test" not in " ".join(spec.command or ()) for spec in matched)
    dev_summary_path = Path(dev_summary["path"])
    dev_summary_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        sweep.validate_inputs(loaded)


def test_frozen_sweep_uses_format_compliant_q4_fixed_baselines() -> None:
    root = Path(__file__).parents[1]
    config = sweep.load_config(root / "configs" / "experiments" / "budget_sweep_dev.json")
    fixed = [
        policy
        for policy in config["policies"]
        if policy["id"].startswith("q4_fixed_")
    ]

    assert len(fixed) == 4
    assert {policy["prompt_id"] for policy in fixed} == {"budget_rubric_v1"}
    assert all(policy["id"].endswith("_rubric") for policy in fixed)
    assert Path(config["result_root"]).name == "flashvid_budget_sweep_dev_strict"

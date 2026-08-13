from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.validate_perception_memory_config import TEST_SHA256, validate_config


CONFIG_PATH = Path("configs/experiments/perception_memory_eva_sft.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_perception_memory_config_passes_all_registered_gates() -> None:
    config = _config()
    assert validate_config(config) == []
    assert {
        dataset: config["datasets"][dataset]["final"]["sha256"]
        for dataset in TEST_SHA256
    } == TEST_SHA256


def test_config_rejects_test_hash_or_split_isolation_drift() -> None:
    config = _config()
    config["datasets"]["lvbench"]["final"]["sha256"] = "0" * 64
    config["data_isolation"]["train_dev_final_pairwise_disjoint"] = False
    errors = validate_config(config)
    assert any("lvbench.final SHA-256 changed" in error for error in errors)
    assert any("train_dev_final_pairwise_disjoint" in error for error in errors)


def test_config_requires_visual_csv_episodes_and_keeps_dev_gates() -> None:
    config = _config()
    config["process_sft_data"]["minimum_stable_questions"] = 359
    config["process_sft_data"]["completion_gate_kind"] = "legacy_prefix_judge"
    config["process_sft_data"]["planner_complete_episode_required"] = False
    config["dev_selection"]["initial_seeds"] = [17]
    config["dev_selection"]["maximum_mean_total_token_ratio"] = 1.0
    errors = validate_config(config)
    assert any("must not block SFT startup" in error for error in errors)
    assert any("visual_csv" in error for error in errors)
    assert any("Planner complete episodes" in error for error in errors)
    assert any("seed 42" in error for error in errors)
    assert any("total-token ratio" in error for error in errors)


def test_config_rejects_changed_rescue_variants() -> None:
    config = _config()
    config["process_sft_data"]["coverage_variant_ids"] = ["rescue_custom"]
    errors = validate_config(config)
    assert any("rescue coverage variants changed" in error for error in errors)


def test_config_rejects_test_without_dev_gate_or_domestic_network_policy() -> None:
    config = deepcopy(_config())
    config["final_test"]["requires_dev_gate_passed"] = False
    config["orchestration"]["automatic_test_run_on_failed_dev_gate"] = True
    config["network_policy"]["hf_endpoint"] = "https://huggingface.co"
    config["network_policy"]["local_pc_downloads_forbidden"] = False
    errors = validate_config(config)
    assert any("Test requires" in error for error in errors)
    assert any("failed Dev" in error for error in errors)
    assert any("hf-mirror" in error for error in errors)
    assert any("local downloads" in error for error in errors)


def test_runbook_uses_safe_background_and_resume_commands() -> None:
    text = Path("docs/perception_memory_eva_sft_runbook.md").read_text(
        encoding="utf-8"
    )
    assert "HF_ENDPOINT=https://hf-mirror.com" in text
    assert "setsid nohup" in text
    assert "--resume" in text
    assert "Test300 只能运行一次" in text
    assert "不要用旧 v4 结果替代" in text

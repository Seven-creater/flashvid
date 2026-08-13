from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import scripts.gate_perception_memory as gate_cli

from flashvid_eval.role_separated_orchestration import (
    EXPECTED_ABLATION_CELLS,
    EXPECTED_ABLATION_EXECUTION_MODES,
    ROLE_NAMES,
    SOURCE_NAMES,
    TEST_MANIFEST_SHA256,
    RoleSeparatedConfigError,
    build_source_lock,
    derive_frozen_dev10,
    materialize_role_ablation,
    training_quantity_report,
    validate_config,
)
from scripts.gate_perception_memory import _training_quantity_report


CONFIG_PATH = Path("configs/experiments/role_separated_process_sft.json")
LOCK_PATH = Path(
    "configs/source_locks/role_separated_process_sft_sources.lock.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_role_separated_config_and_source_lock_are_reproducible() -> None:
    config = _config()
    assert validate_config(config) == []
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    expected = build_source_lock(config, config_sha256=digest)
    actual = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert actual == expected
    assert actual["source_count"] == 7
    assert tuple(item["name"] for item in actual["sources"]) == SOURCE_NAMES
    assert all(len(item["commit"]) == 40 for item in actual["sources"])
    assert all(item["reuse_boundary"] for item in actual["sources"])
    assert all(item["excluded_boundary"] for item in actual["sources"])


def test_only_planner_is_trainable_and_shared_adapter_is_rejected() -> None:
    config = _config()
    config["roles"]["observer"]["adapter"] = "trainable_planner_lora"
    config["roles"]["observer"]["trainable"] = True
    config["shared_adapter_across_roles_forbidden"] = False
    errors = validate_config(config)
    assert any("observer must use the frozen base" in item for item in errors)
    assert any("observer must not be trainable" in item for item in errors)
    assert any("may not be shared" in item for item in errors)


def test_training_quantity_is_report_only_and_never_a_gate_condition() -> None:
    config = _config()
    config["training_quantity"].update(
        {
            "observed_questions": 1,
            "observed_prefixes": 1,
            "candidate_fix_questions": 0,
        }
    )
    assert validate_config(config) == []
    assert training_quantity_report(config) == {
        "policy": "report_only",
        "blocking": False,
        "observed_questions": 1,
        "observed_prefixes": 1,
        "candidate_fix_questions": 0,
        "advisory_targets": {
            "questions": 360,
            "per_dataset": 100,
            "candidate_fix_questions": 90,
        },
    }
    assert _training_quantity_report(config["training_quantity"])["blocking"] is False

    blocked = deepcopy(config)
    blocked["training_quantity"]["blocks_training"] = True
    assert any("may not block" in item for item in validate_config(blocked))


@pytest.mark.parametrize(
    ("field", "value", "error_text"),
    (
        ("quality_contract_version", "v2", "quality contract version changed"),
        ("visual_path_classifier_version", "v2", "classifier version changed"),
        ("candidate_strata_ratio_min", 0.8, "must remain 0.9"),
        ("candidate_strata_ratio_max", 1.2, "must remain 1.1"),
        ("observed_continue_stop_ratio_min", 0.8, "must remain 0.9"),
        ("observed_continue_stop_ratio_max", 1.2, "must remain 1.1"),
        ("visual_path_all_families_required", False, "families must remain required"),
        ("visual_path_max_min_ratio", 1.2, "must remain 1.1"),
    ),
)
def test_process_sft_quality_contract_is_frozen(
    field: str, value: object, error_text: str
) -> None:
    config = _config()
    config["process_sft"][field] = value

    assert any(error_text in error for error in validate_config(config))


def test_gate_reports_low_training_quantity_without_adding_a_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate_cli,
        "evaluate_dev_gate",
        lambda **_: {"phase": "dev", "passed": True, "conditions": {}},
    )
    method = {
        "method_id": "placeholder",
        "runs": [{"seed": 42, "paths": {dataset: "unused" for dataset in TEST_MANIFEST_SHA256}}],
    }
    report = gate_cli.evaluate_config(
        {
            "phase": "dev",
            "expected_manifest_sha256": {
                dataset: "a" * 64 for dataset in TEST_MANIFEST_SHA256
            },
            "baseline": method,
            "candidates": [dict(method, method_id="candidate")],
            "training_quantity": {
                "observed_questions": 1,
                "observed_prefixes": 2,
                "candidate_fix_questions": 0,
            },
        }
    )
    assert report["passed"] is True
    assert report["conditions"] == {}
    assert report["training_quantity"]["blocking"] is False


def test_role_ablation_is_complete_dev_only_and_contains_no_test_manifest() -> None:
    config = _config()
    plan = materialize_role_ablation(config)
    assert plan["split"] == "dev"
    assert plan["test_access"] == "forbidden"
    assert len(plan["runs"]) == 6
    observed = {
        run["id"]: tuple(run["bindings"][role] for role in ROLE_NAMES)
        for run in plan["runs"]
    }
    assert observed == EXPECTED_ABLATION_CELLS
    assert all(
        set(run["pm_role_config"]) == {"planner", "observer", "verifier", "answerer"}
        for run in plan["runs"]
    )
    assert {
        run["id"]: run["execution"]["mode"] for run in plan["runs"]
    } == EXPECTED_ABLATION_EXECUTION_MODES
    fixed = [
        run
        for run in plan["runs"]
        if run["execution"]["mode"].startswith("fixed_")
    ]
    assert all(
        run["execution"]["status"] == "waiting_for_frozen_base_inputs"
        and run["execution"]["cli"] == "scripts/run_role_ablation_pair.py"
        for run in fixed
    )
    assert all(
        all(spec["derived_samples"] == 10 for spec in run["manifests"].values())
        for run in plan["runs"]
    )
    serialized = json.dumps(plan, sort_keys=True)
    assert not any(digest in serialized for digest in TEST_MANIFEST_SHA256.values())
    with pytest.raises(RoleSeparatedConfigError, match="Dev-only"):
        materialize_role_ablation(config, split="test")


def test_gate_threshold_or_test_selection_drift_is_rejected() -> None:
    config = _config()
    config["gates"]["dev"]["minimum_gain_out_of_150"] = 2
    config["gates"]["test"]["maximum_total_token_ratio"] = 0.71
    config["role_ablation"]["may_select_prompt"] = True
    config["data"]["test"]["lvbench"]["sha256"] = "0" * 64
    errors = validate_config(config)
    assert any("Dev gain must be +3/150" in item for item in errors)
    assert any("Test total-token ratio" in item for item in errors)
    assert any("may not select prompts" in item for item in errors)
    assert any("Test SHA-256 changed" in item for item in errors)


def test_dev30_manifests_are_derived_only_from_verified_dev50(tmp_path: Path) -> None:
    source = tmp_path / "dev50.jsonl"
    lines = [json.dumps({"sample_id": f"sample-{index}"}) for index in range(50)]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "frozen" / "dev10.jsonl"
    artifact = derive_frozen_dev10(
        source, expected_source_sha256=source_sha, output=output
    )
    assert artifact["samples"] == 10
    assert output.read_text(encoding="utf-8").splitlines() == lines[:10]
    with pytest.raises(RoleSeparatedConfigError, match="SHA-256 changed"):
        derive_frozen_dev10(
            source,
            expected_source_sha256="0" * 64,
            output=tmp_path / "other.jsonl",
        )

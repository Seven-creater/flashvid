#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


DATASETS = ("lvbench", "lsdbench", "cgbench")
TEST_SHA256 = {
    "lvbench": "2e71b4ae1fb1fe88c5eeb8d93d099f4a149eff31626b9f60a74a8b5e05743bbb",
    "lsdbench": "37dbd946d8abe870ac4eec7d8d1f787c99c148f43546a97371cb13a8856e68d2",
    "cgbench": "4f60b2876d896741529733841e9497b123c75fee0f5a428c761ac689d8b1b0a6",
}
EXPECTED_SPLIT_COUNTS = {"train": 200, "dev": 50, "final": 100}
REQUIRED_FUNNEL = {
    "target_hit",
    "evidence_state_valid",
    "evidence_complete",
    "judge_correct",
}
FORBIDDEN_TOOLS = {
    "clip",
    "ocr",
    "asr",
    "object_detector",
    "external_retrieval_model",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Validate immutable scope and all pre-registered execution gates."""

    errors: list[str] = []
    _require(config.get("schema_version") == 1, "schema_version must be 1", errors)

    datasets = _mapping(config.get("datasets"), "datasets", errors)
    _require(set(datasets) == set(DATASETS), "datasets must be the fixed three", errors)
    for dataset in DATASETS:
        spec = _mapping(datasets.get(dataset), f"datasets.{dataset}", errors)
        _require(
            set(spec) >= set(EXPECTED_SPLIT_COUNTS),
            f"datasets.{dataset} must contain train/dev/final",
            errors,
        )
        for split, count in EXPECTED_SPLIT_COUNTS.items():
            split_spec = _mapping(
                spec.get(split), f"datasets.{dataset}.{split}", errors
            )
            digest = str(split_spec.get("sha256") or "")
            _require(
                bool(_SHA256.fullmatch(digest)),
                f"datasets.{dataset}.{split}.sha256 must be lowercase SHA-256",
                errors,
            )
            _require(
                split_spec.get("samples") == count,
                f"datasets.{dataset}.{split} must contain {count} samples",
                errors,
            )
        final = _mapping(spec.get("final"), f"datasets.{dataset}.final", errors)
        _require(
            final.get("sha256") == TEST_SHA256[dataset],
            f"datasets.{dataset}.final SHA-256 changed",
            errors,
        )
        _require(
            final.get("role") == "fixed_engineering_test",
            f"datasets.{dataset}.final role must stay fixed_engineering_test",
            errors,
        )
        _require(
            final.get("replacement_forbidden") is True,
            f"datasets.{dataset}.final replacements must be forbidden",
            errors,
        )

    isolation = _mapping(config.get("data_isolation"), "data_isolation", errors)
    for key in (
        "train_dev_final_pairwise_disjoint",
        "cross_dataset_absolute_video_disjoint",
        "test_manifest_mutation_forbidden",
        "test_sample_replacement_forbidden",
    ):
        _require(isolation.get(key) is True, f"data_isolation.{key} must be true", errors)
    _require(isolation.get("unit") == "video_uid", "split unit must be video_uid", errors)

    diagnostics = _mapping(config.get("diagnostics_gate"), "diagnostics_gate", errors)
    _require(
        diagnostics.get("hard_gate_before_runtime_or_trajectory_generation") is True,
        "paired badcase audit must be a pre-generation hard gate",
        errors,
    )
    _require(
        diagnostics.get("required_paired_samples") == 300,
        "diagnostics must pair all 300 test samples",
        errors,
    )
    _require(
        diagnostics.get("required_samples_per_dataset") == 100,
        "diagnostics must pair 100 samples per dataset",
        errors,
    )
    _require(
        set(diagnostics.get("required_funnel") or []) == REQUIRED_FUNNEL,
        "diagnostics must require the four-stage evidence funnel",
        errors,
    )
    _require(
        diagnostics.get("old_v4_rows_may_not_satisfy_gate") is True,
        "old v4 rows must not satisfy the current SFT audit gate",
        errors,
    )

    runtime = _mapping(config.get("runtime"), "runtime", errors)
    _require(runtime.get("backend") == "perception_memory_eva", "wrong backend", errors)
    _require(runtime.get("allowed_models") == ["Qwen3.5-9B"], "only Qwen3.5-9B is allowed", errors)
    _require(runtime.get("controller_media_count") == 0, "Controller must be text-only", errors)
    _require(runtime.get("max_evidence_steps") == 6, "max evidence steps must be 6", errors)
    _require(runtime.get("candidate_rerun") is False, "candidate rerun must be disabled", errors)
    _require(
        set(runtime.get("forbidden_semantic_tools") or []) == FORBIDDEN_TOOLS,
        "forbidden semantic tool set changed",
        errors,
    )
    for key in (
        "perception_current_frame_batch_only",
        "perception_candidate_blind",
        "retrieval_candidate_blind",
        "completeness_candidate_blind",
        "judge_candidate_blind",
        "drop_images_after_perception",
        "targeted_confirmation_before_candidate_change",
        "candidate_change_requires_two_blind_judges",
        "candidate_change_requires_valid_evidence_ids",
    ):
        _require(runtime.get(key) is True, f"runtime.{key} must be true", errors)

    data = _mapping(config.get("process_sft_data"), "process_sft_data", errors)
    _require(data.get("source_split") == "train600_only", "SFT data must use Train600 only", errors)
    _require(data.get("minimum_stable_questions") == 360, "need >=360 stable questions", errors)
    _require(data.get("minimum_stable_per_dataset") == 100, "need >=100 stable questions per dataset", errors)
    _require(data.get("minimum_candidate_fixes") == 90, "need >=90 candidate fixes", errors)
    _require(data.get("minimum_candidate_fixes_per_dataset") == 20, "need >=20 candidate fixes per dataset", errors)
    _require(data.get("stable_evidence_only_judges_required") == 3, "stable prefix gate must be 3/3", errors)
    _require(
        data.get("coverage_variants_for_samples_without_stable_positive") == 4,
        "exactly four rescue coverage variants are required",
        errors,
    )
    _require(
        data.get("coverage_variant_ids")
        == [
            "rescue_global32",
            "rescue_global64",
            "rescue_first_half64",
            "rescue_second_half64",
        ],
        "rescue coverage variants changed",
        errors,
    )
    _require(data.get("complete_prefix_may_target_stop_only_after_3_of_3") is True, "stop targets require 3/3", errors)
    _require(data.get("retain_failed_tail_deletion_as_continue_example") is True, "failed deletion prefixes must be retained", errors)

    dev = _mapping(config.get("dev_selection"), "dev_selection", errors)
    _require(dev.get("initial_seeds") == [42], "Dev must start with seed 42", errors)
    _require(dev.get("promotion_seeds") == [17, 73], "promoted checkpoints must add seeds 17/73", errors)
    _require(dev.get("aggregate_seeds") == [17, 42, 73], "Dev aggregate seeds changed", errors)
    _require(dev.get("minimum_mean_correct_gain_out_of_150") == 3, "Dev gain gate must be +3/150", errors)
    _require(dev.get("per_dataset_non_decrease") is True, "Dev datasets must not regress", errors)
    _require(dev.get("maximum_mean_total_token_ratio") == 0.7, "Dev total-token ratio must be <=0.70", errors)
    _require(dev.get("maximum_mean_visual_token_ratio") == 0.7, "Dev visual-token ratio must be <=0.70", errors)
    _require(dev.get("freeze_one_winner_only") is True, "Dev must freeze one winner", errors)

    final = _mapping(config.get("final_test"), "final_test", errors)
    _require(final.get("requires_dev_gate_passed") is True, "Test requires a passed Dev gate", errors)
    _require(final.get("run_once") is True, "Test300 must run once", errors)
    _require(final.get("checkpoint_selection_from_test_forbidden") is True, "Test checkpoint selection must be forbidden", errors)
    _require(final.get("minimum_correct_gain_vs_same_runtime_untrained_out_of_300") == 6, "Test gain gate must be +6/300", errors)
    _require(final.get("minimum_total_correct_out_of_300") == 157, "Test total gate must be 157/300", errors)
    _require(final.get("minimum_dataset_correct") == {"lvbench": 45, "lsdbench": 63, "cgbench": 43}, "Test dataset floors changed", errors)
    _require(final.get("maximum_mean_total_token_ratio") == 0.7, "Test total-token ratio must be <=0.70", errors)
    _require(final.get("maximum_mean_visual_token_ratio") == 0.7, "Test visual-token ratio must be <=0.70", errors)

    network = _mapping(config.get("network_policy"), "network_policy", errors)
    _require(network.get("execution_host") == "server_only", "execution must be server-only", errors)
    _require(network.get("local_pc_downloads_forbidden") is True, "local downloads must be forbidden", errors)
    _require(network.get("hf_endpoint") == "https://hf-mirror.com", "HF endpoint must be hf-mirror", errors)

    orchestration = _mapping(config.get("orchestration"), "orchestration", errors)
    for key in ("setsid", "nohup", "resume", "config_fingerprint_required", "stop_on_failed_gate"):
        _require(orchestration.get(key) is True, f"orchestration.{key} must be true", errors)
    _require(orchestration.get("automatic_test_run_on_failed_dev_gate") is False, "failed Dev must never launch Test300", errors)
    stages = orchestration.get("stage_order") or []
    _require(stages and stages[0] == "validate_config_and_split_isolation", "split validation must be first", errors)
    _require(len(stages) > 1 and stages[1] == "audit_current_sft_badcases", "current SFT audit must be second", errors)
    _require(stages and stages[-1] == "run_test300_once_if_dev_gate_passed", "Test300 must be the final conditional stage", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Perception-Memory EVA frozen experiment gates.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_config(value if isinstance(value, Mapping) else {})
    if errors:
        raise SystemExit("invalid Perception-Memory EVA config:\n- " + "\n- ".join(errors))
    print(json.dumps({"status": "passed", "config": str(args.config)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

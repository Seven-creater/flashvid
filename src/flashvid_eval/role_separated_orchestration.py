"""Frozen orchestration contracts for role-separated process SFT experiments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


DATASETS = ("lvbench", "lsdbench", "cgbench")
TEST_MANIFEST_SHA256 = {
    "lvbench": "2e71b4ae1fb1fe88c5eeb8d93d099f4a149eff31626b9f60a74a8b5e05743bbb",
    "lsdbench": "37dbd946d8abe870ac4eec7d8d1f787c99c148f43546a97371cb13a8856e68d2",
    "cgbench": "4f60b2876d896741529733841e9497b123c75fee0f5a428c761ac689d8b1b0a6",
}
SOURCE_NAMES = (
    "efficient_video_agent",
    "time_search_r",
    "frame_thinker",
    "long_vt",
    "long_video_r1",
    "long_video_agent",
    "video_mind",
)
ROLE_NAMES = ("controller", "perception", "judge")
EXPECTED_ABLATION_CELLS = {
    "base_base_base": ("base", "base", "base"),
    "sft_base_base": ("candidate", "base", "base"),
    "base_sft_base": ("base", "candidate", "base"),
    "base_base_sft": ("base", "base", "candidate"),
    "sft_sft_base": ("candidate", "candidate", "base"),
    "sft_base_sft": ("candidate", "base", "candidate"),
    "base_sft_sft": ("base", "candidate", "candidate"),
    "sft_sft_sft": ("candidate", "candidate", "candidate"),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RoleSeparatedConfigError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Validate frozen sources, role isolation, gates, and Test isolation."""

    errors: list[str] = []
    _require(config.get("schema_version") == 1, "schema_version must be 1", errors)

    runtime = _mapping(config.get("runtime_contract"), "runtime_contract", errors)
    _require(
        runtime.get("backend") == "perception_memory_eva",
        "runtime backend must remain perception_memory_eva",
        errors,
    )
    _require(
        runtime.get("same_runtime_baseline_required") is True,
        "baseline and SFT must use the same runtime",
        errors,
    )
    _require(
        runtime.get("prompt_selection_on_test_forbidden") is True,
        "Test may not select prompts",
        errors,
    )
    _require(
        runtime.get("test_access_before_dev_winner_forbidden") is True,
        "Test must stay inaccessible until a Dev winner is frozen",
        errors,
    )

    roles = _mapping(config.get("roles"), "roles", errors)
    _require(set(roles) == set(ROLE_NAMES), "roles must be controller/perception/judge", errors)
    controller = _mapping(roles.get("controller"), "roles.controller", errors)
    perception = _mapping(roles.get("perception"), "roles.perception", errors)
    judge = _mapping(roles.get("judge"), "roles.judge", errors)
    _require(controller.get("adapter") == "trainable_controller_lora", "only controller LoRA may be trained", errors)
    _require(controller.get("media_count") == 0, "controller must remain text-only", errors)
    for name, role in (("perception", perception), ("judge", judge)):
        _require(role.get("adapter") == "frozen_base", f"{name} must use the frozen base", errors)
        _require(role.get("trainable") is False, f"{name} must not be trainable", errors)
    _require(
        config.get("shared_adapter_across_roles_forbidden") is True,
        "one adapter may not be shared across roles",
        errors,
    )

    process = _mapping(config.get("process_sft"), "process_sft", errors)
    _require(
        process.get("templates")
        == [
            "direct_answer",
            "single_frame_select",
            "timestamp_grounded_select",
            "hierarchical_refinement",
            "multi_interval_exploration",
        ],
        "process SFT templates must retain the five pre-registered families",
        errors,
    )
    for key in (
        "real_tool_loop_required",
        "incomplete_prefix_targets_continue",
        "complete_prefix_stop_requires_3_of_3",
        "tool_observations_masked",
        "test300_training_forbidden",
    ):
        _require(process.get(key) is True, f"process_sft.{key} must be true", errors)

    quantity = _mapping(config.get("training_quantity"), "training_quantity", errors)
    _require(quantity.get("policy") == "report_only", "training quantity must be report-only", errors)
    _require(quantity.get("blocks_training") is False, "training quantity may not block training", errors)
    for key in ("observed_questions", "observed_prefixes", "candidate_fix_questions"):
        value = quantity.get(key)
        _require(
            value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0),
            f"training_quantity.{key} must be null or a non-negative integer",
            errors,
        )

    data = _mapping(config.get("data"), "data", errors)
    dev = _mapping(data.get("dev"), "data.dev", errors)
    test = _mapping(data.get("test"), "data.test", errors)
    _require(set(dev) == set(DATASETS), "Dev must cover the fixed three datasets", errors)
    _require(set(test) == set(DATASETS), "Test must cover the fixed three datasets", errors)
    for dataset in DATASETS:
        dev_spec = _mapping(dev.get(dataset), f"data.dev.{dataset}", errors)
        test_spec = _mapping(test.get(dataset), f"data.test.{dataset}", errors)
        _require(dev_spec.get("samples") == 50, f"{dataset} Dev must have 50 samples", errors)
        _require(bool(_SHA256.fullmatch(str(dev_spec.get("sha256") or ""))), f"{dataset} Dev SHA-256 invalid", errors)
        _require(test_spec.get("samples") == 100, f"{dataset} Test must have 100 samples", errors)
        _require(test_spec.get("sha256") == TEST_MANIFEST_SHA256[dataset], f"{dataset} Test SHA-256 changed", errors)
        _require(test_spec.get("read_only") is True, f"{dataset} Test must be read-only", errors)

    gates = _mapping(config.get("gates"), "gates", errors)
    dev_gate = _mapping(gates.get("dev"), "gates.dev", errors)
    test_gate = _mapping(gates.get("test"), "gates.test", errors)
    _require(dev_gate.get("baseline") == "same_runtime_untrained", "Dev baseline must use the same runtime", errors)
    _require(dev_gate.get("minimum_gain_out_of_150") == 3, "Dev gain must be +3/150", errors)
    _require(dev_gate.get("per_dataset_non_decrease") is True, "Dev datasets may not regress", errors)
    _require(dev_gate.get("maximum_total_token_ratio") == 0.7, "Dev total-token ratio must be <=0.70", errors)
    _require(dev_gate.get("maximum_visual_token_ratio") == 0.7, "Dev visual-token ratio must be <=0.70", errors)
    _require(dev_gate.get("maximum_engineering_failure_rate") == 0.01, "Dev engineering failure rate must be <=1%", errors)
    _require(test_gate.get("requires_frozen_dev_winner") is True, "Test requires one frozen Dev winner", errors)
    _require(test_gate.get("minimum_gain_out_of_300") == 6, "Test gain must be +6/300", errors)
    _require(test_gate.get("minimum_total_correct") == 157, "Test minimum must be 157/300", errors)
    _require(test_gate.get("minimum_dataset_correct") == {"lvbench": 45, "lsdbench": 63, "cgbench": 43}, "Test dataset floors changed", errors)
    _require(test_gate.get("per_dataset_non_decrease") is True, "Test datasets may not regress", errors)
    _require(test_gate.get("maximum_total_token_ratio") == 0.7, "Test total-token ratio must be <=0.70", errors)
    _require(test_gate.get("maximum_visual_token_ratio") == 0.7, "Test visual-token ratio must be <=0.70", errors)
    _require(test_gate.get("maximum_engineering_failure_rate") == 0.01, "Test engineering failure rate must be <=1%", errors)
    _require(test_gate.get("run_once") is True, "Test may run only once", errors)

    ablation = _mapping(config.get("role_ablation"), "role_ablation", errors)
    _require(ablation.get("split") == "dev", "role ablation must be Dev-only", errors)
    _require(ablation.get("seeds") == [42], "role ablation must use frozen seed 42", errors)
    _require(ablation.get("may_select_prompt") is False, "role ablation may not select prompts", errors)
    _require(ablation.get("test_access") == "forbidden", "role ablation may not access Test", errors)
    raw_cells = ablation.get("cells")
    if not isinstance(raw_cells, list):
        errors.append("role_ablation.cells must be an array")
    else:
        cells: dict[str, tuple[str, str, str]] = {}
        for item in raw_cells:
            if not isinstance(item, Mapping):
                errors.append("role_ablation cell must be an object")
                continue
            cell_id = str(item.get("id") or "")
            bindings = item.get("bindings")
            if not isinstance(bindings, Mapping) or set(bindings) != set(ROLE_NAMES):
                errors.append(f"role_ablation.{cell_id} bindings invalid")
                continue
            cells[cell_id] = tuple(str(bindings[role]) for role in ROLE_NAMES)
        _require(cells == EXPECTED_ABLATION_CELLS, "role ablation must contain the full frozen 2^3 matrix", errors)

    sources = config.get("sources")
    if not isinstance(sources, list):
        errors.append("sources must be an array")
    else:
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, Mapping):
                errors.append("source entry must be an object")
                continue
            name = str(source.get("name") or "")
            seen.add(name)
            url = str(source.get("repository_url") or "")
            _require(url.startswith("https://github.com/") and url.endswith(".git"), f"{name} repository URL must be official GitHub HTTPS", errors)
            _require(bool(_COMMIT.fullmatch(str(source.get("commit") or ""))), f"{name} commit must be a full 40-character SHA", errors)
            _require(bool(str(source.get("reuse_boundary") or "").strip()), f"{name} reuse boundary is required", errors)
            _require(bool(str(source.get("excluded_boundary") or "").strip()), f"{name} excluded boundary is required", errors)
        _require(tuple(source.get("name") for source in sources if isinstance(source, Mapping)) == SOURCE_NAMES, "the seven official sources and order are frozen", errors)
        _require(len(seen) == len(SOURCE_NAMES), "source names must be unique", errors)
    return errors


def require_valid_config(config: Mapping[str, Any]) -> None:
    errors = validate_config(config)
    if errors:
        raise RoleSeparatedConfigError("invalid role-separated process SFT config:\n- " + "\n- ".join(errors))


def build_source_lock(config: Mapping[str, Any], *, config_sha256: str) -> dict[str, Any]:
    require_valid_config(config)
    if not _SHA256.fullmatch(config_sha256):
        raise RoleSeparatedConfigError("config_sha256 must be a lowercase SHA-256")
    sources = [dict(item) for item in config["sources"]]
    return {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "config_sha256": config_sha256,
        "source_count": len(sources),
        "source_set_sha256": sha256_json(sources),
        "network_policy": {
            "code_fetch": "server_only",
            "model_data_dependencies": "hf-mirror_or_domestic_mirror_only",
            "local_pc_downloads_forbidden": True,
        },
        "sources": sources,
    }


def training_quantity_report(config: Mapping[str, Any]) -> dict[str, Any]:
    require_valid_config(config)
    quantity = config["training_quantity"]
    return {
        "policy": "report_only",
        "blocking": False,
        "observed_questions": quantity.get("observed_questions"),
        "observed_prefixes": quantity.get("observed_prefixes"),
        "candidate_fix_questions": quantity.get("candidate_fix_questions"),
        "advisory_targets": dict(quantity.get("advisory_targets") or {}),
    }


def materialize_role_ablation(
    config: Mapping[str, Any], *, split: str = "dev"
) -> dict[str, Any]:
    require_valid_config(config)
    if split != "dev":
        raise RoleSeparatedConfigError(
            "role ablation is Dev-only; Test cannot be used for role or prompt selection"
        )
    ablation = config["role_ablation"]
    dev_manifests = config["data"]["dev"]
    runs = []
    for cell in ablation["cells"]:
        runs.append(
            {
                "id": cell["id"],
                "split": "dev",
                "seed": 42,
                "bindings": dict(cell["bindings"]),
                "manifests": {
                    dataset: dict(dev_manifests[dataset]) for dataset in DATASETS
                },
                "runtime": {
                    "backend": "perception_memory_eva",
                    "same_prompt_hash_required": True,
                    "same_runtime_bundle_required": True,
                    "candidate_rerun": False,
                },
                "execution": {
                    "status": "requires_role_endpoint_router",
                    "cli": "scripts/evaluate_mcq.py",
                    "note": (
                        "This is an orchestration skeleton only. Bind the three role "
                        "endpoints without changing the frozen runtime or prompt."
                    ),
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "split": "dev",
        "test_access": "forbidden",
        "prompt_selection": "forbidden",
        "training_quantity": training_quantity_report(config),
        "runs": runs,
    }


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RoleSeparatedConfigError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise RoleSeparatedConfigError(f"{path}: config root must be an object")
    return value

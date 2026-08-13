"""Frozen orchestration contracts for role-separated process SFT experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
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
ROLE_NAMES = ("planner", "observer", "verifier", "answerer")
EXPECTED_ABLATION_CELLS = {
    "base_all": ("base", "base", "base", "base"),
    "planner_old_lora": ("candidate", "base", "base", "base"),
    "observer_old_lora": ("base", "candidate", "base", "base"),
    "verifier_old_lora": ("base", "base", "candidate", "base"),
    "answerer_old_lora": ("base", "base", "base", "candidate"),
    "old_lora_all": ("candidate", "candidate", "candidate", "candidate"),
}
EXPECTED_ABLATION_EXECUTION_MODES = {
    "base_all": "end_to_end_source_for_frozen_inputs",
    "planner_old_lora": "end_to_end_system_intervention",
    "observer_old_lora": "fixed_observer_pair_with_base_downstream",
    "verifier_old_lora": "fixed_verifier_pair",
    "answerer_old_lora": "fixed_answerer_pair",
    "old_lora_all": "end_to_end_interaction_not_single_role_causal",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_frozen_dev10(
    source: Path,
    *,
    expected_source_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Derive one immutable first-10 Dev manifest after verifying Dev50."""

    if not source.is_file():
        raise FileNotFoundError(source)
    actual_source_sha256 = file_sha256(source)
    if actual_source_sha256 != expected_source_sha256:
        raise RoleSeparatedConfigError(f"Dev50 manifest SHA-256 changed: {source}")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 50:
        raise RoleSeparatedConfigError(f"Dev manifest must contain exactly 50 rows: {source}")
    parsed = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RoleSeparatedConfigError(
                f"Dev manifest row {index} is invalid JSON: {source}"
            ) from error
        if not isinstance(row, Mapping):
            raise RoleSeparatedConfigError(
                f"Dev manifest row {index} is not an object: {source}"
            )
        parsed.append(row)
    identities = [
        str(row.get("sample_id", row.get("id", ""))).strip() for row in parsed[:10]
    ]
    if any(not identity for identity in identities) or len(set(identities)) != 10:
        raise RoleSeparatedConfigError("derived Dev10 identities must be non-empty and unique")
    payload = "".join(f"{line}\n" for line in lines[:10]).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if output.exists() and file_sha256(output) != payload_sha256:
        raise RoleSeparatedConfigError(f"refusing to overwrite changed Dev10: {output}")
    if not output.exists():
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    return {
        "path": str(output.resolve()),
        "sha256": payload_sha256,
        "samples": 10,
        "source_path": str(source.resolve()),
        "source_sha256": actual_source_sha256,
        "derivation": "first_10_rows_of_each_sha256_frozen_dev_manifest",
        "sample_ids_sha256": sha256_json(identities),
    }


def freeze_role_ablation_dev30(
    config: Mapping[str, Any], *, output_dir: Path
) -> dict[str, dict[str, Any]]:
    """Verify and derive each dataset once for every role-ablation cell."""

    require_valid_config(config)
    artifacts: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        specification = config["data"]["dev"][dataset]
        artifacts[dataset] = derive_frozen_dev10(
            Path(specification["path"]),
            expected_source_sha256=specification["sha256"],
            output=output_dir / f"{dataset}_dev10.jsonl",
        )
    return artifacts


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
    _require(
        set(roles) == set(ROLE_NAMES),
        "roles must be planner/observer/verifier/answerer",
        errors,
    )
    planner = _mapping(roles.get("planner"), "roles.planner", errors)
    observer = _mapping(roles.get("observer"), "roles.observer", errors)
    verifier = _mapping(roles.get("verifier"), "roles.verifier", errors)
    answerer = _mapping(roles.get("answerer"), "roles.answerer", errors)
    _require(
        planner.get("adapter") == "trainable_planner_lora",
        "only Planner LoRA may be trained",
        errors,
    )
    _require(planner.get("media_count") == 0, "Planner must remain text-only", errors)
    for name, role in (
        ("observer", observer),
        ("verifier", verifier),
        ("answerer", answerer),
    ):
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
            "single_frame_select",
            "timestamp_grounded_select",
            "hierarchical_refinement",
            "multi_interval_exploration",
        ],
        "process SFT templates must retain the four visual path families",
        errors,
    )
    _require(
        process.get("minimum_real_frame_selects_per_question") == 1,
        "every training question requires a real frame_select",
        errors,
    )
    _require(
        process.get("quality_contract_version")
        == "role_separated_process_sft_quality_v1",
        "process SFT quality contract version changed",
        errors,
    )
    _require(
        process.get("visual_path_classifier_version")
        == "visual_path_classifier_v1",
        "visual path classifier version changed",
        errors,
    )
    _require(
        process.get("candidate_training_strata")
        == ["candidate_correct", "candidate_wrong"],
        "candidate training strata changed",
        errors,
    )
    for key in (
        "candidate_strata_ratio_min",
        "observed_continue_stop_ratio_min",
    ):
        _require(
            process.get(key) == 0.9,
            f"process_sft.{key} must remain 0.9",
            errors,
        )
    for key in (
        "candidate_strata_ratio_max",
        "observed_continue_stop_ratio_max",
        "visual_path_max_min_ratio",
    ):
        _require(
            process.get(key) == 1.1,
            f"process_sft.{key} must remain 1.1",
            errors,
        )
    _require(
        process.get("visual_path_all_families_required") is True,
        "all four visual path families must remain required",
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
    _require(
        ablation.get("samples_per_dataset") == 10,
        "role ablation must use 10 frozen Dev samples per dataset",
        errors,
    )
    _require(
        ablation.get("total_samples") == 30,
        "role ablation must contain exactly 30 Dev samples",
        errors,
    )
    _require(ablation.get("seeds") == [42], "role ablation must use frozen seed 42", errors)
    _require(ablation.get("may_select_prompt") is False, "role ablation may not select prompts", errors)
    _require(ablation.get("test_access") == "forbidden", "role ablation may not access Test", errors)
    _require(
        ablation.get("manifest_derivation")
        == "first_10_rows_of_each_sha256_frozen_dev_manifest",
        "role ablation Dev30 derivation changed",
        errors,
    )
    causal = _mapping(
        ablation.get("causal_contract"), "role_ablation.causal_contract", errors
    )
    _require(
        {
            cell_id: causal.get(f"{cell_id}_mode")
            for cell_id in EXPECTED_ABLATION_EXECUTION_MODES
        }
        == EXPECTED_ABLATION_EXECUTION_MODES,
        "role ablation execution modes changed",
        errors,
    )
    for key in (
        "frame_content_sha256_required",
        "same_frozen_request_sha256_required",
        "offline_scoring_after_both_arms",
    ):
        _require(causal.get(key) is True, f"role_ablation.{key} must be true", errors)
    artifacts = _mapping(
        ablation.get("artifacts"), "role_ablation.artifacts", errors
    )
    _require(
        set(artifacts) == {"base", "candidate"},
        "role ablation requires base and candidate artifacts",
        errors,
    )
    for artifact_name in ("base", "candidate"):
        artifact = _mapping(
            artifacts.get(artifact_name),
            f"role_ablation.artifacts.{artifact_name}",
            errors,
        )
        _require(
            set(artifact) == {"base_url", "model", "artifact_sha256"},
            f"role ablation {artifact_name} artifact schema changed",
            errors,
        )
        _require(
            str(artifact.get("base_url") or "").startswith("http://127.0.0.1:"),
            f"role ablation {artifact_name} must use a loopback endpoint",
            errors,
        )
        _require(
            bool(str(artifact.get("model") or "").strip()),
            f"role ablation {artifact_name} model is required",
            errors,
        )
        _require(
            bool(_SHA256.fullmatch(str(artifact.get("artifact_sha256") or ""))),
            f"role ablation {artifact_name} artifact SHA-256 invalid",
            errors,
        )
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
        _require(
            cells == EXPECTED_ABLATION_CELLS,
            "role ablation must contain the six frozen four-role cells",
            errors,
        )

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
    artifacts = ablation["artifacts"]
    runs = []
    for cell in ablation["cells"]:
        role_config = {
            role: dict(artifacts[cell["bindings"][role]]) for role in ROLE_NAMES
        }
        mode = EXPECTED_ABLATION_EXECUTION_MODES[cell["id"]]
        fixed_role = (
            cell["id"].removesuffix("_old_lora")
            if mode.startswith("fixed_")
            else None
        )
        runs.append(
            {
                "id": cell["id"],
                "split": "dev",
                "seed": 42,
                "bindings": dict(cell["bindings"]),
                "pm_role_config": role_config,
                "manifests": {
                    dataset: {
                        **dict(dev_manifests[dataset]),
                        "derived_samples": 10,
                        "derivation": ablation["manifest_derivation"],
                    }
                    for dataset in DATASETS
                },
                "runtime": {
                    "backend": "perception_memory_eva",
                    "same_prompt_hash_required": True,
                    "same_runtime_bundle_required": True,
                    "candidate_rerun": False,
                },
                "execution": {
                    "mode": mode,
                    "status": (
                        "waiting_for_frozen_base_inputs"
                        if fixed_role is not None
                        else "ready_for_gpu_when_authorized"
                    ),
                    "cli": (
                        "scripts/run_role_ablation_pair.py"
                        if fixed_role is not None
                        else "scripts/evaluate_mcq.py"
                    ),
                    "paired_role": fixed_role,
                    "paired_bindings": (
                        {
                            "control": dict(artifacts["base"]),
                            "treatment": dict(artifacts["candidate"]),
                        }
                        if fixed_role is not None
                        else None
                    ),
                    "frozen_input_cli": (
                        "scripts/freeze_role_ablation_inputs.py"
                        if cell["id"] == "base_all"
                        else None
                    ),
                    "note": (
                        "Fixed-role cells require the immutable Base input bundle and "
                        "a paired control/treatment run over identical request SHA-256."
                        if fixed_role is not None
                        else (
                            "This is an end-to-end system intervention; pass the "
                            "materialized role_config.json to --pm-role-config."
                        )
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

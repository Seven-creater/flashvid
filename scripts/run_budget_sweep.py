#!/usr/bin/env python3
"""Configuration-driven launcher for the training-free FlashVID budget sweep.

The launcher deliberately delegates every inference run to ``evaluate_mcq.py``.
It owns only input validation, immutable run metadata, smoke-manifest creation,
and safe detached orchestration on POSIX hosts.  It never starts model services
and contains no training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALLOWED_STAGES = {"smoke", "dev", "final"}
ALLOWED_STRATEGIES = {
    "fixed_r010",
    "fixed_r025",
    "fixed_r050",
    "fixed_r100",
    "model_requested",
    "random_uniform",
    "random_matched",
    "route_rule",
    "uncertainty_escalation",
}
ALLOWED_PROMPTS = {
    "legacy_v1",
    "budget_rubric_v1",
    "budget_escalation_v1",
}
ALLOWED_RANDOM_SEEDS = {17, 42, 73}
RATIO_KEYS = ("0.10", "0.25", "0.50", "1.00")
DEV_FRONTIER_GROUPS = ("q9_controller", "q4_controller")
OFFLINE_EXECUTIONS = {"candidate_only", "reuse"}
SHA_PLACEHOLDER_PREFIXES = ("REPLACE_", "TODO", "UNKNOWN")


@dataclass(frozen=True)
class RunSpec:
    stage: str
    dataset: str
    policy_id: str
    output_dir: Path | None
    command: tuple[str, ...] | None
    offline_reason: str | None = None

    @property
    def is_offline(self) -> bool:
        return self.command is None


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _artifact_path(
    artifact: Any,
    label: str,
    *,
    check_files: bool,
    kind: str = "file",
) -> tuple[Path, str | None]:
    item = _require_mapping(artifact, label)
    path = Path(_require_nonempty_string(item.get("path"), f"{label}.path"))
    expected = item.get("sha256")
    if kind == "directory":
        if check_files and not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
        return path, None
    expected = _require_nonempty_string(expected, f"{label}.sha256")
    if expected.upper().startswith(SHA_PLACEHOLDER_PREFIXES):
        if check_files:
            raise ValueError(f"{label}.sha256 is still a placeholder: {expected}")
        return path, expected
    if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError(f"{label}.sha256 must be a 64-character hexadecimal digest")
    if check_files:
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    return path, expected.lower()


def _read_config_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise ValueError(f"cyclic config extends chain: {resolved}")
    chain.add(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    current = _require_mapping(payload, f"config {resolved}")
    parent_name = current.get("extends")
    if parent_name is None:
        return current
    parent_path = Path(_require_nonempty_string(parent_name, "extends"))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    parent = _read_config_with_extends(parent_path, chain)
    merged = dict(parent)
    merged.update({key: value for key, value in current.items() if key != "extends"})
    return merged


def load_config(path: Path) -> dict[str, Any]:
    payload = _read_config_with_extends(path)
    config = _require_mapping(payload, "config")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _require_nonempty_string(config.get("experiment_id"), "experiment_id")
    _require_nonempty_string(config.get("result_root"), "result_root")
    datasets = _require_mapping(config.get("datasets"), "datasets")
    if set(datasets) != {"lvbench", "lsdbench", "cgbench"}:
        raise ValueError("datasets must contain exactly lvbench, lsdbench, and cgbench")
    controllers = _require_mapping(config.get("controllers"), "controllers")
    for required in ("q9", "q4base", "ck39"):
        if required not in controllers:
            raise ValueError(f"controllers is missing {required}")
    prompts = config.get("prompt_ids")
    if not isinstance(prompts, list) or set(prompts) != ALLOWED_PROMPTS:
        raise ValueError(f"prompt_ids must contain exactly {sorted(ALLOWED_PROMPTS)}")
    concurrency = config.get("concurrency")
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    seeds = _require_mapping(config.get("seeds"), "seeds")
    random_seeds = seeds.get("random_uniform")
    if not isinstance(random_seeds, list) or set(random_seeds) != ALLOWED_RANDOM_SEEDS:
        raise ValueError("seeds.random_uniform must contain exactly 17, 42, and 73")
    policies = config.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("policies must be a non-empty list")
    identifiers: set[str] = set()
    for index, raw_policy in enumerate(policies):
        policy = _require_mapping(raw_policy, f"policies[{index}]")
        policy_id = _require_nonempty_string(policy.get("id"), f"policies[{index}].id")
        if policy_id in identifiers:
            raise ValueError(f"duplicate policy id: {policy_id}")
        identifiers.add(policy_id)
        execution = policy.get("execution", "evaluate")
        if execution not in {"evaluate", *OFFLINE_EXECUTIONS}:
            raise ValueError(f"unsupported execution mode for {policy_id}: {execution}")
        stages = policy.get("stages", ["smoke", "dev", "final"])
        if not isinstance(stages, list) or not stages or not set(stages) <= ALLOWED_STAGES:
            raise ValueError(f"invalid stages for policy {policy_id}")
        if execution == "evaluate":
            if policy.get("controller") not in controllers:
                raise ValueError(f"unknown controller for policy {policy_id}")
            if policy.get("strategy") not in ALLOWED_STRATEGIES:
                raise ValueError(f"invalid budget strategy for policy {policy_id}")
            if policy.get("prompt_id") not in ALLOWED_PROMPTS:
                raise ValueError(f"invalid prompt_id for policy {policy_id}")
            if policy.get("strategy") in {"random_uniform", "random_matched"} and "random_seed" in policy:
                if policy["random_seed"] not in ALLOWED_RANDOM_SEEDS:
                    raise ValueError(f"invalid random seed for policy {policy_id}")
            if policy.get("strategy") == "random_matched":
                direct = policy.get("budget_distribution")
                derived = policy.get("budget_distribution_from_dev")
                if (direct is None) == (derived is None):
                    raise ValueError(
                        f"{policy_id} requires exactly one of budget_distribution "
                        "or budget_distribution_from_dev"
                    )
                if direct is not None:
                    validate_budget_distribution(direct, f"policies.{policy_id}.budget_distribution")
                else:
                    source = _require_mapping(
                        derived,
                        f"policies.{policy_id}.budget_distribution_from_dev",
                    )
                    _require_mapping(
                        source.get("summary"),
                        f"policies.{policy_id}.budget_distribution_from_dev.summary",
                    )
        forbidden_keys = {
            "train",
            "training",
            "sft",
            "budget_policy_base_url",
            "budget_policy_model",
        }
        present = forbidden_keys & set(policy)
        if present:
            raise ValueError(
                f"training or BudgetPolicy service is forbidden in {policy_id}: {sorted(present)}"
            )
    return config


def validate_inputs(config: dict[str, Any], *, check_files: bool = True) -> None:
    _artifact_path(
        config["endpoint_config"],
        "endpoint_config",
        check_files=check_files,
    )
    for controller_id, raw in config["controllers"].items():
        controller = _require_mapping(raw, f"controllers.{controller_id}")
        _require_nonempty_string(controller.get("base_url"), f"controllers.{controller_id}.base_url")
        _require_nonempty_string(controller.get("model"), f"controllers.{controller_id}.model")
    for dataset, raw in config["datasets"].items():
        item = _require_mapping(raw, f"datasets.{dataset}")
        _artifact_path(item.get("annotations"), f"datasets.{dataset}.annotations", check_files=check_files)
        _artifact_path(
            {"path": item.get("video_root")},
            f"datasets.{dataset}.video_root",
            check_files=check_files,
            kind="directory",
        )
        manifests = _require_mapping(item.get("manifests"), f"datasets.{dataset}.manifests")
        for split in ("dev", "final"):
            _artifact_path(
                manifests.get(split),
                f"datasets.{dataset}.manifests.{split}",
                check_files=check_files,
            )
        candidates = _require_mapping(item.get("candidates"), f"datasets.{dataset}.candidates")
        for split in ("dev", "final"):
            candidate = _require_mapping(
                candidates.get(split),
                f"datasets.{dataset}.candidates.{split}",
            )
            _artifact_path(
                candidate,
                f"datasets.{dataset}.candidates.{split}",
                check_files=check_files,
            )
            if candidate.get("normalization_cache") is not None:
                _artifact_path(
                    candidate["normalization_cache"],
                    f"datasets.{dataset}.candidates.{split}.normalization_cache",
                    check_files=check_files,
                )
    for policy in config["policies"]:
        if policy.get("strategy") == "random_matched" and policy.get("budget_distribution_from_dev"):
            source = policy["budget_distribution_from_dev"]
            summary_path, _ = _artifact_path(
                source["summary"],
                f"policies.{policy['id']}.budget_distribution_from_dev.summary",
                check_files=check_files,
            )
            if check_files:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                matched = _require_mapping(payload.get("matched_random"), "Dev summary matched_random")
                validate_budget_distribution(
                    matched.get("budget_distribution"),
                    "Dev summary matched_random.budget_distribution",
                )
        if policy.get("execution") == "reuse":
            by_dataset = _require_mapping(
                policy.get("result_by_dataset"),
                f"policies.{policy['id']}.result_by_dataset",
            )
            if set(by_dataset) != set(config["datasets"]):
                raise ValueError(
                    f"policies.{policy['id']}.result_by_dataset must cover all datasets"
                )
            for dataset, artifact in by_dataset.items():
                _artifact_path(
                    artifact,
                    f"policies.{policy['id']}.result_by_dataset.{dataset}",
                    check_files=check_files,
                )
    selection = _require_mapping(
        config.get("final_selection", {}),
        "final_selection",
    )
    if selection.get("mode") == "dev_non_dominated":
        final_selection_summary_path(selection, check_files=check_files)


def freeze_config(config: dict[str, Any], result_root: Path, *, resume: bool) -> str:
    digest = canonical_sha256(config)
    destination = result_root / "frozen_config.json"
    payload = {
        "schema_version": 1,
        "config_sha256": digest,
        "config": config,
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"frozen sweep config changed; refusing resume or overwrite: {destination}"
            )
        if not resume:
            raise FileExistsError(
                f"frozen sweep already exists; use --resume: {destination}"
            )
        return digest
    result_root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return digest


def build_smoke_manifest(source: Path, destination: Path, *, write: bool) -> tuple[Path, str]:
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 10:
        raise ValueError(f"dev manifest needs at least 10 samples for smoke: {source}")
    content = "".join(line + "\n" for line in lines[:10])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            current = destination.read_text(encoding="utf-8")
            if current != content:
                raise RuntimeError(f"existing smoke manifest differs: {destination}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(destination)
    return destination, digest


def validate_budget_distribution(value: Any, label: str) -> dict[str, float]:
    distribution = _require_mapping(value, label)
    if set(distribution) != set(RATIO_KEYS):
        raise ValueError(f"{label} must contain exactly {list(RATIO_KEYS)}")
    normalized: dict[str, float] = {}
    for key in RATIO_KEYS:
        raw = distribution[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{label}.{key} must be numeric")
        probability = float(raw)
        if probability < 0 or probability > 1:
            raise ValueError(f"{label}.{key} must be between 0 and 1")
        normalized[key] = probability
    if abs(sum(normalized.values()) - 1.0) > 1e-6:
        raise ValueError(f"{label} probabilities must sum to 1")
    return normalized


def budget_distribution_for_policy(policy: dict[str, Any]) -> dict[str, float] | None:
    if policy.get("strategy") != "random_matched":
        return None
    if policy.get("budget_distribution") is not None:
        return validate_budget_distribution(
            policy["budget_distribution"],
            f"policies.{policy['id']}.budget_distribution",
        )
    source = policy["budget_distribution_from_dev"]
    summary_path = Path(source["summary"]["path"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matched = _require_mapping(payload.get("matched_random"), "Dev summary matched_random")
    source_policy = matched.get("source_policy_id")
    if not isinstance(source_policy, str) or not source_policy:
        raise ValueError("Dev summary matched_random.source_policy_id is required")
    return validate_budget_distribution(
        matched.get("budget_distribution"),
        "Dev summary matched_random.budget_distribution",
    )


def final_selection_summary_path(
    selection: dict[str, Any],
    *,
    check_files: bool,
) -> Path:
    value = selection.get("dev_summary")
    if isinstance(value, dict):
        path, _ = _artifact_path(
            value,
            "final_selection.dev_summary",
            check_files=check_files,
        )
        return path
    path = Path(_require_nonempty_string(value, "final_selection.dev_summary"))
    if check_files and not path.is_file():
        raise FileNotFoundError(f"Dev summary required for final selection: {path}")
    return path


def _pending_summary_artifact(artifact: Any, label: str) -> dict[str, Any] | None:
    if isinstance(artifact, dict):
        item = _require_mapping(artifact, label)
        path = Path(_require_nonempty_string(item.get("path"), f"{label}.path"))
        expected = _require_nonempty_string(item.get("sha256"), f"{label}.sha256")
    else:
        path = Path(_require_nonempty_string(artifact, label))
        expected = None
    reasons: list[str] = []
    if expected and expected.upper().startswith(SHA_PLACEHOLDER_PREFIXES):
        reasons.append("sha256_not_frozen")
    if not path.is_file():
        reasons.append("file_missing")
    elif expected and not reasons and file_sha256(path).lower() != expected.lower():
        reasons.append("sha256_mismatch")
    if not reasons:
        return None
    return {
        "label": label,
        "path": str(path),
        "expected_sha256": expected,
        "reasons": reasons,
    }


def pending_frozen_summaries(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """List Dev-derived summaries that must be frozen before this stage."""

    pending: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for policy in config["policies"]:
        if stage not in policy.get("stages", ["smoke", "dev", "final"]):
            continue
        derived = policy.get("budget_distribution_from_dev")
        if policy.get("strategy") != "random_matched" or not derived:
            continue
        label = f"policies.{policy['id']}.budget_distribution_from_dev.summary"
        item = _pending_summary_artifact(derived["summary"], label)
        if item is not None and (item["label"], item["path"]) not in seen:
            pending.append(item)
            seen.add((item["label"], item["path"]))
    selection = _require_mapping(config.get("final_selection", {}), "final_selection")
    if stage == "final" and selection.get("mode") == "dev_non_dominated":
        item = _pending_summary_artifact(
            selection.get("dev_summary"),
            "final_selection.dev_summary",
        )
        if item is not None and (item["label"], item["path"]) not in seen:
            pending.append(item)
    return pending


def _dev_controller_frontier_ids(summary: Any) -> set[str]:
    """Read only executable controller frontiers, never the system frontier."""

    payload = _require_mapping(summary, "Dev summary")
    pareto = _require_mapping(payload.get("pareto"), "Dev summary pareto")
    selected: set[str] = set()
    for group_name in DEV_FRONTIER_GROUPS:
        group = _require_mapping(
            pareto.get(group_name),
            f"Dev summary pareto.{group_name}",
        )
        entries = group.get("non_dominated_methods")
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry for entry in entries
        ):
            raise ValueError(
                f"Dev summary pareto.{group_name}.non_dominated_methods "
                "must be a list of policy ids"
            )
        selected.update(entries)
    return selected


def _available_expanded_ids(config: dict[str, Any], stage: str) -> set[str]:
    available: set[str] = set()
    for policy in config["policies"]:
        if stage not in policy.get("stages", ["smoke", "dev", "final"]):
            continue
        if policy.get("strategy") in {"random_uniform", "random_matched"} and "random_seed" not in policy:
            available.update(
                f"{policy['id']}_seed{seed}"
                for seed in config["seeds"]["random_uniform"]
            )
        else:
            available.add(policy["id"])
    return available


def select_policy_ids(config: dict[str, Any], stage: str) -> set[str]:
    available = _available_expanded_ids(config, stage)
    if stage != "final":
        return available
    selection = _require_mapping(config.get("final_selection", {}), "final_selection")
    mode = selection.get("mode", "explicit")
    always = set(selection.get("always_include", []))
    if mode == "explicit":
        selected = set(selection.get("policy_ids", [])) | always
    elif mode == "dev_non_dominated":
        summary = final_selection_summary_path(selection, check_files=True)
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
        selected = _dev_controller_frontier_ids(summary_payload) | always
        if selection.get("include_cost_matched_random"):
            cost_match = _require_mapping(
                summary_payload.get("cost_matched_random"),
                "Dev summary cost_matched_random",
            )
            selected_method = _require_nonempty_string(
                cost_match.get("selected_method"),
                "Dev summary cost_matched_random.selected_method",
            )
            selected.add(selected_method)
    else:
        raise ValueError(f"unsupported final_selection.mode: {mode}")
    random_bases = {
        policy["id"]
        for policy in config["policies"]
        if policy.get("strategy") in {"random_uniform", "random_matched"} and "random_seed" not in policy
    }
    for base in selected & random_bases:
        selected.update(
            f"{base}_seed{seed}" for seed in config["seeds"]["random_uniform"]
        )
    all_known: set[str] = set(random_bases)
    for known_stage in ALLOWED_STAGES:
        all_known.update(_available_expanded_ids(config, known_stage))
    unknown = selected - all_known
    if unknown:
        raise ValueError(f"Dev summary selected unknown policy ids: {sorted(unknown)}")
    return available & selected


def _expanded_policies(config: dict[str, Any], stage: str) -> Iterable[tuple[dict[str, Any], int | None, str]]:
    selected = select_policy_ids(config, stage)
    for policy in config["policies"]:
        if stage not in policy.get("stages", ["smoke", "dev", "final"]):
            continue
        if policy.get("strategy") in {"random_uniform", "random_matched"} and "random_seed" not in policy:
            for seed in config["seeds"]["random_uniform"]:
                expanded_id = f"{policy['id']}_seed{seed}"
                if expanded_id in selected:
                    yield policy, int(seed), expanded_id
        else:
            seed = policy.get("random_seed")
            if policy["id"] in selected:
                yield policy, int(seed) if seed is not None else None, policy["id"]


def build_run_specs(
    config: dict[str, Any],
    stage: str,
    *,
    resume: bool,
    write_smoke: bool,
    controller_filter: str | None = None,
) -> list[RunSpec]:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"invalid stage: {stage}")
    result_root = Path(config["result_root"])
    experiment_config_hash = canonical_sha256(config)
    endpoint_path = Path(config["endpoint_config"]["path"])
    endpoint_hash = str(config["endpoint_config"]["sha256"])
    sample_seed = int(config["seeds"].get("sample", 42))
    specs: list[RunSpec] = []
    expanded_policies = list(_expanded_policies(config, stage))
    active_controllers = {
        str(policy["controller"])
        for policy, _, _ in expanded_policies
        if policy.get("execution", "evaluate") == "evaluate"
    }
    if controller_filter is None and len(active_controllers) > 1:
        phases = ", ".join(sorted(active_controllers))
        raise ValueError(
            f"stage {stage} requires multiple externally managed controllers "
            f"({phases}); run one phase at a time with --controller"
        )
    for policy, random_seed, expanded_id in expanded_policies:
        execution = policy.get("execution", "evaluate")
        if (
            controller_filter is not None
            and execution == "evaluate"
            and policy.get("controller") != controller_filter
        ):
            continue
        for dataset, dataset_config in config["datasets"].items():
            if execution in OFFLINE_EXECUTIONS:
                reason = "frozen candidate baseline is scored offline" if execution == "candidate_only" else "reuse frozen result"
                specs.append(RunSpec(stage, dataset, expanded_id, None, None, reason))
                continue
            split = "dev" if stage in {"smoke", "dev"} else "final"
            manifest_artifact = dataset_config["manifests"][split]
            candidate_artifact = dataset_config["candidates"][split]
            manifest = Path(manifest_artifact["path"])
            manifest_hash = str(manifest_artifact["sha256"])
            if stage == "smoke":
                manifest, manifest_hash = build_smoke_manifest(
                    manifest,
                    result_root / "smoke" / "manifests" / f"{dataset}_dev10.jsonl",
                    write=write_smoke,
                )
            output_dir = result_root / stage / expanded_id / dataset
            if (
                not resume
                and output_dir.is_dir()
                and any(output_dir.iterdir())
            ):
                raise FileExistsError(
                    f"non-empty run output exists; use --resume: {output_dir}"
                )
            controller = config["controllers"][policy["controller"]]
            command = [
                sys.executable,
                "scripts/evaluate_mcq.py",
                "--dataset",
                dataset,
                "--backend",
                "flashvid_hybrid",
                "--agent-version",
                "flashvid_budget_v1",
                "--annotations",
                str(dataset_config["annotations"]["path"]),
                "--video-root",
                str(dataset_config["video_root"]),
                "--manifest",
                str(manifest),
                "--expected-manifest-sha256",
                manifest_hash,
                "--candidate-results",
                str(candidate_artifact["path"]),
                "--controller-base-url",
                str(controller["base_url"]),
                "--controller-model",
                str(controller["model"]),
                "--perception-endpoints",
                str(endpoint_path),
                "--expected-endpoint-config-sha256",
                endpoint_hash,
                "--budget-strategy",
                str(policy["strategy"]),
                "--controller-prompt-id",
                str(policy["prompt_id"]),
                "--concurrency",
                str(config["concurrency"]),
                "--seed",
                str(sample_seed),
                "--output-dir",
                str(output_dir),
                "--experiment-config-sha256",
                experiment_config_hash,
            ]
            if random_seed is not None:
                command.extend(["--budget-random-seed", str(random_seed)])
            budget_distribution = budget_distribution_for_policy(policy)
            if budget_distribution is not None:
                command.extend(
                    [
                        "--budget-match-distribution",
                        canonical_json(budget_distribution),
                    ]
                )
            normalization_cache = candidate_artifact.get("normalization_cache")
            if normalization_cache:
                command.extend(
                    [
                        "--candidate-normalization-cache",
                        str(normalization_cache["path"]),
                        "--candidate-normalization-read-only",
                    ]
                )
            override = controller.get("config_override")
            if override:
                command.extend(["--controller-config-override", str(override)])
            if resume:
                command.append("--resume")
            specs.append(
                RunSpec(stage, dataset, expanded_id, output_dir, tuple(command))
            )
    return specs


def shell_line(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def write_and_launch(
    specs: list[RunSpec],
    result_root: Path,
    stage: str,
    *,
    phase: str | None = None,
) -> tuple[Path, int]:
    if os.name == "nt":
        raise RuntimeError("execution requires a POSIX server; use --dry-run on Windows")
    runnable = [spec for spec in specs if spec.command]
    suffix = f"_{phase}" if phase else ""
    launcher = result_root / "launchers" / f"run_{stage}{suffix}.sh"
    log = result_root / "logs" / f"{stage}{suffix}.log"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in runnable:
        assert spec.command is not None
        lines.append(shell_line(spec.command))
    lines.append("")
    launcher.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    launcher.chmod(0o755)
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            ["setsid", "nohup", "bash", str(launcher)],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    return launcher, process.pid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and run the training-free FlashVID budget sweep."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(ALLOWED_STAGES), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--controller",
        choices=("q9", "q4base", "ck39"),
        help="Run one externally managed controller phase; offline baselines remain included.",
    )
    parser.add_argument(
        "--allow-missing-inputs",
        action="store_true",
        help="Preview a server config off-server; valid only with --dry-run (smoke still needs its dev manifest).",
    )
    args = parser.parse_args()
    if args.allow_missing_inputs and not args.dry_run:
        parser.error("--allow-missing-inputs is only valid with --dry-run")
    config = load_config(args.config)
    validate_inputs(config, check_files=not args.allow_missing_inputs)
    if args.allow_missing_inputs:
        pending = pending_frozen_summaries(config, args.stage)
        if pending:
            print(
                json.dumps(
                    {
                        "experiment_id": config["experiment_id"],
                        "stage": args.stage,
                        "config_sha256": canonical_sha256(config),
                        "preview_status": "valid_template_waiting_for_frozen_dev_inputs",
                        "pending_frozen_inputs": pending,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.stage == "smoke":
            raise ValueError(
                "smoke dry-run needs real dev manifests to derive immutable first-10 manifests"
            )
    specs = build_run_specs(
        config,
        args.stage,
        resume=args.resume,
        write_smoke=not args.dry_run,
        controller_filter=args.controller,
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "stage": args.stage,
        "config_sha256": canonical_sha256(config),
        "runnable": sum(not spec.is_offline for spec in specs),
        "offline": sum(spec.is_offline for spec in specs),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for spec in specs:
        if spec.command:
            print(shell_line(spec.command))
        else:
            print(f"# offline {spec.policy_id}/{spec.dataset}: {spec.offline_reason}")
    if args.dry_run:
        return
    result_root = Path(config["result_root"])
    freeze_config(config, result_root, resume=args.resume)
    launcher, pid = write_and_launch(
        specs,
        result_root,
        args.stage,
        phase=args.controller,
    )
    print(json.dumps({"launcher": str(launcher), "pid": pid}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and sequentially execute the frozen Qwen-only experiment matrix.

This orchestrator never starts or stops services and never inspects GPUs. Every
inference task is delegated to ``scripts/evaluate_mcq.py`` without a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping


PHASES = {
    "protocol_audit",
    "protocol_smoke",
    "blind_diagnostics",
    "direct_dev",
    "agent_smoke",
    "agent_dev",
    "final_test",
    "trajectory",
    "teacher_dev",
    "sft_dev",
    "final_matrix",
}
DATASETS = ("lvbench", "lsdbench", "cgbench")
SUPPORTED_AGENTS = (
    "a0_eva_clean",
    "a1_storyboard_zoom",
    "a2_multi_clue_memory",
    "a3_hierarchical_search",
    "a4_independent_arbitration",
)
DISABLED_AGENTS = {"a5_specialist_composition"}
DIRECT_SAMPLING_IDS = {"uniform32", "uniform64", "uniform128", "fps2"}
DIRECT_SAMPLING_SPECS = {
    "uniform32": {"id": "uniform32", "num_frames": 32},
    "uniform64": {"id": "uniform64", "num_frames": 64},
    "uniform128": {"id": "uniform128", "num_frames": 128},
    "fps2": {"id": "fps2", "fps": 2.0, "max_frames": 768},
}
FROZEN_PROTOCOL_SPECS = {
    "no_think": {
        "enable_thinking": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "think": {
        "enable_thinking": True,
        "max_tokens": 8192,
        "length_retry_max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}
BASELINE_MODES = {
    "question_choices",
    "choices_only",
    "permuted_choices",
    "mismatched_video",
}
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class FrozenWinner:
    winner_id: str
    model_key: str
    protocol: str
    seed: int
    strategy: str
    agent_config: Artifact
    selection_report: Artifact
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class FrozenCheckpoint:
    checkpoint_id: str
    epoch: int
    served_name: str
    base_url: str
    served_stack_sha256: str
    adapter_path: Path
    adapter_artifact_sha256: str
    frozen_winner_sha256: str
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class FrozenSFTWinner:
    checkpoint: FrozenCheckpoint
    selection_report: Artifact
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class SearchVariant:
    variant_id: str
    overview_frames: int
    local_fps: float
    max_intervals: int
    max_turns: int
    hierarchy_nodes: int | None = None
    hierarchy_depth: int | None = None
    local_window_s: float | None = None
    effective_schedule_fingerprint: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    phase: str
    task_id: str
    dataset: str
    split: str
    model_key: str
    model: str
    model_artifact_sha256: str
    base_url: str
    manifest: str
    manifest_sha256: str
    output_dir: str
    resume: bool
    concurrency: int
    agent_config_sha256: str | None
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.command) < 2 or Path(self.command[1]).name != "evaluate_mcq.py":
            raise ValueError("orchestrator tasks may only invoke evaluate_mcq.py")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    digest = _nonempty(value, label).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return digest


def _safe_id(value: str) -> str:
    return (_SAFE_ID_RE.sub("_", value).strip("._") or "task")[:180]


def _path_arg(path: Path) -> str:
    """Keep server-style absolute paths intact during Windows dry-runs."""
    return path.as_posix()


def search_variants(config: Mapping[str, Any]) -> list[SearchVariant]:
    search = config["agent_search"]
    declared = search.get("variants")
    if isinstance(declared, list):
        return [
            SearchVariant(
                variant_id=_nonempty(item.get("id"), "agent_search.variants.id"),
                overview_frames=int(item["overview_frames"]),
                local_fps=float(item["local_fps"]),
                max_intervals=int(item["max_intervals"]),
                max_turns=int(item["max_turns"]),
                hierarchy_nodes=int(item["hierarchy_nodes"]),
                hierarchy_depth=int(item["hierarchy_depth"]),
                local_window_s=float(item["local_window_s"]),
            )
            for raw in declared
            for item in [_mapping(raw, "agent_search.variants item")]
        ]
    variants: list[SearchVariant] = []
    for overview, fps, intervals, turns in product(
        search["overview_frames"],
        search["local_fps"],
        search["max_intervals"],
        search["max_turns"],
    ):
        fps_id = str(float(fps)).replace(".", "p")
        variants.append(
            SearchVariant(
                variant_id=f"ov{int(overview):03d}_fps{fps_id}_int{int(intervals):02d}_turn{int(turns):02d}",
                overview_frames=int(overview),
                local_fps=float(fps),
                max_intervals=int(intervals),
                max_turns=int(turns),
            )
        )
    return variants


def select_search_variant(config: Mapping[str, Any], variant_id: str) -> SearchVariant:
    matches = [item for item in search_variants(config) if item.variant_id == variant_id]
    if not matches:
        examples = ", ".join(item.variant_id for item in search_variants(config)[:3])
        raise ValueError(f"unknown search variant {variant_id!r}; examples: {examples}")
    return matches[0]


def _variant_settings(
    base_settings: Mapping[str, Any], variant: SearchVariant
) -> dict[str, Any]:
    settings = dict(base_settings)
    settings.update(
        {
            "overview_frames": variant.overview_frames,
            "local_fps": variant.local_fps,
            "max_intervals": variant.max_intervals,
            "max_turns": variant.max_turns,
        }
    )
    for key in ("hierarchy_nodes", "hierarchy_depth", "local_window_s"):
        value = getattr(variant, key)
        if value is not None:
            settings[key] = value
    return settings


def effective_schedule(
    strategy: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Return only settings that can change a strategy's evidence trajectory."""

    if strategy == "a4_independent_arbitration":
        evidence_strategy = str(
            settings.get("evidence_strategy", "a3_hierarchical_search")
        )
        if evidence_strategy == "a4_independent_arbitration":
            raise ValueError("A4 evidence strategy cannot recursively select A4")
        return {
            "strategy": strategy,
            "direct_sampling": str(settings.get("direct_sampling", "uniform64")),
            "evidence": effective_schedule(evidence_strategy, settings),
        }
    if strategy == "a0_eva_clean":
        return {
            "strategy": strategy,
            "overview_frames": int(settings.get("overview_frames", 64)),
            "max_intervals": int(settings.get("max_intervals", 3)),
            "max_turns": int(settings.get("max_turns", 6)),
            "resize": float(settings.get("resize", 0.75)),
            "max_frames_per_call": int(settings.get("max_frames_per_call", 128)),
        }
    if strategy in {"a1_storyboard_zoom", "a2_multi_clue_memory"}:
        max_intervals = int(settings.get("max_intervals", 3))
        max_turns = int(settings.get("max_turns", 6))
        return {
            "strategy": strategy,
            "overview_frames": int(settings.get("overview_frames", 64)),
            "local_fps": float(settings.get("local_fps", 1.0)),
            "local_window_s": float(settings.get("local_window_s", 120.0)),
            "local_interval_limit": min(max_intervals, max(1, max_turns - 2)),
            "resize": float(settings.get("resize", 0.75)),
            "max_frames_per_call": int(settings.get("max_frames_per_call", 128)),
        }
    if strategy == "a3_hierarchical_search":
        max_turns = int(settings.get("max_turns", 6))
        hierarchy_depth = int(settings.get("hierarchy_depth", 3))
        return {
            "strategy": strategy,
            "root_nodes": 8,
            "branch_nodes": 2,
            "hierarchy_depth": min(hierarchy_depth, max(0, max_turns - 2)),
            "local_fps": float(settings.get("local_fps", 1.0)),
            "local_window_s": float(settings.get("local_window_s", 120.0)),
            "resize": float(settings.get("resize", 0.75)),
            "max_frames_per_call": int(settings.get("max_frames_per_call", 128)),
        }
    raise ValueError(f"unsupported strategy for effective schedule: {strategy}")


def load_config(path: Path) -> dict[str, Any]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _nonempty(config.get("experiment_id"), "experiment_id")
    _nonempty(config.get("result_root"), "result_root")
    _nonempty(config.get("source_workspace"), "source_workspace")

    datasets = _mapping(config.get("datasets"), "datasets")
    if tuple(datasets) != DATASETS and set(datasets) != set(DATASETS):
        raise ValueError(f"datasets must contain exactly {DATASETS}")
    for dataset in DATASETS:
        item = _mapping(datasets[dataset], f"datasets.{dataset}")
        _nonempty(item.get("annotations"), f"datasets.{dataset}.annotations")
        _nonempty(item.get("video_root"), f"datasets.{dataset}.video_root")
        for split in ("train", "dev", "final"):
            artifact = _mapping(item.get(split), f"datasets.{dataset}.{split}")
            _nonempty(artifact.get("path"), f"datasets.{dataset}.{split}.path")
            _sha(artifact.get("sha256"), f"datasets.{dataset}.{split}.sha256")

    models = _mapping(config.get("models"), "models")
    if set(models) != {"q4", "q9"}:
        raise ValueError("models must contain exactly q4 and q9")
    for key, model in models.items():
        item = _mapping(model, f"models.{key}")
        served_name = _nonempty(item.get("served_name"), f"models.{key}.served_name")
        if "qwen3.5" not in served_name.lower():
            raise ValueError(f"models.{key} is not a Qwen3.5 endpoint")
        _nonempty(item.get("base_url"), f"models.{key}.base_url")
        _sha(item.get("artifact_sha256"), f"models.{key}.artifact_sha256")

    protocols = _mapping(config.get("protocols"), "protocols")
    if set(protocols) != {"no_think", "think"}:
        raise ValueError("protocols must contain exactly no_think and think")
    for protocol_id, protocol in protocols.items():
        item = _mapping(protocol, f"protocols.{protocol_id}")
        if not isinstance(item.get("enable_thinking"), bool):
            raise ValueError(f"protocols.{protocol_id}.enable_thinking must be boolean")
        if not isinstance(item.get("max_tokens"), int) or item["max_tokens"] <= 0:
            raise ValueError(f"protocols.{protocol_id}.max_tokens must be positive")
        if item != FROZEN_PROTOCOL_SPECS[protocol_id]:
            raise ValueError(
                f"{protocol_id} protocol differs from the evaluator's frozen request spec"
            )

    sampling = config.get("direct_sampling")
    if not isinstance(sampling, list) or not sampling:
        raise ValueError("direct_sampling must be a non-empty list")
    sampling_ids = [str(_mapping(item, "direct_sampling item").get("id")) for item in sampling]
    if len(sampling_ids) != len(set(sampling_ids)) or not set(sampling_ids) <= DIRECT_SAMPLING_IDS:
        raise ValueError("direct_sampling contains duplicate or unsupported ids")
    for item in sampling:
        if item != DIRECT_SAMPLING_SPECS[str(item["id"])]:
            raise ValueError(f"direct_sampling {item['id']} differs from the frozen evaluator spec")

    diagnostics = _mapping(config.get("diagnostics"), "diagnostics")
    modes = diagnostics.get("modes")
    seeds = diagnostics.get("seeds")
    if not isinstance(modes, list) or not set(modes) <= BASELINE_MODES:
        raise ValueError("diagnostics.modes contains unsupported modes")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("diagnostics.seeds must be a non-empty integer list")

    search = _mapping(config.get("agent_search"), "agent_search")
    frameworks = search.get("framework_order")
    if not isinstance(frameworks, list) or not frameworks:
        raise ValueError("agent_search.framework_order must be a non-empty list")
    unknown = set(frameworks) - set(SUPPORTED_AGENTS) - DISABLED_AGENTS
    if unknown:
        raise ValueError(f"unknown agent frameworks: {sorted(unknown)}")
    search_seeds = search.get("seeds")
    if not isinstance(search_seeds, list) or not search_seeds or not all(
        isinstance(seed, int) for seed in search_seeds
    ):
        raise ValueError("agent_search.seeds must be a non-empty integer list")
    for key in ("overview_frames", "max_intervals", "max_turns"):
        values = search.get(key)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, int) and value > 0 for value in values
        ):
            raise ValueError(f"agent_search.{key} must be a non-empty positive integer list")
    local_fps = search.get("local_fps")
    if not isinstance(local_fps, list) or not local_fps or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in local_fps
    ):
        raise ValueError("agent_search.local_fps must be a non-empty positive number list")
    variants = search.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("agent_search.variants must be a non-empty pre-registered list")
    variant_ids: list[str] = []
    for index, raw in enumerate(variants):
        item = _mapping(raw, f"agent_search.variants[{index}]")
        variant_ids.append(_nonempty(item.get("id"), f"agent_search.variants[{index}].id"))
        for key in ("overview_frames", "max_intervals", "max_turns", "hierarchy_nodes", "hierarchy_depth"):
            if not isinstance(item.get(key), int) or item[key] <= 0:
                raise ValueError(f"agent_search.variants[{index}].{key} must be positive")
        for key in ("local_fps", "local_window_s"):
            if not isinstance(item.get(key), (int, float)) or isinstance(item[key], bool) or item[key] <= 0:
                raise ValueError(f"agent_search.variants[{index}].{key} must be positive")
        if item["overview_frames"] not in search["overview_frames"]:
            raise ValueError("variant overview_frames is outside the frozen search axis")
        if float(item["local_fps"]) not in [float(value) for value in search["local_fps"]]:
            raise ValueError("variant local_fps is outside the frozen search axis")
        if item["max_intervals"] not in search["max_intervals"] or item["max_turns"] not in search["max_turns"]:
            raise ValueError("variant interval/turn setting is outside the frozen search axes")
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("agent_search.variants ids must be unique")

    execution = _mapping(config.get("execution"), "execution")
    if not isinstance(execution.get("concurrency"), int) or execution["concurrency"] <= 0:
        raise ValueError("execution.concurrency must be positive")
    if not isinstance(execution.get("sample_seed"), int):
        raise ValueError("execution.sample_seed must be an integer")
    sft = _mapping(config.get("sft"), "sft")
    train600 = _mapping(sft.get("train600"), "sft.train600")
    _nonempty(train600.get("path"), "sft.train600.path")
    _sha(train600.get("sha256"), "sft.train600.sha256")
    if not isinstance(sft.get("trajectories_per_sample"), int) or sft[
        "trajectories_per_sample"
    ] <= 0:
        raise ValueError("sft.trajectories_per_sample must be positive")
    return config


def validate_inputs(config: Mapping[str, Any], *, check_files: bool = True) -> None:
    if not check_files:
        return
    workspace = Path(config["source_workspace"])
    evaluator = workspace / "scripts" / "evaluate_mcq.py"
    if not workspace.is_dir() or not evaluator.is_file():
        raise FileNotFoundError(f"source workspace/evaluator is missing: {evaluator}")
    for dataset in DATASETS:
        item = config["datasets"][dataset]
        annotations = Path(item["annotations"])
        video_root = Path(item["video_root"])
        if not annotations.is_file():
            raise FileNotFoundError(annotations)
        if not video_root.is_dir():
            raise FileNotFoundError(video_root)
        for split in ("train", "dev", "final"):
            path = Path(item[split]["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = file_sha256(path)
            expected = str(item[split]["sha256"]).lower()
            if actual != expected:
                raise RuntimeError(
                    f"{dataset} {split} manifest SHA-256 mismatch: expected {expected}, got {actual}"
                )
    train600 = config["sft"]["train600"]
    train600_path = Path(train600["path"])
    if not train600_path.is_file():
        raise FileNotFoundError(train600_path)
    train600_hash = file_sha256(train600_path)
    if train600_hash != str(train600["sha256"]).lower():
        raise RuntimeError("frozen Train600 SHA-256 mismatch")
    merged = hashlib.sha256()
    row_count = 0
    for dataset in DATASETS:
        train_path = Path(config["datasets"][dataset]["train"]["path"])
        for line in train_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            merged.update((canonical_json(row) + "\n").encode("utf-8"))
            row_count += 1
    if row_count != 600 or merged.hexdigest() != train600_hash:
        raise RuntimeError("frozen Train600 does not match the three Train200 manifests")


def _artifact(payload: Any, label: str, base: Path, *, check_files: bool) -> Artifact:
    item = _mapping(payload, label)
    path = Path(_nonempty(item.get("path"), f"{label}.path"))
    if not path.is_absolute():
        path = (base / path).resolve()
    expected = _sha(item.get("sha256"), f"{label}.sha256")
    if check_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return Artifact(path, expected)


def load_frozen_winner(
    path: Path,
    config: Mapping[str, Any],
    config_hash: str,
    *,
    check_files: bool = True,
) -> FrozenWinner:
    if check_files and not path.is_file():
        raise FileNotFoundError(path)
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "frozen winner")
    if payload.get("schema_version") != 2:
        raise ValueError(
            "frozen winner schema_version must be 2 and originate from the Dev selector"
        )
    frozen_config_hash = _sha(
        payload.get("experiment_config_sha256"),
        "frozen winner experiment_config_sha256",
    )
    if frozen_config_hash != config_hash:
        raise RuntimeError("frozen winner was selected from a different experiment config")
    model_key = _nonempty(payload.get("model_key"), "frozen winner model_key")
    if model_key not in config["models"]:
        raise ValueError(f"frozen winner has unknown model_key: {model_key}")
    protocol = _nonempty(payload.get("protocol"), "frozen winner protocol")
    if protocol not in config["protocols"]:
        raise ValueError(f"frozen winner has unknown protocol: {protocol}")
    seed = payload.get("seed")
    if not isinstance(seed, int):
        raise ValueError("frozen winner seed must be an integer")
    source = path.resolve()
    agent_config = _artifact(
        payload.get("agent_config"),
        "frozen winner agent_config",
        source.parent,
        check_files=check_files,
    )
    selection_report = _artifact(
        payload.get("selection_report"),
        "frozen winner selection_report",
        source.parent,
        check_files=check_files,
    )
    report = _mapping(
        json.loads(selection_report.path.read_text(encoding="utf-8")),
        "winner selection report",
    )
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise ValueError("winner selection report is not a passed schema-v1 report")
    report_state = _sha(
        report.get("selection_state_sha256"),
        "winner selection state",
    )
    computed_state = canonical_sha256(
        {key: value for key, value in report.items() if key != "selection_state_sha256"}
    )
    if report_state != computed_state or payload.get("selection_state_sha256") != report_state:
        raise RuntimeError("winner selection report state hash is invalid")
    if report.get("experiment_config_sha256") != config_hash:
        raise RuntimeError("winner selection report belongs to a different experiment config")
    if report.get("blocking_errors") != []:
        raise ValueError("winner selection report contains blocking errors")
    policy = _mapping(report.get("policy"), "winner selection policy")
    expected_policy = {
        "protocol_and_direct_selected_per_model": True,
        "agent_teacher_model_key": model_key,
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
    }
    if policy != expected_policy:
        raise ValueError("winner selection report used a non-frozen selection policy")
    strict_order = payload.get("strict_stage_order")
    if strict_order != list(SUPPORTED_AGENTS):
        raise ValueError("frozen winner does not prove strict A0-through-A4 evaluation")
    agent_selection = _mapping(report.get("agent_selection"), "winner agent selection")
    if agent_selection.get("strict_stage_order") != list(SUPPORTED_AGENTS):
        raise ValueError("selection report did not evaluate A0 through A4 in strict order")
    stages = agent_selection.get("stages")
    if not isinstance(stages, list) or [
        stage.get("stage") if isinstance(stage, dict) else None for stage in stages
    ] != list(SUPPORTED_AGENTS):
        raise ValueError("selection report stage trace is incomplete or out of order")
    if payload.get("source_run_plans") != report.get("source_run_plans"):
        raise RuntimeError("frozen winner source plans differ from the selection report")
    report_winner = _mapping(report.get("winner"), "winner selection report winner")
    expected_winner_fields = {
        "winner_id": payload.get("winner_id"),
        "model_key": model_key,
        "protocol": protocol,
        "seed": seed,
        "strategy": payload.get("strategy"),
        "variant_id": payload.get("variant_id"),
        "agent_config": payload.get("agent_config"),
    }
    if report_winner != expected_winner_fields:
        raise RuntimeError("frozen winner differs from its passed selection report")
    final_incumbent = _mapping(
        agent_selection.get("final_incumbent"),
        "winner final incumbent",
    )
    if (
        final_incumbent.get("kind") != "agent"
        or final_incumbent.get("strategy") != payload.get("strategy")
        or final_incumbent.get("variant_id") != payload.get("variant_id")
        or final_incumbent.get("agent_config") != payload.get("agent_config")
    ):
        raise RuntimeError("frozen winner differs from the final promoted incumbent")
    agent_payload = _mapping(
        json.loads(agent_config.path.read_text(encoding="utf-8")),
        "winner agent config",
    )
    agent_settings = _mapping(agent_payload.get("agent", agent_payload), "winner agent settings")
    strategy = _nonempty(agent_settings.get("strategy"), "winner agent strategy")
    if payload.get("strategy") != strategy:
        raise RuntimeError("frozen winner strategy differs from its agent config")
    if strategy not in SUPPORTED_AGENTS:
        if strategy in DISABLED_AGENTS:
            raise ValueError("A5 is disabled and cannot be a frozen winner")
        raise ValueError(f"unsupported frozen winner strategy: {strategy}")
    winner_id = _safe_id(str(payload.get("winner_id") or f"{strategy}_{model_key}_{protocol}"))
    return FrozenWinner(
        winner_id=winner_id,
        model_key=model_key,
        protocol=protocol,
        seed=seed,
        strategy=strategy,
        agent_config=agent_config,
        selection_report=selection_report,
        source_path=source,
        source_sha256=file_sha256(source),
    )


def load_frozen_checkpoint(
    path: Path,
    config: Mapping[str, Any],
    config_hash: str,
    winner: FrozenWinner,
    *,
    check_files: bool = True,
) -> FrozenCheckpoint:
    if check_files and not path.is_file():
        raise FileNotFoundError(path)
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "frozen checkpoint")
    if payload.get("schema_version") != 1:
        raise ValueError("frozen checkpoint schema_version must be 1")
    claimed_fingerprint = _sha(
        payload.get("checkpoint_fingerprint"), "checkpoint_fingerprint"
    )
    computed_fingerprint = canonical_sha256(
        {key: value for key, value in payload.items() if key != "checkpoint_fingerprint"}
    )
    if claimed_fingerprint != computed_fingerprint:
        raise RuntimeError("frozen checkpoint self-fingerprint is invalid")
    if payload.get("experiment_config_sha256") != config_hash:
        raise RuntimeError("frozen checkpoint belongs to a different experiment config")
    q9_hash = str(config["models"]["q9"]["artifact_sha256"])
    if payload.get("base_model_artifact_sha256") != q9_hash:
        raise RuntimeError("frozen checkpoint uses a different Qwen3.5-9B base")
    frozen = _mapping(payload.get("frozen_winner"), "checkpoint frozen_winner")
    if frozen.get("sha256") != winner.source_sha256:
        raise RuntimeError("checkpoint was trained from a different Agent winner")
    if frozen.get("winner_id") != winner.winner_id:
        raise RuntimeError("checkpoint winner_id differs from the frozen Agent winner")
    if frozen.get("agent_config") != {
        "path": str(winner.agent_config.path),
        "sha256": winner.agent_config.sha256,
    }:
        raise RuntimeError("checkpoint Agent config differs from the frozen winner")
    train_data = _mapping(payload.get("train_data"), "checkpoint train_data")
    train_path = Path(_nonempty(train_data.get("path"), "checkpoint train_data.path"))
    train_hash = _sha(train_data.get("sha256"), "checkpoint train_data.sha256")
    adapter = _mapping(payload.get("adapter"), "checkpoint adapter")
    adapter_path = Path(_nonempty(adapter.get("path"), "checkpoint adapter.path"))
    adapter_hash = _sha(
        adapter.get("artifact_sha256"), "checkpoint adapter.artifact_sha256"
    )
    adapter_files = adapter.get("files")
    if not isinstance(adapter_files, list) or not adapter_files:
        raise ValueError("checkpoint adapter.files must be a non-empty list")
    served_stack_hash = _sha(
        payload.get("served_stack_sha256"), "checkpoint served_stack_sha256"
    )
    epoch = payload.get("epoch")
    if epoch not in {1, 2, 3}:
        raise ValueError("checkpoint epoch must be 1, 2, or 3")
    if check_files:
        if not train_path.is_file() or file_sha256(train_path) != train_hash:
            raise RuntimeError("checkpoint SFT data changed or is unavailable")
        if not adapter_path.is_dir():
            raise FileNotFoundError(adapter_path)
        actual_files = [
            {
                "path": item.relative_to(adapter_path).as_posix(),
                "size": item.stat().st_size,
                "sha256": file_sha256(item),
            }
            for item in sorted(adapter_path.rglob("*"))
            if item.is_file() and item.name != "READY.txt"
        ]
        if actual_files != adapter_files:
            raise RuntimeError("checkpoint adapter files changed after freezing")
        if canonical_sha256(actual_files) != adapter_hash:
            raise RuntimeError("checkpoint adapter artifact SHA-256 is invalid")
    source = path.resolve()
    return FrozenCheckpoint(
        checkpoint_id=_safe_id(_nonempty(payload.get("checkpoint_id"), "checkpoint_id")),
        epoch=int(epoch),
        served_name=_nonempty(payload.get("served_name"), "checkpoint served_name"),
        base_url=_nonempty(payload.get("base_url"), "checkpoint base_url"),
        served_stack_sha256=served_stack_hash,
        adapter_path=adapter_path,
        adapter_artifact_sha256=adapter_hash,
        frozen_winner_sha256=winner.source_sha256,
        source_path=source,
        source_sha256=file_sha256(source),
    )


def load_frozen_sft_winner(
    path: Path,
    config: Mapping[str, Any],
    config_hash: str,
    winner: FrozenWinner,
    *,
    check_files: bool = True,
) -> FrozenSFTWinner:
    if check_files and not path.is_file():
        raise FileNotFoundError(path)
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "frozen SFT winner")
    if payload.get("schema_version") != 1:
        raise ValueError("frozen SFT winner schema_version must be 1")
    claimed = _sha(payload.get("winner_fingerprint"), "SFT winner_fingerprint")
    computed = canonical_sha256(
        {key: value for key, value in payload.items() if key != "winner_fingerprint"}
    )
    if claimed != computed:
        raise RuntimeError("frozen SFT winner self-fingerprint is invalid")
    if payload.get("experiment_config_sha256") != config_hash:
        raise RuntimeError("frozen SFT winner belongs to a different experiment")
    report_artifact = _artifact(
        payload.get("selection_report"),
        "SFT winner selection_report",
        path.resolve().parent,
        check_files=check_files,
    )
    report = _mapping(
        json.loads(report_artifact.path.read_text(encoding="utf-8")),
        "SFT selection report",
    )
    if report.get("status") != "passed" or report.get("experiment_config_sha256") != config_hash:
        raise ValueError("SFT selection report is not a passed report for this experiment")
    report_state = _sha(
        report.get("selection_state_sha256"), "SFT selection_state_sha256"
    )
    if report_state != canonical_sha256(
        {key: value for key, value in report.items() if key != "selection_state_sha256"}
    ):
        raise RuntimeError("SFT selection report state hash is invalid")
    selected = _mapping(report.get("selected"), "SFT selection selected checkpoint")
    if selected.get("checkpoint_id") != payload.get("checkpoint_id") or selected.get(
        "epoch"
    ) != payload.get("epoch"):
        raise RuntimeError("frozen SFT winner differs from its selection report")
    checkpoint_artifact = _artifact(
        payload.get("checkpoint_config"),
        "SFT winner checkpoint_config",
        path.resolve().parent,
        check_files=check_files,
    )
    if selected.get("checkpoint_config") != {
        "path": str(checkpoint_artifact.path),
        "sha256": checkpoint_artifact.sha256,
    }:
        raise RuntimeError("selected checkpoint config differs from SFT winner")
    checkpoint = load_frozen_checkpoint(
        checkpoint_artifact.path,
        config,
        config_hash,
        winner,
        check_files=check_files,
    )
    if checkpoint.checkpoint_id != payload.get("checkpoint_id"):
        raise RuntimeError("selected checkpoint identity differs from checkpoint config")
    return FrozenSFTWinner(
        checkpoint=checkpoint,
        selection_report=report_artifact,
        source_path=path.resolve(),
        source_sha256=file_sha256(path.resolve()),
    )


def _manifest(config: Mapping[str, Any], dataset: str, split: str) -> Artifact:
    item = config["datasets"][dataset][split]
    return Artifact(Path(item["path"]), str(item["sha256"]).lower())


def _agent_config(
    config: Mapping[str, Any],
    strategy: str,
    *,
    check_files: bool,
) -> Artifact:
    if strategy not in SUPPORTED_AGENTS:
        raise ValueError(f"unsupported or disabled strategy: {strategy}")
    path = Path(config["source_workspace"]) / "configs" / "agents" / f"{strategy}.json"
    if check_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
        settings = _mapping(payload.get("agent", payload), f"{path} agent")
        if settings.get("strategy") != strategy:
            raise RuntimeError(f"agent config strategy mismatch: {path}")
        digest = file_sha256(path)
    else:
        digest = "0" * 64
    return Artifact(path, digest)


def _base_agent_settings(artifact: Artifact, strategy: str) -> dict[str, Any]:
    if artifact.path.is_file():
        payload = _mapping(json.loads(artifact.path.read_text(encoding="utf-8")), str(artifact.path))
        settings = _mapping(payload.get("agent", payload), f"{artifact.path} agent")
        return dict(settings)
    return {
        "strategy": strategy,
        "overview_frames": 64,
        "local_fps": 1.0,
        "local_window_s": 120.0,
        "max_intervals": 3,
        "max_turns": 6,
        "max_frames_per_call": 128,
        "resize": 0.75,
        "hierarchy_nodes": 8,
        "hierarchy_depth": 3,
        "evidence_strategy": "a3_hierarchical_search",
    }


def materialize_agent_variant(
    config: Mapping[str, Any],
    strategy: str,
    base_config: Artifact,
    variant: SearchVariant,
    *,
    output_path: Path,
    write: bool,
    purpose: str,
) -> Artifact:
    settings = _base_agent_settings(base_config, strategy)
    if settings.get("strategy") != strategy:
        raise RuntimeError(f"base agent config strategy mismatch: {base_config.path}")
    uses_a3 = strategy == "a3_hierarchical_search" or (
        strategy == "a4_independent_arbitration"
        and settings.get("evidence_strategy", "a3_hierarchical_search")
        == "a3_hierarchical_search"
    )
    if uses_a3:
        settings["hierarchy_nodes"] = 8
    settings = _variant_settings(settings, variant)
    schedule = effective_schedule(strategy, settings)
    effective_fingerprint = canonical_sha256(schedule)
    payload = {
        "schema_version": 1,
        "config_id": f"{strategy}_{variant.variant_id}_{purpose}",
        "experiment_config_sha256": canonical_sha256(config),
        "source_agent_config": {
            "path": _path_arg(base_config.path),
            "sha256": base_config.sha256,
        },
        "search_variant": {
            **asdict(variant),
            "effective_schedule_fingerprint": effective_fingerprint,
        },
        "effective_schedule": schedule,
        "purpose": purpose,
        "agent": settings,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and file_sha256(output_path) != digest:
            raise RuntimeError(f"refusing to overwrite changed agent variant: {output_path}")
        if not output_path.exists():
            temporary = output_path.with_suffix(output_path.suffix + ".partial")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(output_path)
    return Artifact(output_path, digest)


def trajectory_variants(
    config: Mapping[str, Any],
    count: int,
    *,
    strategy: str,
    base_config: Artifact,
) -> list[SearchVariant]:
    search = config["agent_search"]
    if strategy in {"a3_hierarchical_search", "a4_independent_arbitration"}:
        raw_catalog = [
            SearchVariant(
                variant_id=(
                    f"root08_branch02_depth{depth}_fps{str(float(fps)).replace('.', 'p')}"
                    f"_window{int(window):03d}"
                ),
                overview_frames=64,
                local_fps=float(fps),
                max_intervals=4,
                max_turns=max(4, int(depth) + 2),
                hierarchy_nodes=8,
                hierarchy_depth=int(depth),
                local_window_s=float(window),
            )
            for depth, fps, window in product(
                (1, 2, 3), search["local_fps"], (60.0, 120.0)
            )
        ]
    else:
        raw_catalog = [
            SearchVariant(
                variant_id=(
                    f"ov{int(overview):03d}_fps{str(float(fps)).replace('.', 'p')}"
                    f"_int{int(intervals):02d}_turn{int(turns):02d}"
                ),
                overview_frames=int(overview),
                local_fps=float(fps),
                max_intervals=int(intervals),
                max_turns=int(turns),
            )
            for overview, fps, intervals, turns in product(
                search["overview_frames"],
                search["local_fps"],
                search["max_intervals"],
                search["max_turns"],
            )
        ]
    base_settings = _base_agent_settings(base_config, strategy)
    catalog: list[SearchVariant] = []
    seen_effective: set[str] = set()
    for candidate in raw_catalog:
        fingerprint = canonical_sha256(
            effective_schedule(strategy, _variant_settings(base_settings, candidate))
        )
        if fingerprint in seen_effective:
            continue
        seen_effective.add(fingerprint)
        catalog.append(
            SearchVariant(
                **{
                    **asdict(candidate),
                    "effective_schedule_fingerprint": fingerprint,
                }
            )
        )
    if count <= 0 or len(catalog) < count:
        raise ValueError(f"cannot select {count} trajectory variants from {len(catalog)} variants")
    if count == 1:
        return [catalog[len(catalog) // 2]]
    indices = [round(index * (len(catalog) - 1) / (count - 1)) for index in range(count)]
    selected = [catalog[index] for index in indices]
    if len({item.variant_id for item in selected}) != count:
        raise RuntimeError("trajectory variant selection produced duplicates")
    return selected


def _smoke_manifest(
    config: Mapping[str, Any],
    dataset: str,
    *,
    write: bool,
) -> Artifact:
    source = _manifest(config, dataset, "dev")
    if not source.path.is_file():
        raise FileNotFoundError(
            f"smoke phases require the real Dev manifest, even in dry-run: {source.path}"
        )
    if file_sha256(source.path) != source.sha256:
        raise RuntimeError(f"{dataset} Dev manifest changed before smoke derivation")
    rows = [line for line in source.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 10:
        raise ValueError(f"{dataset} Dev manifest has fewer than 10 samples")
    content = "".join(f"{line}\n" for line in rows[:10])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path = Path(config["result_root"]) / "frozen" / "smoke" / f"{dataset}_dev10.jsonl"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and file_sha256(path) != digest:
            raise RuntimeError(f"refusing to overwrite changed smoke manifest: {path}")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
    return Artifact(path, digest)


def _task_command(
    config: Mapping[str, Any],
    config_hash: str,
    *,
    phase: str,
    task_id: str,
    dataset: str,
    split: str,
    model_key: str,
    manifest: Artifact,
    backend: str,
    seed: int,
    resume: bool,
    extra: Iterable[str],
    agent_config_sha256: str | None = None,
    model_override: Mapping[str, Any] | None = None,
) -> TaskSpec:
    dataset_config = config["datasets"][dataset]
    model = model_override or config["models"][model_key]
    concurrency = int(config["execution"]["concurrency"])
    output_dir = Path(config["result_root"]) / phase / dataset / _safe_id(task_id)
    command = [
        sys.executable,
        "scripts/evaluate_mcq.py",
        "--dataset",
        dataset,
        "--backend",
        backend,
        "--annotations",
        str(dataset_config["annotations"]),
        "--video-root",
        str(dataset_config["video_root"]),
        "--base-url",
        str(model["base_url"]),
        "--model",
        str(model["served_name"]),
        "--model-artifact-sha256",
        str(model["artifact_sha256"]),
        "--manifest",
        _path_arg(manifest.path),
        "--expected-manifest-sha256",
        manifest.sha256,
        "--sample",
        {"train": "200", "dev": "50", "smoke": "10", "final": "100"}[split],
        "--seed",
        str(seed),
        "--output-dir",
        _path_arg(output_dir),
        "--concurrency",
        str(concurrency),
        "--experiment-config-sha256",
        config_hash,
    ]
    command.extend(str(item) for item in extra)
    if backend == "qwen_agent":
        command.extend(
            [
                "--frame-root",
                _path_arg(Path(config["result_root"]) / "frame_cache" / model_key),
            ]
        )
        if agent_config_sha256 is None:
            raise ValueError("qwen_agent task requires an agent config SHA-256")
        command.extend(
            ["--expected-agent-config-sha256", agent_config_sha256]
        )
    if resume:
        command.append("--resume")
    return TaskSpec(
        phase=phase,
        task_id=task_id,
        dataset=dataset,
        split=split,
        model_key=model_key,
        model=str(model["served_name"]),
        model_artifact_sha256=str(model["artifact_sha256"]),
        base_url=str(model["base_url"]),
        manifest=_path_arg(manifest.path),
        manifest_sha256=manifest.sha256,
        output_dir=_path_arg(output_dir),
        resume=resume,
        concurrency=concurrency,
        agent_config_sha256=agent_config_sha256,
        command=tuple(command),
    )


def build_tasks(
    config: Mapping[str, Any],
    phase: str,
    *,
    config_hash: str | None = None,
    resume: bool | None = None,
    frozen_winner: FrozenWinner | None = None,
    frozen_checkpoint: FrozenCheckpoint | None = None,
    frozen_sft_winner: FrozenSFTWinner | None = None,
    model_filter: str | None = None,
    protocol_filter: str | None = None,
    framework_filter: str | None = None,
    search_variant_id: str | None = None,
    final_model_group: str | None = None,
    check_files: bool = True,
    write_smoke: bool = False,
    write_variants: bool = False,
) -> list[TaskSpec]:
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    digest = config_hash or canonical_sha256(config)
    do_resume = bool(config["execution"].get("resume", False)) if resume is None else resume
    if model_filter is not None and model_filter not in config["models"]:
        raise ValueError(f"unknown model filter: {model_filter}")
    if protocol_filter is not None and protocol_filter not in config["protocols"]:
        raise ValueError(f"unknown protocol filter: {protocol_filter}")
    if framework_filter is not None and framework_filter not in SUPPORTED_AGENTS:
        if framework_filter in DISABLED_AGENTS:
            raise ValueError("A5 is disabled")
        raise ValueError(f"unknown framework filter: {framework_filter}")
    if phase not in {"protocol_smoke", "protocol_audit"} and phase not in {
        "final_test",
        "trajectory",
        "teacher_dev",
        "sft_dev",
        "final_matrix",
    }:
        if protocol_filter is None:
            raise ValueError(f"{phase} requires an explicit frozen --protocol selection")
    if phase == "agent_dev":
        if framework_filter is None:
            raise ValueError("agent_dev requires --framework so A0-A4 run one promotion stage at a time")
        if search_variant_id is None:
            raise ValueError("agent_dev requires --search-variant")
    if phase == "final_matrix" and (
        model_filter is not None
        or protocol_filter is not None
        or framework_filter is not None
        or search_variant_id is not None
    ):
        raise ValueError("final_matrix does not accept search filters")
    if phase == "final_matrix" and final_model_group not in {"q9", "q4", "sft9"}:
        raise ValueError("final_matrix requires --final-model-group q9|q4|sft9")
    if phase != "final_matrix" and final_model_group is not None:
        raise ValueError("final_model_group is only valid for final_matrix")
    model_keys = [key for key in ("q9", "q4") if model_filter in {None, key}]
    protocols = [
        item for item in config["protocols"] if protocol_filter in {None, item}
    ]
    sample_seed = int(config["execution"]["sample_seed"])
    tasks: list[TaskSpec] = []

    def add_baseline(
        dataset: str,
        model_key: str,
        protocol: str,
        mode: str,
        *,
        task_suffix: str,
        sampling: str | None = None,
        permutation_seed: int | None = None,
        mismatch_seed: int | None = None,
        split: str = "dev",
        manifest_override: Artifact | None = None,
    ) -> None:
        manifest = manifest_override or _manifest(config, dataset, "dev")
        extra = ["--baseline-mode", mode, "--qwen-protocol", protocol]
        if sampling:
            extra.extend(["--direct-sampling", sampling])
        if permutation_seed is not None:
            extra.extend(["--option-permutation-seed", str(permutation_seed)])
        if mismatch_seed is not None:
            mismatch = (
                Path(config["result_root"])
                / "frozen"
                / "mismatched_video_maps"
                / f"{dataset}_dev_seed{mismatch_seed}.json"
            )
            if check_files and not mismatch.is_file():
                raise FileNotFoundError(
                    f"build the frozen mismatched-video map before this phase: {mismatch}"
                )
            extra.extend(["--mismatched-video-map", str(mismatch)])
        task_id = _safe_id(f"{model_key}_{mode}_{protocol}_{task_suffix}")
        tasks.append(
            _task_command(
                config,
                digest,
                phase=phase,
                task_id=task_id,
                dataset=dataset,
                split=split,
                model_key=model_key,
                manifest=manifest,
                backend="qwen_baseline",
                seed=sample_seed,
                resume=do_resume,
                extra=extra,
            )
        )

    if phase in {"protocol_smoke", "protocol_audit"}:
        split = "smoke" if phase == "protocol_smoke" else "dev"
        smoke_manifests = (
            {dataset: _smoke_manifest(config, dataset, write=write_smoke) for dataset in DATASETS}
            if phase == "protocol_smoke" else {}
        )
        for dataset in DATASETS:
            for model_key in model_keys:
                for protocol in protocols:
                    add_baseline(
                        dataset,
                        model_key,
                        protocol,
                        "direct",
                        task_suffix="uniform64",
                        sampling="uniform64",
                        split=split,
                        manifest_override=smoke_manifests.get(dataset),
                    )
    elif phase == "blind_diagnostics":
        modes = list(config["diagnostics"]["modes"])
        diagnostic_seeds = [int(seed) for seed in config["diagnostics"]["seeds"]]
        for dataset in DATASETS:
            for model_key in model_keys:
                for protocol in protocols:
                    for mode in modes:
                        if mode == "permuted_choices":
                            for diagnostic_seed in diagnostic_seeds:
                                add_baseline(
                                    dataset,
                                    model_key,
                                    protocol,
                                    mode,
                                    task_suffix=f"seed{diagnostic_seed}",
                                    permutation_seed=diagnostic_seed,
                                )
                        elif mode == "mismatched_video":
                            for diagnostic_seed in diagnostic_seeds:
                                add_baseline(
                                    dataset,
                                    model_key,
                                    protocol,
                                    mode,
                                    task_suffix=f"uniform64_seed{diagnostic_seed}",
                                    sampling="uniform64",
                                    mismatch_seed=diagnostic_seed,
                                )
                        else:
                            add_baseline(
                                dataset,
                                model_key,
                                protocol,
                                mode,
                                task_suffix="base",
                            )
    elif phase == "direct_dev":
        for dataset in DATASETS:
            for model_key in model_keys:
                for protocol in protocols:
                    for sampling in config["direct_sampling"]:
                        sampling_id = str(sampling["id"])
                        add_baseline(
                            dataset,
                            model_key,
                            protocol,
                            "direct",
                            task_suffix=sampling_id,
                            sampling=sampling_id,
                        )
    elif phase in {"agent_smoke", "agent_dev"}:
        split = "smoke" if phase == "agent_smoke" else "dev"
        frameworks = [
            item
            for item in config["agent_search"]["framework_order"]
            if item in SUPPORTED_AGENTS and framework_filter in {None, item}
        ]
        seeds = (
            [sample_seed]
            if phase == "agent_smoke"
            else [int(seed) for seed in config["agent_search"]["seeds"]]
        )
        for dataset in DATASETS:
            manifest = (
                _smoke_manifest(config, dataset, write=write_smoke)
                if split == "smoke"
                else _manifest(config, dataset, "dev")
            )
            for model_key in model_keys:
                for protocol in protocols:
                    for framework in frameworks:
                        base_agent_config = _agent_config(
                            config, framework, check_files=check_files
                        )
                        variants: list[SearchVariant | None]
                        if phase == "agent_dev":
                            assert search_variant_id is not None
                            variants = (
                                list(search_variants(config))
                                if search_variant_id == "all"
                                else [select_search_variant(config, search_variant_id)]
                            )
                        else:
                            variants = [None]
                        for variant in variants:
                            agent_config = (
                                materialize_agent_variant(
                                    config,
                                    framework,
                                    base_agent_config,
                                    variant,
                                    output_path=(
                                        Path(config["result_root"])
                                        / "frozen"
                                        / "agent_variants"
                                        / framework
                                        / f"{variant.variant_id}.json"
                                    ),
                                    write=write_variants,
                                    purpose="dev_search",
                                )
                                if variant is not None
                                else base_agent_config
                            )
                            for agent_seed in seeds:
                                variant_suffix = (
                                    f"_{variant.variant_id}"
                                    if variant is not None
                                    else ""
                                )
                                task_id = _safe_id(
                                    f"{model_key}_{framework}_{protocol}{variant_suffix}_seed{agent_seed}"
                                )
                                tasks.append(
                                    _task_command(
                                        config,
                                        digest,
                                        phase=phase,
                                        task_id=task_id,
                                        dataset=dataset,
                                        split=split,
                                        model_key=model_key,
                                        manifest=manifest,
                                        backend="qwen_agent",
                                        seed=agent_seed,
                                        resume=do_resume,
                                        agent_config_sha256=agent_config.sha256,
                                        extra=(
                                            "--qwen-protocol",
                                            protocol,
                                            "--agent-config",
                                            _path_arg(agent_config.path),
                                        ),
                                    )
                                )
    elif phase in {"teacher_dev", "sft_dev"}:
        if frozen_winner is None:
            raise ValueError(f"{phase} requires --frozen-winner-config")
        if frozen_winner.model_key != "q9":
            raise ValueError(f"{phase} requires the frozen Qwen3.5-9B winner")
        if model_filter not in {None, "q9"}:
            raise ValueError(f"{phase} is a Qwen3.5-9B-only phase")
        if protocol_filter not in {None, frozen_winner.protocol}:
            raise ValueError("protocol filter conflicts with frozen winner")
        if phase == "sft_dev" and frozen_checkpoint is None:
            raise ValueError("sft_dev requires --sft-checkpoint-config")
        if phase == "teacher_dev" and frozen_checkpoint is not None:
            raise ValueError("teacher_dev cannot receive an SFT checkpoint")
        if frozen_checkpoint is None:
            model_key = "q9"
            model_override = None
            task_prefix = f"teacher_{frozen_winner.winner_id}"
        else:
            model_key = f"sft_{frozen_checkpoint.checkpoint_id}"
            model_override = {
                "served_name": frozen_checkpoint.served_name,
                "artifact_sha256": frozen_checkpoint.served_stack_sha256,
                "base_url": frozen_checkpoint.base_url,
            }
            task_prefix = frozen_checkpoint.checkpoint_id
        for dataset in DATASETS:
            manifest = _manifest(config, dataset, "dev")
            tasks.append(
                _task_command(
                    config,
                    digest,
                    phase=phase,
                    task_id=_safe_id(f"{task_prefix}_{dataset}"),
                    dataset=dataset,
                    split="dev",
                    model_key=model_key,
                    manifest=manifest,
                    backend="qwen_agent",
                    seed=frozen_winner.seed,
                    resume=do_resume,
                    agent_config_sha256=frozen_winner.agent_config.sha256,
                    model_override=model_override,
                    extra=(
                        "--qwen-protocol",
                        frozen_winner.protocol,
                        "--agent-config",
                        _path_arg(frozen_winner.agent_config.path),
                    ),
                )
            )
    elif phase == "final_matrix":
        if frozen_winner is None or frozen_sft_winner is None:
            raise ValueError(
                "final_matrix requires both the frozen untrained winner and a gate-passed SFT winner"
            )
        selection = _mapping(
            json.loads(frozen_winner.selection_report.path.read_text(encoding="utf-8")),
            "final matrix Dev selection report",
        )
        protocol_selection = _mapping(
            selection.get("protocol_selection"), "protocol_selection"
        )
        direct_selection = _mapping(selection.get("direct_selection"), "direct_selection")
        selected_protocols: dict[str, str] = {}
        selected_sampling: dict[str, str] = {}
        for key in ("q9", "q4"):
            protocol_item = _mapping(protocol_selection.get(key), f"protocol_selection.{key}")
            direct_item = _mapping(direct_selection.get(key), f"direct_selection.{key}")
            selected_protocols[key] = _nonempty(
                protocol_item.get("protocol"), f"protocol_selection.{key}.protocol"
            )
            selected_sampling[key] = _nonempty(
                direct_item.get("sampling"), f"direct_selection.{key}.sampling"
            )
            if selected_sampling[key] not in DIRECT_SAMPLING_IDS:
                raise ValueError(f"unsupported frozen Direct sampling for {key}")
        if selected_protocols["q9"] != frozen_winner.protocol:
            raise RuntimeError("final matrix Q9 protocol differs from frozen Agent winner")
        a0_config = _agent_config(config, "a0_eva_clean", check_files=check_files)
        for dataset in DATASETS:
            manifest = _manifest(config, dataset, "final")
            for key in ("q9", "q4"):
                if key != final_model_group:
                    continue
                protocol = selected_protocols[key]
                tasks.append(
                    _task_command(
                        config,
                        digest,
                        phase=phase,
                        task_id=f"{key}_no_video",
                        dataset=dataset,
                        split="final",
                        model_key=key,
                        manifest=manifest,
                        backend="qwen_baseline",
                        seed=sample_seed,
                        resume=do_resume,
                        extra=(
                            "--qwen-protocol",
                            protocol,
                            "--baseline-mode",
                            "question_choices",
                        ),
                    )
                )
                tasks.append(
                    _task_command(
                        config,
                        digest,
                        phase=phase,
                        task_id=f"{key}_direct_{selected_sampling[key]}",
                        dataset=dataset,
                        split="final",
                        model_key=key,
                        manifest=manifest,
                        backend="qwen_baseline",
                        seed=sample_seed,
                        resume=do_resume,
                        extra=(
                            "--qwen-protocol",
                            protocol,
                            "--baseline-mode",
                            "direct",
                            "--direct-sampling",
                            selected_sampling[key],
                        ),
                    )
                )
                for label, agent_config in (
                    ("eva_clean", a0_config),
                    ("best_untrained", frozen_winner.agent_config),
                ):
                    tasks.append(
                        _task_command(
                            config,
                            digest,
                            phase=phase,
                            task_id=f"{key}_{label}",
                            dataset=dataset,
                            split="final",
                            model_key=key,
                            manifest=manifest,
                            backend="qwen_agent",
                            seed=sample_seed,
                            resume=do_resume,
                            agent_config_sha256=agent_config.sha256,
                            extra=(
                                "--qwen-protocol",
                                protocol,
                                "--agent-config",
                                _path_arg(agent_config.path),
                            ),
                        )
                    )
            checkpoint = frozen_sft_winner.checkpoint
            if final_model_group == "sft9":
                tasks.append(
                    _task_command(
                        config,
                        digest,
                        phase=phase,
                        task_id=f"sft9_{checkpoint.checkpoint_id}",
                        dataset=dataset,
                        split="final",
                        model_key=f"sft_{checkpoint.checkpoint_id}",
                        model_override={
                            "served_name": checkpoint.served_name,
                            "artifact_sha256": checkpoint.served_stack_sha256,
                            "base_url": checkpoint.base_url,
                        },
                        manifest=manifest,
                        backend="qwen_agent",
                        seed=sample_seed,
                        resume=do_resume,
                        agent_config_sha256=frozen_winner.agent_config.sha256,
                        extra=(
                            "--qwen-protocol",
                            frozen_winner.protocol,
                            "--agent-config",
                            _path_arg(frozen_winner.agent_config.path),
                        ),
                    )
                )
    elif phase in {"final_test", "trajectory"}:
        if frozen_winner is None:
            raise ValueError(f"{phase} requires --frozen-winner-config")
        if phase == "trajectory" and frozen_winner.model_key != "q9":
            raise ValueError("trajectory generation requires the frozen Qwen3.5-9B winner")
        if model_filter is not None and model_filter != frozen_winner.model_key:
            raise ValueError("model filter conflicts with frozen winner")
        split = "final" if phase == "final_test" else "train"
        trajectory_count = (
            1 if phase == "final_test" else int(config["sft"]["trajectories_per_sample"])
        )
        trajectory_schedule = (
            []
            if phase == "final_test"
            else trajectory_variants(
                config,
                trajectory_count,
                strategy=frozen_winner.strategy,
                base_config=frozen_winner.agent_config,
            )
        )
        for dataset in DATASETS:
            manifest = _manifest(config, dataset, split)
            for trajectory_index in range(trajectory_count):
                seed = (
                    frozen_winner.seed
                    if phase == "final_test"
                    else frozen_winner.seed + trajectory_index * 1009
                )
                suffix = "final" if phase == "final_test" else f"trajectory_{trajectory_index:02d}"
                task_id = _safe_id(f"{frozen_winner.winner_id}_{suffix}")
                if phase == "trajectory":
                    variant = trajectory_schedule[trajectory_index]
                    agent_config = materialize_agent_variant(
                        config,
                        frozen_winner.strategy,
                        frozen_winner.agent_config,
                        variant,
                        output_path=(
                            Path(config["result_root"])
                            / "frozen"
                            / "trajectory_variants"
                            / frozen_winner.winner_id
                            / f"trajectory_{trajectory_index:02d}_{variant.variant_id}.json"
                        ),
                        write=write_variants,
                        purpose=f"trajectory_{trajectory_index:02d}",
                    )
                else:
                    agent_config = frozen_winner.agent_config
                extra = [
                    "--qwen-protocol",
                    frozen_winner.protocol,
                    "--agent-config",
                    _path_arg(agent_config.path),
                ]
                if phase == "trajectory":
                    extra.extend(
                        [
                            "--defer-scoring",
                            "--train600-manifest-sha256",
                            str(config["sft"]["train600"]["sha256"]),
                            "--trajectory-schedule-id",
                            f"trajectory_{trajectory_index:02d}_{variant.variant_id}",
                        ]
                    )
                tasks.append(
                    _task_command(
                        config,
                        digest,
                        phase=phase,
                        task_id=task_id,
                        dataset=dataset,
                        split=split,
                        model_key=frozen_winner.model_key,
                        manifest=manifest,
                        backend="qwen_agent",
                        seed=seed,
                        resume=do_resume,
                        agent_config_sha256=agent_config.sha256,
                        extra=extra,
                    )
                )
    return tasks


def build_run_plan(
    config_path: Path,
    config: Mapping[str, Any],
    phase: str,
    tasks: list[TaskSpec],
    *,
    frozen_winner: FrozenWinner | None,
    frozen_checkpoint: FrozenCheckpoint | None,
    frozen_sft_winner: FrozenSFTWinner | None,
    model_filter: str | None,
    protocol_filter: str | None,
    framework_filter: str | None,
    search_variant_id: str | None,
    final_model_group: str | None,
) -> dict[str, Any]:
    config_hash = canonical_sha256(config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": phase,
        "model_filter": model_filter,
        "protocol_filter": protocol_filter,
        "framework_filter": framework_filter,
        "search_variant_id": search_variant_id,
        "final_model_group": final_model_group,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        "frozen_winner": (
            {
                "path": str(frozen_winner.source_path),
                "sha256": frozen_winner.source_sha256,
                "winner_id": frozen_winner.winner_id,
                "agent_config_sha256": frozen_winner.agent_config.sha256,
                "selection_report_sha256": frozen_winner.selection_report.sha256,
            }
            if frozen_winner is not None
            else None
        ),
        "frozen_checkpoint": (
            {
                "path": str(frozen_checkpoint.source_path),
                "sha256": frozen_checkpoint.source_sha256,
                "checkpoint_id": frozen_checkpoint.checkpoint_id,
                "epoch": frozen_checkpoint.epoch,
                "served_stack_sha256": frozen_checkpoint.served_stack_sha256,
            }
            if frozen_checkpoint is not None
            else None
        ),
        "frozen_sft_winner": (
            {
                "path": str(frozen_sft_winner.source_path),
                "sha256": frozen_sft_winner.source_sha256,
                "checkpoint_id": frozen_sft_winner.checkpoint.checkpoint_id,
                "selection_report_sha256": frozen_sft_winner.selection_report.sha256,
            }
            if frozen_sft_winner is not None
            else None
        ),
        "disabled_frameworks": sorted(DISABLED_AGENTS),
        "task_count": len(tasks),
        "tasks": [asdict(task) for task in tasks],
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def freeze_run_plan(path: Path, plan: Mapping[str, Any], *, resume: bool) -> str:
    expected = str(plan["plan_sha256"])
    if path.exists():
        existing = _mapping(json.loads(path.read_text(encoding="utf-8")), "existing run plan")
        actual = str(existing.get("plan_sha256") or "")
        if actual != canonical_sha256({key: value for key, value in existing.items() if key != "plan_sha256"}):
            raise RuntimeError(f"existing run plan has an invalid self-hash: {path}")
        if actual != expected:
            raise RuntimeError(f"resume plan/config hash conflict: {path}")
        if not resume:
            raise FileExistsError(f"run plan already exists; pass --resume: {path}")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return expected


def assert_safe_execution(tasks: list[TaskSpec]) -> None:
    endpoints: dict[str, set[str]] = {}
    for task in tasks:
        endpoints.setdefault(task.base_url, set()).add(task.model)
        if Path(task.command[1]).name != "evaluate_mcq.py":
            raise RuntimeError("refusing to execute a non-evaluator command")
    conflicts = {url: names for url, names in endpoints.items() if len(names) > 1}
    if conflicts:
        raise RuntimeError(
            "multiple served models share one endpoint; execute one --model-key phase at a time: "
            + canonical_json({key: sorted(value) for key, value in conflicts.items()})
        )


def preflight_endpoints(tasks: list[TaskSpec], *, timeout_s: float = 10.0) -> None:
    checked: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.base_url.rstrip("/"), task.model)
        if key in checked:
            continue
        checked.add(key)
        endpoint = f"{key[0]}/models"
        request = urllib.request.Request(
            endpoint,
            headers={"Authorization": "Bearer no", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.load(response)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(f"model endpoint preflight failed for {endpoint}: {exc}") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        served = {
            str(item.get("id"))
            for item in data or []
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        if task.model not in served:
            raise RuntimeError(
                f"endpoint {endpoint} does not serve exact model {task.model!r}; available={sorted(served)}"
            )


def execute_tasks(
    tasks: list[TaskSpec],
    source_workspace: Path,
    *,
    endpoint_preflight: bool = True,
) -> None:
    assert_safe_execution(tasks)
    if endpoint_preflight:
        preflight_endpoints(tasks)
    if not source_workspace.is_dir():
        raise FileNotFoundError(source_workspace)
    for task in tasks:
        subprocess.run(list(task.command), cwd=source_workspace, check=True)


def audit_protocol_smoke(tasks: list[TaskSpec]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    length_truncations: list[dict[str, str]] = []
    row_count = 0
    for task in tasks:
        result_files = sorted(
            Path(task.output_dir).glob(f"{task.dataset}_*.jsonl")
        )
        if len(result_files) != 1:
            issues.append(
                {
                    "task_id": task.task_id,
                    "sample_id": "",
                    "reason": f"expected_one_result_jsonl_found_{len(result_files)}",
                }
            )
            continue
        rows = [
            json.loads(line)
            for line in result_files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 10:
            issues.append(
                {
                    "task_id": task.task_id,
                    "sample_id": "",
                    "reason": f"expected_10_rows_found_{len(rows)}",
                }
            )
        seen: set[str] = set()
        protocol_index = task.command.index("--qwen-protocol") + 1
        protocol = task.command[protocol_index]
        for row in rows:
            row_count += 1
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                issues.append(
                    {
                        "task_id": task.task_id,
                        "sample_id": sample_id,
                        "reason": "missing_or_duplicate_sample_id",
                    }
                )
            seen.add(sample_id)
            attempts = row.get("request_attempts")
            had_length = bool(row.get("length_retry_used")) or (
                isinstance(attempts, list)
                and any(
                    isinstance(attempt, Mapping)
                    and attempt.get("finish_reason") == "length"
                    for attempt in attempts
                )
            )
            if protocol == "think" and had_length:
                length_truncations.append(
                    {"task_id": task.task_id, "sample_id": sample_id}
                )
            failure = next(
                (
                    key
                    for key in (
                        "error",
                        "error_type",
                        "parse_error",
                        "model_parse_failure",
                        "data_unavailable",
                        "control_unavailable",
                    )
                    if row.get(key)
                ),
                None,
            )
            if failure is None and row.get("prediction") is None:
                failure = "prediction_missing"
            if failure is None and row.get("annotation_leak_check") != "passed":
                failure = "annotation_leak_check_not_passed"
            if failure is None and int(row.get("candidate_rerun") or 0) != 0:
                failure = "candidate_rerun_nonzero"
            if failure is None and row.get("media_items") != 1:
                failure = "direct_video_missing"
            if failure is None and row.get("sampling_id") != "uniform64":
                failure = "sampling_id_mismatch"
            if failure is None and row.get("sampled_frames_estimated") != 64:
                failure = "sampled_frame_request_mismatch"
            if failure is None and row.get("visual_usage_complete") is not True:
                failure = "visual_token_accounting_incomplete"
            if failure is not None:
                issues.append(
                    {
                        "task_id": task.task_id,
                        "sample_id": sample_id,
                        "reason": str(failure),
                    }
                )
    status = "passed" if not issues and not length_truncations else "blocked"
    return {
        "schema_version": 1,
        "status": status,
        "task_count": len(tasks),
        "row_count": row_count,
        "engineering_issues": issues,
        "thinking_length_truncations": length_truncations,
        "required_action": (
            "set the frozen think protocol max_tokens to 32768 and rerun the entire protocol smoke"
            if length_truncations
            else None
        ),
    }


def shell_line(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Qwen-only Agent experiment matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-key", choices=("q9", "q4"))
    parser.add_argument(
        "--protocol",
        choices=("no_think", "think"),
        help="Frozen protocol selected after protocol_audit; required by later Dev phases.",
    )
    parser.add_argument(
        "--framework",
        choices=SUPPORTED_AGENTS,
        help="Run one A0-A4 promotion stage; required by agent_dev.",
    )
    parser.add_argument(
        "--search-variant",
        help="One pre-registered variant id, or 'all'; required by agent_dev.",
    )
    parser.add_argument("--frozen-winner-config", type=Path)
    parser.add_argument("--sft-checkpoint-config", type=Path)
    parser.add_argument("--frozen-sft-winner", type=Path)
    parser.add_argument("--final-model-group", choices=("q9", "q4", "sft9"))
    parser.add_argument(
        "--allow-missing-inputs",
        action="store_true",
        help="Preview server paths locally; smoke phases still require real Dev manifests.",
    )
    args = parser.parse_args()
    if args.allow_missing_inputs and not args.dry_run:
        parser.error("--allow-missing-inputs is only valid with --dry-run")
    if args.phase in {
        "final_test",
        "trajectory",
        "teacher_dev",
        "sft_dev",
        "final_matrix",
    } and args.frozen_winner_config is None:
        parser.error(f"{args.phase} requires explicit --frozen-winner-config")
    if args.phase == "sft_dev" and args.sft_checkpoint_config is None:
        parser.error("sft_dev requires --sft-checkpoint-config")
    if args.phase != "sft_dev" and args.sft_checkpoint_config is not None:
        parser.error("--sft-checkpoint-config is only valid with sft_dev")
    if args.phase == "final_matrix" and (
        args.frozen_sft_winner is None or args.final_model_group is None
    ):
        parser.error("final_matrix requires --frozen-sft-winner and --final-model-group")
    if args.phase != "final_matrix" and (
        args.frozen_sft_winner is not None or args.final_model_group is not None
    ):
        parser.error("SFT winner/final model group are only valid with final_matrix")
    if args.phase in {"blind_diagnostics", "direct_dev", "agent_smoke", "agent_dev"}:
        if args.protocol is None:
            parser.error(f"{args.phase} requires explicit --protocol selected after audit")
    if args.phase == "agent_dev" and args.framework is None:
        parser.error("agent_dev requires --framework")
    if args.phase == "agent_dev" and args.search_variant is None:
        parser.error("agent_dev requires --search-variant")

    config = load_config(args.config)
    check_files = not args.allow_missing_inputs
    validate_inputs(config, check_files=check_files)
    config_hash = canonical_sha256(config)
    winner = None
    if args.frozen_winner_config is not None:
        winner = load_frozen_winner(
            args.frozen_winner_config,
            config,
            config_hash,
            check_files=check_files,
        )
    checkpoint = None
    if args.sft_checkpoint_config is not None:
        if winner is None:
            raise AssertionError("validated SFT checkpoint phase lost its winner")
        checkpoint = load_frozen_checkpoint(
            args.sft_checkpoint_config,
            config,
            config_hash,
            winner,
            check_files=check_files,
        )
    sft_winner = None
    if args.frozen_sft_winner is not None:
        if winner is None:
            raise AssertionError("validated final matrix lost its untrained winner")
        sft_winner = load_frozen_sft_winner(
            args.frozen_sft_winner,
            config,
            config_hash,
            winner,
            check_files=check_files,
        )
    effective_resume = args.resume or bool(config["execution"].get("resume", False))
    tasks = build_tasks(
        config,
        args.phase,
        config_hash=config_hash,
        resume=effective_resume,
        frozen_winner=winner,
        frozen_checkpoint=checkpoint,
        frozen_sft_winner=sft_winner,
        model_filter=args.model_key,
        protocol_filter=args.protocol,
        framework_filter=args.framework,
        search_variant_id=args.search_variant,
        final_model_group=args.final_model_group,
        check_files=check_files,
        write_smoke=not args.dry_run,
        write_variants=not args.dry_run,
    )
    plan = build_run_plan(
        args.config,
        config,
        args.phase,
        tasks,
        frozen_winner=winner,
        frozen_checkpoint=checkpoint,
        frozen_sft_winner=sft_winner,
        model_filter=args.model_key,
        protocol_filter=args.protocol,
        framework_filter=args.framework,
        search_variant_id=args.search_variant,
        final_model_group=args.final_model_group,
    )
    print(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "phase": args.phase,
                "config_sha256": config_hash,
                "plan_sha256": plan["plan_sha256"],
                "task_count": len(tasks),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for task in tasks:
        print(shell_line(task.command))
    if args.dry_run:
        return

    assert_safe_execution(tasks)
    suffix_parts = [
        item
        for item in (
            args.model_key,
            args.protocol,
            args.framework,
            args.search_variant,
            checkpoint.checkpoint_id if checkpoint is not None else None,
            args.final_model_group,
        )
        if item
    ]
    suffix = f"_{'_'.join(_safe_id(item) for item in suffix_parts)}" if suffix_parts else ""
    plan_path = Path(config["result_root"]) / "run_plans" / f"{args.phase}{suffix}.json"
    freeze_run_plan(plan_path, plan, resume=effective_resume)
    execute_tasks(tasks, Path(config["source_workspace"]))
    if args.phase == "protocol_smoke":
        audit = audit_protocol_smoke(tasks)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        if audit["status"] != "passed":
            raise RuntimeError(
                "protocol smoke failed; inspect engineering_issues and thinking_length_truncations"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from flashvid_eval.privacy import assert_deferred_result_public
from flashvid_eval.qwen_sft import canonical_sha256, sha256_file


DATASETS = ("lvbench", "lsdbench", "cgbench")
SCHEDULES_PER_DATASET = 12
SAMPLES_PER_DATASET = 200


def _require_sha256(value: Any, field: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return raw


def _read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield line_number, row


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _command_value(command: list[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"trajectory task has invalid {flag}")
    return command[indices[0] + 1]


def _validate_plan(path: Path, config_hash: str) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("trajectory run plan must be schema v1")
    claimed = str(plan.get("plan_sha256") or "")
    computed = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if claimed != computed or plan.get("config_sha256") != config_hash:
        raise RuntimeError("trajectory plan hash/config mismatch")
    if plan.get("phase") != "trajectory":
        raise ValueError("input plan is not a trajectory phase")
    return plan


def collect_bundle(
    *,
    config_path: Path,
    trajectory_plan: Path,
    counterfactual_paths: list[Path],
    rescue_index_path: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    config_hash = canonical_sha256(config)
    plan = _validate_plan(trajectory_plan, config_hash)
    model_hash = _require_sha256(
        config["models"]["q9"]["artifact_sha256"], "Qwen3.5-9B artifact"
    )
    train600_hash = _require_sha256(
        config["sft"]["train600"]["sha256"], "Train600 manifest"
    )
    dataset_hash_by_name = {
        dataset: _require_sha256(
            config["datasets"][dataset]["train"]["sha256"],
            f"{dataset} Train200 manifest",
        )
        for dataset in DATASETS
    }
    dataset_hashes = set(dataset_hash_by_name.values())
    tasks = plan.get("tasks")
    expected_task_count = len(DATASETS) * SCHEDULES_PER_DATASET
    if not isinstance(tasks, list) or len(tasks) != expected_task_count:
        raise ValueError("trajectory plan must contain 12 schedules x 3 datasets")
    trajectory_files: list[dict[str, str]] = []
    agent_hashes: set[str] = set()
    runner_hashes: set[str] = set()
    seen_task_ids: set[tuple[str, str]] = set()
    task_counts: Counter[str] = Counter()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("trajectory task is not an object")
        dataset = str(task.get("dataset") or "")
        task_id = str(task.get("task_id") or "")
        identity = (dataset, task_id)
        if dataset not in DATASETS or not task_id or identity in seen_task_ids:
            raise ValueError("trajectory plan has invalid/duplicate task identity")
        seen_task_ids.add(identity)
        task_counts[dataset] += 1
        dataset_hash = dataset_hash_by_name[dataset]
        if task.get("manifest_sha256") != dataset_hash:
            raise RuntimeError("trajectory task uses a different Train200 manifest")
        command = list(task.get("command") or [])
        if _command_value(command, "--train600-manifest-sha256") != train600_hash:
            raise RuntimeError("trajectory task uses a different Train600")
        agent_hash = _require_sha256(
            task.get("agent_config_sha256"), "trajectory Agent config"
        )
        if _command_value(command, "--expected-agent-config-sha256") != agent_hash:
            raise RuntimeError("trajectory task Agent hash is not command-bound")
        output_dir = Path(str(task.get("output_dir") or ""))
        result_path = _one(output_dir, f"{dataset}_*.jsonl")
        frozen_path = _one(output_dir, f"frozen_inputs_{dataset}_*.json")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected_runner = _require_sha256(
            frozen.get("run_fingerprint"), "trajectory runner fingerprint"
        )
        frozen_agent = frozen.get("agent_config")
        if (
            not isinstance(frozen_agent, Mapping)
            or frozen.get("dataset") != dataset
            or frozen.get("manifest", {}).get("sha256") != dataset_hash
            or frozen.get("model") != "Qwen3.5-9B"
            or frozen.get("model_artifact_sha256") != model_hash
            or frozen.get("experiment_config_sha256") != config_hash
            or frozen.get("scoring_deferred") is not True
            or frozen.get("train600_manifest_sha256") != train600_hash
            or frozen_agent.get("sha256") != agent_hash
        ):
            raise RuntimeError(f"trajectory frozen-input mismatch: {frozen_path}")
        rows = 0
        seen_samples: set[str] = set()
        for line_number, row in _read_rows(result_path):
            assert_deferred_result_public(row)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen_samples:
                raise ValueError(f"{result_path}:{line_number}: duplicate/empty sample")
            seen_samples.add(sample_id)
            if (
                row.get("scoring_deferred") is not True
                or row.get("model") != "Qwen3.5-9B"
                or row.get("model_artifact_sha256") != model_hash
                or row.get("dataset_manifest_sha256")
                != task.get("manifest_sha256")
                or row.get("train600_manifest_sha256") != train600_hash
                or row.get("manifest_sha256") != train600_hash
                or row.get("agent_config_sha256") != agent_hash
                or row.get("trajectory_runner_fingerprint") != expected_runner
                or row.get("runner_fingerprint") != expected_runner
            ):
                raise RuntimeError(f"{result_path}:{line_number}: provenance mismatch")
            rows += 1
        if rows != SAMPLES_PER_DATASET:
            raise ValueError(
                f"{result_path}: expected {SAMPLES_PER_DATASET} rows, found {rows}"
            )
        trajectory_files.append(
            {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)}
        )
        agent_hashes.add(agent_hash)
        runner_hashes.add(expected_runner)

    if task_counts != Counter({dataset: SCHEDULES_PER_DATASET for dataset in DATASETS}):
        raise ValueError(f"trajectory plan schedule distribution mismatch: {task_counts}")

    all_dataset_hashes = set(dataset_hashes)
    rescue_files = 0
    if rescue_index_path is not None:
        rescue = json.loads(rescue_index_path.read_text(encoding="utf-8"))
        if not isinstance(rescue, dict) or rescue.get("schema_version") != 1:
            raise ValueError("rescue index must be schema v1")
        claimed = str(rescue.get("rescue_index_sha256") or "")
        computed = canonical_sha256(
            {key: value for key, value in rescue.items() if key != "rescue_index_sha256"}
        )
        if claimed != computed:
            raise RuntimeError("rescue index self-hash mismatch")
        config_ref = rescue.get("experiment_config")
        base_ref = rescue.get("source_base_bundle")
        if (
            not isinstance(config_ref, Mapping)
            or config_ref.get("canonical_sha256") != config_hash
            or not isinstance(base_ref, Mapping)
            or rescue.get("train600_manifest_sha256") != train600_hash
            or rescue.get("model_artifact_sha256") != model_hash
            or rescue.get("variant_id") != "rescue"
        ):
            raise RuntimeError("rescue index provenance mismatch")
        base_bundle_path = Path(str(base_ref.get("path") or ""))
        if not base_bundle_path.is_file() or sha256_file(base_bundle_path) != base_ref.get(
            "sha256"
        ):
            raise RuntimeError("rescue source base bundle changed")
        base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
        if base_bundle.get("bundle_sha256") != base_ref.get("bundle_sha256"):
            raise RuntimeError("rescue source base bundle identity mismatch")
        base_files = sorted(
            base_bundle.get("trajectory_files") or [], key=lambda item: item["path"]
        )
        if base_files != sorted(trajectory_files, key=lambda item: item["path"]):
            raise RuntimeError("rescue was discovered from a different base matrix")
        rescue_agent = rescue.get("agent_config")
        manifests = rescue.get("manifests")
        if not isinstance(rescue_agent, Mapping) or not isinstance(manifests, Mapping):
            raise ValueError("rescue index has no Agent/manifests")
        rescue_agent_path = Path(str(rescue_agent.get("path") or ""))
        rescue_agent_hash = _require_sha256(
            rescue_agent.get("sha256"), "rescue Agent config"
        )
        if (
            not rescue_agent_path.is_file()
            or sha256_file(rescue_agent_path) != rescue_agent_hash
        ):
            raise RuntimeError("rescue Agent config changed")
        for dataset in DATASETS:
            manifest_ref = manifests.get(dataset)
            if not isinstance(manifest_ref, Mapping):
                raise ValueError(f"rescue index has no {dataset} manifest")
            expected_rows = int(manifest_ref.get("count") or 0)
            if expected_rows < 0:
                raise ValueError(f"{dataset} rescue manifest count cannot be negative")
            manifest_path = Path(str(manifest_ref.get("path") or ""))
            manifest_hash = _require_sha256(
                manifest_ref.get("sha256"), f"{dataset} rescue manifest"
            )
            if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_hash:
                raise RuntimeError(f"{dataset} rescue manifest changed")
            manifest_ids = {
                str(row.get("sample_id") or "")
                for _, row in _read_rows(manifest_path)
            }
            if "" in manifest_ids or len(manifest_ids) != expected_rows:
                raise ValueError(f"{dataset} rescue manifest count/ID mismatch")
            if expected_rows == 0:
                continue
            output_dir = Path(str(manifest_ref.get("result_dir") or ""))
            result_path = _one(output_dir, f"{dataset}_*.jsonl")
            frozen_path = _one(output_dir, f"frozen_inputs_{dataset}_*.json")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            runner_hash = _require_sha256(
                frozen.get("run_fingerprint"), "rescue runner fingerprint"
            )
            if (
                frozen.get("dataset") != dataset
                or frozen.get("manifest", {}).get("sha256") != manifest_hash
                or frozen.get("model") != "Qwen3.5-9B"
                or frozen.get("model_artifact_sha256") != model_hash
                or frozen.get("experiment_config_sha256") != config_hash
                or frozen.get("scoring_deferred") is not True
                or frozen.get("train600_manifest_sha256") != train600_hash
                or frozen.get("trajectory_schedule_id") != rescue.get("schedule_id")
                or frozen.get("trajectory_variant_id") != "rescue"
                or frozen.get("agent_config", {}).get("sha256") != rescue_agent_hash
            ):
                raise RuntimeError(f"rescue frozen-input mismatch: {frozen_path}")
            seen_samples: set[str] = set()
            for line_number, row in _read_rows(result_path):
                assert_deferred_result_public(row)
                sample_id = str(row.get("sample_id") or "")
                if not sample_id or sample_id in seen_samples:
                    raise ValueError(
                        f"{result_path}:{line_number}: duplicate/empty rescue sample"
                    )
                seen_samples.add(sample_id)
                if (
                    row.get("variant_id") != "rescue"
                    or row.get("trajectory_runner_fingerprint") != runner_hash
                    or row.get("runner_fingerprint") != runner_hash
                    or row.get("dataset_manifest_sha256") != manifest_hash
                    or row.get("train600_manifest_sha256") != train600_hash
                    or row.get("manifest_sha256") != train600_hash
                    or row.get("agent_config_sha256") != rescue_agent_hash
                    or row.get("model_artifact_sha256") != model_hash
                    or row.get("scoring_deferred") is not True
                ):
                    raise RuntimeError(f"{result_path}:{line_number}: rescue provenance mismatch")
            if seen_samples != manifest_ids:
                raise ValueError(f"{dataset} rescue result coverage mismatch")
            trajectory_files.append(
                {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)}
            )
            all_dataset_hashes.add(manifest_hash)
            agent_hashes.add(rescue_agent_hash)
            runner_hashes.add(runner_hash)
            rescue_files += 1

    for path in counterfactual_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        sidecar = path.with_suffix(path.suffix + ".frozen.json")
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        frozen = json.loads(sidecar.read_text(encoding="utf-8"))
        run_fingerprint = _require_sha256(
            frozen.get("run_fingerprint"), "counterfactual runner fingerprint"
        )
        dataset = str(frozen.get("dataset") or "")
        dataset_hash = str(frozen.get("manifest_sha256") or "")
        agent_hash = _require_sha256(
            frozen.get("agent_config_sha256"), "counterfactual Agent config"
        )
        if (
            dataset not in DATASETS
            or dataset_hash not in all_dataset_hashes
            or frozen.get("train600_manifest_sha256") != train600_hash
            or frozen.get("model") != "Qwen3.5-9B"
            or frozen.get("model_artifact_sha256") != model_hash
        ):
            raise RuntimeError(f"counterfactual sidecar provenance mismatch: {sidecar}")
        rows = 0
        seen_trajectory_ids: set[str] = set()
        for line_number, row in _read_rows(path):
            assert_deferred_result_public(row)
            trajectory_id = str(row.get("trajectory_id") or "")
            if not trajectory_id or trajectory_id in seen_trajectory_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate/empty trajectory_id"
                )
            seen_trajectory_ids.add(trajectory_id)
            if (
                row.get("counterfactual_run_fingerprint") != run_fingerprint
                or row.get("dataset_manifest_sha256") != dataset_hash
                or row.get("train600_manifest_sha256") != train600_hash
                or row.get("manifest_sha256") != train600_hash
                or row.get("agent_config_sha256") != agent_hash
                or row.get("model_artifact_sha256") != model_hash
                or row.get("scoring_deferred") is not True
            ):
                raise RuntimeError(f"{path}:{line_number}: provenance mismatch")
            rows += 1
        if not rows:
            raise ValueError(f"counterfactual file is empty: {path}")
        trajectory_files.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
        )
        agent_hashes.add(agent_hash)
        runner_hashes.add(run_fingerprint)

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "config_sha256": config_hash,
        "train600_manifest_sha256": train600_hash,
        "trajectory_run_plan": {
            "path": str(trajectory_plan.resolve()),
            "sha256": sha256_file(trajectory_plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "counterfactual_files": len(counterfactual_paths),
        "rescue_files": rescue_files,
        "trajectory_files": sorted(trajectory_files, key=lambda item: item["path"]),
        "expected_provenance": {
            "model": "Qwen3.5-9B",
            "model_artifact_sha256": model_hash,
            "dataset_manifest_sha256s": sorted(all_dataset_hashes),
            "agent_config_sha256s": sorted(agent_hashes),
            "runner_fingerprints": sorted(runner_hashes),
        },
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and validate frozen Qwen trajectory inputs for SFT filtering."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory-run-plan", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, action="append", default=[])
    parser.add_argument("--rescue-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = collect_bundle(
        config_path=args.config,
        trajectory_plan=args.trajectory_run_plan,
        counterfactual_paths=args.counterfactual,
        rescue_index_path=args.rescue_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8")) != bundle:
            raise RuntimeError(f"refusing to overwrite changed input bundle: {args.output}")
    else:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode("utf-8"))
        temporary.replace(args.output)
    print(
        json.dumps(
            {
                "trajectory_files": len(bundle["trajectory_files"]),
                "runner_fingerprints": len(
                    bundle["expected_provenance"]["runner_fingerprints"]
                ),
                "bundle_sha256": bundle["bundle_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()

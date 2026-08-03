from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from flashvid_eval.budget_reporting import (
    DATASETS,
    MethodSpec,
    aggregate_dataset_metrics,
    pareto_acceptance,
    read_jsonl_latest,
    summarize_records,
)


RESULT_NAME = "{dataset}_flashvid_hybrid.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, str]:
    return {
        relative: _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if (relative := path.relative_to(root).as_posix()) not in excluded_names
    }


def _validate_selected_materialization(
    selection: dict[str, Any],
    checkpoint: str,
    selected_root: Path,
) -> Path:
    metadata_path = selected_root / "selected_checkpoint.json"
    if not metadata_path.is_file():
        raise ValueError(f"selected validation metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("selected validation metadata must be a JSON object")
    if metadata.get("selected_checkpoint") != checkpoint:
        raise ValueError(
            "selected validation materialization does not match checkpoint selection"
        )

    expected_files = metadata.get("files")
    if not isinstance(expected_files, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in expected_files.items()
    ):
        raise TypeError("selected validation metadata files must be a hash mapping")
    selected_files = _tree_hashes(
        selected_root,
        excluded_names=frozenset({metadata_path.name}),
    )
    if selected_files != expected_files:
        raise ValueError("selected validation tree hashes do not match metadata")

    raw_source = metadata.get("source")
    if not isinstance(raw_source, str) or not raw_source:
        raise TypeError("selected validation metadata source must be a path")
    source = Path(raw_source).resolve()
    if not source.is_dir():
        raise ValueError(f"selected validation source not found: {source}")
    if _tree_hashes(source) != expected_files:
        raise ValueError("selected validation source tree hashes do not match metadata")

    source_files_by_checkpoint = selection.get("source_files")
    if not isinstance(source_files_by_checkpoint, dict):
        raise TypeError("checkpoint selection source_files must be a mapping")
    source_files = source_files_by_checkpoint.get(checkpoint)
    if not isinstance(source_files, list) or not source_files:
        raise ValueError(
            f"checkpoint selection has no source files for {checkpoint}"
        )
    for raw_path in source_files:
        if not isinstance(raw_path, str) or not raw_path:
            raise TypeError("checkpoint selection source file paths must be strings")
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(source).as_posix()
        except ValueError as error:
            raise ValueError(
                f"checkpoint selection source file is outside materialized source: {path}"
            ) from error
        if relative not in expected_files or not path.is_file():
            raise ValueError(
                f"checkpoint selection source file is absent from metadata: {path}"
            )
        if _sha256(path) != expected_files[relative]:
            raise ValueError(
                f"checkpoint selection source file hash changed: {path}"
            )
    return metadata_path


def _load_method(
    root: Path,
    *,
    expected_counts: dict[str, int],
) -> tuple[
    dict[str, Any],
    dict[str, set[str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    datasets: dict[str, Any] = {}
    sample_ids: dict[str, set[str]] = {}
    answers: dict[str, dict[str, str]] = {}
    files: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        path = root / dataset / RESULT_NAME.format(dataset=dataset)
        if not path.is_file():
            raise ValueError(f"validation result not found: {path}")
        loaded = read_jsonl_latest(path)
        expected = expected_counts[dataset]
        if loaded.invalid_lines:
            raise ValueError(f"{path}: {loaded.invalid_lines} invalid JSONL lines")
        if loaded.duplicate_sample_ids:
            raise ValueError(
                f"{path}: duplicate sample IDs: {loaded.duplicate_sample_ids}"
            )
        if len(loaded.records) != expected:
            raise ValueError(
                f"{path}: expected {expected} records, got {len(loaded.records)}"
            )
        wrong_dataset = [
            str(row.get("sample_id"))
            for row in loaded.records
            if str(row.get("dataset") or "").strip().lower() != dataset
        ]
        if wrong_dataset:
            raise ValueError(f"{path}: records have wrong dataset: {wrong_dataset[:3]}")
        metrics = summarize_records(
            loaded.records,
            expected=expected,
            duplicates=loaded.duplicate_sample_ids,
            invalid_lines=loaded.invalid_lines,
        )
        if metrics["tokens"]["retained_token_coverage"] != expected:
            raise ValueError(f"{path}: retained visual tokens are not complete")
        datasets[dataset] = metrics
        missing_sample_ids = [
            index
            for index, row in enumerate(loaded.records)
            if not str(row.get("sample_id") or "").strip()
        ]
        if missing_sample_ids:
            raise ValueError(
                f"{path}: records are missing sample_id at rows {missing_sample_ids[:3]}"
            )
        sample_ids[dataset] = {
            str(row["sample_id"]) for row in loaded.records
        }
        answers[dataset] = {
            str(row["sample_id"]): str(
                row.get("answer")
                or row.get("correct_answer")
                or row.get("right_answer")
                or ""
            ).strip().upper()
            for row in loaded.records
        }
        files[dataset] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    return (
        {
            "datasets": datasets,
            "aggregate": aggregate_dataset_metrics(datasets),
        },
        sample_ids,
        answers,
        files,
    )


def evaluate_sft_pareto_gate(
    *,
    selection_path: Path,
    selected_root: Path,
    untrained_root: Path,
    fixed100_root: Path,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    decision = selection.get("selection", selection)
    checkpoint = str(decision.get("selected_checkpoint") or "")
    if decision.get("status") != "complete" or not checkpoint:
        raise ValueError(f"checkpoint selection is not complete: {decision}")

    metadata_path = _validate_selected_materialization(
        selection,
        checkpoint,
        selected_root,
    )

    methods: dict[str, Any] = {}
    identities: dict[str, dict[str, set[str]]] = {}
    answers: dict[str, dict[str, dict[str, str]]] = {}
    files: dict[str, dict[str, dict[str, str]]] = {}
    for name, root in (
        ("sft", selected_root),
        ("untrained_4b", untrained_root),
        ("fixed_100", fixed100_root),
    ):
        method, method_ids, method_answers, method_files = _load_method(
            root,
            expected_counts=expected_counts,
        )
        methods[name] = method
        identities[name] = method_ids
        answers[name] = method_answers
        files[name] = method_files

    for dataset in DATASETS:
        reference_ids = identities["sft"][dataset]
        reference_answers = answers["sft"][dataset]
        for name in ("untrained_4b", "fixed_100"):
            if identities[name][dataset] != reference_ids:
                raise ValueError(
                    f"{dataset}: {name} and selected SFT use different sample IDs"
                )
            if answers[name][dataset] != reference_answers:
                raise ValueError(
                    f"{dataset}: {name} and selected SFT use different answers"
                )

    specs = {
        "sft": MethodSpec("sft", "validation", {}, role="sft_dynamic"),
        "untrained_4b": MethodSpec(
            "untrained_4b",
            "validation",
            {},
            role="untrained_dynamic",
        ),
        "fixed_100": MethodSpec(
            "fixed_100",
            "validation",
            {},
            role="fixed_100",
            fixed_ratio=1.0,
        ),
    }
    acceptance = pareto_acceptance(methods, specs, "validation")
    passed = acceptance.get("status") == "complete" and acceptance.get("passed") is True
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "selected_checkpoint": checkpoint,
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
        },
        "selected_validation_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": _sha256(metadata_path),
        },
        "inputs": files,
        "methods": methods,
        "acceptance": acceptance,
    }


def _parse_expected(values: list[str]) -> dict[str, int]:
    expected = {dataset: 50 for dataset in DATASETS}
    for value in values:
        if "=" not in value:
            raise ValueError("--expected must use DATASET=COUNT")
        dataset, raw_count = value.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset in --expected: {dataset}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("expected count must be positive")
        expected[dataset] = count
    return expected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hard-gate a selected SFT checkpoint against the approved 150-sample "
            "validation Pareto criteria."
        )
    )
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--untrained-root", type=Path, required=True)
    parser.add_argument("--fixed100-root", type=Path, required=True)
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = evaluate_sft_pareto_gate(
            selection_path=args.checkpoint_selection,
            selected_root=args.selected_root,
            untrained_root=args.untrained_root,
            fixed100_root=args.fixed100_root,
            expected_counts=_parse_expected(args.expected),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "passed": False,
            "error": str(error),
        }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

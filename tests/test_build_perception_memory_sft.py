from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import build_perception_memory_sft as builder

ROLE_RUNTIME_VERSION = "role_separated_visual_csv_v2"
ROLE_PROMPT_BUNDLE_SHA256 = "6" * 64
CONTROLLER_CONSTRAINT_VERSION = "eva_tool_call_regex_v2"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _trajectory(dataset: str, sample_id: str, trajectory_id: str, marker: str) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "model": "Qwen3.5-9B",
        "model_artifact_sha256": "a" * 64,
        "diagnostics_gate_sha256": "9" * 64,
        "candidate_results_sha256": "8" * 64,
        "config_sha256": "b" * 64,
        "experiment_config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "train600_manifest_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "prompt_hashes": {"controller": "f" * 64},
        "request_trace": [
            {
                "prompt_hash": marker * 64,
                "messages": [
                    {"role": "system", "content": "Frozen protocol."},
                    {"role": "user", "content": "Question and choices."},
                ],
            }
        ],
    }


def _role_separated_trajectory(
    dataset: str, sample_id: str, trajectory_id: str, marker: str
) -> dict:
    row = _trajectory(dataset, sample_id, trajectory_id, marker)
    row.pop("diagnostics_gate_sha256")
    row["training_source_lock_sha256"] = "7" * 64
    row["role_separated_runtime_version"] = ROLE_RUNTIME_VERSION
    row["role_prompt_schema_bundle_sha256"] = ROLE_PROMPT_BUNDLE_SHA256
    row["controller_output_constraint_version"] = CONTROLLER_CONSTRAINT_VERSION
    return row


def _role_contract_kwargs() -> dict[str, str]:
    return {
        "expected_training_source_lock_sha256": "7" * 64,
        "expected_role_separated_runtime_version": ROLE_RUNTIME_VERSION,
        "expected_role_prompt_schema_bundle_sha256": ROLE_PROMPT_BUNDLE_SHA256,
        "expected_controller_output_constraint_version": (
            CONTROLLER_CONSTRAINT_VERSION
        ),
    }


def _record(trajectory_id: str, role: str) -> dict:
    target = "stop" if role == "planner" else "memory"
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": target, "loss": True},
        ],
        "metadata": {
            "trajectory_id": trajectory_id,
            "process_role": role,
            "terminal_prefix_index": 0,
        },
    }


def _install_export_stubs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    gate_calls: list[dict] = []

    def fake_export(row: dict, **_kwargs: object) -> tuple[dict, ...]:
        trajectory_id = row["trajectory_id"]
        return (_record(trajectory_id, "planner"),)

    def fake_gate(trajectories: list[dict], records: list[dict], **kwargs: int) -> dict:
        selected_by_dataset: dict[str, int] = {}
        for row in trajectories:
            dataset = str(row["dataset"])
            selected_by_dataset[dataset] = selected_by_dataset.get(dataset, 0) + 1
        gate_calls.append(
            {
                "trajectory_ids": [row["trajectory_id"] for row in trajectories],
            "record_ids": [row["metadata"]["record_id"] for row in records],
                "kwargs": kwargs,
            }
        )
        return {
            "selected_trajectories": len(trajectories),
            "selected_by_dataset": selected_by_dataset,
            "candidate_fixes": 0,
            "candidate_fixes_by_dataset": {
                dataset: 0 for dataset in selected_by_dataset
            },
            "candidate_training_strata": {
                "candidate_correct": 0,
                "candidate_wrong": 0,
            },
            "visual_path_distribution": {
                "single_frame_select": 0,
                "timestamp_grounded_select": 0,
                "hierarchical_refinement": 0,
                "multi_interval_exploration": 0,
            },
            "prefixes": {"complete": len(trajectories)},
            "planner_decisions": {
                "observed_incomplete_continue": 0,
                "stop": len(trajectories),
            },
            "role_episodes": {"planner": len(trajectories)},
            "assistant_targets": {"stop": len(trajectories)},
            "sft_records": len(records),
        }

    monkeypatch.setattr(builder, "build_perception_memory_sft_records", fake_export)
    monkeypatch.setattr(builder, "enforce_perception_memory_selection_gate", fake_gate)
    return gate_calls


def test_build_sorts_records_and_freezes_hash_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_calls = _install_export_stubs(monkeypatch)
    first = tmp_path / "part-b.jsonl"
    second = tmp_path / "part-a.jsonl"
    row_b = _trajectory("lsdbench", "sample-b", "trajectory-b", "1")
    row_a = _trajectory("lvbench", "sample-a", "trajectory-a", "2")
    _write_jsonl(first, [row_b])
    _write_jsonl(second, [row_a])
    output = tmp_path / "sft.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = builder.build(
        selected_paths=[first, second],
        output=output,
        summary_path=summary_path,
        completion_gate_kind="legacy_prefix_judge",
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    identities = [
        (
            row["metadata"]["trajectory_id"],
            row["metadata"]["process_role"],
        )
        for row in records
    ]
    assert identities == [
        ("trajectory-a", "planner"),
        ("trajectory-b", "planner"),
    ]
    assert len({row["metadata"]["record_id"] for row in records}) == 2
    assert gate_calls[0]["trajectory_ids"] == ["trajectory-a", "trajectory-b"]
    assert gate_calls[0]["kwargs"] == {"completion_gate_kind": "legacy_prefix_judge"}
    assert summary["selection_quality_gate"]["quantity_is_advisory"] is True
    assert summary == json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["outputs"]["sft_jsonl"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    input_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in summary["selected_inputs"]
    }
    assert input_hashes == {
        first.name: hashlib.sha256(first.read_bytes()).hexdigest(),
        second.name: hashlib.sha256(second.read_bytes()).hexdigest(),
    }
    coverage = summary["provenance_coverage"]
    assert coverage["models"] == {
        "covered_rows": 2,
        "missing_rows": 0,
        "values": ["Qwen3.5-9B"],
    }
    assert coverage["hash_fields"]["config_sha256"]["values"] == ["b" * 64]
    assert coverage["hash_fields"]["manifest_sha256"]["values"] == ["c" * 64]
    assert coverage["hash_fields"]["model_artifact_sha256"]["values"] == [
        "a" * 64
    ]
    assert coverage["request_prompt_hashes"] == {
        "covered_requests": 2,
        "missing_requests": 0,
        "values": ["1" * 64, "2" * 64],
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                _trajectory("lvbench", "same", "one", "1"),
                _trajectory("lvbench", "same", "two", "2"),
            ],
            "duplicate selected sample",
        ),
        (
            [
                _trajectory("lvbench", "one", "same", "1"),
                _trajectory("lsdbench", "two", "same", "2"),
            ],
            "duplicate trajectory_id",
        ),
    ],
)
def test_build_rejects_duplicate_sample_and_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
    message: str,
) -> None:
    _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(selected, rows)

    with pytest.raises(ValueError, match=message):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "sft.jsonl",
            summary_path=tmp_path / "summary.json",
            completion_gate_kind="legacy_prefix_judge",
        )

    assert not (tmp_path / "sft.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


def test_build_rejects_duplicate_stable_record_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _trajectory("lvbench", "sample", "trajectory", "1")
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(selected, [row])
    duplicate = _record("trajectory", "planner")
    monkeypatch.setattr(
        builder,
        "build_perception_memory_sft_records",
        lambda _row, **_kwargs: (duplicate, deepcopy(duplicate)),
    )

    with pytest.raises(ValueError, match="duplicate process-SFT record_id"):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "sft.jsonl",
            summary_path=tmp_path / "summary.json",
            completion_gate_kind="legacy_prefix_judge",
        )


def test_overwrite_is_explicit_and_invalid_input_preserves_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected.jsonl"
    rows = [
        _trajectory("lvbench", "good", "trajectory-good", "1"),
        _trajectory("lsdbench", "bad", "trajectory-bad", "2"),
    ]
    _write_jsonl(selected, rows)
    output = tmp_path / "sft.jsonl"
    summary_path = tmp_path / "summary.json"
    output.write_bytes(b"old-sft\n")
    summary_path.write_bytes(b"old-summary\n")

    with pytest.raises(FileExistsError, match="--overwrite"):
        builder.build(
            selected_paths=[selected],
            output=output,
            summary_path=summary_path,
        )

    def fail_on_bad(row: dict, **_kwargs: object) -> tuple[dict, ...]:
        if row["sample_id"] == "bad":
            raise ValueError("invalid selected trajectory")
        return (_record(row["trajectory_id"], "planner"),)

    monkeypatch.setattr(builder, "build_perception_memory_sft_records", fail_on_bad)
    with pytest.raises(ValueError, match="invalid selected trajectory"):
        builder.build(
            selected_paths=[selected],
            output=output,
            summary_path=summary_path,
            overwrite=True,
        )

    assert output.read_bytes() == b"old-sft\n"
    assert summary_path.read_bytes() == b"old-summary\n"


def test_invalid_row_never_creates_partial_formal_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(selected, [_trajectory("lvbench", "sample", "trajectory", "1")])
    monkeypatch.setattr(
        builder,
        "build_perception_memory_sft_records",
        lambda _row, **_kwargs: (_ for _ in ()).throw(ValueError("broken trajectory")),
    )
    output = tmp_path / "nested" / "sft.jsonl"
    summary_path = tmp_path / "nested" / "summary.json"

    with pytest.raises(ValueError, match="broken trajectory"):
        builder.build(
            selected_paths=[selected],
            output=output,
            summary_path=summary_path,
            overwrite=True,
        )

    assert not output.exists()
    assert not summary_path.exists()
    assert not output.parent.exists()


def test_small_corpus_is_advisory_and_does_not_block_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_calls = _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(
        selected,
        [_trajectory("lvbench", "sample", "trajectory", "1")],
    )

    summary = builder.build(
        selected_paths=[selected],
        output=tmp_path / "sft.jsonl",
        summary_path=tmp_path / "summary.json",
        completion_gate_kind="legacy_prefix_judge",
    )

    assert gate_calls[0]["kwargs"] == {"completion_gate_kind": "legacy_prefix_judge"}
    assert summary["selection_quality_gate"]["passed"] is True
    assert summary["quantity_distribution"]["selected_trajectories"] == 1


def test_role_separated_build_uses_training_source_lock_instead_of_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(
        selected,
        [_role_separated_trajectory("lvbench", "sample", "trajectory", "1")],
    )

    summary = builder.build(
        selected_paths=[selected],
        output=tmp_path / "sft.jsonl",
        summary_path=tmp_path / "summary.json",
        completion_gate_kind="legacy_prefix_judge",
        **_role_contract_kwargs(),
    )

    coverage = summary["provenance_coverage"]["hash_fields"]
    assert coverage["diagnostics_gate_sha256"]["covered_rows"] == 0
    assert coverage["training_source_lock_sha256"]["values"] == ["7" * 64]
    assert summary["export_policy"]["training_source_lock_sha256"] == "7" * 64
    assert summary["export_policy"]["role_runtime_contract"] == {
        "role_separated_runtime_version": ROLE_RUNTIME_VERSION,
        "role_prompt_schema_bundle_sha256": ROLE_PROMPT_BUNDLE_SHA256,
        "controller_output_constraint_version": CONTROLLER_CONSTRAINT_VERSION,
    }


def test_build_rejects_mixed_or_wrong_training_source_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    row = _role_separated_trajectory("lvbench", "sample", "trajectory", "1")
    row["diagnostics_gate_sha256"] = "9" * 64
    _write_jsonl(selected, [row])

    with pytest.raises(ValueError, match="exactly one complete source gate"):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "mixed.jsonl",
            summary_path=tmp_path / "mixed.json",
            completion_gate_kind="legacy_prefix_judge",
            **_role_contract_kwargs(),
        )

    row.pop("diagnostics_gate_sha256")
    _write_jsonl(selected, [row])
    with pytest.raises(ValueError, match="not bound to the expected"):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "wrong.jsonl",
            summary_path=tmp_path / "wrong.json",
            completion_gate_kind="legacy_prefix_judge",
            expected_training_source_lock_sha256="6" * 64,
            expected_role_separated_runtime_version=ROLE_RUNTIME_VERSION,
            expected_role_prompt_schema_bundle_sha256=ROLE_PROMPT_BUNDLE_SHA256,
            expected_controller_output_constraint_version=(
                CONTROLLER_CONSTRAINT_VERSION
            ),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("role_separated_runtime_version", None),
        ("role_prompt_schema_bundle_sha256", None),
        ("controller_output_constraint_version", None),
        ("role_separated_runtime_version", "drifted-runtime"),
        ("role_prompt_schema_bundle_sha256", "4" * 64),
        ("controller_output_constraint_version", "drifted-constraint"),
    ),
)
def test_role_separated_build_rejects_missing_or_drifted_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str | None,
) -> None:
    _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    row = _role_separated_trajectory("lvbench", "sample", "trajectory", "1")
    if replacement is None:
        row.pop(field)
    else:
        row[field] = replacement
    _write_jsonl(selected, [row])

    with pytest.raises(ValueError, match="not bound to the expected"):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "sft.jsonl",
            summary_path=tmp_path / "summary.json",
            completion_gate_kind="legacy_prefix_judge",
            **_role_contract_kwargs(),
        )


def test_experiment_config_filter_excludes_rows_without_rewriting_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_calls = _install_export_stubs(monkeypatch)
    selected = tmp_path / "selected.jsonl"
    included = _trajectory("lvbench", "included", "trajectory-included", "1")
    excluded = _trajectory("cgbench", "excluded", "trajectory-excluded", "2")
    excluded["experiment_config_sha256"] = "7" * 64
    _write_jsonl(selected, [included, excluded])

    summary = builder.build(
        selected_paths=[selected],
        output=tmp_path / "sft.jsonl",
        summary_path=tmp_path / "summary.json",
        completion_gate_kind="legacy_prefix_judge",
        include_experiment_config_sha256="b" * 64,
    )

    assert gate_calls[0]["trajectory_ids"] == ["trajectory-included"]
    audit = summary["selected_filter"]
    assert audit["input_rows"] == 2
    assert audit["included_rows"] == 1
    assert audit["excluded_rows"] == 1
    assert audit["excluded_trajectories"] == [
        {
            "dataset": "cgbench",
            "sample_id": "excluded",
            "trajectory_id": "trajectory-excluded",
            "experiment_config_sha256": "7" * 64,
        }
    ]

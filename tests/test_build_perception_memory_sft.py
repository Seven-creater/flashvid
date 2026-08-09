from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import build_perception_memory_sft as builder


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


def _record(trajectory_id: str, prefix_index: int, target: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": target, "loss": True},
        ],
        "metadata": {
            "trajectory_id": trajectory_id,
            "prefix_index": prefix_index,
            "episode_target_type": target,
        },
    }


def _install_export_stubs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    gate_calls: list[dict] = []

    def fake_export(row: dict) -> tuple[dict, ...]:
        trajectory_id = row["trajectory_id"]
        return (
            _record(trajectory_id, 0, "memory"),
            _record(trajectory_id, -1, "tool"),
        )

    def fake_gate(trajectories: list[dict], records: list[dict], **kwargs: int) -> dict:
        gate_calls.append(
            {
                "trajectory_ids": [row["trajectory_id"] for row in trajectories],
                "record_ids": [row["metadata"]["record_id"] for row in records],
                "thresholds": kwargs,
            }
        )
        return {
            "selected_trajectories": len(trajectories),
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
        minimum_total=2,
        minimum_per_dataset=0,
        minimum_candidate_fixes=0,
        minimum_candidate_fixes_per_dataset=0,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    identities = [
        (
            row["metadata"]["trajectory_id"],
            row["metadata"]["prefix_index"],
            row["metadata"]["episode_target_type"],
        )
        for row in records
    ]
    assert identities == [
        ("trajectory-a", -1, "tool"),
        ("trajectory-a", 0, "memory"),
        ("trajectory-b", -1, "tool"),
        ("trajectory-b", 0, "memory"),
    ]
    assert len({row["metadata"]["record_id"] for row in records}) == 4
    assert gate_calls[0]["trajectory_ids"] == ["trajectory-a", "trajectory-b"]
    assert gate_calls[0]["thresholds"] == {
        "minimum_total": 2,
        "minimum_per_dataset": 0,
        "minimum_candidate_fixes": 0,
        "minimum_candidate_fixes_per_dataset": 0,
    }
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
            minimum_total=0,
            minimum_per_dataset=0,
            minimum_candidate_fixes=0,
            minimum_candidate_fixes_per_dataset=0,
        )

    assert not (tmp_path / "sft.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


def test_build_rejects_duplicate_stable_record_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _trajectory("lvbench", "sample", "trajectory", "1")
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(selected, [row])
    duplicate = _record("trajectory", 0, "memory")
    monkeypatch.setattr(
        builder,
        "build_perception_memory_sft_records",
        lambda _row: (duplicate, deepcopy(duplicate)),
    )

    with pytest.raises(ValueError, match="duplicate process-SFT record_id"):
        builder.build(
            selected_paths=[selected],
            output=tmp_path / "sft.jsonl",
            summary_path=tmp_path / "summary.json",
            minimum_total=0,
            minimum_per_dataset=0,
            minimum_candidate_fixes=0,
            minimum_candidate_fixes_per_dataset=0,
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

    def fail_on_bad(row: dict) -> tuple[dict, ...]:
        if row["sample_id"] == "bad":
            raise ValueError("invalid selected trajectory")
        return (_record(row["trajectory_id"], 0, "memory"),)

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
        lambda _row: (_ for _ in ()).throw(ValueError("broken trajectory")),
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

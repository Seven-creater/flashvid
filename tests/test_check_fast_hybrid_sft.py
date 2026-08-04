from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_fast_hybrid_sft import validate_fast_hybrid_sft_data


DATASETS = ("lvbench", "lsdbench", "cgbench")
CONFIG_SHA = "c" * 64
DATASET_SHA = {"lvbench": "a" * 64, "lsdbench": "b" * 64, "cgbench": "d" * 64}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    manifest_path = tmp_path / "train600.jsonl"
    selected_path = tmp_path / "selected_pruned.jsonl"
    sft_path = tmp_path / "fast_hybrid_sft.jsonl"
    summary_path = tmp_path / "fast_hybrid_sft_summary.json"
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")

    manifest_rows = [
        {"dataset": dataset, "sample_id": f"{dataset}-one", "answer": "A"}
        for dataset in DATASETS
    ]
    _write_jsonl(manifest_path, manifest_rows)
    manifest_sha = _sha256(manifest_path)

    selected_rows: list[dict] = []
    sft_rows: list[dict] = []
    for dataset in DATASETS:
        sample_id = f"{dataset}-one"
        trajectory_id = f"{dataset}:{sample_id}:compressed:0"
        selected_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "manifest_sha256": manifest_sha,
                "train600_manifest_sha256": manifest_sha,
                "dataset_manifest_sha256": DATASET_SHA[dataset],
                "config_sha256": CONFIG_SHA,
                "scoring_deferred": True,
                "_selection_stable": True,
                "_selection_confirmation_count": 3,
                "_compression_finalized": True,
                "annotation_leak_check": "passed",
                "prediction": "A",
                "final_prediction": "A",
                "candidate_answer": "A" if dataset == "lvbench" else "B",
                "fallback_to_candidate": dataset == "lvbench",
                "tool_steps": [
                    {
                        "start_time": 0.0,
                        "end_time": 5.0,
                        "nframes": 1,
                        "resize": 1.0,
                        "actual_timestamps": [1.0],
                        "frame_paths": [str(frame)],
                    }
                ],
            }
        )
        common_metadata = {
            "schema_version": 1,
            "dataset": dataset,
            "sample_id": sample_id,
            "trajectory_id": trajectory_id,
            "manifest_sha256": manifest_sha,
            "train600_manifest_sha256": manifest_sha,
            "dataset_manifest_sha256": DATASET_SHA[dataset],
            "config_sha256": CONFIG_SHA,
        }
        sft_rows.extend(
            [
                {
                    "messages": [
                        {"role": "system", "content": "Use EVA tools."},
                        {"role": "user", "content": "Question and choices"},
                        {
                            "role": "assistant",
                            "content": (
                                '<tool_call>{"tool":"frame_select","arguments":'
                                '{"start_time":0,"end_time":5,"nframes":1,'
                                '"resize":1.0}}</tool_call>'
                            ),
                            "loss": True,
                        },
                    ],
                    "metadata": {
                        **common_metadata,
                        "assistant_target_types": ["tool"],
                        "episode_id": f"{trajectory_id}#turn-000",
                        "turn_index": 0,
                        "episode_target_type": "tool",
                    },
                },
                {
                    "messages": [
                        {"role": "system", "content": "Use EVA tools."},
                        {"role": "user", "content": "Question and choices"},
                        {"role": "tool", "content": "Observed frame: <image>"},
                        {"role": "assistant", "content": "Answer: A", "loss": True},
                    ],
                    "metadata": {
                        **common_metadata,
                        "assistant_target_types": ["final"],
                        "episode_id": f"{trajectory_id}#turn-001",
                        "turn_index": 1,
                        "episode_target_type": "final",
                    },
                    "images": [str(frame)],
                },
            ]
        )

    _write_jsonl(selected_path, selected_rows)
    _write_jsonl(sft_path, sft_rows)
    summary = {
        "selected": 3,
        "selected_by_dataset": {dataset: 1 for dataset in DATASETS},
        "sft_records": 6,
        "selected_sha256": _sha256(selected_path),
        "sft_sha256": _sha256(sft_path),
        "final_targets": 3,
        "tool_targets": 3,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "selected_path": selected_path,
        "sft_path": sft_path,
        "summary_path": summary_path,
        "frame": frame,
        "selected_rows": selected_rows,
        "sft_rows": sft_rows,
        "summary": summary,
    }


def _validate(fixture: dict[str, object]) -> dict:
    return validate_fast_hybrid_sft_data(
        train_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
        selected_path=fixture["selected_path"],  # type: ignore[arg-type]
        sft_path=fixture["sft_path"],  # type: ignore[arg-type]
        summary_path=fixture["summary_path"],  # type: ignore[arg-type]
        config_sha256=CONFIG_SHA,
        expected_counts={dataset: 1 for dataset in DATASETS},
        minimum_total=3,
        minimum_per_dataset=1,
    )


def test_fast_hybrid_checker_accepts_episode_export_and_correct_candidate_fallback(
    tmp_path: Path,
) -> None:
    report = _validate(_fixture(tmp_path))

    assert report["status"] == "passed"
    assert report["selected_by_dataset"] == {dataset: 1 for dataset in DATASETS}
    assert report["episode_target_counts"] == {"final": 3, "tool": 3}
    assert report["frame_contract_checked"] is True


def test_fast_hybrid_checker_rejects_non_stable_selection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected_rows = fixture["selected_rows"]
    assert isinstance(selected_rows, list)
    selected_rows[0]["_selection_confirmation_count"] = 2
    _write_jsonl(fixture["selected_path"], selected_rows)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="3/3 confirmation"):
        _validate(fixture)


def test_fast_hybrid_checker_rejects_missing_frame(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = fixture["frame"]
    assert isinstance(frame, Path)
    frame.unlink()

    with pytest.raises(ValueError, match="missing Fast Hybrid frame file"):
        _validate(fixture)


def test_fast_hybrid_checker_rejects_missing_final_episode(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    sft_rows = fixture["sft_rows"]
    assert isinstance(sft_rows, list)
    del sft_rows[1]
    _write_jsonl(fixture["sft_path"], sft_rows)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exactly one final episode"):
        _validate(fixture)


def test_fast_hybrid_checker_rejects_summary_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = fixture["summary"]
    assert isinstance(summary, dict)
    summary["sft_sha256"] = "0" * 64
    summary_path = fixture["summary_path"]
    assert isinstance(summary_path, Path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary mismatch: sft_sha256"):
        _validate(fixture)

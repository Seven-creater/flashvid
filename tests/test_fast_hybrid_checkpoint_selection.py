from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.fast_hybrid_checkpoint_selection import (
    CheckpointRun,
    select_fast_hybrid_checkpoint,
)


DATASETS = ("lvbench", "lsdbench", "cgbench")


def _write_run(
    root: Path,
    name: str,
    *,
    correct_by_dataset: dict[str, int],
    total_tokens: int,
    visual_tokens: int,
    prompt_hash: str = "p",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for dataset in DATASETS:
        path = root / name / f"{dataset}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(50):
            answer = "A"
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset}-{index}",
                    "answer": answer,
                    "prediction": answer if index < correct_by_dataset[dataset] else "B",
                    "candidate_answer": "A",
                    "candidate_results_sha256": "c" * 64,
                    "agent_version": "fast_hybrid_v2",
                    "official_eva_commit": "o" * 40,
                    "experiment_config_sha256": "e" * 64,
                    "prompt_hashes": {"verification": prompt_hash},
                    "annotation_leak_check": "passed",
                    "candidate_rerun": 0,
                    "candidate_cost_complete": True,
                    "end_to_end_total_tokens": total_tokens,
                    "end_to_end_total_tokens_complete": True,
                    "end_to_end_visual_tokens": visual_tokens,
                    "end_to_end_visual_tokens_complete": True,
                    "end_to_end_latency_s": 1.0,
                    "tool_calls": [{}],
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        paths[dataset] = path
    return paths


def test_selects_accuracy_gain_with_strict_cost_reduction(tmp_path: Path) -> None:
    teacher = _write_run(
        tmp_path,
        "teacher",
        correct_by_dataset={dataset: 20 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
    )
    epoch1 = _write_run(
        tmp_path,
        "epoch1",
        correct_by_dataset={"lvbench": 21, "lsdbench": 20, "cgbench": 20},
        total_tokens=90,
        visual_tokens=70,
    )
    epoch2 = _write_run(
        tmp_path,
        "epoch2",
        correct_by_dataset={"lvbench": 22, "lsdbench": 20, "cgbench": 20},
        total_tokens=95,
        visual_tokens=75,
    )
    report = select_fast_hybrid_checkpoint(
        teacher_paths=teacher,
        checkpoints=[
            CheckpointRun("checkpoint-epoch1", 1, epoch1),
            CheckpointRun("checkpoint-epoch2", 2, epoch2),
        ],
    )
    assert report["status"] == "passed"
    assert report["selected"]["checkpoint_id"] == "checkpoint-epoch2"
    assert report["selected"]["clearly_reduced_30pct"] is False


def test_rejects_dataset_drop_or_non_decreasing_cost(tmp_path: Path) -> None:
    teacher = _write_run(
        tmp_path,
        "teacher",
        correct_by_dataset={dataset: 20 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
    )
    dropped = _write_run(
        tmp_path,
        "dropped",
        correct_by_dataset={"lvbench": 19, "lsdbench": 23, "cgbench": 20},
        total_tokens=90,
        visual_tokens=70,
    )
    flat_cost = _write_run(
        tmp_path,
        "flat_cost",
        correct_by_dataset={"lvbench": 21, "lsdbench": 20, "cgbench": 20},
        total_tokens=100,
        visual_tokens=70,
    )
    report = select_fast_hybrid_checkpoint(
        teacher_paths=teacher,
        checkpoints=[
            CheckpointRun("dropped", 1, dropped),
            CheckpointRun("flat", 2, flat_cost),
        ],
    )
    assert report["status"] == "blocked"
    assert report["selected"] is None


def test_rejects_changed_prompt_or_candidate(tmp_path: Path) -> None:
    teacher = _write_run(
        tmp_path,
        "teacher",
        correct_by_dataset={dataset: 20 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
    )
    changed = _write_run(
        tmp_path,
        "changed",
        correct_by_dataset={dataset: 21 for dataset in DATASETS},
        total_tokens=90,
        visual_tokens=70,
        prompt_hash="changed",
    )
    with pytest.raises(ValueError, match="prompt hash changed"):
        select_fast_hybrid_checkpoint(
            teacher_paths=teacher,
            checkpoints=[CheckpointRun("changed", 1, changed)],
        )


def test_rejects_incomplete_visual_cost_instead_of_zero_filling(tmp_path: Path) -> None:
    teacher = _write_run(
        tmp_path,
        "teacher",
        correct_by_dataset={dataset: 20 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
    )
    checkpoint = _write_run(
        tmp_path,
        "checkpoint",
        correct_by_dataset={dataset: 21 for dataset in DATASETS},
        total_tokens=90,
        visual_tokens=70,
    )
    first_path = checkpoint["lvbench"]
    rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["end_to_end_visual_tokens"] = None
    rows[0]["end_to_end_visual_tokens_complete"] = False
    first_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="end_to_end_visual_tokens"):
        select_fast_hybrid_checkpoint(
            teacher_paths=teacher,
            checkpoints=[CheckpointRun("checkpoint", 1, checkpoint)],
        )


def test_teacher_infrastructure_failure_is_excluded_only_from_cost_pairing(
    tmp_path: Path,
) -> None:
    teacher = _write_run(
        tmp_path,
        "teacher",
        correct_by_dataset={dataset: 20 for dataset in DATASETS},
        total_tokens=100,
        visual_tokens=80,
    )
    checkpoint = _write_run(
        tmp_path,
        "checkpoint",
        correct_by_dataset={"lvbench": 21, "lsdbench": 20, "cgbench": 20},
        total_tokens=90,
        visual_tokens=70,
    )
    first_path = teacher["lvbench"]
    rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    rows[49].update(
        {
            "error": "TimeoutError: timed out",
            "annotation_leak_check": "not_run",
            "end_to_end_total_tokens_complete": False,
            "end_to_end_visual_tokens": None,
            "end_to_end_visual_tokens_complete": False,
        }
    )
    for key in ("agent_version", "official_eva_commit", "prompt_hashes"):
        rows[49].pop(key)
    first_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = select_fast_hybrid_checkpoint(
        teacher_paths=teacher,
        checkpoints=[CheckpointRun("checkpoint", 1, checkpoint)],
    )

    assert report["status"] == "passed"
    assert report["teacher"]["summary"]["engineering_failures"] == 1
    assert report["teacher"]["summary"]["cost_samples"] == 149

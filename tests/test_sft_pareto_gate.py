from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.check_sft_pareto_gate import evaluate_sft_pareto_gate, main
from scripts.select_sft_checkpoint import _materialize_selected


DATASETS = ("lvbench", "lsdbench", "cgbench")


def _row(
    dataset: str,
    sample_id: str,
    *,
    correct: bool,
    retained: float,
    ratio: float,
) -> dict[str, object]:
    prediction = "A" if correct else "B"
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "answer": "A",
        "prediction": prediction,
        "final_prediction": prediction,
        "candidate_answer": "B",
        "candidate_source": "parsed",
        "candidate_rerun": 0,
        "annotation_leak_check": "passed",
        "retained_visual_tokens": retained,
        "raw_visual_tokens": 100,
        "tool_steps": [
            {
                "retention_ratio": ratio,
                "retained_visual_tokens": retained,
                "raw_visual_tokens": 100,
            }
        ],
    }


def _write_method(
    root: Path,
    *,
    correct_by_dataset: dict[str, int],
    retained: float,
    ratios: tuple[float, ...] = (0.1, 0.5),
    count: int = 3,
) -> None:
    for dataset in DATASETS:
        output = root / dataset / f"{dataset}_flashvid_hybrid.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            _row(
                dataset,
                f"{dataset}-{index}",
                correct=index < correct_by_dataset[dataset],
                retained=retained,
                ratio=ratios[index % len(ratios)],
            )
            for index in range(count)
        ]
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def _fixture(
    tmp_path: Path,
    *,
    sft_correct: dict[str, int] | None = None,
    sft_retained: float = 60,
    sft_ratios: tuple[float, ...] = (0.1, 0.5),
    fixed_correct: dict[str, int] | None = None,
    fixed_retained: float = 100,
) -> tuple[Path, Path, Path, Path]:
    selected_root = tmp_path / "selected"
    selected_source = tmp_path / "checkpoint-3"
    untrained_root = tmp_path / "untrained"
    fixed_root = tmp_path / "fixed"
    selection_path = tmp_path / "checkpoint_selection.json"
    _write_method(
        selected_source,
        correct_by_dataset=sft_correct or {dataset: 2 for dataset in DATASETS},
        retained=sft_retained,
        ratios=sft_ratios,
    )
    selection_path.write_text(
        json.dumps(
            {
                "selection": {
                    "status": "complete",
                    "selected_checkpoint": "checkpoint-3",
                },
                "source_files": {
                    "checkpoint-3": [
                        str(
                            (
                                selected_source
                                / dataset
                                / f"{dataset}_flashvid_hybrid.jsonl"
                            ).resolve()
                        )
                        for dataset in DATASETS
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    _materialize_selected(selected_source, selected_root, "checkpoint-3")
    _write_method(
        untrained_root,
        correct_by_dataset={"lvbench": 1, "lsdbench": 1, "cgbench": 1},
        retained=60,
    )
    _write_method(
        fixed_root,
        correct_by_dataset=fixed_correct or {dataset: 3 for dataset in DATASETS},
        retained=fixed_retained,
        ratios=(1.0,),
    )
    return selection_path, selected_root, untrained_root, fixed_root


def _evaluate(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    selection, selected, untrained, fixed = paths
    return evaluate_sft_pareto_gate(
        selection_path=selection,
        selected_root=selected,
        untrained_root=untrained,
        fixed100_root=fixed,
        expected_counts={dataset: 3 for dataset in DATASETS},
    )


def test_sft_pareto_gate_accepts_approved_accuracy_path(tmp_path: Path) -> None:
    report = _evaluate(_fixture(tmp_path))
    assert report["status"] == "passed"
    assert report["acceptance"]["passed"] is True
    assert report["acceptance"]["conditions"] == {
        "improves_untrained_accuracy_without_more_cost": True,
        "or_preserves_untrained_accuracy_with_20pct_cost_reduction": False,
        "within_3_correct_of_fixed_100": True,
        "at_least_30pct_cost_reduction_vs_fixed_100": True,
        "no_dataset_drops_more_than_2": True,
        "dynamic_budget_not_single_constant": True,
    }


def test_sft_pareto_gate_rejects_insufficient_gain_and_cost_reduction(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        sft_correct={"lvbench": 2, "lsdbench": 2, "cgbench": 1},
    )
    report = _evaluate(paths)
    assert report["status"] == "failed"
    conditions = report["acceptance"]["conditions"]
    assert not conditions["improves_untrained_accuracy_without_more_cost"]
    assert not conditions["or_preserves_untrained_accuracy_with_20pct_cost_reduction"]


def test_sft_pareto_gate_rejects_single_budget_policy(tmp_path: Path) -> None:
    report = _evaluate(_fixture(tmp_path, sft_ratios=(0.5,)))
    assert report["status"] == "failed"
    assert not report["acceptance"]["conditions"][
        "dynamic_budget_not_single_constant"
    ]


def test_sft_pareto_gate_rejects_less_than_30pct_savings_vs_fixed100(
    tmp_path: Path,
) -> None:
    report = _evaluate(_fixture(tmp_path, fixed_retained=80))
    assert report["status"] == "failed"
    assert not report["acceptance"]["conditions"][
        "at_least_30pct_cost_reduction_vs_fixed_100"
    ]


def test_sft_pareto_gate_rejects_more_than_three_correct_drop_vs_fixed100(
    tmp_path: Path,
) -> None:
    report = _evaluate(
        _fixture(
            tmp_path,
            sft_correct={dataset: 1 for dataset in DATASETS},
            sft_retained=40,
        )
    )
    assert report["status"] == "failed"
    assert not report["acceptance"]["conditions"][
        "within_3_correct_of_fixed_100"
    ]


def test_sft_pareto_gate_rejects_more_than_two_correct_drop_on_one_dataset(
    tmp_path: Path,
) -> None:
    report = _evaluate(
        _fixture(
            tmp_path,
            sft_correct={"lvbench": 0, "lsdbench": 3, "cgbench": 3},
        )
    )
    assert report["status"] == "failed"
    assert not report["acceptance"]["conditions"][
        "no_dataset_drops_more_than_2"
    ]


def test_sft_pareto_gate_requires_identical_validation_samples(
    tmp_path: Path,
) -> None:
    selection, selected, untrained, fixed = _fixture(tmp_path)
    path = untrained / "lvbench" / "lvbench_flashvid_hybrid.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_id"] = "different"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different sample IDs"):
        _evaluate((selection, selected, untrained, fixed))


def test_sft_pareto_gate_rejects_tampered_materialized_tree(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    selected = paths[1]
    result = selected / "lvbench" / "lvbench_flashvid_hybrid.jsonl"
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tree hashes"):
        _evaluate(paths)


def test_sft_pareto_gate_rejects_changed_source_tree(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metadata = json.loads(
        (paths[1] / "selected_checkpoint.json").read_text(encoding="utf-8")
    )
    source = Path(metadata["source"])
    result = source / "lvbench" / "lvbench_flashvid_hybrid.jsonl"
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source tree hashes"):
        _evaluate(paths)


def test_sft_pareto_gate_binds_selection_source_files(tmp_path: Path) -> None:
    selection, selected, untrained, fixed = _fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["source_files"]["checkpoint-3"] = [
        str(
            (
                untrained / "lvbench" / "lvbench_flashvid_hybrid.jsonl"
            ).resolve()
        )
    ]
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="outside materialized source"):
        _evaluate((selection, selected, untrained, fixed))


def test_sft_pareto_cli_writes_failed_report_for_type_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, selected, untrained, fixed = _fixture(tmp_path)
    metadata_path = selected / "selected_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"] = []
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sft_pareto_gate.py",
            "--checkpoint-selection",
            str(selection),
            "--selected-root",
            str(selected),
            "--untrained-root",
            str(untrained),
            "--fixed100-root",
            str(fixed),
            "--expected",
            "lvbench=3",
            "--expected",
            "lsdbench=3",
            "--expected",
            "cgbench=3",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["passed"] is False
    assert "hash mapping" in report["error"]

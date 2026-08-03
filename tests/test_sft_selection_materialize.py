from __future__ import annotations

import json
from pathlib import Path

from scripts.select_sft_checkpoint import _materialize_selected


def test_selected_validation_results_are_materialized_idempotently(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoint-1"
    result_dir = source / "lvbench"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "lvbench_flashvid_hybrid.jsonl"
    result_file.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    target = tmp_path / "sft4p4_model"

    _materialize_selected(source, target, "checkpoint-1")
    assert (target / "lvbench" / result_file.name).read_text(
        encoding="utf-8"
    ) == result_file.read_text(encoding="utf-8")
    metadata = json.loads(
        (target / "selected_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["selected_checkpoint"] == "checkpoint-1"

    _materialize_selected(source, target, "checkpoint-1")


def test_selected_validation_results_are_atomically_rematerialized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoint-1"
    result_dir = source / "lvbench"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "lvbench_flashvid_hybrid.jsonl"
    result_file.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    target = tmp_path / "sft4p4_model"

    _materialize_selected(source, target, "checkpoint-1")
    result_file.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
    _materialize_selected(source, target, "checkpoint-1")

    assert (target / "lvbench" / result_file.name).read_text(
        encoding="utf-8"
    ) == result_file.read_text(encoding="utf-8")
    metadata = json.loads(
        (target / "selected_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["files"]["lvbench/lvbench_flashvid_hybrid.jsonl"]
    assert not list(tmp_path.glob(".sft4p4_model.previous.*"))

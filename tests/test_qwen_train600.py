from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.freeze_qwen_train600 import freeze_train600, sha256_file


DATASETS = ("lvbench", "lsdbench", "cgbench")
COUNTS = {"train": 2, "dev": 1, "final": 1}


def _row(dataset: str, sample_id: str, video: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "video": video,
        "question": "What happens?",
        "choices": {"A": "opens", "B": "closes"},
        "answer": "A",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, object] = {}
    for dataset in DATASETS:
        split_specs: dict[str, object] = {}
        for split, count in COUNTS.items():
            path = tmp_path / f"{dataset}_{split}.jsonl"
            rows = [
                _row(
                    dataset,
                    f"{split}-shared-id-{index}",
                    f"{dataset}/{split}-{index}.mp4",
                )
                for index in range(count)
            ]
            _write_jsonl(path, rows)
            split_specs[split] = {"path": str(path), "sha256": sha256_file(path)}
        datasets[dataset] = split_specs
    config = {"schema_version": 1, "datasets": datasets}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_freeze_train600_is_deterministic_read_only_and_video_isolated(
    tmp_path: Path,
) -> None:
    config_path, config = _config(tmp_path)
    source_hashes = {
        spec["path"]: sha256_file(Path(spec["path"]))
        for dataset in config["datasets"].values()
        for spec in dataset.values()
    }
    output = tmp_path / "frozen/train600.jsonl"
    metadata = tmp_path / "frozen/train600.frozen.json"

    first = freeze_train600(
        config_path=config_path,
        output_path=output,
        metadata_path=metadata,
        expected_split_counts=COUNTS,
    )
    second = freeze_train600(
        config_path=config_path,
        output_path=output,
        metadata_path=metadata,
        expected_split_counts=COUNTS,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6
    assert [row["dataset"] for row in rows] == [
        "lvbench",
        "lvbench",
        "lsdbench",
        "lsdbench",
        "cgbench",
        "cgbench",
    ]
    assert len({(row["dataset"], row["sample_id"]) for row in rows}) == 6
    assert first["video_isolation"] == "passed"
    assert first["output"]["sha256"] == sha256_file(output)
    assert second["output_status"] == "existing_identical"
    assert second["metadata_status"] == "existing_identical"
    assert source_hashes == {
        path: sha256_file(Path(path)) for path in source_hashes
    }


def test_freeze_train600_rejects_split_overlap_and_cross_dataset_absolute_video(
    tmp_path: Path,
) -> None:
    config_path, config = _config(tmp_path)
    lvbench = config["datasets"]["lvbench"]
    train_path = Path(lvbench["train"]["path"])
    dev_path = Path(lvbench["dev"]["path"])
    dev_rows = [json.loads(line) for line in dev_path.read_text().splitlines()]
    dev_rows[0]["video"] = json.loads(train_path.read_text().splitlines()[0])["video"]
    _write_jsonl(dev_path, dev_rows)
    lvbench["dev"]["sha256"] = sha256_file(dev_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="video overlap"):
        freeze_train600(
            config_path=config_path,
            output_path=tmp_path / "overlap.jsonl",
            metadata_path=tmp_path / "overlap.json",
            expected_split_counts=COUNTS,
        )

    config_path, config = _config(tmp_path / "absolute")
    shared_video = str((tmp_path / "shared.mp4").resolve())
    changed = deepcopy(config)
    for dataset in ("lvbench", "lsdbench"):
        train = Path(changed["datasets"][dataset]["train"]["path"])
        rows = [json.loads(line) for line in train.read_text().splitlines()]
        rows[0]["video"] = shared_video
        _write_jsonl(train, rows)
        changed["datasets"][dataset]["train"]["sha256"] = sha256_file(train)
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="shared across datasets"):
        freeze_train600(
            config_path=config_path,
            output_path=tmp_path / "absolute.jsonl",
            metadata_path=tmp_path / "absolute.json",
            expected_split_counts=COUNTS,
        )

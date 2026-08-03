from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qwen_search_preflight_freezes_counts_hashes_and_video_splits(tmp_path: Path) -> None:
    split_specs = {"train": 200, "dev": 50, "final": 100}
    dataset: dict[str, dict[str, str]] = {}
    for split, count in split_specs.items():
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "dataset": "demo",
                        "sample_id": f"{split}-{index}",
                        "video": f"{split}-video-{index}.mp4",
                    }
                )
                + "\n"
                for index in range(count)
            ),
            encoding="utf-8",
        )
        dataset[split] = {"path": str(path), "sha256": _sha256(path)}
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"datasets": {"demo": dataset}}),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight_qwen_agent_search.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--source-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"] is True

    dataset["dev"]["sha256"] = "0" * 64
    config.write_text(json.dumps({"datasets": {"demo": dataset}}), encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "manifest hash mismatch" in failed.stderr

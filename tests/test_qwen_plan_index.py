from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flashvid_eval.qwen_plan_index import (
    build_selection_plan_index,
    canonical_sha256,
    load_selection_plan_index,
    write_frozen_index,
)


ROOT = Path(__file__).parents[1]
SELECTOR = ROOT / "scripts" / "select_qwen_agent_dev_winner.py"


def _plan(
    path: Path,
    *,
    config_sha: str,
    phase: str,
    task_id: str,
    dataset: str = "lvbench",
) -> Path:
    payload = {
        "schema_version": 1,
        "phase": phase,
        "config_sha256": config_sha,
        "task_count": 1,
        "tasks": [{"task_id": task_id, "dataset": dataset}],
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _canonical_plans(tmp_path: Path, config_sha: str) -> list[Path]:
    return [
        _plan(
            tmp_path / "protocol_audit.json",
            config_sha=config_sha,
            phase="protocol_audit",
            task_id="q9_no_think",
        ),
        _plan(
            tmp_path / "direct_dev.json",
            config_sha=config_sha,
            phase="direct_dev",
            task_id="q9_uniform32",
        ),
        _plan(
            tmp_path / "agent_dev.json",
            config_sha=config_sha,
            phase="agent_dev",
            task_id="q9_a0",
        ),
    ]


def test_index_uses_only_explicit_plans_even_when_recovery_sibling_exists(
    tmp_path: Path,
) -> None:
    config = {"experiment_id": "test"}
    config_sha = canonical_sha256(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plans = _canonical_plans(tmp_path, config_sha)
    _plan(
        tmp_path / "protocol_audit_recovery.json",
        config_sha=config_sha,
        phase="protocol_audit",
        task_id="q9_no_think",
    )
    rejection = _plan(
        tmp_path / "q4_smoke.json",
        config_sha=config_sha,
        phase="protocol_smoke",
        task_id="q4_think",
    )

    payload = build_selection_plan_index(
        config_path=config_path,
        config_sha256=config_sha,
        run_plan_paths=plans,
        q4_think_smoke_rejection=rejection,
    )
    index_path = tmp_path / "index.json"
    write_frozen_index(index_path, payload)
    loaded = load_selection_plan_index(index_path, config_sha)

    assert len(loaded["run_plans"]) == 3
    assert all("recovery" not in item["path"] for item in loaded["run_plans"])
    assert loaded["q4_think_smoke_rejection"]["path"] == str(rejection.resolve())


def test_index_rejects_duplicate_logical_tasks_across_original_and_recovery(
    tmp_path: Path,
) -> None:
    config_sha = canonical_sha256({"experiment_id": "test"})
    plans = _canonical_plans(tmp_path, config_sha)
    recovery = _plan(
        tmp_path / "protocol_audit_recovery.json",
        config_sha=config_sha,
        phase="protocol_audit",
        task_id="q9_no_think",
    )
    with pytest.raises(ValueError, match="duplicate logical Dev task"):
        build_selection_plan_index(
            config_path=tmp_path / "config.json",
            config_sha256=config_sha,
            run_plan_paths=[*plans, recovery],
        )


def test_index_rejects_tampered_referenced_plan(tmp_path: Path) -> None:
    config = {"experiment_id": "test"}
    config_sha = canonical_sha256(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plans = _canonical_plans(tmp_path, config_sha)
    payload = build_selection_plan_index(
        config_path=config_path,
        config_sha256=config_sha,
        run_plan_paths=plans,
    )
    index_path = tmp_path / "index.json"
    write_frozen_index(index_path, payload)
    plans[0].write_text(plans[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference changed"):
        load_selection_plan_index(index_path, config_sha)


def test_selector_rejects_mixing_index_and_legacy_plan_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--config",
            str(config_path),
            "--selection-plan-index",
            str(tmp_path / "index.json"),
            "--run-plan",
            str(tmp_path / "legacy.json"),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--winner-output",
            str(tmp_path / "winner.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert "mutually exclusive" in process.stderr

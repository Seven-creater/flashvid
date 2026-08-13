from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qwen_sft_smoke_gate import bind_smoke_report, validate_smoke_report


MODEL_SHA = "a" * 64


def _passed_report(root: Path) -> Path:
    path = root / "preflight" / "training_update.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 1, "status": "passed", "global_step": 1}),
        encoding="utf-8",
    )
    return path


def test_passed_smoke_is_bound_to_formal_output_training_bytes_and_model(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text('{"messages": []}\n', encoding="utf-8")
    report = _passed_report(smoke)

    bound = bind_smoke_report(
        report,
        smoke_output_dir=smoke,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
    )
    gate = validate_smoke_report(
        report,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
    )

    assert bound["formal_training_gate"] == gate
    assert gate["formal_output_dir"] == str(formal.resolve())
    assert gate["train_data"]["path"] == str(train_data.resolve())


def test_formal_gate_rejects_changed_training_data(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text("first\n", encoding="utf-8")
    report = _passed_report(smoke)
    bind_smoke_report(
        report,
        smoke_output_dir=smoke,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
    )
    train_data.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after the smoke"):
        validate_smoke_report(
            report,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
        )


def test_formal_gate_rejects_different_output_or_model(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text("same\n", encoding="utf-8")
    report = _passed_report(smoke)
    bind_smoke_report(
        report,
        smoke_output_dir=smoke,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
    )

    with pytest.raises(ValueError, match="different formal output"):
        validate_smoke_report(
            report,
            formal_output_dir=tmp_path / "other-formal",
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
        )
    with pytest.raises(ValueError, match="different base-model"):
        validate_smoke_report(
            report,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256="b" * 64,
        )


def test_failed_or_unbound_smoke_report_cannot_start_formal(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text("same\n", encoding="utf-8")
    failed = smoke / "preflight" / "training_update.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(json.dumps({"status": "failed", "global_step": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="passed one-step smoke"):
        bind_smoke_report(
            failed,
            smoke_output_dir=smoke,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
        )

    unbound = _passed_report(tmp_path / "other-smoke")
    with pytest.raises(ValueError, match="no formal-training binding"):
        validate_smoke_report(
            unbound,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
        )


def test_image_bearing_probe_is_bound_and_cannot_change(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "planner.jsonl"
    probe_data = tmp_path / "observer_probe.jsonl"
    train_data.write_text("planner\n", encoding="utf-8")
    probe_data.write_text("image probe\n", encoding="utf-8")
    report = _passed_report(smoke)

    bound = bind_smoke_report(
        report,
        smoke_output_dir=smoke,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
        smoke_probe_data=probe_data,
    )
    assert bound["formal_training_gate"]["smoke_probe_data"]["path"] == str(
        probe_data.resolve()
    )
    validate_smoke_report(
        report,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
        smoke_probe_data=probe_data,
    )

    probe_data.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="smoke-probe data changed"):
        validate_smoke_report(
            report,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
            smoke_probe_data=probe_data,
        )


def test_bound_image_probe_must_be_supplied_to_formal_gate(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    formal = tmp_path / "formal"
    train_data = tmp_path / "planner.jsonl"
    probe_data = tmp_path / "observer_probe.jsonl"
    train_data.write_text("planner\n", encoding="utf-8")
    probe_data.write_text("image probe\n", encoding="utf-8")
    report = _passed_report(smoke)
    bind_smoke_report(
        report,
        smoke_output_dir=smoke,
        formal_output_dir=formal,
        train_data=train_data,
        base_model_artifact_sha256=MODEL_SHA,
        smoke_probe_data=probe_data,
    )

    with pytest.raises(ValueError, match="requires the bound smoke-probe"):
        validate_smoke_report(
            report,
            formal_output_dir=formal,
            train_data=train_data,
            base_model_artifact_sha256=MODEL_SHA,
        )

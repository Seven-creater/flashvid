from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_script("fast_hybrid_bulk_common")
teacher = _load_script("launch_fast_hybrid_teacher_matrix")
repair = _load_script("repair_fast_hybrid_teacher_provenance")
file_sha256 = common.file_sha256
TeacherJob = teacher.TeacherJob
audit_outputs = teacher.audit_outputs
repair_teacher_provenance = repair.repair_teacher_provenance


SHA_CONFIG = "a" * 64
SHA_MODEL = "b" * 64
SHA_MANIFEST = "c" * 64
SHA_TRAIN = "d" * 64
SHA_CANDIDATE = "e" * 64
SHA_RUN = "f" * 64


def _fixture(tmp_path: Path, *, error_type: str = "TimeoutError") -> tuple[Path, Path, Path, Path]:
    output_dir = tmp_path / "raw" / "base" / "budget_006000_seed_17" / "lsdbench"
    output_dir.mkdir(parents=True)
    output = output_dir / "lsdbench_fast_hybrid_eva.jsonl"
    frozen = {
        "dataset": "lsdbench",
        "manifest": {"path": "manifest.jsonl", "sha256": SHA_MANIFEST},
        "candidate_results": {
            "path": "candidate.jsonl",
            "sha256": SHA_CANDIDATE,
            "direct_rerun": 0,
        },
        "agent_version": "fast_hybrid_v2",
        "scoring_deferred": True,
        "teacher_temperature": 0.2,
        "generation_seed": 17,
        "experiment_config_sha256": SHA_CONFIG,
        "model_artifact_sha256": SHA_MODEL,
        "train600_manifest_sha256": SHA_TRAIN,
        "trajectory_schedule_id": "budget_006000_seed_17",
        "trajectory_variant_id": "base",
        "trajectory_replica_id": 0,
        "run_fingerprint": SHA_RUN,
    }
    (output_dir / "frozen_inputs_lsdbench.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    common = {
        "candidate_results_sha256": SHA_CANDIDATE,
        "teacher_model_sha256": SHA_MODEL,
        "model_artifact_sha256": SHA_MODEL,
        "experiment_config_sha256": SHA_CONFIG,
        "manifest_sha256": SHA_MANIFEST,
        "train600_manifest_sha256": SHA_TRAIN,
        "trajectory_schedule_id": "budget_006000_seed_17",
        "trajectory_variant_id": "base",
        "trajectory_replica_id": 0,
        "teacher_temperature": 0.2,
        "generation_seed": 17,
        "scoring_deferred": True,
        "candidate_rerun": 0,
        "run_fingerprint": SHA_RUN,
    }
    good = {"sample_id": "good", **common, "annotation_leak_check": "passed"}
    failed_rows = [
        {
            "sample_id": sample_id,
            "error": f"{error_type}: timed out",
            "error_type": error_type,
            "prediction": prediction,
            "scoring_deferred": True,
            "candidate_rerun": 0,
            "run_fingerprint": SHA_RUN,
            "annotation_leak_check": "not_run",
        }
        for sample_id, prediction in (("timeout-1", "A"), ("timeout-2", "B"))
    ]
    output.write_text(
        "".join(
            json.dumps(row) + "\n" for row in (good, *failed_rows)
        ),
        encoding="utf-8",
    )
    command = (
        "python",
        "evaluate.py",
        "--expected-manifest-sha256",
        SHA_MANIFEST,
        "--seed",
        "17",
        "--trajectory-schedule-id",
        "budget_006000_seed_17",
        "--trajectory-variant-id",
        "base",
        "--trajectory-replica-id",
        "0",
        "--experiment-config-sha256",
        SHA_CONFIG,
        "--model-artifact-sha256",
        SHA_MODEL,
        "--train600-manifest-sha256",
        SHA_TRAIN,
    )
    job = TeacherJob(
        job_id="base:budget_006000_seed_17:lsdbench",
        endpoint="http://127.0.0.1:8200/v1",
        command=command,
        log_path=tmp_path / "teacher.log",
        output_path=output,
        expected_ids=("good", "timeout-1", "timeout-2"),
        dataset="lsdbench",
        phase="base",
        schedule_id="budget_006000_seed_17",
        planner_seed=17,
        active_manifest_sha256=SHA_MANIFEST,
    )
    plan = {
        "schema_version": 1,
        "kind": "fast_hybrid_teacher_matrix",
        "config_sha256": SHA_CONFIG,
        "candidate_sha256s": {
            "lvbench": "1" * 64,
            "lsdbench": SHA_CANDIDATE,
            "cgbench": "2" * 64,
        },
        "jobs": [
            {
                "job_id": job.job_id,
                "endpoint": job.endpoint,
                "command": list(job.command),
                "log_path": str(job.log_path),
                "output_path": str(job.output_path),
                "expected_ids": list(job.expected_ids),
                "dataset": job.dataset,
                "phase": job.phase,
                "schedule_id": job.schedule_id,
                "planner_seed": job.planner_seed,
                "active_manifest_sha256": job.active_manifest_sha256,
            }
        ],
    }
    plan_path = tmp_path / "teacher.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    audit = audit_outputs([job])
    assert audit["status"] == "failed"
    failed_audit = tmp_path / "teacher_audit.json"
    failed_audit.write_text(json.dumps(audit), encoding="utf-8")
    return plan_path, failed_audit, tmp_path / "teacher_repaired_audit.json", output


def test_repairs_only_missing_timeout_provenance_and_preserves_failed_audit(
    tmp_path: Path,
) -> None:
    plan, failed_audit, repaired_audit, output = _fixture(tmp_path)
    failed_audit_sha = file_sha256(failed_audit)
    before_rows = [json.loads(line) for line in output.read_text().splitlines()]

    report = repair_teacher_provenance(
        plan_path=plan,
        failed_audit_path=failed_audit,
        output_audit_path=repaired_audit,
    )

    assert report["status"] == "passed"
    assert report["repaired_rows"] == 2
    assert file_sha256(failed_audit) == failed_audit_sha
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    timeout = rows[1]
    assert timeout["error"] == "TimeoutError: timed out"
    assert timeout["prediction"] == "A"
    assert timeout["annotation_leak_check"] == "not_run"
    assert timeout["trajectory_schedule_id"] == "budget_006000_seed_17"
    assert timeout["generation_seed"] == 17
    assert timeout["manifest_sha256"] == SHA_MANIFEST
    assert timeout["model_artifact_sha256"] == SHA_MODEL
    assert rows[2]["trajectory_schedule_id"] == "budget_006000_seed_17"

    allowed = set(report["repairs"][0]["added_fields"])
    for before, after in zip(before_rows, rows):
        if before["sample_id"] == "good":
            assert before == after
            continue
        assert {key: value for key, value in before.items() if key not in allowed} == {
            key: value for key, value in after.items() if key not in allowed
        }

    repeated = repair_teacher_provenance(
        plan_path=plan,
        failed_audit_path=failed_audit,
        output_audit_path=repaired_audit,
    )
    assert repeated == report


def test_refuses_to_relabel_non_timeout_failure(tmp_path: Path) -> None:
    plan, failed_audit, repaired_audit, output = _fixture(
        tmp_path, error_type="ValueError"
    )
    before_sha = file_sha256(output)
    with pytest.raises(RuntimeError, match="refusing non-timeout"):
        repair_teacher_provenance(
            plan_path=plan,
            failed_audit_path=failed_audit,
            output_audit_path=repaired_audit,
        )
    assert file_sha256(output) == before_sha


def test_rejects_forged_repaired_audit(tmp_path: Path) -> None:
    plan, failed_audit, repaired_audit, _output = _fixture(tmp_path)
    repaired_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fast_hybrid_teacher_repaired_audit",
                "status": "passed",
                "jobs": 1,
                "rows": 3,
                "issues": [],
                "source_plan_sha256": file_sha256(plan),
                "source_failed_audit_sha256": file_sha256(failed_audit),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match immutable inputs"):
        repair_teacher_provenance(
            plan_path=plan,
            failed_audit_path=failed_audit,
            output_audit_path=repaired_audit,
        )

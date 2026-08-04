from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_autopilot_is_one_shot_resumable_and_bounded() -> None:
    text = _text("scripts/run_fast_hybrid_sft_autopilot.sh")
    assert "teacher_audit_passed" in text
    assert "sleep 60" in text
    assert "resume_base_teacher_once" in text
    assert "repair_teacher_audit_once" in text
    assert "repair_fast_hybrid_teacher_provenance.py" in text
    assert "_repaired_audit.json" in text
    assert "TEACHER_CODE_PROJECT_DIR" in text
    assert '"$TEACHER_CODE_PROJECT_DIR/scripts/launch_fast_hybrid_teacher_matrix.py"' in text
    assert '--repo-root "$TEACHER_CODE_PROJECT_DIR"' in text
    assert "--resume" in text
    assert "run_fast_hybrid_post_teacher.sh" in text
    assert "run_fast_hybrid_train_eval.sh" in text
    assert "while true" not in text
    assert "rm -rf" not in text
    assert "pkill" not in text
    assert "kill -9" not in text


def test_train_eval_enforces_smoke_dev_winner_and_single_test_gate() -> None:
    text = _text("scripts/run_fast_hybrid_train_eval.sh")
    assert "recover_lvbench_sft_inputs.sh" in text
    assert "freeze_fast_hybrid_eval_protocol.py" in text
    assert "--phase dev --mode teacher" in text
    assert "--smoke --formal-output-dir" in text
    assert "--load-weights-preflight" in text
    assert "verify_sft_checkpoints.py" in text
    assert "freeze_fast_hybrid_sft_checkpoint.py" in text
    assert "select_fast_hybrid_sft_winner.py" in text
    assert "selection_code == 2" in text
    assert "--phase test --mode checkpoint --winner" in text
    assert text.count("--phase test --mode checkpoint --winner") == 1
    assert "summarize_fast_hybrid_sft.py" in text
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in text
    assert 'reference["sha256"]' in text


def test_post_teacher_recovers_only_allowlisted_owned_base_services() -> None:
    text = _text("scripts/run_fast_hybrid_post_teacher.sh")
    assert "SERVICE_OWNER_PROJECT_DIR" in text
    assert "ensure_base_service 8200 0,1,2,3 1" in text
    assert "ensure_base_service 8201 4,5,6,7 0" in text
    assert 'stop_qwen_agent.sh" "$port"' in text
    assert "launch_qwen_agent_service.sh" in text
    assert "ALLOW_SHARED_GPUS" in text
    assert "pkill" not in text
    assert "kill -9" not in text


def test_train_eval_only_stops_owned_project_services() -> None:
    text = _text("scripts/run_fast_hybrid_train_eval.sh")
    assert "SERVICE_OWNER_PROJECT_DIR" in text
    assert 'stop_qwen_agent.sh" 8200' in text
    assert 'stop_qwen_agent.sh" 8201' in text
    assert "ALLOW_SHARED_GPUS=1" in text
    assert "CUDA_DEVICES=0,1,2,3" in text
    assert "CUDA_DEVICES=4,5,6,7" in text
    assert "pkill" not in text
    assert "kill -9" not in text
    assert "rm -rf" not in text


def test_lvbench_recovery_accepts_read_only_candidate_from_another_worktree() -> None:
    text = _text("scripts/recover_lvbench_sft_inputs.sh")
    assert "LV_ORIGINAL_CANDIDATE" in text
    assert "LV_RECOVERY_BASE_URL" in text
    assert "LV_RECOVERY_TIMEOUT" in text
    assert "expected-original-sha256" in text
    assert "--resume --retry-errors" in text


def test_server_paths_never_use_foreign_sources_or_runtime_installs() -> None:
    for path in (
        "scripts/run_fast_hybrid_post_teacher.sh",
        "scripts/run_fast_hybrid_train_eval.sh",
        "scripts/run_fast_hybrid_sft_autopilot.sh",
        "scripts/recover_lvbench_sft_inputs.sh",
    ):
        text = _text(path)
        assert "huggingface.co" not in text
        assert "snapshot_download" not in text
        assert "pip install" not in text
        assert "git clone" not in text


def test_new_python_clis_are_directly_executable() -> None:
    for path in (
        "scripts/freeze_fast_hybrid_sft_checkpoint.py",
        "scripts/run_fast_hybrid_sft_eval.py",
        "scripts/select_fast_hybrid_sft_winner.py",
        "scripts/summarize_fast_hybrid_sft.py",
        "scripts/extract_hf_mirror_zip_member.py",
        "scripts/repair_direct_candidate_file.py",
        "scripts/repair_fast_hybrid_teacher_provenance.py",
    ):
        subprocess.run(
            [sys.executable, path, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )


def test_sft_eval_retries_a_resume_safe_child_once() -> None:
    text = _text("scripts/run_fast_hybrid_sft_eval.py")
    assert "retry_failed_processes=True" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_fast_hybrid_sft_shells_parse() -> None:
    for path in (
        "scripts/run_fast_hybrid_post_teacher.sh",
        "scripts/run_fast_hybrid_train_eval.sh",
        "scripts/run_fast_hybrid_sft_autopilot.sh",
        "scripts/recover_lvbench_sft_inputs.sh",
    ):
        subprocess.run(["bash", "-n", path], check=True)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import run_perception_memory_sft_autopilot as autopilot


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=tmp_path / "repo",
        repair_run_root=tmp_path / "repair",
        run_root=tmp_path / "run",
        expected_git_branch="codex/perception-memory-eva",
        expected_git_head="a" * 40,
        answers=tmp_path / "train600.jsonl",
        expected_answers_sha256=autopilot.TRAIN600_SHA256,
        base_url=[f"http://127.0.0.1:{8200 + index}/v1" for index in range(8)],
        owned_service=[f"{8200 + index}={1000 + index}" for index in range(8)],
        service_log_root=tmp_path / "logs",
        python="/env/bin/python",
        model="Qwen3.5-9B",
        judge_concurrency=64,
        request_timeout=80.0,
        sft_env_dir=tmp_path / "sft-env",
        model_path=tmp_path / "Qwen3.5-9B",
        expected_model_artifact_sha256="b" * 64,
        hf_endpoint="https://hf-mirror.com",
        poll_seconds=30.0,
        wait_timeout=43200.0,
        resume=False,
    )


def test_commands_preserve_frozen_judges_and_selection_gate(tmp_path: Path) -> None:
    args = _args(tmp_path)
    trajectories = tmp_path / "merged.jsonl"
    judgments = tmp_path / "judgments.jsonl"

    judge = autopilot._judge_command(
        args, trajectories, judgments, resume=True, retry_errors=True
    )
    assert judge.count("--base-url") == 8
    assert [
        judge[index + 1]
        for index, value in enumerate(judge)
        if value == "--judge-seed"
    ] == ["17", "42", "73"]
    assert "--resume" in judge and "--retry-errors" in judge

    build = autopilot._build_command(args)
    assert build[build.index("--minimum-total") + 1] == "360"
    assert build[build.index("--minimum-per-dataset") + 1] == "100"
    assert build[build.index("--minimum-candidate-fixes") + 1] == "90"
    assert build[build.index("--minimum-candidate-fixes-per-dataset") + 1] == "20"


def test_training_smoke_is_bound_and_first_formal_run_is_not_resume(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    smoke = autopilot._training_command(args, smoke=True)
    assert "--smoke" in smoke
    assert "--load-weights-preflight" in smoke
    assert "--formal-output-dir" in smoke

    first = autopilot._training_command(args, smoke=False, resume=False)
    resumed = autopilot._training_command(args, smoke=False, resume=True)
    assert "--smoke-report" in first
    assert "--resume" not in first
    assert "--resume" in resumed

    env = autopilot._training_env(args)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert env["USE_FSDP2"] == "1"
    assert env["SFT_ENV_DIR"] == str(args.sft_env_dir)
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_owned_service_scope_must_be_exactly_ports_8200_to_8207() -> None:
    values = [f"{8200 + index}={100 + index}" for index in range(8)]
    assert autopilot._parse_service_bindings(values)[8207] == 107
    with pytest.raises(ValueError, match="8200-8207"):
        autopilot._parse_service_bindings(values[:-1])
    with pytest.raises(ValueError, match="unique"):
        autopilot._parse_service_bindings(values[:-1] + ["8207=100"])


def _fake_proc_service(
    tmp_path: Path, *, pid: int, port: int, uid: int, argv: list[str]
) -> Path:
    proc_root = tmp_path / "proc"
    root = proc_root / str(pid)
    (root / "fd").mkdir(parents=True)
    (root / "status").write_text(
        f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n", encoding="utf-8"
    )
    (root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
    # fields 3..22 inclusive; starttime is the twentieth token after comm.
    (root / "stat").write_text(
        f"{pid} (python) " + " ".join(["S", *(["0"] * 18), "12345"]),
        encoding="utf-8",
    )
    return proc_root


def test_service_capture_rejects_wrong_owner_command_or_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, port, uid = 1234, 8200, 4321
    log_root = tmp_path / "logs"
    log_root.mkdir()
    expected_log = log_root / f"transformers_{port}.log"
    expected_log.write_text("", encoding="utf-8")
    argv = [
        "/env/bin/transformers",
        "serve",
        "Qwen3.5-9B",
        "--port",
        str(port),
    ]
    proc_root = _fake_proc_service(tmp_path, pid=pid, port=port, uid=uid, argv=argv)
    monkeypatch.setattr(autopilot, "_current_uid", lambda: uid)
    monkeypatch.setattr(autopilot.os, "readlink", lambda _path: str(expected_log))

    snapshot = autopilot._capture_service(
        pid=pid, port=port, log_root=log_root, proc_root=proc_root
    )
    assert snapshot["pid"] == pid and snapshot["port"] == port
    assert snapshot["start_ticks"] == 12345

    monkeypatch.setattr(autopilot, "_current_uid", lambda: uid + 1)
    with pytest.raises(RuntimeError, match="current UID"):
        autopilot._capture_service(
            pid=pid, port=port, log_root=log_root, proc_root=proc_root
        )

    monkeypatch.setattr(autopilot, "_current_uid", lambda: uid)
    monkeypatch.setattr(autopilot.os, "readlink", lambda _path: str(tmp_path / "foreign.log"))
    with pytest.raises(RuntimeError, match="experiment log"):
        autopilot._capture_service(
            pid=pid, port=port, log_root=log_root, proc_root=proc_root
        )


def test_existing_run_requires_resume_and_matching_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.answers.write_text("{}\n", encoding="utf-8")
    args.run_root.mkdir()
    monkeypatch.setattr(autopilot, "_file_sha256", lambda _path: autopilot.TRAIN600_SHA256)

    with pytest.raises(FileExistsError, match="pass --resume"):
        autopilot.run_pipeline(args)

    args.resume = True
    (args.run_root / "status.json").write_text(
        json.dumps({"config_sha256": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        autopilot.run_pipeline(args)


def test_successful_pipeline_is_resumable_without_reissuing_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.repo_root.mkdir()
    args.answers.write_text("{}\n", encoding="utf-8")
    merged = tmp_path / "merged.jsonl"
    merged.write_text('{"trajectory_id":"t1"}\n', encoding="utf-8")

    real_sha = autopilot._file_sha256

    def bound_sha(path: Path) -> str:
        return (
            autopilot.TRAIN600_SHA256
            if Path(path) == args.answers
            else real_sha(Path(path))
        )

    monkeypatch.setattr(autopilot, "_file_sha256", bound_sha)
    monkeypatch.setattr(autopilot, "_verify_git", lambda *_args: None)
    monkeypatch.setattr(
        autopilot,
        "_capture_service",
        lambda *, pid, port, log_root: {
            "pid": pid,
            "port": port,
            "uid": 1,
            "start_ticks": pid,
            "argv_sha256": str(pid),
            "log_path": str(log_root / f"transformers_{port}.log"),
        },
    )
    repair_audit = {
        "repair_status": {"path": "status.json", "sha256": "c" * 64},
        "merged": autopilot._jsonl_record(merged),
        "merge_summary_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        autopilot, "_wait_for_repair", lambda _args: (merged, repair_audit)
    )
    monkeypatch.setattr(
        autopilot,
        "_audit_judgments",
        lambda _trajectories, output: {
            "expected_prefixes": 1,
            "rows": 1,
            "failures": 0,
            "output_sha256": real_sha(output),
        },
    )
    stops: list[bool] = []
    monkeypatch.setattr(
        autopilot,
        "_stop_services",
        lambda *_args, **_kwargs: stops.append(True),
    )
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, log_path, env=None):
        del cwd, log_path, env
        command = list(command)
        commands.append(command)
        joined = " ".join(command)
        if "judge_perception_memory_prefixes.py" in joined:
            (args.run_root / "prefix_judgments.jsonl").write_text(
                '{"prefix_id":"p1","judge_status":"complete"}\n',
                encoding="utf-8",
            )
        elif "select_perception_memory_trajectories.py" in joined:
            directory = args.run_root / "selection"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "labeled.jsonl").write_text("{}\n", encoding="utf-8")
            (directory / "selected.jsonl").write_text("{}\n", encoding="utf-8")
            (directory / "summary.json").write_text("{}\n", encoding="utf-8")
        elif "build_perception_memory_sft.py" in joined:
            directory = args.run_root / "sft_data"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "perception_memory_sft.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (directory / "summary.json").write_text("{}\n", encoding="utf-8")
        elif "--smoke" in command:
            report = args.run_root / "checkpoints/smoke/preflight/training_update.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"status":"passed"}\n', encoding="utf-8")
        elif "train_qwen_agent_9b_lora.sh" in joined:
            (args.run_root / "checkpoints/formal/checkpoint-1").mkdir(
                parents=True, exist_ok=True
            )
        return 0

    monkeypatch.setattr(autopilot, "_run_command", fake_run)
    first = autopilot.run_pipeline(args)
    assert first["status"] == "passed"
    assert first["current_stage"] == "complete"
    assert stops == [True]
    formal = [item for item in commands if "train_qwen_agent_9b_lora.sh" in " ".join(item)][-1]
    assert "--resume" not in formal

    commands.clear()
    args.resume = True
    resumed = autopilot.run_pipeline(args)
    assert resumed["status"] == "passed"
    assert commands == []

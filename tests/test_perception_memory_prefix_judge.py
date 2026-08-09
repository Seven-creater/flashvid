import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval import perception_memory_prefix_judge as prefix_module
from flashvid_eval.perception_memory_prefix_judge import (
    PerceptionMemoryPrefixJudge,
    PrefixJudgeConfig,
    bind_prefix_jobs,
    build_prefix_judge_messages,
    parse_prefix_judge_response,
    implementation_dependency_hashes,
)
from scripts import judge_perception_memory_prefixes as judge_script


def _memory(*, complete: bool = False) -> dict:
    events = [
        {
            "evidence_id": "E0001",
            "interval": [0.0, 10.0],
            "timestamp": 3.0,
            "fact": "The person opens the left door.",
            "source": "timestamped_fact",
        }
    ]
    if complete:
        events.append(
            {
                "evidence_id": "E0002",
                "interval": [10.0, 20.0],
                "timestamp": 13.0,
                "fact": "The person walks through the left door.",
                "source": "timestamped_fact",
            }
        )
    return {
        "event_ledger": events,
        "option_ledger": {
            "A": {"supports": [], "contradicts": ["E0001"]},
            "B": {
                "supports": [item["evidence_id"] for item in events],
                "contradicts": [],
            },
        },
        "unresolved": [] if complete else ["What happens next?"],
        "observed_intervals": [[0.0, 10.0]]
        + ([[10.0, 20.0]] if complete else []),
    }


def _trajectory(trajectory_id: str = "lvbench:s1:family:0") -> dict:
    return {
        "dataset": "lvbench",
        "sample_id": "s1",
        "trajectory_id": trajectory_id,
        "config_sha256": "c" * 64,
        "run_fingerprint": "r" * 64,
        "scoring_deferred": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        # Candidate may exist in the deferred trace, but it must never enter a job.
        "candidate_answer": "A",
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": "s1",
            "video": "/data/video.mp4",
            "question": "Which door does the person use?",
            "choices": {"A": "right", "B": "left"},
        },
        "perception_states": [
            {"step_index": 0, "memory_after": _memory()},
            {"step_index": 1, "memory_after": _memory(complete=True)},
        ],
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, model, messages, max_tokens=32, **kwargs):
        self.calls.append(
            {
                "model": model,
                "messages": deepcopy(messages),
                "max_tokens": max_tokens,
                **deepcopy(kwargs),
            }
        )
        return ChatResult(
            content='{"answer":"B","evidence_ids":["E0001"]}',
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            raw={},
            latency_s=0.1,
            finish_reason="stop",
        )


def test_strict_prefix_parser_requires_valid_cited_evidence() -> None:
    decision = parse_prefix_judge_response(
        '{"answer":"B","evidence_ids":["E0001"]}',
        ("A", "B"),
        ("E0001",),
    )
    assert decision is not None
    assert decision.answer == "B"
    assert decision.evidence_ids == ("E0001",)
    assert (
        parse_prefix_judge_response(
            '{"answer":"B","evidence_ids":["E9999"]}',
            ("A", "B"),
            ("E0001",),
        )
        is None
    )
    assert (
        parse_prefix_judge_response(
            '{"answer":"B","evidence_ids":["E0001"],"extra":1}',
            ("A", "B"),
            ("E0001",),
        )
        is None
    )


def test_binding_uses_only_public_sample_and_memory() -> None:
    row = _trajectory()
    jobs = bind_prefix_jobs([row])
    assert len(jobs) == 2
    assert jobs[0].sample.candidate_answer is None
    assert jobs[0].memory.evidence_ids == frozenset({"E0001"})
    messages = build_prefix_judge_messages(jobs[0].sample, jobs[0].memory)
    serialized = json.dumps(messages)
    assert "direct candidate" not in serialized.casefold()
    assert "candidate_answer" not in serialized
    assert "image_url" not in serialized
    assert "/data/video.mp4" not in serialized

    leaked = deepcopy(row)
    leaked["public_sample"]["answer"] = "B"
    with pytest.raises(ValueError, match="public_sample"):
        bind_prefix_jobs([leaked])

    leaked = deepcopy(row)
    leaked["public_sample"]["choices"]["B"] += " SECRET_ANNOTATION_SENTINEL"
    with pytest.raises(ValueError, match="sentinel"):
        bind_prefix_jobs([leaked])


def test_three_seed_judge_disables_tools_records_requests_and_resumes() -> None:
    job = bind_prefix_jobs([_trajectory()])[0]
    client = FakeClient()
    judge = PerceptionMemoryPrefixJudge(client, PrefixJudgeConfig())
    updates: list[dict] = []
    result = judge.judge(job, on_update=updates.append)

    assert result["judge_status"] == "complete"
    assert [item["judge_seed"] for item in result["judge_confirmations"]] == [
        17,
        42,
        73,
    ]
    assert len(updates) == 3
    assert len(client.calls) == 3
    for call, confirmation in zip(client.calls, result["judge_confirmations"]):
        assert call["extra_body"] == {"tool_choice": "none"}
        assert call["chat_template_kwargs"] == {"enable_thinking": False}
        assert call["seed"] == confirmation["judge_seed"]
        assert confirmation["request_kwargs"]["tool_choice"] == "none"
        assert confirmation["media_count"] == 0
        assert confirmation["candidate_blind"] is True
        assert confirmation["parsed_valid"] is True

    resumed_client = FakeClient()
    resumed = PerceptionMemoryPrefixJudge(resumed_client).judge(job, existing=result)
    assert resumed == result
    assert resumed_client.calls == []


def test_retry_errors_replaces_only_failed_seed_and_preserves_successes() -> None:
    job = bind_prefix_jobs([_trajectory()])[0]
    complete = PerceptionMemoryPrefixJudge(FakeClient()).judge(job)
    failed = deepcopy(complete)
    for confirmation in failed["judge_confirmations"]:
        if confirmation["judge_seed"] == 42:
            confirmation.update(
                {
                    "parsed_valid": True,
                    "error": "RuntimeError: temporary API failure",
                    "error_type": "RuntimeError",
                }
            )
        elif confirmation["judge_seed"] == 73:
            confirmation.update(
                {
                    "prediction": None,
                    "evidence_ids": [],
                    "parsed_valid": False,
                    "error": None,
                    "error_type": None,
                }
            )
    failed["judge_status"] = "complete_with_failures"
    preserved = {
        item["judge_seed"]: deepcopy(item)
        for item in failed["judge_confirmations"]
        if item["judge_seed"] == 17
    }

    skipped_client = FakeClient()
    skipped = PerceptionMemoryPrefixJudge(skipped_client).judge(job, existing=failed)
    assert skipped["judge_status"] == "complete_with_failures"
    assert skipped_client.calls == []

    retry_client = FakeClient()
    retried = PerceptionMemoryPrefixJudge(retry_client).judge(
        job,
        existing=failed,
        retry_errors=True,
    )
    assert retried["judge_status"] == "complete"
    assert [call["seed"] for call in retry_client.calls] == [42, 73]
    by_seed = {item["judge_seed"]: item for item in retried["judge_confirmations"]}
    assert by_seed[17] == preserved[17]
    assert by_seed[42]["parsed_valid"] is True
    assert by_seed[73]["parsed_valid"] is True


def test_config_fingerprint_covers_implementation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = implementation_dependency_hashes()
    assert set(dependencies) == {
        "prefix_judge",
        "perception_memory_eva",
        "privacy",
        "qwen_agent_core",
        "eva_official",
    }
    assert all(len(value) == 64 for value in dependencies.values())
    monkeypatch.setattr(
        prefix_module,
        "implementation_dependency_hashes",
        lambda: {"dependency": "a" * 64},
    )
    first = PrefixJudgeConfig().fingerprint()
    job = bind_prefix_jobs([_trajectory()])[0]
    persisted = PerceptionMemoryPrefixJudge(FakeClient()).judge(job)
    monkeypatch.setattr(
        prefix_module,
        "implementation_dependency_hashes",
        lambda: {"dependency": "b" * 64},
    )
    assert PrefixJudgeConfig().fingerprint() != first
    with pytest.raises(RuntimeError, match="prefix_judge_config_sha256"):
        PerceptionMemoryPrefixJudge(FakeClient()).judge(job, existing=persisted)


def test_resume_rejects_changed_prefix_source() -> None:
    first = bind_prefix_jobs([_trajectory()])[0]
    result = PerceptionMemoryPrefixJudge(FakeClient()).judge(first)
    changed = _trajectory()
    changed["perception_states"][0]["memory_after"]["event_ledger"][0][
        "fact"
    ] = "The person closes the left door."
    second = bind_prefix_jobs([changed])[0]
    with pytest.raises(RuntimeError, match="source_sha256"):
        PerceptionMemoryPrefixJudge(FakeClient()).judge(second, existing=result)


def test_concurrent_script_compacts_progress_and_resume_skips_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        json.dumps(_trajectory(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output = tmp_path / "prefix_judgments.jsonl"
    clients: list[FakeClient] = []

    def client_factory(*_args, **_kwargs):
        client = FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr(judge_script, "OpenAICompatibleClient", client_factory)

    def args(resume: bool) -> SimpleNamespace:
        return SimpleNamespace(
            trajectories=[trajectory_path],
            output=output,
            base_url="http://unused/v1",
            api_key="no",
            timeout=1.0,
            model="Qwen3.5-9B",
            judge_seed=[17, 42, 73],
            max_tokens=512,
            temperature=0.2,
            concurrency=2,
            resume=resume,
            retry_errors=False,
        )

    summary = judge_script.run(args(False))
    assert summary["prefixes"] == 2
    assert summary["complete"] == 2
    assert len(clients[0].calls) == 6
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert output.with_suffix(".jsonl.progress.jsonl").read_text(encoding="utf-8") == ""

    resumed = judge_script.run(args(True))
    assert resumed["complete"] == 2
    assert clients[1].calls == []


def test_concurrent_script_retry_errors_only_reissues_failed_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        json.dumps(_trajectory(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output = tmp_path / "prefix_judgments.jsonl"
    clients: list[FakeClient] = []

    def client_factory(*_args, **_kwargs):
        client = FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr(judge_script, "OpenAICompatibleClient", client_factory)

    def args(*, resume: bool, retry_errors: bool) -> SimpleNamespace:
        return SimpleNamespace(
            trajectories=[trajectory_path],
            output=output,
            base_url="http://unused/v1",
            api_key="no",
            timeout=1.0,
            model="Qwen3.5-9B",
            judge_seed=[17, 42, 73],
            max_tokens=512,
            temperature=0.2,
            concurrency=2,
            resume=resume,
            retry_errors=retry_errors,
        )

    judge_script.run(args(resume=False, retry_errors=False))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    failed_prefix = rows[0]["prefix_id"]
    preserved = {}
    for confirmation in rows[0]["judge_confirmations"]:
        if confirmation["judge_seed"] == 42:
            confirmation.update(
                {
                    "prediction": None,
                    "evidence_ids": [],
                    "parsed_valid": False,
                    "error": "RuntimeError: temporary API failure",
                    "error_type": "RuntimeError",
                }
            )
        else:
            preserved[confirmation["judge_seed"]] = deepcopy(confirmation)
    rows[0]["judge_status"] = "complete_with_failures"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = judge_script.run(args(resume=True, retry_errors=True))
    assert summary["complete"] == 2
    assert summary["complete_with_failures"] == 0
    assert summary["retry_errors"] is True
    assert [call["seed"] for call in clients[1].calls] == [42]
    final_rows = {
        row["prefix_id"]: row
        for row in (
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        )
    }
    final_by_seed = {
        item["judge_seed"]: item
        for item in final_rows[failed_prefix]["judge_confirmations"]
    }
    assert final_by_seed[17] == preserved[17]
    assert final_by_seed[73] == preserved[73]

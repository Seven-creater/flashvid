from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_visual_csv import (
    VisualCsvConfig,
    VisualCsvVerifier,
    attach_visual_csv_results,
    bind_visual_csv_jobs,
    build_visual_csv_messages,
    label_visual_csv_prefixes,
    parse_visual_csv_response,
)
from scripts import judge_perception_memory_visual_csv as visual_csv_cli


def _response(
    answer: str = "A",
    indices: list[int] | None = None,
    *,
    evidence_complete: bool = True,
    missing_evidence: list[str] | None = None,
) -> str:
    if indices is None:
        indices = [0]
    if missing_evidence is None:
        missing_evidence = [] if evidence_complete else ["later action"]
    return json.dumps(
        {
            "answer": answer,
            "frame_indices": indices,
            "evidence_complete": evidence_complete,
            "missing_evidence": missing_evidence,
        }
    )


class _Client:
    def __init__(self, response: str | None = None) -> None:
        self.response = response or _response()
        self.calls: list[dict] = []

    def chat(self, model: str, messages: list[dict], **kwargs: object) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return ChatResult(
            content=self.response,
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            raw={},
            latency_s=0.01,
            finish_reason="stop",
        )


class _SeedFailureClient(_Client):
    def __init__(
        self,
        *,
        infrastructure_seeds: tuple[int, ...] = (),
        invalid_seeds: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.infrastructure_seeds = set(infrastructure_seeds)
        self.invalid_seeds = set(invalid_seeds)

    def chat(self, model: str, messages: list[dict], **kwargs: object) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        seed = kwargs.get("seed")
        if seed in self.infrastructure_seeds:
            raise RuntimeError("temporary API failure")
        return ChatResult(
            content="not-json" if seed in self.invalid_seeds else self.response,
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            raw={},
            latency_s=0.01,
            finish_reason="stop",
        )


def _trajectory(tmp_path: Path) -> dict:
    first = (tmp_path / "frame-0.jpg").resolve()
    second = (tmp_path / "frame-1.jpg").resolve()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    return {
        "dataset": "lvbench",
        "sample_id": "sample-1",
        "trajectory_id": "lvbench:sample-1:visual:0",
        "run_fingerprint": "f" * 64,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_answer": "SECRET_CANDIDATE_VALUE",
        "ground_truth": "SECRET_GROUND_TRUTH_VALUE",
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": "sample-1",
            "video": "video.mp4",
            "question": "What does the person do?",
            "choices": {"A": "opens the door", "B": "sits down"},
        },
        "perception_states": [
            {
                "step_index": 0,
                "frame_paths": [str(first)],
                "timestamps": [1.0],
                "memory_after": {"event_ledger": "SECRET_LEDGER_VALUE"},
                "raw_reasoning": "SECRET_PRIOR_REASONING_VALUE",
            },
            {
                "step_index": 1,
                "frame_paths": [str(first), str(second)],
                "timestamps": [1.0, 2.0],
                "memory_after": {"event_ledger": "SECRET_LEDGER_VALUE"},
                "raw_reasoning": "SECRET_PRIOR_REASONING_VALUE",
            },
        ],
    }


def test_jobs_accumulate_only_real_frames_and_messages_are_private_free(
    tmp_path: Path,
) -> None:
    jobs = bind_visual_csv_jobs([_trajectory(tmp_path)])

    assert [len(job.frame_paths) for job in jobs] == [1, 2]
    assert jobs[1].timestamps == (1.0, 2.0)
    serialized = json.dumps(build_visual_csv_messages(jobs[1]), ensure_ascii=False)
    for forbidden in (
        "SECRET_CANDIDATE_VALUE",
        "SECRET_GROUND_TRUTH_VALUE",
        "SECRET_LEDGER_VALUE",
        "SECRET_PRIOR_REASONING_VALUE",
        "Direct candidate",
    ):
        assert forbidden not in serialized
    assert serialized.count('"type": "image_url"') == 2


def test_missing_frame_and_misaligned_prefix_fail_closed(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    Path(trajectory["perception_states"][0]["frame_paths"][0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        bind_visual_csv_jobs([trajectory])

    trajectory = _trajectory(tmp_path)
    trajectory["perception_states"][1]["timestamps"].pop()
    with pytest.raises(ValueError, match="aligned"):
        bind_visual_csv_jobs([trajectory])


@pytest.mark.parametrize(
    "text",
    (
        _response(indices=[2]),
        _response(indices=[]),
        _response(indices=[0, 0]),
        _response(evidence_complete=True, missing_evidence=["should be empty"]),
        _response(evidence_complete=False, missing_evidence=[]),
        '{"answer":"A","frame_indices":[0],"evidence_complete":true}',
    ),
)
def test_parser_rejects_bad_indices_or_completeness_contract(text: str) -> None:
    assert parse_visual_csv_response(text, ("A", "B"), 2) is None


def test_three_seed_verification_persists_predictions_indices_and_boolean(
    tmp_path: Path,
) -> None:
    job = bind_visual_csv_jobs([_trajectory(tmp_path)])[0]
    client = _Client(
        _response(evidence_complete=False, missing_evidence=["later action"])
    )
    verifier = VisualCsvVerifier(client)

    row = verifier.verify(job)

    confirmations = row["visual_csv_confirmations"]
    assert [item["judge_seed"] for item in confirmations] == [17, 42, 73]
    assert [item["prediction"] for item in confirmations] == ["A", "A", "A"]
    assert [item["frame_indices"] for item in confirmations] == [[0], [0], [0]]
    assert all(item["evidence_complete"] is False for item in confirmations)
    assert all(item["missing_evidence"] == ["later action"] for item in confirmations)
    assert all(item["candidate_blind"] is True for item in confirmations)
    assert row["visual_csv_status"] == "complete"
    assert row["visual_csv_judge_seeds"] == [17, 42, 73]
    assert [call["seed"] for call in client.calls] == [17, 42, 73]
    serialized = json.dumps(row, ensure_ascii=False)
    for forbidden in (
        "SECRET_CANDIDATE_VALUE",
        "SECRET_GROUND_TRUTH_VALUE",
        "SECRET_LEDGER_VALUE",
        "SECRET_PRIOR_REASONING_VALUE",
    ):
        assert forbidden not in serialized


def test_resume_rejects_source_config_and_seed_drift(tmp_path: Path) -> None:
    job = bind_visual_csv_jobs([_trajectory(tmp_path)])[0]
    row = VisualCsvVerifier(_Client()).verify(job)

    VisualCsvVerifier(_Client()).verify(job, existing=row)
    with pytest.raises(RuntimeError, match="config"):
        VisualCsvVerifier(
            _Client(), VisualCsvConfig(max_tokens=257)
        ).verify(job, existing=row)

    changed_job = deepcopy(job)
    object.__setattr__(changed_job, "source_sha256", "a" * 64)
    with pytest.raises(RuntimeError, match="source"):
        VisualCsvVerifier(_Client()).verify(changed_job, existing=row)

    duplicate = deepcopy(row)
    duplicate["visual_csv_confirmations"][1]["judge_seed"] = 17
    with pytest.raises(RuntimeError, match="duplicate"):
        VisualCsvVerifier(_Client()).verify(job, existing=duplicate)

    tampered = deepcopy(row)
    tampered["visual_csv_confirmations"][0]["request_messages"][1]["content"][0][
        "text"
    ] = "changed request"
    with pytest.raises(RuntimeError, match="provenance"):
        VisualCsvVerifier(_Client()).verify(job, existing=tampered)


def test_retry_errors_reissues_only_infrastructure_seed(tmp_path: Path) -> None:
    job = bind_visual_csv_jobs([_trajectory(tmp_path)])[0]
    first_client = _SeedFailureClient(
        infrastructure_seeds=(42,), invalid_seeds=(73,)
    )
    failed = VisualCsvVerifier(first_client).verify(job)
    assert failed["visual_csv_status"] == "complete_with_failures"

    skipped_client = _Client()
    skipped = VisualCsvVerifier(skipped_client).verify(job, existing=failed)
    assert skipped["visual_csv_status"] == "complete_with_failures"
    assert skipped_client.calls == []

    retry_client = _Client()
    retried = VisualCsvVerifier(retry_client).verify(
        job, existing=failed, retry_errors=True
    )
    assert [call["seed"] for call in retry_client.calls] == [42]
    assert retried["visual_csv_status"] == "complete_with_failures"
    by_seed = {
        item["judge_seed"]: item
        for item in retried["visual_csv_confirmations"]
    }
    assert by_seed[42]["parsed_valid"] is True
    assert by_seed[73]["failure_class"] == "model_parse_failure"


def test_offline_attach_and_label_uses_three_predictions_not_model_boolean(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    jobs = bind_visual_csv_jobs([trajectory])
    verifier = VisualCsvVerifier(
        _Client(_response(evidence_complete=False, missing_evidence=["more"])),
    )
    rows = [verifier.verify(job) for job in jobs]

    attached = attach_visual_csv_results([trajectory], rows)
    labeled = label_visual_csv_prefixes(
        attached, {("lvbench", "sample-1"): "A"}
    )

    assert all(
        state["evidence_complete"] is True
        for state in labeled[0]["perception_states"]
    )
    assert all(
        confirmation["evidence_complete"] is False
        for state in labeled[0]["perception_states"]
        for confirmation in state["visual_csv_confirmations"]
    )
    serialized = json.dumps(labeled, ensure_ascii=False)
    assert "SECRET_GROUND_TRUTH_VALUE" not in serialized
    assert "SECRET_CANDIDATE_VALUE" not in serialized
    assert "SECRET_LEDGER_VALUE" in serialized
    assert labeled[0]["offline_label_join"] == (
        "ground_truth_used_for_boolean_only_not_serialized"
    )

    wrong = label_visual_csv_prefixes(
        attached, {("lvbench", "sample-1"): "B"}
    )
    assert all(
        state["evidence_complete"] is False
        for state in wrong[0]["perception_states"]
    )


def test_cli_is_multi_endpoint_resumable_and_fingerprint_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        json.dumps(_trajectory(tmp_path), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "visual.csv.jsonl"
    clients: list[_Client] = []

    def client_factory(*args: object, **kwargs: object) -> _Client:
        del args, kwargs
        client = _Client()
        clients.append(client)
        return client

    monkeypatch.setattr(visual_csv_cli, "OpenAICompatibleClient", client_factory)
    args = argparse.Namespace(
        trajectories=[trajectory_path],
        output=output,
        base_urls=["http://one/v1", "http://two/v1"],
        api_key="no",
        model="Qwen3.5-9B",
        judge_seed=[17, 42, 73],
        max_tokens=256,
        temperature=0.2,
        timeout=80.0,
        concurrency=2,
        local_media_paths=True,
        resume=False,
        retry_errors=False,
    )

    summary = visual_csv_cli.run(args)
    assert summary["endpoint_count"] == 2
    assert summary["complete"] == 2
    assert sum(len(client.calls) for client in clients) == 6

    clients.clear()
    args.resume = True
    summary = visual_csv_cli.run(args)
    assert summary["complete"] == 2
    assert sum(len(client.calls) for client in clients) == 0

    args.max_tokens = 257
    with pytest.raises(RuntimeError, match="config"):
        visual_csv_cli.run(args)


def test_cli_retry_errors_only_reissues_infrastructure_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        json.dumps(_trajectory(tmp_path), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "visual.csv.jsonl"
    clients: list[_Client] = []
    fail_first_run = True

    def client_factory(*args: object, **kwargs: object) -> _Client:
        del args, kwargs
        client: _Client = (
            _SeedFailureClient(infrastructure_seeds=(42,))
            if fail_first_run
            else _Client()
        )
        clients.append(client)
        return client

    monkeypatch.setattr(visual_csv_cli, "OpenAICompatibleClient", client_factory)
    args = argparse.Namespace(
        trajectories=[trajectory_path],
        output=output,
        base_urls=["http://one/v1"],
        api_key="no",
        model="Qwen3.5-9B",
        judge_seed=[17, 42, 73],
        max_tokens=256,
        temperature=0.2,
        timeout=80.0,
        concurrency=2,
        local_media_paths=True,
        resume=False,
        retry_errors=False,
    )

    summary = visual_csv_cli.run(args)
    assert summary["complete_with_failures"] == 2
    clients.clear()
    fail_first_run = False

    args.resume = True
    summary = visual_csv_cli.run(args)
    assert summary["complete_with_failures"] == 2
    assert sum(len(client.calls) for client in clients) == 0

    clients.clear()
    args.retry_errors = True
    summary = visual_csv_cli.run(args)
    assert summary["complete"] == 2
    assert summary["retry_errors"] is True
    assert [call["seed"] for client in clients for call in client.calls] == [42, 42]

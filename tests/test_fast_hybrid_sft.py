from __future__ import annotations

from pathlib import Path

import pytest

from flashvid_eval.fast_hybrid_sft import (
    build_fast_hybrid_sft_records,
    validate_fast_hybrid_frame_files,
)
from flashvid_eval.privacy import AnnotationLeakError


def _trajectory(frame: Path) -> dict:
    tool = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":0,"end_time":10,"nframes":1,"resize":1.0}}'
        "</tool_call>"
    )
    base_messages = [
        {"role": "system", "content": "Use EVA tools."},
        {"role": "user", "content": "Question: what happens?\nA: one\nB: two\nC: three"},
    ]
    second_messages = [
        *base_messages,
        {"role": "assistant", "content": f"I should inspect.\n{tool}"},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "<tool_response>Frame at 1 second:"},
                {"type": "image_url", "image_url": {"url": frame.as_uri()}},
                {"type": "text", "text": "</tool_response>"},
            ],
        },
    ]
    return {
        "dataset": "lsdbench",
        "sample_id": "one",
        "trajectory_id": "lsdbench:one:budget_006000_seed_17:0",
        "family_id": "budget_006000_seed_17",
        "manifest_sha256": "a" * 64,
        "train600_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "scoring_deferred": True,
        "_selection_stable": True,
        "prediction": "C",
        "final_prediction": "C",
        "candidate_answer": "B",
        "total_tokens": 123,
        "visual_tokens": 10,
        "request_trace": [
            {
                "stage": "verification",
                "prompt_hash": "d" * 64,
                "seed": 17,
                "attempt_index": 1,
                "messages": base_messages,
                "content": f"private-looking prose is removed\n{tool}",
                "finish_reason": "stop",
                "usage": {"total_tokens": 50},
            },
            {
                "stage": "verification",
                "prompt_hash": "e" * 64,
                "seed": 17,
                "attempt_index": 1,
                "messages": second_messages,
                "content": "The evidence is enough.\nAnswer: C",
                "finish_reason": "stop",
                "usage": {"total_tokens": 73},
            },
        ],
        "tool_steps": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 1,
                "resize": 1.0,
                "timestamps": [1.0],
                "actual_timestamps": [1.0],
                "frame_paths": [str(frame)],
            }
        ],
    }


def test_fast_hybrid_export_masks_context_and_compresses_targets(tmp_path: Path) -> None:
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")
    records = build_fast_hybrid_sft_records(_trajectory(frame))

    assert len(records) == 2
    assert records[0]["metadata"]["assistant_target_types"] == ["tool"]
    assert records[1]["metadata"]["assistant_target_types"] == ["final"]
    tool_target = records[0]["messages"][-1]
    final_target = records[1]["messages"][-1]
    assert tool_target["loss"] is True
    assert tool_target["content"].startswith("<tool_call>")
    assert "private-looking prose" not in tool_target["content"]
    assert final_target == {"role": "assistant", "content": "Answer: C", "loss": True}
    assert records[1]["images"] == [str(frame)]
    assert all(
        message.get("loss") is not True
        for record in records
        for message in record["messages"][:-1]
    )


def test_fast_hybrid_export_rejects_private_scoring_fields(tmp_path: Path) -> None:
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")
    trajectory = _trajectory(frame)
    trajectory["answer"] = "C"
    with pytest.raises(AnnotationLeakError, match="answer"):
        build_fast_hybrid_sft_records(trajectory)


def test_fast_hybrid_export_requires_stable_selection_marker(tmp_path: Path) -> None:
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")
    trajectory = _trajectory(frame)
    trajectory.pop("_selection_stable")
    with pytest.raises(ValueError, match="3/3-stable"):
        build_fast_hybrid_sft_records(trajectory)


def test_fast_hybrid_export_masks_rejected_intermediate_answer(tmp_path: Path) -> None:
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")
    trajectory = _trajectory(frame)
    final_request = dict(trajectory["request_trace"][-1])
    final_request.update(
        {
            "stage": "change_confirmation",
            "prompt_hash": "f" * 64,
            "content": "Answer: B",
        }
    )
    trajectory["request_trace"][-1]["content"] = "Answer: C"
    trajectory["prediction"] = "B"
    trajectory["final_prediction"] = "B"
    trajectory["request_trace"].append(final_request)

    records = build_fast_hybrid_sft_records(trajectory)

    assert [record["metadata"]["assistant_target_types"] for record in records] == [
        ["tool"],
        ["final"],
    ]
    trained = [
        message["content"]
        for record in records
        for message in record["messages"]
        if message.get("loss") is True
    ]
    assert trained[-1] == "Answer: B"
    assert "Answer: C" not in trained


def test_frame_audit_requires_existing_one_to_one_files(tmp_path: Path) -> None:
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"image")
    trajectory = _trajectory(frame)
    validate_fast_hybrid_frame_files(trajectory)
    trajectory["tool_steps"][0]["actual_timestamps"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="count differs"):
        validate_fast_hybrid_frame_files(trajectory)

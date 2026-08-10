from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.perception_memory_eva import PERCEPTION_NORMALIZATION_VERSION
from flashvid_eval.perception_memory_prefix_judge import bind_prefix_jobs
from flashvid_eval.perception_memory_repair import (
    merge_replay_results,
    prepare_repair_scope,
)
from flashvid_eval.perception_memory_replay import canonical_sha256, file_sha256
from flashvid_eval.runner import parse_question_time_range
from scripts import reconcile_perception_memory_replay as repair_script
from scripts.run_perception_memory_explicit_time_rescue import _validate_manifest_row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source(
    source_id: str,
    *,
    sample_id: str,
    question: str = "What does the person do?",
    interval: tuple[float, float] = (0.0, 10.0),
) -> dict:
    return {
        "dataset": "lvbench",
        "sample_id": sample_id,
        "video": f"videos/{sample_id}.mp4",
        "trajectory_id": source_id,
        "candidate_answer": "A",
        "scoring_deferred": True,
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": sample_id,
            "video": f"videos/{sample_id}.mp4",
            "question": question,
            "choices": {"A": "waits", "B": "opens the door"},
        },
        "tool_steps": [
            {
                "start_time": interval[0],
                "end_time": interval[1],
                "nframes": 1,
                "resize": 0.75,
                "frame_paths": [f"/cache/{sample_id}.jpg"],
                "actual_timestamps": [sum(interval) / 2.0],
            }
        ],
    }


def _memory() -> dict:
    return {
        "event_ledger": [
            {
                "evidence_id": "E0001",
                "interval": [0.0, 10.0],
                "timestamp": 5.0,
                "fact": "The person opens the door.",
                "source": "timestamped_fact",
            }
        ],
        "option_ledger": {
            "A": {"supports": [], "contradicts": ["E0001"]},
            "B": {"supports": ["E0001"], "contradicts": []},
        },
        "unresolved": [],
        "observed_intervals": [[0.0, 10.0]],
    }


def _result(source: dict, source_file_sha256: str, *, error: str | None = None) -> dict:
    source_id = source["trajectory_id"]
    result = {
        "schema_version": 1,
        "dataset": source["dataset"],
        "sample_id": source["sample_id"],
        "trajectory_id": f"{source_id}:perception_memory_replay_v3",
        "source_trajectory_id": source_id,
        "source_row_sha256": canonical_sha256(source),
        "source_file_sha256": source_file_sha256,
        "candidate_answer": source["candidate_answer"],
        "config_sha256": "c" * 64,
        "run_fingerprint": "r" * 64,
        "scoring_deferred": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "perception_normalization_version": PERCEPTION_NORMALIZATION_VERSION,
        "public_sample": source["public_sample"],
        "perception_states": [{"step_index": 0, "memory_after": _memory()}],
        "error": error,
        "error_type": "ValueError" if error else None,
    }
    return result


def _fixture(tmp_path: Path) -> tuple[list[Path], list[Path], list[dict]]:
    success = _source("lvbench:s0:teacher:0", sample_id="s0")
    cached = _source("lvbench:s1:teacher:0", sample_id="s1")
    explicit = _source(
        "lvbench:s2:teacher:0",
        sample_id="s2",
        question="What happens at 65:02?",
        interval=(64.02, 66.02),
    )
    explicit["tool_steps"].append(
        {
            "start_time": 3901.0,
            "end_time": 3903.0,
            "nframes": 1,
            "resize": 0.75,
            "frame_paths": ["/cache/s2-correct.jpg"],
            "actual_timestamps": [3902.0],
        }
    )
    source_0 = _write_jsonl(tmp_path / "source-0.jsonl", [success, cached])
    source_1 = _write_jsonl(tmp_path / "source-1.jsonl", [explicit])
    base_0 = _write_jsonl(
        tmp_path / "base-0.jsonl",
        [
            _result(success, file_sha256(source_0)),
            _result(
                cached,
                file_sha256(source_0),
                error="ValueError: invalid_json_or_schema",
            ),
        ],
    )
    base_1 = _write_jsonl(
        tmp_path / "base-1.jsonl",
        [
            _result(
                explicit,
                file_sha256(source_1),
                error=(
                    "ValueError: perception step 0 returned invalid evidence after "
                    "retry: invalid_frame_reference"
                ),
            )
        ],
    )
    return [source_0, source_1], [base_0, base_1], [success, cached, explicit]


def _prepare(tmp_path: Path) -> tuple[list[Path], list[Path], list[dict], Path]:
    sources, bases, rows = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    summary = prepare_repair_scope(
        source_paths=sources,
        base_results_paths=bases,
        output_dir=prepared,
        expected_rows=3,
        expected_explicit_time_rows=1,
        expected_explicit_time_samples=1,
    )
    assert summary["cached_repair"] == 1
    assert summary["explicit_time_rescue"] == 1
    return sources, bases, rows, prepared


def test_public_parser_supports_long_video_mm_ss() -> None:
    assert parse_question_time_range("What happens at 65:02?") == (3901.0, 3903.0)


def test_prepare_freezes_exact_failed_lanes_and_shard_lineage(tmp_path: Path) -> None:
    sources, bases, rows, prepared = _prepare(tmp_path)
    cached = _read_jsonl(prepared / "cached_repair_input.jsonl")
    rescue = _read_jsonl(prepared / "explicit_time_mismatch.jsonl")
    scope = json.loads((prepared / "frozen_scope.json").read_text(encoding="utf-8"))

    assert cached == [rows[1]]
    assert rescue[0]["source_row"] == rows[2]
    assert rescue[0]["classification"] == "source_explicit_time_parse_mismatch"
    assert rescue[0]["reason"] == "explicit_time_source_interval_mismatch"
    assert rescue[0]["parsed_time_source"] == "public_question"
    assert rescue[0]["parsed_time_range"] == [3901.0, 3903.0]
    assert rescue[0]["failed_step_index"] == 0
    assert rescue[0]["failed_step_requested_interval"] == [64.02, 66.02]
    assert rescue[0]["legacy_decimalized_interval"] == [64.02, 66.02]
    assert rescue[0]["source_requested_intervals"] == [
        [64.02, 66.02],
        [3901.0, 3903.0],
    ]
    rescued_sample, source_intervals = _validate_manifest_row(rescue[0])
    assert rescued_sample.candidate_answer == "A"
    assert source_intervals == ((64.02, 66.02), (3901.0, 3903.0))
    assert scope["base_failure_ids"] == [
        "lvbench:s1:teacher:0",
        "lvbench:s2:teacher:0",
    ]
    assert len(scope["source"]["files"]) == len(sources) == 2
    assert len(scope["base"]["files"]) == len(bases) == 2
    assert all(
        row["source_file_sha256"]
        == file_sha256(sources[0] if row["sample_id"] != "s2" else sources[1])
        for row in _read_jsonl(bases[0]) + _read_jsonl(bases[1])
    )


def test_prepare_requires_complete_equal_scope_and_never_overwrites(
    tmp_path: Path,
) -> None:
    sources, bases, _rows = _fixture(tmp_path)
    with pytest.raises(ValueError, match="complete source/base scope"):
        prepare_repair_scope(
            source_paths=sources,
            base_results_paths=bases,
            output_dir=tmp_path / "incomplete",
            expected_rows=2917,
            expected_explicit_time_rows=1,
            expected_explicit_time_samples=1,
        )
    prepare_repair_scope(
        source_paths=sources,
        base_results_paths=bases,
        output_dir=tmp_path / "prepared",
        expected_rows=3,
        expected_explicit_time_rows=1,
        expected_explicit_time_samples=1,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_repair_scope(
            source_paths=sources,
            base_results_paths=bases,
            output_dir=tmp_path / "prepared",
            expected_rows=3,
            expected_explicit_time_rows=1,
            expected_explicit_time_samples=1,
        )


def test_prepare_rejects_tampered_per_row_shard_lineage(tmp_path: Path) -> None:
    sources, bases, _rows = _fixture(tmp_path)
    base_rows = _read_jsonl(bases[1])
    base_rows[0]["source_file_sha256"] = file_sha256(sources[0])
    _write_jsonl(bases[1], base_rows)
    with pytest.raises(ValueError, match="source_file_sha256 mismatch"):
        prepare_repair_scope(
            source_paths=sources,
            base_results_paths=bases,
            output_dir=tmp_path / "prepared",
            expected_rows=3,
            expected_explicit_time_rows=1,
            expected_explicit_time_samples=1,
        )


def test_prepare_does_not_call_unrelated_failed_interval_a_time_parse_mismatch(
    tmp_path: Path,
) -> None:
    source = _source(
        "lvbench:s4:teacher:0",
        sample_id="s4",
        question="What happens at 65:02?",
        interval=(100.0, 110.0),
    )
    source_path = _write_jsonl(tmp_path / "source-unrelated.jsonl", [source])
    base = _result(
        source,
        file_sha256(source_path),
        error=(
            "ValueError: perception step 0 returned invalid evidence after retry: "
            "invalid_frame_reference"
        ),
    )
    base_path = _write_jsonl(tmp_path / "base-unrelated.jsonl", [base])
    prepared = tmp_path / "prepared-unrelated"

    summary = prepare_repair_scope(
        source_paths=[source_path],
        base_results_paths=[base_path],
        output_dir=prepared,
        expected_rows=1,
        expected_explicit_time_rows=0,
        expected_explicit_time_samples=0,
    )

    assert summary["cached_repair"] == 1
    assert summary["explicit_time_rescue"] == 0


def _unbindable_fixture(
    tmp_path: Path,
) -> tuple[list[Path], list[Path], list[dict], Path]:
    success = _source("lvbench:s0:teacher:0", sample_id="s0")
    unbindable = _source("lvbench:s3:teacher:0", sample_id="s3")
    source_path = _write_jsonl(tmp_path / "source.jsonl", [success, unbindable])
    good = _result(success, file_sha256(source_path))
    bad = _result(unbindable, file_sha256(source_path))
    bad["perception_states"][0]["memory_after"]["event_ledger"] = []
    base_path = _write_jsonl(tmp_path / "base.jsonl", [good, bad])
    prepared = tmp_path / "prepared-unbindable"
    prepare_repair_scope(
        source_paths=[source_path],
        base_results_paths=[base_path],
        output_dir=prepared,
        expected_rows=2,
        expected_explicit_time_rows=0,
        expected_explicit_time_samples=0,
    )
    return [source_path], [base_path], [success, unbindable], prepared


def test_prepare_classifies_error_free_unbindable_base_as_cached_repair(
    tmp_path: Path,
) -> None:
    _sources, _bases, rows, prepared = _unbindable_fixture(tmp_path)
    cached = _read_jsonl(prepared / "cached_repair_input.jsonl")
    scope = json.loads((prepared / "frozen_scope.json").read_text(encoding="utf-8"))
    source_id = rows[1]["trajectory_id"]

    assert [row["trajectory_id"] for row in cached] == [source_id]
    assert scope["base_success_ids"] == [rows[0]["trajectory_id"]]
    assert scope["base_failure_ids"] == [source_id]
    assert scope["base_failure_reasons"][source_id]["kind"] == "prefix_unbindable"
    assert (
        "memory_after.event_ledger must be non-empty"
        in scope["base_failure_reasons"][source_id]["detail"]
    )


def test_merge_replaces_frozen_unbindable_base_and_never_emits_it(
    tmp_path: Path,
) -> None:
    sources, bases, rows, prepared = _unbindable_fixture(tmp_path)
    source_id = rows[1]["trajectory_id"]
    replacement = _result(rows[1], file_sha256(prepared / "cached_repair_input.jsonl"))
    replacement_path = _write_jsonl(tmp_path / "replacement.jsonl", [replacement])
    merged_dir = tmp_path / "merged-unbindable"

    summary = merge_replay_results(
        source_paths=sources,
        base_results_paths=bases,
        frozen_scope_path=prepared / "frozen_scope.json",
        replacement_result_paths=[replacement_path],
        output_dir=merged_dir,
    )

    merged = _read_jsonl(merged_dir / "merged_success.jsonl")
    by_id = {row["source_trajectory_id"]: row for row in merged}
    assert summary["base_success"] == 1
    assert summary["replacement_success"] == 1
    assert by_id[source_id]["source_file_sha256"] == file_sha256(
        prepared / "cached_repair_input.jsonl"
    )
    assert by_id[source_id]["perception_states"][0]["memory_after"]["event_ledger"]
    assert bind_prefix_jobs(merged)


def test_merge_isolates_unbindable_replacement_as_double_failure(
    tmp_path: Path,
) -> None:
    sources, bases, rows, prepared = _unbindable_fixture(tmp_path)
    source_id = rows[1]["trajectory_id"]
    replacement = _result(rows[1], file_sha256(prepared / "cached_repair_input.jsonl"))
    replacement["perception_states"][0]["memory_after"]["event_ledger"] = []
    replacement_path = _write_jsonl(
        tmp_path / "replacement-unbindable.jsonl", [replacement]
    )
    merged_dir = tmp_path / "merged-double-failure"

    summary = merge_replay_results(
        source_paths=sources,
        base_results_paths=bases,
        frozen_scope_path=prepared / "frozen_scope.json",
        replacement_result_paths=[replacement_path],
        output_dir=merged_dir,
    )

    merged = _read_jsonl(merged_dir / "merged_success.jsonl")
    failures = _read_jsonl(merged_dir / "double_failures.jsonl")
    assert source_id not in {row["source_trajectory_id"] for row in merged}
    assert summary["double_failure_ids"] == [source_id]
    assert failures[0]["base_failure_reason"]["kind"] == "prefix_unbindable"
    assert failures[0]["replacement_failure_reason"]["kind"] == ("prefix_unbindable")
    assert bind_prefix_jobs(merged)


def _replacement_results(
    prepared: Path, sources: list[Path], rows: list[dict]
) -> tuple[dict, dict]:
    cached_input = prepared / "cached_repair_input.jsonl"
    manifest = prepared / "explicit_time_mismatch.jsonl"
    scope = prepared / "frozen_scope.json"
    cached = _result(rows[1], file_sha256(cached_input))
    rescue = _result(rows[2], file_sha256(sources[1]))
    rescue.update(
        {
            "rescue_manifest_sha256": file_sha256(manifest),
            "frozen_scope_sha256": file_sha256(scope),
        }
    )
    return cached, rescue


def test_merge_preserves_base_success_and_binds_all_replacements(
    tmp_path: Path,
) -> None:
    sources, bases, rows, prepared = _prepare(tmp_path)
    cached, rescue = _replacement_results(prepared, sources, rows)
    replacements = _write_jsonl(tmp_path / "replacements.jsonl", [cached, rescue])
    merged_dir = tmp_path / "merged"
    summary = merge_replay_results(
        source_paths=sources,
        base_results_paths=bases,
        frozen_scope_path=prepared / "frozen_scope.json",
        replacement_result_paths=[replacements],
        output_dir=merged_dir,
    )

    merged = _read_jsonl(merged_dir / "merged_success.jsonl")
    assert summary["base_success"] == 1
    assert summary["replacement_success"] == 2
    assert summary["double_failures"] == 0
    assert [row["source_trajectory_id"] for row in merged] == [
        row["trajectory_id"] for row in rows
    ]
    assert bind_prefix_jobs(merged)
    assert _read_jsonl(merged_dir / "double_failures.jsonl") == []
    with pytest.raises(FileExistsError, match="already exists"):
        merge_replay_results(
            source_paths=sources,
            base_results_paths=bases,
            frozen_scope_path=prepared / "frozen_scope.json",
            replacement_result_paths=[replacements],
            output_dir=merged_dir,
        )


def test_merge_keeps_double_failure_out_of_prefix_success(tmp_path: Path) -> None:
    sources, bases, rows, prepared = _prepare(tmp_path)
    cached, rescue = _replacement_results(prepared, sources, rows)
    cached["error"] = "ValueError: invalid_json_or_schema"
    cached["error_type"] = "ValueError"
    replacements = _write_jsonl(tmp_path / "replacements.jsonl", [cached, rescue])
    merged_dir = tmp_path / "merged"
    summary = merge_replay_results(
        source_paths=sources,
        base_results_paths=bases,
        frozen_scope_path=prepared / "frozen_scope.json",
        replacement_result_paths=[replacements],
        output_dir=merged_dir,
    )

    merged = _read_jsonl(merged_dir / "merged_success.jsonl")
    failures = _read_jsonl(merged_dir / "double_failures.jsonl")
    assert summary["status"] == "completed_with_failures"
    assert summary["merged_success"] == 2
    assert [row["source_trajectory_id"] for row in failures] == [
        rows[1]["trajectory_id"]
    ]
    assert rows[1]["trajectory_id"] not in {
        row["source_trajectory_id"] for row in merged
    }
    assert bind_prefix_jobs(merged)


def test_merge_rejects_duplicates_extras_and_changed_frozen_inputs(
    tmp_path: Path,
) -> None:
    sources, bases, rows, prepared = _prepare(tmp_path)
    cached, rescue = _replacement_results(prepared, sources, rows)
    duplicate_0 = _write_jsonl(tmp_path / "duplicate-0.jsonl", [cached, rescue])
    duplicate_1 = _write_jsonl(tmp_path / "duplicate-1.jsonl", [cached])
    with pytest.raises(ValueError, match="duplicate replacement"):
        merge_replay_results(
            source_paths=sources,
            base_results_paths=bases,
            frozen_scope_path=prepared / "frozen_scope.json",
            replacement_result_paths=[duplicate_0, duplicate_1],
            output_dir=tmp_path / "duplicate-merged",
        )

    extra = dict(cached)
    extra["source_trajectory_id"] = "lvbench:extra:teacher:0"
    extra["sample_id"] = "extra"
    extra_path = _write_jsonl(tmp_path / "extra.jsonl", [extra, rescue])
    with pytest.raises(ValueError, match="outside failed scope"):
        merge_replay_results(
            source_paths=sources,
            base_results_paths=bases,
            frozen_scope_path=prepared / "frozen_scope.json",
            replacement_result_paths=[extra_path],
            output_dir=tmp_path / "extra-merged",
        )

    sources[0].write_text(
        sources[0].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source shards differ"):
        merge_replay_results(
            source_paths=sources,
            base_results_paths=bases,
            frozen_scope_path=prepared / "frozen_scope.json",
            replacement_result_paths=[duplicate_0],
            output_dir=tmp_path / "changed-merged",
        )


def test_cli_prepare_uses_2917_default_but_supports_audited_test_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sources, bases, _rows = _fixture(tmp_path)
    exit_code = repair_script.main(
        [
            "prepare",
            "--source",
            *(str(path) for path in sources),
            "--base-results",
            *(str(path) for path in bases),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--expected-rows",
            "3",
            "--expected-explicit-time-rows",
            "1",
            "--expected-explicit-time-samples",
            "1",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

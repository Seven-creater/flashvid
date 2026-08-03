from __future__ import annotations

from scripts.check_raw_trajectories import DATASETS, validate_raw_trajectories


def _record(dataset: str, index: int) -> dict:
    return {
        "dataset": dataset,
        "sample_id": f"{dataset}-sample-{index // 2}",
        "trajectory_id": f"{dataset}-trajectory-{index}",
        "prediction": "A",
        "final_prediction": "A",
        "trajectory_valid": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "training_messages": [{"role": "assistant", "content": "Answer: A"}],
        "tool_steps": [
            {
                "api_multimodal_video_tokens_actual": 96,
                "raw_visual_tokens_actual": None,
                "retained_visual_tokens_actual": None,
                "token_count_source": (
                    "flashvid_plugin_contract_estimate_with_vllm_api_multimodal_audit"
                ),
            }
        ],
    }


def _matrix(count: int = 2) -> dict[str, list[dict]]:
    return {
        dataset: [_record(dataset, index) for index in range(count)]
        for dataset in DATASETS
    }


def test_raw_trajectory_gate_accepts_complete_clean_matrix() -> None:
    report = validate_raw_trajectories(
        _matrix(), expected_per_dataset=2, maximum_failure_rate=0.01
    )
    assert report["status"] == "passed"
    assert report["counts"]["valid_records"] == 6
    assert report["counts"]["failed_records"] == 0
    assert report["counts"]["duplicate_trajectory_id_count"] == 0


def test_raw_trajectory_gate_counts_multi_error_row_once_and_blocks_over_one_percent() -> None:
    matrix = _matrix(count=33)
    broken = matrix["lvbench"][0]
    broken.update(
        {
            "api_error": "timeout",
            "parse_error": "invalid tool call",
            "annotation_leak_check": "failed",
            "trajectory_valid": False,
        }
    )
    report = validate_raw_trajectories(
        matrix, expected_per_dataset=33, maximum_failure_rate=0.01
    )
    assert report["status"] == "failed"
    assert report["counts"]["failed_records"] == 1
    assert report["constraints"]["engineering_failure_rate"]["actual"] == 1 / 99
    assert not report["constraints"]["engineering_failure_rate"]["passed"]
    assert report["failure_breakdown"]["api_error"] == 1
    assert report["failure_breakdown"]["parse_error"] == 1
    assert report["constraints"]["annotation_leak_failures"]["actual"] == 1


def test_raw_trajectory_gate_blocks_incomplete_dataset_and_duplicate_id() -> None:
    matrix = _matrix()
    matrix["cgbench"].pop()
    matrix["lsdbench"][0]["trajectory_id"] = matrix["lvbench"][0]["trajectory_id"]
    report = validate_raw_trajectories(
        matrix, expected_per_dataset=2, maximum_failure_rate=1.0
    )
    assert report["status"] == "failed"
    assert not report["constraints"]["dataset_record_counts"]["passed"]
    assert report["counts"]["duplicate_trajectory_id_count"] == 1


def test_raw_trajectory_gate_blocks_missing_api_multimodal_audit() -> None:
    matrix = _matrix()
    matrix["lvbench"][0]["tool_steps"][0][
        "api_multimodal_video_tokens_actual"
    ] = None
    report = validate_raw_trajectories(
        matrix, expected_per_dataset=2, maximum_failure_rate=1.0
    )
    assert report["status"] == "failed"
    assert report["failure_breakdown"]["missing_api_multimodal_measurement"] == 1

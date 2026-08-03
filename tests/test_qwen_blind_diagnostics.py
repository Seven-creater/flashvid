from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_qwen_blind_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("summarize_qwen_blind_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_blind_summary_exposes_position_text_and_media_controls() -> None:
    rows = [
        {
            "dataset": "demo",
            "model": "Qwen3.5-9B",
            "baseline_mode": "question_choices",
            "protocol_id": "no_think_v1",
            "sample_id": "1",
            "answer": "A",
            "prediction": "A",
            "displayed_prediction": "B",
            "correct": True,
            "choices": {"A": "opens door", "B": "sits"},
            "media_items": 0,
        },
        {
            "dataset": "demo",
            "model": "Qwen3.5-9B",
            "baseline_mode": "question_choices",
            "protocol_id": "no_think_v1",
            "sample_id": "2",
            "answer": "B",
            "prediction": None,
            "displayed_prediction": None,
            "correct": False,
            "choices": {"A": "walks", "B": "closes the window"},
            "media_items": 0,
            "parse_error": "strict_json_answer_missing",
        },
    ]
    report = module.summarize_run(rows)
    assert report["accuracy"] == 0.5
    assert report["answer_letter_counts"] == {"A": 1, "B": 1}
    assert report["prediction_letter_counts"] == {"A": 1, "NONE": 1}
    assert report["media_items_total"] == 0
    assert report["option_length"]["mean_correct"] is not None
    assert report["top_correct_option_tokens"]

from __future__ import annotations

import hashlib
import re

import pytest

from flashvid_eval.flashvid_hybrid import CONTROLLER_SYSTEM_PROMPT
from flashvid_eval.prompt_registry import (
    get_prompt,
    prompt_ids,
    prompt_manifest,
)


def test_legacy_prompt_is_exact_frozen_copy() -> None:
    spec = get_prompt("legacy_v1")
    assert spec.text == CONTROLLER_SYSTEM_PROMPT
    assert spec.sha256 == hashlib.sha256(spec.text.encode("utf-8")).hexdigest()


def test_registry_contains_only_frozen_prompt_ids_and_hashes() -> None:
    assert prompt_ids() == (
        "legacy_v1",
        "budget_rubric_v1",
        "budget_escalation_v1",
    )
    assert prompt_manifest() == {
        prompt_id: get_prompt(prompt_id).sha256 for prompt_id in prompt_ids()
    }
    with pytest.raises(ValueError, match="unknown prompt_id"):
        get_prompt("latest")


@pytest.mark.parametrize("prompt_id", ("budget_rubric_v1", "budget_escalation_v1"))
def test_new_prompts_keep_symmetric_eva_format_and_candidate_gate(
    prompt_id: str,
) -> None:
    text = get_prompt(prompt_id).text
    assert text.count("<tool_call>") == 4
    assert text.count("</tool_call>") == 4
    for ratio in ("0.10", "0.25", "0.50", "1.00"):
        assert text.count(f'"retention_ratio":{ratio}') == 1
    assert "Treat the Direct candidate as a hypothesis" in text
    assert "Change it only after visible evidence contradicts it" in text
    assert "Answer: X" in text
    assert re.search(r'"retention_ratio":0\.5(?=[,}])', text) is None
    assert "On the first turn, request visual evidence" in text
    assert "until at least one <tool_response> has been received" in text


def test_rubric_requires_lowest_sufficient_budget_without_midpoint_anchor() -> None:
    text = get_prompt("budget_rubric_v1").text
    assert "choose the lowest ratio sufficient" in text
    assert "Consider all\nfour on every call" in text
    assert "Do not spend a larger budget merely because it may be safer" in text


def test_escalation_prompt_freezes_start_and_upgrade_rules() -> None:
    text = get_prompt("budget_escalation_v1").text
    assert "Start broad or ordinary questions at 10%" in text
    assert "action or\nexplicit-time questions at 25%" in text
    assert "OCR or fine-detail questions at 100%" in text
    assert "Escalate by only\none level" in text
    assert "must not use a lower ratio" in text

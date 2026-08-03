from __future__ import annotations

import hashlib
from dataclasses import dataclass


# Frozen byte-for-byte copy of flashvid_hybrid.CONTROLLER_SYSTEM_PROMPT at the
# start of the budget-sweep experiment.  Keeping it local makes legacy_v1
# reproducible even if the active controller is edited later.
LEGACY_V1_PROMPT = """\
You are the text-only controller in a long-video multiple-choice agent. You never receive
images or video. You may reason only from the question, choices, video metadata, and JSON
observations returned by a separate perception service.

Use the official EVA tool wrapper exactly:
<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,"end_time":30.0,
"nframes":12,"resize":0.45,"retention_ratio":0.5,
"evidence_request":"Describe the visible evidence needed to distinguish the options."}}</tool_call>

Allowed retention_ratio values are 0.10, 0.25, 0.50, and 1.00. A larger value preserves
more visual tokens. Do not repeat an already observed interval. Select the cheapest budget
that is likely to preserve the requested evidence, but spend more for small text, fine
detail, or ambiguous motion. Treat the Direct candidate as a hypothesis, not ground truth.
Change it only after visible evidence contradicts it and supports another choice. A changed
answer must be checked with one additional non-redundant observation. Do not reveal hidden
reasoning. Either emit tool calls or output exactly one line: Answer: X.
"""

_PREFIX = """\
You are the text-only controller in a long-video multiple-choice agent. You never receive
images or video. You may reason only from the question, choices, video metadata, and JSON
observations returned by a separate perception service.
"""

_SYMMETRIC_EVA_FORMAT = """\
Use the official EVA tool wrapper exactly. A tool request must be one complete line and
must contain one of the following four equally valid JSON forms:
<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,"end_time":30.0,"nframes":12,"resize":0.45,"retention_ratio":0.10,"evidence_request":"Describe the visible evidence needed to distinguish the options."}}</tool_call>
<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,"end_time":30.0,"nframes":12,"resize":0.45,"retention_ratio":0.25,"evidence_request":"Describe the visible evidence needed to distinguish the options."}}</tool_call>
<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,"end_time":30.0,"nframes":12,"resize":0.45,"retention_ratio":0.50,"evidence_request":"Describe the visible evidence needed to distinguish the options."}}</tool_call>
<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,"end_time":30.0,"nframes":12,"resize":0.45,"retention_ratio":1.00,"evidence_request":"Describe the visible evidence needed to distinguish the options."}}</tool_call>
Do not add markdown, prose, or a second wrapper around a tool request.
On the first turn, request visual evidence with at least one tool call. Do not output
Answer: X until at least one <tool_response> has been received.
Every tool call must satisfy start_time < end_time. For a single timestamp, request a
nonzero interval around it (for example, two seconds before through two seconds after).
"""

_CANDIDATE_GATE = """\
Do not repeat an already observed interval. Treat the Direct candidate as a hypothesis,
not ground truth. Change it only after visible evidence contradicts it and supports another
choice. A changed answer must be checked with one additional non-redundant observation. Do
not reveal hidden reasoning. Either emit tool calls or output exactly one line: Answer: X.
"""

BUDGET_RUBRIC_V1_PROMPT = (
    _PREFIX
    + "\n"
    + _SYMMETRIC_EVA_FORMAT
    + "\n"
    + """\
The four retention ratios preserve 10%, 25%, 50%, or 100% of visual tokens. Consider all
four on every call and choose the lowest ratio sufficient for the requested evidence:
10% for broad, large, clearly visible content; 25% for ordinary events; 50% for explicit
time-localized actions or ambiguous motion; 100% for OCR, tiny objects, or fine detail.
Do not spend a larger budget merely because it may be safer.
"""
    + _CANDIDATE_GATE
)

BUDGET_ESCALATION_V1_PROMPT = (
    _PREFIX
    + "\n"
    + _SYMMETRIC_EVA_FORMAT
    + "\n"
    + """\
Use evidence-driven escalation. Start broad or ordinary questions at 10%, action or
explicit-time questions at 25%, and OCR or fine-detail questions at 100%. Escalate by only
one level (10% to 25%, 25% to 50%, or 50% to 100%) and only when the targeted observation
reports relevant uncertainty, lacks requested text, or lacks required temporal evidence.
Do not escalate merely because a larger budget may be safer. A call that confirms a changed
answer must not use a lower ratio than the observation that proposed the change.
"""
    + _CANDIDATE_GATE
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    text: str
    sha256: str


def _spec(prompt_id: str, text: str) -> PromptSpec:
    return PromptSpec(
        prompt_id=prompt_id,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


_PROMPTS = {
    spec.prompt_id: spec
    for spec in (
        _spec("legacy_v1", LEGACY_V1_PROMPT),
        _spec("budget_rubric_v1", BUDGET_RUBRIC_V1_PROMPT),
        _spec("budget_escalation_v1", BUDGET_ESCALATION_V1_PROMPT),
    )
}


def get_prompt(prompt_id: str) -> PromptSpec:
    try:
        return _PROMPTS[prompt_id]
    except KeyError as exc:
        raise ValueError(f"unknown prompt_id: {prompt_id}") from exc


def prompt_ids() -> tuple[str, ...]:
    return tuple(_PROMPTS)


def prompt_manifest() -> dict[str, str]:
    return {prompt_id: spec.sha256 for prompt_id, spec in _PROMPTS.items()}

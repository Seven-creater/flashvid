from __future__ import annotations

import re
from collections.abc import Iterable


def extract_answer_letter(text: str, valid_letters: Iterable[str]) -> str | None:
    """Extract the last explicit answer letter from an MLLM response.

    The parser prefers an explicit ``Answer: X`` marker, then accepts a lone
    option letter or a parenthesized letter. It never returns a letter outside
    the options supplied by the dataset adapter.
    """

    valid = {str(letter).upper() for letter in valid_letters}
    if not valid:
        return None
    text = text or ""
    explicit = re.findall(r"(?i)\banswer\s*(?:is|:|=)\s*\(?([A-H])\)?", text)
    for candidate in reversed(explicit):
        candidate = candidate.upper()
        if candidate in valid:
            return candidate

    marked = re.findall(r"(?<![A-Z])\(([A-H])\)(?![A-Z])|(?<![A-Z])\b([A-H])\b", text.upper())
    for first, second in reversed(marked):
        candidate = first or second
        if candidate in valid:
            return candidate
    return None


def extract_strict_answer_letter(text: str, valid_letters: Iterable[str]) -> str | None:
    """Extract a final answer only from the last non-empty response line.

    This is intentionally stricter than ``extract_answer_letter`` and only
    accepts either ``Answer: X`` on the last line or a lone option letter.
    """

    valid = {str(letter).upper() for letter in valid_letters}
    if not valid:
        return None
    lines = [line.strip().strip("`") for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    final_line = lines[-1]
    explicit = re.fullmatch(r"(?i)answer\s*:\s*\(?([A-H])\)?\s*[.!?]?\s*", final_line)
    if explicit:
        candidate = explicit.group(1).upper()
        return candidate if candidate in valid else None
    lone = re.fullmatch(r"\(?([A-H])\)?", final_line.upper())
    if lone:
        candidate = lone.group(1).upper()
        return candidate if candidate in valid else None
    return None

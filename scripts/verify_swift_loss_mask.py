#!/usr/bin/env python3
"""Verify ms-swift 4.4.2 role loss masks with the real model template.

This is deliberately a hard gate.  It loads the local Qwen3.5 processor (not
the model weights), encodes a small batch with the exact training template, and
uses one-message-at-a-time marker probes to prove that system/user/tool content
has label -100 while assistant tool/final content has trainable labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence


REQUIRED_SWIFT_VERSION = "4.4.2"
MASKED_ROLES = {"system", "user", "tool"}
TRAINED_ROLES = {"assistant"}
TRAINING_MAX_LENGTH = 16384


class TemplateLike(Protocol):
    template_meta: Any

    def set_mode(self, mode: str) -> None: ...

    def encode(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_int_list(value: Any, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise RuntimeError(f"template output {field!r} is not a one-dimensional int list")
    return value


def _changed_span(base: list[int], probe: list[int]) -> tuple[int, int]:
    prefix = 0
    common = min(len(base), len(probe))
    while prefix < common and base[prefix] == probe[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(base) - prefix
        and suffix < len(probe) - prefix
        and base[-1 - suffix] == probe[-1 - suffix]
    ):
        suffix += 1
    end = len(probe) - suffix
    if end <= prefix:
        raise RuntimeError("marker probe did not change tokenization")
    return prefix, end


def _encode(
    template: TemplateLike,
    record: Mapping[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    payload: dict[str, Any] = {"messages": messages}
    for key in ("images", "videos"):
        if key in record:
            payload[key] = deepcopy(record[key])
    encoded = template.encode(payload)
    if "input_ids" not in encoded or "labels" not in encoded:
        raise RuntimeError("template.encode did not return input_ids and labels")
    input_ids = _to_int_list(encoded["input_ids"], "input_ids")
    labels = _to_int_list(encoded["labels"], "labels")
    if len(input_ids) != len(labels) or not input_ids:
        raise RuntimeError("template returned empty or length-mismatched labels")
    return input_ids, labels


def verify_records_with_template(
    records: Sequence[Mapping[str, Any]],
    template: TemplateLike,
    *,
    sample_count: int,
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    template.set_mode("train")
    checked_records = 0
    masked_probes = 0
    assistant_probes = 0
    probe_details: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        if checked_records >= sample_count:
            break
        source_messages = record.get("messages")
        if not isinstance(source_messages, list) or not source_messages:
            continue
        messages: list[dict[str, Any]] = []
        for source_message in source_messages:
            if not isinstance(source_message, Mapping):
                raise RuntimeError(f"record {record_index} has a non-object message")
            role = str(source_message.get("role") or "")
            content = source_message.get("content")
            if role not in MASKED_ROLES | TRAINED_ROLES or not isinstance(content, str):
                raise RuntimeError(
                    f"record {record_index} has unsupported role/content for loss-mask verification"
                )
            if role == "assistant":
                source_loss = source_message.get("loss")
                if not isinstance(source_loss, bool):
                    raise RuntimeError(
                        f"record {record_index} assistant message lacks boolean loss"
                    )
                normalized = {
                    "role": role,
                    "content": content,
                    "loss": source_loss,
                }
                messages.append(normalized)
            else:
                if "loss" in source_message:
                    raise RuntimeError("loss is only valid on assistant messages")
                messages.append({"role": role, "content": content})

        base_ids, _ = _encode(template, record, messages)
        record_masked = 0
        record_assistant = 0
        for message_index, message in enumerate(messages):
            role = message["role"]
            marker = (
                f"\nSWIFT_LOSS_MASK_PROBE_R{record_index}_M{message_index}_"
                f"{role.upper()}_QXJZ"
            )
            probe_messages = deepcopy(messages)
            probe_messages[message_index]["content"] += marker
            probe_ids, probe_labels = _encode(template, record, probe_messages)
            start, end = _changed_span(base_ids, probe_ids)
            span_labels = probe_labels[start:end]
            should_train = role == "assistant" and message.get("loss") is True
            if not should_train:
                if any(label != -100 for label in span_labels):
                    raise RuntimeError(
                        f"record {record_index} message {message_index} role={role} "
                        "has trainable labels in its marker span"
                    )
                masked_probes += 1
                record_masked += 1
            else:
                if not any(label != -100 for label in span_labels):
                    raise RuntimeError(
                        f"record {record_index} message {message_index} role=assistant "
                        "has no trainable label in its marker span"
                    )
                assistant_probes += 1
                record_assistant += 1
        if record_masked == 0 or record_assistant < 1:
            raise RuntimeError(
                f"record {record_index} did not exercise both masked roles and "
                "at least one explicitly trainable assistant target"
            )
        probe_details.append(
            {
                "record_index": record_index,
                "masked_role_probes": record_masked,
                "assistant_probes": record_assistant,
                "token_count": len(base_ids),
            }
        )
        checked_records += 1

    if checked_records < sample_count:
        raise RuntimeError(
            f"requested {sample_count} real records, only {checked_records} were verifiable"
        )
    return {
        "checked_records": checked_records,
        "masked_role_probes": masked_probes,
        "assistant_loss_probes": assistant_probes,
        "probe_details": probe_details,
    }


def verify_all_record_encodings(
    records: Sequence[Mapping[str, Any]],
    template: TemplateLike,
    *,
    max_length: int = TRAINING_MAX_LENGTH,
) -> dict[str, Any]:
    """Prove every exported row survives the exact training template."""

    if not records:
        raise RuntimeError("SFT data is empty")
    template.set_mode("train")
    lengths: list[int] = []
    trainable_counts: list[int] = []
    for record_index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"record {record_index} has no messages")
        input_ids, labels = _encode(template, record, messages)
        if len(input_ids) > max_length:
            raise RuntimeError(
                f"record {record_index} encoded to {len(input_ids)} tokens, "
                f"above max_length={max_length}"
            )
        trainable = sum(label != -100 for label in labels)
        if trainable == 0:
            raise RuntimeError(f"record {record_index} has no trainable labels")
        lengths.append(len(input_ids))
        trainable_counts.append(trainable)
    ordered = sorted(lengths)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "encoded_records": len(records),
        "maximum_encoded_tokens": max(lengths),
        "p95_encoded_tokens": ordered[p95_index],
        "minimum_trainable_labels": min(trainable_counts),
        "max_length": max_length,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            records.append(value)
    return records


def _load_real_template(model: Path) -> tuple[TemplateLike, str]:
    version = importlib.metadata.version("ms-swift")
    if version != REQUIRED_SWIFT_VERSION:
        raise RuntimeError(
            f"expected ms-swift {REQUIRED_SWIFT_VERSION}, found {version}"
        )
    if not model.is_dir():
        raise FileNotFoundError(f"local model directory not found: {model}")
    from swift import get_processor, get_template

    processor = get_processor(str(model), download_model=False)
    template = get_template(
        processor,
        max_length=TRAINING_MAX_LENGTH,
        loss_scale="default",
        template_backend="swift",
    )
    template_type = str(getattr(template.template_meta, "template_type", "unknown"))
    return template, template_type


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_valid_cached_report(
    output: Path,
    *,
    sft_data: Path,
    model: Path,
    sample_count: int,
) -> dict[str, Any] | None:
    """Return an immutable matching report without repeating media tokenization."""

    if not output.is_file():
        return None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
        encoded_records = sum(
            1 for line in sft_data.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        valid = (
            isinstance(report, dict)
            and report.get("status") == "passed"
            and report.get("ms_swift_version") == REQUIRED_SWIFT_VERSION
            and Path(str(report.get("model") or "")).resolve() == model.resolve()
            and Path(str(report.get("sft_data") or "")).resolve() == sft_data.resolve()
            and report.get("sft_data_sha256") == _sha256(sft_data)
            and report.get("template_type") == "qwen3_5"
            and report.get("template_backend") == "swift"
            and report.get("loss_scale") == "default"
            and int(report.get("checked_records", 0)) >= sample_count
            and int(report.get("masked_role_probes", 0)) >= sample_count
            and int(report.get("assistant_loss_probes", 0)) >= sample_count
            and int(report.get("encoded_records", 0)) == encoded_records
            and 0 < int(report.get("maximum_encoded_tokens", 0)) <= TRAINING_MAX_LENGTH
            and int(report.get("minimum_trainable_labels", 0)) > 0
            and int(report.get("max_length", 0)) == TRAINING_MAX_LENGTH
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return report if valid else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hard-fail unless ms-swift 4.4.2's real Qwen3.5 training template "
            "masks system/user/tool labels and trains assistant labels."
        )
    )
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-valid",
        action="store_true",
        help="reuse an existing report only when all paths, hashes and invariants match",
    )
    args = parser.parse_args(argv)

    try:
        if not args.sft_data.is_file():
            raise FileNotFoundError(args.sft_data)
        if args.reuse_valid:
            cached = load_valid_cached_report(
                args.output,
                sft_data=args.sft_data,
                model=args.model,
                sample_count=args.samples,
            )
            if cached is not None:
                print(json.dumps({"cache_reused": True, **cached}, ensure_ascii=False, indent=2))
                return 0
        template, template_type = _load_real_template(args.model)
        records = _read_jsonl(args.sft_data)
        verification = verify_records_with_template(
            records,
            template,
            sample_count=args.samples,
        )
        encoding_verification = verify_all_record_encodings(records, template)
        report: dict[str, Any] = {
            "status": "passed",
            "ms_swift_version": REQUIRED_SWIFT_VERSION,
            "model": str(args.model.resolve()),
            "template_type": template_type,
            "template_backend": "swift",
            "loss_scale": "default",
            "sft_data": str(args.sft_data.resolve()),
            "sft_data_sha256": _sha256(args.sft_data),
            **verification,
            **encoding_verification,
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "required_ms_swift_version": REQUIRED_SWIFT_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }

    _atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Reject over-length SFT trajectories using the exact ms-swift template."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.verify_swift_loss_mask import REQUIRED_SWIFT_VERSION, _encode


DEFAULT_MAX_LENGTH = 16_384
ENCODING_HEADROOM = 262_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError("SFT data is empty")
    return rows


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, Mapping):
        raise ValueError("SFT row has no metadata object")
    return value


def _validate_trajectory_targets(
    rows: Sequence[Mapping[str, Any]], indexes: Sequence[int], trajectory_id: str
) -> None:
    """Accept either the legacy split episodes or one role-separated episode set."""

    metadata = [_metadata(rows[index]) for index in indexes]
    process_roles = [str(item.get("process_role") or "") for item in metadata]
    if any(process_roles):
        if any(role not in {"planner", "observer"} for role in process_roles):
            raise ValueError(
                f"retained trajectory {trajectory_id!r} has an invalid process role"
            )
        planner = [item for item in metadata if item.get("process_role") == "planner"]
        if len(planner) != 1:
            raise ValueError(
                f"retained trajectory {trajectory_id!r} must contain exactly one "
                "Planner episode"
            )
        if planner[0].get("episode_schema") != "planner_complete_episode_v1":
            raise ValueError(
                f"retained trajectory {trajectory_id!r} has an invalid Planner schema"
            )
        targets = planner[0].get("assistant_target_types")
        if (
            not isinstance(targets, list)
            or not targets
            or any(target not in {"tool", "plan", "stop"} for target in targets)
            or targets.count("tool") < 1
            or targets.count("stop") != 1
            or targets[-1] != "stop"
        ):
            raise ValueError(
                f"retained trajectory {trajectory_id!r} must contain at least one "
                "tool target and end with exactly one stop target"
            )
        for item in metadata:
            if item.get("process_role") != "observer":
                continue
            observer_targets = item.get("assistant_target_types")
            if observer_targets != ["memory"]:
                raise ValueError(
                    f"retained trajectory {trajectory_id!r} has an invalid "
                    "Observer episode"
                )
        return

    target_types = [str(item.get("episode_target_type") or "") for item in metadata]
    if target_types.count("final") != 1 or "tool" not in target_types:
        raise ValueError(
            f"retained trajectory {trajectory_id!r} must contain at least one "
            "tool target and exactly one final target"
        )


def select_length_safe_trajectories(
    rows: Sequence[Mapping[str, Any]],
    encoded_lengths: Sequence[int],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[list[int], dict[str, Any]]:
    """Return retained row indexes and an auditable trajectory-level report."""

    if len(rows) != len(encoded_lengths):
        raise ValueError("rows and encoded_lengths must have equal length")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        trajectory_id = str(_metadata(row).get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError(f"row {index} has no metadata.trajectory_id")
        groups.setdefault(trajectory_id, []).append(index)

    retained: list[int] = []
    rejected: list[dict[str, Any]] = []
    for trajectory_id, indexes in groups.items():
        over_length = [index for index in indexes if encoded_lengths[index] > max_length]
        if over_length:
            first = _metadata(rows[indexes[0]])
            rejected.append(
                {
                    "trajectory_id": trajectory_id,
                    "dataset": first.get("dataset"),
                    "sample_id": first.get("sample_id"),
                    "row_indexes": indexes,
                    "maximum_encoded_tokens": max(encoded_lengths[index] for index in indexes),
                    "over_length_episodes": [
                        {
                            "row_index": index,
                            "episode_id": _metadata(rows[index]).get("episode_id"),
                            "encoded_tokens": encoded_lengths[index],
                        }
                        for index in over_length
                    ],
                }
            )
            continue

        _validate_trajectory_targets(rows, indexes, trajectory_id)
        retained.extend(indexes)

    retained.sort()
    if not retained:
        raise ValueError("every SFT trajectory exceeded max_length")
    return retained, {
        "input_rows": len(rows),
        "input_trajectories": len(groups),
        "retained_rows": len(retained),
        "retained_trajectories": len(groups) - len(rejected),
        "rejected_rows": len(rows) - len(retained),
        "rejected_trajectories": len(rejected),
        "maximum_input_tokens": max(encoded_lengths),
        "maximum_retained_tokens": max(encoded_lengths[index] for index in retained),
        "rejections": rejected,
    }


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


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


def _cached_report(
    audit: Path,
    output: Path,
    *,
    source_sha256: str,
    model: Path,
    model_artifact_sha256: str,
    max_length: int,
) -> dict[str, Any] | None:
    if not audit.is_file() or not output.is_file():
        return None
    try:
        value = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "status": "passed",
        "source_sha256": source_sha256,
        "model": str(model.resolve()),
        "model_artifact_sha256": model_artifact_sha256,
        "max_length": max_length,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    if value.get("output_sha256") != _sha256(output):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args(argv)

    source_sha256 = _sha256(args.input)
    cached = _cached_report(
        args.audit,
        args.output,
        source_sha256=source_sha256,
        model=args.model,
        model_artifact_sha256=args.model_artifact_sha256,
        max_length=args.max_length,
    )
    if cached is not None:
        print(json.dumps(cached, ensure_ascii=False, indent=2))
        return 0

    try:
        version = importlib.metadata.version("ms-swift")
        if version != REQUIRED_SWIFT_VERSION:
            raise RuntimeError(
                f"expected ms-swift {REQUIRED_SWIFT_VERSION}, found {version}"
            )
        if not args.model.is_dir():
            raise FileNotFoundError(args.model)
        from swift import get_processor, get_template

        processor = get_processor(str(args.model), download_model=False)
        template = get_template(
            processor,
            max_length=max(args.max_length, ENCODING_HEADROOM),
            loss_scale="default",
            template_backend="swift",
        )
        template.set_mode("train")
        rows = _read_jsonl(args.input)
        lengths: list[int] = []
        for index, row in enumerate(rows):
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"row {index} has no messages")
            input_ids, labels = _encode(template, row, messages)
            if not any(label != -100 for label in labels):
                raise ValueError(f"row {index} has no trainable labels")
            lengths.append(len(input_ids))
        retained, selection = select_length_safe_trajectories(
            rows, lengths, max_length=args.max_length
        )
        _atomic_write_jsonl(args.output, [rows[index] for index in retained])
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "source": str(args.input.resolve()),
            "source_sha256": source_sha256,
            "output": str(args.output.resolve()),
            "output_sha256": _sha256(args.output),
            "model": str(args.model.resolve()),
            "model_artifact_sha256": args.model_artifact_sha256,
            "ms_swift_version": version,
            "template_type": str(
                getattr(template.template_meta, "template_type", "unknown")
            ),
            "max_length": args.max_length,
            **selection,
        }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "source": str(args.input.resolve()),
            "source_sha256": source_sha256,
            "model": str(args.model.resolve()),
            "model_artifact_sha256": args.model_artifact_sha256,
            "max_length": args.max_length,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _atomic_write_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

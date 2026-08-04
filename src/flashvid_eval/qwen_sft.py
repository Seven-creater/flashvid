from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname


SCHEMA_VERSION = 2
TRAIN_COUNTS = {"lvbench": 200, "lsdbench": 200, "cgbench": 200}
ALLOWED_TARGETS = frozenset({"plan", "tool", "memory", "stop", "final"})
HIDDEN_REASONING_KEYS = frozenset(
    {
        "reasoning_content",
        "reasoning",
        "analysis",
        "chain_of_thought",
        "cot",
        "thinking",
        "thought",
        "thoughts",
        "reflection",
        "rationale",
    }
)
PRIVATE_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "right_answer",
        "ground_truth",
        "gt",
        "time_range",
        "clue_intervals",
        "question_type",
    }
)
PRIVATE_SENTINELS = ("ANNOTATION_SENTINEL", "GROUND_TRUTH_SENTINEL")
HIDDEN_TEXT_MARKERS = ("<think>", "</think>", "<analysis>", "</analysis>")
_ERROR_FIELDS = ("error", "error_type", "api_error", "frame_error", "parse_error")
_PARSER_MARKERS = ("parse", "invalid_answer", "invalid_json", "malformed")
_MESSAGE_OPTIONAL_FIELDS = ("name", "tool_call_id", "tool_calls", "function_call")
_TOOL_CALL_RE = re.compile(r"<tool_call>.+?</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class TrainingManifest:
    answers: dict[tuple[str, str], str]
    sha256: str
    counts: dict[str, int]


@dataclass(frozen=True)
class StableSelection:
    selected: tuple[dict[str, Any], ...]
    stable_candidates: tuple[dict[str, Any], ...]
    no_stable_sample_ids: tuple[str, ...]
    rejected_families: dict[str, int]
    stable_family_count: int


@dataclass(frozen=True)
class ExpectedTrajectoryProvenance:
    model: str
    model_artifact_sha256: str
    dataset_manifest_sha256s: frozenset[str]
    agent_config_sha256s: frozenset[str]
    runner_fingerprints: frozenset[str]

    def __post_init__(self) -> None:
        if self.model != "Qwen3.5-9B":
            raise ValueError("Qwen trajectory SFT provenance requires Qwen3.5-9B")
        _require_sha256(self.model_artifact_sha256, "model_artifact_sha256")
        if (
            not self.dataset_manifest_sha256s
            or not self.agent_config_sha256s
            or not self.runner_fingerprints
        ):
            raise ValueError("expected manifest, Agent, and runner fingerprint sets cannot be empty")
        for value in self.dataset_manifest_sha256s:
            _require_sha256(value, "dataset_manifest_sha256")
        for value in self.agent_config_sha256s:
            _require_sha256(value, "agent_config_sha256")
        for value in self.runner_fingerprints:
            _require_sha256(value, "runner_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_artifact_sha256": self.model_artifact_sha256,
            "dataset_manifest_sha256s": sorted(self.dataset_manifest_sha256s),
            "agent_config_sha256s": sorted(self.agent_config_sha256s),
            "runner_fingerprints": sorted(self.runner_fingerprints),
        }


def _require_sha256(value: Any, field: str) -> str:
    raw = str(value or "").strip()
    normalized = raw.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ) or raw != normalized:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_trajectory_input_bundle(
    path: Path, config_sha256: str
) -> tuple[list[Path], ExpectedTrajectoryProvenance]:
    """Load an immutable trajectory bundle and verify every referenced byte."""

    bundle = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("input bundle must be a schema-v1 object")
    claimed = str(bundle.get("bundle_sha256") or "")
    computed = canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    if claimed != computed:
        raise RuntimeError("input bundle self-hash mismatch")
    if bundle.get("config_sha256") != config_sha256:
        raise RuntimeError("input bundle experiment config mismatch")
    for field in ("config", "trajectory_run_plan"):
        reference = bundle.get(field)
        if not isinstance(reference, Mapping):
            raise ValueError(f"input bundle has no {field} reference")
        source_path = Path(str(reference.get("path") or ""))
        if not source_path.is_file() or sha256_file(source_path) != reference.get(
            "sha256"
        ):
            raise RuntimeError(f"bundled {field} changed: {source_path}")

    trajectory_paths: list[Path] = []
    for item in bundle.get("trajectory_files") or []:
        if not isinstance(item, Mapping):
            raise ValueError("bundled trajectory reference is not an object")
        trajectory_path = Path(str(item.get("path") or ""))
        if not trajectory_path.is_file() or sha256_file(trajectory_path) != item.get(
            "sha256"
        ):
            raise RuntimeError(f"bundled trajectory file changed: {trajectory_path}")
        trajectory_paths.append(trajectory_path)
    provenance = bundle.get("expected_provenance")
    if not trajectory_paths or not isinstance(provenance, Mapping):
        raise ValueError("input bundle has no trajectories/provenance")
    return trajectory_paths, ExpectedTrajectoryProvenance(
        model=str(provenance.get("model") or ""),
        model_artifact_sha256=str(provenance.get("model_artifact_sha256") or ""),
        dataset_manifest_sha256s=frozenset(
            provenance.get("dataset_manifest_sha256s") or []
        ),
        agent_config_sha256s=frozenset(
            provenance.get("agent_config_sha256s") or []
        ),
        runner_fingerprints=frozenset(provenance.get("runner_fingerprints") or []),
    )


def composite_trajectory_id(
    dataset: str,
    sample_id: str,
    family_id: str,
    replica_id: str | int,
) -> str:
    values = (dataset, sample_id, family_id, str(replica_id))
    if any(not str(value).strip() for value in values):
        raise ValueError("trajectory identity components cannot be empty")
    return ":".join(quote(str(value), safe="") for value in values)


def parse_composite_trajectory_id(value: str) -> tuple[str, str, str, str]:
    parts = str(value).split(":")
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            "trajectory_id must be dataset:sample:family:replica with percent encoding"
        )
    return tuple(unquote(part) for part in parts)  # type: ignore[return-value]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            records.append(value)
    return records


def _answer(record: Mapping[str, Any]) -> str:
    value = (
        record.get("answer")
        or record.get("correct_answer")
        or record.get("right_answer")
        or (record.get("reward_model") or {}).get("ground_truth")
    )
    answer = str(value or "").strip().upper()
    if len(answer) != 1 or answer < "A" or answer > "H":
        raise ValueError("manifest row has no valid A-H answer")
    return answer


def load_training_manifest(
    path: str | Path,
    *,
    expected_counts: Mapping[str, int] = TRAIN_COUNTS,
) -> TrainingManifest:
    records = read_jsonl(path)
    answers: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        dataset = str(record.get("dataset") or "").strip().lower()
        sample_id = str(record.get("sample_id") or record.get("id") or "").strip()
        if dataset not in expected_counts or not sample_id:
            raise ValueError(f"manifest row {index} has invalid dataset/sample_id")
        identity = (dataset, sample_id)
        if identity in answers:
            raise ValueError(f"duplicate train sample: {dataset}/{sample_id}")
        answers[identity] = _answer(record)
        counts[dataset] += 1

    expected = dict(expected_counts)
    if dict(counts) != expected:
        raise ValueError(f"expected Train600 counts {expected}, found {dict(counts)}")
    return TrainingManifest(answers, sha256_file(path), dict(counts))


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < (0.0 if not positive else 1e-12):
        raise ValueError(f"{field} must be {'positive' if positive else 'non-negative'}")
    return result


def trajectory_total_tokens(record: Mapping[str, Any]) -> float:
    direct = record.get("total_tokens")
    if direct is not None:
        return _number(direct, "total_tokens")
    request_trace = record.get("request_trace")
    if not isinstance(request_trace, list) or not request_trace:
        raise ValueError("trajectory requires total_tokens or request_trace usage")
    total = 0.0
    for index, request in enumerate(request_trace):
        if not isinstance(request, Mapping):
            raise ValueError(f"request_trace[{index}] must be an object")
        usage = request.get("usage", request)
        if not isinstance(usage, Mapping) or usage.get("total_tokens") is None:
            raise ValueError(f"request_trace[{index}] has no total_tokens")
        total += _number(usage["total_tokens"], f"request_trace[{index}].total_tokens")
    return total


def trajectory_visual_tokens(record: Mapping[str, Any]) -> float:
    direct = record.get("visual_tokens")
    if direct is not None:
        return _number(direct, "visual_tokens")
    steps = record.get("tool_steps")
    if not isinstance(steps, list):
        raise ValueError("trajectory requires visual_tokens or tool_steps")
    total = 0.0
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("visual_tokens") is None:
            raise ValueError(f"tool_steps[{index}] has no visual_tokens")
        total += _number(step["visual_tokens"], f"tool_steps[{index}].visual_tokens")
    return total


def validate_tool_steps(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_steps = record.get("tool_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("tool_steps must be a non-empty list")
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool_steps[{index}] must be an object")
        start = _number(raw.get("start_time"), f"tool_steps[{index}].start_time")
        end = _number(raw.get("end_time"), f"tool_steps[{index}].end_time")
        if end <= start:
            raise ValueError(f"tool_steps[{index}] end_time must exceed start_time")
        nframes = raw.get("nframes")
        if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes <= 0:
            raise ValueError(f"tool_steps[{index}].nframes must be positive int")
        resize = _number(raw.get("resize"), f"tool_steps[{index}].resize", positive=True)
        timestamps = raw.get("actual_timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError(f"tool_steps[{index}].actual_timestamps must be non-empty")
        normalized_timestamps = [
            _number(value, f"tool_steps[{index}].actual_timestamps")
            for value in timestamps
        ]
        if len(normalized_timestamps) > nframes:
            raise ValueError(f"tool_steps[{index}] has more timestamps than nframes")
        if any(value < start - 0.1 or value > end + 0.1 for value in normalized_timestamps):
            raise ValueError(f"tool_steps[{index}] timestamp lies outside its interval")
        step = dict(raw)
        step.update(
            {
                "start_time": start,
                "end_time": end,
                "nframes": nframes,
                "resize": resize,
                "actual_timestamps": normalized_timestamps,
            }
        )
        steps.append(step)
    return tuple(steps)


def _walk_forbidden(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace(" ", "_")
            child_path = f"{path}.{key}"
            if normalized in PRIVATE_KEYS:
                return child_path
            found = _walk_forbidden(item, child_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _walk_forbidden(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and any(marker in value for marker in PRIVATE_SENTINELS):
        return path
    return None


def _message_private_path(messages: Any, path: str) -> str | None:
    if not isinstance(messages, list):
        return _walk_forbidden(messages, path)
    for index, message in enumerate(messages):
        item_path = f"{path}[{index}]"
        if not isinstance(message, Mapping):
            return _walk_forbidden(message, item_path)
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and any(
                marker in content for marker in PRIVATE_SENTINELS
            ):
                return f"{item_path}.content"
            continue
        found = _walk_forbidden(message, item_path)
        if found:
            return found
    return None


def _trajectory_private_path(record: Mapping[str, Any]) -> str | None:
    for key in record:
        if str(key).strip().casefold().replace(" ", "_") in PRIVATE_KEYS:
            return f"$.{key}"
    found = _message_private_path(record.get("training_messages"), "$.training_messages")
    if found:
        return found
    found = _walk_forbidden(record.get("tool_steps"), "$.tool_steps")
    if found:
        return found
    request_trace = record.get("request_trace")
    if isinstance(request_trace, list):
        for index, request in enumerate(request_trace):
            if not isinstance(request, Mapping):
                continue
            base = f"$.request_trace[{index}]"
            for key in ("messages", "request_messages"):
                if key in request:
                    found = _message_private_path(request[key], f"{base}.{key}")
                    if found:
                        return found
            for key in ("request", "input", "prompt"):
                if key in request:
                    found = _walk_forbidden(request[key], f"{base}.{key}")
                    if found:
                        return found
    return None


def materialize_trajectory_identity(
    record: Mapping[str, Any],
    *,
    schedule_id: str,
    replica_id: str | int,
    judge_seed: str | int,
    manifest_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    variant_id: str = "base",
) -> dict[str, Any]:
    """Attach generation provenance and normalize AgentTrace tool-step aliases."""

    dataset = str(record.get("dataset") or "").strip().lower()
    sample_id = str(record.get("sample_id") or "").strip()
    if not dataset or not sample_id:
        raise ValueError("AgentTrace requires dataset and sample_id before materialization")
    schedule = str(schedule_id).strip()
    variant = str(variant_id).strip()
    replica = str(replica_id).strip()
    if not schedule or not variant or not replica:
        raise ValueError("schedule, variant, and replica cannot be empty")
    family_id = schedule if variant == "base" else f"{schedule}~{variant}"
    raw_steps = record.get("tool_steps")
    if not isinstance(raw_steps, list):
        raise ValueError("AgentTrace requires tool_steps")
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool_steps[{index}] must be an object")
        request = raw.get("request")
        request = request if isinstance(request, Mapping) else {}
        step = dict(raw)
        aliases = {
            "start_time": raw.get(
                "start_time", raw.get("resolved_start_time", request.get("start_time"))
            ),
            "end_time": raw.get(
                "end_time", raw.get("resolved_end_time", request.get("end_time"))
            ),
            "actual_timestamps": raw.get("actual_timestamps", raw.get("timestamps")),
            "nframes": raw.get(
                "nframes", raw.get("resolved_nframes", request.get("nframes"))
            ),
            "resize": raw.get("resize", request.get("resize")),
        }
        step.update(aliases)
        steps.append(step)
    materialized = dict(record)
    materialized.update(
        {
            "dataset": dataset,
            "sample_id": sample_id,
            "schedule_id": schedule,
            "variant_id": variant,
            "family_id": family_id,
            "replica_id": replica,
            "judge_seed": judge_seed,
            "trajectory_id": composite_trajectory_id(
                dataset, sample_id, family_id, replica
            ),
            "manifest_sha256": manifest_sha256,
            "train600_manifest_sha256": manifest_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "config_sha256": config_sha256,
            "tool_steps": steps,
        }
    )
    return materialized


def _identity(record: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    dataset = str(record.get("dataset") or "").strip().lower()
    sample_id = str(record.get("sample_id") or "").strip()
    schedule_id = str(record.get("schedule_id") or "").strip()
    variant_id = str(record.get("variant_id") or "base").strip()
    family_id = str(record.get("family_id") or "").strip()
    replica_id = str(record.get("replica_id") if record.get("replica_id") is not None else "").strip()
    if not dataset or not sample_id or not schedule_id or not variant_id or not replica_id:
        raise ValueError("trajectory requires dataset/sample_id/schedule_id/variant_id/replica_id")
    expected_family = schedule_id if variant_id == "base" else f"{schedule_id}~{variant_id}"
    if not family_id:
        family_id = expected_family
    if family_id != expected_family:
        raise ValueError("family_id does not match schedule_id and variant_id")
    trajectory_id = str(record.get("trajectory_id") or "")
    expected_id = composite_trajectory_id(dataset, sample_id, family_id, replica_id)
    if trajectory_id != expected_id:
        raise ValueError(f"trajectory_id mismatch: expected {expected_id!r}")
    if parse_composite_trajectory_id(trajectory_id) != (
        dataset,
        sample_id,
        family_id,
        replica_id,
    ):
        raise AssertionError("composite trajectory ID failed round trip")
    return dataset, sample_id, schedule_id, variant_id, family_id, replica_id


def validate_trajectory_provenance(
    record: Mapping[str, Any], expected: ExpectedTrajectoryProvenance
) -> None:
    if record.get("scoring_deferred") is not True:
        raise ValueError("trajectory must have scoring_deferred=true")
    if record.get("model") != expected.model:
        raise ValueError("trajectory model is not the frozen Qwen3.5-9B Teacher")
    if record.get("model_artifact_sha256") != expected.model_artifact_sha256:
        raise ValueError("trajectory model_artifact_sha256 mismatch")
    dataset_manifest_hash = str(record.get("dataset_manifest_sha256") or "")
    if dataset_manifest_hash not in expected.dataset_manifest_sha256s:
        raise ValueError("trajectory dataset_manifest_sha256 is not frozen")
    agent_hash = str(record.get("agent_config_sha256") or "")
    if agent_hash not in expected.agent_config_sha256s:
        raise ValueError("trajectory agent_config_sha256 is not frozen")
    variant_id = str(record.get("variant_id") or "base")
    if variant_id in {"base", "rescue"}:
        runner_field = "trajectory_runner_fingerprint"
    else:
        runner_field = "counterfactual_run_fingerprint"
    runner_hash = str(record.get(runner_field) or "")
    if runner_hash not in expected.runner_fingerprints:
        raise ValueError(f"trajectory {runner_field} is not frozen")


def _validate_trajectory_identity_and_hashes(
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config_sha256: str,
    expected_provenance: ExpectedTrajectoryProvenance | None = None,
) -> None:
    _identity(record)
    if record.get("manifest_sha256") != manifest_sha256:
        raise ValueError("trajectory manifest_sha256 mismatch")
    if record.get("train600_manifest_sha256") != manifest_sha256:
        raise ValueError("trajectory train600_manifest_sha256 mismatch")
    if record.get("config_sha256") != config_sha256:
        raise ValueError("trajectory config_sha256 mismatch")
    if expected_provenance is not None:
        validate_trajectory_provenance(record, expected_provenance)


def _validate_trajectory_accounting(record: Mapping[str, Any]) -> None:
    validate_tool_steps(record)
    total_tokens = trajectory_total_tokens(record)
    trajectory_visual_tokens(record)
    request_trace = record.get("request_trace")
    if record.get("total_tokens") is not None and isinstance(request_trace, list) and request_trace:
        traced_total = 0.0
        for index, request in enumerate(request_trace):
            if not isinstance(request, Mapping):
                raise ValueError(f"request_trace[{index}] must be an object")
            usage = request.get("usage", request)
            if not isinstance(usage, Mapping) or usage.get("total_tokens") is None:
                raise ValueError(f"request_trace[{index}] has no total_tokens")
            traced_total += _number(
                usage["total_tokens"], f"request_trace[{index}].total_tokens"
            )
        if not math.isclose(total_tokens, traced_total, rel_tol=0, abs_tol=1e-6):
            raise ValueError("total_tokens does not match request_trace usage")
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s"):
        if record.get(field) is not None:
            _number(record[field], field)


def validate_trajectory_schema(
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config_sha256: str,
    expected_provenance: ExpectedTrajectoryProvenance | None = None,
) -> None:
    _validate_trajectory_identity_and_hashes(
        record,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        expected_provenance=expected_provenance,
    )
    _validate_trajectory_accounting(record)


def _prediction(record: Mapping[str, Any]) -> str | None:
    value = record.get("final_prediction", record.get("prediction"))
    answer = str(value or "").strip().upper()
    return answer if len(answer) == 1 and "A" <= answer <= "H" else None


def _rejection_reason(record: Mapping[str, Any], correct_answer: str) -> str | None:
    if record.get("annotation_leak_check") != "passed":
        return "annotation_leak"
    if _trajectory_private_path(record):
        return "annotation_leak"
    if record.get("fallback_used") is True or record.get("fallback_to_candidate") is True:
        return "fallback"
    errors = [str(record.get(field) or "") for field in _ERROR_FIELDS]
    failure = str(record.get("failure_stage") or "")
    error_text = " ".join(errors + [failure]).casefold()
    if any(errors) or failure:
        return "parser" if any(marker in error_text for marker in _PARSER_MARKERS) else "infrastructure"
    prediction = _prediction(record)
    if prediction is None:
        return "parser"
    if prediction != correct_answer:
        return "incorrect"
    return None


def trajectory_rejection_reason(
    record: Mapping[str, Any], correct_answer: str
) -> str | None:
    """Return why an unscored trace cannot enter SFT after the offline label join."""

    answer = str(correct_answer).strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "H":
        raise ValueError("correct_answer must be an A-H option")
    return _rejection_reason(record, answer)


def _accounting_rejection_reason(record: Mapping[str, Any]) -> str | None:
    try:
        _validate_trajectory_accounting(record)
    except (TypeError, ValueError) as error:
        message = str(error).casefold()
        if "tool_steps" in message or "timestamp" in message or "nframes" in message:
            return "invalid_tool_trace"
        if "token" in message or "usage" in message:
            return "invalid_token_accounting"
        return "invalid_trajectory"
    return None


def _tool_shape(record: Mapping[str, Any]) -> str:
    steps = validate_tool_steps(record)
    return canonical_sha256(
        [
            {
                "start_time": step["start_time"],
                "end_time": step["end_time"],
                "nframes": step["nframes"],
                "resize": step["resize"],
                "actual_timestamps": step["actual_timestamps"],
            }
            for step in steps
        ]
    )


def _nested_confirmation_reason(
    record: Mapping[str, Any], correct_answer: str, required: int
) -> str | None:
    confirmations = record.get("judge_confirmations")
    if not isinstance(confirmations, list) or len(confirmations) != required:
        return "incomplete_confirmation"
    seeds: set[str] = set()
    for confirmation in confirmations:
        if not isinstance(confirmation, Mapping):
            return "incomplete_confirmation"
        seed = str(confirmation.get("judge_seed") or confirmation.get("seed") or "")
        if not seed or seed in seeds:
            return "incomplete_confirmation"
        seeds.add(seed)
        if confirmation.get("annotation_leak_check") != "passed":
            return "annotation_leak"
        if confirmation.get("fallback_used") is True or confirmation.get(
            "fallback_to_candidate"
        ) is True:
            return "fallback"
        error_text = " ".join(
            [str(confirmation.get(key) or "") for key in _ERROR_FIELDS]
            + [str(confirmation.get("failure_stage") or "")]
        )
        if error_text.strip():
            return "parser" if any(marker in error_text.casefold() for marker in _PARSER_MARKERS) else "infrastructure"
        value = confirmation.get("final_prediction", confirmation.get("prediction"))
        prediction = str(value or "").strip().upper()
        if len(prediction) != 1 or not "A" <= prediction <= "H":
            return "parser"
        if prediction != correct_answer:
            return "unstable"
    return None


def _representative(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], float]:
    costs = [trajectory_total_tokens(row) for row in rows]
    middle = float(median(costs))
    representative = min(
        rows,
        key=lambda row: (
            abs(trajectory_total_tokens(row) - middle),
            trajectory_visual_tokens(row),
            _number(row.get("latency_s", 0), "latency_s"),
            str(row["trajectory_id"]),
        ),
    )
    return dict(representative), middle


def select_stable_correct_trajectories(
    trajectories: Iterable[Mapping[str, Any]],
    manifest: TrainingManifest,
    *,
    config_sha256: str,
    expected_schedules: int = 12,
    required_confirmations: int = 3,
    expected_provenance: ExpectedTrajectoryProvenance | None = None,
) -> StableSelection:
    if expected_schedules <= 0 or required_confirmations <= 0:
        raise ValueError("expected_schedules and required_confirmations must be positive")
    rows = [dict(row) for row in trajectories]
    families: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    base_schedules: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for row in rows:
        _validate_trajectory_identity_and_hashes(
            row,
            manifest_sha256=manifest.sha256,
            config_sha256=config_sha256,
            expected_provenance=expected_provenance,
        )
        dataset, sample_id, schedule_id, variant_id, family_id, _ = _identity(row)
        identity = (dataset, sample_id)
        if identity not in manifest.answers:
            raise ValueError(f"trajectory is not in Train600: {dataset}/{sample_id}")
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in seen_ids:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen_ids.add(trajectory_id)
        families[(dataset, sample_id, family_id)].append(row)
        if variant_id == "base":
            base_schedules[identity].add(schedule_id)

    missing_samples = sorted(set(manifest.answers) - set(base_schedules))
    if missing_samples:
        raise ValueError(f"trajectory matrix misses Train600 samples: {missing_samples[:3]}")
    bad_schedule_counts = {
        f"{dataset}/{sample_id}": len(schedules)
        for (dataset, sample_id), schedules in base_schedules.items()
        if len(schedules) != expected_schedules
    }
    if bad_schedule_counts:
        raise ValueError(
            f"each Train600 sample requires {expected_schedules} base schedules: "
            f"{dict(list(sorted(bad_schedule_counts.items()))[:3])}"
        )

    stable_by_sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for (dataset, sample_id, family_id), family_rows in sorted(families.items()):
        answer = manifest.answers[(dataset, sample_id)]
        reason = None
        for row in family_rows:
            reason = _rejection_reason(row, answer)
            if reason is None:
                reason = _accounting_rejection_reason(row)
            if reason:
                break
        if reason:
            rejected[reason] += 1
            continue
        nested = [row for row in family_rows if isinstance(row.get("judge_confirmations"), list)]
        if nested:
            if len(family_rows) != 1:
                rejected["mixed_confirmation_encoding"] += 1
                continue
            reason = _nested_confirmation_reason(nested[0], answer, required_confirmations)
        else:
            if len(family_rows) != required_confirmations:
                reason = "incomplete_confirmation"
            else:
                replica_ids = {str(row.get("replica_id")) for row in family_rows}
                judge_seeds = {
                    str(row.get("judge_seed") or "") for row in family_rows
                }
                reason = None
                if len(replica_ids) != required_confirmations or "" in judge_seeds or len(judge_seeds) != required_confirmations:
                    reason = "incomplete_confirmation"
                elif len({_tool_shape(row) for row in family_rows}) != 1:
                    reason = "unstable_tool_trace"
        if reason:
            rejected[reason] += 1
            continue
        representative, median_cost = _representative(family_rows)
        representative["_selection_family_id"] = family_id
        representative["_selection_median_total_tokens"] = median_cost
        representative["_selection_confirmation_count"] = required_confirmations
        representative["_selection_stable"] = True
        stable_by_sample[(dataset, sample_id)].append(representative)

    selected: list[dict[str, Any]] = []
    stable_candidates: list[dict[str, Any]] = []
    no_stable: list[str] = []
    for identity in sorted(manifest.answers):
        choices = stable_by_sample.get(identity, [])
        if not choices:
            no_stable.append(f"{identity[0]}/{identity[1]}")
            continue
        stable_candidates.extend(
            sorted(choices, key=lambda row: str(row["_selection_family_id"]))
        )
        selected.append(
            min(
                choices,
                key=lambda row: (
                    row["_selection_median_total_tokens"],
                    trajectory_visual_tokens(row),
                    len(validate_tool_steps(row)),
                    _number(row.get("latency_s", 0), "latency_s"),
                    row["_selection_family_id"],
                ),
            )
        )
    return StableSelection(
        selected=tuple(selected),
        stable_candidates=tuple(stable_candidates),
        no_stable_sample_ids=tuple(no_stable),
        rejected_families=dict(sorted(rejected.items())),
        stable_family_count=sum(len(values) for values in stable_by_sample.values()),
    )


def generate_counterfactual_specs(
    trajectory: Mapping[str, Any],
    *,
    frame_fractions: Sequence[float] = (0.75, 0.5, 0.25),
    replicas: int = 3,
) -> tuple[dict[str, Any], ...]:
    dataset, sample_id, schedule_id, _, _, _ = _identity(trajectory)
    steps = list(validate_tool_steps(trajectory))
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    fractions = tuple(float(value) for value in frame_fractions)
    if any(value <= 0 or value >= 1 for value in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError("frame fractions must be unique values between zero and one")

    variants: list[tuple[str, int, float]] = []
    for prefix_length in range(1, len(steps)):
        variants.append((f"prefix_{prefix_length}", prefix_length, 1.0))
    for prefix_length in range(1, len(steps) + 1):
        for fraction in fractions:
            variants.append(
                (f"prefix_{prefix_length}_frames_{int(round(fraction * 100)):03d}", prefix_length, fraction)
            )

    specs: list[dict[str, Any]] = []
    seen_shapes: set[str] = set()
    for variant_id, prefix_length, fraction in variants:
        planned_calls = []
        for source in steps[:prefix_length]:
            planned_calls.append(
                {
                    "start_time": source["start_time"],
                    "end_time": source["end_time"],
                    "nframes": max(1, int(math.floor(source["nframes"] * fraction + 0.5))),
                    "resize": source["resize"],
                    "source_actual_timestamps": source["actual_timestamps"],
                    **(
                        {"evidence_request": source["evidence_request"]}
                        if source.get("evidence_request") is not None
                        else {}
                    ),
                }
            )
        shape = canonical_sha256(planned_calls)
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        family_id = f"{schedule_id}~{variant_id}"
        spec = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "sample_id": sample_id,
            "schedule_id": schedule_id,
            "variant_id": variant_id,
            "family_id": family_id,
            "base_trajectory_id": trajectory["trajectory_id"],
            "manifest_sha256": trajectory["manifest_sha256"],
            "train600_manifest_sha256": trajectory["train600_manifest_sha256"],
            "dataset_manifest_sha256": trajectory["dataset_manifest_sha256"],
            "config_sha256": trajectory["config_sha256"],
            "prefix_length": prefix_length,
            "frame_fraction": fraction,
            "planned_calls": planned_calls,
            "replica_trajectory_ids": [
                composite_trajectory_id(dataset, sample_id, family_id, replica)
                for replica in range(replicas)
            ],
            "required_judge_confirmations": replicas,
        }
        spec["counterfactual_fingerprint"] = canonical_sha256(spec)
        specs.append(spec)
    return tuple(specs)


def _target_type(message: Mapping[str, Any]) -> str:
    return str(
        message.get("target_type")
        or message.get("training_target")
        or message.get("turn_type")
        or ""
    ).strip().casefold()


def _local_media_path(value: Any, field: str) -> str:
    url = value.get("url") if isinstance(value, Mapping) else value
    if not isinstance(url, str) or not url:
        raise ValueError(f"{field} must contain a file URI")
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"{field} only accepts local file:// URIs")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field} file URI cannot contain query or fragment")
    path = Path(url2pathname(unquote(parsed.path)))
    if not path.is_absolute():
        raise ValueError(f"{field} must resolve to an absolute local path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{field} local file does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} local path is not a file: {resolved}")
    return str(resolved)


def _normalize_sft_content(content: Any) -> tuple[str, list[str], list[str]]:
    if isinstance(content, str):
        if "<image>" in content or "<video>" in content:
            raise ValueError("raw SFT content cannot contain unbound media placeholders")
        return content, [], []
    if not isinstance(content, list):
        raise ValueError("message content must be text or an OpenAI content list")
    parts: list[str] = []
    images: list[str] = []
    videos: list[str] = []
    for index, item in enumerate(content):
        if not isinstance(item, Mapping):
            raise ValueError(f"content[{index}] must be an object")
        item_type = str(item.get("type") or "")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"content[{index}].text must be a string")
            if "<image>" in text or "<video>" in text:
                raise ValueError("source text cannot contain media placeholders")
            parts.append(text)
        elif item_type == "image_url":
            images.append(_local_media_path(item.get("image_url"), "image_url"))
            parts.append("<image>")
        elif item_type == "video_url":
            videos.append(_local_media_path(item.get("video_url"), "video_url"))
            parts.append("<video>")
        else:
            raise ValueError(f"unsupported OpenAI content type: {item_type!r}")
    return "\n".join(parts), images, videos


def _strict_response_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            return None
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _validate_tool_calls(content: str) -> bool:
    matches = list(_TOOL_CALL_RE.finditer(content))
    if not matches:
        return False
    if content.count("<tool_call>") != len(matches) or content.count(
        "</tool_call>"
    ) != len(matches):
        raise ValueError("request_trace response has malformed tool markup")
    for match in matches:
        try:
            payload = json.loads(match.group(0)[len("<tool_call>") : -len("</tool_call>")])
        except json.JSONDecodeError as error:
            raise ValueError("request_trace tool call is not valid JSON") from error
        if not isinstance(payload, Mapping) or payload.get("tool") != "frame_select":
            raise ValueError("request_trace contains a non-frame_select tool call")
        arguments = payload.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("request_trace frame_select arguments must be an object")
        has_nframes = "nframes" in arguments
        has_fps = "fps" in arguments
        if has_nframes == has_fps:
            raise ValueError("frame_select requires exactly one of nframes or fps")
        start = _number(arguments.get("start_time"), "frame_select.start_time")
        end = _number(arguments.get("end_time"), "frame_select.end_time")
        if end <= start:
            raise ValueError("frame_select end_time must exceed start_time")
        if has_nframes:
            nframes = arguments["nframes"]
            if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes <= 0:
                raise ValueError("frame_select nframes must be a positive integer")
        else:
            _number(arguments["fps"], "frame_select.fps", positive=True)
        _number(arguments.get("resize", 1.0), "frame_select.resize", positive=True)
    return True


def _classify_request_target(request: Mapping[str, Any]) -> str:
    content = request.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("request_trace response content must be non-empty text")
    hidden = _find_hidden_reasoning(content, "$.request_trace.content")
    if hidden:
        raise ValueError("request_trace response contains hidden reasoning markup")
    has_tool_marker = "<tool_call>" in content or "</tool_call>" in content
    has_tool = _validate_tool_calls(content) if has_tool_marker else False
    if has_tool_marker and not has_tool:
        raise ValueError("request_trace response has malformed tool markup")

    payload = _strict_response_object(content)
    answer = None
    if payload is not None and set(payload) == {"answer"}:
        value = str(payload["answer"] or "").strip().upper()
        if len(value) == 1 and "A" <= value <= "H":
            answer = value
    is_stop = bool(
        payload
        and len(payload) == 1
        and str(payload.get("action", payload.get("decision", ""))).strip().casefold()
        == "stop"
    )
    active = sum((has_tool, answer is not None, is_stop))
    if active > 1:
        raise ValueError("request_trace response mixes incompatible agent actions")
    if has_tool:
        return "tool"
    if answer is not None:
        return "final"
    if is_stop:
        return "stop"

    request_kind = str(request.get("request_kind") or "").strip().casefold()
    if request_kind == "observer":
        if payload is not None:
            has_intervals = "intervals" in payload
            has_selected_node = "selected_node" in payload
            if has_intervals and has_selected_node:
                raise ValueError(
                    "observer response mixes interval and hierarchy planning actions"
                )
            if has_intervals:
                if not isinstance(payload["intervals"], list):
                    raise ValueError("observer planning intervals must be a list")
                return "plan"
            if has_selected_node:
                selected_node = payload["selected_node"]
                if isinstance(selected_node, bool) or not isinstance(selected_node, int):
                    raise ValueError("observer selected_node must be an integer")
                return "plan"
        return "memory"
    if request_kind == "planner":
        return "plan"
    if request_kind in {"judge", "direct"}:
        raise ValueError(
            f"{request_kind} response has neither strict answer nor tool request"
        )
    raise ValueError(f"unsupported request_kind for SFT: {request_kind!r}")


def _episode_messages(request: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    source = request.get("messages")
    if not isinstance(source, list) or not source:
        raise ValueError("request_trace entry has no message snapshot")
    messages: list[dict[str, Any]] = []
    system_count = 0
    previous_role: str | None = None
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request messages[{index}] must be an object")
        role = str(raw.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"request messages[{index}] has unsupported role {role!r}")
        if role == "system":
            system_count += 1
            if index != 0:
                raise ValueError("system message may only appear first in an SFT episode")
        elif index == 0:
            raise ValueError("SFT episode must begin with exactly one system message")
        if role == "assistant" and previous_role not in {"user", "tool"}:
            raise ValueError("historical assistant message has invalid predecessor")
        if role == "tool" and previous_role != "assistant":
            raise ValueError("historical tool message must follow an assistant message")
        hidden = _find_hidden_reasoning(raw, f"$.request_trace.messages[{index}]")
        if hidden:
            raise ValueError(f"hidden reasoning in request message at {hidden}")
        forbidden = (
            _walk_forbidden(raw)
            if role != "assistant"
            else (
                f"$.request_trace.messages[{index}].content"
                if isinstance(raw.get("content"), str)
                and any(marker in raw["content"] for marker in PRIVATE_SENTINELS)
                else None
            )
        )
        if forbidden:
            raise ValueError(f"private annotation in request message at {forbidden}")
        content = raw.get("content", "")
        if not isinstance(content, (str, list)):
            raise ValueError("request message content must be text or a content list")
        message = {"role": role, "content": deepcopy(content)}
        for key in _MESSAGE_OPTIONAL_FIELDS:
            if key in raw:
                message[key] = deepcopy(raw[key])
        if role == "assistant":
            message["target_type"] = "context"
        messages.append(message)
        previous_role = role
    if system_count != 1:
        raise ValueError("SFT episode must contain exactly one system message")
    if previous_role not in {"user", "tool"}:
        raise ValueError("request snapshot must end in a user or tool message")
    messages.append(
        {
            "role": "assistant",
            "content": str(request["content"]),
            "target_type": target,
        }
    )
    return messages


def _terminal_request_indices(request_trace: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, request in enumerate(request_trace):
        if not isinstance(request, Mapping):
            raise ValueError(f"request_trace[{index}] must be an object")
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request_trace[{index}] has no messages")
        prompt_hash = str(request.get("prompt_hash") or canonical_sha256(messages))
        key = (
            str(request.get("branch") or ""),
            str(request.get("request_kind") or ""),
            prompt_hash,
            str(request.get("seed") if request.get("seed") is not None else ""),
        )
        groups[key].append(index)

    selected: list[int] = []
    for indices in groups.values():
        if len(indices) > 1:
            attempts = [request_trace[index].get("attempt_index") for index in indices]
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in attempts
            ) or len(set(attempts)) != len(attempts):
                raise ValueError("duplicate request prompt lacks unique retry attempt_index")
            chosen = max(indices, key=lambda index: int(request_trace[index]["attempt_index"]))
        else:
            chosen = indices[0]
        if str(request_trace[chosen].get("finish_reason") or "") == "length":
            raise ValueError("terminal request attempt is still truncated")
        selected.append(chosen)
    return tuple(sorted(selected))


def derive_training_episodes(trajectory: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Convert each terminal AgentTrace request into one independent SFT episode."""

    raw_trace = trajectory.get("request_trace")
    if not isinstance(raw_trace, list) or not raw_trace:
        raise ValueError("selected AgentTrace requires a non-empty request_trace")
    request_trace = [dict(request) if isinstance(request, Mapping) else request for request in raw_trace]
    tool_steps = trajectory.get("tool_steps")
    evidence_memory = trajectory.get("evidence_memory")
    if not isinstance(tool_steps, list) or not isinstance(evidence_memory, list):
        raise ValueError("AgentTrace requires tool_steps and evidence_memory lists")

    terminal_indices = _terminal_request_indices(request_trace)
    targets = {
        turn_index: _classify_request_target(request_trace[turn_index])
        for turn_index in terminal_indices
    }
    final_indices = [
        turn_index for turn_index, target in targets.items() if target == "final"
    ]
    if not final_indices:
        raise ValueError("AgentTrace has no strict final-answer training episode")
    final_turn = max(final_indices)
    if final_turn != max(terminal_indices):
        raise ValueError("the terminal model request is not the final answer judge")
    final_payload = _strict_response_object(str(request_trace[final_turn]["content"]))
    final_answer = str((final_payload or {}).get("answer") or "").strip().upper()
    trace_answer = str(
        trajectory.get("final_prediction", trajectory.get("prediction")) or ""
    ).strip().upper()
    if final_answer != trace_answer:
        raise ValueError("final judge answer does not match AgentTrace final_prediction")

    episodes: list[dict[str, Any]] = []
    for turn_index in terminal_indices:
        request = request_trace[turn_index]
        target = targets[turn_index]
        if target == "final" and turn_index != final_turn:
            continue
        branch = str(request.get("branch") or "")
        request_kind = str(request.get("request_kind") or "")
        messages = _episode_messages(request, target)
        evidence_indices = [
            index
            for index, item in enumerate(evidence_memory)
            if isinstance(item, Mapping) and item.get("content") == request.get("content")
        ]
        has_tool_input = any(
            isinstance(message, Mapping) and message.get("role") == "tool"
            for message in request.get("messages", [])
        )
        branch_tool_indices = [
            index
            for index, item in enumerate(tool_steps)
            if isinstance(item, Mapping) and str(item.get("branch") or "") == branch
        ]
        tool_indices = branch_tool_indices if has_tool_input or request_kind == "observer" else []
        prompt_hash = str(
            request.get("prompt_hash") or canonical_sha256(request.get("messages"))
        )
        episode_id = f"{trajectory['trajectory_id']}#turn-{turn_index:03d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "turn_index": turn_index,
                "branch": branch,
                "request_kind": request_kind,
                "prompt_hash": prompt_hash,
                "attempt_index": int(request.get("attempt_index") or 0),
                "finish_reason": request.get("finish_reason"),
                "target_type": target,
                "tool_step_indices": tool_indices,
                "evidence_memory_indices": evidence_indices,
                "messages": messages,
                "usage": deepcopy(request.get("usage") or {}),
            }
        )
    if not episodes:
        raise ValueError("AgentTrace produced no terminal training episodes")
    return tuple(episodes)


def build_sft_record(
    trajectory: Mapping[str, Any],
    *,
    require_complete_trajectory: bool = True,
    episode_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = trajectory.get("training_messages")
    if not isinstance(source, list) or not source:
        raise ValueError("selected trajectory requires training_messages")
    messages: list[dict[str, Any]] = []
    images: list[str] = []
    videos: list[str] = []
    trained_targets: list[str] = []
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ValueError(f"training_messages[{index}] must be an object")
        role = str(raw.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        hidden = _find_hidden_reasoning(raw, f"$.training_messages[{index}]")
        if hidden:
            raise ValueError(f"hidden reasoning field in training messages at {hidden}")
        forbidden = (
            _walk_forbidden(raw)
            if role != "assistant"
            else (
                f"$.training_messages[{index}].content"
                if isinstance(raw.get("content"), str)
                and any(marker in raw["content"] for marker in PRIVATE_SENTINELS)
                else None
            )
        )
        if forbidden:
            raise ValueError(f"private annotation in training message at {forbidden}")
        content, message_images, message_videos = _normalize_sft_content(
            raw.get("content", "")
        )
        images.extend(message_images)
        videos.extend(message_videos)
        message: dict[str, Any] = {"role": role, "content": content}
        for key in _MESSAGE_OPTIONAL_FIELDS:
            if key in raw:
                message[key] = raw[key]
        if role == "assistant":
            target = _target_type(raw)
            if not target:
                raise ValueError("every assistant message requires target_type")
            if target in HIDDEN_REASONING_KEYS:
                raise ValueError("hidden reasoning assistant turns cannot enter SFT")
            trainable = target in ALLOWED_TARGETS
            message["loss"] = trainable
            if trainable:
                trained_targets.append(target)
        messages.append(message)
    if require_complete_trajectory:
        if "final" not in trained_targets:
            raise ValueError("SFT trajectory requires a trainable final assistant turn")
        if not any(target in {"plan", "tool", "memory", "stop"} for target in trained_targets):
            raise ValueError("SFT trajectory requires at least one trainable agent action")
    elif len(trained_targets) != 1:
        raise ValueError("request-level SFT episode must have exactly one trainable target")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": trajectory["dataset"],
        "sample_id": trajectory["sample_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "family_id": trajectory.get("_selection_family_id", trajectory.get("family_id")),
        "manifest_sha256": trajectory["manifest_sha256"],
        "train600_manifest_sha256": trajectory["train600_manifest_sha256"],
        "dataset_manifest_sha256": trajectory["dataset_manifest_sha256"],
        "config_sha256": trajectory["config_sha256"],
        "total_tokens": trajectory_total_tokens(trajectory),
        "visual_tokens": trajectory_visual_tokens(trajectory),
        "assistant_target_types": trained_targets,
    }
    if episode_metadata is not None:
        metadata.update(deepcopy(dict(episode_metadata)))
    record = {"messages": messages, "metadata": metadata}
    if images:
        record["images"] = images
    if videos:
        record["videos"] = videos
    return record


def build_sft_records(trajectory: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    source = trajectory.get("training_messages")
    if isinstance(source, list) and source:
        return (build_sft_record(trajectory),)
    records: list[dict[str, Any]] = []
    for episode in derive_training_episodes(trajectory):
        materialized = dict(trajectory)
        materialized["training_messages"] = episode["messages"]
        metadata = {
            key: episode[key]
            for key in (
                "episode_id",
                "turn_index",
                "branch",
                "request_kind",
                "prompt_hash",
                "attempt_index",
                "finish_reason",
                "target_type",
                "tool_step_indices",
                "evidence_memory_indices",
                "usage",
            )
        }
        metadata["episode_target_type"] = metadata.pop("target_type")
        records.append(
            build_sft_record(
                materialized,
                require_complete_trajectory=False,
                episode_metadata=metadata,
            )
        )
    return tuple(records)


def validate_exported_sft_record(record: Mapping[str, Any]) -> None:
    messages = record.get("messages")
    metadata = record.get("metadata")
    if not isinstance(messages, list) or not messages or not isinstance(metadata, Mapping):
        raise ValueError("SFT record requires messages and metadata")
    forbidden = _message_private_path(messages, "$.messages")
    if not forbidden:
        forbidden = _walk_forbidden(metadata, "$.metadata")
    if forbidden:
        raise ValueError(f"private annotation found in exported SFT at {forbidden}")
    hidden = _find_hidden_reasoning(record)
    if hidden:
        raise ValueError(f"hidden reasoning found in exported SFT at {hidden}")
    targets = metadata.get("assistant_target_types")
    if not isinstance(targets, list) or any(target not in ALLOWED_TARGETS for target in targets):
        raise ValueError("SFT metadata has invalid assistant targets")
    trainable = 0
    image_placeholders = 0
    video_placeholders = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("exported SFT message content must be normalized text")
        image_placeholders += content.count("<image>")
        video_placeholders += content.count("<video>")
        if role == "assistant":
            if not isinstance(message.get("loss"), bool):
                raise ValueError("every exported assistant message requires a bool loss field")
            trainable += int(message["loss"])
        elif "loss" in message:
            raise ValueError("loss may only be attached to assistant messages")
    for key, expected in (
        ("images", image_placeholders),
        ("videos", video_placeholders),
    ):
        media = record.get(key)
        if expected == 0:
            if media is not None:
                raise ValueError(f"pure-text SFT episode must not contain {key}")
            continue
        if not isinstance(media, list) or len(media) != expected:
            raise ValueError(f"{key} count does not match message placeholders")
        for path in media:
            if not isinstance(path, str):
                raise ValueError(f"{key} entries must be strings")
            local = Path(path)
            if not local.is_absolute() or not local.is_file():
                raise ValueError(f"{key} entry is not an existing absolute file: {path}")
    episode_target = metadata.get("episode_target_type")
    if episode_target is not None:
        if episode_target not in ALLOWED_TARGETS or targets != [episode_target]:
            raise ValueError("episode target metadata does not match assistant target")
        if trainable != 1:
            raise ValueError("request-level SFT episode must train exactly one assistant turn")
    elif trainable != len(targets) or "final" not in targets:
        raise ValueError("assistant loss mask does not match recorded targets")


def _find_hidden_reasoning(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace(" ", "_")
            child_path = f"{path}.{key}"
            if normalized in HIDDEN_REASONING_KEYS:
                return child_path
            found = _find_hidden_reasoning(item, child_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_hidden_reasoning(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and any(
        marker in value.casefold() for marker in HIDDEN_TEXT_MARKERS
    ):
        return path
    return None


def source_fingerprint(
    *,
    manifest_sha256: str,
    config_sha256: str,
    trajectory_files: Mapping[str, str],
    phase: str,
    expected_provenance: Mapping[str, Any] | None = None,
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "trajectory_files": dict(sorted(trajectory_files.items())),
            "expected_provenance": dict(expected_provenance or {}),
            "implementation_sha256": sha256_file(Path(__file__)),
        }
    )


def checkpoint_gate(
    teacher: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    def metric(source: Mapping[str, Any], *names: str) -> float:
        aggregate = source.get("aggregate", source)
        if not isinstance(aggregate, Mapping):
            raise ValueError("checkpoint aggregate must be an object")
        for name in names:
            if aggregate.get(name) is not None:
                return _number(aggregate[name], name)
        raise ValueError(f"missing checkpoint metric: {' or '.join(names)}")

    for key in ("manifest_sha256", "sample_ids_sha256"):
        left, right = teacher.get(key), checkpoint.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"teacher/checkpoint {key} mismatch")
    teacher_correct = metric(teacher, "correct")
    checkpoint_correct = metric(checkpoint, "correct")
    teacher_total = metric(teacher, "mean_total_tokens", "total_tokens")
    checkpoint_total = metric(checkpoint, "mean_total_tokens", "total_tokens")
    teacher_visual = metric(teacher, "mean_visual_tokens", "visual_tokens")
    checkpoint_visual = metric(checkpoint, "mean_visual_tokens", "visual_tokens")
    conditions = {
        "accuracy_strictly_higher": checkpoint_correct > teacher_correct,
        "total_tokens_at_most_70pct": teacher_total > 0 and checkpoint_total <= 0.70 * teacher_total,
        "visual_tokens_at_most_70pct": teacher_visual > 0 and checkpoint_visual <= 0.70 * teacher_visual,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "teacher": {
            "correct": teacher_correct,
            "total_tokens": teacher_total,
            "visual_tokens": teacher_visual,
        },
        "checkpoint": {
            "correct": checkpoint_correct,
            "total_tokens": checkpoint_total,
            "visual_tokens": checkpoint_visual,
        },
        "deltas": {
            "correct": checkpoint_correct - teacher_correct,
            "total_token_reduction": 1.0 - checkpoint_total / teacher_total if teacher_total else None,
            "visual_token_reduction": 1.0 - checkpoint_visual / teacher_visual if teacher_visual else None,
        },
    }

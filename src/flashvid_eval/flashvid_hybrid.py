from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .answers import extract_strict_answer_letter
from .budget_strategies import (
    BudgetContext,
    BudgetDecision,
    choose_budget,
    freeze_matched_distribution,
)
from .client import OpenAICompatibleClient
from .datasets import VideoIndex
from .eva_official import select_frames as official_select_frames
from .flashvid_budget import (
    ALLOWED_RETENTION_RATIOS,
    BudgetEndpointPool,
    flashvid_token_counts,
    pack_frames_to_mp4,
    perception_cache_key,
)
from .media import probe_video
from .prompt_registry import get_prompt, prompt_ids
from .runner import (
    _initial_tool_call,
    _question_route,
    _ranges_overlap,
    _ranges_redundant,
    _usage_fields,
)
from .schemas import ModelSample
from .schemas import Sample, ScoringRecord


_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_UNCERTAINTY_RE = re.compile(
    r"(?i)\b(?:unclear|uncertain|insufficient|not visible|cannot tell|could not determine)\b"
)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_PRIVATE_REQUEST_KEYS = {
    "answer",
    "correct_answer",
    "right_answer",
    "ground_truth",
    "time_range",
    "clue_intervals",
    "question_type",
}

_CORE_IMPLEMENTATION_FILES = (
    "scripts/evaluate_mcq.py",
    "src/flashvid_eval/runner.py",
    "src/flashvid_eval/flashvid_hybrid.py",
    "src/flashvid_eval/budget_strategies.py",
    "src/flashvid_eval/prompt_registry.py",
)


def core_implementation_sha256() -> dict[str, str]:
    """Hash the model-facing implementation that defines a Hybrid run.

    These hashes are part of the resume fingerprint.  A new process therefore
    refuses to append to results produced by different evaluator code, even
    when the prompts and runtime configuration are unchanged.
    """

    project_root = Path(__file__).resolve().parents[2]
    hashes: dict[str, str] = {}
    for relative in _CORE_IMPLEMENTATION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise RuntimeError(f"core implementation file not found: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


# Public compatibility alias for the frozen pre-sweep prompt.  New experiments
# resolve their prompt by ID; legacy_v1 remains byte-for-byte unchanged.
CONTROLLER_SYSTEM_PROMPT = get_prompt("legacy_v1").text


PERCEPTION_SYSTEM_PROMPT = """\
You are a video perception service, not the answering agent. Inspect only the supplied
short clip. Report visible facts without guessing, using compact JSON with exactly these
keys: observed_facts, visible_text, temporal_changes, option_evidence, uncertainties.
The first three and uncertainties are arrays of short strings. option_evidence is an object
whose values are arrays of short strings. Use at most 4 items in each top-level array and
at most 2 items per option; keep every item under 25 words. Empty arrays are valid. Never
choose a final answer and never invent events outside the clip. Return one complete JSON
object with no markdown or commentary.
"""

# Some Qwen3.5 responses remain verbose even under JSON mode.  A 2k ceiling
# prevents syntactically truncated objects; the parser below still enforces a
# compact bounded observation before anything reaches the controller or SFT.
PERCEPTION_MAX_TOKENS = 2048


def _short_string_array_schema(max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "maxLength": 160},
        "maxItems": max_items,
    }


PERCEPTION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "flashvid_perception_observation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "observed_facts": _short_string_array_schema(4),
                "visible_text": _short_string_array_schema(4),
                "temporal_changes": _short_string_array_schema(4),
                "option_evidence": {
                    "type": "object",
                    "properties": {
                        letter: _short_string_array_schema(2)
                        for letter in "ABCDEFGH"
                    },
                    "additionalProperties": False,
                },
                "uncertainties": _short_string_array_schema(4),
            },
            "required": [
                "observed_facts",
                "visible_text",
                "temporal_changes",
                "option_evidence",
                "uncertainties",
            ],
            "additionalProperties": False,
        },
    },
}

_PERCEPTION_REQUEST_GATE_VERSION = "candidate_blind_v1"
_EVA_SELECTOR_CACHE_VERSION = "official_select_frame_fallback_v1"


@dataclass(frozen=True)
class FlashVIDHybridConfig:
    budget_policy: str = "model"
    fixed_retention_ratio: float = 0.50
    max_turns: int = 6
    max_perception_calls: int = 5
    controller_temperature: float = 0.0
    trajectory_index: int = 0
    prompt_version: str = "flashvid_budget_v1"
    budget_strategy: str | None = None
    budget_random_seed: int | None = None
    budget_match_distribution: dict[str, float] | None = None
    controller_prompt_id: str = "legacy_v1"

    def __post_init__(self) -> None:
        if self.budget_policy not in {"fixed", "random", "model"}:
            raise ValueError(f"unsupported budget policy: {self.budget_policy}")
        if self.controller_prompt_id not in prompt_ids():
            raise ValueError(
                f"unsupported controller prompt: {self.controller_prompt_id}"
            )
        if self.budget_strategy == "random_matched":
            if self.budget_match_distribution is None:
                raise ValueError(
                    "random_matched requires budget_match_distribution"
                )
            frozen_distribution = freeze_matched_distribution(
                self.budget_match_distribution
            )
            object.__setattr__(
                self,
                "budget_match_distribution",
                {
                    f"{ratio:.2f}": probability
                    for ratio, probability in frozen_distribution.as_mapping().items()
                },
            )
            choose_budget(
                "random_matched",
                BudgetContext(
                    route="global_overview",
                    step_index=0,
                    planned_call={
                        "evidence_request": "validation",
                        "retention_ratio": 0.50,
                    },
                ),
                random_seed=self.budget_random_seed,
                matched_distribution=frozen_distribution,
                sample_key="validation",
            )
        elif self.budget_strategy is not None:
            if self.budget_match_distribution is not None:
                raise ValueError(
                    "budget_match_distribution is only valid for random_matched"
                )
            # Validate the strategy and seed without relying on model-facing data.
            choose_budget(
                self.budget_strategy,
                BudgetContext(
                    route="global_overview",
                    step_index=0,
                    planned_call={
                        "evidence_request": "validation",
                        "retention_ratio": 0.50,
                    },
                ),
                random_seed=self.budget_random_seed,
            )
        elif self.budget_random_seed is not None:
            raise ValueError("budget_random_seed requires budget_strategy")
        elif self.budget_match_distribution is not None:
            raise ValueError(
                "budget_match_distribution requires random_matched"
            )
        if float(self.fixed_retention_ratio) not in ALLOWED_RETENTION_RATIOS:
            raise ValueError(
                f"fixed_retention_ratio must be one of {ALLOWED_RETENTION_RATIOS}"
            )
        if not 1 <= int(self.max_turns) <= 12:
            raise ValueError("max_turns must be between 1 and 12")
        if not 1 <= int(self.max_perception_calls) <= 10:
            raise ValueError("max_perception_calls must be between 1 and 10")
        if float(self.controller_temperature) < 0:
            raise ValueError("controller_temperature cannot be negative")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def controller_prompt_hash(prompt_id: str = "legacy_v1") -> str:
    return prompt_hash(
        get_prompt(prompt_id).text
        + "\n"
        + inspect.getsource(build_controller_user_prompt)
    )


def perception_prompt_hash() -> str:
    return prompt_hash(
        PERCEPTION_SYSTEM_PROMPT
        + "\n"
        + inspect.getsource(build_perception_text)
        + "\n"
        + inspect.getsource(candidate_blind_evidence_request)
        + "\n"
        + json.dumps(
            PERCEPTION_RESPONSE_FORMAT,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + f"\nmax_tokens={PERCEPTION_MAX_TOKENS}"
    )


def _format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"{letter}: {text}" for letter, text in choices.items())


def build_controller_user_prompt(
    sample: ModelSample,
    metadata: dict[str, float | int],
    route: str,
) -> str:
    candidate = sample.candidate_answer or "NONE"
    return (
        f"Video duration: {float(metadata['duration']):.3f} seconds; "
        f"resolution: {int(metadata['width'])}x{int(metadata['height'])}.\n"
        f"Question-derived route: {route}.\n"
        f"Direct candidate hypothesis: {candidate}.\n"
        "The candidate is only a fallback hypothesis. Find visual evidence before accepting "
        "or changing it.\n\n"
        f"Question: {sample.question}\nChoices:\n{_format_choices(sample.choices)}"
    )


def build_perception_text(
    question: str,
    choices: dict[str, str],
    evidence_request: str,
    timestamps: Iterable[float],
) -> str:
    # This function deliberately has no candidate or annotation argument.
    timestamp_text = ", ".join(f"{float(value):.3f}" for value in timestamps)
    return (
        f"Original-frame timestamps in clip order (seconds): [{timestamp_text}]\n"
        f"Evidence request: {evidence_request}\n"
        f"Question: {question}\nChoices:\n{_format_choices(choices)}\n"
        "Return JSON only."
    )


def candidate_blind_evidence_request(route: str) -> str:
    """Return a trusted request that cannot encode a Direct-candidate hint."""

    if route == "ocr_detail":
        return (
            "Transcribe all readable text and describe the fine visual details relevant "
            "to distinguishing every option."
        )
    if route in {"action_event", "temporal_event"}:
        return (
            "Describe the visible actions, state changes, and their temporal order. "
            "Report evidence for and against every option without choosing an answer."
        )
    if route == "explicit_question_time":
        return (
            "Describe all visible facts and changes in this question-specified time "
            "window that distinguish the options."
        )
    return (
        "Describe all visible facts, entities, actions, text, and changes relevant to "
        "distinguishing every option, without choosing an answer."
    )


def perception_request_context_hash(
    sample: ModelSample,
    video: Path,
    evidence_request: str,
) -> str:
    stat = video.stat()
    payload = {
        "gate_version": _PERCEPTION_REQUEST_GATE_VERSION,
        "selector_version": _EVA_SELECTOR_CACHE_VERSION,
        "prompt_sha256": perception_prompt_hash(),
        "dataset": sample.dataset,
        "question": sample.question,
        "choices": sample.choices,
        "evidence_request": evidence_request,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_perception_messages(
    media_path: Path,
    question: str,
    choices: dict[str, str],
    evidence_request: str,
    timestamps: Iterable[float],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": media_path.resolve().as_uri()}},
                {
                    "type": "text",
                    "text": build_perception_text(
                        question,
                        choices,
                        evidence_request,
                        timestamps,
                    ),
                },
            ],
        },
    ]


def parse_budget_tool_calls(
    text: str,
    *,
    require_retention_ratio: bool = False,
) -> list[dict[str, Any]]:
    """Accept only official wrapped frame_select calls with the extended fields."""

    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    for block in _TOOL_BLOCK_RE.findall(text or ""):
        cursor = 0
        while cursor < len(block):
            start = block.find("{", cursor)
            if start < 0:
                break
            try:
                payload, consumed = decoder.raw_decode(block[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            cursor = start + consumed
            if not isinstance(payload, dict) or payload.get("tool") != "frame_select":
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                continue
            if require_retention_ratio and "retention_ratio" not in arguments:
                continue
            try:
                ratio = float(arguments.get("retention_ratio", 0.50))
                call = {
                    "start_time": float(arguments["start_time"]),
                    "end_time": float(arguments["end_time"]),
                    "nframes": int(arguments["nframes"]),
                    "resize": float(arguments.get("resize", 1.0)),
                    "retention_ratio": ratio,
                    "evidence_request": str(
                        arguments.get(
                            "evidence_request",
                            "Describe visible facts that distinguish the options.",
                        )
                    ).strip()[:500],
                }
            except (KeyError, TypeError, ValueError):
                continue
            if (
                call["end_time"] <= call["start_time"]
                or not 1 <= call["nframes"] <= 64
                or not 0.05 <= call["resize"] <= 2.0
                or ratio not in ALLOWED_RETENTION_RATIOS
                or not call["evidence_request"]
            ):
                continue
            calls.append(call)
    return calls


def canonical_tool_call(call: dict[str, Any]) -> str:
    payload = {"tool": "frame_select", "arguments": dict(call)}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def parse_perception_json(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    if start < 0:
        raise ValueError("perception response contains no JSON object")
    payload, _ = decoder.raw_decode(candidate[start:])
    if not isinstance(payload, dict):
        raise ValueError("perception response is not a JSON object")
    required = (
        "observed_facts",
        "visible_text",
        "temporal_changes",
        "option_evidence",
        "uncertainties",
    )
    if set(payload) != set(required):
        raise ValueError(f"perception response keys must be exactly {required}")
    for key in ("observed_facts", "visible_text", "temporal_changes", "uncertainties"):
        if not isinstance(payload[key], list):
            raise ValueError(f"perception field {key} must be a list")
        payload[key] = [str(item)[:160] for item in payload[key]][:4]
    if not isinstance(payload["option_evidence"], dict):
        raise ValueError("perception field option_evidence must be an object")
    payload["option_evidence"] = {
        str(key)[:32]: [str(item)[:160] for item in value][:2]
        for key, value in payload["option_evidence"].items()
        if isinstance(value, list)
    }
    return payload


def _contains_private_request_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace(" ", "_")
            if normalized in _PRIVATE_REQUEST_KEYS:
                return True
            if _contains_private_request_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_private_request_key(item) for item in value)
    return False


def runtime_request_audit(
    sample: ModelSample,
    controller_messages: list[dict[str, Any]],
    tool_steps: list[dict[str, Any]],
) -> tuple[str, str | None]:
    if hasattr(sample, "answer") or hasattr(sample, "metadata"):
        return "failed", "model_sample_contains_private_fields"
    controller_media = 0
    for message in controller_messages:
        content = message.get("content")
        if isinstance(content, list):
            controller_media += sum(
                isinstance(item, dict)
                and item.get("type") in {"image_url", "video_url"}
                for item in content
            )
    if controller_media:
        return "failed", "controller_received_media"
    if _contains_private_request_key(controller_messages):
        return "failed", "controller_private_key"
    for step in tool_steps:
        audit = step.get("request_audit")
        if not isinstance(audit, dict):
            return "failed", "perception_audit_missing"
        if int(audit.get("media_count", -1)) != 1:
            return "failed", "perception_media_count"
        if int(audit.get("candidate_field_count", -1)) != 0:
            return "failed", "perception_candidate_field"
        serialized = json.dumps(audit.get("messages"), ensure_ascii=False)
        if "Direct candidate" in serialized or "candidate_answer" in serialized:
            return "failed", "perception_candidate_leak"
        if _contains_private_request_key(audit.get("messages")):
            return "failed", "perception_private_key"
    return "passed", None


def deterministic_random_ratio(
    sample_id: str,
    trajectory_index: int,
    perception_step: int,
) -> float:
    if trajectory_index < 4:
        return float(ALLOWED_RETENTION_RATIOS[trajectory_index])
    patterns = (
        (0, 1, 2, 3, 0),
        (0, 1, 0, 2, 3),
        (0, 2, 1, 3, 2),
        (0, 3, 1, 2, 3),
        (0, 2, 3, 1, 0),
    )
    random_index = (trajectory_index - 4) % 20
    group, shift = divmod(random_index, 4)
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    label_shift = digest[0] % 4
    position_shift = digest[1] % len(patterns[group])
    position = (perception_step + position_shift) % len(patterns[group])
    ratio_index = (patterns[group][position] + shift + label_shift) % 4
    return float(ALLOWED_RETENTION_RATIOS[ratio_index])


def budget_sequence_bank_hash() -> str:
    rows = [
        [
            deterministic_random_ratio("frozen-sequence-hash", trajectory, step)
            for step in range(5)
        ]
        for trajectory in range(4, 24)
    ]
    return prompt_hash(json.dumps(rows, separators=(",", ":")))


class _CacheLock:
    def __init__(self, thread_lock: threading.Lock, lock_path: Path):
        self.thread_lock = thread_lock
        self.lock_path = lock_path
        self.handle = None

    def __enter__(self) -> "_CacheLock":
        self.thread_lock.acquire()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+b")
        try:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            try:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            self.handle.close()
        self.thread_lock.release()


def _trajectory_seed(sample_id: str, trajectory_index: int) -> int:
    digest = hashlib.sha256(
        f"{sample_id}:{trajectory_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _json_usage_sum(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _api_multimodal_video_tokens(usage: dict[str, Any]) -> int | None:
    """Read vLLM's measured video placeholder span from OpenAI usage details."""

    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    multimodal = details.get("multimodal_tokens")
    if not isinstance(multimodal, dict):
        return None
    value = multimodal.get("video")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _short_observation(payload: dict[str, Any], timestamps: list[float]) -> str:
    wrapper = {
        "timestamps": [round(value, 3) for value in timestamps],
        **payload,
    }
    return f"<tool_response>{json.dumps(wrapper, ensure_ascii=False, separators=(',', ':'))}</tool_response>"


class FlashVIDHybridEvaluator:
    """Text-only controller plus candidate-blind FlashVID perception."""

    def __init__(
        self,
        controller_client: OpenAICompatibleClient,
        controller_model: str,
        perception_pool: BudgetEndpointPool,
        video_root: Path,
        frame_root: Path,
        cache_root: Path,
        media_root: Path,
        config: FlashVIDHybridConfig,
        *,
        candidate_sources: dict[str, str] | None = None,
        candidate_file_hash: str | None = None,
        normalized_candidate_hash: str | None = None,
        manifest_hash: str | None = None,
        experiment_config_hash: str | None = None,
    ):
        self.controller_client = controller_client
        self.controller_model = controller_model
        self.perception_pool = perception_pool
        self.index = VideoIndex(video_root)
        self.frame_root = frame_root.resolve()
        self.cache_root = cache_root.resolve()
        self.media_root = media_root.resolve()
        self.config = config
        self.candidate_sources = candidate_sources or {}
        self.candidate_file_hash = candidate_file_hash
        self.normalized_candidate_hash = normalized_candidate_hash
        self.manifest_hash = manifest_hash
        self.experiment_config_hash = experiment_config_hash
        self.implementation_sha256 = core_implementation_sha256()
        self._endpoint_clients: dict[str, OpenAICompatibleClient] = {}
        self._endpoint_semaphores: dict[str, threading.BoundedSemaphore] = {}
        for endpoint in perception_pool.endpoints:
            self._endpoint_clients[endpoint.base_url] = OpenAICompatibleClient(
                endpoint.base_url,
                endpoint.api_key,
            )
            self._endpoint_semaphores[endpoint.base_url] = threading.BoundedSemaphore(
                endpoint.max_concurrency
            )
        self._cache_guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)

    def prompt_hashes(self) -> dict[str, str]:
        return {
            "controller": controller_prompt_hash(self.config.controller_prompt_id),
            "perception": perception_prompt_hash(),
        }

    def run_fingerprint(self) -> str:
        payload = {
            "config": asdict(self.config),
            "controller_model": self.controller_model,
            "perception_endpoints": [
                {
                    "retention_ratio": endpoint.retention_ratio,
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "backend_revision": endpoint.backend_revision,
                }
                for endpoint in self.perception_pool.endpoints
            ],
            "prompt_hashes": self.prompt_hashes(),
            "candidate_file_sha256": self.candidate_file_hash,
            "normalized_candidate_sha256": self.normalized_candidate_hash,
            "manifest_sha256": self.manifest_hash,
            "experiment_config_sha256": self.experiment_config_hash,
            "budget_sequence_bank_sha256": budget_sequence_bank_hash(),
            "implementation_sha256": self.implementation_sha256,
        }
        return prompt_hash(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def static_audit_fields(self) -> dict[str, Any]:
        """Return frozen fields that do not depend on a successful model call."""
        return {
            "candidate_results_sha256": self.candidate_file_hash,
            "normalized_candidate_sha256": self.normalized_candidate_hash,
            "controller_model": self.controller_model,
            "controller_prompt_id": self.config.controller_prompt_id,
            "controller_prompt_hash": controller_prompt_hash(
                self.config.controller_prompt_id
            ),
            "perception_prompt_hash": perception_prompt_hash(),
            "run_fingerprint": self.run_fingerprint(),
            "run_config_sha256": self.run_fingerprint(),
            "manifest_sha256": self.manifest_hash,
            "experiment_config_sha256": self.experiment_config_hash,
            "budget_policy": self.config.budget_policy,
            "budget_strategy": self.config.budget_strategy,
            "budget_random_seed": self.config.budget_random_seed,
            "budget_sequence": [],
        }

    def freeze_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "controller_model": self.controller_model,
            "perception_endpoints": [
                {
                    "retention_ratio": endpoint.retention_ratio,
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "max_concurrency": endpoint.max_concurrency,
                    "backend_revision": endpoint.backend_revision,
                }
                for endpoint in self.perception_pool.endpoints
            ],
            "prompt_hashes": self.prompt_hashes(),
            "candidate_file_sha256": self.candidate_file_hash,
            "normalized_candidate_sha256": self.normalized_candidate_hash,
            "manifest_sha256": self.manifest_hash,
            "experiment_config_sha256": self.experiment_config_hash,
            "budget_sequence_bank_sha256": budget_sequence_bank_hash(),
            "implementation_sha256": self.implementation_sha256,
            "run_fingerprint": self.run_fingerprint(),
            "candidate_rerun": 0,
        }
        path = output_dir / "frozen_config.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != payload:
                raise RuntimeError(f"frozen config changed during resume: {path}")
        else:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (output_dir / "prompt_hashes.json").write_text(
            json.dumps(self.prompt_hashes(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _select_ratio(
        self,
        requested: float,
        sample_id: str,
        trajectory_index: int,
        step: int,
        *,
        policy_override: str | None,
        fixed_ratio_override: float | None,
    ) -> float:
        policy = policy_override or self.config.budget_policy
        if policy == "fixed":
            ratio = (
                float(fixed_ratio_override)
                if fixed_ratio_override is not None
                else float(self.config.fixed_retention_ratio)
            )
        elif policy == "random":
            ratio = deterministic_random_ratio(sample_id, trajectory_index, step)
        elif policy == "model":
            ratio = float(requested)
        else:
            raise ValueError(f"unsupported budget policy: {policy}")
        if ratio not in ALLOWED_RETENTION_RATIOS:
            raise ValueError(f"invalid executed retention ratio: {ratio}")
        return ratio

    def _select_budget_decision(
        self,
        requested: float,
        sample_id: str,
        trajectory_index: int,
        step: int,
        *,
        route: str,
        planned_call: dict[str, Any],
        previous_observations: list[dict[str, Any]],
        previous_budgets: list[float],
        policy_override: str | None,
        fixed_ratio_override: float | None,
        is_change_confirmation: bool,
    ) -> BudgetDecision:
        if self.config.budget_strategy is not None:
            public_call = {
                key: value
                for key, value in planned_call.items()
                if key
                in {
                    "start_time",
                    "end_time",
                    "nframes",
                    "resize",
                    "retention_ratio",
                    "evidence_request",
                }
            }
            # The controller sees the Direct candidate, so its free-form
            # evidence_request can encode candidate information.  Budget
            # strategies receive the trusted candidate-blind request instead;
            # the original text remains available only in the controller audit.
            public_call["evidence_request"] = candidate_blind_evidence_request(
                route
            )
            public_call["sample_key"] = hashlib.sha256(
                sample_id.encode("utf-8")
            ).hexdigest()[:16]
            if is_change_confirmation:
                public_call["purpose"] = "confirm_change"
            context = BudgetContext(
                route=route,
                step_index=step,
                planned_call=public_call,
                previous_observations=tuple(previous_observations),
                previous_budgets=tuple(previous_budgets),
            )
            matched_distribution = (
                freeze_matched_distribution(
                    self.config.budget_match_distribution or {}
                )
                if self.config.budget_strategy == "random_matched"
                else None
            )
            return choose_budget(
                self.config.budget_strategy,
                context,
                random_seed=self.config.budget_random_seed,
                matched_distribution=matched_distribution,
                sample_key=(
                    sample_id
                    if self.config.budget_strategy == "random_matched"
                    else None
                ),
            )
        policy = policy_override or self.config.budget_policy
        ratio = self._select_ratio(
            requested,
            sample_id,
            trajectory_index,
            step,
            policy_override=policy_override,
            fixed_ratio_override=fixed_ratio_override,
        )
        return BudgetDecision(ratio, f"legacy_{policy}")

    def _cache_lock(self, key: str) -> _CacheLock:
        with self._cache_guard:
            thread_lock = self._key_locks.setdefault(key, threading.Lock())
        return _CacheLock(
            thread_lock,
            self.cache_root / key[:2] / f"{key}.lock",
        )

    @staticmethod
    def _safe_sample_id(sample_id: str) -> str:
        cleaned = _SAFE_ID_RE.sub("_", sample_id).strip("._")
        return (cleaned or "sample")[:60]

    def _perceive(
        self,
        sample: ModelSample,
        video: Path,
        metadata: dict[str, float | int],
        call: dict[str, Any],
        trajectory_index: int,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_id = self._safe_sample_id(sample.sample_id)
        safe_dataset = self._safe_sample_id(sample.dataset)
        endpoint = self.perception_pool.choose(float(call["retention_ratio"]))
        request_context_hash = perception_request_context_hash(
            sample,
            video,
            str(call["evidence_request"]),
        )
        key = perception_cache_key(
            video=video,
            start_time=float(call["start_time"]),
            end_time=float(call["end_time"]),
            nframes=int(call["nframes"]),
            resize=float(call["resize"]),
            retention_ratio=float(call["retention_ratio"]),
            model=endpoint.model,
            prompt_hash=request_context_hash,
            evidence_request=str(call["evidence_request"]),
            backend_revision=endpoint.backend_revision,
        )
        cache_path = self.cache_root / key[:2] / f"{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(key):
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                trace = {
                    **cached["trace"],
                    "cache_hit": True,
                    "cache_key": key,
                }
                trace["cached_perception_usage"] = trace.get("perception_usage", {})
                trace["cached_perception_latency_s"] = trace.get(
                    "perception_latency_s",
                    0.0,
                )
                trace["perception_executed_usage"] = _usage_fields({})
                trace["perception_executed_latency_s"] = 0.0
                trace["api_attempts"] = 0
                trace["cache_miss_requests"] = 0
                return cached["observation"], {
                    **trace,
                }

            sample_digest = hashlib.sha256(
                f"{sample.dataset}:{sample.sample_id}".encode("utf-8")
            ).hexdigest()[:12]
            invocation = (
                f"{key[:12]}_{os.getpid()}_{threading.get_ident()}"
            )
            scope = f"{safe_dataset}_{safe_id}_{sample_digest}"
            frame_dir = (
                self.frame_root
                / "flashvid_budget_v1"
                / scope
                / f"trajectory_{trajectory_index:02d}"
                / f"step_{step:02d}_{invocation}"
            )
            frames, timestamps, backend = official_select_frames(
                video,
                float(call["start_time"]),
                float(call["end_time"]),
                int(call["nframes"]),
                float(call["resize"]),
                frame_dir,
            )
            media_dir = (
                self.media_root
                / scope
                / f"trajectory_{trajectory_index:02d}"
                / f"step_{step:02d}_{invocation}"
            )
            media_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = media_dir / "frames.mp4"
            sidecar_path = media_dir / "timestamps.json"
            try:
                media_meta = pack_frames_to_mp4(
                    frames,
                    timestamps,
                    mp4_path,
                    sidecar_path,
                )
                resolved_mp4 = mp4_path.resolve()
                try:
                    resolved_mp4.relative_to(self.media_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"temporary media escaped allowed root: {resolved_mp4}"
                    ) from exc

                messages = build_perception_messages(
                    resolved_mp4,
                    sample.question,
                    sample.choices,
                    str(call["evidence_request"]),
                    timestamps,
                )
                request_text = build_perception_text(
                    sample.question,
                    sample.choices,
                    str(call["evidence_request"]),
                    timestamps,
                )
                client = self._endpoint_clients[endpoint.base_url]
                semaphore = self._endpoint_semaphores[endpoint.base_url]
                error: Exception | None = None
                result = None
                api_attempts = 0
                for _ in range(2):
                    api_attempts += 1
                    try:
                        with semaphore:
                            result = client.chat(
                                endpoint.model,
                                messages,
                                max_tokens=PERCEPTION_MAX_TOKENS,
                                temperature=0.0,
                                response_format=PERCEPTION_RESPONSE_FORMAT,
                            )
                        observation = parse_perception_json(result.content)
                        break
                    except Exception as exc:
                        error = exc
                else:
                    raise RuntimeError(
                        "perception failed twice at ratio "
                        f"{call['retention_ratio']}: {error}"
                    ) from error
                assert result is not None
                api_multimodal_tokens = _api_multimodal_video_tokens(result.usage)
                raw_tokens, retained_tokens, effective_ratio = flashvid_token_counts(
                    metadata,
                    len(frames),
                    float(call["resize"]),
                    float(call["retention_ratio"]),
                )
                trace = {
                    "start_time": float(call["start_time"]),
                    "end_time": float(call["end_time"]),
                    "actual_timestamps": [round(value, 6) for value in timestamps],
                    "nframes_requested": int(call["nframes"]),
                    "nframes_actual": len(frames),
                    "resize": float(call["resize"]),
                    "retention_ratio": float(call["retention_ratio"]),
                    "raw_visual_tokens": int(raw_tokens),
                    "retained_visual_tokens": int(retained_tokens),
                    "raw_visual_tokens_estimated": int(raw_tokens),
                    "retained_visual_tokens_estimated": int(retained_tokens),
                    "raw_visual_tokens_actual": None,
                    "retained_visual_tokens_actual": None,
                    "api_multimodal_video_tokens_actual": api_multimodal_tokens,
                    "effective_retention_ratio": float(effective_ratio),
                    "token_count_source": (
                        "flashvid_plugin_contract_estimate_with_vllm_api_multimodal_audit"
                    ),
                    "evidence_request": str(call["evidence_request"]),
                    "frame_backend": backend,
                    "perception_endpoint": endpoint.base_url,
                    "perception_model": endpoint.model,
                    "perception_backend_revision": endpoint.backend_revision,
                    "perception_usage": _usage_fields(result.usage),
                    "perception_prompt_tokens_details": result.usage.get(
                        "prompt_tokens_details"
                    ),
                    "perception_executed_usage": _usage_fields(result.usage),
                    "perception_latency_s": result.latency_s,
                    "perception_executed_latency_s": result.latency_s,
                    "api_attempts": api_attempts,
                    "cache_miss_requests": 1,
                    "media": {
                        "frame_count": media_meta["frame_count"],
                        "timestamps": [
                            item["timestamp_s"] for item in media_meta["frames"]
                        ],
                    },
                    "request_audit": {
                        "messages": [
                            {
                                "role": "system",
                                "content": PERCEPTION_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": "<TEMP_MEDIA>"},
                                    },
                                    {"type": "text", "text": request_text},
                                ],
                            },
                        ],
                        "media_count": 1,
                        "candidate_field_count": 0,
                        "request_context_sha256": request_context_hash,
                        "sha256": hashlib.sha256(
                            json.dumps(
                                [
                                    PERCEPTION_SYSTEM_PROMPT,
                                    request_text,
                                    "<TEMP_MEDIA>",
                                ],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                    },
                    "cache_hit": False,
                    "cache_key": key,
                }
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=f".{key}.",
                    suffix=".partial",
                    dir=cache_path.parent,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(
                        json.dumps(
                            {"observation": observation, "trace": trace},
                            ensure_ascii=False,
                        )
                    )
                try:
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                return observation, trace
            finally:
                mp4_path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)
                for frame in frames:
                    frame.unlink(missing_ok=True)
                for directory in (
                    frame_dir,
                    media_dir,
                    frame_dir.parent,
                    media_dir.parent,
                ):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass

    def flashvid_hybrid(
        self,
        sample: ModelSample,
        *,
        trajectory_index: int | None = None,
        policy_override: str | None = None,
        fixed_ratio_override: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        trajectory_index = (
            self.config.trajectory_index
            if trajectory_index is None
            else int(trajectory_index)
        )
        video = self.index.resolve(sample.video)
        metadata = probe_video(video)
        route = _question_route(sample.question, {})
        initial_call, _ = _initial_tool_call(
            sample.question,
            float(metadata["duration"]),
            "hybrid_v3c",
            {},
        )
        initial_call.update(
            {
                "retention_ratio": float(self.config.fixed_retention_ratio),
                "evidence_request": (
                    "Inspect the visible events and details needed to distinguish all answer options."
                ),
            }
        )
        controller_system_prompt = get_prompt(
            self.config.controller_prompt_id
        ).text
        controller_messages: list[dict[str, Any]] = [
            {"role": "system", "content": controller_system_prompt},
            {
                "role": "user",
                "content": build_controller_user_prompt(sample, metadata, route),
            },
        ]
        training_messages: list[dict[str, Any]] = [
            {"role": "system", "content": controller_system_prompt},
            {
                "role": "user",
                "content": build_controller_user_prompt(sample, metadata, route),
            },
        ]
        controller_usage_items: list[dict[str, Any]] = []
        perception_usage_items: list[dict[str, Any]] = []
        perception_executed_usage_items: list[dict[str, Any]] = []
        controller_latency = 0.0
        perception_latency = 0.0
        perception_executed_latency = 0.0
        tool_steps: list[dict[str, Any]] = []
        policy_observations: list[dict[str, Any]] = []
        policy_budgets: list[float] = []
        observed_intervals: list[tuple[float, float]] = []
        raw_visual_tokens = 0
        retained_visual_tokens = 0
        cache_hits = 0
        fallback_reason: str | None = None
        final_answer: str | None = None
        final_raw = ""
        proposed_change: str | None = None
        confirmation_observations = 0
        turns = 0
        controller_raw_responses: list[str] = []

        # Let the model plan the first interval and budget. The deterministic v3c
        # action above is used only when the initial response is malformed.
        first_result = self.controller_client.chat(
            self.controller_model,
            controller_messages,
            max_tokens=512,
            temperature=self.config.controller_temperature,
            seed=_trajectory_seed(sample.sample_id, trajectory_index),
        )
        turns += 1
        controller_usage_items.append(_usage_fields(first_result.usage))
        controller_latency += first_result.latency_s
        controller_raw_responses.append(first_result.content)
        requested_calls = parse_budget_tool_calls(
            first_result.content,
            require_retention_ratio=(
                self.config.budget_strategy == "model_requested"
            ),
        )
        if requested_calls:
            pending_calls = requested_calls[:1]
        else:
            pending_calls = [initial_call]
            fallback_reason = "initial_tool_parse_fallback"

        while (
            turns <= self.config.max_turns
            and len(tool_steps) < self.config.max_perception_calls
        ):
            accepted: list[dict[str, Any]] = []
            for requested in pending_calls[:2]:
                start_time = max(
                    0.0,
                    min(float(requested["start_time"]), float(metadata["duration"]) - 0.1),
                )
                end_time = min(
                    float(metadata["duration"]),
                    max(start_time + 0.1, float(requested["end_time"])),
                )
                interval = (start_time, end_time)
                if any(_ranges_redundant(interval, old) for old in observed_intervals):
                    continue
                if any(
                    _ranges_overlap(
                        interval,
                        (float(item["start_time"]), float(item["end_time"])),
                    )
                    for item in accepted
                ):
                    continue
                executed = dict(requested)
                executed["start_time"] = start_time
                executed["end_time"] = end_time
                budget_decision = self._select_budget_decision(
                    float(requested["retention_ratio"]),
                    (
                        f"{sample.dataset}:{sample.sample_id}"
                        if self.config.budget_strategy == "random_matched"
                        else sample.sample_id
                    ),
                    trajectory_index,
                    len(tool_steps) + len(accepted),
                    route=route,
                    planned_call=executed,
                    previous_observations=policy_observations,
                    previous_budgets=policy_budgets,
                    policy_override=policy_override,
                    fixed_ratio_override=fixed_ratio_override,
                    is_change_confirmation=proposed_change is not None,
                )
                executed["retention_ratio"] = budget_decision.ratio
                executed["_requested_retention_ratio"] = float(
                    requested["retention_ratio"]
                )
                executed["_budget_strategy"] = (
                    self.config.budget_strategy
                    or policy_override
                    or self.config.budget_policy
                )
                executed["_budget_reason_code"] = budget_decision.reason_code
                executed["controller_evidence_request"] = str(
                    requested["evidence_request"]
                )
                executed["evidence_request"] = candidate_blind_evidence_request(
                    route
                )
                accepted.append(executed)
            if not accepted:
                fallback_reason = fallback_reason or "invalid_or_repeated_tool_call"
                break

            accepted = accepted[
                : self.config.max_perception_calls - len(tool_steps)
            ]
            execution_batch: list[
                tuple[dict[str, Any], str, dict[str, Any]]
            ] = []
            for call in accepted:
                controller_evidence_request = str(
                    call["controller_evidence_request"]
                )
                perception_call = {
                    key: value
                    for key, value in call.items()
                    if key != "controller_evidence_request"
                    and not key.startswith("_")
                }
                budget_audit = {
                    "requested_retention_ratio": call[
                        "_requested_retention_ratio"
                    ],
                    "executed_retention_ratio": call["retention_ratio"],
                    "budget_strategy": call["_budget_strategy"],
                    "budget_reason_code": call["_budget_reason_code"],
                }
                execution_batch.append(
                    (
                        perception_call,
                        controller_evidence_request,
                        budget_audit,
                    )
                )
            canonical_batch = "\n".join(
                canonical_tool_call(call) for call, _, _ in execution_batch
            )
            controller_messages.append(
                {"role": "assistant", "content": canonical_batch}
            )
            training_messages.append(
                {"role": "assistant", "content": canonical_batch}
            )

            for (
                perception_call,
                controller_evidence_request,
                budget_audit,
            ) in execution_batch:
                if len(tool_steps) >= self.config.max_perception_calls:
                    break
                observation, trace = self._perceive(
                    sample,
                    video,
                    metadata,
                    perception_call,
                    trajectory_index,
                    len(tool_steps),
                )
                trace["controller_evidence_request"] = controller_evidence_request
                trace["evidence_request_gate"] = _PERCEPTION_REQUEST_GATE_VERSION
                trace.update(budget_audit)
                observed_intervals.append(
                    (
                        float(perception_call["start_time"]),
                        float(perception_call["end_time"]),
                    )
                )
                raw_visual_tokens += int(trace["raw_visual_tokens"])
                retained_visual_tokens += int(trace["retained_visual_tokens"])
                logical_perception_usage = dict(
                    trace.get("perception_usage") or {}
                )
                perception_usage_items.append(logical_perception_usage)
                perception_executed_usage_items.append(
                    dict(
                        trace.get("perception_executed_usage")
                        or logical_perception_usage
                    )
                )
                logical_perception_latency = float(
                    trace.get("perception_latency_s", 0.0)
                )
                perception_latency += logical_perception_latency
                perception_executed_latency += float(
                    trace.get(
                        "perception_executed_latency_s",
                        logical_perception_latency,
                    )
                )
                cache_hits += int(bool(trace["cache_hit"]))
                tool_steps.append(trace)
                policy_observations.append(observation)
                policy_budgets.append(float(perception_call["retention_ratio"]))
                tool_response = _short_observation(
                    observation,
                    list(trace["actual_timestamps"]),
                )
                controller_messages.append({"role": "tool", "content": tool_response})
                training_messages.append({"role": "tool", "content": tool_response})
                if proposed_change is not None:
                    confirmation_observations += 1

            if turns >= self.config.max_turns:
                break
            result = self.controller_client.chat(
                self.controller_model,
                controller_messages,
                max_tokens=512,
                temperature=self.config.controller_temperature,
                seed=_trajectory_seed(sample.sample_id, trajectory_index),
            )
            turns += 1
            controller_usage_items.append(_usage_fields(result.usage))
            controller_latency += result.latency_s
            controller_raw_responses.append(result.content)
            final_raw = result.content
            strict = extract_strict_answer_letter(result.content, sample.option_letters)
            if strict is not None:
                controller_messages.append(
                    {"role": "assistant", "content": f"Answer: {strict}"}
                )
                if (
                    sample.candidate_answer is not None
                    and strict != sample.candidate_answer
                ):
                    if _UNCERTAINTY_RE.search(result.content):
                        fallback_reason = "uncertain_changed_answer"
                        break
                    if proposed_change is None:
                        proposed_change = strict
                        confirmation_observations = 0
                        controller_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"A proposed change from {sample.candidate_answer} to {strict} "
                                    "requires one additional non-redundant visual check. Emit one "
                                    "frame_select tool call targeting the distinguishing evidence."
                                ),
                            }
                        )
                        if turns >= self.config.max_turns:
                            fallback_reason = "change_unconfirmed"
                            break
                        confirmation_plan = self.controller_client.chat(
                            self.controller_model,
                            controller_messages,
                            max_tokens=512,
                            temperature=self.config.controller_temperature,
                            seed=_trajectory_seed(sample.sample_id, trajectory_index),
                        )
                        turns += 1
                        controller_usage_items.append(
                            _usage_fields(confirmation_plan.usage)
                        )
                        controller_latency += confirmation_plan.latency_s
                        controller_raw_responses.append(
                            confirmation_plan.content
                        )
                        pending_calls = parse_budget_tool_calls(
                            confirmation_plan.content,
                            require_retention_ratio=(
                                self.config.budget_strategy == "model_requested"
                            ),
                        )
                        if not pending_calls:
                            fallback_reason = "change_confirmation_tool_missing"
                            break
                        continue
                    if (
                        strict == proposed_change
                        and confirmation_observations >= 1
                    ):
                        final_answer = strict
                        training_messages.append(
                            {"role": "assistant", "content": f"Answer: {strict}"}
                        )
                        break
                    fallback_reason = "change_confirmation_disagreed"
                    break
                final_answer = strict
                training_messages.append(
                    {"role": "assistant", "content": f"Answer: {strict}"}
                )
                break
            pending_calls = parse_budget_tool_calls(
                result.content,
                require_retention_ratio=(
                    self.config.budget_strategy == "model_requested"
                ),
            )
            if not pending_calls:
                fallback_reason = "controller_answer_or_tool_parse_failed"
                break

        if (
            final_answer is None
            and tool_steps
            and fallback_reason != "initial_tool_parse_fallback"
        ):
            recovery_trigger = fallback_reason
            final_instruction = (
                "The perception-call budget is exhausted. Use only the observations "
                "already returned. Do not call a tool and output exactly one line: Answer: X."
            )
            controller_messages.append(
                {"role": "user", "content": final_instruction}
            )
            training_messages.append(
                {"role": "user", "content": final_instruction}
            )
            final_result = self.controller_client.chat(
                self.controller_model,
                controller_messages,
                max_tokens=64,
                temperature=self.config.controller_temperature,
                seed=_trajectory_seed(sample.sample_id, trajectory_index),
            )
            turns += 1
            controller_usage_items.append(_usage_fields(final_result.usage))
            controller_latency += final_result.latency_s
            controller_raw_responses.append(final_result.content)
            final_raw = final_result.content
            forced = extract_strict_answer_letter(
                final_result.content,
                sample.option_letters,
            )
            if forced is None:
                fallback_reason = "final_decision_parse_failed"
            else:
                controller_messages.append(
                    {"role": "assistant", "content": f"Answer: {forced}"}
                )
                if (
                    sample.candidate_answer is not None
                    and forced != sample.candidate_answer
                    and not (
                        forced == proposed_change
                        and confirmation_observations >= 1
                        and fallback_reason != "change_confirmation_disagreed"
                    )
                ):
                    fallback_reason = "forced_change_unconfirmed"
                else:
                    final_answer = forced
                    if recovery_trigger:
                        fallback_reason = f"forced_final_after_{recovery_trigger}"
                    training_messages.append(
                        {"role": "assistant", "content": f"Answer: {forced}"}
                    )

        if final_answer is None and sample.candidate_answer in sample.option_letters:
            final_answer = sample.candidate_answer
            decision_source = "candidate_fallback"
            training_messages.append(
                {"role": "assistant", "content": f"Answer: {final_answer}"}
            )
        elif final_answer is not None and final_answer == sample.candidate_answer:
            decision_source = "candidate_confirmed"
        elif final_answer is not None:
            decision_source = "confirmed_visual_change"
        else:
            decision_source = "no_valid_answer"

        controller_usage = _json_usage_sum(controller_usage_items)
        perception_usage = _json_usage_sum(perception_usage_items)
        perception_executed_usage = _json_usage_sum(
            perception_executed_usage_items
        )
        total_usage = {
            key: controller_usage[key] + perception_usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        executed_usage = {
            key: controller_usage[key] + perception_executed_usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        effective_ratio = (
            retained_visual_tokens / raw_visual_tokens if raw_visual_tokens else 0.0
        )
        api_multimodal_values = [
            step.get("api_multimodal_video_tokens_actual") for step in tool_steps
        ]
        api_multimodal_complete = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in api_multimodal_values
        )
        api_multimodal_total = (
            sum(int(value) for value in api_multimodal_values)
            if api_multimodal_complete
            else None
        )
        parse_error_reasons = {
            "initial_tool_parse_fallback",
            "invalid_or_repeated_tool_call",
            "controller_answer_or_tool_parse_failed",
            "change_confirmation_tool_missing",
            "final_decision_parse_failed",
        }
        parse_error = (
            fallback_reason if fallback_reason in parse_error_reasons else None
        )
        annotation_leak_status, annotation_leak_reason = runtime_request_audit(
            sample,
            controller_messages,
            tool_steps,
        )
        failure_stage = (
            "annotation_leak"
            if annotation_leak_status != "passed"
            else ("controller_parse" if parse_error else None)
        )
        trajectory_valid = parse_error is None and annotation_leak_status == "passed"
        sft_messages = training_messages if trajectory_valid else []
        return {
            "backend": "flashvid_hybrid",
            "agent_version": self.config.prompt_version,
            "prediction": final_answer,
            "final_prediction": final_answer,
            "raw_response": final_raw,
            "candidate_answer": sample.candidate_answer,
            "candidate_source": self.candidate_sources.get(
                sample.sample_id,
                "parsed" if sample.candidate_answer else "none",
            ),
            "candidate_results_sha256": self.candidate_file_hash,
            "normalized_candidate_sha256": self.normalized_candidate_hash,
            "manifest_sha256": self.manifest_hash,
            "experiment_config_sha256": self.experiment_config_hash,
            "run_config_sha256": self.run_fingerprint(),
            "candidate_rerun": 0,
            "candidate_changed": bool(
                sample.candidate_answer
                and final_answer
                and final_answer != sample.candidate_answer
            ),
            "fallback_to_candidate": decision_source == "candidate_fallback",
            "fallback_reason": fallback_reason,
            "parse_error": parse_error,
            "failure_stage": failure_stage,
            "trajectory_valid": trajectory_valid,
            "decision_source": decision_source,
            "question_route": route,
            "tool_steps": tool_steps,
            "tool_calls": tool_steps,
            "tool_call_count": len(tool_steps),
            "budget_sequence": [
                float(step["retention_ratio"]) for step in tool_steps
            ],
            "observed_intervals": [
                [round(start, 6), round(end, 6)]
                for start, end in observed_intervals
            ],
            "turn_count": turns,
            "rounds": turns,
            "raw_visual_tokens": raw_visual_tokens,
            "retained_visual_tokens": retained_visual_tokens,
            "raw_visual_tokens_estimated": raw_visual_tokens,
            "retained_visual_tokens_estimated": retained_visual_tokens,
            "raw_visual_tokens_actual": None,
            "retained_visual_tokens_actual": None,
            "api_multimodal_video_tokens_actual": api_multimodal_total,
            "api_multimodal_measurement_complete": api_multimodal_complete,
            "visual_token_count_source": (
                "flashvid_plugin_contract_estimate_with_vllm_api_multimodal_audit"
            ),
            "effective_retention_ratio": effective_ratio,
            "visual_tokens": retained_visual_tokens,
            "controller_model": self.controller_model,
            "perception_models": sorted(
                {step["perception_model"] for step in tool_steps}
            ),
            "controller_prompt_id": self.config.controller_prompt_id,
            "controller_prompt_hash": controller_prompt_hash(
                self.config.controller_prompt_id
            ),
            "perception_prompt_hash": perception_prompt_hash(),
            "run_fingerprint": self.run_fingerprint(),
            "controller_requests": len(controller_usage_items),
            "perception_requests": sum(
                int(step.get("api_attempts", 0)) for step in tool_steps
            ),
            "perception_logical_requests": len(perception_usage_items),
            "perception_tool_calls": len(tool_steps),
            "perception_cache_miss_requests": sum(
                int(step.get("cache_miss_requests", 0)) for step in tool_steps
            ),
            "perception_http_attempts": sum(
                int(step.get("api_attempts", 0)) for step in tool_steps
            ),
            "controller_usage": controller_usage,
            "perception_usage": perception_usage,
            "perception_executed_usage": perception_executed_usage,
            "usage": total_usage,
            "executed_usage": executed_usage,
            **total_usage,
            "controller_latency_s": controller_latency,
            "perception_latency_s": perception_latency,
            "perception_executed_latency_s": perception_executed_latency,
            "latency_s": controller_latency + perception_latency,
            "cache_hits": cache_hits,
            "perception_cache_keys": [step["cache_key"] for step in tool_steps],
            "trajectory_index": trajectory_index,
            "budget_policy": policy_override or self.config.budget_policy,
            "budget_strategy": self.config.budget_strategy,
            "budget_random_seed": self.config.budget_random_seed,
            "training_messages": sft_messages,
            "controller_request_audit": {
                "messages": controller_messages,
                "raw_assistant_responses": controller_raw_responses,
                "media_count": 0,
                "sha256": hashlib.sha256(
                    json.dumps(
                        controller_messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "annotation_leak_check": annotation_leak_status,
            "annotation_leak_reason": annotation_leak_reason,
            "elapsed_s": time.perf_counter() - started,
        }


def evaluate_flashvid_trajectories(
    samples: list[Sample],
    evaluator: FlashVIDHybridEvaluator,
    candidate_answers: dict[str, str],
    output_dir: Path,
    *,
    trajectories_per_sample: int = 24,
    concurrency: int = 32,
    resume: bool = False,
    retry_errors: bool = False,
) -> dict[str, Any]:
    """Generate the preregistered four fixed plus random budget traces.

    Ground truth is joined only after ``flashvid_hybrid`` returns. It is never
    present in the ``ModelSample`` passed to model-facing code.
    """

    if trajectories_per_sample < 4:
        raise ValueError("trajectories_per_sample must be at least 4")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = samples[0].dataset
    output_path = output_dir / f"{dataset}_flashvid_hybrid_trajectories.jsonl"
    expected_run_fingerprint = evaluator.run_fingerprint()
    cached: dict[str, dict[str, Any]] = {}
    if resume and output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            trajectory_id = str(record.get("trajectory_id") or "")
            if trajectory_id:
                if record.get("run_fingerprint") != expected_run_fingerprint:
                    raise RuntimeError(
                        f"resume fingerprint mismatch for {trajectory_id}: "
                        f"expected {expected_run_fingerprint}, "
                        f"found {record.get('run_fingerprint')}"
                    )
                cached[trajectory_id] = record

    work: list[tuple[Sample, int]] = []
    for sample in samples:
        for index in range(trajectories_per_sample):
            trajectory_id = f"{sample.sample_id}:{index}"
            previous = cached.get(trajectory_id)
            if previous is None or (
                retry_errors
                and bool(
                    previous.get("error")
                    or previous.get("verifier_error")
                    or previous.get("failure_stage")
                    or previous.get("parse_error")
                    or previous.get("trajectory_valid") is False
                )
            ):
                work.append((sample, index))

    fixed = ALLOWED_RETENTION_RATIOS

    def run_one(item: tuple[Sample, int]) -> dict[str, Any]:
        sample, index = item
        candidate = candidate_answers.get(sample.sample_id)
        model_sample = ModelSample.from_sample(sample, candidate)
        started = time.perf_counter()
        try:
            if index < len(fixed):
                result = evaluator.flashvid_hybrid(
                    model_sample,
                    trajectory_index=index,
                    policy_override="fixed",
                    fixed_ratio_override=float(fixed[index]),
                )
                result["trajectory_policy"] = f"constant_{fixed[index]:.2f}"
            else:
                result = evaluator.flashvid_hybrid(
                    model_sample,
                    trajectory_index=index,
                    policy_override="random",
                )
                result["trajectory_policy"] = "stratified_random"
        except Exception as exc:
            fallback = candidate if candidate in sample.option_letters else None
            result = {
                "prediction": fallback,
                "final_prediction": fallback,
                "candidate_answer": fallback,
                "candidate_rerun": 0,
                "fallback_to_candidate": fallback is not None,
                "decision_source": "candidate_fallback" if fallback else "no_valid_answer",
                "error": f"{type(exc).__name__}: {exc}",
                "turn_count": 0,
                "tool_steps": [],
                "raw_visual_tokens": 0,
                "retained_visual_tokens": 0,
                "usage": {},
                "latency_s": time.perf_counter() - started,
                "training_messages": [],
                "annotation_leak_check": "not_run",
                "run_fingerprint": expected_run_fingerprint,
            }
        scoring = ScoringRecord.from_sample(sample)
        result.update(
            {
                "dataset": scoring.dataset,
                "sample_id": scoring.sample_id,
                "trajectory_index": index,
                "trajectory_id": f"{sample.sample_id}:{index}",
                "video": sample.video,
                "answer": scoring.answer,
                "correct": result.get("prediction") == scoring.answer,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        return result

    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(run_one, item) for item in work]
            for future in as_completed(futures):
                record = future.result()
                cached[record["trajectory_id"]] = record
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    ordered = [
        cached[f"{sample.sample_id}:{index}"]
        for sample in samples
        for index in range(trajectories_per_sample)
        if f"{sample.sample_id}:{index}" in cached
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    summary = {
        "dataset": dataset,
        "backend": "flashvid_hybrid",
        "mode": "trajectory_generation",
        "samples": len(samples),
        "trajectories_per_sample": trajectories_per_sample,
        "expected": len(samples) * trajectories_per_sample,
        "completed": len(ordered),
        "correct_trajectories": sum(bool(item.get("correct")) for item in ordered),
        "errors": sum(bool(item.get("error")) for item in ordered),
        "output": str(output_path),
    }
    (output_dir / f"{dataset}_trajectory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

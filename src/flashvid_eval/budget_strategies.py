from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ALLOWED_RATIOS = (0.10, 0.25, 0.50, 1.00)
SUPPORTED_RANDOM_SEEDS = (17, 42, 73)

_PRIVATE_KEYS = {
    "answer",
    "candidate",
    "candidate_answer",
    "candidate_raw_response",
    "clue_intervals",
    "correct_answer",
    "direct_explanation",
    "direct_response",
    "ground_truth",
    "question_type",
    "right_answer",
    "time_range",
}

_ROUTE_ALIASES = {
    "explicit_time": "explicit_question_time",
    "fine_detail": "ocr_detail",
    "ordinary_event": "ordinary_event",
}

_ROUTE_RATIOS = {
    "global_overview": 0.10,
    "ordinary_event": 0.25,
    "temporal_event": 0.25,
    "explicit_question_time": 0.50,
    "action_event": 0.50,
    "ocr_detail": 1.00,
}

_ESCALATION_START_RATIOS = {
    "global_overview": 0.10,
    "ordinary_event": 0.10,
    "temporal_event": 0.10,
    "explicit_question_time": 0.25,
    "action_event": 0.25,
    "ocr_detail": 1.00,
}

_TEXT_REQUEST_TERMS = (
    "caption",
    "character",
    "label",
    "letter",
    "number",
    "ocr",
    "read",
    "sign",
    "subtitle",
    "text",
    "word",
    "written",
)
_TEMPORAL_REQUEST_TERMS = (
    "action",
    "after",
    "before",
    "change",
    "first",
    "motion",
    "next",
    "order",
    "sequence",
    "then",
    "transition",
)


@dataclass(frozen=True)
class BudgetContext:
    """Candidate-blind inputs available to a budget strategy.

    Every field is execution state rather than scoring state.  Recursive key
    validation prevents annotations or Direct-candidate material from being
    smuggled through a nested planned call or observation.
    """

    route: str
    step_index: int
    planned_call: Mapping[str, Any]
    previous_observations: Sequence[Mapping[str, Any]] = ()
    previous_budgets: Sequence[float] = ()

    def __post_init__(self) -> None:
        normalized_route = _normalize_route(self.route)
        if int(self.step_index) < 0:
            raise ValueError("step_index must be non-negative")
        if len(self.previous_observations) != len(self.previous_budgets):
            raise ValueError(
                "previous_observations and previous_budgets must have equal length"
            )
        _reject_private_keys(self.planned_call, path="planned_call")
        _reject_private_keys(self.previous_observations, path="previous_observations")
        normalized_budgets = tuple(_normalize_ratio(value) for value in self.previous_budgets)
        _canonical_json(
            {
                "route": normalized_route,
                "step_index": int(self.step_index),
                "planned_call": self.planned_call,
                "previous_observations": self.previous_observations,
                "previous_budgets": normalized_budgets,
            }
        )
        object.__setattr__(self, "route", normalized_route)
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "planned_call", dict(self.planned_call))
        object.__setattr__(
            self,
            "previous_observations",
            tuple(dict(item) for item in self.previous_observations),
        )
        object.__setattr__(self, "previous_budgets", normalized_budgets)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "step_index": self.step_index,
            "planned_call": self.planned_call,
            "previous_observations": self.previous_observations,
            "previous_budgets": self.previous_budgets,
        }


@dataclass(frozen=True)
class BudgetDecision:
    ratio: float
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio", _normalize_ratio(self.ratio))
        if not self.reason_code:
            raise ValueError("reason_code cannot be empty")


@dataclass(frozen=True)
class MatchedBudgetDistribution:
    """Frozen four-tier distribution for a cost-matched random baseline."""

    probabilities: tuple[float, float, float, float]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(ALLOWED_RATIOS):
            raise ValueError("matched distribution must contain four probabilities")
        values = tuple(float(value) for value in self.probabilities)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("matched-distribution probabilities must be finite and non-negative")
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("matched-distribution probabilities must sum to 1")
        payload = json.dumps(
            [f"{probability:.17g}" for probability in values],
            separators=(",", ":"),
        )
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(
            self, "sha256", hashlib.sha256(payload.encode("utf-8")).hexdigest()
        )

    def as_mapping(self) -> dict[float, float]:
        return dict(zip(ALLOWED_RATIOS, self.probabilities, strict=True))


def freeze_matched_distribution(
    weights: Mapping[float | str, float | int],
) -> MatchedBudgetDistribution:
    """Validate and normalize four probabilities or deterministic tier counts."""

    normalized_weights: dict[float, float] = {}
    for raw_ratio, raw_weight in weights.items():
        try:
            ratio = _normalize_ratio(float(raw_ratio))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid matched-distribution ratio: {raw_ratio}") from exc
        if ratio in normalized_weights:
            raise ValueError(f"duplicate matched-distribution ratio: {raw_ratio}")
        if isinstance(raw_weight, bool):
            raise ValueError("matched-distribution weights must be finite numbers")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("matched-distribution weights must be finite numbers") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("matched-distribution weights must be finite and non-negative")
        normalized_weights[ratio] = weight

    if set(normalized_weights) != set(ALLOWED_RATIOS):
        raise ValueError(
            f"matched distribution must contain exactly the four ratios {ALLOWED_RATIOS}"
        )
    total = math.fsum(normalized_weights.values())
    if total <= 0:
        raise ValueError("matched-distribution weights must have a positive sum")
    probabilities = tuple(normalized_weights[ratio] / total for ratio in ALLOWED_RATIOS)
    # Force the normalized representation to sum to one despite binary-float
    # rounding, so the final cumulative interval always ends exactly at 1.
    probability_list = list(probabilities)
    probability_list[-1] = 1.0 - math.fsum(probability_list[:-1])
    if probability_list[-1] < 0:
        raise ValueError("matched-distribution normalization produced an invalid probability")
    probabilities = (
        probability_list[0],
        probability_list[1],
        probability_list[2],
        probability_list[3],
    )
    return MatchedBudgetDistribution(probabilities=probabilities)


def choose_budget(
    strategy: str,
    context: BudgetContext,
    *,
    random_seed: int | None = None,
    matched_distribution: MatchedBudgetDistribution | None = None,
    sample_key: str | None = None,
) -> BudgetDecision:
    """Choose one of the four FlashVID budgets without model training.

    Supported strategies are the four explicit fixed policies,
    ``model_requested``, ``random_uniform``, ``random_matched``, ``route_rule`` and
    ``uncertainty_escalation``.
    ``random_uniform`` is deterministic for the full public context and seed.
    ``random_matched`` is deterministic only from sample key, step and seed, so
    it cannot adapt to observations while matching a frozen empirical mix.
    """

    fixed = {
        "fixed_r010": 0.10,
        "fixed_r025": 0.25,
        "fixed_r050": 0.50,
        "fixed_r100": 1.00,
    }
    if strategy == "model_requested":
        _reject_unused_random_arguments(
            strategy,
            random_seed=random_seed,
            matched_distribution=matched_distribution,
            sample_key=sample_key,
        )
        if "retention_ratio" not in context.planned_call:
            raise ValueError("model_requested requires planned_call.retention_ratio")
        return BudgetDecision(
            float(context.planned_call["retention_ratio"]),
            "model_requested",
        )
    if strategy in fixed:
        _reject_unused_random_arguments(
            strategy,
            random_seed=random_seed,
            matched_distribution=matched_distribution,
            sample_key=sample_key,
        )
        return BudgetDecision(fixed[strategy], strategy)
    if strategy == "random_uniform":
        if matched_distribution is not None or sample_key is not None:
            raise ValueError(
                "random_uniform does not accept matched_distribution or sample_key"
            )
        if random_seed not in SUPPORTED_RANDOM_SEEDS:
            raise ValueError(
                f"random_uniform seed must be one of {SUPPORTED_RANDOM_SEEDS}"
            )
        payload = _canonical_json(context.canonical_payload())
        digest = hashlib.sha256(f"{random_seed}\n{payload}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(ALLOWED_RATIOS)
        return BudgetDecision(
            ALLOWED_RATIOS[index], f"random_uniform_seed_{random_seed}"
        )
    if strategy == "random_matched":
        if random_seed not in SUPPORTED_RANDOM_SEEDS:
            raise ValueError(
                f"random_matched seed must be one of {SUPPORTED_RANDOM_SEEDS}"
            )
        if matched_distribution is None:
            raise ValueError("random_matched requires a frozen matched_distribution")
        normalized_sample_key = str(sample_key or "").strip()
        if not normalized_sample_key:
            raise ValueError("random_matched requires a non-empty sample_key")
        ratio = _sample_matched_ratio(
            matched_distribution,
            seed=random_seed,
            sample_key=normalized_sample_key,
            step_index=context.step_index,
        )
        return BudgetDecision(
            ratio,
            f"random_matched_seed_{random_seed}_{matched_distribution.sha256[:12]}",
        )
    _reject_unused_random_arguments(
        strategy,
        random_seed=random_seed,
        matched_distribution=matched_distribution,
        sample_key=sample_key,
    )
    if strategy == "route_rule":
        return _route_rule(context)
    if strategy == "uncertainty_escalation":
        return _uncertainty_escalation(context)
    raise ValueError(f"unsupported budget strategy: {strategy}")


def _sample_matched_ratio(
    distribution: MatchedBudgetDistribution,
    *,
    seed: int,
    sample_key: str,
    step_index: int,
) -> float:
    digest = hashlib.sha256(
        f"{seed}\n{sample_key}\n{step_index}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for ratio, probability in zip(
        ALLOWED_RATIOS, distribution.probabilities, strict=True
    ):
        cumulative += probability
        if draw < cumulative:
            return ratio
    return ALLOWED_RATIOS[-1]


def _reject_unused_random_arguments(
    strategy: str,
    *,
    random_seed: int | None,
    matched_distribution: MatchedBudgetDistribution | None,
    sample_key: str | None,
) -> None:
    if any(value is not None for value in (random_seed, matched_distribution, sample_key)):
        raise ValueError(f"{strategy} does not accept random baseline arguments")


def _route_rule(context: BudgetContext) -> BudgetDecision:
    ratio = _ROUTE_RATIOS[context.route]
    if context.route == "global_overview":
        reason = "route_global_overview"
    elif context.route in {"ordinary_event", "temporal_event"}:
        reason = "route_ordinary_event"
    elif context.route in {"explicit_question_time", "action_event"}:
        reason = "route_explicit_time_or_action"
    else:
        reason = "route_ocr_or_fine_detail"
    return BudgetDecision(ratio, reason)


def _uncertainty_escalation(context: BudgetContext) -> BudgetDecision:
    start_ratio = _ESCALATION_START_RATIOS[context.route]
    if not context.previous_observations:
        return BudgetDecision(start_ratio, f"start_{context.route}")

    last_observation = context.previous_observations[-1]
    previous_ratio = context.previous_budgets[-1]
    trigger = _escalation_trigger(context, last_observation)
    if trigger is not None:
        next_ratio = _next_ratio(previous_ratio)
        reason = trigger if next_ratio > previous_ratio else "hold_max_budget"
        return BudgetDecision(next_ratio, reason)

    ratio = start_ratio
    if _is_change_confirmation(context.planned_call) and previous_ratio > ratio:
        return BudgetDecision(previous_ratio, "hold_change_confirmation_floor")
    return BudgetDecision(ratio, "reset_sufficient_evidence")


def _escalation_trigger(
    context: BudgetContext, observation: Mapping[str, Any]
) -> str | None:
    uncertainties = _nonempty_items(observation.get("uncertainties"))
    if uncertainties:
        # The perception request is targeted, so its uncertainty list refers to
        # the evidence_request for this tool step rather than unrelated content.
        return "escalate_relevant_uncertainty"

    request = str(context.planned_call.get("evidence_request", "")).casefold()
    needs_text = context.route == "ocr_detail" or any(
        term in request for term in _TEXT_REQUEST_TERMS
    )
    if needs_text and not _nonempty_items(observation.get("visible_text")):
        return "escalate_missing_target_text"

    needs_temporal = context.route in {
        "action_event",
        "explicit_question_time",
        "temporal_event",
    } or any(term in request for term in _TEMPORAL_REQUEST_TERMS)
    if needs_temporal and not _nonempty_items(observation.get("temporal_changes")):
        return "escalate_missing_temporal_evidence"
    return None


def _is_change_confirmation(planned_call: Mapping[str, Any]) -> bool:
    if planned_call.get("is_change_confirmation") is True:
        return True
    purpose = str(planned_call.get("purpose", "")).strip().casefold()
    return purpose in {"change_confirmation", "confirm_change"}


def _nonempty_items(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _next_ratio(ratio: float) -> float:
    index = ALLOWED_RATIOS.index(_normalize_ratio(ratio))
    return ALLOWED_RATIOS[min(index + 1, len(ALLOWED_RATIOS) - 1)]


def _normalize_route(route: str) -> str:
    normalized = _ROUTE_ALIASES.get(str(route).strip(), str(route).strip())
    if normalized not in _ROUTE_RATIOS:
        raise ValueError(f"unsupported question route: {route}")
    return normalized


def _normalize_ratio(value: float) -> float:
    ratio = float(value)
    for allowed in ALLOWED_RATIOS:
        if abs(ratio - allowed) < 1e-9:
            return allowed
    raise ValueError(f"budget ratio must be one of {ALLOWED_RATIOS}: {value}")


def _reject_private_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().casefold()
            if key in _PRIVATE_KEYS:
                raise ValueError(f"private field is forbidden in budget context: {path}.{key}")
            _reject_private_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_private_keys(nested, path=f"{path}[{index}]")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("budget context must be JSON serializable") from exc

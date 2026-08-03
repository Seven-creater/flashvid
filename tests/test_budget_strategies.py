from __future__ import annotations

import pytest

from flashvid_eval.budget_strategies import (
    ALLOWED_RATIOS,
    BudgetContext,
    MatchedBudgetDistribution,
    choose_budget,
    freeze_matched_distribution,
)


def _context(
    *,
    route: str = "global_overview",
    observations: tuple[dict, ...] = (),
    budgets: tuple[float, ...] = (),
    planned_call: dict | None = None,
) -> BudgetContext:
    return BudgetContext(
        route=route,
        step_index=len(budgets),
        planned_call=planned_call
        or {
            "start_time": 0.0,
            "end_time": 30.0,
            "nframes": 12,
            "resize": 0.45,
            "evidence_request": "Observe the relevant event.",
        },
        previous_observations=observations,
        previous_budgets=budgets,
    )


def test_fixed_strategies_cover_exactly_four_budgets() -> None:
    context = _context()
    decisions = {
        name: choose_budget(name, context).ratio
        for name in ("fixed_r010", "fixed_r025", "fixed_r050", "fixed_r100")
    }
    assert decisions == {
        "fixed_r010": 0.10,
        "fixed_r025": 0.25,
        "fixed_r050": 0.50,
        "fixed_r100": 1.00,
    }


def test_model_requested_preserves_the_controller_budget_only() -> None:
    context = _context(
        planned_call={
            "retention_ratio": 0.25,
            "evidence_request": "Observe the event.",
        }
    )
    decision = choose_budget("model_requested", context)
    assert decision.ratio == 0.25
    assert decision.reason_code == "model_requested"


def test_random_uniform_is_deterministic_for_registered_seeds() -> None:
    context = _context(route="action_event")
    for seed in (17, 42, 73):
        first = choose_budget("random_uniform", context, random_seed=seed)
        second = choose_budget("random_uniform", context, random_seed=seed)
        assert first == second
        assert first.ratio in {0.10, 0.25, 0.50, 1.00}
        assert first.reason_code == f"random_uniform_seed_{seed}"
    with pytest.raises(ValueError, match="seed"):
        choose_budget("random_uniform", context, random_seed=1)


def test_random_matched_normalizes_counts_and_is_stable_by_sample_step() -> None:
    distribution = freeze_matched_distribution(
        {"0.10": 10, "0.25": 20, "0.50": 30, "1.00": 40}
    )
    assert distribution.probabilities == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert sum(distribution.probabilities) == 1.0
    context = _context(route="action_event")
    first = choose_budget(
        "random_matched",
        context,
        random_seed=42,
        matched_distribution=distribution,
        sample_key="lvbench:sample-1",
    )
    second = choose_budget(
        "random_matched",
        context,
        random_seed=42,
        matched_distribution=distribution,
        sample_key="lvbench:sample-1",
    )
    assert first == second
    assert first.ratio in ALLOWED_RATIOS
    assert distribution.sha256[:12] in first.reason_code


def test_random_matched_does_not_depend_on_observation_content() -> None:
    distribution = freeze_matched_distribution(
        {0.10: 1, 0.25: 1, 0.50: 1, 1.00: 1}
    )
    low_context = _context(
        observations=({"uncertainties": []},), budgets=(0.10,)
    )
    uncertain_context = _context(
        observations=({"uncertainties": ["Everything is unclear"]},),
        budgets=(1.00,),
    )
    kwargs = {
        "random_seed": 73,
        "matched_distribution": distribution,
        "sample_key": "lsdbench:stable-sample",
    }
    assert choose_budget("random_matched", low_context, **kwargs) == choose_budget(
        "random_matched", uncertain_context, **kwargs
    )


def test_random_matched_never_selects_zero_probability_tiers() -> None:
    distribution = freeze_matched_distribution(
        {0.10: 0, 0.25: 0, 0.50: 11, 1.00: 0}
    )
    for sample_index in range(50):
        decision = choose_budget(
            "random_matched",
            _context(),
            random_seed=17,
            matched_distribution=distribution,
            sample_key=f"cgbench:{sample_index}",
        )
        assert decision.ratio == 0.50


@pytest.mark.parametrize(
    "weights",
    (
        {0.10: 1, 0.25: 1, 0.50: 1},
        {0.10: 0, 0.25: 0, 0.50: 0, 1.00: 0},
        {0.10: -1, 0.25: 1, 0.50: 1, 1.00: 1},
        {0.10: 1, 0.25: 1, 0.50: 1, 0.75: 1, 1.00: 1},
    ),
)
def test_matched_distribution_rejects_invalid_four_tier_weights(
    weights: dict,
) -> None:
    with pytest.raises(ValueError, match="matched"):
        freeze_matched_distribution(weights)


def test_frozen_distribution_cannot_be_constructed_with_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        MatchedBudgetDistribution((0.10, 0.20, 0.30, 0.30))


def test_random_matched_requires_distribution_key_and_registered_seed() -> None:
    context = _context()
    distribution = freeze_matched_distribution(
        {0.10: 1, 0.25: 1, 0.50: 1, 1.00: 1}
    )
    with pytest.raises(ValueError, match="distribution"):
        choose_budget(
            "random_matched", context, random_seed=17, sample_key="sample"
        )
    with pytest.raises(ValueError, match="sample_key"):
        choose_budget(
            "random_matched",
            context,
            random_seed=17,
            matched_distribution=distribution,
        )
    with pytest.raises(ValueError, match="seed"):
        choose_budget(
            "random_matched",
            context,
            random_seed=99,
            matched_distribution=distribution,
            sample_key="sample",
        )


@pytest.mark.parametrize(
    ("route", "ratio", "reason"),
    (
        ("global_overview", 0.10, "route_global_overview"),
        ("ordinary_event", 0.25, "route_ordinary_event"),
        ("temporal_event", 0.25, "route_ordinary_event"),
        ("explicit_question_time", 0.50, "route_explicit_time_or_action"),
        ("action_event", 0.50, "route_explicit_time_or_action"),
        ("ocr_detail", 1.00, "route_ocr_or_fine_detail"),
    ),
)
def test_route_rule_mapping(route: str, ratio: float, reason: str) -> None:
    decision = choose_budget("route_rule", _context(route=route))
    assert (decision.ratio, decision.reason_code) == (ratio, reason)


def test_uncertainty_escalation_starts_by_route() -> None:
    assert choose_budget("uncertainty_escalation", _context()).ratio == 0.10
    assert (
        choose_budget(
            "uncertainty_escalation", _context(route="action_event")
        ).ratio
        == 0.25
    )
    assert (
        choose_budget("uncertainty_escalation", _context(route="ocr_detail")).ratio
        == 1.00
    )


@pytest.mark.parametrize(
    ("route", "request_text", "observation", "reason"),
    (
        (
            "global_overview",
            "Observe the event.",
            {"uncertainties": ["The object is unclear."]},
            "escalate_relevant_uncertainty",
        ),
        (
            "global_overview",
            "Read the sign text.",
            {"uncertainties": [], "visible_text": []},
            "escalate_missing_target_text",
        ),
        (
            "action_event",
            "Observe the action sequence.",
            {"uncertainties": [], "temporal_changes": []},
            "escalate_missing_temporal_evidence",
        ),
    ),
)
def test_uncertainty_escalation_requires_defined_evidence_failure(
    route: str, request_text: str, observation: dict, reason: str
) -> None:
    context = _context(
        route=route,
        observations=(observation,),
        budgets=(0.25,),
        planned_call={"evidence_request": request_text},
    )
    decision = choose_budget("uncertainty_escalation", context)
    assert decision.ratio == 0.50
    assert decision.reason_code == reason


def test_sufficient_observation_resets_low_and_confirmation_never_downgrades() -> None:
    observation = {
        "uncertainties": [],
        "visible_text": ["EXIT"],
        "temporal_changes": ["The door opens."],
    }
    ordinary = _context(
        observations=(observation,), budgets=(0.50,), planned_call={}
    )
    assert choose_budget("uncertainty_escalation", ordinary).ratio == 0.10

    confirmation = _context(
        observations=(observation,),
        budgets=(0.50,),
        planned_call={"purpose": "confirm_change"},
    )
    decision = choose_budget("uncertainty_escalation", confirmation)
    assert decision.ratio == 0.50
    assert decision.reason_code == "hold_change_confirmation_floor"


@pytest.mark.parametrize(
    "private_payload",
    (
        {"candidate_answer": "A"},
        {"time_range": [1, 2]},
        {"nested": {"clue_intervals": [[1, 2]]}},
        {"question_type": "OCR"},
        {"answer": "B"},
    ),
)
def test_budget_context_rejects_candidate_and_annotations(
    private_payload: dict,
) -> None:
    with pytest.raises(ValueError, match="private field"):
        _context(planned_call=private_payload)


def test_budget_context_rejects_private_fields_in_observations() -> None:
    with pytest.raises(ValueError, match="private field"):
        _context(
            observations=({"observed_facts": [], "ground_truth": "A"},),
            budgets=(0.10,),
        )

from __future__ import annotations

import pytest

from neural_decompiler.scale_comparison import (
    HistoricalConsistencyError,
    behavior_decision,
    compare_paired_cases,
    normalized_depths,
    paired_bootstrap_interval,
    validate_historical_consistency,
)


def _case(
    country: str,
    template: str,
    *,
    top1: bool,
    rank: int,
    logit: float,
    margin: float,
    block_logit_change: float = 4.0,
    block_selectivity_change: float = 1.0,
) -> dict[str, object]:
    return {
        "country": country,
        "template_id": template,
        "tokenized_prompt": ["The", " capital"],
        "token_ids": [1, 2],
        "correct_target": {"text": " Test", "id": 3},
        "control_targets": [
            {"text": " One", "id": 4},
            {"text": " Two", "id": 5},
            {"text": " Three", "id": 6},
        ],
        "final_model_prediction": {"token": " Test" if top1 else " the", "token_id": 3 if top1 else 7},
        "target_is_final_top1": top1,
        "candidate_stages": {
            "correct": [
                {"label": "after_layer_0", "target_logit": logit - block_logit_change, "target_rank": rank + 1},
                {"label": "after_layer_1", "target_logit": logit, "target_rank": rank},
            ]
        },
        "selectivity_stages": [
            {"label": "after_layer_0", "correct_minus_control_logit": margin - block_selectivity_change},
            {"label": "after_layer_1", "correct_minus_control_logit": margin},
        ],
        "final_metrics": {
            "final_correct_minus_control_logit": margin,
            "final_block_selectivity_change": block_selectivity_change,
            "final_block_target_logit_change": block_logit_change,
            "c001_final_block_replicated": True,
        },
    }


def test_normalized_depths_include_embedding_and_each_post_block() -> None:
    assert normalized_depths(6) == pytest.approx((0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0))
    assert normalized_depths(12)[-1] == 1.0
    assert len(normalized_depths(12)) == 13


def test_normalized_depths_reject_nonpositive_layer_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalized_depths(0)


def test_behavior_decision_enforces_frozen_canonical_boundary() -> None:
    assert behavior_decision(12, 24, baseline_top1_count=0)["criterion_met"] is True
    assert behavior_decision(11, 24, baseline_top1_count=0)["criterion_met"] is False
    assert behavior_decision(12, 24, baseline_top1_count=12)["criterion_met"] is False


def test_compare_paired_cases_reports_improvement_worsening_and_ties() -> None:
    baseline = [
        _case("A", "canonical", top1=False, rank=20, logit=10.0, margin=1.0),
        _case("B", "canonical", top1=True, rank=1, logit=12.0, margin=3.0),
        _case("C", "canonical", top1=False, rank=5, logit=11.0, margin=2.0),
    ]
    comparison = [
        _case("A", "canonical", top1=True, rank=2, logit=14.0, margin=4.0),
        _case("B", "canonical", top1=False, rank=4, logit=11.0, margin=2.0),
        _case("C", "canonical", top1=False, rank=5, logit=13.0, margin=2.0),
    ]

    result = compare_paired_cases(baseline, comparison)

    assert result["case_count"] == 3
    assert result["target_rank_improved_count"] == 1
    assert result["target_rank_worsened_count"] == 1
    assert result["target_rank_unchanged_count"] == 1
    assert result["top1_gained_count"] == 1
    assert result["top1_lost_count"] == 1
    assert result["median_target_rank_improvement"] == 0.0
    assert result["median_final_selectivity_margin_change"] == 0.0
    assert result["cases"][0]["target_rank_improvement"] == 18
    assert result["cases"][0]["final_selectivity_margin_change"] == 3.0


def test_compare_paired_cases_rejects_unmatched_case_keys() -> None:
    with pytest.raises(ValueError, match="same country-template keys"):
        compare_paired_cases(
            [_case("A", "canonical", top1=False, rank=2, logit=1.0, margin=1.0)],
            [_case("B", "canonical", top1=False, rank=2, logit=1.0, margin=1.0)],
        )


def test_paired_bootstrap_is_fixed_seed_and_preserves_constant_effect() -> None:
    first = paired_bootstrap_interval([2.0, 2.0, 2.0], seed=3003, resamples=10_000)
    second = paired_bootstrap_interval([2.0, 2.0, 2.0], seed=3003, resamples=10_000)
    assert first == second
    assert first == {
        "estimate": 2.0,
        "lower_95": 2.0,
        "upper_95": 2.0,
        "seed": 3003,
        "resamples": 10_000,
        "method": "paired bootstrap percentile interval for the median",
    }


def test_historical_consistency_accepts_matching_cases() -> None:
    cases = [_case("A", "canonical", top1=False, rank=2, logit=10.0, margin=1.0)]
    result = validate_historical_consistency(cases, cases, atol=1e-5, rtol=1e-5)
    assert result["passed"] is True
    assert result["case_count"] == 1


def test_historical_consistency_derives_raw_final_block_change_for_old_schema() -> None:
    historical = [_case("A", "canonical", top1=False, rank=2, logit=10.0, margin=1.0)]
    fresh = [_case("A", "canonical", top1=False, rank=2, logit=10.0, margin=1.0)]
    del historical[0]["final_metrics"]["final_block_target_logit_change"]

    result = validate_historical_consistency(historical, fresh, atol=1e-5, rtol=1e-5)

    assert result["passed"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda case: case.update(token_ids=[1, 99]), "tokenization"),
        (lambda case: case["final_model_prediction"].update(token_id=99), "top-1"),
        (lambda case: case["candidate_stages"]["correct"][-1].update(target_rank=99), "rank"),
        (lambda case: case["final_metrics"].update(final_correct_minus_control_logit=1.1), "metric"),
    ],
)
def test_historical_consistency_stops_on_material_difference(mutation, message: str) -> None:
    historical = [_case("A", "canonical", top1=False, rank=2, logit=10.0, margin=1.0)]
    fresh = [_case("A", "canonical", top1=False, rank=2, logit=10.0, margin=1.0)]
    mutation(fresh[0])
    with pytest.raises(HistoricalConsistencyError, match=message):
        validate_historical_consistency(historical, fresh, atol=1e-5, rtol=1e-5)

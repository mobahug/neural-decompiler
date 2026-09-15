from __future__ import annotations

import pytest

from neural_decompiler.logit_lens import StageMetric
from neural_decompiler.selectivity import (
    CaseOutcome,
    balanced_control_indices,
    calculate_selectivity_stages,
    summarize_outcomes,
)


def metric(label: str, logit: float, rank: int) -> StageMetric:
    return StageMetric(label, logit, rank, 0.0)


def test_balanced_controls_exclude_correct_target_and_balance_exposure() -> None:
    controls = balanced_control_indices(item_count=8, offsets=(1, 3))

    assert controls[0] == (1, 3)
    assert controls[7] == (0, 2)
    assert all(index not in row for index, row in enumerate(controls))
    appearances = [sum(target in row for row in controls) for target in range(8)]
    assert appearances == [2] * 8


@pytest.mark.parametrize("offsets", [(0, 1), (1, 1), (1, 8)])
def test_balanced_controls_reject_invalid_offsets(offsets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="offset"):
        balanced_control_indices(item_count=8, offsets=offsets)


def test_selectivity_stage_subtracts_mean_balanced_control_logit() -> None:
    correct = [metric("embedding", 10.0, 4), metric("after_layer_0", 14.0, 1)]
    controls = [
        [metric("embedding", 6.0, 8), metric("after_layer_0", 7.0, 7)],
        [metric("embedding", 8.0, 6), metric("after_layer_0", 10.0, 4)],
        [metric("embedding", 10.0, 4), metric("after_layer_0", 13.0, 2)],
    ]

    stages = calculate_selectivity_stages(correct, controls)

    assert stages[0].control_mean_logit == 8.0
    assert stages[0].correct_minus_control_logit == 2.0
    assert stages[1].control_mean_logit == 10.0
    assert stages[1].correct_minus_control_logit == 4.0
    assert stages[1].correct_beats_control_count == 3
    assert stages[1].control_count == 3


def test_selectivity_stages_reject_misaligned_labels() -> None:
    correct = [metric("embedding", 1.0, 1)]
    controls = [[metric("after_layer_0", 0.0, 2)]]

    with pytest.raises(ValueError, match="labels"):
        calculate_selectivity_stages(correct, controls)


def test_summary_applies_preregistered_criteria_per_template() -> None:
    outcomes = [
        CaseOutcome("A", "canonical", 4.0, 2.0, 3, 3, True, True),
        CaseOutcome("B", "canonical", 2.0, 1.0, 2, 3, False, True),
        CaseOutcome("C", "canonical", 1.0, 0.5, 2, 3, False, False),
        CaseOutcome("D", "canonical", -1.0, -0.5, 0, 3, False, True),
        CaseOutcome("A", "paraphrase", 3.0, 1.0, 3, 3, True, True),
        CaseOutcome("B", "paraphrase", 1.0, 0.5, 2, 3, False, True),
        CaseOutcome("C", "paraphrase", -1.0, -0.5, 1, 3, False, False),
        CaseOutcome("D", "paraphrase", -2.0, -1.0, 0, 3, False, False),
    ]

    summary = summarize_outcomes(outcomes, required_template_ids=("canonical", "paraphrase"))

    canonical = summary["by_template"]["canonical"]
    assert canonical["case_count"] == 4
    assert canonical["median_final_correct_minus_control_logit"] == 1.5
    assert canonical["median_final_block_selectivity_change"] == 0.75
    assert canonical["positive_final_margin_count"] == 3
    assert canonical["pairwise_correct_over_control_wins"] == 7
    assert canonical["pairwise_comparisons"] == 12
    assert canonical["intended_target_top1_count"] == 1
    assert canonical["c001_final_block_replication_count"] == 3
    assert canonical["criteria"]["positive_median_final_margin"] is True
    assert canonical["criteria"]["positive_median_final_block_change"] is True
    assert canonical["criteria"]["positive_margin_wilson_lower_above_half"] is False
    assert "positive_final_margin_wilson_95" not in summary["overall_descriptive"]
    assert "c001_final_block_replication_count" not in summary["overall_descriptive"]
    assert summary["overall_descriptive"]["inference_warning"] == (
        "Templates reuse the same countries and are not independent observations; "
        "inferential criteria are evaluated per template."
    )
    assert summary["primary_hypothesis_supported"] is False


def test_summary_locks_wilson_reference_and_template_conjunction() -> None:
    canonical = [
        CaseOutcome(str(index), "canonical", 1.0, 1.0, 3, 3, False, True)
        for index in range(24)
    ]
    paraphrase = [
        CaseOutcome(str(index), "paraphrase", 1.0, -1.0, 3, 3, False, True)
        for index in range(24)
    ]

    summary = summarize_outcomes(
        canonical + paraphrase,
        required_template_ids=("canonical", "paraphrase"),
    )

    assert summary["by_template"]["canonical"]["positive_final_margin_wilson_95"][
        "lower"
    ] == pytest.approx(0.8620237953)
    assert summary["by_template"]["canonical"]["criteria"][
        "template_supports_primary_hypothesis"
    ] is True
    assert summary["by_template"]["paraphrase"]["criteria"][
        "template_supports_primary_hypothesis"
    ] is False
    assert summary["primary_hypothesis_supported"] is False

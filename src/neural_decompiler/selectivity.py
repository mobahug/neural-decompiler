"""Interfaces for controlled target-selectivity measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Sequence

from neural_decompiler.logit_lens import StageMetric


@dataclass(frozen=True)
class CaseOutcome:
    country: str
    template_id: str
    final_margin: float
    final_block_change: float
    pairwise_wins: int
    pairwise_total: int
    target_is_top1: bool
    c001_replicated: bool


@dataclass(frozen=True)
class SelectivityStage:
    """Correct-target support relative to balanced incorrect targets."""

    label: str
    correct_logit: float
    control_mean_logit: float
    correct_minus_control_logit: float
    correct_rank: int
    best_control_rank: int
    correct_beats_control_count: int
    control_count: int


def balanced_control_indices(
    item_count: int,
    offsets: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Create cyclic derangements with equal control exposure."""

    if item_count < 2:
        raise ValueError("item_count must be at least 2")
    offsets = tuple(int(offset) for offset in offsets)
    if not offsets:
        raise ValueError("At least one control offset is required")
    if len(set(offsets)) != len(offsets) or any(
        offset <= 0 or offset >= item_count for offset in offsets
    ):
        raise ValueError(
            "Each control offset must be unique and between 1 and item_count - 1"
        )
    return tuple(
        tuple((index + offset) % item_count for offset in offsets)
        for index in range(item_count)
    )


def calculate_selectivity_stages(
    correct: Sequence[StageMetric],
    controls: Sequence[Sequence[StageMetric]],
) -> tuple[SelectivityStage, ...]:
    """Subtract the mean incorrect-target logit at every aligned stage."""

    if not correct:
        raise ValueError("Correct-target stages cannot be empty")
    if not controls:
        raise ValueError("At least one control trajectory is required")
    correct_labels = tuple(stage.label for stage in correct)
    for control in controls:
        if tuple(stage.label for stage in control) != correct_labels:
            raise ValueError("Correct and control stage labels must align exactly")

    rows: list[SelectivityStage] = []
    for stage_index, correct_stage in enumerate(correct):
        control_stages = [control[stage_index] for control in controls]
        control_mean = sum(stage.target_logit for stage in control_stages) / len(
            control_stages
        )
        rows.append(
            SelectivityStage(
                label=correct_stage.label,
                correct_logit=correct_stage.target_logit,
                control_mean_logit=control_mean,
                correct_minus_control_logit=correct_stage.target_logit
                - control_mean,
                correct_rank=correct_stage.target_rank,
                best_control_rank=min(stage.target_rank for stage in control_stages),
                correct_beats_control_count=sum(
                    correct_stage.target_logit > stage.target_logit
                    for stage in control_stages
                ),
                control_count=len(control_stages),
            )
        )
    return tuple(rows)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _summarize_group(outcomes: Sequence[CaseOutcome]) -> dict[str, object]:
    if not outcomes:
        raise ValueError("Cannot summarize an empty outcome group")
    final_margins = [outcome.final_margin for outcome in outcomes]
    final_block_changes = [outcome.final_block_change for outcome in outcomes]
    positive_count = sum(value > 0.0 for value in final_margins)
    lower, upper = _wilson_interval(positive_count, len(outcomes))
    wins = sum(outcome.pairwise_wins for outcome in outcomes)
    comparisons = sum(outcome.pairwise_total for outcome in outcomes)
    top1_count = sum(outcome.target_is_top1 for outcome in outcomes)
    replication_count = sum(outcome.c001_replicated for outcome in outcomes)
    positive_median_margin = median(final_margins) > 0.0
    positive_median_block_change = median(final_block_changes) > 0.0
    lower_above_half = lower > 0.5
    return {
        "case_count": len(outcomes),
        "median_final_correct_minus_control_logit": median(final_margins),
        "median_final_block_selectivity_change": median(final_block_changes),
        "positive_final_margin_count": positive_count,
        "positive_final_margin_fraction": positive_count / len(outcomes),
        "positive_final_margin_wilson_95": {"lower": lower, "upper": upper},
        "pairwise_correct_over_control_wins": wins,
        "pairwise_comparisons": comparisons,
        "pairwise_win_fraction": wins / comparisons,
        "intended_target_top1_count": top1_count,
        "intended_target_top1_fraction": top1_count / len(outcomes),
        "c001_final_block_replication_count": replication_count,
        "c001_final_block_replication_fraction": replication_count / len(outcomes),
        "criteria": {
            "positive_median_final_margin": positive_median_margin,
            "positive_median_final_block_change": positive_median_block_change,
            "positive_margin_wilson_lower_above_half": lower_above_half,
            "template_supports_primary_hypothesis": positive_median_margin
            and positive_median_block_change
            and lower_above_half,
        },
    }


def _summarize_paired_templates_descriptively(
    outcomes: Sequence[CaseOutcome],
) -> dict[str, object]:
    """Pool templates only for description, never for interval-based inference."""

    if not outcomes:
        raise ValueError("Cannot summarize an empty outcome group")
    wins = sum(outcome.pairwise_wins for outcome in outcomes)
    comparisons = sum(outcome.pairwise_total for outcome in outcomes)
    top1_count = sum(outcome.target_is_top1 for outcome in outcomes)
    positive_count = sum(outcome.final_margin > 0.0 for outcome in outcomes)
    return {
        "case_count": len(outcomes),
        "median_final_correct_minus_control_logit": median(
            outcome.final_margin for outcome in outcomes
        ),
        "median_final_block_selectivity_change": median(
            outcome.final_block_change for outcome in outcomes
        ),
        "positive_final_margin_count": positive_count,
        "positive_final_margin_fraction": positive_count / len(outcomes),
        "pairwise_correct_over_control_wins": wins,
        "pairwise_comparisons": comparisons,
        "pairwise_win_fraction": wins / comparisons,
        "intended_target_top1_count": top1_count,
        "intended_target_top1_fraction": top1_count / len(outcomes),
        "inference_warning": (
            "Templates reuse the same countries and are not independent observations; "
            "inferential criteria are evaluated per template."
        ),
    }


def summarize_outcomes(
    outcomes: Sequence[CaseOutcome],
    required_template_ids: Sequence[str],
) -> dict[str, object]:
    """Summarize outcomes and apply the preregistered per-template rule."""

    template_ids = tuple(required_template_ids)
    if not template_ids or len(set(template_ids)) != len(template_ids):
        raise ValueError("Required template IDs must be unique and non-empty")
    unexpected = {outcome.template_id for outcome in outcomes} - set(template_ids)
    if unexpected:
        raise ValueError(f"Unexpected template IDs: {sorted(unexpected)}")
    by_template: dict[str, dict[str, object]] = {}
    for template_id in template_ids:
        group = [outcome for outcome in outcomes if outcome.template_id == template_id]
        by_template[template_id] = _summarize_group(group)
    return {
        "overall_descriptive": _summarize_paired_templates_descriptively(outcomes),
        "by_template": by_template,
        "primary_hypothesis_supported": all(
            bool(summary["criteria"]["template_supports_primary_hypothesis"])
            for summary in by_template.values()
        ),
    }

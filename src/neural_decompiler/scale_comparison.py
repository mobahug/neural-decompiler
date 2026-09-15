"""Pure contracts for observational comparisons across model scales."""

from __future__ import annotations

from math import isclose
import random
from statistics import mean, median
from typing import Mapping, Sequence


class HistoricalConsistencyError(RuntimeError):
    """Raised when a fresh baseline run disagrees with its historical artifact."""


def normalized_depths(n_layers: int) -> tuple[float, ...]:
    """Return embedding depth 0 and one normalized depth per post-block stage."""

    if n_layers <= 0:
        raise ValueError("Layer count must be positive")
    return (0.0, *(index / n_layers for index in range(1, n_layers + 1)))


def behavior_decision(
    comparison_top1_count: int,
    case_count: int,
    *,
    baseline_top1_count: int,
    minimum_top1_count: int = 12,
) -> dict[str, object]:
    """Apply the frozen canonical behavioral-substrate decision boundary."""

    if case_count <= 0:
        raise ValueError("Case count must be positive")
    if not 0 <= comparison_top1_count <= case_count:
        raise ValueError("Comparison top-1 count must be within the case count")
    if not 0 <= baseline_top1_count <= case_count:
        raise ValueError("Baseline top-1 count must be within the case count")
    if not 0 <= minimum_top1_count <= case_count:
        raise ValueError("Minimum top-1 count must be within the case count")
    criterion_met = (
        comparison_top1_count >= minimum_top1_count
        and comparison_top1_count > baseline_top1_count
    )
    return {
        "template": "canonical",
        "minimum_comparison_top1_count": minimum_top1_count,
        "case_count": case_count,
        "baseline_top1_count": baseline_top1_count,
        "comparison_top1_count": comparison_top1_count,
        "comparison_exceeds_baseline": comparison_top1_count > baseline_top1_count,
        "criterion_met": criterion_met,
        "interpretation": (
            "Predeclared project decision criterion, not a universal scientific threshold."
        ),
    }


def _case_key(case: Mapping[str, object]) -> tuple[str, str]:
    return str(case["country"]), str(case["template_id"])


def _target_rank(case: Mapping[str, object]) -> int:
    candidate_stages = case["candidate_stages"]
    return int(candidate_stages["correct"][-1]["target_rank"])


def _final_margin(case: Mapping[str, object]) -> float:
    return float(case["final_metrics"]["final_correct_minus_control_logit"])


def _final_metric(case: Mapping[str, object], name: str) -> float:
    metrics = case["final_metrics"]
    if name in metrics:
        return float(metrics[name])
    if name == "final_block_target_logit_change":
        stages = case["candidate_stages"]["correct"]
        return float(stages[-1]["target_logit"]) - float(stages[-2]["target_logit"])
    raise KeyError(name)


def compare_paired_cases(
    baseline_cases: Sequence[Mapping[str, object]],
    comparison_cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare cases paired by country and template without pooling templates."""

    baseline = {_case_key(case): case for case in baseline_cases}
    comparison = {_case_key(case): case for case in comparison_cases}
    if len(baseline) != len(baseline_cases) or len(comparison) != len(comparison_cases):
        raise ValueError("Country-template keys must be unique within each model")
    if baseline.keys() != comparison.keys():
        raise ValueError("Models must contain the same country-template keys")
    if not baseline:
        raise ValueError("At least one paired case is required")

    rows: list[dict[str, object]] = []
    for key in baseline:
        baseline_case = baseline[key]
        comparison_case = comparison[key]
        baseline_rank = _target_rank(baseline_case)
        comparison_rank = _target_rank(comparison_case)
        rank_improvement = baseline_rank - comparison_rank
        margin_change = _final_margin(comparison_case) - _final_margin(baseline_case)
        baseline_top1 = bool(baseline_case["target_is_final_top1"])
        comparison_top1 = bool(comparison_case["target_is_final_top1"])
        rows.append(
            {
                "country": key[0],
                "template_id": key[1],
                "baseline_target_rank": baseline_rank,
                "comparison_target_rank": comparison_rank,
                "target_rank_improvement": rank_improvement,
                "baseline_final_selectivity_margin": _final_margin(baseline_case),
                "comparison_final_selectivity_margin": _final_margin(comparison_case),
                "final_selectivity_margin_change": margin_change,
                "baseline_target_top1": baseline_top1,
                "comparison_target_top1": comparison_top1,
                "top1_change": (
                    "gained" if comparison_top1 and not baseline_top1
                    else "lost" if baseline_top1 and not comparison_top1
                    else "unchanged"
                ),
            }
        )

    rank_changes = [int(row["target_rank_improvement"]) for row in rows]
    margin_changes = [float(row["final_selectivity_margin_change"]) for row in rows]
    return {
        "case_count": len(rows),
        "target_rank_improved_count": sum(value > 0 for value in rank_changes),
        "target_rank_worsened_count": sum(value < 0 for value in rank_changes),
        "target_rank_unchanged_count": sum(value == 0 for value in rank_changes),
        "top1_gained_count": sum(row["top1_change"] == "gained" for row in rows),
        "top1_lost_count": sum(row["top1_change"] == "lost" for row in rows),
        "top1_unchanged_count": sum(row["top1_change"] == "unchanged" for row in rows),
        "median_target_rank_improvement": float(median(rank_changes)),
        "mean_target_rank_improvement": float(mean(rank_changes)),
        "median_final_selectivity_margin_change": float(median(margin_changes)),
        "mean_final_selectivity_margin_change": float(mean(margin_changes)),
        "cases": rows,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_interval(
    paired_differences: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    """Return a deterministic percentile interval for the paired median effect."""

    values = tuple(float(value) for value in paired_differences)
    if not values:
        raise ValueError("At least one paired difference is required")
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    generator = random.Random(seed)
    sample_size = len(values)
    estimates = [
        float(median(values[generator.randrange(sample_size)] for _ in range(sample_size)))
        for _ in range(resamples)
    ]
    return {
        "estimate": float(median(values)),
        "lower_95": _percentile(estimates, 0.025),
        "upper_95": _percentile(estimates, 0.975),
        "seed": seed,
        "resamples": resamples,
        "method": "paired bootstrap percentile interval for the median",
    }


def _indexed_cases(
    cases: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
    indexed = {_case_key(case): case for case in cases}
    if len(indexed) != len(cases):
        raise HistoricalConsistencyError(f"Duplicate {label} country-template keys")
    return indexed


def validate_historical_consistency(
    historical_cases: Sequence[Mapping[str, object]],
    fresh_cases: Sequence[Mapping[str, object]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Require a fresh 70M run to reproduce the Experiment 002 artifact."""

    historical = _indexed_cases(historical_cases, label="historical")
    fresh = _indexed_cases(fresh_cases, label="fresh")
    if historical.keys() != fresh.keys():
        raise HistoricalConsistencyError(
            "Historical and fresh runs do not contain the same country-template keys"
        )

    metric_names = (
        "final_correct_minus_control_logit",
        "final_block_selectivity_change",
        "final_block_target_logit_change",
    )
    maximum_difference = 0.0
    for key, old in historical.items():
        new = fresh[key]
        if (
            old["tokenized_prompt"] != new["tokenized_prompt"]
            or old["token_ids"] != new["token_ids"]
            or old["correct_target"] != new["correct_target"]
            or old["control_targets"] != new["control_targets"]
        ):
            raise HistoricalConsistencyError(f"70M tokenization mismatch for {key}")
        if old["final_model_prediction"]["token_id"] != new["final_model_prediction"]["token_id"]:
            raise HistoricalConsistencyError(f"70M top-1 mismatch for {key}")
        if _target_rank(old) != _target_rank(new):
            raise HistoricalConsistencyError(f"70M target-rank mismatch for {key}")
        for metric_name in metric_names:
            old_value = _final_metric(old, metric_name)
            new_value = _final_metric(new, metric_name)
            difference = abs(new_value - old_value)
            maximum_difference = max(maximum_difference, difference)
            if not isclose(old_value, new_value, abs_tol=atol, rel_tol=rtol):
                raise HistoricalConsistencyError(
                    f"70M final metric mismatch for {key}: {metric_name} "
                    f"historical={old_value:.9g}, fresh={new_value:.9g}"
                )
    return {
        "passed": True,
        "case_count": len(fresh),
        "atol": atol,
        "rtol": rtol,
        "maximum_absolute_metric_difference": maximum_difference,
    }

#!/usr/bin/env python3
"""Run Experiment 002: held-out capital-target selectivity."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict
from datetime import datetime, timezone
import html
from importlib import metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any, Sequence

import torch

from neural_decompiler.logit_lens import (
    FinalProjectionMismatch,
    StageMetric,
    metrics_from_logits,
    project_cached_logits,
    validate_final_projection,
)
from neural_decompiler.selectivity import (
    CaseOutcome,
    balanced_control_indices,
    calculate_selectivity_stages,
    summarize_outcomes,
)


DEFAULT_MODEL = "EleutherAI/pythia-70m-deduped"
PINNED_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs/experiment-002"
CapitalPair = namedtuple("CapitalPair", "country capital")
CAPITAL_PAIRS = (
    CapitalPair("Spain", "Madrid"),
    CapitalPair("Portugal", "Lisbon"),
    CapitalPair("Austria", "Vienna"),
    CapitalPair("Belgium", "Brussels"),
    CapitalPair("Netherlands", "Amsterdam"),
    CapitalPair("Denmark", "Copenhagen"),
    CapitalPair("Norway", "Oslo"),
    CapitalPair("Sweden", "Stockholm"),
    CapitalPair("Poland", "Warsaw"),
    CapitalPair("Hungary", "Budapest"),
    CapitalPair("Greece", "Athens"),
    CapitalPair("Ireland", "Dublin"),
    CapitalPair("Canada", "Ottawa"),
    CapitalPair("Japan", "Tokyo"),
    CapitalPair("China", "Beijing"),
    CapitalPair("Russia", "Moscow"),
    CapitalPair("Egypt", "Cairo"),
    CapitalPair("Thailand", "Bangkok"),
    CapitalPair("Chile", "Santiago"),
    CapitalPair("Iraq", "Baghdad"),
    CapitalPair("Iran", "Tehran"),
    CapitalPair("Philippines", "Manila"),
    CapitalPair("Switzerland", "Bern"),
    CapitalPair("Syria", "Damascus"),
)
TEMPLATES = {
    "canonical": "The capital of {country} is",
    "paraphrase": "In {country}, the capital is",
}
CONTROL_OFFSETS = (1, 7, 13)
EXPERIMENT_001_COUNTRIES = ("France", "Germany", "Italy", "Finland")
C001_REPLICATION_MINIMUM = 18


def load_model() -> Any:
    """Load only the pinned Pythia-70M checkpoint on CPU."""

    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(
        DEFAULT_MODEL,
        revision=PINNED_REVISION,
        device="cpu",
    )


def control_pairs() -> tuple[tuple[str, ...], ...]:
    """Return three capital-token controls for each correct pair."""

    indices = balanced_control_indices(len(CAPITAL_PAIRS), CONTROL_OFFSETS)
    return tuple(
        tuple(CAPITAL_PAIRS[index].capital for index in row) for row in indices
    )


def _single_token_id(model: Any, token: str) -> int:
    try:
        return int(model.to_single_token(token))
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Protocol token {token!r} must be exactly one token") from error


def _token_strings(model: Any, tokens: torch.Tensor) -> list[str]:
    strings = model.to_str_tokens(tokens)
    if strings and isinstance(strings[0], list):
        strings = strings[0]
    return [str(token) for token in strings]


def validate_protocol_tokenization(model: Any) -> list[dict[str, object]]:
    """Validate all inclusion criteria before any next-token output is inspected."""

    validated: list[dict[str, object]] = []
    for pair in CAPITAL_PAIRS:
        country_token = f" {pair.country}"
        capital_token = f" {pair.capital}"
        country_id = _single_token_id(model, country_token)
        capital_id = _single_token_id(model, capital_token)
        prompt_tokens: dict[str, dict[str, object]] = {}
        for template_id, template in TEMPLATES.items():
            prompt = template.format(country=pair.country)
            tokens = model.to_tokens(prompt)
            prompt_tokens[template_id] = {
                "prompt": prompt,
                "tokenized_prompt": _token_strings(model, tokens),
                "token_ids": [int(token_id) for token_id in tokens[0].tolist()],
            }
        validated.append(
            {
                "country": pair.country,
                "capital": pair.capital,
                "country_token": {"text": country_token, "id": country_id},
                "capital_token": {"text": capital_token, "id": capital_id},
                "prompts": prompt_tokens,
            }
        )
    return validated


def _largest_positive_transition(
    stages: Sequence[StageMetric],
) -> tuple[str | None, bool]:
    deltas = [
        (previous.label, current.label, current.target_logit - previous.target_logit)
        for previous, current in zip(stages, stages[1:])
    ]
    positive = [transition for transition in deltas if transition[2] > 0.0]
    if not positive:
        return None, False
    from_stage, to_stage, _ = max(positive, key=lambda transition: transition[2])
    final_from = "embedding" if len(stages) == 2 else stages[-2].label
    replicated = from_stage == final_from and to_stage == stages[-1].label
    return f"{from_stage}_to_{to_stage}", replicated


def analyze_case(
    model: Any,
    pair: CapitalPair,
    control_capitals: Sequence[str],
    *,
    template_id: str,
    template: str,
) -> dict[str, object]:
    """Measure one held-out prompt against balanced incorrect capital tokens."""

    if len(control_capitals) != len(CONTROL_OFFSETS):
        raise ValueError(
            f"Expected {len(CONTROL_OFFSETS)} controls, received {len(control_capitals)}"
        )
    correct_text = f" {pair.capital}"
    correct_id = _single_token_id(model, correct_text)
    control_targets = [
        {"text": f" {capital}", "id": _single_token_id(model, f" {capital}")}
        for capital in control_capitals
    ]
    prompt = template.format(country=pair.country)
    tokens = model.to_tokens(prompt)
    with torch.inference_mode():
        actual_logits, cache = model.run_with_cache(tokens)
        actual_final_logits = actual_logits[0, -1, :]
        projected_logits, labels = project_cached_logits(model, cache)
        validation = validate_final_projection(
            projected_logits[-1], actual_final_logits, correct_id
        )
        if not validation.logits_allclose:
            raise FinalProjectionMismatch(
                "Final projected logits are outside the configured tolerance: "
                f"maximum absolute difference "
                f"{validation.max_abs_logit_difference:.9g}"
            )
        correct_stages = metrics_from_logits(projected_logits, correct_id, labels)
        control_stages = [
            metrics_from_logits(projected_logits, int(target["id"]), labels)
            for target in control_targets
        ]

    selectivity_stages = calculate_selectivity_stages(correct_stages, control_stages)
    final_stage = selectivity_stages[-1]
    final_block_change = (
        final_stage.correct_minus_control_logit
        - selectivity_stages[-2].correct_minus_control_logit
    )
    c001_transition, c001_replicated = _largest_positive_transition(correct_stages)
    actual_probabilities = torch.softmax(actual_final_logits.float(), dim=-1)
    top1_token_id = int(actual_final_logits.argmax().item())
    return {
        "country": pair.country,
        "capital": pair.capital,
        "template_id": template_id,
        "prompt": prompt,
        "tokenized_prompt": _token_strings(model, tokens),
        "token_ids": [int(token_id) for token_id in tokens[0].tolist()],
        "correct_target": {"text": correct_text, "id": correct_id},
        "control_targets": control_targets,
        "final_model_prediction": {
            "token": str(model.to_string(top1_token_id)),
            "token_id": top1_token_id,
            "logit": float(actual_final_logits[top1_token_id].item()),
            "probability": float(actual_probabilities[top1_token_id].item()),
        },
        "target_is_final_top1": top1_token_id == correct_id,
        "candidate_stages": {
            "correct": [asdict(stage) for stage in correct_stages],
            "controls": [
                {
                    "capital": capital,
                    "token": target,
                    "stages": [asdict(stage) for stage in stages],
                }
                for capital, target, stages in zip(
                    control_capitals, control_targets, control_stages, strict=True
                )
            ],
        },
        "selectivity_stages": [asdict(stage) for stage in selectivity_stages],
        "final_metrics": {
            "final_correct_minus_control_logit": final_stage.correct_minus_control_logit,
            "final_block_selectivity_change": final_block_change,
            "correct_beats_control_count": final_stage.correct_beats_control_count,
            "control_count": final_stage.control_count,
            "c001_largest_positive_change_transition": c001_transition,
            "c001_final_block_replicated": c001_replicated,
        },
        "final_projection_validation": asdict(validation),
    }


def _resolved_revision(model: Any) -> str | None:
    candidates = (model, getattr(model, "model", None), getattr(model, "hf_model", None))
    for candidate in candidates:
        revision = getattr(getattr(candidate, "config", None), "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def ensure_model_identity(model: Any) -> str:
    """Require the pinned checkpoint and raw TransformerBridge coordinate system."""

    resolved_revision = _resolved_revision(model)
    if resolved_revision != PINNED_REVISION:
        raise RuntimeError(
            "Pinned model revision mismatch: "
            f"expected {PINNED_REVISION}, received {resolved_revision}"
        )
    if bool(getattr(model, "compatibility_mode", False)):
        raise RuntimeError("TransformerBridge compatibility mode must remain disabled")
    return resolved_revision


def _dependency_versions() -> dict[str, str]:
    packages = {
        "transformer_lens": "transformer-lens",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "matplotlib": "matplotlib",
    }
    versions = {"python": sys.version.split()[0]}
    versions.update(
        {
            output_name: metadata.version(distribution_name)
            for output_name, distribution_name in packages.items()
        }
    )
    return versions


def _git_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": revision, "working_tree_dirty": dirty}


def _case_outcome(case: dict[str, object]) -> CaseOutcome:
    metrics = case["final_metrics"]
    return CaseOutcome(
        country=str(case["country"]),
        template_id=str(case["template_id"]),
        final_margin=float(metrics["final_correct_minus_control_logit"]),
        final_block_change=float(metrics["final_block_selectivity_change"]),
        pairwise_wins=int(metrics["correct_beats_control_count"]),
        pairwise_total=int(metrics["control_count"]),
        target_is_top1=bool(case["target_is_final_top1"]),
        c001_replicated=bool(metrics["c001_final_block_replicated"]),
    )


def _observations(summary: dict[str, object]) -> list[str]:
    support = bool(summary["primary_hypothesis_supported"])
    observations = [
        "The preregistered primary support rule was satisfied."
        if support
        else "The preregistered primary support rule was not satisfied."
    ]
    for template_id, template_summary in summary["by_template"].items():
        observations.append(
            f"For {template_id}, the median final correct-minus-control logit was "
            f"{template_summary['median_final_correct_minus_control_logit']:+.6f}, "
            "and the median final-block selectivity change was "
            f"{template_summary['median_final_block_selectivity_change']:+.6f}."
        )
    canonical = summary["by_template"]["canonical"]
    observations.append(
        "The Experiment 001 final-block raw-logit pattern occurred in "
        f"{canonical['c001_final_block_replication_count']} of "
        f"{canonical['case_count']} canonical held-out cases."
    )
    return observations


def build_results(model: Any) -> dict[str, object]:
    """Run the frozen protocol sequentially and assemble machine-readable results."""

    resolved_revision = ensure_model_identity(model)
    tokenization = validate_protocol_tokenization(model)
    controls = control_pairs()
    cases: list[dict[str, object]] = []
    for template_id, template in TEMPLATES.items():
        for pair, control_capitals in zip(CAPITAL_PAIRS, controls, strict=True):
            cases.append(
                analyze_case(
                    model, pair, control_capitals,
                    template_id=template_id, template=template,
                )
            )
    summary = summarize_outcomes(
        [_case_outcome(case) for case in cases],
        required_template_ids=tuple(TEMPLATES),
    )
    canonical = summary["by_template"]["canonical"]
    summary["c001_confirmatory_criterion"] = {
        "template": "canonical",
        "minimum_replications": C001_REPLICATION_MINIMUM,
        "observed_replications": canonical["c001_final_block_replication_count"],
        "case_count": canonical["case_count"],
        "criterion_met": canonical["c001_final_block_replication_count"]
        >= C001_REPLICATION_MINIMUM,
    }
    dtype = str(getattr(model.cfg, "dtype", torch.float32)).removeprefix("torch.")
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "id": "EXPERIMENT-002",
            "title": "Held-out Capital-Target Selectivity",
            "analysis_status": "confirmatory",
            "analysis_code": _git_metadata(),
            "original_experiment_001_countries_excluded": list(EXPERIMENT_001_COUNTRIES),
        },
        "model": {
            "name": DEFAULT_MODEL,
            "requested_revision": PINNED_REVISION,
            "resolved_revision": resolved_revision,
            "device": "cpu",
            "dtype": dtype,
            "bridge_type": type(model).__name__,
            "compatibility_mode": bool(getattr(model, "compatibility_mode", False)),
        },
        "dependencies": _dependency_versions(),
        "protocol": {
            "research_question": (
                "Does country context selectively increase relative support for the "
                "correct single-token capital over balanced incorrect capital tokens, "
                "and does C001 replicate on held-out country-capital pairs?"
            ),
            "templates": TEMPLATES,
            "capital_pairs": [pair._asdict() for pair in CAPITAL_PAIRS],
            "control_offsets": list(CONTROL_OFFSETS),
            "control_design": (
                "Cyclic derangements; every capital is correct twice (once per template) "
                "and appears as an incorrect control six times (three per template)."
            ),
            "primary_outcome": "correct_minus_balanced_control_logit",
            "primary_support_rule": (
                "For both templates independently: median final correct-minus-control "
                "logit > 0; median final-block selectivity change > 0; and the lower "
                "bound of the 95% Wilson interval for the fraction of cases with a "
                "positive final margin > 0.5."
            ),
            "c001_replication_rule": (
                "At least 18 of 24 canonical held-out cases have their largest positive "
                "adjacent-stage correct-target raw-logit change across the final block."
            ),
            "tokenization_inclusion_rule": (
                "Country and capital must each be one leading-space tokenizer token."
            ),
            "tokenization_preflight": tokenization,
        },
        "summary": summary,
        "cases": cases,
        "observations": _observations(summary),
    }


def write_results_json(result: dict[str, object], path: Path) -> None:
    """Write a stable, human-diffable machine result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_plots(result: dict[str, object], output_dir: Path) -> None:
    """Create stage-level and final-block selectivity SVGs."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = result["cases"]
    if not cases:
        raise ValueError("Cannot plot an experiment result without cases")
    labels = [stage["label"] for stage in cases[0]["selectivity_stages"]]
    x_values = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for template_id in result["protocol"]["templates"]:
        template_cases = [case for case in cases if case["template_id"] == template_id]
        if not template_cases:
            continue
        values_by_stage = [
            [case["selectivity_stages"][index]["correct_minus_control_logit"] for case in template_cases]
            for index in range(len(labels))
        ]
        medians = [median(values) for values in values_by_stage]
        lower = [_percentile(values, 0.25) for values in values_by_stage]
        upper = [_percentile(values, 0.75) for values in values_by_stage]
        (line,) = axis.plot(x_values, medians, marker="o", label=template_id)
        axis.fill_between(x_values, lower, upper, color=line.get_color(), alpha=0.15)
    axis.axhline(0.0, color="#333333", linewidth=1, linestyle="--")
    axis.set_title("Experiment 002: correct-target selectivity by residual stage")
    axis.set_xlabel("Residual-stream stage")
    axis.set_ylabel("Correct minus balanced-control logit (median; band = IQR)")
    axis.set_xticks(x_values, labels, rotation=30, ha="right")
    axis.grid(alpha=0.25)
    axis.legend(title="Template")
    figure.savefig(output_dir / "selectivity-by-stage.svg", format="svg")
    plt.close(figure)

    countries = list(dict.fromkeys(case["country"] for case in cases))
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    offsets = {"canonical": -0.13, "paraphrase": 0.13}
    for template_id in result["protocol"]["templates"]:
        template_cases = [case for case in cases if case["template_id"] == template_id]
        if not template_cases:
            continue
        by_country = {case["country"]: case for case in template_cases}
        xs = [index + offsets.get(template_id, 0.0) for index, country in enumerate(countries) if country in by_country]
        ys = [
            by_country[country]["final_metrics"]["final_block_selectivity_change"]
            for country in countries if country in by_country
        ]
        axis.scatter(xs, ys, s=42, label=template_id, alpha=0.85)
    axis.axhline(0.0, color="#333333", linewidth=1, linestyle="--")
    axis.set_title("Experiment 002: final-block change in target selectivity")
    axis.set_xlabel("Held-out country")
    axis.set_ylabel("Change in correct-minus-control logit, final block")
    axis.set_xticks(range(len(countries)), countries, rotation=55, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Template")
    figure.savefig(output_dir / "final-block-selectivity.svg", format="svg")
    plt.close(figure)


def _inline_svg(path: Path) -> str:
    svg = path.read_text(encoding="utf-8")
    return svg[svg.index("<svg") :]


def _case_table_rows(cases: Sequence[dict[str, object]]) -> str:
    rows = []
    for case in cases:
        prediction = case["final_model_prediction"]
        metrics = case["final_metrics"]
        validation = case["final_projection_validation"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(case['template_id'])}</td>"
            f"<td>{html.escape(case['country'])}</td>"
            f"<td>{html.escape(case['capital'])}</td>"
            f"<td>{html.escape(repr(prediction['token']))}</td>"
            f"<td>{'yes' if case['target_is_final_top1'] else 'no'}</td>"
            f"<td>{metrics['final_correct_minus_control_logit']:+.6f}</td>"
            f"<td>{metrics['final_block_selectivity_change']:+.6f}</td>"
            f"<td>{metrics['correct_beats_control_count']}/{metrics['control_count']}</td>"
            f"<td>{'yes' if metrics['c001_final_block_replicated'] else 'no'}</td>"
            f"<td>{validation['max_abs_logit_difference']:.3g}</td>"
            "</tr>"
        )
    return "".join(rows)


def _case_details(case: dict[str, object]) -> str:
    stage_rows = "".join(
        "<tr>"
        f"<td>{html.escape(stage['label'])}</td>"
        f"<td>{stage['correct_logit']:.6f}</td>"
        f"<td>{stage['control_mean_logit']:.6f}</td>"
        f"<td>{stage['correct_minus_control_logit']:+.6f}</td>"
        f"<td>{stage['correct_rank']}</td>"
        f"<td>{stage['best_control_rank']}</td>"
        "</tr>"
        for stage in case["selectivity_stages"]
    )
    validation = case["final_projection_validation"]
    controls = ", ".join(
        f"{target['text'].strip()} (ID {target['id']})"
        for target in case["control_targets"]
    )
    token_pairs = ", ".join(
        f"{html.escape(repr(token))} ({token_id})"
        for token, token_id in zip(case["tokenized_prompt"], case["token_ids"], strict=True)
    )
    return f"""
    <details>
      <summary>{html.escape(case['template_id'])}: {html.escape(case['prompt'])}</summary>
      <p><strong>Prompt tokens:</strong> {token_pairs}</p>
      <p><strong>Correct target:</strong> {html.escape(repr(case['correct_target']['text']))}
         (ID {case['correct_target']['id']}); <strong>controls:</strong> {html.escape(controls)}</p>
      <table>
        <thead><tr><th>Stage</th><th>Correct logit</th><th>Control mean</th><th>Margin</th><th>Correct rank</th><th>Best control rank</th></tr></thead>
        <tbody>{stage_rows}</tbody>
      </table>
      <h4>Final-projection validation</h4>
      <p>Target-rank agreement: {validation['target_rank_matches']};
         top-1 agreement: {validation['top1_token_id_matches']};
         logits within tolerance: {validation['logits_allclose']};
         maximum absolute difference: {validation['max_abs_logit_difference']:.9g}.</p>
    </details>
    """


def write_html_report(result: dict[str, object], output_dir: Path) -> None:
    """Create a self-contained HTML report with both SVG figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_svg = _inline_svg(output_dir / "selectivity-by-stage.svg")
    final_svg = _inline_svg(output_dir / "final-block-selectivity.svg")
    summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(template_id)}</td>"
        f"<td>{summary['case_count']}</td>"
        f"<td>{summary['median_final_correct_minus_control_logit']:+.6f}</td>"
        f"<td>{summary['median_final_block_selectivity_change']:+.6f}</td>"
        f"<td>{summary['positive_final_margin_count']}/{summary['case_count']}</td>"
        f"<td>{summary['positive_final_margin_wilson_95']['lower']:.3f}–{summary['positive_final_margin_wilson_95']['upper']:.3f}</td>"
        f"<td>{summary['intended_target_top1_count']}/{summary['case_count']}</td>"
        f"<td>{'yes' if summary['criteria']['template_supports_primary_hypothesis'] else 'no'}</td>"
        "</tr>"
        for template_id, summary in result["summary"]["by_template"].items()
    )
    observations = "".join(
        f"<li>{html.escape(observation)}</li>" for observation in result["observations"]
    )
    details = "".join(_case_details(case) for case in result["cases"])
    model = result["model"]
    support = bool(result["summary"]["primary_hypothesis_supported"])
    outcome = "SUPPORTED" if support else "NOT SUPPORTED"
    outcome_class = "ok" if support else "warning"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment 002: Held-out Capital-Target Selectivity</title>
  <style>
    body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 1180px; margin: 2rem auto; padding: 0 1rem; color: #18212b; }}
    h1, h2, h3, h4 {{ line-height: 1.2; }}
    section {{ border-top: 1px solid #ccd5df; margin-top: 2.5rem; padding-top: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
    th, td {{ border: 1px solid #ccd5df; padding: .4rem .5rem; text-align: right; }}
    th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align: left; }}
    .notice {{ background: #eef5ff; border-left: 4px solid #3977b8; padding: .8rem 1rem; }}
    .ok {{ background: #e6f5eb; color: #176b36; padding: .7rem 1rem; }}
    .warning {{ background: #fff3cd; color: #6b4d00; padding: .7rem 1rem; }}
    .chart svg {{ width: 100%; height: auto; }}
    details {{ margin: .7rem 0; padding: .6rem; border: 1px solid #d9e0e7; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    code {{ background: #f3f5f7; padding: .1rem .25rem; }}
  </style>
</head>
<body>
  <h1>Experiment 002: Held-out Capital-Target Selectivity</h1>
  <p><strong>Model:</strong> {html.escape(model['name'])} at
     <code>{html.escape(str(model['resolved_revision']))}</code>
     ({html.escape(model['device'])}, {html.escape(model['dtype'])})</p>
  <p><strong>Run:</strong> {html.escape(result['run_timestamp_utc'])}</p>
  <p class="{outcome_class}"><strong>Preregistered primary hypothesis: {outcome}.</strong></p>
  <div class="notice">
    <strong>Interpretation limit.</strong> The primary quantity is the
    correct minus balanced-control logit: the correct target's logit minus the mean
    logit of three balanced incorrect capital-token controls.
    This controlled observational/input-comparison study does not identify an internal causal mechanism,
    factual storage location, attention head, MLP, or circuit.
  </div>
  <section>
    <h2>Protocol and decision rule</h2>
    <p>{html.escape(result['protocol']['research_question'])}</p>
    <p><strong>Primary rule:</strong> {html.escape(result['protocol']['primary_support_rule'])}</p>
    <p>The 24 countries are held out from Experiment 001. Every capital token appears
       equally often as a correct target and as an incorrect control within each template.</p>
  </section>
  <section>
    <h2>Summary</h2>
    <table>
      <thead><tr><th>Template</th><th>Cases</th><th>Median final margin</th><th>Median final-block change</th><th>Positive margins</th><th>Wilson 95%</th><th>Correct top-1</th><th>Primary rule</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
    <ul>{observations}</ul>
  </section>
  <div class="chart"><h2>Selectivity through layers</h2>{stage_svg}</div>
  <div class="chart"><h2>Final-block selectivity changes</h2>{final_svg}</div>
  <section>
    <h2>Case results</h2>
    <table>
      <thead><tr><th>Template</th><th>Country</th><th>Capital</th><th>Actual top-1</th><th>Target top-1?</th><th>Final margin</th><th>Final-block change</th><th>Control wins</th><th>C001 pattern?</th><th>Parity max diff</th></tr></thead>
      <tbody>{_case_table_rows(result['cases'])}</tbody>
    </table>
  </section>
  <section>
    <h2>Per-case stage measurements</h2>
    {details}
  </section>
</body>
</html>
"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def print_trace(result: dict[str, object]) -> None:
    """Print concise human-readable case and aggregate measurements."""

    print(f"Model: {result['model']['name']}")
    print(f"Revision: {result['model']['resolved_revision']}")
    for case in result["cases"]:
        metrics = case["final_metrics"]
        prediction = case["final_model_prediction"]
        validation = case["final_projection_validation"]
        print(f"\n[{case['template_id']}] {case['prompt']}")
        print(
            f"  Correct target: {case['correct_target']['text']!r}; "
            f"actual top-1: {prediction['token']!r}; "
            f"target_top1={case['target_is_final_top1']}"
        )
        print(
            "  Final correct-minus-control logit: "
            f"{metrics['final_correct_minus_control_logit']:+.6f}"
        )
        print(
            "  Final-block selectivity change: "
            f"{metrics['final_block_selectivity_change']:+.6f}; "
            f"control wins={metrics['correct_beats_control_count']}/{metrics['control_count']}"
        )
        print(
            "  Final projection: "
            f"allclose={validation['logits_allclose']}, "
            f"rank_match={validation['target_rank_matches']}, "
            f"top1_match={validation['top1_token_id_matches']}, "
            f"max_abs_difference={validation['max_abs_logit_difference']:.9g}"
        )
    print("\nAggregate observations:")
    for observation in result["observations"]:
        print(f"  - {observation}")


def main() -> None:
    model = load_model()
    result = build_results(model)
    write_results_json(result, OUTPUT_DIR / "results.json")
    write_plots(result, OUTPUT_DIR)
    write_html_report(result, OUTPUT_DIR)
    print_trace(result)
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

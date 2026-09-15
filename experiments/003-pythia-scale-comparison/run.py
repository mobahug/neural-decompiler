#!/usr/bin/env python3
"""Run Experiment 003: observational Pythia scale-and-behavior comparison."""

from __future__ import annotations

import gc
import html
import importlib.util
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any, NamedTuple, Sequence

import torch

from neural_decompiler.scale_comparison import (
    behavior_decision,
    compare_paired_cases,
    normalized_depths,
    paired_bootstrap_interval,
    validate_historical_consistency,
)
from neural_decompiler.selectivity import CaseOutcome, summarize_outcomes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-003"
HISTORICAL_RESULTS_PATH = ROOT / "outputs/experiment-002/results.json"
BOOTSTRAP_SEED = 3003
BOOTSTRAP_RESAMPLES = 10_000
HISTORICAL_ATOL = 1e-5
HISTORICAL_RTOL = 1e-5
BEHAVIOR_MINIMUM = 12


class ModelSpec(NamedTuple):
    key: str
    name: str
    revision: str


MODEL_SPECS = (
    ModelSpec(
        "pythia-70m-deduped",
        "EleutherAI/pythia-70m-deduped",
        "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
    ),
    ModelSpec(
        "pythia-160m-deduped",
        "EleutherAI/pythia-160m-deduped",
        "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
    ),
)


def _load_experiment_002() -> Any:
    path = ROOT / "experiments/002-capital-selectivity/run.py"
    spec = importlib.util.spec_from_file_location("experiment_002_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Experiment 002 analysis path from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPERIMENT_002 = _load_experiment_002()
CAPITAL_PAIRS = EXPERIMENT_002.CAPITAL_PAIRS
TEMPLATES = EXPERIMENT_002.TEMPLATES
CONTROL_OFFSETS = EXPERIMENT_002.CONTROL_OFFSETS


def load_model(spec: ModelSpec) -> Any:
    """Load one pinned model in raw TransformerBridge coordinates on CPU."""

    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(
        spec.name,
        revision=spec.revision,
        device="cpu",
        dtype=torch.float32,
    )


def _resolved_revision(model: Any) -> str | None:
    candidates = (model, getattr(model, "model", None), getattr(model, "hf_model", None))
    for candidate in candidates:
        revision = getattr(getattr(candidate, "config", None), "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def ensure_model_identity(model: Any, spec: ModelSpec) -> str:
    """Require the requested revision and disabled compatibility mode."""

    resolved = _resolved_revision(model)
    if resolved != spec.revision:
        raise RuntimeError(
            f"Pinned revision mismatch for {spec.name}: expected {spec.revision}, "
            f"received {resolved}"
        )
    if bool(getattr(model, "compatibility_mode", False)):
        raise RuntimeError("TransformerBridge compatibility mode must remain disabled")
    return resolved


def _adjacent_extrema(stages: Sequence[dict[str, object]]) -> dict[str, object]:
    transitions = [
        {
            "from_stage": previous["label"],
            "to_stage": current["label"],
            "change": float(current["target_logit"]) - float(previous["target_logit"]),
        }
        for previous, current in zip(stages, stages[1:])
    ]
    if not transitions:
        raise ValueError("At least two residual stages are required")
    return {
        "largest_positive": max(transitions, key=lambda row: float(row["change"])),
        "largest_negative": min(transitions, key=lambda row: float(row["change"])),
    }


def _decorate_case(case: dict[str, object], n_layers: int) -> dict[str, object]:
    depths = normalized_depths(n_layers)
    correct_stages = case["candidate_stages"]["correct"]
    if len(correct_stages) != len(depths):
        raise ValueError("Residual stage count does not match model layer count")
    for stage, depth in zip(correct_stages, depths, strict=True):
        stage["normalized_depth"] = depth
    for control in case["candidate_stages"]["controls"]:
        for stage, depth in zip(control["stages"], depths, strict=True):
            stage["normalized_depth"] = depth
    for stage, depth in zip(case["selectivity_stages"], depths, strict=True):
        stage["normalized_depth"] = depth

    extrema = _adjacent_extrema(correct_stages)
    case["final_metrics"]["final_block_target_logit_change"] = (
        float(correct_stages[-1]["target_logit"])
        - float(correct_stages[-2]["target_logit"])
    )
    case["final_metrics"]["largest_positive_target_logit_change"] = extrema["largest_positive"]
    case["final_metrics"]["largest_negative_target_logit_change"] = extrema["largest_negative"]
    case["final_target"] = {
        "logit": float(correct_stages[-1]["target_logit"]),
        "rank": int(correct_stages[-1]["target_rank"]),
        "probability": float(correct_stages[-1]["logit_lens_probability"]),
    }
    return case


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


def build_model_result(model: Any, spec: ModelSpec) -> dict[str, object]:
    """Run the same Experiment 002 analysis path for one model."""

    resolved = ensure_model_identity(model, spec)
    tokenization = EXPERIMENT_002.validate_protocol_tokenization(model)
    controls = EXPERIMENT_002.control_pairs()
    n_layers = int(model.cfg.n_layers)
    cases: list[dict[str, object]] = []
    for template_id, template in TEMPLATES.items():
        for pair, control_capitals in zip(CAPITAL_PAIRS, controls, strict=True):
            case = EXPERIMENT_002.analyze_case(
                model,
                pair,
                control_capitals,
                template_id=template_id,
                template=template,
            )
            cases.append(_decorate_case(case, n_layers))
    summary = summarize_outcomes(
        [_case_outcome(case) for case in cases],
        required_template_ids=tuple(TEMPLATES),
    )
    dtype = str(getattr(model.cfg, "dtype", torch.float32)).removeprefix("torch.")
    return {
        "model": {
            "id": spec.key,
            "name": spec.name,
            "requested_revision": spec.revision,
            "resolved_revision": resolved,
            "device": "cpu",
            "dtype": dtype,
            "n_layers": n_layers,
            "bridge_type": type(model).__name__,
            "compatibility_mode": bool(getattr(model, "compatibility_mode", False)),
        },
        "tokenization_preflight": tokenization,
        "summary": summary,
        "cases": cases,
    }


def _dependency_versions() -> dict[str, str]:
    packages = {
        "transformer_lens": "transformer-lens",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "matplotlib": "matplotlib",
    }
    versions = {"python": sys.version.split()[0]}
    versions.update({name: metadata.version(distribution) for name, distribution in packages.items()})
    return versions


def _git_metadata() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": revision, "working_tree_dirty": dirty}


def _with_bootstrap(paired: dict[str, object], *, seed: int) -> dict[str, object]:
    rows = paired["cases"]
    paired["bootstrap"] = {
        "target_rank_improvement": paired_bootstrap_interval(
            [float(row["target_rank_improvement"]) for row in rows],
            seed=seed,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "final_selectivity_margin_change": paired_bootstrap_interval(
            [float(row["final_selectivity_margin_change"]) for row in rows],
            seed=seed,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
    }
    return paired


def assemble_results(
    baseline: dict[str, object],
    comparison: dict[str, object],
    *,
    historical_consistency: dict[str, object],
    timestamp: str | None = None,
) -> dict[str, object]:
    """Assemble paired, template-separated cross-model results."""

    paired_by_template = {}
    for template_index, template_id in enumerate(TEMPLATES):
        baseline_cases = [case for case in baseline["cases"] if case["template_id"] == template_id]
        comparison_cases = [case for case in comparison["cases"] if case["template_id"] == template_id]
        paired_by_template[template_id] = _with_bootstrap(
            compare_paired_cases(baseline_cases, comparison_cases),
            seed=BOOTSTRAP_SEED + template_index,
        )

    baseline_key = str(baseline["model"]["id"])
    comparison_key = str(comparison["model"]["id"])
    baseline_canonical = baseline["summary"]["by_template"]["canonical"]
    comparison_canonical = comparison["summary"]["by_template"]["canonical"]
    decision = behavior_decision(
        int(comparison_canonical["intended_target_top1_count"]),
        int(comparison_canonical["case_count"]),
        baseline_top1_count=int(baseline_canonical["intended_target_top1_count"]),
        minimum_top1_count=BEHAVIOR_MINIMUM,
    )
    c001 = {
        model_result["model"]["id"]: {
            "canonical_count": model_result["summary"]["by_template"]["canonical"]["c001_final_block_replication_count"],
            "canonical_case_count": model_result["summary"]["by_template"]["canonical"]["case_count"],
            "paraphrase_count": model_result["summary"]["by_template"]["paraphrase"]["c001_final_block_replication_count"],
            "paraphrase_case_count": model_result["summary"]["by_template"]["paraphrase"]["case_count"],
        }
        for model_result in (baseline, comparison)
    }
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "id": "EXPERIMENT-003",
            "title": "Pythia Scale-and-Behavior Comparison",
            "analysis_status": "confirmatory observational comparison",
            "analysis_code": _git_metadata(),
        },
        "dependencies": _dependency_versions(),
        "protocol": {
            "research_question": (
                "Does scaling from Pythia-70M to Pythia-160M improve factual-capital "
                "completion and correct-target selectivity, and does C001 persist?"
            ),
            "models_processed_sequentially": [baseline_key, comparison_key],
            "templates": TEMPLATES,
            "capital_pairs": [pair._asdict() for pair in CAPITAL_PAIRS],
            "control_offsets": list(CONTROL_OFFSETS),
            "controls_per_case": 3,
            "normalized_depth": "embedding=0; post-block i=(i+1)/n_layers; final block=1",
            "behavioral_decision_rule": (
                "Pythia-160M is a substantially better project substrate only if the "
                "canonical intended target is top-1 for at least 12 of 24 cases and "
                "the count exceeds Pythia-70M. This is a project decision threshold."
            ),
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "interpretation": (
                    "Descriptive paired uncertainty only; 24 countries are a small, "
                    "tokenization-constrained convenience sample."
                ),
            },
            "interpretation_boundary": (
                "Observational only. Normalized-depth alignment does not imply "
                "functional equivalence, and scale differences are not causal localization."
            ),
        },
        "historical_70m_consistency": historical_consistency,
        "models": {baseline_key: baseline, comparison_key: comparison},
        "summary": {
            "behavioral_substrate_decision": decision,
            "paired_comparison_by_template": paired_by_template,
            "c001_by_model": c001,
        },
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def write_plots(result: dict[str, object], output_dir: Path) -> None:
    """Write the three frozen SVG comparisons."""

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model_results = list(result["models"].values())
    labels = [model_result["model"]["id"] for model_result in model_results]

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.34
    for offset, template_id in ((-width / 2, "canonical"), (width / 2, "paraphrase")):
        values = [model_result["summary"]["by_template"][template_id]["intended_target_top1_count"] for model_result in model_results]
        axis.bar([value + offset for value in x], values, width=width, label=template_id)
    axis.axhline(BEHAVIOR_MINIMUM, color="#9a3412", linestyle="--", label="canonical threshold")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 24)
    axis.set_ylabel("Intended capital top-1 count (of 24)")
    axis.set_title("Experiment 003: actual next-token behavior")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "behavior-comparison.svg", format="svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, template_id in zip(axes, TEMPLATES, strict=True):
        paired = result["summary"]["paired_comparison_by_template"][template_id]["cases"]
        axis.scatter(
            [row["baseline_final_selectivity_margin"] for row in paired],
            [row["comparison_final_selectivity_margin"] for row in paired],
            alpha=0.8,
        )
        values = [
            float(row[key])
            for row in paired
            for key in ("baseline_final_selectivity_margin", "comparison_final_selectivity_margin")
        ]
        lower, upper = min(values), max(values)
        axis.plot([lower, upper], [lower, upper], color="#555", linestyle="--")
        axis.set_title(template_id)
        axis.set_xlabel("70M final selectivity margin")
        axis.set_ylabel("160M final selectivity margin")
        axis.grid(alpha=0.25)
    figure.suptitle("Correct-capital selectivity across model scale")
    figure.savefig(output_dir / "selectivity-comparison.svg", format="svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, template_id in zip(axes, TEMPLATES, strict=True):
        for model_result in model_results:
            cases = [case for case in model_result["cases"] if case["template_id"] == template_id]
            stage_count = len(cases[0]["selectivity_stages"])
            depths = [cases[0]["selectivity_stages"][index]["normalized_depth"] for index in range(stage_count)]
            stage_medians = [
                median(case["selectivity_stages"][index]["correct_minus_control_logit"] for case in cases)
                for index in range(stage_count)
            ]
            axis.plot(depths, stage_medians, marker="o", label=model_result["model"]["id"])
        axis.axhline(0, color="#555", linestyle="--")
        axis.set_title(template_id)
        axis.set_xlabel("Normalized depth (descriptive only)")
        axis.set_ylabel("Median correct-minus-control logit")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Selectivity trajectories at normalized model depth")
    figure.savefig(output_dir / "normalized-depth-trajectories.svg", format="svg")
    plt.close(figure)


def _inline_svg(path: Path) -> str:
    svg = path.read_text(encoding="utf-8")
    return svg[svg.index("<svg"):]


def write_html_report(result: dict[str, object], output_dir: Path) -> None:
    """Write a self-contained report separating behavior, selectivity, and trajectory."""

    decision = result["summary"]["behavioral_substrate_decision"]
    model_rows = []
    for model_id, model_result in result["models"].items():
        canonical = model_result["summary"]["by_template"]["canonical"]
        paraphrase = model_result["summary"]["by_template"]["paraphrase"]
        model_rows.append(
            f"<tr><td>{html.escape(model_id)}</td><td>{model_result['model']['n_layers']}</td>"
            f"<td>{canonical['intended_target_top1_count']}/24</td>"
            f"<td>{paraphrase['intended_target_top1_count']}/24</td>"
            f"<td>{canonical['median_final_correct_minus_control_logit']:+.6f}</td>"
            f"<td>{paraphrase['median_final_correct_minus_control_logit']:+.6f}</td></tr>"
        )
    comparison_rows = []
    for template_id, comparison in result["summary"]["paired_comparison_by_template"].items():
        rank_ci = comparison["bootstrap"]["target_rank_improvement"]
        margin_ci = comparison["bootstrap"]["final_selectivity_margin_change"]
        comparison_rows.append(
            f"<tr><td>{html.escape(template_id)}</td>"
            f"<td>{comparison['median_target_rank_improvement']:+.3f}</td>"
            f"<td>[{rank_ci['lower_95']:+.3f}, {rank_ci['upper_95']:+.3f}]</td>"
            f"<td>{comparison['median_final_selectivity_margin_change']:+.6f}</td>"
            f"<td>[{margin_ci['lower_95']:+.6f}, {margin_ci['upper_95']:+.6f}]</td>"
            f"<td>{comparison['target_rank_improved_count']}/{comparison['target_rank_worsened_count']}/{comparison['target_rank_unchanged_count']}</td></tr>"
        )
    c001_rows = "".join(
        f"<tr><td>{html.escape(model_id)}</td><td>{summary['canonical_count']}/{summary['canonical_case_count']}</td>"
        f"<td>{summary['paraphrase_count']}/{summary['paraphrase_case_count']}</td></tr>"
        for model_id, summary in result["summary"]["c001_by_model"].items()
    )
    pair_rows = "".join(
        f"<tr><td>{html.escape(template_id)}</td><td>{html.escape(str(row['country']))}</td>"
        f"<td>{row['baseline_target_rank']}</td><td>{row['comparison_target_rank']}</td>"
        f"<td>{row['target_rank_improvement']:+d}</td>"
        f"<td>{row['baseline_final_selectivity_margin']:+.6f}</td>"
        f"<td>{row['comparison_final_selectivity_margin']:+.6f}</td>"
        f"<td>{row['final_selectivity_margin_change']:+.6f}</td><td>{row['top1_change']}</td></tr>"
        for template_id, comparison in result["summary"]["paired_comparison_by_template"].items()
        for row in comparison["cases"]
    )
    status = "PASSED" if decision["criterion_met"] else "NOT PASSED"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 003: Pythia Scale-and-Behavior Comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#17202a}}section{{border-top:1px solid #ccd5df;margin-top:2rem;padding-top:1rem}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccd5df;padding:.4rem;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.notice{{background:#eef5ff;border-left:4px solid #3977b8;padding:.8rem 1rem}}.outcome{{background:#fff3cd;padding:.8rem 1rem}}.chart svg{{width:100%;height:auto}}code{{background:#f3f5f7;padding:.1rem .25rem}}</style></head><body>
<h1>Experiment 003: Pythia Scale-and-Behavior Comparison</h1>
<p class="outcome"><strong>Behavioral substrate criterion: {status}.</strong> 160M produced the intended capital for {decision['comparison_top1_count']} of 24 canonical prompts; the requirement was at least {decision['minimum_comparison_top1_count']} and more than 70M's {decision['baseline_top1_count']}.</p>
<div class="notice"><strong>Interpretation boundary:</strong> This experiment is observational. Model-scale differences do not show that additional layers caused a behavior, identify a storage location or circuit, or make normalized depths functionally equivalent.</div>
<section><h2>BEHAVIOR</h2><p>Does each model actually produce the intended capital as its next token?</p><table><thead><tr><th>Model</th><th>Layers</th><th>Canonical top-1</th><th>Paraphrase top-1</th><th>Canonical median margin</th><th>Paraphrase median margin</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table><div class="chart">{_inline_svg(output_dir / 'behavior-comparison.svg')}</div></section>
<section><h2>SELECTIVITY</h2><p>Correct target logit minus the mean logit of three balanced incorrect capital controls. Bootstrap intervals are descriptive paired intervals over 24 convenience-sampled countries.</p><table><thead><tr><th>Template</th><th>Median rank improvement</th><th>Bootstrap 95%</th><th>Median margin change</th><th>Bootstrap 95%</th><th>Rank improved/worse/tied</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table><div class="chart">{_inline_svg(output_dir / 'selectivity-comparison.svg')}</div></section>
<section><h2>TRAJECTORY</h2><p>Normalized depth is descriptive only: embedding=0 and post-block i=(i+1)/n_layers.</p><div class="chart">{_inline_svg(output_dir / 'normalized-depth-trajectories.svg')}</div></section>
<section><h2>GENERALIZATION</h2><p>C001 counts cases whose largest positive adjacent-stage raw target-logit change occurs across the final block. This does not measure selectivity or causality.</p><table><thead><tr><th>Model</th><th>Canonical</th><th>Paraphrase</th></tr></thead><tbody>{c001_rows}</tbody></table></section>
<section><h2>Paired cases</h2><table><thead><tr><th>Template</th><th>Country</th><th>70M rank</th><th>160M rank</th><th>Rank improvement</th><th>70M margin</th><th>160M margin</th><th>Margin change</th><th>Top-1 change</th></tr></thead><tbody>{pair_rows}</tbody></table></section>
</body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    """Write JSON, SVGs, then a self-contained HTML report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_plots(result, output_dir)
    write_html_report(result, output_dir)


def _load_historical_results() -> dict[str, object]:
    if not HISTORICAL_RESULTS_PATH.exists():
        raise RuntimeError(
            "Experiment 002 results are required for the frozen 70M consistency check. "
            "Run experiments/002-capital-selectivity/run.py first."
        )
    return json.loads(HISTORICAL_RESULTS_PATH.read_text(encoding="utf-8"))


def print_trace(result: dict[str, object]) -> None:
    decision = result["summary"]["behavioral_substrate_decision"]
    for model_id, model_result in result["models"].items():
        print(f"\n{model_id} ({model_result['model']['n_layers']} layers)")
        for template_id, summary in model_result["summary"]["by_template"].items():
            print(
                f"  {template_id}: top-1={summary['intended_target_top1_count']}/24; "
                f"median final margin={summary['median_final_correct_minus_control_logit']:+.6f}"
            )
    print(f"\nBehavioral substrate criterion: {'PASSED' if decision['criterion_met'] else 'NOT PASSED'}")
    print(f"Historical 70M consistency: {result['historical_70m_consistency']['passed']}")
    print(f"Wrote results to {OUTPUT_DIR}")


def main() -> None:
    historical = _load_historical_results()

    baseline_model = load_model(MODEL_SPECS[0])
    baseline = build_model_result(baseline_model, MODEL_SPECS[0])
    consistency = validate_historical_consistency(
        historical["cases"],
        baseline["cases"],
        atol=HISTORICAL_ATOL,
        rtol=HISTORICAL_RTOL,
    )
    del baseline_model
    gc.collect()

    comparison_model = load_model(MODEL_SPECS[1])
    comparison = build_model_result(comparison_model, MODEL_SPECS[1])
    del comparison_model
    gc.collect()

    result = assemble_results(
        baseline,
        comparison,
        historical_consistency=consistency,
    )
    write_outputs(result, OUTPUT_DIR)
    print_trace(result)


if __name__ == "__main__":
    main()

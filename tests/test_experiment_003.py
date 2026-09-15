from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
runner = _load(
    ROOT / "experiments/003-pythia-scale-comparison/run.py",
    "experiment_003_runner",
)
experiment_002 = _load(
    ROOT / "experiments/002-capital-selectivity/run.py",
    "experiment_002_reference",
)


def _case(country: str, template_id: str, *, top1: bool, rank: int, margin: float) -> dict[str, object]:
    return {
        "country": country,
        "capital": f"Capital{country}",
        "template_id": template_id,
        "target_is_final_top1": top1,
        "candidate_stages": {
            "correct": [
                {"label": "embedding", "normalized_depth": 0.0, "target_logit": 1.0, "target_rank": rank + 1},
                {"label": "after_layer_0", "normalized_depth": 1.0, "target_logit": 3.0, "target_rank": rank},
            ]
        },
        "selectivity_stages": [
            {"label": "embedding", "normalized_depth": 0.0, "correct_minus_control_logit": margin - 1.0},
            {"label": "after_layer_0", "normalized_depth": 1.0, "correct_minus_control_logit": margin},
        ],
        "final_metrics": {
            "final_correct_minus_control_logit": margin,
            "final_block_selectivity_change": 1.0,
            "final_block_target_logit_change": 2.0,
            "c001_final_block_replicated": True,
        },
    }


def _model_result(model_id: str, *, canonical_top1: int, rank_shift: int, margin_shift: float) -> dict[str, object]:
    cases = []
    for template_id in ("canonical", "paraphrase"):
        for index in range(24):
            cases.append(
                _case(
                    f"Country{index}",
                    template_id,
                    top1=index < (canonical_top1 if template_id == "canonical" else 3),
                    rank=20 - rank_shift,
                    margin=1.0 + margin_shift,
                )
            )
    return {
        "model": {"id": model_id, "name": model_id, "n_layers": 1},
        "summary": {
            "by_template": {
                "canonical": {"case_count": 24, "intended_target_top1_count": canonical_top1, "c001_final_block_replication_count": 24, "median_final_correct_minus_control_logit": 1.0 + margin_shift},
                "paraphrase": {"case_count": 24, "intended_target_top1_count": 3, "c001_final_block_replication_count": 20, "median_final_correct_minus_control_logit": 1.0 + margin_shift},
            }
        },
        "cases": cases,
    }


def test_model_specs_pin_only_70m_and_160m() -> None:
    assert [(spec.key, spec.name, spec.revision) for spec in runner.MODEL_SPECS] == [
        ("pythia-70m-deduped", "EleutherAI/pythia-70m-deduped", "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"),
        ("pythia-160m-deduped", "EleutherAI/pythia-160m-deduped", "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e"),
    ]


def test_protocol_reuses_experiment_002_pairs_templates_and_controls() -> None:
    assert runner.CAPITAL_PAIRS == experiment_002.CAPITAL_PAIRS
    assert runner.TEMPLATES == experiment_002.TEMPLATES
    assert runner.CONTROL_OFFSETS == experiment_002.CONTROL_OFFSETS


def test_model_identity_requires_exact_revision_and_raw_bridge_coordinates() -> None:
    spec = runner.MODEL_SPECS[1]
    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash=spec.revision)),
        compatibility_mode=False,
    )
    assert runner.ensure_model_identity(model, spec) == spec.revision

    model.model.config._commit_hash = "wrong"
    with pytest.raises(RuntimeError, match="Pinned revision mismatch"):
        runner.ensure_model_identity(model, spec)

    model.model.config._commit_hash = spec.revision
    model.compatibility_mode = True
    with pytest.raises(RuntimeError, match="compatibility mode"):
        runner.ensure_model_identity(model, spec)


def test_assemble_results_applies_behavior_rule_and_paired_bootstrap() -> None:
    baseline = _model_result("pythia-70m-deduped", canonical_top1=0, rank_shift=0, margin_shift=0.0)
    comparison = _model_result("pythia-160m-deduped", canonical_top1=12, rank_shift=5, margin_shift=2.0)

    result = runner.assemble_results(
        baseline,
        comparison,
        historical_consistency={"passed": True},
        timestamp="2026-09-15T00:00:00+00:00",
    )

    assert result["experiment"]["id"] == "EXPERIMENT-003"
    assert result["summary"]["behavioral_substrate_decision"]["criterion_met"] is True
    canonical = result["summary"]["paired_comparison_by_template"]["canonical"]
    assert canonical["target_rank_improved_count"] == 24
    assert canonical["median_target_rank_improvement"] == 5.0
    assert canonical["median_final_selectivity_margin_change"] == 2.0
    assert canonical["bootstrap"]["target_rank_improvement"]["resamples"] == 10_000
    assert canonical["bootstrap"]["target_rank_improvement"]["seed"] == 3003
    assert result["summary"]["c001_by_model"]["pythia-70m-deduped"]["canonical_count"] == 24
    assert result["summary"]["c001_by_model"]["pythia-160m-deduped"]["canonical_count"] == 24


def test_write_outputs_creates_json_three_svgs_and_self_contained_html(tmp_path: Path) -> None:
    result = runner.assemble_results(
        _model_result("pythia-70m-deduped", canonical_top1=0, rank_shift=0, margin_shift=0.0),
        _model_result("pythia-160m-deduped", canonical_top1=12, rank_shift=5, margin_shift=2.0),
        historical_consistency={"passed": True},
        timestamp="2026-09-15T00:00:00+00:00",
    )

    runner.write_outputs(result, tmp_path)

    assert json.loads((tmp_path / "results.json").read_text())["experiment"]["id"] == "EXPERIMENT-003"
    for filename in (
        "behavior-comparison.svg",
        "selectivity-comparison.svg",
        "normalized-depth-trajectories.svg",
    ):
        assert (tmp_path / filename).read_text().lstrip().startswith("<?xml")
    report = (tmp_path / "report.html").read_text()
    assert "BEHAVIOR" in report
    assert "SELECTIVITY" in report
    assert "TRAJECTORY" in report
    assert "GENERALIZATION" in report
    assert "observational" in report.lower()
    assert "<svg" in report

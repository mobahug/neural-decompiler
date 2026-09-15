from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def _load_runner() -> ModuleType:
    path = ROOT / "experiments/004-prompt-elicitation/run.py"
    spec = importlib.util.spec_from_file_location("experiment_004_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _fake_case(country: str, capital: str, format_id: str) -> dict[str, object]:
    rank = int(format_id[1:])
    return {
        "country": country,
        "capital": capital,
        "format_id": format_id,
        "prompt": f"prompt:{format_id}:{country}",
        "tokenized_prompt": [country],
        "token_ids": [1],
        "correct_target": {"text": f" {capital}", "id": 2},
        "control_targets": [],
        "actual_top1": {"text": f" {capital}" if format_id == "F4" else " the", "id": 2 if format_id == "F4" else 3},
        "final_model_prediction": {
            "token": f" {capital}" if format_id == "F4" else " the",
            "token_id": 2 if format_id == "F4" else 3,
            "logit": 3.0,
            "probability": 0.2,
        },
        "target_is_final_top1": format_id == "F4",
        "final_target": {"logit": 2.0, "rank": rank, "probability": 0.02},
        "candidate_stages": {
            "correct": [
                {"label": "embedding", "target_logit": 1.0, "target_rank": rank + 1, "logit_lens_probability": 0.01},
                {"label": "after_layer_0", "target_logit": 2.0, "target_rank": rank, "logit_lens_probability": 0.02},
            ],
            "controls": [],
        },
        "selectivity_stages": [
            {"label": "embedding", "correct_minus_control_logit": 0.0},
            {"label": "after_layer_0", "correct_minus_control_logit": float(6 - rank)},
        ],
        "final_metrics": {
            "final_correct_minus_control_logit": float(6 - rank),
            "final_block_selectivity_change": float(6 - rank),
            "c001_final_block_replicated": True,
        },
        "final_projection_validation": {
            "logits_allclose": True,
            "target_rank_matches": True,
            "top1_token_id_matches": True,
            "max_abs_logit_difference": 0.0,
        },
    }


def test_cli_has_separate_phases_and_no_heldout_format_override() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(["development"]).phase == "development"
    assert parser.parse_args(["heldout"]).phase == "heldout"
    with pytest.raises(SystemExit):
        parser.parse_args(["heldout", "--format", "F4"])


def test_model_identity_requires_pinned_revision_and_raw_coordinates() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash=runner.MODEL_REVISION)),
        compatibility_mode=False,
    )
    assert runner.ensure_model_identity(model) == runner.MODEL_REVISION
    model.model.config._commit_hash = "wrong"
    with pytest.raises(RuntimeError, match="Pinned revision mismatch"):
        runner.ensure_model_identity(model)
    model.model.config._commit_hash = runner.MODEL_REVISION
    model.compatibility_mode = True
    with pytest.raises(RuntimeError, match="compatibility mode"):
        runner.ensure_model_identity(model)


def test_development_analysis_runs_exactly_sixty_cases() -> None:
    calls: list[tuple[str, str]] = []

    def analyze(_model: object, pair: object, _controls: object, *, format_id: str, template: str) -> dict[str, object]:
        assert "{country}" in template
        calls.append((pair.country, format_id))
        return _fake_case(pair.country, pair.capital, format_id)

    cases = runner.analyze_cases(
        object(),
        runner.DEVELOPMENT_PAIRS,
        [item.id for item in runner.PROMPT_FORMATS],
        analyzer=analyze,
        n_layers=1,
    )
    summaries = runner.summarize_formats(cases)

    assert len(cases) == 60
    assert len(set(calls)) == 60
    assert {row["format_id"] for row in summaries} == {"F1", "F2", "F3", "F4", "F5"}
    assert next(row for row in summaries if row["format_id"] == "F4")["top1_count"] == 12


def test_heldout_analysis_uses_only_format_from_validated_lock() -> None:
    calls: list[str] = []

    def analyze(_model: object, pair: object, _controls: object, *, format_id: str, template: str) -> dict[str, object]:
        calls.append(format_id)
        return _fake_case(pair.country, pair.capital, format_id)

    cases = runner.analyze_locked_heldout(
        object(),
        {"selected_format": "F4"},
        analyzer=analyze,
        n_layers=1,
    )

    assert len(cases) == 12
    assert calls == ["F4"] * 12


def test_development_writer_creates_artifact_and_candidate_lock(tmp_path: Path) -> None:
    cases = [
        _fake_case(pair.country, pair.capital, format_id)
        for format_id in ("F1", "F2", "F3", "F4", "F5")
        for pair in runner.DEVELOPMENT_PAIRS
    ]
    result = runner.assemble_development_results(
        cases,
        resolved_revision=runner.MODEL_REVISION,
        timestamp="2026-09-15T00:00:00+00:00",
        git_commit="e" * 40,
    )

    artifact, candidate = runner.write_development_outputs(result, tmp_path)

    assert json.loads(artifact.read_text())["phase"] == "development"
    lock = json.loads(candidate.read_text())
    assert lock["selected_format"] == "F4"
    assert lock["development_results"]["sha256"] == runner.sha256_file(artifact)
    assert candidate.name == "selection-lock.candidate.json"


def test_heldout_refuses_when_tracked_lock_is_missing(tmp_path: Path) -> None:
    with pytest.raises(runner.SelectionLockError, match="missing"):
        runner.load_verified_lock(
            lock_path=tmp_path / "selection-lock.json",
            development_path=tmp_path / "development-results.json",
        )


@pytest.mark.parametrize(
    ("successes", "label"),
    [(6, "strong_support"), (3, "partial_insufficient"), (0, "heldout_failure")],
)
def test_heldout_decision_uses_frozen_top1_threshold(successes: int, label: str) -> None:
    cases = [_fake_case(pair.country, pair.capital, "F4") for pair in runner.HELDOUT_PAIRS]
    for index, case in enumerate(cases):
        case["target_is_final_top1"] = index < successes
    summary = runner.summarize_heldout(cases)
    assert summary["top1_count"] == successes
    assert summary["decision"] == label
    assert summary["strong_support_threshold"] == 6


def test_final_outputs_include_three_svgs_and_conditional_report(tmp_path: Path) -> None:
    development_cases = [
        _fake_case(pair.country, pair.capital, format_id)
        for format_id in ("F1", "F2", "F3", "F4", "F5")
        for pair in runner.DEVELOPMENT_PAIRS
    ]
    development = runner.assemble_development_results(
        development_cases,
        resolved_revision=runner.MODEL_REVISION,
        timestamp="2026-09-15T00:00:00+00:00",
        git_commit="e" * 40,
    )
    heldout = [_fake_case(pair.country, pair.capital, "F4") for pair in runner.HELDOUT_PAIRS]
    result = runner.assemble_final_results(
        development,
        {"selected_format": "F4", "protocol_code_commit": "e" * 40},
        heldout,
        timestamp="2026-09-15T01:00:00+00:00",
        git_commit="f" * 40,
    )

    runner.write_final_outputs(result, tmp_path)

    stored = json.loads((tmp_path / "results.json").read_text())
    assert stored["summary"]["heldout"]["top1_count"] == 12
    assert stored["summary"]["heldout"]["decision"] == "strong_support"
    for filename in (
        "development-prompt-comparison.svg",
        "heldout-behavior.svg",
        "heldout-selectivity.svg",
    ):
        assert (tmp_path / filename).read_text().lstrip().startswith("<?xml")
    report = (tmp_path / "report.html").read_text()
    assert "Conditionally confirmatory" in report
    assert "12/12" in report
    assert "C001 remains observational" in report
    assert "does not identify a causal mechanism" in report
    assert report.count("<svg") == 3

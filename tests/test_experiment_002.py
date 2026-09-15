from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from neural_decompiler.logit_lens import FinalProjectionMismatch

def load_runner() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/002-capital-selectivity/run.py"
    spec = importlib.util.spec_from_file_location("experiment_002_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def sample_result() -> dict[str, object]:
    validation = {
        "max_abs_logit_difference": 0.0,
        "mean_actual_minus_projected_offset": 0.0,
        "max_abs_difference_after_offset": 0.0,
        "atol": 0.00001,
        "rtol": 0.00001,
        "logits_allclose": True,
        "projected_target_rank": 2,
        "actual_target_rank": 2,
        "target_rank_matches": True,
        "projected_top1_token_id": 9,
        "actual_top1_token_id": 9,
        "top1_token_id_matches": True,
    }
    stages = [
        {
            "label": "after_layer_4",
            "correct_logit": 10.0,
            "control_mean_logit": 9.5,
            "correct_minus_control_logit": 0.5,
            "correct_rank": 4,
            "best_control_rank": 5,
            "correct_beats_control_count": 2,
            "control_count": 3,
        },
        {
            "label": "after_layer_5",
            "correct_logit": 14.0,
            "control_mean_logit": 12.5,
            "correct_minus_control_logit": 1.5,
            "correct_rank": 2,
            "best_control_rank": 3,
            "correct_beats_control_count": 3,
            "control_count": 3,
        },
    ]
    template_summary = {
        "case_count": 1,
        "median_final_correct_minus_control_logit": 1.5,
        "median_final_block_selectivity_change": 1.0,
        "positive_final_margin_count": 1,
        "positive_final_margin_fraction": 1.0,
        "positive_final_margin_wilson_95": {"lower": 0.206549, "upper": 1.0},
        "pairwise_correct_over_control_wins": 3,
        "pairwise_comparisons": 3,
        "pairwise_win_fraction": 1.0,
        "intended_target_top1_count": 0,
        "intended_target_top1_fraction": 0.0,
        "c001_final_block_replication_count": 1,
        "c001_final_block_replication_fraction": 1.0,
        "criteria": {
            "positive_median_final_margin": True,
            "positive_median_final_block_change": True,
            "positive_margin_wilson_lower_above_half": False,
            "template_supports_primary_hypothesis": False,
        },
    }
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": "2026-09-15T12:00:00+00:00",
        "experiment": {
            "id": "EXPERIMENT-002",
            "analysis_status": "confirmatory",
            "original_experiment_001_countries_excluded": [
                "France",
                "Germany",
                "Italy",
                "Finland",
            ],
        },
        "model": {
            "name": "EleutherAI/pythia-70m-deduped",
            "requested_revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
            "resolved_revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
            "device": "cpu",
            "dtype": "float32",
        },
        "dependencies": {"python": "3.12.13", "torch": "2.14.0"},
        "protocol": {
            "research_question": "Does the country context selectively support its capital?",
            "templates": {
                "canonical": "The capital of {country} is",
                "paraphrase": "In {country}, the capital is",
            },
            "control_offsets": [1, 7, 13],
            "primary_outcome": "correct_minus_balanced_control_logit",
            "primary_support_rule": "The rule must pass for both templates.",
        },
        "summary": {
            "by_template": {
                "canonical": template_summary,
                "paraphrase": template_summary,
            },
            "primary_hypothesis_supported": False,
            "c001_confirmatory_criterion": {
                "template": "canonical",
                "minimum_replications": 18,
                "observed_replications": 24,
                "case_count": 24,
                "criterion_met": True,
            },
        },
        "cases": [
            {
                "country": "Spain",
                "capital": "Madrid",
                "template_id": "canonical",
                "prompt": "The capital of Spain is",
                "tokenized_prompt": ["The", " capital", " of", " Spain", " is"],
                "token_ids": [1, 2, 3, 4, 5],
                "correct_target": {"text": " Madrid", "id": 6},
                "control_targets": [
                    {"text": " Lisbon", "id": 7},
                    {"text": " Tokyo", "id": 8},
                    {"text": " Moscow", "id": 10},
                ],
                "final_model_prediction": {
                    "token": " the",
                    "token_id": 9,
                    "logit": 15.0,
                    "probability": 0.2,
                },
                "target_is_final_top1": False,
                "candidate_stages": {},
                "selectivity_stages": stages,
                "final_metrics": {
                    "final_correct_minus_control_logit": 1.5,
                    "final_block_selectivity_change": 1.0,
                    "correct_beats_control_count": 3,
                    "control_count": 3,
                    "c001_largest_positive_change_transition": "after_layer_4_to_after_layer_5",
                    "c001_final_block_replicated": True,
                },
                "final_projection_validation": validation,
            }
        ],
        "observations": [
            "The preregistered primary support rule was not satisfied."
        ],
    }


def test_protocol_has_24_held_out_pairs_and_balanced_controls() -> None:
    assert len(runner.CAPITAL_PAIRS) == 24
    assert not ({"France", "Germany", "Italy", "Finland"} & {
        pair.country for pair in runner.CAPITAL_PAIRS
    })
    assert runner.CONTROL_OFFSETS == (1, 7, 13)
    controls = runner.control_pairs()
    assert all(pair.capital not in row for pair, row in zip(runner.CAPITAL_PAIRS, controls, strict=True))
    appearances = {
        pair.capital: sum(pair.capital in row for row in controls)
        for pair in runner.CAPITAL_PAIRS
    }
    assert set(appearances.values()) == {3}


class OneLayerCache:
    def accumulated_resid(
        self,
        *,
        return_labels: bool,
        apply_ln: bool,
    ) -> tuple[torch.Tensor, list[str]]:
        assert return_labels is True
        assert apply_ln is True
        return (
            torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
            ["0_pre", "final_post"],
        )


class FakeBridge:
    cfg = SimpleNamespace(n_layers=1, dtype=torch.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
        )
    )
    token_ids = {
        " Madrid": 1,
        " Lisbon": 2,
        " Tokyo": 3,
        " Moscow": 4,
    }

    def to_single_token(self, token: str) -> int:
        return self.token_ids[token]

    def to_tokens(self, prompt: str) -> torch.Tensor:
        assert prompt == "The capital of Spain is"
        return torch.tensor([[10, 11, 12, 13, 14]])

    def to_str_tokens(self, tokens: torch.Tensor) -> list[str]:
        return ["The", " capital", " of", " Spain", " is"]

    def to_string(self, token_id: int) -> str:
        return {1: " Madrid", 2: " Lisbon", 3: " Tokyo", 4: " Moscow", 5: " the"}.get(token_id, "x")

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            [
                [0.0, 2.0, 1.0, 0.0, -1.0, 3.0],
                [0.0, 4.0, 2.0, 3.0, 1.0, 5.0],
            ]
        )
        return residual @ weights

    def run_with_cache(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, OneLayerCache]:
        assert torch.is_grad_enabled() is False
        logits = torch.zeros((1, 5, 6))
        logits[0, -1, :] = torch.tensor([0.0, 4.0, 2.0, 3.0, 1.0, 5.0])
        return logits, OneLayerCache()


def test_analyze_case_measures_correct_target_against_three_controls() -> None:
    pair = runner.CapitalPair("Spain", "Madrid")

    result = runner.analyze_case(
        FakeBridge(),
        pair,
        ("Lisbon", "Tokyo", "Moscow"),
        template_id="canonical",
        template="The capital of {country} is",
    )

    assert result["correct_target"] == {"text": " Madrid", "id": 1}
    assert [target["id"] for target in result["control_targets"]] == [2, 3, 4]
    assert result["final_metrics"] == {
        "final_correct_minus_control_logit": 2.0,
        "final_block_selectivity_change": 0.0,
        "correct_beats_control_count": 3,
        "control_count": 3,
        "c001_largest_positive_change_transition": "embedding_to_after_layer_0",
        "c001_final_block_replicated": True,
    }
    assert result["target_is_final_top1"] is False
    assert result["final_model_prediction"]["token"] == " the"
    assert result["final_projection_validation"]["logits_allclose"] is True


class OffsetFakeBridge(FakeBridge):
    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return super().unembed(residual) + 0.25


def test_analyze_case_stops_when_raw_final_logits_are_not_allclose() -> None:
    with pytest.raises(FinalProjectionMismatch, match="outside the configured tolerance"):
        runner.analyze_case(
            OffsetFakeBridge(),
            runner.CapitalPair("Spain", "Madrid"),
            ("Lisbon", "Tokyo", "Moscow"),
            template_id="canonical",
            template="The capital of {country} is",
        )


def test_model_identity_requires_the_exact_pinned_revision() -> None:
    wrong = FakeBridge()
    wrong.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="wrong"))

    with pytest.raises(RuntimeError, match="Pinned model revision mismatch"):
        runner.ensure_model_identity(wrong)


def test_c001_replication_rule_locks_17_versus_18_case_boundary() -> None:
    assert runner.evaluate_c001_replication(17, 24)["criterion_met"] is False
    assert runner.evaluate_c001_replication(18, 24)["criterion_met"] is True


def test_results_json_preserves_protocol_and_case_measurements(tmp_path: Path) -> None:
    result = sample_result()

    runner.write_results_json(result, tmp_path / "results.json")

    decoded = json.loads((tmp_path / "results.json").read_text())
    assert decoded["experiment"]["analysis_status"] == "confirmatory"
    assert decoded["protocol"]["control_offsets"] == [1, 7, 13]
    assert decoded["cases"][0]["final_metrics"]["final_block_selectivity_change"] == 1.0
    assert decoded["cases"][0]["final_projection_validation"]["logits_allclose"] is True


def test_report_writers_create_selectivity_svgs_and_self_contained_html(tmp_path: Path) -> None:
    result = sample_result()

    runner.write_plots(result, tmp_path)
    runner.write_html_report(result, tmp_path)

    stage_svg = (tmp_path / "selectivity-by-stage.svg").read_text()
    final_svg = (tmp_path / "final-block-selectivity.svg").read_text()
    report = (tmp_path / "report.html").read_text()
    assert "<svg" in stage_svg
    assert "<svg" in final_svg
    assert "Experiment 002: Held-out Capital-Target Selectivity" in report
    assert "The capital of Spain is" in report
    assert "correct minus balanced-control" in report
    assert "does not identify an internal causal mechanism" in report
    assert "Final-projection validation" in report
    assert "24 of 24" in report
    assert "required at least 18" in report
    assert "criterion was met" in report

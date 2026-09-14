from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from neural_decompiler.logit_lens import StageMetric


def load_runner() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/001-capital-recall/run.py"
    spec = importlib.util.spec_from_file_location("experiment_001_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class RejectingTargetBridge:
    def to_single_token(self, target: str) -> int:
        raise ValueError("not one token")


def test_analyze_prompt_rejects_a_multi_token_target() -> None:
    with pytest.raises(
        ValueError,
        match="Target ' Paris' must tokenize to exactly one token",
    ):
        runner.analyze_prompt(
            RejectingTargetBridge(),
            "The capital of France is",
            " Paris",
        )


def test_observations_report_measured_extrema_without_causal_language() -> None:
    stages = [
        StageMetric("embedding", 0.0, 10, 0.01),
        StageMetric("after_layer_0", 2.0, 3, 0.10),
        StageMetric("after_layer_1", 1.5, 5, 0.05),
    ]

    observations = runner.derive_observations(" Paris", stages)

    assert observations == [
        "The Paris target logit increased most strongly from embedding to after_layer_0 (+2.000000).",
        "The Paris target logit decreased most strongly from after_layer_0 to after_layer_1 (-0.500000).",
        "The Paris target rank improved most strongly from embedding to after_layer_0 (10 to 3).",
        "The Paris target rank worsened most strongly from after_layer_0 to after_layer_1 (3 to 5).",
    ]
    words = set(" ".join(observations).lower().replace(".", "").split())
    assert not ({"stored", "retrieved", "caused"} & words)


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
            torch.tensor(
                [
                    [[[1.0, 0.0], [1.0, 0.0]]],
                    [[[0.0, 1.0], [0.0, 1.0]]],
                ]
            ),
            ["0_pre", "final_post"],
        )


class SuccessfulFakeBridge:
    cfg = SimpleNamespace(n_layers=1, model_name="fake-bridge")

    def to_single_token(self, target: str) -> int:
        assert target == " Paris"
        return 1

    def to_tokens(self, prompt: str) -> torch.Tensor:
        assert prompt == "The capital of France is"
        return torch.tensor([[10, 11]])

    def to_str_tokens(self, tokens: torch.Tensor) -> list[str]:
        assert tokens.tolist() == [[10, 11]]
        return ["The capital", " of France is"]

    def to_string(self, token_id: int) -> str:
        return {0: " London", 1: " Paris", 2: " Rome"}[token_id]

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
        bias = torch.tensor([0.5, -0.5, 1.0])
        return residual @ weight + bias

    def run_with_cache(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, OneLayerCache]:
        assert torch.is_grad_enabled() is False
        assert tokens.tolist() == [[10, 11]]
        logits = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 2.5, 2.0]]])
        return logits, OneLayerCache()


def test_analyze_prompt_records_tokens_actual_prediction_and_projection() -> None:
    result = runner.analyze_prompt(
        SuccessfulFakeBridge(),
        "The capital of France is",
        " Paris",
    )

    assert result["tokenized_prompt"] == ["The capital", " of France is"]
    assert result["token_ids"] == [10, 11]
    assert result["target_token"] == {"text": " Paris", "id": 1}
    assert result["final_model_prediction"]["token_id"] == 1
    assert result["target_is_final_top1"] is True
    assert result["stages"][-1]["target_rank"] == 1
    assert result["final_projection_validation"]["top1_token_id_matches"] is True


@pytest.fixture
def sample_result() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": "2026-09-14T12:00:00+00:00",
        "model": {
            "name": "EleutherAI/pythia-70m-deduped",
            "requested_revision": "main",
            "resolved_revision": None,
            "device": "cpu",
            "dtype": "float32",
        },
        "dependencies": {"python": "3.12.13", "torch": "2.8.0"},
        "prompts": [
            {
                "prompt": "The capital of France is",
                "tokenized_prompt": ["The", " capital", " of", " France", " is"],
                "token_ids": [1, 2, 3, 4, 5],
                "target_token": {"text": " Paris", "id": 6},
                "final_model_prediction": {
                    "token": " Paris",
                    "token_id": 6,
                    "logit": 4.0,
                    "probability": 0.4,
                },
                "target_is_final_top1": True,
                "stages": [
                    {
                        "label": "embedding",
                        "target_logit": 1.0,
                        "target_rank": 50,
                        "logit_lens_probability": 0.01,
                    },
                    {
                        "label": "after_layer_0",
                        "target_logit": 4.0,
                        "target_rank": 1,
                        "logit_lens_probability": 0.4,
                    },
                ],
                "transitions": [
                    {
                        "from_stage": "embedding",
                        "to_stage": "after_layer_0",
                        "target_logit_delta": 3.0,
                        "target_rank_delta": -49,
                    }
                ],
                "final_projection_validation": {
                    "max_abs_logit_difference": 0.0,
                    "mean_actual_minus_projected_offset": 0.0,
                    "max_abs_difference_after_offset": 0.0,
                    "atol": 1e-5,
                    "rtol": 1e-5,
                    "logits_allclose": True,
                    "projected_target_rank": 1,
                    "actual_target_rank": 1,
                    "target_rank_matches": True,
                    "projected_top1_token_id": 6,
                    "actual_top1_token_id": 6,
                    "top1_token_id_matches": True,
                },
                "observations": [
                    "The Paris target logit increased most strongly from embedding to after_layer_0 (+3.000000)."
                ],
            }
        ],
    }


def test_write_results_json_preserves_required_machine_readable_fields(
    tmp_path: Path,
    sample_result: dict[str, object],
) -> None:
    path = tmp_path / "results.json"

    runner.write_results_json(sample_result, path)

    decoded = json.loads(path.read_text())
    assert decoded["schema_version"] == "1.0"
    assert decoded["model"]["name"] == "EleutherAI/pythia-70m-deduped"
    assert decoded["prompts"][0]["target_token"]["text"] == " Paris"
    assert decoded["prompts"][0]["stages"][0]["label"] == "embedding"
    assert "final_projection_validation" in decoded["prompts"][0]

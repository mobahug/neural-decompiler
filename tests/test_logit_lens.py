from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neural_decompiler.logit_lens import (
    FinalProjectionMismatch,
    analyze_cached_prompt,
    metrics_from_logits,
    stage_labels,
    validate_final_projection,
)


def test_stage_labels_describe_embedding_and_completed_layers() -> None:
    assert stage_labels(["0_pre", "1_pre", "2_pre", "final_post"], 3) == [
        "embedding",
        "after_layer_0",
        "after_layer_1",
        "after_layer_2",
    ]


def test_metrics_from_logits_uses_one_based_rank_and_softmax() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])

    metrics = metrics_from_logits(
        logits,
        target_token_id=2,
        labels=["embedding", "after_layer_0"],
    )

    assert [metric.target_rank for metric in metrics] == [2, 2]
    assert [metric.target_logit for metric in metrics] == [1.0, 2.0]
    assert metrics[0].logit_lens_probability == pytest.approx(0.244728, abs=1e-6)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_metrics_from_logits_rejects_non_finite_values(non_finite: float) -> None:
    logits = torch.tensor([[0.0, non_finite, 1.0]])

    with pytest.raises(ValueError, match="Stage logits contain non-finite values"):
        metrics_from_logits(logits, target_token_id=2, labels=["embedding"])


def test_final_projection_records_constant_offset_without_hiding_it() -> None:
    projected = torch.tensor([1.0, 3.0, 2.0])
    actual = torch.tensor([1.25, 3.25, 2.25])

    result = validate_final_projection(
        projected,
        actual,
        target_token_id=2,
        atol=1e-6,
        rtol=1e-6,
    )

    assert result.logits_allclose is False
    assert result.target_rank_matches is True
    assert result.top1_token_id_matches is True
    assert result.mean_actual_minus_projected_offset == pytest.approx(0.25)
    assert result.max_abs_difference_after_offset == pytest.approx(0.0)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_final_projection_rejects_non_finite_values(non_finite: float) -> None:
    projected = torch.tensor([1.0, non_finite, 2.0])
    actual = torch.tensor([1.0, non_finite, 2.0])

    with pytest.raises(FinalProjectionMismatch, match="non-finite"):
        validate_final_projection(projected, actual, target_token_id=2)


@pytest.mark.parametrize(
    "projected",
    [torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 3.0, 2.0])],
)
def test_final_projection_rejects_rank_or_top1_disagreement(
    projected: torch.Tensor,
) -> None:
    actual = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(FinalProjectionMismatch):
        validate_final_projection(projected, actual, target_token_id=2)


class FakeCache:
    def accumulated_resid(
        self,
        *,
        return_labels: bool,
        apply_ln: bool,
    ) -> tuple[torch.Tensor, list[str]]:
        assert return_labels is True
        assert apply_ln is True
        residuals = torch.tensor(
            [
                [[[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 1.0]]],
            ]
        )
        return residuals, ["0_pre", "final_post"]


class FakeBridge:
    cfg = SimpleNamespace(n_layers=1)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
        bias = torch.tensor([0.5, -0.5, 1.0])
        return residual @ weight + bias


def test_analyze_cached_prompt_normalizes_then_uses_complete_unembedding() -> None:
    actual = torch.tensor([0.5, 2.5, 2.0])

    projection = analyze_cached_prompt(
        FakeBridge(),
        FakeCache(),
        actual,
        target_token_id=1,
    )

    assert projection.stages[0].target_logit == -0.5
    assert projection.stages[1].target_logit == 2.5
    assert projection.validation.logits_allclose is True

from __future__ import annotations

import pytest
import torch

from neural_decompiler.logit_lens import (
    FinalProjectionMismatch,
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

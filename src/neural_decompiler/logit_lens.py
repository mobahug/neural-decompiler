"""Pure helpers for observational logit-lens measurements."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class StageMetric:
    """Target-token measurements at one residual-stream stage."""

    label: str
    target_logit: float
    target_rank: int
    logit_lens_probability: float


@dataclass(frozen=True)
class FinalProjectionValidation:
    """Numerical comparison between projected and actual final logits."""

    max_abs_logit_difference: float
    mean_actual_minus_projected_offset: float
    max_abs_difference_after_offset: float
    atol: float
    rtol: float
    logits_allclose: bool
    projected_target_rank: int
    actual_target_rank: int
    target_rank_matches: bool
    projected_top1_token_id: int
    actual_top1_token_id: int
    top1_token_id_matches: bool


class FinalProjectionMismatch(RuntimeError):
    """Raised when projected final logits change target rank or top-1 token."""


def stage_labels(cache_labels: Sequence[str], n_layers: int) -> list[str]:
    """Convert TransformerLens accumulated-residual labels to public labels."""

    expected_count = n_layers + 1
    if len(cache_labels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} residual stages for {n_layers} layers, "
            f"received {len(cache_labels)}"
        )

    public_labels: list[str] = []
    for index, label in enumerate(cache_labels):
        if index == 0 and label == "0_pre":
            public_labels.append("embedding")
        elif index == n_layers and label == "final_post":
            public_labels.append(f"after_layer_{n_layers - 1}")
        elif match := re.fullmatch(r"(\d+)_pre", label):
            layer_input = int(match.group(1))
            if layer_input != index:
                raise ValueError(
                    f"Unexpected residual label {label!r} at stage {index}"
                )
            public_labels.append(f"after_layer_{layer_input - 1}")
        else:
            raise ValueError(f"Unsupported accumulated-residual label: {label!r}")
    return public_labels


def _target_rank(logits: Tensor, target_token_id: int) -> int:
    target_logit = logits[target_token_id]
    return int((logits > target_logit).sum().item()) + 1


def metrics_from_logits(
    logits: Tensor,
    target_token_id: int,
    labels: Sequence[str],
) -> list[StageMetric]:
    """Calculate target-token measurements from one vocabulary vector per stage."""

    if logits.ndim != 2:
        raise ValueError(f"Expected [stage, vocabulary] logits, received {logits.shape}")
    if logits.shape[0] != len(labels):
        raise ValueError("Stage label count does not match the logits stage count")
    if not 0 <= target_token_id < logits.shape[1]:
        raise ValueError(f"Target token ID {target_token_id} is outside the vocabulary")

    probabilities = torch.softmax(logits.float(), dim=-1)
    return [
        StageMetric(
            label=label,
            target_logit=float(stage_logits[target_token_id].item()),
            target_rank=_target_rank(stage_logits, target_token_id),
            logit_lens_probability=float(probabilities[index, target_token_id].item()),
        )
        for index, (label, stage_logits) in enumerate(zip(labels, logits, strict=True))
    ]


def validate_final_projection(
    projected_logits: Tensor,
    actual_logits: Tensor,
    target_token_id: int,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> FinalProjectionValidation:
    """Compare final projected logits with actual output and enforce rank parity."""

    if projected_logits.ndim != 1 or actual_logits.ndim != 1:
        raise ValueError("Final projected and actual logits must be vocabulary vectors")
    if projected_logits.shape != actual_logits.shape:
        raise ValueError(
            "Final projected and actual logits must have the same vocabulary shape"
        )

    difference = actual_logits.float() - projected_logits.float()
    mean_offset = difference.mean()
    centered_difference = difference - mean_offset
    projected_target_rank = _target_rank(projected_logits, target_token_id)
    actual_target_rank = _target_rank(actual_logits, target_token_id)
    projected_top1 = int(projected_logits.argmax().item())
    actual_top1 = int(actual_logits.argmax().item())

    result = FinalProjectionValidation(
        max_abs_logit_difference=float(difference.abs().max().item()),
        mean_actual_minus_projected_offset=float(mean_offset.item()),
        max_abs_difference_after_offset=float(centered_difference.abs().max().item()),
        atol=atol,
        rtol=rtol,
        logits_allclose=bool(
            torch.allclose(projected_logits.float(), actual_logits.float(), atol=atol, rtol=rtol)
        ),
        projected_target_rank=projected_target_rank,
        actual_target_rank=actual_target_rank,
        target_rank_matches=projected_target_rank == actual_target_rank,
        projected_top1_token_id=projected_top1,
        actual_top1_token_id=actual_top1,
        top1_token_id_matches=projected_top1 == actual_top1,
    )
    if not result.target_rank_matches or not result.top1_token_id_matches:
        raise FinalProjectionMismatch(
            "Final residual projection disagrees with actual model output: "
            f"target ranks {projected_target_rank} vs {actual_target_rank}; "
            f"top-1 token IDs {projected_top1} vs {actual_top1}; "
            f"maximum absolute logit difference {result.max_abs_logit_difference:.9g}"
        )
    return result

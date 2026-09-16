from __future__ import annotations

import json

import pytest
import torch

from neural_decompiler.behavior import BehaviorCase, BehaviorSpec, MetricSpec


def test_behavior_is_domain_independent_and_json_safe() -> None:
    case = BehaviorCase(
        case_id="case-1",
        input_text="A B",
        target_positions=(-1,),
        expected_outputs=(" C",),
        control_refs=("control-1",),
        metadata={"split": "held-out"},
    )
    spec = BehaviorSpec(
        behavior_id="copy-next",
        version="1",
        cases=(case,),
        metric=MetricSpec("target_logit", {"reduction": "mean"}),
        inclusion_criteria=("single-token target",),
        success_criteria=("positive held-out margin",),
    )

    encoded = json.dumps(spec.to_dict(), allow_nan=False, sort_keys=True)

    assert json.loads(encoded)["cases"][0]["input_text"] == "A B"
    assert "capital" not in encoded.lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_text": "a", "input_token_ids": (1,)},
        {},
    ],
)
def test_case_requires_exactly_one_input_form(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        BehaviorCase("case", **kwargs)


def test_behavior_rejects_duplicate_case_ids() -> None:
    first = BehaviorCase("same", input_text="a")
    second = BehaviorCase("same", input_text="b")
    with pytest.raises(ValueError, match="unique"):
        BehaviorSpec(
            "behavior",
            "1",
            (first, second),
            MetricSpec("metric"),
            ("included",),
            ("succeeds",),
        )


@pytest.mark.parametrize(
    "bad",
    [torch.tensor(1), {"set"}, float("nan"), {1: "non-string-key"}],
)
def test_behavior_rejects_non_json_safe_metadata(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BehaviorCase("case", input_text="a", metadata={"bad": bad})


def test_behavior_requires_nonempty_criteria() -> None:
    case = BehaviorCase("case", input_text="a")
    with pytest.raises(ValueError, match="inclusion_criteria"):
        BehaviorSpec(
            "behavior", "1", (case,), MetricSpec("metric"), (), ("success",)
        )

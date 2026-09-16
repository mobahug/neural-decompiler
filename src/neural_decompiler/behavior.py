"""Compact, task-independent behavior declarations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


def validate_json_safe(value: Any, *, path: str = "value") -> None:
    """Reject values that cannot be represented reproducibly in JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            validate_json_safe(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON-safe value {type(value).__name__}")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "metric name")
        validate_json_safe(self.parameters, path="metric.parameters")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class BehaviorCase:
    case_id: str
    input_text: str | None = None
    input_token_ids: tuple[int, ...] | None = None
    target_positions: tuple[int, ...] = ()
    expected_outputs: tuple[Any, ...] = ()
    control_refs: tuple[str, ...] = ()
    counterfactual_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        if (self.input_text is None) == (self.input_token_ids is None):
            raise ValueError("BehaviorCase requires exactly one input form")
        if self.input_text is not None and not isinstance(self.input_text, str):
            raise TypeError("input_text must be a string")
        if self.input_token_ids is not None:
            if not self.input_token_ids or not all(
                isinstance(token, int) and token >= 0 for token in self.input_token_ids
            ):
                raise ValueError("input_token_ids must be nonempty nonnegative integers")
        if not all(isinstance(position, int) for position in self.target_positions):
            raise TypeError("target_positions must contain integers")
        for field_name, refs in (
            ("control_refs", self.control_refs),
            ("counterfactual_refs", self.counterfactual_refs),
        ):
            if any(not isinstance(ref, str) or not ref for ref in refs):
                raise ValueError(f"{field_name} must contain nonempty strings")
        validate_json_safe(self.expected_outputs, path="expected_outputs")
        validate_json_safe(self.metadata, path="metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_text": self.input_text,
            "input_token_ids": (
                list(self.input_token_ids) if self.input_token_ids is not None else None
            ),
            "target_positions": list(self.target_positions),
            "expected_outputs": list(self.expected_outputs),
            "control_refs": list(self.control_refs),
            "counterfactual_refs": list(self.counterfactual_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BehaviorSpec:
    behavior_id: str
    version: str
    cases: tuple[BehaviorCase, ...]
    metric: MetricSpec
    inclusion_criteria: tuple[str, ...]
    success_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.behavior_id, "behavior_id")
        _require_text(self.version, "version")
        if not self.cases:
            raise ValueError("cases must not be empty")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("BehaviorCase IDs must be unique")
        for name, criteria in (
            ("inclusion_criteria", self.inclusion_criteria),
            ("success_criteria", self.success_criteria),
        ):
            if not criteria or any(
                not isinstance(criterion, str) or not criterion.strip()
                for criterion in criteria
            ):
                raise ValueError(f"{name} must contain nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "version": self.version,
            "cases": [case.to_dict() for case in self.cases],
            "metric": self.metric.to_dict(),
            "inclusion_criteria": list(self.inclusion_criteria),
            "success_criteria": list(self.success_criteria),
        }

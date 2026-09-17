"""Frozen, model-independent declarations and metrics for candidate screening."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .behavior import validate_json_safe
from .capture import CapturePlan, CaptureRequest, InstrumentationSettings, run_capture
from .components import ComponentKind, ComponentRef, resolve_component
from .interventions import (
    Intervention,
    InterventionOperation,
    InterventionPlan,
    ReplacementSource,
    run_interventions,
)
from .models import PYTHIA_160M, PYTHIA_70M


_GATE_NAMES = (
    "overall_accuracy", "wilson_lower_bound", "per_template_accuracy", "template_range",
    "development_drop", "positive_template_margin", "contrast_flip", "cue_dependence",
    "beats_baselines", "integrity",
)


class Split(str, Enum):
    DEVELOPMENT = "selection-development"
    HOLDOUT = "selection-holdout"
    FUTURE_RESERVE = "future-reserve"


class CompactnessPartition(str, Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _freeze_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = dict(value)
    validate_json_safe(copied, path=name)
    return MappingProxyType(copied)


def _json(value: dict[str, Any], path: str) -> dict[str, Any]:
    validate_json_safe(value, path=path)
    return value


@dataclass(frozen=True)
class PromptCondition:
    """Exact prompt and canonical A/B target sequences at the final position."""

    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    a_token_ids: tuple[int, ...]
    b_token_ids: tuple[int, ...]
    target_position: int = -1
    a_text: str = ""
    b_text: str = ""

    def __post_init__(self) -> None:
        _require_text(self.prompt_text, "prompt_text")
        for name in ("a_text", "b_text"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        for name in ("prompt_token_ids", "a_token_ids", "b_token_ids"):
            tokens = tuple(getattr(self, name))
            if not tokens or any(not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError(f"{name} must contain nonnegative token IDs")
            object.__setattr__(self, name, tokens)
        if self.target_position != -1:
            raise ValueError("target_position must be -1")

    def to_dict(self) -> dict[str, Any]:
        return _json({
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "a_token_ids": list(self.a_token_ids),
            "b_token_ids": list(self.b_token_ids),
            "target_position": self.target_position,
            "a_text": self.a_text,
            "b_text": self.b_text,
        }, "prompt_condition")

    @property
    def single_token(self) -> bool:
        return len(self.a_token_ids) == 1 and len(self.b_token_ids) == 1


@dataclass(frozen=True)
class ScreeningCase:
    """Frozen four-condition matched case."""

    case_id: str
    candidate_id: str
    template_id: str
    split: Split
    lexical_key: str
    rule_class: str
    primary_orientation: str
    x_a: PromptCondition
    x_b: PromptCondition
    s_a: PromptCondition
    s_b: PromptCondition
    local_heuristic_choices: Mapping[str, Any]
    compactness_partition: CompactnessPartition | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "candidate_id", "template_id", "lexical_key", "rule_class"):
            _require_text(getattr(self, name), name)
        if not isinstance(self.split, Split):
            raise TypeError("split must be a Split")
        if self.primary_orientation not in {"A", "B"}:
            raise ValueError("primary_orientation must be A or B")
        for name in ("x_a", "x_b", "s_a", "s_b"):
            if not isinstance(getattr(self, name), PromptCondition):
                raise TypeError(f"{name} must be a PromptCondition")
        choices = _freeze_mapping(self.local_heuristic_choices, "local_heuristic_choices")
        if set(choices) != {"A", "B"}:
            raise ValueError("local_heuristic_choices must contain A and B")
        object.__setattr__(self, "local_heuristic_choices", choices)
        if self.compactness_partition is not None and not isinstance(self.compactness_partition, CompactnessPartition):
            raise TypeError("compactness_partition must be a CompactnessPartition or None")

    def to_dict(self) -> dict[str, Any]:
        return _json({
            "case_id": self.case_id, "candidate_id": self.candidate_id,
            "template_id": self.template_id, "split": self.split.value,
            "lexical_key": self.lexical_key, "rule_class": self.rule_class,
            "primary_orientation": self.primary_orientation,
            "conditions": {"x_a": self.x_a.to_dict(), "x_b": self.x_b.to_dict(),
                           "s_a": self.s_a.to_dict(), "s_b": self.s_b.to_dict()},
            "local_heuristic_choices": dict(self.local_heuristic_choices),
            "compactness_partition": self.compactness_partition.value if self.compactness_partition else None,
        }, "screening_case")


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    template_ids: tuple[str, ...]
    local_heuristic: str = ""

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        templates = tuple(self.template_ids)
        if len(templates) != 3 or len(set(templates)) != 3:
            raise ValueError("candidate definitions require exactly three unique templates")
        for template_id in templates:
            _require_text(template_id, "template_id")
        object.__setattr__(self, "template_ids", templates)
        if self.local_heuristic:
            _require_text(self.local_heuristic, "local_heuristic")

    def to_dict(self) -> dict[str, Any]:
        return _json({"candidate_id": self.candidate_id, "template_ids": list(self.template_ids),
                      "local_heuristic": self.local_heuristic}, "candidate_definition")


@dataclass(frozen=True)
class ScreeningManifest:
    candidates: tuple[CandidateDefinition, ...]
    cases: tuple[ScreeningCase, ...]
    schema_version: int = 1
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        candidates, cases = tuple(self.candidates), tuple(self.cases)
        if not candidates:
            raise ValueError("manifest must contain candidates")
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        case_ids = tuple(case.case_id for case in cases)
        if len(candidate_ids) != len(set(candidate_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("manifest IDs must be unique")
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        if self.content_sha256 is not None:
            _require_text(self.content_sha256, "content_sha256")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cases", cases)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def cases_for(self, candidate_id: str, split: Split, template_id: str | None = None) -> tuple[ScreeningCase, ...]:
        return tuple(case for case in self.cases if case.candidate_id == candidate_id and case.split is split
                     and (template_id is None or case.template_id == template_id))

    def to_dict(self) -> dict[str, Any]:
        return _json({"schema_version": self.schema_version, "content_sha256": self.content_sha256,
                      "candidates": [candidate.to_dict() for candidate in self.candidates],
                      "cases": [case.to_dict() for case in self.cases]}, "screening_manifest")


def fixed_contrast(logp_a: float, logp_b: float) -> float:
    """Return fixed-orientation ``c(x) = log P(A|x) - log P(B|x)``."""
    value = float(logp_a) - float(logp_b)
    if not math.isfinite(value):
        raise ValueError("contrast must be finite")
    return value


def correctness_margin(contrast: float, intended: str) -> float:
    """Return intended-orientation margin without changing the stored contrast."""
    if intended not in {"A", "B"}:
        raise ValueError("intended must be A or B")
    value = _finite(contrast, "contrast")
    return value if intended == "A" else -value


@dataclass(frozen=True)
class ConditionMeasurement:
    logp_a: float
    logp_b: float
    intended: str
    top1_token_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "logp_a", _finite(self.logp_a, "logp_a"))
        object.__setattr__(self, "logp_b", _finite(self.logp_b, "logp_b"))
        if self.intended not in {"A", "B"}:
            raise ValueError("intended must be A or B")
        if self.top1_token_id is not None and (not isinstance(self.top1_token_id, int) or self.top1_token_id < 0):
            raise ValueError("top1_token_id must be a nonnegative integer or None")

    @property
    def contrast(self) -> float:
        return fixed_contrast(self.logp_a, self.logp_b)

    @property
    def correctness_margin(self) -> float:
        return correctness_margin(self.contrast, self.intended)

    @property
    def correct(self) -> bool:
        return self.correctness_margin > 0.0

    def to_dict(self) -> dict[str, Any]:
        return _json({"logp_a": self.logp_a, "logp_b": self.logp_b, "intended": self.intended,
                      "contrast": self.contrast, "correctness_margin": self.correctness_margin,
                      "correct": self.correct, "top1_token_id": self.top1_token_id}, "condition_measurement")


@dataclass(frozen=True)
class CaseMeasurement:
    x_a: ConditionMeasurement
    x_b: ConditionMeasurement
    s_a: ConditionMeasurement
    s_b: ConditionMeasurement
    primary_orientation: str
    case_id: str = ""
    template_id: str = ""

    def __post_init__(self) -> None:
        for name in ("x_a", "x_b", "s_a", "s_b"):
            if not isinstance(getattr(self, name), ConditionMeasurement):
                raise TypeError(f"{name} must be a ConditionMeasurement")
        if (self.x_a.intended, self.x_b.intended, self.s_a.intended, self.s_b.intended) != ("A", "B", "A", "B"):
            raise ValueError("conditions must preserve A/B intended orientations")
        if self.primary_orientation not in {"A", "B"}:
            raise ValueError("primary_orientation must be A or B")
        if self.case_id:
            _require_text(self.case_id, "case_id")
        if self.template_id:
            _require_text(self.template_id, "template_id")

    @property
    def primary(self) -> ConditionMeasurement:
        return self.x_a if self.primary_orientation == "A" else self.x_b

    @property
    def primary_correct(self) -> bool:
        return self.primary.correct

    @property
    def cue_shuffle_correct(self) -> bool:
        return (self.s_a if self.primary_orientation == "A" else self.s_b).correct

    @property
    def contrast_flip(self) -> bool:
        return self.x_a.contrast > 0.0 and self.x_b.contrast < 0.0

    @property
    def d_full(self) -> float:
        return self.x_a.contrast - self.x_b.contrast

    @property
    def d_cue(self) -> float:
        return 0.5 * ((self.x_a.contrast - self.s_a.contrast) + (self.s_b.contrast - self.x_b.contrast))

    def to_dict(self) -> dict[str, Any]:
        return _json({"case_id": self.case_id, "template_id": self.template_id,
                      "primary_orientation": self.primary_orientation, "x_a": self.x_a.to_dict(),
                      "x_b": self.x_b.to_dict(), "s_a": self.s_a.to_dict(), "s_b": self.s_b.to_dict(),
                      "primary_correct": self.primary_correct, "cue_shuffle_correct": self.cue_shuffle_correct,
                      "contrast_flip": self.contrast_flip, "d_full": self.d_full, "d_cue": self.d_cue}, "case_measurement")


def measure_case(logp: Mapping[str, Mapping[str, float]], primary_orientation: str, *, case_id: str = "", template_id: str = "") -> CaseMeasurement:
    """Measure all four fixed protocol conditions while preserving A/B orientation."""
    if primary_orientation not in {"A", "B"}:
        raise ValueError("primary_orientation must be A or B")
    def condition(name: str, intended: str) -> ConditionMeasurement:
        try:
            scores = logp[name]
            return ConditionMeasurement(scores["A"], scores["B"], intended)
        except (KeyError, TypeError) as error:
            raise ValueError(f"logp must contain {name}.A and {name}.B") from error
    return CaseMeasurement(condition("x_a", "A"), condition("x_b", "B"), condition("s_a", "A"),
                           condition("s_b", "B"), primary_orientation, case_id, template_id)


def wilson_lower_bound(successes: int, total: int) -> float:
    """Return the frozen two-sided 95% Wilson lower confidence bound."""
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total, with total positive")
    z = 1.959963984540054
    p = successes / total
    numerator = p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return numerator / (1 + z * z / total)


def passes_overall_accuracy_gate(successes: int, total: int) -> bool:
    return successes / total >= 0.85 and wilson_lower_bound(successes, total) >= 0.78


def _rate(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class CandidateBehaviorSummary:
    candidate_id: str
    case_count: int
    primary_correct_count: int
    template_accuracies: Mapping[str, float]
    template_mean_margins: Mapping[str, float]
    contrast_flip_rate: float
    cue_shuffle_accuracy: float
    mean_d_full: float
    mean_d_cue: float
    baseline_accuracies: Mapping[str, float]
    integrity_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        if not isinstance(self.case_count, int) or self.case_count < 0 or not isinstance(self.primary_correct_count, int) or not 0 <= self.primary_correct_count <= self.case_count:
            raise ValueError("primary_correct_count must be between zero and case_count")
        accuracies = _freeze_mapping(self.template_accuracies, "template_accuracies")
        margins = _freeze_mapping(self.template_mean_margins, "template_mean_margins")
        if not accuracies or set(accuracies) != set(margins):
            raise ValueError("template accuracies and margins must have the same nonempty keys")
        for key, value in accuracies.items():
            _require_text(key, "template_id"); _rate(value, f"template_accuracies.{key}")
        for key, value in margins.items():
            _finite(value, f"template_mean_margins.{key}")
        baselines = _freeze_mapping(self.baseline_accuracies, "baseline_accuracies")
        for key, value in baselines.items():
            _require_text(key, "baseline_id"); _rate(value, f"baseline_accuracies.{key}")
        failures = tuple(self.integrity_failures)
        if any(not isinstance(value, str) or not value.strip() for value in failures):
            raise ValueError("integrity_failures must contain nonempty strings")
        object.__setattr__(self, "template_accuracies", accuracies)
        object.__setattr__(self, "template_mean_margins", margins)
        object.__setattr__(self, "baseline_accuracies", baselines)
        object.__setattr__(self, "integrity_failures", failures)
        object.__setattr__(self, "contrast_flip_rate", _rate(self.contrast_flip_rate, "contrast_flip_rate"))
        object.__setattr__(self, "cue_shuffle_accuracy", _rate(self.cue_shuffle_accuracy, "cue_shuffle_accuracy"))
        object.__setattr__(self, "mean_d_full", _finite(self.mean_d_full, "mean_d_full"))
        object.__setattr__(self, "mean_d_cue", _finite(self.mean_d_cue, "mean_d_cue"))

    @property
    def accuracy(self) -> float:
        if not self.case_count:
            raise ValueError("accuracy is undefined for zero cases")
        return self.primary_correct_count / self.case_count

    def to_dict(self) -> dict[str, Any]:
        return _json({"candidate_id": self.candidate_id, "case_count": self.case_count,
                      "primary_correct_count": self.primary_correct_count, "accuracy": self.accuracy,
                      "template_accuracies": dict(self.template_accuracies), "template_mean_margins": dict(self.template_mean_margins),
                      "contrast_flip_rate": self.contrast_flip_rate, "cue_shuffle_accuracy": self.cue_shuffle_accuracy,
                      "mean_d_full": self.mean_d_full, "mean_d_cue": self.mean_d_cue,
                      "baseline_accuracies": dict(self.baseline_accuracies), "integrity_failures": list(self.integrity_failures)}, "candidate_behavior_summary")


def summarize_behavior(measurements: Sequence[CaseMeasurement], *, candidate_id: str, baseline_accuracies: Mapping[str, float] | None = None, integrity_failures: Sequence[str] = ()) -> CandidateBehaviorSummary:
    """Summarize primary-only accuracy and paired effects without model work."""
    rows = tuple(measurements)
    if not rows or any(not isinstance(row, CaseMeasurement) for row in rows):
        raise ValueError("measurements must be nonempty CaseMeasurement values")
    if any(not row.template_id for row in rows):
        raise ValueError("measurements require template_id")
    grouped: dict[str, list[CaseMeasurement]] = {}
    for row in rows:
        grouped.setdefault(row.template_id, []).append(row)
    return CandidateBehaviorSummary(
        candidate_id=candidate_id, case_count=len(rows), primary_correct_count=sum(row.primary_correct for row in rows),
        template_accuracies={key: sum(row.primary_correct for row in values) / len(values) for key, values in grouped.items()},
        template_mean_margins={key: sum(row.primary.correctness_margin for row in values) / len(values) for key, values in grouped.items()},
        contrast_flip_rate=sum(row.contrast_flip for row in rows) / len(rows),
        cue_shuffle_accuracy=sum(row.cue_shuffle_correct for row in rows) / len(rows),
        mean_d_full=sum(row.d_full for row in rows) / len(rows), mean_d_cue=sum(row.d_cue for row in rows) / len(rows),
        baseline_accuracies=baseline_accuracies or {}, integrity_failures=tuple(integrity_failures))


@dataclass(frozen=True)
class BehavioralGateResult:
    checks: Mapping[str, bool]
    passed: bool

    def __post_init__(self) -> None:
        checks = _freeze_mapping(self.checks, "behavioral_gate.checks")
        if tuple(checks) != _GATE_NAMES or any(not isinstance(value, bool) for value in checks.values()):
            raise ValueError("behavioral gate checks must contain every frozen boolean gate in order")
        if self.passed != all(checks.values()):
            raise ValueError("passed must equal all behavioral checks")
        object.__setattr__(self, "checks", checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.checks.items() if not value)

    def to_dict(self) -> dict[str, Any]:
        return _json({"checks": dict(self.checks), "passed": self.passed, "failed_checks": list(self.failed_checks)}, "behavioral_gate_result")


def evaluate_behavioral_gates(development: CandidateBehaviorSummary, holdout: CandidateBehaviorSummary) -> BehavioralGateResult:
    """Evaluate every frozen behavioral rule; no successful later gate rescues a failure."""
    if development.candidate_id != holdout.candidate_id:
        raise ValueError("development and holdout candidate IDs must match")
    if holdout.case_count != 120:
        raise ValueError("behavioral gate requires exactly 120 holdout cases")
    checks = {
        "overall_accuracy": holdout.accuracy >= 0.85,
        "wilson_lower_bound": wilson_lower_bound(holdout.primary_correct_count, holdout.case_count) >= 0.78,
        "per_template_accuracy": min(holdout.template_accuracies.values()) >= 0.75,
        "template_range": max(holdout.template_accuracies.values()) - min(holdout.template_accuracies.values()) <= 0.15,
        "development_drop": development.accuracy - holdout.accuracy <= 0.10,
        "positive_template_margin": min(holdout.template_mean_margins.values()) > 0.0,
        "contrast_flip": holdout.contrast_flip_rate >= 0.80,
        "cue_dependence": holdout.accuracy - holdout.cue_shuffle_accuracy >= 0.20 or holdout.mean_d_cue >= 0.5 * holdout.mean_d_full,
        "beats_baselines": all(holdout.accuracy - value >= 0.15 for value in holdout.baseline_accuracies.values()),
        "integrity": not holdout.integrity_failures,
    }
    return BehavioralGateResult(checks, all(checks.values()))


def aligned_patch_shift(c_a: float, c_b: float, c_a_from_b: float, c_b_from_a: float) -> float:
    return 0.5 * ((_finite(c_a, "c_a") - _finite(c_a_from_b, "c_a_from_b")) + (_finite(c_b_from_a, "c_b_from_a") - _finite(c_b, "c_b")))


@dataclass(frozen=True)
class PairPatchMeasurement:
    d_full: float
    d_patch: float
    template_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "d_full", _finite(self.d_full, "d_full"))
        object.__setattr__(self, "d_patch", _finite(self.d_patch, "d_patch"))
        _require_text(self.template_id, "template_id")

    def to_dict(self) -> dict[str, Any]:
        return _json({"d_full": self.d_full, "d_patch": self.d_patch, "template_id": self.template_id}, "pair_patch_measurement")


def aggregate_recovery(rows: Sequence[PairPatchMeasurement]) -> float:
    measurements = tuple(rows)
    if not measurements or any(not isinstance(row, PairPatchMeasurement) for row in measurements):
        raise ValueError("rows must be nonempty PairPatchMeasurement values")
    denominator = sum(row.d_full for row in measurements)
    if denominator == 0:
        raise ValueError("aggregate recovery denominator must not be zero")
    return sum(row.d_patch for row in measurements) / denominator


@dataclass(frozen=True)
class DenominatorValidation:
    overall_denominator: float
    template_denominators: Mapping[str, float]
    valid: bool
    recovery: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "overall_denominator", _finite(self.overall_denominator, "overall_denominator"))
        denominators = _freeze_mapping(self.template_denominators, "template_denominators")
        if not denominators:
            raise ValueError("template_denominators must not be empty")
        for key, value in denominators.items():
            _require_text(key, "template_id"); _finite(value, f"template_denominators.{key}")
        expected = self.overall_denominator >= 0.25 and all(value >= 0.10 for value in denominators.values())
        if self.valid != expected:
            raise ValueError("valid must match frozen compactness denominator floors")
        if self.recovery is not None:
            if not self.valid:
                raise ValueError("recovery must be None for invalid denominators")
            object.__setattr__(self, "recovery", _finite(self.recovery, "recovery"))
        object.__setattr__(self, "template_denominators", denominators)

    def to_dict(self) -> dict[str, Any]:
        return _json({"overall_denominator": self.overall_denominator, "template_denominators": dict(self.template_denominators),
                      "valid": self.valid, "recovery": self.recovery}, "denominator_validation")


def evaluate_compactness_denominators(overall_denominator: float, template_denominators: Mapping[str, float]) -> DenominatorValidation:
    overall = _finite(overall_denominator, "overall_denominator")
    denominators = _freeze_mapping(template_denominators, "template_denominators")
    valid = overall >= 0.25 and bool(denominators) and all(_finite(value, f"template_denominators.{key}") >= 0.10 for key, value in denominators.items())
    return DenominatorValidation(overall, denominators, valid)


def random_reference(random_recoveries: Sequence[float]) -> float:
    values = tuple(_finite(value, "random recovery") for value in random_recoveries)
    if not values:
        raise ValueError("random_recoveries must not be empty")
    return max(statistics.median(values), 0.05)


def required_random_beating_recovery(random_recoveries: Sequence[float]) -> float:
    return 2.0 * random_reference(random_recoveries)


@dataclass(frozen=True)
class CompactnessGateResult:
    checks: Mapping[str, bool]
    passed: bool
    selected_k: int | None = None
    evaluations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checks = _freeze_mapping(self.checks, "compactness_gate.checks")
        if not checks or any(not isinstance(value, bool) for value in checks.values()):
            raise ValueError("compactness checks must be nonempty booleans")
        if self.passed != all(checks.values()):
            raise ValueError("passed must equal all compactness checks")
        if self.passed and (not isinstance(self.selected_k, int) or not 1 <= self.selected_k <= 12):
            raise ValueError("passing compactness requires selected_k from 1 through 12")
        if not self.passed and self.selected_k is not None:
            raise ValueError("failing compactness cannot have selected_k")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "evaluations", _freeze_mapping(self.evaluations, "compactness_gate.evaluations"))

    def to_dict(self) -> dict[str, Any]:
        return _json({"checks": dict(self.checks), "passed": self.passed, "selected_k": self.selected_k,
                      "evaluations": {key: dict(value) for key, value in self.evaluations.items()}}, "compactness_gate_result")


def evaluate_compactness_gate(top_k_measurements: Mapping[int, Sequence[PairPatchMeasurement]], random_recoveries: Mapping[int, Sequence[float]]) -> CompactnessGateResult:
    """Return the smallest top-k set satisfying all frozen compactness rules."""
    evaluations: dict[str, Mapping[str, Any]] = {}
    selected_k: int | None = None
    selected_checks: dict[str, bool] | None = None
    for k in sorted(top_k_measurements):
        if not isinstance(k, int) or not 1 <= k <= 12 or k not in random_recoveries:
            raise ValueError("each compactness k must be 1..12 with random recoveries")
        rows = tuple(top_k_measurements[k])
        templates: dict[str, list[PairPatchMeasurement]] = {}
        for row in rows:
            templates.setdefault(row.template_id, []).append(row)
        overall_d = sum(row.d_full for row in rows) / len(rows) if rows else 0.0
        template_ds = {key: sum(row.d_full for row in values) / len(values) for key, values in templates.items()}
        denominators = evaluate_compactness_denominators(overall_d, template_ds) if template_ds else None
        recovery = aggregate_recovery(rows) if denominators and denominators.valid else None
        template_rs = {key: aggregate_recovery(values) for key, values in templates.items()} if recovery is not None else {}
        checks = {
            "denominators": bool(denominators and denominators.valid),
            "overall_recovery": recovery is not None and recovery >= 0.70,
            "beats_random": recovery is not None and recovery >= required_random_beating_recovery(random_recoveries[k]),
            "positive_template_recovery": recovery is not None and all(sum(row.d_patch for row in values) > 0 and template_rs[key] > 0 for key, values in templates.items()),
        }
        evaluations[str(k)] = {"denominators": denominators.to_dict() if denominators else None, "recovery": recovery,
                          "template_recoveries": template_rs, "random_reference": random_reference(random_recoveries[k]), "checks": checks}
        if selected_checks is None and all(checks.values()):
            selected_k, selected_checks = k, checks
    if selected_checks is None:
        selected_checks = {"denominators": False, "overall_recovery": False, "beats_random": False, "positive_template_recovery": False}
    return CompactnessGateResult(selected_checks, selected_k is not None, selected_k, evaluations)


@dataclass(frozen=True)
class FinalistSelection:
    selected_candidate_id: str | None
    ranking: tuple[str, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        ranking = tuple(self.ranking)
        if len(ranking) != len(set(ranking)):
            raise ValueError("ranking candidate IDs must be unique")
        for candidate_id in ranking:
            _require_text(candidate_id, "ranking candidate_id")
        if self.selected_candidate_id is not None:
            _require_text(self.selected_candidate_id, "selected_candidate_id")
            if self.selected_candidate_id not in ranking:
                raise ValueError("selected candidate must appear in ranking")
        if self.rationale:
            _require_text(self.rationale, "rationale")
        object.__setattr__(self, "ranking", ranking)

    def to_dict(self) -> dict[str, Any]:
        return _json({"selected_candidate_id": self.selected_candidate_id, "ranking": list(self.ranking), "rationale": self.rationale}, "finalist_selection")


# The manifest below is deliberately generated from literal, closed pools.  The
# builder may inspect tokenizer behavior but it has no model or logit dependency.
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SEED = 20260916
# degree-inflection was eliminated pre-output on 2026-09-17 (tokenizer infeasibility).
_CANDIDATE_ORDER = ("regular-plural", "ordinal-suffix")
MANIFEST_CASE_COUNT = 720
TEMPLATES = {
    "regular-plural": {
        "cardinal": (("The display contains one", "The display contains two"), ("The tray holds one", "The tray holds two")),
        "quantifier": (("The catalog lists each", "The catalog lists several"), ("The inventory records each", "The inventory records several")),
        "coordinated-adjective": (("Mira and Noah packed one bright", "Mira and Noah packed two bright"), ("Lena and Omar displayed one small", "Lena and Omar displayed two small")),
    },
    "ordinal-suffix": {
        "bare-numeral": (("NUMBER", "NUMBER"), ("No. NUMBER", "No. NUMBER")),
        "dated-event": (("The event happened on September NUMBER", "The event happened on September NUMBER"), ("The meeting occurred on March NUMBER", "The meeting occurred on March NUMBER")),
        "ranked-list": (("Her final rank was NUMBER", "Her final rank was NUMBER"), ("The athlete placed NUMBER", "The athlete placed NUMBER")),
    },
}

PLURAL_POOLS = {
    "selection-development": {"simple": ("cat", "dog", "book", "lamp", "chair", "river", "cloud", "train", "spoon", "cup", "garden", "window", "planet", "robot", "candle"), "change": ("city", "baby", "story", "party", "berry", "box", "bus", "dish", "watch", "class", "puppy", "brush", "fox", "church", "bench")},
    "selection-holdout": {"simple": ("island", "pencil", "button", "ticket", "basket", "mirror", "camera", "tunnel", "village", "jacket", "carpet", "bottle", "blanket", "ladder", "rocket"), "change": ("family", "lady", "hobby", "cherry", "country", "glass", "kiss", "match", "peach", "wish", "library", "factory", "mystery", "gallery", "branch")},
    "future-reserve": {"simple": ("anchor", "beacon", "castle", "desert", "engine", "forest", "harbor", "insect", "kernel", "market", "needle", "ocean", "pocket", "quilt", "ribbon", "table", "stone", "field", "road", "door"), "change": ("army", "diary", "enemy", "fairy", "glory", "sky", "fly", "ally", "penny", "reply", "beach", "bush", "cross", "dress", "inch")},
}
ORDINAL_NUMBER_PAIRS = {
    "selection-development": ((21, 11), (22, 12), (23, 13), (31, 111), (32, 112), (33, 113), (41, 211), (42, 212), (43, 213), (51, 311), (52, 312), (53, 313), (61, 411), (62, 412), (63, 413), (71, 511), (72, 512), (73, 513), (81, 611), (82, 612)),
    "selection-holdout": ((91, 711), (92, 712), (93, 713), (101, 811), (102, 812), (103, 813), (121, 911), (122, 912), (123, 913), (131, 1011), (132, 1012), (133, 1013), (141, 1111), (142, 1112), (143, 1113), (151, 1211), (152, 1212), (153, 1213), (161, 1311), (162, 1312)),
    "future-reserve": ((163, 1313), (171, 1411), (172, 1412), (173, 1413), (181, 1511), (182, 1512), (183, 1513), (191, 1611), (192, 1612), (193, 1613), (201, 1711), (202, 1712), (203, 1713), (221, 1811), (222, 1812), (223, 1813), (231, 1911), (232, 1912), (233, 1913), (241, 2011)),
}
def regular_plural(base: str) -> str:
    if base.endswith("y") and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return base + "es"
    return base + "s"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def manifest_content_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _tokenizer_mapping(tokenizers: Any) -> dict[str, Any]:
    if isinstance(tokenizers, Mapping):
        result = dict(tokenizers)
    else:
        result = {PYTHIA_70M.model_id: tokenizers}
    expected = {PYTHIA_70M.model_id, PYTHIA_160M.model_id}
    if set(result) != expected:
        raise ValueError("manifest freezing requires tokenizers for both pinned Pythia models")
    return result


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(encoded, Sequence) or any(not isinstance(value, int) for value in encoded):
        raise ValueError("tokenizer must return integer IDs")
    return list(encoded)


def _assert_tokenizer_compatibility(tokenizers: Mapping[str, Any]) -> dict[str, Any]:
    first = tokenizers[PYTHIA_70M.model_id]
    second = tokenizers[PYTHIA_160M.model_id]
    first_vocab, second_vocab = first.get_vocab(), second.get_vocab()
    if first_vocab != second_vocab:
        raise ValueError("pinned tokenizer vocabularies are incompatible")
    first_special = getattr(first, "special_tokens_map", {})
    second_special = getattr(second, "special_tokens_map", {})
    if first_special != second_special or tuple(getattr(first, "all_special_ids", ())) != tuple(getattr(second, "all_special_ids", ())):
        raise ValueError("pinned tokenizer special-token settings are incompatible")
    vocabulary = _canonical_json(dict(sorted(first_vocab.items())))
    return {"model_ids": [PYTHIA_70M.model_id, PYTHIA_160M.model_id], "vocabulary_sha256": hashlib.sha256(vocabulary.encode("utf-8")).hexdigest(), "special_tokens": first_special, "special_token_ids": list(getattr(first, "all_special_ids", ())) }


def _word_rule(base: str, kind: str) -> str:
    if kind == "simple":
        return "simple-suffix"
    return "consonant-y" if base.endswith("y") else "sibilant-es"


def _ordinal_suffix(number: int) -> str:
    if number % 100 in {11, 12, 13}:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def _build_condition(tokenizers: Mapping[str, Any], prompt: str, a_text: str, b_text: str, token_strings: dict[str, str]) -> dict[str, Any]:
    first = tokenizers[PYTHIA_70M.model_id]
    prompt_ids = _token_ids(first, prompt)
    if not prompt_ids:
        raise ValueError("prompt must not encode to an empty token sequence")
    alternatives: dict[str, list[int]] = {}
    for label, alternative in (("A", a_text), ("B", b_text)):
        full_ids = _token_ids(first, prompt + alternative)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("contextual retokenization prevents frozen target suffixes")
        suffix = full_ids[len(prompt_ids):]
        if not suffix:
            raise ValueError("target suffix must not be empty")
        alternatives[label] = suffix
    if len(alternatives["A"]) != len(alternatives["B"]):
        raise ValueError("A/B alternatives must have equal token lengths")
    for tokenizer in tokenizers.values():
        if _token_ids(tokenizer, prompt) != prompt_ids:
            raise ValueError("pinned tokenizer encodings differ")
        for label, alternative in (("A", a_text), ("B", b_text)):
            if _token_ids(tokenizer, prompt + alternative) != prompt_ids + alternatives[label]:
                raise ValueError("pinned tokenizer encodings differ")
    def strings(ids: Sequence[int]) -> list[str]:
        values = list(first.convert_ids_to_tokens(list(ids)))
        if len(values) != len(ids) or any(first.convert_tokens_to_ids(value) != token for token, value in zip(ids, values)):
            raise ValueError("token-string roundtrip failed")
        for token, value in zip(ids, values):
            prior = token_strings.setdefault(str(token), value)
            if prior != value:
                raise ValueError("token ID has inconsistent token string")
        return values
    return {"prompt_text": prompt, "prompt_token_ids": prompt_ids, "prompt_token_strings": strings(prompt_ids), "a_text": a_text, "b_text": b_text, "a_token_ids": alternatives["A"], "b_token_ids": alternatives["B"], "a_token_strings": strings(alternatives["A"]), "b_token_strings": strings(alternatives["B"]), "target_position": -1}


def _word_eligible(tokenizers: Mapping[str, Any], split: str, base: str) -> bool:
    """Check every frozen plural template before a pool item is selected."""
    token_strings: dict[str, str] = {}
    a, b = base, regular_plural(base)
    try:
        for variants in TEMPLATES["regular-plural"].values():
            for prompts in variants:
                for prompt in prompts:
                    condition = _build_condition(tokenizers, prompt, " " + a, " " + b, token_strings)
                    if split == Split.DEVELOPMENT.value and (
                        len(condition["a_token_ids"]) != 1 or len(condition["b_token_ids"]) != 1
                    ):
                        return False
    except ValueError as error:
        message = str(error)
        if "pinned tokenizer" in message or "roundtrip" in message:
            raise
        return False
    return True


def _selected_words(tokenizers: Mapping[str, Any], split: str) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for kind in ("simple", "change"):
        eligible = [(base, kind) for base in PLURAL_POOLS[split][kind] if _word_eligible(tokenizers, split, base)]
        if len(eligible) < 10:
            raise ValueError(f"curated regular-plural {split} {kind} pool cannot meet tokenizer constraints")
        selected.extend(eligible[:10])
    return selected


def _case_payload(*, case_id: str, candidate_id: str, template_id: str, split: str, lexical_key: str, rule_class: str, primary_orientation: str, x_a: dict[str, Any], x_b: dict[str, Any], local_choices: Mapping[str, Any], compactness: str | None) -> dict[str, Any]:
    # The local target remains fixed while the cue is exchanged in s_a/s_b.
    return {"case_id": case_id, "candidate_id": candidate_id, "template_id": template_id, "split": split, "lexical_key": lexical_key, "rule_class": rule_class, "primary_orientation": primary_orientation, "conditions": {"x_a": x_a, "x_b": x_b, "s_a": x_b, "s_b": x_a}, "local_heuristic_choices": dict(local_choices), "compactness_partition": compactness}


def build_manifest_payload(tokenizer: Any) -> dict[str, Any]:
    """Build, but never score, the complete manifest from closed pools and tokenizers."""
    tokenizers = _tokenizer_mapping(tokenizer)
    provenance = _assert_tokenizer_compatibility(tokenizers)
    token_strings: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    candidates = [
        {"candidate_id": "regular-plural", "template_ids": list(TEMPLATES["regular-plural"]), "local_heuristic": "append-s"},
        {"candidate_id": "ordinal-suffix", "template_ids": list(TEMPLATES["ordinal-suffix"]), "local_heuristic": "final-digit-suffix"},
    ]
    for candidate_id in _CANDIDATE_ORDER:
        candidate_cases: list[dict[str, Any]] = []
        for split in (member.value for member in Split):
            if candidate_id == "ordinal-suffix":
                selected: list[tuple[Any, str]] = [(pair, "ordinal-" + _ordinal_suffix(pair[0])) for pair in ORDINAL_NUMBER_PAIRS[split]]
            else:
                selected = _selected_words(tokenizers, split)
            for template_id, variants in TEMPLATES[candidate_id].items():
                for variant_index, prompts in enumerate(variants):
                    for item_index, item in enumerate(selected):
                        primary = "A" if item_index < 10 else "B"
                        if candidate_id == "ordinal-suffix":
                            (number_a, number_b), rule_class = item
                            prompt_a = prompts[0].replace("NUMBER", str(number_a))
                            prompt_b = prompts[1].replace("NUMBER", str(number_b))
                            a_text, b_text = _ordinal_suffix(number_a), _ordinal_suffix(number_b)
                            local = {"A": _ordinal_suffix(number_a % 10), "B": _ordinal_suffix(number_b % 10)}
                            lexical_key = f"{number_a}:{number_b}"
                        else:
                            base, kind = item
                            prompt_a, prompt_b = prompts
                            a_word, b_word = base, regular_plural(base)
                            # The shallow rule follows the count cue and appends a bare "s".
                            local = {"A": base, "B": base + "s"}
                            a_text, b_text = " " + a_word, " " + b_word
                            rule_class = _word_rule(base, kind)
                            lexical_key = base
                        x_a = _build_condition(tokenizers, prompt_a, a_text, b_text, token_strings)
                        x_b = _build_condition(tokenizers, prompt_b, a_text, b_text, token_strings)
                        number = variant_index * 20 + item_index + 1
                        candidate_cases.append(_case_payload(case_id=f"{candidate_id}-{split}-{template_id}-{number:02d}", candidate_id=candidate_id, template_id=template_id, split=split, lexical_key=lexical_key, rule_class=rule_class, primary_orientation=primary, x_a=x_a, x_b=x_b, local_choices=local, compactness=None))
        development = [case for case in candidate_cases if case["split"] == Split.DEVELOPMENT.value]
        for number, case in enumerate(development):
            case["compactness_partition"] = CompactnessPartition.DISCOVERY.value if number < 60 else CompactnessPartition.VALIDATION.value
        cases.extend(candidate_cases)
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "seed": MANIFEST_SEED, "models": [{"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, {"model_id": PYTHIA_160M.model_id, "revision": PYTHIA_160M.revision}], "tokenizer_provenance": provenance, "token_string_by_id": token_strings, "protocol": {"cases_per_template": 40, "cases_per_split": 120, "single_token_development": True}, "candidates": candidates, "cases": cases}
    payload["content_sha256"] = manifest_content_digest(payload)
    validate_manifest(payload)
    return payload


def freeze_manifest(path: Path, tokenizer: Any) -> None:
    """Write one canonical self-digesting artifact after tokenizer-only validation."""
    payload = build_manifest_payload(tokenizer)
    payload["content_sha256"] = manifest_content_digest(payload)
    validate_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(f"{path} has unknown or missing fields: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _parse_condition(value: Any, token_strings: Mapping[str, Any]) -> PromptCondition:
    if not isinstance(value, Mapping):
        raise ValueError("condition must be an object")
    _require_exact_keys(value, {"prompt_text", "prompt_token_ids", "prompt_token_strings", "a_text", "b_text", "a_token_ids", "b_token_ids", "a_token_strings", "b_token_strings", "target_position"}, "condition")
    for text_key in ("prompt_text", "a_text", "b_text"):
        _require_text(value[text_key], text_key)
    def ids_and_strings(id_key: str, string_key: str) -> tuple[int, ...]:
        ids, strings = value[id_key], value[string_key]
        if not isinstance(ids, list) or not ids or any(not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError(f"{id_key} must be nonempty nonnegative integer IDs")
        if not isinstance(strings, list) or len(strings) != len(ids) or any(not isinstance(token, str) or not token for token in strings):
            raise ValueError(f"{string_key} must align with {id_key}")
        if any(token_strings.get(str(token)) != string for token, string in zip(ids, strings)):
            raise ValueError("stored token IDs and token strings disagree")
        return tuple(ids)
    prompt_ids = ids_and_strings("prompt_token_ids", "prompt_token_strings")
    a_ids = ids_and_strings("a_token_ids", "a_token_strings")
    b_ids = ids_and_strings("b_token_ids", "b_token_strings")
    if len(a_ids) != len(b_ids):
        raise ValueError("A/B token lengths differ")
    return PromptCondition(value["prompt_text"], prompt_ids, a_ids, b_ids, value["target_position"], value["a_text"], value["b_text"])


def validate_manifest(manifest: Mapping[str, Any] | ScreeningManifest) -> ScreeningManifest:
    """Validate every declarative integrity invariant without loading model weights."""
    if isinstance(manifest, ScreeningManifest):
        return manifest
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping or ScreeningManifest")
    _require_exact_keys(manifest, {"schema_version", "seed", "models", "tokenizer_provenance", "token_string_by_id", "protocol", "candidates", "cases", "content_sha256"}, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION or manifest["seed"] != MANIFEST_SEED:
        raise ValueError("manifest schema version or seed is not frozen")
    digest = manifest_content_digest(manifest)
    if manifest["content_sha256"] != digest:
        raise ValueError("manifest content_sha256 does not match canonical payload")
    models = manifest["models"]
    expected_models = [{"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, {"model_id": PYTHIA_160M.model_id, "revision": PYTHIA_160M.revision}]
    if models != expected_models:
        raise ValueError("manifest model provenance is incompatible")
    provenance = manifest["tokenizer_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("tokenizer provenance must be an object")
    _require_exact_keys(provenance, {"model_ids", "vocabulary_sha256", "special_tokens", "special_token_ids"}, "tokenizer_provenance")
    if provenance["model_ids"] != [PYTHIA_70M.model_id, PYTHIA_160M.model_id] or not isinstance(provenance["vocabulary_sha256"], str):
        raise ValueError("tokenizer provenance is incompatible")
    if not isinstance(manifest["token_string_by_id"], Mapping):
        raise ValueError("token_string_by_id must be an object")
    protocol = manifest["protocol"]
    if protocol != {"cases_per_template": 40, "cases_per_split": 120, "single_token_development": True}:
        raise ValueError("manifest protocol is not frozen")
    candidates_raw = manifest["candidates"]
    if not isinstance(candidates_raw, list) or len(candidates_raw) != len(_CANDIDATE_ORDER):
        raise ValueError("manifest requires exactly the two frozen candidates")
    candidates: list[CandidateDefinition] = []
    for raw, expected_id in zip(candidates_raw, _CANDIDATE_ORDER):
        if not isinstance(raw, Mapping):
            raise ValueError("candidate must be an object")
        _require_exact_keys(raw, {"candidate_id", "template_ids", "local_heuristic"}, "candidate")
        if raw["candidate_id"] != expected_id or raw["template_ids"] != list(TEMPLATES[expected_id]):
            raise ValueError("candidate IDs or template order is not canonical")
        candidates.append(CandidateDefinition(raw["candidate_id"], tuple(raw["template_ids"]), raw["local_heuristic"]))
    cases_raw = manifest["cases"]
    if not isinstance(cases_raw, list) or len(cases_raw) != MANIFEST_CASE_COUNT:
        raise ValueError(f"manifest requires exactly {MANIFEST_CASE_COUNT} cases")
    cases: list[ScreeningCase] = []
    case_ids: set[str] = set()
    lexical: dict[tuple[str, str], set[str]] = {}
    orientations: dict[tuple[str, str, str], list[str]] = {}
    for raw in cases_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("case must be an object")
        _require_exact_keys(raw, {"case_id", "candidate_id", "template_id", "split", "lexical_key", "rule_class", "primary_orientation", "conditions", "local_heuristic_choices", "compactness_partition"}, "case")
        if raw["case_id"] in case_ids:
            raise ValueError("case IDs must be unique")
        case_ids.add(raw["case_id"])
        try:
            split = Split(raw["split"])
        except (TypeError, ValueError) as error:
            raise ValueError("case split is invalid") from error
        if raw["candidate_id"] not in _CANDIDATE_ORDER or raw["template_id"] not in TEMPLATES[raw["candidate_id"]]:
            raise ValueError("case candidate/template is invalid")
        conditions = raw["conditions"]
        if not isinstance(conditions, Mapping) or set(conditions) != {"x_a", "x_b", "s_a", "s_b"}:
            raise ValueError("case must contain exactly four conditions")
        parsed = {name: _parse_condition(conditions[name], manifest["token_string_by_id"]) for name in conditions}
        if split is Split.DEVELOPMENT and any(len(condition.a_token_ids) != 1 or len(condition.b_token_ids) != 1 for condition in parsed.values()):
            raise ValueError("development alternatives must be single tokens")
        compactness_raw = raw["compactness_partition"]
        if split is Split.FUTURE_RESERVE:
            if compactness_raw is not None:
                raise ValueError("future-reserve cases cannot have compactness labels")
            compactness = None
        else:
            if split is Split.HOLDOUT and compactness_raw is not None:
                raise ValueError("holdout cases cannot have compactness labels")
            try:
                compactness = CompactnessPartition(compactness_raw) if compactness_raw is not None else None
            except ValueError as error:
                raise ValueError("compactness partition is invalid") from error
            if split is Split.DEVELOPMENT and compactness is None:
                raise ValueError("development cases require a compactness partition")
        choices = raw["local_heuristic_choices"]
        if not isinstance(choices, Mapping) or set(choices) != {"A", "B"}:
            raise ValueError("local heuristic choices must be A/B")
        cases.append(ScreeningCase(raw["case_id"], raw["candidate_id"], raw["template_id"], split, raw["lexical_key"], raw["rule_class"], raw["primary_orientation"], parsed["x_a"], parsed["x_b"], parsed["s_a"], parsed["s_b"], choices, compactness))
        lexical.setdefault((raw["candidate_id"], split.value), set()).add(raw["lexical_key"])
        orientations.setdefault((raw["candidate_id"], split.value, raw["template_id"]), []).append(raw["primary_orientation"])
    for candidate in candidates:
        split_sets = [lexical.get((candidate.candidate_id, split.value), set()) for split in Split]
        if any(len(values) != 20 for values in split_sets) or any(left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1:]):
            raise ValueError("split lexical keys must be 20 and pairwise disjoint")
        for split in Split:
            if len([case for case in cases if case.candidate_id == candidate.candidate_id and case.split is split]) != 120:
                raise ValueError("candidate split must contain exactly 120 cases")
            for template in candidate.template_ids:
                rows = [case for case in cases if case.candidate_id == candidate.candidate_id and case.split is split and case.template_id == template]
                if len(rows) != 40 or orientations[(candidate.candidate_id, split.value, template)].count("A") != 20:
                    raise ValueError("template strata require 40 cases and balanced primary orientation")
        development = [case for case in cases if case.candidate_id == candidate.candidate_id and case.split is Split.DEVELOPMENT]
        if sum(case.compactness_partition is CompactnessPartition.DISCOVERY for case in development) != 60 or sum(case.compactness_partition is CompactnessPartition.VALIDATION for case in development) != 60:
            raise ValueError("development compactness partition must be 60/60")
    return ScreeningManifest(tuple(candidates), tuple(cases), schema_version=MANIFEST_SCHEMA_VERSION, content_sha256=manifest["content_sha256"])


def load_manifest(path: Path) -> ScreeningManifest:
    """Read the committed sole-authority JSON and reject all malformed variants."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read manifest: {path}") from error
    return validate_manifest(payload)


# ---------------------------------------------------------------------------
# Behavioral execution: teacher-forced scoring, frozen baselines, and gates.
# Future-reserve cases are loadable but no execution path below accepts them.

BASELINE_IDS = ("majority", "lexical-prior", "local-heuristic", "cue-shuffle")
EXECUTABLE_SPLITS = (Split.DEVELOPMENT, Split.HOLDOUT)


def _model_device(model: Any) -> Any:
    return getattr(getattr(model, "cfg", None), "device", "cpu")


def _log_probabilities(model: Any, tokens: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(tokens, prepend_bos=False)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("model must return [batch, position, vocabulary] logits")
    return logits.float().log_softmax(dim=-1)


def teacher_forced_log_probability(model: Any, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> float:
    """Sum log P(target_t | prompt, target_<t) from one forward over prompt + target."""
    prompt, target = tuple(prompt_ids), tuple(target_ids)
    if not prompt or not target:
        raise ValueError("prompt_ids and target_ids must be nonempty")
    full = torch.tensor([prompt + target], dtype=torch.long, device=_model_device(model))
    log_probs = _log_probabilities(model, full)
    vocabulary = int(log_probs.shape[-1])
    if any(token >= vocabulary for token in target):
        raise ValueError("target token ID is outside the model vocabulary")
    start = len(prompt) - 1
    terms = [log_probs[0, start + offset, token] for offset, token in enumerate(target)]
    return _finite(torch.stack(terms).sum().item(), "teacher-forced log probability")


def score_condition(model: Any, condition: PromptCondition, intended: str) -> ConditionMeasurement:
    """Score A and B in separate teacher-forced forwards using the stored token IDs only."""
    if not isinstance(condition, PromptCondition):
        raise TypeError("condition must be a PromptCondition")
    logp_a = teacher_forced_log_probability(model, condition.prompt_token_ids, condition.a_token_ids)
    logp_b = teacher_forced_log_probability(model, condition.prompt_token_ids, condition.b_token_ids)
    top1: int | None = None
    if condition.single_token:
        prompt = torch.tensor([condition.prompt_token_ids], dtype=torch.long, device=_model_device(model))
        top1 = int(_log_probabilities(model, prompt)[0, -1].argmax().item())
    return ConditionMeasurement(logp_a, logp_b, intended, top1)


def score_behavior_case(model: Any, case: ScreeningCase) -> CaseMeasurement:
    """Measure the four frozen conditions of one executable case."""
    if not isinstance(case, ScreeningCase):
        raise TypeError("case must be a ScreeningCase")
    if case.split not in EXECUTABLE_SPLITS:
        raise ValueError(f"split {case.split.value} is not executable during candidate selection")
    return CaseMeasurement(
        x_a=score_condition(model, case.x_a, "A"), x_b=score_condition(model, case.x_b, "B"),
        s_a=score_condition(model, case.s_a, "A"), s_b=score_condition(model, case.s_b, "B"),
        primary_orientation=case.primary_orientation, case_id=case.case_id, template_id=case.template_id)


def executable_cases(manifest: ScreeningManifest, candidate_id: str, split: Split) -> tuple[ScreeningCase, ...]:
    """Return manifest-ordered cases for one executable split; the reserve is refused."""
    if candidate_id not in manifest.candidate_ids:
        raise ValueError(f"unknown candidate {candidate_id}")
    if split not in EXECUTABLE_SPLITS:
        raise ValueError(f"split {split.value} is not executable during candidate selection")
    return manifest.cases_for(candidate_id, split)


def run_behavioral_screen(model: Any, manifest: ScreeningManifest, candidate_ids: Sequence[str] | None = None, *, on_case: Any = None, scorer: Any = None, completed: Mapping[str, CaseMeasurement] | None = None) -> dict[str, dict[str, tuple[CaseMeasurement, ...]]]:
    """Score development and holdout cases in manifest order; reserve cases never reach the scorer."""
    score = scorer or score_behavior_case
    done = dict(completed or {})
    ids = tuple(candidate_ids) if candidate_ids is not None else manifest.candidate_ids
    if any(candidate_id not in manifest.candidate_ids for candidate_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique manifest candidates")
    results: dict[str, dict[str, tuple[CaseMeasurement, ...]]] = {}
    for candidate_id in manifest.candidate_ids:
        if candidate_id not in ids:
            continue
        per_split: dict[str, tuple[CaseMeasurement, ...]] = {}
        for split in EXECUTABLE_SPLITS:
            rows: list[CaseMeasurement] = []
            for case in executable_cases(manifest, candidate_id, split):
                measurement = done.get(case.case_id)
                if measurement is None:
                    measurement = score(model, case)
                    if not isinstance(measurement, CaseMeasurement) or measurement.case_id != case.case_id:
                        raise ValueError("scorer must return the measurement of the requested case")
                    if on_case is not None:
                        on_case(measurement)
                rows.append(measurement)
            per_split[split.value] = tuple(rows)
        results[candidate_id] = per_split
    return results


def _intended_text(case: ScreeningCase, orientation: str) -> str:
    condition = case.x_a if orientation == "A" else case.x_b
    return (condition.a_text if orientation == "A" else condition.b_text).strip()


def _majority(orientations: Sequence[str]) -> str:
    return "A" if list(orientations).count("A") >= list(orientations).count("B") else "B"


def lexical_prior_predictions(manifest: ScreeningManifest, candidate_id: str) -> tuple[Mapping[str, str], str]:
    """Development-only lexical majorities plus the global development majority backoff (ties favor A)."""
    development = executable_cases(manifest, candidate_id, Split.DEVELOPMENT)
    per_key: dict[str, list[str]] = {}
    for case in development:
        per_key.setdefault(case.lexical_key, []).append(case.primary_orientation)
    return MappingProxyType({key: _majority(values) for key, values in per_key.items()}), _majority([case.primary_orientation for case in development])


def integrity_failures(manifest: ScreeningManifest, candidate_id: str, split: Split, measurements: Sequence[CaseMeasurement]) -> tuple[str, ...]:
    """Detect leakage, duplicates, missing cases, or mismatched case metadata."""
    cases = {case.case_id: case for case in executable_cases(manifest, candidate_id, split)}
    failures: list[str] = []
    seen: set[str] = set()
    for row in measurements:
        if row.case_id in seen:
            failures.append(f"duplicate measurement {row.case_id}")
        seen.add(row.case_id)
        case = cases.get(row.case_id)
        if case is None:
            failures.append(f"measurement {row.case_id} is not a {split.value} case of {candidate_id}")
        elif row.template_id != case.template_id or row.primary_orientation != case.primary_orientation:
            failures.append(f"measurement {row.case_id} disagrees with manifest metadata")
    missing = sorted(set(cases) - seen)
    if missing:
        failures.append(f"missing {len(missing)} {split.value} measurements")
    return tuple(failures)


def baseline_accuracies(manifest: ScreeningManifest, candidate_id: str, split: Split, measurements: Sequence[CaseMeasurement]) -> dict[str, float]:
    """Evaluate every frozen trivial baseline on the primary condition of each case."""
    cases = executable_cases(manifest, candidate_id, split)
    if not cases:
        raise ValueError("cannot evaluate baselines without cases")
    by_id = {row.case_id: row for row in measurements}
    lexical, backoff = lexical_prior_predictions(manifest, candidate_id)
    majority = _majority([case.primary_orientation for case in cases])
    total = len(cases)
    return {
        "majority": sum(case.primary_orientation == majority for case in cases) / total,
        "lexical-prior": sum(lexical.get(case.lexical_key, backoff) == case.primary_orientation for case in cases) / total,
        "local-heuristic": sum(case.local_heuristic_choices[case.primary_orientation] == _intended_text(case, case.primary_orientation) for case in cases) / total,
        "cue-shuffle": sum(by_id[case.case_id].cue_shuffle_correct for case in cases if case.case_id in by_id) / total,
    }


def behavioral_result_for_candidate(manifest: ScreeningManifest, candidate_id: str, development: Sequence[CaseMeasurement], holdout: Sequence[CaseMeasurement]) -> dict[str, Any]:
    """Summarize both executable splits and apply every frozen gate to holdout."""
    summaries: dict[str, CandidateBehaviorSummary] = {}
    for split, rows in ((Split.DEVELOPMENT, development), (Split.HOLDOUT, holdout)):
        summaries[split.value] = summarize_behavior(
            rows, candidate_id=candidate_id, baseline_accuracies=baseline_accuracies(manifest, candidate_id, split, rows),
            integrity_failures=integrity_failures(manifest, candidate_id, split, rows))
    gates = evaluate_behavioral_gates(summaries[Split.DEVELOPMENT.value], summaries[Split.HOLDOUT.value])
    return _json({"candidate_id": candidate_id, "development": summaries[Split.DEVELOPMENT.value].to_dict(),
                  "holdout": summaries[Split.HOLDOUT.value].to_dict(), "gates": gates.to_dict()}, "behavioral_result")


# ---------------------------------------------------------------------------
# Exploratory compactness probe: exact bidirectional replacement at the final
# prompt position over the closed head/MLP universe, development data only.

COMPACTNESS_SEED = 20260917
COMPACTNESS_MAX_K = 12
COMPACTNESS_RANDOM_SETS = 100
COMPACTNESS_BATCH_SIZE = 16
_ATTN_SETTINGS = InstrumentationSettings(use_attn_result=True)


class CompactnessIntegrityError(RuntimeError):
    """An intervention did not perform the exact declared replacement."""


def canonical_component_id(ref: ComponentRef) -> str:
    if ref.kind is ComponentKind.ATTN_HEAD:
        return f"L{ref.layer:02d}.H{ref.head:02d}"
    if ref.kind is ComponentKind.MLP_OUT:
        return f"L{ref.layer:02d}.MLP"
    raise ValueError("compactness components are attention heads and whole MLP outputs only")


def component_universe(model: Any) -> tuple[ComponentRef, ...]:
    """Every attention-head result then the whole MLP output, layer by layer."""
    cfg = getattr(model, "cfg", None)
    n_layers, n_heads = int(getattr(cfg, "n_layers", 0)), int(getattr(cfg, "n_heads", 0))
    if n_layers <= 0 or n_heads <= 0:
        raise ValueError("model cfg must declare positive n_layers and n_heads")
    universe: list[ComponentRef] = []
    for layer in range(n_layers):
        universe.extend(ComponentRef(ComponentKind.ATTN_HEAD, layer=layer, head=head) for head in range(n_heads))
        universe.append(ComponentRef(ComponentKind.MLP_OUT, layer=layer))
    return tuple(universe)


def validate_compactness_cases(cases: Sequence[ScreeningCase]) -> tuple[ScreeningCase, ...]:
    """Only partitioned, single-token, final-position development cases may be probed."""
    rows = tuple(cases)
    if not rows:
        raise ValueError("compactness requires cases")
    for case in rows:
        if not isinstance(case, ScreeningCase) or case.split is not Split.DEVELOPMENT or case.compactness_partition is None:
            raise ValueError("compactness accepts only partitioned selection-development cases")
        for condition in (case.x_a, case.x_b):
            if not condition.single_token:
                raise ValueError("compactness requires single-token A/B alternatives")
            if condition.target_position != -1:
                raise ValueError("compactness requires target position -1")
    return rows


@dataclass(frozen=True)
class PairSources:
    """Unpatched contrasts and final-position activations of one matched pair."""

    case_id: str
    template_id: str
    c_a: float
    c_b: float
    a_slices: Mapping[str, torch.Tensor]
    b_slices: Mapping[str, torch.Tensor]

    @property
    def d_full(self) -> float:
        return self.c_a - self.c_b


def _contrast_from_logits(logits: torch.Tensor, condition: PromptCondition) -> float:
    log_probs = logits.float().log_softmax(dim=-1)
    return fixed_contrast(float(log_probs[condition.a_token_ids[0]]), float(log_probs[condition.b_token_ids[0]]))


def _batches(cases: Sequence[ScreeningCase], batch_size: int) -> list[tuple[ScreeningCase, ...]]:
    groups: dict[tuple[int, int], list[ScreeningCase]] = {}
    for case in cases:
        groups.setdefault((len(case.x_a.prompt_token_ids), len(case.x_b.prompt_token_ids)), []).append(case)
    batches: list[tuple[ScreeningCase, ...]] = []
    for key in sorted(groups):
        rows = groups[key]
        batches.extend(tuple(rows[start:start + batch_size]) for start in range(0, len(rows), batch_size))
    return batches


def capture_pair_sources(model: Any, cases: Sequence[ScreeningCase], universe: Sequence[ComponentRef], *, batch_size: int = COMPACTNESS_BATCH_SIZE, capture: Any = None) -> dict[str, PairSources]:
    """Capture every universe component at position -1 for x_A and x_B of each pair."""
    run = capture or run_capture
    rows = validate_compactness_cases(cases)
    requests = tuple(CaptureRequest(ref, positions=(-1,)) for ref in universe)
    plan = CapturePlan(requests, _ATTN_SETTINGS)
    device = _model_device(model)
    contrasts: dict[str, dict[str, float]] = {}
    slices: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for batch in _batches(rows, batch_size):
        for side in ("a", "b"):
            conditions = [case.x_a if side == "a" else case.x_b for case in batch]
            tokens = torch.tensor([condition.prompt_token_ids for condition in conditions], dtype=torch.long, device=device)
            result = run(model, tokens, plan, prepend_bos=False)
            if not result.hook_settings.effective_use_attn_result or result.hook_settings.compatibility_mode:
                raise CompactnessIntegrityError("capture must run with use_attn_result and without compatibility mode")
            for index, (case, condition) in enumerate(zip(batch, conditions)):
                contrasts.setdefault(case.case_id, {})[side] = _contrast_from_logits(result.logits[index, -1], condition)
                slices.setdefault(case.case_id, {})[side] = {
                    canonical_component_id(request.component): result.activations[request].tensor[index:index + 1].clone()
                    for request in requests}
    return {case.case_id: PairSources(case.case_id, case.template_id, contrasts[case.case_id]["a"], contrasts[case.case_id]["b"],
                                      MappingProxyType(slices[case.case_id]["a"]), MappingProxyType(slices[case.case_id]["b"]))
            for case in rows}


def _check_execution(model: Any, execution: Any, ref: ComponentRef, replacement: torch.Tensor, sequence_length: int) -> None:
    expected_hook = resolve_component(ref, model).hook_name
    problems = []
    if execution.component != ref or execution.hook_name != expected_hook:
        problems.append("component/hook mismatch")
    if execution.operation is not InterventionOperation.REPLACE or execution.source is not ReplacementSource.REFERENCE:
        problems.append("operation/source mismatch")
    if tuple(execution.shape) != tuple(replacement.shape) or execution.dtype != str(replacement.dtype):
        problems.append("shape/dtype mismatch")
    if tuple(execution.normalized_positions) != (sequence_length - 1,):
        problems.append("position mismatch")
    if execution.outside_max_abs_change != 0.0:
        problems.append("outside change")
    if not torch.equal(execution.after.to(replacement.device), replacement):
        problems.append("inexact replacement")
    if problems:
        raise CompactnessIntegrityError(f"{canonical_component_id(ref)}: {', '.join(problems)}")


def patched_contrasts(model: Any, cases: Sequence[ScreeningCase], sources: Mapping[str, PairSources], components: Sequence[ComponentRef], *, direction: str, batch_size: int = COMPACTNESS_BATCH_SIZE, intervene: Any = None) -> dict[str, float]:
    """Contrast of each pair's target prompt with components replaced from its counterpart run."""
    if direction not in {"a_from_b", "b_from_a"}:
        raise ValueError("direction must be a_from_b or b_from_a")
    run = intervene or run_interventions
    device = _model_device(model)
    refs = tuple(components)
    if len({canonical_component_id(ref) for ref in refs}) != len(refs):
        raise ValueError("component set must not repeat components")
    results: dict[str, float] = {}
    for batch in _batches(cases, batch_size):
        conditions = [case.x_a if direction == "a_from_b" else case.x_b for case in batch]
        tokens = torch.tensor([condition.prompt_token_ids for condition in conditions], dtype=torch.long, device=device)
        replacements = []
        for ref in refs:
            key = canonical_component_id(ref)
            side = "b_slices" if direction == "a_from_b" else "a_slices"
            replacements.append(torch.cat([getattr(sources[case.case_id], side)[key] for case in batch], dim=0).to(device))
        plan = InterventionPlan(tuple(Intervention(ref, (-1,), InterventionOperation.REPLACE, replacement, ReplacementSource.REFERENCE)
                                      for ref, replacement in zip(refs, replacements)), _ATTN_SETTINGS)
        result = run(model, tokens, plan, prepend_bos=False) if refs else None
        if result is None:
            with torch.inference_mode():
                logits = model(tokens, prepend_bos=False)
        else:
            if len(result.executions) != len(refs) or result.hook_settings.compatibility_mode or not result.hook_settings.effective_use_attn_result:
                raise CompactnessIntegrityError("intervention run did not execute every declared replacement")
            for ref, replacement, execution in zip(refs, replacements, result.executions):
                _check_execution(model, execution, ref, replacement, int(tokens.shape[1]))
            logits = result.logits
        for index, (case, condition) in enumerate(zip(batch, conditions)):
            results[case.case_id] = _contrast_from_logits(logits[index, -1], condition)
    return results


def patched_pair_shifts(model: Any, cases: Sequence[ScreeningCase], sources: Mapping[str, PairSources], components: Sequence[ComponentRef], **options: Any) -> tuple[PairPatchMeasurement, ...]:
    """Bidirectional aligned shift d_patch(S) for every case, in manifest order."""
    a_from_b = patched_contrasts(model, cases, sources, components, direction="a_from_b", **options)
    b_from_a = patched_contrasts(model, cases, sources, components, direction="b_from_a", **options)
    rows = []
    for case in cases:
        source = sources[case.case_id]
        shift = aligned_patch_shift(source.c_a, source.c_b, a_from_b[case.case_id], b_from_a[case.case_id])
        rows.append(PairPatchMeasurement(source.d_full, shift, case.template_id))
    return tuple(rows)


def rank_discovery_components(singleton_effects: Mapping[str, Sequence[PairPatchMeasurement]]) -> tuple[str, ...]:
    """Descending mean discovery effect; exact ties break by canonical component ID."""
    means = {key: statistics.fmean(row.d_patch for row in rows) for key, rows in singleton_effects.items()}
    return tuple(sorted(means, key=lambda key: (-means[key], key)))


def deterministic_random_sets(universe_ids: Sequence[str], size: int, *, count: int = COMPACTNESS_RANDOM_SETS, seed: int = COMPACTNESS_SEED) -> tuple[tuple[str, ...], ...]:
    rng = random.Random(seed)
    ordered = list(universe_ids)
    if not 1 <= size <= len(ordered):
        raise ValueError("random set size must be between 1 and the universe size")
    return tuple(tuple(sorted(rng.sample(ordered, size))) for _ in range(count))


def _validation_denominators(rows: Sequence[PairPatchMeasurement]) -> DenominatorValidation:
    templates: dict[str, list[float]] = {}
    for row in rows:
        templates.setdefault(row.template_id, []).append(row.d_full)
    return evaluate_compactness_denominators(statistics.fmean(row.d_full for row in rows), {key: statistics.fmean(values) for key, values in templates.items()})


def run_compactness_probe(model: Any, manifest: ScreeningManifest, candidate_id: str, *, on_progress: Any = None, **options: Any) -> dict[str, Any]:
    """Discovery-only ranking, validation-only cumulative and random recovery, one frozen gate."""
    cases = validate_compactness_cases(executable_cases(manifest, candidate_id, Split.DEVELOPMENT))
    discovery = tuple(case for case in cases if case.compactness_partition is CompactnessPartition.DISCOVERY)
    validation = tuple(case for case in cases if case.compactness_partition is CompactnessPartition.VALIDATION)
    if len(discovery) != 60 or len(validation) != 60:
        raise ValueError("compactness requires a 60/60 discovery/validation partition")
    universe = component_universe(model)
    by_id = {canonical_component_id(ref): ref for ref in universe}
    capture_options = {key: options[key] for key in ("batch_size", "capture") if key in options}
    patch_options = {key: options[key] for key in ("batch_size", "intervene") if key in options}
    sources = capture_pair_sources(model, cases, universe, **capture_options)

    def progress(stage: str, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    singleton: dict[str, tuple[PairPatchMeasurement, ...]] = {}
    for index, ref in enumerate(universe):
        singleton[canonical_component_id(ref)] = patched_pair_shifts(model, discovery, sources, (ref,), **patch_options)
        progress("discovery", index + 1, len(universe))
    ranking = rank_discovery_components(singleton)
    validation_rows = tuple(PairPatchMeasurement(sources[case.case_id].d_full, 0.0, case.template_id) for case in validation)
    denominators = _validation_denominators(validation_rows)
    top_k: dict[int, tuple[PairPatchMeasurement, ...]] = {}
    random_results: dict[int, dict[str, Any]] = {}
    random_recoveries: dict[int, list[float]] = {}
    if denominators.valid:
        for k in range(1, COMPACTNESS_MAX_K + 1):
            top_k[k] = patched_pair_shifts(model, validation, sources, tuple(by_id[key] for key in ranking[:k]), **patch_options)
            progress("top-k", k, COMPACTNESS_MAX_K)
        for k in range(1, COMPACTNESS_MAX_K + 1):
            sets = deterministic_random_sets(tuple(by_id), k, count=COMPACTNESS_RANDOM_SETS)
            recoveries = [aggregate_recovery(patched_pair_shifts(model, validation, sources, tuple(by_id[key] for key in members), **patch_options)) for members in sets]
            random_recoveries[k] = recoveries
            random_results[k] = {"sets": [list(members) for members in sets], "recoveries": recoveries, "median": statistics.median(recoveries), "reference": random_reference(recoveries)}
            progress("random", k, COMPACTNESS_MAX_K)
        gate = evaluate_compactness_gate(top_k, random_recoveries)
    else:
        gate = CompactnessGateResult({"denominators": False, "overall_recovery": False, "beats_random": False, "positive_template_recovery": False}, False)
    return _json({
        "candidate_id": candidate_id, "seed": COMPACTNESS_SEED, "max_k": COMPACTNESS_MAX_K, "random_sets_per_k": COMPACTNESS_RANDOM_SETS,
        "component_universe": list(by_id), "discovery_case_ids": [case.case_id for case in discovery], "validation_case_ids": [case.case_id for case in validation],
        "unpatched": {case.case_id: {"c_a": sources[case.case_id].c_a, "c_b": sources[case.case_id].c_b, "d_full": sources[case.case_id].d_full, "template_id": case.template_id} for case in cases},
        "validation_denominators": denominators.to_dict(),
        "singleton_discovery": {key: {"mean_d_patch": statistics.fmean(row.d_patch for row in rows), "effects": [row.to_dict() for row in rows]} for key, rows in singleton.items()},
        "ranking": list(ranking),
        "top_k": {str(k): {"components": list(ranking[:k]), "measurements": [row.to_dict() for row in rows]} for k, rows in top_k.items()},
        "random": {str(k): value for k, value in random_results.items()},
        "gate": gate.to_dict(),
    }, "compactness_result")

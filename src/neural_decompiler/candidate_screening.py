"""Frozen, model-independent declarations and metrics for candidate screening."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .behavior import validate_json_safe


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

    def __post_init__(self) -> None:
        _require_text(self.prompt_text, "prompt_text")
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
        }, "prompt_condition")


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

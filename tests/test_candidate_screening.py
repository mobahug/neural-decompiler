import copy
import json
from pathlib import Path

import pytest

import neural_decompiler.candidate_screening as candidate_screening
from neural_decompiler.models import PYTHIA_160M, PYTHIA_70M

from neural_decompiler.candidate_screening import (
    CandidateBehaviorSummary,
    CompactnessPartition,
    PairPatchMeasurement,
    PromptCondition,
    ScreeningCase,
    Split,
    aggregate_recovery,
    correctness_margin,
    evaluate_behavioral_gates,
    evaluate_compactness_gate,
    evaluate_compactness_denominators,
    fixed_contrast,
    measure_case,
    passes_overall_accuracy_gate,
    random_reference,
    required_random_beating_recovery,
    wilson_lower_bound,
    load_manifest,
    validate_manifest,
)


MANIFEST_PATH = Path(__file__).parents[1] / "screening/behavior-candidates/manifest-v1.json"


def test_committed_manifest_has_exact_candidates_splits_and_counts() -> None:
    """A missing, reordered, or incompletely stratified frozen case set is invalid."""
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.candidate_ids == ("regular-plural", "ordinal-suffix")
    assert len(manifest.cases) == 720
    for candidate in manifest.candidates:
        assert len(candidate.template_ids) == 3
        for split in Split:
            assert len(manifest.cases_for(candidate.candidate_id, split)) == 120
            assert all(
                len(manifest.cases_for(candidate.candidate_id, split, template)) == 40
                for template in candidate.template_ids
            )


def test_development_cases_are_single_token_and_partitioned() -> None:
    """Development compactness may not consume holdout/reserve cases or multi-token rows."""
    manifest = load_manifest(MANIFEST_PATH)
    for candidate_id in manifest.candidate_ids:
        cases = manifest.cases_for(candidate_id, Split.DEVELOPMENT)
        assert all(
            len(condition.a_token_ids) == len(condition.b_token_ids) == 1
            for case in cases
            for condition in (case.x_a, case.x_b, case.s_a, case.s_b)
        )
        assert sum(case.compactness_partition is CompactnessPartition.DISCOVERY for case in cases) == 60
        assert sum(case.compactness_partition is CompactnessPartition.VALIDATION for case in cases) == 60


@pytest.mark.parametrize("field", ["prompt_text", "split", "a_token_ids", "content_sha256"])
def test_manifest_validation_rejects_tampering(field: str) -> None:
    """Changing scientific content or its digest cannot pass integrity validation."""
    payload = json.loads(MANIFEST_PATH.read_text())
    changed = copy.deepcopy(payload)
    if field == "content_sha256":
        changed[field] = "0" * 64
    elif field == "split":
        changed["cases"][0][field] = Split.HOLDOUT.value
    elif field == "prompt_text":
        changed["cases"][0]["conditions"]["x_a"][field] += " altered"
    else:
        changed["cases"][0]["conditions"]["x_a"][field][0] += 1
    with pytest.raises(ValueError):
        validate_manifest(changed)


def test_development_word_eligibility_rejects_equal_length_multitoken_targets() -> None:
    """A two-token/two-token row cannot enter the single-token compactness split."""
    class TwoTokenNounTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            if text.endswith((" quilt", " quilts")):
                return [1, 2, 3]
            return [1]

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [f"token-{token}" for token in ids]

        def convert_tokens_to_ids(self, token: str) -> int:
            return int(token.removeprefix("token-"))

    tokenizers = {
        PYTHIA_70M.model_id: TwoTokenNounTokenizer(),
        PYTHIA_160M.model_id: TwoTokenNounTokenizer(),
    }

    assert not candidate_screening._word_eligible(tokenizers, Split.DEVELOPMENT.value, "quilt")
    assert candidate_screening._word_eligible(tokenizers, Split.HOLDOUT.value, "quilt")


def test_local_heuristic_is_wrong_on_exactly_half_of_every_template_stratum() -> None:
    """The shallow rule must fail on the 20 predeclared exception cases per template, never more or fewer."""
    manifest = load_manifest(MANIFEST_PATH)
    payload = json.loads(MANIFEST_PATH.read_text())
    intended = {
        (case["case_id"], "A"): case["conditions"]["x_a"]["a_text"].strip()
        for case in payload["cases"]
    } | {
        (case["case_id"], "B"): case["conditions"]["x_b"]["b_text"].strip()
        for case in payload["cases"]
    }
    for candidate in manifest.candidates:
        for split in Split:
            for template in candidate.template_ids:
                rows = manifest.cases_for(candidate.candidate_id, split, template)
                wrong = [
                    case for case in rows
                    if case.local_heuristic_choices[case.primary_orientation]
                    != intended[(case.case_id, case.primary_orientation)]
                ]
                assert len(rows) == 40 and len(wrong) == 20
                assert all(case.primary_orientation == "B" for case in wrong)
                assert all(case.rule_class != "simple-suffix" for case in wrong)


def test_split_lexical_keys_are_pairwise_disjoint_and_reserve_is_enumerable() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    for candidate_id in manifest.candidate_ids:
        keys = [
            {case.lexical_key for case in manifest.cases_for(candidate_id, split)}
            for split in Split
        ]
        assert all(len(values) == 20 for values in keys)
        assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
        reserve = manifest.cases_for(candidate_id, Split.FUTURE_RESERVE)
        assert len(reserve) == 120 and all(case.compactness_partition is None for case in reserve)


def test_fixed_contrast_and_correctness_margin_do_not_conflate_orientation():
    """A fixed A-minus-B contrast must not become an intended-minus-foil score."""
    assert fixed_contrast(-1.0, -3.0) == pytest.approx(2.0)
    assert correctness_margin(2.0, intended="A") == pytest.approx(2.0)
    assert correctness_margin(-2.0, intended="B") == pytest.approx(2.0)


def test_case_measurement_keeps_correctness_positive_while_contrast_flips():
    """Counterfactual correctness remains positive even though fixed contrast flips."""
    measured = measure_case(
        logp={
            "x_a": {"A": -1.0, "B": -3.0},
            "x_b": {"A": -3.0, "B": -1.0},
            "s_a": {"A": -3.0, "B": -1.0},
            "s_b": {"A": -1.0, "B": -3.0},
        },
        primary_orientation="A",
    )
    assert measured.x_a.contrast == pytest.approx(2.0)
    assert measured.x_b.contrast == pytest.approx(-2.0)
    assert measured.x_a.correctness_margin == pytest.approx(2.0)
    assert measured.x_b.correctness_margin == pytest.approx(2.0)
    assert measured.d_full == pytest.approx(4.0)
    assert measured.d_cue == pytest.approx(4.0)


def test_wilson_gate_makes_103_of_120_the_effective_minimum():
    """The Wilson lower bound, rather than a rounded percentage, sets the boundary."""
    assert wilson_lower_bound(102, 120) == pytest.approx(0.775325, abs=1e-6)
    assert wilson_lower_bound(103, 120) == pytest.approx(0.784805, abs=1e-6)
    assert not passes_overall_accuracy_gate(102, 120)
    assert passes_overall_accuracy_gate(103, 120)


def test_frozen_case_declarations_preserve_four_conditions_and_json_values():
    """A case cannot omit one paired/control condition or expose mutable metadata."""
    condition = PromptCondition(
        prompt_text="one cat",
        prompt_token_ids=(1, 2),
        a_token_ids=(3,),
        b_token_ids=(4,),
    )
    choices = {"A": "append-s", "B": "append-s"}
    case = ScreeningCase(
        case_id="plural-t1-001",
        candidate_id="regular-plural",
        template_id="t1",
        split=Split.DEVELOPMENT,
        lexical_key="cat",
        rule_class="regular-s",
        primary_orientation="A",
        x_a=condition,
        x_b=condition,
        s_a=condition,
        s_b=condition,
        local_heuristic_choices=choices,
        compactness_partition=CompactnessPartition.DISCOVERY,
    )
    choices["A"] = "changed-after-construction"

    assert case.local_heuristic_choices["A"] == "append-s"
    assert case.to_dict()["conditions"] == {
        "x_a": condition.to_dict(),
        "x_b": condition.to_dict(),
        "s_a": condition.to_dict(),
        "s_b": condition.to_dict(),
    }
    assert json.dumps(case.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="target_position"):
        PromptCondition("prompt", (1,), (2,), (3,), target_position=0)


def _passing_summary(*, correct_count=103, **changes):
    values = {
        "candidate_id": "regular-plural",
        "case_count": 120,
        "primary_correct_count": correct_count,
        "template_accuracies": {"t1": 0.85, "t2": 0.85, "t3": 0.875},
        "template_mean_margins": {"t1": 1.0, "t2": 1.0, "t3": 1.0},
        "contrast_flip_rate": 0.80,
        "cue_shuffle_accuracy": 0.65,
        "mean_d_full": 1.0,
        "mean_d_cue": 0.5,
        "baseline_accuracies": {
            "majority": 0.70,
            "lexical-prior": 0.70,
            "local-heuristic": 0.70,
        },
        "integrity_failures": (),
    }
    values.update(changes)
    return CandidateBehaviorSummary(**values)


def test_behavioral_gate_records_every_frozen_rule_when_all_rules_pass():
    """The result matrix must retain every numbered gate, rather than a bare boolean."""
    result = evaluate_behavioral_gates(_passing_summary(correct_count=110), _passing_summary())

    assert result.passed
    assert tuple(result.checks) == (
        "overall_accuracy",
        "wilson_lower_bound",
        "per_template_accuracy",
        "template_range",
        "development_drop",
        "positive_template_margin",
        "contrast_flip",
        "cue_dependence",
        "beats_baselines",
        "integrity",
    )
    assert all(result.checks.values())
    assert json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("changed_holdout", "changed_development", "failed_check"),
    [
        ({"case_count": 119, "primary_correct_count": 103}, {}, "overall_accuracy"),
        ({"primary_correct_count": 102}, {}, "wilson_lower_bound"),
        ({"template_accuracies": {"t1": 0.74, "t2": 0.90, "t3": 0.90}}, {}, "per_template_accuracy"),
        ({"template_accuracies": {"t1": 0.75, "t2": 0.90, "t3": 0.90}}, {}, "template_range"),
        ({}, {"primary_correct_count": 116}, "development_drop"),
        ({"template_mean_margins": {"t1": -0.01, "t2": 1.0, "t3": 1.0}}, {}, "positive_template_margin"),
        ({"contrast_flip_rate": 0.79}, {}, "contrast_flip"),
        ({"cue_shuffle_accuracy": 0.66, "mean_d_cue": 0.49}, {}, "cue_dependence"),
        ({"baseline_accuracies": {"majority": 0.71, "lexical-prior": 0.70, "local-heuristic": 0.70}}, {}, "beats_baselines"),
        ({"integrity_failures": ("duplicate lexical item",)}, {}, "integrity"),
    ],
)
def test_behavioral_gate_does_not_allow_any_later_rule_to_rescue_a_failed_rule(
    changed_holdout, changed_development, failed_check
):
    """Each frozen requirement independently prevents selection when it is violated."""
    development = _passing_summary(correct_count=110, **changed_development)
    holdout = _passing_summary(**changed_holdout)

    if holdout.case_count == 120:
        result = evaluate_behavioral_gates(development, holdout)
        assert not result.checks[failed_check]
        assert not result.passed
    else:
        with pytest.raises(ValueError, match="exactly 120"):
            evaluate_behavioral_gates(development, holdout)


def test_compactness_uses_ratio_of_aggregate_shifts_and_never_divides_bad_denominators():
    """Recovery must not average per-row ratios or paper over weak contrasts."""
    rows = (
        PairPatchMeasurement(d_full=4.0, d_patch=3.0, template_id="t1"),
        PairPatchMeasurement(d_full=2.0, d_patch=1.0, template_id="t2"),
    )
    assert aggregate_recovery(rows) == pytest.approx(4.0 / 6.0)

    denominator = evaluate_compactness_denominators(
        0.249, {"t1": 0.20, "t2": 0.20, "t3": 0.20}
    )
    assert not denominator.valid
    assert denominator.recovery is None
    assert json.dumps(denominator.to_dict(), allow_nan=False)


def test_random_compactness_reference_has_a_nonvacuous_floor():
    """A non-positive random median cannot make any positive recovery look impressive."""
    recoveries = (-0.4, -0.1, 0.0)
    assert random_reference(recoveries) == pytest.approx(0.05)
    assert required_random_beating_recovery(recoveries) == pytest.approx(0.10)


def test_compactness_gate_selects_the_smallest_valid_set_and_serializes_it():
    """A valid k records raw diagnostics with JSON-safe, deterministic keys."""
    result = evaluate_compactness_gate(
        {
            1: (
                PairPatchMeasurement(d_full=1.0, d_patch=0.8, template_id="t1"),
                PairPatchMeasurement(d_full=1.0, d_patch=0.8, template_id="t2"),
                PairPatchMeasurement(d_full=1.0, d_patch=0.8, template_id="t3"),
            )
        },
        {1: (0.1, 0.1, 0.1)},
    )

    assert result.passed
    assert result.selected_k == 1
    assert result.to_dict()["evaluations"]["1"]["recovery"] == pytest.approx(0.8)
    assert json.dumps(result.to_dict(), allow_nan=False)


def test_result_declarations_reject_nonfinite_values_before_json_serialization():
    """A NaN measurement must fail at its protocol boundary, not in a later report."""
    with pytest.raises(ValueError, match="finite"):
        PairPatchMeasurement(d_full=float("nan"), d_patch=1.0, template_id="t1")

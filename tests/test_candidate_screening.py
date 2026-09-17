import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
        for template in {case.template_id for case in cases}:
            for partition in CompactnessPartition:
                rows = [case for case in cases if case.template_id == template and case.compactness_partition is partition]
                assert len(rows) == 20
                assert sum(case.primary_orientation == "A" for case in rows) == 10
                assert sum(case.rule_class == "simple-suffix" for case in rows) in (0, 10)


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
    """Synthetic count-based summary; template corrects are split as evenly as possible."""
    base, extra = divmod(correct_count, 3)
    values = {
        "candidate_id": "regular-plural",
        "case_count": 120,
        "primary_correct_count": correct_count,
        "template_case_counts": {"t1": 40, "t2": 40, "t3": 40},
        "template_correct_counts": {"t1": base, "t2": base + (extra > 1), "t3": base + (extra > 0)},
        "template_mean_margins": {"t1": 1.0, "t2": 1.0, "t3": 1.0},
        "contrast_flip_count": 96,
        "cue_shuffle_correct_count": 78,
        "mean_d_full": 1.0,
        "mean_d_cue": 0.5,
        "baseline_correct_counts": {
            "majority": 84,
            "lexical-prior": 84,
            "local-heuristic": 84,
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
        ({"case_count": 119, "primary_correct_count": 103, "template_case_counts": {"t1": 39, "t2": 40, "t3": 40}}, {}, "overall_accuracy"),
        ({"primary_correct_count": 102, "template_correct_counts": {"t1": 34, "t2": 34, "t3": 34}}, {}, "wilson_lower_bound"),
        ({"primary_correct_count": 101, "template_correct_counts": {"t1": 29, "t2": 36, "t3": 36}}, {}, "per_template_accuracy"),
        ({"template_correct_counts": {"t1": 30, "t2": 37, "t3": 36}}, {}, "template_range"),
        ({}, {"primary_correct_count": 116, "template_correct_counts": {"t1": 39, "t2": 38, "t3": 39}}, "development_drop"),
        ({"template_mean_margins": {"t1": -0.01, "t2": 1.0, "t3": 1.0}}, {}, "positive_template_margin"),
        ({"contrast_flip_count": 95}, {}, "contrast_flip"),
        ({"cue_shuffle_correct_count": 80, "mean_d_cue": 0.49}, {}, "cue_dependence"),
        ({"baseline_correct_counts": {"majority": 86, "lexical-prior": 84, "local-heuristic": 84}}, {}, "beats_baselines"),
        ({"integrity_failures": ("duplicate lexical item",)}, {}, "integrity"),
        ({}, {"integrity_failures": ("missing 1 selection-development measurements",)}, "integrity"),
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


@pytest.mark.parametrize(
    ("changed_holdout", "changed_development", "boundary_check"),
    [
        ({"primary_correct_count": 102, "template_correct_counts": {"t1": 30, "t2": 36, "t3": 36}}, {}, "template_range"),  # exactly 15 points
        ({}, {"primary_correct_count": 115, "template_correct_counts": {"t1": 38, "t2": 38, "t3": 39}}, "development_drop"),  # exactly 10 points
        ({"baseline_correct_counts": {"majority": 85, "lexical-prior": 84, "local-heuristic": 84}}, {}, "beats_baselines"),  # exactly 15 points
        ({"cue_shuffle_correct_count": 79, "mean_d_cue": 0.0}, {}, "cue_dependence"),  # exactly 20 points
        ({"contrast_flip_count": 96}, {}, "contrast_flip"),  # exactly 80%
    ],
)
def test_boundary_accuracies_are_decided_exactly_not_by_float_rounding(changed_holdout, changed_development, boundary_check):
    """A candidate sitting exactly on a frozen threshold passes that gate."""
    development = _passing_summary(correct_count=110, **changed_development)
    holdout = _passing_summary(**changed_holdout)
    result = evaluate_behavioral_gates(development, holdout)
    assert result.checks[boundary_check]


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


# ---------------------------------------------------------------------------
# Behavioral execution


class CausalFakeModel:
    """Prefix-sensitive causal fake: position t is peaked at (sum of tokens[:t+1]) mod vocab."""

    vocab = 16

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(device="cpu")
        self.calls: list[list[int]] = []

    def __call__(self, tokens: torch.Tensor, *, prepend_bos: bool = False) -> torch.Tensor:
        assert not prepend_bos
        self.calls.append(tokens[0].tolist())
        sums = tokens.cumsum(dim=-1) % self.vocab
        grid = torch.arange(self.vocab, dtype=torch.float32)
        return -(grid.view(1, 1, -1) - sums.unsqueeze(-1).float()).abs()


def test_teacher_forced_sequence_score_uses_each_alternatives_own_prefix() -> None:
    model = CausalFakeModel()
    score = candidate_screening.teacher_forced_log_probability(model, prompt_ids=(4, 5), target_ids=(6, 7))
    full = model(torch.tensor([[4, 5, 6, 7]])).log_softmax(dim=-1)
    assert score == pytest.approx(float(full[0, 1, 6] + full[0, 2, 7]))
    other = candidate_screening.teacher_forced_log_probability(model, prompt_ids=(4, 5), target_ids=(9, 7))
    assert other != pytest.approx(score)
    with pytest.raises(ValueError, match="vocabulary"):
        candidate_screening.teacher_forced_log_probability(model, (4,), (99,))


def test_single_token_condition_records_top1_but_multitoken_does_not() -> None:
    model = CausalFakeModel()
    single = PromptCondition("p", (4, 5), (9,), (2,), a_text=" a", b_text=" b")
    multi = PromptCondition("p", (4, 5), (9, 1), (2, 1))
    measured_single = candidate_screening.score_condition(model, single, "A")
    measured_multi = candidate_screening.score_condition(model, multi, "B")
    assert measured_single.top1_token_id == 9
    assert measured_single.logp_a > measured_single.logp_b
    assert measured_multi.top1_token_id is None
    assert measured_multi.intended == "B"


def test_behavioral_screen_scores_all_and_only_development_and_holdout_cases() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    seen: list[str] = []

    def spy(model: object, case: ScreeningCase) -> candidate_screening.CaseMeasurement:
        seen.append(case.case_id)
        return _measurement_for(case, correct=True)

    results = candidate_screening.run_behavioral_screen(object(), manifest, scorer=spy)
    expected = [
        case.case_id
        for candidate_id in manifest.candidate_ids
        for split in (Split.DEVELOPMENT, Split.HOLDOUT)
        for case in manifest.cases_for(candidate_id, split)
    ]
    assert seen == expected
    assert not any("future-reserve" in case_id for case_id in seen)
    assert set(results) == set(manifest.candidate_ids)
    assert all(len(results[cid][split.value]) == 120 for cid in results for split in (Split.DEVELOPMENT, Split.HOLDOUT))
    reserve = manifest.cases_for("ordinal-suffix", Split.FUTURE_RESERVE)[0]
    with pytest.raises(ValueError, match="not executable"):
        candidate_screening.score_behavior_case(CausalFakeModel(), reserve)
    with pytest.raises(ValueError, match="not executable"):
        candidate_screening.executable_cases(manifest, "ordinal-suffix", Split.FUTURE_RESERVE)


def _measurement_for(case: ScreeningCase, *, correct: bool, flip: bool = True, margin: float = 2.0) -> candidate_screening.CaseMeasurement:
    """Build a measurement whose primary condition is correct/incorrect with a fixed margin."""
    sign = 1.0 if correct else -1.0
    a_pref = {"A": -1.0, "B": -1.0 - margin}
    b_pref = {"A": -1.0 - margin, "B": -1.0}
    if case.primary_orientation == "A":
        x_a = a_pref if correct else b_pref
        x_b = b_pref if flip else a_pref
    else:
        x_b = b_pref if correct else a_pref
        x_a = a_pref if flip else b_pref
    del sign
    return candidate_screening.measure_case(
        {"x_a": x_a, "x_b": x_b, "s_a": x_b, "s_b": x_a},
        case.primary_orientation, case_id=case.case_id, template_id=case.template_id)


def test_baselines_use_development_only_lexical_priors_and_the_manifest_heuristic() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    lexical, backoff = candidate_screening.lexical_prior_predictions(manifest, "regular-plural")
    assert set(lexical) == {case.lexical_key for case in manifest.cases_for("regular-plural", Split.DEVELOPMENT)}
    assert lexical["cat"] == "A" and lexical["city"] == "B" and backoff == "A"
    holdout = manifest.cases_for("regular-plural", Split.HOLDOUT)
    rows = [_measurement_for(case, correct=True) for case in holdout]
    baselines = candidate_screening.baseline_correct_counts(manifest, "regular-plural", Split.HOLDOUT, rows)
    assert set(baselines) == set(candidate_screening.BASELINE_IDS)
    assert baselines == {"majority": 60, "lexical-prior": 60, "local-heuristic": 60, "cue-shuffle": 0}


def test_integrity_failures_detect_reserve_leakage_duplicates_and_missing_rows() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    holdout = manifest.cases_for("ordinal-suffix", Split.HOLDOUT)
    rows = [_measurement_for(case, correct=True) for case in holdout]
    assert candidate_screening.integrity_failures(manifest, "ordinal-suffix", Split.HOLDOUT, rows) == ()
    reserve_row = _measurement_for(manifest.cases_for("ordinal-suffix", Split.FUTURE_RESERVE)[0], correct=True)
    failures = candidate_screening.integrity_failures(manifest, "ordinal-suffix", Split.HOLDOUT, rows[:-1] + [reserve_row, rows[0]])
    assert any("future-reserve" in failure for failure in failures)
    assert any("duplicate" in failure for failure in failures)
    assert any("missing 1" in failure for failure in failures)


@pytest.mark.parametrize(
    "correct_count,bad_template,expected_pass,expected_failed",
    [
        (103, None, True, ()),
        (102, None, False, ("wilson_lower_bound",)),
        (109, "bare-numeral", False, ("per_template_accuracy", "template_range")),
    ],
)
def test_behavioral_result_reports_every_gate_and_failed_reason(correct_count, bad_template, expected_pass, expected_failed) -> None:
    """102/120 fails only through Wilson; a 29/40 template fails even at 109/120 overall."""
    manifest = load_manifest(MANIFEST_PATH)

    def round_robin_wrong(cases: tuple[ScreeningCase, ...], count: int) -> set[str]:
        # Spread the wrong cases over templates so only the overall gates move.
        by_template: dict[str, list[ScreeningCase]] = {}
        for case in cases:
            by_template.setdefault(case.template_id, []).append(case)
        wrong: set[str] = set()
        while len(wrong) < count:
            for rows in by_template.values():
                if len(wrong) < count:
                    wrong.add(rows.pop(0).case_id)
        return wrong

    development_cases = manifest.cases_for("ordinal-suffix", Split.DEVELOPMENT)
    development_wrong = round_robin_wrong(development_cases, 120 - correct_count)
    development = [_measurement_for(case, correct=case.case_id not in development_wrong) for case in development_cases]
    holdout_cases = manifest.cases_for("ordinal-suffix", Split.HOLDOUT)
    if bad_template is None:
        wrong = round_robin_wrong(holdout_cases, 120 - correct_count)
    else:
        wrong = set(sorted(case.case_id for case in holdout_cases if case.template_id == bad_template)[:11])
    holdout = [_measurement_for(case, correct=case.case_id not in wrong) for case in holdout_cases]
    result = candidate_screening.behavioral_result_for_candidate(manifest, "ordinal-suffix", development, holdout)
    assert tuple(result["gates"]["checks"]) == candidate_screening._GATE_NAMES
    assert result["gates"]["passed"] is expected_pass
    assert tuple(result["gates"]["failed_checks"]) == expected_failed
    assert result["holdout"]["primary_correct_count"] == correct_count
    assert set(result["holdout"]["baseline_accuracies"]) == set(candidate_screening.BASELINE_IDS)


# ---------------------------------------------------------------------------
# Exploratory compactness probe

from neural_decompiler.components import ComponentKind, ComponentRef  # noqa: E402
from instrumentation_fakes import TinyBridge  # noqa: E402


def _contrast_bridge() -> TinyBridge:
    """TinyBridge whose A/B contrast depends on the final token and on patched activations."""
    return TinyBridge(readout=lambda normalized: torch.cat((normalized, (normalized[..., :2] * 4.0) ** 2), dim=-1))


def _dev_case(number: int, *, last_a: int, last_b: int, template: str = "t1", partition: CompactnessPartition = CompactnessPartition.DISCOVERY, length: int = 3) -> ScreeningCase:
    """Single-token development case for the TinyBridge vocabulary of five tokens."""
    prompt_a = tuple([1] * (length - 1) + [last_a])
    prompt_b = tuple([1] * (length - 1) + [last_b])
    x_a = PromptCondition(f"a{number}", prompt_a, (3,), (0,), a_text=" a", b_text=" b")
    x_b = PromptCondition(f"b{number}", prompt_b, (3,), (0,), a_text=" a", b_text=" b")
    return ScreeningCase(
        case_id=f"fake-selection-development-{template}-{number:03d}", candidate_id="fake", template_id=template,
        split=Split.DEVELOPMENT, lexical_key=f"k{number}", rule_class="simple-suffix",
        primary_orientation="A" if number % 2 else "B", x_a=x_a, x_b=x_b, s_a=x_b, s_b=x_a,
        local_heuristic_choices={"A": "a", "B": "b"}, compactness_partition=partition)


def test_component_universe_is_all_heads_then_whole_mlps_in_canonical_order() -> None:
    universe = candidate_screening.component_universe(TinyBridge())
    assert universe == (
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=0), ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=1),
        ComponentRef(ComponentKind.MLP_OUT, layer=0),
        ComponentRef(ComponentKind.ATTN_HEAD, layer=1, head=0), ComponentRef(ComponentKind.ATTN_HEAD, layer=1, head=1),
        ComponentRef(ComponentKind.MLP_OUT, layer=1),
    )
    assert [candidate_screening.canonical_component_id(ref) for ref in universe] == ["L00.H00", "L00.H01", "L00.MLP", "L01.H00", "L01.H01", "L01.MLP"]


def test_compactness_rejects_multitoken_holdout_or_unpartitioned_cases() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    holdout = manifest.cases_for("regular-plural", Split.HOLDOUT)[0]
    with pytest.raises(ValueError, match="selection-development"):
        candidate_screening.validate_compactness_cases((holdout,))
    multi = PromptCondition("p", (1, 2), (3, 4), (0, 4))
    case = _dev_case(1, last_a=2, last_b=4)
    bad = ScreeningCase(case.case_id, case.candidate_id, case.template_id, case.split, case.lexical_key, case.rule_class,
                        case.primary_orientation, multi, multi, multi, multi, dict(case.local_heuristic_choices), case.compactness_partition)
    with pytest.raises(ValueError, match="single-token"):
        candidate_screening.validate_compactness_cases((bad,))
    with pytest.raises(ValueError, match="target_position"):
        PromptCondition("p", (1,), (2,), (3,), target_position=2)


def test_bidirectional_patch_shift_matches_single_case_runs_and_is_exact() -> None:
    model = _contrast_bridge()
    cases = [_dev_case(number, last_a=4, last_b=2, length=3 + number % 2) for number in range(1, 21)]
    universe = candidate_screening.component_universe(model)
    sources = candidate_screening.capture_pair_sources(model, cases, universe)
    assert model.cfg.use_attn_result is False
    assert all(sources[case.case_id].d_full != 0.0 for case in cases)
    batched = candidate_screening.patched_pair_shifts(model, cases, sources, universe[:3])
    single = tuple(candidate_screening.patched_pair_shifts(model, (case,), sources, universe[:3])[0] for case in cases)
    assert [row.to_dict() for row in batched] == [row.to_dict() for row in single]
    assert all(row.d_full > 0.0 and row.d_patch != 0.0 and row.d_patch != row.d_full for row in batched)
    everything = candidate_screening.patched_pair_shifts(model, cases, sources, universe)
    # Replacing the entire final-position universe from the counterpart still leaves the
    # token embedding, so the shift is not forced to equal d_full; it must be finite and stored raw.
    assert all(math.isfinite(row.d_patch) for row in everything)


def test_patch_integrity_failure_raises_instead_of_zero_effect() -> None:
    model = _contrast_bridge()
    cases = [_dev_case(1, last_a=4, last_b=2)]
    universe = candidate_screening.component_universe(model)
    sources = candidate_screening.capture_pair_sources(model, cases, universe)

    def leaky(model_, tokens, plan, *, prepend_bos):
        result = candidate_screening.run_interventions(model_, tokens, plan, prepend_bos=prepend_bos)
        execution = result.executions[0]
        tampered = type(execution)(**{**execution.__dict__, "outside_max_abs_change": 0.5})
        return type(result)(result.logits, (tampered,), result.hook_settings)

    with pytest.raises(candidate_screening.CompactnessIntegrityError, match="outside change"):
        candidate_screening.patched_pair_shifts(model, cases, sources, universe[:1], intervene=leaky)


def test_discovery_ranking_and_random_sets_are_deterministic() -> None:
    rows = {
        "L00.MLP": (PairPatchMeasurement(1.0, 0.5, "t1"), PairPatchMeasurement(1.0, 0.5, "t1")),
        "L00.H01": (PairPatchMeasurement(1.0, 0.5, "t1"), PairPatchMeasurement(1.0, 0.5, "t1")),
        "L01.H00": (PairPatchMeasurement(1.0, 0.9, "t1"), PairPatchMeasurement(1.0, 0.1, "t1")),
        "L00.H00": (PairPatchMeasurement(1.0, -0.2, "t1"), PairPatchMeasurement(1.0, 0.0, "t1")),
    }
    assert candidate_screening.rank_discovery_components(rows) == ("L00.H01", "L00.MLP", "L01.H00", "L00.H00")
    universe = tuple(f"c{i}" for i in range(10))
    first = candidate_screening.deterministic_random_sets(universe, 3, count=5)
    assert first == candidate_screening.deterministic_random_sets(universe, 3, count=5)
    assert all(len(set(members)) == 3 and list(members) == sorted(members) for members in first)
    with pytest.raises(ValueError):
        candidate_screening.deterministic_random_sets(universe, 11)


def test_compactness_probe_uses_discovery_for_ranking_and_validation_for_recovery(monkeypatch) -> None:
    monkeypatch.setattr(candidate_screening, "COMPACTNESS_MAX_K", 3)
    monkeypatch.setattr(candidate_screening, "COMPACTNESS_RANDOM_SETS", 4)
    model = _contrast_bridge()
    cases = []
    for number in range(1, 121):
        template = ("t1", "t2", "t3")[(number - 1) % 3]
        partition = CompactnessPartition.DISCOVERY if number <= 60 else CompactnessPartition.VALIDATION
        cases.append(_dev_case(number, last_a=4, last_b=2, template=template, partition=partition, length=3 + number % 2))
    manifest = candidate_screening.ScreeningManifest((candidate_screening.CandidateDefinition("fake", ("t1", "t2", "t3")),), tuple(cases))
    seen: list[tuple[str, tuple[str, ...]]] = []

    def spy(model_, tokens, plan, *, prepend_bos):
        seen.append(("patch", tuple(candidate_screening.canonical_component_id(item.component) for item in plan.interventions)))
        return candidate_screening.run_interventions(model_, tokens, plan, prepend_bos=prepend_bos)

    discovery_ids = {case.case_id for case in cases[:60]}
    validation_ids = {case.case_id for case in cases[60:]}
    calls: list[tuple[str, set[str]]] = []
    original = candidate_screening.patched_contrasts

    def tracking(model_, cases_, sources, components, **options):
        calls.append((",".join(candidate_screening.canonical_component_id(ref) for ref in components), {case.case_id for case in cases_}))
        return original(model_, cases_, sources, components, **options)

    monkeypatch.setattr(candidate_screening, "patched_contrasts", tracking)
    result = candidate_screening.run_compactness_probe(model, manifest, "fake", intervene=spy)
    singleton_calls = [ids for label, ids in calls if "," not in label]
    assert all(ids == discovery_ids for ids in singleton_calls[: 6 * 2])
    later_calls = [ids for label, ids in calls][6 * 2:]
    assert later_calls and all(ids == validation_ids for ids in later_calls)
    assert result["discovery_case_ids"] == sorted(discovery_ids, key=lambda c: int(c[-3:]))
    assert set(result["validation_case_ids"]) == validation_ids
    assert result["ranking"] and set(result["ranking"]) == set(result["component_universe"])
    assert set(result["top_k"]) == {"1", "2", "3"} and set(result["random"]) == {"1", "2", "3"}
    assert all(len(result["random"][k]["recoveries"]) == 4 for k in result["random"])
    assert set(result["gate"]["checks"]) == {"denominators", "overall_recovery", "beats_random", "positive_template_recovery"}
    assert json.dumps(result, allow_nan=False)

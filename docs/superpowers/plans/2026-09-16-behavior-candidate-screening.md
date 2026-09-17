# Behavior Candidate Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the frozen Pythia behavior screen (two candidates after the 2026-09-17 tokenizer-infeasibility amendment) that can return zero or one proposed reverse-engineering target without creating Experiment 005.

**Architecture:** A single focused `candidate_screening` module owns immutable manifest parsing, scoring semantics, frozen gates, compactness arithmetic, phase-state validation, and deterministic reporting. One runner exposes only `validate`, `behavioral`, `compactness`, and `report`; it reuses the existing pinned-model, capture, intervention, component, and provenance modules. One committed JSON manifest is the authority for every concrete case, tokenization decision, split, control, seed, and threshold; one generated JSON result and one generated Markdown report are the only scientific outputs.

**Tech Stack:** Python 3.12, standard-library dataclasses/enum/json/hashlib/random/statistics, PyTorch 2.x, TransformerLens 3.9.x `TransformerBridge`, pytest 8.x.

**Spec:** `docs/superpowers/specs/2026-09-16-behavior-candidate-screening-design.md`

## Global Constraints

- The candidate set is exactly `regular-plural` and `ordinal-suffix`; `degree-inflection` was eliminated by the 2026-09-17 tokenizer-infeasibility amendment recorded under Task 2, and eliminated candidates cannot re-enter this protocol version.
- Primary execution is pinned `EleutherAI/pythia-70m-deduped` revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32.
- Run pinned `EleutherAI/pythia-160m-deduped` revision `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e` only when zero candidates pass every 70M behavioral gate; use byte-identical cases, metrics, baselines, and thresholds.
- Use exactly three template families per candidate and 40 cases per template in each of `selection-development`, `selection-holdout`, and `future-reserve`.
- Seed manifest splitting with `20260916`; seed compactness random component sets with `20260917`.
- All 120 `selection-development` cases for every candidate must have exactly one target token for alternative A and exactly one target token for alternative B under the pinned Pythia tokenizer. Compactness therefore intervenes at the final prompt position and scores one next-token A/B contrast. There is no multi-token compactness path in protocol v1.
- `selection-holdout` and `future-reserve` may use equal-length multi-token alternatives, but their A and B token counts must match within every condition and they are never eligible for compactness.
- Preserve fixed-orientation contrast `c(x) = log P(A|x) - log P(B|x)` separately from correctness margin `m(x) = y(x)c(x)` in every API and artifact.
- Keep zero scientific retries, untouched future reserve, discovery-only component ranking, deterministic random controls, the denominator floors, and the finalist prior-art audit exactly as specified.
- Keep TransformerBridge compatibility mode disabled; use canonical `ComponentRef` sites, `CapturePlan`, and exact `InterventionPlan` replacements.
- Keep generated outputs under `outputs/behavior-candidate-screening/`; never create an Experiment 005 directory, claim, or preregistration.
- Follow red-green-refactor for every implementation task. The default test suite stays offline.

## File map

- `src/neural_decompiler/candidate_screening.py`: frozen screening types, manifest construction/validation, teacher-forced scoring, behavioral gates, compactness arithmetic, phase-state rules, finalist selection, and Markdown rendering.
- `src/neural_decompiler/models.py`: add the already established pinned Pythia-160M `ModelSpec` used only by the automatic fallback.
- `src/neural_decompiler/__init__.py`: export only the stable screen declaration needed by callers.
- `screening/behavior-candidates/manifest-v1.json`: the sole committed scientific manifest, including its self-verifying content digest.
- `screening/behavior-candidates/run.py`: the only screening CLI and model orchestration entry point.
- `tests/test_candidate_screening.py`: pure manifest, metric, gate, compactness, state, and report tests.
- `tests/test_behavior_candidate_screening_runner.py`: phase isolation, model-order, reserve non-execution, resume/refusal, and fake-runner tests.
- `.gitignore`: ignore the two generated outputs while retaining the directory policy.
- `README.md`: document the screening commands and scientific boundary.
- `outputs/behavior-candidate-screening/results.json`: the single generated machine-readable state/result artifact.
- `outputs/behavior-candidate-screening/report.md`: the single generated human-readable candidate matrix/report.

---

### Task 1: Frozen protocol types and metric semantics

**Files:**
- Create: `src/neural_decompiler/candidate_screening.py`
- Create: `tests/test_candidate_screening.py`

**Interfaces:**
- Produces: `Split`, `CompactnessPartition`, `PromptCondition`, `ScreeningCase`, `CandidateDefinition`, `ScreeningManifest`, `ConditionMeasurement`, `CaseMeasurement`, `CandidateBehaviorSummary`, `BehavioralGateResult`, `PairPatchMeasurement`, `DenominatorValidation`, `CompactnessGateResult`, and `FinalistSelection`.
- Produces: `fixed_contrast(logp_a, logp_b)`, `correctness_margin(contrast, intended)`, `measure_case(...)`, `wilson_lower_bound(successes, total)`, `passes_overall_accuracy_gate(...)`, `summarize_behavior(...)`, `evaluate_behavioral_gates(...)`, `aligned_patch_shift(...)`, `aggregate_recovery(...)`, `evaluate_compactness_denominators(...)`, `random_reference(...)`, `required_random_beating_recovery(...)`, and `evaluate_compactness_gate(...)`.
- All public result objects implement `to_dict()` and reject non-finite values through `behavior.validate_json_safe`.

- [ ] **Step 1: Write failing tests for the two distinct margins and the effective count threshold**

```python
def test_fixed_contrast_and_correctness_margin_do_not_conflate_orientation():
    assert fixed_contrast(-1.0, -3.0) == pytest.approx(2.0)
    assert correctness_margin(2.0, intended="A") == pytest.approx(2.0)
    assert correctness_margin(-2.0, intended="B") == pytest.approx(2.0)


def test_case_measurement_keeps_correctness_positive_while_contrast_flips():
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
    assert wilson_lower_bound(102, 120) == pytest.approx(0.775325, abs=1e-6)
    assert wilson_lower_bound(103, 120) == pytest.approx(0.784805, abs=1e-6)
    assert not passes_overall_accuracy_gate(102, 120)
    assert passes_overall_accuracy_gate(103, 120)
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `uv run pytest tests/test_candidate_screening.py -q`

Expected: collection fails with `ModuleNotFoundError: neural_decompiler.candidate_screening`.

- [ ] **Step 3: Implement immutable declarations and the exact scalar formulas**

```python
class Split(str, Enum):
    DEVELOPMENT = "selection-development"
    HOLDOUT = "selection-holdout"
    FUTURE_RESERVE = "future-reserve"


class CompactnessPartition(str, Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"


def fixed_contrast(logp_a: float, logp_b: float) -> float:
    value = float(logp_a) - float(logp_b)
    if not math.isfinite(value):
        raise ValueError("contrast must be finite")
    return value


def correctness_margin(contrast: float, intended: str) -> float:
    if intended not in {"A", "B"}:
        raise ValueError("intended must be A or B")
    return contrast if intended == "A" else -contrast


def wilson_lower_bound(successes: int, total: int) -> float:
    z = 1.959963984540054
    p = successes / total
    numerator = p + z * z / (2 * total) - z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )
    return numerator / (1 + z * z / total)
```

`ScreeningCase` must contain the four concrete conditions `x_a`, `x_b`, `s_a`, and `s_b`; canonical alternatives A/B; `primary_orientation`; candidate/template/split IDs; lexical key; rule class; local-heuristic choices; and optional compactness partition. `PromptCondition` stores exact prompt text, prompt token IDs, A token IDs, B token IDs, and target position `-1`.

- [ ] **Step 4: Add failing behavioral-gate tests covering every frozen rule**

Create synthetic summaries that independently fail overall count, Wilson bound, per-template 75%, template range 15 points, development-to-holdout drop 10 points, positive template margin, 80% contrast flip, cue effect, each baseline margin, and integrity validation. Assert a gate record exists for every numbered condition and that no later gate rescues an earlier failure.

- [ ] **Step 5: Implement summary and gate evaluation without model code**

```python
def passes_overall_accuracy_gate(successes: int, total: int) -> bool:
    return successes / total >= 0.85 and wilson_lower_bound(successes, total) >= 0.78


def evaluate_behavioral_gates(
    development: CandidateBehaviorSummary,
    holdout: CandidateBehaviorSummary,
) -> BehavioralGateResult:
    if holdout.case_count != 120:
        raise ValueError("behavioral gate requires exactly 120 holdout cases")
    checks = {
        "overall_accuracy": holdout.accuracy >= 0.85,
        "wilson_lower_bound": wilson_lower_bound(
            holdout.primary_correct_count, holdout.case_count
        ) >= 0.78,
        "per_template_accuracy": min(holdout.template_accuracies.values()) >= 0.75,
        "template_range": (
            max(holdout.template_accuracies.values())
            - min(holdout.template_accuracies.values()) <= 0.15
        ),
        "development_drop": development.accuracy - holdout.accuracy <= 0.10,
        "positive_template_margin": min(holdout.template_mean_margins.values()) > 0.0,
        "contrast_flip": holdout.contrast_flip_rate >= 0.80,
        "cue_dependence": (
            holdout.accuracy - holdout.cue_shuffle_accuracy >= 0.20
            or holdout.mean_d_cue >= 0.5 * holdout.mean_d_full
        ),
        "beats_baselines": all(
            holdout.accuracy - value >= 0.15
            for value in holdout.baseline_accuracies.values()
        ),
        "integrity": not holdout.integrity_failures,
    }
    return BehavioralGateResult(checks=checks, passed=all(checks.values()))
```

Primary accuracy uses exactly one predeclared primary condition per case, balanced 20 A-primary and 20 B-primary cases within each template. Counterfactual flip, `d_full`, and `d_cue` always use both conditions. This preserves `n = 120` while still measuring both orientations.

- [ ] **Step 6: Run focused and full offline tests**

Run: `uv run pytest tests/test_candidate_screening.py -q && uv run pytest -q`

Expected: all focused tests and the historical suite pass.

- [ ] **Step 7: Commit the metric foundation**

```bash
git add src/neural_decompiler/candidate_screening.py tests/test_candidate_screening.py
git commit -m "feat: define frozen behavior screening metrics"
```

### Task 2: One frozen, tokenizer-validated manifest

#### Amendment — 2026-09-17: tokenizer-eligibility pool extension

Before any manifest case was written or inspected and before any model weight,
logit, probability, or scientific output was inspected, tokenizer-only
validation of the pinned 70M and 160M revisions established that only nine
forms in the original `regular-plural` / `future-reserve` / `simple` pool met
the equal-length target constraint. Append these fixed literals in order after
the existing pool: `table`, `stone`, `field`, `road`, `door`. No token-length
rule or threshold changes. The builder still selects the first ten
tokenizer-eligible simple forms in the exact ordered pool.

#### Amendment — 2026-09-17: complete-census degree-pool extension

The complete tokenizer-only census found seven eligible forms in
`degree-inflection` / `selection-development` / `change` and nine in
`degree-inflection` / `future-reserve` / `change`, before any manifest output
or model weight, logit, or probability inspection. Before encoding or testing
new forms, append these fixed ordered literals: development — `gentle`,
`noble`, `humble`, `simple`, `dense`, `loose`, `sane`, `stale`, `rude`,
`crude`, `angry`, `dirty`; future reserve — `polite`, `remote`, `mature`,
`severe`, `vague`, `acute`, `fierce`, `huge`, `lame`, `empty`, `ugly`,
`sleepy`. All template, token-length, seed, threshold, and first-ten-eligible
rules remain unchanged.

#### Amendment — 2026-09-17: corrected development eligibility census

The previous census omitted development's stricter one-token predicate from
word eligibility. Before encoding or testing the following fixed forms, append
development/simple in this order: `black`, `brown`, `green`, `round`,
`straight`, `narrow`, `quiet`, `brisk`, `faint`, `harsh`, `cheap`, `dear`,
`full`, `hollow`, `smart`, `solid`, `strict`, `swift`, `tight`, `tough`; then
append development/change: `rainy`, `windy`, `tidy`, `crazy`, `lively`,
`merry`, `cheery`, `cloudy`, `dusty`, `frosty`, `silky`, `spicy`, `itchy`,
`witty`, `silly`, `jolly`, `nasty`, `thirsty`, `filthy`, `stormy`. These are
fixed before tokenizer testing. The first-ten-eligible, templates, token
lengths, seed, and thresholds are otherwise unchanged.

#### Amendment — 2026-09-17: `degree-inflection` eliminated for tokenizer infeasibility

With the corrected one-token predicate, and still before any manifest,
weight, logit, or probability was inspected, the `degree-inflection` /
`selection-development` / `change` stratum has only four eligible forms
(`large`, `easy`, `big`, `simple`) and an exhaustive scan of the compatible
single-token vocabulary yields no further ordinary disjoint spelling-change
adjective. The design's candidate set is amended to exactly `regular-plural`
and `ordinal-suffix`. Remove the degree templates, pools, and spelling function
from the builder; the manifest validator requires exactly two candidates in
that order and exactly 720 cases. All other rules — three templates per
candidate, 40 cases per template stratum, single-token development
alternatives, the 60/60 compactness partition, seeds, and thresholds — are
unchanged. The `regular-plural` local heuristic follows the count cue: it
records the bare noun for the singular condition and `base + s` for the plural
condition, so its recorded choice is wrong on exactly 20 of 40 cases per
template, matching the ordinal final-digit rule.

#### Amendment — 2026-09-17: pre-output review corrections

An independent review of the implementation before any model output found
two protocol-level defects, both corrected before the first scientific run:

1. The compactness discovery/validation partition was labeled by manifest
   order, which is not the template-stratified 60/60 split the design
   requires. The builder now assigns even pool items to discovery and odd pool
   items to validation within every template stratum under both surface
   variants, giving 20/20 per template with 10 A-primary and 10 B-primary
   cases per partition; the validator enforces this. The manifest was
   re-frozen tokenizer-only.
2. Accuracy gates compared floating-point rates, so an outcome sitting exactly
   on a frozen threshold (for example a 15-point template range) could be
   decided against the design. Summaries now carry integer counts and the
   gates compare exact rationals; the integrity gate also covers development
   integrity failures and requires 120 development cases.

Finalist audits are write-once per candidate; changing a recorded audit
requires a tracked report-phase incident note that moves the prior audits and
decision into the invalidated-run history. Split membership is predeclared by
the literal per-split pools and number pairs; the recorded seed 20260916
selects nothing.

**Files:**
- Modify: `src/neural_decompiler/candidate_screening.py`
- Modify: `src/neural_decompiler/models.py`
- Modify: `src/neural_decompiler/__init__.py`
- Create: `screening/behavior-candidates/manifest-v1.json`
- Modify: `tests/test_candidate_screening.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `PYTHIA_160M`, `MANIFEST_SCHEMA_VERSION = 1`, `build_manifest_payload(tokenizer)`, `manifest_content_digest(payload)`, `load_manifest(path)`, and `validate_manifest(manifest)`.
- `build_manifest_payload` may be used to create and verify the committed manifest, but scientific execution reads only the committed JSON.
- `load_manifest` verifies the embedded digest after removing only the `content_sha256` field from the digest input.

- [ ] **Step 1: Add failing tests for the exact fallback model and manifest structure**

```python
def test_pythia_160m_fallback_is_exact_and_immutable():
    assert PYTHIA_160M.model_id == "EleutherAI/pythia-160m-deduped"
    assert PYTHIA_160M.revision == "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e"


def test_committed_manifest_has_exact_candidates_splits_and_counts():
    manifest = load_manifest(MANIFEST_PATH)
    assert tuple(manifest.candidate_ids) == (
        "regular-plural", "degree-inflection", "ordinal-suffix"
    )
    for candidate in manifest.candidates:
        assert len(candidate.template_ids) == 3
        for split in Split:
            assert len(manifest.cases_for(candidate.candidate_id, split)) == 120
            for template_id in candidate.template_ids:
                assert len(manifest.cases_for(candidate.candidate_id, split, template_id)) == 40


def test_every_development_case_is_single_token_and_partitioned_60_60():
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
```

- [ ] **Step 2: Implement the pinned 160M declaration and strict manifest schema**

```python
PYTHIA_160M = ModelSpec(
    model_id="EleutherAI/pythia-160m-deduped",
    revision="582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
)
```

Reject unknown fields, duplicate IDs, noncanonical candidate order, missing conditions, overlapping lexical keys/numbers between splits, unbalanced primary orientation, incorrect template/count strata, future-reserve compactness labels, and any mismatch between stored token IDs and the stored token strings.

- [ ] **Step 3: Encode the three exact template families and case-generation rules**

Use these template families verbatim. Each family has two frozen surface variants; applying both variants to exactly 20 lexical items or number pairs produces 40 cases per template:

```python
TEMPLATES = {
    "regular-plural": {
        "cardinal": (
            ("The display contains one", "The display contains two"),
            ("The tray holds one", "The tray holds two"),
        ),
        "quantifier": (
            ("The catalog lists each", "The catalog lists several"),
            ("The inventory records each", "The inventory records several"),
        ),
        "coordinated-adjective": (
            ("Mira and Noah packed one bright", "Mira and Noah packed two bright"),
            ("Lena and Omar displayed one small", "Lena and Omar displayed two small"),
        ),
    },
    "degree-inflection": {
        "two-item-comparative": (
            ("Of the two options, this one is", "Of all the options, this one is the"),
            ("Between these two, this one is", "Among all three, this one is the"),
        ),
        "three-item-superlative": (
            ("Compared with the other item, this one is", "Among the three items, this one is the"),
            ("Next to the second item, this one is", "Of the complete set, this one is the"),
        ),
        "explicit-analogy": (
            ("small, smaller; ADJECTIVE,", "small, smallest; ADJECTIVE,"),
            ("short becomes shorter; ADJECTIVE becomes", "short becomes shortest; ADJECTIVE becomes"),
        ),
    },
    "ordinal-suffix": {
        "bare-numeral": (("NUMBER", "NUMBER"), ("No. NUMBER", "No. NUMBER")),
        "dated-event": (
            ("The event happened on September NUMBER", "The event happened on September NUMBER"),
            ("The meeting occurred on March NUMBER", "The meeting occurred on March NUMBER"),
        ),
        "ranked-list": (
            ("Her final rank was NUMBER", "Her final rank was NUMBER"),
            ("The athlete placed NUMBER", "The athlete placed NUMBER"),
        ),
    },
}
```

Use these exact ordered lexical pools. The builder selects the first 10
tokenizer-eligible simple forms and first 10 eligible spelling-change forms for
each plural/degree split, preserving order. The pools are deliberately larger
than the selected set so tokenization filtering is deterministic rather than a
post-output judgment:

```python
PLURAL_POOLS = {
    "selection-development": {
        "simple": ("cat", "dog", "book", "lamp", "chair", "river", "cloud", "train", "spoon", "cup", "garden", "window", "planet", "robot", "candle"),
        "change": ("city", "baby", "story", "party", "berry", "box", "bus", "dish", "watch", "class", "puppy", "brush", "fox", "church", "bench"),
    },
    "selection-holdout": {
        "simple": ("island", "pencil", "button", "ticket", "basket", "mirror", "camera", "tunnel", "village", "jacket", "carpet", "bottle", "blanket", "ladder", "rocket"),
        "change": ("family", "lady", "hobby", "cherry", "country", "glass", "kiss", "match", "peach", "wish", "library", "factory", "mystery", "gallery", "branch"),
    },
    "future-reserve": {
        "simple": ("anchor", "beacon", "castle", "desert", "engine", "forest", "harbor", "insect", "kernel", "market", "needle", "ocean", "pocket", "quilt", "ribbon"),
        "change": ("army", "diary", "enemy", "fairy", "glory", "sky", "fly", "ally", "penny", "reply", "beach", "bush", "cross", "dress", "inch"),
    },
}

DEGREE_POOLS = {
    "selection-development": {
        "simple": ("tall", "short", "long", "fast", "slow", "dark", "light", "cold", "warm", "young", "old", "rich", "poor", "high", "low"),
        "change": ("large", "nice", "safe", "brave", "wise", "happy", "easy", "busy", "heavy", "funny", "big", "hot", "thin", "fat", "sad"),
    },
    "selection-holdout": {
        "simple": ("near", "quick", "kind", "calm", "bold", "bright", "strong", "small", "hard", "soft", "clean", "deep", "sharp", "weak", "sweet"),
        "change": ("close", "strange", "pale", "rare", "fine", "pretty", "noisy", "tiny", "lucky", "early", "wet", "flat", "red", "fit", "dim"),
    },
    "future-reserve": {
        "simple": ("cool", "mild", "rough", "smooth", "thick", "fresh", "clear", "plain", "great", "grand", "crisp", "dull", "fair", "firm", "loud"),
        "change": ("wide", "late", "cute", "ripe", "tame", "dry", "shy", "lazy", "messy", "sunny", "mad", "slim", "grim", "glad", "sparse"),
    },
}

ORDINAL_NUMBER_PAIRS = {
    "selection-development": (
        (21, 11), (22, 12), (23, 13), (31, 111), (32, 112),
        (33, 113), (41, 211), (42, 212), (43, 213), (51, 311),
        (52, 312), (53, 313), (61, 411), (62, 412), (63, 413),
        (71, 511), (72, 512), (73, 513), (81, 611), (82, 612),
    ),
    "selection-holdout": (
        (91, 711), (92, 712), (93, 713), (101, 811), (102, 812),
        (103, 813), (121, 911), (122, 912), (123, 913), (131, 1011),
        (132, 1012), (133, 1013), (141, 1111), (142, 1112), (143, 1113),
        (151, 1211), (152, 1212), (153, 1213), (161, 1311), (162, 1312),
    ),
    "future-reserve": (
        (163, 1313), (171, 1411), (172, 1412), (173, 1413), (181, 1511),
        (182, 1512), (183, 1513), (191, 1611), (192, 1612), (193, 1613),
        (201, 1711), (202, 1712), (203, 1713), (221, 1811), (222, 1812),
        (223, 1813), (231, 1911), (232, 1912), (233, 1913), (241, 2011),
    ),
}
```

The manifest builder applies only these closed build-time spelling functions and
writes the resulting surface forms explicitly into the JSON; scientific
execution never derives a target form:

```python
def regular_plural(base: str) -> str:
    if base.endswith("y") and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return base + "es"
    return base + "s"


DOUBLE_FINAL = frozenset({"big", "hot", "thin", "fat", "sad", "wet", "flat", "red", "fit", "dim", "mad", "slim", "grim", "glad"})


def regular_degree(base: str) -> tuple[str, str]:
    if base.endswith("e"):
        return base + "r", base + "st"
    if base.endswith("y") and base[-2] not in "aeiou":
        return base[:-1] + "ier", base[:-1] + "iest"
    if base in DOUBLE_FINAL:
        return base + base[-1] + "er", base + base[-1] + "est"
    return base + "er", base + "est"
```

The local heuristic is computed separately (`base + s`, `base + er`, or
`base + est`) so its deliberately wrong predictions remain auditable. Unit
tests assert every selected `change` item differs from that shallow heuristic.

For plural and degree candidates, each split uses exactly 20 disjoint lexical keys: 10 simple-suffix forms and 10 predeclared spelling-change forms. Each key appears in both frozen surface variants and across all three template families, but never in another split. For ordinals, each split uses exactly 20 disjoint matched number pairs: the A-number takes `st`, `nd`, or `rd`; the B-number has the corresponding `11`, `12`, or `13` ending and takes `th`. Ten of each template's 40 primary cases must be A-primary for each non-`th` class collectively, ten must be B-primary exception cases, and the remaining twenty repeat the balanced orientations under the second surface variant. The final-digit heuristic must be wrong on exactly 20/40 cases per template.

The manifest builder must use only curated literal pools stored in this module, deterministic sorting, and tokenizer inspection. It must not query model logits. If the curated pools cannot produce the exact counts while satisfying token constraints, stop before writing the manifest and amend the design; never substitute cases after model output.

- [ ] **Step 4: Implement tokenizer-only freezing and self-digesting output**

```python
def manifest_content_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def freeze_manifest(path: Path, tokenizer: Any) -> None:
    payload = build_manifest_payload(tokenizer)
    payload["content_sha256"] = manifest_content_digest(payload)
    write_json(path, payload)
```

Load tokenizers for both pinned model revisions and first require identical vocabulary, special-token configuration, and encoding for every manifest string. For every condition, encode the prompt without special tokens, then encode `prompt + alternative`; require the latter to begin with the exact prompt IDs and store only the suffix as the alternative token IDs. Reject contextual retokenization, empty suffixes, unequal A/B suffix lengths, any 70M/160M tokenization difference, duplicated case IDs, or token strings that fail round-trip decoding. Store both compatible model revisions, tokenizer vocabulary digest/settings, exact token IDs/strings, target position `-1`, rule class, primary orientation, local-heuristic choice, split, compactness partition, and all four concrete prompt strings.

- [ ] **Step 5: Generate the candidate manifest without loading model weights**

Run:

```bash
uv run python -c 'from pathlib import Path; from transformers import AutoTokenizer; from neural_decompiler.candidate_screening import freeze_manifest; from neural_decompiler.models import PYTHIA_70M, PYTHIA_160M; t70=AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision); t160=AutoTokenizer.from_pretrained(PYTHIA_160M.model_id, revision=PYTHIA_160M.revision); freeze_manifest(Path("screening/behavior-candidates/manifest-v1.json"), {PYTHIA_70M.model_id: t70, PYTHIA_160M.model_id: t160})'
```

Expected: one canonical JSON file containing 720 matched cases (two candidates after the 2026-09-17 amendment), the exact tokenizer provenance, and a valid embedded SHA-256 digest; no model is loaded and no probability is computed.

- [ ] **Step 6: Add tamper, reserve, lexical-disjointness, and tokenizer-policy tests**

Copy the decoded payload in memory, change one prompt, split, token ID, threshold, seed, or digest, and assert `validate_manifest` fails. Assert development/holdout/reserve lexical-key sets are pairwise disjoint for each candidate. Assert future-reserve case IDs can be enumerated but no execution API accepts `Split.FUTURE_RESERVE`.

- [ ] **Step 7: Run manifest, model, and full offline tests**

Run: `uv run pytest tests/test_candidate_screening.py tests/test_models.py -q && uv run pytest -q`

Expected: all tests pass, with no network or model-weight requirement after the manifest has been committed.

- [ ] **Step 8: Review and commit the frozen manifest**

```bash
git add src/neural_decompiler/candidate_screening.py src/neural_decompiler/models.py src/neural_decompiler/__init__.py screening/behavior-candidates/manifest-v1.json tests/test_candidate_screening.py tests/test_models.py
git commit -m "feat: freeze behavior candidate manifest"
```

Record the case counts, candidate IDs, split lexical-set digests, tokenizer revision, and manifest content digest in the commit review. Do not proceed if any item is implicit or generated from model behavior.

### Task 3: Behavioral scoring, baselines, and gates

**Files:**
- Modify: `src/neural_decompiler/candidate_screening.py`
- Modify: `tests/test_candidate_screening.py`

**Interfaces:**
- Produces: `teacher_forced_log_probability(model, prompt_ids, target_ids)`, `score_condition(model, condition)`, `score_behavior_case(model, case)`, `run_behavioral_screen(model, manifest, candidate_ids, on_case=None)`, and `behavioral_result_for_candidate(...)`.
- Consumes: immutable `ScreeningCase` declarations and the model returned by `models.load_model`.
- The optional `on_case` callback receives a JSON-safe completed measurement for incremental persistence; it cannot alter scoring.

- [ ] **Step 1: Write failing teacher-forcing tests using a deterministic fake model**

```python
def test_teacher_forced_sequence_score_uses_each_alternatives_own_prefix():
    model = PrefixSensitiveFakeModel()
    score = teacher_forced_log_probability(
        model,
        prompt_ids=(4, 5),
        target_ids=(6, 7),
    )
    expected = (
        torch.log_softmax(model(torch.tensor([[4, 5, 6]])), dim=-1)[0, 1, 6]
        + torch.log_softmax(model(torch.tensor([[4, 5, 6]])), dim=-1)[0, 2, 7]
    )
    assert score == pytest.approx(float(expected))


def test_single_token_condition_records_top1_but_multitoken_does_not():
    single = score_condition(FakeModel(), single_token_condition())
    multi = score_condition(FakeModel(), two_token_condition())
    assert single.top1_token_id is not None
    assert multi.top1_token_id is None
```

- [ ] **Step 2: Run focused tests and verify the scoring functions are absent**

Run: `uv run pytest tests/test_candidate_screening.py -q`

Expected: the new tests fail on missing imports or attributes.

- [ ] **Step 3: Implement exact teacher-forced scoring**

```python
def teacher_forced_log_probability(model: Any, prompt_ids: tuple[int, ...], target_ids: tuple[int, ...]) -> float:
    full = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=model.cfg.device)
    with torch.inference_mode():
        logits = model(full, prepend_bos=False)
        log_probs = logits.log_softmax(dim=-1)
    start = len(prompt_ids) - 1
    terms = [log_probs[0, start + offset, token_id] for offset, token_id in enumerate(target_ids)]
    return float(torch.stack(terms).sum().item())
```

Score A and B in separate forwards so later target tokens are teacher-forced on their own alternative prefix. Use the manifest's exact token IDs, never retokenize during scientific execution, and assert final-logit vocabulary size covers every target ID.

- [ ] **Step 4: Add failing baseline and reserve-isolation tests**

Test majority, development lexical prior with global-majority backoff for unseen holdout keys, manifest-declared local heuristic, cue-shuffle accuracy, and counterfactual flip. Inject a scorer that records case IDs and assert the call set contains all and only development plus holdout IDs; no future-reserve ID may reach it.

- [ ] **Step 5: Implement deterministic behavioral orchestration**

Evaluate candidates, templates, cases, and conditions in manifest order. For each case, record four condition measurements, primary correctness, counterfactual flip, `d_full`, `d_cue`, tokenizer metadata, and provenance. Compute lexical-prior mappings only from development measurements and lexical keys. Apply all frozen gates to holdout without changing cases, templates, or thresholds.

- [ ] **Step 6: Add exact pass/fail matrix tests**

Create three synthetic 120-case candidate result sets: one at 103/120 that passes every other gate, one at 102/120 that fails both effective overall gates, and one with 110/120 overall but a 29/40 template that fails the per-template gate. Assert result JSON lists every gate and failed reason rather than returning only a boolean.

- [ ] **Step 7: Run focused and full tests**

Run: `uv run pytest tests/test_candidate_screening.py -q && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit behavioral execution**

```bash
git add src/neural_decompiler/candidate_screening.py tests/test_candidate_screening.py
git commit -m "feat: add frozen behavioral candidate screen"
```

### Task 4: Exploratory compactness execution

**Files:**
- Modify: `src/neural_decompiler/candidate_screening.py`
- Modify: `tests/test_candidate_screening.py`
- Modify: `tests/instrumentation_fakes.py`

**Interfaces:**
- Produces: `component_universe(model)`, `capture_pair_sources(model, cases)`, `patched_pair_shift(model, case, components, sources)`, `rank_discovery_components(...)`, `deterministic_random_sets(...)`, and `run_compactness_probe(...)`.
- Consumes: `CaptureRequest`, `CapturePlan`, `run_capture`, `Intervention`, `InterventionPlan`, `run_interventions`, `ComponentKind.ATTN_HEAD`, and `ComponentKind.MLP_OUT`.
- Returns raw singleton discovery effects, fixed cumulative top-`k` validation effects for `k=1..12`, all 100 random-set recoveries per `k`, denominator diagnostics, and one `CompactnessGateResult` per candidate.

- [ ] **Step 1: Write failing tests for the closed component universe and single-token boundary**

```python
def test_component_universe_is_all_heads_then_whole_mlps_in_canonical_order():
    model = TinyBridge(n_layers=2, n_heads=2)
    assert component_universe(model) == (
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=0),
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=1),
        ComponentRef(ComponentKind.MLP_OUT, layer=0),
        ComponentRef(ComponentKind.ATTN_HEAD, layer=1, head=0),
        ComponentRef(ComponentKind.ATTN_HEAD, layer=1, head=1),
        ComponentRef(ComponentKind.MLP_OUT, layer=1),
    )


def test_compactness_rejects_any_multitoken_or_nonfinal_target_case():
    with pytest.raises(ValueError, match="single-token"):
        validate_compactness_cases((development_case(a_ids=(1, 2)),))
    with pytest.raises(ValueError, match="target position -1"):
        validate_compactness_cases((development_case(target_position=2),))
```

- [ ] **Step 2: Add failing arithmetic tests for bidirectional patching and denominator floors**

```python
def test_bidirectional_patch_shift_and_ratio_of_aggregate_means():
    rows = [
        PairPatchMeasurement(d_full=4.0, d_patch=3.0, template_id="t1"),
        PairPatchMeasurement(d_full=2.0, d_patch=1.0, template_id="t2"),
    ]
    assert aggregate_recovery(rows) == pytest.approx(4.0 / 6.0)


@pytest.mark.parametrize("overall,template", [(0.249, 0.2), (0.5, 0.099), (-1.0, 1.0)])
def test_compactness_denominator_floor_fails_without_division(overall, template):
    result = evaluate_compactness_denominators(overall, {"t1": template, "t2": 0.2, "t3": 0.2})
    assert not result.valid
    assert result.recovery is None


def test_nonpositive_random_median_uses_point_zero_five_floor():
    assert random_reference([-0.4, -0.1, 0.0]) == pytest.approx(0.05)
    assert required_random_beating_recovery([-0.4, -0.1, 0.0]) == pytest.approx(0.10)
```

- [ ] **Step 3: Implement bidirectional exact replacement using existing instrumentation**

For each pair, capture every component at position `-1` from `x_A` and `x_B` with `use_attn_result=True`. To compute `c_A<-B(S)`, run `x_A` with one exact `REFERENCE` replacement per component in `S` using the cached `x_B` slices; reverse the sources for `c_B<-A(S)`. Score A/B from the final prompt logits because compactness alternatives are single tokens.

```python
def aligned_patch_shift(c_a: float, c_b: float, c_a_from_b: float, c_b_from_a: float) -> float:
    return 0.5 * ((c_a - c_a_from_b) + (c_b_from_a - c_b))
```

Assert every execution reports `outside_max_abs_change == 0.0`, exact expected shape/dtype, `ReplacementSource.REFERENCE`, and the declared component/hook. A failed intervention integrity check fails the candidate; it is never converted to a zero effect.

- [ ] **Step 4: Implement discovery ranking and validation-only recovery**

Rank components by descending discovery `mean(d_patch({component}))`, breaking exact ties by canonical ID `L{layer:02d}.H{head:02d}` before `L{layer:02d}.MLP`. Freeze cumulative sets `top[:k]` for `k=1..12`; never rerank on validation. Compute overall and per-template recovery as ratio of aggregate means, reject overall denominator below `0.25` nats or any template denominator below `0.10` nats, and report raw recovery without clipping.

- [ ] **Step 5: Implement deterministic random component controls**

```python
def deterministic_random_sets(universe: tuple[ComponentRef, ...], size: int) -> tuple[tuple[ComponentRef, ...], ...]:
    rng = random.Random(20260917)
    ordered = list(universe)
    return tuple(
        tuple(sorted(rng.sample(ordered, size), key=canonical_component_id))
        for _ in range(100)
    )
```

Persist all sampled IDs and all raw recoveries. For each `k`, set `B_k = max(median(random_recoveries), 0.05)` and require top-`k` recovery to be at least `2 * B_k`. The passing set is the smallest `k <= 12` that also reaches `0.70` overall and has positive aggregate shift/recovery in every template.

- [ ] **Step 6: Batch only by exact paired prompt lengths**

Group development pairs by `(len(x_A.prompt_token_ids), len(x_B.prompt_token_ids))` and process a fixed maximum of 16 pairs per batch. Do not add padding, attention-mask abstractions, or generalized batch scheduling. Assert batched measurements equal one-case measurements in the fake model. This keeps the implementation small while avoiding one forward per individual case.

- [ ] **Step 7: Add discovery/holdout/reserve leakage tests**

Inject spies into capture and intervention calls. Assert component singleton ranking receives only the 60 discovery IDs, every cumulative/random evaluation receives only the 60 compactness-validation IDs, and neither selection-holdout nor future-reserve IDs appear in any compactness call.

- [ ] **Step 8: Run compactness, instrumentation, and full tests**

Run: `uv run pytest tests/test_candidate_screening.py tests/test_capture.py tests/test_interventions.py -q && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 9: Commit compactness execution**

```bash
git add src/neural_decompiler/candidate_screening.py tests/test_candidate_screening.py tests/instrumentation_fakes.py
git commit -m "feat: add exploratory compactness probe"
```

### Task 5: One phased runner, fallback enforcement, and run-state integrity

**Files:**
- Create: `screening/behavior-candidates/run.py`
- Create: `tests/test_behavior_candidate_screening_runner.py`
- Modify: `src/neural_decompiler/candidate_screening.py`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `run.py validate`, `run.py behavioral`, `run.py compactness`, and `run.py report` only.
- Produces: `new_results_state(...)`, `load_results_state(...)`, `write_results_state(...)`, `assert_phase_allowed(...)`, `resume_pending_case_ids(...)`, and `invalidate_run_with_incident(...)`.
- The runner supports `--resume` only for a provenance-identical incomplete phase and `--incident-note TRACKED_PATH` only for invalidating an entire completed model screen after a committed software fix.

- [ ] **Step 1: Write failing parser and phase-order tests**

```python
def test_cli_exposes_only_four_phases_and_no_candidate_or_threshold_override():
    parser = runner.build_parser()
    for phase in ("validate", "behavioral", "compactness", "report"):
        assert parser.parse_args([phase]).phase == phase
    with pytest.raises(SystemExit):
        parser.parse_args(["behavioral", "--candidate", "ordinal-suffix"])
    with pytest.raises(SystemExit):
        parser.parse_args(["behavioral", "--threshold", "0.80"])


def test_compactness_requires_completed_behavioral_gate_results():
    state = valid_state_without_behavioral_results()
    with pytest.raises(PhaseError, match="behavioral"):
        assert_phase_allowed("compactness", state)
```

- [ ] **Step 2: Implement canonical state creation and atomic writes**

The JSON artifact records schema version, manifest path/digest, protocol/code commit, Git dirty state, model order, tokenizer provenance, dependency versions, per-case measurements, phase status (`not_started`, `running`, `complete`, `invalidated`), gate matrices, compactness artifacts, audit entries, and final selection. Write to a sibling temporary file and replace `results.json` atomically only after canonical JSON serialization succeeds.

- [ ] **Step 3: Add failing tests for automatic 70M-to-160M order**

```python
def test_behavioral_stops_after_70m_when_any_candidate_passes():
    loads = []
    runner.run_behavioral_phase(
        model_loader=lambda spec: loads.append(spec.model_id) or FakeModel(spec),
        screen=fake_screen_with_70m_pass,
    )
    assert loads == [PYTHIA_70M.model_id]


def test_behavioral_runs_160m_only_when_zero_70m_candidates_pass():
    loads = []
    runner.run_behavioral_phase(
        model_loader=lambda spec: loads.append(spec.model_id) or FakeModel(spec),
        screen=fake_screen_with_no_70m_pass,
    )
    assert loads == [PYTHIA_70M.model_id, PYTHIA_160M.model_id]
```

- [ ] **Step 4: Implement model order and candidate eligibility**

`behavioral` always runs every retained candidate on 70M. It unloads the model and runs every retained candidate on 160M only if the 70M pass count is zero. `compactness` uses the model scale whose behavioral phase produced at least one passing candidate and probes only those passers. No CLI flag can choose a scale or candidate.

- [ ] **Step 5: Implement zero-retry, resume, and incident behavior**

Before scientific execution, require a clean Git tree, a 40-character committed protocol/code SHA, and exact manifest digest. A completed phase refuses a second run. `--resume` accepts only a `running` phase with identical manifest digest, commit SHA, model revision, runtime, and completed-case digests; it evaluates only missing IDs in canonical order. `--incident-note` must name a tracked Markdown file that contains the old run ID, affected phase/model, defect, invalid artifact digest, fix commit, and full-rerun decision. It moves the old run under `invalidated_runs`, clears the entire affected model screen for all candidates, and starts a new run ID; candidate-only reruns are impossible.

- [ ] **Step 6: Add crash/resume, tamper, reserve, and retry-refusal tests**

Simulate a crash after case 17, resume to exactly the remaining cases, and assert the completed artifact equals an uninterrupted run except for run timestamps. Tamper with a completed case, manifest digest, code commit, or model revision and assert resume fails before model load. Assert future-reserve IDs never appear in `executed_case_ids`. Assert a second completed run fails without a valid tracked incident note.

- [ ] **Step 7: Implement the four phase handlers**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(MANIFEST_PATH)
    if args.phase == "validate":
        return run_validate_phase(manifest)
    state = load_results_state(RESULTS_PATH)
    assert_phase_allowed(args.phase, state)
    if args.phase == "behavioral":
        return run_behavioral_phase(manifest, state, resume=args.resume, incident_note=args.incident_note)
    if args.phase == "compactness":
        return run_compactness_phase(manifest, state, resume=args.resume, incident_note=args.incident_note)
    return run_report_phase(manifest, state, audit_json=args.audit_json)
```

`validate` never loads model weights or creates scientific measurements. `report` never loads a model. `behavioral` and `compactness` use `models.load_model`, `seed_runtime`, existing provenance collectors, sequential model scales, and incremental canonical state writes.

- [ ] **Step 8: Add output ignore rule and run focused/full tests**

Add `outputs/behavior-candidate-screening/*` to `.gitignore`.

Run: `uv run pytest tests/test_behavior_candidate_screening_runner.py tests/test_candidate_screening.py -q && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 9: Commit the phased runner**

```bash
git add screening/behavior-candidates/run.py src/neural_decompiler/candidate_screening.py tests/test_behavior_candidate_screening_runner.py .gitignore
git commit -m "feat: add phased candidate screening runner"
```

### Task 6: Candidate matrix, finalist audit, and deterministic selection

**Files:**
- Modify: `src/neural_decompiler/candidate_screening.py`
- Modify: `screening/behavior-candidates/run.py`
- Modify: `tests/test_candidate_screening.py`
- Modify: `tests/test_behavior_candidate_screening_runner.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `parse_audit_json(text)`, `validate_finalist_audits(state)`, `select_finalist(state)`, and `render_markdown_report(state)`.
- `report --audit-json JSON` is the only audit-record input. It updates the same `results.json`; it does not create a second artifact.
- Novelty statuses are exactly `NO_CLOSE_MECHANISM_LOCATED`, `PARTIAL_OVERLAP_WITH_EXPLICIT_GAP`, and `SUBSTANTIALLY_MAPPED`.

- [ ] **Step 1: Write failing audit and deterministic-selection tests**

```python
def test_report_remains_pending_until_every_compactness_passer_is_audited():
    state = state_with_compactness_passers("degree-inflection", "ordinal-suffix")
    result = finalize_selection(state)
    assert result.status == "PENDING_FINALIST_AUDIT"
    assert result.selected_candidate is None


def test_selection_uses_the_frozen_lexicographic_order():
    state = fully_audited_state(
        degree_status="PARTIAL_OVERLAP_WITH_EXPLICIT_GAP",
        ordinal_status="NO_CLOSE_MECHANISM_LOCATED",
    )
    assert select_finalist(state).candidate_id == "ordinal-suffix"


def test_substantially_mapped_finalist_is_excluded():
    state = fully_audited_state(ordinal_status="SUBSTANTIALLY_MAPPED")
    assert select_finalist(state).candidate_id != "ordinal-suffix"
```

- [ ] **Step 2: Implement strict audit parsing into the existing result artifact**

Audit JSON must map each compactness-passing candidate to: exact behavior/variants searched, model families, component/edge/feature/subspace/program coverage, causal/reconstruction/self-repair/held-out coverage, search-through date, novelty status, explicit contribution gap, and at least one primary-source URL or a statement that no close work was located after the recorded searches. Reject entries for non-finalists, unknown fields, missing gaps for partial overlap, or unrecognized statuses.

Example invocation shape:

```bash
uv run python screening/behavior-candidates/run.py report --audit-json '{"ordinal-suffix":{"novelty_status":"NO_CLOSE_MECHANISM_LOCATED","searched_through":"2026-09-16","behavior_variants":["ordinal suffix selection","numeric suffix generation"],"model_families":["Pythia","GPT-2","GPT-NeoX"],"explanation_levels":["component","edge","feature","subspace","program"],"validation_modes":["causal","reconstruction","self-repair","held-out intervention"],"primary_sources":[],"explicit_gap":"No naturally learned ordinal-suffix generation mechanism was located."}}'
```

- [ ] **Step 3: Implement the exact finalist ordering and zero-target outcome**

Filter out every behavioral failure, compactness failure, missing audit, and `SUBSTANTIALLY_MAPPED` candidate. Sort survivors by novelty status, worst-template holdout accuracy descending, overall holdout accuracy descending, validated component count ascending, counterfactual flip rate descending, and candidate ID ascending. Return zero targets normally when the survivor list is empty.

- [ ] **Step 4: Write failing report-content tests**

Assert the Markdown contains model identity, manifest/code digests, every candidate, every behavioral and compactness gate, baseline values, denominator diagnostics, random-set summaries, audit status, failed reasons, reserve non-execution proof, invalidated-run history, and exactly one final status: zero target, pending audit, or one proposed target. Assert it says compactness is exploratory and that no Experiment 005 exists.

- [ ] **Step 5: Implement deterministic Markdown rendering**

Render from validated `results.json` only; never recompute a metric during rendering. Use a compact candidate matrix followed by per-candidate gate detail, compactness/random diagnostics, audit comparison, provenance, limitations, and the stopping decision. Write only `outputs/behavior-candidate-screening/report.md`.

- [ ] **Step 6: Document commands and boundaries in the root README**

Add the four exact commands, the generated paths, the automatic fallback rule, the zero-retry/incident rule, and the statement that this is candidate selection rather than Experiment 005. Do not add a dashboard, notebook, plugin, database, or generalized benchmark API.

- [ ] **Step 7: Run focused and full offline tests**

Run: `uv run pytest tests/test_candidate_screening.py tests/test_behavior_candidate_screening_runner.py -q && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit reporting and documentation**

```bash
git add src/neural_decompiler/candidate_screening.py screening/behavior-candidates/run.py tests/test_candidate_screening.py tests/test_behavior_candidate_screening_runner.py README.md
git commit -m "feat: add candidate screening report and audit gate"
```

### Task 7: Pre-output freeze and scientific execution

**Files:**
- Generate: `outputs/behavior-candidate-screening/results.json`
- Generate: `outputs/behavior-candidate-screening/report.md`
- Create only after a verified software defect: `research/incidents/<date>-behavior-screening-<run-id>.md`
- Do not create or modify any Experiment 005 file.

**Interfaces:**
- `validate` proves manifest integrity without model output.
- `behavioral` performs the one allowed frozen behavioral screen and automatic fallback.
- `compactness` runs only for behavioral passers using development data.
- `report` first records the matrix; after finalist audits it records the zero/one-target decision.

- [ ] **Step 1: Run the complete offline verification suite**

Run: `uv run pytest -q`

Expected: all offline tests pass; the opt-in live test remains skipped.

- [ ] **Step 2: Run the pinned live instrumentation contract before candidate output**

Run:

```bash
NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 uv run pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q
```

Expected: the one live contract passes for the exact pinned 70M revision.

- [ ] **Step 3: Commit all protocol, manifest, code, tests, and docs and require a clean tree**

Run: `git status --short --branch && git rev-parse HEAD`

Expected: no short-status entries and a 40-character commit recorded as `protocol_code_commit`. No model probability from a manifest case may have been inspected before this point.

- [ ] **Step 4: Validate the frozen manifest**

Run: `uv run python screening/behavior-candidates/run.py validate`

Expected: exact candidate/split/template counts, embedded digest, tokenizer metadata, 120 single-token development cases per candidate, 60/60 compactness partitions, lexical disjointness, and future-reserve non-executability all pass.

- [ ] **Step 5: Run the behavioral phase exactly once**

Run: `uv run python screening/behavior-candidates/run.py behavioral`

Expected: both retained candidates are evaluated on development and holdout at 70M. The runner stops if at least one passes; otherwise it automatically repeats the byte-identical protocol for both at 160M. It records 240 development and 240 holdout case IDs per executed model scale and zero future-reserve IDs.

- [ ] **Step 6: Inspect integrity and gate outcomes without changing protocol code**

Run: `uv run python screening/behavior-candidates/run.py report`

Expected: `report.md` shows all behavioral measurements and either no behavioral passer or the exact passing set. A failure is retained; no template, case, threshold, baseline, or candidate is repaired.

- [ ] **Step 7: Run compactness only if at least one behavioral candidate passed**

Run: `uv run python screening/behavior-candidates/run.py compactness`

Expected: only behavioral passers are probed; discovery uses 60 development IDs, validation uses the other 60, every top/random intervention is bidirectional, and holdout/future-reserve execution counts remain zero.

- [ ] **Step 8: Perform and record finalist-level prior-art audits**

For every compactness passer, search primary sources through the audit date across the exact behavior and variants, Pythia/model-family work, component/edge/feature/subspace/program explanations, causal validation, reconstruction, self-repair, and held-out interventions. Invoke `report --audit-json` with the complete entries. If there are no compactness passers, invoke plain `report`; the final result is zero target and no audit is required.

- [ ] **Step 9: Verify the final artifact and stop**

Run:

```bash
uv run pytest -q
git status --short --branch
```

Verify `results.json` and `report.md` agree on every gate, audit, reserve count, model revision, manifest/code digest, and final zero/one-target decision. Generated outputs remain ignored. Do not design, preregister, create, or run Experiment 005 in this plan.

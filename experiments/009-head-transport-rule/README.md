# Experiment 009: The Transport Rule of `L03.H04`

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md).

## Inputs

- Exposed pool (fitting data): the forty cue tokens, eighteen frames, and eighty nouns of Experiment 008.
- `confirmation-v1.json` — frozen on 2026-09-18 by tokenizer rules under the design's grammatical-compatibility policy,
  before any Experiment 009 model output, content sha256 `6330b5a22f81455d90b42a412b8bccc325b5150fa6de58d90e341bfc8589095f`:
  23 fresh cue tokens — `either neither` (singular-selecting; `an` dropped because only 8 of the 79 nouns are
  vowel-initial), `eleven thirteen fourteen fifteen sixteen seventeen` (plural-numeral), `more most other enough
  certain` (plural-quantity), `my your his her our their` (number-neutral), `small blue new cold` (bare-adjective,
  plural-forcing) — and six fresh frames (`The shelf carries`, `The ledger names`, `The manual describes`, `The
  bulletin mentions`, `Ida and Tomas counted … shiny`, `Yusuf and Petra wrapped … thin`); every token licensed in
  every frame over the 79 single-token nouns; 138 token prompts. Executed only by `confirm`.

## Commands

```bash
uv run python experiments/009-head-transport-rule/run.py validate
uv run python experiments/009-head-transport-rule/run.py freeze-confirmation
uv run python experiments/009-head-transport-rule/run.py explore
uv run python experiments/009-head-transport-rule/run.py lock
uv run python experiments/009-head-transport-rule/run.py confirm
uv run python experiments/009-head-transport-rule/run.py report
```

- `explore` (Tier A, once): for every exposed (token, frame) the full, axis-only, and orthogonal-only E-patches with the
  head's internals captured; the exact OV decomposition (P1/P2/P3/remainder) and the locked mechanism level; stage
  fractions, additivity gaps, the non-additivity stage; the transport rule and the ridge-scalar baseline with
  leave-one-cue-out; the forty-token contrast rule (Experiment 007 estimator) with leave-one-cue-out; the quality gates;
  the exports under `outputs/experiment-009/parameters/`.
- `lock`: writes `outputs/experiment-009/candidate-lock.json` and `candidate-predictions.md` — every fresh (token,
  frame) prediction from the exported rules, without running any fresh prompt. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (Tier C, once): validates the committed artifacts (digests, state binding, unchanged scientific paths,
  clean tree, untouched ledger), recomputes every locked prediction from the on-disk rules **before any fresh prompt
  runs** (refuses on any difference above `1e-9`), then runs the fresh frames' reference and cue-pair prompts, the
  plural cue's E-patch, and every fresh token's three patches and behavioral prompt; applies the floors and the
  outcome rule.
- `report` renders `outputs/experiment-009/report.md`.

Boundaries: no fresh prompt before `confirm`; every phase refuses to run out of order or twice; `lock` refuses if a
scientific path changed since `explore`; numerical and identity failures and software defects are incidents recorded
with their commit; the head weights are read from the model, the OV levels from captured activations; nothing is
fitted on confirmation data. Stated limit: the fresh bare adjective `small` is also the trailing adjective of the
exposed frame `Lena and Omar displayed {cue} small`; the frozen disjointness rule covers cue tokens and noun forms, so
this is compliant, and the token enters no fit.

## Status — 2026-09-18: complete; outcome `HEAD_MECHANISM_CONFIRMED_P1 | TRANSPORT_RULE_FAILED | NOT_LOCKED`

The lock (`877221f`) and the integrity check (`78d71c2`) preceded the single `confirm`, which ran at `78d71c2` on a
clean tree (results state sha256 `d5e7954d31088355003f7ee4afd4e9baf512d8725bf9fcd3b43095bccfcf309a`; every locked
prediction reproduced from the on-disk rules before any fresh prompt, max difference 0.0). The final report is copied
verbatim to [`evidence/final-report-2026-09-18.md`](evidence/final-report-2026-09-18.md). Ledger after confirm: 930
prompt keys, 80 noun keys. Experiment 009 is closed; nothing is amended or rerun.

- **Precondition:** fresh-frame cue effect 473/474 (floor 427).
- **Y1 — mechanism, locked level P1: PASS.** On the 138 fresh (token, frame) pairs the fixed-normalization linear OV
  read-out of the residual arriving at `L03.H04` predicts the head's measured number-axis output with Spearman
  **0.957** and MAE **0.066** (τ_M 0.321; floors 0.90 / τ_M). P2 (exact cue-position LayerNorm): 0.963 / 0.055; P3
  (patched attention): 0.998 / 0.006. The claim, in the design's words, held prospectively: given the residual
  arriving at the head, a fixed-normalization linear `W_V W_O` read-out is sufficient to explain the selective
  transport of every new cue in every new frame; LayerNorm-scale and attention modulation are unnecessary; the
  opposing components the head reads were constructed upstream.
- **Y2 — weight-only transport rule: FAIL** (`spearman`, `singular_measured`, `plural_measured`, `plural_predicted`).
  Over the 23 fresh tokens the rank-1 rule reached Spearman 0.765 (floor 0.80) with MAE 0.081 (τ₂ 0.589). The
  frozen category checks failed on both sides: the grammatically singular-selecting `either` and `neither` were
  measured at 0.50 and 0.45 (both above 0.35 — the network does not treat them like `this`, whose transport is 0.03;
  their E-patch contrasts are −2.9 and −3.3 nats, plural-leaning), and 11 of the 15 plural-expectation tokens were
  measured ≥ 0.65 (`more` 0.61, `most` 0.60, `enough` 0.49, `blue` 0.58 fell short; 12 required), the same count the
  predictions already missed. The possessives came out at 0.36–0.57 (predicted 0.58–0.66). The numerals were predicted
  well (0.89–0.95 predicted, 0.90–1.00 measured).
- **Y3 — contrast rule: `NOT_LOCKED`** (its gate failed at Tier A). The Experiment 007 sixteen-token program's fresh
  contrast predictions are reported in the table (it under-predicts every plural-like token by 0.5–1.5 nats, as in
  Experiment 007).
- **Non-additivity stage on the fresh tokens (descriptive):** `R3` for the numerals and the possessives, `T` for the
  bare adjectives, `NONE` for `more`, `most`, `enough`, `other`, `our`, `their`, `either`.

What this settles and what it does not. The head-level mechanism is now a prospectively confirmed statement: the
selective transport of number by `L03.H04` is a fixed-normalization linear read-out of components already present in
the cue residual it receives, for 40 exposed and 23 never-seen cues in 24 frames. What remains open is *upstream*:
which components of the token-local encoding, transformed by layers 0–2, land on the head's read direction. The
one-direction rule from `E` alone predicts the numerals and the bulk ordering but not the determiners, and the
grammatical expectations were wrong for `either`/`neither` (the model places them mid-way, like `that` and `the`). The
`Ridge-scalar` baseline's much lower leave-one-cue-out error at Tier A (0.099 versus 0.162) says the information is
linearly present in `E` across several directions; a next experiment should fit the head's read direction back
through layers 0–2 to `E` rather than a single response-weighted direction. C002 is unchanged (out of scope here).

### Tier A (2026-09-18, for the record)

`explore` ran once on protocol/code commit `b4c281a` (run `89239d06811dee39`; A0 passed; results state sha256
`61d1dfa1c0ea009c0b15b4d65784212ac5c2a42401dcbe56409e404c7265745b`). The Tier A report is copied verbatim to
[`evidence/exploration-report-2026-09-18.md`](evidence/exploration-report-2026-09-18.md); the candidate prediction
table to [`evidence/candidate-predictions-2026-09-18.md`](evidence/candidate-predictions-2026-09-18.md). Ledger: 774
prompt keys, 80 noun keys — the exposed pool only; no fresh prompt ran. Replication of Experiments 006 (192) and 007
(144) exact; head reconstruction max relative error 4.9e-7; every OV identity held.

- **Mechanism (Q1).** Over the 720 exposed (token, frame) pairs the fixed-normalization linear OV read-out **P1**
  explains the head's number-axis output change with R² 0.909 and MAE 0.078 (RMSE 0.107) → **locked level P1**,
  τ_M = 0.321. P2 (exact LayerNorm) adds almost nothing (R² 0.919; the cue-position LayerNorm scale ratio is 0.92–1.09
  for every token); P3 (patched attention) is near-exact (R² 0.999, MAE 0.005); the remainder (other positions) is
  0.005. For the suppressed tokens P1 already gives the low transport measured (`this` 0.12 vs 0.03, `a` 0.14 vs 0.19,
  `another` 0.18 vs 0.17). Reading, as the design words it: given the residual arriving at the head, a
  fixed-normalization linear `W_V W_O` read-out is sufficient; LayerNorm-scale and attention modulation are
  unnecessary; the opposing components the head reads in opposite directions were constructed upstream.
- **Where the non-additivity enters (M2).** For `this`/`a`/`another`, at the head output the axis-only patch
  transports 0.38 / 0.48 / 0.41 of the plural cue's signal, the orthogonal-only patch −0.21 / +0.02 / −0.07, the whole
  0.03 / 0.19 / 0.17 — mostly *linear* cancellation inside the read-out (`this`: 0.38 − 0.21 ≈ 0.17 vs 0.03 measured;
  gap −0.14), then the gaps grow downstream (`g_R3` −0.36 / −0.43 / −0.28, `g_c` −0.41 / −0.60 / −0.44). Modal
  non-additivity stage: `a` → `T` (78% of frames), `this` and `another` → `R3`. Non-additivity of this size at `c` is
  also common among ordinary tokens (saturation), as in Experiment 008.
- **Transport rule (Q2, weight-only, fitted on the forty tokens).** Rank-1 cross-moment direction with
  template slopes 0.153 / 0.222 / 0.166; LOCO error 0.162, Spearman 0.776, normalized RMSE 0.256 → **gate passed**;
  τ₂ = 0.589. Its leave-one-out predictions reproduce the bulk ordering but not the determiner selectivity (`this`
  predicted 0.51 vs 0.03 measured, `a` 0.56 vs 0.19, `another` 0.47 vs 0.17, `those` 0.56 vs 0.92); the
  `Ridge-scalar` baseline's LOCO error is 0.099 — the selectivity is linearly recoverable from `E` with more than one
  direction. The frozen rule is the rank-1 one; its prospective predictions are locked as they are.
- **Contrast rule.** The Experiment 007 family refitted on forty tokens selects rank 1 with LOCO error 1.107 and fails
  its gate (Spearman 0.643 < 0.70; normalized RMSE 0.282) → **`NOT_LOCKED`**. The 007 sixteen-token program has an
  exposed MAE of 0.872 and is carried as the reported baseline.
- **Candidate lock**: `outputs/experiment-009/candidate-lock.json`, content sha256
  `2d637fa1c11181570f30edf812863103785b0ba665a7ae525ce16f50ff950198`; predictions artifact
  `candidate-predictions.md`, sha256 `a2214243e832e55ff05760bc57b4f5e1334d0fe2e6af8163049405a196b6a29d` (23 tokens × 6
  frames: predicted `q_T` and the 007 program's contrast prediction). Predicted transport: numerals 0.89–0.95;
  `either` 0.30, `neither` 0.43; possessives 0.58–0.66; `certain` 0.66, `more` 0.60, `other` 0.58, `most` 0.57,
  `enough` 0.56; bare adjectives 0.66–0.73. Under the frozen category check the rule's own predictions already put only
  11 of the 15 plural-expectation tokens at or above 0.65 (12 required), so the transport-rule axis can pass Y2 only
  if that count is met by the predictions — it is not; Y2 is expected to fail on `plural_predicted` whatever is
  measured. This is stated before `confirm` and is derivable from the committed predictions.

Installing the two artifacts as `preregistration-lock.json` and `predictions.md` and committing them is the
preregistration act; the reviewer's sign-off precedes `confirm`.

### Preregistration-integrity check before `confirm` (reporting incident, no scientific change)

The lock was installed at commit `877221f` (both files byte-identical to the candidates). The reviewer asked three
questions about discrepancies between the status prose and the committed artifacts; all three are prose errors in the
status message, not defects in the frozen set, the lock, or the frozen rule. Nothing scientific was changed.

1. **Exact confirmation tokens (23, from `confirmation-v1.json`, sha256 `6330b5a2…095f`, the same digest the lock
   records):** singular-selecting `either`, `neither` (2); plural-numeral `eleven`, `thirteen`, `fourteen`,
   `fifteen`, `sixteen`, `seventeen` (6 — `twelve` is an exposed token and is skipped by the disjointness rule);
   plural-quantity `more`, `most`, `other`, `enough`, `certain` (5); number-neutral `my`, `your`, `his`, `her`, `our`,
   `their` (6); bare-adjective `small`, `blue`, `new`, `cold` (4). 2 + 6 + 5 + 6 + 4 = 23; 23 × 6 frames = 138 token
   prompts. The status message's shorthand "eleven…seventeen" wrongly suggested seven numerals.
2. **The locked Y2 category rule** is the design's revision 2 text (commit `fa62ab9`, written before the freeze
   `197d8e6` and before Tier A): "all but at most one singular-selecting token measured ≤ 0.35, and at least 80% of
   the plural-numeral, plural-quantity, and bare-adjective tokens measured ≥ 0.65 (`exact_count_floor(0.8, n)`), with
   the rule's predictions meeting the same counts". The implementation applies exactly this (`max(n_singular − 1, 0)`
   and `exact_count_floor(0.8, n_plural)`), and the lock records `singular_max_q 0.35`, `plural_min_q 0.65`,
   `plural_rate 0.8`, `y2_spearman 0.80`. With the frozen set the denominators are `n_singular = 2` (at least 1 of 2)
   and `n_plural = 15` (at least 12 of 15). The "≥ 2 of 3 singular, ≥ 8 of 11 plural" wording was revision 1's, replaced
   in revision 2 before any implementation; it appears nowhere in the committed rule.
3. **The singular threshold after dropping `an`:** the "all but at most one" rule is defined for any category size,
   and the `an` rule (≥ 10 vowel-initial nouns; cue-final frames; vowel-initial nouns) was frozen in revision 2 and
   applied deterministically at the freeze (8 vowel-initial nouns among the 79). Nothing was revised after the
   predictions existed; the design's interpretation limits already state that the singular check is weak.

Consequence already visible in the committed predictions: `either` 0.296 ≤ 0.35 and `neither` 0.429 > 0.35 (1 of 2,
allowed); 11 of the 15 plural-expectation tokens are predicted ≥ 0.65 (`more` 0.60, `most` 0.57, `other` 0.58,
`enough` 0.56 fall short; 12 required), so the transport-rule axis fails the `plural_predicted` count whatever
`confirm` measures. The lock, predictions, thresholds, categories, and rules are untouched.

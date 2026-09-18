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

## Status — 2026-09-18: Tier A executed once; candidate lock and predictions written, awaiting installation and the reviewer's sign-off

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

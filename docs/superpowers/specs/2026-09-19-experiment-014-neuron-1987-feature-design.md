# Experiment 014: What Drives Block-2 Neuron 1987 — A Compact Input-Side Predictor, Locked Before Interpretation — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 2 — approved conceptually at revision 1 subject to the changes under "Revision history" (the
operational "below threshold" wording, the frozen LayerNorm equations, the predicted-never-measured invariant on the
arriving change, MAE descriptive with amplitude error among firing pairs, balanced accuracy in place of raw firing
agreement, and a rigidly specified axis-only alternative), which this revision makes. No Experiment 014 directory,
confirmation set, lock, or model run exists. Experiments 005–013 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter, feature-reconstructing. Experiments 012–013 found one block-2 MLP neuron, 1987,
carrying about three times the read-direction mass of any other in every template. This experiment reconstructs its
computation — residual input → LayerNorm → one input weight vector → preactivation → GELU → one output vector — and
asks which input feature controls the preactivation, by committing a **compact input-side predictor** of the neuron's
activation change for unseen cues and frames before any of them runs, and testing preregistered alternatives (a
number-axis scalar, a specific off-axis direction, an interaction of directions, context-dependent gating) against
each other. No semantic label is attached before the predictions are scored; the interpretation is offered afterwards
from the frozen firing rule.

## Purpose and question

The neuron's output direction reads positively onto the head's read direction (`r(W_out[1987]) / r(E(pl) − E(ref))`
= +0.204 in the cardinal template), so every unit of activation the E-patch adds contributes +0.20 of the plural cue's
signal. In the exposed ledgers it is the dominant single carrier of the block-2 MLP correction that Experiment 012
predicted and Experiment 013 explained at the level of arriving residual changes. What Experiment 013 did not ask is
what the neuron itself computes.

**Design check on exposed data (outside any results state; a 47-token × 12-frame subset of the exposed pool with the
patched runs re-executed; to be recomputed on the full exposed pool inside Tier A).**

- **Operating point.** In every frame the neuron is **below the preregistered firing threshold at the reference cue**
  (`Δa ≥ 0.5` is the frozen rule; the reference activation itself is −0.17 to +0.25, the reference preactivation −1.35
  to +0.39 — not literally zero under GELU). The E-patch changes its activation bimodally: for the numerals and the count
  quantifiers (`three`, `seven`, `ten`, `twelve`, `eighteen`, `forty`, `hundred`, `thousand`, `million`, `dozen`,
  `few`, `many`, `numerous`, `countless`, `myriad`) `Δa` is +1.2 to +2.1 in every frame; for determiners, possessives
  and adjectives — including the plural determiners `these`, `those`, `both`, `all`, `some` — `Δa` is −0.1 to +0.05.
  93% of the variance of `Δa` is between tokens. The neuron's term is +0.25 to +0.45 of the plural cue's signal for the
  firing cues and about −0.02 for the rest.
- **Alternatives, on the same pairs.** (i) *Number-axis sensitivity:* the arriving change's component along `d̂_E`,
  projected onto the neuron's input direction, predicts `Δa` with Spearman 0.21 and negative explained variance, and
  reaches balanced firing accuracy 0.59 (it fires for almost everything: sensitivity 0.92, specificity 0.26).
  (ii) *A specific direction:* the preactivation is exactly linear in the LayerNorm output, so given the arriving
  residual change the neuron's response is fixed by its own input weights and the frame's reference state. Evaluating
  the exact LayerNorm and `W_in[:, 1987]` at the predicted state reproduces the *measured* preactivation change with R²
  0.964, and `GELU(pre_ref + Δp̂re) − GELU(pre_ref)` predicts `Δa` with Spearman 0.946, R² 0.905 per pair (0.970 / 0.940
  per token), balanced firing accuracy 0.993 (sensitivity 1.00, specificity 0.99), MAE 0.10 over all pairs and 0.24
  among the firing pairs (mean firing `Δa` 1.74). The first-order LayerNorm Jacobian — one state-dependent direction
  `J_LN(x_ref)ᵀ W_in[:, 1987]` — reproduces the exact form with R² 0.955 and reaches Spearman 0.918, R² 0.868 per pair
  (0.946 / 0.926 per token) and balanced accuracy 0.986 itself. (iii) *Interaction of directions:* nothing beyond the
  LayerNorm is needed. (iv) *Gating:* the reference operating point varies by frame but stays below threshold; the firing
  cues fire in every frame (`ten`: +1.3 to +2.7).
- **What the direction reads** (through the Jacobian, descriptive). `cos(γ₂ ⊙ W_in[:, 1987], d̂_E) = 0.24`. Decomposing
  the arriving change: its number-axis component is a large positive push for every plural-leaning cue (+0.7 to +2.3),
  and its off-axis component is a large *negative* push for the non-firing cues (−1.2 to −2.9 for determiners,
  adjectives, possessives, `these`, `those`, `both`) but near zero to positive for the firing cues (−0.2 to +1.2). The
  neuron therefore fires when the axis excitation survives the off-axis veto — one direction read against the
  representation's two kinds of structure, not a scalar of the number axis. By source, the encoding difference
  dominates (mean |contribution| 1.15), block 1's heads push down for every cue (0.42), block 1's MLP contributes
  ±0.3–0.6 either way.

> Which input feature controls block-2 neuron 1987, and does a compact input-side predictor — one weight-defined
> direction, one reference operating point per frame, the frozen-pattern arriving change of Experiment 013 —
> prospectively predict the neuron's activation change and its firing for cue words and frames never measured on it?
> Is the simple number-axis account rejected on the same fresh data?

## The quantities under test (frozen definitions)

Notation as in Experiments 012–013; `x₂(k)` the frame's reference residual before block 2 at position `k`; `Δ̂x₂` the
Experiment 013 frozen-pattern arriving change at the cue position (`ΔE + Δ̂₁ + Σ_{layer-1 heads} Δ̂h`, from the
weights and the frame's reference state); `u = γ₂ ⊙ W_in[:, 1987]`, `b = b_in[1987]`, `o = W_out[1987]`.

**Measured (exact, from the captured residual before block 2 at `p_c` in the reference and patched runs):**

```text
pre_ref(f) = ⟨ln2₂(x₂(p_c)), W_in[:, 1987]⟩ + b        the operating point
Δpre(w, f) = ⟨ln2₂(x₂'(p_c)), W_in[:, 1987]⟩ + b − pre_ref
Δa(w, f)   = GELU(pre_ref + Δpre) − GELU(pre_ref)        the activation change (Y1/Y2 target)
fires(w,f) = [Δa ≥ 0.5]                                  the frozen firing rule
term(w, f) = Δa · r(o) / r(E(pl_T) − E(ref_T))            the neuron's contribution to the head-readable correction
```

**Predicted (weights, the locked axes, the frozen 012/013 models, and one reference state per frame — no patched run).**
The LayerNorm is `LN(x) = γ₂ ⊙ (x − μ(x)) / σ(x) + β₂` with `μ` the mean over the 512 coordinates, `σ(x) = √(mean((x − μ)²) + ε)`
(population variance, the model's `ε`), exactly as `plural_mechanism.exact_layer_norm`; `pre(x) = ⟨LN(x), W_in[:, 1987]⟩ + b`.

```text
Δp̂re(w, f) = pre(x₂(p_c) + Δ̂x₂) − pre(x₂(p_c))                                   the predictor: the exact LayerNorm and the
                                                                                 neuron's own input weights at the predicted state
Δâ(w, f)   = GELU(pre_ref + Δp̂re) − GELU(pre_ref)                                 exact GELU (the model's activation)
fîres(w,f) = [Δâ ≥ 0.5]
```

The compact single-direction form, frozen for the accounting and reported beside the predictor (first-order
LayerNorm Jacobian at the reference, with `x̂ = (x₂ − μ)/σ`, `v = Δ̂x₂`, `v_c = v − mean(v)·1`, `u = γ₂ ⊙ W_in[:, 1987]`):

```text
Δp̂re_J(w, f) = ⟨W_in[:, 1987], J_LN(x₂) v⟩ = [ ⟨u, v_c⟩ − ⟨u, x̂⟩ · mean(x̂ ⊙ v_c) ] / σ      centering, scale, and the variance derivative
```

The alternative, committed beside the predictor and defined as rigidly: **the same equations with `Δ̂x₂` replaced by
its component along the frozen number direction**, `Δ̂x₂,axis = ⟨Δ̂x₂, d̂_E⟩ d̂_E` — the same reference state, the same
LayerNorm, the same `W_in`, `b`, GELU and threshold, no fitted scale or intercept:

```text
Δp̂re_axis(w, f) = pre(x₂(p_c) + Δ̂x₂,axis) − pre(x₂(p_c));   Δâ_axis = GELU(pre_ref + Δp̂re_axis) − GELU(pre_ref)
```

**Invariant (formal):** `Δ̂x₂` is Experiment 013's frozen-pattern *predicted* arriving change — `ΔE + Δ̂₁ + Σ_{layer-1 heads} Δ̂h`
from the weights and the frame's *reference* state — and never a measured residual of a patched run. The prediction
function takes only the reference state and the token identity; the test suite asserts that no capture or
intervention entry point is reachable while the prediction table is computed, and `confirm` reproduces the locked table
from those inputs before any fresh prompt. If even one patched residual entered `Δ̂x₂`, the experiment would be a
read-out, not a reconstruction.

Every predicted quantity for the exposed frames is computable at the lock (their reference states are locked as in
Experiment 013, extended with `pre_ref`); for the fresh frames it is computed at `confirm` stage 1 from the frame's
reference run and digested before any fresh cue prompt, exactly as in Experiment 013.

**Accounting (descriptive, per pair, exact given the captures):** the measured `Δpre` against `Δp̂re` (the pattern-change
remainder, since the LayerNorm is exact) and against `Δp̂re_J` (the LayerNorm-linearization remainder); the split of
`Δp̂re_J` by source (`ΔE`, `Δ̂₁`, the layer-1 heads — the Jacobian is linear in `v`) and by axis / off-axis part; MAE of
`Δâ` over all pairs and the amplitude error `|Δâ − Δa|` among the pairs that fire, so that "did it know whether the neuron
fires" and "did it know how much" are reported apart.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked model, the Experiment 013 frozen-pattern model
and its reference-state captures, the results-state, ledger, lock, prediction-artifact, two-stage confirmation and
incident conventions — all exactly as in Experiments 012–013. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration record only; nothing is fitted)

The 135 exposed cue tokens (Experiment 013's 111 and its 24 confirmed) and the 42 exposed frames (its 36 and its 6),
with the 80 nouns. `explore`:

- captures the 42 reference states (as Experiment 013, plus `pre_ref`, `σ_ref`), locks them;
- re-measures the E-patch of every exposed token in every exposed frame that Experiment 013 recorded (its 2724 + 1008
  pairs) with the residual before block 2 captured; replicates Experiment 013's per-pair `c_L`, `c_M`, `c_H` within
  `1e-6` (a committed extract); computes `Δpre`, `Δa`, `fires`, `term` exactly, and `Δp̂re`, `Δâ`, `Δâ_axis` from the
  locked states;
- records, for the full pool, the exposed statistics of the predictor, its Jacobian form and the axis-only alternative
  (Spearman, R², balanced firing accuracy with sensitivity and specificity, the majority-class baseline, MAE, the
  amplitude error among firing pairs; per pair and per token), the source and axis/off-axis accounting, the two
  remainders, the operating points, the firing set by lexical class, and the neuron's share of the block-2 read change
  for firing and non-firing cues.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 014 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 135 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists, with quotas; classes for coverage and
  reporting only, **no class carries an expectation**.
  - `determiner-like` (quota 5): `second`, `third`, `former`, `latter`, `only`, `main`, `whole`, `entire`, `very`.
  - `numeral` (quota 5): `dozens`, `hundreds`, `thousands`, `millions`, `billions`, `twice`, `double`, `triple`,
    `couple`, `pair`.
  - `quantity` (quota 5): `minimal`, `infinite`, `excess`, `lesser`, `considerable`, `insufficient`, `extensive`.
  - `possessive-or-pronoun` (quota 4): `anybody`, `somebody`, `everybody`, `anyone`, `whoever`, `yourself`,
    `themselves`.
  - `adjective` (quota 5): `narrow`, `sharp`, `smooth`, `rough`, `golden`, `silver`, `purple`, `yellow`, `tiny`,
    `giant`.
- **Fresh frames (6, two per template), literal:** cardinal `The bakery bakes {cue}`, `The library lends {cue}`;
  quantifier `The journal publishes {cue}`, `The committee approves {cue}`; coordinated-adjective
  `Omar and Lucia piled {cue} neat`, `Farah and Tobias lifted {cue} wet`. Texts must differ from every exposed frame
  text; the original cue tokens are used for their own cue prompts.
- **Two fresh sets, both executed only by `confirm`:** the 24 fresh tokens in the 42 exposed frames (Y1, 1008 pairs;
  every prediction in the lock) and in the 6 fresh frames (Y2, 144 pairs; predictions at stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes and models, the 42 locked reference states — no forward pass on any fresh prompt):**
  the complete prediction table for every fresh token × exposed frame — token, frame, template, `pre_ref`, `Δp̂re`, `Δâ`,
  `fîres`, `Δp̂re_J`, `Δp̂re_axis`, `Δâ_axis`, `fîres_axis`, the source split of `Δp̂re_J` (`ΔE`, `Δ̂₁`, heads) and its axis /
  off-axis parts, the predicted `term` — and the token means; the predicted firing counts per set (so the class balance
  the predictor commits to is on record before execution); the frozen floors; the exposed statistics; the model
  revision and the protocol commit; `predictions.md` with the token means and the predicted firing set.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiment 013's procedure):**
  *stage 1* — the fresh frames' reference runs (exposed tokens only) with the residual before block 2 captured, their
  validity (plural-cue head change `≥ 0.25 σ_T`, cue-pair check), the frame-conditional prediction table with the same
  columns, serialized and digested; *stage 2*, only after the digest is re-read from disk — every fresh cue's E-patch
  in every frame of both sets with the residual before block 2 captured; `Δpre`, `Δa`, `fires`, `term`, the accounting;
  scoring against the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–013; a token is scored in a set iff it has at least three
  valid frames there; Y2 needs at least four valid fresh frames and sixteen scored tokens; Y1 needs sixteen scored
  tokens.
- **Aggregation, fixed:** Y1 and Y2 on **token means** of `Δâ` and `Δa` over identical frame sets (at most 24 points
  each) for the ordering and the explained variance, and on **pairs** of the respective set for the firing
  classification; Y3 on the same token means and pairs as Y1 ∪ Y2. No MAE ceiling: the target is bimodal (below
  threshold ≈ −0.1, firing ≈ +1.2 to +2.1), so the ordering, the explained variance and the classification are the
  tests, and a compressed predictor fails the explained-variance floor; MAE and the amplitude error among firing pairs
  are reported descriptively.
- **Classification metric, fixed:** **balanced accuracy** `½(sensitivity + specificity)` of `fîres` against `fires`
  over the pairs, with the frozen threshold `Δa ≥ 0.5` on both sides; the majority-class baseline is reported beside it.
  **Guard:** if the measured pairs of a set contain no firing pair or no non-firing pair, the classification criterion
  of that set is non-evaluable — the family is then judged on the ordering and the explained variance alone and its
  label carries `_CLASSIFICATION_NOT_EVALUABLE`.
- **Y1 — the compact predictor, strict boundary (new tokens × exposed frames; numbers committed before `confirm`):**
  `Δā̂(w)` versus `Δā(w)`: **Spearman ≥ 0.80** and **R² ≥ 0.50**; and **balanced firing accuracy ≥ 0.90** over the
  pairs. Pass → `NEURON_FEATURE_PREDICTED_TOKENS`; fail → `NEURON_FEATURE_NOT_PREDICTED_TOKENS` (naming the floor).
  Exposed-subset values: 0.970 / 0.940 / 0.993 (majority baseline 0.68).
- **Y2 — the same, frame-conditional (new tokens × previously untested frames; numbers digested at stage 1):** the
  same three floors. Pass → `NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL`; fail → the `_NOT_` label.
- **Y3 — the number-axis alternative is rejected on the fresh data:** over the scored token means of both sets,
  `Δā̂_axis` versus `Δā`: **R² < 0.30**, **and** its balanced firing accuracy over the pairs of both sets **< 0.70**.
  Both → `AXIS_ONLY_REJECTED`; otherwise `AXIS_ONLY_NOT_REJECTED`; with the classification guard triggered, the
  balanced-accuracy condition is non-evaluable and the family is `AXIS_ONLY_NOT_EVALUABLE`. Exposed-subset values: R²
  0.09 per token (−0.15 per pair), balanced accuracy 0.59 (sensitivity 0.92, specificity 0.26). This family tests the
  alternative, not the predictor, and the alternative has no free parameter to tune after the fact; a `NOT_REJECTED`
  with a passing Y1 would mean the number axis alone carries the neuron on the fresh set, contradicting the exposed
  accounting.
- **Descriptive (no floor):** the Jacobian form's own statistics; the accounting of `Δp̂re_J` by source and axis /
  off-axis part per token; the LayerNorm-linearization and pattern-change remainders; MAE over all pairs and the
  amplitude error among firing pairs; the operating points of the fresh frames; the firing set of the fresh tokens by
  lexical class beside the predicted set (the frozen rule `Δa ≥ 0.5`); the neuron's share of the block-2 read change;
  the interpretation, offered here and only here: what the firing set has in common, in the network's own terms
  (which exposed and fresh cues turn the neuron on) — no label is preregistered.
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities, software defects) stop the phase, are recorded with
  their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- The predictor is compact but not weight-only: it uses the frame's reference operating point and the Experiment 013
  frozen-pattern *predicted* arriving change (weights, locked axes, reference states) — never a measured patched
  residual. Y1 carries the strict boundary for the token dimension; Y2 is conditional on the observed reference state,
  as in Experiment 013.
- Passing Y1–Y3 shows that the neuron's activation change under a cue swap is prospectively reconstructed from its
  own input weights applied, through the exact LayerNorm at the frame's reference state, to the upstream change the
  previously decoded circuit predicts, and that a simple grammatical-number axis cannot account for the activation
  boundary. It does not say why the network has this direction, what the neuron does in other contexts, or anything
  about behaviour; the semantic gloss of the firing set is descriptive and comes after the scoring.
- Five lexical classes, three templates, this checkpoint; one neuron.

## Minimal implementation boundary

A module `neuron_feature.py` reusing `attention_paths.py` (reference states, frozen-pattern arriving change, stage
machinery, scoring patterns), `layer_correction.py` (`LayerWeights`), `head_transport.py`, `read_assembly.py`,
`encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; a committed extract of Experiment 013's per-pair
`c_L`, `c_M`, `c_H` for replication; the confirmation builder with the frozen lists, frames and the two prompt lists; a
runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages.
Tests: `Δa` from the captured residual equals the neuron's contribution in `layer_correction.neuron_ledger`; the
Jacobian form equals the exact-LayerNorm difference to first order (a small change) and the exact form is used for the
prediction; the axis alternative equals the predictor when the arriving change lies along `d̂_E`; the prediction table
is computed with every capture and intervention entry point disabled (the invariant); balanced accuracy and its guard
on synthetic tables; every floor branch; the stage barrier; phase isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and no
fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1** (commit `a62fa07`): initial draft. Reviewed: approved conceptually with six changes — say "below the
  preregistered firing threshold" rather than "off"; freeze the exact LayerNorm equations (γ, centering, variance
  derivative, ε) of the predictor; make the predicted-never-measured status of the arriving change a formal invariant;
  keep MAE descriptive and report amplitude error among firing pairs; replace raw firing agreement by balanced accuracy
  with a class-balance guard, in Y1/Y2 and in Y3; define the axis-only alternative as rigidly as the predictor, with no
  fitted scale or intercept.
- **Revision 2**: all six made. The predictor is the exact LayerNorm and the neuron's input weights evaluated at the
  predicted state (no linearization); the first-order Jacobian form is frozen as the compact feature for the accounting
  and reported beside it; the alternative is the same computation on the number-axis component of the same predicted
  change; balanced accuracy ≥ 0.90 (Y1/Y2) and < 0.70 (Y3) with the guard and the majority baseline; the invariant is
  stated and tested. Design-check figures restated for the exact form (Spearman 0.946, R² 0.905, balanced accuracy
  0.993) and the Jacobian form (0.918 / 0.868 / 0.986). No floor of revision 1 was loosened.

# Experiment 007: A Response-Supervised Low-Rank Cue Subspace — Prospective Prediction on the Untouched Holdout

**Date:** 2026-09-18

**Status:** Revision 3, for review. Not approved. No Experiment 007
directory, lock, or model run exists. Experiment 006 is closed at
`QUALITY_GATE_FAILED` and is not amended by this document. Revisions 2 and 3
change only the items listed under "Revision history"; the hypothesis, the
folds, the rank rule, the quality gate, τ, the confirmation set, and the
outcome rule are unchanged from revision 1.

**Scope:** the same count-cued singular/plural contrast in pinned
`EleutherAI/pythia-70m-deduped`, the same inherited circuit (E = `L00.MLP` at
the cue position, T = `L03.H04`, R = {`L04.MLP`, `L05.MLP`} at the noun
position), the same exposed pool, and **exactly** the confirmation set frozen
for Experiment 006, inherited byte-for-byte as an untouched holdout.

## Purpose

Experiment 006 falsified one specific hypothesis: that the cue-to-readout
computation can be captured by the leading ≤ 4 *unsupervised* principal
directions of the `L00.MLP` cue representations. Under leave-one-cue-out over
the sixteen exposed cue tokens the whole PCA family predicted poorly (rank 1
error 2.305 ± 0.296 nats, rank 4 only about 9% better, normalized RMSE 0.623),
so no lock was written. A post-hoc diagnostic suggested — without proving —
that the variance-leading directions of `E(w)` are poorly aligned with the
downstream-relevant directions, since PCA ranks 8 and 15 fit far better
in-sample.

Experiment 007 asks the question that result leaves open:

> Does a small **response-supervised** subspace of the token-local `L00.MLP`
> representation — directions chosen to predict the measured downstream
> causal response rather than to explain the variance of `E` — prospectively
> predict the E-patch response and the behavioral shift of unseen cue tokens,
> in unseen frames, with unseen nouns?

The experiment is primarily about the decompiler. The circuit is not
re-searched; its criteria are inherited unchanged and reported, and C002's
maturity is out of scope unless those already-frozen criteria happen to pass
prospectively.

## Inherited fixed elements

- Model, runtime, instrumented forward path, contrast `c(x)`, exact final
  LayerNorm (population variance, `ε = 1e-5`), reference cues (`one` for
  cardinal and coordinated-adjective, `each` for quantifier), reference-relative
  coordinates, the E-patch residual response `Δr_Epatch(w, f)` (final
  pre-LayerNorm residual change at the noun position when `E(ref_T)` is
  replaced by the weight-only `E(w)` in frame `f`'s reference prompt), and the
  cue-level statistics — all exactly as in the Experiment 006 design.
- Exploratory pool: the twelve exposed frames (six manifest, six Experiment
  005 extension frames), the sixteen exposed cue tokens, the sixty Experiment
  005 nouns, and the Experiment 006 E-patch responses (recomputed
  deterministically; they must match Experiment 006's recorded per-(token,
  frame) mean shifts to 1e-6, else incident).
- Confirmation set: `experiments/006-low-rank-cue-decompilation/confirmation-v1.json`,
  content sha256 `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`,
  read in place and never copied, altered, filtered, reordered, or selected
  from. Its twenty fresh nouns, six fresh frames, and twenty-four fresh cue
  tokens have never produced a model output. The lock records the digest and
  `confirm` refuses if it differs by one byte.
- Seeds: runtime `20260916`, control `20260920`.

## The program: a shared supervised subspace with template-specific readout

```text
ΔE_T(w)       = E(w) − E(ref_T)                       token-local, weights only
z(w, T)       = U_rᵀ ΔE_T(w)                          shared U_r ∈ ℝ^{d_model × r}, orthonormal columns
Δr̂(w, f)     = V_T z(w, T)                            template-specific V_T ∈ ℝ^{d_model × r}
Δĉ_N(w, f)   = −u_N · [ LN(ρ_f + V_T z(w, T)) − LN(ρ_f) ]
```

`U_r` is **shared across templates**, so `z` is a single compact cue
representation; only the readout `V_T` is template-specific. `ρ_f` is the
clean final pre-LayerNorm residual of an exposed frame's reference prompt and
`ρ_template` (the mean over the template's exposed frames) stands in for a
fresh frame. For `w = ref_T`, `z = 0` and every prediction is exactly zero.

### Estimator (deterministic, no tunable hyperparameter besides `r`)

The subspace is learned to predict the response. Stack, over every exposed
token `w` and frame `f`, the predictor rows `x(w, f) = ΔE_T(w)` and response
rows `y(w, f) = Δr_Epatch(w, f)` into `X, Y ∈ ℝ^{n × d_model}` with
`n = 16 × 12 = 192` (the reference token's rows are zero on both sides and are
kept). Neither matrix is centered and no intercept is fitted anywhere,
because the coordinates are already reference-relative: the zero point is
meaningful, but the sample means are not zero, so the matrix `XᵀY` below is an
**uncentered cross-moment**, not a covariance.

`U_r` is the **supervised cross-moment SVD subspace**:

```text
C   = Xᵀ Y                 ∈ ℝ^{d_model × d_model}, the uncentered cross-moment
C   = P Σ Qᵀ               singular value decomposition, σ₁ ≥ σ₂ ≥ … (float64)
U_r = P[:, :r]             the first r left singular vectors of C
```

equivalently `U_r = argmax_{UᵀU = I_r} ‖Uᵀ XᵀY‖²_F`. There is no deflation
and no iteration, so there is no partial-least-squares variant to choose
between: the estimator is the leading left singular subspace of the
cross-moment. It is response-supervised (directions of `ΔE` that carry the
largest uncentered cross-moment with the measured E-patch response), shared
across templates, orthonormal by construction, and nested — `U_r` is the
first `r` columns of `U_4`, so ranks 1–4 share directions. Every prediction
of the program depends only on the span of `U_r` (`V_T` absorbs any rotation
or sign change within it); for reproducible exported tensors, each column's
sign is fixed so that its largest-magnitude entry is positive.

**Singular-gap rule.** The span of `U_r` is ill-defined when `σ_r = σ_{r+1}`,
and a rank whose prediction is not uniquely defined in some fold could decide
the selection. So the check applies to every SVD that contributes to
selection, not only to the final one:

```text
for every outer LOCO fold (16 fits on fifteen tokens) and for the final fit on all sixteen tokens:
    require rank(C) ≥ 5                              (σ₅ must exist; the exposed X spans ≤ 15 dimensions, so this is generic)
    for every evaluated r ∈ {1, 2, 3, 4}:
        gap_r = (σ_r − σ_{r+1}) / σ_1
        require gap_r ≥ 1e-8
```

`rank(C)` is the number of singular values exceeding
`max(C.shape) × eps(float64) × σ_1`. Every `gap_r` is recorded. Any violation
is a numerical/identifiability incident (see "Incidents"), never
`QUALITY_GATE_FAILED`; none is expected with `d_model = 512` and `n = 192`.

`V_T` is then the least-squares map, without intercept, from
`z(w, T) = U_rᵀ ΔE_T(w)` to `Δr_Epatch(w, f)` over the exposed tokens and the
frames of template `T`, solved deterministically by the Moore–Penrose
pseudoinverse in float64 rather than by the normal equations, so that the
solve is defined without a full-column-rank assumption:

```text
Z_T   = X_T U_r                              one row per (training token, frame of T): 60 × r in a fold, 64 × r in the final fit
V_Tᵀ  = pinv(Z_T) Y_T                        SVD pseudoinverse, rcond = max(Z_T.shape) × eps(float64)
rank(Z_T) = #{ singular values of Z_T > rcond × σ_max(Z_T) }
```

`rank(Z_T)` is recorded for every template in every fold and in the final
fit. **Identifiability rule:** `rank(Z_T) < r` for any template, in any fold
or in the final fit, is an incident. This is part of the hypothesis, not a
numerical convenience: `r` is meant to be an actual shared `r`-dimensional
representation read by every template, so every coordinate of `z` must be
identifiable inside each template's own rows (`Z_T` has at least fourteen
distinct nonzero rows in every fold — far more than `r ≤ 4` — so the rule is
generically satisfied). When `rank(Z_T) = r` the pseudoinverse
solution coincides with the ordinary least-squares solution
`(Z_Tᵀ Z_T)⁻¹ Z_Tᵀ Y_T`; at most `r = 4` coefficients per output coordinate
are estimated from 64 rows, so no regularization is needed at that stage.
The same pseudoinverse solve, tolerance, and rank recording are used for the
`V_T` of the `PCA-006(r)` baseline; a rank deficiency there is reported, not
an incident, because the baseline is never selected against.

Neither step has a penalty parameter, so nothing is tuned inside or outside
the folds; `r ∈ {1, 2, 3, 4}` is the only choice and it is made by the rule
below.

### Baselines (frozen, reported, never selected against)

- `PCA-006(r)`: the Experiment 006 family at the same rank, refitted on the
  same folds (reference-relative `Δz`, unsupervised `U_r`).
- `E005-scalar`: the frozen Experiment 005 scalar program.
- `Ridge-full`: the full-dimensional regularized linear baseline
  `Δr̂(w, f) = B_T ΔE_T(w)` with **one map per template**,
  `B_T ∈ ℝ^{d_model × d_model}`, the full-rank analogue of the primary
  family's `V_T U_rᵀ` (a single shared `B` would be a stricter model than the
  decompiler and could not answer the intended question). Each `B_T` is a
  ridge regression without intercept, `B_Tᵀ = (X_Tᵀ X_T + λ I)⁻¹ X_Tᵀ Y_T`,
  with one `λ` shared by the three templates within a fold and chosen by a
  fully nested cue-group procedure. For each outer held-out cue token:
  1. remove every row of that token (all twelve frames) from `X` and `Y`;
  2. over the remaining fifteen training tokens, run an inner
     leave-one-cue-out;
  3. in each inner fold, form the grid
     `λ ∈ {10⁻², 10⁻¹, 1, 10, 10²} × tr(X_innerᵀ X_inner)/d_model` from the
     inner-training rows only (`X_inner` = the pooled rows of the fourteen
     inner-training tokens across the three templates), fit the three `B_T`
     on those rows at every grid position, and predict the inner held-out
     token's E-patch shift in every exposed frame and noun;
  4. score each grid position by the mean, over the fifteen inner held-out
     tokens, of the same per-token aggregate mean absolute error used for the
     primary family;
  5. select the grid position (the multiplier) with the smallest inner score;
     ties go to the larger multiplier;
  6. refit the three `B_T` on all fifteen outer-training tokens at
     `λ = multiplier × tr(X_outerᵀ X_outer)/d_model`, the trace taken over
     those fifteen tokens' pooled rows;
  7. predict the outer held-out token.
  The selected multiplier of every outer fold is recorded. Nothing about `λ`
  is ever chosen using the outer held-out token or the confirmation set, and
  because the grid is a fixed set of multipliers of the training trace, even
  the scale of the grid is leakage-free. For the confirmation report (Y4),
  `Ridge-full` is fitted once on all sixteen exposed tokens with the
  multiplier chosen by the same procedure run as a leave-one-cue-out over the
  sixteen tokens, and the resulting `B_T` are frozen in the lock. This
  baseline tests whether a full-dimensional regularized linear map
  generalizes materially better than the selected low-rank map under the
  same outer folds; it is not a bound on all linear maps.

## Rank selection and the pre-lock quality gate (unchanged from Experiment 006)

Outer leave-one-cue-out over the sixteen exposed tokens; the cue token is the
unit. For each rank and each held-out token, fit `U_r` and every `V_T` on the
other fifteen tokens and predict the held-out token's E-patch shift in every
exposed frame and noun. Each held-out token yields one aggregate mean absolute
error and one signed cue-level pair (mean predicted, mean measured). Then:

```text
error_r, SE_r over the 16 held-out tokens
eligible  = {1} ∪ {r > 1 : error_r ≤ 0.8 × error_1}
r_best    = argmin_{r ∈ eligible} error_r
threshold = error_{r_best} + SE_{r_best}
selected  = min {r ∈ eligible : error_r ≤ threshold}
```

The lock may be written only if the selected model, on the sixteen cue-level
pairs, satisfies the 20% rule when `r > 1`, has Spearman ≥ 0.70, and has
normalized LOCO RMSE ≤ 0.50. If the gate fails, the experiment ends at Tier A
with `QUALITY_GATE_FAILED` and the confirmation set stays untouched, exactly
as in Experiment 006. τ = max(0.5 nats, 3 × RMSE of the sixteen cue-level
errors of the selected rank), a calibration width only.

The same LOCO table is computed for `PCA-006(r)` and `Ridge-full` (its outer
folds are the same sixteen held-out tokens; the nested `λ` selection lives
inside each fold), and the comparison is recorded before the lock: the
supervised model is expected to beat `PCA-006` at the same rank by at least
20% on LOCO error (a stated prediction, not a gate), and `Ridge-full`'s LOCO
error is recorded next to the selected rank's for the three-way reading given
under "Interpretation limits".

## Confirmation (once, after the lock)

Identical to Experiment 006's program axis on the inherited confirmation set:
Y1 (E-patch prediction for the 24 fresh tokens in the 6 fresh frames × 20
fresh nouns; Spearman ≥ 0.80, MAE ≤ τ, confident-token sign agreement in
≥ 5/6 frames), Y2 (behavioral shift; Spearman ≥ 0.70, MAE ≤ 1.5 τ), Y3
(template-specific original cue pair in each fresh frame; per-template mean
within τ; `d_full > 0` in ≥ 108/120), bands (≥ 18 of 24 tokens inside ± τ in
≥ 5 of 6 frames on Y1), Y4 (the Y1/Y2 statistics of `PCA-006` at the
selected rank, `E005-scalar`, and the template-specific `Ridge-full`, each
fitted on all sixteen exposed tokens and frozen in the lock; reported, never
selected against), Y5 (residuals, including the E-route versus behavioral
difference). The
cue-effect gate on the fresh frames (`d_full > 0` in ≥ 108/120 pairs) is the
precondition.

The inherited circuit families (P1, P3-fidelity, P4, P5, P7, P8, P9 with
Experiment 006's unchanged floors, including P3's 0.90 correlation) are run on
the fresh nouns over the manifest frames and over the fresh frames and are
**reported**; they do not enter the outcome label. If they pass on both sets,
C002 is reviewed for promotion citing them; otherwise C002 is unchanged.

### Outcome rule (program axis)

- `QUALITY_GATE_FAILED` — pre-lock; nothing fresh executed.
- `CUE_EFFECT_NOT_REPLICATED` — the fresh-frame precondition fails.
- `DECOMPILED` — Y1–Y3 pass and the bands are hit.
- `DECOMPILED_MISCALIBRATED` — Y1–Y3 pass; bands missed.
- `PROGRAM_NOT_SUPPORTED` — a Y floor fails; the failing family names where
  (encoding subspace, transport/readout linearity, or frame context).

### Incidents

An incident is not an outcome label. The frozen incidents are: an exposed
E-patch response that fails to match Experiment 006's recorded mean to 1e-6;
`rank(C) < 5` or a singular gap `gap_r < 1e-8` in any contributing SVD;
`rank(Z_T) < r` for any template in any fold or the final fit; a lock digest
or confirmation-set digest mismatch; and any software defect. An incident
stops the phase where it occurs and is recorded in the results state and the
report. It follows the incident rule of Experiments 005 and 006: incident
note, invalidation of the affected artifacts, a committed fix or a documented
protocol amendment made before any data beyond the incident is looked at,
and a rerun of the affected phase only under that amendment; a defect found
after the confirmation run invalidates it and permits a rerun of the entire
confirmation only under a new protocol version whose report also carries the
invalidated result. Resolving an incident never touches the confirmation set
and never changes a threshold, the rank rule, or the quality gate. There are
no scientific retries.

## Interpretation limits

- Sixteen exposed cue tokens define the subspace; the outer leave-one-cue-out
  procedure is the arbiter of generalization before the lock, and the
  twenty-four fresh tokens are the only prospective test.
- A supervised subspace found from responses is a description of the
  downstream-relevant directions of `E`, not a claim about how the model
  represents number in general.
- The program is linear from `z` to the residual increment; nonlinearity
  appears as residual.
- `Ridge-full` is a full-dimensional regularized linear baseline, not a bound
  on all linear maps: a finite-grid ridge estimator is one estimator, and
  other regularizers or estimators could generalize better. It tests whether
  a full-dimensional regularized linear map generalizes materially better
  than the selected low-rank map under the same outer folds. Reading:
  supervised low-rank ≈ `Ridge-full` — compact dimensionality is plausible;
  `Ridge-full` ≫ low-rank — the relevant mapping may be linear but not
  ≤ 4-dimensional; both poor — linear generalization from `E(w)` itself is
  questionable.
- Nothing generalizes beyond the pinned checkpoint, the three template
  families, single-token regular nouns, and the tokens tested.

## Minimal implementation boundary

Reuse the Experiment 006 module for the exposed pool, E-patch responses, LOCO
scaffolding, cue-level statistics, the rank rule, the quality gate, τ, the
confirmation measurements, Y floors, bands, lock validation, and the report;
add the cross-moment SVD subspace estimator and the template-specific ridge
baseline (nested cue-group `λ`) as pure tensor functions with tests (recovery
of a planted low-rank supervised map; orthonormality; nestedness
`U_r = U_4[:, :r]`; determinism and sign convention; invariance of predictions
to column signs; zero prediction at the reference; the singular-gap check
raising on a planted tie in a fold; the pseudoinverse solve with the frozen
`rcond` and `rank(Z_T)` recording, raising on a planted rank-deficient
template; `λ` grid computed from the inner-training rows only and one `B_T`
per template), a program module that loads only exported tensors, and a
runner with phases
`validate`, `explore`, `calibrate`, `lock`, `confirm`, `report` (no freeze
phase: the confirmation set is inherited). No new prompts, no component
search, no other model.

## Approval and stopping condition

Design first; no implementation until approved. The confirmation set is
already frozen. Tier C runs once after the lock commit, and only if the
pre-lock quality gate passes.

## Revision history

- **Revision 1** (commit `10c8faf`): initial draft. Reviewed: not approved,
  with one main blocker (the supervised estimator named "uncentered SIMPLS"
  was specified verbally; partial-least-squares variants differ in their
  normalization and orthogonality constraints, so two implementations could
  both claim to follow the text and produce different `U_r`), one secondary
  blocker (`Ridge-full` read as a single shared `B`, a stricter model than the
  decompiler), and one wording error ("bounds what any linear map achieves").
- **Revision 2**: (1) the estimator is now the explicit supervised
  cross-moment SVD subspace — `U_r` = first `r` left singular vectors of the
  uncentered cross-moment `XᵀY`, equivalently
  `argmax_{UᵀU = I} ‖Uᵀ XᵀY‖²_F` — with no deflation, a fixed sign convention,
  nested ranks, and `V_T` by no-intercept least squares from `z`; `XᵀY` is
  described as an uncentered cross-moment, not a covariance. (2) `Ridge-full`
  is template-specific (`B_T` per template) with a fully nested cue-group `λ`
  selection: the grid is a fixed set of multipliers of
  `tr(X_trainᵀ X_train)/d_model` computed from the inner-training rows only,
  the multiplier is chosen by inner leave-one-cue-out over the fifteen
  training tokens, and the three `B_T` are refitted on the outer-training
  tokens before predicting the outer held-out token; the lock-time fit for Y4
  is defined the same way over the sixteen exposed tokens. (3) The
  interpretation of `Ridge-full` is corrected to "full-dimensional
  regularized linear baseline" with the three-way reading. No other element
  changed. Reviewed: scientifically sound; two numerical identifiability
  ambiguities remained (the `V_T` solve assumed full column rank of `Z_T`;
  the singular-gap check covered only the final fit although sixteen
  outer-fold SVDs per rank feed the selection).
- **Revision 3** (commit `f64a609` reviewed): (1) `V_T` is solved by the
  float64 Moore–Penrose pseudoinverse with `rcond = max(Z_T.shape) × eps(float64)`,
  `rank(Z_T)` is recorded for every template in every fold and the final fit,
  and `rank(Z_T) < r` anywhere is an identifiability incident (the stricter
  option: all `r` coordinates must be identifiable in every template because
  `r` denotes a shared representation read by every template); the same solve
  is used for the `PCA-006(r)` readout, where a deficiency is reported only.
  (2) The singular-gap rule `gap_r = (σ_r − σ_{r+1})/σ_1 ≥ 1e-8` applies to
  every evaluated `r ∈ {1..4}` in every outer LOCO fold and in the final
  sixteen-token fit, with `rank(C) ≥ 5` required so that `σ_{r+1}` exists; any
  violation is a numerical/identifiability incident, not
  `QUALITY_GATE_FAILED`. (3) An "Incidents" subsection freezes what an
  incident is and how it is resolved, mirroring Experiments 005 and 006. No
  other element changed.

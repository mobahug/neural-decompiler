# Experiment 007: A Response-Supervised Low-Rank Cue Subspace — Prospective Prediction on the Untouched Holdout

**Date:** 2026-09-18

**Status:** Draft for review. Not approved. No Experiment 007 directory, lock,
or model run exists. Experiment 006 is closed at `QUALITY_GATE_FAILED` and is
not amended by this document.

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
kept). `U_r` is the first `r` weight directions of **uncentered SIMPLS**
(partial least squares, multi-response) on `(X, Y)`: the deterministic
sequence of unit vectors `u_k` maximizing the covariance `‖Yᵀ X u_k‖` subject
to orthogonality to the earlier score directions, orthonormalized. No
intercept and no centering, because the coordinates are already
reference-relative. `V_T` is then the ordinary least-squares map, without
intercept, from `z(w, T)` to `Δr_Epatch(w, f)` over the exposed tokens and the
frames of template `T` (at most `r = 4` coefficients per output coordinate
from 64 rows, so no regularization is needed at that stage). Because SIMPLS
has no penalty parameter, nothing is tuned inside or outside the folds;
`r ∈ {1, 2, 3, 4}` is the only choice and it is made by the rule below.

### Baselines (frozen, reported, never selected against)

- `PCA-006(r)`: the Experiment 006 family at the same rank, refitted on the
  same folds (reference-relative `Δz`, unsupervised `U_r`).
- `E005-scalar`: the frozen Experiment 005 scalar program.
- `Ridge-full`: the full linear map `Δr̂ = B ΔE_T(w)` with `B` estimated by
  ridge regression, the ridge parameter chosen by an inner leave-one-cue-out
  over the fifteen training tokens of each outer fold from the fixed grid
  `λ ∈ {10⁻², 10⁻¹, 1, 10, 10²} × tr(XᵀX)/d_model`; nothing about `λ` is ever
  chosen using the confirmation set. This baseline answers whether the
  limitation is low dimensionality or the instability of any linear map from
  `E(w)` across unseen tokens.

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

The same LOCO table is computed for `PCA-006(r)` and `Ridge-full`, and the
comparison is recorded before the lock: the supervised model is expected to
beat `PCA-006` at the same rank by at least 20% on LOCO error (a stated
prediction, not a gate), and `Ridge-full`'s LOCO error is the reference for
"low dimensionality versus linear instability".

## Confirmation (once, after the lock)

Identical to Experiment 006's program axis on the inherited confirmation set:
Y1 (E-patch prediction for the 24 fresh tokens in the 6 fresh frames × 20
fresh nouns; Spearman ≥ 0.80, MAE ≤ τ, confident-token sign agreement in
≥ 5/6 frames), Y2 (behavioral shift; Spearman ≥ 0.70, MAE ≤ 1.5 τ), Y3
(template-specific original cue pair in each fresh frame; per-template mean
within τ; `d_full > 0` in ≥ 108/120), bands (≥ 18 of 24 tokens inside ± τ in
≥ 5 of 6 frames on Y1), Y4 (all baselines' Y1/Y2 statistics, reported), Y5
(residuals, including the E-route versus behavioral difference). The
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

## Interpretation limits

- Sixteen exposed cue tokens define the subspace; the outer leave-one-cue-out
  procedure is the arbiter of generalization before the lock, and the
  twenty-four fresh tokens are the only prospective test.
- A supervised subspace found from responses is a description of the
  downstream-relevant directions of `E`, not a claim about how the model
  represents number in general.
- The program is linear from `z` to the residual increment; nonlinearity
  appears as residual. `Ridge-full` bounds what any linear map achieves under
  the same folds.
- Nothing generalizes beyond the pinned checkpoint, the three template
  families, single-token regular nouns, and the tokens tested.

## Minimal implementation boundary

Reuse the Experiment 006 module for the exposed pool, E-patch responses, LOCO
scaffolding, cue-level statistics, the rank rule, the quality gate, τ, the
confirmation measurements, Y floors, bands, lock validation, and the report;
add the SIMPLS subspace estimator and the ridge baseline (nested-fold λ) as
pure tensor functions with tests (recovery of a planted low-rank supervised
map; orthonormality; determinism; zero prediction at the reference), a
program module that loads only exported tensors, and a runner with phases
`validate`, `explore`, `calibrate`, `lock`, `confirm`, `report` (no freeze
phase: the confirmation set is inherited). No new prompts, no component
search, no other model.

## Approval and stopping condition

Design first; no implementation until approved. The confirmation set is
already frozen. Tier C runs once after the lock commit, and only if the
pre-lock quality gate passes.

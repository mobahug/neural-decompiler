# Experiment 012: What Layers 1–2 Add — Is the Correction to the Encoding Read a Token-Local MLP Computation? — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 1 — draft for review. No Experiment 012 directory, confirmation set, lock, or model run exists.
Experiments 005–011 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter. The quantity under test is a **token-local forward model of the two MLPs of
blocks 1 and 2 at the cue position**, applied to the cue's layer-0 encoding difference with attention held at the
reference and the base state taken from exposed reference prompts; nothing is fitted to any cue. A confirmation set of
new cue tokens and new frames is frozen by tokenizer rules before any Experiment 012 model output; every numeric
prediction is committed before the single confirmation run, and the prediction phase performs **no forward pass on
any fresh prompt**.

## Purpose and question

Experiment 011 confirmed prospectively that the weight-defined encoding read `g_E` predicts `L03.H04`'s transport of
never-executed cues (Spearman 0.848, MAE 0.077 over 24 tokens). Its residual is systematic: `q̄_T − ḡ_E` was positive
for 17 of 24 tokens (mean +0.04), and its rank order follows the net contribution of layers 1–2 to the head-readable
signal (Spearman 0.66), which averaged +0.18 on the fresh set. The encoding read omits that contribution by
construction. The exposed ledger — Experiment 010's 63 tokens × 24 frames and Experiment 011's 24 × 6, 1630 (token,
frame) pairs with the exact attribution `ρ_f(Δr_c) = ρ_f(ΔE) + Σ_k ρ_f(Δout_k)` over the 18 components of layers 1–2 —
says where the correction lives, descriptively:

- `L02.MLP` carries the mean of the correction (+0.126 of the layers' +0.125, in units of the plural cue's signal;
  positive in 83% of pairs). `L01.MLP` is zero-mean with spread ±0.06. Each of the sixteen attention heads has mean
  absolute contribution ≤ 0.034.
- The correction is additively separable into a token effect and a frame effect (R² 0.89 for `L02.MLP`, 0.91 for the
  net layer change), and the frame effect is mostly the template's: quantifier frames +0.22, cardinal +0.07,
  coordinated-adjective +0.09 for `L02.MLP`.
- It is **not** a gain on the number-axis content of the encoding: the token-level rank correlation between the net
  layer change and `f_E` is 0.48 (numerals from `ten` upward receive +0.25 to +0.44, `these`, `those`, `three`, `two`
  with equally high `f_E` receive +0.03 to +0.11, `both` and `either` receive −0.09 and −0.13).

The next link of the chain is therefore not another predictor of transport but the computation that produces this
correction. Because the block-0 MLP is parallel to block-0 attention (GPT-NeoX parallel residual), the E-patch changes
the cue position's residual before block 1 by **exactly** `ΔE = E(w) − E(ref_T)`; block 1's MLP output change at the
cue position is therefore an exact function of `ΔE` and the reference state, and block 2's MLP output change is a
function of `ΔE`, the reference state, block 1's MLP change, and block 1's attention change at the cue position.
This suggests a zero-parameter candidate for the computation: **the two MLPs, evaluated at the cue position on the
encoding difference, with attention held fixed**.

> Is the layers-1–2 correction to the head-readable number signal a token-local nonlinear function of the cue's
> layer-0 encoding computed by the MLPs of blocks 1 and 2 at the cue position — with the frame entering only through
> the base state and attention heads of layers 1–2 contributing no systematic amount — for cue words and frames the
> model has never been measured on, **before any of their layer-1–2 or head behaviour is observed**?

**Design check on exposed data (outside any results state; to be recomputed inside Tier A).** On Experiment 010's 63
tokens × 24 frames, with Experiment 011's locked read direction: (i) block 1's MLP change computed from the frame's own
reference state and `ΔE` equals the recorded `ρ_f(Δout_{L01.MLP})` to 6.6e-7 relative — the exactness above, verified;
(ii) the token-local model with each frame's own reference state (omitting only block 1's attention change in block
2's input) reproduces the recorded net layer change per token with Spearman 0.950 and MAE 0.035 (spread 0.118); (iii)
with the **template's mean exposed reference state** instead of the frame's own — the form usable before any fresh
prompt runs — Spearman 0.946 and MAE 0.048 per token, 0.884 and 0.073 per pair; the residual is largest in the
quantifier template (pair Spearman 0.72, MAE 0.09), where the correction is largest; (iv) the composite "encoding read
plus modelled correction" predicts the head's exact P1 fraction per token with Spearman 0.977 and MAE 0.037, against
0.887 and 0.085 for the encoding read alone — while against measured transport `q_T` the two are equivalent (0.907
versus 0.925 rank, 0.071 versus 0.064 MAE), because `g_E`'s plural-cue normalization happens to absorb the average
correction and the remaining gap to `q_T` is the head's own attention modulation (Experiment 009's P3 − P1). Experiment
012 tests (iii) and (iv) prospectively; it does not aim to improve the prediction of `q_T`.

## The quantities under test (frozen definitions)

Notation as in Experiments 010–011: `ρ_f` the frame's exact read functional, `r(x) = ⟨x − mean(x)·1, γ₃ ⊙ m⟩` its
frame-free inner read (the frame scalars cancel in every fraction below), `d̂_T`, `d̂_E`, `γ₃ ⊙ m` and `σ_T` **exactly
the vectors and scalar locked by Experiment 011** (digest-bound; not re-estimated — the read direction is fixed once,
and `confirm` re-checks the recomputed axes against them as Experiment 011 did), `ref_T` and `pl_T` the template's
reference and plural cues, `E(w)` the weight-only layer-0 encoding, `ΔE = E(w) − E(ref_T)`.

**Measured (exact, from the E-patch of `w` into frame `f`'s reference prompt; Experiment 010's identity):**

```text
c_M(w, f) = [ρ_f(Δout_{L01.MLP}) + ρ_f(Δout_{L02.MLP})] / ρ_f(Δr_c(pl_T, f))      the MLP correction
c_H(w, f) = Σ_{16 heads of layers 1–2} ρ_f(Δout_k) / ρ_f(Δr_c(pl_T, f))            the head correction
c_L(w, f) = c_M + c_H                                                               the net layer change (010's f_layers)
```

**Predicted (token-local model; weights, the locked axes, and locked exposed base states — no fresh prompt):** with
`x̄₁,T` and `x̄₂,T` the residual stream at the cue position **before blocks 1 and 2 in the reference prompt**, averaged
over the exposed frames of template `T`, and `ln2ₗ`, `MLPₗ` block `l`'s MLP LayerNorm and MLP:

```text
Δ̂₁(w, T) = MLP₁(ln2₁(x̄₁,T + ΔE)) − MLP₁(ln2₁(x̄₁,T))
Δ̂₂(w, T) = MLP₂(ln2₂(x̄₂,T + ΔE + Δ̂₁)) − MLP₂(ln2₂(x̄₂,T))
D̂_T      = r(E(pl_T) − E(ref_T)) + r(Δ̂₁ + Δ̂₂)(pl_T, T)                             the plural cue's modelled total
ĉ(w, T)  = r(Δ̂₁ + Δ̂₂)(w, T) / D̂_T                                                  the predicted layer correction
q̂(w, T)  = [r(ΔE) + r(Δ̂₁ + Δ̂₂)(w, T)] / D̂_T                                        the composite P1-level prediction
```

`ĉ` and `q̂` depend on the template only. Their companions, defined the same way and locked beside them: the split
into the `MLP₁` and `MLP₂` parts; the model evaluated on the number-axis part `ΔE_∥` alone and on `ΔE_⊥` alone, with
the interaction `ĉ(ΔE) − ĉ(ΔE_∥) − ĉ(ΔE_⊥)` (the model is nonlinear; this is an exact evaluation, not a decomposition
claim); and the encoding read `g_E` of Experiment 011 for the fresh tokens, reported for continuity.

**Denominator validity (frozen before any fresh measurement):** a template's predictions are `PREDICTION_UNDEFINED`
if `D̂_T` is below `0.25 × max_T |D̂_T|` or below `0.25 σ_r` (Experiment 011's rule and `σ_r`); such a template's fresh
frames are excluded before the lock, and with fewer than two templates the experiment stops at `PREDICTION_UNDEFINED`.

**What the model omits, exactly.** With the frame's own reference state, `Δ̂₁` is the measured `Δout_{L01.MLP}`
(no approximation), and `Δ̂₂` omits one term of block 2's input: block 1's attention output change at the cue
position. With the template base, both omit the frame's deviation from the template mean. `confirm` therefore also
evaluates, **after** the fresh run and descriptively, the model at each fresh frame's own reference state (the
"own-base ladder": template base → own base → own base plus the measured block-1 attention change, the last being the
measured `Δout_{L02.MLP}` up to float error and checked as an identity), so that a Y1 failure is attributable to the
base-point term or to the attention-input term.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, P1 and
its OV identity checks, the exact attribution functional `ρ_f` and its identity checks (ρ identity, P1 cross-check,
neuron sum), the Experiment 011 locked axes and read weight, the results-state, ledger, lock, prediction-artifact, and
incident conventions — all exactly as in Experiments 010–011. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration only; nothing is fitted)

The 87 exposed cue tokens (Experiment 010's 63 and Experiment 011's 24) and the 30 exposed frames (Experiment 010's
24 and Experiment 011's 6), ten per template, with the 80 nouns for the replication checks. `explore`:

- runs every E-patch of the 87 × 30 pool with the head's internals **and the cue position's residual stream before
  blocks 1 and 2** captured in the reference and patched runs; replicates the per-component fractions `f_k` against
  committed extracts of Experiments 010 (63 × 24) and 011 (24 × 6) within 1e-6; the other pairs are new exposed
  measurements (E-patches into exposed reference prompts add no prompt key to the ledger);
- computes the base states `x̄₁,T`, `x̄₂,T` (means over the template's ten exposed frames) and locks them;
- computes the exact MLP neuron ledgers of blocks 1 and 2 from the captured residuals (`Δa_j` from the captured
  inputs; identity `Σ_j Δa_j W_out[j] = Δout` within 1e-4 relative, an incident otherwise);
- evaluates the template-base model and the own-base ladder on the 87 tokens; fixes the tolerances from exposed
  residuals at exactly the aggregation level at which the floors are scored, **token means over one frame set**
  (`F(w)` = the token's informative exposed frames, own-reference frames excluded):
  `τ_c = max(0.10, 3 × RMSE over the 87 tokens of (c̄̂(w) − c̄_L(w)))` and
  `τ_P = max(0.10, 3 × RMSE over the 87 tokens of (q̄̂(w) − P̄1(w)))`, where `P1(w, f) = ρ_f(Δr_c) / ρ_f(Δr_c(pl_T, f))`
  is the head's exact P1 fraction; the base states are in-sample for these residuals (means over the same frames) and
  this is stated in the lock;
- records the exposed descriptive statistics (Spearman, MAE, explained variance, mean signed residual of `ĉ` against
  `c̄_L` and against `c̄_M`; the heads' share `mean|c_H| / mean|c_M|`; the `∥`/`⊥` evaluations; the neuron ledgers).

## Confirmation set (frozen by tokenizer rules before any Experiment 012 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 87 exposed tokens and from
  every exposed noun form; the first eligible entries of the frozen candidate lists, with quotas. The lists are
  organized by lexical class **for coverage and reporting only — no class carries an expectation**; the only
  expectations are the committed numbers.
  - `determiner-like` (quota 5): `half`, `whichever`, `whatever`, `which`, `what`, `same`, `own`, `last`, `next`, `first`.
  - `numeral` (quota 5): `fifty`, `sixty`, `seventy`, `eighty`, `ninety`, `zero`, `thousand`, `million`, `billion`.
  - `quantity` (quota 5): `scarce`, `least`, `myriad`, `manifold`, `sparse`, `limited`, `surplus`, `endless`.
  - `possessive-or-pronoun` (quota 4): `hers`, `theirs`, `ours`, `thy`, `whom`.
  - `adjective` (quota 5): `black`, `white`, `young`, `empty`, `wooden`, `tall`, `thick`, `quiet`, `broken`.
  Every fresh cue is licensed in every fresh frame (no orthographic exception is needed in these lists). The target is
  head-level; the behavioral contrast is reported only for continuity, over the 79 single-token exposed nouns.
- **Fresh frames (6, two per template), literal:** cardinal `The warehouse ships {cue}`, `The florist arranges {cue}`;
  quantifier `The lecture reviews {cue}`, `The brochure advertises {cue}`; coordinated-adjective
  `Hugo and Amara folded {cue} flat`, `Nadia and Erik hauled {cue} dusty`. Texts must differ from every exposed frame
  text; the original cue tokens are used for their own cue prompts.
- The set (tokens with classes, frames, prompts, digest) is committed as `confirmation-v1.json` before `explore`;
  `confirm` refuses if any fresh prompt appears in the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (the prediction phase; weights, the locked axes, and the locked base states — no forward pass on any fresh
  prompt):** for every fresh token and defined template: `ĉ`, its `MLP₁`/`MLP₂` parts, the `∥`/`⊥` evaluations and
  interaction, `q̂`, `g_E`; the token means over the token's licensed fresh frames; `D̂_T`; the numeric `τ_c` and `τ_P`;
  the exposed descriptive statistics; `predictions.md` with one row per fresh token and template and the token means.
  The lock phase may read weights, the Experiment 011 lock, and the exposed results state (for the base states); it has
  no access to a capture or intervention API. `confirm` recomputes every prediction from the weights, the locked axes,
  and the locked base states before any fresh prompt runs and refuses on any difference above 1e-9.
- **Tier C (`confirm`, once):** for every fresh frame, the reference run with the head's internals and the residuals
  before blocks 1 and 2, the plural cue's E-patch (normalization and frame validity), and the sg/pl cue prompts
  (frame-level cue-effect check: `d_full > 0` in at least `exact_count_floor(108/120, 79)` of the frame's 79 noun
  pairs); then, in every valid frame, for every fresh token the E-patch with the internals and residuals captured:
  measured `c_M`, `c_H`, `c_L`, `P1`, `q_T`, the exact attribution ledger, the block-2 neuron ledger, the own-base
  ladder, and the behavioral contrast (reported).

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiment 011 — a fresh frame is valid iff its plural cue's head change is
  informative (`≥ 0.25 σ_T`) and its cue-pair check passes; a fresh token is scored iff it has at least three valid
  frames, and `F(w)` is exactly those frames for every token mean. No candidate is ever excluded on the basis of its
  own values. `PRECONDITION_FAILED` if fewer than four of the six frames are valid or fewer than sixteen tokens are
  scored; nothing is then judged.
- **Y1 — the layer correction is the token-local MLP computation (numbers committed before `confirm`):** over the
  scored fresh tokens, `c̄̂(w)` versus measured `c̄_L(w)` (the full net layer change, heads included, so that a
  systematic head contribution counts against the hypothesis): Spearman ≥ 0.80 and MAE ≤ `τ_c`. Pass →
  `LAYER_CORRECTION_TOKEN_LOCAL_MLP`; fail → `LAYER_CORRECTION_NOT_TOKEN_LOCAL` (naming the floor). The same
  comparison against `c̄_M` (MLP part only), the explained variance, and the mean signed residual are reported beside
  it, never judged.
- **Y2 — the composite account reaches the head's linear read-out:** over the scored fresh tokens, `q̄̂(w)` versus
  the measured P1 fraction `P̄1(w)` (exact from the arriving residuals; frame scalars cancel): Spearman ≥ 0.90 and
  MAE ≤ `τ_P`. Pass → `COMPOSITE_PREDICTS_P1`; fail → `COMPOSITE_FAILS_P1`. Experiment 011's `g_E` against the same
  `P̄1(w)`, and both against measured `q̄_T(w)`, are reported beside it, never judged.
- **Descriptive (no floor):** the heads' share on the fresh set against the exposed baseline; the own-base ladder
  (base-point term and attention-input term per token); the `∥`/`⊥` evaluations of the model; the block-2 neuron
  ledger — `n_80` on absolute mass, positive and negative mass, the top-20 neurons per template and their overlap
  across templates and between exposed and fresh tokens (Jaccard), and for the exposed top-20 the cosine of each
  neuron's LayerNorm-input weight with `d̂_E` (what the neurons read, descriptively); class-wise means as descriptive
  strata. Experiment 010's lesson stands: no relay or concentration bound is set without an exposed baseline, and none
  is set here.
- Outcome = `Y1 | Y2`. Incidents (replication, identities, software defects) stop the phase, are recorded with their
  commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- The prediction uses the weights, the axes locked by Experiment 011, and **exposed reference states** (template means
  of ten exposed frames each); the design never calls it "weight-only". What is not observed before the lock is
  anything about the fresh tokens or the fresh frames.
- Passing Y1 shows that the correction layers 1–2 add to the encoding read is, up to the two omissions of measured
  size, the MLPs' own nonlinear evaluation of the encoding difference at the cue position — a token-local computation
  — for this preregistered heterogeneous cue set (five lexical classes), not for arbitrary English words. It does not
  say which features of the encoding the block-2 neurons read; the neuron ledger describes that, without a floor.
- Passing Y2 closes the account up to the head: encoding read plus MLP correction equals what the head reads linearly.
  It says nothing new about the head's attention modulation (Experiment 009's P3 − P1) or the behavioral contrast.
- Twenty-four tokens give a coarse Spearman; the floors are those of Experiments 009 and 011 so the tests are
  comparable. Nothing generalizes beyond the pinned checkpoint, the three templates, and the tokens tested.

## Minimal implementation boundary

A module `layer_correction.py` reusing `read_assembly.py` (pool, functional, attribution ledger, neuron
concentration), `encoding_read.py` (confirmation and lock patterns, scoring, validity), `head_transport.py`,
`cue_suppression.py`, and `plural_mechanism.py`; a `LayerWeights` dataclass holding blocks 1–2's MLP LayerNorm and MLP
weights; capture of `RESID_PRE.L1` and `RESID_PRE.L2` at the cue position added to the reference and patched runs
(existing capture kinds); committed extracts of Experiments 010's and 011's per-component fractions; the confirmation
builder with the frozen lists and frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`,
`confirm`, `report`. Tests: the model's `Δ̂₁` equals the fake's measured block-1 MLP change with the own base; the
neuron-sum identity; the lock's predictions reproduce from the weights and locked states; every floor branch on
synthetic tables; the confirmation policy; phase isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`.

## Revision history

- **Revision 1**: initial draft.

# Experiment 013: What Layers 1–2's Attention Adds — Does Frozen-Pattern Value Transport Predict the Residual of the Token-Local Model? — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 1 — draft for review. No Experiment 013 directory, confirmation set, lock, or model run exists.
Experiments 005–012 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter, residual-predicting. Experiment 012's frozen token-local MLP model is taken as
given; the quantity under test is the **residual it left**, and the candidate for that residual is the part of the
computation Experiment 012 deliberately omitted: the frame's own base state and the layer-1–2 attention paths at the
cue position, modelled with **attention patterns held at the reference** (only the cue position's *values* change).
Nothing is fitted; nothing about the 012 model is changed. Predictions for new cue tokens in exposed frames are
committed before any fresh prompt runs (Y1); predictions for new tokens in new frames are frame-conditional — computed
from each new frame's reference prompt and digested into the results state before any fresh cue enters the network
(Y2); the attribution between the MLP path and the direct head path is tested at the pair level (Y3).

## Purpose and question

Experiment 012 confirmed prospectively that the two MLPs of blocks 1–2, evaluated on `ΔE` with attention held fixed
and template-mean base states, predict most of the net layers-1–2 correction (Spearman 0.864, R² 0.795 on 24 new
tokens × 6 new frames). Its own-base ladder measured what it omitted: a base-point term (the frame's own state against
the template mean; mean −0.037, |.| 0.047), an attention-input term (block 1's attention change reaching block 2's
MLP; +0.051, |.| 0.065) and the heads' direct writes (−0.013; gross share 0.23 of the MLP part). The omissions were
visible: the pair level was looser (R² 0.65), one sign was missed (`which`), and `thy`'s net correction was right
while its attribution was wrong (measured MLP part +0.02 and heads −0.10, against a token-local MLP value of −0.11 at
the frame's own base).

Under the E-patch only the cue position's residual changes at the input of block 1 (`Δx₁(p_c) = ΔE` exactly; earlier
positions are untouched). For a head `h` of layer `l` at query `p_c`, the exact output change therefore splits as in
Experiment 009's OV identity:

```text
Δout_h(p_c) = A_h^ref(p_c, p_c) · Δv_h(p_c) W_O^h        the value path, patterns held at the reference
            + Σ_k ΔA_h(p_c, k) · v_h^patch(k) W_O^h       the pattern-change path
```

where `Δv_h(p_c) = [ln1_l(x_l(p_c) + Δ_l) − ln1_l(x_l(p_c))] W_V^h` needs only the frame's reference residual at
`p_c`, the reference pattern weight the head puts on the cue position, and the change `Δ_l` arriving at that layer.
The first term is computable from the reference prompt alone; the second needs the patched run.

> Is the residual that Experiment 012's token-local model leaves — on cue words and frames the model has never been
> measured on — predicted by the omitted computation modelled with attention patterns held at the reference: the
> frame's own base state, the frozen-pattern value path of the sixteen heads of layers 1–2 feeding block 2's MLP and
> writing directly onto the read direction? And is the attribution between the MLP path and the heads' direct path
> predicted at the same time, including compensation cases like `thy`?

**Design check on exposed data (outside any results state; to be recomputed inside Tier A).** With Experiment 012's
recorded pairs (87 × 30 exposed and 24 × 6 confirmed) and the 36 frames' reference prompts re-captured: the
frozen-pattern model at the frame's own state predicts the *residual* `c_L − ĉ_012` per token with Spearman 0.947 and
explained variance 0.86 on the 87 exposed tokens (residual spread sd 0.060) and 0.948 / 0.87 on the 24 confirmed
tokens (sd 0.073); the base-point term alone explains 0.35 / 0.38 of it. The full model reaches Spearman 0.984 /
0.990 and R² 0.976 / 0.973 against the measured `c_L` (against Experiment 012's 0.944 / 0.864 and 0.773 / 0.795). The
attribution is predicted: the MLP part with R² 0.98 / 0.98, the heads' direct part with 0.88 / 0.64 (spread sd 0.05),
the attention-input term with 0.75 / 0.74; every one of the sixteen heads' direct terms has pair-level Spearman
0.64–0.95. `thy`'s compensation is reproduced without any fitting: measured MLP +0.02 / heads −0.10, predicted −0.01 /
−0.07, frame by frame (quantifier frames measured (+0.34, −0.28) against predicted (+0.32, −0.20)). The pattern-change
path is what remains (about 0.02 per token mean). Experiment 013 tests the residual prediction and the attribution
prospectively.

## The quantities under test (frozen definitions)

Units as in Experiment 012's Y1: every quantity is a ratio of inner reads at the cue position with the weight-only
denominator `r(E(pl_T) − E(ref_T))`; `r`, `d̂_T`, `d̂_E`, `γ₃ ⊙ m`, `σ_T` are Experiment 011's locked values; `ĉ_012(w, T)` is
Experiment 012's locked model — its equations, its all-frames base states `x̄₁,T`, `x̄₂,T` (means over the 30 frames
exposed at that time) and its read — applied to any token, unchanged.

**Measured (exact; Experiment 010's identity with the cue position's captured residuals and patterns):**

```text
c_L(w, f) = Σ_k r(Δout_k) / r(E(pl_T) − E(ref_T))                  net layer change, heads included (as in 012)
c_M, c_H  = the MLP part / the heads' direct part                  (c_L = c_M + c_H)
r(w, f)   = c_L(w, f) − ĉ_012(w, T)                                 the residual of the frozen 012 model (Y1/Y2 target)
```

**Predicted (weights, the locked axes, the frozen 012 model, and one reference state per frame — no patched run):**
with `x₁, x₂` the cue position's reference residuals before blocks 1 and 2 in frame `f`, `A_h^ref(p_c, p_c)` the
reference pattern weight of head `h` on the cue position, and `MLP_l`, `ln1_l`, `ln2_l` block `l`'s weights:

```text
Level 1 (own base):   Δ̂₁ = MLP₁(ln2₁(x₁ + ΔE)) − MLP₁(ln2₁(x₁))                                (exact by the architecture)
Frozen-pattern heads: Δ̂h = A_h^ref(p_c, p_c) · [ln1_l(x_l + Δ_l) − ln1_l(x_l)] W_V^h W_O^h      for the eight heads of layer l
                      layer 1 with Δ_1 = ΔE;  layer 2 with Δ_2 = ΔE + Δ̂₁ + Σ_{h ∈ layer 1} Δ̂h
Level 2 (frozen-pattern attention):
                      Δ̂₂ = MLP₂(ln2₂(x₂ + Δ_2)) − MLP₂(ln2₂(x₂))
                      ĉ_M(w, f) = [r(Δ̂₁) + r(Δ̂₂)] / r(E(pl_T) − E(ref_T))                          predicted MLP part
                      ĉ_H(w, f) = Σ_{16 heads} r(Δ̂h) / r(E(pl_T) − E(ref_T))                        predicted direct head part
                      ĉ_L(w, f) = ĉ_M + ĉ_H
r̂(w, f)   = ĉ_L(w, f) − ĉ_012(w, T)                                                             the predicted residual
```

The ladder is exact by construction: `c_L − ĉ_012 = [ĉ_own − ĉ_012] + [ĉ_L − ĉ_own] + [c_L − ĉ_L]` — the base-point
term (Experiment 012's Level 1 minus Level 0), the frozen-pattern attention term, and the pattern-change remainder.
The remainder is further split exactly from the patched run's captures: the heads' direct pattern-change part
`Σ_h Σ_k ΔA_h(p_c, k) v_h^patch(k) W_O^h` (Experiment 009's identity, checked per head against the captured head
outputs, an incident above `1e-4` relative) and the indirect part through block 2's MLP (`c_M − ĉ_M`). Every term is
recorded per pair. Denominator validity is inherited from Experiment 012's lock (all three templates defined).

**Compensation cases (frozen rule, descriptive):** a (token, frame) pair is a *compensation case* if `ĉ_M` and `ĉ_H`
have opposite signs and each exceeds 0.05 in magnitude; the report states, over the fresh compensation cases, the
fraction whose measured `c_M` and `c_H` carry the predicted signs, and lists them. `thy` in Experiment 012's frames is
the exposed example (reported at Tier A); no fresh case is named in advance.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, P1 and
its OV identity checks, the exact attribution functional and its identity checks, the Experiment 011 locked axes and
read weight, the Experiment 012 locked model (equations, base states, read) and its lock digest, the results-state,
ledger, lock, prediction-artifact, and incident conventions — all exactly as in Experiments 010–012. Seeds: runtime
`20260916`, control `20260924`.

## Exposed pool (calibration only; nothing is fitted)

The 111 exposed cue tokens (Experiment 012's 87 and its 24 confirmed) and the 36 exposed frames (Experiment 012's 30
and its 6), with the 80 nouns; the 2724 recorded (token, frame) pairs — Experiment 012's 87 × 30 and 24 × 6 — as a
committed extract of its per-pair ledger (`c_L`, `c_M`, `c_H`, `c_k`, `ĉ_own`, and the locked or leave-one-frame-out
`ĉ_012`). `explore`:

- re-captures the 36 reference prompts with the residuals at every position `≤ p_c` before blocks 1 and 2 and the
  layer-1–2 attention pattern rows at `p_c`, and locks the 36 frames' reference states (`x₁`, `x₂`, and the sixteen
  `A_h^ref(p_c, p_c)`);
- re-measures the 2724 recorded pairs with the patched run's residuals and pattern rows at `p_c` captured; replicates
  Experiment 012's extract within `1e-6` for every recorded field; checks the per-head OV identity in every pair;
- evaluates Levels 1 and 2 and the pattern-change split for every pair; fixes the tolerances from the exposed
  residual-prediction errors at the aggregation level of each floor: `τ_r = max(0.05, 3 × RMSE over the 111 tokens of
  (r̄̂(w) − r̄(w)))` (token means over the token's recorded frames), and `τ_A = max(0.05, 3 × RMSE over the 2724 pairs
  of (ĉ_H − c_H))` for the pair-level attribution. The lower floor (0.05, against 0.10 in Experiments 011–012) follows
  the target: the residual's spread (sd ≈ 0.06–0.07 per token) is half that of `c_L`. No leave-one-frame-out is needed:
  the model uses each frame's own state, not a template mean;
- records the exposed descriptive statistics (Spearman, MAE, R², bias of `r̂` against `r`, of `ĉ_L` against `c_L`, of
  `ĉ_M`/`ĉ_H` against `c_M`/`c_H`, per head, per template; the size of the pattern-change remainder and its direct /
  indirect split; the compensation cases; `thy`).

## Confirmation set (frozen by tokenizer rules before any Experiment 013 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 111 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists, with quotas; classes for coverage and
  reporting only, **no class carries an expectation**.
  - `determiner-like` (quota 5): `same`, `own`, `last`, `next`, `first`, `second`.
  - `numeral` (quota 5): `zero`, `thousand`, `million`, `billion`, `trillion`, `dozens`, `hundreds`, `thousands`,
    `millions`.
  - `quantity` (quota 5): `limited`, `surplus`, `endless`, `plenty`, `vast`, `minimal`, `infinite`, `excess`, `lesser`,
    `considerable`, `insufficient`.
  - `possessive-or-pronoun` (quota 4): `whom`, `someone`, `nobody`, `everyone`, `anybody`, `somebody`, `everybody`,
    `anyone`.
  - `adjective` (quota 5): `tall`, `thick`, `quiet`, `broken`, `wide`, `narrow`, `sharp`, `smooth`, `rough`, `golden`.
  Every fresh cue is licensed in every frame it is run in.
- **Fresh frames (6, two per template), literal:** cardinal `The pantry stocks {cue}`, `The courier delivers {cue}`;
  quantifier `The audit examines {cue}`, `The podcast discusses {cue}`; coordinated-adjective
  `Mateo and Ines washed {cue} clean`, `Sofia and Anders bought {cue} loose`. Texts must differ from every exposed
  frame text; the original cue tokens are used for their own cue prompts.
- **Two fresh sets, both executed only by `confirm`:** (a) the 24 fresh tokens in the **36 exposed frames** (Y1, 864
  pairs; the frames' reference states are exposed and locked at Tier A, so every prediction is in the lock); (b) the
  24 fresh tokens in the **6 fresh frames** (Y2, 144 pairs; the predictions need each fresh frame's reference state).
- The set (tokens with classes, frames, both prompt lists, digest) is committed as `confirmation-v1.json` before
  `explore`; `confirm` refuses if any fresh prompt appears in the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, the frozen 012 model, and the 36 locked reference states — no forward pass on
  any fresh prompt):** for every fresh token: `ĉ_012(w, T)` per template; and for every fresh token × exposed frame:
  `ĉ_M`, `ĉ_H`, `ĉ_L`, `r̂`, the sixteen heads' frozen-pattern terms; the token means over the 36 frames; the numeric
  `τ_r`, `τ_A`; the exposed descriptive statistics; `predictions.md` with the token means and the per-template values.
  The lock phase has no access to a capture or intervention API. `confirm` recomputes every locked prediction before
  any fresh prompt runs and refuses on any difference above `1e-9`.
- **Tier C (`confirm`, once), in two stages with a digest between them:**
  - *Stage 1 (reference states of the fresh frames):* for each fresh frame, the reference run only (reference cue;
    every token in it is exposed) with the residuals at positions `≤ p_c` and the pattern rows captured; the plural
    cue's E-patch and the sg/pl cue prompts (validity, as Experiment 011–012: plural head change `≥ 0.25 σ_T`, cue
    effect `≥ exact_count_floor(108/120, 79)`); then the **frame-conditional predictions** `ĉ_M`, `ĉ_H`, `ĉ_L`, `r̂` for
    every fresh token in that frame, written to the results state and digested **before any fresh cue prompt runs**.
    No fresh cue token has entered the network at the end of stage 1.
  - *Stage 2 (fresh cues):* every fresh token's E-patch in every valid frame of both sets, with the head's internals,
    the residuals and pattern rows at `p_c` captured; the exact ledger (`c_L`, `c_M`, `c_H`, `c_k`, P1, `q_T`), the
    ladder terms, the per-head OV identity, the compensation cases; scoring.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 011–012 for the fresh frames; the exposed frames are valid by
  construction (all 36 were informative in Experiment 012). A token is scored in a set iff it has at least three valid
  frames there; `F(w)` is exactly those frames for every mean. `PRECONDITION_FAILED` for Y2 if fewer than four fresh
  frames are valid or fewer than sixteen tokens are scored; Y1 has no frame precondition (36 exposed frames) and needs
  sixteen scored tokens.
- **Y1 — residual prediction, strict boundary (new tokens × exposed frames; numbers committed before `confirm`):**
  over the scored tokens, `r̄̂(w) = mean_{f ∈ F(w)} r̂(w, f)` versus `r̄(w) = mean_{f ∈ F(w)} r(w, f)`: **Spearman ≥ 0.80**,
  **MAE ≤ τ_r**, **R² ≥ 0.50**. Pass → `RESIDUAL_PREDICTED_TOKENS`; fail → `RESIDUAL_NOT_PREDICTED_TOKENS` (naming the
  floor). The frozen 012 model's own error on the same tokens (`ĉ_012` against `c_L`) and the full model's (`ĉ_L`
  against `c_L`) are reported beside it, never judged.
- **Y2 — residual prediction, frame-conditional (new tokens × new frames; numbers digested at stage 1):** the same
  three floors on the same token-mean comparison over the fresh frames. Pass → `RESIDUAL_PREDICTED_FRAMES`; fail →
  `RESIDUAL_NOT_PREDICTED_FRAMES`.
- **Y3 — attribution (pairs of both sets):** the predicted direct head part against the measured one, `ĉ_H(w, f)`
  versus `c_H(w, f)` over all scored (token, valid frame) pairs: **Spearman ≥ 0.70 and MAE ≤ τ_A**; and the predicted
  MLP part `ĉ_M` versus `c_M`: **Spearman ≥ 0.90**. Pass → `ATTRIBUTION_PREDICTED`; fail → `ATTRIBUTION_NOT_PREDICTED`.
  The exposed pair-level values are 0.86 and 0.99 (0.81 and 0.98 on the confirmed 012 set).
- **Descriptive (no floor):** the ladder — base-point, frozen-pattern attention, pattern-change remainder with its
  direct and indirect split — per token and in the mean; per-head agreement; the compensation cases and their measured
  signs; `thy` at Tier A; class means; the composite in the head's units (`g_E + ĉ_L` in Experiment 012's Y2 form)
  against the measured P1 fraction, reported for continuity.
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities, software defects) stop the phase, are recorded with
  their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Y2's predictions are not committed before every fresh prompt: they are committed before every fresh *cue* prompt,
  from the fresh frame's reference run (exposed tokens only). Y1 carries the strict boundary for the token dimension;
  the design never describes Y2 as more than frame-conditional.
- Passing Y1–Y3 shows that what Experiment 012 omitted is, up to the pattern-change remainder, value transport by the
  layer-1–2 heads with unchanged attention patterns plus the frame's own base state — and that the split between the
  MLP path and the heads' direct path is predicted, including compensation cases. It does not say what the heads'
  patterns compute (they are taken from the reference run), what the block-2 neurons detect, or anything about the
  behavioral contrast. Nothing generalizes beyond the pinned checkpoint, the three templates, and the tokens tested.
- The residual is a small quantity (sd ≈ 0.06–0.07 per token); the R² floor and the Spearman floor carry the test and
  the MAE ceiling is derived at that scale.

## Minimal implementation boundary

A module `attention_paths.py` reusing `layer_correction.py` (the 012 model, its lock loader, pair measurement,
scoring patterns), `head_transport.py` (`HeadWeights` for the sixteen heads, the OV identity), `read_assembly.py`,
`encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; a committed extract of Experiment 012's per-pair
ledger; the confirmation builder with the frozen lists, frames and the two prompt lists; a runner with phases
`validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, with `confirm`'s two stages and the
inter-stage digest. Tests: the frozen-pattern head term equals the captured head output change when the pattern is
unchanged (fake); the OV identity split; the ladder sums exactly; the lock's predictions reproduce; every floor branch
on synthetic tables; stage-1 predictions are digested before any stage-2 prompt; phase isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and no
fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1**: initial draft.

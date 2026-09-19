# Experiment 015: The Cue-Induced Attention-Pattern Change from the Decoded Query/Key Computation — A Token-Local Cue-Change Q/K Model, Conditional on the Reference Frame State, Locked Before Execution — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 2 — approved conceptually at revision 1 subject to the changes under "Revision history" (Level 1
kept apart from the scientific result as an identity check; the model named as a token-local *cue-change* model
conditional on the reference frame state; `c_ΔA` frozen to the last index; the Q/K prediction of `ΔA` separated from
the V/O/readout used to score its consequence; Y2 kept frame-conditional; the aggregate character of Y2 stated and the
six per-frame results reported regardless; the axis-only alternative frozen in every detail; the diagonal-proportional
comparator kept as a named descriptive competitor with a predeclared interpretation rule), which this revision makes.
No Experiment 015 directory, confirmation set, lock, or model run exists. Experiments 005–014 are closed and are not
amended by this document.

**Kind:** Prospective, zero-parameter, program-completing. Experiment 013 wrote the layer-1–2 attention contribution
at the cue position as an exact two-term split per head — a frozen-pattern value path it predicted, and a
pattern-change term `Σ_k ΔA_h(p_c,k) v_h'(k) W_O^h` it *measured* from the patched run. Experiment 014 inherited the
same frozen-pattern state. This experiment writes the pattern change itself as a program — LayerNorm → query and key
projections → rotary rotation → scaled dot products → softmax — driven by a reduced, token-local description of the
cue change instead of the patched run, commits its predictions for unseen cues and frames before any of them runs,
and tests a rigid number-axis alternative on the same fresh data. Nothing is fitted; no grammatical label is attached
to any head or cue before scoring.

## Purpose and question

The pattern-change term is small in read units but it is the one piece of the cue-position computation through
layers 1–2 that Experiments 012–014 never predicted: on Experiment 013's exposed ledger it has standard deviation
0.036 of the plural cue's signal (against 0.217 for the whole layer contribution `c_L` and 0.068 for the heads' part
`c_H`), and it *is* the heads' misprediction — its correlation with `c_H − ĉ_H` over the 2724 exposed pairs is 0.986.
In pattern space the change is not small at all: a cue swap moves the self-attention weight `A_h(p_c,p_c)` of single
heads by up to 0.83, and the total variation of the eight cue rows summed over heads averages 1.66 at layer 1 and
1.06 at layer 2 per pair.

**Two levels, kept apart.**

```text
Level 1 — identity, not a claim
  the full weight-defined Q/K/V recomputation from the frame's actual reference state and ΔE
  → must reproduce the captured patterns and residuals within the frozen numerical tolerance
  → a failure is an implementation incident that stops the phase; a success counts toward nothing

Level 0 — the hypothesis
  the Experiment 012 template-mean base + ΔE of the new cue + the reduced Q/K computation,
  applied to the frame's observed reference attention state
  → predicts the cue-induced change of the attention pattern
```

*Why Level 1 is exact.* The E-patch acts only at the cue position, and block 0's attention reads only the embeddings,
so with the parallel residual `x₁'(p_c) = x₁(p_c) + ΔE` exactly, while every position `k < p_c` is unchanged in every
layer (causal masking). The cue row of every layer-1 head, the layer-1 head outputs at `p_c`, the residual before
block 2 at `p_c`, and the cue row of every layer-2 head are therefore exact functions of the weights, the frame's
*reference* residuals at positions `≤ p_c`, and `ΔE`. Recomputing them with the weight-only program below reproduces
the model to float32 precision on the whole exposed pool (5628 re-executed pairs: cue rows to `9.8e-6`, the residual
before block 2 to `1.4e-5`, the per-head output split to `1.7e-5` relative). Level 1 closes Experiment 013's
accounting identically (its remainder becomes a computed term); it proves only that the program is implemented
correctly (rotary convention, scale, masking), and it never enters Y1–Y3.

*Why the claim is Level 0, and what "token-local" means here.* Experiment 012 reduced the MLP corrections to a
token-local model evaluated at locked template-mean base states. The same move applied to the query/key computation
gives a **token-local cue-change model conditional on the reference frame state**: the cue position's query, key and
value *changes* are computed at the Experiment 012 locked template-mean bases from `ΔE` alone — token-only given the
template — while the frame contributes its reference run and nothing else: its reference keys, values and logit rows
at positions `≤ p_c`. Level 0 does not predict the attention pattern from the token alone; it predicts the *change*
the cue swap induces, given the frame's observed reference attention state. Rotary position embeddings cooperate: at
one position the query and the key receive the same rotation `R_p`, and `(R_p q)ᵀ(R_p k) = qᵀ R_pᵀ R_p k = qᵀ k` because
the rotation is orthogonal — so the cue's self-logit change is not caused by position and is exactly token-local at
the base; only the off-diagonal logit changes couple the token-local query change to the frame's rotated keys.

**Design check on exposed data (outside any results state; every exposed token in every exposed frame, 5628 patched
runs re-executed; to be recomputed inside Tier A).**

- **Level 0.** Over the 135 token means of the predicted pattern-change read `ĉ_ΔA` (defined below) against the
  measured `c_ΔA`: Spearman 0.975, R² 0.869, MAE 0.0062 (measured spread 0.021); per pair 0.899 / 0.792. In pattern
  space, pooling every entry `ΔÂ_h(p_c,k)` against `ΔA_h(p_c,k)` over the pairs: R² 0.917 at layer 1 and 0.713 at
  layer 2; the total-variation error is 24% (layer 1) and 46% (layer 2) of the measured total variation; the sixteen
  self-weight changes are reproduced with pooled R² 0.889 / 0.664. Per template (14 frames each) the layer-2 entry R²
  is 0.672–0.763; per frame it ranges 0.387–0.910 (median 0.732) at layer 2 and 0.633–0.983 (median 0.930) at layer 1
  — Level 0 is an aggregate description, not a per-frame guarantee.
- **The number-axis alternative, given the frame's own reference state and the exact arriving change (its best
  chance):** token-mean `ĉ_ΔA` R² −2.49 (Spearman 0.30); pooled entry R² 0.186 at layer 1, 0.041 at layer 2, 0.138
  over both.
- **The diagonal-proportional comparators (descriptive, named).** *Oracle*: the exact self weight of each head with the
  other keys rescaled in proportion to the reference row — pooled entry R² 0.765 / 0.793, token-mean `ĉ_ΔA` R² 0.474:
  most, but not all, of the row change is a change of self-attention. *Level-0 diagonal-proportional*: the same
  rescaling applied to the **Level-0 predicted** self weight, i.e. Level 0 without its off-diagonal coupling — pooled
  entry R² 0.685 / 0.523, token-mean R² 0.653. The full Level 0 exceeds it by 0.23 (layer 1) and 0.19 (layer 2) of
  entry R² on the exposed pool.
- **Other rungs, each at the frame's own state with the exact arriving change:** *query-only* (self key frozen):
  entry R² 0.23 / −0.25; *self-key-only*: 0.25 / −0.70; *additive* (both sides moved, the interaction `⟨Δq, Δk⟩`
  dropped from the diagonal): 0.26 / −0.74; *LayerNorm linearised* (first-order Jacobian, exact softmax): −0.57 /
  −1.47 — worse than predicting no change, because the cue swap is a replacement, not a perturbation: `‖Δx‖ / ‖x −
  μ(x)‖` averages 0.95 at layer 1 and 0.81 at layer 2. *Frozen-pattern arrival* (layer-2 rows from Experiment 013's
  arriving change, i.e. with the layer-1 pattern change omitted): 0.853 at layer 2 — the layer-1 pattern change does
  reach the layer-2 patterns.
- **Diagonal decomposition (descriptive, 303-pair subset).** The change of the self logit is dominated by the query
  side (`⟨Δq, k_ref⟩` rms 4.5 logits at layer 1, 7.0 at layer 2) with the key side (1.1 / 1.4) and the interaction
  (1.2 / 2.1) each large enough that dropping either breaks the softmax row; no partial evaluation of the bilinear
  form survives.

> Given a frame's reference attention state, can the cue-induced change of the layer-1 and layer-2 attention
> patterns at the cue position be predicted prospectively — for cue words and frames never measured on it — from a
> reduced, token-local query/key computation driven by the cue's encoding change `ΔE` alone at locked template-mean
> bases, and does the predicted change account for the pattern-change term Experiment 013 left as a measured
> remainder? Is the number-axis account of the same change rejected on the same fresh data?

## The quantities under test (frozen definitions)

Notation as in Experiments 012–014. Layers `ℓ ∈ {1, 2}`; heads `h ∈ {0..7}`; **the sixteen heads** are
`H = {L01.H00 … L01.H07, L02.H00 … L02.H07}`; `x_ℓ(k)` the frame's reference residual before block `ℓ` at position `k ≤
p_c`; `x_ℓ'(k)` the patched run's; `ΔE = E(w) − E(ref_T)` the weight-only encoding difference; `x̄_ℓ^T` the Experiment
012 locked template-mean bases; `d̂_E` the locked number direction (the Experiment 011 lock's `axes_vectors.R0`, unit
norm, in residual coordinates); `r(·)` the locked weight-only read `⟨x − mean(x), γ₃ ⊙ m⟩`; `D_T = r(E(pl_T) − E(ref_T))`
the template's denominator (1.6299 cardinal and coordinated-adjective, 1.4961 quantifier).

**The weight-only query/key/value program (frozen; every constant is the pinned checkpoint's).** With `LN_ℓ` block
`ℓ`'s first LayerNorm (`γ`, `β`, `ε = 1e-5`, population variance, exactly `plural_mechanism.exact_layer_norm`):

```text
n_ℓ(x)      = LN_ℓ(x)                                                  the normalized residual
q̃_h(x)      = n_ℓ(x) W_Q^h + b_Q^h,   k̃_h(x) = n_ℓ(x) W_K^h + b_K^h,   v_h(x) = n_ℓ(x) W_V^h + b_V^h     (64-dimensional)
R_p         = the rotary rotation at position p on the first 16 of the 64 head coordinates: for i = 0..7 the pair
              (i, i+8) is rotated by θ_i(p) = p · 10000^(−i/8) — [x_i, x_{i+8}] → [x_i cos θ − x_{i+8} sin θ, x_i sin θ + x_{i+8} cos θ];
              coordinates 16..63 pass through (partial_rotary_factor 0.25, HuggingFace half-split convention)
q_h(x, p)   = R_p q̃_h(x),   k_h(x, p) = R_p k̃_h(x)
s_h(p_c, k) = ⟨q_h(x_ℓ(p_c), p_c), k_h(x_ℓ(k), k)⟩ / 8                 the logits of the cue row, k ≤ p_c (√64 = 8)
A_h(p_c, ·) = softmax_{k ≤ p_c} s_h(p_c, ·)                             causal; the prompts carry no BOS token
out_h(p_c)  = Σ_{k ≤ p_c} A_h(p_c, k) v_h(x_ℓ(k)) W_O^h                  the head's output at the cue position
```

**Level 1 — identities (checks, never claims).** *I1 (incident above `1e-4` absolute):* the recomputed reference rows
equal the captured `ATTN_PATTERN.L{1,2}` rows at `p_c` of the reference run, and the recomputed patched rows from the
captured patched residuals equal the captured patched rows. *I2 (incident above `1e-4` relative):* `x₂(p_c) + ΔE +
[MLP₁(ln2₁(x₁(p_c) + ΔE)) − MLP₁(ln2₁(x₁(p_c)))] + Δout₁(p_c)`, with `Δout₁` the eight-head layer-1 output change
recomputed from the program on `x₁(p_c) + ΔE` against the frame's reference keys and values, equals the captured
patched residual before block 2. *I3 (incident above `1e-4` relative):* Experiment 013's per-head split (`frozen +
pattern_change = measured`), inherited. A violated identity stops the phase as an implementation incident; a
satisfied one is recorded and counts toward no floor.

**Measured (exact, from the captured rows and residuals of the reference and patched runs) — frozen to the last
index:**

```text
ΔA_h(p_c, k) = A_h'(p_c, k) − A_h(p_c, k)          for every h ∈ H and every key position k ∈ {0, …, p_c}; row p_c only
                                                   (A' the patched run's captured row, A the reference run's captured row)
v_h'(k)      = v_h(x_ℓ(k)) for k < p_c              the reference values (positions < p_c are unchanged)
v_h'(p_c)    = v_h(x_ℓ'(p_c))                       the patched cue value, from the captured patched residual x_ℓ'(p_c)
P_h(w, f)    = Σ_{k=0}^{p_c} ΔA_h(p_c, k) v_h'(k) W_O^h        Experiment 013's pattern-change term of head h (a 512-vector)
c_ΔA(w, f)   = Σ_{h ∈ H} r(P_h(w, f)) / D_T                  the sixteen heads' terms read on the locked read weight and summed,
                                                             divided by the template's denominator — one number per pair;
                                                             equal to Experiment 013's `remainder_direct_pattern_change`
c_ΔA,ℓ(w, f) = Σ_{h in layer ℓ} r(P_h) / D_T                  the per-layer parts (descriptive); c_ΔA = c_ΔA,1 + c_ΔA,2
c̄_ΔA(w)      = mean over the token's scored frames of c_ΔA(w, f)   the token mean, over the same frame set as the prediction
```

**Predicted, Level 0 — the token-local cue-change model conditional on the reference frame state (weights, the
locked axes, read and bases, the frame's reference run; no patched quantity).** Two parts, kept apart: the Q/K part
predicts the pattern change; the V/O/readout part turns the predicted change into its consequence.

*(a) Q/K part — predicts `ΔÂ`.* At layer 1 the state change is `Δ̂x₁ = ΔE`; at layer 2 it is the decoded arriving
change `Δ̂x₂` defined in (c). For each layer, with `x̄ = x̄_ℓ^T` and `Δ̂x = Δ̂x_ℓ`:

```text
Δq̂_h  = q̃_h(x̄ + Δ̂x) − q̃_h(x̄),   Δk̂_h = k̃_h(x̄ + Δ̂x) − k̃_h(x̄)                token-local changes at the template-mean base
Δŝ_h(p_c, k)   = ⟨R_{p_c} Δq̂_h, k_h(x_ℓ(k), k)⟩ / 8                     for k < p_c: the token-local query change against the frame's reference keys
Δŝ_h(p_c, p_c) = [⟨q̃_h(x̄ + Δ̂x), k̃_h(x̄ + Δ̂x)⟩ − ⟨q̃_h(x̄), k̃_h(x̄)⟩] / 8   rotation-free (same-position lemma), fully token-local
Â_h(p_c, ·)    = softmax_{k ≤ p_c} [ s_h(p_c, ·) + Δŝ_h(p_c, ·) ]        the frame's reference logit row moved by the token-local changes
ΔÂ_h(p_c, k)   = Â_h(p_c, k) − A_h(p_c, k)                              the Y1/Y2 pattern-space target
```

*(b) V/O/readout part — scores the consequence of the predicted `ΔÂ`; it does not determine `ΔÂ`.* With `Δv̂_h = v_h(x̄
+ Δ̂x) − v_h(x̄)` the token-local value change at the base:

```text
v̂_h(k)    = v_h(x_ℓ(k)) for k < p_c;   v̂_h(p_c) = v_h(x_ℓ(p_c)) + Δv̂_h
P̂_h(w, f) = Σ_{k=0}^{p_c} ΔÂ_h(p_c, k) v̂_h(k) W_O^h;    ĉ_ΔA(w, f) = Σ_{h ∈ H} r(P̂_h) / D_T;    ĉ̄_ΔA(w) as for c̄_ΔA
```

*(c) The decoded arriving change at block 2* — Experiment 012's Level-0 MLP term plus the layer-1 attention output
change assembled from (a) and (b) (the frozen-pattern value path *and* the predicted pattern change), so that no term
of it is supplied from a patched run and Experiment 013's frozen pattern is no longer used. The layer-2 Q/K
prediction depends on this input; the dependence is on the layer-1 V/O consequence, not on the Q/K program itself:

```text
Δ̂out₁(p_c) = Σ_{h ∈ layer 1} [ Σ_k Â_h(p_c, k) v̂_h(k) − Σ_k A_h(p_c, k) v_h(x₁(k)) ] W_O^h
Δ̂x₂        = ΔE + [MLP₁(ln2₁(x̄₁^T + ΔE)) − MLP₁(ln2₁(x̄₁^T))] + Δ̂out₁(p_c)
```

**The alternative, committed beside the predictor and defined as rigidly — the number-axis-only program (Y3).**
Frozen in every detail now: the axis is the Experiment 011 lock's `R0` vector `d̂_E` (unit norm, residual
coordinates); the projection is `Δx_ℓ,axis = ⟨Δx_ℓ, d̂_E⟩ d̂_E` in residual coordinates before the LayerNorm, with no
scale, intercept or calibration of any kind; the reference treatment is the frame's *own* state `x_ℓ(p_c)` (not the
template base) with the *exact* Level-1 arriving change `Δx₁ = ΔE`, `Δx₂ = x₂'(p_c) − x₂(p_c)` as recomputed by
Level 1 — the alternative is thus given more than the predictor (the exact change rather than a reduced one); the
cue's query, self key and value are taken at `x_ℓ(p_c) + Δx_ℓ,axis`, the other keys and values are the frame's
reference ones, and the LayerNorm, projections, rotation, scale and softmax are those of the program:

```text
Â_h^axis(p_c, k)   = softmax_k s_h^axis,   s_h^axis(p_c, k) = ⟨q_h(x_ℓ(p_c) + Δx_ℓ,axis, p_c), k_h(x_ℓ(k), k)⟩ / 8 for k < p_c,
                                          s_h^axis(p_c, p_c) = ⟨q̃_h(x_ℓ(p_c) + Δx_ℓ,axis), k̃_h(x_ℓ(p_c) + Δx_ℓ,axis)⟩ / 8
ΔÂ^axis, v̂^axis(p_c) = v_h(x_ℓ(p_c) + Δx_ℓ,axis), P̂^axis, ĉ_ΔA^axis, ĉ̄_ΔA^axis as in (a)–(b)
```

Nothing about the alternative may be adjusted after the fresh data.

**The diagonal-proportional comparators (descriptive, named; frozen definitions).** For a given predicted or exact
self weight `a_h` of head `h`, the *proportional* row is `Ã_h(p_c, p_c) = a_h` and `Ã_h(p_c, k) = A_h(p_c, k) · (1 −
a_h) / (1 − A_h(p_c, p_c))` for `k < p_c` — the self weight changes, the remaining mass is redistributed in the
proportions of the reference row (no off-diagonal query·key coupling). *Oracle diagonal-proportional* uses the
Level-1 (exact) self weight: it measures the share of the row change that is a change of self-attention. *Level-0
diagonal-proportional* uses the Level-0 predicted self weight `Â_h(p_c, p_c)` from (a): it is Level 0 stripped of its
off-diagonal term `⟨R_{p_c} Δq̂_h, k_h(x_ℓ(k), k)⟩`, from the same inputs. Both are scored with the same statistics as
Level 0 (pooled entry R² per layer, total-variation ratio, token-mean `ĉ_ΔA` with V/O part (b)). **Predeclared
interpretation rule (no gate):** the report says that Level 0 decodes the query/key interaction *beyond* the self-logit
effect only if its pooled entry R² exceeds the Level-0 diagonal-proportional comparator's by at least 0.10 at both
layers on the fresh pairs of Y1 ∪ Y2; otherwise it uses the narrower wording that most of the cue-induced pattern
change is captured by the self-logit change with proportional redistribution. Exposed margins: 0.23 / 0.19.

**Invariant (formal):** the Level-0 prediction is a function of the token identity, the template's locked bases, the
locked axes and read, the weights, and the frame's *reference* residuals at positions `≤ p_c` — never a residual,
row or value of a patched run. The prediction function takes only those inputs; the test suite asserts that no
capture or intervention entry point is reachable while a prediction table is computed, and `confirm` reproduces the
locked table from the lock's own contents before any fresh prompt. The identities I1–I3, the axis-only alternative's
exact arriving change and the oracle comparator use patched quantities and are checks or comparators, not
predictions.

Every Level-0 quantity for the exposed frames is computable at the lock from the locked reference states (this
experiment locks, per exposed frame, the reference residuals before blocks 1 and 2 at *every* position `≤ p_c`, from
which the keys, values and reference rows follow); for the fresh frames it is computed at `confirm` stage 1 from the
frame's reference run and digested before any fresh cue prompt, as in Experiments 013–014.

**Accounting (descriptive, per pair, exact given the captures):** the Level-1 identities; the comparators and rungs
(oracle and Level-0 diagonal-proportional; query-only, self-key-only, additive, LayerNorm-linearised, frozen-pattern
arrival) each at the frame's own state with the exact arriving change unless stated; the three diagonal terms `⟨Δq,
k_ref⟩`, `⟨q_ref, Δk⟩`, `⟨Δq, Δk⟩` per head; `‖Δx_ℓ‖ / ‖x_ℓ − μ‖`; the total-variation ratio `Σ ½‖ΔÂ_h − ΔA_h‖₁ / Σ
½‖ΔA_h‖₁` per layer; the sixteen self-weight changes against their predictions; `c_ΔA,ℓ` per layer; MAE of `ĉ_ΔA`; the
six fresh frames' individual results; and Experiment 013's ladder re-derived with the pattern term computed — how much
of `c_L` Level 0 accounts for, and that Level 1 accounts for all of it.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked bases and model, the Experiment 013 capture of
reference states and pattern rows, the results-state, ledger, lock, prediction-artifact, two-stage confirmation and
incident conventions — all exactly as in Experiments 012–014. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration record only; nothing is fitted)

The 159 exposed cue tokens (Experiment 014's 135 and its 24 confirmed) and the 48 exposed frames (its 42 and its 6),
with the 80 nouns. `explore`:

- captures the 48 reference states with the residuals before blocks 1 and 2 at every position `≤ p_c` and the
  layer-1–2 pattern rows at `p_c`, checks I1 on each, and locks the states (the cue-position residuals must agree
  with the Experiment 014 lock within `1e-9`, its own tolerance);
- re-measures the E-patch of every recorded pair of Experiment 014's ledger (its 3732 + 1008 + 144 pairs) with the
  patched residuals before blocks 1 and 2 and the patched pattern rows captured; replicates Experiment 014's per-pair
  `c_L`, `c_M`, `c_H` within `1e-6` (a committed extract) and Experiment 013's sixteen self-weight changes and its
  `remainder_direct_pattern_change` within `1e-6` on its 3732 pairs (a committed extract); checks I1–I3 on every pair;
  computes `ΔA`, `P_h`, `c_ΔA` exactly and the Level-0, axis-only and comparator predictions from the locked states;
- records, for the full recorded pool, the exposed statistics of Level 0, of the alternative and of the comparators
  (token means and pairs of `ĉ_ΔA`; pooled entry R² and total-variation ratio per layer; self-weight R²), the rungs,
  the diagonal decomposition, and the re-derived ladder.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 015 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 159 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists, with quotas; classes for coverage and
  reporting only, **no class carries an expectation**.
  - `determiner-like` (quota 5): `fourth`, `fifth`, `particular`, `previous`, `final`, `initial`, `upper`,
    `respective`, `individual`.
  - `ordinal-or-numeral` (quota 5): `sixth`, `seventh`, `eighth`, `ninth`, `tenth`, `twentieth`, `quarter`, `twin`,
    `dual`, `tens`.
  - `quantity` (quota 5): `scant`, `adequate`, `moderate`, `substantial`, `enormous`, `unlimited`, `countable`,
    `spare`, `remaining`, `total`.
  - `possessive-or-pronoun` (quota 4): `himself`, `herself`, `itself`, `ourselves`, `oneself`, `myself`, `yourselves`.
  - `adjective` (quota 5): `plastic`, `frozen`, `ancient`, `modern`, `loud`, `bright`, `orange`, `pink`, `brown`,
    `grey`, `heavy`, `gentle`.
- **Fresh frames (6, two per template), literal:** cardinal `The farmer grows {cue}`, `The printer produces {cue}`;
  quantifier `The clinic treats {cue}`, `The agency recruits {cue}`; coordinated-adjective
  `Emma and Marco gathered {cue} dry`, `Sam and Julia loaded {cue} warm`. Texts must differ from every exposed frame
  text; the original cue tokens are used for their own cue prompts.
- **Two fresh sets, both executed only by `confirm`:** the 24 fresh tokens in the 48 exposed frames (Y1, 1152 pairs;
  every prediction in the lock) and in the 6 fresh frames (Y2, 144 pairs; predictions at stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, read and bases, the 48 locked reference states — no forward pass on any fresh
  prompt):** the complete prediction table for every fresh token × exposed frame — token, frame, template, the full
  predicted rows `ΔÂ_h(p_c, ·)` for the sixteen heads, the sixteen predicted self-weight changes, `ĉ_ΔA` per layer and
  in total, the axis-only rows and `ĉ_ΔA^axis`, the Level-0 diagonal-proportional rows and their `ĉ_ΔA`, the predicted
  arriving change's read `r(Δ̂x₂)/D_T` — and the token means of `ĉ_ΔA`; the frozen floors; the exposed statistics; the
  model revision and the protocol commit; `predictions.md` with the token means and the predicted self-weight changes
  per head averaged over the exposed frames.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiment 013's procedure):**
  *stage 1* — the fresh frames' reference runs (exposed tokens only) with the residuals at every position `≤ p_c`
  and the pattern rows captured, I1, their validity (plural-cue head change `≥ 0.25 σ_T`, cue-pair check), the
  frame-conditional prediction table with the same columns, serialized and digested; *stage 2*, only after the digest
  is re-read from disk — every fresh cue's E-patch in every frame of both sets with the patched residuals and rows
  captured; I1–I3; `ΔA`, `P_h`, `c_ΔA`, the accounting; scoring against the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–014; a token is scored in a set iff it has at least three
  valid frames there; Y2 needs at least four valid fresh frames and sixteen scored tokens; Y1 needs sixteen scored
  tokens.
- **Aggregation, fixed:** the read-unit criteria on **token means** `ĉ̄_ΔA(w)` versus `c̄_ΔA(w)` over identical frame
  sets (at most 24 points each); the pattern-space criteria on **every entry** `ΔÂ_h(p_c, k)` against `ΔA_h(p_c, k)`
  pooled over the scored pairs of the set and the eight heads of a layer (one R² per layer; rows are frame-specific,
  so they are not averaged over frames). No MAE ceiling: the explained-variance floors carry the magnitude test in
  both spaces; MAE and the total-variation ratio are reported descriptively.
- **Y1 — Level 0, strict prospective prediction (fresh cues × exposed frames; numbers committed before `confirm`):**
  token means: **Spearman ≥ 0.80** and **R² ≥ 0.50**; raw entries: **R² ≥ 0.50 at layer 1 and R² ≥ 0.50 at layer 2**,
  separately. All four → `PATTERN_CHANGE_PREDICTED_TOKENS`; otherwise `PATTERN_CHANGE_NOT_PREDICTED_TOKENS` (naming
  the floor). Exposed-pool values: 0.975 / 0.869 / 0.917 / 0.713.
- **Y2 — Level 0, frame-conditional prospective prediction (fresh cues × new frames; the frames' reference states are
  observed first, the predictions digested, and only then the fresh cue prompts run):** the same four floors. Pass →
  `PATTERN_CHANGE_PREDICTED_FRAMES_CONDITIONAL`; fail → the `_NOT_` label. Y2 is an **aggregate** claim over the
  valid fresh frames — it does not say that Level 0 works in every frame (an exposed frame reaches layer-2 entry R²
  0.39) — and it is never to be called unseen-frame prediction: the frame's reference attention state is an input.
  The six fresh frames' individual results (entry R² per layer, total-variation ratio, `c_ΔA` versus `ĉ_ΔA` over their
  scored tokens) are reported descriptively regardless of the outcome.
- **Y3 — the number-axis alternative is rejected on the fresh data:** over the scored token means of both sets,
  `ĉ̄_ΔA^axis` versus `c̄_ΔA`: **R² < 0.30**, **and** the entry R² of `ΔÂ^axis` against `ΔA` pooled over the pairs of
  both sets and all sixteen heads **< 0.30**. Both → `AXIS_ONLY_REJECTED`; otherwise `AXIS_ONLY_NOT_REJECTED`; with
  fewer than two scored token means the family is `AXIS_ONLY_NOT_EVALUABLE`. Exposed-pool values: R² −2.49 (token
  means), 0.138 (pooled entries). This family tests the alternative, not the predictor; the alternative receives the
  exact arriving change and has no free parameter, so a rejection here falsifies "the cue-induced attention-pattern
  changes are driven by the grammatical-number axis" cleanly; a `NOT_REJECTED` with a passing Y1 would mean the number
  axis alone re-routes the cue's attention on the fresh set, contradicting the exposed accounting.
- **Descriptive (no floor):** the Level-1 identities on the fresh pairs; the two diagonal-proportional comparators
  with the predeclared interpretation rule applied; the rungs and the diagonal decomposition; the total-variation
  ratios; the self-weight R² per head; `c_ΔA,ℓ` per layer; MAE; the six fresh frames individually; the re-derived
  Experiment 013 ladder on the fresh pairs (how much of `c_L` Level 0 accounts for; Level 1 accounts for all of it);
  and, offered here and only here, the interpretation: which heads move most under which cues, in the network's own
  terms — no label is preregistered for any head or cue.
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities, software defects) stop the phase, are recorded with
  their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Level 0 is token-local in the cue *change* and conditional on the reference frame state: it takes the frame's
  reference keys, values and logit rows at positions `≤ p_c` (reference-run quantities, as Experiments 013–014 took
  reference states) and the Experiment 012 bases. Y1 carries the strict boundary for the token dimension; Y2 is
  conditional on the observed reference state and aggregate over frames.
- Passing Y1–Y3 shows that, given a frame's reference attention state, the cue-induced re-routing of the layer-1–2
  heads at the cue position is prospectively reconstructed by the decoded query/key program driven by a token-local
  description of the cue change — so that, with the pattern change computed rather than supplied, the cue-position
  computation through layers 1–2 is written end to end as a weight-only program whose only frame input is the
  reference run — while a simple grammatical-number axis cannot account for the re-routing. Whether Level 0 decodes
  the interaction beyond the self-logit effect is decided by the predeclared comparator rule, not by Y1–Y3. It does not
  say what the heads attend to in other contexts, why the network routes this way, or anything about behaviour; which
  heads move for which cues is described after the scoring.
- Level 1 is exact by construction and proves nothing beyond the correctness of the program's implementation; it is
  reported as an identity, not as evidence, and counts toward no floor.
- Five lexical classes, three templates, this checkpoint; sixteen heads.

## Minimal implementation boundary

A module `attention_patterns.py` reusing `attention_paths.py` (reference-state capture extended to all positions,
stage machinery, scoring patterns), `layer_correction.py` (`LayerWeights`, bases, read), `head_transport.py`,
`read_assembly.py`, `encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; the weight-only Q/K/V program with
the rotary rotation as a small frozen class; committed extracts of Experiment 014's per-pair `c_L`, `c_M`, `c_H` and
of Experiment 013's per-pair self-weight changes and pattern-change read for replication; the confirmation builder
with the frozen lists, frames and the two prompt lists; a runner with phases `validate`, `freeze-confirmation`,
`explore`, `lock`, `confirm`, `report`, `confirm` in two stages. Tests: the program reproduces the captured rows on the
fake and (smoke) on the pinned model; the same-position rotation lemma; I2 on the fake; Level 0 equals Level 1 when
the template base equals the frame's own state; the axis alternative equals Level 1 when the change lies along
`d̂_E`; the Level-0 diagonal-proportional comparator equals Level 0 when the off-diagonal logit changes are zero; the
prediction table is computed with every capture and intervention entry point disabled (the invariant); pooled entry
R² and total-variation ratio on synthetic rows; every floor branch and the comparator rule; the stage barrier; phase
isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and no
fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1** (commits `2854be9`, `497c33f`): initial draft. Reviewed: approved conceptually with eight changes —
  keep the Level-1 exact recomputation entirely apart from the scientific result (an implementation/identity check
  that stops the phase on failure and counts toward no floor); name the model a token-local *cue-change* model
  conditional on the reference frame state, since the frame's reference keys and logit rows are supplied; freeze
  `c_ΔA` to the last index (heads, positions, reference/patched values, summation, token means); separate the Q/K
  prediction of `ΔA` from the V/O/readout used to score its consequence; keep Y2's frame-conditional qualification
  and never call it unseen-frame prediction; state that Y2 is an aggregate claim and report the six frames
  individually regardless of outcome; freeze the axis, projection, scaling, reference treatment and calibration of
  the axis-only alternative now; keep the diagonal-proportional comparator as a named descriptive competitor with a
  predeclared interpretation rule.
- **Revision 2**: all eight made. The two levels are set out as a diagram with Level 1 an identity (I1–I3, incidents)
  and Level 0 the hypothesis; the model is renamed throughout; `c_ΔA` and `c̄_ΔA` are written index by index; the
  Level-0 definition is split into the Q/K part (a), the V/O/readout part (b) and the arriving change (c); Y1 is
  labelled strict prospective, Y2 frame-conditional prospective and aggregate, with the six per-frame results
  descriptive; the axis-only alternative is frozen (the 011 `R0` vector, residual-coordinate projection, no scale or
  intercept, own state, exact arriving change); the oracle and Level-0 diagonal-proportional comparators are defined
  with the rule "Level 0 decodes the interaction beyond the self-logit effect only if its pooled entry R² exceeds the
  Level-0 comparator's by ≥ 0.10 at both layers on the fresh pairs" (exposed margins 0.23 / 0.19; oracle comparator
  0.765 / 0.793, Level-0 comparator 0.685 / 0.523). No floor of revision 1 was changed.

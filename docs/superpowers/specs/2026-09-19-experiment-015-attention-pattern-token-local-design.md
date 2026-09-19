# Experiment 015: The Cue-Induced Attention-Pattern Change from the Decoded Query/Key Computation — A Token-Local Q/K Model, Locked Before Execution — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 1 — draft for review. No Experiment 015 directory, confirmation set, lock, or model run exists.
Experiments 005–014 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter, program-completing. Experiment 013 wrote the layer-1–2 attention contribution
at the cue position as an exact two-term split per head — a frozen-pattern value path it predicted, and a
pattern-change term `Σ_k ΔA_h(p_c,k) v_h'(k) W_O^h` it *measured* from the patched run. Experiment 014 inherited the
same frozen-pattern state. This experiment writes the pattern change itself as a program — LayerNorm → query and key
projections → rotary rotation → scaled dot products → softmax — driven by the decoded state change instead of the
patched run, commits its predictions for unseen cues and frames before any of them runs, and tests a rigid number-axis
alternative on the same fresh data. Nothing is fitted; no grammatical label is attached to any head or cue before
scoring.

## Purpose and question

The pattern-change term is small in read units but it is the one piece of the cue-position computation through
layers 1–2 that Experiments 012–014 never predicted: on Experiment 013's exposed ledger it has standard deviation
0.036 of the plural cue's signal (against 0.217 for the whole layer contribution `c_L` and 0.068 for the heads' part
`c_H`), and it *is* the heads' misprediction — its correlation with `c_H − ĉ_H` over the 2724 exposed pairs is 0.986.
In pattern space the change is not small at all: a cue swap moves the self-attention weight `A_h(p_c,p_c)` of single
heads by up to 0.83, and the total variation of the eight cue rows summed over heads averages 1.66 at layer 1 and
1.06 at layer 2 per pair.

**Two facts fix what a prospective test can and cannot claim here.**

1. *Exactness of the own-state recomputation (no risk; the backbone, not the claim).* The E-patch acts only at the cue
   position, and block 0's attention reads only the embeddings, so with the parallel residual `x₁'(p_c) = x₁(p_c) + ΔE`
   exactly, while every position `k < p_c` is unchanged in every layer (causal masking). The cue row of every layer-1
   head, the layer-1 head outputs at `p_c`, the residual before block 2 at `p_c`, and the cue row of every layer-2 head
   are therefore exact functions of the weights, the frame's *reference* residuals at positions `≤ p_c`, and `ΔE`.
   Recomputing them with the weight-only program below reproduces the model to float32 precision on the whole exposed
   pool (5628 re-executed pairs: cue rows to `9.8e-6`, the residual before block 2 to `1.4e-5`, the per-head output
   split to `1.7e-5` relative). This recomputation — "Level 1" — closes Experiment 013's accounting identically (its
   remainder becomes a computed term) and enters this experiment as an incident-guarded identity. It carries no risk,
   so it is not a preregistered claim.
2. *The claim must be a reduced description.* Experiment 012 reduced the MLP corrections to a **token-local** model
   evaluated at locked template-mean base states; Experiment 013 froze the patterns. The natural reduction of the
   pattern change is the same move applied to the query/key computation: compute the cue position's query, key and
   value *changes* at the Experiment 012 locked template-mean bases from `ΔE` alone — token-only given the template —
   and let the frame contribute nothing but reference-run quantities (its keys, values and reference logit rows at
   positions `≤ p_c`). Rotary position embeddings cooperate: the same-position rotation cancels in the self logit
   (`⟨R_p a, R_p b⟩ = ⟨a, b⟩`), so the change of the diagonal logit is exactly token-local at the base, and only the
   off-diagonal logit changes couple the token-local query change to the frame's keys.

**Design check on exposed data (outside any results state; every exposed token in every exposed frame, 5628 patched
runs re-executed; to be recomputed inside Tier A).**

- **Level 0 — the token-local Q/K model.** Over the 135 token means of the predicted pattern-change read `ĉ_ΔA` (sum of
  the sixteen heads' terms, both layers) against the measured `c_ΔA`: Spearman 0.975, R² 0.869, MAE 0.0062 (measured
  spread 0.021); per pair 0.899 / 0.792. In pattern space, pooling every entry `ΔÂ_h(p_c,k)` against `ΔA_h(p_c,k)`
  over the pairs: R² 0.917 at layer 1 and 0.713 at layer 2; the total-variation error is 24% (layer 1) and 46%
  (layer 2) of the measured total variation; the sixteen self-weight changes are reproduced with pooled R² 0.889 /
  0.664. Per template (14 frames each) the layer-2 entry R² is 0.672–0.763; per frame it ranges 0.387–0.910 (median
  0.732) at layer 2 and 0.633–0.983 (median 0.930) at layer 1.
- **Alternatives, on the same pairs, each given the frame's own reference state and the exact arriving change (their
  best chance).** *Number-axis only* — the same program on the `d̂_E` component of the state change: token-mean `ĉ_ΔA`
  R² −2.49 (Spearman 0.30); pooled entry R² 0.186 at layer 1, 0.041 at layer 2, 0.138 over both. *Query-only*
  (self-key frozen): entry R² 0.23 / −0.25; *self-key-only*: 0.25 / −0.70; *additive* (both sides moved, the
  interaction `⟨Δq, Δk⟩` dropped from the diagonal): 0.26 / −0.74; *LayerNorm linearised* (first-order Jacobian,
  exact softmax): −0.57 / −1.47 — worse than predicting no change, because the cue swap is a replacement, not a
  perturbation: `‖Δx‖ / ‖x − μ(x)‖` averages 0.95 at layer 1 and 0.81 at layer 2. *Diagonal-proportional* (the
  exact self weight, the other keys rescaled in proportion to the reference row): 0.77 / 0.79 — the change is mostly
  but not only a change of self-attention. *Frozen-pattern arrival* (layer-2 rows from Experiment 013's arriving
  change, i.e. with the layer-1 pattern change omitted): 0.853 at layer 2 — the layer-1 pattern change does reach the
  layer-2 patterns.
- **Diagonal decomposition (descriptive, 303-pair subset).** The change of the self logit is dominated by the query
  side (`⟨Δq, k_ref⟩` rms 4.5 logits at layer 1, 7.0 at layer 2) with the key side (1.1 / 1.4) and the interaction
  (1.2 / 2.1) each large enough that dropping either breaks the softmax row; no partial evaluation of the bilinear
  form survives.

> Can the cue-induced change of the layer-1 and layer-2 attention patterns at the cue position — and through it the
> pattern-change term of the heads' contribution to the read direction — be predicted prospectively, for cue words
> and frames never measured on it, by the decoded query/key program evaluated on a token-local state change at
> locked template-mean bases, with the frame contributing only its reference run? Is the number-axis account of the
> same change rejected on the same fresh data?

## The quantities under test (frozen definitions)

Notation as in Experiments 012–014. Layers `ℓ ∈ {1, 2}`, heads `h ∈ {0..7}` (the sixteen heads `L01.H00 … L02.H07`);
`x_ℓ(k)` the frame's reference residual before block `ℓ` at position `k ≤ p_c`; `x_ℓ'(k)` the patched run's; `ΔE =
E(w) − E(ref_T)` the weight-only encoding difference; `x̄_ℓ^T` the Experiment 012 locked template-mean bases; `d̂_E`
the locked number direction; `r(·)` the locked weight-only read; `D_T = r(E(pl_T) − E(ref_T))` the template's
denominator.

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
out_h(p_c)  = Σ_{k ≤ p_c} A_h(p_c, k) v_h(x_ℓ(k)) W_O^h
```

*Identity I1 (incident above `1e-4` absolute):* the recomputed reference rows equal the captured `ATTN_PATTERN.L{1,2}`
rows at `p_c` of the reference run, and the recomputed patched rows from the captured patched residuals equal the
captured patched rows. *Identity I2 (incident above `1e-4` relative):* `x₂(p_c) + ΔE + [MLP₁(ln2₁(x₁(p_c) + ΔE)) −
MLP₁(ln2₁(x₁(p_c)))] + Δout₁(p_c)`, with `Δout₁` the sixteen-term layer-1 output change recomputed from the program on
`x₁(p_c) + ΔE`, equals the captured patched residual before block 2 (Level 1 — the model itself). *Identity I3:*
Experiment 013's per-head split (`frozen + pattern_change = measured`, `1e-4` relative), inherited.

**Measured (exact, from the captured rows and residuals of the reference and patched runs):**

```text
ΔA_h(p_c, k) = A_h'(p_c, k) − A_h(p_c, k)                              the pattern change, k ≤ p_c, for the sixteen heads
P_h(w, f)    = Σ_{k ≤ p_c} ΔA_h(p_c, k) v_h'(k) W_O^h                     Experiment 013's pattern-change term; v' = v at k < p_c
c_ΔA(w, f)   = Σ_{h ∈ 16} r(P_h) / D_T                                    its read (Experiment 013's `remainder_direct_pattern_change`)
```

**Predicted, Level 0 — the token-local Q/K model (weights, the locked axes and bases, the frame's reference run; no
patched quantity).** At layer 1 the state change is `Δ̂x₁ = ΔE`; at layer 2 it is the decoded arriving change defined
below. For each layer, with `x̄ = x̄_ℓ^T` and `Δ̂x = Δ̂x_ℓ`:

```text
Δq̂_h  = q̃_h(x̄ + Δ̂x) − q̃_h(x̄),   Δk̂_h = k̃_h(x̄ + Δ̂x) − k̃_h(x̄),   Δv̂_h = v_h(x̄ + Δ̂x) − v_h(x̄)        token-local at the base
Δŝ_h(p_c, k)   = ⟨R_{p_c} Δq̂_h, k_h(x_ℓ(k), k)⟩ / 8                     for k < p_c: the frame's reference keys
Δŝ_h(p_c, p_c) = [⟨q̃_h(x̄ + Δ̂x), k̃_h(x̄ + Δ̂x)⟩ − ⟨q̃_h(x̄), k̃_h(x̄)⟩] / 8   rotation-free (same-position lemma), fully token-local
Â_h(p_c, ·)    = softmax_{k ≤ p_c} [ s_h(p_c, ·) + Δŝ_h(p_c, ·) ]        the frame's reference logits moved by the token-local changes
ΔÂ_h(p_c, k)   = Â_h(p_c, k) − A_h(p_c, k)
v̂_h(k)         = v_h(x_ℓ(k)) for k < p_c;   v̂_h(p_c) = v_h(x_ℓ(p_c)) + Δv̂_h
P̂_h(w, f)      = Σ_k ΔÂ_h(p_c, k) v̂_h(k) W_O^h;    ĉ_ΔA(w, f) = Σ_{h ∈ 16} r(P̂_h) / D_T
```

The decoded arriving change at block 2 (Experiment 012's Level-0 MLP term, plus this experiment's own layer-1
attention change — the frozen-pattern value path *and* the predicted pattern change — so that no term of it is
supplied from a patched run and Experiment 013's frozen pattern is no longer used):

```text
Δ̂out₁(p_c) = Σ_{h ∈ layer 1} [ Σ_k Â_h(p_c, k) v̂_h(k) − Σ_k A_h(p_c, k) v_h(x₁(k)) ] W_O^h
Δ̂x₂        = ΔE + [MLP₁(ln2₁(x̄₁^T + ΔE)) − MLP₁(ln2₁(x̄₁^T))] + Δ̂out₁(p_c)
```

**The alternative, committed beside the predictor and defined as rigidly — the number-axis-only program at the
frame's own state, given the exact arriving change (its best chance):** with `Δx₁ = ΔE` and `Δx₂` the exact Level-1
change of the residual before block 2, `Δx_ℓ,axis = ⟨Δx_ℓ, d̂_E⟩ d̂_E`, the cue row from `q_h, k_h, v_h` evaluated at
`x_ℓ(p_c) + Δx_ℓ,axis` against the frame's reference keys — the same LayerNorm, projections, rotation, scale and
softmax, no fitted scale or intercept:

```text
Â_h^axis(p_c, ·) = softmax_k ⟨q_h(x_ℓ(p_c) + Δx_ℓ,axis, p_c), k_h(·)⟩ / 8   (with the self key from the same state);   ΔÂ^axis, P̂^axis, ĉ_ΔA^axis as above
```

**Invariant (formal):** the Level-0 prediction is a function of the token identity, the template's locked bases, the
locked axes and read, the weights, and the frame's *reference* residuals at positions `≤ p_c` — never a residual,
row or value of a patched run. The prediction function takes only those inputs; the test suite asserts that no
capture or intervention entry point is reachable while a prediction table is computed, and `confirm` reproduces the
locked table from the lock's own contents before any fresh prompt. The identities I1–I3 use patched quantities and
are checks, not predictions.

Every Level-0 quantity for the exposed frames is computable at the lock from the locked reference states (this
experiment locks, per exposed frame, the reference residuals before blocks 1 and 2 at *every* position `≤ p_c`, from
which the keys, values and reference rows follow); for the fresh frames it is computed at `confirm` stage 1 from the
frame's reference run and digested before any fresh cue prompt, as in Experiments 013–014.

**Accounting (descriptive, per pair, exact given the captures):** the Level-1 identities; the alternatives'
statistics (query-only, self-key-only, additive, LayerNorm-linearised, diagonal-proportional, frozen-pattern arrival)
each at the frame's own state; the three diagonal terms `⟨Δq, k_ref⟩`, `⟨q_ref, Δk⟩`, `⟨Δq, Δk⟩` per head; `‖Δx_ℓ‖ /
‖x_ℓ − μ‖`; the total-variation ratio `Σ ½‖ΔÂ_h − ΔA_h‖₁ / Σ ½‖ΔA_h‖₁` per layer; the sixteen self-weight changes
against their predictions; the pattern-change read per layer; MAE of `ĉ_ΔA`; and Experiment 013's ladder re-derived
with the pattern term computed — how much of `c_L` the fully decoded program (Levels 0 and 1) accounts for.

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
  computes `ΔA`, `P_h`, `c_ΔA` exactly and the Level-0 and axis-only predictions from the locked states;
- records, for the full recorded pool, the exposed statistics of the Level-0 model and of the alternative (token means
  and pairs of `ĉ_ΔA`; pooled entry R² and total-variation ratio per layer; self-weight R²), the descriptive rungs, the
  diagonal decomposition, and the re-derived ladder.

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
- **Lock (weights, the locked axes, bases and read, the 48 locked reference states — no forward pass on any fresh
  prompt):** the complete prediction table for every fresh token × exposed frame — token, frame, template, the full
  predicted rows `ΔÂ_h(p_c, ·)` for the sixteen heads, the sixteen predicted self-weight changes, `ĉ_ΔA` per layer and
  in total, the axis-only rows and `ĉ_ΔA^axis`, the predicted arriving change's read `r(Δ̂x₂)/D_T` — and the token means
  of `ĉ_ΔA`; the frozen floors; the exposed statistics; the model revision and the protocol commit; `predictions.md`
  with the token means and the predicted self-weight changes per head averaged over the exposed frames.
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
- **Aggregation, fixed:** the read-unit criteria on **token means** of `ĉ_ΔA` and `c_ΔA` over identical frame sets (at
  most 24 points each); the pattern-space criteria on **every entry** `ΔÂ_h(p_c, k)` against `ΔA_h(p_c, k)` pooled over
  the scored pairs of the set and the eight heads of a layer (one R² per layer; rows are frame-specific, so they are
  not averaged over frames). No MAE ceiling: the explained-variance floors carry the magnitude test in both spaces; MAE
  and the total-variation ratio are reported descriptively.
- **Y1 — the token-local Q/K model, strict boundary (new tokens × exposed frames; numbers committed before
  `confirm`):** token means `ĉ̄_ΔA(w)` versus `c̄_ΔA(w)`: **Spearman ≥ 0.80** and **R² ≥ 0.50**; and the pooled entry
  **R² ≥ 0.50 at layer 1 and at layer 2**. All four → `PATTERN_CHANGE_PREDICTED_TOKENS`; otherwise
  `PATTERN_CHANGE_NOT_PREDICTED_TOKENS` (naming the floor). Exposed-pool values: 0.975 / 0.869 / 0.917 / 0.713.
- **Y2 — the same, frame-conditional (new tokens × previously untested frames; numbers digested at stage 1):** the
  same four floors. Pass → `PATTERN_CHANGE_PREDICTED_FRAMES_CONDITIONAL`; fail → the `_NOT_` label.
- **Y3 — the number-axis alternative is rejected on the fresh data:** over the scored token means of both sets,
  `ĉ̄_ΔA^axis` versus `c̄_ΔA`: **R² < 0.30**, **and** the entry R² of `ΔÂ^axis` against `ΔA` pooled over the pairs of
  both sets and all sixteen heads **< 0.30**. Both → `AXIS_ONLY_REJECTED`; otherwise `AXIS_ONLY_NOT_REJECTED`; with
  fewer than two scored token means the family is `AXIS_ONLY_NOT_EVALUABLE`. Exposed-pool values: R² −2.49 (token
  means), 0.138 (pooled entries). This family tests the alternative, not the predictor, and the alternative has no free
  parameter to tune after the fact; a `NOT_REJECTED` with a passing Y1 would mean the number axis alone re-routes the
  cue's attention on the fresh set, contradicting the exposed accounting.
- **Descriptive (no floor):** the Level-1 identities on the fresh pairs; the rungs (query-only, self-key-only,
  additive, linearised, diagonal-proportional, frozen-pattern arrival) and the diagonal decomposition; the
  total-variation ratios; the self-weight R² per head; the pattern-change read per layer; MAE; the re-derived
  Experiment 013 ladder on the fresh pairs (how much of `c_L` Level 0 and Level 1 account for); and, offered here and
  only here, the interpretation: which heads move most under which cues, in the network's own terms — no label is
  preregistered for any head or cue.
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities, software defects) stop the phase, are recorded with
  their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Level 0 is token-local but not frame-free: it takes the frame's reference keys, values and logit rows at positions
  `≤ p_c` (reference-run quantities, as Experiments 013–014 took reference states), and the Experiment 012 bases. Y1
  carries the strict boundary for the token dimension; Y2 is conditional on the observed reference state.
- Passing Y1–Y3 shows that the cue-induced re-routing of the layer-1–2 heads at the cue position is prospectively
  reconstructed from the decoded query/key program driven by a token-local state change — and that with the
  pattern change computed rather than supplied, the cue-position computation through layers 1–2 is written end to end
  as a weight-only program whose only frame input is the reference run — while a simple grammatical-number axis
  cannot account for the re-routing. It does not say what the heads attend to in other contexts, why the network
  routes this way, or anything about behaviour; which heads move for which cues is described after the scoring.
- The Level-1 recomputation is exact by construction and proves nothing beyond the correctness of the program's
  implementation (rotary convention, scale, masking); it is reported as an identity, not as evidence.
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
`d̂_E`; the prediction table is computed with every capture and intervention entry point disabled (the invariant);
pooled entry R² and total-variation ratio on synthetic rows; every floor branch; the stage barrier; phase isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and no
fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1**: initial draft.

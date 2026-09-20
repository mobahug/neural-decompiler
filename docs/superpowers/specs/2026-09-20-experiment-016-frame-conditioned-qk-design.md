# Experiment 016: What the Frame Contributes — The Frame-Conditioned Token-Local Q/K Model: Reference Operands, Normalization Scale, and Block 1's Operating Point — A Prospective Test

**Date:** 2026-09-20

**Status:** Revision 1 — draft for review. No Experiment 016 directory, confirmation set, lock, or model run exists.
Experiments 005–015 are closed and are not amended by this document; Experiment 015's Y2 failure stands as recorded.

**Kind:** Prospective, zero-parameter, decomposition-completing. Experiment 015 showed that a token-local cue-change
Q/K model at template-mean bases predicts the cue-induced attention-pattern change for unseen cues in exposed frames
but misses its frame-conditional floors at layer 2 (aggregate entry R² 0.42; one new frame −0.27) while the exact
recomputation from the frame's own state (Level 1) holds to 1e-5. This experiment asks *through which channels* the
frame's cue-position state enters the reduced prediction, commits a frame-conditioned token-local model whose only
frame inputs are three exact reference-run quantities — the frame's reference query and key at the cue position as the
operands of the self-logit, the frame's LayerNorm scale before and after the cue change, and block 1's MLP evaluated at
the frame's own operating point — and tests it on unseen cues and twelve unseen frames, with a rigid "scale-only"
alternative committed for rejection and the three channels' ablations predeclared as a descriptive ladder. Nothing is
fitted; no label is attached to any head, frame, or cue before scoring.

## Purpose and question

Experiment 015's Level 0 replaced the frame's cue-position residual `x_ℓ(p_c)` by the Experiment 012 template-mean base
`x̄_ℓ^T` wherever the *change* was computed (the query, key and value changes, the self-logit change, block 1's MLP
change), and took from the frame only its reference keys, values and logit row. Its ladder showed where the loss sits:
Experiment 013's own-base frozen-pattern model reached R² 0.94 on `c_L` in the fresh frames where Level 0 reached 0.69,
and Level 1 — the same program at the frame's own state — 1.00. The information the reduced model lacks is therefore in
the frame's cue-position state; the question is *what part of it*.

**Design check on exposed data (outside any results state; every Experiment 015 pair — 6180 pairs, 183 cues, 54 frames
including the six frames that were fresh in 015 — re-analysed from the locked and digested reference states and the
stored measured rows; no forward pass; to be recomputed inside Tier A).** The frame's cue-position residual was fed to
the Level-0 machinery through candidate channels, each an exact reference-run quantity, and the layer-2 pattern change
was scored as in Experiment 015 (pooled entry R² over the eight heads' rows; the token-mean read `c_ΔA`).

- **What does not suffice (compact descriptors).** Matching only the frame's residual mean and scale in the base: layer-2
  entry R² 0.49 on the six 015-fresh frames (Level 0: 0.42). The frame deviation's component along the cue change: 0.43.
  Its component along the number axis `d̂_E`: 0.43. A shared low-dimensional subspace of frame deviations fitted on the 48
  exposed frames: 0.53 (k = 1), 0.52 (k = 2), 0.51 (k = 4), 0.56 (k = 8) — the frame variation the prediction needs is not
  low-dimensional across frames. Single attention heads' arrivals at the cue position: at best 0.79 on the exposed frames
  (`L01.H01`; head and neuron searches on every third exposed pair); all sixteen layer-0–1 heads' arrivals 0.87 (fresh)
  / 0.94 (exposed). The operating points of the 8, 32, 128
  or 512 most frame-sensitive block-1 neurons: 0.83, 0.83, 0.87, 0.92 (fresh) against 0.97 for all 2048. The frame
  dependence is distributed; no small set of heads, neurons or directions carries it.
- **What does suffice (three exact channels).** With everything else at the template means, adding to Level 0
  (1) the frame's own reference query and key of the cue position as the operands of the self-logit's bilinear form,
  (2) the frame's LayerNorm scale `σ` of the cue residual before and after the change (two scalars per layer per pair)
  in the query/key/value changes, and (3) block 1's MLP change evaluated at the frame's own cue residual, gives layer-2
  entry R² **0.988** over all 6180 pairs (per-frame minimum 0.937, median 0.991), **0.967** on the six 015-fresh frames
  (minimum 0.937; `The agency recruits`, the frame that failed 015, 0.937 from −0.27), layer-1 entry R² 0.996, the
  sixteen self-weight changes R² 0.986, and the token-mean read `c_ΔA` with Spearman 0.998 / R² 0.994 (183 tokens) and
  0.996 / 0.997 (the 24 015-fresh tokens in the six 015-fresh frames). What this model still discards is the frame
  deviation's *direction* in the change — the term `γ ⊙ (x_f − x̄)_c (1/σ_f' − 1/σ_f)`, the renormalization of the frame's
  context by the cue change — and that term is the whole remainder to the exact chain: 0.012 of entry R² on the exposed
  frames, 0.033 on the fresh ones.
- **The channels' weights (ablation from the full model, layer-2 entry R², six 015-fresh frames / 48 exposed frames):**
  without the reference operands 0.650 / 0.754 (−0.32 / −0.23); without the frame's operating point for block 1's MLP
  0.805 / 0.891 (−0.16 / −0.10); without the frame's scale 0.888 / 0.963 (−0.08 / −0.03). Each channel alone over Level
  0: operands 0.587 / 0.865, MLP operating point 0.590 / 0.770, scale 0.523 / 0.712. **Scale only** — "the frame matters
  through how large its residual is" — is the natural compact account and it fails: 0.523 / 0.712 against 0.967 / 0.988.

> Through which channels does a frame's cue-position state enter the cue-induced attention-pattern change, and does
> the frame-conditioned token-local model — the cue's query, key and value changes token-local up to the frame's
> normalization scale, the self-logit formed with the frame's own reference operands, block 1's MLP at the frame's own
> operating point, everything else at the template means — prospectively predict the change for cue words and frames
> never measured on it? Is the scale-only account rejected on the same fresh data?

## The quantities under test (frozen definitions)

Notation as in Experiment 015: layers `ℓ ∈ {1, 2}`, the sixteen heads `H`, `x_ℓ(k)` the frame's reference residual
before block `ℓ` at position `k ≤ p_c`, `x̄_ℓ^T` the Experiment 012 template-mean bases, `ΔE = E(w) − E(ref_T)`, `d̂_E`,
`r(·)`, `D_T` as before; the weight-only program `LN_ℓ`, `q̃_h`, `k̃_h`, `v_h`, `R_p`, `s_h`, `A_h`, `out_h` exactly as
frozen in Experiment 015 (partial rotary rotation, scale `1/√64`, softmax over `k ≤ p_c`). LayerNorm statistics of a
vector `x`: `μ(x)` the mean over the 512 coordinates, `σ(x) = √(mean((x − μ)²) + ε)`; `x_c = x − μ(x)`.

**Measured (exact, as in Experiment 015):** `ΔA_h(p_c, k) = A_h'(p_c, k) − A_h(p_c, k)` from the captured rows; `P_h`,
`c_ΔA = Σ_{h∈H} r(P_h)/D_T`, `c̄_ΔA(w)` the token mean over identical frame sets. Experiment 015's identities I1–I3 stay
incident-guarded checks that count toward no floor.

**Level 0-F — the frame-conditioned token-local model (the hypothesis; weights, the locked axes, read and bases, the
frame's reference run at positions ≤ p_c; no patched quantity).** Let `x_ℓ = x_ℓ(p_c)` be the frame's cue residual,
`x̄ = x̄_ℓ^T`, and `Δ̂x_ℓ` the change arriving at the layer (`Δ̂x₁ = ΔE`; `Δ̂x₂` below). Three channels:

```text
(1) reference operands   q̃_ref = q̃_h(x_ℓ),  k̃_ref = k̃_h(x_ℓ)          the frame's own cue query and key (unrotated; the rotation cancels on the diagonal)
(2) frame scale          σ_f = σ(x_ℓ),  σ_f' = σ(x_ℓ + Δ̂x_ℓ)            two scalars per layer per pair
(3) operating point      Δ̂₁ = MLP₁(ln2₁(x₁ + ΔE)) − MLP₁(ln2₁(x₁))       block 1's MLP change at the frame's own cue residual
```

and the model:

```text
n_Δ           = γ_ℓ ⊙ [ (x̄ + Δ̂x)_c / σ_f'  −  x̄_c / σ_f ]                 the change of the normalized residual: template direction, frame scale
Δq̃_h  = n_Δ W_Q^h,   Δk̃_h = n_Δ W_K^h,   Δv̂_h = n_Δ W_V^h               token-local up to the two scalars
Δŝ_h(p_c, k)   = ⟨R_{p_c} Δq̃_h, k_h(x_ℓ(k), k)⟩ / 8                       k < p_c: the frame's reference keys (as in 015)
Δŝ_h(p_c, p_c) = [ ⟨Δq̃_h, k̃_ref⟩ + ⟨q̃_ref, Δk̃_h⟩ + ⟨Δq̃_h, Δk̃_h⟩ ] / 8        the self-logit with the frame's own operands
Â_h(p_c, ·)    = softmax_{k ≤ p_c} [ s_h(p_c, ·) + Δŝ_h(p_c, ·) ];   ΔÂ_h = Â_h − A_h
v̂_h(k)         = v_h(x_ℓ(k)) for k < p_c;   v̂_h(p_c) = v_h(x_ℓ) + Δv̂_h
P̂_h = Σ_k ΔÂ_h(p_c, k) v̂_h(k) W_O^h;   ĉ_ΔA = Σ_{h∈H} r(P̂_h) / D_T                                             (V/O part, scores the consequence)
Δ̂out₁(p_c)    = Σ_{h ∈ layer 1} [ Σ_k Â_h v̂_h − Σ_k A_h v_h(x₁(k)) ] W_O^h
Δ̂x₂           = ΔE + Δ̂₁ + Δ̂out₁(p_c)                                     the arriving change: channel (3) plus the layer-1 attention change
```

The template-mean direction of the change is the reduction: the frame's deviation `x_ℓ − x̄` enters the *change* of
the query, key and value only through `σ_f` and `σ_f'`. Setting `σ_f = σ(x̄)`, `σ_f' = σ(x̄ + Δ̂x)`, `q̃_ref = q̃_h(x̄)`,
`k̃_ref = k̃_h(x̄)` and `Δ̂₁` at `x̄₁` recovers Experiment 015's Level 0 exactly; setting all three channels and adding
the discarded term `γ_ℓ ⊙ (x_ℓ − x̄)_c (1/σ_f' − 1/σ_f)` recovers Level 1 exactly (algebraic identity, tested).

**The alternative, committed beside the predictor and defined as rigidly — scale only (Y3):** channel (2) alone —
`n_Δ` as above with the frame's `σ_f`, `σ_f'`, but the self-logit's operands at the template base (`q̃_h(x̄)`, `k̃_h(x̄)`)
and block 1's MLP at `x̄₁`. No scale, intercept or calibration; the same program otherwise.

**The ablation ladder (descriptive, predeclared):** the full model minus one channel at a time (`−operands`, `−scale`,
`−operating point`), each channel alone over Level 0, Experiment 015's Level 0 (all three off), and Level 1 (exact,
identity). Predeclared ordering to be reported against, no gate: the ablation cost at layer 2 on the fresh pairs is
largest for the reference operands, then the operating point, then the scale (exposed: 0.32 > 0.16 > 0.08 on the six
015-fresh frames); the remainder from the full model to Level 1 (the discarded renormalization term) is ≤ 0.05 of
layer-2 entry R² (exposed 0.012 / 0.033).

**Invariant (formal):** every Level 0-F prediction is a function of the token identity, the locked bases, axes and read,
the weights, and the frame's *reference* residuals at positions `≤ p_c` — never a residual, row or value of a patched
run. The three channels are functions of `x_ℓ(p_c)` (reference) and `Δ̂x_ℓ` (predicted). The prediction table is
computed with every capture and intervention entry point disabled; `confirm` reproduces the locked table before any
fresh prompt.

**Accounting (descriptive, per pair):** Experiment 015's identities and comparators (the oracle and Level-0-F
diagonal-proportional rows, the comparator rule at margin 0.10), the ablation ladder, the discarded renormalization
term's size, `‖Δ̂x‖/‖x − μ‖`, `σ_f'/σ_f`, the total-variation ratios, the self-weight changes per head, `c_ΔA` per layer,
the twelve fresh frames individually, and the ladder of decoded `c_L` (Experiment 012, 013 own-base, 015 Level 0,
Level 0-F, Level 1).

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, the head weights, the
Experiment 011 locked axes and read weight, the Experiment 012 locked bases, the Experiment 015 program and its
captures, the results-state, ledger, lock, prediction-artifact, two-stage confirmation and incident conventions — all
exactly as in Experiments 012–015. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration record only; nothing is fitted)

The 183 exposed cue tokens (Experiment 015's 159 and its 24 confirmed) and the 54 exposed frames (its 48 and its 6),
with the 80 nouns. `explore`:

- captures the 54 reference states (as Experiment 015: the residuals before blocks 1 and 2 at every position `≤ p_c`,
  the pattern rows at `p_c`; I1), checks the cue-position residuals against the Experiment 015 lock and stage-1 digests
  within `1e-9`, and locks the states;
- re-measures the E-patch of every recorded pair of Experiment 015 (its 4884 + 1152 + 144 pairs) with the patched
  residuals and rows captured; replicates Experiment 015's per-pair `c_ΔA`, the sixteen self-weight changes and
  `c_L`, `c_M`, `c_H` within `1e-6` (a committed extract); checks I1–I3 and the Level-0/Level-1 recovery identities on
  every pair; computes the measured rows and the Level 0-F, scale-only and ladder predictions from the locked states;
- records the exposed statistics (token means and pairs of `ĉ_ΔA`, pooled entry R² and total-variation ratio per
  layer, per frame, self-weight R²), the ablation ladder, the renormalization remainder, and the decoded-`c_L` ladder.

The floors are frozen in this design; no exposed statistic sets a threshold.

## Confirmation set (frozen by tokenizer rules before any Experiment 016 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 183 exposed tokens and every
  exposed noun form; the first eligible entries of the frozen candidate lists, with quotas; classes for coverage and
  reporting only, **no class carries an expectation**. Words left unchosen by earlier experiments' lists were never
  executed and are eligible.
  - `determiner-like` (quota 5): `initial`, `upper`, `respective`, `individual`, `specific`, `separate`, `distinct`,
    `remaining`, `spare`, `select`.
  - `ordinal-or-numeral` (quota 4): `twentieth`, `quarter`, `twin`, `dual`, `tens`, `double`, `triple`, `couple`, `pair`,
    `single`.
  - `quantity` (quota 5): `unlimited`, `countable`, `total`, `insufficient`, `extensive`, `overall`, `average`, `typical`,
    `rare`, `frequent`.
  - `possessive-or-pronoun` (quota 4): `oneself`, `myself`, `yourselves`, `whoever`, `yourself`, `themselves`, `who`,
    `thee`.
  - `adjective` (quota 6): `bright`, `orange`, `pink`, `brown`, `grey`, `heavy`, `gentle`, `tiny`, `giant`, `silver`,
    `purple`, `yellow`.
- **Fresh frames (12, four per template), literal:** cardinal `The garden yields {cue}`, `The factory assembles {cue}`,
  `The kitchen serves {cue}`, `The workshop builds {cue}`; quantifier `The essay explores {cue}`, `The teacher explains
  {cue}`, `The guide recommends {cue}`, `The seminar addresses {cue}`; coordinated-adjective `Liam and Ava stored {cue}
  cold`, `Zoe and Theo shipped {cue} fresh`, `Nina and Arjun mixed {cue} thick`, `Elif and Marco tied {cue} tight`.
  Texts must differ from every exposed frame text; the original cue tokens are used for their own cue prompts. Twelve
  frames because the claim under test is about frames: Experiment 015's six could be decided by one.
- **Two fresh sets, both executed only by `confirm`:** the fresh tokens in the 54 exposed frames (Y1, at most 1296
  pairs; every prediction in the lock) and in the 12 fresh frames (Y2, at most 288 pairs; predictions at stage 1).
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt appears in
  the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** as listed under "Exposed pool".
- **Lock (weights, the locked axes, read and bases, the 54 locked reference states — no forward pass on any fresh
  prompt):** the complete prediction table for every fresh token × exposed frame — the sixteen Level 0-F rows, the
  self-weight changes, `ĉ_ΔA` per layer and in total, the scale-only rows and read, the Level 0-F diagonal-proportional
  rows and read, the three ablations' `ĉ_ΔA`, `σ_f'/σ_f` per layer, the decoded `c_L` at Level 0-F — and the token
  means; the frozen floors; the exposed statistics; `predictions.md`.
- **Tier C (`confirm`, once), two stages with the digested table as the barrier (Experiment 013's procedure):**
  *stage 1* — the twelve fresh frames' reference runs (exposed tokens only), I1, validity (plural-cue head change
  `≥ 0.25 σ_T`, cue-pair check), the frame-conditional prediction table, serialized and digested; *stage 2*, only after
  the digest is re-read from disk — every fresh cue's E-patch in every frame of both sets with the patched residuals
  and rows captured; I1–I3; the measured rows, `P_h`, `c_ΔA`; scoring against the two tables.

## Floors and outcome (frozen)

- **Validity (outcome-independent):** as Experiments 012–015; a token is scored in a set iff it has at least three
  valid frames there; Y2 needs at least eight valid fresh frames and sixteen scored tokens; Y1 needs sixteen scored
  tokens.
- **Aggregation, fixed:** as Experiment 015 — the read-unit criteria on token means over identical frame sets; the
  pattern-space criteria on every entry pooled over the scored pairs of the set and the eight heads of a layer (one R²
  per layer). The floors are higher than Experiment 015's because the claim is different: not that a reduced model
  captures most of the change, but that three exact channels make it nearly complete.
- **Y1 — Level 0-F, strict prospective (fresh cues × exposed frames; numbers committed before `confirm`):** token means
  `ĉ̄_ΔA` versus `c̄_ΔA`: **Spearman ≥ 0.90** and **R² ≥ 0.85**; raw entries: **R² ≥ 0.85 at layer 1 and at layer 2**.
  All four → `FRAME_CHANNELS_PREDICTED_TOKENS`; otherwise `FRAME_CHANNELS_NOT_PREDICTED_TOKENS` (naming the floor).
  Exposed-pool values: 0.998 / 0.994 / 0.996 / 0.988.
- **Y2 — Level 0-F, frame-conditional prospective, aggregate over the valid fresh frames (fresh cues × new frames;
  numbers digested at stage 1):** the same four floors. Pass → `FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL`; fail →
  the `_NOT_` label. The twelve frames are reported individually regardless of the outcome; the aggregate is the claim.
  Exposed values on the six 015-fresh frames: 0.996 / 0.997 / 0.996 / 0.967 (per-frame minimum 0.937).
- **Y3 — the scale-only alternative is rejected on the fresh data:** over the pairs of both sets, the scale-only
  model's pooled layer-2 entry R² is **below 0.85** *and* at least **0.15 below** Level 0-F's. Both → `SCALE_ONLY_REJECTED`;
  otherwise `SCALE_ONLY_NOT_REJECTED`; with fewer than two scored token means `SCALE_ONLY_NOT_EVALUABLE`. Exposed:
  0.712 against 0.988 (48 frames), 0.523 against 0.967 (six 015-fresh frames). The alternative has no free parameter;
  a `NOT_REJECTED` with a passing Y1 would mean the frame acts on the pattern change through its scale alone,
  contradicting the exposed ablations.
- **Descriptive (no floor):** the ablation ladder against its predeclared ordering; the renormalization remainder; the
  comparators and the comparator rule; the per-layer reads; the twelve frames individually; the decoded-`c_L` ladder;
  which heads move for which cues in the new frames (in the network's own terms, after scoring).
- Outcome = `Y1 | Y2 | Y3`. Incidents (replication, identities, software defects) stop the phase, are recorded with
  their commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Level 0-F is frame-conditional by construction: its three channels are reference-run quantities of the frame's
  cue position, and its keys, values and logit rows are the frame's. Y1 carries the strict boundary for the token
  dimension; Y2 is conditional on the observed reference state and aggregate over frames. Nothing here predicts a
  frame's attention state from its text alone.
- Passing Y1–Y3 shows that the frame's contribution to the cue-induced attention-pattern change is carried by three
  identifiable quantities of its cue-position state — the operands of the self-logit, the normalization scale, and
  block 1's operating point — and not by the direction of its residual in the query/key/value change, nor by its scale
  alone; with them the token-local change (template direction) predicts the change nearly exactly on unseen cues and
  frames. It does not say why the frames differ in these quantities, nor anything about behaviour. The design check
  also records that no compact descriptor tried (scalars, a shared subspace, a few heads, a few neurons) carries the
  frame dependence; that is a description of this model, not a preregistered claim.
- Twelve new frames, five lexical classes, three templates, this checkpoint; sixteen heads.

## Minimal implementation boundary

A module `frame_channels.py` reusing `attention_patterns.py` (the program, `ReferenceRow`, statistics, stage machinery,
scoring patterns, comparators), `layer_correction.py`, `attention_paths.py`, `neuron_feature.py` (pool chain),
`plural_mechanism.py`; the Level 0-F model with its three channel switches (so that the ablations, the scale-only
alternative and the Experiment 015 Level 0 recovery are the same code with switches, and the Level 1 recovery with the
renormalization term is tested as an identity); a committed extract of Experiment 015's per-pair `c_ΔA`, self-weight
changes and `c_L`, `c_M`, `c_H` for replication; the confirmation builder with the frozen lists and twelve frames; a
runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages.
Tests: the switch-off recovery of Experiment 015's Level 0 to 1e-12 on the fake; the all-channels-plus-remainder
recovery of Level 1 to 1e-12; the invariant; the floors, Y3 and the ladder on synthetic tables; the stage barrier; phase
isolation; the pinned-model smoke on a neutral prompt as in Experiment 015.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`, and no
fresh cue prompt runs before its frame's stage-1 predictions are digested.

## Revision history

- **Revision 1**: initial draft.

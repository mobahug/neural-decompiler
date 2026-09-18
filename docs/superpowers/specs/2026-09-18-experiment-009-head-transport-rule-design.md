# Experiment 009: The Transport Rule of `L03.H04` — Mechanism Decomposition and a Prospective Test on Frozen New Cues

**Date:** 2026-09-18

**Status:** Revision 1, for review. Not approved. No Experiment 009 directory, confirmation set, lock, or model run
exists. Experiments 005–008 are closed and are not amended by this document.

**Kind:** Prospective. A confirmation set of new cue tokens and new frames is frozen by tokenizer rules before any
Experiment 009 model output; every rule is fitted on the forty exposed tokens only; predictions are locked; the
fresh set runs once.

## Purpose and questions

Experiment 008 localized the suppression of the apparent number signal of `this`, `a`, `another`: the signal survives
the cue-position layers 1–2 and the sentence context, `L03.H04` attends to the cue normally, and the head's output on
its number axis carries only 3–19% of the plural cue's signal for those tokens, ≈ 100% for numerals and quantities,
and 150–180% for `these` and `those`. The transmitted fraction is graded
(`this < a ≈ another < that < the < numerals < these ≈ those`). Experiment 008 also found that the encoding-axis
component of `ΔE_T(w)` alone drives 50–64% of the plural effect for those tokens while the whole `ΔE_T(w)` drives none,
without saying where between the encoding and the contrast the non-additivity enters.

Experiment 009 asks two questions, one mechanistic and one prospective:

> **Q1 (how).** Is the head's output change an exact linear read-out of the change in the residual it receives at the
> cue position (linear cancellation inside the OV computation), or does the selectivity require the layer-3 LayerNorm
> nonlinearity, the attention redistribution, or changes outside the cue position? And where along
> `E → R1 → T → R2 → R3 → c` does the non-additivity of the component patches enter?

> **Q2 (prediction).** Can a rule derived from the forty exposed cue tokens predict, before any of them is run, how
> completely `L03.H04` transports the number signal of new determiners and numerals in new frames — and can the
> linear-from-`E` decompiler, refitted on forty tokens, predict their behavioral contrast shift?

## Competing hypotheses for Q1 (kept genuinely competing; decided by exact decompositions, not by fitting)

For frame `f` (query position `p_t`, cue position `p_c`) and head `h = L03.H04`, with `r_k` the layer-3 input residual
at key position `k`, `ν_k = LN₃(r_k)`, `v_k = ν_k W_V^h + b_V^h`, attention weights `A_k` from `p_t`, and the per-head
result `T = Σ_k A_k v_k W_O^h`, the measured head-output change under an E-patch is exactly

```text
ΔT = Σ_k A_k^ref Δv_k W_O   +   Σ_k ΔA_k v_k^patched W_O
```

Three nested predictors of `ΔT` from the cue position alone, in order of complexity:

- **P1 — linear OV read-out (fixed LayerNorm statistics):**
  `ΔT₁ = A_c^ref · [γ₃ ⊙ (Δr_c − mean(Δr_c)) / σ_c^ref] W_V W_O`. Linear in the head's input change; the head reads
  one direction `m = W_V W_O d̂_T` (in LayerNorm-normalized coordinates). If P1 suffices, the suppression is *linear
  cancellation inside the OV computation*: the orthogonal part of the input residual carries a negative `m` component.
- **P2 — exact LayerNorm at the cue position, reference attention:**
  `ΔT₂ = A_c^ref · (ν_c^patched − ν_c^ref) W_V W_O`. Adds the LayerNorm scale and mean nonlinearity at the cue
  position (norm-based gating: a large orthogonal component inflates `σ_c` and shrinks the axis signal the head sees).
- **P3 — exact LayerNorm and the patched attention pattern, cue-position values only:**
  `ΔT₃ = A_c^patched v_c^patched W_O − A_c^ref v_c^ref W_O + Σ_{k≠c} (A_k^patched − A_k^ref) v_k^ref W_O`. Adds
  attention redistribution.
- **Remainder** `ΔT − ΔT₃ = Σ_{k≠c} A_k^patched Δv_k W_O`: value changes at other positions (only the adjective
  position after the cue in coordinated-adjective frames can change) plus numerics.

Each level is computed from captured activations and weights with no fitted parameter. The **upstream** question is
answered by the component patches traced through the stages (M2 below).

## Inherited fixed elements

Model, runtime, instrumented forward path, contrast `c`, exact final LayerNorm, reference cues `ref_T` (`one`, `each`),
reference prompts, weight-only `E(w)`, the E-patch intervention, the E-patch shift `Δc(w, f)` (mean over nouns), the
fixed circuit, the site-axis estimator (plural positive), the stage sites and signal fractions
`ŝ_s = ⟨Δ_s(w, f), d̂_s⟩ / ⟨Δ_s(pl_T, f), d̂_s⟩` with their uninformative floors, the oriented trace, the
component split `ΔE = ΔE_∥ + ΔE_⊥` along `d̂_E`, the results-state, ledger, lock, and incident conventions — all
exactly as in Experiments 006–008. The stage axes are re-estimated on the eighteen exposed frames' clean cue pairs at
`explore` (Experiment 008 found `d̂_E` identical to Experiment 005's frozen axis). Seeds: runtime `20260916`, control
`20260922`.

## Exposed pool (fitting data; all previously executed)

The forty cue tokens, eighteen frames, and eighty nouns (79 single-token) of Experiment 008. Nothing else may be
executed before `confirm`.

## Confirmation set (frozen by tokenizer rules before any Experiment 009 model output)

- **Fresh cue tokens (up to 24), in five categories with quotas, chosen as the first eligible entries of the frozen
  candidate lists** (single token with a leading space in the pinned tokenizer; disjoint from the forty exposed tokens
  and from every exposed noun form; a prompt in every fresh frame must round-trip):
  - `singular-selecting` (quota 3): `an`, `either`, `neither`, `whichever`, `whatever`.
  - `plural-numeral` (quota 6): `eleven`, `thirteen`, `fourteen`, `fifteen`, `sixteen`, `twenty`, `thirty`, `forty`,
    `fifty`, `sixty`, `ninety`, `thousand`, `million`.
  - `plural-quantity` (quota 5): `more`, `most`, `other`, `certain`, `enough`, `additional`, `extra`, `assorted`,
    `sufficient`, `innumerable`.
  - `number-neutral` (quota 6): `my`, `our`, `their`, `your`, `its`, `his`, `her`, `which`, `whose`, `what`.
  - `control` (quota 4): `small`, `blue`, `new`, `cold`, `green`, `cheap`, `warm`, `dark`.
  A category whose quota cannot be met by eligible entries is frozen with the entries available (recorded); the set is
  frozen even if it has fewer than 24 tokens. The categories carry pre-registered expectations used only in the
  category-level check below: singular-selecting low transport (`q_T ≤ 0.35`), plural-numeral and plural-quantity
  high transport (`q_T ≥ 0.65`), number-neutral and control unconstrained (their predictions are the sharpest test of
  the graded rule).
- **Fresh frames (6, two per template), literal:** cardinal `The shelf carries {cue}`, `The ledger names {cue}`;
  quantifier `The manual describes {cue}`, `The bulletin mentions {cue}`; coordinated-adjective
  `Ida and Tomas counted {cue} shiny`, `Yusuf and Petra wrapped {cue} thin`. Their texts must differ from every exposed
  frame text; positions `p_c`, `p_t` follow the template; the original cue tokens are used for their own cue prompts.
- **Nouns:** the eighty exposed nouns (79 single-token). The prediction targets are per-(token, frame) means over
  nouns; no claim is noun-specific, and Experiments 006–007 already tested noun generalization. This is a stated limit.
- The set is committed (`confirmation-v1.json`, tokenizer-only, digest recorded) before `explore`; `confirm` refuses if
  any fresh prompt appears in the ledger earlier.

## Measurements

### Tier A (`explore`, exposed pool; once)

- **M1 — E-patch traces with the head's internals** for all 40 tokens × 18 frames: the stage captures of Experiment 008
  (`R0`, `R1`, `T`, `R2`, `M4`, `M5`, `R3`, contrast), plus everything the OV decomposition needs: the layer-3 input
  residual at every key position, the layer-3 attention row of head 4 from `p_t`, and the per-head result — for the
  clean reference run and the patched run. The Experiment 008 replication check applies (mean shifts within 1e-6 of
  the committed extracts).
- **M2 — component patches traced**: `E(ref_T) + ΔE_∥` and `E(ref_T) + ΔE_⊥` with the same captures. Stage-wise
  additivity gap `g_s(w, f) = [Δ_s(full) − Δ_s(∥) − Δ_s(⊥)] · d̂_s / ⟨Δ_s(pl_T, f), d̂_s⟩` for `s ∈ {R1, T, R2, R3}` and
  `g_c` on the contrast (Experiment 008's `g`). The **non-additivity stage** `ν*(w, f)` is the first stage with
  `|g_s| > g_max = 0.25`, or `NONE`.
- **M3 — OV decomposition** for every patched run: `ΔT₁`, `ΔT₂`, `ΔT₃`, the remainder, each projected on `d̂_T` and
  normalized like `q_T`; the exact identity `ΔT₃ + remainder = ΔT` is checked (1e-4 relative) in every run.
- **M4 — rule fitting on the forty tokens (leave-one-cue-out by token, as in Experiment 007):**
  - *Transport rule* for `q_T` from `E` alone: primary family `q̂_T(w, T) = β_T ⟨ΔE_T(w), v⟩` with `v` the unit
    response-weighted direction `v ∝ Σ_i x_i y_i` over the pooled rows (`x_i = ΔE_T(w)`, `y_i = q_T(w, f)`; the rank-1
    cross-moment direction for a scalar response) and `β_T` by no-intercept least squares per template; baseline
    `Ridge-scalar` (template-specific ridge from `ΔE` to `q_T` with Experiment 007's nested cue-group `λ`).
  - *Contrast rule* for `Δc` from `E` alone: Experiment 007's supervised cross-moment family at ranks 1–4 refitted on
    the forty tokens (same estimator, singular-gap and identifiability rules, rank rule, τ), with Experiment 007's
    locked sixteen-token program as a frozen baseline.
  - Quality gates (unchanged from Experiment 007, per rule): Spearman ≥ 0.70 and normalized LOCO RMSE ≤ 0.50 over the
    forty cue-level values; the 20% rule for ranks above 1. A rule that fails its gate is **not locked** and its axis
    is reported as `NOT_LOCKED`; the other axes proceed.
- **Mechanism level selection (frozen rule):** over the 720 exposed (token, frame) pairs, for each level `j ∈ {1, 2, 3}`
  compute the MAE and the R² of `q̂_T^{(j)}` against the measured `q_T`. The locked level is the lowest `j` with
  `R² ≥ 0.90` and `MAE ≤ 0.10`; if none qualifies, the mechanism axis is `NOT_LOCKED` and the decomposition is reported
  (the remainder's share tells whether other positions matter).

### Lock

The candidate lock records the frozen confirmation set digest, the stage axes, `d̂_T`, `m`, the locked mechanism
level with its exposed MAE/R² and `τ_M = max(0.10, 3 × RMSE of its exposed residuals)`, the transport rule (`v`,
`β_T`, τ₂ = max(0.10, 3 × LOCO RMSE)), the contrast rule (rank, program, τ₃), the per-(fresh token, fresh frame)
predictions `q̂_T` and `Δĉ`, the category expectations, and every floor. Installing and committing it is the
preregistration act.

### Tier C (`confirm`, once)

For every fresh token in every fresh frame: the E-patch run with the same captures as M1 (measured `q_T`, `Δc`,
the OV decomposition at the locked level), and the behavioral prompt (`Δc_beh`, reported). Also the fresh frames'
original cue pairs (the cue-effect precondition `d_full > 0` in ≥ 108/120 of frame × noun pairs on the fresh frames,
computed over the exposed nouns as in Experiment 007 — here 6 frames × 79 nouns = 474 pairs, floor
`exact_count_floor(108/120, 474)`).

## Floors and outcome rules

- **Precondition:** the fresh-frame cue effect holds (`CUE_EFFECT_NOT_REPLICATED` otherwise; nothing else is judged).
- **Mechanism axis (Y1, no fitted parameter):** on the fresh (token, frame) pairs, the locked level's `q̂_T^{(j)}`
  against the measured `q_T`: Spearman ≥ 0.90 and MAE ≤ `τ_M`. Pass → `HEAD_MECHANISM_CONFIRMED_P<j>` (P1: linear
  OV cancellation; P2: LayerNorm-gated OV; P3: attention-modulated OV). Fail → `HEAD_MECHANISM_NOT_SUPPORTED` with the
  level at which the fresh data would have passed, if any (reported, not a claim).
- **Transport-rule axis (Y2):** over the fresh tokens' six-frame means, `q̂_T` versus measured `q_T`: Spearman ≥ 0.80,
  MAE ≤ τ₂, and the category check (at least 2 of 3 singular-selecting tokens measured `≤ 0.35`; at least 8 of the 11
  plural tokens measured `≥ 0.65`; the rule's predictions must agree with those measured categories in the same
  counts). Pass → `TRANSPORT_RULE_PREDICTED`; fail → `TRANSPORT_RULE_FAILED`, naming the failing category.
- **Contrast-rule axis (Y3):** Experiment 007's Y1 floors on the fresh tokens (Spearman ≥ 0.80, MAE ≤ τ₃,
  confident-token sign rule in ≥ 5 of 6 frames). Pass → `CONTRAST_RULE_PREDICTED`; fail → `CONTRAST_RULE_FAILED`.
  The Experiment 007 sixteen-token program's fresh performance is reported beside it.
- **Non-additivity stage (M2, descriptive on both pools):** the modal `ν*` over the singular-selecting and the
  exposed suppressed tokens is reported with its agreeing fraction; it enters no label.
- The outcome is the triple of axis labels (with `NOT_LOCKED` where a gate failed). Incidents (replication mismatch,
  identity failures, OV identity failure, software defects) stop the phase, are recorded with their commit, and are
  never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Three singular-selecting candidates exist in English at this position; the category check is correspondingly weak,
  and the number-neutral determiners are the informative prospective cases. Nothing generalizes beyond the pinned
  checkpoint, the three templates, single-token regular nouns, and the tokens tested.
- P1–P3 decompose the head's output change given its *measured* input change; they say how the head transforms what
  it receives, not why upstream layers place each token where they do. The E-only rules (Y2, Y3) address the latter
  and can fail while Y1 passes.
- The stage axes summarize each stage by one direction; the OV decomposition and the direct effects are exact and are
  weighted over the axis fractions where they disagree.

## Minimal implementation boundary

A module `head_transport.py` reusing `cue_suppression.py` (pool, stage axes, traces, component patches, fractions),
`supervised_subspace.py` (cross-moment estimator, ridge, LOCO, gates, exports, lock validation patterns),
`cue_decompilation.py` (confirmation builder pattern, results state), and `plural_mechanism.py`; the OV decomposition
as pure tensor functions checked against the captured head result; a confirmation builder with the frozen candidate
lists and frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`.
Tests: the OV identity on the fake (levels sum to the measured head result), planted linear/LN/attention cases that
each level does or does not explain, the additivity-stage rule, the rule fits and gates, the confirmation freeze and
its refusals, and phase isolation.

## Approval and stopping condition

Design first; no implementation until approved. Order after approval: plan → code and tests → tokenizer-only
confirmation freeze and commit → implementation review → `explore` once → `lock` → the user's lock commit → `confirm`
once → report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`.

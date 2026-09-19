# Experiment 011: Does the Weight-Only Encoding Read Predict `L03.H04`'s Transport of New Cues? — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 2 — conceptually approved at revision 1 subject to three clarifications under "Revision
history", which this revision makes (the prospective boundary and the weight-computable encoding, the aggregation
level of Y1 and τ_g, and outcome-independent validity rules). No Experiment 011 directory, confirmation set, lock, or
model run exists. Experiments 005–010 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter. The quantity under test is a **weight-defined read functional applied to the
cue's layer-0 encoding**; nothing is fitted to any cue. A confirmation set of new cue tokens and new frames is frozen
by tokenizer rules before any Experiment 011 model output; every numeric prediction is committed before the single
confirmation run, and the prediction phase performs **no forward pass on any fresh prompt**.

## Purpose and question

Experiments 009 and 010 established, on exposed data: (i) `L03.H04`'s number transport is a fixed-normalization
linear read-out of the residual arriving at the cue position (confirmed prospectively, Spearman 0.957), and (ii) the
mixture the head reads is already largely present in the layer-0 MLP encoding — for `this`, `a`, `another`, `every`
the number-axis part of `E` reads positively and the rest of `E` reads negatively through the head's direction in
every frame, layers 1–2 net at most ±0.07, and the encoding read alone ranks the 63 exposed cues' transport with
Spearman 0.925. That last number is descriptive: the 63 cues were all exposed, and the head's direction `d̂_T` was
estimated on their frames.

> Does the encoding read `g_E(w, T)` — the projection of `E(w) − E(ref_T)` onto the head's read direction, normalized
> by the template's plural cue — predict how much number signal `L03.H04` transports for cue words it has never been
> measured on, in frames it has never seen, **before any head or downstream behaviour of those cues is observed**?
> And does the head's fixed-normalization linear read-out (P1) hold on those new cues as it did in Experiment 009?

**The prospective boundary, stated exactly.** In this model the block-0 MLP reads the second LayerNorm of the token
embedding alone (GPT-NeoX parallel residual, rotary positions), so the layer-0 encoding of a cue is the weight-only
function `E(w) = MLP₀(ln2₀(W_E[w]))` used since Experiment 005 and verified against the captured `L00.MLP` output in
every E-patch since (1e-5). `g_E` is therefore computable from the weight matrices (`W_E`, `ln2₀`, `MLP₀`, `γ₃`, `W_V`,
`W_O`) and the frozen axis `d̂_T` **without running any fresh prompt**: the prediction phase touches weights only. The
claim is not "prediction without using the model" — the encoding is the model's own layer-0 representation of the cue
— but prediction of unseen `L03.H04` transport from unseen cues' layer-0 encodings before observing any head or
downstream behaviour of those cues, with a read functional fixed entirely by earlier experiments.

If both hold, the chain "layer-0 MLP builds a distributed encoding → its projection onto the head's read direction
sets how much number the head transmits → the head reads it linearly" has been tested prospectively at every link.

## The quantity under test (frozen definition)

With `m = W_V^h W_O^h d̂_T` (`h = L03.H04`), `γ₃` the layer-3 attention LayerNorm weight, `d̂_T` the head-output number
axis estimated on the exposed frames' clean cue pairs, `ref_T` the template's reference cue (`one`, `each`), and
`pl_T` its plural cue (`two`, `several`):

```text
r(x)      = ⟨ x − mean(x)·1 , γ₃ ⊙ m ⟩                          the weight-only inner read
g_E(w, T) = r(E(w) − E(ref_T)) / r(E(pl_T) − E(ref_T))            the encoding read, normalized by the plural cue
```

`g_E` uses the weights (`W_E`, `ln2₀`, `MLP₀`, `γ₃`, `W_V`, `W_O`) and the frozen axis `d̂_T`; no reference-run
statistics, no fitted coefficient. Its companion splits, defined the same way: `g_∥` (the part of `ΔE` along the
encoding number axis `d̂_E`) and `g_⊥ = g_E − g_∥`. `g_E` depends on the template only (through `ref_T` and `pl_T`), not
on the frame.

**Denominator validity (frozen before any fresh measurement).** The three template denominators
`r(E(pl_T) − E(ref_T))` are weight-only and known at `explore`. A template's predictions are `PREDICTION_UNDEFINED` if
its denominator is below `0.25 × max_T |r(E(pl_T) − E(ref_T))|` or below `0.25 σ_r`, where `σ_r` is the site-axis scale
of `r` over the exposed frames' clean cue pairs; such a template's fresh frames are excluded before the lock, and if
fewer than two templates remain the experiment stops at `PREDICTION_UNDEFINED` with no confirmation run.

The measured target is Experiment 008/009/010's transport fraction `q_T(w, f) = ⟨ΔT(w, f), d̂_T⟩ / ⟨ΔT(pl_T, f), d̂_T⟩`
from the E-patch of `w` into frame `f`'s reference prompt. **Frame validity is decided by the template's plural cue
alone, never by a candidate's own value:** a fresh frame is valid iff its plural cue's E-patch head change satisfies
`|⟨ΔT(pl_T, f), d̂_T⟩| ≥ 0.25 σ_T` and its original cue pair passes the frame-level cue-effect check below; in a valid
frame **every** licensed candidate is scored, whatever its `q_T`.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, stage axes (plural
positive, re-estimated on the exposed frames at `explore`), the head weights, the P1 predictor and its OV identity
checks, the exact attribution functional `ρ_f` (Experiment 010), the results-state, ledger, lock, prediction-artifact,
and incident conventions — all exactly as in Experiments 009–010. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration only; nothing is fitted)

The 63 cue tokens and 24 frames of Experiment 010 (and the 80 nouns for the replication checks). `explore` recomputes
`q_T` for all 63 × 24 (replicated against a committed extract of Experiment 010's recorded `q_T` within 1e-6),
estimates `d̂_T` and `d̂_E`, computes `g_E` for the 63 tokens, and fixes two tolerances from exposed residuals only, at
exactly the aggregation level at which the floors are scored:

- **token means over one frame set.** For a token `w` and a frame set `F(w)`, `ḡ_E(w) = mean_{f ∈ F(w)} g_E(w, T(f))`
  and `q̄_T(w) = mean_{f ∈ F(w)} q_T(w, f)` over the **same** frames (`g_E` varies only with the template, so `ḡ_E` is
  the template values weighted by the number of frames of each template in `F(w)`). On the exposed pool `F(w)` is the
  token's informative exposed frames (own-reference frames excluded, as in Experiment 010).
- `τ_g = max(0.10, 3 × RMSE over the 63 exposed tokens of (ḡ_E(w) − q̄_T(w)))` — token-mean residuals against
  token-mean targets, the level of Y1;
- `τ_M = max(0.10, 3 × RMSE over the 63 × 24 informative pairs of (P1 fraction − q_T))` — pair level, the level of Y2.

Both numeric values are written into the lock before any fresh measurement.

## Confirmation set (frozen by tokenizer rules before any Experiment 011 model output)

- **Fresh cue tokens (at most 24)**: single token with a leading space; disjoint from the 63 exposed tokens and from
  every exposed noun form; the first eligible entries of the frozen candidate lists, with quotas. The lists are
  organized by lexical class **for coverage and reporting only — no class carries an expectation**; the only
  expectations are the committed `g_E` predictions.
  - `determiner-like` (quota 5): `an`, `such`, `much`, `little`, `less`, `half`, `whichever`, `whatever`.
  - `numeral` (quota 5): `eighteen`, `nineteen`, `twenty`, `thirty`, `forty`, `fifty`, `sixty`, `seventy`, `eighty`,
    `ninety`, `zero`.
  - `quantity` (quota 5): `additional`, `extra`, `sufficient`, `ample`, `abundant`, `scarce`, `fewest`, `least`,
    `myriad`, `manifold`.
  - `possessive-or-pronoun` (quota 4): `its`, `whose`, `mine`, `yours`, `hers`, `theirs`.
  - `adjective` (quota 5): `green`, `cheap`, `warm`, `dark`, `huge`, `black`, `white`, `young`, `empty`, `wooden`.
  `an` is licensed only in cue-final frames (in the coordinated-adjective frames it would precede a consonant-initial
  adjective); every other cue in every fresh frame. Because the target is head-level, no noun licensing applies; the
  behavioral contrast is reported only for continuity, over the 79 single-token exposed nouns.
- **Fresh frames (6, two per template), literal:** cardinal `The cabinet stores {cue}`, `The vendor sells {cue}`;
  quantifier `The survey covers {cue}`, `The monitor tracks {cue}`; coordinated-adjective
  `Leo and Maya labeled {cue} round`, `Priya and Jonas bundled {cue} soft`. Texts must differ from every exposed frame
  text; the original cue tokens are used for their own cue prompts.
- The set (tokens with classes and licensed frames, frames, prompts, digest) is committed as `confirmation-v1.json`
  before `explore`; `confirm` refuses if any fresh prompt appears in the ledger earlier.

## Measurements

- **Tier A (`explore`, exposed pool, once):** the 63 × 24 E-patches with the head's internals captured (as Experiment
  010), replication against the Experiment 010 extract, `d̂_T`, `d̂_E`, `g_E`, `g_∥`, `g_⊥` for the 63 tokens, the exposed
  Spearman/MAE of `g_E` against `q̄_T` (descriptive), `τ_g`, the P1 fractions and `τ_M`.
- **Lock (the prediction phase; weights only, no forward pass on any fresh prompt):** `g_E(w, T)` (and `g_∥`, `g_⊥`)
  for every fresh token and template, the token means `ḡ_E(w)` over the token's licensed fresh frames, `d̂_T`, `d̂_E`,
  `γ₃ ⊙ m`, the template denominators, the numeric `τ_g` and `τ_M`, the exposed descriptive statistics, and Experiment
  009's frozen rank-1 transport rule's predictions for the fresh tokens as the reported baseline; `predictions.md` with
  one row per fresh token and template and the token means. The lock phase may read weights and the exposed results
  state; it has no access to a capture or intervention API. `confirm` recomputes every prediction from the weights and
  the locked axes before any fresh prompt runs and refuses on any difference above 1e-9.
- **Tier C (`confirm`, once):** for every fresh frame, the reference run with the head's internals, the plural cue's
  E-patch (normalization and frame validity), and the sg/pl cue prompts (frame-level cue-effect check: `d_full > 0` in
  at least `exact_count_floor(108/120, 79)` of the frame's 79 noun pairs); then, in every valid frame, for every
  licensed fresh token the E-patch with the head's internals captured: measured `q_T`, the P1 fraction (which needs
  the fresh frame's reference LayerNorm scale and attention weight and is therefore computed here), the exact
  attribution ledger (`f_E`, `f_∥`, `f_⊥`, `f_k`, net, gross), and the behavioral contrast (reported).

## Floors and outcome (frozen)

- **Validity (outcome-independent):** a fresh frame is valid iff its plural cue's head change is informative and its
  cue-pair check passes (both depend on the template's original cues only); a fresh token is scored iff it has at least
  three valid licensed frames, and `F(w)` is exactly those frames for both `ḡ_E(w)` and `q̄_T(w)`. No candidate is ever
  excluded on the basis of its own `q_T`. `PRECONDITION_FAILED` if fewer than four of the six frames are valid or
  fewer than sixteen tokens are scored; nothing is then judged.
- **Y1 — encoding read → transport (zero-parameter, numbers committed before `confirm`):** over the scored fresh
  tokens, `ḡ_E(w)` versus measured `q̄_T(w)`: Spearman ≥ 0.80 and MAE ≤ `τ_g`. Pass → `ENCODING_READ_PREDICTS_TRANSPORT`;
  fail → `ENCODING_READ_FAILS` (naming the floor). Experiment 009's rank-1 rule (fitted on 40 tokens) is reported beside
  it, never judged.
- **Y2 — prospective replication of the frozen P1 mechanism:** the P1 equation, its identity checks, and `τ_M` are
  frozen before the fresh head data exist, but P1's per-example values need the fresh frame's reference LayerNorm scale
  and attention weight and are computed at `confirm`; they are not in `predictions.md`. Over the fresh (scored token,
  valid frame) pairs, the P1 fraction versus measured `q_T`: Spearman ≥ 0.90 and MAE ≤ `τ_M`. Pass →
  `HEAD_P1_REPLICATED`; fail → `HEAD_P1_NOT_REPLICATED`.
- **Descriptive (no floor):** the fresh tokens' attribution ledger, with the sign of `f_⊥` for tokens whose locked `g_E`
  is ≤ 0.35 and the fraction of fresh tokens whose net layer change lies within the exposed range; the exposed
  baseline for gross change (median `G` 0.44 in Experiment 010) is stated so that no relay claim is made without it.
- Outcome = `Y1 | Y2`. Incidents (replication, identities, software defects) stop the phase, are recorded with their
  commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- `g_E` is a weight-defined functional of the cue's layer-0 encoding given the frozen axis `d̂_T` (estimated from
  exposed clean runs); the design never calls it "weight-only" without that qualification, and never claims that no
  representation of the fresh cue is used — the encoding is that representation, computed from the weights.
- Passing Y1 shows that the encoding's projection onto the head's direction predicts transport for new cues in new
  frames; it does not identify which neurons or which upstream computation build the projection, and it does not
  reach the behavioral contrast (Experiment 007's failed link is not retested here).
- Twenty-four tokens give a coarse Spearman; the floors are the same as Experiment 009's so that the two prospective
  transport tests are comparable.
- Nothing generalizes beyond the pinned checkpoint, the three templates, and the tokens tested.

## Minimal implementation boundary

A module `encoding_read.py` reusing `read_assembly.py` (pool, functional, attribution ledger), `head_transport.py`
(head weights, P1, confirmation and lock patterns, prediction artifact), `cue_suppression.py`, and
`plural_mechanism.py`; a committed extract of Experiment 010's `q_T` means; the confirmation builder with the frozen
lists and frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`.
Tests: `g_E` equals the ratio of inner reads on the fake; the lock's predictions reproduce from the weights; every
floor branch on synthetic tables; the confirmation policy; phase isolation.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` → the user's lock commit → the reviewer's sign-off → `confirm` once →
report. No token, frame, or prediction may be changed after the lock; no fresh prompt runs before `confirm`.

## Revision history

- **Revision 1** (commit `adbc6f7`): initial draft. Reviewed: conceptually approved with three clarifications — the
  prospective boundary (the encoding of a fresh cue is the model's representation; state what is and is not observed
  before the lock), the aggregation level of Y1 and of `τ_g` (token means over the same frame set, both), and
  outcome-independent validity (no candidate filtered by its own `q_T`; frozen handling of small denominators and
  minimum counts); Y2 described as a prospective replication of the frozen P1 equation whose per-example values are
  computed at `confirm`.
- **Revision 2**: (1) the boundary is stated exactly — `E(w)` is the weight-only block-0 MLP encoding used since
  Experiment 005, so the lock phase computes every prediction from weights and the frozen axes with no forward pass on
  any fresh prompt and no access to capture or intervention APIs; the claim is prediction of unseen head transport
  from unseen cues' layer-0 encodings before any head or downstream behaviour is observed; the functional is called
  weight-defined / zero-parameter, never "weight-only" without the axis qualification. (2) `ḡ_E(w)` and `q̄_T(w)` are
  means over one identical frame set; Y1 is scored at that level; `τ_g` is derived from the 63 exposed token-mean
  residuals at the same level and its numeric value is written into the lock. (3) Frame validity depends only on the
  template's plural cue and cue pair; every licensed candidate in a valid frame is scored; a token needs three valid
  frames; at least four valid frames and sixteen scored tokens are required; template denominators below
  `0.25 × max` or `0.25 σ_r` make that template `PREDICTION_UNDEFINED` before the lock. No floor or threshold changed.

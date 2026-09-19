# Experiment 011: Does the Weight-Only Encoding Read Predict `L03.H04`'s Transport of New Cues? — A Prospective Test

**Date:** 2026-09-19

**Status:** Revision 1, for review. Not approved. No Experiment 011 directory, confirmation set, lock, or model run
exists. Experiments 005–010 are closed and are not amended by this document.

**Kind:** Prospective, zero-parameter. The quantity under test is derived from the model's weights and one frozen
axis; nothing is fitted to any cue. A confirmation set of new cue tokens and new frames is frozen by tokenizer rules
before any Experiment 011 model output; every prediction is committed before the single confirmation run.

## Purpose and question

Experiments 009 and 010 established, on exposed data: (i) `L03.H04`'s number transport is a fixed-normalization
linear read-out of the residual arriving at the cue position (confirmed prospectively, Spearman 0.957), and (ii) the
mixture the head reads is already largely present in the layer-0 MLP encoding — for `this`, `a`, `another`, `every`
the number-axis part of `E` reads positively and the rest of `E` reads negatively through the head's direction in
every frame, layers 1–2 net at most ±0.07, and the encoding read alone ranks the 63 exposed cues' transport with
Spearman 0.925. That last number is descriptive: the 63 cues were all exposed, and the head's direction `d̂_T` was
estimated on their frames.

> Does the weight-only encoding read `g_E(w, T)` — the projection of `E(w) − E(ref_T)` onto the head's read direction,
> normalized by the template's plural cue — predict, before any of them is run, how much number signal `L03.H04`
> transports for cue words it has never been measured on, in frames it has never seen? And does the head's
> fixed-normalization linear read-out (P1) hold on those new cues as it did in Experiment 009?

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
statistics, no fitted coefficient. Its companion splits, also weight-only: `g_∥` (the part of `ΔE` along the encoding
number axis `d̂_E`) and `g_⊥ = g_E − g_∥`.

The measured target is Experiment 008/009/010's transport fraction `q_T(w, f) = ⟨ΔT(w, f), d̂_T⟩ / ⟨ΔT(pl_T, f), d̂_T⟩`
from the E-patch of `w` into frame `f`'s reference prompt, with the frame uninformative below `0.25 σ_T`.

## Inherited fixed elements

Model, runtime, instrumented path, E-patch, reference cues and prompts, weight-only `E(w)`, stage axes (plural
positive, re-estimated on the exposed frames at `explore`), the head weights, the P1 predictor and its OV identity
checks, the exact attribution functional `ρ_f` (Experiment 010), the results-state, ledger, lock, prediction-artifact,
and incident conventions — all exactly as in Experiments 009–010. Seeds: runtime `20260916`, control `20260924`.

## Exposed pool (calibration only; nothing is fitted)

The 63 cue tokens and 24 frames of Experiment 010 (and the 80 nouns for the replication checks). `explore` recomputes
`q_T` for all 63 × 24 (replicated against a committed extract of Experiment 010's recorded `q_T` within 1e-6),
estimates `d̂_T` and `d̂_E`, computes `g_E` for the 63 tokens, and fixes two tolerances from exposed residuals only:

- `τ_g = max(0.10, 3 × RMSE over the 63 tokens of (g_E(w) − q̄_T(w)))`, where `q̄_T(w)` is the token's mean over its
  informative frames;
- `τ_M = max(0.10, 3 × RMSE over the 63 × 24 informative pairs of (P1 fraction − q_T))` (as in Experiment 009).

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
- **Lock:** `g_E(w, T)` (and `g_∥`, `g_⊥`) for every fresh token and template, `d̂_T`, `d̂_E`, `γ₃ ⊙ m`, `τ_g`, `τ_M`, the
  exposed baseline statistics, and Experiment 009's frozen rank-1 transport rule's predictions for the fresh tokens as
  the reported baseline; `predictions.md` with one row per fresh token and template. `confirm` recomputes every
  prediction from the weights and the locked axes before any fresh prompt runs and refuses on any difference above
  1e-9.
- **Tier C (`confirm`, once):** for every fresh frame, the reference run with the head's internals, the plural cue's
  E-patch (normalization), and the sg/pl cue prompts (sanity precondition `d_full > 0` in ≥ 108/120 of the
  6 × 79 pairs); for every fresh token in its licensed frames, the E-patch with the head's internals captured: measured
  `q_T`, the P1 fraction, the exact attribution ledger (`f_E`, `f_∥`, `f_⊥`, `f_k`, net, gross), and the behavioral
  contrast (reported).

## Floors and outcome (frozen)

- **Precondition:** the fresh-frame cue effect holds and every fresh frame is informative at the head stage
  (`|⟨ΔT(pl_T, f), d̂_T⟩| ≥ 0.25 σ_T`); otherwise `PRECONDITION_FAILED` and nothing is judged.
- **Y1 — encoding read → transport (zero-parameter):** over the fresh tokens' means across their licensed frames,
  `g_E` versus measured `q̄_T`: Spearman ≥ 0.80 and MAE ≤ `τ_g`. Pass → `ENCODING_READ_PREDICTS_TRANSPORT`; fail →
  `ENCODING_READ_FAILS` (naming the floor). Experiment 009's rank-1 rule (fitted on 40 tokens) is reported beside it,
  never judged.
- **Y2 — the head's linear read-out on new cues:** over the fresh (token, frame) pairs, the P1 fraction versus
  measured `q_T`: Spearman ≥ 0.90 and MAE ≤ `τ_M`. Pass → `HEAD_P1_REPLICATED`; fail → `HEAD_P1_NOT_REPLICATED`.
- **Descriptive (no floor):** the fresh tokens' attribution ledger, with the sign of `f_⊥` for tokens whose locked `g_E`
  is ≤ 0.35 and the fraction of fresh tokens whose net layer change lies within the exposed range; the exposed
  baseline for gross change (median `G` 0.44 in Experiment 010) is stated so that no relay claim is made without it.
- Outcome = `Y1 | Y2`. Incidents (replication, identities, software defects) stop the phase, are recorded with their
  commit, and are never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- `g_E` depends on `d̂_T`, an axis estimated from exposed clean runs; it is weight-only given that axis, not
  weight-only in the absolute sense, and the design says so wherever it is named.
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

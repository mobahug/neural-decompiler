# Experiment 009: The Transport Rule of `L03.H04` — Mechanism Decomposition and a Prospective Test on Frozen New Cues

**Date:** 2026-09-18

**Status:** Revision 2 — conceptually approved at revision 1 subject to the three corrections under "Revision
history", which this revision makes (grammatical-compatibility policy, explicit exact decomposition, immutable
pre-confirm prediction artifact). No Experiment 009 directory, confirmation set, lock, or model run exists.
Experiments 005–008 are closed and are not amended by this document.

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

For frame `f` (query position `p_t`, cue position `p_c`) and head `h = L03.H04`: `r_k` is the layer-3 input residual
at key position `k` (`RESID_PRE.L3`), `ν_k = LN₃(r_k) = γ₃ ⊙ (r_k − mean(r_k)) / σ_k + β₃` with
`σ_k = sqrt(var(r_k) + ε)` (population variance over the `d_model` coordinates, `ε = 1e-5`), `v_k = ν_k W_V^h + b_V^h`,
`A_k` the attention weight of head `h` from query `p_t` to key `k`, and `T = Σ_k A_k v_k W_O^h` the per-head result
(TransformerLens `attn_result`, which excludes `b_O`). Superscripts `ref` and `patch` denote the clean reference run
and the E-patched run; `Δx = x^patch − x^ref`. The **frozen exact identity** is

```text
ΔT = Σ_k A_k^ref (v_k^patch − v_k^ref) W_O   +   Σ_k (A_k^patch − A_k^ref) v_k^patch W_O
```

(the second term uses the **patched** value vectors; the `ΔA·Δv` cross term therefore sits in the second term, not
in a hidden remainder). `b_V` cancels in every `Δv`. Three nested predictors of `ΔT` from the cue position alone, in
order of complexity:

- **P1 — fixed-normalization linear OV read-out:**
  `ΔT₁ = A_c^ref · [γ₃ ⊙ (Δr_c − mean(Δr_c)) / σ_c^ref] W_V W_O`, where `mean(Δr_c)` is the mean of the `d_model`
  coordinates of `Δr_c` (centering is linear) and the denominator is held at the reference run's scale `σ_c^ref`.
  This is the exact `Δν_c W_V W_O` scaled by `A_c^ref` when `σ_c^patch = σ_c^ref`; it is linear in the head's input
  change and reads one direction, `m = W_V W_O d̂_T`, in normalized coordinates.
- **P2 — exact LayerNorm at the cue position, reference attention:**
  `ΔT₂ = A_c^ref · (ν_c^patch − ν_c^ref) W_V W_O`. Adds the LayerNorm scale and mean nonlinearity at the cue position
  (norm-based gating: a large orthogonal component inflates `σ_c` and shrinks every component the head sees).
- **P3 — exact LayerNorm and the patched attention pattern, cue-position values only:**
  `ΔT₃ = A_c^patch v_c^patch W_O − A_c^ref v_c^ref W_O + Σ_{k≠c} (A_k^patch − A_k^ref) v_k^ref W_O`. Adds attention
  redistribution over unchanged other positions.
- **Remainder** `ΔT − ΔT₃ = Σ_{k≠c} A_k^patch (v_k^patch − v_k^ref) W_O` (follows from the identity above): value
  changes at positions other than the cue — only the adjective position after the cue in coordinated-adjective frames
  can change; it is identically zero in cue-final frames up to numerics.

Each level is computed from captured activations and weights with no fitted parameter; the identity
`ΔT₃ + remainder = ΔT` is checked in every run. **Reading of the outcomes.** P1 sufficient: given the residual arriving
at the head, a fixed-normalization linear `W_V W_O` read-out of the incoming cue residual is sufficient to explain the
selective transport — LayerNorm-scale and attention modulation are unnecessary; the opposing components the head
reads in opposite directions were constructed upstream, and the head merely reads them (it is not claimed that the OV
circuit generated them). P1 fails, P2 sufficient: LayerNorm-dependent gating at the cue position. P1–P2 fail, P3
sufficient: attention modulation matters. None sufficient: the cue-position head story is incomplete (the remainder's
share says whether other positions carry it). The **upstream** question is answered by the component patches traced
through the stages (M2 below).

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

The experiment tests the mechanism, not whether the model detects ungrammatical English, so every fresh cue must be
grammatical as a bare determiner-like word directly before a count noun in the fresh frames, with a **frozen,
deterministic grammatical-compatibility policy** that restricts a cue to the frames and noun forms it licenses. The
policy is fixed here, before any output is seen; where it restricts, the affected quantities are averaged over the
licensed subset and the subset is recorded in the frozen set.

- **Fresh cue tokens (at most 24), in five categories with quotas, chosen as the first eligible entries of the frozen
  candidate lists** (single token with a leading space in the pinned tokenizer; disjoint from the forty exposed tokens
  and from every exposed noun form; a prompt in every licensed fresh frame must round-trip):
  - `singular-selecting` (quota 3): `either`, `neither`, then `an`. `an` is licensed only in cue-final frames (in the
    coordinated-adjective frames it would precede a consonant-initial adjective) and, for every contrast-based
    quantity, only over nouns whose singular form begins with a vowel letter (`a, e, i, o, u`); it is included only if
    at least ten single-token exposed nouns qualify, otherwise the category is frozen with `either`, `neither`.
    Head-level quantities (`q_T`, the OV decomposition) do not involve nouns and use every licensed frame.
    Expectation: low transport (`q_T ≤ 0.35`).
  - `plural-numeral` (quota 6): `eleven`, `thirteen`, `fourteen`, `fifteen`, `sixteen`, `seventeen`, `eighteen`,
    `nineteen`, `twenty`, `thirty`, `forty`, `fifty`, `sixty`, `seventy`, `eighty`, `ninety` — numerals that stand
    alone as determiners (no bare `hundred`/`thousand`/`million`). Expectation: high transport (`q_T ≥ 0.65`).
  - `plural-quantity` (quota 5): `more`, `most`, `other`, `enough`, `certain`, `additional`, `extra`, `assorted`,
    `sufficient`, `innumerable` — grammatical before a bare plural count noun. Expectation: high transport.
  - `number-neutral` (quota 6): the possessives `my`, `your`, `his`, `her`, `our`, `their`, `its` — grammatical before
    singular and plural count nouns in every fresh frame (`which`, `whose`, `what` are dropped: declarative frames do
    not license them). Expectation: unconstrained — the sharpest prospective test of the graded rule.
  - `bare-adjective` (quota 4): `small`, `blue`, `new`, `cold`, `green`, `cheap`, `warm`, `dark`. A bare adjective
    before a count noun forces the plural (`carries small boxes`, not `carries small box`), so these are **not** neutral
    controls — Experiment 007's `big`/`red`/`old`/`fresh` behaved plural-like for this reason. Expectation: high
    transport, by the bare-noun-phrase rule rather than by number semantics.
  A category whose quota cannot be met by eligible entries is frozen with the entries available (recorded); the set is
  frozen even if it has fewer than 24 tokens. Expectations are used only in the category-level check below.
- **Fresh frames (6, two per template), literal:** cardinal `The shelf carries {cue}`, `The ledger names {cue}`;
  quantifier `The manual describes {cue}`, `The bulletin mentions {cue}`; coordinated-adjective
  `Ida and Tomas counted {cue} shiny`, `Yusuf and Petra wrapped {cue} thin`. Every fresh cue except `an` is licensed
  in every fresh frame (possessives, numerals, quantity words, and adjectives all precede an adjective grammatically).
  Their texts must differ from every exposed frame text; positions `p_c`, `p_t` follow the template; the original cue
  tokens are used for their own cue prompts.
- **Nouns:** the eighty exposed nouns (79 single-token); for `an`, the vowel-initial subset. The prediction targets are
  per-(token, frame) means over the licensed nouns; no claim is noun-specific, and Experiments 006–007 already tested
  noun generalization. This is a stated limit.
- The set — tokens with categories and expectations, frames, the licensed frame and noun subsets per token, and the
  cue prompts — is committed (`confirmation-v1.json`, tokenizer-only, digest recorded) before `explore`; `confirm`
  refuses if any fresh prompt appears in the ledger earlier.

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

### Lock — the immutable pre-confirm prediction artifact

The visible sequence is: forty exposed cues → rules fitted and gated → rules frozen → predictions generated for
**every** fresh cue → prediction artifact committed with its digest → only then `confirm` → scoring against the frozen
predictions. The `lock` phase writes two files from the exported rules without running any fresh prompt:

- `candidate-lock.json` — the frozen confirmation set digest, the stage axes, `d̂_T`, `m`, the locked mechanism level
  with its exposed MAE/R² and `τ_M = max(0.10, 3 × RMSE of its exposed residuals)`, the transport rule (`v`, `β_T`,
  digest of its exported tensors, τ₂ = max(0.10, 3 × LOCO RMSE)), the contrast rule (rank, program digest, τ₃), the
  category expectations, every floor, and the per-(fresh token, fresh frame) predictions `q̂_T` and `Δĉ` (the latter
  per licensed noun and as the mean over licensed nouns), with each token's six-frame means;
- `candidate-predictions.md` — the human-readable table (token, category, expectation, predicted `q_T` per frame and
  mean, predicted contrast shift per frame and mean, the rule digests), whose sha256 is recorded in the lock.

Installing both as `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
`confirm` (i) validates the lock against the results state and the digests, (ii) recomputes every locked prediction
from the on-disk rules **before any fresh prompt runs** and refuses (nothing executed) if any differs by more than
1e-9, and (iii) scores the fresh measurements only against the locked numbers. Nothing measured at `confirm` can
influence a prediction.

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
- **Transport-rule axis (Y2):** over the fresh tokens' means across their licensed frames, `q̂_T` versus measured
  `q_T`: Spearman ≥ 0.80, MAE ≤ τ₂, and the category check — all but at most one singular-selecting token measured
  `≤ 0.35`, and at least 80% of the plural-numeral, plural-quantity, and bare-adjective tokens measured `≥ 0.65`
  (`exact_count_floor(0.8, n)`), with the rule's predictions meeting the same counts. Pass → `TRANSPORT_RULE_PREDICTED`;
  fail → `TRANSPORT_RULE_FAILED`, naming the failing floor.
- **Contrast-rule axis (Y3):** Experiment 007's Y1 floors on the fresh tokens (Spearman ≥ 0.80, MAE ≤ τ₃,
  confident-token sign rule in ≥ 5 of 6 frames). Pass → `CONTRAST_RULE_PREDICTED`; fail → `CONTRAST_RULE_FAILED`.
  The Experiment 007 sixteen-token program's fresh performance is reported beside it.
- **Non-additivity stage (M2, descriptive on both pools):** the modal `ν*` over the singular-selecting and the
  exposed suppressed tokens is reported with its agreeing fraction; it enters no label.
- The outcome is the triple of axis labels (with `NOT_LOCKED` where a gate failed). Incidents (replication mismatch,
  identity failures, OV identity failure, software defects) stop the phase, are recorded with their commit, and are
  never an outcome label; a confirm incident permits no re-run in this protocol version.

## Interpretation limits

- Only two or three singular-selecting determiners remain unexposed in English at this position (`either`, `neither`,
  possibly `an`), so that category check is weak; the possessives are the informative prospective cases. The
  bare-adjective category tests the bare-noun-phrase plural rule, not number semantics. Nothing generalizes beyond
  the pinned checkpoint, the three templates, single-token regular nouns, and the tokens tested.
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

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
implementation review → `explore` once → `lock` (the prediction artifact) → the user's lock commit → **the reviewer's
sign-off** → `confirm` once → report. No token, frame, or prediction may be changed after the lock; no fresh prompt
runs before `confirm`.

## Revision history

- **Revision 1** (commit `71e5ce9`): initial draft. Reviewed: conceptually approved with three corrections — (1) the
  fresh set risked ungrammatical constructions (`an` before consonant-initial nouns and adjectives, bare `million`,
  bare adjectives with singular count nouns, wh-determiners in declarative frames); (2) the "exact" head decomposition
  left the `ΔA·Δv` cross term ambiguous and P1's centering and scale implicit; (3) the predictions must exist as an
  immutable committed artifact before `confirm`.
- **Revision 2**: (1) a frozen grammatical-compatibility policy — `either`, `neither`, and `an` (cue-final frames and
  vowel-initial nouns only, included only if ten such nouns exist) as singular-selecting; numerals that stand alone
  as determiners only; possessives as the number-neutral set; adjectives re-labelled `bare-adjective` with the
  plural expectation the bare-noun-phrase rule implies; licensed frame and noun subsets recorded in the frozen set;
  the 474-pair cross-product is not forced. (2) The identity is frozen as
  `ΔT = Σ A^ref Δv W_O + Σ ΔA v^patch W_O`, with P1's `mean(Δr_c)` centering and reference-scale denominator explicit
  and the remainder derived from it. (3) The lock phase writes the machine-readable lock and a human-readable
  prediction table with recorded digests; `confirm` reproduces every locked prediction before any fresh prompt and
  scores only against the locked numbers. (4) P1 success is worded as "fixed-normalization linear OV read-out of the
  incoming residual is sufficient; LayerNorm-scale and attention modulation are unnecessary", without implying that
  the OV circuit generated the opposing components. (5) The reviewer's sign-off precedes `confirm`.

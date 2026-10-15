# Experiment 102: blind transfer of an explicit read-discovery procedure to Pythia-160M, with specificity controls and simple baselines — design

**Status:** revision 1, 2026-10-15, **proposed, not decided.** This document contains no code, has run no model, freezes
no frame and computes no tolerance. It was written by the operator (Claude, an AI) on the user's authorization of
2026-10-14 to *design* the next experiment and to preregister its hypotheses, controls, thresholds, failure criteria and
untouched confirmation data before any run. It has an independent AI design review (companion file
[2026-10-15-experiment-102-design-reviews.md](2026-10-15-experiment-102-design-reviews.md)); that review is not a human
validation. Nothing here runs until the user decides §14 and authorizes each phase of §10.

Inputs: Experiment 101's closed record (`experiments/101-read-component-intervention/evidence/closure-2026-10-14.md`,
branch `exp101-phases`, `a9b6b33`); the critical assessment
(`research/synthesis/exp101-critical-assessment-2026-10-14.md`); two AI method surveys of 2026-10-14 (the 160M
discovery survey and the baselines-and-controls survey, `research/synthesis/exp102-surveys/`, summarized where used); and two post-hoc, exploratory checks of
101's saved states and the 70M weights (`research/synthesis/checks/exp101_posthoc_{directions,construction}.{py,out}`).
Numbers from those checks are labelled *post hoc*; X101-1 (directions) and X101-2 (construction) name them, as in
the C004 draft.

## 1. Why this experiment

**What 101 showed.** In 36 new coordinated frames of pinned Pythia-70M-deduped, setting the read coordinate
`r(x) = ⟨x, w_c⟩` of the residual entering layer 3 at the cue position — `w_c` being the covector through which
`L03.H04` writes onto its number axis — removed and installed the `one`→`two` shift against an equal-length off-read
control in 36/36 frames (`READ_COMPONENT_EFFECTS_AS_PREDICTED`). The draft claim C004 records this, narrowly.

**What 101 cannot show.** Whether the *procedure* that found `w_c` is a method. The 70M answer was found over twenty
experiments with human choices at every step: the head was selected on two coordinated development frames after a
screen had anticipated it, and was then kept as a constant (160M survey, §3). A method generalizes if, written down
and frozen, it finds a causal variable in a model where nobody knows the answer. 101 also leaves the specificity
questions open (critical assessment §1–§5): no site or layer control, an axis estimated on the tested pair, and no
comparison with the simple directions any practitioner would try first.

**The user's priority** (2026-10-14): "prioritize proving that our decompilation method generalizes, rather than
continually improving the explanation of one grammatical behavior"; compare against simple baselines; make the method
for finding the circuit explicit; assess individual predictions, not only aggregates.

**The question of 102.** Does an explicit, mechanical discovery procedure (MD-1, §4), frozen before any
number-agreement prompt has run on Pythia-160M-deduped, return a read variable that passes preregistered causal tests on
fresh frames in 160M — and is that variable specific to its site, better than the simplest data-derived direction at
equal edit length, robust to re-estimating its axis without the tested pair, and sufficient, frame by frame, to predict
the effect of other edits? The same battery runs on 70M with C004's variable, giving C004 its first Level-4 tests.

**What 102 is not.** It does not test a second behavior, a second model family, other templates or a population of
frames. If 160M transfers, the next test of the method is a different behavior (§11).

## 2. Hypotheses

Each hypothesis is stated with what would refute it. `T` is the read set MD-1 returns (one head in 70M), `L*` its layer,
`V` the span of the read covectors (§5).

| id | hypothesis | model(s) | refuted, within scope, by |
|---|---|---|---|
| **H1** (method transfer) | MD-1 finds a compact read in 160M — consumption concentrated at one layer, at most 3 heads, replicated on a held-out half of the discovery frames — and on fresh frames its read coordinate removes and installs the `one`→`two` shift beyond an equal-length off-read control | 160M | a discovery stop (§4), or R or I failing on the fresh frames |
| **H2** (site specificity) | the same edit vector applied one layer later, at `RESID_PRE.(L*+1)` and the same position, removes and installs at least 0.5 less of the shift than at `L*` | both | S failing |
| **H3** (value over a naive direction) | at equal edit length, the read beats the frame's own cue-change direction by at least 0.10 of the shift in both removal and installation | both | B failing while the read and the cue change are not near-collinear (§11) |
| **H4** (held-out axis) | a read built from an axis estimated without the cue `two` (from `three` and `many` against `one`) removes and installs the `one`→`two` shift beyond its own matched control | both | H failing |
| **H5** (projection sufficiency) | an edit's effect at the site depends only on how far it moves the read coordinate: the equal-length cue-change edit's effect is predicted, frame by frame, from its realized change in `r` through that frame's own measured read dose–response | both (k = 1) | PS failing: a second effective channel exists at the site |
| **H6** (per-item completeness) | removal and installation are complete noun by noun, not only in the noun mean | both | the per-noun criteria of §8.5 failing (secondary, no force on the primary label) |

**What counts as the method generalizing (H1 + H2).** A 160M label of `METHOD_TRANSFERS` (§8.4) requires a discovery
that completes, R and I, and S. B, H and PS qualify it; they do not gate it. Whatever `T` MD-1 returns counts; no
human may substitute another head.

**What does not count.** Finding "the analogue of `L03.H04`" by looking through 160M's heads; any exploration of 160M
number-agreement prompts before MD-1 and its thresholds are frozen; any re-estimation after the lock.

## 3. Models, behavior and data

### 3.1 Models

| | 70M | 160M |
|---|---|---|
| checkpoint | `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | `EleutherAI/pythia-160m-deduped`, revision `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e` |
| layers × heads, d_model | 6 × 8, 512 | 12 × 12, 768 |
| role in 102 | development model: MD-1's retrodiction (§4.4); C004's variable under the specificity battery | the held-out model: MD-1 runs once, blind to its outcome |

Both snapshots are in the local HF cache (no download); 160M has no `refs/main` and is loaded by commit SHA. The
tokenizer files are byte-identical (`tokenizer.json` sha256 `c24618a1…f25624`), so every token id, position and
tokenizer-only eligibility rule carries over. TransformerBridge, float32 on CPU, 4 threads, the pinned library versions
(transformer_lens 3.9.0, torch 2.14.0, transformers 5.17.0), as in 101. No number-agreement prompt has ever run on
160M (160M survey §1).

### 3.2 The behavior, cues, nouns and outcome

As in 101: coordinated frames `NAME and NAME VERBED {cue} ADJ`, cue at `p_c`, adjective at `p_t = p_c + 1`; reference
cue `one` (581), primary plural cue `two` (767), stratum `three` (1264) and `many` (1142); the 79 scorable **exposed**
nouns; `c = log P(singular) − log P(plural)` at `p_t`, and `Δc` its change from the reference prompt averaged over the
nouns. The template, cue pair and nouns were selected on 70M; the scope says so (§11).

### 3.3 The three data sets

| set | frames | status | used for |
|---|---|---|---|
| **D** (discovery and calibration) | the 36 exposed coordinated frames of 020's pool (101's calibration set) | spent on 70M; **never run on 160M** | MD-1 on both models; calibration of both models |
| **C** (confirmation) | 101's unused candidates **37–60** (24 frames) and the first **12** eligible of the new candidates 61–96 (§15) | **never executed on any model** | the one confirmation run per model |
| **Reserve** | the new candidates not taken into C (73–96 if 61–72 are all eligible at the freeze) | never executed; **untouched by 102** | reserved for a later replication (§11) |

- **D's split.** D1 and D2 are 18 frames each: the frames sorted by `sha256("102|split|" + frame_id)`, the first 18
  forming D1. Fixed now; no other split is allowed.
- **Why spent-on-70M frames serve 160M's discovery.** No 160M outcome on them has been observed; MD-1 is mechanical,
  so no choice is left for knowledge of 70M's per-frame results to bias. C is fresh for both models.
- **C's freeze** is tokenizer-only and follows 101's §11 rules (§15), with one addition: an adjective may not be any of
  101's 60 candidates' adjectives.

## 4. The discovery procedure MD-1

MD-1 is code with fixed thresholds. Its version, source digest and thresholds enter the lock; it runs once per model on
D, under a guarded launcher, and its outputs are frozen before any C prompt runs. Its rules were chosen with knowledge
of 70M's values (the threshold column), so its 70M retrodiction is a consistency check, not evidence of generality.

All patching is bidirectional on the `one`/`two` pair: *installation* patches a site of the `one` run with the `two`
run's value, *blocking* patches the `two` run with the `one` run's value. For a patch `π`, its per-frame recovery is
`½·[Δc(install π)/Δc_full + (1 − Δc(block π)/Δc_full)]`; a statistic over frames is the **median** of per-frame values,
over frames with `Δc_full ≤ −1` nat.

| step | data | rule | stop label | 70M reference value |
|---|---|---|---|---|
| (a) behavior | D | `Δc_full ≤ −1` nat in ≥ 30 of 36 frames | `BEHAVIOR_ABSENT` | 101 calibration: 36/36 |
| (b) residual profile | D | `R_c(L)`: recovery of patching `RESID_PRE.L` at `p_c`, L = 0 … n_layers − 1, and `R_c(n_layers) := 0`; `R_t(L)` likewise at `p_t` (descriptive). `R_c(0)` is an identity (G0) | — | 005 A3, 2 frames: `R_c(3)` 0.956, `R_c(4)` 0.125 |
| (c) layer | D | `Δ_L = R_c(L) − R_c(L+1)`; `L* = argmax Δ_L` (ties: the lower layer). **CONCENTRATED** if `Δ_{L*} ≥ 0.5` and `R_c(L*) ≥ 0.7` | `CONSUMPTION_NOT_CONCENTRATED` | 0.831; 0.956 |
| (d) heads | rank on D1, verify on D2 | candidates: heads of layer `L*` whose attention from `p_t` to `p_c` (mean of the two runs, median over D1) is ≥ 0.3; rank them by their singleton recovery `s_h` at `p_t` on D1 (ties: canonical order). `T` = the shortest prefix, of at most 3 heads, whose **joint** patch at `p_t` recovers ≥ 0.70 on D1 | `NO_COMPACT_READ` (no prefix of ≤ 3 reaches 0.70) | `L03.H04`: attention 0.747, head-alone 0.816 (005, 2 frames) |
| (d′) verify | D2 | `T`'s joint recovery on D2 ≥ 0.60 | `DISCOVERY_NOT_REPLICATED` | — |
| (e) read | D (`one`/`two`) | for each `h ∈ T`: `d̂_T^h` = unit mean over D of `h`'s output change at `p_t` (`two` minus `one`); `w_c^h = C(γ_{L*} ⊙ W_V^h W_O^h d̂_T^h)`; `V` = span of the `w_c^h` (orthonormalized in canonical head order) | — | 011's recipe; X101-2: `w_c ≈ Kδ`, cos 0.95 |
| (e′) held-out axis | D (`three`, `many` vs `one`) | `d̂_T^{h,(−)}` = unit mean of `h`'s output change for `three` and `many` against `one`; `V^(−)` likewise | — | — |
| (e″) P1 diagnostic (k = 1) | D × {`two`, `three`, `many`} (108 pairs) | R² between `⟨ΔT, d̂_T⟩` and `(A_c/σ_c)·⟨Δx, w_c⟩`: `P1_SUFFICIENT` if ≥ 0.80, else `P1_NOT_ESTABLISHED` (recorded; no stop) | — | 009's P1 R² 0.909 |
| (f) freeze | — | the discovery record binds every value above, `L*`, `T`, the `d̂_T`, `w_c` and `V` in float64 with digests | — | — |

**Reported, not decisive** (both models): the full 144-head (or 48-head) singleton ranking on D1 and on D2; `R_t(L)`;
the value-path share of `T`; the attention and singleton scores of every head; which head the **naive selections**
would pick — the head with the highest attention from `p_t` to `p_c` in any layer, the head with the highest direct
logit attribution to `c` at `p_t`, and the highest singleton `s_h` in any layer. If the attention-selected head differs
from `T`'s first head, its read (the same recipe at its own layer) enters the confirmation as the secondary arm pair
ATTN-READ (§6): the causal ranking is thereby compared with the naive one on the outcome itself. In 70M the
attention-selected head would be `L05.H07` (0.999), not `L03.H04` (005, A4).

### 4.4 Step 0: the 70M retrodiction

MD-1 runs unchanged on 70M with D before any 160M pass. It **passes** if (a) passes, `L* = 3` is CONCENTRATED,
`T = {L03.H04}` and (d′) passes. `cos(w_c^{MD-1}, w_c^{011})` is reported: at ≥ 0.95, "MD-1 recovers 011's read"; below,
"MD-1 recovers the head and the site; its axis differs" (011's axis also used `each`/`several` in cue-final frames,
which MD-1 cannot use). A failure on the head, layer or `k` means MD-1 is not the procedure that found the 70M answer:
MD-1 is revised and re-run on 70M, every version recorded, before any 160M pass. No 160M prompt runs until a version
passes; a revision after any 160M pass is forbidden.

## 5. The read variable and the edits

For a frame, at `RESID_PRE.L*` and `p_c`: `x_ref`, `x_pl` the plain runs' residuals; `Δx = x_pl − x_ref`; `C` the
centring projector; `P_V` the orthogonal projector onto `V`.
- **Read part** `R = P_V Δx` (for k = 1, `(Δr/‖w_c‖²)·w_c` with `Δr = ⟨Δx, w_c⟩`), of length `s = ‖R‖`.
- **Off-read part** `off_c = CΔx − R`, `û_off = off_c/‖off_c‖`; overshoot flagged when `‖off_c‖ < s`, as in 101.
- **Cue-change direction** `Δx̂_c = CΔx/‖CΔx‖`.
- **Realized read change** of an edit `e` on the `two` prompt: `ρ(e) = −⟨e, w_c⟩/Δr` (the fraction of `Δr` removed);
  on the `one` prompt `ρ(e) = ⟨e, w_c⟩/Δr` (the fraction installed). Computed in float64 from the float32 edit (k = 1).
- **Held-out** `R^(−) = P_{V^(−)} Δx`, `s^(−)`, `û_off^(−)` likewise.
- **70M's variable** is C004's: 011's `w_c` at `RESID_PRE.L3` (G1 against 101's lock). MD-1's 70M read enters as two
  bridge arms (§6).

Every edit is an exact REPLACE of one residual vector, through the existing intervention with its landing and
zero-outside-change checks; every arm is one forward pass reading all 79 contrasts at `p_t`.

## 6. The arms

All arms run in every frame of D (calibration) and C (confirmation), for the cue `two`; the stratum arms for `three`
and `many`.

| arm | prompt | edit | family |
|---|---|---|---|
| REF, PL, NOOP | `one`, `two`, `two` | none; none; `x_pl` replaced by itself | references; G3 |
| NEC-READ, SUF-READ | `two`, `one` | `x_pl − R`; `x_ref + R` | **R, I** |
| NEC-OFF, SUF-OFF | `two`, `one` | `x_pl − s·û_off`; `x_ref + s·û_off` | **R, I** (matched control) |
| NEC-DX, SUF-DX | `two`, `one` | `x_pl − s·Δx̂_c`; `x_ref + s·Δx̂_c` | **B, PS** |
| NEC-Lq, SUF-Lq, q ∈ {0.25, 0.5, 0.75} | `two`, `one` | `x_pl − q·R`; `x_ref + q·R` | **PS** (the dose–response ladder) |
| NEC-SITE, SUF-SITE | `two`, `one` | `R` subtracted from / added to `RESID_PRE.(L*+1)` at `p_c` (the same vector, one layer later) | **S** |
| NEC-HREAD, SUF-HREAD, NEC-HOFF, SUF-HOFF | `two`, `one` | as READ and OFF with `R^(−)`, `s^(−)`, `û_off^(−)` | **H** |
| NEC-AMP, SUF-NEG | `two`, `one` | `x_pl + R`; `x_ref − R` | D (secondary) |
| PC-SWAP-PL, PC-SWAP-REF | `two`, `one` | the whole `x(p_c)` from the other prompt | secondary |
| NEC/SUF-MD, -PC1, -LDA, -GRAD | `two`, `one` | length `s` along the frozen mean cue change, the first principal component of D's cue changes, a shrinkage-LDA direction, and the mean gradient of `c` at the site over D — each frozen from D at calibration | secondary baselines |
| NEC-ISO-j, SUF-ISO-j, j = 1 … 8 | `two`, `one` | length `s` along mean-zero unit directions orthogonal to `V` (SHA-256 counter-mode, tag `102|iso|{model}|{frame_id}|{j}`) | secondary |
| NEC-W^h for each other head `h` of layer `L*` | `two` | the recipe of §4(e) applied to head `h` (its own `d̂_T^h` from D), its coordinate set to the reference's | secondary ("dead-salmon" check of the recipe) |
| HEAD-NEC, HEAD-SUF (+ -V variants) | `two`, `one` | `T`'s outputs at `p_t` replaced by the read edit's exact effect on them (as 101) | secondary |
| ABH-NEC, ABH-SUF | `two`, `one` | the read edit at `p_c` with `T`'s outputs at `p_t` clamped to their unedited values (all but the head) | secondary |
| NEC-PT | `two` | `R` subtracted at `RESID_PRE.L*`, `p_t` | secondary site |
| NEC-UP | `two` | `R` subtracted at `RESID_PRE.(L*−1)`, `p_c` (skipped if `L* = 0`); the realized `ρ` at `L*` captured | secondary (mediation) |
| ATTN-READ (NEC, SUF) | `two`, `one` | the attention-selected head's read, at its own layer's input at `p_c` (only if it differs from `T`) | secondary (discovery baseline) |
| BRIDGE (NEC, SUF), 70M only | `two`, `one` | READ with MD-1's 70M `w_c` | secondary |
| stratum: PL, NEC/SUF-READ, NEC/SUF-OFF | `three`, `many` vs `one` | as for `two` | secondary |

**Per-frame fractions** (each over the frame's `Δc_full`): `F(arm) = Δc(arm)/Δc_full`; `F_rem = F(NEC-READ)`,
`F_ins = F(SUF-READ)`, and so on.

**Compute.** About 80 runs per frame and model (k = 1): ≈ 2,900 per model per phase. 70M ≈ 0.07–0.19 s per run, 160M
an estimated 0.2–0.5 s (to be measured in a smoke run). With MD-1 (≈ 6k 70M passes; ≈ 12k 160M passes) the experiment
needs about 2–4 h of CPU in all. No GPU, no cloud cost, no download.

## 7. Calibration on D

After the 160M discovery record is committed, a calibration phase runs every arm of §6 on D, for both models. These
data are spent; the phase is exploratory.
- **What it freezes:** the baseline directions (MD, PC1, LDA with shrinkage λ = tr(Σ)/d as in X101-1, GRAD from one
  backward pass per D frame), the other heads' axes `d̂_T^h`, and the PS tolerances `ε_rem`, `ε_ins`: the nearest-rank
  90th percentiles (33rd of 36) of `|F(NEC-DX) − F̂_rem|` and `|F(SUF-DX) − F̂_ins|` on D (§8.3), with a floor of 0.02.
  If either exceeds **0.20**, PS is uninformative for that model and is not tested.
- **What it reports:** `Δr`, `s`, `‖off_c‖`, overshoot flags, natural ranges of `r`, every direction's cosine with `ŵ`
  and its `ρ`; every family's exposed distribution and its sign-test power at the observed share.
- **Revision triggers,** the only ones: a gate failure; `‖off_c‖ < s` in more than 6 of 36 D frames; a non-finite
  fraction; `ε > 0.20` (which makes PS uninformative; it does not revise anything else). A revision driven by the size
  of any family's exposed result is forbidden.

## 8. Statistics and decision rules

The unit is the frame, n = 36 (C), cue `two`, per model. A frame with `Δc_full > −1` nat counts as a failure in every
count and enters every median as the worst value; ties count as failures. Each count's p-value is its exact binomial
tail under a symmetric sign. Medians have frame-bootstrap 95 % intervals (10,000 resamples, seed 20261015), reported
with no decision force. The two models are separate families: their questions are different (method transfer;
C004's specificity).

### 8.1 Precondition

At least **30 of 36** C frames with `Δc_full ≤ −1` nat; otherwise `BEHAVIOR_PRECONDITION_NOT_MET`.

### 8.2 The families (serial gatekeeping)

**Stage 1 — R and I** (as 101), by Holm at 0.025: **26 and 25 of 36**.
- R: `N_f = F(NEC-OFF) − F(NEC-READ)`; count of `N_f > 0`; median `N ≥ 0.05`.
- I: `S_f = F(SUF-READ) − F(SUF-OFF)`; count of `S_f > 0`; median `S ≥ 0.05`.

**Stage 2 — S, B and H**, tested only if R and I both pass, by Holm at 0.025 over the three: **26, 26 and 25 of 36**
(tails 0.0057, 0.0057, 0.0144). Testing a second family at the full level only after every hypothesis of the first is
rejected keeps the familywise error at 0.025.
- S: `Z_f = min(F(NEC-SITE) − F(NEC-READ), F(SUF-READ) − F(SUF-SITE))`; count of `Z_f > 0`; median `Z ≥ 0.50`.
- B: `B_f = min(F(NEC-DX) − F(NEC-READ), F(SUF-READ) − F(SUF-DX))`; count of `B_f > 0`; median `B ≥ 0.10`.
- H: `H_f = min(F(NEC-HOFF) − F(NEC-HREAD), F(SUF-HREAD) − F(SUF-HOFF))`; count of `H_f > 0`; median `H ≥ 0.05`.

A family passes if Holm rejects its count's null and its median meets its floor; a failed floor does not change another
family's threshold.

### 8.3 Stage 3 — projection sufficiency (PS), an accuracy family

Tested only if R and I pass and k = 1, for each model.
- **The prediction.** The frame's removal ladder `{(0, 1), (0.25, F(NEC-L25)), (0.5, F(NEC-L50)), (0.75, F(NEC-L75)),
  (1, F_rem)}` is interpolated linearly at the realized `ρ(NEC-DX)`, giving `F̂_rem(DX)`; the installation ladder
  `{(0, 0), …, (1, F_ins)}` at `ρ(SUF-DX)` gives `F̂_ins(DX)`. A `ρ` outside [0, 1] counts as a disagreement.
- **A frame agrees** if `|F(NEC-DX) − F̂_rem(DX)| ≤ max(ε_rem, 0.02)` and `|F(SUF-DX) − F̂_ins(DX)| ≤ max(ε_ins, 0.02)`.
- **PS passes** if at least **27 of 36** frames agree (101's P count).
- The same statistic is reported, with no force, for every other direction of §6 and for NEC-UP (with its realized
  `ρ` at `L*`).

### 8.4 Labels

Per model, a compound label `PRIMARY | S | B | H | PS`.

| outcome | primary component |
|---|---|
| any gate fails | `NOT_INTERPRETABLE` |
| precondition fails | `BEHAVIOR_PRECONDITION_NOT_MET` |
| neither R nor I | `READ_EFFECTS_NOT_ESTABLISHED` |
| R only / I only | `READ_REMOVAL_ONLY` / `READ_INSTALLATION_ONLY` |
| R and I | `READ_EFFECTS_ESTABLISHED` |

Stage-2 and stage-3 components (only after `READ_EFFECTS_ESTABLISHED`; otherwise `NOT_REACHED`): `SITE_SPECIFIC` /
`SITE_SPECIFICITY_NOT_ESTABLISHED`; `BASELINE_EXCEEDED` / `BASELINE_NOT_EXCEEDED`; `HELD_OUT_AXIS_TRANSFERS` /
`HELD_OUT_AXIS_NOT_ESTABLISHED`; `PROJECTION_SUFFICIENT` / `PROJECTION_SUFFICIENCY_NOT_CONFIRMED` /
`PROJECTION_SUFFICIENCY_NOT_EVALUABLE` / `PROJECTION_SUFFICIENCY_NOT_TESTED` (k > 1).

**The 160M method verdict:**

| outcome | verdict |
|---|---|
| (a) fails on D | `BEHAVIOR_ABSENT_160M` (the method is not testable on this behavior in 160M; nothing follows about it) |
| a discovery stop | `DISCOVERY_STOPPED_<NOT_CONCENTRATED \| NO_COMPACT_READ \| NOT_REPLICATED>` |
| C precondition fails | `BEHAVIOR_PRECONDITION_NOT_MET_ON_C` |
| R or I fails | `METHOD_READ_NOT_CAUSAL` |
| R, I pass; S fails | `METHOD_READ_CAUSAL_NOT_SITE_SPECIFIC` |
| R, I, S pass | `METHOD_TRANSFERS`, with the B, H and PS components |

No label says that a variable has no causal role or that the method cannot work; a failure is scoped to MD-1 as
specified, this behavior and these models.

### 8.5 Individual predictions (preregistered, secondary label component)

Recorded as `PER_ITEM_COMPLETE` or `PER_ITEM_INCOMPLETE` per model; they cannot change the primary label or the
verdict.
- **Noun-level completeness.** Over cells (frame, noun) with the noun's own PL shift `≤ −1` nat: the share with
  `|F_fn(NEC-READ)| ≤ 0.2` and the share with `|F_fn(SUF-READ) − 1| ≤ 0.2`, each **≥ 0.80**; and at most **4** of 79
  nouns whose median `F_fn(SUF-READ)` across frames is below 0.8 with a frame-bootstrap upper bound below 0.9.
- **Noun-level PS** (k = 1): the share of cells whose `F_fn(NEC-DX)` lies within `ε_noun` of the noun's own ladder
  prediction, `ε_noun` set like `ε` at the cell level on D; reported against **0.75**.
- **Frame-level rank agreement:** Spearman across frames of `F(NEC-DX)` against `F̂_rem(DX)`, and of `A_amp` against
  the ladder's slope at `ρ = 1`; descriptive.

### 8.6 Power

If 80 % of frames show the predicted sign, a family at 26/36 passes with probability 0.91 and at 25/36 with 0.96; at
75 %, 0.73 and 0.83; at 70 %, 0.47 and 0.61. For 70M, 101's exposed calibration and confirmation predict R, I, S and B
far above these thresholds (*post hoc* estimates for B: removal along the cue change leaves about 0.75 of the shift,
installation produces about 0.6, against ≈ 0 and ≈ 1 along the read). For 160M nothing is known; that is the point.

## 9. Gates

- **G0, discovery identities.** Patching `RESID_PRE.L0` at `p_c` reproduces the other prompt's `Δc` bit for bit (both
  directions, every D frame); the D1/D2 split reproduces from its rule; every discovery number recomputes from the
  saved measurements bit for bit.
- **G1, geometry.** Each `w_c^h` recomputed from the weights and the recorded `d̂_T^h` within 1e-9; `V` orthonormal
  within 1e-12; every ISO and baseline direction unit and mean-zero within 1e-12, ISO orthogonal to `V`; 70M's `w_c`
  equal to 101's within 1e-9.
- **G2, the edits.** Landing bit for bit, zero change outside the slice. In float64 from the float32 vector: realized
  read coordinates within 1e-4 of target; centred lengths within 1e-4 relative; parallel/orthogonal as specified within
  1e-4; realized `ρ` recorded for every edit.
- **G3, the hook path.** NOOP reproduces PL bit for bit; the next-layer residual at the edited position equals `x′` plus
  the block's attention and MLP outputs within 1e-4 (max-abs).
- **G5, head arms.** Recomputed head outputs reproduce the captured ones: attention rows within 1e-4, outputs along
  `d̂_T` within 1e-3 (017's tolerances; the 160M calibration reports the realized margins, and a 160M failure is an
  incident, not a reason to loosen).
- **G6, determinism.** Every result recomputes from the saved measurements bit for bit at the report; the discovery
  record's digest is checked before calibration, the lock's before confirm.
- **G7, isolation and accounting.** Every run has a model-scoped, arm-tagged key `model|frame_id|word|token_id|arm`,
  executed exactly once per phase; no C key appears in any earlier ledger of either model (005–025, 101, and 102's own
  retrodiction, discovery and calibration ledgers); the reserve's keys appear in none of 102's ledgers.

## 10. Phases, authorizations and human commits

Each phase runs once, with a read-only AI review between phases; **each phase that runs a model needs the user's
explicit authorization**.
1. **Design** (this document): independent AI review; revisions; the user's decision on §14. An independent human expert
   review is recommended here, before anything is frozen.
2. **Plan and implementation:** new modules only (the pinned modules of 101 and 022–025 are not edited); a runner with
   tier-A and tier-B tests on fakes; a 160M contract test (12/12/768/3072, the parallel block identity, NOOP) on a
   neutral, non-experimental prompt; canonical component labels (`L10.H03`, never `L010.H03`); model-scoped ledgers.
   Implementation review.
3. **Freeze C** (tokenizer only): `confirmation-v1.json`; committed.
4. **Retrodiction** on 70M (spent D; *authorization 1*): MD-1 must pass §4.4. Review.
5. **160M discovery** (MD-1 once on D; *authorization 2*): the discovery record; an integrity-only review (did MD-1 run
   as frozen?). **The user commits the discovery record.** A discovery stop ends the 160M part here.
6. **Calibration** of both models on D (*authorization 3*): the frozen directions and `ε`; revision triggers only.
   Review.
7. **Lock** (weights and committed records only): binds MD-1's version and outputs, C, every direction, `ε`, §8 and
   §9. **The user commits the lock.**
8. **Confirm** (*authorization 4*): each model once on C, under a guarded launcher (101's confirm launcher as template).
9. **Report and closure,** with an independent AI confirmation review that recomputes every label.

No phase is repeated. A failure is reported as a failure; no frame is replaced, no threshold is revised after an
outcome is seen.

## 11. What each outcome would mean

| 160M outcome | what it would support | what it would not |
|---|---|---|
| `METHOD_TRANSFERS` with `BASELINE_EXCEEDED`, `PROJECTION_SUFFICIENT` | a frozen, explicit procedure found, in a model where the answer was unknown, a site-specific one-dimensional read that carries the tested shift, beats the naive cue-change direction per unit length, and predicts other edits frame by frame | generality to other behaviors, models or templates; that the read is the model's algorithm |
| `METHOD_TRANSFERS`, `BASELINE_NOT_EXCEEDED`, cue change and read nearly collinear (median cosine ≥ 0.9) | the method finds the variable, but in 160M the naive direction finds it too: the mechanism adds explanation, not search power | that the method is needed to find the direction |
| `METHOD_TRANSFERS`, `BASELINE_NOT_EXCEEDED`, low cosine | a second effective direction exists at the site; the claim must be about a subspace | a one-dimensional variable |
| `METHOD_READ_CAUSAL_NOT_SITE_SPECIFIC` | the read direction acts where it is written, not only where the head reads | the "read of a head" reading |
| `METHOD_READ_NOT_CAUSAL` | the procedure's output is not a causal variable in 160M | that 160M has no such variable |
| a discovery stop | MD-1's assumptions (one consumption layer, ≤ 3 heads) fail in 160M; the shape of the failure (§4's reported profiles) is the finding | that the behavior has no mechanism |
| `BEHAVIOR_ABSENT_160M` | nothing about the method | — |

For 70M, the same components are C004's first Level-4 tests: S (site), H (held-out axis), B and PS (baselines and
sufficiency). A failure narrows or contests C004 through its own review; it does not change 101's frozen record.

**After 102** (proposals, not decisions): if 160M transfers, the next test of the method is a **different behavior**
(MD-1 frozen, applied unchanged), and a second model family or seed (a PolyPythias 70M seed, after checking its data
variant against the deduped models). The reserve frames serve a replication within this behavior.

## 12. What is definitional and what is not

- **Definitional, given the head:** that editing `⟨x, w_c⟩` moves the head's output along `d̂_T` (the manipulation
  check), and that edits orthogonal to `w_c` do not, to first order. *Post hoc* (X101-2): `w_c` is, to first order, the
  mean cue change mapped through the head's read metric `K = CΓW WᵀΓC` (cos 0.95); an edit along `w_c` writes the
  head's output along `M d̂_T`, 0.34 of whose norm is off `d̂_T`, so even through the head the inert-remainder test is
  not wholly vacuous.
- **Not definitional:** which head and layer MD-1 selects (the other heads' recipe arms show what a wrong selection
  gives); that the remainder is near-inert through every reader; the head's share; completeness; site specificity;
  the comparison with the cue-change direction; projection sufficiency through readers from which `w_c` was not
  derived; the held-out axis.

## 13. Risks

- **Distributed or redundant transport in 160M** (12 heads per layer): the single-site, ≤ 3-head scope may not hold.
  This is a finding about MD-1, recorded through the stop labels and the reported profiles.
- **Multi-layer consumption:** `CONSUMPTION_NOT_CONCENTRATED`.
- **The winner's curse** over 144 heads: the D1/D2 rule; the full rankings are reported.
- **Rank-1 adequacy** of `d̂_T`: the P1 diagnostic; PS is not tested for k > 1.
- **Self-repair** through other readers, stronger in a deeper model, would weaken R relative to I.
- **Saturation** at larger `|Δc_full|`: D is secondary for this reason; the ladder records it.
- **Overshoot** if the read is a larger share of `Δx` in 160M: flagged, as in 101.
- **Numerics:** 160M head-arm tolerances (G5) are 70M's; a failure is an incident.
- **Selection on 70M:** the template, cue pair and nouns survived because they worked on 70M, which favours transfer;
  stated in every label's scope.
- **Not blind in authorship:** the operator knows 70M's answer. Mitigations: MD-1 and its thresholds are frozen
  before any 160M pass; the discovery runs mechanically under a guarded launcher; every head's scores and every
  forking path are published; the user commits the discovery record before calibration.
- **Code hazards:** pinned module blobs; the two-digit label format; model-free ledger keys in older code; the 160M
  cache's missing `refs/main`.

## 14. Decisions for the reviewer

- **D1. Scope of 102.** As proposed: the 160M method-transfer test and the 70M specificity battery in one experiment,
  with separate families. Alternative: 160M only, or 70M first as its own experiment.
- **D2. MD-1's rules and thresholds** (§4): CONCENTRATED at 0.5/0.7; attention ≥ 0.3; joint recovery ≥ 0.70 on D1 and
  ≥ 0.60 on D2; k ≤ 3; medians over frames.
- **D3. The primary baseline:** the frame's own cue-change direction at equal centred length, in both directions.
  Alternatives: the frozen mean difference as primary; interchange semantics.
- **D4. The site control:** the same vector at `RESID_PRE.(L*+1)`, `p_c`, with a floor of 0.5. Alternatives: the
  coordinate re-derived at `L*+1`; `p_t` as primary.
- **D5. The held-out axis** from `three` and `many` against `one`. No non-numeral singular/plural pair exists in
  coordinated frames (`a`, `this`, `another` lose the transport, 008–009).
- **D6. Gatekeeping:** R and I, then S, B and H by Holm, then PS as an accuracy family; separate families per model.
- **D7. No stage 1:** with no program table, the confirmation is one run per model. Alternative: 101's two stages, with
  REF and PL states committed before any edit.
- **D8. Frames:** C = 101's 37–60 and the first 12 eligible of 61–96; the rest of 61–96 reserved. Alternative: a
  grammar-generated pool sampled with a committed seed, which would define a population of frames.
- **D9. Per-item criteria** (§8.5) as a secondary label component; alternative: give them label force.
- **D10. A second held-out model now** (a PolyPythias seed; needs a download and the user's approval), or later as
  proposed.
- **D11. Authorizations:** four model-running phases, each authorized separately, and the user's commits of the
  discovery record and the lock.

## 15. The frames

**C = 101's candidates 37–60** (frozen eligible in 101's `confirmation-v1.json`, never executed) **and the first 12
eligible of candidates 61–96 below**, in this order, chosen by the tokenizer only under 101 §11's rules plus: the
adjective is none of 101's 60 candidates' adjectives, and the text equals none of them. A pre-check with the tokenizer
alone (no model; 2026-10-15) found all 36 eligible, so C would take 61–72 and the reserve would be 73–96; the freeze
re-checks and decides. If fewer than 12 are eligible, the freeze writes nothing and the design is revised.

| # | text | # | text |
|---|---|---|---|
| 61 | Abe and Cara loaded {cue} bulky | 79 | Toby and Vera fixed {cue} twisted |
| 62 | Bart and Della rinsed {cue} muddy | 80 | Wade and Zoe hugged {cue} stuffed |
| 63 | Cody and Erin wiped {cue} dusty | 81 | Yuri and Alma lifted {cue} solid |
| 64 | Drew and Fiona baked {cue} tasty | 82 | Zane and Bess carried {cue} delicate |
| 65 | Evan and Gail grilled {cue} tender | 83 | Abel and Edna hired {cue} strong |
| 66 | Frank and Hazel roasted {cue} hot | 84 | Boyd and Faith chose {cue} mild |
| 67 | Glen and Ida raised {cue} lively | 85 | Chad and Iva picked {cue} tart |
| 68 | Hal and Joy rescued {cue} loyal | 86 | Dale and Jean nursed {cue} weak |
| 69 | Ike and Kay ordered {cue} plain | 87 | Earl and Lily treated {cue} sore |
| 70 | Jay and Lena sold {cue} vintage | 88 | Ford and Nina adopted {cue} healthy |
| 71 | Kirk and Mae traded {cue} antique | 89 | Grant and Pearl bought {cue} compact |
| 72 | Lars and Nora hauled {cue} hollow | 90 | Heath and Rae replaced {cue} dim |
| 73 | Milo and Opal groomed {cue} curly | 91 | Ivan and Sage painted {cue} spotted |
| 74 | Nick and Pam fostered {cue} skinny | 92 | Joel and Tia sewed {cue} floral |
| 75 | Otto and Ruth kept {cue} cracked | 93 | Kurt and Willa returned {cue} short |
| 76 | Pat and Rhea sanded {cue} scratched | 94 | Lane and Wren hushed {cue} noisy |
| 77 | Reed and Sky stacked {cue} folded | 95 | Mitch and Bree helped {cue} poor |
| 78 | Sid and Tara tied {cue} rolled | 96 | Noel and Cass met {cue} rich |

No model is run until the design is decided, the implementation and its review are complete, and the user authorizes
the phase.

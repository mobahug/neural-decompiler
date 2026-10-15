# Experiment 102: blind transfer of an explicit read-discovery procedure to Pythia-160M, with specificity controls and simple baselines — design

**Status:** revision 2, 2026-10-15, **proposed, not decided.** This document contains no code, has run no model,
freezes no frame and computes no tolerance. It was written by the operator (Claude, an AI) on the user's authorization
of 2026-10-14 to *design* the next experiment and to preregister its hypotheses, controls, thresholds, failure criteria
and untouched confirmation data before any run. Revision 1 (commit `1cf99b4`) had an independent AI design review:
**NOT READY**, with two blockers, eleven SHOULDs and nine notes. Revision 2 applies all of them (§16). The review is in
the companion file [2026-10-15-experiment-102-design-reviews.md](2026-10-15-experiment-102-design-reviews.md); it is an
AI review, not a human validation. Nothing here runs until the user decides §14 and authorizes each phase of §10.

Inputs: Experiment 101's closed record (`experiments/101-read-component-intervention/evidence/closure-2026-10-14.md`,
branch `exp101-phases`, `a9b6b33`); the critical assessment
(`research/synthesis/exp101-critical-assessment-2026-10-14.md`); two AI method surveys of 2026-10-14
(`research/synthesis/exp102-surveys/`); and two post-hoc, exploratory checks of 101's saved states and the 70M weights,
X101-1 (directions) and X101-2 (construction), `research/synthesis/checks/exp101_posthoc_{directions,construction}.{py,out}`,
as in the C004 draft. Numbers from those checks are labelled *post hoc*; estimates are labelled *(est.)*.

## 1. Why this experiment

**What 101 showed.** In 36 new coordinated frames of pinned Pythia-70M-deduped, setting the read coordinate
`r(x) = ⟨x, w_c⟩` of the residual entering layer 3 at the cue position — `w_c` being the covector through which
`L03.H04` writes onto its number axis — removed and installed the `one`→`two` shift against an equal-length off-read
control in 36/36 frames (`READ_COMPONENT_EFFECTS_AS_PREDICTED`). The draft claim C004 records this, narrowly.

**What 101 cannot show.** Whether the *procedure* that found `w_c` is a method. The 70M answer was found over twenty
experiments with human choices at every step: the head was selected on two coordinated development frames after a
screen had anticipated it, and was then kept as a constant (160M survey §3). A method generalizes if, written down and
frozen, it finds a causal variable in a model where nobody knows the answer. 101 also leaves the specificity questions
open (critical assessment §1–§5).

**The user's priority** (2026-10-14): prove that the decompilation method generalizes rather than keep improving the
explanation of one behavior; compare against simple baselines; make the method for finding the circuit explicit;
assess individual predictions, not only aggregates.

**What 102 tests, exactly.** The **read-discovery recipe**: causal patching to find the consumption layer and the
transport heads, then a weight-derived read of those heads (MD-1, §4). It does **not** test C003's decompiled programs,
which have no 160M counterpart. Its question: does MD-1, frozen before any number-agreement prompt has run on
Pythia-160M-deduped, return a read variable that, on fresh frames in 160M, carries most of the tested shift, acts at its
position and not where the head writes, moves the outcome in the signed direction, and predicts — frame by frame and in
advance — the effect of other edits? The same battery runs on 70M with C004's variable, together with C004's own open
tests.

**What 102 is not.** It does not test a second behavior, a second model family, other templates or a population of
frames. If 160M transfers, the next test of the recipe is a different behavior (§11).

## 2. Hypotheses

`T` is the read set MD-1 returns (one head in 70M), `L*` its layer, `V` the span of its read covectors (§5).

| id | hypothesis | model(s) | refuted, within scope, by |
|---|---|---|---|
| **H1** (recipe transfer) | MD-1 finds a compact read in 160M (one consumption layer, at most 3 heads, replicated on the held-out half of D), and on fresh frames its read coordinate removes and installs the `one`→`two` shift beyond an equal-length off-read control, carrying at least half of it | 160M | a discovery stop (§4), or R, I or M failing |
| **H2** (position specificity) | the same edit vector written at the prediction position `p_t` (same layer input) has less than half the effect it has at `p_c`, in both directions | both | the position family failing |
| **H3** (signed direction) | reversing the read change in the `one` prompt makes it more singular | both | A_neg failing |
| **H4** (per-length efficiency) | at equal edit length the read beats the frame's own cue-change direction in removal and installation | both | B failing while the two directions are not near-collinear |
| **H5** (held-out axis) | a read built from an axis estimated without `two` (from `three` and `many` against `one`) is non-inferior to the read, within 0.10 of the shift | both | H failing |
| **H6** (projection sufficiency) | an edit's effect at the site depends only on how far it moves the read coordinate: the equal-length edits along the cue change, the mean cue difference and the mean gradient are each predicted by the frame's own read dose–response, within a fixed margin | both (k = 1) | PS failing |
| **H7** (prospective per-frame prediction) | each fresh frame's response to the read, cue-change, mean-difference and gradient edits is predicted **before** the edits run, from its unedited states and D's frozen dose–response | both (k = 1) | PT failing |
| **H8** (per-item completeness) | removal and installation are complete noun by noun | both | §8.7's criteria failing (secondary; no force on the primary label) |
| **PC** (positive control) | MD-1's own 70M read passes R and I on the fresh frames | 70M | PC failing, which makes any 160M verdict unreadable as a statement about MD-1 |

**The verdict** `MD1_READ_TRANSFERS_TO_160M` (§8.6) requires a completed discovery, R, I, the magnitude criterion M,
the position family and A_neg. B, H, PS, PT and the per-item component qualify it; they do not gate it. Whatever `T`
MD-1 returns counts; no human may substitute another head.

**What does not count.** Finding "the analogue of `L03.H04`" by looking through 160M's heads; any 160M forward pass on a
D, C or reserve text before its authorized phase; any re-estimation after the lock.

## 3. Models, behavior and data

### 3.1 Models

| | 70M | 160M |
|---|---|---|
| checkpoint | `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | `EleutherAI/pythia-160m-deduped`, revision `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e` |
| layers × heads, d_model | 6 × 8, 512 | 12 × 12, 768 |
| role | development model: MD-1's retrodiction (§4.4) and positive control; C004's variable under the battery | held-out model: MD-1 runs once, blind to its outcome |

Both snapshots are in the local HF cache (no download); 160M has no `refs/main` and is loaded by commit SHA. The
tokenizer files are byte-identical (`tokenizer.json` sha256 `c24618a1…f25624`), so every token id, position and
tokenizer-only rule carries over. TransformerBridge, float32 on CPU, 4 threads, the pinned library versions as in 101.
No number-agreement prompt has ever run on 160M (160M survey §1).

### 3.2 Behavior, cues, nouns and outcome

As in 101: coordinated frames `NAME and NAME VERBED {cue} ADJ`, cue at `p_c`, adjective at `p_t = p_c + 1`; reference
`one` (581), primary plural `two` (767), stratum `three` (1264) and `many` (1142); the 79 scorable **exposed** nouns;
`c = log P(singular) − log P(plural)` at `p_t`, `Δc` its change from the reference prompt averaged over the nouns. The
template, cue pair and nouns were selected on 70M; every label's scope says so.

### 3.3 Data

| set | frames | status | used for |
|---|---|---|---|
| **D** | the 36 exposed coordinated frames of 020's pool (101's calibration set) | spent on 70M; **never run on 160M** | MD-1 on both models; calibration of both models |
| **C** | 101's unused candidates **37–60** (24) and the first **12** eligible of the new candidates 61–96 (§15) | **never executed on any model** | stage 1 and stage 2 of the one confirmation per model |
| **Reserve** | the new candidates not taken into C (73–96 if 61–72 are all eligible at the freeze) | never executed; **untouched by 102** | a later replication; to be re-checked linguistically before any use (§15) |
| **Smoke** | two smoke-only texts fixed in the plan, none of D, C, the reserve or any earlier frame text | — | launcher smoke tests only; never analysed |

- **D's split.** D1 and D2 are 18 frames each: the frames sorted by `sha256("102|split|" + frame_id)`, the first 18 in
  D1. No other split is allowed.
- **D and 005.** D contains 005's two development frames (`coordinated-adjective-1`, `-2`), on which `L03.H04` was
  originally selected, and 005's extension frames. The 70M retrodiction is therefore partly in-sample to the head's
  original selection; it is a consistency check, not evidence of generality.
- **Why spent-on-70M frames serve 160M's discovery.** No 160M output on them exists; MD-1 is mechanical, so knowledge of
  70M's per-frame results cannot steer it. C is fresh for both models.

## 4. The discovery procedure MD-1

MD-1 is code with fixed rules. Its version, source digest, thresholds and the D1/D2 lists enter a record the **user
commits before the 160M discovery** (§10); the 160M launcher refuses to run otherwise. It runs once per model on D under
a guarded launcher, and its outputs are frozen before any C prompt runs. Its thresholds were chosen with knowledge of
70M's values (the last column).

**Definitions.** All patching is bidirectional on the `one`/`two` pair. *Installation* replaces a site of the `one` run
with the `two` run's value; *blocking* replaces the `two` run's site with the `one` run's value. For a patch `π` and a
frame `f`, `rec_f(π) = ½·[Δc(install π)/Δc_full + 1 − Δc(block π)/Δc_full]`. A statistic over frames is the **median**
of per-frame values over the frames with `Δc_full ≤ −1` nat. A **head patch** replaces that head's additive output at
`p_t` (its per-head result `z_h W_O^h`, before the sum over heads) and nothing else; a **joint** patch replaces every
head of a set at once.

| step | data | rule | stop label | 70M reference |
|---|---|---|---|---|
| (a) behavior | D | `Δc_full ≤ −1` nat in ≥ 30 of 36 frames | `BEHAVIOR_ABSENT` | 101 calibration: 36/36 |
| (b) residual profile | D | `R_c(L)` = median `rec_f` of patching `RESID_PRE.L` at `p_c`, L = 0 … n_layers − 1; `R_c(n_layers) := 0`. `R_t(L)` likewise at `p_t` (descriptive). `R_c(0)` is an identity (G0) | — | 005 A3, 2 frames: L1 0.965, L2 0.938, L3 0.956, L4 0.125, L5 0.125 |
| (c) layer | D | `Δ_L = R_c(L) − R_c(L+1)` (a difference of medians); `L* = argmax Δ_L`, ties to the lower layer. **CONCENTRATED** if `Δ_{L*} ≥ 0.5` and `R_c(L*) ≥ 0.7` | `CONSUMPTION_NOT_CONCENTRATED` | `Δ_3` 0.831, `R_c(3)` 0.956 |
| (d) heads | rank on D1 | candidates: the heads of layer `L*` whose attention from `p_t` to `p_c` (the mean of the two runs' weights, median over D1) is **≥ 0.1**; ranked by singleton `s_h` = median `rec_f` of the head patch on D1 (ties: canonical order). `T` = the shortest prefix, of at most 3 heads (or of all candidates if fewer), whose joint patch has median `rec_f` ≥ **0.70** on D1 | `NO_COMPACT_READ` (no candidate, or no prefix reaches 0.70) | `L03.H04`: attention 0.747, alone 0.816 (005, 2 frames); *post hoc* on D its edited output carries about 0.93 of the removal |
| (d′) verify | D2 | `T`'s joint patch has median `rec_f` ≥ **0.60** on D2 | `DISCOVERY_NOT_REPLICATED` | — |
| (e) read | D (`one`/`two`) | for `h ∈ T`: `d̂_T^h` = unit vector of the unweighted mean over the 36 frames of `h`'s result at `p_t` (`two` run minus `one` run); `w_c^h = C(γ_{L*} ⊙ W_V^h W_O^h d̂_T^h)`; `V` = span of the `w_c^h`, orthonormalized by Gram–Schmidt in canonical head order | — | 011's recipe; X101-2: `w_c ≈ Kδ`, cos 0.95 |
| (e′) held-out axis | D (`three`, `many` vs `one`) | `d̂_T^{h,(−)}` = unit vector of the unweighted mean over the 72 pairs of `h`'s result change at `p_t`; `w_c^{h,(−)}` and `V^(−)` likewise | — | — |
| (e″) P1 diagnostic (k = 1) | D × {`two`, `three`, `many`} (108 pairs) | `y = ⟨ΔT_h, d̂_T^h⟩`, `ŷ = (A_c/σ_c)·⟨Δx, w_c^h⟩` with `A_c` the head's attention `p_t→p_c` and `σ_c` the LayerNorm scale of `x(p_c)`, both from the `one` run; `R² = 1 − Σ(y − ŷ)²/Σ(y − ȳ)²` about the identity line. `P1_SUFFICIENT` if ≥ 0.80, else `P1_NOT_ESTABLISHED`; recorded, no stop | — | none comparable (009's 0.909 was over 720 pairs including suppressed cues) |
| (f) record | — | every value above, `L*`, `T`, the `d̂_T`, `w_c` and `V` in float64 with digests | — | — |

**If `L* = n_layers − 1`** the layer control (§6) is not evaluable and is recorded so; nothing else changes.

**Reported, with no force** (both models): the full singleton ranking on D1 and on D2; `R_t(L)`; `T`'s value-path
share; every head's attention and `s_h`; an attribution-patching (first-order) head ranking; and the heads that **naive
selections** would pick — the highest attention from `p_t` to `p_c` in any layer (ATTN), the highest direct logit
attribution to `c` at `p_t` (DLA: the head's output change projected through the final LayerNorm, at the `one` run's
scale, and the unembedding onto the noun-mean contrast), and the highest `s_h` in any layer (TOP). Each naive head
outside `T` enters the confirmation through its own read (§6). In 70M the attention-selected head would be `L05.H07`
(0.999), not `L03.H04` (005 A4).

**A robustness map** (a preregistered output, no force): `L*` and `T` recomputed from the saved discovery measurements
with each threshold moved by ±0.1 (0.5, 0.7, 0.1, 0.70, 0.60), one at a time.

### 4.4 Step 0: the 70M retrodiction and the revision cap

MD-1 runs unchanged on 70M with D before any 160M pass. It **passes** if (a) passes, `L* = 3` is CONCENTRATED,
`T = {L03.H04}` and (d′) passes. `cos(w_c^{MD-1}, w_c^{011})` is reported (011's axis also used `each`/`several` in
cue-final frames, which MD-1 cannot use); the method's own 70M read is then positive-controlled with force on C (PC,
§8.5).

**Revision cap.** If the retrodiction fails, MD-1 may be revised **once**, for a stated principle (for example, "the
attention criterion should not exclude heads behind an attention sink"), never to match a 70M number; the revision and
its reason are recorded, and version 2 must pass. A second failure returns the design to the user. No 160M prompt runs
before a version passes, and no revision is allowed after any 160M pass.

## 5. The read variable and the edits

For a frame, at `RESID_PRE.L*` and `p_c`: `x_ref`, `x_pl` the plain runs' residuals; `Δx = x_pl − x_ref`; `C` the
centring projector; `P_V` the orthogonal projector onto `V`.
- **Read part** `R = P_V Δx` (k = 1: `(Δr/‖w_c‖²)·w_c`, `Δr = ⟨Δx, w_c⟩`), of length `s = ‖R‖`.
- **Off-read part** `off_c = CΔx − R`, `û_off = off_c/‖off_c‖`; overshoot flagged when `‖off_c‖ < s`, as in 101.
- **Realized read change** of an edit `e` (k = 1): `ρ(e) = −⟨e, w_c⟩/Δr` on the `two` prompt (fraction removed) and
  `⟨e, w_c⟩/Δr` on the `one` prompt (fraction installed), in float64 from the float32 edit.
- **Baseline directions**, each centred, unit, and oriented so that `⟨u, δ_D⟩ > 0`, where `δ_D` is the mean of `CΔx`
  over D for the cue `two`; all frozen at calibration from D except DX:
  - **DX** (per frame): `CΔx/‖CΔx‖`, the frame's own cue change;
  - **MD**: `δ_D/‖δ_D‖`, the mean cue difference;
  - **PC1**: the first right-singular vector of the 108 centred cue changes on D (`two`, `three`, `many` against `one`);
  - **LDA**: `(Σ_W + λI)⁻¹(μ_pl − μ_sg)` with the classes the centred `x(p_c)` of the `one` runs (36) and of the plural
    runs (108) on D, `Σ_W` the pooled within-class covariance, `λ = tr(Σ_W)/d`;
  - **GRAD**: the centred mean, over D's frames and both prompts, of `∇_x c̄` at `RESID_PRE.L*`, `p_c` (one backward pass
    per prompt; `c̄` the noun-mean contrast at `p_t`);
  - **NAT**: `P_nat` projects onto the top-k right-singular vectors of the 108 centred cue changes on D, k the smallest
    reaching 95 % of the summed squared singular values; `ŵ_nat = P_nat ŵ/‖P_nat ŵ‖` and the out-of-span direction
    `u_dorm = C(I − P_nat)ŵ`, normalized. `k` and `‖P_nat ŵ‖` are reported.
- **Held-out** `R^(−) = P_{V^(−)} Δx`, `s^(−)`, `û_off^(−)` as above.
- **70M's primary variable** is C004's: 011's `w_c` at `RESID_PRE.L3` (G1 against 101's lock). MD-1's 70M read is the
  positive control (PC).

Every edit is an exact REPLACE of one residual vector, through the existing intervention with its landing and
zero-outside-change checks; every arm is one forward pass reading all 79 contrasts at `p_t`, and also records the KL
divergence of the full next-token distribution at `p_t` from the arm's own prompt run and the summed probability
change of non-noun tokens.

## 6. The arms

All arms run in every frame of D (calibration) and C (stage 2), for the cue `two`; stratum arms for `three` and `many`.

**Primary arms** (their gates can void the label, §9):

| arm | prompt | edit | family |
|---|---|---|---|
| REF, PL, NOOP | `one`, `two`, `two` | none; none; `x_pl` replaced by itself | references; G3 |
| NEC-READ, SUF-READ | `two`, `one` | `x_pl − R`; `x_ref + R` | R, I, M |
| NEC-OFF, SUF-OFF | `two`, `one` | `x_pl − s·û_off`; `x_ref + s·û_off` | R, I (matched control) |
| NEC-PT, SUF-PT | `two`, `one` | `R` subtracted from / added to `RESID_PRE.L*` at **`p_t`** | position |
| SUF-NEG | `one` | `x_ref − R` | A_neg |
| NEC-Lq, SUF-Lq, q ∈ {0.25, 0.5, 0.75} | `two`, `one` | `x_pl − q·R`; `x_ref + q·R` | PS ladder |
| NEC-u, SUF-u for u ∈ {DX, MD, GRAD} | `two`, `one` | `x_pl − s·u`; `x_ref + s·u` (equal length) | B (DX), PS, PT |
| NEC-HREAD, SUF-HREAD, NEC-HOFF, SUF-HOFF | `two`, `one` | READ and OFF with `R^(−)`, `s^(−)`, `û_off^(−)` | H |
| 70M only: NEC/SUF-BREAD, NEC/SUF-BOFF | `two`, `one` | READ and OFF with MD-1's 70M read | PC |

**Secondary arms** (a gate failure voids that arm alone):

| arm | edit | purpose |
|---|---|---|
| NEC-AMP | `x_pl + R` | A_amp, descriptive (saturation; off-range, X101-1) |
| PC-SWAP-PL, PC-SWAP-REF | the whole `x(p_c)` from the other prompt | share through `p_c` |
| NEC/SUF-PC1, NEC/SUF-LDA | equal length `s` | further baselines |
| NEC/SUF-IX-u, u ∈ {MD, PC1, LDA, GRAD} | interchange: `x_pl − ⟨CΔx, u⟩u`; `x_ref + ⟨CΔx, u⟩u` | what a practitioner would run; predicted to pass R and I with edits 2–3× longer *(est.)* |
| NEC/SUF-DX-FP | DX at the length whose post-LayerNorm footprint `‖γ_{L*} ⊙ e‖/σ(x′)` equals READ's | B under a footprint budget |
| NEC/SUF-NAT, NEC/SUF-DORM | `∓(Δr/⟨ŵ_nat, w_c⟩)·ŵ_nat` (moves `r` by exactly `Δr` within the natural span); `∓ s·u_dorm` | C004's subspace-illusion test (B8) |
| NEC/SUF-ISO-j, j = 1 … 8 | length `s` along mean-zero unit directions orthogonal to `V` (SHA-256 counter-mode, tag `102|iso|{model}|{frame_id}|{j}`) | isotropic controls |
| NEC/SUF-W^h, NEC-WS^h, each other head `h` of `L*` | the recipe of §4(e) applied to `h` (its `d̂_T^h` from D), oriented: equal length `s` both ways; and the swap of its own coordinate (removal) | selectivity of MD-1's head choice; "dead-salmon" check of the recipe |
| NEC/SUF-ATTN, -DLA, -TOP | the read of each naive-selected head outside `T` (§4), swapped at its own layer's input at `p_c` | the discovery procedure against naive head selection, on the outcome |
| HEAD-NEC, HEAD-SUF (+ -V) | `T`'s outputs at `p_t` replaced by the read edit's exact effect on them (as 101) | head share |
| ABH-NEC, ABH-SUF | the read edit at `p_c` with `T`'s outputs at `p_t` clamped to their unedited values | the other readers' share |
| NEC/SUF-SITE | `R` subtracted from / added to `RESID_PRE.(L*+1)` at `p_c` | layer control (not evaluable if `L*` is the last layer) |
| NEC/SUF-PREV | `R` at `RESID_PRE.L*`, `p_c − 1` | the head's attention-gain law |
| NEC-UP | `R` at `RESID_PRE.(L*−1)`, `p_c` (if `L* ≥ 1`); realized `ρ` at `L*` captured | mediation through `r` |
| stratum: PL, NEC/SUF-READ, NEC/SUF-OFF | `three`, `many` against `one` | transfer across cues |

**Fractions:** `F(arm) = Δc(arm)/Δc_full`; `F_rem = F(NEC-READ)`, `F_ins = F(SUF-READ)`.

**Compute.** About 110–125 runs per frame and model, about 4,000–4,500 per model per phase; 70M ≈ 0.07–0.19 s and
160M an estimated 0.2–0.5 s per run (measured in the contract phase). With MD-1 (≈ 6k 70M passes, ≈ 12.4k 160M
passes), the experiment needs about 3–5 h of CPU in all. No GPU, no cloud cost, no download.

## 7. Calibration on D

After the 160M discovery record is committed, a calibration phase runs every arm of §6 on D for both models. These data
are spent; the phase is exploratory. It freezes, by fixed rules:
- the baseline directions (MD, PC1, LDA, GRAD), `P_nat`, the other heads' axes and the naive heads' reads;
- **PS's margins:** `ε_rem = max(0.05, e₉₀^rem)`, `ε_ins = max(0.05, e₉₀^ins)`, where `e₉₀` is the nearest-rank 90th
  percentile (33rd of 36) over D of the ladder's **leave-one-rung-out** interpolation error — predicting `F(NEC-L50)`
  from the rungs {0, 0.25, 0.75, 1}, and `F(SUF-L50)` likewise. No DX, MD or GRAD arm enters `ε`. If `e₉₀ > 0.10`, PS is
  uninformative for that model. D's DX, MD and GRAD discrepancies are reported as exploratory evidence on H6.
- **PT's pooled dose–response:** for removal and installation, the median over D of `F` at each rung
  ρ ∈ {0, 0.25, 0.5, 0.75, 1}, linearly interpolated; and per predicted arm a tolerance `τ_arm`, the nearest-rank 90th
  percentile of D's **leave-one-frame-out** prediction errors (the pooled curve recomputed without the frame), floor
  0.02, cap 0.20 (above the cap that arm is uninformative for PT);
- the per-noun margin `ε_noun = max(0.10, e₉₀^noun)` from the cell-level leave-one-rung-out error (uninformative above
  0.20).

It reports `Δr`, `s`, `‖off_c‖`, overshoot flags, natural ranges of `r`, every direction's cosine with `ŵ` and its `ρ`,
every edit's Mahalanobis length (D's within-cue covariance) and post-LayerNorm footprint, and every family's exposed
distribution with its sign-test power. **Revision triggers,** the only ones: a gate failure; `‖off_c‖ < s` in more than 6
of 36 D frames; a non-finite fraction. A revision driven by the size of any exposed result is forbidden; an
uninformative PS or PT is recorded, not revised.

## 8. Statistics and decision rules

The unit is the frame, n = 36 (C), cue `two`, per model. A frame with `Δc_full > −1` nat counts as a failure in every
count and enters every median as the worst value; ties count as failures. A count's p-value is its exact binomial tail
under a symmetric sign. Medians have frame-bootstrap 95 % intervals (10,000 resamples, seed 20261015), with no decision
force. The two models' families are separate (method transfer; C004's tests).

### 8.1 Precondition

At least **30 of 36** C frames with `Δc_full ≤ −1` nat (checked in stage 1); otherwise
`BEHAVIOR_PRECONDITION_NOT_MET`.

### 8.2 Serial gatekeeping (each stage at 0.025, Holm within the stage: 26 and 25 of 36)

Each later stage is tested only if every hypothesis of the stage before it is rejected, which keeps the familywise
error at 0.025 per model; a later stage's qualifiers never move an earlier stage's threshold.
1. **Stage 1 — R and I** (as 101). R: `N_f = F(NEC-OFF) − F(NEC-READ)`, count of `N_f > 0`, median `N ≥ 0.05`. I:
   `S_f = F(SUF-READ) − F(SUF-OFF)`, count of `S_f > 0`, median `S ≥ 0.05`.
2. **Stage 2 — position and direction.** Position: `Z′_f = min[(1 − F_rem) − 2(1 − F(NEC-PT)), F_ins − 2·F(SUF-PT)]`,
   count of `Z′_f > 0` (the effect at `p_t` is less than half the effect at `p_c`, both ways), median `Z′ ≥ 0.25`.
   Direction: count of `A_neg = Δc(SUF-NEG) > 0` (nats), no floor.
3. **Stage 3 — per-length efficiency and the held-out axis.** B: `B_f = min(F(NEC-DX) − F_rem, F_ins − F(SUF-DX))`,
   count of `B_f > 0`, median `B ≥ 0.10`. H: `H_f = min(F(NEC-HOFF) − F(NEC-HREAD), F(SUF-HREAD) − F(SUF-HOFF))`, count
   of `H_f > 0`, median `H ≥ 0.05`, **and non-inferiority**: `F(NEC-HREAD) − F_rem ≤ 0.10` and
   `F_ins − F(SUF-HREAD) ≤ 0.10` in at least 26 of 36 frames, with both median differences ≤ 0.10. `cos(V, V^(−))` (the
   principal-angle cosine) is reported; at ≥ 0.98 the H result is recorded as near-redundant with R and I.

A family passes if Holm rejects its count's null and its median floor (and, for H, non-inferiority) holds.

**Magnitude M** (a criterion, not a test, applied after stage 1): median `F_rem ≤ 0.5` and median `F_ins ≥ 0.5`.

### 8.3 PS — projection sufficiency, with fixed margins (k = 1; evaluated if R and I pass)

For each edit `u ∈ {DX, MD, GRAD}` and each frame: `F̂_rem(u)` interpolates the frame's removal ladder
`{(0, 1), (0.25, F(NEC-L25)), (0.5, F(NEC-L50)), (0.75, F(NEC-L75)), (1, F_rem)}` linearly at `ρ(NEC-u)`, and
`F̂_ins(u)` the installation ladder `{(0, 0), …, (1, F_ins)}` at `ρ(SUF-u)`. A `ρ` outside [0, 1] counts as a
disagreement. A frame agrees for `u` if `|F(NEC-u) − F̂_rem(u)| ≤ ε_rem` and `|F(SUF-u) − F̂_ins(u)| ≤ ε_ins`. **PS
passes** if, for each of DX, MD and GRAD, at least **27 of 36** frames agree and the median signed deviation `F − F̂` lies
within **±0.05** on each side. GRAD is the adversarial edit: its components outside the read are where a hidden channel
would act. The same statistic is reported, with no force, for every other secondary edit (PC1, LDA, NAT, DORM, ISO,
W^h, UP with its realized `ρ`).

### 8.4 PT — the prospective per-frame table (k = 1; stage 1)

After the lock, stage 1 runs REF, PL and the stratum PL prompts on C (unedited), computes every primary edit's `ρ` per
frame, and predicts `F` for NEC/SUF-READ, -DX, -MD and -GRAD from D's pooled dose–response (§7). The table (the
predictions, the `ρ`s and the precondition) is written, digested and **committed by the user before stage 2 runs any
edit**; stage 2 refuses to start otherwise. **PT passes** if every informative predicted arm has at least **27 of 36**
frames within its `τ_arm`; it is uninformative if more than half the arms are. PT is an accuracy family, as 101's P was.

### 8.5 PC — the positive control (70M)

MD-1's own 70M read, with its own matched off-read control, is scored by stage 1's rules (R and I, Holm 26/25 at
0.025, medians ≥ 0.05) as a separate family. If it fails, every 160M verdict carries `METHOD_POSITIVE_CONTROL_FAILED`
and cannot be read as a statement about MD-1.

### 8.6 Labels and the 160M verdict

Per model, a compound label `PRIMARY | MAGNITUDE | POSITION | DIRECTION | B | H | PS | PT | PER_ITEM` (70M adds `| PC`).

| component | values |
|---|---|
| PRIMARY | `NOT_INTERPRETABLE` (a primary gate failed), `BEHAVIOR_PRECONDITION_NOT_MET`, `READ_EFFECTS_NOT_ESTABLISHED`, `READ_REMOVAL_ONLY`, `READ_INSTALLATION_ONLY`, `READ_EFFECTS_ESTABLISHED` |
| MAGNITUDE | `READ_CARRIES_MAJORITY`, `READ_PARTIAL` |
| POSITION | `POSITION_SPECIFIC`, `POSITION_SPECIFICITY_NOT_ESTABLISHED`, `NOT_REACHED` |
| DIRECTION | `REVERSAL_DIRECTIONAL`, `REVERSAL_NOT_ESTABLISHED`, `NOT_REACHED` |
| B | `PER_LENGTH_ADVANTAGE`, `NO_ADVANTAGE_COLLINEAR` (median `cos(CΔx, ŵ)` ≥ 0.9), `NO_ADVANTAGE`, `NOT_REACHED` |
| H | `HELD_OUT_AXIS_NONINFERIOR`, `HELD_OUT_AXIS_NOT_ESTABLISHED`, `NOT_REACHED` |
| PS | `PROJECTION_SUFFICIENT`, `PROJECTION_SUFFICIENCY_REJECTED`, `PROJECTION_SUFFICIENCY_NOT_EVALUABLE`, `NOT_TESTED_MULTI_HEAD`, `NOT_REACHED` |
| PT | `FRAMES_PREDICTED`, `FRAMES_NOT_PREDICTED`, `PT_NOT_EVALUABLE`, `NOT_TESTED_MULTI_HEAD` |
| PER_ITEM | `PER_ITEM_COMPLETE`, `PER_ITEM_INCOMPLETE` |

**The 160M verdict**, always with the scope string *"MD-1 v{n}; coordinated `one`/`two` noun-number agreement; one
template; Pythia-70M-deduped → Pythia-160M-deduped; the read-discovery recipe, not the decompiled programs"*:

| outcome | verdict | what it answers |
|---|---|---|
| a primary gate fails | `NOT_INTERPRETABLE` | nothing |
| (a) fails on D | `BEHAVIOR_ABSENT_160M` | nothing about MD-1 |
| a discovery stop | `MD1_DISCOVERY_STOPPED_<NOT_CONCENTRATED \| NO_COMPACT_READ \| NOT_REPLICATED>` | **H1 is answered negatively for MD-1 as frozen** |
| C precondition fails | `BEHAVIOR_PRECONDITION_NOT_MET_ON_C` | nothing about MD-1 |
| R or I fails | `MD1_READ_NOT_CAUSAL_160M` | H1 negative |
| R, I pass; M fails | `MD1_READ_PARTIAL_160M` | H1 negative on magnitude |
| R, I, M pass; position or A_neg fails | `MD1_READ_NOT_SPECIFIC_160M` (the failing component named) | H1 holds causally; H2 or H3 negative |
| R, I, M, position and A_neg pass | `MD1_READ_TRANSFERS_TO_160M`, with B, H, PS, PT and PER_ITEM | H1–H3 supported within scope |

A failed 70M PC appends `METHOD_POSITIVE_CONTROL_FAILED`. No label says that a variable has no causal role.

### 8.7 Individual predictions

- **Per frame, with force:** PT (prospective, §8.4) and PS (§8.3).
- **Per noun, recorded as `PER_ITEM_COMPLETE` or `PER_ITEM_INCOMPLETE`, no force** (the 79 nouns share frames and are
  correlated): over cells (frame, noun) with the noun's PL shift ≤ −1 nat, the share with `|F_fn(NEC-READ)| ≤ 0.2` and
  the share with `|F_fn(SUF-READ) − 1| ≤ 0.2`, each **≥ 0.80**; at most **4** of 79 nouns with median
  `F_fn(SUF-READ)` below 0.8 and a frame-bootstrap upper bound below 0.9; noun-level PS on DX within `ε_noun` in a
  share **≥ 0.75** of cells.
- **Descriptive:** Spearman across frames of `F(NEC-DX)` against PT's prediction; the AUROC of PT for the frames whose
  `A_amp` is not negative.

### 8.8 Power

At 80 % of frames with the predicted sign, a count at 26/36 passes with probability 0.911 and at 25/36 with 0.958 (75 %:
0.725, 0.833; 70 %: 0.466, 0.609). The verdict needs stages 1 and 2: about **0.76** jointly at 80 % per statistic
(independence approximation) and about **0.99** at 90 %. The minimal effect of interest is a per-frame success
probability of 0.8. For 70M, 101's record predicts R, I, the position control and B well above threshold (B: removal
along DX leaves about 0.75 of the shift and installs about 0.6 *(est.)*, against ≈ 0 and ≈ 1 along the read); its
Level-4 value lies in H, PS, PT, the position control and the NAT/DORM arms. For 160M nothing is known.

## 9. Gates

- **G0, discovery identities.** Patching `RESID_PRE.L0` at `p_c` reproduces the other prompt's `Δc` bit for bit (both
  directions, every D frame); the D1/D2 split reproduces from its rule; every discovery number recomputes from the
  saved measurements bit for bit.
- **G1, geometry.** Each `w_c^h` recomputed from the weights and the recorded `d̂_T^h` within 1e-9; `V` orthonormal
  within 1e-12; every ISO and baseline direction unit and mean-zero within 1e-12, ISO orthogonal to `V`; 70M's `w_c`
  equal to 101's within 1e-9.
- **G2, the edits.** Landing bit for bit, zero change outside the slice; in float64 from the float32 vector, realized
  read coordinates within 1e-4 of target, centred lengths within 1e-4 relative, parallel/orthogonal within 1e-4;
  realized `ρ` recorded for every edit.
- **G3, the hook path.** NOOP reproduces PL bit for bit; the next-layer residual at the edited position equals `x′` plus
  the block's attention and MLP outputs within the model's tolerance (70M: 1e-4 max-abs).
- **G5, head arms.** Recomputed head outputs reproduce the captured ones (70M: attention rows 1e-4, outputs along `d̂_T`
  1e-3, 017's tolerances).
- **160M tolerances** for G3 and G5 are set in the contract phase on four neutral, non-experimental prompts, as the
  larger of 70M's tolerance and 10× the largest discrepancy measured there for the same identity, and frozen before the
  discovery.
- **G4** (101's program identities) does not apply: 102 has no decoded program.
- **G6, determinism.** Stage 2's REF and PL runs reproduce stage 1's bit for bit; every result recomputes from the saved
  measurements bit for bit at the report; the committed MD-1 record, discovery record, lock and stage-1 table are
  digest-checked before the phases that depend on them.
- **G7, isolation and accounting.** Every run has a model-scoped, arm-tagged key `model|frame_id|word|token_id|arm`,
  executed once per phase. No C or reserve prompt's **token-id sequence** appears in any earlier record that stores
  sequences or keys (005–025, 101, and 102's retrodiction, discovery and calibration ledgers); for earlier ledgers kept
  only as digests, novelty rests on §15's text rule. No 160M run touches a D, C or reserve text before its authorized
  phase; smoke runs use the smoke-only texts.
- **Which failures void the label.** G0–G3, G6 and G7 on the primary arms make the label `NOT_INTERPRETABLE`. A gate
  failure on a secondary arm voids that arm only and is reported.

## 10. Phases, authorizations and human commits

Each phase runs once, with a read-only AI review between phases. **Every phase that runs a model needs the user's
explicit authorization;** the user commits four records.
1. **Design** (this document): AI review; revisions; an independent **human expert review is recommended here**, before
   anything is frozen; the user's decision on §14.
2. **Plan, implementation and contract:** new modules only (the pinned modules of 101 and 022–025 are not edited);
   tier-A and tier-B tests on fakes; a 160M contract test (12/12/768/3072, the parallel block identity, NOOP) on neutral
   prompts, which also measures run time and sets 160M's G3/G5 tolerances; canonical component labels (`L10.H03`,
   never `L010.H03`); model-scoped ledgers. Implementation review.
3. **Freeze C** (tokenizer only, with the committed linguistic check of §15): `confirmation-v1.json`; committed.
4. **Retrodiction** on 70M (spent D; *authorization 1*), with the revision cap of §4.4. Review. **The user commits
   the MD-1 record** (source digest, thresholds, D1/D2 lists, retrodiction result).
5. **160M discovery** (MD-1 once on D; *authorization 2*): an integrity-only review (did MD-1 run as committed?). **The
   user commits the discovery record.** A discovery stop ends the 160M part.
6. **Calibration** of both models on D (*authorization 3*): §7's frozen directions, margins, pooled curves and
   tolerances; revision triggers only. Review.
7. **Lock** (weights and committed records only): binds MD-1, the discovery record, C, every direction, margin and
   tolerance, §8 and §9. **The user commits the lock.**
8. **Stage 1** (*authorization 4*): REF, PL and stratum PL on C for both models; the PT table. **The user commits the
   table.**
9. **Stage 2** (*authorization 5*): every arm on C, each model once, under a guarded launcher (101's confirm launcher as
   template, with smoke on the smoke-only texts).
10. **Report and closure,** with an independent AI confirmation review that recomputes every label.

No phase is repeated. A failure is reported as a failure; no frame is replaced; no threshold is revised after an outcome
is seen.

## 11. What each outcome would mean

| 160M verdict | what it would support | what it would not |
|---|---|---|
| `MD1_READ_TRANSFERS_TO_160M` with `PER_LENGTH_ADVANTAGE`, `PROJECTION_SUFFICIENT`, `FRAMES_PREDICTED` | a frozen read-discovery recipe found, in a model where the answer was unknown, a position-specific, signed, one-dimensional read carrying most of the tested shift, more efficient per unit length than the naive direction, sufficient against an adversarial gradient edit, and predictive of fresh frames in advance — for this behavior and template | generality to other behaviors, model families or templates; anything about the decompiled programs; that the read is the model's algorithm |
| transfers, `NO_ADVANTAGE_COLLINEAR` | the recipe finds the variable, but in 160M the naive direction finds it too: the mechanism adds explanation, not search power | that the recipe is needed to find the direction |
| transfers, `NO_ADVANTAGE` or `PROJECTION_SUFFICIENCY_REJECTED` | a second effective direction exists at the site; a claim must be about a subspace | a one-dimensional variable |
| `MD1_READ_NOT_SPECIFIC_160M` | the read is causal but acts where the head writes too (position), or its reversal is not signed (direction) | the "read of a transport head" reading |
| `MD1_READ_PARTIAL_160M` | the read carries a minority of the shift; transport is shared beyond `T` | a complete read |
| `MD1_READ_NOT_CAUSAL_160M` | the recipe's output is not a causal variable in 160M | that 160M has no such variable |
| a discovery stop | a negative answer to H1 for MD-1 as frozen (one consumption layer, ≤ 3 heads); the reported profiles describe the failure | that the behavior has no mechanism |
| `BEHAVIOR_ABSENT_160M`, precondition failures | nothing about MD-1 | — |

For 70M, the components are C004's first tests of position specificity, held-out axis, per-length efficiency,
projection sufficiency, prospective per-frame prediction, the natural-span (subspace-illusion) arms and output
specificity (KL). A failure narrows or contests C004 through its own review; 101's frozen record does not change.

**After 102** (proposals): if 160M transfers, the next test of the recipe is a **different behavior** with MD-1 frozen
and applied unchanged, then a second model family or seed (for example a PolyPythias 70M seed, after checking its data
variant) and a negative-control checkpoint on which MD-1 should stop. The reserve frames serve a replication within this
behavior.

## 12. What is definitional, what is implied, and what is at risk

- **Definitional, given the head:** that editing `⟨x, w_c⟩` moves the head's output along `d̂_T`, and that edits
  orthogonal to `w_c` do not, to first order. *Post hoc* (X101-2): `w_c ≈ Kδ` (cos 0.95), the mean cue change through the
  head's read metric; an edit along `w_c` writes `M d̂_T`, 0.34 of whose norm is off `d̂_T`; and `cos(ŵ, d̂_T)` = 0.20, so
  the read covector at `p_c` is far from the head's write axis at `p_t`.
- **Largely implied once the discovery and R/I pass** (review 1): B follows from geometry and the inert off-read part
  (it measures per-length efficiency, not a separate mechanism); H is near-redundant when the numeral axes are
  collinear (`cos(V, V^(−))` reported); the layer control is bounded by MD-1's own layer rule (hence secondary).
- **At risk, and not bounded by MD-1:** the discovery itself (layer, heads, replication); R, I and M on fresh frames;
  the position control at `p_t`, which MD-1 never measures; A_neg; PS against the adversarial GRAD and the fixed MD
  edits with fixed margins; PT's prospective per-frame predictions; the positive control; the naive-selection reads.

## 13. Risks

- **Distributed or redundant transport in 160M** (12 heads per layer): MD-1's scope (one layer, ≤ 3 heads) may not hold;
  recorded through the stop labels, the profiles and the robustness map.
- **Attention sinks** could hide a genuine reader: the attention filter is 0.1, and the robustness map reports 0.0 and
  0.2.
- **The winner's curse** over 144 heads: the D1/D2 rule; full rankings published.
- **Rank-1 adequacy:** the P1 diagnostic; PS and PT are not tested for k > 1.
- **Self-repair** in a deeper model would weaken R relative to I.
- **Saturation** at larger `|Δc_full|`: A_amp is descriptive; A_neg is the direction gate.
- **Overshoot** if the read is a larger share of `Δx`: flagged.
- **Numerics:** 160M tolerances are measured and frozen in the contract phase; a failure is an incident.
- **Selection on 70M** of the template, cue pair and nouns favours transfer; stated in every scope.
- **Not blind in authorship:** the operator knows 70M's answer. Mitigations: MD-1 committed by the user before any 160M
  pass; mechanical execution under a guarded launcher; every head's scores, the robustness map and every forking path
  published; the user commits the discovery record.
- **Code hazards:** pinned module blobs; label formats; model-free ledger keys in older code; the 160M cache's missing
  `refs/main`.

## 14. Decisions for the reviewer

- **D1. Scope:** the 160M recipe-transfer test and the 70M battery in one experiment, with separate families.
- **D2. MD-1's rules:** CONCENTRATED at 0.5/0.7; attention ≥ 0.1; joint recovery ≥ 0.70 on D1 and ≥ 0.60 on D2; k ≤ 3;
  medians; the revision cap.
- **D3. Specificity:** the position control at `p_t` as the verdict's gate (relative form, floor 0.25), the layer control
  secondary.
- **D4. Direction:** A_neg as a gate; A_amp descriptive.
- **D5. B** as a per-length-efficiency qualifier, with the interchange and footprint forms secondary.
- **D6. H** with non-inferiority at 0.10.
- **D7. PS** with fixed margins (floor 0.05, uninformative above 0.10) on DX, MD and GRAD, and a signed criterion.
- **D8. Stage 1 restored** with the prospective table PT and a fourth user commit.
- **D9. The 70M positive control** with force on the verdict's interpretation.
- **D10. Frames:** C = 101's 37–60 and the first 12 eligible of 61–96 (five adjectives replaced, §15); the reserve kept.
- **D11. Per-item criteria** secondary, without force.
- **D12. A second held-out model or a negative-control checkpoint now** (needs a download), or later as proposed.
- **D13. Authorizations:** five model-running phases, each authorized separately, and four user commits.

## 15. The frames

**C = 101's candidates 37–60** (frozen eligible in 101's `confirmation-v1.json`, never executed; their adjectives passed
101's linguistic review) **and the first 12 eligible of candidates 61–96 below**, in order, chosen by the tokenizer only
under 101 §11's rules plus: the adjective is none of 101's 60 candidates' adjectives, and the text equals none of them.
A tokenizer-only pre-check (no model; 2026-10-15) found all 36 eligible, so C would take 61–72 and the reserve would be
73–96; the freeze re-checks and decides. If fewer than 12 are eligible, the freeze writes nothing and the design is
revised. Revision 2 replaced five candidates whose adjective can be read as a noun (review 1, N4): 65 `tender`, 69
`plain`, 70 `vintage`, 71 `antique`, 72 `hollow`.

| # | text | # | text |
|---|---|---|---|
| 61 | Abe and Cara loaded {cue} bulky | 79 | Toby and Vera fixed {cue} twisted |
| 62 | Bart and Della rinsed {cue} muddy | 80 | Wade and Zoe hugged {cue} stuffed |
| 63 | Cody and Erin wiped {cue} dusty | 81 | Yuri and Alma lifted {cue} solid |
| 64 | Drew and Fiona baked {cue} tasty | 82 | Zane and Bess carried {cue} delicate |
| 65 | Evan and Gail shook {cue} sandy | 83 | Abel and Edna hired {cue} strong |
| 66 | Frank and Hazel roasted {cue} hot | 84 | Boyd and Faith chose {cue} mild |
| 67 | Glen and Ida raised {cue} lively | 85 | Chad and Iva picked {cue} tart |
| 68 | Hal and Joy rescued {cue} loyal | 86 | Dale and Jean nursed {cue} weak |
| 69 | Ike and Kay crossed {cue} icy | 87 | Earl and Lily treated {cue} sore |
| 70 | Jay and Lena climbed {cue} rocky | 88 | Ford and Nina adopted {cue} healthy |
| 71 | Kirk and Mae greeted {cue} bald | 89 | Grant and Pearl bought {cue} compact |
| 72 | Lars and Nora planted {cue} slender | 90 | Heath and Rae replaced {cue} dim |
| 73 | Milo and Opal groomed {cue} curly | 91 | Ivan and Sage painted {cue} spotted |
| 74 | Nick and Pam fostered {cue} skinny | 92 | Joel and Tia sewed {cue} floral |
| 75 | Otto and Ruth kept {cue} cracked | 93 | Kurt and Willa returned {cue} short |
| 76 | Pat and Rhea sanded {cue} scratched | 94 | Lane and Wren hushed {cue} noisy |
| 77 | Reed and Sky stacked {cue} folded | 95 | Mitch and Bree helped {cue} poor |
| 78 | Sid and Tara tied {cue} rolled | 96 | Noel and Cass met {cue} rich |

**Linguistic check of C's new adjectives** (an AI judgment, to be confirmed by the human reviewer and committed with the
freeze): `bulky`, `muddy`, `dusty`, `tasty`, `sandy`, `hot`, `lively`, `loyal`, `icy`, `rocky`, `bald` and `slender` are
plain attributive adjectives in these frames, none commonly a count noun, and each verb takes a plural object
naturally. **The reserve is not yet clean:** `solid`, `tart`, `sore`, `compact`, `short`, `poor` and `rich` (81, 85, 87,
89, 93, 95, 96) can be read as nouns and must be replaced before any use of the reserve.

## 16. What changed in revision 2

| review 1 finding | change |
|---|---|
| **B1** PS could not fail (ε calibrated on the tested discrepancy; one plane only) | fixed margins from the ladder's leave-one-rung-out error, floor 0.05, uninformative above 0.10; a signed criterion; MD and the adversarial GRAD added as PS edits (§7, §8.3) |
| **B2** the site control was near-implied by MD-1, could not discriminate a generic direction, mislabelled incompleteness, and passed by construction at the last layer | the position control at `p_t`, both ways, relative form, is the verdict's gate; the layer control secondary and not evaluable at the last layer; the magnitude criterion M and `MD1_READ_PARTIAL_160M` (§6, §8.2, §8.6) |
| S1 B follows from geometry | B reframed as per-length efficiency; interchange arms, footprint arms, per-baseline statistics, orientation rule, definitions of GRAD and LDA (§5, §6, §8.2) |
| S2 naive head selection barely tested | ATTN, DLA and TOP reads; other heads at equal length both ways plus their swaps; an attribution-patching ranking (§4, §6, §8.7) |
| S3 no disruption test in 160M | A_neg is a stage-2 gate (§8.2) |
| S4 H did not test material loss | non-inferiority at 0.10; `cos(V, V^(−))` reported (§8.2) |
| S5 Holm coupled the verdict's gate to its qualifiers | serial gatekeeping: R, I; then position and A_neg; then B and H (§8.2) |
| S6 no prospective individual prediction | stage 1 restored with the prospective table PT, committed by the user; `ε_noun` fixed like ε (§7, §8.4, §8.7) |
| S7 MD-1 governance | revision cap of one, for a principle; the user commits MD-1 before the 160M discovery; the positive control PC (§4.4, §8.5, §10) |
| S8 MD-1 ambiguities | recovery, medians, Δ, hooks, `d̂_T` averaging, (e′) pooling, the P1 R² defined; attention filter 0.1; the robustness map (§4) |
| S9 gates could void the label for secondary reasons; 160M tolerances; smoke on D | primary-only voiding; 160M tolerances set in the contract phase; smoke-only texts (§3.3, §9) |
| S10 labels overclaimed | `MD1_READ_TRANSFERS_TO_160M` with a mandatory scope string; the recipe, not the programs, stated in §1 and §11; a stop named as a negative answer; the collinearity split moved into the labels |
| S11 C004's open tests missing | NAT and DORM arms; KL and non-noun probability recorded for every arm (§5, §6) |
| N1 companion file missing | created and committed with revision 1 |
| N2 P1 reference not comparable | marked as not comparable (§4) |
| N3 D contains 005's frames | stated (§3.3) |
| N4 noun-readable adjectives in C | five replaced; the linguistic check stated; the reserve flagged (§15) |
| N5 G7 by token sequences | G7 matches token-id sequences (§9) |
| N6 joint power | stated, with a minimal effect of interest (§8.8) |
| N7 negative-control checkpoint | listed as D12 and in §11 |
| N8 the 70M battery is low-risk | its Level-4 value located (§8.8, §11) |
| N9 estimates unmarked; G4 skipped silently | marked *(est.)*; G4's inapplicability stated (§9) |

No model is run until the design is decided, the implementation and its review are complete, and the user authorizes
the phase.

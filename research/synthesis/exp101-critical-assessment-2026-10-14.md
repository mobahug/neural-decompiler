# Experiment 101: a critical assessment of the remaining confounders (2026-10-14)

**Status.** Written by the operator (Claude, an AI) after Experiment 101 closed, as input to the user's decisions on
C004 and the next experiment. It is an analysis, not a review and not a human validation. Every number is from the
committed 101 record, or from the post-hoc, exploratory diagnostics in
[`checks/exp101_posthoc_directions.py`](checks/exp101_posthoc_directions.py) (output in `.out` beside it), which read the
saved measurements only and run no model. Proposals are marked as such.

## The result in one paragraph

In 36 new coordinated frames of pinned Pythia-70M-deduped, editing only the read coordinate `r(x) = ⟨x, w_c⟩` of the
residual entering layer 3 at the cue position — `w_c` being the weight-derived covector through which `L03.H04` writes
onto its number axis `d̂_T` — removed (`F_rem` median −0.009) and installed (`F_ins` 1.004) the whole `one`→`two` shift
of the mean noun-number contrast, while equal-length edits along the cue's own off-read change did nothing (1.000 left,
−0.033 installed), in every frame; the direction checks and the frozen program's predictions also passed
(`READ_COMPONENT_EFFECTS_AS_PREDICTED`; closure in
`experiments/101-read-component-intervention/evidence/closure-2026-10-14.md`).

**The question this assessment serves:** did we find a generally useful way of locating causal variables inside an LLM,
or did we isolate a signal in one mechanism we had already studied for twenty experiments? 101 cannot answer it; the
assessment below says what can.

## The confounders, one by one

### 1. The in-sample axis

- **The concern.** `d̂_T` — and so `w_c` — was estimated from the head's output changes on exposed `one`/`two` (and
  `each`/`several`) pairs. 101 then tested `one`/`two`. Freshness is in the frames only.
- **What 101 shows.** The stratum `three`, which did not enter `d̂_T`, behaves like `two` on every statistic (R 36/36,
  I 36/36, D 28/36 and 36/36); `many` passes R and I but would fail D (23/36). The axis transfers to new frames and,
  descriptively, partly to new cues.
- **What remains.** Whether a read derived **without** `one`/`two` works as well on `one`/`two`. If `w_c` only works
  because it was fitted to this pair, the "decoded variable" is partly a fitted readout.
- **The decisive test** (proposal): derive `d̂_T` (hence `w_c`) from other cue pairs only, freeze it, and rerun the
  removal and installation tests for `one`/`two` on new frames. Prediction under the mechanistic reading: the same
  result within a stated margin; under the fitted-readout reading: a material loss.

### 2. Site specificity (layer and position)

- **The concern.** 101 has no site or layer negative control. If the same `w_c` edit worked at another layer or
  position, "the read at layer 3" would be the wrong description.
- **What exists.** 005's exploratory residual profile (whole-residual patching at `p_c`, coordinated frames): recovery
  0.956 at the layer-3 input but 0.125 at the layer-4 input — the cue's information at `p_c` is consumed at layer 3
  (`experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md:29–36`).
- **The decisive tests** (proposal): the same read edit at `RESID_PRE.L4` (`p_c`), where the head has already read
  (predicted ≈ no effect); at a non-cue position (predicted ≈ no effect); and at `RESID_PRE.L2`/`L1` (`p_c`), where the
  information has not yet passed layers 1–2 (prediction: smaller or transformed — a genuinely open question, because
  layers 1–2 rewrite the cue state).

### 3. Off-distribution interventions

- **The concern.** Edited residuals are composites the model never produces naturally.
- **What the saved data show (post hoc).** The natural read coordinate is `r(one)` −1.06…−0.34, `r(two)` 0.88…1.58,
  `r(three)` 0.74…1.42, `r(many)` 0.63…1.30 across 72 frames. The removal and installation swaps set `r` to the other
  prompt's natural value — **inside** the natural range. The direction arms are not: NEC-AMP's target (2.58…3.90) is
  above every natural plural `r`, and SUF-NEG's (−3.21…−1.97) below every natural `r(one)`, in 36/36 frames of both
  sets. D is therefore an off-manifold extrapolation test; R and I are not, in `r`, though the composite states are
  novel in their remaining coordinates.
- **What remains.** Whether the composites are "on-manifold" in the full space. A proposal: natural-variation tests
  (pairs of natural prompts whose `r` differ, with matched off-read parts) and graded doses within the natural range.

### 4. The simpler mathematical explanation

- **The concern.** `w_c` is, by construction, the covector through which the head writes onto `d̂_T`. Editing
  `⟨x, w_c⟩` therefore moves the head's output along `d̂_T` almost by definition (101's manipulation check: NEC-READ
  −0.008, SUF-READ 0.979 of `PL`'s `ΔT`). Is the result just this identity?
- **What is implied by construction.** That the head's output along `d̂_T` follows `r` (up to the LayerNorm scale and the
  attention row).
- **How close `w_c` is to a steering vector** (post hoc, weights and saved states only,
  [`checks/exp101_posthoc_construction.py`](checks/exp101_posthoc_construction.py)): to first order `w_c ∝ K·δ`, where
  `δ` is the mean cue change and `K = CΓW WᵀΓC` is the head's read metric after LayerNorm (`W = W_V W_O`, `Γ = diag(γ₃)`,
  `C` the centring projector). Measured: cos(`w_c`, `Kδ`) = 0.948 on the calibration frames and 0.952 on the
  confirmation frames (per frame 0.88–0.97), while cos(`δ`, `ŵ`) is only 0.41. So what the mechanistic derivation adds
  over the plain mean-difference direction is exactly `K` — the head's own metric.
- **How much of "the remainder is inert" is definitional for the head itself.** An edit along `w_c` writes the head's
  output along `M d̂_T`, `M = WᵀΓCΓW`; post hoc, cos(`M d̂_T`, `d̂_T`) = 0.939, so 0.34 of that output's norm is off the
  axis. Because `d̂_T` is the mean output change, the remainder's off-axis output cancels the read part's on average. The
  head-route part of the inertness test is therefore partly, not wholly, vacuous; the non-head readers carry its
  unambiguous content.
- **What the complementary arms cannot exclude.** Under additivity (101: `F_rem + F_ins` = 0.989), any illusory effect of
  the read part through a dormant pathway (B8) must be cancelled by the remainder, which reproduces 101's pattern
  exactly. The illusion is excluded for the head's `d̂_T` channel, not for the other readers (the ~11–12 %).
- **What is empirical — and not implied.**
  1. **The remainder is inert through every reader.** The off-read part is ~83–86 % of the cue change's centred energy
     (post hoc: the share of ‖mean Δx‖² along `ŵ` is 0.166–0.168), and it reaches MLP3, the other heads, `L05.H07`
     and block 5 as well; yet removing it changes nothing (`F_swap_pl` 0.044 vs `F_rem` −0.009). Nothing in the
     head's definition forces the other readers to ignore it.
  2. **The head's output is the bottleneck** for ~88–89 % of the effect (the head-only arms).
  3. **Completeness:** the read part alone reproduces the whole-position swap (`F_ins` 1.004 vs `F_swap_ref` 1.008).
- **A sharp, untested consequence** (proposal): if `r` is a sufficient statistic of `x(p_c)` for this behaviour, then
  the effect of **any** edit `e` at the site is predicted by the change it makes in `r` alone, `⟨e, w_c⟩`, through the
  dose–response that the read arms measured. An edit of length `a` along a unit direction `u` moves `r` by
  `(a/s)·cos(u, ŵ)` of the read change `Δr` (with `s = |Δr|/‖w_c‖` the read edit's length); the response is non-linear
  (removing half of `Δr` leaves 0.70 of the shift; installing half gives 0.72). That is an interchange-intervention-style
  test of `r` as a causal variable, stronger than "a direction works", and it makes the baselines below quantitative.

### 5. Simpler baselines and prior work

- **The prior.** One-dimensional causal directions for agreement already exist in small Pythia models: CausalGym
  (Arora, Jurafsky and Potts, 2024; the literature review's A7) finds them with DAS, probes and difference-in-means,
  including Pythia-70M, and warns that DAS's efficacy is not selective (it can produce effects for arbitrary mappings).
  "A direction works" is not new.
- **What the saved data show (post hoc).** Naive data-derived directions are **not** `ŵ`: the cosine of `ŵ` with each
  frame's cue change is 0.29–0.44; with the mean cue change 0.41; with its first principal component (84–85 % of the
  variance) 0.41; with a class-mean "probe" direction 0.41–0.42; with a shrinkage-LDA direction 0.42–0.43. The
  data-derived directions are dominated by the inert off-read part. So the weight-derived read is not what a simple
  data method returns — and whether those methods would nevertheless remove the shift at equal norm is an open,
  testable question. Under item 4's prediction, an equal-norm edit along the mean cue change moves `r` by only ~41 % of
  `Δr`, which the measured dose–response maps to removing roughly a quarter of the shift (or installing roughly 60 %) —
  far below the read edit's full effect at the same length.
- **Why 101's R and I cannot answer this.** In interchange form — setting a direction's coordinate to the other
  prompt's value — the mean-difference direction would also move `r` by about `Δr` (PC1 explains ~85 % of the cue
  changes' variance), so it would plausibly also pass R and I, with an edit two to three times longer. R and I show
  that `r` mediates the shift; only comparisons at **equal edit length** separate `w_c` from the simple directions.
- **What would count** (proposal): the mechanistic derivation adds value if `ŵ` achieves the full effect at a norm where
  data-derived directions achieve a fraction predicted by their `ŵ`-projection; it "adds nothing but is not wrong" if a
  data-derived direction (e.g. DAS trained on exposed data) recovers `ŵ` or matches it per unit norm; it is undermined if
  a direction nearly orthogonal to `ŵ` matches the effect.

### 6. The other readers (~11–12 %)

The head-only arms leave about 11–12 % of the effect to other readers of `x(p_c)` (other layer-3 heads, MLP3, `L05.H07`,
block 5's path). The value-only and full head arms agree, so the head's attention change contributes little. A
decomposition by component is cheap on spent data (proposal) and bears on whether "the variable" is the head's or the
site's.

### 7. Saturation and the direction family

D passed by 3 frames (28/36 against 25) and `many` would fail it: the head's transported signal nearly doubles under
NEC-AMP (×1.94) while the contrast moves −0.136 nats — the plural prompt is near saturation, and NEC-AMP leaves the
natural range of `r` (item 3). D's evidence is real but is the weakest link, and it is an extrapolation test.

### 8. The nouns and individual predictions

The outcome averaged 79 **exposed** nouns. The program's per-noun readout errors reach about 1.06–1.20 nats on some
nouns even with exact inputs, while the error of the noun mean stays small; P agreed in mean fractions only. Future
predictions should be scored per item (per noun and per frame), with preregistered per-item error criteria, and on fresh
nouns where possible.

### 9. The inferential population and replication

The exact binomial tails (2⁻³⁶) treat 36 frames of one hand-written template as exchangeable units under a
sign-symmetric null; they say nothing about a population of templates, cue pairs or models. There is one confirmation
run, no replication, and every review so far is an AI review — no human expert has examined the result.

## What this means

- **What we have:** a preregistered causal result for one decoded variable at one site in one model, with the strongest
  controls the project has run, and an empirical content (inert remainder, bottleneck, completeness) beyond the
  definitional identity. That justifies a narrowly scoped claim (draft C004), not more.
- **What we do not have:** any evidence that the **procedure** that found `w_c` would find a causal variable in a model
  or behaviour we have not studied. The 70M answer was found over twenty experiments with human choices at every step.
- **The most informative next step** (proposal, detailed in the 102 design,
  `docs/superpowers/specs/2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`): make the discovery
  procedure explicit and mechanical, run it unchanged on Pythia-160M with no knowledge of the 70M answer allowed into its
  decisions, test the variable it returns with fresh preregistered interventions, and in the same experiment run the
  specificity controls (site, layer, held-out-axis) and the equal-norm baselines whose outcomes item 4 predicts
  quantitatively — on both models.

**Sources added on 2026-10-15:** the construction check above, and two AI method surveys of 2026-10-14 (the 160M
discovery survey; the baselines-and-controls survey; both in `exp102-surveys/`), whose relevant conclusions are folded into items 4 and 5 and into
the 102 design. The surveys are AI-authored scratch memos, not reviews; their estimates marked *(est.)* there are not
measurements.

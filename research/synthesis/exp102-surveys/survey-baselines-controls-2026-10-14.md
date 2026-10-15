# Experiment 102 methods survey: simpler explanations, baselines and controls for 101

**Provenance.** An AI-authored memo (a Claude survey agent), written read-only on 2026-10-14. Every recommendation in it
is a **[proposal]**. No model was run.

**Sources read:**
- CL: `experiments/101-read-component-intervention/evidence/closure-2026-10-14.md`;
- D101: the 101 design;
- REV: `…/evidence/confirmation-review-2026-10-14/REVIEW.md`;
- RS: `research/synthesis/research-state-2026-10-14.md`;
- LIT: `research/paper/literature-review/review-2026-09-28.md`;
- DR005: 005's discovery report;
- six web searches.

**Markers.** *(est.)* is my interpolation from 101's medians. *(sib.)* is a value from the sibling scratch file
`scratchpad/exp102/diagnostics-ŵ-vs-data-directions.txt`, which was computed from stored states; I have not verified it.

**Notation:**
- `x` is `RESID_PRE.L3` at `p_c`, with d = 512.
- `C` is the centring projector, `Γ = diag(γ₃)`, and `W = W_V W_O` of `L03.H04`.
- `w_c = CΓW d̂_T`, and `ŵ` is its unit direction.
- `Δx_f = x_pl − x_ref`, and `Δr_f = ⟨Δx_f, w_c⟩`.
- The read part is `R_f = (Δr_f/‖w_c‖²)w_c`, of length `s_f`. The off-read part is `O_f = CΔx_f − R_f`.
- `A` is the head's attention to `p_c`, and `σ` is the LayerNorm scale there.
- `F` is an arm's `Δc` as a fraction of `Δc_full`.
- LM is the learned-mechanism reading.

## 0. Key points

1. **Part of R and I follows from how `w_c` was built** (§1.1). The evidential core is narrower:
   - the remainder is inert in every frame;
   - the readers other than `L03.H04` respond to the read part and not to the remainder;
   - the effect is complete in every frame.
2. **To first order, `w_c ∝ K·δ`.** δ is the mean cue change, and K is the head's OV Gram matrix after LayerNorm. What
   the derivation adds over a steering vector is exactly K.
   - The mean difference, PC1 and the probe all sit near cosine 0.41 to `ŵ` *(sib.)*.
   - As interchanges, they are predicted to pass R and I too.
   - Only comparisons at equal length discriminate.
3. **101's complementary arms do not by themselves exclude the subspace illusion.** Under additivity, `O_f` cancels any
   illusory effect of `R_f`. The illusion is excluded only for the head's `d̂_T` channel (§1.5).
4. **One master prediction, projection sufficiency, scores every baseline** (§2).
5. **The best value per cost** (§6):
   - spent-data analytics;
   - an axis derived without one/two;
   - site and layer nulls;
   - arms that separate the head from the other readers.
6. **For Pythia-160M, transfer a frozen pipeline, not the answer** (§5).

## 1. Simpler explanations as falsifiable alternatives

### 1.1 "A convenient readout": what is construction and what is evidence

Under P1, with A and σ fixed, an edit `e` at `p_c` gives

```
ΔT(e) = (A/σ)·WᵀΓC e   ⇒   ⟨ΔT(e), d̂_T⟩ = (A/σ)·⟨e, w_c⟩                                  (1)
d̂_T ∝ E_est[ΔT]       ⇒   w_c ∝ CΓWWᵀΓC·δ_w = K·δ_w   (K PSD, rank ≤ 64)                    (2)
```

- δ_w is the (A/σ)-weighted mean of `CΔx` over the pairs that estimated `d̂_T`: one/two and each/several, on exposed
  frames.
- (2) ignores attention changes and the remainder from the adjective position.
- `cos(w_c, Kδ_w)` is free to check from stored states.

**Construction, with no or little evidential weight:**
- **(a) The manipulation check.** NEC-READ −0.008, SUF-READ 0.979, NEC-OFF 1.026, SUF-OFF 0.003 and NEC-AMP 1.943 (REV §5)
  are identity (1), plus small corrections from LayerNorm and attention.
- **(b) The remainder through the head (partly construction).** `O_f` changes the head's `d̂_T` output by exactly zero.
  Its off-axis output cancels `R_f`'s on average:
  - a `w_c` edit writes along `M d̂_T`, with `M = WᵀΓCΓW`;
  - the head's total off-axis change has mean zero on the estimation pairs.

  The head's part of "the remainder is inert" therefore tests only whether the downstream ignores `(I − d̂d̂ᵀ)M d̂_T`,
  plus frame-specific deviations. Its average part is vacuous insofar as `d̂_T` is an eigenvector of M. The free check
  is `cos(M d̂_T, d̂_T)`.
- **(c) The separation.** `r(Δx) ∝ ⟨Δx, Kδ_w⟩ > 0` for any change that resembles δ_w, so the separation of one from two
  is designed in.
- **(d) The slice.** `R_f` is the shortest edit that moves r by Δr. "14 % of the squared length" is
  `cos²(CΔx, ŵ)` = 0.086–0.189: geometry.

**Evidence:**
- **(e) Per-frame inertness.** `O_f` is inert in each fresh frame (`F_rem` −0.17 to +0.08), not just on average.
- **(f) The non-head readers.** These are the readers from which `w_c` was not derived.
  - Through `R_f` they deliver about 0.1 of the shift (from the head shares 0.882 and 0.894, if the routes add).
  - The whole of `O_f`, about 86 % of the squared length, nets about 0 through all readers (−0.037 plural, +0.008
    reference; REV §6.1).
  - The split of that zero between the head and the other readers is unmeasured.
- **(g) Completeness.** `F_ins` is 0.81–1.07.
- **(h) The head share and transfer.** The head share of about 88–89 % is consistent with C002's exploratory 0.816 and 0.865 (DR005 A6). The
  effect transfers to `three`, and partly to `many`.

**Alternative H_C (a construction artifact).** Only `L03.H04` responds to r, and the downstream does not read the head
specifically along `d̂_T`. Tests:
- **ALL-BUT-HEAD arms.** These are the complement of 101's head arms: the `p_c` edit, with `L03.H04`'s output at `p_t`
  clamped to its unedited value.
  - LM: the other readers give about +0.1 to `R_f` and about 0 to `O_f`.
  - H_C: their response does not follow the split, or scales with squared length.
- **A write-side split** at `RESID_PRE.L4`, `p_t`, in the `one` prompt. Inject the head's own plural change in two
  parts: `⟨ΔT_f, d̂_T⟩d̂_T` and its off-axis rest.
  - LM: the on-axis part gives about 0.88, and the rest gives about 0.
  - H_C (the downstream uses the head's whole 64-dimensional message): the rest gives at least 0.1.
- **Recipe specificity (a dead-salmon check, LIT D15).** Apply `CΓW^h d̂_T^h` for the seven other layer-3 heads, and
  `CΓWz` for random head axes.
  - LM: each effect follows its `w_c` projection (§2).
  - "Any pullback works": large effects beyond it.
- **Free:** a null for `cos(CΔx_f, ŵ)` (0.29–0.44), built from other heads' and random-axis covectors. Optionally, test
  whether it co-emerges with the behavior across Pythia checkpoints.

### 1.2 Mean difference (a steering vector)

H_MD: any direction aligned with the mean cue difference would do. Comparing `ŵ` with δ̂ isolates K.

- **As an interchange, δ̂ is predicted to pass.** It moves r by `⟨CΔx_f, δ̂⟩⟨δ̂, w_c⟩ ≈ Δr_f`, because PC1 explains
  0.85 of the variance *(sib.)*. So it should pass 101's R and I, with an edit 2–3× longer *(est.)*. R and I therefore
  show that the read coordinate mediates, not that `w_c` beats δ̂.
- **At equal length, projection sufficiency predicts `D_f(0.41)`.** `c(r)` saturates on the plural side: the half dose
  leaves 0.70, and NEC-AMP adds only −0.136 nats. So, against about 0 and about 1 for `ŵ`:
  - removal along δ̂ should leave about 0.75 *(est.)*;
  - installation along it should give about 0.6 *(est.)*.

  H_MD predicts no gap.
- **The per-frame `CΔx_f/‖CΔx_f‖` needs no estimation.** It is the purest steering vector (CAA/ActAdd). The difference
  of means is the best generic steering vector in arXiv 2502.02716.

### 1.3 A linear probe

H_LP: a classifier direction would do.
- **What it is.** Logistic regression and LDA give about `(Σ_W + λI)⁻¹δ`, a mean difference weighted by discriminability
  rather than by the head's read.
- **Why it should be weaker.** It up-weights low-variance directions. Mass-mean directions are more causally implicated
  than logistic-regression ones (Marks and Tegmark). Directions can also be near-perfectly readable yet causally inert
  (LIT D1).
- **Here it is redundant.** Its cosine with `ŵ` is 0.41–0.43 *(sib.)*.
- **Predictions.** LM: `F ≤ D(cos(u_probe, ŵ))`. H_LP: `F(u_probe) ≥ F(ŵ) − m`.
- **LEACE** (the probe coordinate, moved along δ) adds nothing beyond δ̂.

### 1.4 The gradient, or attribution

H_GR. Through the head, `∇_x c = (A/σ)(C − ẑẑᵀ)ΓW q`, where `q = ∂c/∂T` and `ẑ` is the unit centred x. Compare
`w_c = CΓW d̂_T`. So `cos(ŵ, ĝ_head)` asks whether what the head writes on average (`d̂_T`) is what the downstream
reads (q).

- **As a direction.** The gradient is first-order optimal.
  - LM: `ŵ` is close to `P_nat ĝ` (§1.5), and `F(ĝ) ≤ D(cos(ĝ, ŵ)) + ε`.
  - H_GR: the frozen gradient beats `ŵ` at equal length by at least m.
- **As a predictor, against the program P.**
  - First-order attribution patching (AtP) at `x_pl` should under-predict removal, because the gradient is small on the
    saturated plural side. AtP* documents saturation false negatives.
  - Integrated gradients along the edit path sum exactly, so they are the fair competitor.
  - If integrated gradients match P frame by frame, the bespoke program adds nothing for these edits (RS §5 Q1).

### 1.5 The subspace illusion, or a dormant direction

LIT B8: a 1-D patch can pair a varying but disconnected direction with a connected but dormant one.

**Excluded already:**
- generic disruption (family D);
- an artifact of being "louder" after LayerNorm;
- that the natural change acts, in aggregate, other than through r: SUF-READ ≈ PC-SWAP-REF (1.004 against 1.008), and
  NEC-READ ≈ PC-SWAP-PL (−0.009 against 0.044);
- any illusion in the head's `d̂_T` channel. `w_c` is the head's literal input covector, r varies naturally, and `w_c`
  was never fitted to the counterfactual, which is how such illusions are usually produced.

**Not excluded:**
- **Additivity masks it.** `F_rem + F_ins` = 0.989. If `R_f` worked through a dormant route, `O_f` would carry its
  negative and net about 0. That is exactly 101's pattern.
- **`ŵ` may lie partly outside the natural span.** `ŵ ∝ Kδ_w` need not lie in the span of natural cue variation, so `R_f`
  may contain an unnatural component that `O_f` cancels.
- **What it could explain:** the non-head 0.1, the opposition of −0.037, and the head's off-axis output (§1.1 b).
- **D's edits leave the natural range.** NEC-AMP and SUF-NEG move r beyond the natural range in 36 of 36 frames
  *(sib.)*.

**The test.**
- `P_nat` projects onto the top-k principal components of cue-induced changes over all exposed cues and frames, with k
  frozen at 95 % of variance. Report `‖P_nat ŵ‖` free; RS §4.5 suggested it.
- **READ-NAT:** `e = Δr_f·ŵ_nat/⟨ŵ_nat, w_c⟩`, with `ŵ_nat = P_nat ŵ` normalized. It moves r by exactly `Δr_f`,
  within the span.
- **DORMANT:** length `s_f` along `(I − P_nat)ŵ`.
- Run both also as ALL-BUT-HEAD arms, and report Mahalanobis lengths.

**Predictions.**
- LM: READ-NAT reproduces READ, head share included, and the non-head response to DORMANT is about 0.
- The illusion: the opposite on both counts.

### 1.6 Site and layer non-specificity: see §3

### 1.7 An axis estimated in sample

H_IS: `w_c` works because `d̂_T` came from one/two.

**Control:**
- Build `w^(−) = CΓW d̂_T^(−)` from each/several only.
- Build a second version from a frozen pool with no numerals, excluding `a`, `this` and `another`, whose transport
  collapses (the 009 design, line 18).
- Report `cos(w^(−), w_c)` and leave-one-pair-out stability, both free.
- Rerun R, I and the off-read controls with `w^(−)`.

**Predictions.**
- LM: cosine of at least 0.9, with medians of N and S of at least 0.85.
- H_IS: a material drop.

This control also removes construction effect (b). `three` already transfers, but it is a numeral.

## 2. Baselines

**Fairness rules:**
1. **The same data budget.** Every direction comes from the `d̂_T` pool (a second tier uses all exposed cues) and is
   frozen by digest before any fresh prompt. Per-frame oracles are labelled as oracles.
2. **Two forms for each unit direction u:** equal length, `x ∓ s_f·u`, and interchange, `x ∓ ⟨CΔx_f, u⟩u`. Report the
   edit length and the realized Δr (exact LayerNorm) for both.
3. **Budget.** Equal centred L2 length is primary. The post-LayerNorm footprint and the Mahalanobis length are secondary.
4. **Same material.** Use the same frames, nouns and prompts. Record `L03.H04`'s whole ΔT at `p_t` (RS §4.5).
5. **Projection sufficiency, H_PS:** `F_f(u) = D_f(ρ_{u,f})`.
   - ρ is the realized change in the `w_c` coordinate as a fraction of `Δr_f`.
   - `D_f` is the frame's dose–response to pure read edits on a ladder ρ ∈ {±0.25, ±0.5, ±0.75, ±1}. 101 gives
     D(0.5) ≈ 0.71 and D(1) ≈ 1.
   - Under H_PS no direction beats `ŵ` per unit length, and `ED50(ŵ)/ED50(u) = |cos(u, ŵ)|`.
   - Removal discriminates directions best, because the plural side saturates. Installation detects residual
     sensitivity best, because the singular side is steep.

**Decisions** are paired by frame, with m and ε calibrated on exposed frames:
- **Favours `ŵ`:** `ŵ` beats u by at least m in at least 26 of 36 frames, and H_PS holds within ε.
- **Favours the baseline:** u beats `ŵ` by at least m.
- **Adds nothing but is not wrong:** TOST equivalence within ±m_eq. An m_eq of 0.05–0.10 is plausible, since `F_ins`
  has a between-frame SD of about 0.05.
  - With `cos(u, ŵ)` of at least 0.9, this is convergence: the weights find without fitting what the data finds, and
    their value is explanatory.
  - With a cosine below 0.9, H_PS fails: a second effective direction exists, so any claim must be about a subspace.

| baseline | construction (at `RESID_PRE.L3`, `p_c`) | LM prediction | favours the baseline if |
|---|---|---|---|
| mean difference | δ̂, the mean `CΔx` over the pool; per frame, `CΔx_f` normalized | equal length: removal leaves about 0.75, installation gives about 0.6 *(est.)*. As an interchange it passes R and I with an edit 2–3× longer | equal-length removal within m of `ŵ` |
| PCA | PC1 of the uncentred {`CΔx`}; top-k subspaces | about δ̂ (cosine 0.41 *(sib.)*) | within m of `ŵ` |
| probe | L2 logistic regression (λ by leave-frame-out cross-validation) and shrinkage LDA; a control task (Hewitt and Liang) | accuracy about 1; F no higher than δ̂'s | F ≥ `F(ŵ)` − m |
| gradient | frozen ḡ from exposed frames; a per-frame oracle; integrated gradients along `Δx_f`; `P_nat ḡ`; the adversary `P_nat ḡ`⊥`ŵ` | `cos(ŵ, P_nat ḡ)` of at least 0.8; the adversary installs at most ε | F exceeds `D(ρ)` by more than ε |
| DAS | 1-D, trained by interchange on the pool and on a diverse pool; at least 5 seeds; a control task (LIT A7, B2) | linearized, the in-span part of any exact solution is parallel to the in-span gradient, so `u_DAS ≈ ŵ_nat`; unidentified on one/two alone | better fresh interchange accuracy with a shorter edit |
| random | 101's 8 isotropic directions, plus 8 random head axes (`CΓWz`) | `D(ρ)`, about 0 | — |
| other heads | `w^h` for the seven other layer-3 heads | `D(ρ_h)` | effects beyond the projection |

## 3. Negative controls: site and layer

Two alternatives are tested here:
- **H_S:** `w_c` is a generic "plural" direction that works at any site;
- **H_L:** it works at any layer.

LM and H_C both predict site specificity. So these controls separate "generic" from "read here", and §1.1 separates LM
from H_C. They use 101's own vectors, each with a footprint-matched copy and a whole-position swap at the same site.

| control | LM prediction | alternative |
|---|---|---|
| `RESID_PRE.L4`, `p_c` | `F_ins` ≲ 0.15 and `F_rem` ≳ 0.85. Only readers at layers 4–5 remain: `L05.H07` attends at 0.999, and 005's layer-4 patch recovered 0.125 (DR005) | H_S/H_L: `F_ins` of at least 0.3 |
| `RESID_PRE.L3`, `p_t` | small, preregistered as the integrated-gradient effect of `R_f` at `p_t`; report `cos(ŵ, d̂_T)` | comparable to the effect at `p_c` |
| `RESID_PRE.L3`, `p_c−1` and other positions | the gain law, `ρ_j = (a_{t,j}/σ_j)/(a_{t,c}/σ_c)`, from the stage-1 attention | unrelated to attention |
| `RESID_PRE.L2`, `p_c` | mediated: `F ≈ D_f(realized Δr at L3 / Δr_f)`. State the `L02.H07` route (attention 0.893) in advance | departs from the realized-Δr prediction |
| write side, `RESID_PRE.L4`, `p_t` | `d̂_T` at its natural size gives about the head share; the off-axis change gives about 0 | the downstream reads off `d̂_T` |

**The target is a double dissociation.** `ŵ` acts at (L3, `p_c`) but not at (L4, `p_c`) or (L3, `p_t`), and `d̂_T` acts at
(L4, `p_t`). A generic number direction predicts no dissociation.

## 4. Individual predictions

Keep apart **(A) causal completeness per noun**, which is a measurement, and **(B) the program's per-noun accuracy**,
which is capped by ceiling errors of 0.7–1.2 nats (CL, SF2). Calibrate each threshold on 101's spent data, then
preregister it.

1. **Completeness.** Use the cells with `y_{fn,PL}` ≤ −1 nat.
   - Report the shares with `|F^rem_{fn}| ≤ 0.2` and with `|F^ins_{fn} − 1| ≤ 0.2`.
   - A *noun exception* has a frame-median `F^ins` below 0.8, with a bootstrap upper bound below 0.9.
   - LM predicts few. Noun-specific routes predict a stable subset.
2. **Skill.** In each frame and arm, compute `R²_within = 1 − Σ_n(y − ŷ)²/Σ_n(y − ȳ_f)²` (skill beyond the frame mean),
   and the Spearman correlation over nouns with a sign count across frames.
3. **Error structure.** Fit `e_{fna} = μ_a + u_{fa} + v_{na} + ε`. Do noun biases from exposed frames predict fresh errors
   (Spearman of at least 0.5)? If so, the readout's error is fixed and can be corrected.
4. **Calibration.** Per arm, the Deming slope in [0.9, 1.1], with a bootstrap crossed over frames and nouns. Report Lin's
   concordance on `Δc`, not on the fractions.
5. **Coverage.** Split-conformal intervals per noun, clustered by frame, should reach at least the nominal level minus
   0.05.
6. **Frame ranks,** computable now from 101's table: Spearman of `F̂_rem` against `F_rem`, and the AUROC of `Â_amp` for
   the sign of `A_amp`. Did the program foresee the eight saturating frames (06, 09, 15, 22, 23, 32, 34 and 36)?
7. **The ceiling split:** `(ŷ − y^ceil) + (y^ceil − y)`.

**Population.** Sample the frames from a grammar-generated pool with a committed seed, across at least two templates,
so that the tails describe a defined population (REV §6.3). Add fresh nouns if any are eligible.

## 5. Transfer to Pythia-160M (12 × 12 heads, d = 768)

**Transfer the method, not the answer.** Freeze a pipeline M, and hash it before 160M is loaded:
1. a behavior screen;
2. a position map;
3. attribution of each head's transport from `p_c` to `p_t`;
4. a selection rule: the top head; if its share is below 0.5, the smallest set reaching 0.8, as a read subspace; if no
   set of at most three heads reaches it, report "distributed";
5. `d̂_T` from a fixed list of pairs;
6. `w_c`;
7. the arms and statistics.

The sibling scratch memo `survey-160m-discovery.md` sketches such an algorithm.

**Transfer means** that, whichever head M picks:
- M's prediction table holds on fresh frames. It is built from the weights and exposed states only and committed before
  any confirmatory edit, as in 101's stage 1.
- The read passes §1–§3: it beats the equal-length cue change on removal, the site nulls hold, and the held-out axis
  works.

Searching mid-layer heads that attend to the cue for "the `L03.H04` analogue" is not transfer. It replicates a motif
with the researcher in the loop, and a number-carrying head is already known elsewhere (LIT A2).

**Guards:**
1. **A blind positive control.** M on 70M from scratch, without 011's lock, must rank `L03.H04` first and recover `w_c`
   at a cosine of at least 0.95.
2. **An answer we do not know:** a second 70M seed. PolyPythias has 9 extra seeds per size, 14M–410M. Check its data
   variant against the deduped model.
3. **A negative-control model.** An early checkpoint, or a model without the behavior, must stop at the precondition.
4. **Structural predictions only,** for example "a slice of at most 30 % carries at least 0.8". Label any number taken
   from 70M.
5. **Separate roles.** The discovery agent does not see 70M's results. Publish the full head ranking and the forking
   paths.
6. **Kill criteria, kept as findings:** a failed R or I, equivalence with the steering baseline, or a site control that
   is not null.

## 6. Ranked by information per unit cost

Cost codes:
- **Z:** spent data and weights only.
- **L:** at most 10 arms × 36 frames. At about 0.07 s per forward pass, design and review dominate the cost.
- **M:** a new harness.
- **H:** a new optimization harness or model.

1. **[Z] Spent-data analytics.** `cos(w_c, Kδ_w)`, `cos(M d̂_T, d̂_T)`, `‖P_nat ŵ‖`, `d̂_T^(−)` and its stability,
   the alignment null, and §4 on 101's data. They set every prediction below.
2. **[L] The held-out axis `w^(−)`.** It is the only direct test of H_IS, and it removes construction effect (b).
3. **[L] Site and layer nulls:** L4 at `p_c` (raw and matched), L3 at `p_t`, L3 at `p_c−1`, and the mediation test from
   L2.
4. **[L–M] ALL-BUT-HEAD, READ-NAT and DORMANT.** They test the part of 101 that is not construction, and the illusion's
   open gap.
5. **[L] The equal-length cue change, the mean difference and the D ladder.** They are largely predictable from 101, but
   they measure K and make H_PS usable.
6. **[M] The gradient family,** with integrated gradients and AtP against P.
7. **[M] The write-side split and random head axes.**
8. **[L] The other heads' recipe,** as the dead-salmon guard before series II uses the recipe as a tool.
9. **[L] Probe and PCA.** They are redundant at a cosine of 0.41 *(sib.)*.
10. **[H] Full DAS,** only if `P_nat ĝ` diverges from `ŵ_nat`.
11. **[H] The blind pipeline:** 70M, then a 70M seed, then 160M.

**A compact primary family, Holm-corrected:**
- R and I with `w^(−)`;
- the L4 null at `p_c`;
- equal-length removal along `ŵ` against the frame's own normalized cue change;
- READ-NAT's non-head share against READ's.

Everything else is secondary, reported with intervals.

## Sources

**Repository:**
- CL lines 20–44 and 67–80; REV §3, §5, §6 and App. A;
- D101 §2, §4 and §7; RS §4.5 and §5;
- LIT A2, A7, B2, B8, D1, D15 and line 994;
- DR005 lines 29–50; the 009 design, lines 18 and 52–56.

**Web (this session):**
- [CausalGym](https://aclanthology.org/2024.acl-long.785), [arXiv 2402.12560](https://arxiv.org/abs/2402.12560v1);
- [Marks & Tegmark](https://arxiv.org/pdf/2310.06824);
- [AtP*](https://arxiv.org/pdf/2403.00745);
- [steering evaluation](https://arxiv.org/pdf/2502.02716);
- [PolyPythias](https://arxiv.org/abs/2503.09543);
- [Tigges et al.](https://aclanthology.org/2024.blackboxnlp-1.5): searched, not relied on.

**Prior knowledge, not re-verified:** arXiv 2312.06681 (CAA), 2308.10248 (ActAdd), 2306.03819 (LEACE), 1703.01365
(integrated gradients), 1909.03368 (control tasks).

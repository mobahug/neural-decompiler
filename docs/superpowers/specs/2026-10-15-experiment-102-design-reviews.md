# Experiment 102 design: the reviews

Companion file of `2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`. Every review here is an
independent AI review (a Claude agent, read-only), not a human validation.

| review | of | verdict | source sha256 |
|---|---|---|---|
| 1 (2026-10-15) | revision 1, sha256 `84b24685…796f` (commit `1cf99b4`) | NOT READY: 2 blockers, 11 SHOULDs, 9 notes | `edd86fe9bbb7633766db40af82a1f64f4d98bd0ef747ad848ceb03c335e88dfb` |

Revision 2 of the design applies every finding; its section "What changed in revision 2" maps each one.

---

## Review 1 (verbatim)

# Experiment 102 design: independent review

**AI review, not human validation.** A Claude agent (Opus 5.5) wrote this review on 2026-10-15, read-only.
- Nothing in the repository was created, changed, staged or committed.
- No model, runner, test or forward pass ran, and no network was used.
- The only computations were arithmetic on numbers already in the record: binomial tails, Holm thresholds, power, and ladder interpolations from 101's recorded medians.

**The design under review.**
- File: `docs/superpowers/specs/2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`, revision 1, 446 lines, uncommitted on branch `c004-exp102-design`.
- **The file changed while this review ran.**
  - At the start (HEAD `a9b6b33`) its sha256 was **`2c2da3633c01ad042391611656a49165dad3c3ad7593969e912c92db85bf5f0c`** (5,915 words).
  - At the end (HEAD `b821773`, which committed the C004 draft, the assessment, the checks and the surveys) it is **`84b24685dc87d2bd8ce5168bd70f35984196eca46612f2fb3d2cdacbb804796f`** (5,916 words, 36,238 bytes). This is byte-identical to the scratchpad copy `exp102/design-rev1.md`.
  - The only change I found is line 13, which now names `research/synthesis/exp102-surveys/`.
- **Every line cited below** was re-checked against `84b24685…`, and the line numbers are unchanged.
- **The two surveys** are cited as "160M survey" and "baselines survey". They refer to the committed copies in `research/synthesis/exp102-surveys/` (`survey-160m-discovery-2026-10-14.md` and `survey-baselines-controls-2026-10-14.md`). Both are byte-identical to the scratch copies.

**What I read.**
- Experiment 101: the design (§2, §4, §6–§8, §11–§14), the closure, the calibration review (§10) and the confirmation review.
- `docs/RESEARCH_METHODOLOGY.md`.
- The C004 draft and the critical assessment.
- X101-1 and X101-2 (`.py` and `.out`).
- Both scratch surveys.
- Experiment 005: the discovery report and design §A2–A6.
- The 009 README and `models.py`.
- 101's `confirmation-v1.json` and `calibration-v1.json`.
- The HF cache metadata: configs, tokenizer hashes and refs.

## Verdict: NOT READY

The core of the design is sound and genuinely falsifiable:
- MD-1 is frozen and mostly mechanical. It runs once on frames that are spent for 70M and can stop at four places.
- Removal (R) and installation (I) are then tested on frames that no model has ever run.
- Selection happens on D and testing on C, the families are separate per model, and the scope is stated honestly.

The design falls short in two places:
- **The add-on tests.** Two of the four preregistered add-ons cannot do what the user asked of them:
  - PS (H5) cannot fail under its own refutation condition.
  - S (H2), which gates the 160M verdict, is largely implied by MD-1's own layer criterion, and it mislabels its most likely failure.

  Both fixes are local changes to the text. With them and the SHOULD items applied, I expect READY WITH CHANGES or better.
- **The specification.** Several MD-1 details and gate rules still need specifying (S7–S9).

**Checked and correct.**
- **Statistics.**
  - Holm thresholds are 26/25 for two families and 26/26/25 for three, with tails 0.00567 and 0.01441; 24/36 would give 0.0326.
  - §8.6's power figures are correct: 0.911/0.958 at 80 %, 0.725/0.833 at 75 %, 0.466/0.609 at 70 %.
  - Serial gatekeeping as described controls the familywise error at 0.025 per model.
  - The worst-value and tie rules are conservative.
- **The 70M reference values.** 005's A3, A4 and A6 values match the discovery report (:29–51), and Δ₃ = 0.956 − 0.125 = 0.831. X101-2's off-axis share is 0.3445.
- **The 160M model.**
  - It is pinned at `582159a2`, has no `refs/main`, and is 12 layers × 12 heads, d_model 768, MLP 3072, with a parallel residual.
  - Its `tokenizer.json` is identical to 70M's (`c24618a1…f25624`).
- **Compute.** About 78 arms per frame on 70M and 80 on 160M, and about 12.4k MD-1 passes on 160M.
- **Data hygiene.**
  - Candidates 37–60 are all `eligible: true` in 101's `confirmation-v1.json`, whose picks are 1–36.
  - 101's calibration reports cover only D's 36 frames, and 101's freeze loaded no model.
  - Using frames spent on 70M for the 160M discovery is acceptable: no 160M output exists, MD-1 is mechanical and C is fresh for both models. The remaining selection effect is stated (line 386).
  - The reserve is well defined by rule.
- **The 70M retrodiction should pass.**
  - On D, PC-SWAP gives R_c(3) ≈ ½(0.995 + 0.965) ≈ 0.98 (calibration review :377).
  - `L03.H04`'s edited output carries about 0.93 of the removal (`F_head_rem` 0.069, :502), which suggests a singleton recovery s_h ≈ 0.9 against the 0.70 threshold.
  - The one unmeasured quantity is R_c(4), 0.125 on two frames. CONCENTRATED fails only if it exceeds about 0.48.

## BLOCKER

### B1. PS (H5) cannot be refuted by a stable second channel, and its strongest refutation is labelled "not evaluable"

**Where:** §7 lines 199–201, §8.3 lines 236–245, §8.4 line 262.

**Problem.** ε is the 90th percentile, on D, of the very discrepancy that PS tests. A second effective channel is a property of the model, so it appears on D and C alike, and ε absorbs it.
- **The geometry.** NEC-DX = cosθ·R + s·sinθ·û_off, with cosθ = ρ(DX) = 0.305–0.429 on D (X101-1). DX's off-read slice is therefore 0.90–0.95·s long, along exactly the OFF arm's direction.
- **What ε absorbs.** Suppose that direction carried g of the shift per length s. Then F(NEC-DX) and F(SUF-DX) would deviate from the ladder by about 0.93·g in every frame.
  - g = 0.05, 0.10 or 0.15 gives ε ≈ 0.05, 0.09 or 0.14. PS then passes on C (at least 27/36), labelled `PROJECTION_SUFFICIENT`.
  - From g ≈ 0.20, ε exceeds 0.20 and the label becomes `PROJECTION_SUFFICIENCY_NOT_EVALUABLE`.
- **D passes by construction.** The nearest-rank rule guarantees at least 30/36 joint agreement on D. PS on C therefore only tests that C is no worse than D.
- **Only a plane is tested.** DX lies in span(R, û_off), and both of those arms already run. PS therefore tests additivity within a plane that R and I already probe, not "any edit".

**Why it matters.**
- H5's stated refutation is "a second effective channel exists at the site" (line 56). That is exactly the case this test cannot detect.
- Calibrating τ on exposed data was legitimate for 101's P, which measured a program's fidelity. It is not legitimate for a scientific sufficiency claim.

**Fix.**
- **The tolerance.** Replace §7's ε rule with:

  > ε_rem = ε_ins = max(0.05, e₉₀), where e₉₀ is the nearest-rank 90th percentile, on D, of the ladder's leave-one-rung-out interpolation error. The error is computed by predicting F(NEC-L50) from the rungs {0, 0.25, 0.75, 1}, and likewise for installation. No DX arm enters ε. If e₉₀ > 0.10, PS is uninformative.

  D's DX discrepancies are then reported as exploratory evidence about H5.
- **The signed criterion.** Add: "and each side's median signed deviation lies within ±0.05".
- **The edits.** Make GRAD and MD (frozen from D, with components outside each frame's plane) PS edits alongside DX. GRAD is the most adversarial direction for a hidden channel. PS passes only if each of DX, GRAD and MD reaches 27/36.
- **The interpretation.** With these changes, §11's "predicts other edits" (line 347) becomes true.

### B2. S (H2), the verdict's gate, is near-implied by MD-1 (c), cannot discriminate a generic direction, misattributes incompleteness, and passes by construction when L* is the last layer

**Where:** §2 line 53, §6 line 172, §8.2 line 229, §8.4 line 272, §11 line 350.

**Problem.**
1. **S follows from MD-1's own criterion.**
   - MD-1 chooses L* so that the whole-residual patch at `RESID_PRE.(L*+1)` recovers at most R_c(L*) − 0.5 on D (line 119). S's floor is the same 0.5.
   - A complete read therefore passes S, unless R at L*+1 acts more strongly than the whole natural difference there, or D→C shifts. S can fail essentially only through incompleteness.
   - For 70M, the record predicts Z ≈ 0.86 or more (F_rem −0.035 and F_ins 0.980 on D, with L4 recovery 0.125).
2. **S cannot discriminate a generic direction.** MD-1 chose L* because the site (L*+1, p_c) has little leverage left, so any direction, generic or not, does little there. S therefore cannot discriminate the critical assessment's alternative (§2), a generic plural direction that works wherever it is added.
3. **S mislabels an incomplete read.**
   - Z = min(F(NEC-SITE) − F_rem, F_ins − F(SUF-SITE)) with a median of at least 0.5 requires the read itself to remove and install at least 0.5.
   - A site-specific read that carries 0.45 therefore fails S. It is labelled `METHOD_READ_CAUSAL_NOT_SITE_SPECIFIC`, read as "acts where it is written" (line 350), which is false in that case.
4. **S passes by construction at the last layer.** If L* = n_layers − 1, `RESID_PRE.(L*+1)` does not exist. Edited as the final residual at p_c, it cannot reach p_t, so F(NEC-SITE) = 1 and F(SUF-SITE) = 0 by construction.

**Why it matters.** The user asked for specificity controls. The verdict's only specificity gate is a re-test of the selection criterion, with an effect-size floor in disguise.

**Fix.**
- **Make the primary site control a position control.**
  - Use `RESID_PRE.L*` at `p_t`, in both directions, by adding SUF-PT (`x_ref(p_t) + R`) next to NEC-PT.
  - A generic number direction predicts an effect there and a read of p_c does not. MD-1 does not bound this effect.
  - Use the statistic `Z′_f = min[(1 − F_rem) − 2(1 − F(NEC-PT)), F_ins − 2·F(SUF-PT)]`, whose count of `Z′_f > 0` means "less than half the effect at p_c, in both directions".
- **Keep L*+1 as a secondary check** that consumption is complete, in the same relative form.
- **Add a magnitude criterion M to the verdict:** median F_rem ≤ 0.5 and median F_ins ≥ 0.5. If R and I pass but M fails, the label is `METHOD_READ_PARTIAL`, not a site label.
- **Add to §4(c):** "if L* = n_layers − 1, stop: `CONSUMPTION_AT_FINAL_LAYER`."
- **Optionally, add NEC/SUF at p_c − 1.** The attention-gain law predicts that effect quantitatively (baselines survey §3).

## SHOULD

### S1. B (H3) follows from the OFF arms, the ladder and geometry; it is not a comparison with a naive method

**Where:** §2 line 54, §6 line 170, §8.2 line 230, §14 D3.

**Problem.**
- **B follows from geometry.**
  - R is the minimum-norm edit that moves r by Δr. At equal L2 length, any other direction moves r by only cosθ as much.
  - B can therefore fail only if something other than r matters. That is PS's question, asked of the same two arms.
- **The record predicts the outcome.**
  - D's ladder (half doses 0.734 and 0.727, calibration review :376) predicts B = 0.54, 0.43 and 0.36 at cosθ = 0.305, 0.377 and 0.429, against a floor of 0.10. This matches line 294.
  - The OFF control already refutes "any cue-aligned direction would do": û_off is about 0.9 cosine to the mean cue change.

**Fix.**
- **Wording.** Say that B measures efficiency per unit length.
- **Interchange arms.** Add the interchange form for MD, PC1, LDA and GRAD: set the coordinate along u to the other prompt's value, at length |⟨CΔx,u⟩|. The baselines survey's fairness rule 2 requires this (survey :209). It is what a practitioner would actually run, and it is predicted to pass R and I with edits 2–3× longer.
- **Statistics.** Preregister, for every baseline, its B_f count and median and its interchange R/I counts, with no label force.
- **Edit sizes.** Report every edit's Mahalanobis length (D's within-cue covariance) and post-LayerNorm footprint, and B under a budget matched on that footprint, as secondary.
- **Definitions.**
  - Orient each u so that ⟨u, mean CΔx_D⟩ > 0.
  - Define GRAD as the centred mean over D of ∇c̄ at x_pl and x_ref.
  - State LDA's classes.
- **Optional.** A one-dimensional DAS direction trained on D, as a secondary baseline.

### S2. The discovery procedure is compared with naive procedures only in part

**Where:** §4 lines 127–133, §6 lines 178 and 183.

**Problem.**
- Only the attention-selected head's read reaches the outcome, and only when it differs from T.
- The heads chosen by DLA, and by the top singleton recovery in any layer, are reported but never tested.
- The other-head reads (W^h) run removal only, at their own lengths, which confounds selectivity with edit length.

**Fix.** This addresses the user's "simple baselines for the circuit-finding method".
- **More naive reads.** Add DLA-READ and TOP-READ arms (NEC and SUF) whenever those heads differ from T. Report an attribution-patching head ranking as a further naive selector.
- **W^h at equal length.** Run W^h at equal length s, in both directions.
- **Selectivity statistic.** Preregister "T's read removes and installs more than the median other-head read of layer L*, at equal length", counted against 26/36, with no force.
- **Same statistics.** Report the primary R/I statistics, descriptively, for ATTN-READ, DLA-READ and TOP-READ.

### S3. Nothing excludes disruption along the selected direction in 160M

**Where:** §6 line 174.

**Problem.**
- 101 made its direction family D primary because disruption toward neutral "mimics both a removal effect and an installation effect" (101 design :145–148). 102 demotes D to secondary with no statistics, and the R/I floors are 0.05.
- A direction whose perturbation pushes c toward neutral in both prompts gives F_rem ≈ F_ins ≈ 0.5. It passes R and I against an inert OFF control and sits at S's floor.

**Fix.**
- Add A_neg > 0 in 25/36 as a 160M gate. It was 36/36 in 101, and the plural side's saturation does not affect it.
- Keep A_amp descriptive, because of saturation and the off-range extrapolation shown by X101-1.
- B2's criterion M covers the rest.

### S4. H (H4) does not test the stated concern, a material loss

**Where:** §8.2 line 231.

**Problem.** A median of 0.05 beyond the held-out read's own control passes a held-out read that carries a tenth of the shift. The critical assessment (lines 34–36) asks for "the same result within a stated margin".

**Fix.**
- **Non-inferiority.** Require F(NEC-HREAD) ≤ F_rem + 0.10 and F(SUF-HREAD) ≥ F_ins − 0.10 in at least 26/36 frames, with both median differences within ±0.10.
- **The axis cosine.** Report cos(V, V^(−)). Preregister that at 0.98 or above the test is near-redundant with R and I.
- **The 70M comparator.**
  - In 70M the in-sample comparator is 011's axis. It was estimated on `one`/`two` plus `each`/`several`, including cue-final frames, so the recipe and the frames differ as well as the cue.
  - Also compare with MD-1's own 70M read (BRIDGE).

### S5. Stage-2 Holm couples the verdict's gate to its qualifiers

**Where:** lines 59–60 against lines 226–227.

**Problem.**
- S, B and H share one Holm family, so S needs 26/36 or 25/36 depending on B's and H's p-values. For example, S at 25/36 fails when B and H are weak.
- B and H therefore partly gate `METHOD_TRANSFERS`, which contradicts line 60.

**Fix.** Use serial gatekeeping, each stage at 0.025:
1. R and I (26/25);
2. the verdict's remaining gates, the position control and A_neg (26/25);
3. B and H (26/25).

PS stays an accuracy family. The familywise error stays at 0.025, and the qualifiers no longer move any gate's threshold.

### S6. No individual prediction is prospective

**Where:** §8.3, §8.5, §14 D7.

**Problem.**
- PS's "prediction" is an interpolation within the same run.
- 101's standard was a per-frame table committed before any edit. 102 drops it.

**Fix (D7's alternative).**
1. After the lock, run REF and PL on C and compute every edit's ρ per frame.
2. Predict each frame's and arm's F from D's frozen dose–response, using a pooled ladder with a leave-one-frame-out tolerance from D.
3. Commit the table.
4. Run the edits.
5. Score the table with a preregistered count.

**The per-noun criteria.**
- Keeping §8.5's completeness criteria without force is defensible: the 79 nouns are correlated, because they share frames.
- ε_noun (line 286) carries B1's defect. Fix it the same way.

### S7. MD-1's governance

**Where:** §4 lines 106 and 135–142; §10 lines 321 and 340.

**Problems and fixes.**
- **Contradiction.** "Runs once per model" and "No phase is repeated" contradict the open revise-and-rerun loop of the retrodiction. Cap the loop: at most one revision, and a third version requires a design revision. A revision's reason must be a stated principle, not a 70M number.
- **Commitment.** The user should commit the passing MD-1 before authorization 2: its source digest, its thresholds and the D1/D2 lists. The 160M launcher should refuse to run otherwise.
- **The positive control.**
  - The 160M survey (:167) required cos(w_c^MD-1, w_c^011) ≥ 0.95 for the retrodiction to pass, and the design dropped that condition.
  - MD-1's own 70M read is therefore tested only by secondary BRIDGE arms. The method's output is never positive-controlled with force on the model whose answer is known.
  - Either give BRIDGE the R/I statistics as a preregistered positive control, or restore the condition. If the control is adopted and fails, a 160M `METHOD_TRANSFERS` cannot be read as a result about the method.

### S8. MD-1 is not yet fully mechanical

**Where:** §4.

**Specify:**
- whether Δ_L is the median of per-frame differences or a difference of medians;
- the frames and weights that enter d̂_T^h;
- for (e′), whether the 72 pairs are pooled or the two cues' means are averaged;
- for (e″), which run supplies A_c and σ_c, and whether R² is the squared Pearson correlation or 1 − SSE/SST about the identity line;
- the hook of the head patch: each head's result at p_t only;
- the case of fewer candidate heads than needed.

**The attention filter** (≥ 0.3, line 120) is shaped by 70M. In Pythia, attention sinks can push a genuine 160M reader below it, which would stop MD-1 spuriously. Consider 0.1, or ranking by s_h alone (D2).

**A robustness map.** Publish, as a preregistered output, L* and T under ±0.1 on each threshold, computed from the saved measurements.

### S9. Gates can void the primary label for secondary reasons, and 160M tolerances need setting

**Where:** §8.4 line 253, §9.

**Problem.**
- "Any gate fails → NOT_INTERPRETABLE" lets a G5 failure on the secondary head arms void the 160M verdict.
- 70M's layer-3 attention scores already carry float32 offsets of 1.2–3.3e4 (calibration review :452–453). G3's absolute 1e-4 and G5's tolerances are therefore risky at d = 768.

**Fix.**
- **Which gates void the label.**
  - Only primary-arm gates void the label: G0–G3 on the READ, OFF, site, DX, ladder and H arms, plus G6 and G7.
  - A secondary-arm gate failure voids that arm alone.
- **160M tolerances.** Set 160M's G3 and G5 tolerances in the contract phase, on neutral prompts (for example, 10× the measured float32 floor), and freeze them before discovery.
- **No D or C text before discovery.**
  - Forbid any 160M smoke or rehearsal run on any D or C text.
  - 101's confirm smoke reran D's frame `coordinated-adjective-1` (confirmation review :68, :384–386). Reusing that launcher would leak D on 160M.

### S10. Labels overclaim the scope

**Where:** §8.4 lines 264–276, §11.

**Fix.**
- **Rename the verdict.** `METHOD_TRANSFERS` names a general property from one behavior and one template, across one pair of models with the same family, data and tokenizer. Use `MD1_READ_TRANSFERS_TO_160M`, with a mandatory scope string.
- **State what is tested.** Say in §1 and §11 that 102 tests the read-discovery recipe (patching for layer and heads, plus a weight-derived value read). It does not test C003's decompiled programs, since P has no 160M counterpart.
- **Name a stop as a failure.** §11 should state that a discovery stop answers H1 negatively for MD-1 as frozen. "The shape of the failure is the finding" invites spin.
- **Give the cosine split force.** Move §11's 0.9 cosine split (line 348) into §8.4.

### S11. "C004's first Level-4 tests" omit C004's own listed discriminating tests

**Where:** line 40, lines 355–356.

**Problem.** The C004 draft lists two tests as open (draft :211, :335):
- edits restricted to, and outside, the span of natural cue changes (READ-NAT and DORMANT, against the subspace illusion);
- the edits' effects on the rest of the next-token distribution.

**Fix.**
- **The span arms.** Add the 4 arms, with P_nat frozen from D at 95 % variance (survey :176–182).
- **The distribution.** For each arm, record the KL divergence at p_t and the change in log-probability of non-noun tokens. Both are free.
- **The alternative.** Otherwise, narrow line 40 to "Level-4 tests of site, held-out axis and per-length efficiency".

## NOTE

- **N1.** Lines 6–7 cite a companion review file that does not exist yet.
- **N2.** The P1 reference (line 124) is not comparable.
  - 009's R² of 0.909 is over 720 exposed (token, frame) pairs, including suppressed cues (009 README :102–104).
  - 102's 108 strongly plural pairs vary far less. 70M may therefore read `P1_NOT_ESTABLISHED` even where P1 holds.
- **N3.** D contains 005's two development frames (`coordinated-adjective-1` and `-2`) and its extension frames. The retrodiction is therefore partly in-sample to the original selection of the head. Say so.
- **N4.** C's new frames include adjectives that can be read as nouns: `tender` (65), and `plain`, `vintage`, `antique` and `hollow` (69–72).
  - 101 replaced such adjectives (101 design :484, :500–501).
  - The tokenizer pre-check lives only in the scratchpad.
  - Run a linguistic check before the freeze, and commit it.
- **N5.** G7 should match prompts by token-id sequence, for two reasons:
  - frames 37–60 will get new frame ids;
  - the committed ledgers are digests only (`calibration-v1.json` stores `keys` as a count and a sha256).
- **N6.** §8.6's power is per family. The verdict needs R, I and then S, which together reach about 0.88 at 80 % sign shares. State a minimal effect of interest for 160M.
- **N7.** Consider a negative-control checkpoint, such as an early 160M step, on which MD-1 must stop (survey :313). It would measure the method's false-positive rate.
- **N8.** The 70M battery carries little risk:
  - R and I are known from 101;
  - S is predicted at about 0.86, and B at 0.36–0.54.

  Its Level-4 value lies in H, PS, the position control and S11's arms. §11 should say so.
- **N9.** Two small labelling fixes:
  - Line 294's B figures are the survey's estimates, not measurements; mark them "(est.)".
  - The gates skip G4 with no note.

## The most important weakness, and the user's question

**The most important weakness.** If MD-1's 160M discovery passes, nearly every downstream component follows from it or from 101's known inert off-read part:
- S follows from MD-1's own layer criterion.
- B follows from geometry and the OFF arm.
- PS follows from a tolerance calibrated on the discrepancy it tests.
- H follows from near-collinear numeral axes.

The risky content of 102 is therefore MD-1's discovery, plus R and I on fresh frames. That core is a real test, and its hygiene is good.

**Would the revised design answer "does our decompilation method generalize?"** It would answer a narrower question, cleanly and falsifiably, once these fixes are in:
- B1 and B2: a position control, and a fixed-margin sufficiency test with an adversarial direction.
- S1–S3: interchange and naive-selection baselines, and a test that excludes disruption.

The narrower question is whether a frozen read-discovery recipe, built on 70M, finds a site-specific causal read in an unstudied sibling model, for the same behavior and template, and does so more efficiently than naive directions and head choices.

A positive answer would not show generality across behaviors, model families or templates, nor generality of the decompiled programs. A negative answer, either a discovery stop or a failed R or I, would be an informative "no" for MD-1 as frozen. The user's broader question still needs a second behavior, as §11 says.

# Experiment 102 design: the reviews

Companion file of `2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`. Every review here is an
independent AI review (a Claude agent, read-only), not a human validation.

| review | of | verdict | source sha256 |
|---|---|---|---|
| 1 (2026-10-15) | revision 1, sha256 `84b24685…796f` (commit `1cf99b4`) | NOT READY: 2 blockers, 11 SHOULDs, 9 notes | `edd86fe9bbb7633766db40af82a1f64f4d98bd0ef747ad848ceb03c335e88dfb` |
| 2 (2026-10-15, re-check) | revision 2, sha256 `efb1b137…c28936` (commit `53b1458`) | READY WITH CHANGES: no blocker, 6 SHOULDs, 12 notes | `e839c14a53b99259bff896530d5bd9a50fa74cd51469c152bfa5586a45f653e2` |
| 3 (2026-10-15, change check) | revision 3, sha256 `7563f985…c0a4` (commit `58ecf45`) | READY WITH CHANGES: no blocker, 2 SHOULDs, 9 notes | `d7690df8a12e6c91399470512bda2f78a3ccf9d65c4ad5083ce9ca8c83eb6856` |

Revision 2 of the design applies every finding of review 1 (its §16); revision 3 every finding of review 2 (its §17);
revision 4 every finding of review 3 (its §18). Revision 4's own edits have not been reviewed.

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

---

## Review 2: re-check of revision 2 (verbatim)

# Experiment 102 design, revision 2: independent re-check

**This is an AI review, not a human validation.** A Claude agent (Opus 5.5) wrote it on 2026-10-15, read-only.
- Nothing in the repository was created, changed, staged or committed.
- No model, runner, test, tokenizer pass or forward pass ran, and no network was used.
- The only computations were arithmetic on numbers already in the record. Estimates are marked *(est.)*.

**The file under review.**
- `docs/superpowers/specs/2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`, revision 2, 557 lines, at HEAD `53b1458` of branch `c004-exp102-design`.
- Its sha256 is **`efb1b137dc12fb7ae621822f2146186ffdece6bf0d44124ba90cf2b4c0c28936`**, as expected.
- Revision 1 is at `1cf99b4`, sha256 `84b24685…796f`, which equals review 1's record.
- "l." below means a line of revision 2.

**What I read.**
- Revision 1 and review 1.
- From 101: the design, the closure, the confirmation review (§3, §5–6, App. A) and the calibration review (§7, §10, §12).
- The critical assessment, both surveys, and the outputs of X101-1 and X101-2.
- The C004 draft and 005's discovery report.
- The scratch pre-check script, which I did not run.

## Verdict: READY WITH CHANGES

Review 1's blockers are fixed:
- PS can now fail: its margins are fixed and its edits include adversarial ones.
- The verdict's specificity gate is a position control that MD-1 does not bound.
- M separates a partial read from a non-specific one.
- Gatekeeping is serial.
- MD-1 is specified and committed by the user.
- Stage 1 is restored.

No blocker remains. The six SHOULDs below are local changes to the text; one of them adds two cheap secondary arms. SHOULD 1 bears on how D3 should be decided.

## 1. Review 1's findings

| finding | status | reason (lines) |
|---|---|---|
| B1 | RESOLVED | ε comes from read-ladder arms only (floor 0.05, cap 0.10); a signed criterion is added; MD and GRAD are added (237–240, 285–294). New issues: SHOULD 2 |
| B2 | PARTLY RESOLVED | The position control runs both ways, in relative form (199, 272–274). The layer control is secondary and not evaluable at the last layer (134, 221). M and `…PARTIAL…` are added (283, 336). But the 0.25 floor re-couples magnitude and specificity (SHOULD 1b), and the gate has a new confound (SHOULD 1) |
| S1 | PARTLY RESOLVED | The arms, the orientation rule and the definitions are in place (55, 167–178, 213–214, 249). The "per-baseline statistics" that §16 cites (l.535) are absent from §8 (SHOULD 5) |
| S2 | PARTLY RESOLVED | The ATTN/DLA/TOP reads, the W^h arms both ways with their swaps, and the attribution-patching ranking are added (136–142, 217–218). No selectivity or naive-read statistic exists, although §16 cites §8.7 for one (l.536) |
| S3 | RESOLVED | A_neg is a stage-2 gate (200, 274) |
| S4 | RESOLVED | Non-inferiority at 0.10 and `cos(V, V^(−))` (276–279) |
| S5 | RESOLVED | Serial stages (266–281). Familywise error ≤ 0.025 per model holds: a stage is reached only if every hypothesis before it is rejected |
| S6 | RESOLVED | PT is committed by the user (241–246, 296–302), and ε_noun is fixed. New issues: SHOULD 3 |
| S7 | RESOLVED | The revision cap, the user's commit of MD-1, and PC (110–113, 147–157, 304–308). A timing gap remains: SHOULD 6 |
| S8 | RESOLVED | Everything listed is defined (115–145). The map's computability: NOTE 3 |
| S9 | RESOLVED | Only primary-arm gates void the label; 160M tolerances are set at the contract phase; smoke runs use their own texts (98, 377–379, 387–390) |
| S10 | RESOLVED | The renamed verdict with its scope string; the recipe, not the programs; a stop is a negative answer; the collinearity split is in the labels (35–41, 326–338, 420–431). A negative overclaim remains (SHOULD 4) |
| S11 | RESOLVED | The NAT and DORM arms, KL and the non-noun mass (176–178, 184–186, 215) |
| N1–N9 | RESOLVED | The companion file is at `1cf99b4`; 131; 102–104; 499–500 and 523–527 (see NOTE 8); 384–388 (see NOTE 7); 355–357 (see NOTE 1); 438–439 and 489; 358–360; 359 and 380 |

## 2. Remaining findings

### SHOULD 1. The position gate can fail through routes that the read-of-a-head reading itself predicts

**Where:** H2 (l.53), §8.2 (l.272–274), §11 (l.427), D3.

H2 assumes that a read of `p_c` does nothing when written at `p_t`. Under the mechanism, `R` written at `RESID_PRE.L*`, `p_t` acts through three routes that MD-1 never measures:

1. **The write-axis overlap.**
   - `R` persists at `p_t` with a component `s·cos(ŵ, d̂_T)` along the head's write axis.
   - As a share of the head's natural write, that is `g_w = cos(ŵ, d̂_T)·σ_c/(A_c‖w_c‖)`. In 70M: 0.20·0.57/(0.747·0.766) ≈ **0.20** *(est.)*.
   - The installation side is steep: the half dose installs 0.72. `F(SUF-PT)` could therefore reach about 0.2–0.3.
   - 70M probably passes, but with `Z′` near 0.4 rather than 1.
   - A copy-like 160M head (OV ≈ identity on the number subspace) has `g_w` near 1, and H2 fails by geometry alone.
2. **Self-read.** The head reads `w_c` wherever it attends, with gain `(A_tt/σ_t)/(A_tc/σ_c)`. The filter admits heads with `A_tc` = 0.1.
3. **The query.** `R` changes `T`'s query at `p_t`, and with it `T`'s attention to `p_c`.

§11 would read the resulting `MD1_READ_NOT_SPECIFIC_160M` as not supporting "the read of a transport head". That reading is wrong in exactly these cases.

**Fix.**
- **Before D3,** tell the user that the gate's prediction under the mechanism is not ≈ 0.
- **At discovery,** record for each `h ∈ T`, from D and the weights before any C prompt: `cos(ŵ^h, d̂_T^h)`, `g_w`, `A_tt` and `g_s`.
- **Add two secondary arms:**
  - **ABH-PT:** `R` at `p_t` with `T`'s outputs at `p_t` clamped;
  - **WPT:** `±⟨R, d̂_T⟩d̂_T` at `RESID_PRE.(L*+1)`, `p_t`.
- **Record** `T`'s attention row and `⟨ΔT, d̂_T⟩` in the PT arms.
- **§11's row becomes:** "the read edit also acts at `p_t`: through `T`'s own write axis or self-attention where WPT and ABH-PT account for it (a shared number direction), otherwise as a generic direction."
- **Optionally preregister:** if the median of `g_w + g_s` is at least 0.5 at discovery, POSITION is `NOT_EVALUABLE_AS_SITE_TEST`.

**1b. The floor (l.273).** With M at its boundary (0.5/0.5), a median `Z′` of at least 0.25 caps the `p_t` effect at 0.125, a quarter of the `p_c` effect rather than half. For a complete read the cap is 0.375. That is B2.3's coupling, weakened. **Fix:** a median `Z′` of at least 0.10.

### SHOULD 2. PS: ε is set by the ladder's curvature, two edge rules are missing, and a failure is over-read

**Where:** l.237–240, l.285–294, l.322 and l.426.

- **The scale.**
  - The leave-one-rung-out error at 0.5 spans a gap of 0.5. For smooth curves it is about 4× the error of the 0.25-spaced interpolation that PS actually uses.
  - 101's half doses are 0.734 and 0.727, so a quadratic through ρ = 0, ½ and 1 gives a leave-one-rung-out error of about 0.06, against an actual interpolation error of about 0.016 *(est.)*.
  - `e₉₀` is therefore likely 0.07 or more and may exceed the 0.10 cap. For 70M, curvature, not sufficiency, sets ε, and PS could come out uninformative.
- **The known remainder.**
  - Under additivity, DX's off-read slice (about 0.93·s, along `û_off`) and 101's per-frame SUF-OFF fractions (median −0.033, down to −0.090 on C) predict 26, 29 and 31 of 36 frames agreeing at ε = 0.05, 0.06 and 0.07, with a signed installation median of about −0.03 *(est.)*.
  - PS on 70M therefore turns on a recorded near-inert remainder, not on a hidden channel.
- **Fixed directions.** For MD and GRAD, `ρ = cos(u, ŵ)` in every frame. GRAD is oriented by `δ_D`, not by `ŵ`, so a slightly negative cosine would fail PS in 36 of 36 frames by rule (l.289).
- **The signed median.** "Worst value" (l.257) is undefined for a two-sided band.

**Fix.**
- **ε.** Set ε = max(0.05, e₉₀/4), or add rungs at 0.125 and 0.375. PS is uninformative only if `e₉₀/4` exceeds 0.10 on either side.
- **The remainder: decide in D7.** Either predict `F̂ = ladder(ρ) +` the frame's OFF effect scaled by the off-read length (for DX and MD), or keep the 0.05 floor and state in §8.8 that 70M is predicted near 26 of 36.
- **Fixed directions.** Extend the ladders to ρ = −1, with NEC-AMP and SUF-NEG as the rungs.
- **The signed median.** Enter non-evaluable frames once as +∞ and once as −∞, and require both to pass.
- **The label.** Rename `…REJECTED` to `PROJECTION_SUFFICIENCY_NOT_SUPPORTED`.
- **§11's row becomes:** "r is not sufficient within ε (the side, sign and size are reported); a subspace claim needs a median |deviation| above 0.10."

### SHOULD 3. PT is mostly a frame-independent coverage test, and its tolerance absorbs bias

**Where:** H7 (l.58), l.39, l.241–244, §8.4 and l.424.

- **One number for most arms.** READ has ρ ≡ 1, and MD and GRAD have ρ ≡ `cos(u, ŵ)`. Their "per-frame" prediction is one number for every frame, fixed at the lock. Only DX uses the frame's unedited states.
- **Absorbed bias.** `τ_arm` is the 90th percentile of D's own errors for that arm, so a stable bias is absorbed: B1's defect again. An arm whose bias exceeds 0.20 becomes "uninformative" instead of failing.
- **The consequence.** PT can fail only through a shift from D to C. "Predicted … from its unedited states" and "predictive of fresh frames in advance" therefore overclaim.

**Fix.**
- **The tolerance.** Set `τ_arm` to the 90th percentile over D of `|ladder_f(ρ) − pooled₋f(ρ)|`, the read's own spread between frames at the arm's ρ, so that a biased arm fails.
- **The wording.** Reword H7 and §11 as "D-calibrated per-frame predictions, frame-specific only through ρ (DX)".

### SHOULD 4. `MD1_READ_NOT_CAUSAL_160M` contradicts l.340

**Where:** l.335 and l.429.

- **The contradiction.** Line 340 says "No label says that a variable has no causal role", yet this label says exactly that.
- **The scope.** It also covers outcomes where only removal or only installation passes.

**Fix.**
- **The label.** Rename it to `MD1_READ_EFFECTS_NOT_ESTABLISHED_160M`, adding `_REMOVAL_ONLY` or `_INSTALLATION_ONLY` when one direction passes.
- **The §11 text.** Use "did not pass R and I on fresh 160M frames".

### SHOULD 5. The baseline and naive-selection statistics are not preregistered

**Where:** §8; §16 (l.535–536).

- **What is missing.** §8 defines B for DX only. These arms have no statistic:
  - the interchange arms;
  - DX-FP, PC1 and LDA, and MD's B;
  - the W^h arms;
  - the ATTN, DLA and TOP arms.
- **Why it matters.** These are the user's "simple baselines". Without a preregistered statistic they become forking paths.

**Fix:** add a §8.9, with no force:
- **Per baseline.** For each u in {MD, PC1, LDA, GRAD, DX-FP}: the count and median of `B_f^u`.
- **Interchange arms.** Their R/I-style counts and medians, and their median edit length relative to `s`.
- **Selectivity.** The number of frames, against 26 of 36, in which READ beats the median W^h at equal length in both directions.
- **Each naive head.**
  - The medians of `F_rem` and `F_ins`.
  - The count of frames in which `T`'s read beats it in both directions.
  - Its edit length.
  - If its layer is below `L*`, the realized ρ at `L*`.

### SHOULD 6. The positive control arrives after the one 160M discovery

**Where:** l.149–157, §8.5.

- **The gap.**
  - A defect in step (e) would surface only at stage 2.
  - By then MD-1 is frozen and the 160M read is committed, and no revision is allowed (l.157).
  - The retrodiction checks the layer and the head, not the read.

**Fix.**
- **A read check before 160M.** Add to §4.4's pass rule either an exposed R/I check of MD-1's 70M read on D (BREAD and BOFF, 144 runs on spent frames) or `cos(w_c^{MD-1}, w_c^{011}) ≥ 0.9`.
- **Defects versus revisions.** An implementation defect is fixed, re-reviewed and recorded without counting against the revision cap; only a change of rule counts.

## NOTES

1. **Power.** Holm passes a two-hypothesis stage at 80 % with probability 0.915: both counts at least 25 and one at least 26. Both stages together give about **0.84**, not 0.76 (l.356).
2. **The contract phase runs 160M** (l.399), yet D13 counts five model-running phases.
   - Define "neutral" as no cue word and no template frame.
   - Run the smoke-only texts on 160M only after the MD-1 commit (l.37–38).
   - "No phase is repeated" (l.417) needs an exception for a version-2 retrodiction.
3. **The robustness map** (l.144–145) needs joint patches that MD-1 never ran. Run every prefix of at most 3 heads for attention thresholds {0, 0.1, 0.2}, or mark those cells "not computable".
4. **Definitions to settle:**
   - "Centred" for PC1 and `P_nat` (l.171, 176) should mean `C` applied to each vector, not centring across samples. X101-1's PC1 share of 0.84 implies the former.
   - Orient `u_dorm` by `ŵ`, since `⟨u_dorm, δ_D⟩ ≈ 0`.
   - The W^h orientation (l.217).
   - DLA's frames and statistic.
   - `γ_{L*}` is the attention LayerNorm's gain.
   - DX-FP's root: the smallest positive one.
   - For k > 1: which principal angle is used, and the rule for B's collinearity.
5. **Labels.**
   - MAGNITUDE, PT and PER_ITEM lack a `NOT_REACHED` or `NOT_EVALUABLE` value.
   - No 160M per-model label is defined after a discovery stop, and none for a PC that is not evaluable.
   - A PC failure should not qualify a stop, since no read is involved.
   - H's "near-redundant" should be a label suffix.
   - `READ_CARRIES_MAJORITY` is wrong at exactly 0.5.
6. **Voiding** (l.389). Gate failures on the MD, GRAD, ladder, H and BREAD/BOFF arms void the verdict, although these arms only qualify it. Such a failure should void the qualifier instead.
7. **G7** (l.385–386). The wording "appears in any earlier record that stores sequences" would also catch 37–60 in 101's freeze file. Say "executed in any earlier ledger".
8. **Frames.**
   - The replacements 65 `sandy`, 69 `icy`, 70 `rocky`, 71 `bald` and 72 `slender` are sound adjectives with verbs that take plural objects.
   - 71 "greeted one [another]" and 65 "shook one…" allow a reciprocal reading. 101's frames 21 (`hugged`) and 53 (`met`) behaved normally with the same kind of verb.
   - N4's noun standard was not applied to 37–60: 58 `fast` and 39 `lush` can be nouns, and 50 `early`, 41 `quick`, 43 `slow` and 48 `real` can be adverbs ("picked one early"). The human check should cover all 36 frames.
   - The reserve's flag list misses `strong`, `weak`, `healthy` and `delicate`.
   - If four or more of 61–72 fail, the freeze order would pull the flagged 81 into C.
9. **Smaller gaps.**
   - "The AUROC of PT for the frames whose A_amp is not negative" (l.350) is undefined: PT predicts no `A_amp`.
   - The stratum statistics are unspecified.
   - 70M's H compares recipes as well as cues. Add HREAD against BREAD.
10. **160M's calibration on D is in-sample to `T`'s selection on D1.** Report PT's curves and the R/I effects for D1 and D2 separately.
11. **Gating order.**
    - If the position gate fails, B and H are never reached, so H, which tests C004's top confounder, would go untested.
    - For 70M at least, a Holm over {position, A_neg, B, H} after stage 1 (26/26/26/25) reaches H and keeps the familywise error.
    - Say whether "rejected" includes the median floors.
    - Say whether stage 2 runs after a failed precondition on C.
12. **A_neg, revisions and footprints.**
    - SUF-NEG's target lies below every natural `r(one)` (X101-1), so read a DIRECTION failure as "not signed beyond the natural range".
    - Calibration revisions after the 160M discovery may not touch MD-1 or `T`.
    - Report `‖γ⊙R‖/σ` at `p_t` against the value at `p_c`.

## Checked and correct

- **Holm thresholds.** The tails are 0.00567 and 0.01441; 24 of 36 would give 0.0326.
- **Power and familywise error.** The per-count power figures are right, and serial gatekeeping keeps the familywise error at 0.025 or less.
- **Tolerances.** The 160M tolerance rule is sound.
- **Record values.** 005's A3, A4 and A6, Δ₃ = 0.831, and X101's 0.948/0.952, 0.2013 and 0.3445 all match the record.
- **Compute.** This is plausible:
  - about 123 runs per frame for 160M and 115 for 70M;
  - about 12.3k MD-1 passes on 160M (1,728 for the profile plus 10,368 for the head patches);
  - 3–5 h in all.
- **Structure.**
  - G0's identity holds by construction.
  - The verdict table is ordered, and its rows are mutually exclusive.
  - The rule for D's split is deterministic.

---

## Review 3: change check of revision 3 (verbatim)

# Experiment 102 design, revision 3: change check

**This is an independent AI review (Claude, Opus 5.5), not a human validation.** It was written on 2026-10-15, read-only.
- Nothing in the repository changed. No model, runner, test, tokenizer or forward pass ran, and no network was used.
- The only computations were binomial tails and arithmetic on 101's recorded per-frame values.

**The file:** `docs/superpowers/specs/2026-10-15-experiment-102-blind-read-discovery-transfer-design.md`, revision 3, 659 lines, at HEAD `58ecf45`.
- Its sha256 is `7563f9856ad7b77eaf8e2bc57cd7207755064e3e390a8b883a3ca0055700c0a4`, as expected.
- It was diffed against `53b1458` and read against review 2.
- "l." below means a line of revision 3.

## Verdict: READY WITH CHANGES

There is no blocker, and every review-2 finding is applied. Two new SHOULDs remain, each a short text change, plus nine notes.

## 1. Review 2's findings

| finding | status | lines |
|---|---|---|
| SHOULD 1 | RESOLVED (new issues: S-A, N3, N4) | 58, 137, 178, 221, 237–239, 307–310, 321, 394, 499–501, 521, 556 |
| SHOULD 1b | RESOLVED | 308 |
| SHOULD 2 | RESOLVED (MD kept on the ladder by decision: N5; new issue: N6) | 267–271, 331–343, 378, 498, 562 |
| SHOULD 3 | RESOLVED (new issue: S-B) | 62, 273–278, 349–355, 496 |
| SHOULD 4 | RESOLVED; the old name appears only in §17 | 391, 503 |
| SHOULD 5 | RESOLVED | 423–433 |
| SHOULD 6 | RESOLVED (new issue: N2) | 157–163, 168 |
| NOTE 1 | RESOLVED; recomputed: 0.9148, 0.837, 0.830 | 416 |
| NOTE 2 | RESOLVED | 102–103, 167, 465, 472, 486, 571 |
| NOTE 3 | RESOLVED | 150–153 |
| NOTE 4 | RESOLVED (`Δx̂_c` is undefined: N8) | 125, 145, 174, 187, 195–199, 243 |
| NOTE 5 | RESOLVED | 359–363, 367–380 |
| NOTE 6 | RESOLVED (Holm with a voided family: N1) | 211–253, 460 |
| NOTE 7 | RESOLVED | 457 |
| NOTE 8 | RESOLVED as specified (consistency gaps: N7) | 581–592, 600, 606 |
| NOTE 9 | RESOLVED | 407–410 |
| NOTE 10 | RESOLVED | 283, 536 |
| NOTE 11 | RESOLVED | 293, 299, 318–327 |
| NOTE 12 | RESOLVED | 59, 205, 265, 375, 500 |

## 2. What checks out in the changed text

- **`R_⊥`.**
  - It is well defined: `R` and every `d̂_T^h` live in the residual space.
  - It is used consistently in H2, (e‴), §5, §6, §8.2, §8.6, §11–§13, D3, G1 and G5.
  - The clamp removes the self-read and query routes, and the projection removes the write-axis overlap. The PT, ABH-PT and WPT arms measure these routes separately.
  - For k = 1, `‖R_⊥‖/‖R‖ = √(1 − cos²(ŵ, d̂_T))` is the same in every frame: 0.980 in 70M.
  - The 0.5 cut (cos > 0.87) is sensible, but see SHOULD A.
- **`…_POSITION_UNTESTED`** fits "counts as passed for gatekeeping only":
  - it claims no rejection;
  - direction needs 25/36 whether position is dropped from Holm or counted as passed;
  - stage 3 is then reached.
- **`λ_f`** is right on both sides.
  - For k = 1, `P_V(s·DX) = ρR`, so DX's off-read part is `λ_f·s·û_off` with `λ_f = ‖off_c‖/‖CΔx‖ = √(1−ρ²)`. That is the off-read length divided by the OFF arm's length, along the same `û_off`.
  - The removal term is relative to F = 1 and the installation term to F = 0. The signs match NEC-OFF and SUF-OFF.
  - In 70M, λ_f ≈ 0.93.
- **The rest checks out:**
  - the ε /4 rule (interpolation error scales with the square of the rung gap);
  - the ±∞ rule;
  - completeness and mutual exclusivity of the labels and verdict table (except N1);
  - the read check (30/36, medians ≥ 0.5; 101's R and I medians were 1.006 and 1.035);
  - the Holm thresholds 26/25 and 26/26/26/25;
  - the power figures 0.915, 0.84 and 0.83.
- **No leftovers:** no removed name, floor 0.25 or old ε rule appears outside the §16/§17 logs.
- **Frames:** 65 and 71 match. The replacement rule cannot pull a listed flagged frame into C.

## 3. Remaining findings

**SHOULD A. The position gate compares unequal lengths** (l.58, 221, 307–310, 556).
- **The problem.**
  - The gate compares `R_⊥` at `p_t`, of length r·s where r = `‖R_⊥‖/‖R‖`, with the full `R` at `p_c`.
  - To first order, a direction acting at `p_t` with per-length efficacy e, relative to the read at `p_c`, therefore passes whenever e < 1/(2r).
  - That bound is 0.51 at 70M's r = 0.98, but 0.58 at r = 0.87, 0.70 at r = 0.71, and 1.0 at the cut.
  - Every other control in the design is compared at equal length.
- **Fix.** Use `s·R_⊥/‖R_⊥‖` in NEC-/SUF-PTX and keep the unscaled vector as a secondary arm. Alternatively, put `r·(1 − F_rem)` and `r·F_ins` in Z′. The 0.5 cut then also bounds `cos(R̂_⊥, ŵ) = r`.

**SHOULD B. PT's DX prediction lacks the off-read term that PS now adds** (l.349–352 vs 334–338).
- **The problem.**
  - PT predicts DX from the read ladder alone, with τ taken from the read's own spread.
  - On 101's spent frames, `λ_f·F_ins_off` has median −0.031, reaches −0.083, and lies beyond 0.05 in 10 of 36 frames *(est.)*.
  - That is the order of the read's spread: 101's leave-one-frame-out spread of `F_ins` is 0.08.
  - PT on SUF-DX in 70M can therefore fail on the recorded remainder, the reading PS's term was added to avoid. Calibration cannot revise it.
- **Fix.** Add `λ_f·median_D(F(NEC-OFF) − 1)` and `λ_f·median_D F(SUF-OFF)`, both known at stage 1, and state the 70M expectation in §8.8.

**NOTES**
- **N1** (l.318–325).
  - Say how a family that is not evaluable (position, or B or H after a gate failure on a qualifier arm) enters its Holm stage. For example, if it is dropped, the thresholds are 26/26/25 for three families, 26/25 for two and 25 for one.
  - Say that 26/26/26/25 are step-down thresholds by rank. l.323 reads as if each belonged to one family.
- **N2** (l.165–170, 465).
  - A re-run after a defect fix is not "the one permitted repeat". Permit it explicitly for the 70M retrodiction.
  - A defect found after the 160M discovery should give `NOT_INTERPRETABLE`, not a second discovery.
- **N3** (l.238, 433). ABH-PT is one arm, but its Z′ needs both directions. Make it NEC-/SUF-ABH-PT.
- **N4** (l.178, 441).
  - If `d̂_T^h` is not mean-free, the centred part of `R_⊥` keeps `⟨R, d̂⟩(1 − ‖Cd̂‖²)` along `Cd̂`. Project onto `C d̂_T^h`.
  - For 70M, name which `d̂_T` is used (011's recorded one).
- **N5** (l.334, 418–421, 562). In 70M, MD has cos 0.41 with ŵ (X101-1), so 0.91·s of it is off-read, yet it is scored on the ladder alone. State its 70M PS expectation, or add an MD-OFF arm and term.
- **N6** (l.267–271, 331–333). The [−1, 0] segment has a gap of 1, so ε's /4 does not cover ρ < 0: the error can reach 4·e₉₀ at ρ = −0.5. Add rungs at ρ = −0.5 (`x_pl + ½R`, `x_ref − ½R`) and use `max(0.05, e₉₀)` there, or state the limit.
- **N7** (§15).
  - The reciprocal criterion that changed 65 and 71 also reads on 44 `told` and 53 `met` (in C) and on 73 `groomed` and 80 `hugged` (the 1st and 8th replacements). None is flagged.
  - Review 2's claim that 53 behaved normally is wrong: 101 executed only frames 1–36 (`picks`). Only frame 21 `hugged` is evidence. Either drop the criterion, citing 21, or flag these frames.
  - Arguably also flag 37 `sick`, 49 `lost` and 55 `pretty` (adverb) in C, and 74 `skinny`, 84 `mild` and 92 `floral` in the pool.
  - Extend "skipping the flagged ones" to the tokenizer selection (l.576–579).
- **N8** (definitions).
  - Define `Δx̂_c` (l.198, 335).
  - "Cannot be evaluated" (l.341) means `Δc_full > −1`.
  - PS "uninformative" maps to `PROJECTION_SUFFICIENCY_NOT_EVALUABLE`.
  - Say when DX is uninformative for PT, since `τ(ρ_f)` varies by frame (l.276).
  - Give PC's value when 70M's precondition or a verdict gate fails.
- **N9** (l.393, 499). `MD1_READ_NOT_SPECIFIC_160M` and "behaves as a generic number direction there" assert non-specificity from a failed test; 24/36 with a median Z′ of 0.3 would fail. Use `…_SPECIFICITY_NOT_ESTABLISHED_160M`.

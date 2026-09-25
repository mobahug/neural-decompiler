# Experiment 024: Does an Operational Nounness Score Predict When the Frozen Downstream Routing Stops Holding?

**Status (2026-09-25): implemented, frozen and calibrated, each independently reviewed and committed; lock next.**
- **Implementation:** complete and independently reviewed (PASS WITH NOTES, no blockers; every finding addressed).
  Commits `6f80665`…`44c3c2b`.
- **Freeze:** the production `freeze` ran exactly once at `44c3c2b`, tokenizer and text only. It was independently
  reviewed (PASS WITH NOTES) and `confirmation-v1.json` was installed byte for byte in `566eb6f`:
  - file sha256 `68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60`;
  - content sha256 `87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87`.

  See [Freeze](#freeze-2026-09-25).
- **Calibration:** the production `calibrate` ran exactly once at `bf0049c`, from spent exposed data and the weights
  only. It was independently reviewed (PASS WITH NOTES) and `calibration-v1.json` was installed byte for byte in
  `1084efc`:
  - file sha256 `81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d`;
  - content sha256 `09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8`.

  **The primary requirement is ρ ≥ `0.3136960600375234`.** See [Calibration](#calibration-2026-09-25).
- **Not run:** `lock`, `confirm` and `report`. **No fresh 024 prompt has run**, and no readout error of any 024 cue
  has been observed.

Implements the design
[`docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md`](../../docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md)
(revision 2, `9d03dee`) through the plan
[`docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md`](../../docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md)
(revision 1, `608088c`) and the reviewer's six implementation requirements.

The question: does a frozen, weight-derived operational nounness score prospectively predict, on 40 never-executed cue
words, how much the frozen block-4/5 attention readout errs in the 108 exposed frames?

- **The score:** `nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue)`.
  - `E_w` is the input embedding row, in float64.
  - `μ_noun` is the mean of the 158 singular and plural rows of the 79 scorable exposed nouns.
  - `μ_cue` is the mean of the 139 exposed calibration cues' rows, leave-one-out for a calibration cue.
- **The response:** each cue's `MSE = Σ SSE_C / Σ n` over its 108 exposed pairs.
  - `SSE_C = Σ_nouns (Δc − C)²`.
  - `C` is Experiment 020's frozen Level-0 readout fed the measured `Δx3`.
- **The primary test:** the Spearman `ρ` across the 40 cues. A PASS needs `ρ ≥ max(F_ρ, null₉₇.₅)`.
  - `F_ρ` is element `[249]` of 10,000 SHA-indexed draws of 40 calibration cues.
  - `null₉₇.₅` is element `[97499]` of 100,000 SHA-indexed permutations.
  - The four-way results, in precedence order: `NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS`.
- **The E–N disambiguation guard:** an exact one-sided permutation test of
  `D_EN = mean(log MSE_E) − mean(log MSE_N)`, comparing the 8 ordinary singular nouns with the 8 non-noun controls.
  - All `C(16, 8) = 12,870` assignments are enumerated, in exact integer arithmetic on the observed binary64 values.
  - It passes iff `K ≤ 321`, where `K` counts the assignments at or above the observed contrast, ties included.
  - An all-tied population fails.
- **The outcome**, in the frozen hierarchy:
  - `NOUNNESS_PREDICTION_NOT_ESTABLISHED`;
  - `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED`;
  - `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.

  Every outcome is predictive and associational. The secondary B/C/D/E contrasts and the block-4/5 ladder run after
  the result is written, and can never alter it.

## Code

- **Module:** `src/neural_decompiler/readout_routing.py` (`rr`).
  - It calls 022's `upstream_localization.py` and 023's `block0_completion.py`, pinned by git blob with 022's ten.
  - It binds 023's reviewed exposed cells, confirmation and lock by file and content digest.
  - The production configuration is literal and immutable: 8 per class, a guard of 8 + 8 with 12,870 assignments and
    a largest passing `K` of 321, `B` = 10,000 and `P` = 100,000. A test world can only pass its own configuration
    explicitly.
- **Runner:** `experiments/024-readout-routing-nounness/run.py`.
  - Phases: `validate`, `freeze`, `calibrate`, `lock`, `confirm` and `report`.
  - It takes no option and always runs the production configuration.

## Phases (each only when separately authorized)

1. **`freeze`** (tokenizer only) writes `confirmation-v1.json`, which is committed by hand and reviewed.
   - It picks the first 8 eligible entries of each ordered list.
   - A shortfall, or picks other than the design's expected 40, writes nothing and creates no state.
2. **`calibrate`** (once; 023's committed exposed cells and the weights; no forward pass) writes the candidate
   `calibration-v1.json`. It is installed byte-identically, committed and reviewed.
3. **`lock`** (weights only) writes the candidate `preregistration-lock.json` and `preregistration.md`. They are
   installed byte-identically, committed and reviewed.
4. **`confirm`** (once; never resumed):
   - `validate_lock`, then the model digests;
   - I7 before any prompt: the lock's scores, centroids and predictions, bit for bit;
   - the 4,320 prompts, once each, with the measurements saved first;
   - the accounting, and `C` recomputed from the saved `Δx3` bit for bit;
   - I1, I3 and I4;
   - the result and the completed phase in one atomic write;
   - then the descriptive records.
5. **`report`**.

The local outputs stay in `outputs/experiment-024/`.

## Freeze (2026-09-25)

**The run.** The production `freeze` ran once at `44c3c2b`, from 09:34:55 to 09:35:05 UTC, with exit status 0, under a
guarded launcher: 0 model loads, 0 module calls, 0 captures, and the only repository write was the confirmation file.
- Evidence: [`evidence/freeze-2026-09-25/`](evidence/freeze-2026-09-25/README.md).
- The independent review (PASS WITH NOTES, no blockers):
  [`evidence/freeze-review-2026-09-25/REVIEW.md`](evidence/freeze-review-2026-09-25/REVIEW.md).

**The 40 frozen cues:**
- **N:** honest, polite, rude, sleepy, wise, lucky, merry, nervous.
- **B / D:** gallons / gallon, ounces / ounce, acres / acre, herds / herd, crowds / crowd, bundles / bundle,
  clusters / cluster, litres / litre.
- **C / E:** apples / apple, horses / horse, doctors / doctor, kings / king, rabbits / rabbit, poets / poet,
  dragons / dragon, lions / lion.

These are the design's expected picks, chosen mechanically; no reserve was needed.

**Eligibility.**
- 30 rejections, each with its reason.
- Exclusions: 351 earlier cue ids, 161 target-noun forms and 316 exposed-frame tokens. None overlaps the 40 cues.

**The manifest.** 4,320 keys (40 cues × 108 exposed frames), sha256 `fcc437fc…`, with 0 overlap with the 39,312 spent
keys.

**Review notes, preserved as the protocol decisions of 2026-09-25:**
1. **No scores in the freeze.** The freeze intentionally contains no nounness scores. It is tokenizer and text only:
   the population was fixed before the weights were consulted.
2. **Nounness is bound later, as the approved design says.**
   - Calibration binds the embedding, the reference populations and the centroids.
   - The lock binds the 40 scores, the extrapolation count and the outcome semantics.
   - Confirm re-derives them all before any fresh prompt.
3. **One canonical score implementation.** An independent implementation agreed with the canonical module to about
   2.2e-16 but not byte for byte. The protocol relies on the canonical module for the score digest (`a703ac16…` is the
   expected value).
4. **The preregistered extrapolation statement is unchanged:** 5 of the 8 E cues lie above the calibration maximum
   (+0.135020).
5. **A descriptive pre-outcome note.** Before any outcome, 23 of the 40 cues lie above the calibration maximum (6 B, 4 D,
   8 C, 5 E, 0 N). This has no outcome force and does not alter the design.
6. **Extra reads.** The additional file reads the review identified during the freeze were integrity checks only. None
   fed the selection.

## Calibration (2026-09-25)

**The run.** The production `calibrate` ran once at `bf0049c`, from 13:08:12 to 13:08:45 UTC, with exit status 0,
under a guarded launcher:
- 1 weights load, 0 module calls, 0 captures, no fresh prompt;
- it read only 023's committed exposed cells and the weights.

The candidate record was installed byte for byte as `calibration-v1.json` in `1084efc`.
- Evidence: [`evidence/calibrate-2026-09-25/`](evidence/calibrate-2026-09-25/README.md).
- The independent review (PASS WITH NOTES, no blockers):
  [`evidence/calibration-review-2026-09-25/REVIEW.md`](evidence/calibration-review-2026-09-25/REVIEW.md).

**The calibrated values (full precision, from spent exposed data only):**
- **The line** (exposed-data-fitted, prospectively frozen; secondary): log MSE = `-2.481260357420486` +
  `1.2992017464639374` · nounness, residual sd `0.2570037337349923` (139 cues). The exposed Spearman is
  `0.5296617364493499`.
- **F_ρ** = `0.24411074612857814`: element [249] of 10,000 SHA-indexed draws, with 0 undefined draws.
- **null₉₇.₅** = `0.3136960600375234`: element [97499] of 100,000 SHA-indexed permutations.
- **The primary requirement: ρ ≥ `0.3136960600375234`.** The chance-level null binds, not the exposed-like floor.

**Calibration notes, preserved as recorded in the review:**
1. **F_ρ is lower than the design's rough estimate** (0.2441 against about 0.257) and remains valid. The design's
   approximate diagnostics are left as they were; design revision 2 is not edited to carry the realized values.
2. **15 null values tie at the threshold.** The frozen rule is an empirical 100,000-permutation order statistic
   (ρ ≥ element [97499]). It stays authoritative and unambiguous, and it is not changed because of the ties.
3. **ENVELOPE_ONLY_FAILURE is unreachable for this realized calibration**, because null₉₇.₅ > F_ρ. The generic
   four-way result machinery is unchanged.
4. **Only spent exposed data and the weights.** The nounness construction, the per-cue MSE, the line and both
   thresholds come from 023's committed exposed cells and the pinned weights.
5. **The 40 fresh nounness scores are deferred to the lock**, as the approved design intends. They are not in the
   calibration record.
6. **Last-bit differences only.** The reviewer's independent arithmetic differed from the canonical nounness and MSE
   only at the last-bit scale (at most about 2e-16). That changed no ordering or statistic, and the canonical module
   remains authoritative for every digest.

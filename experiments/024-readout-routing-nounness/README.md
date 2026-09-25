# Experiment 024: Does an Operational Nounness Score Predict When the Frozen Downstream Routing Stops Holding?

**Status (2026-09-25): implemented, frozen, calibrated, locked and confirmed, each phase run once and independently
reviewed; `report` not yet run.**
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
- **Lock:** the production `lock` ran exactly once, successfully, at `be74d23`, from the committed freeze, the
  committed calibration record and the weights only. It was independently reviewed (PASS WITH NOTES, no blockers).
  The exact candidates were installed byte-identically in `3affe55`, a commit of exactly these two paths:
  - `preregistration-lock.json`: file sha256 `5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d`,
    content sha256 `a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739`;
  - `preregistration.md`: sha256 `007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54`.

  See [Lock](#lock-2026-09-25).
- **Confirm:** the production `confirm` ran exactly once, at `af160ce`, with exit status 0 and no incident. It executed
  all 4,320 frozen keys once each. It was independently reviewed (PASS WITH NOTES, no blockers), and the reviewer
  derived the result from the saved measurements with its own code.
  - **Primary:** ρ = `0.6108818011257036` ≥ `0.3136960600375234`, **PASS**.
  - **E–N guard:** K = 1 of 12,870 (PASS iff K ≤ 321), **PASS**.
  - **Frozen outcome: `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.**

  See [Confirm](#confirm-2026-09-25).
- **Not run:** `report`.

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
3. **`lock`** (weights only) writes the candidate lock and preregistration (`candidate-lock.json`,
   `candidate-preregistration.md`). After the independent review they are installed byte-identically as
   `preregistration-lock.json` and `preregistration.md` and committed.
4. **`confirm`** (once; never resumed):
   - before the model loads, the confirm launcher asserts the module blob directly (see the Lock notes);
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

## Lock (2026-09-25)

**The run.** The production `lock` ran once at `be74d23`, from 17:13:39 to 17:14:05 UTC, with exit status 0, under a
guarded launcher:
- 1 weights load, 0 module calls, 0 captures, no prompt;
- it wrote only the two candidates and the state.

The independent review was PASS WITH NOTES, with no blockers. The exact candidates were then installed byte for byte in
`3affe55`, which touches exactly `preregistration-lock.json` and `preregistration.md`.
- Evidence: [`evidence/lock-2026-09-25/`](evidence/lock-2026-09-25/README.md). The launcher's record `lock_run.json`
  has sha256 `b7a51118ddb998fa005d7dc32cba7bce2a18278a2ec79f94ea19db973be5f5e2`.
- The independent review:
  [`evidence/lock-review-2026-09-25/REVIEW.md`](evidence/lock-review-2026-09-25/REVIEW.md).

**What the lock binds (full precision):**
- the analysis module `src/neural_decompiler/readout_routing.py`, git blob
  `e6cb37767d1d06c6ff40804a88eab569723afdb5`, with the 12 pinned module blobs;
- the 40 frozen scores in the freeze's order, digest
  `a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995`, each with its line prediction and above-maximum
  flag;
- the calibration maximum `0.13502027836111233`:
  - exactly 5 of the 8 E cues lie above it (apple, horse, doctor, poet, dragon);
  - every E scores above every N;
- **the primary requirement ρ ≥ `0.3136960600375234`** (F_ρ `0.24411074612857814`, bound by the null);
- the exact E–N guard: 12,870 assignments, PASS iff K ≤ 321, and K ≥ 322 fails;
- the outcome hierarchy and the semantics, the secondary text included;
- the 4,320-key manifest `fcc437fc…`, the committed freeze and calibration record, 023's cells and the dependencies.

**Review notes, preserved as the protocol decisions of 2026-09-25:**
1. **The direct module-blob check.** The installed lock stores the canonical analysis-module blob `e6cb3776…`.
   - `validate_lock` freezes that implementation only indirectly, through scientific-path immutability: a clean tree
     and no scientific path changed since `be74d23`. It does not compare `lock.module.blob` with `rr.own_blob()`.
   - The code is not changed.
   - The confirm launcher must independently assert `rr.own_blob() == installed_lock["module"]["blob"]` before the model
     loads, and record the expected and observed blob in the confirm evidence. A mismatch stops before the model load,
     the ledger or any prompt.
2. **Weight-digest mismatches refuse without an incident.** A parameter or embedding digest mismatch refuses
   confirmation before the ledger and any prompt, and need not record an incident, because nothing has been spent.
   Only an I7 mismatch records a phase incident.
3. **The envelope-only band is empty.** F_ρ < null₉₇.₅, so ENVELOPE_ONLY_FAILURE cannot occur for this calibration; any
   ρ below `0.3136960600375234` is a GUARD_FAILURE.
4. **A non-finite fresh MSE is conservatively an incident**, not NOT_INTERPRETABLE.
5. **The canonical module stays authoritative for the score digest.** The independent high-precision recomputation of
   the 40 scores agreed numerically (at most 2.2e-16, identical ordering) but not byte-digest-wise: correctly rounded
   values would give `e494736a…`.
6. **The launcher's record is a self-report.** Its sha256 is recorded here, and the review corroborated zero prompts
   from the state itself: undoing only the lock's entries reproduces the pre-lock state byte for byte.

## Confirm (2026-09-25)

**The run.** The production `confirm` was invoked exactly once, at `af160ce`, through an external guarded launcher.
- Before the model loaded, the launcher asserted the module blob directly: `rr.own_blob()` equals the lock's
  `module.blob`, `e6cb37767d1d06c6ff40804a88eab569723afdb5`.
- Confirm ran from 19:31:56 to 19:38:45 UTC and exited with status 0. No incident occurred and nothing was retried.
- 1 model load; I7 reproduced bit for bit before the first prompt.
- All **4,320 frozen keys (40 cues × 108 frames) were executed exactly once**, with 0 collisions with the 39,312 spent
  keys.
- Every validity check passed:
  - C recomputed from the saved Δx3 bit for bit on all 4,320 pairs;
  - I1 1.53e-5, I3 2.12e-5 and I4 6.42e-5, against 1e-4, 1e-4 and 1e-3.
- The measurements `outputs/experiment-024/stage2-measurements.pt` (76,314,087 bytes, sha256
  `b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93`) and the results state (`076ab9f9…`, file
  `e636210c…`) are preserved unchanged and gitignored. Their hashes are recorded in the evidence.
- Evidence: [`evidence/confirm-2026-09-25/`](evidence/confirm-2026-09-25/README.md).
- The independent reviews:
  - post-install: [`evidence/postinstall-review-2026-09-25/REVIEW.md`](evidence/postinstall-review-2026-09-25/REVIEW.md);
  - confirmation:
    [`evidence/confirmation-review-2026-09-25/REVIEW.md`](evidence/confirmation-review-2026-09-25/REVIEW.md).

  Both are PASS WITH NOTES, with no blockers.

**The preregistered result:**

| test | observed | locked requirement | result |
|---|---|---|---|
| primary: Spearman ρ between the frozen nounness score and the fresh per-cue MSE (40 cues) | `0.6108818011257036` (= 1628/2665) | ρ ≥ `0.3136960600375234` | **PASS** |
| E–N disambiguation: `D_EN = mean(log MSE_E) − mean(log MSE_N)`, exact one-sided permutation test | `D_EN = 0.5680373703974243`; K = 1 of 12,870 | K ≤ 321 | **PASS** |
| **frozen outcome** | **`NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`** | | |

**What the PASS means.** The prospective association exceeded the locked primary threshold.
- The frozen rule was ρ ≥ max(F_ρ, null₉₇.₅), with F_ρ = `0.24411074612857814` and null₉₇.₅ = `0.3136960600375234`.
  The null bound.
- The stored reading "at least as strongly as the exposed-like relationship" is not the formal interpretation. F_ρ is
  the 2.5th percentile of the exposed-like draws, so the rule says the association was not unusually weak, not that it
  matched the exposed relationship.
- The preregistration is not edited.

The guard: every one of the eight ordinary singular nouns (E) had a larger log MSE than every one of the eight non-noun
controls (N). The observed split is the unique most extreme of the 12,870 assignments.

**The claim, in its scope.** On 40 fresh cues, the frozen operational weight-derived nounness score prospectively
predicted larger error in the frozen block-4/5 attention/readout approximation. The association passed the
preregistered E-vs-N disambiguation guard against simple plurality-only and measure-only explanations.
- **The guard's formal scope is limited to those two preregistered simple alternatives.** E and N differ in other
  lexical and semantic properties besides nounness: all eight N cues are adjectives, and the E cues are concrete,
  mostly animate nouns.
- **What was observed is the error of a previously frozen approximation of the downstream readout**, not rerouting
  measured directly.
- **The scope:** 108 exposed frames, 79 nouns, one checkpoint (Pythia-70M-deduped), 8 cues per class. 23 of the 40
  cues lie beyond the calibration's nounness range, so the test is partly an extrapolation.
- **Not claimed:**
  - that nounness causes rerouting, or that rerouting was measured directly;
  - that the model detects nouns;
  - that nounness is a general linguistic representation;
  - that E-vs-N rules out all lexical or semantic confounds;
  - that 024 explains all routing-regime changes, or establishes general algorithm choice.

**Descriptive results (no outcome force; they cannot alter the result):**
- **MSE by class** (mean): N 0.0714, B 0.1189, D 0.1127, C 0.1340, E 0.1250.
- **Class contrasts on log MSE** (bootstrap intervals, elements [249] and [9750] of 10,000):

  | contrast | value | interval |
  |---|---|---|
  | noun effect | +0.538 | [+0.395, +0.677] |
  | measure effect among nouns | −0.126 | [−0.252, +0.004], spans zero |
  | plurality effect among nouns | +0.076 | [−0.058, +0.203], spans zero |
  | measure × plurality interaction | +0.020 | [−0.244, +0.271] |
- **Subsets:** ρ is 0.586 on cue-final frames and 0.698 on coordinated frames; on normalized MSE it is 0.592.
- **R²_C by class:** 0.93–0.96.
- **The frozen exposed line:** it slightly under-predicts the fresh error, by a mean residual of +0.093 in log MSE (+0.034
  over the 23 extrapolated cues, +0.173 over the 17 within range). Its largest residuals are gallon (+0.72) and acre
  (+0.63), both singular measure nouns.
- **Block-4 attention's share of the readout error:** 0.58–0.73 by class; block 5 carries the rest.
- **The extrapolation flags:** 23 of 40 fresh cues lie beyond the calibration's nounness maximum (0 N, 6 B, 4 D, 8 C,
  5 E).

**Notes:**
1. **Last-bit differences.** The reviewer's own per-cue MSE differs from the canonical value by at most 2.8e-17 in 3 of
   40 cues, from the canonical summation order, which stays authoritative. Ranks, ρ and K are identical.
2. **The noun ledger holds all 80 pool nouns**, as in Experiment 020, and no fresh noun.
3. **Nothing is committed that the repository does not track.** It keeps no binary measurement artifact and has no
   convention for backing up large artifacts externally, so the measurement file stays local, with its hash recorded.

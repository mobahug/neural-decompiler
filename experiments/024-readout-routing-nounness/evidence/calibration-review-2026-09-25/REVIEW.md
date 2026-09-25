# Independent calibration review — 2026-09-25

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the calibration read-only, at `bf0049c`.
- **Method:** it wrote its own implementations, including exact-rational and exactly rounded routes, and called the
  canonical module only where a canonical byte digest was the thing being checked. It reported numerical
  disagreement separately from byte identity.
- **Result:** every frozen number, digest, ordering and threshold reproduced. It found that the existing candidate could
  be installed byte for byte without regeneration; that was done in `1084efc`.

**What it reviewed:**
- the candidate `outputs/experiment-024/candidate-calibration.json`, file sha256 `81fb499e…`, content sha256
  `09bf093e…`;
- the results state, digest `5ed26580…` (file sha256 `4b319d62…`);
- the arrays file, sha256 `a4cd548f…`.

All were unchanged at the end of the review. The scripts and outputs are in this directory; every run used `rguard.py`,
which refuses any repository write.

| item | result |
|---|---|
| 1. State (`review_state`) | **Repository.** HEAD, origin and `ls-remote` are `bf0049c`; the tree is clean; the freeze is unchanged (committed once, `566eb6f`). **Results state.** Its digest recomputes; calibrate is complete once, with no incident, stop or cross-check failure; lock, confirm and report have not started; the ledger is empty. **Run evidence.** The `run_id` reproduces from the inputs, the commit and the creation second, consistent with one state. The launcher record shows one load, 0 module calls, 0 captures, 0 forbidden reads, and writes only to `outputs/experiment-024`, and the file times fall inside the run window. |
| 2. Bindings (`review_state`) | All match the reviewer's own hashes: the freeze; 023's cells data, index and index content; 023's and 022's confirmation and lock; all 12 module blobs (its own `sha1(b"blob <len>\0"+bytes)`, equal to git's tree), including `readout_decompilation.py`; the locked-states digest (`26c21d63…`); all 21 frozen-input digests, each located independently; the parameter (113 parameters) and embedding digests. 022's table is bound only by digest and was never opened. |
| 3. Nounness (`review_weights`) | **Ids and centroids.** The 158 noun-form ids and 139 cue ids are distinct and in canonical order. μ_noun and μ_cue match, and the exactly rounded means equal the torch means in all 512 coordinates. **Leave-one-out.** The canonical recomputation equals the record bit for bit, 139 of 139. The exact-integer route differs by at most 1.94e-16 with identical ordering (smallest gap between scores 1.1e-6). The check tells the variants apart: a subtraction-based leave-one-out would change 21 scores, and none would change all 139. **Maximum.** 0.13502027836111233, the cue "paper". |
| 4. Per-cue MSE (`review_state`, `review_stats`) | From the raw artifact alone: 139 cues, no pronoun, canonical order, 79 nouns in every cell. The `fsum` values equal the record 139 of 139; the exact-rational route differs by at most 2.2e-16 relative. Range 0.02775 to 0.15684, all distinct and positive. 0 attempts to open 022's table or 023's local outputs. |
| 5. The line (`review_stats`) | An exact-rational OLS reproduces slope `1.2992017464639374`, intercept `-2.481260357420486` and residual sd `0.2570037337349923` exactly. The exposed Spearman is `0.5296617364493499`. The rounded prose values appear nowhere in `rr` or `run.py`. |
| 6. The 10,000 draws (`review_stats`) | Own SHA indices equal the saved ones (`955930ce…`); draw values `7c03376c…`, flags `76b5b970…`; 0 undefined. Every saved value bitwise-equals the exactly rounded Spearman (at most 5.6e-17 from the exact real value). F_ρ is element [249] = `0.24411074612857814`, between neighbours 0.24371 and 0.24429. Min, median and max are −0.04808, 0.52852 and 0.84872; the direction check passes. |
| 7. The null (`review_stats`) | Own Fisher–Yates permutations equal the saved int8 array (`a68f7291…`). The null values equal the exact `1 − S/10660` bit for bit (`92e63029…`). null₉₇.₅ = 836/2665 (S = 7316) = `0.3136960600375234`. 15 values tie at the threshold (note 1). No 99.5 % value is used anywhere. |
| 8. The threshold | T_primary = max(F_ρ, null₉₇.₅) = `0.3136960600375234`, bound by the null. |
| 9. The E–N guard | Bound, not evaluated: 8 + 8, 12,870 assignments, K ≤ 321, ties against the guard, exact dyadic arithmetic, natural log, E first. The record holds no K, D_EN, threshold or result, and no synthetic outcome was created. |
| 10. The fresh-score boundary (`review_weights`) | The record binds no fresh score, and the 40 fresh ids are disjoint from the cue, noun and pronoun ids. A weights-only reproduction, with every forward blocked (0 module calls during and after the load), gives the expected lock values: digest `a703ac16…` (own exact route within 2.2e-16); calibration maximum `0.13502027836111233`; 5 of 8 E cues above it (apple, horse, doctor, poet, dragon; the nearest is 0.0064 away); every E above every N (lion 0.1085 > nervous −0.0451). |
| 11. No fresh measurement | No stage-2 file, lock, preregistration or report exists; the ledger is empty; no manifest key appears anywhere. The arrays file holds exactly the five calibration arrays. The tier-C test re-measures only four allow-listed spent pairs, and all earlier scratch work loaded weights only. |

## Notes (non-blocking; the calibration is unchanged)
1. **Null ties.** 15 null values tie at the threshold, at elements [97497] to [97511]. So 2,503 of 100,000 values are
   ≥ t (2.503 %) and 2,488 are > t. The frozen rule, ρ ≥ element [97499], stays unambiguous, and the excess over 2.5 %
   is well inside Monte Carlo error (about 0.05 percentage points). This comes from the design's discrete null, not the
   implementation.
2. **F_ρ sits below the design estimate.** F_ρ = 0.2441, against the design's rough 0.257. The null binds anyway, so
   the envelope-only band is empty for this calibration. As a descriptive rate, 5.93 % of the exposed calibration
   draws fall below the threshold.
3. **The implementer's evidence proves determinism, not correctness.** Its "from scratch" recomputation used the
   canonical `rr` functions, so it shows determinism rather than agreement with the specification; the reviewer's
   independent implementations close that gap. The run evidence is archived with this commit.
4. **The install commit stays non-scientific.** It touches only `calibration-v1.json` (then the evidence and the
   READMEs). Any change to `src/` or `run.py` would make `lock` refuse.
5. **A guard misfire in the reviewer's own run.** Its audit hook once blocked a `filelock` temporary-directory cleanup
   in `$TMPDIR`. Nothing was created in the repository, and the final runs show 0 refused writes and 0 forbidden reads.

**Install statement (the reviewer's).** Install the existing candidate byte for byte as `calibration-v1.json`, without
regeneration.
- It is canonical JSON plus a newline, and no `.gitattributes` or autocrlf setting could change its bytes.
- The checks `lock` will run pass now: the state's record digest, `verify_calibration_record`, the inputs, the freeze
  binding, and the recomputed bindings and dependencies.

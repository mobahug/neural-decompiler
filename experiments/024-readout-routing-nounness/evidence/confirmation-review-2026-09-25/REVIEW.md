# Independent confirmation review — 2026-09-25

**Verdict: PASS WITH NOTES — no blockers.** A separate agent reviewed the one production `confirm` read-only, at
`af160ce`.
- **Method:** it treated `stage2-measurements.pt` as immutable evidence and derived the scientific result from it with
  its own code. It did not trust the recorded ρ, K or outcome. It called the canonical module only where the frozen
  definition itself was being applied (the 020 readout, to rebuild C; the gate formulas) or where byte identity was
  checked.
- **Guards:** every script ran under `rguard.py`, which refused 0 repository writes. The only model access was a
  weights-only load sealed by `rseal.py`.
- **Afterwards:** the tree stayed clean, HEAD, origin and the remote stayed `af160ce`, and all six output files kept
  their hashes and modification times.

**Independently derived from the saved measurements:**
- **ρ = 1628/2665 = `0.6108818011257036`.** Neither vector has ties; its own average-rank implementation,
  `rr.spearman` and `rr.spearman_direct` agree exactly.
- **The locked threshold is `0.3136960600375234`**, max(F_ρ 0.24411074612857814, null₉₇.₅), bound by the null. Compared
  exactly, ρ ≥ threshold: **PASS**, with a margin of +0.297.
- **D_EN = 20465703117234853 / 2⁵⁵ = `0.5680373703974243`.** It comes from exact integer enumeration of all 12,870
  assignments of the binary64 natural-log MSE values, with ties counted by `D_perm ≥ D_observed`.
- **K = 1 / 12,870**, so p = 1/12,870; PASS, since K ≤ 321. The next assignment is 0.0195 below the observed one, and
  the float cross-check differs by 1.1e-16. Every E log-MSE exceeds every N log-MSE, by at least 0.078.
- **Outcome: `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.** It was derived only from the
  validity result, its own ρ, the locked threshold, its own K and its own transcription of the installed outcome table.
  No secondary or descriptive statistic participates.

**The reviewer's statements:**
1. The confirmation result is **scientifically valid** under the frozen Experiment 024 protocol.
2. The exact saved measurement artifact and result state **must be preserved unchanged**:
   - `stage2-measurements.pt`: `b21babe1…`, 76,314,087 bytes;
   - `results.json`: file `e636210c…`, state `076ab9f9…`.
3. Experiment 024 is **ready for report and closure** after the evidence is installed.

| item | result |
|---|---|
| 1. Integrity (45/45, `s01`) | HEAD, origin and the remote are `af160ce`; the tree is clean. The freeze, calibration, lock and preregistration are byte-identical, and their content digests recompute. Confirm completed once (19:32:10 → 19:37:04); report has not started; no incident, stop or failure exists anywhere. `assert_phase_allowed` refuses confirm and allows report. No scientific path changed since the lock (32 paths, all non-scientific). The module blob `e6cb3776…` is the same in the lock, its own sha1, git and `rr.own_blob()`, and all 12 pins match. |
| 2. Single execution (36/36, `s02`) | An AST read of the launcher finds exactly one `runner.confirm()` call, at module level, outside any loop or handler. The sentinel is exclusive and created first. The counts: 1 invocation, 1 load, 4,320 captures, 0 patched runs, 0 interventions. The dry run shows every counter at 0 and a stub-only prompt log. Environment: 4 threads, the pinned checkpoint re-hashed. **The state transition:** reverting only confirm's four fields reproduces the pre-confirm state `58891c24…` and its file `14459cdd…` byte for byte. |
| 3. Accounting (25/25, `s03`) | Its own manifest has 4,320 keys, digest `fcc437fc…`. Ledger = manifest = prompt log, key for key, with 0 missing, 0 duplicated and 0 extra. The 39,312 spent keys (disjoint sources, rebuilt from the files) have 0 collisions; no executed ledger of experiments 005–023 on the machine contains any key. |
| 4. The artifact (15/15, `s04`) | sha256 `b21babe1…`, 76,314,087 bytes; 20 tensors; 2,880 + 1,440 pairs; all finite. Its own tensor digests equal the state's. It was saved and fsynced before any scoring (`run.py` 550 → 552 → 563 → 577), and the file times agree. |
| 5. C rebuilt (22/22, `s05`) | Its own row mapping equals `rr.target_units`, and the saved positions match. Under a sealed, weights-only load (0 module calls; 499 classes stubbed; 16 entry points refused, 0 attempts), `ul.contrast_of` on the saved Δx3 equals the saved C for **4,320 of 4,320 rows, max difference 0.0**. |
| 6. Identities (`s05`, `s06b`) | I1 `1.5317713646822995e-05` (tolerance 1e-4); I3 `2.1161813141654138e-05` (1e-4); I4 `6.424818726813442e-05` (1e-3); Level-1 identity `0.009929305188346005` (2e-2, descriptive). Its own loop, `rr.target_gates` and the recorded values agree exactly, locations included. I7 re-run under the seal is bitwise equal. No incident condition is met. |
| 7. Per-cue MSE (`s07`) | 40 values, all finite and positive; `s07_results.json` holds the full table. Its own `math.fsum` values equal the canonical ones bit for bit in 37 of 40 cues; nervous, gallon and lion differ by at most 2.8e-17, from the canonical summation order. A mimic of that order matches 40 of 40. |
| 8. Primary Spearman (`s07`) | 1628/2665 = `0.6108818011257036`, no ties, disagreement 0.0: PASS. |
| 9. E–N guard (`s07`) | D_EN 20465703117234853/2⁵⁵, K = 1: PASS. With correctly rounded MSE values, D_EN would be one ulp higher; K is unchanged. |
| 10. Outcome (`s07`) | Derived as above, and equal to `rr.outcome` and the recorded label. `rr.score` calls no secondary function. |
| 11. Atomic state (14/14, `s11`) | The state writes are at `run.py` 547, 555, 560, 569, 574 and 597, plus the three descriptive writes. The result is assigned only at 594, after scoring. Removing only the descriptive records from the final state reproduces **`37f2f88b…`**, the digest logged at the result write. No incident coexists with the result. |
| 12. Descriptives (16/16, `s12`) | Every recorded value reproduces with 0 difference. The noun effect is +0.538, interval [0.395, 0.677]; the measure effect −0.126, [−0.252, +0.004]; the plurality effect +0.076, [−0.058, +0.203]; the interaction +0.020. Cue-final ρ 0.586, coordinated 0.698, nMSE ρ 0.592. The line's residuals: mean +0.093, with gallon +0.72 and acre +0.63 largest. 23 of 40 cues lie above the calibration maximum. The block-4 share is 0.58–0.73. |
| 13. Claim scope | Supported exactly as the predictive claim; see below. |
| 14. Evidence | Listed; archived here and in `../confirm-2026-09-25/` and `../postinstall-review-2026-09-25/`. |

## Non-blocking notes
1. **The README was stale** at review time. It is updated in the same commit that archives this review.
2. **The launcher's record is a self-report**, corroborated four ways: the state reversal, the key-for-key accounting,
   the re-derived tensor digests and the file times.
3. **Last-bit MSE differences in 3 of 40 cues** (≤ 2.8e-17) come from the canonical summation order, which stays
   authoritative. Ranks, ρ and K are identical on every route.
4. **The noun ledger has 80 keys**: every pool noun, including the non-scorable `selection-holdout:peach`. This matches
   020's ledger and holds no fresh noun.
5. **The null threshold decided the pass**, so ENVELOPE_ONLY_FAILURE could not occur.
6. **The frozen line under-predicts** on average (+0.093 in log MSE), more within range (+0.173) than over the
   extrapolated cues (+0.034). This is descriptive.
7. **`results.json` has mode 0600**, a side effect of mkstemp plus `os.replace`. No action is needed.

## Claim-scope notes
- **The supported claim.** On 40 fresh cues, the frozen operational weight-derived nounness score prospectively
  predicted larger error in the frozen block-4/5 attention/readout approximation. The association passed the
  preregistered E-vs-N disambiguation guard against simple plurality-only and measure-only explanations.
- **Where PASS is overstated.** The stored PASS reading ("at least as strongly as the exposed-like relationship") is
  stronger than the rule. The rule is ρ ≥ max(F_ρ, null₉₇.₅), and F_ρ is the 2.5th percentile of the exposed-like draws.
  State simply that the prospective association exceeded the locked primary threshold.
- **The limits of E versus N.** E and N differ in more than nounness: all eight N cues are adjectives, and the E cues are
  concrete, mostly animate nouns. The guard formally rules out only plurality-only and measure-only. It does not
  identify nounness as the causal variable, or which part of the score matters.
- **"Readout error" is the error of the frozen-attention approximation**, not measured rerouting. The scope is 108
  exposed frames, 79 nouns, one checkpoint and 8 cues per class, and 23 of 40 cues are extrapolated.
- **Overclaims to avoid:**
  - nounness causes rerouting;
  - measured rerouting was observed;
  - the model detects nouns;
  - nounness is a general linguistic representation;
  - E-vs-N rules out all lexical or semantic confounds;
  - 024 explains all routing-regime changes, or establishes general algorithm choice.

  No such wording was found in the preregistration, README, design or plan.

**The reviewer's decision (2026-09-25).** The review is accepted, and the confirmation result is accepted as
scientifically valid under the frozen protocol.

The files here (`rguard.py`, `rseal.py`, `s01`–`s07`, `s06b`, `s11`, `s12`, `s14` `.py`/`.out`, and `s05`/`s07`/`s12`
`_results.json`) are the reviewer's, as written and run. `SHA256SUMS` was computed at archive time, including
`s14.out`.

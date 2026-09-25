# Independent lock review — 2026-09-25

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the lock read-only, at `be74d23`.
- **Method:**
  - It wrote its own code for every check, including an exact-integer, 60-digit route for the 40 scores taken straight
    from the pinned fp16 checkpoint.
  - It called the canonical module only where byte identity was the thing being checked: the rendering, the digests,
    `fresh_quantities`/I7 and `validate_lock`. It reported numerical disagreement separately from byte identity.
  - Every run used `review_guard.py` (for pytest, `review_pytest_guard.py`), which refuses any repository write and
    stubs every forward after the load. 0 writes were refused.
- **Result:** every bound number, digest, ordering, threshold and rule reproduced. The existing candidates can be
  installed byte for byte without regeneration; that was done in `3affe55`.

**What it reviewed:**
- `outputs/experiment-024/candidate-lock.json`, file sha256 `5c2a9b90…`, content sha256 `a39668bf…`;
- `outputs/experiment-024/candidate-preregistration.md`, sha256 `007c9e6c…`;
- the results state, digest `58891c24…` (file sha256 `14459cdd…`), and the calibration arrays `a4cd548f…`.

All were unchanged at the end of the review. HEAD, origin and `ls-remote` stayed `be74d23`, and the tree stayed clean.

**The scripts in this directory:**
- `r01_state_integrity.py`: items 1, 2, 3, 5, 6, 7, 9 and 12 on the file side; 100 checks.
- `r02_prereg_numbers.py`: the preregistration number sweep.
- `r03_scores_i7.py`: items 4 and 10, and the `validate_lock` acceptance and refusals; 51 checks.
- `r04_state_transition.py`: the pre-lock state reconstruction.
- `r05_outcome.py`: item 8.
- `pytest_tierA.out`: 30 passed.
- `pytest_tierB_runner.out`: 31 passed in 408 s.

| item | result |
|---|---|
| 1. State | **Repository.** HEAD, origin/main and `ls-remote` are `be74d23`; the tree is clean, with no stash and no untracked file. **Committed files.** The freeze (`68510e1b`/`87f8aff1`) and calibration (`81fb499e`/`09bf093e`) files are canonical, equal to the HEAD blobs, and each committed once. **Results state.** Its digest recomputes to `58891c24`. Calibrate and lock are complete with no incident or stop; confirm and report have not started; both ledgers are empty; the outputs hold exactly five files, none of them a stage-2 file. **Zero prompts during lock:** (a) undoing only the lock's entries in the state reproduces the pre-lock state `5ed26580` (file `4b319d62`) byte for byte, so the lock changed nothing else, the ledger included; (b) a sealed reload ran `ModelPrograms.from_model`, `score_bindings` and `fresh_quantities` with 0 module calls; (c) `Runner.lock` has no code path to `capture_prompt`, `measure_prompt` or `stage_two_022`; (d) the launcher record agrees. **Limits.** The launcher record is a self-report. It counts `Module.__call__` only, and `invocations: 1` is a constant (note 3). |
| 2. Integrity and rendering | **Hashes.** Both file hashes match. Its own content digest (key removed) is `a39668bf`, and the `pm` route agrees. **Format.** Canonical JSON plus a single newline; all 96 floats are finite. **Rendering.** The preregistration equals `render_preregistration(lock)` byte for byte, and its digest equals the state's. **Number sweep.** Every float in the preregistration parses back bit-exactly to the lock: F_ρ, the null, the threshold, the line, and all 40 rows (class, word, id, score, prediction, flag). All 31 backticked tokens are lock strings. The only remaining numbers are 020, 023, 024, the "4" of "block-4/5", and `+0.135020`, the maximum to 6 decimals. |
| 3. Bindings | **Files.** The record (path, file, content), the freeze with its counts and manifest, and 023's cells (`d2ee71e5`/`62818147`/`d38305cb`, its own hashes). **020's readout.** The exposed-states digest `26c21d63` recomputed now, and the frozen-input digests; `lock.inputs` equals the base digests computed now. **Model.** Parameters `fd953f1c` (113 parameters) and embedding `9cd6f39b`. **Pinned modules.** All 12 blobs: its own sha1 equals git's HEAD tree, `FROZEN_BLOBS`, the lock and the record. **The analysis module.** `readout_routing.py` has blob `e6cb3776`, equal to the file, HEAD and `be74d23`, and it is imported from the repository (editable install). **Secondary definitions.** `CONTRASTS`, the `024\|contrast` tag, elements [249]/[9750], `secondary` and `ladder` are frozen through that blob and confirm's refusal of scientific paths; the constants are also re-checked by `validate_lock`. `scientific_changes` classified 16 constructed paths correctly. The tier-B tamper test passed. |
| 4. The 40 scores | **Exact route.** From the pinned safetensors fp16 (sha `3da38833`, its LFS id), in exact integers and 60-digit arithmetic: at most 2.196e-16 from the lock, identical ordering, smallest gap between neighbours 3.56e-4. A separate float64 numpy route: 2.220e-16. **Loader and digest.** The loader's `W_E` is bit-identical to the fp16 rows, with 0 module calls during and after the load. The canonical digest is `a703ac16`; correctly rounded exact values would give `e494736a`, so this is numerical agreement, not byte identity. **Classes** (min / max / mean): N −0.229351 / −0.045076 / −0.146095; B +0.006685 / +0.396574 / +0.199902; D −0.065554 / +0.302937 / +0.101159; C +0.206190 / +0.345916 / +0.295034; E +0.108452 / +0.298294 / +0.174030. **Calibration maximum.** 0.13502027836111233, the cue "paper" (exact 0.1350202783611122…). **Above it.** Exactly 5 of 8 E (apple, horse, doctor, poet, dragon); the nearest fresh score is 6.38e-3 away. Every E is above every N. Descriptively, 23 of the 40 cues. |
| 5. The frozen predictor | Every `predicted_log_mse` equals `intercept + slope·x` exactly in float64, within 0.555 ulp of the exact rational value. The line equals the record's, and an exact-rational fit reproduces it bit for bit. The exposed Spearman `0.5296617364493499` reproduces. No fresh MSE exists anywhere. |
| 6. The primary test | F_ρ = element [249] = `0.24411074612857814`; null = element [97499] = `0.3136960600375234`, an exact rational; the threshold `0.3136960600375234` is bound by the null; 0 undefined draws. Its own SHA regeneration of all 10,000 × 40 draws and all 100,000 permutations equals the saved arrays, and so do the five array digests. `score()` reads the thresholds only from the validated lock, and no confirm path can replace them. |
| 7. The E–N guard (bound, not evaluated) | **Spec.** The units are exactly the frozen 8 E then 8 N, and the spec equals `PRODUCTION_GUARD`: 321/12,870 ≤ 0.025 < 322/12,870. **Enumeration.** 12,870 distinct subsets, the observed one first, with complement pairing. **Code.** Exact dyadic integers; ties count against the guard (`>=`); PASS iff K ≤ 321; all-tied gives K 12,870, a FAIL. **Tests.** Tier A 30/30, including 321 PASS, 322 FAIL, 12,870 FAIL and the adversarial case (float K 9,867, exact K 6,435). No K was evaluated on any data. |
| 8. The outcome | `rr.outcome` matches the design table on all 12 (primary, guard) pairs. `outcome`, `en_guard` and `classify_primary` are called only inside `score()`. `_descriptives` writes only descriptives and failures, so no secondary path reaches the label. |
| 9. The manifest | 4,320 unique keys, `fcc437fc`, in the stored canonical order. The spent keys: 020's ledger 30,132, 020's confirmation set 3,060, and 022's and 023's manifests 3,060 each, 39,312 in all. There are 0 collisions, no fresh id appears in any spent key, and the ledger is empty. |
| 10. I7 | **Rehearsal.** Run sealed, as confirm runs it: `bitwise_equal` True, nothing differing (the centroids, scores, predictions, flags, maximum, 5 of 8 and the sentence). A one-ulp drift is detected, and a wrong parameter digest is refused. **Code.** An I7 mismatch records a phase incident and returns before the ledger write and before `stage_two_022`. `test_i7_refuses_a_drift_before_any_fresh_prompt` passed. |
| 11. Confirm's write order | **Order in `run.py`:** `validate_lock` → the model digests → I7 → the ledger written → `stage_two_022` → the durable save with its digests → the accounting (written, then enforced) → re-read and digest check → C bit for bit → the gates (written, then enforced) → `score` (MSE, ρ with its cross-check, exact K, outcome) → the 020/022/023 re-check → one atomic write of the result and the completed phase → the descriptives. **Failure path.** If that write fails, the result is removed and an incident recorded. An incident never coexists with a result, and a second confirm is refused. **Tests.** Tier B 31/31, including the write-order test, all 5 post-measurement incidents, the failed result write, and both the descriptive failure and the interruption. |
| 12. No leakage | No 024 stage-2 file exists. The only tracked file with fresh keys is the freeze manifest. The state and the lock carry no Δc, Δx3, MSE, ρ or K, and none of the 4,320 keys has run. |

## Notes (non-blocking; the lock is unchanged)
1. **Confirm never compares `lock.module.blob` (`e6cb3776…`) with `rr.own_blob()`.** Code identity at confirm holds
   only indirectly: a clean tree, no scientific path changed since `be74d23`, and the editable install. The reviewer
   verified that `src/` and `run.py` are unchanged from `44c3c2b` through `be74d23`. It suggested that the confirm
   launcher assert the blob and record it.
2. **A weight-digest mismatch is refused, not recorded.** A parameter or embedding digest mismatch at confirm raises
   `PhaseError` without an incident. Only an I7 mismatch records one. Both happen before the ledger and any prompt.
3. **The launcher evidence has limits.** `lock_run.json` is a self-report and was not yet hash-anchored. It counts
   `Module.__call__` only, not a direct `.forward()`, and seals at the first parameter digest. Commit it with its sha256;
   items 1 and 4 close the gap independently.
4. **The envelope-only band is empty.** F_ρ is below null₉₇.₅, so ENVELOPE_ONLY_FAILURE cannot occur; any ρ below
   0.3137 is a GUARD_FAILURE. The preregistration states that the null binds.
5. **A non-finite fresh MSE is an incident**, not NOT_INTERPRETABLE. This is documented and conservative; an MSE of 0
   still gives NOT_INTERPRETABLE.
6. **Install hygiene.** The install must touch only the two files, and then the README and `evidence/`. Anything
   scientific makes confirm refuse, before the model loads. The runtime check needs 4 torch threads (the default here).
7. **A false flag in the reviewer's own guard.** It first flagged `filelock`'s probe deletions in the system temp
   directory (a relative path with `dir_fd`). Nothing in the repository was touched; the hook was fixed and the runs
   repeated cleanly.

**The reviewer's decisions on the notes (2026-09-25).** The review is accepted.
- **Note 1 is acted on without a code change.** The confirm launcher must independently assert
  `rr.own_blob() == installed_lock["module"]["blob"]` before the model loads, and record the expected and observed blob
  in the confirm evidence. A mismatch stops before the model load, the ledger or any prompt.
- **Note 2 is acceptable as it is.** A weight-digest mismatch before the ledger and any prompt may simply refuse
  confirmation; nothing has been spent at that point.
- **Note 3:** the evidence commit records `lock_run.json`'s sha256 (`b7a51118…`; see
  [`../lock-2026-09-25/README.md`](../lock-2026-09-25/README.md)).

**Install statement (the reviewer's).** Install the existing candidates byte for byte, without regeneration, as
`preregistration-lock.json` and `preregistration.md`.
- **Accepted.** `validate_lock`, called as a pure function on the exact candidates as if installed, accepted them: with
  a clean tree, and with the install commit's own paths changed (the two files, the README and `evidence/`).
- **Refused:**
  - a change to `readout_routing.py`, `run.py` or 023's cells;
  - a scientific file renamed away;
  - a commit that is not an ancestor;
  - a dirty tree, or untracked files;
  - an edited preregistration;
  - a lock resealed with a replaced threshold, even with a forged state;
  - a forged record;
  - a non-production configuration.

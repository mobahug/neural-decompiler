# Experiment 101: Is the Read Component at the Cue Position Causal for the `one`/`two` Choice?

**Status (2026-10-14): CLOSED — `READ_COMPONENT_EFFECTS_AS_PREDICTED`.** Every phase ran exactly once on the branch
`exp101-phases`, each independently reviewed by a separate AI agent (no review is a human validation): the freeze
(`a857f9a`), the calibration (`d502777`), the lock (committed by the user, `961e814`), stage 1 (its prediction table
committed by the user, `59d8ec4`), confirm (at `59d8ec4`) and the report. Confirm: precondition 36/36; R 36/36 (median N
1.006); I 36/36 (median S 1.035); D `A_amp` < 0 28/36 and `A_neg` > 0 36/36; P 33/36 within τ; every gate held; the
confirmation review recomputed the label bit for bit. **Scope:** the read component of the residual entering layer 3 at
the cue position, as every reader of that residual sees it, in coordinated frames, for the cue pair one/two — not
`L03.H04` alone (≈88–89 % of the effect runs through its output), not beyond this pair, site, checkpoint and these 36
frames. See the [closure](evidence/closure-2026-10-14.md) and the [final report](evidence/final-report-2026-10-14.md).
A claim review is a separate decision and has not been made.

Experiment 101 opens the second series (Neural Decompiler II), numbered from 101. It is the causal test of claim
[C003](../../research/claims/C003-frame-conditioned-decoded-program.md) that the paper-readiness assessment named:
remove, install, double and reverse the component of the residual entering layer 3 at the cue position that
`L03.H04` reads, in 36 new coordinated frames, against an equal-length slice of the cue's own off-read change.

- **Design:** [revision 3](../../docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md)
  (`a87e284`), as the reviewer decided it (§14, `516df20`).
- **Plan:** [revision 3](../../docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md)
  (`96f0cc8`), with the reviewer's decision on G4's form (`1e48c1c`).

## Code

- **Module:** `src/neural_decompiler/read_intervention.py` (`ri`).
  - It calls the frozen modules and never edits them; 20 are pinned by git blob at `516df20`.
  - It binds 025's committed confirmation file by file and content digest (its manifest joins the forbidden set; its
    cues and candidate lists feed the freeze's exclusions).
  - The production configuration is literal and immutable: 36 frames, the cues `one`/`two` with the stratum
    `three`/`many`, 8 isotropic directions, the Holm thresholds 26 and 25, D 25, P 27, the precondition 30. A test world
    can only pass its own configuration explicitly.
  - `ri.run_arm` and `ri.capture_reference` are the only functions that run a 101 prompt; the guarded launchers wrap
    them by identity.
- **Runner:** `experiments/101-read-component-intervention/run.py`.
  - Phases: `validate`, `freeze`, `calibrate`, `lock`, `stage1`, `confirm` and `report`.
  - It takes no option and always runs the production configuration.
- **Tests:**
  - `tests/test_read_intervention.py` holds tier A (pure) and the tier-C contracts (the pinned model, opt-in, spent
    exposed prompts only; a guard refuses any 101 frame prompt by its token sequence, and every outcome function).
  - `tests/test_experiment_101_runner.py` holds tier B (the six-layer fake, on 025's frozen fake world).

## Phases (each only when separately authorized)

0. **The operator restores** `outputs/experiment-020/results.json` from
   `experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz` with the command in that
   folder's README; every phase verifies its sha256.
1. **`freeze`** (tokenizer only) writes `confirmation-v1.json`, which is committed alone and reviewed. A shortfall, or
   picks other than candidates 1–36, writes nothing. **Done on 2026-10-14:** candidates 1–36, file sha256 `6da26cd7…`,
   content sha256 `ef356506…`, installed in `a857f9a`
   ([evidence/freeze-2026-10-14/](evidence/freeze-2026-10-14/README.md)); AI review PASS WITH NOTES
   ([evidence/freeze-review-2026-10-14/](evidence/freeze-review-2026-10-14/README.md)).
2. **`calibrate`** (exposed, exploratory) re-captures the 36 exposed coordinated reference states, runs every arm on
   them, computes G1–G6, `τ_rem` and `τ_ins`, and evaluates the four revision triggers. A trigger stops for review and
   writes no calibration file; no trigger widens a tolerance. The candidate `calibration-v1.json` is installed,
   committed and reviewed. **Done on 2026-10-14:** complete, no trigger; τ_rem 0.025153865322968608, τ_ins
   0.03511072680382421 (P evaluable); G4's tightest margin Level-1's, 0.010654 against 2e-2 (1.88×); file sha256
   `6b1db338…`, content sha256 `1eb1fe68…`, installed in `d502777`
   ([evidence/calibrate-2026-10-14/](evidence/calibrate-2026-10-14/README.md)); AI review PASS WITH NOTES
   ([evidence/calibration-review-2026-10-14/](evidence/calibration-review-2026-10-14/README.md)).
3. **`lock`** (weights only; every forward refused) writes the candidate `preregistration-lock.json` and
   `preregistration.md`, installed byte-identically, committed and reviewed. **Ran on 2026-10-14** at `3621eec`: candidate lock content
   sha256 `8da97e35…` (file `8dc6e6d3…`), preregistration `3761b813…`; AI review PASS WITH NOTES, no blocker
   ([evidence/lock-2026-10-14/](evidence/lock-2026-10-14/README.md),
   [evidence/lock-review-2026-10-14/](evidence/lock-review-2026-10-14/README.md)). **Not installed: the user commits
   it.**
4. **`stage1`** runs only the 36 reference prompts, seals the model, computes the program's table from the states and
   the weights, and writes `stage1/states.f64`, `stage1/states.json` and `stage1/table.json`. The user commits them.
   **Ran on 2026-10-14** at `961e814`, on the user's authorization: exit 0, the 36 S1 keys once each; files
   `1c7e51c0…`, `2b2c3a61…`, `71d339cd…` ([evidence/stage1-2026-10-14/](evidence/stage1-2026-10-14/README.md)); the
   launcher re-check READY FOR PRODUCTION and the stage-1 review PASS WITH NOTES, no blocker.
5. **`confirm`** (once; never resumed) passes the barrier, writes the ledger of all 1,980 S2 keys, runs each once,
   saves the measurements, checks the accounting, computes G2–G6, scores R, I, D and P, and writes the result in one
   atomic write; then the secondary records. **Ran on 2026-10-14** at `59d8ec4` on the user's authorization: exit
   0, `READ_COMPONENT_EFFECTS_AS_PREDICTED` ([evidence/confirm-2026-10-14/](evidence/confirm-2026-10-14/README.md));
   confirmation review PASS WITH NOTES, no blocker, label recomputed
   ([evidence/confirmation-review-2026-10-14/](evidence/confirmation-review-2026-10-14/README.md)).
6. **`report`** recomputes the table (G6) and renders the report. **Ran on 2026-10-14:** G6 bit for bit; `report.md` `54f883ad…`
   ([evidence/report-2026-10-14/](evidence/report-2026-10-14/README.md)).

A gate failure after the measurements is an integrity failure under the incident procedure: no frame is dropped,
replaced or ignored, no outcome is scored, and the report renders `NOT_INTERPRETABLE`.

The local outputs stay in `outputs/experiment-101/`.

## Protocol notes

1. **G4's form** was decided by the reviewer on 2026-10-05 (plan § G4's form and the reviewer's decision, `1e48c1c`),
   with two frozen rules: calibration may test the G4 tolerances but never widen them (a calibration G4 failure stops
   before the lock and needs a reviewed amendment), and G4 is an integrity gate, not an outcome filter. Calibration is
   where the frozen tolerances meet the layer-3 edits for the first time; the tightest recorded margin before 101 was
   Level-1's, 1.7–1.85×.
2. **Who commits.** The design's one named human step is the lock: "The user commits the lock" (design §9). The phase
   files before it (the freeze's confirmation file and, after its review, the calibration file) are installed and
   committed on `exp101-phases` by the operator, each alone and byte-identically, as 023–025's were; the branch is
   local until the user pushes it. The lock and the stage-1 files (the prediction freeze) are committed by the user.
3. **The freeze review's notes** (all non-blocking; [REVIEW.md](evidence/freeze-review-2026-10-14/REVIEW.md)):
   - SF1 (preserve the freeze evidence in the repository) is answered by the evidence commit after `a857f9a`; the first
     dry run's launcher was not preserved, and a reconstruction is recorded as such.
   - N1 (audit-hook argument mapping) is applied to the calibrate and lock launchers: `dir_fd` is read per event, and a
     mode or a length is never taken for a descriptor.
   - N2 (pure helpers such as `ri.main_fractions`, `ri.predicted_change`, `ri.ratio` and `ri.g2_values`): the later
     phases compute with them on their own data (calibration's fractions and gates, the lock's and stage 1's tables),
     so a launcher refuses them only where its phase does not need them.
   - N4 (provenance): the implementation commits `1e48c1c`…`40829d2` and every phase commit are local until the user
     pushes them; no commit on this branch is rewritten.
   - N7: the calibrate launcher's unguarded smoke reference ran the S1 re-capture and the 55 arms of the spent exposed
     frame `coordinated-adjective-1` (020's pool) after the freeze; it is engineering only, outside every 101 ledger,
     like tier C's runs on spent frames, and the calibrate evidence records it.
   - N8: later test-gate logs record their HEAD and time.
   - N12, a process lapse: the install commit `a857f9a` was made after the review's report stated its verdict but before
     the reviewer had handed back; the final verdict was unchanged. Later phases wait for the hand-back.
4. **The calibration answers the G4 question with evidence.** Every frozen G4 tolerance held on the 1,800 exposed G4
   runs; no amendment was needed or made. The tightest margin is Level-1's, 1.88× (0.010654 against 2e-2, one stratum
   arm in the frame `018-3`; `two`'s arms at most 0.00501). The calibration review traced the two outliers to float32
   rounding in the measured blocks 4–5 (020's attention-score floor), not to the edit, and estimated the chance that one
   of the confirm's G4 runs exceeds 2e-2 at about 0.3 % (0.02–1 %; about 10 % at the 95th percentile of a heavy-tailed
   frame bootstrap). Such a failure would make the confirmation `NOT_INTERPRETABLE` (rule 2). This is for the user's
   lock decision; no tolerance changes (rule 1).
5. **The calibration review's notes** ([REVIEW.md](evidence/calibration-review-2026-10-14/REVIEW.md)): SF1 (preserve the
   calibrate evidence; complete its README) is answered by the evidence commit after `d502777`; N2 (calibration's G6 is
   a same-process check; the cross-process G6 runs at confirm), N3 (I4 is vacuous on reference-prompt runs, by design),
   N4 (`coordinated-adjective-1`'s spent keys ran three times in all, with bit-identical results), N5 (the run log
   records admissions), N6 (a harmless `.git/index` refresh) and N7 (descriptive observations) are recorded in the
   calibrate evidence.
6. **The lock review's notes** ([REVIEW.md](evidence/lock-review-2026-10-14/REVIEW.md)): no blocker to committing the lock.
   Its S1–S3 concern the prepared stage-1 launcher (version 1; not part of the lock) and are applied in version 2, which
   still needs a re-check before stage 1. N2: confirm does not record design §2's per-frame `s/‖off_c‖` and overshoot
   flag; the confirmation review or the report should recompute them from the saved measurements. N4: do not rewrite
   this branch. N9: keep the machine's environment fixed until confirm, since stage 1 compares the geometry with the
   lock bit for bit and a mismatch is a phase incident.
7. **The closure** ([closure-2026-10-14.md](evidence/closure-2026-10-14.md)) answers the confirmation review's SF1 (the
   per-frame overshoot report design §2 asks for: 0 of 36 frames flagged for every cue) and SF2 (the per-arm ceilings
   and E₁), records its notes (the stratum `many` would fail D; D is saturation-limited; the head counts are nearly
   vacuous; the scope's implicit limitations), and lists the competing explanations the design does and does not
   exclude.

## Report and closure

- [The final report](evidence/final-report-2026-10-14.md) (`outputs/experiment-101/report.md`, byte-identical).
- [The closure](evidence/closure-2026-10-14.md): the result, what it supports and does not, the secondary results, the
  competing explanations, the overshoot and per-arm tables, the integrity record, and the decisions that remain.
- [The confirmation-record extract](evidence/confirmation-record-2026-10-14.json) of the git-ignored final state (the
  ledgers by count and digest).

# Experiment 101: Is the Read Component at the Cue Position Causal for the `one`/`two` Choice?

**Status (2026-10-14): FROZEN — the freeze ran once; calibrate is next.** The implementation passed its review and
re-check (READY FOR PHASES, `40829d2`). The phases run on the branch `exp101-phases`. The freeze (tokenizer only) ran
once at `40829d2` and wrote `confirmation-v1.json`, installed byte-identically in `a857f9a`; its independent AI review
found PASS WITH NOTES, with no blocker ([evidence](evidence/README.md)). No calibration, lock, stage 1 or confirmation
has run, and no 101 prompt has been executed.

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
   committed and reviewed.
3. **`lock`** (weights only; every forward refused) writes the candidate `preregistration-lock.json` and
   `preregistration.md`, installed byte-identically, committed and reviewed.
4. **`stage1`** runs only the 36 reference prompts, seals the model, computes the program's table from the states and
   the weights, and writes `stage1/states.f64`, `stage1/states.json` and `stage1/table.json`. The user commits them.
5. **`confirm`** (once; never resumed) passes the barrier, writes the ledger of all 1,980 S2 keys, runs each once,
   saves the measurements, checks the accounting, computes G2–G6, scores R, I, D and P, and writes the result in one
   atomic write; then the secondary records.
6. **`report`** recomputes the table (G6) and renders the report.

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

## Report and closure

Not yet.

# Experiment 101: Is the Read Component at the Cue Position Causal for the `one`/`two` Choice?

**Status (2026-10-05): implementation under review. No phase has run.** No frame is frozen, no calibration has run,
and no 101 prompt has been executed.

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
   picks other than candidates 1–36, writes nothing.
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

None yet.

## Report and closure

Not yet.

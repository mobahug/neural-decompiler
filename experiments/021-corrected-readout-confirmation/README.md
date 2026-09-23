# Experiment 021: Corrected Readout-Decompilation Confirmation

Implements the design
[`docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md`](../../docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md)
(revision 3, `0ac46aa`, final design review passed) through the plan
[`docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md`](../../docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md)
(`fb10d44`, implementation/data-flow review passed).

The Experiment 020 question — does the decoded mechanism predict the model's own singular-versus-plural logit contrast
for unseen cues, unseen frames and unseen nouns — with the same program, model, populations, procedure, labels and
untouched confirmation set, and floors from a preregistered exposed-only calibration.

## Inputs

- The program: `src/neural_decompiler/readout_decompilation.py` exactly as at `ce3766b` (git blob `caa73b40…`),
  checked on the imported file by every phase; nothing in 021 edits or reimplements it.
- Experiment 020's closure (`7f4a4f8`): `closure.json` (`f2b1b5e7…`), the committed exploration extract
  (`99f25ee6…`) and the local, gitignored results state `outputs/experiment-020/results.json` (file `da63b8c2…`,
  state `2e5485dc…`), whose ledger (30,132 exposed keys, 0 of the 3060 manifest keys) and recorded exploration are
  the calibration's only exposed source.
- The confirmation set: `experiments/020-readout-decompilation/confirmation-v1.json` (`e098e2b4…`), read in place.
- The Experiment 011, 012 and 017 locks and the 006–019 input chain, exactly as in Experiment 020.

## Commands

```bash
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py validate
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py calibrate
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py lock
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py confirm
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py report
```

- `validate` (no model): the frozen inputs, the program blob, every Experiment 020 closure digest, the confirmation
  set's isolation, the pools (cues 45/45/36/49, frames 14/14/14 before the screen, nouns 40/19/20) and the rows
  (Y2 64, Y3 84).
- `calibrate` (once, ≈ 1.5 h): re-executes exactly 020's ledger with 020's functions in 020's order (references
  first; each key checked against 020's ledger and the manifest before it runs), the environment check at `1e-9`,
  the identities, the reproduction gate at `1e-9`, the frozen validity screen and the precondition (≥ 6 screened
  frames per template, else a stop for review), then the 10,000 SHA-indexed base draws, the statistics of all 149
  rows, the kernel/direct cross-check (an excess stops the phase with its row, draw and statistic, before any floor),
  the floors and the descriptives. Writes `outputs/experiment-021/candidate-calibration.json`, which is installed
  byte-identical as `calibration-v1.json`, committed, and reviewed before `lock`.
- `lock` (no forward pass): the Y1 prediction table for the 24 fresh cues in the 108 exposed frames, with the floor
  tables of the committed calibration record; installed as `preregistration-lock.json` and `predictions.md` and
  committed by hand before the sign-off.
- `confirm` (once): stage 1 (S1-REF and S1-VALIDITY only), the Y2 floor row selected from the validity verdicts and
  digested, the barrier (both digests re-verified from disk, the row recomputed from the re-read verdicts), stage 2
  with the downstream ceiling, the Y3 row from the measured scorability, the scoring through the one pass predicate.
- `report` renders `outputs/experiment-021/report.md`, verifying the per-row draw values against the record.

## Status — 2026-09-23: implemented and independently reviewed, not run

Tasks 1–4 of the plan are done: `src/neural_decompiler/readout_calibration.py`, the runner, 20 unit tests
(`tests/test_readout_calibration.py`, tier A, ≈ 11 s) and 16 runner tests on a fake-closed Experiment 020 world
(`tests/test_experiment_021_runner.py`, tier B, ≈ 5 min, `CURRENT_EXPERIMENT = "021"`). `validate` passes on the real
inputs. A full-scale run of the calibration on a synthetic table (B = 10,000, the real pool sizes, no model) takes
≈ 6 min and ≈ 0.9 GB, with the kernel agreeing with 020's direct statistics to 2.2e-16. No Experiment 021 model run
has happened; `calibrate` waits for the implementation review.

### Implementation notes (from the independent implementation review; nothing scientific changed)

The review found no blocker. Its should-fix and minor items are all in the code:

- Incident records are always writable: a non-finite value (a one-sided undefined statistic, a structural mismatch)
  is stored as `null`; the kernel/direct cross-check completes every row-draw and records the **worst** location with
  the first one and the count, then stops — before any floor or candidate record exists.
- The kernel pools every sum of squared deviations from per-group two-pass moments (Σ M2ᵢ + Σ nᵢ(x̄ᵢ − x̄)²), never
  from Σy² − (Σy)²/n. The implementation-only agreement guard is applied to |kernel − direct| / max(1, |direct|) ≤
  1e-10: absolute for well-conditioned values, relative where a statistic is itself ill-conditioned (a frame-mean
  `R²` of −16,539 on a synthetic draw with four near-identical frame means differed by 2.2e-9 absolute, 1.3e-13
  relative — float reassociation, not a formula difference).
- `calibrate` resumes only after a recorded incident and never at a commit that carries one; an interruption is
  recorded as an incident; the candidate record's digest reaches the state as the record is written; the after-phase
  re-check of Experiment 020's files runs inside the incident handling of `calibrate`, at the end of `lock` and
  `confirm`, and in every incident record.
- `lock` checks the runtime and thread count against 020's explore record, so the locked rows reproduce at `confirm`.
- `confirm` writes every stage-2 measurement (with digests) and the identity maxima to disk before enforcing the
  identities or scoring, so no failure after stage 2 can lose a fresh measurement.
- The 011/012/017 lock digests (`769bfeac…`, `830abc3b…`, `b4fc9014…`) are asserted and recorded; the calibration
  record's constants, program blob and design are verified before `lock` uses it; `rematerialize` calls 020's own
  `assert_explore_nouns`; the report prints each fresh value's exposed draw median and percentile and places the fresh
  ceiling within its row's distribution.

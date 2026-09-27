# Experiment 025 — the closure data (2026-09-27)

Both scripts are read-only on the repository, the state and every artifact. They load no model, run no prompt and
invoke no phase. They ran from the session's scratch directory and are kept here exactly as written and run.

- **`preclosure.py`, `.out`** — the pre-closure check at the confirm-evidence commit `c8d9fad`, after the report run.
  Its audit hook refused any repository write; there was none. All 24 checks passed:
  - HEAD, origin and the remote are `c8d9fad`; the tree is clean, with no stash;
  - the final state `534dad39…` / `fef818dc…`, all three phases complete, no incident, no descriptive failure;
  - `report.md` `ad9f85e6…` and the measurements `87284bdd…`;
  - reverting the two report entries gives the confirm-final state `1d5d37f0…` / `c827bb69…`, and removing the
    descriptive records from that gives the result-write state `aa3db023…`;
  - the official result as recorded, the 90,720-key ledger and the accounting;
  - the installed freeze, lock and preregistration, each equal to its HEAD blob; the direct module binding;
  - the phase rule refuses another lock and another confirm, and the committed freeze file makes the runner refuse
    another freeze;
  - no scientific path changed since the lock commit; the runner's `validate` passes.

  The check was run three times, all passing: the first version, then with the stash, HEAD-blob, phase-rule and
  scientific-path checks added, then with one note's wording corrected. The archived output is the last run.
- **`build_confirmation_record.py`, `.out`** — the builder of
  [`../confirmation-record-2026-09-26.json`](../confirmation-record-2026-09-26.json), adapted from 024's. Its audit
  hook allowed only that one write, and refused any open of 022's local table or of 023's or 024's local outputs.
  - It asserts that the state's digest verifies, and that the confirm-final and result-write states are reproduced.
  - It checks the launcher's run log against the manifest and the ledger (90,720 each, 0 extra, missing or
    duplicated, 4,320 per condition).
  - It checks the isolation against 020's ledger, 021's spent set and the 022, 023 and 024 manifests: 43,632 spent
    keys, 0 executed, tagged or untagged. The four spent patch-path keys are outside the manifest and the ledger.
  - It records the exit-127 shell event: it never started Python and was not a scientific attempt.
  - The official-result summary is copied from the state, never recomputed. It lists the descriptive-only records
    explicitly.

  The builder ran twice. The first extract differed only in one accounting field,
  `run_log_order_equals_ledger_order: false`. That was true but misleading: the runner writes the ledger as one
  sorted set by design. The field was replaced by `ledger_is_sorted: true` and a note on the order, and the builder was
  rerun. The committed extract is the second run's: content `30da650f3d8ce15fdfef86e51a5e2adb8ebb5bb4ec45f1b5aedaa738796052ca`,
  file `c7fca5c06de7c3722fd223a9a1805967c697626e86b85d85f05b7ff6e3a592ba`.

`SHA256SUMS` lists the four archived files.

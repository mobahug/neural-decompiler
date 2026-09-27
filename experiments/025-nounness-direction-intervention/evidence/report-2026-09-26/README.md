# Experiment 025 — the production report (2026-09-27)

**The official run.** The production `report` was invoked **exactly once**, at commit
`c8d9fad08ab7181460cdd58bee6075668796fd40`, from the confirm-final state `1d5d37f0…`.
- The guarded launcher `launch_report.py --report-once` (pid 3546) called it at 11:13:34.742 UTC. It returned at
  11:13:42.044 UTC with **exit status 0** and **no incident**.
- It rendered the completed confirmation state, and nothing else.
- A check-only run came first (`--check`, pid 3447, 11:13:17 UTC). It did not invoke report.

| | |
|---|---|
| report | `outputs/experiment-025/report.md`, 7,868 bytes, sha256 **`ad9f85e62e74bf10b22060dc6f49ab75ddf3e79b7c81814d29ac6aa19fb75407`**; preserved byte-identically as [`../final-report-2026-09-26.md`](../final-report-2026-09-26.md) |
| final results state | **`534dad3907d94fa78c3a88a9e74bd88051b6ada57a09daff8007d2a3495bcdb8`**, file `fef818dced09bbf4faae43e68bc7da8c97bfeaed5190637099132968f4408134` |
| confirm-final state (report input) | `1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c`, file `c827bb69292f11ff6baf2194b4d8e34ccb0fbad9058c76d1abf2a3581724efb4`; removing only the two report entries from the final state reproduces both byte for byte |
| fresh prompts and measurements | **0**. The ledger (90,720 keys), the confirmation block and the measurements (`87284bdd…`) are unchanged. |

**The result the report states** is the reviewed confirmation result, unchanged: **`CAUSAL_EFFECT_NOT_ESTABLISHED`**,
with A = 26/40, B = 23/40, G = 37/40, threshold = 27, reference tail = 0.01923865414210013.

## The files

- **`launch_report.py`** — the launcher, adapted from 024's.
  - Before anything else it creates an exclusive sentinel per mode (`CHECK_LAUNCHED`, `REPORT_LAUNCHED`), so each
    mode runs once. It then changes to the repository root and asserts it.
  - Its audit hook only records: repository writes outside `outputs/experiment-025/`, the writes inside it, and any
    open of 022's local table or of 023's or 024's local outputs.
  - **Nothing that could run the model may be called**, and each attempt is counted:
    - the runner's model loader and tokenizer loader refuse;
    - the prompt, capture, intervention, measurement, gate and ledger entry points refuse: `pm.capture_prompt`,
      `run_capture`, `run_patched`, `run_interventions` and `record_execution`; `ul.stage_two_022` and
      `measure_prompt`; `capture.run_capture`; `interventions.run_interventions`; and `cr.measure_rotated`,
      `stage_two`, `patch_path_check`, `identity_gates`, `level1_gates`, `run_gates` and `recompute_c`;
    - `torch.nn.Module.__call__` refuses.
  - **The read-only pre-report check** comes next. A mismatch would have stopped before report. It checks:
    - HEAD, origin and the remote are `c8d9fad`, and the tree is clean;
    - the state is `1d5d37f0…` / `c827bb69…`, the measurements `87284bdd…`, and the installed freeze, lock and
      preregistration hashes are unchanged;
    - lock and confirm are complete and report not started, with no incident;
    - there is no `report.md`, the phase rule allows report, the ledger holds 90,720 keys, and the outputs are the
      four post-confirm files.
- **`CHECK_LAUNCHED`, `check.out`, `check_record.json`** — the check-only run: every condition held, and report was not
  invoked.
- **`REPORT_LAUNCHED`, `report.out`, `report.exit`, `report_record.json`** — the official run.
  - **The counts** (every counter is an attempt counter behind a refusal):

    | counter | value |
    |---|---|
    | report invocations | 1 |
    | model loads | **0** |
    | tokenizer loads | **0** |
    | prompt, capture, intervention, measurement, gate and ledger entry points | **0** |
    | module calls | **0** |
  - **The writes:** `report.md` once and one atomic state write (a temporary file, then a rename to `results.json`).
    There was no other repository write, no read of 022's table or 023's or 024's outputs, no relative open, and 0
    hook errors.
- **`verify_report.py`, `.out`** — the read-only post-report verification; all 15 checks passed.
  - `report.md` equals `cr.render_report(the confirm-final state)` byte for byte, and its sha256 is the state's.
  - All three state phases are complete, with no incident.
  - The state differs from the confirm-final state only in `report` and `phases.report`. Reverting them reproduces
    `1d5d37f0…` / `c827bb69…`, and equals the launcher's byte copy.
  - The measurements, the 90,720-key ledger, and the installed freeze, lock and preregistration are unchanged.
  - The report states the frozen result exactly: the label and its reading, the A, B and G rows (26, 23 and 37 of
    40, threshold 27), and the tail `5288280983/274877906944 = 0.01923865414210013`. It contains neither "no causal
    effect" nor "near miss", and its secondary section is headed "Descriptive records (no outcome force)".

**The pre-report state copy.** For the reversal check, the launcher made a byte copy of the confirm-final state in the
session's scratch directory (sha256 `c827bb69…`). It is not archived here: archiving the state is not the 022–024
convention. The [confirmation-record extract](../confirmation-record-2026-09-26.json) and the reversal preserve it.

**The report's header line** lists the phases of its input state, `report not_started`. The frozen renderer renders
the confirm-final state before the report entry is written; 024's report shows the same. The final state records the
report phase as complete.

**What in the report is descriptive.** The adjective and noun columns, the per-cue table, and everything under
"Descriptive records (no outcome force)" are descriptive only. That includes the magnitudes, the counts, the per-template
counts, the even components, the strata, the causal fraction and the ladder. G = 37/40 is a recorded PASS of that
statistic, but under the frozen hierarchy it cannot change the label. None of these can rescue, alter or redefine the
result. The report is preserved unchanged; the renderer, the lock and the preregistration are not edited.

`SHA256SUMS` lists every archived file in this directory. These are the files as written and run in the session's
scratch directory; none was edited.

# Experiment 024 — the production report (2026-09-25)

**The official run.** The production `report` was invoked **exactly once**, at commit
`b80409b1464c8ab59f82d026e104f4510247c69e`.
- The guarded launcher `launch_report.py --report-once` (pid 3700) called it at 20:27:23.911 UTC. It returned at
  20:27:30.987 UTC with **exit status 0** and **no incident**.
- It rendered the completed confirmation state and the installed calibration record, and nothing else.
- A check-only run came first (`--check`, pid 3628, 20:27:12). It did not invoke report.

| | |
|---|---|
| report | `outputs/experiment-024/report.md`, 8,310 bytes, sha256 **`8234ed9b41533c5aa90172cfb1bc0b70c656848eaeff234e9dcfc9a5d09a8193`**; preserved byte-identically as [`../final-report-2026-09-25.md`](../final-report-2026-09-25.md) |
| final results state | **`20ac6095a7e8534ff933a6405ae74b5f707aafd2ec9ac42e0ef5ffb1c5267376`**, file `edbb6dc9c198265202cb567aef7a4d44761c19d658f277e687995759e454ea0c` |
| confirm-final state (report input) | `076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92`, file `e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a`; removing only the two report entries from the final state reproduces both byte for byte |
| fresh prompts and measurements | **0**. The ledger (4,320 keys), the confirmation block and the measurements (`b21babe1…`) are unchanged. |

## The files
- **`launch_report.py`** — the launcher. Before anything else it creates an exclusive sentinel per mode
  (`CHECK_LAUNCHED`, `REPORT_LAUNCHED`), so each mode runs once. Its audit hook only records.
  - **Nothing that could run the model may be called**, and each attempt is counted:
    - the runner's model loader and tokenizer loader refuse;
    - the prompt, capture, intervention, measurement and ledger entry points refuse: `capture_prompt`, `run_capture`,
      `run_patched`, `run_interventions`, `record_execution`, `measure_prompt`, `stage_two_022` and `one_022`;
    - `torch.nn.Module.__call__` refuses.
  - **The read-only pre-report check** comes next. A mismatch would have stopped before report. It checks:
    - HEAD, origin and the remote are `b80409b`, and the tree is clean;
    - the state is `076ab9f9…` / `e636210c…`, the measurements `b21babe1…`, and the lock, preregistration and
      calibration hashes are unchanged;
    - confirm is complete and report not started, with no incident;
    - there is no `report.md`, the phase rule allows report, and the ledger holds 4,320 keys.
- **`CHECK_LAUNCHED`, `check.out`, `check_record.json`** — the check-only run: every condition held, and report was not
  invoked.
- **`REPORT_LAUNCHED`, `report.out`, `report.exit`, `report_record.json`** — the official run.
  - **The counts** (every counter is an attempt counter behind a refusal):

    | counter | value |
    |---|---|
    | report invocations | 1 |
    | model loads | **0** |
    | tokenizer loads | **0** |
    | prompt, capture, measurement and ledger entry points | **0** |
    | module calls | **0** |
  - **The writes:** `report.md` once and one atomic state write. There was no other repository write, no read of
    022's table or 023's outputs, and 0 hook errors.
- **`verify_report.py`, `.out`** — the read-only post-report verification; every check passed.
  - `report.md` equals `rr.render_report(the confirm-final state, the installed calibration record)` byte for byte.
  - The state records this report and the completed phase, and differs from the confirm-final state only in `report`
    and `phases.report`; reverting them reproduces `076ab9f9…` / `e636210c…`.
  - The ledger, the confirmation block and the measurements are unchanged.
  - The report states the frozen result exactly:
    - ρ = `0.6108818011257036`, against F_ρ `0.24411074612857814` and null₉₇.₅ `0.3136960600375234`;
    - the effective threshold `0.3136960600375234`, bound by the null;
    - K = 1 of 12870 (bound 321), exact p = 1/12870; D_EN = `0.5680373703974243`;
    - **`NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`**.

**The pre-report state copy.** For the reversal check, the launcher made a byte copy of the confirm-final state in the
session's scratch directory (sha256 `e636210c…`). It is not archived here: archiving the state is not the 022/023
convention. The [confirmation-record extract](../confirmation-record-2026-09-25.json) and the reversal preserve it.

**The report's PASS wording.** The frozen renderer describes a primary PASS as "at least as strongly as the exposed-like
relationship and beyond chance". For this realized calibration that prose is broader than the formal rule.
- The preregistered decision rule was `ρ ≥ max(F_ρ, null₉₇.₅)`, with F_ρ = `0.24411074612857814` and
  null₉₇.₅ = `0.3136960600375234`. The chance-level null was the binding threshold.
- The prospective result passed the locked primary threshold.
- The report is preserved unchanged for provenance. The renderer, the lock and the preregistration are not edited.

`SHA256SUMS` lists every archived file in this directory. These are the files as written and run in the session's
scratch directory; none was edited.

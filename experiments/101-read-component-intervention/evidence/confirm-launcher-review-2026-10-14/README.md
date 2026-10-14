# Experiment 101 — the independent review of the confirm launcher (2026-10-14)

**An AI review (independent, read-only), not a human validation.**
- **Version 1** (`launch_confirm_v1_as_reviewed.py`, `4f132e8e…`): **READY WITH CHANGES** (`REVIEW.md`) — one should-fix,
  S1: the SIGHUP handler would have turned a closed terminal under `nohup` into an incident after the ledger; notes N1–N9.
- **Version 2** (`launch_confirm.py`, `56a687f3…`, which ran): **READY FOR PRODUCTION** (`REVIEW-v2-diffcheck.md`) —
  S1, N1, N3, N7, N8 and N9 resolved, no new defect; the remaining notes R1–R4 need no change (output went to a regular
  file, as R1 advises).

**Not copied** (scratch): `clone/` (a 0.5 GB scratch clone of the repository), `clone_launcher/` (a launcher copy with
another ROOT), the scenario repositories `main_block/` and `v2/main_block/`, the mirrors and the temporary directories.
**About the stub label:** the scenario tests ran the launcher's main block against a *stub runner* that returns a fixed
label, `READ_COMPONENT_EFFECTS_AS_PREDICTED`, as test data (`scripts/e_main_block_scenario.py`, `v2/e2_main_block_scenario.py`,
summarized in `out/e_main_block.json` and `out/v2_e2_main_block.json`). Those records predate the real confirm (they
are from 13:00–14:05 UTC; confirm ran at 17:37–17:42 UTC) and carry no result.

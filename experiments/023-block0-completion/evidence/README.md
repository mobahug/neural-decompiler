# Experiment 023 — evidence

The run records and independent reviews of the phases run so far. The scripts ran from the session's scratchpad, and
the paths inside them refer to it. They are kept here exactly as written and run, not adapted to this location. Every
script is read-only with respect to the repository.

| folder | what | outcome |
|---|---|---|
| `extract-2026-09-24/` | preflight, the one `extract` run's log, the read-only verification of the candidate, the pre-install check, the post-install checks at the artifact commit | extract ran once at `2111271`, exit 0; E1–E6 pass (E4 max 0.0); artifact installed byte-identically in `83d9c58` |
| `extraction-review-2026-09-24/` | the independent extraction review: its own checker (items 1–8) and logs; `REVIEW.md` | PASS WITH NOTES, no blockers |
| `freeze-2026-09-24/` | pre-freeze check, the tier-C freeze contract, the launcher around the stock `freeze`, its log and file-open log, the read-only verification | freeze ran once at `89536b0`, exit 0; the design's expected picks; committed byte-identically in `a9287ec` |
| `freeze-review-2026-09-24/` | the independent freeze review: clean-room reconstruction (items 1–7) and logs; `REVIEW.md` | PASS WITH NOTES, no blockers |
| `calibrate-2026-09-24/` | pre-calibrate check, the launcher around the stock `calibrate` (022's table and the weight blob blocked; model load and module calls refused), its log and file-open log, the read-only verification; `FLOOR-REVIEW.md` | calibrate ran once at `4e5deeb`, exit 0; 0 undefined draws; floors 0.996897 / 0.998634 / 0.995053 / 0.995186; floor review PASS (marginal by design); the record is committed in `51b5c7d` |
| `archive/` | an exact, recoverable copy of Experiment 020's hash-pinned results state (`gzip -9 -n`) and its README | restores `da63b8c2…`; `13b8d39` |
| `lock-2026-09-24/` | pre-lock check, tier C at `13b8d39` (525 passed), the launcher around the stock `lock` (one weight load, every module call refused after it, 022's table blocked), its log and file-open log, the read-only verification, the before and after snapshots, and the post-install checks (confirm's own `validate_lock`, read-only) | lock ran once at `13b8d39`, exit 0; I5 0, algebra 4.9e-15; installed byte-identically in `f2294ab` |
| `lock-review-2026-09-24/` | the independent lock review: the full byte-identical Y1 regeneration, the lock rebuilt from scratch, I5 and the closed-form T reproduced, and logs; `REVIEW.md`, including the restated no-weights evidence for the freeze | PASS WITH NOTES, no blockers |
| `confirm-2026-09-24/` | pre-confirm check (confirm's own pre-prompt validation, run read-only), tier C at `c7efec7` (525 passed), the accounting launcher around the stock `confirm`, its log, `prompts.jsonl` (every forward call with its key and time) and counts, the read-only verification, the before and after snapshots | confirm ran once at `c7efec7`, exit 0; 3,060 = 3,060 = 3,060; Y1/cue_final ENVELOPE_ONLY_FAILURE; the other three PASS |
| `confirmation-review-2026-09-24/` | the independent confirmation review: accounting, ordering, Y2 rebuild, exact-value results, identities, isolation, the `" shiny"` check, state consistency; `REVIEW.md` | PASS WITH NOTES, no blockers; outcome reproduced |
| `report-2026-09-24/` | the launcher around the stock `report` (no model, 022's table blocked), its log, the read-only verification (exact re-render from the confirmed state) | report rendered once, `9b7a8d7a…` |
| `y2-table.f64`, `y2-table.json`, `final-report-2026-09-24.md` | the closure data: the Y2 table and the report, byte-identical to the outputs (`5344f44`) | `4623adde…`, `e87091ef…`, `9b7a8d7a…` |
| `confirmation-record-2026-09-24.json`, `closure-2026-09-24/` | the confirmation-record extract of the gitignored final state (022's convention, plus the prompt accounting) and its builder | content `0dcc2c44…` |

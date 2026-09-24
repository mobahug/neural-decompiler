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

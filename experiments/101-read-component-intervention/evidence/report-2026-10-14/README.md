# Experiment 101 — the report (2026-10-14)

`report` ran under a guarded launcher (`launch_report.py`, sha256 `e9e176a5…`; no forward pass, every capture,
intervention and measurement function refused, repository writes only `outputs/experiment-101/report.md` and the
results state).
- **Check** (`check_run_*.json`, `check.log`): G6 at the report computed read-only — the table recomputed from the
  committed stage-1 states and the weights equals the committed table bit for bit; nothing written.
- **Production** (`production_run_*.json`, `production.log`, `production.exit`): exit 0; it rendered `report.md`
  (sha256 `54f883adfea90c4ddef306ed7265ce65a82d9b74aca4dde934b64e0ad9d53e2a`, copied byte-identically as
  `../final-report-2026-10-14.md`) and moved the state from `1e3eaeb3…` to `c803fe97…` (the report block only).

# Experiment 101 — the independent calibration review (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent verified the calibrate phase
from the brief `PROMPT.md`.

**Verdict: PASS WITH NOTES — no blocker** (`REVIEW.md`, sha256 `851a856d54976f7836c642cb1585b87eee3b60d011cb2d64be8e1b980a2db746`).
- Its own code, which imports no `neural_decompiler` code and ran first, reproduces the 2,016 keys and their digest, the
  run log's order, every measurement digest and the state digest, every landing and bitwise gate, every tolerance
  gate's maximum and location, the exact-chain Level-1 values (within 7.4e-12), the head deltas (within 1.9e-15), every
  measured quantity, τ and the four triggers.
- The frozen functions, run afterwards in a fresh process under a guard, regenerate the prediction table's digest, the
  full gate record and the candidate file's `content_sha256` bit for bit; `ri.load_calibration_101` accepts the file.
- It ran no forward pass at all (no 101 prompt, no candidate text, no exposed prompt), and changed nothing in the
  repository.

## The findings and what was done

| finding | disposition |
|---|---|
| SF1 — preserve the calibrate evidence in the repository; correct the draft README (placeholders; files not yet existing; "21 checks" in the rehearsal, which recorded 20) | done: `../calibrate-2026-10-14/` holds every record, with `SHA256SUMS`; the README is completed and corrected; `calibration-v1.json` was committed alone and every added file is on a non-scientific path |
| N1 — the Level-1 margin at confirm: about 0.3 % estimated exceedance (0.02–1 %; about 10 % at the 95th percentile of a heavy-tailed frame bootstrap) | recorded for the human's lock decision; no tolerance changes (rule 1) |
| N2 — G6 at calibration is a same-process check | recorded; the cross-process G6 runs at confirm against the stage-1 states, and calibration's environment check (drift 0.0 against 020's states) and the smoke covered cross-process determinism here |
| N3 — I4 is vacuous on the 828 reference-prompt runs | as the plan states; recorded |
| N4 — `coordinated-adjective-1`'s keys ran three times in all | stated in the calibrate README in these terms |
| N5 — the run log records admissions, not completions | stated in the calibrate README |
| N6 — `collect_git_state()` refreshed `.git/index` during the preflight | recorded; harmless (no content change) |
| N7 — descriptive observations (A_amp saturation in 6/36; frozen-readout ceiling errors up to 1.30 nats per noun; τ possibly tight) | recorded; none may revise the calibration |

## Files

| file | what |
|---|---|
| `PROMPT.md`, `REVIEW.md` | the brief and the report |
| `scripts/common.py` | the reviewer's write guard, digest and canonical-JSON functions |
| `scripts/a_keys_ledger.py` → `out/a_keys_ledger.json`, `out/forbidden_101_sequences.json`, `out/exposed_frames.json` | keys, ledger, run log, isolation, token sequences |
| `scripts/b_state_measurements.py` → `out/b_state_measurements.json`, `out/b_values.json` | state, measurements, landing, G2–G6, quantities, triggers |
| `scripts/c_exact_chain.py` → `out/c_exact_chain.json`, `out/c_per_run.json` | the reviewer's float64 layers 3–5: Level-1, `E₁`, ceilings, G5, head deltas |
| `scripts/d_frozen_crosscheck.py` → `out/d_frozen_crosscheck.json`, `out/d_per_run_frozen.json` | the frozen cross-check under the guard |
| `scripts/e_tau_risk_descriptives.py` → `out/e_tau_risk_descriptives.json` | τ, the Level-1 exceedance estimates, the descriptives |
| `scripts/f_level1_decomposition.py` → `out/f_level1_decomposition.json` | what drives the Level-1 outliers |
| `scripts/g_state_inputs.py` → `out/g_state_inputs.json` | the state's inputs |
| `out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt` | the repository before and after the review |
| `SHA256SUMS` | this directory's digests |

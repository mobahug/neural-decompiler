# Experiment 101 — the independent stage-1 review (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent reviewed the production
stage 1 from the brief `PROMPT.md`.

**Verdict: PASS WITH NOTES — no blocker, no should-fix; the three stage-1 files can be committed as they are**
(`REVIEW.md`, sha256 `5f9cb37ee4f52f760d0574cc341348bb1d23bff00b16e1a16cc6881a1753422a`).
- The launcher's evidence (45/45 checks), the state (39/39) and the files (28/28) hold: one invocation, the 36 S1 keys
  once each through `ri.capture_reference`, nothing refused, the ledger exactly the 36 S1 keys, the record's digest
  `d2f1ed03…` reproducing, an mtime sweep showing only the three files, `stage1/` and the state changed.
- **The barrier passes on these bytes:** the runner's own `_barrier`, `_committed_stage1` and `_recompute_table`, run
  read-only. The table recomputes bit for bit at 1, 4 and 8 threads; an independent recomputation (the readout rebuilt
  from the raw checkpoint) agrees to ≤ 1.6e-13.
- **The predictions** (committed before any target result, so describing them leaks nothing), for `two`: Δĉ_full −6.64
  to −3.99 (median −5.00); F̂_rem median −0.01, F̂_ins 0.99; N̂ > 0 and Ŝ > 0 in 36/36; A_amp < 0 in 30/36; A_neg > 0
  in 36/36; no predicted-weak frame for any cue.
- It ran no forward pass and no 101 prompt; its guards (480 token sequences, refused module calls) never fired.

**What was done with its notes.** N1 (commit the three files alone, byte-identical; never edit the state; never rewrite
history) is in the user's commands. N2 (freeze the environment including macOS; the confirm launcher should pin the
platform), N3 (the confirm launcher must refuse a stage-1 incident), N4 (the confirm token names the table's content
sha256) and N5 (dry-run the barrier in the confirm rehearsal) are built into the confirm launcher. N6: the stale README
is updated in this commit; no file is added at a scientific path. N7: the stub artifacts it found in the confirm-launcher
review's scratch directory (six stub `results.json` with a fabricated label, used to test the launcher; three
zero-tensor stubs) are not results and are not copied into the repository; `out/g_leakage.*` lists them by path.

**A deviation from the brief, recorded.** Its leakage sweep scanned two other experiments' git-ignored outputs
(`outputs/experiment-022/calibration-table.pt`, `outputs/experiment-025/stage2-measurements.pt`) for 101 key strings,
read-only, with 0 hits; the brief allowed only 020's results under `outputs/`.

**Not copied:** `tmp/` (empty scratch).

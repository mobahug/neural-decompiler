# Experiment 101 — the independent lock review (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent reviewed the production lock
from the brief `PROMPT.md`.

**Verdict: PASS WITH NOTES — no blocker to committing the lock** (`REVIEW.md`, sha256
`3be162560022d1fb097231b8bc0f3398dcf7494645cdefa782ba38905f39b787`).
- Its own code (no `neural_decompiler` import) rederives every field of the candidate lock from the committed files,
  the raw checkpoint and the documents (81 of 82 checks; the open one is N6). The frozen code then rebuilds
  `candidate-lock.json` and renders `candidate-preregistration.md` byte for byte, reproduces the geometry block bit for
  bit twice and the exposed table's digest `ce055e09…`.
- `ri.validate_lock` accepts the candidate as it would be installed and refuses nine negative controls (a scientific-path
  change, a runner change, a dirty tree, untracked files, a non-ancestor lock commit, an edited preregistration, a
  widened tolerance with and without a matching state, a changed noun order).
- The lock phase ran once and changed only its own block: undoing the state's `lock` block and `phases.lock` gives back
  the calibrated state bit for bit.
- It ran no forward pass, and its guards (an audit hook, the 240 forbidden token sequences, a refusing
  `Module.__call__`, 17 wrapped entry points, a sealed model) never fired.

## The findings and what was done

| finding | disposition |
|---|---|
| S1–S3 — the prepared stage-1 launcher, version 1 (`bf40e043…`): the rehearsal did not exercise the write path under the hook; the hook failed open for relative paths in non-`open` events (`dir_fd = −1` read as a descriptor); the sentinel preceded the preflight and records were overwritten | version 2 (`../lock-2026-10-14/prepared/launch_stage1.py`, `13b1992f…`) fails closed, self-tests its decisions on the recorded event tuples in every mode (`hook_self_test.py/.out.json`: 9 allowed, 10 refused, all as expected, in an isolated test), creates the sentinel after the preflight, never overwrites a record, requires `-B` and asserts the ledger and the tree after the run. The reviewer diff-checked version 2 (not a full review); a re-check before stage 1 is recommended. Version 1 is kept beside it. The same fail-open defect is in the calibrate and lock launchers; it had no effect there (every runner path is absolute, and the reviews' scans found no repository change beyond the outputs). |
| N1 — the Level-1 tail risk at confirm (~0.3 %, up to ~10 %) | unchanged; no amendment is possible inside this protocol version; the user's decision (`../lock-2026-10-14/DECISION.md`) |
| N2 — confirm does not record design §2's per-frame `s/‖off_c‖` and overshoot flag | they are recomputable from the saved `x3_pc` rows of REF and `PL@w`; the confirmation review or the report should produce them |
| N3 — preregistration wording (the third non-claim narrower than the scope sentence; a few implemented rules not restated; the bit-for-bit gates not listed; the head counts; Python bool formatting) | recorded; bound by the module blob, so a change needs a new protocol version; a later claim review should state which population it reads |
| N4 — do not rewrite `exp101-phases` (stage 1 and confirm require `3621eec` as an ancestor) | recorded in `DECISION.md`; push the branch as it is |
| N5 — copy the lock evidence into the repository now | done in this commit, checked against the review's digests (9/9 equal) |
| N6 — `inputs.table_022_file` not verifiable within the review's read limits | recorded; it binds nothing 101 computes |
| N7 — the lock ran before the user's decision on N1 | harmless, as the review says: an amendment needs a new protocol version either way; the user's decision is the commit |
| N8 — minor stage-1 launcher points | applied in version 2 |
| N9 — an environment change would turn stage 1's bit-for-bit geometry check into a phase incident | in `DECISION.md`: keep the machine's OS, Python and packages fixed until confirm |
| N10 — the runner's plain `git status` touches `.git` | harmless; recorded |
| N11 — three wording points in `DECISION.md` | fixed |
| Process — the operator prepared the version-2 launcher and the memo's fixes from the review's draft before the hand-back | nothing was committed before the hand-back; this commit follows it |

## Files

`PROMPT.md`, `REVIEW.md`, the reviewer's `SHA256SUMS` (as it left the directory), `scripts/` and `out/` (`out/audit_event_probe/` holds the reviewer's few-byte probe files, written to record Python's audit events for the stage-1 operations in its own scratch directory).
**Not copied:** `out/stage1_writer_test/states.f64` and `states.json` (5.2 MB), a synthetic table the reviewer wrote to
test `ul.write_table` against the lock's layout: `torch.randn` values, seed 101, under the 101 frame ids (no model
output; `scripts/c_frozen.py:298–313`). They are left out so that no file in the repository can be mistaken for stage-1
states; their digests are in the reviewer's `SHA256SUMS` (`c30e604c…`, `f65e3009…`). `SHA256SUMS.repository` lists
what is here.

# Experiment 101 — the independent re-check of the stage-1 launcher (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent re-checked the guarded
stage-1 launcher version 2 (`launch_stage1.py`, sha256 `13b1992f…`) before its one production run, from the brief
`PROMPT.md`.

**Verdict: READY FOR PRODUCTION — no blocker, no should-fix** (`REVIEW.md`).
- The lock review's S1–S3 and N8 are fixed, with no regression: the forward guards, the sealing, the counted load,
  `write_check` and the preflight are textually version 1's (and the calibrate launcher's, which ran in production).
- The audit hook refuses nothing a correct stage 1 does: 70/70 decisions as expected; the real hook over real library
  calls on a scratch mirror; the model load, two forwards on a neutral sentence ("The quick brown fox…", not a 101
  text) and the weights-only computations judged live with 0 refusals.
- The preflight passed 16/16 in the reviewer's own rehearsal from a copy (`rehearsal/`).
- The production path (the launcher's main block against a stub runner, 12 scenarios) behaved as specified.

**What was done with its notes.** N1 (a stage-1 incident after "complete" is invisible to confirm's phase rule) and N2
(launch detached) were applied: stage 1 was launched with `nohup`, the operator checked `phases.stage1` for incidents
after the run (none), and the confirm launcher refuses unless stage 1 is complete with no incident. N3 (nothing else may
write in the repository during the run) was kept. N4 (the environment, the tokens) was followed.

**Not copied** (scratch, not evidence): `main_block/` (the stub-runner scenario repositories), `mirror*/` (scratch
mirrors, which hold FIFOs), `tmpdir*/` and `work_*/`.

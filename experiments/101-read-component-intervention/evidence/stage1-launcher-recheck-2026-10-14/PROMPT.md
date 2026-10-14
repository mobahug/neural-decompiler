You are the independent, adversarial, read-only re-checker of the guarded launcher for Experiment 101's `stage1` phase,
in the research repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability
project on Pythia-70M-deduped). You are an AI reviewer; your report is an AI review, not a human validation — say so.

## Context

- The lock is installed and committed by the user: commit `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` on branch
  `exp101-phases` (pushed), files `experiments/101-read-component-intervention/preregistration-lock.json` (sha256
  `8dc6e6d3…`, content `8da97e35…`) and `preregistration.md` (`3761b813…`).
- The independent lock review (`experiments/101-read-component-intervention/evidence/lock-review-2026-10-14/REVIEW.md`)
  read version 1 of the stage-1 launcher (`launch_stage1_v1_as_reviewed.py`, sha256 `bf40e043…`) and found three
  should-fix points (S1: the rehearsal did not exercise the write path under the hook; S2: the hook failed open for
  relative paths when `dir_fd = -1`; S3: the sentinel preceded the preflight and records were overwritten) and minor
  points (N8). Version 2 (`launch_stage1.py`, sha256 `13b1992f…`) is meant to fix them; the reviewer only diff-checked
  it. **Your job is the full re-check of version 2 before its one production run.**
- Files: the committed copies in `experiments/101-read-component-intervention/evidence/lock-2026-10-14/prepared/`
  (`launch_stage1.py`, `launch_stage1_v1_as_reviewed.py`, `hook_self_test.py`, `hook_self_test.out.json`), and the
  operator's working copy and its rehearsal record in
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1/
  (`launch_stage1.py`, identical to the committed v2; `rehearsal.log`; `rehearsal_run_20261014T120823753435Z.json`).
- Specifications: plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md`, § Phases
  (`stage1`), § The prediction freeze, § Incidents, stops and refusals, § One-shot protections and guarded launchers;
  the runner `experiments/101-read-component-intervention/run.py` (`stage1`, `_validated_lock`, `_state_for`,
  `write_check`), and `src/neural_decompiler/read_intervention.py` (`capture_reference`, `run_arm`, `stage1_binding`,
  `prediction_table`), `upstream_localization.py` (`write_table`).

## Hard rules

- **Read-only in the repository.** Never create, edit, move, delete, stage or commit anything in the repository,
  including `outputs/`. Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1-launcher-recheck/
  (scripts, logs, `REVIEW.md`).
- **Never run the launcher in production mode, never run the runner's `stage1`, `confirm` or `report`, and never run
  any prompt of the 60 candidate texts (the 36 frozen 101 frames) with any cue.** You MAY run the launcher in
  `STAGE1101_MODE=rehearsal` from a COPY in your own directory (it loads no model and writes only its record beside
  itself), and you may test its hook logic in isolation (extract functions with `ast`, as `hook_self_test.py` does, or
  construct synthetic audit events). Any test that touches the filesystem must use your own scratch directory, never
  the repository.
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python -B` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`; `git --no-optional-locks` for every git call. Never `uv`.

## Check

1. **S1–S3 and N8 are fixed** in version 2, with no regression of what version 1 got right (the sanctioned
   `ri.capture_reference` for exactly the 36 S1 keys after the ledger is on disk; `ri.run_arm` refused; low-level entry
   points and `Module.__call__` refused outside a sanctioned call; sealing; one model load; the allow-list).
2. **The hook fails closed** for every write-type event whose path cannot be resolved, resolves `dir_fd` correctly
   (`-1`/None = the working directory; ≥ 0 = the descriptor's directory), and allows exactly the events a correct stage 1
   raises (Python 3.12 audit signatures; the lock review recorded them in
   `evidence/lock-review-2026-10-14/out/f_audit_events.json`). Think adversarially: list any event a correct run
   raises that the hook would refuse (that would be an incident after the S1 ledger, spending the frames), and any
   repository write it would allow besides the three stage-1 files and the results state.
3. **The preflight** refuses (with nothing recorded and no sentinel) on every refusal-class problem, and passes in the
   real state (the operator's rehearsal record; re-run the rehearsal from your copy if you want).
4. **The production path, by reading the code:** the sentinel after the preflight; `runner.stage1()` once; the runner's
   `write_check` is the launcher's predicate; the post-run checks (the 36 keys once, the ledger exactly the 36 S1 keys,
   the tree exactly the three untracked stage-1 files, the record's digest, the files' sha256s against the record); the
   record written with O_EXCL; the exit code. Check what happens if `stage1()` raises after the ledger (the runner
   records a phase incident; does the launcher record it faithfully and exit non-zero?).
5. **Anything else** that would let a target (S2) prompt run, a file other than the three be written, the phase run
   twice, or a correct run fail.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation". Verdict: READY FOR PRODUCTION, READY WITH
  CHANGES, or NOT READY. Findings most severe first (blocker / should-fix / note) with file:line and a concrete
  scenario. Do not hand back until REVIEW.md is final; then reply with the verdict and the findings.

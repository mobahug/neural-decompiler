You are the independent, adversarial, read-only reviewer of the guarded launcher for Experiment 101's `confirm` phase
— the one-shot, outcome-bearing run — in the research repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler
(a mechanistic-interpretability project on Pythia-70M-deduped). You are an AI reviewer; your report is an AI review,
not a human validation — say so.

## Context

- The lock is committed by the user (`961e814`, branch `exp101-phases`). Stage 1 ran once (exit 0) and wrote the three
  stage-1 files `experiments/101-read-component-intervention/stage1/{states.f64,states.json,table.json}`, which are
  **not committed yet** (the user commits them; a separate AI review of stage 1 is running). Confirm can run only after
  that commit.
- The launcher under review: /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm/launch_confirm.py
  with its isolated hook test (`hook_self_test.py`, `hook_self_test.out.json`) beside it. It reuses the calibrate
  launcher's forward guards, sealing and counted load (`.../exp101/calibrate/launch_calibrate.py`, which ran in
  production) and the stage-1 launcher version 2's fail-closed audit hook (`.../exp101/stage1/launch_stage1.py`, found
  READY by an independent re-check: `.../exp101/stage1-launcher-recheck/REVIEW.md`, whose N1, N2 and N6 the confirm
  launcher is meant to address).
- Specifications: plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md` (§ Phases
  `confirm`, § The prediction freeze, § Gates, as implemented, § Incidents, stops and refusals, § One-shot protections
  and guarded launchers); the runner `experiments/101-read-component-intervention/run.py` (`confirm`, `_barrier`,
  `_committed_stage1`, `_record_incident`, `_secondary`); `src/neural_decompiler/read_intervention.py` (`run_arm`,
  `stage_runs`, `measure_frame`, `run_gates`, `primary_statistics`, `assert_phase_allowed`); 025's confirm launcher
  `experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/launch_confirm_v3.py` as the precedent.

## Hard rules

- **Read-only in the repository.** Never create, edit, move, delete, stage or commit anything in the repository
  (including `outputs/` and the untracked stage-1 files). Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm-launcher-review/
- **Never run the launcher in any mode against the real repository, never run the runner's `confirm`, `stage1` or
  `report`, and never run any 101 prompt** (the 60 candidate texts with any cue). Isolated tests are allowed: extract
  functions with `ast`, stub the runner, or run code against a scratch git repository in your own directory. If you load
  the model at all, use a neutral non-101 sentence only, with a token-sequence guard installed.
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python -B` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`; `git --no-optional-locks`. Never `uv`.

## Check

1. **No target prompt can run outside the protocol:** a forward only inside the sanctioned `ri.run_arm`, only for one of
   the 1,980 S2 keys, each once, only after the S2 ledger is on disk with the barrier passed; `ri.capture_reference`
   refused (confirm never captures a reference); every low-level entry point, `Module.__call__`, instance and class
   `forward` sealed; one model load. Confirm the runner's own order (barrier → ledger → runs) and that the launcher's
   ledger check cannot be satisfied before the runner's barrier passes.
2. **The audit hook** allows exactly the writes a correct confirm makes (the results state through its temporary files
   and renames; `stage2-measurements.pt`; `stage2-partial.pt` on an incident; the outputs mkdir) and refuses everything
   else in the repository, failing closed. Enumerate the Python 3.12 audit events of `rr.write_state_atomic`,
   `rr.save_durably`, `torch.save`, `torch.load`, the model load (including `filelock`'s probe) and the secondary step,
   and say whether each is decided correctly. Any wrong refusal after the S2 ledger would be an incident that spends
   the confirmation — look hard for one.
3. **The preflight** (all refusal-class, before the sentinel): `-B` and `HF_HUB_OFFLINE`; N1/N2; the lock; the module
   binding; the stage-1 files committed and equal to the record; `phases.stage1` complete, `awaiting_commit`, **no
   incidents** (the stage-1 re-check's N1); the ledger exactly the 36 S1 keys; confirm not started; `_state_for` and
   `_validated_lock` with the stage-1 files; the runtime; the S2 keys; the hook self-test; the two production tokens
   (the lock's content sha256, the stage-1 table's content sha256).
4. **Production behaviour:** the sentinel after the preflight; SIGTERM/SIGHUP converted to `KeyboardInterrupt` (does
   the runner then record an incident after the ledger, as it does for SIGINT?); `runner.confirm()` once; the post
   block; the record written with O_EXCL; the exit code. Consider the interaction with `_secondary` (no forward) and
   the final state writes.
5. **The smoke mode:** runs only a spent exposed frame's 55 keys (020's pool), compares with the calibrate launcher's
   unguarded reference, and never builds or runs any 101 prompt, even as a refused probe.
6. **Anything else** that would let a target prompt run twice or outside the ledger, write a file other than the allowed
   ones, lose the record, or make a correct confirm fail.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation". Verdict: READY FOR PRODUCTION, READY WITH
  CHANGES, or NOT READY; findings most severe first (blocker / should-fix / note) with file:line and a concrete scenario.
- Do not hand back until REVIEW.md is final; then reply with the verdict and the findings.

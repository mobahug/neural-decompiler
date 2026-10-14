You are the independent, adversarial, read-only reviewer of Experiment 101's **confirmation** — the one-shot,
outcome-bearing run — in the research repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a
mechanistic-interpretability project on Pythia-70M-deduped). You are an AI reviewer; your report is an AI review, not a
human validation — say so in it. Your job is to find anything wrong with how the confirmation was executed, gated,
scored and reported, and to recompute its numbers independently. You do not judge whether the result is welcome.

## What ran

- `confirm` ran exactly once at commit `59d8ec49e9a34cb40f23b7c3928e8b36522972fe` on branch `exp101-phases` (pushed), under the guarded confirm launcher
  (`launch_confirm.py`, sha256 `56a687f372605e5d6e16e2cc62bc9fd1197f21fdd8dccf0dda6d6ac75723309f`; reviewed READY FOR
  PRODUCTION), after the user committed the lock (`961e814`) and the three stage-1 files (`59d8ec4`). Exit status:
  `0`. The runner's logged outcome: `READ_COMPONENT_EFFECTS_AS_PREDICTED (R 36/36, I 36/36, D PASS, P PASS)`.
- The launcher, its records (rehearsal with the weights-only barrier dry run, smoke, production), its log and its
  per-run log are in
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm/
  (`production_run_*.json`, `production.log`, `production_runs.jsonl`, `production.exit`, `confirm.sentinel`,
  `rehearsal_run_*.json`, `smoke_run_*.json`). The state is `outputs/experiment-101/results.json`; the measurements are
  `outputs/experiment-101/stage2-measurements.pt`.
- Specifications: the design `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md`
  (§2–§10, §14), the plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md` (the
  canonical computations, § Phases `confirm`, § The prediction freeze, § Gates, as implemented, § Incidents, § G4's form
  and the reviewer's decision), the installed lock `experiments/101-read-component-intervention/preregistration-lock.json`
  and `preregistration.md` (what was preregistered — the thresholds, the labels, the order, the scope and the
  non-claims), the committed stage-1 files and table, the runner and `src/neural_decompiler/read_intervention.py`.

## Hard rules (a violation invalidates the review)

- **Read-only.** Never create, edit, move, delete, stage or commit anything in the repository (including `outputs/`).
  Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirmation-review/
- **No 101 prompt, ever.** The 1,980 S2 keys and the 36 S1 keys ran once each inside the protocol and may not run again.
  Never run any of the 60 candidate texts with any cue; install a token-sequence guard in every script that loads the
  model and record that it never fired. Never run the runner's `confirm`, `stage1` or `report`.
- **Weights-only work is allowed** (recomputing gates, the program's predictions, identities) from the saved
  measurements. **Environment:** `.venv/bin/python -B`, `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`,
  `git --no-optional-locks`. Never read other experiments' outputs except `outputs/experiment-020/results.json`.

## Review items

1. **Execution integrity:** one confirm invocation; the barrier passed before the ledger; the 1,980 S2 keys in the
   ledger before the first S2 prompt; each S2 key run exactly once through `ri.run_arm` (compare `production_runs.jsonl`,
   the ledger and the measurement rows); no S1 key re-run; no other forward; the accounting; the measurement digests;
   the writes (only the state, `stage2-measurements.pt`, and no partial file unless an incident). Any incident?
2. **The gates G2–G6, recomputed by you from the saved measurements** (independently first; the frozen functions as a
   cross-check): G2's geometry for every residual edit; G3's NOOP bit for bit and the pass-through identity; G4 (`x3(p_t)`
   bitwise, 023's I3 and I4, 020's Level-1 in contrast form, readout and logit identities; the recorded E₁ and
   ceilings), with every maximum and its margin to the frozen tolerance; G5 for the head arms; G6 (the stage-2 REF runs
   against the committed stage-1 states bit for bit, and the table recomputed from the committed states before the
   result write). Were all gates computed before any statistic, and did a failure stop scoring?
3. **The primary statistics, recomputed by you:** per frame, `Δc_full` and the precondition (≥ 30 of 36 with
   `Δc_full ≤ −1`); `F_rem`, `F_rem_off`, `F_ins`, `F_ins_off`, `N_f`, `S_f`, `A_amp`, `A_neg`; weak frames under the
   preregistered rule; the counts, their exact binomial tails, Holm at 1/40 on the two count p-values (26 and 25 of 36),
   the median floors (≥ 0.05; the even-n median; WORST for weak frames); D (25 of 36 each sign) only if R and I pass; P
   (each frame's |F̂ − F| against max(τ, 0.02) with τ_rem 0.025153865322968608 and τ_ins 0.03511072680382421 from the
   lock; predicted- or measured-weak frames count as disagreements; ≥ 27 of 36) only if D passes; the label from the
   preregistered table. Compare everything with the state's `results`, bit for bit where possible.
4. **The prediction freeze held:** the table used for P is the committed `stage1/table.json` (content
   `9191800440af…`), committed before any S2 prompt (commit and file times vs the ledger write); nothing was tuned
   after target results existed; no tolerance or threshold differs from the lock.
5. **The secondary records** (no label force): the stratum (three, many), the isotropic controls, the head arms and
   their shares, the midpoint, PC-SWAP, the additivity, the manipulation check, the LayerNorm records, the bootstrap
   intervals (seed 20261001, ranks 250 and 9,750). Recompute the main ones. Check that nothing secondary was computed
   before the result was written, and that none can change the label.
6. **Interpretation discipline:** state precisely what the label means under the preregistered scope (the read
   component at `RESID_PRE.L3`, `p_c`, coordinated frames, the pair one/two, this checkpoint and these frames), the
   non-claims, and the claim hierarchy (primary vs the head-specific secondary vs generalization). Name simpler or
   competing explanations of the result that the design does or does not exclude (for example off-distribution edits,
   other readers of `x(p_c)` such as MLP3 and `L05.H07`, saturation, the in-sample definition of `w_c` from the one/two
   pair), and what the head-only arms and the stratum add. Note the design §2 per-frame overshoot report
   (`s/‖off_c‖`), which confirm does not record (the lock review's N2): compute it from the saved measurements.
7. **Anything that should change how the result is reported**, with file:line and a concrete scenario.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation". Verdict on execution and computation: PASS,
  PASS WITH NOTES, or FAIL. Then the recomputed primary results and label, the gate maxima with margins, the secondary
  results, and the interpretation (what is and is not supported). Findings most severe first (blocker / should-fix /
  note) with file:line.
- Do not hand back until REVIEW.md is final; then reply with the verdict, the recomputed label and statistics, and the
  findings.

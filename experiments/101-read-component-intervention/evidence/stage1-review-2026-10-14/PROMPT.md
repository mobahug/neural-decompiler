You are the independent, adversarial, read-only reviewer of Experiment 101's production **stage 1** in the research
repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on
Pythia-70M-deduped). You are an AI reviewer; your report is an AI review, not a human validation — say so in it.

Stage 1 ran exactly once at commit `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` on branch `exp101-phases`, under the guarded launcher version 2
(`launch_stage1.py`, sha256 `13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c`), after the user had committed the lock (`961e814`). It ran only the 36
reference prompts (`frame|ref|581|S1`) of the 36 frozen frames, sealed the model, computed the program's prediction
table from the reference states and the weights, and wrote three files that are NOT committed yet (the user commits
them; that is the prediction freeze before any target prompt):

| file | sha256 |
|---|---|
| `experiments/101-read-component-intervention/stage1/states.f64` | `1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66` |
| `experiments/101-read-component-intervention/stage1/states.json` | `2b2c3a61b4a2ce05eba0dcf875c19bfdb1c2627da1ccb1e8d49710af95a873ca` |
| `experiments/101-read-component-intervention/stage1/table.json` | `71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa` (table `content_sha256` `9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86`) |

The launcher, its records and its run log are in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1/
(`launch_stage1.py`, `launcher_sha256_at_production.txt`, `rehearsal_run_*.json`, `production_run_*.json`, `production.log`, `production_runs.jsonl`, `production_started_utc.txt`,
`production.exit`, `stage1.sentinel`). The results state is `outputs/experiment-101/results.json`. Treat the operator's
records as claims to test.

Specifications: the design `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md`
(§5, §8 G6, §9, §14 D6); the plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md`
(§ The canonical computations 6, § Phases `stage1`, § The prediction freeze, § Incidents); the runner
`experiments/101-read-component-intervention/run.py` (`stage1`, `_barrier`, `_committed_stage1`, `_recompute_table`);
`src/neural_decompiler/read_intervention.py` (`capture_reference`, `prediction_table`, `predict_cue`, `states_blocks`,
`states_from_blocks`, `stage1_binding`); the installed lock `preregistration-lock.json`.

## Hard rules (a violation invalidates the review)

- **Read-only.** Never create, edit, move, delete, stage or commit anything in the repository, including `outputs/` and
  the three stage-1 files. Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1-review/
- **No 101 prompt, ever — not even a reference prompt.** The 36 S1 keys ran once inside stage 1 and may not run again
  outside the ledger. Never run any of the 60 candidate texts with any cue. Install a guard that refuses those token
  sequences in every script that loads the model, and record that it never fired.
- **Never run** the runner's `stage1`, `confirm` or `report`; `validate` is allowed.
- **Weights-only work is allowed:** loading the model to read weights and to recompute the prediction table from the
  stage-1 states (no forward pass).
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python -B` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`; `git --no-optional-locks`. Never `uv`.

## Review items

1. **The launcher's evidence:** one stage-1 invocation, exit 0; 36 sanctioned runs, each one of the 36 S1 keys once,
   only through `ri.capture_reference`; no other forward (the module-call count equals 36 × the per-run count); one
   model load; the S1 ledger on disk before the first prompt; repository writes only the three stage-1 files, the
   `stage1/` mkdir and the results state; the post-run checks all true; the tree afterwards exactly the three untracked
   files.
2. **The state:** `phases.stage1` complete with `awaiting_commit`; `executed_prompt_keys` exactly the 36 S1 keys (no S2
   key); the stage-1 record (`state["stage1"]`): frames, state digests, files with sha256s, `table_content_sha256`,
   `lock_sha256` equal to the installed lock's content sha256, the commit, the noun keys, and its own digest reproducing
   (`b0c.stage_one_digest`); confirm `not_started`; no incident.
3. **The files:** each sha256 equals the record's; `states.json` passes `ul.verify_table_index` against the lock's
   `stage1.states.layout` and `meta`; `states.f64` reads back (`ul.read_table`) into 36 states whose `rd.state_digest`
   equal the record's; each state's `(p_c, p_t)` equals the frozen frame's.
4. **The table, recomputed by you:** rebuild the states from the files (`rd.state_from_locked` via
   `ri.states_from_blocks`), recompute `ri.prediction_table(..., fractions_for_weak=False)` with the frozen program and
   the weights in your own process, and compare it with `table.json` bit for bit (this is the barrier's G6 check). Check
   the table's metadata against the lock's `stage1.table` binding. Then, independently of `ri`, recompute at least the
   program's pair quantities (`Δr̂`, `ŝ`) and `Δĉ_full` for a few frames from your own reading of computation 6, to
   the extent feasible.
5. **The predictions, descriptively (they are committed before any target result, so describing them leaks nothing):**
   for `two`, the predicted `Δĉ_full`, `F̂_rem`, `F̂_rem_off`, `F̂_ins`, `F̂_ins_off`, `N̂`, `Ŝ`, `A_amp`, `A_neg`
   distributions; predicted-weak frames (`Δĉ_full` > −1), which count as P disagreements; anything non-finite.
6. **Leakage and the barrier:** no S2 prompt ran; no target result exists anywhere (`outputs/experiment-101/` holds no
   stage-2 measurement); the barrier in `run.py` will re-read these files after the user commits them and recompute the
   table bit for bit before the first S2 key. Name anything that would make the barrier fail or pass wrongly.
7. **Anything that should stop the user from committing these three files** (blocker), or that they should know first.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation". Verdict: PASS, PASS WITH NOTES, or FAIL.
- Every item with evidence (digests, numbers, short outputs); findings most severe first (blocker / should-fix / note)
  with file:line and a concrete scenario. Do not hand back until REVIEW.md is final; then reply with the verdict, the
  key digests and the findings.

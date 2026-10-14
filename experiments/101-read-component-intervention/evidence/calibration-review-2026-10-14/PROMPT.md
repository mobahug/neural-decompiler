You are the independent, adversarial, read-only verifier of Experiment 101's `calibrate` phase in the research
repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on
Pythia-70M-deduped). You are an AI reviewer; your report is an AI review, not a human validation — say so in it.

`calibrate` ran exactly once under a guarded launcher at commit `8f65062627c77815adf9289c3fe60191d9fc3c17` on branch `exp101-phases`. It is exploratory
and uses spent (exposed) data only: the 36 exposed coordinated frames of Experiment 020's pool, their 36 reference
re-captures and 1,980 arm runs (2,016 keys). Its exit status was 0 (complete: no revision trigger fired; `outputs/experiment-101/candidate-calibration-v1.json` written, τ_rem 0.025153865322968608, τ_ins 0.03511072680382421). The launcher, its records and its
run log are in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/calibrate/
(`launch_calibrate.py`, `rehearsal_run.json`, `smoke_run.json`, `production_run.json`, `production.log`,
`production_runs.jsonl`, `smoke_reference.py`, `smoke_reference.pt`). The phase's outputs are in
`outputs/experiment-101/` (`results.json`, the results state; `calibration-measurements.pt`; and, if the phase
completed, `candidate-calibration-v1.json`).

Specifications:
- the design `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md` (revision 3,
  decided `516df20`), especially §2, §4–§8 and §14;
- the plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md` (revision 3, the G4
  decision `1e48c1c`), especially § Constants (the tolerance table), § The canonical computations, § Phases
  (`calibrate`), § Gates, as implemented, and § G4's form and the reviewer's decision (the two frozen rules: no silent
  tolerance tuning; G4 is an integrity gate, not an outcome filter);
- the module `src/neural_decompiler/read_intervention.py` (`ri`) and the runner
  `experiments/101-read-component-intervention/run.py`;
- the committed `experiments/101-read-component-intervention/confirmation-v1.json` and the freeze evidence under
  `experiments/101-read-component-intervention/evidence/`.

## Hard rules

- **Read-only in the repository.** Do not create, edit, move or delete anything in the repository (including
  `outputs/`), and do not stage or commit. Write your report and scratch scripts only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/calibration-review/
  The report is `REVIEW.md` there.
- **No 101 frame prompt, ever.** Never run any prompt of the 36 frozen 101 frames (any of the 60 candidate texts with
  any cue). Install a guard in every script that loads the model and refuses any prompt whose token sequence is one of
  those (the 60 candidate texts × the cues one/two/three/many) — record that the guard was installed and never fired.
- **Never run a runner phase** (`freeze`, `calibrate`, `lock`, `stage1`, `confirm`, `report`; `validate` is allowed).
- **The model:** you may load the pinned weights (`EleutherAI/pythia-70m-deduped` @
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, local files only) for weight-only recomputation (the read covector, the
  program's predictions with the frozen `ul`/`rd` functions, the gates from the saved measurements). Forward passes
  only on the spent exposed calibration prompts (020's pool), and only for determinism spot checks of at most 3 frames
  — prefer none.
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Never `uv`. Never read `outputs/` of other experiments except
  `outputs/experiment-020/results.json`.

## Verify independently (write your own code first; use `ri` functions only afterwards as a cross-check, and record the
order)

1. **The launcher's evidence:** one `calibrate` invocation; the exit status; one model load; the sanctioned runs equal
   the 2,016 calibration keys, each exactly once (compare `production_runs.jsonl` with the state's
   `calibration_executed` and with your own reconstruction of the keys from 020's pool); no refused call, no forbidden
   read or write; the only repository writes are the allowed outputs; the guards were installed before the runner was
   imported; the calibration ledger was on disk before the first prompt (the launcher refuses otherwise — check the
   code and the record). Check that no key's prompt is a 101 frame prompt (token sequences) and that every untagged key
   is in 020's exposed ledger and in no confirmation manifest.
2. **The rehearsal and smoke records:** the smoke ran one exposed frame through the production guards and compared its
   tensors and logits bit for bit with an unguarded process; check the claims against the records and the code. Note
   that the smoke executed one calibration frame's keys outside the phase (engineering, spent data): say whether that
   compromises anything.
3. **The state** (`outputs/experiment-101/results.json`): `state_sha256` recomputes; the phases; the `calibration`
   block (planned count and digest, measurements digests, accounting, gates, ceilings, stopped/complete fields);
   `executed_prompt_keys` is empty (no confirmation key ran); the inputs and versions.
4. **The measurements** (`calibration-measurements.pt`): the tensor digests equal the state's; every expected tensor
   exists and is finite; the REF rows reproduce 020's locked reference states within `rc.LOCKED_STATE_TOLERANCE`
   (the environment check), and G6's bitwise comparisons against the re-captures hold.
5. **The geometry and the edits, recomputed by you:** `w_c` from 011's lock (`read_weight`, centred) and from the
   weights (`γ₃ ⊙ (W_V W_O d̂_T)` of head 3.4), G1; per frame and cue the pair quantities (`Δr`, `s`, `‖off_c‖`,
   `s/‖off_c‖`, the overshoot flag); every residual edit's realized geometry (G2: `r(x′)` against its target, the
   centred length, parallel/orthogonal), from the saved replacements and plain captures.
6. **The gates' values** (G2–G6) as recorded in the state, recomputed by you from the saved tensors where possible
   (G2, G3's NOOP and pass-through identity, G4's bitwise `x3(p_t)`, G5's landing, G6), and cross-checked with the
   frozen functions for the program-based ones (G4's I3, I4, Level-1, readout; G5's recomputation). The logit identity
   needs the logits, which are not saved: check the recorded values' consistency instead. **G4 viability is the key
   question of this phase:** report every G4 quantity's maximum, where it occurred, and its margin to the frozen
   tolerance (1e-4 relative for I3, 1e-3 for I4, 2e-2 for Level-1, readout and logit), per arm type. State clearly
   whether the frozen tolerances held, and whether any margin is thin enough to be a risk at confirm (36 new frames).
7. **The measured quantities and τ, recomputed by you:** per frame for `two`, `Δc_full`, `F_rem`, `F_rem_off`, `F_ins`,
   `F_ins_off`, `N_f`, `S_f`, `A_amp`, `A_neg` from the saved contrasts (`Δc(arm) = contrasts(arm) − contrasts(REF)`,
   mean over the 79 nouns); the program's predicted fractions (with the frozen `ri.prediction_table` on 020's locked
   states, `fractions_for_weak=True`; check its digest equals the state's `table_content_sha256`); the errors
   `|F̂_rem − F_rem|`, `|F̂_ins − F_ins|` and `τ_rem`, `τ_ins` as the 33rd of 36 sorted values; P's evaluability (τ ≤
   0.20).
8. **The four revision triggers** (design §6): a gate failure; `‖off_c‖ < s` in more than 6 frames; `Δc_full > −1` in
   more than 6 frames; a non-finite fraction. Recompute each. Confirm that no trigger depends on the size of an exposed
   N, S, D or P result, and that no tolerance was changed.
9. **The candidate calibration file** (if written): `content_sha256` recomputes; it binds the confirmation file's
   digests, the module blob, the tolerances (equal to `ri.TOLERANCES`), τ, the triggers (none fired), the keys' count
   and digest, the measurement digests; `ri.load_calibration_101` accepts it. If the phase stopped for review instead,
   identify exactly which trigger fired, its values and locations, and whether it is a real integrity failure or an
   artifact of the implementation.
10. **The descriptive report** (no outcome force): the exposed shares and the power check; the real-swap `Δr`, `‖Δx‖`,
    `‖off_c‖`, `s`, `Δc_full` ranges; anything surprising (for example a sign pattern that would make a primary family
    likely to fail at confirm, saturation, or large ceiling errors of the frozen readout). Flag these as observations;
    calibration may not be revised on them.

11. **The Level-1 margin at confirm (a risk estimate, no decision force).** The tightest G4 margin is Level-1's: the
    recorded maximum is 0.01065 (NEC-HALF@many, `coordinated-adjective-018-3`) against 2e-2, and the next arm maxima
    are 0.0073 and 0.0054. Recompute the Level-1 error of every G4 run (about 1,692) from the saved measurements with
    the frozen functions, report its distribution (quantiles, the top 20 runs with arm and frame), and estimate, with a
    method you state and justify (an empirical exceedance argument, a peaks-over-threshold or extreme-value fit, or a
    bootstrap over frames), the probability that the maximum over a confirm of the same size (36 new frames, the same
    arms) exceeds 2e-2. Say what drives the outliers if you can tell (arm type, cue, frame, LayerNorm scale, the
    float32 attention-score floor that 020 diagnosed). This informs the human's lock decision; it may not change any
    tolerance (rule 1).
12. **The head-only arms and the isotropic controls, descriptively.** On these exposed data, compute the head arms'
    fractions (HEAD-NEC, HEAD-NEC-V against `Δc_full`; HEAD-SUF, HEAD-SUF-V against `Δc(SUF-READ)`), the removal and
    installation head shares under the 0.05 guard as design §7 defines them, and the isotropic comparisons. These are
    exploratory observations on spent data with no label force.
13. **The unguarded smoke reference** (the freeze review's N7): `smoke_reference.py` ran unguarded, after the freeze
    and before this phase, the S1 re-capture and the 55 arms of the spent exposed frame `coordinated-adjective-1`,
    writing digests to the scratchpad only. Say whether anything in it could affect the calibration or the
    confirmation.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation".
- Verdict: PASS, PASS WITH NOTES, or FAIL — on whether the calibration was executed and recorded as the frozen protocol
  requires and whether its numbers are correct (not on whether the exposed results look favourable).
- Every check with evidence (numbers, digests, short outputs); findings most severe first (blocker / should-fix / note)
  with file:line and a concrete scenario.
- Final message: the verdict, τ_rem and τ_ins, every G4 maximum with its margin, the trigger values, the exposed shares,
  and the findings.

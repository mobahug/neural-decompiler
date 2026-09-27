You are the independent, read-only reviewer of Experiment 025's production **confirmation**. The repository is
/Users/gaborhorvath-ulenius/myprojects/neural-decompiler, a mechanistic-interpretability project on
EleutherAI/pythia-70m-deduped @ e93a9faa9c77e5d09219f6c868bfc7a1bd65593c. Experiment 025 is the final experiment of
the sequence.

The confirm ran once at commit `6f72d7d8afcaa079ee38256b7c1250352b1b1bdd`. The provisional official outcome it
recorded:

| field | value |
|---|---|
| label | `CAUSAL_EFFECT_NOT_ESTABLISHED` |
| A | 26/40 |
| B | 23/40 |
| G | 37/40 |
| threshold | 27/40 |
| reference tail | 0.01923865414210013 |

Treat it as provisional until your review is done. Verify everything yourself; the operator's checks are claims to
test.

## Where things are

- **The repository** (read-only for you):
  - the installed freeze `experiments/025-nounness-direction-intervention/confirmation-v1.json`, lock
    `preregistration-lock.json` and `preregistration.md`, and the evidence under `evidence/`;
  - the design `docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md` (c0885e5,
    corrected 26c9925), and the plan `docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md`;
  - the module `src/neural_decompiler/cue_rotation.py` (`cr`), the runner
    `experiments/025-nounness-direction-intervention/run.py`, and the pinned 020–024 modules;
  - the local, git-ignored results: `outputs/experiment-025/results.json` (the results state) and
    `outputs/experiment-025/stage2-measurements.pt` (the durable measurements).
- **The launcher and run records:**
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm025/
  - `launch_confirm_v3.py`, the production launcher, sha256 `58b760bea9d8a7bf66bef4352deb5c7153f7aea46c264de02f49c9ab96d1e307`;
  - v1 `launch_confirm.py` and v2 `launch_confirm_v2.py`, for provenance;
  - `production_command.txt`, `production.log`, `production.exit`, `production_run.json` (the launcher's record), and
    `production_runs.jsonl` (one line per fresh run, with its tagged key; the condition in each key comes from the
    replacement vector's digest);
  - `confirm.sentinel`;
  - `shell_error_production.{log,exit}` and `shell_error_production_command.txt`: the earlier exit-127 event;
  - `smoke_*`, `rehearsal_*`, `smoke_run.json`: the pre-confirm engineering tests, spent prompts only;
  - `verify_confirm.py` and `verify_confirm.json`: the operator's own read-only verification. Treat it as a claim.

## Hard rules (a violation invalidates the review)

- **Read-only.** Never create, edit, move, delete, stage or commit anything in the repository, including
  `outputs/experiment-025/`.
- **Never run** the runner's `confirm`, `report`, `lock` or `freeze`, or the launcher in any mode.
- **No model forward of any kind.** You may load the model for its weights only, to rebuild `ul.ModelPrograms` for
  the gates and C. Refuse `torch.nn.Module.__call__` and every capture and intervention entry point in your
  scripts.
- Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm-review025/
  (scripts, logs and `REVIEW.md`).
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Use `git --no-optional-locks`, and never `uv`.
- **Never read** `outputs/experiment-023/`, `outputs/experiment-024/` or 022's `calibration-table.pt`.
- **Memory:** the machine has 8 GB of RAM and the measurements file is about 0.9 GB. Load it once per process.

## Independence

- Write your own code first for the responses and the outcome:
  - MSE, `ℓ = log(MSE)`, nMSE and `D_attn` from the saved tensors and 020's locked reference states;
  - A, `A_ij`, B, G, the counts, the thresholds and the label.
- Do not call `cr.per_cue`, `cr.statistics`, `cr.count_positive`, `cr.routing_distance`, `cr.outcome_label`,
  `rr.fresh_pair_cells` or `rr.cue_mse` until your own values exist; cross-check afterwards.
- For the gates:
  - you may use the pinned 020–023 program routines (`ul.compose_dx3`, `ul.i3_error`, `ul.contrast_of`,
    `ul.block0_terms`, `rd.ReadoutProgram.level1_detail`, `pm.lexicon_vector`, `ul.reference_embeddings`,
    `ul.reference_rows_017`);
  - drive them with your own loops, indexing and vector reconstruction, not through `cr.run_gates`,
    `cr.identity_gates` or `cr.level1_gates`;
  - reconstruct each run's patched vector from the lock's geometry yourself (the SHA-256 controls and the rotations),
    or at least verify it against the lock's per-cue digests.
- Report the maximum disagreement for every quantity.

## Review items (the reviewer's 15; answer each with evidence)

1. **Provenance.**
   - Verify the production record:
     - launcher v3 sha256 `58b760be…`, and the file today equals it;
     - commit `6f72d7d`; start about 19:23:58 UTC and end about 21:12:44 UTC; exit 0;
     - one model load, no guard refusals, no incident.
   - Separately, examine the shell exit-127 event (`shell_error_production.*`). Determine from every available
     trace (file contents, mtimes and birth times, the sentinel's birth time, the order of the records, the state's
     timestamps) whether:
     - the Python launcher never started;
     - no sentinel was created by it;
     - no model loaded;
     - no state, output or ledger changed.

     Conclude whether it was a `runner.confirm()` invocation or scientific attempt.
2. **The pre-fresh barrier.** From the launcher record, the state and the code (runner confirm order, launcher
   preflight), before the first fresh run:
   1. the repo-root cwd was asserted;
   2. the git, environment and checkpoint bindings passed;
   3. the lock and preregistration validated;
   4. I7′ reproduced the lock;
   5. the four spent patch-path checks passed;
   6. only then was the ledger created.

   Verify that the four patch-path keys stayed outside the ledger and the accounting.
3. **Accounting.**
   - Rebuild the 40 × 108 × 21 = 90,720 tagged keys yourself from the freeze.
   - Compare them with the run log, the state's ledger, and the artifact's per-group rows.
   - Expect 90,720 of each; 0 missing, 0 duplicate, 0 extra; exactly 4,320 per condition.
   - **Condition-vector pairing.** The log's condition per run comes from the replacement vector's digest. The
     runner's stored slot per run comes from its loop order: frames by `frame_id`, cues by token id, conditions in
     frozen order. Show that the log's n-th key equals the n-th key of the runner's loop order for all n; that
     proves each stored slot holds the run of its labelled condition.
   - Also argue from I1: I1 ties each saved x1 to its condition's vector.
4. **The durable artifact.**
   - `stage2-measurements.pt`: 898,673,151 bytes, sha256 `87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788`.
   - Its schema, the dimensions per group, condition and tensor, finiteness, and every stored tensor's
     `rc.tensor_digest` against `state.confirmation.stage2.tensors_sha256`.
   - From the code, verify that the scores were computed from the saved and re-read artifact.
5. **Recompute the validity gates.**
   - I1, I3, I4, C (rebuilt from the saved Δx3), and the Level-1 identity. Expect I1 ≈ 2.15e-5, I3 ≈ 4.35e-5,
     I4 ≈ 9.00e-5, C max 0.0, and Level-1 outcome-bearing max 0.011863 (all within their limits).
   - Level-1 must be verified over exactly the 16 outcome-bearing conditions, with the per-condition maxima.
   - Recompute everything that is feasible in about an hour of CPU. At minimum:
     - all Level-1 values and C on the 16 conditions;
     - I1, I3 and I4 on a large deterministic subset, plus the recorded max-location runs.

     State exactly what you covered.
6. **Independently rebuild ℓ and `D_attn`.**
   - For every cue and condition: MSE (Σ SSE_C / Σ n over 108 frames × 79 nouns), ℓ, nMSE (Σ SSE_C / Σ Q, with
     Q = Σ Δc²), and `D_attn`.
   - `D_attn` is the TV distance per head between the captured L4/L5 rows at `p_t` and 020's locked rows for the
     same frame; the equal mean over the 16 heads; then the mean over the 108 frames.
   - Verify the row alignment (frame, heads, keys up to `p_t`).
7. **A.** Compute `A_i = ½[ℓ(noun+0.32) − ℓ(noun−0.32)]` for all 40 cues.
   - Expect 26/40 strictly positive. Check every recorded value.
   - Handle ties and zeros (they count against).
   - Confirm the threshold is 27 and that A therefore FAILs.
8. **B.** Compute `A_ij` for the 7 random controls and `B_i = A_i − mean_j|A_ij|`.
   - Expect 23/40. Check every value.
   - Confirm B FAILs.
9. **G.** Compute `G_i = ½[D_attn(noun+) − D_attn(noun−)]`.
   - Expect 37/40. Check every value.
   - Confirm G PASSes but cannot change the label.
10. **The outcome precedence.** With all gates passing, A failing, B failing and G passing, the only legal label is
    `CAUSAL_EFFECT_NOT_ESTABLISHED`. Verify that the code (`cr.outcome_label`, `cr.statistics`) cannot pick another
    label from B or G after an A failure.
11. **The adjective/noun split, as a correctness check.**
    - Expected: adjectives A 20/20, B 20/20, G 20/20; nouns A 6/20, B 3/20, G 17/20.
    - Independently verify:
      - the cue order, the stratum labels and the token ids against the freeze;
      - the ± condition mapping: the noun+ vector must raise the nounness score, with s(+) − s(−) = +0.64 at
        float32 precision, for every cue in both strata;
      - the vector lookup and the `t̂` sign: no sign reversal between strata;
      - no indexing offset, reference-row mismatch, or condition-name parsing bug;
      - no accidental use of the plurality or random conditions.
    - Show whether the heterogeneity is real data or an artifact.
    - Do not turn the strata into new primary tests.
12. **The descriptive quantities,** checked only after the official result is reproduced:
    - medians: A 0.1419, B 0.0849, G 0.0156;
    - counts: half-dose A 25/40; plurality A 16/40; nMSE A 24/40; G above random 35/40; + above baseline 23/40;
      − below baseline 30/40;
    - template A counts: cardinal 26, coordinated-adjective 31, quantifier 25;
    - causal fraction 0.341; the ladder values.

    Confirm that none can alter the result.
13. **The result write order.**
    - The validity checks come before scoring, and the binding re-check before the write.
    - There is no incident, the result is one atomic write, and the descriptives come only afterwards and cannot
      alter the result.
    - The result-write digest `aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4` and the final state
      digest `1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c`. Check what you can: for example,
      whether removing the descriptive records and their writes reproduces `aa3db023…`.
14. **The interpretation.** From the preregistration and `cr.SEMANTICS`, give the exact allowed reading of
    `CAUSAL_EFFECT_NOT_ESTABLISHED`:

    > The preregistered test did not establish that moving fresh cue embeddings in the +nounness direction
    > consistently causes greater frozen block-4/5 readout-approximation error than the matched −nounness
    > intervention across the frozen 40-cue population.

    - It must not be rewritten as "no causal effect".
    - These may be reported only as descriptive findings, not as alternative PASSes: G passed strongly; the effect
      is highly heterogeneous by stratum; all 20 adjectives have positive A, B and G; noun cues mostly have
      positive G but not positive A or B.
15. **The stopping rule.** Check the design's stopping condition:
    - after 025, on any clean result, the sequence ends;
    - 024 remains a prospective predictor;
    - no Experiment 026 follows to rescue the result.

## The report (`REVIEW.md`) and the final message

- The verdict: PASS, PASS WITH NOTES, or NOT READY.
- Blockers, separately from notes.
- For each item: the evidence, the numbers, and the maximum disagreements.
- If PASS or PASS WITH NOTES, state whether:
  - the scientific result can be accepted as final;
  - the evidence may be archived;
  - `report` may run exactly once;
  - no confirm rerun is warranted.
- Return the verdict, the blockers and notes, and the key numbers as your final message.

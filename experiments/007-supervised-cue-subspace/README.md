# Experiment 007: A Response-Supervised Low-Rank Cue Subspace

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md)
(revision 3, approved at commit `5449491`) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-007-supervised-cue-subspace-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-007-supervised-cue-subspace-plan.md).

## Inputs

- Exploratory pool (all exposed by Experiment 005 and reused by Experiment 006): the six manifest frames and six
  extension frames, the sixteen exposed cue tokens, and the sixty nouns (59 are single-token on both forms and enter
  every statistic; `peach` is two tokens and is skipped everywhere, as in Experiments 005 and 006).
- The confirmation set is Experiment 006's
  [`confirmation-v1.json`](../006-low-rank-cue-decompilation/confirmation-v1.json), content sha256
  `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`, read in place. It has never produced a model
  output. Every phase refuses if its digest differs from the constant `INHERITED_CONFIRMATION_SHA256`; nothing in
  this experiment copies, rewrites, filters, reorders, or selects from it.
- [`inherited/experiment-006-epatch-means.json`](inherited/experiment-006-epatch-means.json) — a derived extract of
  Experiment 006's recorded exposed E-patch responses (192 per-(token, frame) mean contrast shifts) with the digests
  of their gitignored source. `explore` recomputes the responses and stops as an incident if any mean shift differs
  by more than `1e-6`.

## The program

`ΔE_T(w) = E(w) − E(ref_T)`; `z = U_rᵀ ΔE_T(w)` with `U_r` the first `r` left singular vectors of the uncentered
cross-moment `XᵀY` (`X` rows `ΔE_T(w)`, `Y` rows the measured E-patch residual responses; sign-fixed, nested,
rank(C) ≥ 5 and singular gaps ≥ `1e-8` in every fold); `δ̂ = V_T z` with `V_T` by the float64 pseudoinverse
(`rcond = max(Z_T.shape) × eps`, `rank(Z_T) < r` is an incident); `Δĉ_N = −u_N · [LN(ρ_f + δ̂) − LN(ρ_f)]` with the
exact final LayerNorm. Baselines: `PCA-006(r)` (Experiment 006's family through the same solver), `E005-scalar`,
and `Ridge-full` (one `B_T` per template, `λ` by fully nested leave-one-cue-out from a fixed multiplier grid scaled
by the inner-training trace). Rank selection, the quality gate, and τ are Experiment 006's, unchanged.

## Commands

```bash
uv run python experiments/007-supervised-cue-subspace/run.py validate
uv run python experiments/007-supervised-cue-subspace/run.py explore
uv run python experiments/007-supervised-cue-subspace/run.py calibrate
uv run python experiments/007-supervised-cue-subspace/run.py lock
uv run python experiments/007-supervised-cue-subspace/run.py confirm
uv run python experiments/007-supervised-cue-subspace/run.py report
```

- `validate`: the manifest, the Experiment 005 extension, the inherited confirmation set (digest), and the
  inherited extract, without a model.
- `explore` (Tier A, once): the pinned contract test, the E005-scalar baseline refit, the E-patch residual
  responses for 12 frames × 16 exposed tokens replicated against Experiment 006, leave-one-cue-out tables for the
  supervised family (ranks 1–4), `PCA-006` (ranks 1–4), and `Ridge-full`, the executable rank rule, the pre-lock
  quality gate, τ, the comparison record, and the exports of `selected`, `pca-006`, `ridge-full`, and
  `e005-scalar` under `outputs/experiment-007/parameters/`. If the gate fails the experiment ends here at
  `QUALITY_GATE_FAILED`.
- `calibrate` (Tier B): records τ and the comparison; refuses after a failed gate.
- `lock`: writes `outputs/experiment-007/candidate-lock.json` with every program's predictions for every
  confirmation prompt computed without running them. Installing it as `preregistration-lock.json` and committing it
  is the preregistration act, done by hand.
- `confirm` (Tier C, once): validates the committed lock (frozen-input digests, the inherited confirmation and
  extract digests, all four parameter sets, program sources, unchanged scientific paths since the lock commit,
  clean tree, a ledger without any confirmation prompt or fresh noun), then runs the fresh nouns on the manifest
  frames and the fresh frames with the original cue pairs (circuit families, reported only), and the 24 fresh
  tokens' behavioral prompts and E-patch interventions on the six fresh frames (Y families); applies the floors,
  the bands, and the program-axis outcome rule.
- `report` renders `outputs/experiment-007/report.md`.

Boundaries: the confirmation set is never executed before `confirm`; every phase refuses to run out of order,
twice, or on a dirty tree; numerical and identifiability failures are incidents, never outcome labels (an explore
incident is recorded with its commit and blocks another `explore` at that commit; a confirm incident blocks any
re-run in this protocol version); `confirm` recomputes every preregistered prediction from the on-disk programs and
stops as an incident if any differs from the lock by more than `1e-9`; `linear_cue_program.py` loads only exported
tensors and never imports the network stack.

The `E005-scalar` baseline is refit deterministically on the Experiment 005 development data using the 005 lock's
own mechanism description (its component order fixes float summation order); `explore` requires `k_T`, every axis
and context tensor, and the gains `g_R` printed in the 005 lock statement to be reproduced (an incident otherwise)
and records whether the whole parameter index digest equals the 005 lock's.

## Status — 2026-09-18: Tier A executed once; the pre-lock quality gate passed; candidate lock awaiting installation

`explore` ran once on protocol/code commit `cc56014` (run `81e578c5e6d41814`; A0 passed; CPU float32, 4 BLAS threads). The
rendered Tier A report is copied verbatim to
[`evidence/exploration-report-2026-09-18.md`](evidence/exploration-report-2026-09-18.md). Ledger: 36 prompt keys (the
twelve exposed frames' cue prompts and reference prompts), 60 noun keys; no confirmation prompt, fresh noun, fresh
frame, or fresh cue token was executed.

- Inherited responses: the recomputed 192 E-patch mean shifts match Experiment 006 exactly (max deviation 0.0;
  informational residual-norm deviation 1.8e-6).
- Leave-one-cue-out (supervised cross-moment SVD, cue-level MAE ± SE over the 16 held-out tokens): r1 1.079 ± 0.198,
  r2 1.156 ± 0.195, r3 1.176 ± 0.200, r4 1.061 ± 0.204. Eligible {1} (no rank reaches 0.8 × error₁ = 0.863);
  **selected r = 1**. Every fold: rank(C) ≥ 5, all gaps ≥ 1e-8, rank(Z_T) = r in every template; final fit rank(C) 15,
  gaps 0.454 / 0.363 / 0.078 / 0.0061.
- Quality gate **passed**: Spearman 0.765 (≥ 0.70), normalized LOCO RMSE 0.320 (≤ 0.50); τ = 3.972 nats (3 × RMSE of
  the cue-level errors; dominated by `a` 3.46, `all` 1.84, `every` 1.74, `both` 1.64).
- Comparison (recorded, not gating): the supervised subspace beats `PCA-006` at every rank by about 50% (ratios
  0.47–0.51; the stated ≥ 20% prediction is met). `Ridge-full` LOCO error 0.860 (ratio 0.80 to the selected rank;
  the smallest grid multiplier 0.01 was chosen in all sixteen outer folds). `E005-scalar` cue-level error 1.410.
- `E005-scalar` refit with the 005 lock's mechanism order reproduces the frozen Experiment 005 program exactly: every
  parameter tensor digest and k_T match, and the parameter file's text digest `c51f8fed…` equals the 005 lock's
  `parameters_index_sha256`. (The results state's `matches_experiment_005_lock: False` flag compares a canonical-JSON
  digest with that file digest — a convention mismatch inherited from Experiment 006's code, not a difference in
  parameters.)
- `calibrate` and `lock` ran at the same commit. Candidate lock: `outputs/experiment-007/candidate-lock.json`,
  content sha256 `de0ae866a5625811ef8fcf01f60c0b563aeebb03f7c3f80e058a0bf82216dd96` (rank 1, τ 3.972; predictions for
  all 24 fresh tokens × 6 fresh frames × 20 fresh nouns and the 6 fresh-frame cue pairs, for `selected`, `pca-006`,
  `e005-scalar`, and `ridge-full`). Installing it as `preregistration-lock.json` and committing it is the
  preregistration act; `confirm` then runs once.

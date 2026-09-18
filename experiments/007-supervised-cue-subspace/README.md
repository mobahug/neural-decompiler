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

## Status — 2026-09-18: complete; the single confirmation ended in `PROGRAM_NOT_SUPPORTED`

Experiment 007 is closed. The preregistration lock (content sha256 `de0ae866…dd96`) was committed at `ede0f53`;
`confirm` ran once at that commit on a clean tree (results state sha256
`118dffa052244bafa473b2e74c8964a0403f4d80bd9ca9696cbe3202f9aaa7c0`; every preregistered prediction was reproduced
from the on-disk programs before any fresh prompt ran, max difference 0.0). The final report is copied verbatim to
[`evidence/final-report-2026-09-18.md`](evidence/final-report-2026-09-18.md). Ledger after confirm: 198 prompt keys,
80 noun keys. The outcome is final and is not amended or rerun.

- Precondition: the fresh-frame cue effect held, 120/120 pairs positive (floor 108); on the fresh nouns over the
  manifest frames 120/120 (floor 114).
- **Y1 (E-patch prediction, the decompiler's own route) failed**: Spearman 0.785 over the 24 fresh tokens (floor
  0.80); MAE 1.071 nats (within τ = 3.972); the confident-token sign rule failed for `another` (measured −0.32,
  predicted −2.76; signs agree in 3/6 frames) and `this` (measured +0.01, predicted −2.80; 3/6). Per category the
  Y1 MAE was numeral 0.70, control 0.74, quantity 0.75, inherited 0.89, determiner 1.68.
- Y2 (behavioral shift) passed: Spearman 0.801, MAE 1.195 (≤ 1.5 τ). Y3 passed: every template's fresh-frame cue-pair
  mean within τ (cardinal 4.55 vs 5.30 measured, quantifier 5.68 vs 5.40, coordinated 4.37 vs 4.37) and 120/120
  positive. Bands: 24/24 tokens inside ± τ in ≥ 5/6 frames (τ is wide).
- Outcome rule: a Y floor failed → `PROGRAM_NOT_SUPPORTED`, failing family Y1, located at the encoding subspace.
- Y4 baselines (reported, never selected against): `Ridge-full` Y1 Spearman 0.796, MAE 1.049 (Y2 0.795 / 1.182);
  `E005-scalar` 0.781 / 1.744 (Y2 0.793 / 1.854); `PCA-006` (rank 1) 0.610 / 2.708 (Y2 0.611 / 2.813). `Ridge-full`
  and `E005-scalar` fail the same two confident tokens (`another`, `this`).
- Circuit families (reported; not in the outcome): on both fresh sets P1, P4, P5, P7, P8, P9 passed and P3 failed on
  its correlation floor (0.696 on the fresh nouns × manifest frames, 0.840 on the fresh frames; floor 0.90; F 0.78 /
  0.77 with 120/120 paired signs). C002 is not eligible for review and stays at `LOCALIZED`.

What the result says, within the design's limits: the rank-1 response-supervised direction of `L00.MLP` generalizes
to unseen tokens, frames, and nouns about as well as the tested full-dimensional regularized linear map (Y1 MAE 1.07 vs
1.05; LOCO 1.08 vs 0.86) and far better than the leading unsupervised direction (2.71) or the scalar program (1.74), so
increasing dimensionality through the tested `Ridge-full` estimator does not resolve the failure. The selected
supervised rank-1 program and that regularized full-dimensional linear baseline both fail to explain the anomalously
weak E-patch response of some singular-selecting determiners, especially `this` (measured +0.01, predicted −2.80) and
`another` (−0.32, predicted −2.76), whose `E(w)` projects onto the plural side of the learned direction; with them the Y1
rank correlation misses the 0.80 floor by 0.015 and the confident-sign rule fails. This is a statement about the two
estimators tested, not about every possible linear map: `PCA-006`, although far worse overall, predicts `this` near zero
(0.020 vs 0.007 measured). Nothing here generalizes beyond the pinned checkpoint, the three templates, single-token
regular nouns, and the tokens tested. The 24 confirmation tokens, 6 frames, and 20 nouns are now exposed and may join the
exploratory pool of a future experiment; they are no longer a holdout.

### Tier A (2026-09-18, for the record)

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

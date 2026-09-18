# Experiment 007: A Response-Supervised Low-Rank Cue Subspace

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md)
(revision 3, approved at commit `5449491`) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-007-supervised-cue-subspace-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-007-supervised-cue-subspace-plan.md).

## Inputs

- Exploratory pool (all exposed by Experiment 005 and reused by Experiment 006): the six manifest frames and six
  extension frames, the sixteen exposed cue tokens, and the sixty nouns.
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
twice, or on a dirty tree; numerical and identifiability failures are incidents, never outcome labels;
`linear_cue_program.py` loads only exported tensors and never imports the network stack.

## Status — 2026-09-18: implemented; Tier A not yet run

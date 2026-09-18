# Experiment 006: Low-Rank Decompilation of the Cue Encoding

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md)
(revision 4) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-006-low-rank-cue-decompilation-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-006-low-rank-cue-decompilation-plan.md).

## Inputs

- Exploratory pool (all exposed by Experiment 005): the six manifest frames and six extension frames, the sixteen
  exposed cue tokens, the sixty nouns, and the 72 cue-word prompts.
- `confirmation-v1.json` — the frozen confirmation set, built from tokenizer rules only on 2026-09-18 before any
  Experiment 006 model output, content sha256 `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`:
  twenty fresh nouns (`stone field road door window bridge tower planet letter wheel` / `patch coach church tax
  boss` / `body copy duty spy study`), six fresh frames (`The crate holds`, `The gallery shows`, `The index lists`,
  `The archive keeps`, `Ravi and Elena sorted … plain`, `Nora and Felix stacked … heavy`), and twenty-four fresh cue
  tokens (`any no another single multiple numerous twelve hundred` inherited; `six seven eight nine`; `dozen
  countless various fewer`; `this that these those`; `big red old fresh`). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/006-low-rank-cue-decompilation/run.py validate
uv run python experiments/006-low-rank-cue-decompilation/run.py freeze-confirmation
uv run python experiments/006-low-rank-cue-decompilation/run.py explore
uv run python experiments/006-low-rank-cue-decompilation/run.py calibrate
uv run python experiments/006-low-rank-cue-decompilation/run.py lock
uv run python experiments/006-low-rank-cue-decompilation/run.py confirm
uv run python experiments/006-low-rank-cue-decompilation/run.py report
```

- `explore` (Tier A, once): the pinned contract test, the E005-scalar baseline refit, the fixed circuit's
  families on the exposed pool (60 nouns × 12 frames), the E-patch residual responses for 12 frames × 16 exposed
  tokens, leave-one-cue-out errors for ranks 1–4, the executable rank rule, the pre-lock quality gate, τ, and the
  exports of the selected program, `R1-PCA`, and `E005-scalar` under `outputs/experiment-006/parameters/`.
- `calibrate` (Tier B): records τ and the Experiment 005 circuit bands carried over.
- `lock`: refuses if the quality gate failed; writes `outputs/experiment-006/candidate-lock.json` with every
  program's predictions for every confirmation prompt computed without running them. Installing it as
  `preregistration-lock.json` and committing it is the preregistration act, done by hand.
- `confirm` (Tier C, once): validates the committed lock (digests of the frozen inputs and of all three parameter
  sets, program sources, unchanged scientific paths since the lock commit, clean tree, a ledger without any
  confirmation prompt or fresh noun), then runs fresh nouns on the manifest frames, the fresh frames with the
  original cue pairs, and the 24 fresh tokens' behavioral prompts and E-patch interventions on the six fresh
  frames; applies the floors, bands, and the outcome rule.
- `report` renders `outputs/experiment-006/report.md`.

Boundaries: the confirmation set is never executed before `confirm`; every phase refuses to run out of order,
twice, or on a dirty tree; `low_rank_program.py` loads only exported tensors and never imports the network stack.

## Status — 2026-09-18: Tier A executed once; the pre-lock quality gate failed

`explore` ran on protocol/code commit `adea65f` (run `b09a022e021ba19c`, results state sha256
`ba502188070f5f7144ec78a2b3ad166c15c13047631ccdb18b75cfbb37581982`; A0 passed). The rendered report is copied
verbatim to [`evidence/exploration-report-2026-09-18.md`](evidence/exploration-report-2026-09-18.md). Ledger: 36
prompt keys (the twelve exposed frames' cue prompts and reference prompts), 60 noun keys; no confirmation prompt
and no fresh noun was executed.

- Circuit on the exposed pool (60 nouns × 12 frames, descriptive): cue effect 708/708 positive; P1 0.978; P3
  F 0.804 with 708/708 paired signs but correlation 0.756 (the 0.90 confirmation floor would not be met); P4
  0.915; P5 0.819 / 0.870; P8 0.813; P9 0.917 / 1.037 / 0.190. P7 was recorded as `None` because the reused
  cross-frame control paired only templates with exactly two frames; the pairing was generalized (cyclic) after
  the run, and `explore` cannot be re-run in this protocol version.
- E-patch responses: replacing `L00.MLP` in a reference prompt by the weight-only `E(w)` moves the contrast by a
  token-specific amount that is nearly frame-invariant (for example `all` −3.7 to −5.4 nats over the twelve frames,
  `a` +0.4 to +1.6 in cardinal and coordinated frames, `every` about zero); the named circuit carries 0.6–0.9 of
  each response's squared norm.
- Leave-one-cue-out rank selection: errors 2.305 ± 0.296 (r = 1), 2.400 ± 0.316 (r = 2), 2.358 ± 0.275 (r = 3),
  2.099 ± 0.302 (r = 4); no rank beats `R1-PCA` by 20%, so r = 1 is selected. Quality gate **FAILED**: Spearman
  0.800 (≥ 0.70) but normalized RMSE 0.623 (> 0.50). τ would have been 7.724 nats. The frozen `E005-scalar`
  baseline's cue-level error on the same tokens is 1.410 (not a leave-one-out figure).
- Consequently no lock may be written and the experiment ends at Tier A. Outcome recorded as
  `QUALITY_GATE_FAILED`; no confirmation, no claim change (C002 stays `LOCALIZED`).

Post-hoc diagnostic (exploratory data only, outside the results state, not a gate): fitting the same family on all
sixteen tokens in-sample gives cue-level MAE 2.35 (r = 1), 2.27 (r = 2), 2.09 (r = 3), 2.02 (r = 4), 0.54
(r = 8), 0.40 (r = 15); a rank-free linear map on the full E difference leaves 34% of the response energy
unexplained in-sample. The failure is therefore representational: the leading principal directions of the sixteen
`E(w)` vectors do not carry the number-relevant variation, so a rank ≤ 4 PCA projection cannot represent the
cue-to-readout map, whatever the fitting procedure. A supervised low-rank projection (directions chosen for the
response, not for E's variance) is the obvious next design; it is not part of this protocol.

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

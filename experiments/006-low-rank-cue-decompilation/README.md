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
```

Scientific phases (`explore`, `calibrate`, `lock`, `confirm`, `report`) are added by the plan's later tasks and
enforced in order through `outputs/experiment-006/results.json`.

# Experiment 005: Count-Cued Noun Number Selection — Prospective Mechanism

This directory implements the approved design
[`docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md`](../../docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md)
(revision 4) through the plan
[`docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md`](../../docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md).

## Scientific inputs

- `screening/behavior-candidates/manifest-v1.json` (`regular-plural` cases only), content sha256
  `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`. Every case in a split is one of
  twelve prompts; the noun is only the scored next token.
- `extension-v1.json` — the frozen extension set, built from tokenizer rules only on 2026-09-18 before
  any Experiment 005 model output, content sha256
  `1c6852547f1c2d8ecfe8a736599ba3670fac51e43c93f04057091c56ba179da5`: six new frames (two per template)
  with the original cues, and twelve new cue words (`a`, `the`, `three`, `four`, `five`, `ten`, `many`,
  `few`, `some`, `all`, `both`, `every`) substituted into the six original frames (72 prompts). It
  contains no nouns; extension prompts are scored with the twenty reserve nouns and are executed only by
  `confirm`.

## Commands

```bash
uv run python experiments/005-regular-plural-mechanism/run.py validate
uv run python experiments/005-regular-plural-mechanism/run.py freeze-extension
```

`validate` checks the manifest and the committed extension without loading a model. `freeze-extension`
refuses to overwrite the frozen file. The scientific phases (`discover`, `calibrate`, `revise`, `lock`,
`confirm`, `report`) are added by the implementation plan's later tasks and are enforced in order through
`outputs/experiment-005/results.json`.

## Boundaries

- No reserve noun and no extension prompt is executed before the committed preregistration lock validates
  inside `confirm`; the results state records every executed prompt and scored noun.
- Generated outputs live under `outputs/experiment-005/` and are not committed; evidence copies of the
  final reports go under `evidence/` with the claim.

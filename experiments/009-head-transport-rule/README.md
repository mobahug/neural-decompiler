# Experiment 009: The Transport Rule of `L03.H04`

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md).

## Inputs

- Exposed pool (fitting data): the forty cue tokens, eighteen frames, and eighty nouns of Experiment 008.
- `confirmation-v1.json` — frozen on 2026-09-18 by tokenizer rules under the design's grammatical-compatibility policy,
  before any Experiment 009 model output, content sha256 `6330b5a22f81455d90b42a412b8bccc325b5150fa6de58d90e341bfc8589095f`:
  23 fresh cue tokens — `either neither` (singular-selecting; `an` dropped because only 8 of the 79 nouns are
  vowel-initial), `eleven thirteen fourteen fifteen sixteen seventeen` (plural-numeral), `more most other enough
  certain` (plural-quantity), `my your his her our their` (number-neutral), `small blue new cold` (bare-adjective,
  plural-forcing) — and six fresh frames (`The shelf carries`, `The ledger names`, `The manual describes`, `The
  bulletin mentions`, `Ida and Tomas counted … shiny`, `Yusuf and Petra wrapped … thin`); every token licensed in
  every frame over the 79 single-token nouns; 138 token prompts. Executed only by `confirm`.

## Commands

```bash
uv run python experiments/009-head-transport-rule/run.py validate
uv run python experiments/009-head-transport-rule/run.py freeze-confirmation
uv run python experiments/009-head-transport-rule/run.py explore
uv run python experiments/009-head-transport-rule/run.py lock
uv run python experiments/009-head-transport-rule/run.py confirm
uv run python experiments/009-head-transport-rule/run.py report
```

- `explore` (Tier A, once): for every exposed (token, frame) the full, axis-only, and orthogonal-only E-patches with the
  head's internals captured; the exact OV decomposition (P1/P2/P3/remainder) and the locked mechanism level; stage
  fractions, additivity gaps, the non-additivity stage; the transport rule and the ridge-scalar baseline with
  leave-one-cue-out; the forty-token contrast rule (Experiment 007 estimator) with leave-one-cue-out; the quality gates;
  the exports under `outputs/experiment-009/parameters/`.
- `lock`: writes `outputs/experiment-009/candidate-lock.json` and `candidate-predictions.md` — every fresh (token,
  frame) prediction from the exported rules, without running any fresh prompt. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (Tier C, once): validates the committed artifacts (digests, state binding, unchanged scientific paths,
  clean tree, untouched ledger), recomputes every locked prediction from the on-disk rules **before any fresh prompt
  runs** (refuses on any difference above `1e-9`), then runs the fresh frames' reference and cue-pair prompts, the
  plural cue's E-patch, and every fresh token's three patches and behavioral prompt; applies the floors and the
  outcome rule.
- `report` renders `outputs/experiment-009/report.md`.

Boundaries: no fresh prompt before `confirm`; every phase refuses to run out of order or twice; numerical and
identity failures are incidents recorded with their commit; the head weights are read from the model, the OV levels
from captured activations; nothing is fitted on confirmation data.

## Status — 2026-09-18: confirmation set frozen; Tier A not yet run

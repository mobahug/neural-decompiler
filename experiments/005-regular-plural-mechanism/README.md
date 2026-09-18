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
uv run python experiments/005-regular-plural-mechanism/run.py discover [--screening-results PATH]
uv run python experiments/005-regular-plural-mechanism/run.py calibrate
uv run python experiments/005-regular-plural-mechanism/run.py revise
uv run python experiments/005-regular-plural-mechanism/run.py lock
uv run python experiments/005-regular-plural-mechanism/run.py confirm
uv run python experiments/005-regular-plural-mechanism/run.py report
```

- `validate` checks the manifest and the committed extension without loading a model; `freeze-extension` refuses
  to overwrite the frozen file.
- `discover` (Tier A) runs once: the pinned Bridge contract test (A0), the behavioral baseline replication against
  the screen (A1), the position-resolved patching map, the layer profile, attention, number axes and exact
  direct-effect accounting, the chain/path tests, abstractness controls, neutralization with self-repair
  accounting, isolation, the deterministic hypothesis tree, mechanism-set selection against the Tier A floors,
  and the weight-only program's parameters (exported under `outputs/experiment-005/parameters/`). It records
  mechanism version `M1` or `NO_COMPACT_MECHANISM`.
- `calibrate` (Tier B) evaluates every circuit-axis family on the 114 single-token holdout nouns and derives the
  bands, `RMSE_B`, and `τ`; at most two passes. `revise` applies the design's mechanical revision once, after a
  failed first pass.
- `lock` writes `outputs/experiment-005/candidate-lock.json`, including the program's predictions for every
  extension prompt computed without running them. Installing it as `preregistration-lock.json` and committing it
  is the preregistration act and is done by hand.
- `confirm` (Tier C) validates the committed lock (digests, mechanism version, calibration record, program
  source and parameters, a clean tree whose scientific paths are unchanged since the lock commit, and an
  execution ledger with no reserve noun or extension prompt) and then runs the reserve nouns on the twelve
  manifest prompts, the six new frames, and the 72 cue-word prompts exactly once. It applies the floors, bands,
  discriminating regions, and the two-axis outcome rule.
- `report` renders `outputs/experiment-005/report.md`.

## Boundaries

- No reserve noun and no extension prompt is executed before the committed preregistration lock validates
  inside `confirm`; the results state records every executed prompt and scored noun and every phase refuses to
  run out of order, twice, or on a dirty tree.
- `mechanism_program.py` is the decompiled computation. It loads only the exported tensors, may evaluate the
  declared token-local `E_program` sub-modules (the token embedding and the layer-0 MLP) on single tokens, and
  never imports TransformerLens, Transformers, or the instrumentation modules. The exact final LayerNorm uses
  population variance and the pinned `ε = 1e-5`.
- Numerical note: every measurement runs through the instrumented forward (`use_attn_result=True`), whose
  float32 rounding differs from the plain forward by up to a few 1e-3 nats on ~5-nat contrasts (Pythia-70M logits
  reach ~1.6e3); the A1 prompt-level check tolerates 1e-2 and records the gap. The direct-effect decomposition
  is stated in float64 on the reconstructed residual and reproduces the contrast to 1e-4 nats by identity; the
  model's own float32 unembedding GEMM differs from that reconstruction by up to about 3e-3 nats, recorded as
  `max_model_gap` with a 1e-2 tolerance.
- Generated outputs live under `outputs/experiment-005/` and are not committed; evidence copies of the final
  reports go under `evidence/` with the claim.

## Status — 2026-09-18: Tier A executed once; selection stopped at the program floor

`discover` ran on protocol/code commit `83d1ae4` (attempt 2; attempt 1 stopped in A1 on the instrumented-path
numerics documented above, before any conclusion). Run `050405f9f4f7e629`, results state sha256
`d5977a11954f5ee95e07418191d599cb114018679198e9978d8964910f47795b`; the rendered report is copied verbatim to
[`evidence/discovery-report-2026-09-18.md`](evidence/discovery-report-2026-09-18.md). A0 passed; A1 matched the
screen per case to 0.0 and the report aggregates within tolerance.

Exploratory findings (development data only, no claim):

- Hypothesis tree: `H1` (dedicated transport), unambiguous: `L03.H04` alone recovers 0.816 of the coordinated
  shift; freezing it blocks 0.865 of a counterfactual residual patch at the cue position; the E-alone patch
  reproduces 0.90 of the head's number change and 0.99 of the readout MLPs' change. The cue information at the
  cue position is consumed at layer 3 (residual profile 0.956 at the layer-3 input, 0.125 at the layer-4 input).
- Encoding: `L00.MLP` carries 0.535 of the transport-input axis, the raw embedding 0.009.
- Circuit: with E = `L00.MLP` at the cue position, T = `L03.H04`, R = {`L04.MLP`, `L05.MLP`} (k = 3), the
  counterfactual set recovers 0.982 overall (every template and rule class ≥ 0.975) and the isolation test
  retains 0.783 of the shift (0.927 / 0.760 / 0.664 by template) with the correct sign in 80/120 cases.
- Program (token-local `E_program` = `L00.MLP`, k_T = 0.962): every development pair sign is reproduced, but
  the per-template mean is under-predicted by 0.43 (cardinal), 0.86 (coordinated), and 1.44 nats (quantifier);
  the quantifier gap exceeds the 1.0-nat Tier A program floor at every k ≤ 6, so version M1 is `rejected` and the
  recorded outcome is `NO_COMPACT_MECHANISM`. The shortfall is the cue-dependent shift carried by components
  outside S_M (which recompute under patching but are not part of the program's additive increment) plus the
  lexicon's asymmetric `several` value (n_c = 0.873).

No calibration, lock, or confirmation has run; reserve nouns and extension prompts remain unexecuted. Whether
protocol v1 ends here or the design is amended (a version whose circuit floors pass but whose program misses a
template mean would proceed with the decompilation axis capped) is a design decision recorded separately.

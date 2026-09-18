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
uv run python experiments/005-regular-plural-mechanism/run.py continue
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
- `continue` (design revision 5, protocol v2) applies only after a `discover` that ended in `NO_COMPACT_MECHANISM`
  and before any calibration: it adopts the smallest recorded attempt eligible under the amended rule (recovery
  strata; isolation overall ≥ 0.50 and ≥ 0.40 per template; roles — the P3 sign-retention count is not used for
  selection), re-exports its program parameters, records the version with `PROGRAM_CAPPED`, and freezes the set
  (`revise` is refused). It may run again only to supersede a version adopted under an older rule, before
  calibration. `CIRCUIT_ONLY` then requires X1 and X2; otherwise the outcome is `CIRCUIT_NOT_GENERALIZED`.
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

Protocol v1 ends here and its outcome stands. Design revision 5 (approved 2026-09-18) continues as protocol v2
with `PROGRAM_CAPPED`; its prospective question is whether the E → T → R circuit survives untouched nouns and
unseen frames. The first continuation selection (`M2`, k = 2, isolation overall 0.513 but 0.365 on the
coordinated-adjective template) was superseded before Tier B when the eligibility rule was corrected to apply
the P3 per-template isolation floor; `M3` = k = 3 (E `L00.MLP`, T `L03.H04`, R `L04.MLP` + `L05.MLP`) is the
mechanical choice under the amended rule (`continue` run 2, results state sha256
`b8e89bbdd54fc0cdee3ae04461a53eda4d883608851d9e29408e1f1baddfe263`). Diagnostic only: the clean model flips
87/120 development pairs and the k = 3 isolation retains 80 (≈ 92% conditional retention); the
`retention_diagnostic` field stored inside the M3 version record double-counted holdout flips (185) because of a
filter bug fixed immediately afterwards — the lock and the report recompute it correctly, and the field is not
a selection criterion. Reserve nouns and extension prompts were unexecuted when the continuation was adopted.

## Status — 2026-09-18: protocol v2 executed to completion

- `continue` (twice; M2 superseded, M3 = k = 3 adopted), `calibrate` (pass 1, floors passed, `RMSE_B` 0.609,
  τ 1.826), `lock` (candidate content sha256 `ec2a8879bab35fd6e4d45484321595aed565fb40f19bf608daf2574ea50ccc97`,
  installed and committed by hand as `preregistration-lock.json` in commit `30e8b84`), and `confirm` (once, on
  commit `30e8b84`) all ran. Final results state sha256
  `5a0c6836af57b6132216b3160d429c58b3e75015842e63dc86928161a877f6bf`; the rendered report is copied verbatim to
  [`evidence/final-report-2026-09-18.md`](evidence/final-report-2026-09-18.md). Ledger: 96 prompt keys, 60 noun keys.
- **Outcome: `BEHAVIOR_NOT_REPLICATED`** (axes `CIRCUIT_FAIL` via P3's retention count, `PROGRAM_FAIL` by cap,
  `NOT_GENERALIZED` via P3's retention count on the new frames). The reserve nouns failed the preregistered
  behavioral precondition: 98/120 primary conditions correct (floor 103) and 75/120 contrast flips (floor 96);
  44 of the 45 failures are plural conditions, and in all 45 the number variable still matched the cue (S3).
  By the preregistered rule no other family is confirmatory.
- Recorded, non-confirmatory: on the reserve nouns P1 0.985, P3 0.795 (0.962 / 0.760 / 0.672 by template), P4
  0.873, P5 0.817 / 0.869, P6 0.840 / 0.074, P7 0.984 / −0.007, P8 loss 0.826, P9 0.987 / 0.900 / 0.074, with
  every P band hit except P3's retention count (71/120; the clean model flips only 75/120 reserve pairs) and the
  secondary P8 compensation band; B2's template means fell below their holdout bands. On the six unseen frames X1
  passed (120/120 positive pairs; 12/12 number variables) and P1 0.980, P3 0.824, P4 0.948, P5 0.822 / 0.872, P8
  0.796, P9 0.937 / 1.079 / 0.286, again failing only P3's retention count (62/120; clean flips 65/120).
- Cue words (diagnostic, capped program): the E-patch reproduces each word's behavioral shift closely (the cue
  information travels through `L00.MLP`), but the one-dimensional lexicon does not predict the magnitudes:
  `all`, `some`, `both`, `few`, `many` shift the contrast by −3.8 to −4.8 nats with lexicon values within ±0.48,
  `every` shifts it by ≈ 0, `a` by ≈ 0, `the` by −2.6; the four numerals agree in sign in 6/6 frames. X3 Spearman
  0.699 (floor 0.70), X4 Spearman 0.692 with MAE 1.2–1.6 nats.
- Claim: [`research/claims/C002-count-cued-noun-number-circuit.md`](../../research/claims/C002-count-cued-noun-number-circuit.md)
  at maturity `LOCALIZED`. The experiment is complete; a second scientific attempt requires a new protocol version.

## Archival notes (2026-09-18, after closure; no artifact rewritten)

- The phase table in both evidence reports shows `report: not_started` because the `report` phase renders the
  Markdown without marking itself complete in the results state; the reports were rendered after the phases they
  describe and copied verbatim. This is a runner cosmetic, not a missing phase.
- The superseded `M2` record preserves the statement text rendered at its adoption, whose cap line reads "a
  contextual encoding component is required"; that wording was the generic cap text and was corrected in the same
  session before `M3` was adopted (`M3` names the protocol v1 program-floor failure). `M2` is kept verbatim as
  history and was never calibrated, locked, or confirmed.
- Experiment 005 is closed. Both outcomes stand as recorded; the next experiment is designed separately as
  Experiment 006 and treats all Experiment 005 data, including the reserve nouns and the extension prompts, as
  exploratory input.

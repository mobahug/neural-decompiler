# Experiment 015: The Cue-Induced Attention-Pattern Change from the Decoded Query/Key Computation — A Token-Local Cue-Change Q/K Model, Conditional on the Reference Frame State

Implements the approved design
[`docs/superpowers/specs/2026-09-19-experiment-015-attention-pattern-token-local-design.md`](../../docs/superpowers/specs/2026-09-19-experiment-015-attention-pattern-token-local-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-19-experiment-015-attention-pattern-token-local-plan.md`](../../docs/superpowers/plans/2026-09-19-experiment-015-attention-pattern-token-local-plan.md).
Prospective, zero-parameter, program-completing: the pattern-change term Experiment 013 measured from the patched run —
`Σ_k ΔA_h(p_c,k) v_h'(k) W_O^h` for the sixteen layer-1–2 heads — is written as a weight-only program (LayerNorm →
query/key/value projections → partial rotary rotation → scaled dot products → softmax) and evaluated at two levels
kept strictly apart. **Level 1** recomputes the cue rows from the frame's actual reference residuals and `ΔE`; it is the
model itself (positions before the cue are causally invariant and the parallel residual makes `x₁'(p_c) = x₁(p_c) + ΔE`
exactly), so it must reproduce the captured rows and residuals within a frozen tolerance — an identity check that stops
the phase when violated and counts toward no floor. **Level 0**, the hypothesis, is the token-local cue-change model
conditional on the reference frame state: the cue position's query, key and value *changes* computed at the Experiment
012 locked template-mean bases from `ΔE` alone, applied to the frame's reference keys, values and logit rows. The rigid
number-axis alternative is committed for rejection; the oracle and Level-0 diagonal-proportional comparators are named
descriptive competitors with a predeclared interpretation rule. No label is preregistered for any head or cue.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 159 exposed cue tokens (Experiment
  014's 135 and its 24 confirmed) and the 48 exposed frames (its 42 and its 6), with the 80 nouns; committed extracts
  of Experiment 014's per-pair `c_L`, `c_M`, `c_H`, `c_k` ([`inherited/experiment-014-pair-ledger.json`](inherited/experiment-014-pair-ledger.json),
  4884 pairs, replicated within `1e-6`) and of Experiment 013's per-pair self-attention weight changes and pattern-change
  read ([`inherited/experiment-013-pattern-extract.json`](inherited/experiment-013-pattern-extract.json), 3732 pairs,
  replicated within `1e-6`); the Experiment 011 lock's axes and read weight, the Experiment 012 lock's bases and the
  Experiment 014 lock's cue-position reference states (checked against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-19 by tokenizer rules before any Experiment 015 model output, content
  sha256 `66afcba4cddb872c91a99611a1d0d726e0b2d88e2e2c4e240bfec393fa1ce633`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `fourth fifth particular previous final` (determiner-like), `sixth seventh eighth
  ninth tenth` (ordinal-or-numeral), `scant adequate moderate substantial enormous` (quantity), `himself herself itself
  ourselves` (possessive-or-pronoun), `plastic frozen ancient modern loud` (adjective); six fresh frames (`The farmer grows`, `The printer produces`, `The clinic
  treats`, `The agency recruits`, `Emma and Marco gathered … dry`, `Sam and Julia loaded … warm`); two prompt lists: the
  fresh tokens in the 48 exposed frames (1152, Y1) and in the 6 fresh frames (144, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/015-attention-pattern-token-local/run.py validate
uv run python experiments/015-attention-pattern-token-local/run.py freeze-confirmation
uv run python experiments/015-attention-pattern-token-local/run.py explore
uv run python experiments/015-attention-pattern-token-local/run.py lock
uv run python experiments/015-attention-pattern-token-local/run.py confirm
uv run python experiments/015-attention-pattern-token-local/run.py report
```

- `explore` (once, exposed pool): the 48 frames' reference states — the residuals before blocks 1 and 2 at every
  position up to the cue (locked) and the layer-1–2 pattern rows at the cue (identity I1); the 4884 recorded pairs
  re-measured with the patched residuals and pattern rows captured; replication of both extracts; the identities
  I1–I3 on every pair; the measured pattern change `ΔA`, its per-head term `P_h` and its read `c_ΔA`; Level 0, the
  axis-only alternative and the comparators evaluated from the reference states; the exposed statistics (token means
  and pairs of `ĉ_ΔA`, pooled entry R² and total-variation ratio per layer, self-weight R²), the descriptive rungs
  (query-only, self-key-only, additive, LayerNorm-linearised, frozen-pattern arrival), the diagonal decomposition.
- `lock` (weights, the Experiment 011/012 locks, the 48 locked reference states — no forward pass, no capture or
  intervention call path): the complete prediction table for the fresh tokens × exposed frames (per row the sixteen
  predicted rows `ΔÂ_h(p_c, ·)`, the sixteen predicted self-weight changes, `ĉ_ΔA` per layer and in total, the axis-only
  rows and read, the Level-0 diagonal-proportional rows and read, the predicted arriving change's read), the token
  means; the frozen floors; writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the fresh frames' reference prompts and cue pairs, checks I1, decides validity, computes the
  frame-conditional prediction table and writes it to the results state with its digest; **stage 2** starts only after
  that digest is re-read from disk, runs every fresh cue in both sets with the patched residuals and rows captured,
  checks I1–I3, and scores Y1 (strict prospective; token means and pooled entries over the 48 exposed frames), Y2
  (frame-conditional prospective, aggregate over the valid fresh frames) and Y3 (the axis-only alternative over both
  sets) against the two tables; applies the comparator rule.
- `report` renders `outputs/experiment-015/report.md`.

Boundaries: the invariant that every Level-0 prediction uses the token identity, the locked bases, axes and read, the
weights and the frame's *reference* residuals — never a residual, row or value of a patched run — is tested (the
prediction table is computed with every capture entry point disabled); the Level-1 identities I1–I3 and the
locked-state consistency of re-captured frames are enforced and any failure is an incident recorded with its commit; a
token needs three valid frames in a set to be scored there; no fresh cue prompt runs before its frame's stage-1
predictions are digested; which heads move for which cues is described after the scoring.

## Status

Implementation in progress; no scientific phase has run.

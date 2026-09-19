# Experiment 013: What Layers 1–2's Attention Adds — Does Frozen-Pattern Value Transport Predict the Residual of the Token-Local Model?

Implements the approved design
[`docs/superpowers/specs/2026-09-19-experiment-013-attention-paths-frozen-pattern-design.md`](../../docs/superpowers/specs/2026-09-19-experiment-013-attention-paths-frozen-pattern-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-19-experiment-013-attention-paths-frozen-pattern-plan.md`](../../docs/superpowers/plans/2026-09-19-experiment-013-attention-paths-frozen-pattern-plan.md).
Prospective, zero-parameter, residual-predicting: Experiment 012's frozen token-local MLP model is taken as given, and
the candidate for the residual it left is the computation it omitted — the frame's own base state and the sixteen
heads of layers 1–2 at the cue position with their attention patterns held at the reference. Under the E-patch only
the cue position's residual changes at block 1's input, so each head's output change splits exactly into a value path
with the reference pattern (computable from the reference prompt) and a pattern-change path (measured as the
remainder); the approximation enters only when the frozen-pattern contribution is propagated into block 2's MLP.

## Inputs

- Exposed pool (calibration record only; the tolerances are frozen constants): the 111 exposed cue tokens (Experiment
  012's 87 and its 24 confirmed) and the 36 exposed frames (its 30 and its 6), with the 80 nouns; a committed extract
  of Experiment 012's per-pair ledger ([`inherited/experiment-012-pair-ledger.json`](inherited/experiment-012-pair-ledger.json),
  2724 pairs, replicated within `1e-6`, the frozen 012 model reproduced within `1e-9`); the Experiment 011 lock's axes
  and read weight and the Experiment 012 lock's model (base states, read, defined templates), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-19 by tokenizer rules before any Experiment 013 model output, content
  sha256 `25f891e0a52da6cf60cd428f191010bf5cdc02d279b495b699f8cecc3e6c8f06`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `same own last next first` (determiner-like), `zero thousand million billion
  trillion` (numeral), `limited surplus endless plenty vast` (quantity), `whom someone nobody everyone`
  (possessive-or-pronoun), `tall thick quiet broken wide` (adjective); six fresh frames (`The pantry stocks`, `The
  courier delivers`, `The audit examines`, `The podcast discusses`, `Mateo and Ines washed … clean`, `Sofia and Anders
  bought … loose`); two prompt lists: the fresh tokens in the 36 exposed frames (864, Y1) and in the 6 fresh frames
  (144, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/013-attention-paths-frozen-pattern/run.py validate
uv run python experiments/013-attention-paths-frozen-pattern/run.py freeze-confirmation
uv run python experiments/013-attention-paths-frozen-pattern/run.py explore
uv run python experiments/013-attention-paths-frozen-pattern/run.py lock
uv run python experiments/013-attention-paths-frozen-pattern/run.py confirm
uv run python experiments/013-attention-paths-frozen-pattern/run.py report
```

- `explore` (once, exposed pool): the 36 frames' reference states (residuals at every position `≤ p_c` before blocks
  1 and 2, the pattern rows at `p_c`; the cue position's residuals and the sixteen self-attention weights are locked);
  the 2724 recorded pairs re-measured with the patched pattern rows and head outputs captured; replication of the
  Experiment 012 extract; the per-head OV identity in every pair; Levels 1–2 and the ladder for every pair; the
  calibration record (exposed RMSEs against the design values 0.0233 / 0.0355, an incident beyond 0.001); descriptive
  statistics, the compensation cases, `thy`.
- `lock` (weights, the Experiment 011/012 locks, and the 36 locked reference states — no forward pass, no capture or
  intervention call path): the complete prediction table for the fresh tokens × exposed frames (per row `ĉ_012`, the
  base-point term, the frozen-attention term, `r̂`, `ĉ_M`, `ĉ_H`, `ĉ_L`, the sixteen head terms) and the token means;
  the frozen `τ_r = 0.070`, `τ_A = 0.106`, the floors; writes `candidate-lock.json` and `candidate-predictions.md`.
  Installing them as `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the fresh frames' reference prompts and their cue pairs (exposed tokens), decides validity,
  computes the frame-conditional prediction table for every fresh token and writes it to the results state with its
  digest; **stage 2** starts only after that digest is re-read from disk, runs every fresh cue in both sets with the
  internals, residuals and pattern rows captured, and scores Y1 (strict; token means over the 36 exposed frames), Y2
  (frame-conditional; token means over the valid fresh frames), and Y3 (attribution; pairs of both sets, with the
  degenerate-spread guard) against the two tables.
- `report` renders `outputs/experiment-013/report.md`.

Boundaries: no candidate is ever excluded on the basis of its own measured value; a token needs three valid frames
in a set to be scored there; Y2 needs four valid fresh frames and sixteen scored tokens; Experiment 012's identities,
the per-head OV identity, and the locked-state consistency of re-captured exposed frames are enforced and any failure
is an incident recorded with its commit; the lock phase's only access to the model is its parameters; no fresh cue
prompt runs before its frame's stage-1 predictions are digested; the compensation cases are listed exhaustively by the
frozen rule and never judged.

## Status — 2026-09-19: complete; outcome `RESIDUAL_PREDICTED_TOKENS | RESIDUAL_PREDICTED_FRAMES_CONDITIONAL | ATTRIBUTION_PREDICTED`

The lock was installed byte-identical to the Tier A candidates and committed by hand (`c01126f`; lock content sha256
`ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`, predictions sha256
`d7928546bc2e10200672090f3d6d570afedf83eed938a5776dccc7a0eadecd55`), the read-only post-lock review passed
(hashes, scientific-path immutability since `cc14911`, `validate_lock`, frozen tolerances and floors, complete
pointwise predictions, runtime equal to the explore record, no overlap with executed prompts), the reviewer signed
off, and the single `confirm` ran at `c01126f` on a clean tree (results state sha256
`648aae5d74537de1291d0d8ed7cdc285101602a11d4178e64289c81c36708efd`). Every locked prediction was reproduced from the
weights, the Experiment 011/012 locks and the locked reference states before any fresh prompt (max difference 0.0).
**Stage 1** ran only the six fresh frames' reference prompts and cue pairs, found all six valid (plural-cue head change
1.24–3.12, cue effect 79/80), computed the 144 frame-conditional predictions and digested them
(`ef30588cace646c886ae10fad8b610e20174ff4465867139350b38d6920102b5`); **stage 2** started only after re-reading that
digest from disk and ran the 24 fresh cues in the 36 exposed frames (864 pairs) and in the six fresh frames (144
pairs). The final report is copied verbatim to [`evidence/final-report-2026-09-19.md`](evidence/final-report-2026-09-19.md).
Ledger after confirm: 1134 prompt keys, 80 noun keys. Experiment 013 is closed; nothing is amended or rerun.

- **Identities:** the per-head OV split held in all 1008 fresh pairs (max relative error 8.4e-7); Experiment 012's
  identities held (ρ identity 1.0e-7, P1 cross-check 4e-16, neuron sum 2e-8); every re-captured exposed reference state
  matched its locked copy and every fresh frame's state matched its stage-1 digest.
- **Y1 — residual prediction, strict boundary (24 new cues × 36 exposed frames, token means): PASS on all three
  floors.** The committed `r̂` predicts the measured residual of the frozen 012 model with Spearman **0.877** (floor
  0.80), MAE **0.021** (τ_r 0.070), **R² 0.668** (floor 0.50), bias 0.000; the residual's spread on this set is small
  (sd 0.046). Pairs: 0.875 / R² 0.79 over 864. The committed calls: `surplus` −0.14 → **−0.15** (the largest residual
  measured, as predicted); `thick` +0.06 → +0.07 and `quiet` +0.06 → +0.05 (the other end); the large numerals were
  predicted to be over-shot by the 012 model and were, but by less than committed (`million` −0.12 → −0.06, `billion`
  −0.09 → −0.04, `trillion` −0.08 → −0.06, `thousand` −0.07 → −0.03); `plenty` −0.07 → −0.06. The committed template
  pattern held exactly: predicted per-pair means cardinal +0.04 / quantifier −0.05 / coordinated-adjective −0.04,
  measured +0.04 / −0.05 / −0.04. Signs agreed for 19 of 24 token means (the misses are near zero: `last` −0.01 →
  −0.07, `nobody` +0.01 → −0.02, `everyone`, `first`, `same` within ±0.03). Beside it, never judged: the 012 model
  alone on these cues 0.957 / R² 0.883 against `c_L`, the full model 0.992 / R² 0.966; the base-point term alone
  explains none of the residual here (R² −0.06) — the frozen-pattern attention term carries the prediction.
- **Y2 — residual prediction, frame-conditional (24 new cues × 6 previously untested frames, predictions digested at
  stage 1 from each frame's reference state): PASS.** Spearman **0.919**, MAE **0.022**, **R² 0.747**, bias +0.009
  (spread sd 0.054); pairs 0.930 / R² 0.89 over 144. `thick` +0.13 → +0.12, `quiet` +0.11 → +0.09, `vast` +0.10 →
  +0.10, `wide` +0.09 → +0.07, `surplus` −0.08 → −0.10; the numerals again over-shot in size (−0.08..−0.04 → −0.03..
  −0.02). Template means predicted cardinal +0.14 / quantifier −0.05 / coordinated −0.01, measured +0.14 / −0.05 /
  −0.03. The 012 model alone 0.949 / R² 0.838; the full model 0.983 / R² 0.964; base-point alone R² 0.33.
- **Y3 — attribution (1008 pairs of both sets): PASS.** The predicted direct head part against the measured one:
  Spearman **0.838** (floor 0.70), MAE **0.030** (τ_A 0.106), R² 0.70, spread sd 0.069 (guard 0.017, not degenerate);
  the predicted MLP part 0.986 (floor 0.90), R² 0.97; per head 0.66–0.95; the attention-input term 0.827 / R² 0.68.
- **Compensation cases (descriptive, all listed in the report):** 236 fresh pairs satisfy the frozen rule; the measured
  `c_M` and `c_H` carry the predicted signs in **95%** (225). The eleven misses are all quantifier-frame pairs of
  `thousand`, `million`, `billion`, `trillion`, `vast` and `broken` whose measured `c_H` sits at 0.00–0.02 against a
  predicted −0.05 to −0.09 — the heads' write shrank to zero rather than flipping.
- **Ladder (descriptive):** per pair, Y1 base-point −0.037, frozen-attention +0.020, remainder −0.000 (|.| 0.035:
  direct pattern change 0.031, through block 2's MLP 0.029, layer-2 heads' arrival 0.005); Y2 base-point +0.011,
  frozen-attention +0.018, remainder −0.009 (|.| 0.029). The pattern-change remainder is the one term left
  unaccounted, and it is small.

What this settles and what it does not. Stated at its safe strength: a mechanistically derived frozen-pattern attention
computation — the frame's own base state and the sixteen heads' value paths with their reference patterns, propagated
through block 2's MLP and onto the read direction — prospectively explains a large part of the residual the token-local
012 model left, on 24 never-seen cues both in the 36 exposed frames (strictly prospective) and in six previously
untested frames (conditional on their reference states), and it predicts the split between the MLP path and the heads'
direct path at the pair level, including 95% of the preidentified compensation cases. The account of the forward pass
from the layer-0 encoding to the head's input now reads: 012 prediction + base-point mismatch + frozen-pattern
attention transport + its nonlinear MLP₂ consequence + direct head writes ≈ the measured layers-1–2 correction, with
the remaining error (|.| ≈ 0.03 per pair) the attention-pattern-change term. Limits: the residual is a small quantity
and R² was 0.67 / 0.75, not higher — the numerals' over-shoot was predicted in direction but over-estimated in size, and
near-zero residuals had unreliable signs; the reference patterns are taken from the reference run, not explained; five
lexical classes, three templates, this checkpoint. C002 is unchanged (out of scope here).

### Tier A (2026-09-19, for the record)


`explore` ran once on protocol/code commit `cc14911` (run `db79dab2e9f63182`, results state sha256
`3372b084d996c727c0bd11323e58510334038dfee24e2553c16aba1612fc48bc`; A0 passed). The report is copied to
[`evidence/exploration-report-2026-09-19.md`](evidence/exploration-report-2026-09-19.md) and the candidate prediction
table to [`evidence/candidate-predictions-2026-09-19.md`](evidence/candidate-predictions-2026-09-19.md). Ledger: 108
prompt keys (the 36 exposed frames' cue and reference prompts), 80 noun keys; no fresh prompt ran, and the lock phase
ran no prompt at all.

- Replication of Experiment 012's 2724 recorded pairs: exact for every field; the frozen 012 model reproduced to 0.0.
  The recomputed axes, read weight and denominators equal the Experiment 011 lock; the per-head OV identity held in
  all 2724 pairs (max relative error 8.7e-7); Experiment 012's identities held (ρ identity 1.1e-7, P1 cross-check
  6.6e-16, neuron sum 2e-8).
- **Calibration record (the tolerances themselves are frozen constants):** recomputed exposed RMSEs 0.0233 (token-mean
  residual) and 0.0355 (pair-level `ĉ_H − c_H`) — the design values to four decimals — so `τ_r = 0.070` and
  `τ_A = 0.106` stand. Over the 111 exposed tokens the frozen-pattern model predicts the residual of the 012 model with
  Spearman 0.946, MAE 0.018, R² 0.870, bias +0.004 (residual spread sd 0.065; the base-point term alone R² 0.39); the
  full model reaches Spearman 0.987, R² 0.976 against `c_L` (the 012 model: 0.924, R² 0.78; the 012 model at the
  frame's own base: 0.949, R² 0.89). Pairs: residual 0.885 / R² 0.80; `ĉ_H` vs `c_H` 0.856 / R² 0.73 (spread sd
  0.068, far above the 0.017 guard); `ĉ_M` vs `c_M` 0.986 / R² 0.97; the attention-input term 0.848 / R² 0.75; per head
  0.67–0.95. Ladder means: base-point −0.041, frozen-attention +0.010, remainder −0.002 (|.| 0.036 per pair, 0.018 per
  token: direct pattern change |.| 0.028, through block 2's MLP |.| 0.029, the layer-2 heads' arrival term |.| 0.005).
- Compensation cases by the frozen rule: 742 exposed pairs, measured signs as predicted in 97%. `thy`, frame by frame
  (measured `c_M`, `c_H` against predicted): cardinal (−0.16, −0.07) vs (−0.15, −0.05) and (−0.04, −0.05) vs
  (−0.07, −0.02); quantifier (+0.34, −0.28) vs (+0.32, −0.20) and (+0.18, −0.12) vs (+0.16, −0.11); coordinated (−0.11,
  −0.05) vs (−0.16, −0.02) and (−0.10, −0.02) vs (−0.13, −0.01) — the compensation reproduced without fitting.
- **Candidate lock** `outputs/experiment-013/candidate-lock.json`, content sha256
  `ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`; predictions artifact
  `candidate-predictions.md`, sha256 `d7928546bc2e10200672090f3d6d570afedf83eed938a5776dccc7a0eadecd55`: 864 rows
  (24 fresh tokens × 36 exposed frames), each with `ĉ_012`, the base-point term, the frozen-attention term, `r̂`, `ĉ_M`,
  `ĉ_H`, `ĉ_L` and the sixteen head terms. Committed token-mean residual predictions `r̂` (the 012 model's error the
  frozen-pattern model expects): `surplus` −0.14, `million` −0.12, `billion` −0.09, `trillion` −0.08, `plenty` −0.07,
  `thousand` −0.07 — the 012 model is predicted to over-shoot the large numerals and two quantity words; `thick` +0.06,
  `quiet` +0.06, `whom` +0.05, `next` +0.03, `broken` +0.03 on the other side; the rest within ±0.04. Predicted
  template means: cardinal +0.04, quantifier −0.05, coordinated-adjective −0.04. The spread of the committed `r̂` over
  the 24 token means is 0.054, so Y1's explained-variance floor is a demanding test of a small quantity. 198 of the 864
  rows are compensation cases by the frozen rule (predicted `ĉ_M` and `ĉ_H` of opposite sign, each above 0.05), spread
  over 22 tokens; every one will be listed with its measured signs. The Y2 table (fresh frames) does not exist yet: it
  is computed at `confirm` stage 1 from each fresh frame's reference state and digested before any fresh cue prompt.

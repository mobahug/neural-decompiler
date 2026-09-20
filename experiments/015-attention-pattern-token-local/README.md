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

Scoring conventions fixed before Tier A: Y2 is an aggregate claim over the valid fresh frames (a token needs three
valid frames to be scored; the pooled entry R² is over every scored pair of the set), and the six frames are reported
individually regardless of the outcome; the comparator rule is descriptive and never a gate; the Level-1 identities
count toward no floor.

## Status — 2026-09-20: complete; outcome `PATTERN_CHANGE_PREDICTED_TOKENS | PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`

Tier A (`explore`) ran once at `407dd40` on the 4884 recorded pairs (159 tokens, 48 frames): both extracts replicated
(deviation 0.0 and 2.8e-17), the Level-1 identities held on every pair (I1 ≤ 1.8e-5, I2 3.1e-6, I3 9.4e-7), Level 0
reached token-mean Spearman 0.963 / R² 0.787 and pooled entry R² 0.919 (layer 1) / 0.720 (layer 2), the axis-only
alternative R² −2.08 / entries 0.123. The lock was installed byte-identical to the Tier A candidates and committed by
hand (`5ed51c9`; lock content sha256 `6987704901dccf8c3638c7f770f64d77044320af5a1726e5c517130bc7423ebb`, predictions
sha256 `35efa1e9b8d2575cf69fcb4f5a73baaf0e84ad3548edff188d25d6f933929fe3`; the candidate predictions are copied to
[`evidence/candidate-predictions-2026-09-20.md`](evidence/candidate-predictions-2026-09-20.md)), the read-only post-lock
check passed (hashes, byte identity, scientific-path immutability since `407dd40`, `validate_lock`, frozen floors, the
complete 1152-row table reproduced with zero difference, runtime equal to the explore record, no overlap with executed
prompts), the reviewer cleared the install, and the single `confirm` ran at `5ed51c9` on a clean tree (results state
sha256 `70a8f6699c445f273098d9e429665854cca3afb5fb67ad0651bc52dec7333ae9`). **Stage 1** ran only the six fresh frames'
reference prompts and cue pairs — all six valid (plural head change 1.4–3.0 σ_T, cue effect 79/72) — computed the 144
frame-conditional predictions and digested them (`2dcc809e5cfabbd2648414b9471835c4201d4f2a6f5f862db0c7c66d960c359e`);
**stage 2** started only after re-reading that digest from disk and ran the 24 fresh cues in the 48 exposed frames
(1152 pairs) and the six fresh frames (144 pairs). The final report is copied verbatim to
[`evidence/final-report-2026-09-20.md`](evidence/final-report-2026-09-20.md). Ledger after confirm: 1458 prompt keys,
80 noun keys. Experiment 015 is closed; nothing is amended or rerun.

- **Identities (Level 1, checks only):** on all 1296 fresh pairs the program reproduced the captured rows (I1 ≤ 1.7e-5),
  the exact chain reproduced the captured residual before block 2 (I2 3.4e-6 relative), Experiment 013's split held
  (I3 1.0e-6), and Level 1 accounted for the measured `c_L` to 2e-6. Every re-captured reference state matched its
  locked copy or stage-1 digest.
- **Y1 — strict prospective (24 fresh cues × 48 exposed frames): PASS on all four floors.** Token means `ĉ_ΔA` vs
  `c_ΔA`: Spearman **0.986** (floor 0.80), **R² 0.895** (floor 0.50), MAE 0.0070 on a spread of 0.025; pooled entry R²
  **0.908** at layer 1 and **0.700** at layer 2 (floors 0.50; total-variation ratios 0.26 / 0.47); the sixteen
  self-weight changes pooled R² 0.79; pairs 0.883 / 0.749.
- **Y2 — frame-conditional prospective, aggregate (24 fresh cues × 6 new frames): FAIL on two of four floors.**
  Spearman **0.857** (pass) but **R² 0.490** (floor 0.50); pooled entry R² **0.888** at layer 1 (pass) but **0.423** at
  layer 2 (floor 0.50). The six frames individually (Level 0 entry R² layer 1 / layer 2; `ĉ_ΔA` vs `c_ΔA` R²):
  `cardinal-015-1` 0.72 / 0.47; 0.62 — `cardinal-015-2` 0.94 / 0.62; 0.77 — `coordinated-adjective-015-1` 0.94 / 0.67;
  0.70 — `coordinated-adjective-015-2` 0.94 / 0.74; 0.96 — `quantifier-015-1` 0.96 / 0.66; 0.64 — `quantifier-015-2`
  (`The agency recruits`) **0.79 / −0.27; −0.14**, with a read bias of −0.028: in that frame the ordinals (`tenth`
  predicted −0.045, measured +0.064; `ninth` −0.047 / +0.029), `scant` (+0.060 / +0.128) and `plastic` (+0.011 /
  +0.078) moved the layer-2 rows the other way from the template-mean prediction. The frame is valid by every frozen
  criterion and Level 1 is exact on it, so the miss is the reduced description's: the template-mean base does not
  carry this frame's cue-position state at layer 2 (the ladder makes the same point — Experiment 013's own-base
  frozen-pattern `c_L` reaches R² 0.94 on the fresh frames where Level 0's template-base `c_L` reaches 0.69). Y2 was
  preregistered as an aggregate claim over the valid fresh frames and the aggregate missed; the family label stands.
  Descriptive only, after the fact: over the other five frames the token means would give R² 0.82 — that is not the
  frozen criterion and changes nothing.
- **Y3 — the number-axis alternative is rejected** on the 48 token means and 1296 pairs of both sets: token-mean R²
  **−1.16** (Spearman −0.32; rejected iff below 0.30) and pooled entry R² **0.040** (rejected iff below 0.30). The
  axis-only program predicted a positive pattern-change read (+0.015 to +0.031) for every fresh cue; 17 of the 24
  measured Y1 token means are negative, and Level 0 had committed the negative sign for every one of them (20 predicted
  negative in all).
- **Comparator rule (descriptive, predeclared):** over the 1296 fresh pairs Level 0's pooled entry R² exceeds the
  Level-0 diagonal-proportional comparator's by **0.255** (layer 1: 0.906 vs 0.651) and **0.171** (layer 2: 0.676 vs
  0.505), both ≥ 0.10, so the report may say that Level 0 decodes the query/key interaction beyond the self-logit
  effect. The oracle diagonal-proportional comparator (the exact self weight, proportional redistribution) reaches
  0.74 / 0.81 — most of the row change is a change of self-attention, but the off-diagonal coupling of the token-local
  query change to the frame's keys carries the rest, and Level 0 predicts it.
- **Ladder of decoded `c_L` on the fresh pairs (descriptive):** Experiment 012's token-local MLPs R² 0.725, Experiment
  013's own-base frozen-pattern model 0.938, Level 0 (bases, pattern computed) 0.779, Level 1 (exact) 1.000 — the
  scientifically loaded number is Level 0's, which deliberately discards the frame's own cue-position state; Level 1
  only certifies the program.
- **Rungs on the fresh pairs:** query-only 0.28 / 0.00, self-key-only 0.34 / −0.28, additive 0.37 / −0.26, LayerNorm
  linearised −0.51 / −1.32 (`‖Δx‖/‖x − μ‖` 0.99 / 0.84), frozen-pattern arrival 1.00 / 0.86 — no partial evaluation of
  the bilinear form survives the softmax, and the layer-1 pattern change does reach layer 2.
- **The heads, described only now.** Over the 24 fresh cues the largest measured self-weight changes are `L01.H01`
  (+0.20 on average: the cue attends more to itself), `L01.H02` (−0.135), `L02.H06` (−0.116), `L02.H00` (+0.094),
  `L02.H03` (+0.078); Level 0 reproduces every head's token-mean self-weight change with correlation ≥ 0.96 and the
  sign of 79–100% of them. Which heads move is a property of the weights and the frame the network reveals; no label
  is attached to any head.

What this settles and what it does not. Stated at its safe strength: given a frame's reference attention state, a
reduced query/key computation driven by the cue's encoding change alone — the cue position's query, key and value
changes at the Experiment 012 template-mean bases, applied to the frame's reference keys and logit rows — predicts most
of the cue-induced attention-pattern change of the sixteen layer-1–2 heads for 24 never-seen cues in the 48 exposed
frames (strict boundary), and the change's read on the head's direction with R² 0.90; in six new frames the same
reduced computation keeps the ordering and layer 1 but misses the frozen explained-variance floors at the aggregate,
one frame accounting for most of the shortfall. The number axis alone cannot account for the re-routing: the
axis-only program is wrong in sign for most cues. With the pattern change computed rather than supplied, the
cue-position computation through layers 1–2 is now written end to end as a weight-only program whose only frame input
is the reference run — exactly at Level 1, approximately at Level 0 — and the remaining gap between the two is base
sensitivity at layer 2. This is not "attention predicted from the token": the frame still supplies its reference keys,
values and logit rows; the experiment explains the change, not the attention state. Limits: one checkpoint, three
templates, five lexical classes, sixteen heads; Y2's failure is a frozen result of this protocol version and any
better reduced description (for example a frame-conditional base) is a new experiment.

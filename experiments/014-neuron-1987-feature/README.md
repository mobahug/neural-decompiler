# Experiment 014: What Drives Block-2 Neuron 1987 — A Compact Input-Side Predictor, Locked Before Interpretation

Implements the approved design
[`docs/superpowers/specs/2026-09-19-experiment-014-neuron-1987-feature-design.md`](../../docs/superpowers/specs/2026-09-19-experiment-014-neuron-1987-feature-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-19-experiment-014-neuron-1987-feature-plan.md`](../../docs/superpowers/plans/2026-09-19-experiment-014-neuron-1987-feature-plan.md).
Prospective, zero-parameter, feature-reconstructing: the neuron's computation — residual before block 2 → LayerNorm →
one input weight vector → GELU → one output vector onto the head's read direction — is evaluated exactly at the state
the previously decoded circuit predicts (the frame's reference residual plus Experiment 013's frozen-pattern arriving
change, never a measured patched residual), and the number-axis alternative is committed for rejection on the same
data. No semantic label is preregistered; the firing set is interpreted only after the scoring.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 135 exposed cue tokens (Experiment
  013's 111 and its 24 confirmed) and the 42 exposed frames (its 36 and its 6), with the 80 nouns; a committed extract
  of Experiment 013's per-pair `c_L`, `c_M`, `c_H`, `c_k` ([`inherited/experiment-013-pair-ledger.json`](inherited/experiment-013-pair-ledger.json),
  3732 pairs, replicated within `1e-6`); the Experiment 011 lock's axes and read weight, the Experiment 012 lock's
  model, and the Experiment 013 lock's reference states (checked against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-19 by tokenizer rules before any Experiment 014 model output, content
  sha256 `a28edf831413489b2c8ffd4710bba4d742d4e542348120b0cab90c4f3ab3796d`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `second third former latter only` (determiner-like), `dozens hundreds thousands
  millions billions` (numeral), `minimal infinite excess lesser considerable` (quantity), `anybody somebody everybody
  anyone` (possessive-or-pronoun), `narrow sharp smooth rough golden` (adjective); six fresh frames (`The bakery
  bakes`, `The library lends`, `The journal publishes`, `The committee approves`, `Omar and Lucia piled … neat`, `Farah
  and Tobias lifted … wet`); two prompt lists: the fresh tokens in the 42 exposed frames (1008, Y1) and in the 6 fresh
  frames (144, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/014-neuron-1987-feature/run.py validate
uv run python experiments/014-neuron-1987-feature/run.py freeze-confirmation
uv run python experiments/014-neuron-1987-feature/run.py explore
uv run python experiments/014-neuron-1987-feature/run.py lock
uv run python experiments/014-neuron-1987-feature/run.py confirm
uv run python experiments/014-neuron-1987-feature/run.py report
```

- `explore` (once, exposed pool): the 42 frames' reference states with the neuron's operating point (locked); the 3732
  recorded pairs re-measured with the residual before block 2 captured; replication of the Experiment 013 extract; the
  exact `Δpre`, `Δa`, firing and read term per pair; the predictor, its Jacobian form and the axis-only alternative
  evaluated from the reference states; the exposed statistics (Spearman, R², balanced accuracy with the majority
  baseline, MAE and the amplitude error among firing pairs), the source and axis / off-axis accounting, the remainders,
  the firing set by class.
- `lock` (weights, the Experiment 011/012 locks, the 42 locked reference states — no forward pass, no capture or
  intervention call path): the complete prediction table for the fresh tokens × exposed frames (per row `pre_ref`,
  `Δp̂re`, `Δâ`, the predicted firing, the Jacobian form with its source and axis / off-axis parts, the axis-only
  `Δp̂re`, `Δâ` and firing, the predicted term), the token means and the predicted firing counts; the frozen floors;
  writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as `preregistration-lock.json` and
  `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the fresh frames' reference prompts and cue pairs, decides validity, computes the
  frame-conditional prediction table and writes it to the results state with its digest; **stage 2** starts only after
  that digest is re-read from disk, runs every fresh cue in both sets with the residual before block 2 captured, and
  scores Y1 (strict; token means over the 42 exposed frames plus balanced firing accuracy over the pairs), Y2
  (frame-conditional; the valid fresh frames) and Y3 (the axis-only alternative over both sets) against the two tables.
- `report` renders `outputs/experiment-014/report.md`.

Scoring conventions fixed before Tier A: Y3's degenerate-spread guard is evaluated on the pooled pairs of both sets (a
class missing in one set alone does not make Y3 non-evaluable), and a token scored in Y2 by its three valid frames
feeds Y3 even if Y2's own frame precondition fails; the Jacobian form's activation statistics are reported beside the
predictor's and never judged.

Boundaries: the invariant that every prediction uses the *predicted* arriving change and never a patched residual is
tested (the prediction table is computed with every capture entry point disabled); no candidate is ever excluded on
the basis of its own measured value; a token needs three valid frames in a set to be scored there; Experiment 012's
identities, the neuron's activation identity against the block-2 ledger, and the locked-state consistency of
re-captured frames are enforced and any failure is an incident recorded with its commit; no fresh cue prompt runs
before its frame's stage-1 predictions are digested; the firing set's interpretation comes after the scoring.

## Status — 2026-09-19: complete; outcome `NEURON_FEATURE_PREDICTED_TOKENS | NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`

The lock was installed byte-identical to the Tier A candidates and committed by hand (`d008dae`; lock content sha256
`34b97174ebd68617f7a13a7d876f405761a8faaf42c6675ac8fa42de8c805d01`, predictions sha256
`a0b7448a4c42faee538b180794f95abf6fbf5e3049ab8358ba83875ce05b091e`), the read-only post-lock check passed (hashes,
scientific-path immutability since `bcb0bfe`, `validate_lock`, frozen floors, the complete table with its firing flags
reproduced exactly, runtime equal to the explore record, no overlap with executed prompts), the reviewer signed off, and
the single `confirm` ran at `d008dae` on a clean tree (results state sha256
`96a80d9cd129e7aa37496ee563b7511acbd75884120c46a42f682092acd9448b`). **Stage 1** ran only the six fresh frames'
reference prompts and cue pairs — all six valid, operating points −2.36 to +0.03 — computed the 144 frame-conditional
predictions (22 predicted firing pairs) and digested them (`102103ed6f746e5b24698c6c07441e7965b7fa04cf9a6a67206ed738522e217a`);
**stage 2** started only after re-reading that digest from disk and ran the 24 fresh cues in the 42 exposed frames
(1008 pairs) and the six fresh frames (144 pairs); no patched residual ever entered the predictor. The final report is
copied verbatim to [`evidence/final-report-2026-09-19.md`](evidence/final-report-2026-09-19.md). Ledger after
confirm: 1296 prompt keys, 80 noun keys. Experiment 014 is closed; nothing is amended or rerun.

A note on nomenclature, raised at the sign-off: in the report's classification lines, "majority baseline" is the
*ordinary* accuracy of an always-majority-class classifier (`max(firing, non-firing) / n`; 0.79 on Y1's pairs); the
balanced accuracy is `½(sensitivity + specificity)` and is computed separately — an always-one-class classifier scores
0.5 on it. The line is descriptive; the frozen floor is on the balanced accuracy.

- **Identities:** the neuron's activation identity against the block-2 ledger and Experiment 012's identities held in
  all 1152 fresh pairs (ρ identity 1.0e-7, P1 cross-check 4e-16, neuron sum 2e-8); every re-captured reference state
  matched its locked copy or stage-1 digest.
- **Y1 — strict boundary (24 new cues × 42 exposed frames): PASS on all three floors.** Token means `Δâ` vs `Δa`:
  Spearman **0.950** (floor 0.80), **R² 0.970** (floor 0.50), MAE 0.040, bias +0.002 (spread sd 0.45). Pairs: Spearman
  0.979, R² 0.940, MAE 0.060; `Δp̂re` vs `Δpre` R² 0.954. **Balanced firing accuracy 0.923** (floor 0.90): sensitivity
  0.852, specificity **0.994**; 210 of 1008 pairs fire (majority baseline 0.79); 31 false negatives, 5 false positives.
  Amplitude error among firing pairs 0.20 on a mean `Δa` of 0.99.
- **Y2 — frame-conditional (24 new cues × 6 previously untested frames, predictions digested at stage 1): PASS.**
  Spearman **0.910**, **R² 0.957**, MAE 0.043, bias −0.009; pairs 0.957 / 0.937. **Balanced accuracy 0.927** (sensitivity
  0.870, specificity 0.983; 23 of 144 pairs fire; 3 false negatives, 2 false positives).
- **Y3 — the number-axis alternative is rejected** on the same 48 token means and 1152 pairs: R² **−1.41** (rejected iff
  below 0.30), balanced accuracy **0.601** (rejected iff below 0.70; sensitivity 0.87, specificity **0.33** — it fires for
  818 of 1152 pairs against 233 measured). The predictor on the same token means: 0.977 / R² 0.965.
- **The committed calls.** The three strong predictions fired in every frame: `dozens` +1.49 → **+1.27** (42/42 frames),
  `hundreds` +1.19 → **+0.99** (39/42), `thousands` +1.09 → **+1.16** (42/42). The eighteen predicted vetoes held without
  exception: `rough`, `smooth`, `golden`, `sharp`, `narrow`, `second`, `third`, `former`, `latter`, `only`, `excess`,
  `lesser`, `minimal`, `considerable`, `everybody`, `anybody`, `anyone`, `somebody` — predicted −0.09 to −0.12, measured
  −0.06 to −0.13, **0 firing frames of 756**; `considerable`, whose number-axis part alone would have pushed the
  preactivation up by +1.7 (axis-only `Δâ` +1.25), stayed at −0.13. The committed borderline cases came out borderline,
  with the predictor under-shooting them: `infinite` +0.57 → +0.63 (fires 57% → **69%** of frames), `millions` +0.53 →
  +0.70 (50% → **74%**), `billions` +0.44 → +0.59 (33% → **64%**). Every one of the 31 Y1 false negatives is a
  `billions`, `millions`, `infinite` or (once) `thousands` frame in which the neuron fired while the prediction sat just
  below 0.5 (for example `billions` in `cardinal-1`: predicted 0.46, measured 0.62); the 5 false positives are
  `hundreds` and `infinite` frames the other way. The under-shoot is template-shaped: in the coordinated-adjective
  frames `millions` and `billions` fired in 6 and 7 of 14 frames against 1 predicted, in the cardinal and quantifier
  frames prediction and measurement agree closely. So the model knew *which* cues sit at the boundary and got their
  frame-level firing right in most frames, but placed the plural numerals a little too low.
- **Accounting on the fresh set (mean |Jacobian contribution|):** `ΔE` 0.65, `MLP₁` 0.30, the layer-1 heads 0.47
  (negative for nearly every cue); number-axis part 1.18, off-axis part 1.29. The Jacobian form itself reaches R² 0.75
  per token but balanced accuracy 0.955 (the exact LayerNorm matters for the amplitude, not for the firing). Remainders:
  pattern change |.| 0.165, LayerNorm linearization |.| 0.21–0.26 (preactivation units).
- **The firing set, interpreted only now.** Exposed and fresh together, the neuron fires for the plural cues, every
  cardinal numeral from `three` upward, `dozen`/`dozens`, `hundreds`, `thousands`, `few`, `fewer`, `many`, `multiple`,
  `numerous`, `countless`, `myriad`, `various`, and — at the boundary — `millions`, `billions`, `infinite`; it does not
  fire for any determiner (`these`, `those`, `both`, `all`, `some`, `most`, `every`, `second`, `third`, `only`, …), any
  possessive or pronoun, any adjective, `zero`, or the graded quantity words `excess`, `lesser`, `minimal`,
  `considerable`, `limited`, `surplus`. In the network's own terms, the cues that fire are those whose predicted
  arriving change carries the number-axis excitation *without* the off-axis veto.

What this settles and what it does not. Stated at its safe strength: block-2 neuron 1987's cue-induced activation
change is prospectively reconstructed — for 24 never-seen cues in the 42 exposed frames and in six previously untested
frames — from its own weight-defined input direction applied, through the exact LayerNorm at the frame's reference
operating point, to the upstream state change the previously decoded circuit predicts; and a simple number-axis model
fails on the same data because off-axis structure in that predicted change suppresses the activation for most number-
and plural-associated cues. The claim is about the computation, not a label: the firing set is described afterwards
and is consistent with a count-magnitude reading, but the experiment tests the mechanism (excitation along the number
axis, veto from off-axis structure, one neuron), not the gloss. Limits: the predictor under-shoots the boundary cases,
most in the coordinated-adjective template; it inherits Experiment 013's approximations (the pattern-change remainder
is |.| 0.17 in preactivation units); one neuron, five lexical classes, three templates, this checkpoint. C002 is
unchanged (out of scope here).

### Tier A (2026-09-19, for the record)


`explore` ran once on protocol/code commit `bcb0bfe` (run `00eef3b8bb95c8b8`, results state sha256
`fbae608d39f21599f084b9a4180e69960f250bbf4aafae07a4af32b0bfee672c`; A0 passed). The report is copied to
[`evidence/exploration-report-2026-09-19.md`](evidence/exploration-report-2026-09-19.md) and the candidate prediction
table to [`evidence/candidate-predictions-2026-09-19.md`](evidence/candidate-predictions-2026-09-19.md). Ledger: 126
prompt keys (the 42 exposed frames' cue and reference prompts), 80 noun keys; no fresh prompt ran, and the lock phase
ran no prompt at all.

- Replication of Experiment 013's 3732 recorded pairs (`c_L`, `c_M`, `c_H`, all 18 components): exact. The 36 reference
  states of the Experiment 013 lock re-captured bitwise; Experiment 012's identities and the neuron's activation
  identity against the block-2 ledger held in every pair.
- **The neuron.** `cos(γ₂ ⊙ W_in[:, 1987], d̂_E) = 0.241`; its output reads +0.204 (cardinal, coordinated) / +0.222
  (quantifier) of the plural cue's signal per unit of activation; `b_in = −0.66`. Operating points: reference
  preactivation −1.60 to +0.49 over the 42 frames (template means cardinal +0.10, quantifier −0.85, coordinated
  −0.17); reference activation at most 0.34 — **below the firing threshold 0.5 in every frame**.
- **Exposed record (135 tokens, 3732 pairs; the floors are frozen constants):** the predictor — the exact LayerNorm and
  the neuron's input weights at Experiment 013's frozen-pattern predicted state — gives `Δâ` vs `Δa` Spearman **0.985**,
  R² **0.989**, MAE 0.049 per token (0.976 / 0.963 / 0.094 per pair; `Δp̂re` vs `Δpre` R² 0.968); **balanced firing
  accuracy 0.989** (sensitivity 0.991, specificity 0.987; 1160 of 3732 pairs fire; majority baseline 0.689); amplitude
  error among firing pairs 0.22 on a mean `Δa` of 1.66. The axis-only alternative: R² 0.166, balanced accuracy 0.611
  (sensitivity 0.94, specificity 0.28 — it fires for almost everything). The Jacobian form: R² 0.913 per token,
  balanced accuracy 0.979 (descriptive). Accounting (mean |Jacobian contribution|): `ΔE` 1.08, `MLP₁` 0.33, the layer-1
  heads 0.41 (negative in 94% of pairs); the number-axis part 1.35, the off-axis part 1.14. Remainders: pattern change
  |.| 0.175, LayerNorm linearization |.| 0.229 (in preactivation units, spread 1.3).
- **Measured firing set (≥ half of frames) = predicted firing set, exactly:** the plural cues `two`, `several`; every
  cardinal numeral from `three` upward (`three`–`ten`, `eleven`–`nineteen`, `twenty`–`ninety`, `hundred`, `thousand`,
  `million`, `billion`, `trillion`); `dozen`, `few`, `fewer`, `many`, `multiple`, `numerous`, `countless`, `myriad`,
  `various`. Not firing: every determiner (including `these`, `those`, `both`, `all`, `some`, `most`, `every`), every
  possessive and adjective, and `zero`. The neuron's term for firing cues averages +0.35 of the plural signal; its share
  of the block-2 read change among firing pairs is 1.6 — the rest of block 2's MLP pushes back.
- **Candidate lock** `outputs/experiment-014/candidate-lock.json`, content sha256
  `34b97174ebd68617f7a13a7d876f405761a8faaf42c6675ac8fa42de8c805d01`; predictions artifact
  `candidate-predictions.md`, sha256 `a0b7448a4c42faee538b180794f95abf6fbf5e3049ab8358ba83875ce05b091e`: 1008 rows
  (24 fresh tokens × 42 exposed frames), each with the operating point, `Δp̂re`, `Δâ`, the predicted firing, the
  Jacobian form with its source and axis / off-axis parts, the axis-only prediction, and the predicted term. Committed
  firing predictions: the neuron fires for **`dozens` (+1.49), `hundreds` (+1.19), `thousands` (+1.09)** in every
  frame; **`infinite` +0.57 (57% of frames), `millions` +0.53 (50%), `billions` +0.44 (33%)** are the committed
  borderline cases — the plural `millions`/`billions` predicted far weaker than the exposed `million`/`billion`; the
  other eighteen tokens are predicted not to fire in any frame (`Δâ` −0.09 to −0.12), including `considerable`,
  `excess`, `lesser`, `minimal`, `second`, `third`, `only`. 184 of the 1008 pairs are predicted to fire against 736 for
  the axis-only alternative. The committed mechanism is visible in the table: for `rough`, `smooth`, `golden`,
  `second`, `former` the number-axis part of the arriving change pushes the preactivation up by +1.2 to +1.4 and the
  off-axis part pulls it down by −1.9 to −2.4; for `dozens` both parts are positive (+1.65, +0.67). The Y2 table (fresh
  frames) does not exist yet: it is computed at `confirm` stage 1 from each fresh frame's reference state and digested
  before any fresh cue prompt.

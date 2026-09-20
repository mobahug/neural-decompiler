# Experiment 017: The Transport Head's Own Pattern Change — Predicting `L03.H04`'s Attention Row and Its End-to-End `ΔT` from the Decoded Upstream State

Implements the approved design
[`docs/superpowers/specs/2026-09-20-experiment-017-transport-head-pattern-design.md`](../../docs/superpowers/specs/2026-09-20-experiment-017-transport-head-pattern-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-20-experiment-017-transport-head-pattern-plan.md`](../../docs/superpowers/plans/2026-09-20-experiment-017-transport-head-pattern-plan.md).
Prospective, zero-parameter, chain-closing: Experiments 010–016 decoded how a cue word's weight-only encoding change
`ΔE` becomes the transport head's *input*; the head `L03.H04` itself was described with a frozen attention pattern
since Experiments 009, 011 and 013. This experiment predicts the head's own attention-row change at the transport
position `p_t` and its output change `ΔT` along the locked axis `d̂_T` from the decoded chain alone — Experiment 016's
Level 0-F at the cue position, block 2's MLP at the frame's operating point (channel D: the reference pre-activations
through the exact GELU, the change predicted), one propagation step to `p_t` in coordinated frames, and the head's own
query, key and value changes reduced as Experiment 016 reduced layers 1–2 (the template direction of the change at a
layer-3 template base locked at `explore`, the frame's scale, the frame's own reference operands). Level 0's only
cue-specific input is `ΔE`: no residual, LayerNorm statistic, attention row, MLP state or head state of a fresh cue
forward pass enters it at either position. The three head objects are the row change `ΔA_H(p_t, ·)`, the pattern
term `Π` (the changed row carrying the changed values) and the value term `F` (the reference row carrying the value
change — the frozen-pattern account, the alternative committed for rejection on the cue-final pairs), with
`ΔT = F + Π` exactly. Level 1 — the exact chain from the reference state and `ΔE` — is an identity (I4–I7) and enters
no floor; Level 0, the exact-head rung, the two ablations (`−D`: block 2's operating point at the template base at
`p_c`; template head: Experiment 015's reduction at the head) and Level 1 are switches on one code path.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 207 exposed cue tokens (Experiment
  016's 183 and its 24 confirmed) and the 66 exposed frames (its 54 and its 12), with the 80 nouns; a committed extract
  of Experiment 016's per-pair `c_ΔA` (measured and Level 0-F), sixteen self-weight changes, `c_L`, `c_M`, `c_H` and
  decoded `c_L` ([`inherited/experiment-016-pair-extract.json`](inherited/experiment-016-pair-extract.json), 7764
  pairs, replicated within `1e-6`), carrying also the digests of the twelve 016-fresh frames' stage-1 reference
  states; the Experiment 011 lock's axes and read weight, the Experiment 012 lock's bases and the Experiment 016 lock's
  reference states (checked against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-20 by tokenizer rules before any Experiment 017 model output, content
  sha256 `07fc6ea974e22300d2e4ea87a517204f9474351cddde010e69acc9ce9e969c8e`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `further usual adjacent nearby leading` (determiner-like), `twice once score trio`
  (ordinal-or-numeral), `finite partial whole entire complete` (quantity), `none them us you` (possessive-or-pronoun),
  `violet metal paper leather blunt costly` (adjective); twelve fresh frames, four per template (`The bakery sells`,
  `The museum displays`, `The fleet carries`, `The orchard bears`; `The report lists`, `The lecture covers`, `The survey
  counts`, `The archive holds`; `Clara and Dev packed … hot`, `Anna and Paul carried … cool`, `Eva and Tom hauled … damp`,
  `Alice and Ben served … crisp`); two prompt lists: the fresh tokens in the 66 exposed frames (1584, Y1) and in the 12
  fresh frames (288, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/017-transport-head-pattern/run.py validate
uv run python experiments/017-transport-head-pattern/run.py freeze-confirmation
uv run python experiments/017-transport-head-pattern/run.py explore
uv run python experiments/017-transport-head-pattern/run.py lock
uv run python experiments/017-transport-head-pattern/run.py confirm
uv run python experiments/017-transport-head-pattern/run.py report
```

- `explore` (once, exposed pool): the 66 frames' reference states — the residuals before blocks 1, 2 and 3 at every
  position up to `p_t` and the head's row at `p_t` (locked; the layers-1–2 states checked against the Experiment 016
  lock and stage-1 digests); the four layer-3 template-mean bases (locked); the 7764 recorded pairs re-measured with the
  layer-3 residuals at `p_c` and `p_t` and the head's row captured; replication of the Experiment 016 extract; the
  identities I1–I7 and the Level 1 recovery on every pair (incidents when violated); the measured `ΔA_H`, `F`, `Π`,
  `ΔT`; Level 0, the exact-head rung, the two ablations, the frozen pattern and Level 1 from the reference states; the
  exposed statistics (pooled row-entry R², `Π` and `ΔT` on pairs and token means, per frame, per template, the
  cue-final/coordinated split), the ladder with its predeclared orderings, the `x₃` and scale remainders, the
  decoded-`c_L` ladder.
- `lock` (weights, the Experiment 011/012 locks, the layer-3 bases, the 66 locked reference states — no forward pass, no
  capture or intervention call path): the prediction table for the fresh tokens × exposed frames (the head's row
  change, its self-weight change, `F̂`, `Π̂`, `ΔT̂`, the frozen-pattern `ΔT̂`, the exact-head, `−D` and template-head
  rungs, `σ̂'/σ` at layer 3, the decoded `c_L` with channel D and Experiment 016's), the token means; the frozen floors;
  writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as `preregistration-lock.json` and
  `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the twelve fresh frames' reference prompts and cue pairs, checks I1 and the reference half of
  I5, decides validity, computes the frame-conditional prediction table and writes it to the results state with its
  digest; **stage 2** starts only after that digest is re-read from disk, runs every fresh cue in both sets with the
  layer-3 residuals and the head's row captured, checks the identities and the recovery, and scores Y1 (strict
  prospective), Y2 (frame-conditional prospective, aggregate with the frame-collapse guard) and Y3 (the frozen pattern
  on the pooled cue-final pairs of both sets; the coordinated split descriptive) against the two tables.
- `report` renders `outputs/experiment-017/report.md`.

Boundaries: Level 0 never receives a measured fresh `x₃`, a fresh head state, or any quantity of a fresh cue forward
pass at either `p_c` or `p_t` — its channels come from the reference prompt, every `σ'` is of a predicted residual, and
the prediction table is computed with every capture entry point disabled (a test poisons every patched quantity of a
pair's record, at both positions, and shows every table column unchanged); the identities, the recovery and the
locked-state consistency of re-captured frames are enforced and any failure is an incident recorded with its commit; a
token needs three valid frames in a set to be scored there; Y2 needs eight valid fresh frames; no fresh cue prompt runs
before its frame's stage-1 predictions are digested; Y3 can fail while Y1 and Y2 pass, and its reading in that case is
fixed in the design; which positions the head moves weight between for which cues is described after the scoring.

## Status — 2026-09-20: complete; outcome `HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_NOT_REJECTED`

Tier A (`explore`) ran once at `ebc2206` on the 7764 recorded pairs (207 tokens, 66 frames): the Experiment 016 extract
replicated (deviation 0.0), every exposed frame's layers-1–2 state matched the Experiment 016 lock or its stage-1
digest, the identities held (I4 8.5e-6, I5 1.2e-5, I6 3.3e-5, I7 8.6e-7; I1–I3 as in 016), the switch model's Level 1
recovered the independent exact chain to 6e-12, the four layer-3 template bases were computed over 22 frames per
template and locked, and Level 0 reached head-row entry R² 0.996 (per-frame minimum 0.974), ΔT pairs R² 1.000 / token
means 1.000, Π token means Spearman 0.999 / R² 0.999; the frozen pattern 0.935 on all pairs and 0.918 on the cue-final
pairs (gap 0.081). The lock was installed byte-identical to the Tier A candidates and committed by hand (`4b6f766`;
lock content sha256 `b4fc9014ade7d21d2cd2e5be391ce6234e46fb46887880c0c4d03517d411ed72`, predictions sha256
`14fed442a8bc943f365a7b3768687ae5cf04b4792e5f1606da186c964293f9e6`; the candidate predictions are copied to
[`evidence/candidate-predictions-2026-09-20.md`](evidence/candidate-predictions-2026-09-20.md)), the read-only
post-lock check passed (hashes, byte identity, only the two exempt files changed since `ebc2206`, `validate_lock`,
frozen floors and program record, state digests and confirmation set, the complete 1584-row table reproduced with zero
difference, runtime equal to the explore record, no fresh prompt executed), and the single `confirm` ran at `4b6f766`
on a clean tree straight through both stages (results state sha256
`8a16da7da206f7d981eb5a5b9cc488b48493249d78ed328be28e8f5046f6c443`). **Stage 1** ran only the twelve fresh frames'
reference prompts and cue pairs — all twelve valid (plural head change 0.99–2.63 σ_T, cue effect 79/72; p_c 3–5, p_t =
p_c or p_c + 1) — computed the 288 frame-conditional predictions and digested them
(`7b62fa03c633547f954c9cd27b562b40c88151ff8c6c6a6080c2be151668302f`); **stage 2** started only after re-reading that
digest from disk and ran the 24 fresh cues in the 66 exposed frames (1584 pairs) and the twelve fresh frames (288
pairs). The final report is copied verbatim to [`evidence/final-report-2026-09-20.md`](evidence/final-report-2026-09-20.md).
Ledger after confirm: 2106 prompt keys, 80 noun keys. Experiment 017 is closed; nothing is amended or rerun.

- **Identities and the recovery (checks only):** on all 1872 fresh pairs the exact chain reproduced the captured
  layer-3 residuals at `p_c` and `p_t` (I4 5.7e-6 relative), the head's captured patched row (I5 1.2e-5), the measured
  head change (I6 2.3e-5) and the split `ΔT = F + Π` (I7 8.2e-7); the switch model's Level 1 recovered the independent
  chain to 4e-12; Experiment 016's I1–I3 and recoveries held. Every re-captured reference state matched its locked
  copy or stage-1 digest. The predicted layer-3 scale differed from the captured patched residual's by 0.001 on
  average and the decoded `Δ̂x₃` from the exact change by 2.1% (relative, mean; maximum 14%) — recorded, never fed back.
- **Y1 — strict prospective (24 fresh cues × 66 exposed frames): PASS on all five floors.** Head-row entry R² **0.997**
  (floor 0.95; per-frame minimum 0.970 over the 66 frames); `Π` token means Spearman **0.998** / R² **1.000** (floors
  0.90 / 0.90; MAE 0.0008 on a spread of 0.06); `ΔT` pairs R² **0.999** and token means R² **1.000** (floors 0.95; MAE
  0.007 / 0.002 on a spread of 0.20 σ_T). All 24 token means of `Π` carry the committed sign; the largest token-mean
  miss is 0.004 σ_T of `ΔT` (`trio`). `F̂` predicts `F` with R² 1.000.
- **Y2 — frame-conditional prospective, aggregate with the frame-collapse guard (24 fresh cues × 12 new frames): PASS
  on all six floors.** Row entry R² **0.998**; `Π` token means Spearman **1.000** / R² **1.000**; `ΔT` pairs **1.000**
  and token means **1.000**; **frame guard passed** — every new frame's row entry R² over its 24 pairs: 0.986, 0.998,
  0.994, 0.999 (cardinal), 1.000, 0.999, 1.000, 1.000 (quantifier), 0.999, 0.996, 0.999, 0.986 (coordinated-adjective).
  Per frame, `ΔT̂` predicts `ΔT` with R² 0.990–1.000 and `Π̂` predicts `Π` with R² 0.948–1.000.
- **Y3 — the frozen-pattern account is NOT rejected on the fresh cue-final pairs.** Over the 1248 scored cue-final
  pairs of both sets the frozen pattern's `ΔT` R² is **0.958** (rejected iff below 0.95) and its gap to Level 0's
  0.9996 is **0.042** (rejected iff at least 0.05): both clauses of the preregistered criterion fail, and the label is
  `FROZEN_PATTERN_NOT_REJECTED`. The reading fixed in the design before the run applies verbatim: *the head's pattern
  change was predicted, but the fresh set did not establish that modelling it explicitly was necessary relative to the
  frozen-pattern alternative* — a result about the size of the term on these cues, not a failure of the mechanism, and
  not a success of the mechanism either. Descriptively (no label): on the fresh cues the pattern term carries 0.042 of
  the `ΔT` variance on the cue-final pairs where the exposed pool carried 0.081, so these 24 cues move the head's
  pattern less than the exposed pool's cues do; within each template the frozen pattern is far from complete (over both
  sets, cardinal 0.907 with gap 0.091, quantifier 0.883 with gap 0.117; in the twelve new frames alone cardinal 0.590,
  quantifier 0.415, with per-frame values from 0.83 down to −0.46), and the pooled cue-final value of 0.958 is above
  the ceiling because the criterion pools the two templates, whose `ΔT` levels differ by about a factor of two and are
  carried by `F`. A per-template criterion was not preregistered and none is applied. The predeclared coordinated
  split came out the other way from its expectation: on the 624 coordinated pairs the frozen pattern reaches 0.945 with
  a gap of 0.053, so the criterion would have rejected it there — recorded, entering no label.
- **The rungs (descriptive; rows / `ΔT` / `Π` R² over both sets):** Level 0 0.997 / 0.999 / 0.996; the exact head on
  the same decoded `Δ̂x₃` 0.999 / 1.000 / 0.999 (the layer-3 reduction costs 0.002); without channel D (block 2's MLP
  at the template base at `p_c`) 0.952 / 0.955 / 0.674; with no frame channel at layer 3 (Experiment 015's reduction at
  the head) 0.830 / 0.939 / 0.354; Level 1 (identity) 1 / 1 / 1; the frozen pattern — / 0.956 / ≡ 0. The predeclared
  orderings hold: the layer-3 frame channels' ablation costs more than channel D's on the rows (0.166 > 0.045) and on
  `Π` (0.642 > 0.322). The decoded `c_L` with channel D reaches R² 1.000 on the fresh pairs where Experiment 016's
  Level 0-F reached 0.808 — the whole of the gap Experiment 016 left was block 2's operating point.
- **Where the head moves its weight, described only now (the new frames, in the network's own terms).** For most
  fresh cues the head's row at the transport position loses weight on the cue position and gains it on the token
  immediately before the cue (the verb): cardinal frames −0.06 on the cue and +0.05 on the verb on average, quantifier
  frames −0.24 / +0.25, coordinated frames −0.11 on the cue and +0.09 on the verb with +0.01 on the adjective itself;
  the earlier positions change by less than 0.01. The largest measured self-weight changes are `blunt` (−0.29),
  `metal` (−0.28), `leather` (−0.26), `usual` (−0.24); the pronouns go the other way (`you` +0.17, `them` +0.11, `us`
  +0.01), and `none` and `once` barely move it (−0.00, −0.02). Level 0 reproduces every token's mean self-weight change
  to within 0.004. The sign of the pattern term does not follow the sign of the self-weight change (`trio` −0.14 on
  the self weight with `Π` −0.10; `blunt` −0.29 with `Π` +0.03): the term depends on which values the moved weight
  lands on. No label is attached to any position or cue.

What this settles and what it does not. Stated at its safe strength: the decoded upstream state — the weight-only
encoding change, Experiment 016's frame-conditioned token-local attention changes of layers 1–2, block 1's and block
2's MLPs at the frame's operating points, one propagation step where the transport position follows the cue —
together with the head's own reference operands and scale predicts `L03.H04`'s own attention-row change at the
transport position and its end-to-end output change along `d̂_T` nearly exactly for 24 never-seen cues in the 66
exposed frames and in twelve previously untested frames, no frame falling below row entry R² 0.985, with the pattern
term itself predicted to R² ≥ 0.99 on every set. The chain from the cue's encoding change to the transport head's
output is now written end to end as a weight-only program whose frame inputs are named reference-run quantities, and
it composes: every quantity the head receives was predicted by the earlier experiments' models, not measured. What the
fresh set did *not* establish is that modelling the pattern term is necessary relative to the frozen-pattern account
under the preregistered pooled criterion: on these 24 cues the term is small relative to the value term, and the
account of Experiments 009, 011 and 013 reaches R² 0.958 on the cue-final pairs. This is not the claim that the frame
is a few numbers (the channels' dimensionality is recorded in the design), nor that block 2's frame dependence is
concentrated (Experiment 018), nor anything about behaviour. Limits: one checkpoint, three templates, five lexical
classes, one head, twelve new frames.

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

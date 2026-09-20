# Experiment 016: What the Frame Contributes — The Frame-Conditioned Token-Local Q/K Model

Implements the approved design
[`docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md`](../../docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-20-experiment-016-frame-conditioned-qk-plan.md`](../../docs/superpowers/plans/2026-09-20-experiment-016-frame-conditioned-qk-plan.md).
Prospective, zero-parameter, decomposition-completing: Experiment 015's token-local cue-change Q/K model missed its
frame-conditional floors at layer 2; this experiment adds three literal reference-state channels and nothing else —
(A) the frame's own reference query and key vectors at the cue position as the operands of the self-logit (1024
numbers per layer), (B) the frame's LayerNorm scale before the change and, derived algebraically from the *predicted*
change, after it (two scalars per layer), (C) block 1's MLP evaluated at the frame's own cue residual (the 512-vector
through the 2048 pre-activations) — and tests the resulting Level 0-F on unseen cues and twelve unseen frames with a
per-frame anti-collapse guard, the scale-only alternative committed for rejection, the three ablations as a
descriptive ladder (with each channel alone, the oracle diagonal-proportional comparator and the norm ratios as further
descriptive rungs). The decoded `c_L` at Level 0-F keeps block 2's MLP at the template base — channel C is block 1's
operating point only; Level 1 alone evaluates both MLPs at the frame's own state. The channel switches are one code path: all off recovers Experiment 015's Level 0, all on plus the
discarded renormalization term recovers Level 1 (both checked as identities on every pair). No Level 0-F input is read
from a fresh cue forward pass.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 183 exposed cue tokens (Experiment
  015's 159 and its 24 confirmed) and the 54 exposed frames (its 48 and its 6), with the 80 nouns; a committed extract
  of Experiment 015's per-pair `c_ΔA`, sixteen self-weight changes and `c_L`, `c_M`, `c_H`
  ([`inherited/experiment-015-pair-extract.json`](inherited/experiment-015-pair-extract.json), 6180 pairs, replicated
  within `1e-6`), carrying also the digests of the six 015-fresh frames' stage-1 reference states; the Experiment 011
  lock's axes and read weight, the Experiment 012 lock's bases and the Experiment 015 lock's reference states (checked
  against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-20 by tokenizer rules before any Experiment 016 model output, content
  sha256 `e5a3a4eb392e459117ebd7dfd132fffb66f02bbd0cd75b479f5ba1afd83e3855`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `initial upper respective individual specific` (determiner-like), `twentieth
  quarter twin dual` (ordinal-or-numeral), `unlimited countable total insufficient extensive` (quantity), `oneself
  myself yourselves whoever` (possessive-or-pronoun), `bright orange pink brown grey heavy` (adjective); twelve fresh
  frames, four per template (`The garden yields`, `The factory assembles`, `The kitchen serves`, `The workshop builds`;
  `The essay explores`, `The teacher explains`, `The guide recommends`, `The seminar addresses`; `Liam and Ava stored …
  cold`, `Zoe and Theo shipped … fresh`, `Nina and Arjun mixed … thick`, `Elif and Marco tied … tight`); two prompt lists: the fresh
  tokens in the 54 exposed frames (1296, Y1) and in the 12 fresh frames (288, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/016-frame-conditioned-qk/run.py validate
uv run python experiments/016-frame-conditioned-qk/run.py freeze-confirmation
uv run python experiments/016-frame-conditioned-qk/run.py explore
uv run python experiments/016-frame-conditioned-qk/run.py lock
uv run python experiments/016-frame-conditioned-qk/run.py confirm
uv run python experiments/016-frame-conditioned-qk/run.py report
```

- `explore` (once, exposed pool): the 54 frames' reference states (locked; checked against the Experiment 015 lock
  and stage-1 digests); the 6180 recorded pairs re-measured with the patched residuals and rows captured; replication
  of the Experiment 015 extract; the identities I1–I3 and the two channel recoveries on every pair (incidents when
  violated); the measured rows and `c_ΔA`; Level 0-F, the scale-only alternative, the ablations, Experiment 015's
  Level 0 and Level 1 from the reference states; the exposed statistics (token means and pairs of `ĉ_ΔA`, pooled entry
  R² per layer and per frame, self-weight R²), the ablation ladder with its predeclared ordering, the renormalization
  and scale remainders, the decoded-`c_L` ladder.
- `lock` (weights, the Experiment 011/012 locks, the 54 locked reference states — no forward pass, no capture or
  intervention call path): the prediction table for the fresh tokens × exposed frames (the sixteen Level 0-F rows, the
  self-weight changes, `ĉ_ΔA` per layer and in total, the scale-only rows and read, the diagonal-proportional rows and
  read, the three ablations' reads, Experiment 015's Level-0 read, `σ̂'/σ` per layer, the decoded `c_L`), the token
  means; the frozen floors; writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the twelve fresh frames' reference prompts and cue pairs, checks I1, decides validity, computes
  the frame-conditional prediction table and writes it to the results state with its digest; **stage 2** starts only
  after that digest is re-read from disk, runs every fresh cue in both sets with the patched residuals and rows
  captured, checks the identities and recoveries, and scores Y1 (strict prospective), Y2 (frame-conditional prospective,
  aggregate with the frame-collapse guard) and Y3 (the scale-only alternative over both sets) against the two tables.
- `report` renders `outputs/experiment-016/report.md`.

Boundaries: no Level 0-F input is read from a fresh cue forward pass — the channels come from the reference prompt
and `σ'` from the predicted change (a test poisons the patched capture and shows the table unchanged; the prediction
table is computed with every capture entry point disabled); the identities, recoveries and the locked-state
consistency of re-captured frames are enforced and any failure is an incident recorded with its commit; a token needs
three valid frames in a set to be scored there; Y2 needs eight valid fresh frames; no fresh cue prompt runs before its
frame's stage-1 predictions are digested; which heads move for which cues in the new frames is described after the
scoring.

## Status

Implementation in progress; no scientific phase has run.

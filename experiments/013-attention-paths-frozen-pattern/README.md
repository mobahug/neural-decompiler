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

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

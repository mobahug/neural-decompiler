# Experiment 012: Is the Layers-1–2 Correction to the Encoding Read a Token-Local MLP Computation?

Implements the approved design
[`docs/superpowers/specs/2026-09-19-experiment-012-layer-correction-token-local-design.md`](../../docs/superpowers/specs/2026-09-19-experiment-012-layer-correction-token-local-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-19-experiment-012-layer-correction-token-local-plan.md`](../../docs/superpowers/plans/2026-09-19-experiment-012-layer-correction-token-local-plan.md).
Prospective and zero-parameter: the quantity under test is the two MLPs of blocks 1 and 2 evaluated at the cue position
on the encoding difference `ΔE = E(w) − E(ref_T)` with attention held at the reference and the base state taken from
exposed reference prompts. Because the block-0 MLP is parallel to block-0 attention, the E-patch changes the cue
position's residual before block 1 by exactly `ΔE`, so block 1's part is exact given the base state and block 2's part
omits only block 1's attention change. Every prediction is committed before any fresh prompt runs.

## Inputs

- Exposed pool (calibration only; nothing fitted): the 87 exposed cue tokens (Experiment 010's 63 and Experiment 011's
  24) and the 30 exposed frames (Experiment 010's 24 and Experiment 011's 6), with the 80 nouns; committed extracts of
  Experiments 010's and 011's per-component fractions ([`inherited/`](inherited/), 1512 and 142 pairs, replicated
  within `1e-6` for every component); the Experiment 011 lock's axes `d̂_T`, `d̂_E`, `σ_T`, read weight `γ₃ ⊙ m` and
  weight-only denominators, used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-19 by tokenizer rules before any Experiment 012 model output, content
  sha256 `32892d23d8f853f6af71218a0562a6ffc7c375ae0fae81fc9566a79866964e07`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `half whichever whatever which what` (determiner-like), `fifty sixty seventy eighty
  ninety` (numeral), `scarce least myriad manifold sparse` (quantity), `hers theirs ours thy` (possessive-or-pronoun),
  `black white young empty wooden` (adjective) — every one licensed in every fresh frame, and six fresh frames
  (`The warehouse ships`, `The florist arranges`, `The lecture reviews`, `The brochure advertises`, `Hugo and Amara
  folded … flat`, `Nadia and Erik hauled … dusty`); 144 token prompts. Executed only by `confirm`.

## Commands

```bash
uv run python experiments/012-layer-correction-token-local/run.py validate
uv run python experiments/012-layer-correction-token-local/run.py freeze-confirmation
uv run python experiments/012-layer-correction-token-local/run.py explore
uv run python experiments/012-layer-correction-token-local/run.py lock
uv run python experiments/012-layer-correction-token-local/run.py confirm
uv run python experiments/012-layer-correction-token-local/run.py report
```

- `explore` (once, exposed pool): the 87 × 30 E-patches with the head's internals and the cue position's residuals
  before blocks 1 and 2 captured; replication of the two extracts; the recomputed stage axes checked against the
  Experiment 011 lock; the template base states (locked) and their leave-one-frame-out variants (calibration); the
  exact measured correction `c_L` (heads included), `c_M`, `c_H`, `P1'`, `P1`; the model's leave-one-frame-out
  predictions `ĉ`, `q̂`; `τ_c`, `τ_P`, the exposed explained variance and its 24-token subsample distribution; the
  own-base ladder; the block-1/2 neuron ledgers.
- `lock` (weights, the Experiment 011 axes, and the locked base states — no forward pass, no capture or intervention
  call path): `ĉ` with its `MLP₁`/`MLP₂` parts and `∥`/`⊥` evaluations, `g_E`, `q̂' = g_E + ĉ`, `D̂_T`, `q̂` for every
  fresh token and defined template, the licensed-frame means, the tolerances and floors; writes `candidate-lock.json`
  and `candidate-predictions.md`. Installing them as `preregistration-lock.json` and `predictions.md` and committing
  them is the preregistration act.
- `confirm` (once): validates the artifacts, reproduces every prediction from the weights and the locked states before
  any fresh prompt (refuses above `1e-9`), then runs each fresh frame's reference, plural-cue E-patch and cue pair
  (frame validity from the template's cues only), and every fresh token's E-patch with the internals and residuals
  captured; scores Y1 (`c̄̂` vs `c̄_L`: Spearman ≥ 0.80, MAE ≤ `τ_c`, `R² ≥ 0.50`) and Y2 (`q̄̂` vs `P̄1`: Spearman ≥ 0.90,
  MAE ≤ `τ_P`) on identical frame sets; records the own-base ladder, the heads' share, and the neuron ledgers
  descriptively.
- `report` renders `outputs/experiment-012/report.md`.

Boundaries: no candidate is ever excluded on the basis of its own measured value; a token needs three valid frames;
at least four valid frames and sixteen scored tokens are required; Experiment 010's identity checks, the block-1
exactness, the block-2 identity at the captured input, and the neuron-sum identities are enforced in both phases and
any failure is an incident recorded with its commit; the lock phase's only access to the model is its parameters.

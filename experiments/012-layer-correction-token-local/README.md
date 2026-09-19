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

## Status — 2026-09-19: Tier A executed once; candidate lock and predictions written, awaiting installation and the reviewer's sign-off

`explore` ran once on protocol/code commit `ddbbbd3` (run `cec6c3908508f723`, results state sha256
`069a6306d1325dcd6efd7f2aeac330f26baa2c765021e65d5a4480f1252f381a`; A0 passed). The report is copied to
[`evidence/exploration-report-2026-09-19.md`](evidence/exploration-report-2026-09-19.md) and the candidate prediction
table to [`evidence/candidate-predictions-2026-09-19.md`](evidence/candidate-predictions-2026-09-19.md). Ledger: 90
prompt keys (the 30 exposed frames' cue and reference prompts), 80 noun keys; no fresh prompt ran, and the lock phase
ran no prompt at all.

- Replication of Experiments 010 (1512 pairs) and 011 (142 pairs), every component: exact. The recomputed stage axes,
  `σ_T`, read weight, weight-only denominators and `σ_r` equal the Experiment 011 lock's values. Identity checks on every
  exposed pair: ρ identity 1.1e-7, P1 cross-check 6.6e-16, neuron sum 2e-8 (relative); block-1 exactness, the block-2
  identity at the captured input, and both neuron-sum vector identities held everywhere (incident otherwise).
- Denominators: `r(E(pl_T) − E(ref_T))` 1.630 / 1.496 / 1.630 and the modelled plural totals `D̂_T` 1.745 / 2.213 / 1.851
  (cardinal / quantifier / coordinated-adjective; the plural cue's own modelled correction is 0.07 / 0.48 / 0.14) — all
  three templates defined.
- **Leave-one-frame-out calibration (87 tokens):** `ĉ̄` vs `c̄_L` Spearman 0.944, MAE 0.053, R² 0.773, bias +0.035
  (against `c̄_M` alone: 0.912 / 0.063 / 0.739); **τ_c = 0.207**; 24-token subsamples of R²: 1st percentile 0.45, 5th
  0.58, median 0.77 (frozen floor 0.50). The all-frames base gives the same figures (0.944 / 0.053 / 0.770); the frame's
  own base 0.948 / 0.039 / 0.890. `q̂̄` vs `P̄1`: Spearman 0.982, MAE 0.035, R² 0.979; **τ_P = 0.130**. `c̄_L` spread sd
  0.145 (mean +0.166). Pairs: 0.881 / 0.089; per template cardinal 0.78 / 0.08, quantifier 0.81 / 0.11,
  coordinated-adjective 0.82 / 0.08; the quantifier template carries the largest correction (mean `c_L` +0.36 against
  +0.06 and +0.08). Heads' share `mean|c_H| / mean|c_M|` 0.24. Ladder means: base-point (own − leave-one-out) −0.04,
  attention-input +0.02 (|.| 0.05). The largest token-mean residuals are over-predictions of the small numerals
  (`three` +0.18, `six` +0.16, `ten` +0.15); all under τ_c.
- Reported beside (never judged): `g_E` vs `P̄1` 0.903 / 0.091 and vs `q̄_T` 0.928 / 0.065; `q̂` vs `q̄_T` 0.933 / 0.073 —
  as in the design check, the composite does not improve on `g_E` against `q_T` and is not meant to.
- Block-2 neuron ledger (descriptive): the mass is not concentrated (`n_80` median 502 of 2048, range 402–566; block 1
  median 525); one neuron, 1987, carries about three times the mean absolute mass of the next (0.149 against 0.052) in
  every template, with cosine 0.24 between its effective input direction and `d̂_E`; the top-20 sets overlap across
  templates with Jaccard 0.54 (cardinal–coordinated), 0.29 and 0.25 (quantifier against each).
- **Candidate lock** `outputs/experiment-012/candidate-lock.json`, content sha256
  `830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb`; predictions artifact
  `candidate-predictions.md`, sha256 `f32ce4da97544f395280cfccb800b90f2c1a1cf23ec524ea2eba3dc7069a811f`. Committed
  predictions (mean `ĉ` over the six licensed frames): numerals `seventy` 0.56, `fifty` 0.54, `eighty` 0.45, `sixty`
  0.43, `ninety` 0.38 — the only class whose `∥`/`⊥` interaction is positive; quantity `myriad` 0.59, `manifold` 0.38,
  `sparse` 0.24, `scarce` 0.10, `least` 0.05; adjectives `wooden` 0.21, `white` 0.14, `black` 0.13, `young` 0.11,
  `empty` 0.08; possessive/pronoun `hers` 0.23, `ours` 0.18, `theirs` 0.10, **`thy` −0.08**; determiner-like
  `whichever` 0.19, `half` 0.14, `whatever` 0.05, `what` 0.03, **`which` −0.06**. In every token the model's response to
  `ΔE_⊥` exceeds its response to `ΔE_∥`, and the quantifier template's predicted correction is the largest (for
  example `half` 0.03 / 0.01 / 0.38 across the templates). The composite `q̂` places the numerals at 1.00–1.12 and
  `hers` lowest at 0.34.

Installing the two artifacts as `preregistration-lock.json` and `predictions.md` and committing them is the
preregistration act; the reviewer's sign-off precedes `confirm`.

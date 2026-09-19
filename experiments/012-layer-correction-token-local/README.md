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

## Status — 2026-09-19: complete; outcome `LAYER_CORRECTION_TOKEN_LOCAL_MLP | COMPOSITE_PREDICTS_P1`

The lock was installed byte-identical to the Tier A candidates and committed (`5cea6c4`; lock content sha256
`830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb`, predictions sha256
`f32ce4da97544f395280cfccb800b90f2c1a1cf23ec524ea2eba3dc7069a811f`), the read-only post-lock review passed (hashes,
scientific-path immutability since `ddbbbd3`, `validate_lock`, frozen tolerances and floors, complete pointwise
predictions and normalizations in the lock, no overlap with executed prompts, a read-only `confirm` path), the reviewer
signed off, and the single `confirm` ran at `5cea6c4` on a clean tree (results state sha256
`7561c402d2e24313a3a497412d890fcd93c710ba0b3e96483c73e00439e85585`). Every locked prediction was reproduced from the
weights, the locked axes and the locked base states before any fresh prompt (max difference 0.0); the recomputed axes,
`σ_T`, read weight and `σ_r` matched the lock. The final report is copied verbatim to
[`evidence/final-report-2026-09-19.md`](evidence/final-report-2026-09-19.md). Ledger after confirm: 252 prompt keys,
80 noun keys. Experiment 012 is closed; nothing is amended or rerun.

- **Precondition:** all six fresh frames valid — plural-cue head change 1.31–2.71 (floor 0.253), cue effect 77–79 of
  80 nouns (floor 72) — and all 24 fresh tokens scored in all six frames. Identity checks held on every fresh
  measurement (ρ identity 1.0e-7, P1 cross-check 4e-16, neuron sum 1e-8, relative); the block-1 exactness, the block-2
  identity at the captured input and both neuron-sum vector identities held in all 144 pairs.
- **Y1 — the layer correction is the token-local MLP computation: PASS on all three floors.** Over the 24 token means,
  the committed `ĉ̄` predicts the measured net layer change `c̄_L` (heads included) with Spearman **0.864** (floor 0.80),
  MAE **0.061** (τ_c 0.207), explained variance **R² 0.795** (floor 0.50), bias −0.002. The committed sign was right for
  23 of 24 tokens; four residuals exceed 0.10 (`manifold` predicted 0.38, measured 0.22; `half` 0.14 → 0.01; `which`
  −0.06 → +0.07, the one sign miss; `what` 0.03 → 0.15), none near τ_c. The committed hard calls came out: `thy` −0.08 →
  **−0.08**, the lowest measured correction of the 24 (negative in the four cue-final frames, +0.06 in the quantifier
  frames); `myriad` 0.59 → **0.56**, the highest measured, above every numeral as predicted; the numerals 0.38–0.56 →
  0.34–0.51 with `fifty` and `seventy` highest in both; the quantifier template's correction far above the others
  (predicted +0.44 against +0.07 / +0.13 per pair; measured **+0.36** against +0.18 / +0.11). Pair level: Spearman 0.859,
  MAE 0.098, R² 0.65 over 144 pairs; the cardinal frames' correction was under-predicted (+0.07 vs +0.18).
- **Y2 — the composite account reaches the head's linear read-out: PASS.** `q̄̂` (encoding read plus modelled correction,
  in the head's units) versus the measured P1 fraction `P̄1`: Spearman **0.922** (floor 0.90), MAE **0.062** (τ_P 0.130),
  R² 0.917, bias −0.045. Reported beside it, never judged: `g_E` alone versus `P̄1` 0.856 / 0.127; `g_E` versus `q̄_T`
  0.883 / 0.068; the composite versus `q̄_T` 0.956 / 0.079; `q̂'` versus `P1'` 0.922 / 0.061.
- **Own-base ladder (descriptive, post-confirm):** with each fresh frame's own reference state the model reaches
  Spearman 0.950, MAE 0.051, R² 0.872 against `c̄_L`; the base-point term (own minus locked template base) averages
  −0.037 (|.| 0.047), the attention-input term (block 1's attention change reaching block 2's input) +0.051 (|.| 0.065),
  the heads' direct contribution −0.013 (share `mean|c_H| / mean|c_M|` 0.227, exposed 0.239). The model predicts the
  total `c̄_L` better than the MLP part `c̄_M` alone (0.850 / 0.079 / 0.599): the two omitted attention terms partly
  offset each other in this set. `thy` is the clearest case of the omission: its measured MLP part is +0.02 and its
  heads' part −0.10, so the predicted −0.08 matched the total while the token-local MLPs at the frame's own base give
  −0.11 and block 1's attention change into block 2 supplies +0.13.
- **Descriptive:** in every token the model's response to `ΔE_⊥` exceeds its response to `ΔE_∥` (fresh means 0.28
  against 0.03), and the numerals remain the only class with a positive `∥`/`⊥` interaction (+0.07 to +0.11). The
  block-2 neuron ledger is not concentrated (`n_80` median 512 of 2048, range 434–549); neuron 1987 is again the top
  contributor in every template (mean absolute mass 0.088, next 0.049); top-20 overlap exposed versus fresh: Jaccard
  0.33 (cardinal), 0.67 (quantifier), 0.54 (coordinated-adjective). Class means (`c̄_L` / locked `ĉ̄`): numeral 0.43 /
  0.47, quantity 0.22 / 0.27, adjective 0.16 / 0.13, possessive-or-pronoun 0.16 / 0.11, determiner-like 0.10 / 0.07.

What this settles and what it does not. The systematic residual Experiment 011 left — what layers 1–2 add to the
encoding read before `L03.H04` reads it — is, for 24 never-seen cues in six never-seen frames, predicted in ordering,
absolute error and magnitude by a computation with no fitted parameter: the two MLPs of blocks 1 and 2 evaluated at
the cue position on the encoding difference, attention held at the reference, base states from exposed frames. Block
1's part is exact by the architecture; block 2's part omits attention's effect on its input, and the heads' own writes
are omitted altogether; both omissions are real and measured (about 0.05–0.07 each on average) and did not break the
prediction at the token-mean level for this set, but they are why the pair level is looser (R² 0.65) and why one sign
(`which`) was missed. The composite "encoding read + token-local MLP correction" now accounts for what the head reads
linearly (Y2), extending the decoded chain from the token embedding through the layer-0 encoding and the layer-1–2
MLPs to the confirmed head read-out. The claim is proportional to the set: five lexical classes, three templates, this
checkpoint; the ledger describes but does not test what the block-2 neurons read (the off-axis part of the encoding,
one dominant neuron, a distributed remainder). C002 is unchanged (out of scope here).

### Tier A (2026-09-19, for the record)


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

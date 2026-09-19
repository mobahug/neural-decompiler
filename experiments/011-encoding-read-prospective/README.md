# Experiment 011: Does the Encoding Read Predict `L03.H04`'s Transport of New Cues?

Implements the approved design
[`docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md`](../../docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-19-experiment-011-encoding-read-prospective-plan.md`](../../docs/superpowers/plans/2026-09-19-experiment-011-encoding-read-prospective-plan.md).
Prospective and zero-parameter: the quantity under test is a weight-defined read functional applied to the cue's
layer-0 encoding (`E(w) = MLP₀(ln2₀(W_E[w]))`, weight-only since Experiment 005), so every prediction is computed
without a forward pass on any fresh prompt and committed before the single confirmation run.

## Inputs

- Exposed pool (calibration only; nothing fitted): the 63 tokens and 24 frames of Experiment 010 and a committed
  extract of its transport fractions ([`inherited/experiment-010-transport-fractions.json`](inherited/experiment-010-transport-fractions.json),
  63 × 24, replicated within `1e-6`).
- `confirmation-v1.json` — frozen on 2026-09-19 by tokenizer rules before any Experiment 011 model output, content
  sha256 `174cf22214503836d5da3fc464a5f36fbb7cd9794ab74b00de0c793f12979358`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `an such much little less` (determiner-like; `an` licensed in the four cue-final
  frames only), `eighteen nineteen twenty thirty forty` (numeral), `additional extra sufficient ample abundant`
  (quantity), `its whose mine yours` (possessive-or-pronoun), `green cheap warm dark huge` (adjective) — and six fresh
  frames (`The cabinet stores`, `The vendor sells`, `The survey covers`, `The monitor tracks`, `Leo and Maya labeled …
  round`, `Priya and Jonas bundled … soft`); 142 token prompts. Executed only by `confirm`.

## Commands

```bash
uv run python experiments/011-encoding-read-prospective/run.py validate
uv run python experiments/011-encoding-read-prospective/run.py freeze-confirmation
uv run python experiments/011-encoding-read-prospective/run.py explore
uv run python experiments/011-encoding-read-prospective/run.py lock
uv run python experiments/011-encoding-read-prospective/run.py confirm
uv run python experiments/011-encoding-read-prospective/run.py report
```

- `explore` (once, exposed pool): transport fractions for 63 × 24 replicated against Experiment 010; the axes `d̂_T`,
  `d̂_E`; the read weight `γ₃ ⊙ m`; the template denominators `r(E(pl_T) − E(ref_T))` and their validity; `g_E` for the
  63 tokens (descriptive Spearman against measured transport); `τ_g` from the 63 token-mean residuals and `τ_M` from
  the pair-level P1 residuals.
- `lock` (weights only — no forward pass, no capture or intervention call path): `g_E`, `g_∥`, `g_⊥` for every fresh
  token and defined template, the licensed-frame means, Experiment 009's frozen rank-1 rule as the reported baseline,
  the numeric tolerances and floors; writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (once): validates the artifacts, reproduces every prediction from the weights and the locked axes before
  any fresh prompt (refuses above `1e-9`), then runs each fresh frame's reference, plural-cue E-patch, and cue pair
  (frame validity from the template's cues only), and every licensed fresh token's E-patch with the head's internals
  in every valid frame; scores Y1 (`ḡ_E` vs `q̄_T` over scored tokens, same frame set) and Y2 (P1 vs `q_T` over pairs).
- `report` renders `outputs/experiment-011/report.md`.

Boundaries: no candidate is ever excluded on the basis of its own measured value; a token needs three valid licensed
frames to be scored; at least four valid frames and sixteen scored tokens are required; Experiment 010's identity
checks (ρ identity, P1 cross-check, neuron sum) are enforced in both phases and any failure is an incident recorded
with its commit; the lock phase's only access to the model is its parameters; `σ_T` and the axes are locked and
re-checked at `confirm`. Implementation note on the design's `σ_r` floor: because every exposed frame's singular cue is
the template's reference cue and `r` is linear, `σ_r` as defined equals half the frame-weighted mean of the three
template denominators, so the `0.25 σ_r` floor cannot bind below the `0.25 × max` floor; the rule is applied as
written and the vacuity is noted rather than repaired.

## Status — 2026-09-19: complete; outcome `ENCODING_READ_PREDICTS_TRANSPORT | HEAD_P1_REPLICATED`

The lock was installed and committed by hand (`85a6c04`, byte-identical to the candidates; lock content sha256
`769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`, predictions sha256
`6ce4cd0c1bf8c0bc7b312972513499e97c3bf04cfa455a98dc9448421042d033`), the reviewer's post-lock review found no
scientific defect (hashes, weight-only reproduction, scientific-path immutability since `69c7a14`, frozen axes and
tolerances, outcome-independent validity, read-only `confirm` path), and the single `confirm` ran at `85a6c04` on a
clean tree (results state sha256 `6a73a080bdba2cff81c7b7a83b7b98a82feefaa12dd3c04740c949fb9971dac7`). Every locked
prediction was reproduced from the weights and the locked axes before any fresh prompt (max difference 0.0); the
stage axes and `σ_T` recomputed from the exposed frames matched the locked values. The final report is copied
verbatim to [`evidence/final-report-2026-09-19.md`](evidence/final-report-2026-09-19.md). Ledger after confirm: 232
prompt keys, 80 noun keys. Experiment 011 is closed; nothing is amended or rerun.

- **Precondition:** all six fresh frames valid — plural-cue head change 1.08–2.31 (floor 0.25 σ_T = 0.253), cue
  effect 79/80 nouns in every frame (floor 72) — and all 24 tokens scored (`an` on its four licensed frames; min 16).
  Identity checks held on every fresh measurement (max relative errors: ρ identity 9.3e-8, P1 cross-check 4e-16,
  neuron sum 2e-8); head reconstruction error 0.0 in every frame.
- **Y1 — zero-parameter encoding read: PASS.** Over the 24 token means, the committed `ḡ_E` predicts the head's
  measured transport `q̄_T` with Spearman **0.848** (floor 0.80) and MAE **0.077** (τ_g 0.248). No token's residual
  reaches τ_g; the largest is `abundant` (predicted 0.52, measured 0.71). The committed low predictions came out
  lowest: `an` 0.18 → **0.30** (the lowest measured transport of the 24; the rest-of-encoding part of its read was
  negative in 4/4 frames, −0.25 to −0.35, as in the `a` ledger of Experiment 010) and `mine` 0.35 → 0.45. The
  committed spread inside the adjectives held (`cheap` 0.82 → 0.77, the highest adjective measured; `green` 0.55 →
  0.56), and the numerals came out where predicted (0.84–0.93 → 0.89–0.99, `eighteen` highest in both). The misses are
  in the middle: `much`/`such` 0.42/0.43 → 0.59/0.60 and `abundant` under-predicted, `yours` 0.44 → 0.33 and `its`
  0.48 → 0.40 over-predicted, so `yours`/`its` were measured below `mine`. Residuals are mostly positive (17/24,
  mean +0.04) and track the net layer-1–2 contribution (Spearman 0.66 between residual and net layer change), which
  averages +0.18 on the fresh set (22/24 tokens inside the exposed range −0.08 to +0.40; `twenty` 0.44 and `forty`
  0.41 above it); mean gross layer change 0.56 (exposed median 0.44).
- **Y2 — head read-out replication: PASS.** Over the 142 fresh (token, frame) pairs the fixed-normalization linear
  OV read-out P1 predicts the head's measured transport with Spearman **0.974** (floor 0.90) and MAE **0.076** (τ_M
  0.313) — a second prospective confirmation of Experiment 009's mechanism level, on frames and cues that experiment
  never saw.
- **Baseline (reported only, not in the outcome):** Experiment 009's fitted rank-1 transport rule reaches Spearman
  0.816 and MAE 0.091 on the same 24 token means — it would also have cleared the Y1 floors on this set. The
  zero-parameter read is better on both figures, but the margin is modest; the result is not that the rule was wrong
  here, it is that no fitted parameter was needed.

What this settles and what it does not. For `L03.H04` in this model, a quantity computed from the weights alone — the
cue's layer-0 encoding, block 3's LayerNorm gain, the head's `W_V W_O`, and the frozen number direction — predicted,
before any of these prompts was executed, how strongly the head would transport each of 24 never-seen cue words in six
never-seen frames (rank correlation 0.85, mean error 0.08 in units of the plural cue's transport), including the
suppression of `an` by the opposition inside its own encoding. The claim is proportional to the set: the 24 cues were
drawn by quota from five lexical classes (determiner-like, numeral, quantity, possessive/pronoun, adjective), so the
support is generalization across this preregistered heterogeneous cue set, not to arbitrary English words. The
systematic part of the residual is the net contribution of layers 1–2, which the encoding read omits by construction
and which is positive on average (+0.18); where layers add more, the read under-predicts more. C002 is unchanged (out
of scope here).

### Tier A (2026-09-19, for the record)

`explore` ran once on protocol/code commit `69c7a14` (run recorded in the results state, sha256
`b1b0e47ffdbde872a26c76aa4efa1d3cb699bcc9ac06e23c60c82152a3a23b76`; A0 passed). The report is copied to
[`evidence/exploration-report-2026-09-19.md`](evidence/exploration-report-2026-09-19.md) and the candidate prediction
table to [`evidence/candidate-predictions-2026-09-19.md`](evidence/candidate-predictions-2026-09-19.md). Ledger: 72
prompt keys (exposed frames' cue and reference prompts), 80 noun keys; no fresh prompt ran, and the lock phase ran no
prompt at all.

- Replication of Experiment 010's 1512 transport fractions: exact. Identity checks enforced on every exposed run.
- Template denominators `r(E(pl_T) − E(ref_T))`: cardinal 1.630, quantifier 1.496, coordinated-adjective 1.630 — all
  three defined (floor 0.41).
- Tolerances from exposed residuals at the floors' own levels: **τ_g = 0.248** (token means, 63 tokens) and
  **τ_M = 0.313** (1488 pairs). Descriptive exposed check: `ḡ_E` vs `q̄_T` Spearman 0.925, MAE 0.064; P1 pair MAE 0.073.
  Exposed net layer change range −0.08 to +0.40.
- **Candidate lock** `outputs/experiment-011/candidate-lock.json`, content sha256
  `769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`; predictions artifact
  `candidate-predictions.md`, sha256 `6ce4cd0c1bf8c0bc7b312972513499e97c3bf04cfa455a98dc9448421042d033`. Committed
  predictions (mean `g_E` over licensed frames): numerals `eighteen` 0.93, `twenty` 0.88, `forty` 0.87, `nineteen`
  0.85, `thirty` 0.84; adjectives `cheap` 0.82, `dark` 0.69, `huge` 0.70, `warm` 0.61, `green` 0.55; quantity
  `additional` 0.65, `abundant` 0.52, `extra` 0.52, `ample` 0.51, `sufficient` 0.45; determiner-like `little` 0.62,
  `less` 0.57, `such` 0.43, `much` 0.42, **`an` 0.18** (axis part +0.54, rest −0.36 — the same encoding-borne opposition
  as `a`); possessive/pronoun `whose` 0.53, `its` 0.48, `yours` 0.44, `mine` 0.35. Experiment 009's rank-1 rule is
  locked beside them as the reported baseline.

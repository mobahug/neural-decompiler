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
frames to be scored; at least four valid frames and sixteen scored tokens are required; incidents are recorded with
their commit; the lock phase's only access to the model is its parameters.

## Status — 2026-09-19: confirmation set frozen; Tier A not yet run

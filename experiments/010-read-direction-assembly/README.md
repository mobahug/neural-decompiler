# Experiment 010: How the Head-Readable Number Signal Is Assembled

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-010-read-direction-assembly-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-010-read-direction-assembly-plan.md).
Discovery-only exact attribution: nothing is fitted; every number is a measured or weight-only quantity under
Experiment 009's confirmed P1 read functional, normalized by the template's plural cue.

## Inputs (all exposed by Experiments 005–009)

- 63 cue tokens (the forty of Experiment 008 and the twenty-three of Experiment 009's confirmation set), 24 frames (18 +
  6), 80 nouns (79 single-token, used only for the replication checks).
- Committed extracts of the recorded E-patch means of Experiments 006 (192 pairs, 59 exposed nouns), 007 (144 pairs,
  20 confirmation nouns), and 009 ([`inherited/experiment-009-confirmation-epatch-means.json`](inherited/experiment-009-confirmation-epatch-means.json),
  138 pairs, 79 nouns); `explore` must reproduce all of them within `1e-6`.

## What is computed

For every (token, frame): one E-patch of the frame's reference prompt with the outputs of the sixteen attention heads
and two MLPs of layers 1–2 at the cue position captured. With `ρ_f(x) = (A_c^ref/σ_c^ref)·⟨x − mean(x), γ₃ ⊙ m⟩`,
`m = W_V W_O d̂_T`, the exact identity `ρ_f(Δr_c) = ρ_f(ΔE) + Σ_k ρ_f(Δout_k)` is checked in every run, `ρ_f(Δr_c)` is
checked against Experiment 009's P1 projection, and the neuron terms `c_j = Δa_j ρ_f(W_out[j])` are checked to sum to
`ρ_f(ΔE)`. Fractions are normalized by the plural cue's measured head-output change in the frame, so `f_total` is
Experiment 009's P1 fraction term for term. Reported per token: `f_E`, `f_∥`, `f_⊥`, every `f_k`, `f_L1`, `f_L2`, the
net `f_layers`, the gross `G = Σ|f_k|`, the maximum cumulative deviation `D`, `f_total`, the measured `q_T`, the
weight-only `g_E`, the neuron concentration `n_80` (absolute mass, deterministic ties), signed masses and top-20
neurons per template, the frozen class, and the data-derived stratum; per stratum the consistent opposers and
supporters; the summary; the descriptive predictor check.

## Commands

```bash
uv run python experiments/010-read-direction-assembly/run.py validate
uv run python experiments/010-read-direction-assembly/run.py explore
uv run python experiments/010-read-direction-assembly/run.py report
```

Boundaries: every phase refuses to run out of order or twice; identity, replication, and software failures are
incidents recorded with their commit and block another `explore` at that commit; the ledger records every executed
prompt (the 24 frames' cue and reference prompts) and noun. Design revision 3 (pre-measurement): a token's
own-reference frames are uninformative for it, the two reference cues are reported but excluded from the strata,
consensus, overlap, and concentration flag, `n_80` is undefined on zero mass, token-level `G`/`D` are means of
per-frame values, and the neuron-sum identity is normalized by the functional's natural scale. `g_E` is weight-only
apart from the frozen axis `d̂_T`.

## Status — 2026-09-18: the single run is complete; frozen summary `MIXED`, with a consistent encoding-borne pattern

`explore` ran once on protocol/code commit `0a6836b` (run `bf60efde064bd9f5`, results state sha256
`a9f8d7031207becd4d0390fb0e7ea4d7c37d28e6a500950fed0033d69a27c3da`; A0 passed). The report — ledger, component table,
neurons — is copied verbatim to [`evidence/exploration-report-2026-09-18.md`](evidence/exploration-report-2026-09-18.md).
Ledger: 72 prompt keys (the 24 frames' cue and reference prompts), 80 noun keys. Replication of Experiments 006 (192),
007 (144), and 009 (138) exact; ρ identity max relative error 1.1e-7; P1 cross-check 5.8e-16; neuron-sum 2.0e-8; head
reconstruction 0.0. Experiment 010 is closed; nothing is amended or rerun.

### Frozen classification (mechanical)

- Strata by measured `q_T` (means over informative frames): **low** `a` 0.20, `this` 0.03, `another` 0.17, `every`
  −0.13, `that` 0.33, `her` 0.35; **mid** 18 tokens (`the`, `any`, `no`, `single`, `red`, `old`, `fresh`, `either`,
  `neither`, `more`, `most`, `enough`, `my`, `your`, `his`, `our`, `their`, `blue`); **high** 37 tokens; the two reference
  cues reported separately.
- Every low-stratum token is classed `MIXED`; the summary is **`MIXED`**; no component is a consistent opposer or
  supporter for either stratum; `CONCENTRATED_ENCODING` is absent (`n_80` 320–700 of 2048 everywhere).
- Why `MIXED` rather than `E_BORNE`: the relay condition requires the gross layer change `G = Σ|f_k| ≤ 0.25`, and
  **no token in the pool has `G` below 0.29** (median 0.44, max 0.71): layers 1–2 always move the read score around
  by a few tenths in gross even when their net effect is near zero. The bound was frozen without a baseline for this
  quantity and is too strict for this network; that is recorded as the reason, not corrected after the fact.

### What the ledger shows (descriptive)

Fractions of the template plural cue's measured head-output change, means over the 24 frames:

| token | q_T | f_E | f_∥ (axis part of E) | f_⊥ (rest of E) | net layers 1–2 | gross G | L02.MLP |
|---|---|---|---|---|---|---|---|
| `this` | 0.03 | 0.20 | +0.37 | **−0.17** | −0.06 | 0.32 | +0.01 |
| `a` | 0.20 | 0.10 | +0.46 | **−0.35** | +0.05 | 0.36 | +0.01 |
| `another` | 0.17 | 0.13 | +0.40 | **−0.27** | +0.07 | 0.30 | +0.06 |
| `every` | −0.13 | −0.07 | +0.17 | **−0.24** | −0.07 | 0.31 | −0.10 |
| `that` | 0.33 | 0.46 | +0.39 | +0.07 | −0.08 | 0.42 | +0.01 |
| `these` | 0.97 | 0.83 | +0.52 | +0.32 | +0.03 | 0.38 | +0.08 |
| `three` | 0.93 | 0.88 | +0.87 | +0.01 | +0.08 | 0.43 | +0.08 |
| `cardinal:pl` (`two`) | 1.02 | 0.91 | +0.89 | +0.02 | +0.08 | 0.35 | +0.04 |
| numerals `eleven`–`seventeen` | 0.89–0.96 | 0.73–0.80 | 0.69–0.74 | 0.00–0.06 | +0.24–0.34 | 0.54–0.67 | +0.23–0.31 |

- **The opposition is already inside the encoding.** For `this`, `a`, `another`, `every` the part of `ΔE` along the
  encoding number axis reads +0.37 / +0.46 / +0.40 / +0.17 of the plural cue's signal through the head's direction,
  and the rest of `ΔE` reads −0.17 / −0.35 / −0.27 / −0.24 — negative in **24 of 24 frames for each of the four** —
  leaving `f_E` at 0.20 / 0.10 / 0.13 / −0.07. Layers 1–2 change the read score by at most 0.07 in net for these
  tokens. This is the encoding-borne pattern of H_E in everything but the gross bound.
- **The head reads the encoding almost directly.** With no fitted parameter, the encoding read `f_E` alone ranks the
  63 tokens' measured transport with Spearman 0.925 (MAE 0.111); the weight-only score `g_E` (weights plus the frozen
  axis `d̂_T`) gives Spearman 0.925, MAE 0.064; the full P1 read `f_total` gives 0.909, MAE 0.051.
- **Layers 1–2 amplify strong number signals but do not create the suppression.** `L02.MLP` adds +0.16 on average
  for the high stratum (+0.23–0.31 for the six new numerals) and ≈ 0 for `a`, `this`, `another` (−0.10 for `every`);
  the layer-1 heads and MLP contribute at most a few hundredths each. No component meets the frozen 75%/75%
  consistency rule in either stratum.
- **The encoding's signal is distributed, not concentrated.** `n_80` is 320–700 neurons of 2048 for every token; the
  positive neuron mass is similar across tokens (+1.1 to +1.7 of the plural cue's signal per template), while the
  negative mass separates them: −0.93 / −1.25 / −0.92 / −1.38 for `this` / `a` / `another` / `every` against −0.36 /
  −0.59 for `two` / `several` (cardinal frames). Neuron 1133 is the strongest positive contributor for the plural cues
  (+0.06 to +0.14) and the strongest negative one for `a` (−0.21 to −0.39) and `another` (−0.10 to −0.24); neuron 605 is
  `this`'s strongest opposer (−0.08 to −0.13); neuron 148 dominates every token in the quantifier frames. Top-20
  overlap with the plural cue: 0.37–0.39 (low stratum) versus 0.48–0.57 (high).

### Hypothesis handed to Experiment 011

The number signal that `L03.H04` transports is assembled in the layer-0 MLP encoding itself: `E(w)` carries a
number-axis component and an off-axis component that the head reads with the opposite sign, and the balance of the
two — spread over hundreds of neurons, with a small set of strong opposers for the suppressed cues — is what the head
reads; layers 1–2 mostly relay it (with a gross churn of ≈ 0.3–0.5 that the frozen relay bound did not anticipate)
and amplify strong signals in `L02.MLP`. The candidate zero-parameter, weight-only rule for a prospective test is
`g_E(w, T)`: the encoding read through the head's direction, normalized by the template plural cue (Spearman 0.925
across the 63 exposed tokens here). Experiment 011 must freeze new cue tokens and frames before testing it, and may set
a relay bound from this baseline (median `G` 0.44) rather than from intuition.

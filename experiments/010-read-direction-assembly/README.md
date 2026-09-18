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

## Status — 2026-09-18: implemented; the single run has not happened yet

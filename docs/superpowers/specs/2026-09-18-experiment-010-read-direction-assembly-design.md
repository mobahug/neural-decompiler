# Experiment 010: How the Head-Readable Number Signal Is Assembled — Exact Attribution Through Layers 0–2

**Date:** 2026-09-18

**Status:** Revision 1, for review. Not approved. No Experiment 010 directory or model run exists. Experiments
005–009 are closed and are not amended by this document.

**Kind:** Discovery-only, exact attribution. No predictor is fitted; every number reported is an exact linear
decomposition of a measured quantity, normalized against the template's plural cue. No confirmation set is frozen and
no claim is promoted; the deliverable is a per-token, per-component attribution table with a mechanically derived
summary that becomes the hypothesis of a later prospective experiment.

## Purpose and question

Experiment 009 confirmed prospectively that the selective number transport of `L03.H04` is a fixed-normalization
linear read-out of the residual arriving at the head (Spearman 0.957 on 138 unseen cue–frame pairs): the head reads
one direction, and the opposing components that nearly cancel the number signal of `this`, `a`, `another` were
constructed before the head. It also falsified the idea that a single response-weighted direction of the token-local
encoding predicts transport (`either`/`neither` are transported mid-way, unlike `this`; possessives lower than
predicted), while a full-dimensional ridge from `E` did much better in leave-one-cue-out — the information is
linearly present in `E` across several directions.

> Where, between the token-local encoding and the head's input, are the components that `L03.H04` reads as number
> assembled and opposed? Concretely: how much of the head-readable number signal of a cue is already fixed by `E(w)`
> itself (and by which of its parts), and how much is added or subtracted by the attention heads and MLPs of layers
> 1–2 at the cue position?

## The head-readable number signal (exact, from Experiment 009's confirmed level)

For frame `f` with cue position `c`, reference run `ref`, and the head `h = L03.H04`: with `d̂_T` the head-output number
axis, `m = W_V^h W_O^h d̂_T`, `γ₃` the layer-3 attention LayerNorm weight, and `σ_c^ref` its scale at `c` in the reference
run, the P1 prediction of the head's number-axis output change is

```text
⟨ΔT₁, d̂_T⟩ = (A_c^ref / σ_c^ref) · ρ(Δr_c),        ρ(x) := ⟨ x − mean(x)·1 , γ₃ ⊙ m ⟩
```

`ρ` is a fixed linear functional of the layer-3 input residual change at the cue position — the **read functional**.
Because the residual stream is a sum of component outputs and the E-patch changes nothing before the layer-0 MLP,

```text
Δr_c = ΔE_T(w) + Σ_{k ∈ K} Δout_k(c),        K = { L01.H00 … L01.H07, L01.MLP, L02.H00 … L02.H07, L02.MLP }
ρ(Δr_c) = ρ(ΔE_T(w)) + Σ_k ρ(Δout_k(c))                                     (exact; checked in every run)
```

Every term is a measured activation change projected by the same fixed functional; nothing is fitted. Normalizing by
the plural cue's `ρ(Δr_c)` in the same frame gives fractions that sum to the P1 fraction of Experiment 009:

```text
f_total(w, f) = ρ(Δr_c(w, f)) / ρ(Δr_c(pl_T, f)) = f_E + Σ_k f_k
```

Two further exact splits of the encoding term:

- **axis / orthogonal:** `ρ(ΔE) = ρ(ΔE_∥) + ρ(ΔE_⊥)` along the encoding number axis `d̂_E` (does the part of `E(w)`
  orthogonal to the number axis oppose the number read-out?);
- **neurons:** `ΔE = Σ_j Δa_j(w) · W_out[j]` over the 2048 layer-0 MLP neurons (`a_j = GELU(pre_j)`, weight-only), so
  `ρ(ΔE) = Σ_j Δa_j · ρ(W_out[j])`: which neurons carry the head-readable number signal, which oppose it, and how
  concentrated the signal is.

## Competing hypotheses (decided by the attribution, not by fitting)

- **H_E — encoding-borne.** The opposition is already inside `E(w)`: `f_E` is small for the suppressed cues although
  `f_∥` is large (`ρ(ΔE_⊥)` opposes), and layers 1–2 mostly relay (`Σ_k f_k` small). Then the head's read direction
  `m` already discriminates cues at the encoding, and a zero-parameter, weight-only predictor
  `f_E(w) = ρ(ΔE_T(w)) / ρ(ΔE_T(pl_T))` is the candidate transport rule for a later prospective test.
- **H_L — layer-borne.** `f_E` is large for the suppressed cues and layers 1–2 subtract it: specific components in `K`
  contribute consistently negative fractions for the suppressed cues and not for the numerals (an attention head that
  reads a determiner-like feature, or an MLP that opposes it).
- **H_M — mixed / distributed.** Neither pattern reaches the consensus below.

## Inherited fixed elements

Model, runtime, instrumented path, contrast, E-patch, reference cues and prompts, weight-only `E(w)`, stage axes
(re-estimated on the exposed frames' clean cue pairs, plural positive), `d̂_E`, `d̂_T`, the head weights, the P1
functional, the component split, the results-state, ledger, and incident conventions — all exactly as in Experiments
008–009. Seeds: runtime `20260916`, control `20260923`.

## Exposed pool (all previously executed; nothing fresh)

- **Cue tokens (63):** the forty of Experiment 008 and the twenty-three of Experiment 009's confirmation set.
- **Frames (24):** the eighteen of Experiment 008 and the six of Experiment 009's confirmation set.
- **Nouns:** the eighty exposed nouns (79 single-token), used only for the contrast means that the replication checks
  compare (Experiments 006, 007, and 009 recorded E-patch means on their own frame–token–noun subsets).
- **Data-derived strata (no grammar):** tokens are grouped by their measured Experiment 008/009 transport fraction
  `q_T` (24-frame means recomputed here): **low** `q_T ≤ 0.35`, **mid** `0.35 < q_T < 0.65`, **high** `q_T ≥ 0.65`. The
  strata organize the report; every rule applies to all 63 tokens.

## Measurements (`explore`, once)

- **M1 — E-patch with component captures**, all 63 tokens × 24 frames: replace `L00.MLP` at `c` by `E(w)` in the
  reference prompt; capture every head result and MLP output of layers 1–2 at `c`, `RESID_PRE.L3` at `c`, the head's
  attention row and result at `p_t`, `RESID_POST.L5` at `p_t`, and the logits. Record `ρ(ΔE)`, `ρ(Δout_k)` for every
  `k`, `ρ(Δr_c)`, the identity error `|ρ(Δr_c) − ρ(ΔE) − Σ_k ρ(Δout_k)|`, the measured `q_T`, and the P1 fraction
  (Experiment 009's check that P1 holds is repeated descriptively on the larger pool).
- **M2 — encoding splits** (weight-only, no forward): `ρ(ΔE_∥)`, `ρ(ΔE_⊥)`, and the per-neuron terms
  `Δa_j · ρ(W_out[j])` for every token and template.
- **Replication:** the E-patch mean shifts of the 16 × 12 (Experiment 006 extract), 24 × 6 (Experiment 007 extract),
  and 23 × 6 (a committed extract of Experiment 009's confirmation) pairs must match within 1e-6; the identity
  `Δ_{R0} = ΔE` (1e-4) and the ρ-identity (1e-4 relative to the plural cue's `ρ(Δr_c)`) hold in every run; otherwise
  incidents.

## Definitions

- **Fractions.** `f_E`, `f_∥`, `f_⊥`, `f_k`, `f_total` as above, per (token, frame); the denominator `ρ(Δr_c(pl_T, f))`
  is uninformative if `|ρ(Δr_c(pl_T, f))| < 0.25 · σ_ρ`, where `σ_ρ` is the site-axis scale of `ρ` over the clean cue
  pairs (mean of `ρ(r_c(pl)) − ρ(r_c(sg))` over the frames); token-level values are means over the 24 frames and over
  each template's frames.
- **Layer totals.** `f_L1 = Σ_{k ∈ layer 1} f_k`, `f_L2 = Σ_{k ∈ layer 2} f_k`, `f_layers = f_L1 + f_L2`.
- **Neuron concentration.** For each token and template, the neurons sorted by `|Δa_j · ρ(W_out[j])|`; the smallest
  `n` such that the top-`n` neurons account for 80% of `Σ_j |Δa_j ρ(W_out[j])|` (`n_80`), and the top-20 neurons with
  their signed contributions. The overlap of the top-20 sets between the plural cues and the suppressed cues is
  reported (Jaccard).
- **Consistency of a component.** A component `k` is a **consistent opposer** for a token set `S` if `f_k ≤ −0.10` in
  at least 75% of the frames of at least 75% of the tokens in `S`; a **consistent supporter** if `f_k ≥ +0.10` likewise.

## Frozen classification rules (fixed here, before any Experiment 010 measurement)

Constants: `s_min = 0.35`, `s_high = 0.65`, `c_min = 0.10`, consensus `0.75`, neuron concentration threshold
`n_80 ≤ 40` (of 2048) for "concentrated".

1. **Per token (24-frame means):**
   - `E_BORNE` if `f_E ≤ s_min` and `f_∥ ≥ s_min` and `|f_layers| ≤ c_min`;
   - `LAYER_BORNE` if `f_E ≥ s_min` and `f_layers ≤ −(f_E − s_min)` (layers 1–2 remove at least what would have
     exceeded the low band) and `f_total ≤ s_min`;
   - `RELAYED` if `|f_total − f_E| ≤ c_min` and `f_total ≥ s_high` (an ordinary plural-like cue: the encoding's
     signal passes through);
   - `AMPLIFIED` if `f_layers ≥ +c_min` and `f_total ≥ s_high`;
   - `MIXED` otherwise.
2. **Per stratum:** the modal per-token class with its fraction; the consistent opposers and supporters of the low
   stratum and of the high stratum (rule above), with the components' identities.
3. **Experiment-level summary** over the low stratum: `E_BORNE` if ≥ 75% of its tokens are `E_BORNE`; `LAYER_BORNE`
   (naming the consistent opposers) if ≥ 75% are `LAYER_BORNE` and at least one consistent opposer exists;
   `MIXED` otherwise. `CONCENTRATED_ENCODING` is appended if, for the plural cues and for the low stratum alike,
   `n_80 ≤ 40` in every template.
4. **Descriptive predictor check (no fitting, reported only):** the Spearman correlation and MAE between `f_E(w)`
   (weight-only) and the measured `q_T(w)` over the 63 tokens, and the same for `f_total(w)`. These numbers seed
   Experiment 011's prospective test of the zero-parameter rule if `E_BORNE`; they gate nothing.
5. Incidents (replication, identities, software defects) stop the phase, are recorded with their commit, block
   another `explore` at that commit, and are never a summary label.

## Interpretation limits

- The read functional `ρ` is Experiment 009's P1 level, confirmed at Spearman 0.957 but not exact; attributions of
  the remaining ~4% (LayerNorm scale, attention redistribution) are not made.
- Component attributions are exact decompositions of a sum; they say what each component's output change contributes
  to the read-out, not what would happen if the component were removed (no ablation is run).
- Neuron attributions are weight-only and exact for `E`; they describe the layer-0 MLP's contribution to one
  functional, not "number neurons" in general.
- Nothing generalizes beyond the pinned checkpoint, the three templates, and the 63 tokens; the strata are
  data-derived from exposed measurements.

## Minimal implementation boundary

A module `read_assembly.py` reusing `head_transport.py` (head weights, P1 functional pieces), `cue_suppression.py`
(pool, stage axes, replication), `supervised_subspace.py` and `cue_decompilation.py` (extracts, results-state
patterns), and `plural_mechanism.py`; a committed extract of Experiment 009's confirmation E-patch means; the pool
builder for 63 tokens × 24 frames; the read functional and its exact decompositions as pure tensor functions with
tests (identity on the fake; planted component changes; neuron split reconstructs `ΔE`); the frozen rules on synthetic
tables; a runner with phases `validate`, `explore`, `report`.

## Approval and stopping condition

Design first; no implementation until approved. `explore` runs once after the implementation review; the attribution
table and the summary are the deliverable; the summary's hypothesis (and, if `E_BORNE`, the zero-parameter rule
`f_E`) is handed to the Experiment 011 design, which freezes new cue tokens and frames before any prospective test.

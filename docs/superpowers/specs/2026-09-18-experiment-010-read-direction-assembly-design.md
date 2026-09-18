# Experiment 010: How the Head-Readable Number Signal Is Assembled — Exact Attribution Through Layers 0–2

**Date:** 2026-09-18

**Status:** Revision 3 — pre-measurement clarifications from the independent implementation review (the reference
cues' own-reference frames, the reference cues' exclusion from the strata, undefined concentration on zero mass, and
three frozen readings). Revision 2 was conceptually approved. No Experiment 010 model run exists. Experiments 005–009
are closed and are not amended by this document.

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
axis, `m = W_V^h W_O^h d̂_T`, `γ₃` the layer-3 attention LayerNorm weight, `σ_c^ref(f)` that LayerNorm's scale at `c` in
the frame's reference run, and `A_c^ref(f)` the head's reference attention weight from `p_t` to `c`, Experiment 009's
confirmed P1 prediction of the head's number-axis output change is exactly

```text
⟨ΔT₁, d̂_T⟩ = ρ_f(Δr_c),        ρ_f(x) := (A_c^ref(f) / σ_c^ref(f)) · ⟨ x − mean(x)·1 , γ₃ ⊙ m ⟩
```

`ρ_f` is the **exact P1 read functional**: linear in `x`, with the frame's frozen reference statistics as its scale
(they depend on the reference run only, so they are the same for every token in the frame). It is *not* weight-only.
Because the residual stream is a sum of component outputs and the E-patch changes nothing before the layer-0 MLP,

```text
Δr_c = ΔE_T(w) + Σ_{k ∈ K} Δout_k(c),        K = { L01.H00 … L01.H07, L01.MLP, L02.H00 … L02.H07, L02.MLP }
ρ_f(Δr_c) = ρ_f(ΔE_T(w)) + Σ_k ρ_f(Δout_k(c))                                 (exact by linearity; checked in every run)
```

Every term is a measured activation change under the same fixed functional; nothing is fitted. The fractions are
normalized exactly as Experiment 009 normalized its P1 level — by the template plural cue's **measured** head-output
change in the same frame — so that `f_total` **is** Experiment 009's P1 fraction, term for term:

```text
f_total(w, f) = ρ_f(Δr_c(w, f)) / ⟨ΔT(pl_T, f), d̂_T⟩ = f_E + Σ_k f_k,      f_E = ρ_f(ΔE_T(w)) / ⟨ΔT(pl_T, f), d̂_T⟩
```

(Normalizing by the plural cue's own `ρ_f(Δr_c)` instead would cancel `A_c^ref/σ_c^ref` but would not reproduce 009's
fraction; both denominators are recorded, the first is the one the rules use.)

Two further exact splits of the encoding term, plus a genuinely weight-only score kept **separate**:

- **axis / orthogonal:** `ρ_f(ΔE) = ρ_f(ΔE_∥) + ρ_f(ΔE_⊥)` along the encoding number axis `d̂_E` (does the part of
  `E(w)` orthogonal to the number axis oppose the number read-out?).
- **neurons:** `ΔE = Σ_j Δa_j(w) · W_out[j]` over the 2048 layer-0 MLP neurons (`a_j = GELU(pre_j)`, weight-only), so
  `ρ_f(ΔE) = Σ_j c_j` with `c_j = Δa_j · ρ_f(W_out[j])`: which neurons carry the head-readable number signal, which
  oppose it, and how concentrated the signal is.
- **weight-only m-score (for seeding a later prospective rule; not used by the rules below):**
  `g_E(w, T) = ⟨ΔE_T(w) − mean·1, γ₃ ⊙ m⟩ / ⟨ΔE_T(pl_T) − mean·1, γ₃ ⊙ m⟩` — a ratio of encoding-level reads with no
  activation statistics; it equals `f_E(w, f) / f_E(pl_T, f)` only where the reference scalars cancel, and is reported
  beside `f_E`, never in its place.

## Competing hypotheses (decided by the attribution, not by fitting)

- **H_E — encoding-borne.** The opposition is already inside `E(w)`: `f_E` is small for the suppressed cues although
  `f_∥` is large (`ρ_f(ΔE_⊥)` opposes), and layers 1–2 make only small changes to the read score — small in gross, not
  merely in net. Then the head's read direction already discriminates cues at the encoding, and the weight-only
  `g_E` is the candidate zero-parameter transport rule for a later prospective test.
- **H_L — layer-borne.** `f_E` is large for the suppressed cues and the residual updates of layers 1–2 — themselves
  caused by the E intervention — supply the dominant change in the head-readable score that produces the low final
  value: specific components in `K` contribute consistently negative fractions for the suppressed cues and not for
  the high-transport cues. This is a statement about where the additive residual updates change the read score, not
  a claim that information "originates" in a layer independently of the cue.
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
- **Data-derived strata (no grammar):** tokens are grouped by their measured transport fraction `q_T` (means over
  their informative frames, recomputed here): **low** `q_T ≤ 0.35`, **mid** `0.35 < q_T < 0.65`, **high** `q_T ≥ 0.65`.
  The strata organize the report; every rule applies to every token that has a stratum. **The two reference cues**
  (`one`, `each`) are attributed and reported like every other token but are **excluded from the strata, the
  consensus counts, the neuron overlap, and the concentration flag**: in their own templates their E-patch is the
  identity (nothing to attribute), and in the other templates their `ΔE` is a singular-versus-singular difference,
  not the cue-versus-reference contrast the strata are about.
- **Own-reference frames are uninformative.** A frame in which the token is the frame's reference cue (`ΔE ≡ 0`) is
  excluded from that token's informative frames and from its per-template neuron statistics.

## Measurements (`explore`, once)

- **M1 — E-patch with component captures**, all 63 tokens × 24 frames: replace `L00.MLP` at `c` by `E(w)` in the
  reference prompt; capture every head result and MLP output of layers 1–2 at `c`, `RESID_PRE.L3` at `c`, the head's
  attention row and result at `p_t`, `RESID_POST.L5` at `p_t`, and the logits. Record `ρ(ΔE)`, `ρ(Δout_k)` for every
  `k`, `ρ(Δr_c)`, the identity error `|ρ(Δr_c) − ρ(ΔE) − Σ_k ρ(Δout_k)|`, the measured `q_T`, and the P1 fraction
  (Experiment 009's check that P1 holds is repeated descriptively on the larger pool).
- **M2 — encoding splits** (no additional forward; the reference scalars come from M1's reference runs):
  `ρ_f(ΔE_∥)`, `ρ_f(ΔE_⊥)`, the per-neuron terms `c_j = Δa_j · ρ_f(W_out[j])` for every token and frame, and the
  weight-only `g_E` per token and template.
- **Replication:** the E-patch mean shifts of the 16 × 12 (Experiment 006 extract), 24 × 6 (Experiment 007 extract),
  and 23 × 6 (a committed extract of Experiment 009's confirmation) pairs must match within 1e-6; the identity
  `Δ_{R0} = ΔE` (1e-4) and the ρ-identity (1e-4 relative to the plural cue's `ρ_f(Δr_c)` in the frame) hold in every
  run; the P1 cross-check `ρ_f(Δr_c) = ⟨ΔT₁, d̂_T⟩` (1e-6 relative) and the neuron-sum identity `Σ_j c_j = ρ_f(ΔE)`
  (1e-4 relative to the functional's natural scale `(A_c^ref/σ_c^ref)·‖γ₃ ⊙ m‖·max(‖E(w)‖, ‖E(ref_T)‖)`, because `ΔE`
  is a float32 quantity) hold in every run; otherwise incidents. Per (token, frame) the raw `ρ_f(ΔE)`, every
  `ρ_f(Δout_k)`, `ρ_f(Δr_c)`, the measured head change, both denominators, and the identity errors are recorded.

## Definitions

- **Fractions.** `f_E`, `f_∥`, `f_⊥`, `f_k`, `f_total` as above, per (token, frame); the frame is uninformative if
  `|⟨ΔT(pl_T, f), d̂_T⟩| < 0.25 · σ_T` (Experiment 008/009's convention for the head stage); token-level values are
  means over the informative frames of the 24 and over each template's frames.
- **Layer totals, gross change, and cumulative deviation.** `f_L1 = Σ_{k ∈ layer 1} f_k`, `f_L2 = Σ_{k ∈ layer 2} f_k`,
  `f_layers = f_L1 + f_L2` (the *net* layer contribution); `G = Σ_k |f_k|` (the *gross* layer change); and, along the
  frozen component order `L01.H00 … L01.H07, L01.MLP, L02.H00 … L02.H07, L02.MLP` (heads in index order, then the MLP,
  a convention — within a layer the attention heads and the MLP act in parallel on the same input),
  `D = max_j |Σ_{k ≤ j} f_k|`, the maximum cumulative deviation of the read score from `f_E` while the residual updates
  accumulate. Layer-level cumulative values `|f_L1|` and `|f_L1 + f_L2|` are reported as well. **Token-level `G` and
  `D` are the means over the token's informative frames of the per-frame values** (the stricter relay test: the mean
  of absolute values bounds the absolute value of the mean).
- **Neuron concentration (deterministic).** For each token and frame, `c_j = Δa_j · ρ_f(W_out[j])`; the neurons are
  sorted by `|c_j|` descending with ties broken by neuron index ascending; `n_80` is the smallest `n` such that
  `Σ_{top-n} |c_j| ≥ 0.8 · Σ_j |c_j|` (absolute attribution mass, so cancellation cannot shrink it); `n_80` is
  undefined (`None`) when `Σ_j |c_j| = 0`. Reported per token and template on the template's mean `c_j` over the
  token's informative frames of that template (one deterministic set per token and template): `n_80`, the positive
  mass `Σ_{c_j > 0} c_j`, the negative mass `Σ_{c_j < 0} c_j`, and the top-20 neurons by `|c_j|` with their signed
  mean contributions; the per-frame `n_80` mean is reported beside it.
- **Top-20 overlap.** For each template, the top-20 set of the template's plural cue (from its mean `c_j`) is compared
  with the top-20 set of every token of the frozen low-`q_T` stratum by Jaccard index; the per-template mean Jaccard
  over the low stratum and the same statistic for the high stratum are reported. "Suppressed" always means the frozen
  low stratum.
- **Consistency of a component.** A component `k` is a **consistent opposer** for a token set `S` if `f_k ≤ −0.10` in
  at least 75% of the informative frames of at least 75% of the tokens in `S`; a **consistent supporter** if
  `f_k ≥ +0.10` likewise.

## Frozen classification rules (fixed here, before any Experiment 010 measurement)

Constants: `s_min = 0.35`, `s_high = 0.65`, `c_min = 0.10` (net), `g_max = 0.25` (gross and cumulative), consensus
`0.75`, neuron concentration threshold `n_80 ≤ 40` (of 2048) for "concentrated".

1. **Per token (means over informative frames):** "layers relay" requires small net **and** small gross **and** small
   cumulative change: `|f_layers| ≤ c_min` and `G ≤ g_max` and `D ≤ g_max` (`RELAY`).
   - `E_BORNE` if `f_E ≤ s_min` and `f_∥ ≥ s_min` and `RELAY`;
   - `LAYER_BORNE` if `f_E ≥ s_min` and `f_layers ≤ −(f_E − s_min)` (the layers' residual updates remove at least what
     would have exceeded the low band) and `f_total ≤ s_min`;
   - `RELAYED` if `f_total ≥ s_high` and `RELAY`;
   - `AMPLIFIED` if `f_layers ≥ +c_min` and `f_total ≥ s_high`;
   - `MIXED` otherwise (including every case where large layer updates cancel in net).
2. **Per stratum:** the modal per-token class with its fraction; the consistent opposers and supporters of the low
   stratum and of the high stratum (rule above), with the components' identities.
3. **Experiment-level summary** over the low stratum: `E_BORNE` if ≥ 75% of its tokens are `E_BORNE`; `LAYER_BORNE`
   (naming the consistent opposers) if ≥ 75% are `LAYER_BORNE` and at least one consistent opposer exists;
   `MIXED` otherwise. `CONCENTRATED_ENCODING` is appended if `n_80 ≤ 40` for **each plural cue in its own template
   (`two` in cardinal and coordinated-adjective, `several` in quantifier) and for every low-stratum token in every
   template** (undefined values count as not concentrated).
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

## Revision history

- **Revision 3** (pre-measurement, from the independent implementation review): the reference cues' own-reference
  frames are uninformative for them (their E-patch is the identity), and the two reference cues are excluded from the
  strata, consensus counts, neuron overlap, and concentration flag — with the pool as frozen they would otherwise have
  entered the low stratum with structurally zero records and made the 75% consensus unreachable; `n_80` is undefined
  on zero mass; token-level `G`/`D` are means of per-frame values; the concentration flag's token set is fixed (each
  plural cue in its own template, every low-stratum token in every template); the neuron-sum identity is normalized
  by the functional's natural scale rather than by the token's own `|ρ_f(ΔE)|` (which can be small by hypothesis);
  the raw per-frame records are persisted. No threshold or measurement changed.
- **Revision 1** (commit `a8d422f`): initial draft. Reviewed: direction approved with three corrections — the read
  functional omitted P1's frozen reference scalars `A_c^ref` and `1/σ_c^ref` while claiming to equal 009's P1
  fraction; `E_BORNE` inferred "layers relay" from a small *net* layer contribution, which large cancelling layer
  updates would satisfy; the neuron concentration `n_80` and the top-20 overlap were not defined deterministically
  on absolute attribution mass and across frames.
- **Revision 2**: (1) `ρ_f` carries the frame's frozen reference scalars and the fractions are normalized by the plural
  cue's measured head-output change, so `f_total` equals Experiment 009's P1 fraction term for term; the weight-only
  `g_E` is defined separately and never substituted for `f_E`. (2) Relay requires small net (`|f_layers| ≤ 0.10`),
  small gross (`G = Σ|f_k| ≤ 0.25`), and small maximum cumulative deviation (`D ≤ 0.25`) along the frozen component
  order; `E_BORNE` and `RELAYED` use it; cancelling layer updates fall to `MIXED` or `LAYER_BORNE`. (3) `n_80` is
  defined on absolute attribution mass with deterministic sorting and tie-breaking; positive and negative masses and
  the signed top-20 contributions are reported; the top-20 overlap is frozen as a per-template Jaccard between the
  plural cue's set and each low-stratum token's set, both on template-mean contributions. (4) `LAYER_BORNE` is worded as
  "the layers' residual updates supply the dominant change in the head-readable score", not as information originating
  in a layer. No threshold other than the added gross/cumulative bound changed; no measurement changed.

# Experiment 022: Prospective Localization of the Upstream `Δx3` Error — Omitted Block-0 Attention in the Cue State, and the Coordinated Layer-1–2 Reductions

**Date:** 2026-09-23

**Status:** Revision 3. This is one correction to the approved revision 2 (`0f1009c`), found while writing the
implementation plan and approved on its review.

The calibration stop now triggers at **250 or more** undefined values of a one-sided statistic (C1, C2, C4), and at
**125 or more** of C3's. Revision 2 said "more than 250" and "more than 125". Under the frozen ±∞ convention, exactly
250 undefined values already make `v₍₂₅₀₎ = −∞` and `v₍₉₇₅₁₎ = +∞`, and exactly 125 make C3's bounds infinite.

Nothing else changes. Revision 2 had made the review's precision changes to revision 1 (`2b20db3`): exact tail
directions, C4's wording, the gap rule as per-condition interpretability, eight individually reported conditions, and
validity selecting nothing. They are listed in the revision history.

Nothing is implemented: no Experiment 022 directory, code, results state, confirmation file, calibration record, lock
or model run exists. This document freezes the question, the
attribution, the four claims, the identity gates, the fresh-unit candidate lists and their freeze rule, the calibration
procedure and the floor rule. **No floor has been computed, and no fresh unit has been frozen or executed.** The only
numbers quoted are Experiments 020/021's committed results and an exploratory spike on the exposed pool (below). The
spike ran scratch scripts that are not part of the repository, and it executed only prompt keys already in Experiment
020's exposed ledger.

**Kind:** Prospective, zero-parameter localization. It tests where the committed program's upstream error comes from.
It builds no corrected program, fits nothing, and changes no inherited object.

## Background: what Experiments 020 and 021 left open

Experiment 021 closed at `86f7f8c` with `CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL |
NOUN_READOUT_FIXED`. The committed program's end-to-end flattened `R²` was 0.6167 for fresh cues in the exposed frames
and 0.5797 in fresh frames. Fed the *measured* `Δx3`, the same downstream readout reached 0.9683 and 0.9652. Of the
unexplained variance, 0.352 of 0.383 (Y1) and 0.385 of 0.420 (Y2) was inherited from the upstream prediction of `Δx3`;
Experiment 020 had found the same split on exposed data. The downstream readout is decoded almost exactly. The open
problem is the upstream prediction of `Δx3`.

## What `Δx3` is, and where its error can come from

`Δx3` is the change, caused by the cue, of the residual stream **entering block 3** (`RESID_PRE.L3`) at the cue
position `p_c` and the target position `p_t`. In the cue-final templates (cardinal, quantifier) `p_t = p_c`; in the
coordinated-adjective template `p_t = p_c + 1`.

The transport head `L03.H04` is in block 3, so it belongs to the **downstream** program. Given the measured `Δx3`, that
program computes the head's query, key and attention exactly, and the ceiling (0.97–0.98) already includes it. H04
routing is therefore not a candidate: the upstream error lives in blocks 0–2.

The upstream program is the committed Experiment 017 chain, `HeadChainModel.upstream` at variant `LEVEL0`, built from
the Experiment 011/012/017 locks. It takes a cue input `ΔE` at `p_c` and passes it through:

- Experiment 016's Level 0-F layer-1 and layer-2 rows at `p_c` (template-base normalization, the frame's operands and
  scale);
- block 1's MLP at the frame's operating point, and block 2's MLP (channel D);
- in coordinated frames, the exact one-step propagation to `p_t`, which assumes `x1(p_t)` is unchanged.

Its cue input is Experiment 011's read, `read.encoding_delta = lexicon_vector(cue) − lexicon_vector(ref)`, which is
`MLP₀(LN₂(W_E[cue])) − MLP₀(LN₂(W_E[ref]))`. This is the Experiment 005/011 E-patch, the `L00.MLP` output at `p_c`
only. Swapping the cue in the real prompt also changes three things the chain does not carry: the embedding itself at
`p_c`; block 0's attention output at `p_c`; and, in coordinated frames, block 0's attention output at `p_t`.

## The exploratory spike (exposed pool only; it motivates the claims and scores nothing)

**Exactness.** Block 0 was computed from the prompt's embedding rows and block 0's weights (`atp.LayerProgram` at layer
0). Layers 1–2 were computed exactly with Experiment 017's `exact_chain`, generalized to inputs at both changed
positions. Together they reproduce the measured `Δx1` to 1.3e-5 absolute and the measured `Δx3` to at most 8.5e-6
relative (median 3.1e-6, 4,320 pairs). The composition with every factor included reproduces the ceiling's predicted
`Δc` to at most 3.6e-5 nats. Every quantity below is an exact, weight-only function of the frame's reference prefix and
the cue's token id, so the gap decomposes exactly.

**Ladder** (flattened `Δc` `R²`; 108 exposed frames × 40 exposed cues = 4,320 pairs; order-dependent, descriptive):

| step | all | cardinal | quantifier | coordinated |
|---|---|---|---|---|
| Level 0 (the committed chain) | 0.719 | 0.879 | 0.883 | 0.324 |
| exact layers 1–2, same E-patch input | 0.838 | 0.880 | 0.883 | 0.723 |
| + the embedding change at `p_c` | 0.850 | 0.889 | 0.890 | 0.742 |
| + block-0 attention at `p_c` (= the measured `Δx1(p_c)`) | 0.959 | 0.981 | 0.972 | 0.915 |
| + block-0 attention at `p_t` (= the measured `Δx3`) | 0.978 | 0.981 | 0.972 | 0.979 |
| the committed reduced layers 1–2, fed the full input | — | 0.981 | 0.972 | 0.849 |

**Order-free attribution** (exact Shapley values of the gap between Level 0 and the ceiling; 108 frames × 14 exposed
cues = 1,512 pairs; the factors are defined below):

| share of the gap | cue-final (gap 0.104) | coordinated (gap 0.982) |
|---|---|---|
| `Bv`, block-0 value term at `p_c` | 0.586 | 0.318 |
| `Bp`, block-0 pattern term at `p_c` | 0.350 | 0.079 |
| `T`, block-0 attention into `p_t` | — | 0.155 |
| `emb`, embedding change at `p_c` | 0.063 | 0.004 |
| `R`, layer-1–2 reductions | 0.001 | 0.444 |

**Supporting facts:**
- Median norms: `ΔE` 7.7, embedding change 0.95, value term about 4.0, pattern term about 2.4, block-0 change at `p_t`
  about 3.0.
- The value term is carried by strongly self-attending block-0 heads: `L00.H03` (reference self-weight about 1.0),
  `H02`, `H05` and `H07`.
- The copy into `p_t` comes from heads that attend back to the cue: `L00.H05` (0.60 of its attention on `p_c`), `H06`
  and `H01`.
- In coordinated frames, substituting the measured `Δx3` at `p_c` alone lifts Level 0 from −0.01 to 0.886; at `p_t`
  alone, to 0.317.
- Every cue class recovers (cue-final 0.94–0.995, coordinated 0.95–0.995). Experiment 020's worst cues were this
  omission:
  `whole` goes from −0.66 to 0.92 once block-0 attention at `p_c` is included, and `last` from −0.50 to 0.89.

The spike chose the claims' directions and meaning guards below. It sets no floor: every floor comes from the frozen
calibration, and the spike's values are quoted only as context.

## Question

Does the spike's localization generalize, prospectively, to new cues and new frames? Precisely:

- **Cue-final (H-CF).** The dominant remaining upstream error is omitted block-0 attention in the cue-state
  construction — its value and pattern terms at `p_c` — and the layer-1–2 reductions contribute little.
- **Coordinated (H-CO).** The error is distributed across incomplete layer-0 state construction — including the block-0
  copy pathway from the cue into the target position — and the reduced layer-1–2 computation. It is **not** claimed
  that all coordinated error originates in layer 0.

The operative reading of each claim is 021's: *does the fresh population's attribution lie within the envelope of
exposed-like sets of the same size and composition, and does it keep the claim's qualitative meaning?*

## The frozen objects

### The five factors

For a cue token `t` in frame `f`, with reference token `r` (the template's reference cue; the reference prompt is the
frame with `r`), let `x0_k = W_E[token_k]` be the reference prompt's embedding rows. Pythia has no positional embedding;
rotary position enters inside attention.

- `ΔE = MLP₀(LN₂(W_E[t])) − MLP₀(LN₂(W_E[r]))` — the committed cue input (`read.encoding_delta`), always included.
- `emb`: `Δemb = W_E[t] − W_E[r]`, the embedding's own change at `p_c`.
- Block 0's attention output change at query `q` is `ΔA0(q)`. It is block 0's exact attention program at the reference
  prefix, with the embedding at `p_c` replaced by `W_E[t]`: the query at `q = p_c` and the key and value at `p_c` are
  recomputed; nothing else changes.
- `Bv`: the **value term** at `p_c`, `V(p_c) = Σ_h A_ref[h, p_c→p_c] · (O_h(v′_h(p_c)) − O_h(v_h(p_c)))`. This is the
  reference attention row carrying the cue key's changed value — Experiment 017's `F` term, applied at block 0.
- `Bp`: the **pattern term** at `p_c`, `P(p_c) = ΔA0(p_c) − V(p_c)`: the changed row carrying the new values —
  Experiment 017's `Π` term, applied at block 0.
- `T`: the block-0 attention change at `p_t`, `ΔA0(p_t)`, in coordinated frames. It is a null factor in cue-final
  frames, where `p_t = p_c`.
- `R`: the layer-1–2 program — the committed reduced program (factor off) or the exact weight-only program (factor on).

By GPT-NeoX's parallel residual, `Δx1(p_c) = Δemb + ΔE + ΔA0(p_c)` and, for `p_t ≠ p_c`, `Δx1(p_t) = ΔA0(p_t)`, exactly.

### The compositions

For every subset `S` of `{R, emb, Bv, Bp, T}` (32 subsets; 16 distinct ones in cue-final frames):

- **Input at `p_c`:** `u(p_c) = ΔE + [emb∈S]·Δemb + [Bv∈S]·V(p_c) + [Bp∈S]·P(p_c)`.
- **Input at `p_t`** (coordinated only): `u(p_t) = [T∈S]·ΔA0(p_t)`.
- **Layers 1–2:**
  - With `R ∈ S`: the exact weight-only program at the frame's reference state. This is Experiment 017's `exact_chain`
    with inputs at both changed positions; with an input at `p_c` only, it equals `exact_chain` bit for bit.
  - Without `R`: the committed reduced program, `HeadChainModel.upstream` at `LEVEL0`, with its cue input replaced by
    `u(p_c)`. A nonzero `u(p_t)` enters the one-step propagation's layer 1 at `p_t` exactly: the query, key and value
    at `p_t`, and block 1's MLP at `x1(p_t) + u(p_t)`.
- **Output:** `Δx̂3^S` at `p_c` and `p_t` goes through the frozen downstream readout program (020's
  `ReadoutProgram.blocks_3_to_5`, `LN_final`, the weight-only read), giving `Δĉ^S` for the 79 scorable exposed nouns.

**Endpoints.** The empty subset is the committed Level 0 exactly (identity I5). The full subset is the exact layers-0–2
program, which reproduces the measured `Δx3` (I3); its `Δĉ` is the downstream ceiling (I4).

### Value function, Shapley values, shares

- **Value function.** For a group of pairs, `v(S) = 1 − SSE(S)/SST`, with `SSE(S) = Σ (Δc − Δĉ^S)²` and
  `SST = Σ (Δc − mean Δc)²` over the group's pairs × 79 nouns, `Δc` measured.
- **Gap.** `G = v(full) − v(∅)`.
- **Shapley values.** Exact, over the five factors:
  `φ_f = Σ_{S ⊆ F∖{f}} |S|!(4 − |S|)!/5! · [v(S ∪ {f}) − v(S)]`.
- **Shares.** `σ_f = φ_f / G`. By efficiency they sum to 1.

Because `v` differences are `−ΔSSE/SST`, the shares are Shapley shares of the reduction in squared prediction error;
they do not depend on `SST`.

**Groups.** *Cue-final* is the cardinal and quantifier pairs pooled; *coordinated* is the coordinated-adjective pairs.

### Not a factor

The downstream program, the noun read and `L03.H04` are fixed, identical in every composition, and not attributed.
Finer localization inside `R` (which of 016/017's reductions) is out of scope; it belongs to a later experiment.

## Identity and isolation gates (hard; a violation is an incident, never a label)

| gate | what must hold | tolerance | exposed spike maximum |
|---|---|---|---|
| I1 | block-0 decomposition: `Δemb + ΔE + ΔA0(p_c)` equals the measured `Δx1(p_c)`; `ΔA0(p_t)` equals the measured `Δx1(p_t)` | max abs 1e-4 | 1.3e-5 / 7.4e-6 |
| I2 | `V(p_c) + P(p_c) = ΔA0(p_c)` (algebra) | 1e-12 | — |
| I3 | the full composition's `Δx̂3` reproduces the measured `Δx3` at every changed position | relative 1e-4 | 8.5e-6 |
| I4 | the full composition's `Δĉ` equals the ceiling's (the downstream program fed the measured `Δx3`) | max abs 1e-3 nats | 3.6e-5 |
| I5 | the empty composition equals the committed Level 0 `Δx̂3` (017's chain) | exactly 0.0 | 0.0 |
| I6 | Shapley efficiency: `Σ_f φ_f = G` | 1e-12 · max(1, \|G\|) | — |
| I7 | lock tables reproduced at confirm before any fresh prompt runs | digest equality | — |
| R1 | calibration: the re-measured exposed `Δc` reproduces 021's digest-bound exposed table | max abs 1e-9 | — |
| L1 | no confirmation-manifest key is executed before `confirm`; `calibrate` executes only keys in Experiment 020's exposed ledger | set checks | — |
| L2 | stage 1 executes only S1-REF and S1-VALIDITY; the barrier re-verifies the stage-1 digest from disk before any S2-TARGET | as in 020/021 | — |
| L3 | the calibration and the lock read no measured fresh quantity, and nothing of Experiment 021's spent confirmation set | assertions | — |

The tolerances are frozen from the spike's exposed maxima, with margins of at least 8×.

## Populations and fresh units

| outcome | population | pairs (cue-final / coordinated) |
|---|---|---|
| **Y1** | 24 new cues × the 108 exposed frames × 79 exposed nouns | 1,728 / 864 |
| **Y2** (frame-conditional) | 24 new cues × 18 new frames × 79 exposed nouns, each frame at its own stage-1 reference state | 288 / 144 |

- **No new noun population.** Experiment 021 established the noun readout, and 022 isolates the upstream problem.
- **No validity-based frame selection.** Frames are frozen structurally, never on model behavior. The validity rule
  selects no frame, no calibration row, no threshold and no confirmation population. All 18 new frames are scored. The
  frozen 017–021 validity rule is evaluated
  and recorded descriptively only. This removes the composition-indexed floor rows of 021.

### Fresh cues

The design fixes four ordered class lists. `freeze` takes, per class, the first 6 entries that are a single token with
a leading space and whose token id is outside the exclusion set: every cue token of Experiment 020's exposed pool
(279), every token of Experiment 020's confirmation set (the 24 cues spent in 021), every template reference and plural
cue, and every token of any committed confirmation file. No model output is involved.

| class | ordered candidates (the first 6 eligible are taken) |
|---|---|
| `determiner-like` | alternate, random, standard, regular, normal, common, general, subsequent, given, chosen, selected |
| `quantity` | bulk, spare, dense, lengthy, lots, loads, tons, piles, masses, stacks, gross, net |
| `possessive-or-pronoun` | thou, yourself, themselves, ones, others, naught, whatsoever |
| `adjective` | square, wild, brave, calm, eager, fierce, humble, proud, shy, lazy, busy, sturdy, fragile, polished, shiny, dusty, hollow, noble |

A tokenizer-only pre-check for this design found 11 / 12 / 7 / 18 eligible entries. The expected picks are:
- `determiner-like`: alternate, random, standard, regular, normal, common;
- `quantity`: bulk, spare, dense, lengthy, lots, loads;
- `possessive-or-pronoun`: thou, yourself, themselves, ones, others, naught;
- `adjective`: square, wild, brave, calm, eager, fierce.

The pronoun class is nearly exhausted by earlier experiments. If any class yields fewer than 6 at `freeze`, the freeze
writes nothing and stops for review. No class is refilled from another.

### Fresh frames

The design fixes three ordered frame lists. `freeze` takes, per template, the first 6 that pass structural checks:
- the text appears in no committed frame set;
- the cue slot is a single token;
- `p_c` lies within the exposed range of its template (cue-final 3–5, coordinated 4–7);
- in coordinated frames, `p_t = p_c + 1`;
- the frame is built by the project's own `pm.Frame` construction.

Frame ids are `<template>-022-<k>`.

| template | ordered candidates (the first 6 structurally eligible are taken) |
|---|---|
| cardinal | The pantry keeps {cue} · The studio makes {cue} · The market trades {cue} · The factory builds {cue} · The kitchen bakes {cue} · The quarry yields {cue} · The dairy produces {cue} · The shelter feeds {cue} · The boutique sells {cue} · The hangar shelters {cue} |
| quantifier | The census counts {cue} · The survey names {cue} · The bulletin lists {cue} · The directory shows {cue} · The agenda names {cue} · The appendix cites {cue} · The timetable shows {cue} · The glossary defines {cue} |
| coordinated-adjective | Hugo and Nina sanded {cue} flat · Ines and Karl trimmed {cue} short · Otto and Vera washed {cue} clean · Ivo and Greta sorted {cue} neat · Mats and Alma cooled {cue} fast · Nora and Paul rinsed {cue} dry · Ella and Tomas stacked {cue} low · Anna and Erik glazed {cue} smooth · Leo and Maja folded {cue} flat |

The tokenizer-only pre-check found the first six of each list eligible. Candidates rejected by it and left out of the
lists: five cardinal texts already used; `Rita and Hans packed {cue} snug`, whose suffix is two tokens; and frames with
`p_c` outside the range.

**Manifest:**
- **S1-REF:** 18, one reference prompt per new frame.
- **S1-VALIDITY:** 18, descriptive only.
- **S2-TARGET:** 2,592 (Y1) + 432 (Y2) = 3,024.

`freeze` writes `experiments/022-upstream-error-localization/confirmation-v1.json` with its digest. It is committed
before `calibrate`. The calibration reads only its class and template counts.

## The claims (frozen)

| claim | group | statistic | direction | envelope `E` (exact order statistic; see below) | meaning guard `M` | exposed spike value |
|---|---|---|---|---|---|---|
| **C1** block-0 attention dominates the cue-final gap | cue-final | `s1 = σ_Bv + σ_Bp` | lower bound | `s1 ≥ F1`, `F1 = v₍₂₅₀₎` (2.5 % lower tail) | `s1 ≥ 0.50` | 0.936 |
| **C2** the layer-1–2 reductions contribute little to the cue-final gap | cue-final | `s2 = σ_R` | upper bound | `s2 ≤ F2`, `F2 = v₍₉₇₅₁₎` (2.5 % upper tail: the 250th largest, about the 97.5th percentile) | `s2 ≤ 0.10` | 0.001 |
| **C3** the coordinated gap is split between layer 0 and the reductions | coordinated | `s3 = σ_R` (so `1 − s3` is the layer-0 share, including `T`) | two-sided | `F3lo ≤ s3 ≤ F3hi`, `F3lo = v₍₁₂₅₎`, `F3hi = v₍₉₈₇₆₎` (1.25 % in each tail) | `0.10 ≤ s3 ≤ 0.90` | 0.444 |
| **C4** block-0 cue→target attention contributes positively to the coordinated gap | coordinated | `s4 = σ_T` | lower bound | `s4 ≥ F4`, `F4 = v₍₂₅₀₎` (2.5 % lower tail) | `s4 > 0` | 0.155 |

C4's guard, `s4 > 0`, establishes a **positive** contribution only. It does not establish a large or substantial one,
and no result of C4 is described that way; the calibrated envelope carries the quantitative comparison.

### Exact order statistics (frozen, so no implementation can reverse a tail)

Per population (Y1-like or Y2-like) and per statistic, sort the calibration's `B = 10,000` values in **ascending**
order: `v₍₁₎ ≤ v₍₂₎ ≤ … ≤ v₍₁₀₀₀₀₎`. Ranks are 1-based; in 0-based code, `v₍k₎` is element `k − 1` of the ascending array.

Undefined values (defined below) are placed at `−∞` when a lower bound is computed and at `+∞` when an upper bound is
computed, so they always count against the envelope.

- **Lower bounds (C1, C4):** `F = v₍₂₅₀₎`, element `[249]`. At most 249 values lie strictly below it.
- **Upper bound (C2):** `F = v₍₉₇₅₁₎ = v₍₁₀₀₀₀ − 250 + 1₎`, element `[9750]`. This is the 250th largest; at most 249
  values lie strictly above it. It is **never** `v₍₂₅₀₎`.
- **Two-sided (C3):**
  - `F3lo = v₍₁₂₅₎`, element `[124]`;
  - `F3hi = v₍₉₈₇₆₎ = v₍₁₀₀₀₀ − 125 + 1₎`, element `[9875]`.

Each claim therefore has α = 0.025 in its own direction. Full float64 precision; nothing is rounded.

**Direction checks** (frozen for the implementation): `F1 ≤ median(s1)`, `F4 ≤ median(s4)`, `F2 ≥ median(s2)`, and
`F3lo ≤ median(s3) ≤ F3hi`. A violation means a tail was reversed; it is an implementation incident, and no floor table
is written.

### Condition results (frozen)

Each claim is evaluated **separately on Y1 and on Y2**: eight conditions, `C1–C4 × Y1/Y2`. Y1 and Y2 are never pooled.
New cues in known frames and new cues in new frames are different generalization tests.

**Interpretability (the gap rule).** A condition is evaluated on its group's aggregate population, in full: C1 and C2
on the cue-final pairs of that population, C3 and C4 on its coordinated pairs. It is interpretable iff that aggregate
has `SST > 0` and `G ≥ 0.02`; below that, shares are ratios of small numbers.

The rule is applied to the aggregate only. **It never removes, re-weights or selects a pair, cue, frame or noun.**
Every scored population is the full frozen one.

Each condition has exactly one result, decided in this order:

| result | when | reading |
|---|---|---|
| `NOT_INTERPRETABLE` | the gap rule fails | Localization is not interpretable because too little Level-0→ceiling gap remains (or `SST = 0`). This is neither a pass nor a failure. |
| `GUARD_FAILURE` | `M` fails | The preregistered qualitative localization claim fails for that population, whatever the envelope. |
| `ENVELOPE_ONLY_FAILURE` | `M` holds, `E` fails | A quantitative shift relative to the exposed-like calibration. The qualitative claim survives. |
| `PASS` | `M` and `E` both hold | The attribution lies within the exposed-like envelope and keeps the claim's meaning. |

A non-finite share where the gap rule holds is an implementation incident.

**The eight results are the result of the experiment.** There is no aggregate label, no hidden pooling, and no
requirement that all eight (or any subset of) envelopes pass before any localization conclusion is reported. Each
condition is reported and read on its own. The report's headline is the 2 × 4 table of results, each with its
statistic, envelope, guard, exposed-like percentile and gap `G`.

A guard can make a `PASS` harder, never easier. A claim whose guard is stricter than its envelope (for example
`F1 < 0.50`) is marked guard-bound in the calibration record. Its exposed-like result rates are computed with both
predicates, and any resulting shortfall is reported and never corrected (021's rule).

## Calibration (exposed only, once)

**Pools.** These are Experiment 021's provenance pools, unchanged. Nothing of 021's spent confirmation set enters.
- **Cues:** the 175 exposed cues of the four classes that entered as fresh cues of the 011–019 confirmations (45 / 45 /
  36 / 49).
- **Y2-like frames:** the 42 exposed frames first confirmed in 017–019, which entered no inherited fit (14 per
  template), unscreened.
- **Y1-like frames:** the 108 exposed frames, fixed.

**Re-materialization.** 175 cues × 108 frames = 18,900 pairs, all in Experiment 020's exposed ledger.
- One forward per pair measures `Δc`, `Δx1` and `Δx3` at `p_c` and `p_t` against 020's locked reference states.
- `Δc` must reproduce Experiment 021's digest-bound exposed table (R1). The local file is verified against the digest in
  021's committed calibration record first.
- I1–I6 are enforced on every pair; the maxima are recorded before enforcement.
- The composition table (16 or 32 subsets per pair) is written under `outputs/experiment-022/` and digested.

The estimated cost is about 20 min of forward passes plus about 2 h of compositions. No confirmation or fresh
information is introduced.

**Draws.**
- `B = 10,000` SHA-indexed base draws, with index `int.from_bytes(sha256(f"022|primary|{b}|{s}|{i}")[:8], "big") % n_s`.
- 24 cues per draw, drawn with replacement by class stratum at the fresh set's frozen class counts.
- Y1-like set: the drawn cues × 108 frames.
- Y2-like set: the drawn cues × 18 frames, drawn with replacement at 6 per template from the 42.
- Per draw: `SSE(S)` for every subset and group from per-pair sums (duplicates counted with multiplicity), then the
  Shapley values, `G`, and `s1`–`s4` for Y1 and Y2.
- A draw in which a group's aggregate has `G < 0.02` or `SST = 0` has no interpretable value for that group's
  statistics. It is ranked by the convention above, so it always counts against the envelope, and it is counted in the
  record.

**Kernel check.** On the first 16 draws, the fast Shapley kernel is checked against a direct recomputation from the
materialized draw, at `|k − d| / max(1, |d|) ≤ 1e-10`. This is implementation-only.

**Floor rule.** Per statistic and population, over its 10,000 values, by the exact order statistics above:

| claim | bound |
|---|---|
| C1 | `F1 = v₍₂₅₀₎` |
| C4 | `F4 = v₍₂₅₀₎` |
| C2 | `F2 = v₍₉₇₅₁₎` (the upper tail, never `v₍₂₅₀₎`) |
| C3 | `[v₍₁₂₅₎, v₍₉₈₇₆₎]` |

The direction checks run before anything is written. The meaning guards are separate predicates, not floors.

**Recorded, never corrected:**
- For each condition, the share of draws that are `PASS`, `ENVELOPE_ONLY_FAILURE`, `GUARD_FAILURE` or
  `NOT_INTERPRETABLE`.
- Descriptively, the share of draws in which all four conditions of a population pass, and in which all eight do.
- The medians and the tails, and which claims are guard-bound.

**Calibration stop.** The stop triggers if **250 or more** values of a one-sided statistic (C1, C2, C4), or **125 or
more** of C3's, are undefined. Under the frozen ±∞ convention, such an envelope would be infinite:
- exactly 250 undefined values already make `v₍₂₅₀₎ = −∞` for C1 and C4, and `v₍₉₇₅₁₎ = +∞` for C2;
- exactly 125 make `v₍₁₂₅₎ = −∞` and `v₍₉₈₇₆₎ = +∞` for C3.

`calibrate` then writes no floor table and stops for review. This is a pre-lock calibration stop for review. It is not
scientific evidence, not an incident, and it is never retried automatically.

The candidate calibration record is installed byte-identically, committed, and reviewed before `lock` (021's
procedure).

## Descriptive records (no outcome force)

- The fixed ladder in the spike's order.
- The `R`-only and inputs-only compositions: the complete layer-0 input with the reduced layers 1–2, reported as a
  number only and never as a corrected program.
- Per-template shares (cardinal against quantifier), per-class shares, and `σ_emb`.
- `Δx3` relative error by composition and position.
- Block 0's head profile: reference self-weights, per-head value-term norms, and the attention from `p_t` to `p_c`.
- The new frames' validity verdicts.
- After `report`, a separately labeled **exploratory replication on Experiment 021's spent set**. It uses 021's stored
  stage-2 measurements, digest-verified, and weight-only compositions, with no new prompt. It can change no floor,
  claim, threshold or label, and it is reported as exploratory.

## Phases and leakage boundaries

| phase | may read | may execute | writes | must not |
|---|---|---|---|---|
| `validate` | frozen inputs, locks, 020/021 closure records and digests | nothing | nothing | load a model |
| `freeze` | the tokenizer, the committed frame and cue sets | nothing | the confirmation file (committed) | read any model output |
| `calibrate` (once) | the above, 020's locked exposed states, 021's exposed table (digest-verified), the weights | the 18,900 exposed keys (all in 020's ledger) | the composition table, the draws, the candidate calibration record | run a manifest key; read 021's spent set |
| *(commit)* | | | the record, installed byte-identically; **stop for the floor review** | |
| `lock` | the committed record, 020's locked states, the locks, the weights | nothing (no forward pass) | the Y1 composition tables (digested), the floor tables, the manifest; the candidate lock | read any fresh measured quantity |
| `confirm` stage 1 | the installed lock, the confirmation file, the weights | S1-REF and S1-VALIDITY only | each new frame's reference state, the Y2 composition tables and their digest | execute any S2-TARGET key |
| barrier | the results state on disk | nothing | — | proceed unless the stage-1 digest re-verifies and no S2-TARGET key is in the ledger |
| `confirm` stage 2 | the lock, the digested stage-1 record, the weights | all S2-TARGET keys, once each | the measurements (written before any gate is enforced), the gates, the scores | execute a key twice or outside the manifest |
| `report` | the results state, the calibration draws (digest-verified) | nothing | the report | |
| `replicate-021` (after `report`) | 021's stored stage-2 tables and stage-1 states (digest-verified) | nothing | a separately labeled exploratory record | touch any 022 floor, claim or label |

## Incidents and interpretability

These follow Experiments 020/021: an implementation or protocol failure stops the phase, is recorded with its commit,
and permits no rerun at that commit. An interruption of `confirm` spends the protocol version.

- **Incidents:**
  - any I1–I7, R1 or L1–L3 violation;
  - a missing or non-finite measurement;
  - a structural failure of a new frame's reference capture.
- **Interpretability:** the gap rule above, per condition, on the full aggregate. A `NOT_INTERPRETABLE` result is
  neither an incident nor an exclusion.

## Interpretation (frozen)

Every condition (`C1–C4 × Y1/Y2`) is read on its own, from its own result. No reading depends on how the other
conditions came out.

- **`PASS`.** The fresh population's attribution lies within the exposed-like envelope and keeps the claim's meaning.
  The spike's localization generalized at that claim on that population.
- **`ENVELOPE_ONLY_FAILURE`.** The meaning guard holds, but the share is outside the calibrated envelope. The fresh
  population's share differs from exposed-like sets beyond sampling variability, but the qualitative statement still
  holds: dominant (C1), little (C2), split (C3), positive (C4). It is reported as a quantitative shift, never as a
  refutation of the qualitative localization. For example, if C2's share leaves its tight envelope but stays `≤ 0.10`,
  the report says the reductions' share shifted relative to exposed-like sets and that "the reductions contribute
  little" still holds.
- **`GUARD_FAILURE`.** The preregistered qualitative claim fails for that population:
  - C1: block-0 attention does not dominate the cue-final gap;
  - C2: the reductions do not contribute little;
  - C3: the coordinated error is not split, with one side below 10 %;
  - C4: the block-0 cue→target attention does not contribute positively.
- **`NOT_INTERPRETABLE`.** Localization is not interpretable on that population's group because too little
  Level-0→ceiling gap remains. It is not a pass or a failure, and no example was removed to reach it.
- **C3's direction** is always reported: below the band means the reductions matter less than in exposed-like sets;
  above it, more.
- **C4** never supports "substantial" or "large" wording; its qualitative content is a positive contribution.
- **Incidents** carry no result.
- **What a pass does not show.** It does not show that any corrected program would predict well: 022 builds none; the
  full composition is the identity endpoint only. It also does not show which of 016/017's reductions carries `σ_R`.

## Interpretation limits

- The attribution uses one value function, flattened `Δc` `R²` (equivalently the reduction in squared error). It is
  chosen because `Δc` is what the project explains; `Δx3` errors are diagnostics.
- The value/pattern split is definitional — the reference row against the row change, as in Experiment 017's `F`/`Π` —
  and is stated as such.
- `R` lumps every 016/017 reduction together.
- One checkpoint, three templates, four cue classes, 18 new frames, 79 exposed nouns.

## Review record

**Spike and decision (2026-09-23).** The reviewer chose Option A: a prospective confirmation of the localization, not
a correction-model experiment. Answers to the design questions:

1. Downstream `Δc` is the primary quantity; `Δx3` error is a diagnostic and identity gate.
2. The exact Shapley attribution is primary; the ladder is descriptive.
3. Floors are calibrated 021-style, with about 3–4 claims rather than a large condition matrix.
4. Confirmation uses new cues and new frames and no new nouns. The spent 021 set is allowed only as a clearly labeled
   exploratory replication after the fresh confirmation, influencing nothing.

The reviewer required the claim wording to separate cue-final (block-0 attention dominant, reductions small) from
coordinated (layer 0, including the copy pathway, *and* the reductions). 022 must not claim that all coordinated error
is in layer 0.

**Review of revision 1 (2026-09-23).** The reviewer approved it in principle and set these decisions for revision 2:

1. **Pooling.** Keep all eight conditions, C1–C4 separately on Y1 and Y2. Do not pool: new cues in known frames and
   new cues in new frames are different tests, and pooling could hide a frame-specific weakness like the one 021
   exposed.
2. **Guards.** Accept `Bv+Bp ≥ 0.50`, `R ≤ 0.10` and coordinated `R ∈ [0.10, 0.90]`. C4's `T > 0` establishes only a
   positive contribution, so the claim is worded "block-0 cue→target attention contributes positively".
3. **Validity.** New-frame validity is descriptive only. Frames are frozen structurally; validity selects no frame,
   calibration row, threshold or population.
4. **Gap rule.** Keep `G ≥ 0.02` as an aggregate, per-condition interpretability rule. It never excludes examples, and
   when it fails the result is "not interpretable", not a pass or a fail.
5. **C2.** Accept its possibly tight envelope. The three-way reading reports a quantitative shift while the qualitative
   claim stands.
6. **Tails.** Fix the tail direction explicitly: lower tails for C1 and C4, the upper tail for C2, and 1.25 % per side
   for C3, with the order-statistic indexing stated unambiguously.
7. **Reporting.** Each of the eight conditions is reported individually, with no requirement that all eight envelopes
   pass before any conclusion is reported.

Everything else in revision 1 was preserved.

**Review of the implementation plan (2026-09-23).** Revision 2 was approved. The plan's review approved one correction
to the design: the calibration stop triggers at 250 or more undefined values (C1, C2, C4) and at 125 or more (C3). It
stays a pre-lock stop for review. This is revision 3. The review's other decisions are implementation-level and are
recorded in the plan: the freeze before calibrate, the I3 norm, the R1 pre-check, the exclusion extraction, the draw
order, the committed lock table, the median, the percentile, and the replication gates.

## Implementation boundary (for the plan, after approval)

- **New module** `src/neural_decompiler/upstream_localization.py`:
  - block 0's attention program and the factor decomposition;
  - the multi-position exact chain, and the reduced chain with its input replaced;
  - the compositions, the Shapley kernel and its direct check;
  - the claims and the floor rule;
  - the calibration, lock, scoring and report.
- **Runner** `experiments/022-upstream-error-localization/run.py`, with phases `validate`, `freeze`, `calibrate`,
  `lock`, `confirm`, `report`, `replicate-021`.
- **Frozen modules.** `readout_decompilation.py` (blob `caa73b40…`), `readout_calibration.py` (021's, at its closure
  blob) and the 011–019 modules are called and never edited, with blob checks as in 021.
- **Tests:**
  - the new chains equal `hp.exact_chain` and `HeadChainModel.upstream` bit for bit when only `ΔE` is supplied;
  - the decomposition identity on the pinned model;
  - Shapley efficiency and permutation invariance;
  - the floor rule's exact order statistics: `F1`, `F4` from element `[249]` of the ascending array; `F2` from element
    `[9750]` (a test fails if C2 takes `[249]`); C3 from elements `[124]` and `[9875]`;
  - the direction checks and the undefined-value convention;
  - the calibration stop at exactly 250 undefined values (C1, C2, C4) and exactly 125 (C3), with 249 and 124
    proceeding;
  - the four-way condition result (`NOT_INTERPRETABLE`, `GUARD_FAILURE`, `ENVELOPE_ONLY_FAILURE`, `PASS`) at every
    boundary: exactly `0.50`, `0.10`, `0.90` and `0`, and exactly on each envelope bound;
  - that the gap rule removes no unit;
  - a fake-closed world for the runner (021's pattern).
- **Output directory.** `outputs/experiment-022/` is added to `.gitignore`.

## Stopping condition

Stop after this revision. After approval, the sequence is:

1. The plan.
2. Implementation, with tier B tests.
3. An independent review, then tier C on the clean gated commit.
4. `freeze` (tokenizer only), then commit.
5. `calibrate` once, then install and commit the record; **stop for the floor review**.
6. `lock` (no forward pass), then commit; the independent lock review and the sign-off.
7. `confirm` once.
8. `report`, then closure.
9. `replicate-021`, which is exploratory.

## Revision history

- Revision 1 (2026-09-23, `2b20db3`): proposal for review, after the research-design spike and the reviewer's choice
  of Option A.
- Revision 2 (2026-09-23, `0f1009c`): the review's precision changes; nothing else changes.
  - **Tails.** Each claim's envelope is an exact 1-based order statistic of the ascending draws: `v₍₂₅₀₎` for the lower
    bounds C1 and C4; `v₍₉₇₅₁₎`, the upper tail, for C2; `v₍₁₂₅₎` and `v₍₉₈₇₆₎` for C3. Direction checks and tests are
    added, and the generic "250th value" wording is removed.
  - **C4.** Worded as a positive contribution.
  - **Gap rule.** Per condition, on the full aggregate. It gives `NOT_INTERPRETABLE`, replacing `PRECONDITION_FAILED_*`,
    and removes no unit.
  - **Results.** The Y1/Y2 aggregate labels are replaced by eight individual condition results (`PASS`,
    `ENVELOPE_ONLY_FAILURE`, `GUARD_FAILURE`, `NOT_INTERPRETABLE`), with no pooling and no all-pass requirement.
  - **Validity.** Stated to select nothing.
- Revision 3 (2026-09-23): the calibration stop's threshold is corrected from "more than 250" and "more than 125" to
  "250 or more" (C1, C2, C4) and "125 or more" (C3). Exactly those counts already make the order statistic infinite
  under the ±∞ convention. It stays a pre-lock stop for review: not evidence, not an incident, never retried
  automatically. Nothing else changes.

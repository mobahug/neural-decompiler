# Experiment 023: Prospective Block-0 Completion — Does the Weight-Derived Program Reach the Measured-`Δx3` Ceiling?

**Date:** 2026-09-24

**Status:** Revision 2. The reviewer approved revision 1 (`68698d6`) in principle and asked that five details be made
explicit before the implementation plan. Revision 2 does only that:
- **Sufficient statistics.** The exposed-cells artifact stores `n`, `Σy` and `Σy²` per pair. A draw's `SST` is
  `ΣΣy² − (ΣΣy)² / Σn` over the selected pairs, never a sum of pair-local variances. 022's two-pass moments are kept
  as a hard cross-check.
- **Multiplicity.** A repeated pair contributes repeatedly to every sufficient statistic.
- **Calibration semantics.** The element `[249]` bound, the undefined-draw rule and the stop are stated in one place.
- **The ceiling.** `C` is a frozen comparator, not a mathematical upper bound. `g > 1` and `g < 0` are valid and are
  never incidents.
- **Scope.** The scope is stated positively: new determiner-like, quantity and adjective cues, and new frames.

Nothing else changes. Revision 1 followed the exposed-only design spike and the reviewer's first decisions:
- Option A, prospective block-0 completion;
- population option (a), three cue strata × 8;
- the gap-recovered statistic with its interpretability rule and meaning guard, and no absolute-`R²` gate;
- a compact committed calibration artifact;
- no spent-set replication.

No code, no runner, no freeze, no calibration and no fresh prompt exist yet. The spike used only exposed units, the
weights and stored measurements, and ran no prompt.

## Background: what Experiment 022 settled

Experiment 022 (closed at `0818c4c`) localized the remaining error of the committed upstream `Δx3` prediction.

- **The gap.** Its object was the gap between Level 0 — Experiment 017's chain, fed the committed cue input `ΔE` — and
  the ceiling, the frozen downstream readout fed the measured `Δx3`.
- **All eight conditions passed.** On 24 never-executed cues and 18 never-executed frames:
  - block-0 attention at the cue position carried 0.948 (Y1) and 0.938 (Y2) of the cue-final gap;
  - the correctly wired layer-1–2 reductions carried at most 0.007 of either gap;
  - block 0's change at the target position contributed positively in coordinated frames.
- **The attribution.** It came from an exact five-factor Shapley game.
- **The full composition.** Every factor switched on and layers 1–2 exact: an identity endpoint, not a program.
- **Inputs only.** The descriptive rung "inputs only" — every block-0 term with the committed *reduced* layers 1–2 —
  reached `R²` 0.960, 0.965, 0.952 and 0.953. It was reported as a number and never claimed as a program.

Experiment 023 turns that rung into a frozen program. It asks prospectively whether the program reaches the ceiling.

## The completed program (the frozen object)

For a pair (cue `t`, frame `f`, the template's reference cue `r`), with the cue at `p_c` and the target at `p_t` (`p_t
= p_c` in cue-final frames, `p_t = p_c + 1` in coordinated frames):

**Level 0 (`P0`).** Experiment 017's committed chain feeds the downstream readout. Its layer-1/2 reference rows are
built through `p_c`, as in 017–019 and in 022 revision 4. Its input is `ΔE = MLP₀(LN₂(W_E[t])) − MLP₀(LN₂(W_E[r]))`.
This is 022's empty coalition, identical to `rd.predicted_dx3` (identity I5).

**The block-0 rule.** Pythia adds no positional embedding, and rotary position acts inside attention. So block 0's
input at every position is the token embedding. Block 0's change is therefore a function of the cue's embedding and
the frame's token embeddings alone. It needs no captured state and has no fitted parameter.

- **Cue row (query `p_c`).** Per head `h`, the cue's query, key and value come from `LN₀(W_E[t])`. The row is the
  reference row with its logits moved:
  - by `⟨Δq, k_j⟩/√d` for every earlier key `j < p_c`;
  - on the diagonal, replaced by `⟨q', k'⟩/√d`.

  The value at `p_c` is replaced too. This is the project's own token-local row rule, the Level 0 of 015/016, which the
  committed chain uses at layers 1–2. At block 0 it is exact. It splits into:
  - the value term `V = Σ_h a_h (o'_h − o_h)`;
  - the pattern term `P = Σ_h Σ_k (r'_k − a_k) o'_k`.

  Here `a` is the reference self-weight, `o = W_O v`, and `r'` is the new row.
- **Target row (coordinated frames; query `p_t`, only the key and value at `p_c` change).** This is exactly one logit
  update:

  `T = Σ_h [a_h (o'_h − o_h) + (σ(logit a_h + ⟨q_{p_t,h}, Δk_h⟩/√d) − a_h)(o'_h − rest_h)]`

  `a_h` is the reference weight from `p_t` to `p_c`, and `rest_h` is the reference mix of the other keys.

**The completed program (`P1`).** Its inputs are:
- `u(p_c) = ΔE + Δemb + V + P`, with `Δemb = W_E[t] − W_E[r]`;
- `u(p_t) = T`, in coordinated frames only.

These pass through the committed reduced layer-1–2 program (`ul.reduced_chain`: 017's chain at `LEVEL0` with its input
replaced), then the frozen Experiment 020 readout (blocks 3–5, `LN_final`, the 79-noun read). The output is `Δĉ₁`.
`P1` is Experiment 022's canonical "inputs only" coalition (mask 14 in cue-final frames, 30 in coordinated frames),
bit for bit.

**Allowed inputs of `P0` and `P1`:**
- the weights;
- the frame's text tokens;
- the frame's reference state: 020's locked states for exposed frames, the stage-1 S1-REF capture for new frames.

Nothing measured on a cue prompt ever enters `P0` or `P1`.

**The ceiling (`C`).** The frozen readout is fed the *measured* `Δx3` of the cue prompt, at `p_c` and `p_t`, at the
frame's reference state. It is a comparator and normalizer only and is never an input to `P1`.

`C` is the frozen downstream ceiling *comparator*, not a mathematical upper bound on a finite sample's `R²`. The readout
has its own error, so `P1` can be slightly closer to the measured `Δc` than `C` is, through finite-sample error or
cancellation. That gives `g > 1`, which is permitted and is not an incident.

## The exposed spike (exploratory; it motivates the question and scores nothing)

**Setup.**
- **Pairs:** all 18,900 exposed pairs (175 pool cues × 108 exposed frames, every key in 020's ledger).
- **Source:** the measured `Δc`, `P0`, `P1` and `C` of every pair read from Experiment 022's digest-bound calibration
  table.
- **Compositions:** every candidate rule composed weights-only through the reduced layers 1–2 and the frozen readout;
  no prompt ran.
- **Checks:** the recomputed block-0 terms equal 022's factors to ≤ 1e-14. The recomputed `P1` equals 022's stored
  coalition to ≤ 7e-12. The per-pair `SSE₀` and `SSE₁` equal 022's stored cells bit for bit.

"Gap" below is the share of the Level-0 → exact-block-0 gap each rule recovers. Rules marked † were run on every fifth
cue (3,780 pairs); the low-rank code was scored on a held-out quarter.

| rule | cue-final | coordinated |
|---|---|---|
| Level 0 (`R²`) | 0.8117 | 0.5548 |
| embedding change only (`R²`) | 0.8240 | 0.5774 |
| value term only, frozen pattern | 0.602 | 0.921 |
| the same with template-mean self-weights | 0.583 | 0.917 |
| self logit only | 0.292 | 0.868 |
| self logit relative to the mean query shift | 0.807 | 0.967 |
| oracle self-weight, other keys proportional | 0.822 | 0.966 |
| first-order (linear-response) softmax † | 0.826 | 0.972 |
| four heads (5, 0, 7, 6) † | 0.760 | 0.954 |
| six heads (5, 0, 7, 6, 3, 2) † | 0.913 | 0.988 |
| 32-dimensional principal code of the block-0 input † | 0.690 | 0.931 |
| target term, value part only (cue row exact) | — | 0.947 |
| **the block-0 rule (`P1`)** | **1.000** (`R²` 0.9673) | **1.000** (`R²` 0.9675) |
| ceiling `C` (`R²`) | 0.9676 | 0.9679 |

**What the spike shows.**
- No cheaper rule reaches the exact block-0 rule. The pattern term cannot be dropped, and it is not a self-weight
  effect: the query change redistributes attention among the earlier keys.
- The block-0 contribution is spread over at least six heads and is not low-dimensional.
- With the exact rule, `P1` closes 0.998 (cue-final) and 0.999 (coordinated) of the Level-0 → ceiling gap. It lies
  within 0.0004 `R²` of the ceiling.
- **Descriptive only: 022's spent fresh sets.** On 022's spent fresh populations the same statistic is 0.9983 (Y1
  cue-final), 0.9989 (Y1 coordinated), 0.9952 (Y2 cue-final) and 0.9983 (Y2 coordinated).
- **Expected envelopes.** 2,000 unregistered exposed-like draws, of the composition below, put the 2.5 % tails at about
  0.997, 0.999, 0.995 and 0.995. The calibrated envelopes will therefore be tight.

## Question

Does the completed, fully weight-derived program `P1` prospectively recover essentially all of the explainable
Level-0 → measured-`Δx3`-ceiling gap on new cues? It is asked in the exposed frames (Y1) and in new frames (Y2), per
group (cue-final and coordinated).

## The primary statistic (frozen)

For a population and a group, let:
- `y` be the measured `Δc` of every pair over the 79 exposed nouns;
- `SSE_k = Σ_pairs Σ_nouns (y − k)²`, for `k ∈ {P0, P1, C}`.

Each `SSE_k` is summed from per-pair cells, with a repeated pair counted with its multiplicity.

**`g = (SSE₀ − SSE₁) / (SSE₀ − SSE_C)`**

- The SSE form is authoritative. It equals `(R²₁ − R²₀)/(R²_C − R²₀)` with the total variance cancelled.
- `g` is not clipped. `g > 1` (the program closer to `y` than the ceiling) and `g < 0` are valid numeric results, never
  incidents.

**Sufficient statistics and `SST` (authoritative).** Every pair `i` carries six sufficient statistics over its 79
nouns:
- `n_i` (79);
- `S_i = Σ y`;
- `Q_i = Σ y²`;
- `SSE0_i = Σ (y − P0)²`;
- `SSE1_i = Σ (y − P1)²`;
- `SSEC_i = Σ (y − C)²`.

For any selection of pairs — a fresh group, where each pair counts once, or a calibration draw, where a repeated pair
counts once per selection — the pooled quantities are:
- `N = Σ n_i`, `S = Σ S_i`, `Q = Σ Q_i`;
- **`SST = Q − S² / N`**;
- `SSE_k = Σ SSE{k}_i`, for `k ∈ {0, 1, C}`.

`SST` is never a sum of pair-local variances: the pooled mean changes with every selection. As a hard implementation
cross-check, `SST` must also equal the pooled two-pass identity `Σ M2_i + Σ n_i (mean_i − S/N)²` to a relative 1e-10.
Here `mean_i` and `M2_i` are the pair's two-pass moments, the quantities 021 and 022 used. On the whole exposed pool the
two forms agree to 2.5e-16 (mean −3.8, standard deviation 1.5). `R²_k = 1 − SSE_k / SST`.

**Interpretability.** The condition is evaluated on its full group aggregate. `g` is undefined — the result is
`NOT_INTERPRETABLE` — when `SST ≤ 0` or `SSE₀ − SSE_C < 0.02 · SST`. That is 022's gap rule: the ceiling must beat Level
0 by at least 2 % of the variance. No pair, cue, frame or noun is removed or re-weighted.

**Frozen and reported, without outcome force:**
- `R²₀`, `R²₁` and `R²_C` (`= 1 − SSE_k / SST`) for every condition;
- a `ceiling_limited` flag when `R²_C < 0.80`.

No absolute-`R²` gate exists.

## The four conditions (frozen)

| condition | population | group | statistic | envelope `F` | meaning guard | effective requirement |
|---|---|---|---|---|---|---|
| **K1** | Y1 | cue-final | `g` | `F = v₍₂₅₀₎` of 10,000 exposed-like draws (lower 2.5 % tail, element `[249]`) | `g ≥ 0.90` | `g ≥ max(F, 0.90)` |
| **K2** | Y1 | coordinated | `g` | as K1 | `g ≥ 0.90` | as K1 |
| **K3** | Y2 | cue-final | `g` | as K1 | `g ≥ 0.90` | as K1 |
| **K4** | Y2 | coordinated | `g` | as K1 | `g ≥ 0.90` | as K1 |

The meaning guard `g ≥ 0.90` is the preregistered meaning of "recovers essentially all of the explainable upstream
gap". Undefined draw values are placed at −∞, so they always count against the envelope.

**Results.** The precedence is Experiment 022's:
1. `NOT_INTERPRETABLE`: the gap rule fails, or `SST = 0`;
2. `GUARD_FAILURE`: `g < 0.90`;
3. `ENVELOPE_ONLY_FAILURE`: `0.90 ≤ g < F`;
4. `PASS`: `g ≥ max(F, 0.90)`.

There is no aggregate label. Each condition is read on its own, and Y1 and Y2 are never pooled.

## Populations and fresh units

| population | definition | pairs (cue-final / coordinated) |
|---|---|---|
| **Y1** | 24 new cues × the 108 exposed frames × 79 exposed nouns | 1,728 / 864 |
| **Y2** (frame-conditional) | 24 new cues × 18 new frames × 79 exposed nouns, each frame at its own stage-1 reference state | 288 / 144 |

- **Three cue strata × 8:** determiner-like, quantity and adjective.
- **Scope:** Experiment 023 prospectively tests completion across new determiner-like, quantity and adjective cues and
  new sentence frames only. It makes no claim about "all grammatical-number cue classes" and no new prospective claim for
  possessive or pronoun cues. That stratum is exhausted. A tokenizer-only sweep of every standard personal, possessive, reflexive, demonstrative, interrogative and
  indefinite form found them all used by earlier experiments. The archaic forms are used or split into several tokens.
  Only `I` and `whatsoever` remain as clean candidates, and the reviewer declined a two-word stratum and informal forms.
- No new nouns.
- The validity rule selects nothing; the S1-VALIDITY verdicts are recorded descriptively.

### Fresh cues

`freeze` takes, per stratum, the first 8 entries that are a single token with a leading space and whose id lies
outside the exclusion set. The exclusion set is Experiment 022's freeze exclusion plus 022's own 24 frozen cues:
- every cue token of Experiment 020's exposed pool;
- every token of every committed confirmation file from 006 to 022;
- every template reference cue and plural cue;
- every frame's cue ids and the extension's cue words.

The design-time pre-check counted 327 token ids. No model output is involved. The lists contain the entries of 022's
lists that were never frozen or executed, followed by new words.

| stratum | ordered candidates (the first 8 eligible are taken) |
|---|---|
| `determiner-like` | general, subsequent, given, chosen, selected, present, ultimate, preceding, following, latest, earliest |
| `quantity` | tons, piles, masses, stacks, gross, net, scores, batches, bundles, crowds, herds, multitude, cumulative, aggregate |
| `adjective` | humble, proud, shy, lazy, busy, sturdy, fragile, shiny, dusty, hollow, noble, clever, fuzzy, crisp, stiff, tender, polished |

A tokenizer-only pre-check found every entry eligible (11 / 14 / 17). `polished` is placed last so that it does not
repeat the verb of a picked coordinated frame. The expected picks are:
- `determiner-like`: general, subsequent, given, chosen, selected, present, ultimate, preceding;
- `quantity`: tons, piles, masses, stacks, gross, net, scores, batches;
- `adjective`: humble, proud, shy, lazy, busy, sturdy, fragile, shiny.

If a stratum yields fewer than 8 at `freeze`, the freeze writes nothing and stops for review. No stratum is refilled
from another, and there is no runtime fallback.

### Fresh frames

`freeze` takes, per template, the first 6 texts that pass 022's structural checks:
- the text appears in no committed frame set; the design-time pre-check counted 144 texts, 022's included;
- the cue slot is a single token;
- `p_c` lies in the template's exposed range (cue-final 3–5, coordinated 4–7);
- in coordinated frames, `p_t = p_c + 1`;
- the frame is built by `pm._build_new_frame`.

Frame ids are `<template>-023-<k>`.

| template | ordered candidates (the first 6 structurally eligible are taken) |
|---|---|
| cardinal | The dairy produces {cue} · The shelter feeds {cue} · The boutique sells {cue} · The hangar shelters {cue} · The garage repairs {cue} · The nursery grows {cue} · The warehouse stores {cue} · The workshop carves {cue} · The harbor ships {cue} · The pharmacy stocks {cue} · The florist wraps {cue} · The butcher cuts {cue} · The tailor sews {cue} |
| quantifier | The timetable shows {cue} · The glossary defines {cue} · The inventory lists {cue} · The chart shows {cue} · The memo mentions {cue} · The journal notes {cue} · The register records {cue} · The index cites {cue} · The brochure lists {cue} · The logbook notes {cue} · The digest mentions {cue} · The handbook names {cue} · The spreadsheet counts {cue} · The schedule lists {cue} |
| coordinated-adjective | Ella and Tomas stacked {cue} low · Anna and Erik glazed {cue} smooth · Leo and Maja folded {cue} flat · Lena and Jonas scrubbed {cue} clean · Rosa and Felix polished {cue} bright · Clara and Hans boiled {cue} soft · Mira and Sven chopped {cue} fine · Ida and Lars packed {cue} tight · Eva and Nils painted {cue} blue · Tina and Rolf baked {cue} brown · Sofia and Emil pressed {cue} flat · Maria and Jens brushed {cue} smooth |

The pre-check found every text eligible (13 / 14 / 12; `p_c` 3–4 in the cue-final templates, 5–7 in the coordinated
one). The first six of each list are expected.

**Manifest:**
- S1-REF: 18, one reference prompt per new frame;
- S1-VALIDITY: 18, descriptive only;
- S2-TARGET: 2,592 (Y1) + 432 (Y2) = 3,024.

That makes 3,060 keys. `freeze` writes `experiments/023-block0-completion/confirmation-v1.json` with its digest. It is
committed before `calibrate`, which reads only its stratum and template counts.

## The compact calibration artifact (reviewer requirement)

**The dependency it removes.** Experiment 022's `outputs/experiment-022/calibration-table.pt` is the only source of the
exposed `Δc`, `P0`, `P1` and `C`:
- 424,131,002 bytes, file sha256 `04659d5e…`;
- local and gitignored;
- each tensor digest-bound in 022's committed `calibration-v1.json`, whose record content is `46985fd5…`.

The ceiling cannot be rebuilt without 18,900 exposed forward passes. Experiment 023 must not depend operationally on
that file.

**`extract` (once; read-only on the table; weights only; no prompt).** It writes the artifact:
- `experiments/023-block0-completion/exposed-cells.f64`: raw little-endian float64, C order;
- a canonical-JSON index, `exposed-cells.json`.

**Contents.** For every exposed pair — all 175 cues × 108 frames = 18,900, in 022's canonical order (the strata in
022's order, then token id; frames by `frame_id`) — eight columns, in this order:
- the six sufficient statistics `n`, `S = Σy`, `Q = Σy²`, `SSE₀`, `SSE₁`, `SSE_C`;
- `mean` and `M2`, 022's stored two-pass moments, for the `SST` cross-check only.

That is 1,209,600 bytes. The index binds:
- the identifiers (cue, token id, stratum; frame id, template, group);
- the column formulas;
- 022's table file sha256 and its tensor digests;
- 022's calibration record, file and content digests;
- the extraction module's path and git blob;
- an extraction version string;
- the artifact's own file digest.

The pronoun rows are kept for completeness; 023's draws do not use them.

**Identities at extraction.** Each is hard: a failure writes nothing and stops for review.

| gate | what must hold | tolerance |
|---|---|---|
| E1 | the table's tensors match the digests in 022's committed calibration record | exact |
| E2 | `SSE₀` and `SSE₁` equal 022's stored per-pair `sse` at mask 0 and mask 14 / 30; `n`, `mean` and `M2` equal 022's stored `count`, `mean` and `m2` | bit for bit |
| E3 | `S`, `Q` and `SSE_C`, computed twice from the stored `Δc` and ceiling | bit for bit |
| E4 | `P1` recomputed from the weights and 020's locked states equals the table's `P1` on all 18,900 pairs | max abs 1e-9 |
| E5 | for the first 16 calibration draws, `g`, the gap and `SST` from the cells equal a direct recomputation from the per-noun table | `|k − d| / max(1, |d|) ≤ 1e-10` |
| E6 | for every pair and every draw of E5, `SST = Q − S²/N` agrees with the pooled two-pass identity | relative 1e-10 |

The spike already found E2 exact and E4 at 7e-12.

**The extraction boundary.** The path is one-way:

`022 calibration table → exact verified extraction → 023 exposed-cells.f64/json (committed, digest-bound) → 023 calibration`

After the artifact is extracted, verified by E1–E6, installed byte-identically, independently checked and committed,
nothing in Experiment 023 reads the 0.42-GB table again:
- `calibrate` reads only the committed artifact, verified against the digests the committed index carries;
- the runner opens the 022 table in `extract` only;
- a second `extract` is refused once the artifact is committed.

## Calibration (exposed only, once; no model)

**Pools.** These are Experiment 022's pools without the pronoun stratum:
- cues: the exposed pool's determiner-like (45), quantity (45) and adjective (49) cues;
- Y1-like frames: the 108 exposed frames;
- Y2-like frames: the 42 exposed frames first confirmed in 017–019, 14 per template, unscreened.

**Draws.**
- `B = 10,000` SHA-indexed draws:
  `int.from_bytes(sha256(f"023|primary|{b}|{s}|{i}")[:8], "big") % n_s`.
- 24 cues per draw, with replacement, at 8 per stratum.
- Y1-like: the drawn cues × the 108 frames.
- Y2-like: the drawn cues × 18 frames drawn with replacement at 6 per template from the 42.
- Per draw and group, with every selected pair counted once per selection:
  - `N`, `S`, `Q`, `SSE₀`, `SSE₁` and `SSE_C` are summed from the sufficient statistics;
  - `SST = Q − S²/N`;
  - then `g` and the gap rule.

**Kernel check.** On the first 16 draws, the cell kernel is checked against a direct loop over the drawn pairs'
cells, at 1e-10 (implementation only). The per-noun check is E5, and E6 applies to every draw.

**Exact calibration semantics (frozen):**
- 10,000 deterministic SHA-indexed draws per population.
- A draw's value for a condition is undefined when `SST ≤ 0` or `SSE₀ − SSE_C < 0.02 · SST`. Otherwise it is `g`,
  unclipped: `g > 1` and `g < 0` are valid values.
- Undefined values are placed at −∞ before sorting, so they always count against the lower tail.
- `F = v₍₂₅₀₎`: the 250th value of the ascending 10,000 (1-based), zero-based element `[249]`, as in 022.
- **Direction check,** before anything is written: `F ≤` the median of the defined draws.
- **Calibration stop:** 250 or more undefined values in any condition make `F = −∞`. `calibrate` then writes no floor
  table and stops for review before `lock`. The result is never a usable envelope; this is not an incident and is never
  retried automatically.

**Recorded, never corrected:**
- the share of draws in each of the four results per condition;
- descriptively, the share in which all four conditions pass;
- medians and tails;
- whether the guard binds, i.e. whether `F < 0.90`.

The candidate record is installed byte-identically, committed and reviewed before `lock`.

## Lock (no forward pass)

- **The Y1 prediction table.** `P0` and `P1` for the 2,592 Y1 pairs over the 79 nouns: cue-final `[1728, 2, 79]` then
  coordinated `[864, 2, 79]`, raw little-endian float64 (3.3 MB). It is computed twice from the weights and 020's locked
  states and required bit-identical.
- **Block-0 algebra checks,** at 1e-12: `V + P = ΔA0(p_c)`, and the closed-form `T` equals the exact attention output at
  `p_t`.
- **The lock** binds:
  - the four conditions: statistic, envelope, guard, precedence and readings;
  - the program's definition, with its module blobs (022's `upstream_localization.py`, git blob `46585696…`, used
    unchanged);
  - the artifact and calibration digests;
  - the Y1 table's digests;
  - the Y2 table's construction, byte format, orders and dimensions;
  - the tolerances;
  - the scope statement.
- **The preregistration** is rendered from the lock. All files are installed byte-identically, committed and
  independently reviewed.

## Confirm (once, never resumed)

- **Before any fresh prompt.** `validate_lock` runs with tracked-file enforcement. The runtime is checked against 020's
  explore runtime. I7 rebuilds the Y1 table bit for bit.
- **Stage 1.** The 18 S1-REF and 18 S1-VALIDITY prompts run. The Y2 prediction table (`P0` and `P1` for the 432 pairs)
  is computed from the stage-1 reference states and written once, with its digests in the stage-1 record.
- **Barrier.** The state and the Y2 table are re-read from disk and verified against the digests held in memory. No
  S2-TARGET key may be in the ledger.
- **Stage 2.** Each of the 3,024 target prompts runs once. It measures `Δc` over the 79 nouns and `Δx1` and `Δx3` at
  `p_c` and `p_t`. Every measurement is saved before any gate. The Y2 table is re-verified afterwards.
- **The ceiling.** `C` is computed from the measured `Δx3`. It enters only `SSE_C` and the identities.
- **Identities.** Hard: an incident, with no result.
  - I1: `Δemb + ΔE + ΔA0(p_c)` equals the measured `Δx1(p_c)`, and `ΔA0(p_t)` equals `Δx1(p_t)`.
  - I3: the full composition (layers 1–2 exact) reproduces the measured `Δx3`.
  - I4: its `Δĉ` equals `C`.
  - I5: `P0` equals the committed Level 0 exactly.
- **Scoring.**
  - Per condition: the six sufficient statistics of every fresh pair, each pair once; then `SST = Q − S²/N` with its E6
    cross-check, `g`, the gap, the three `R²`, the flag, and the result;
  - the cell kernel against a direct recomputation from the per-noun measurements, at 1e-10;
  - a `g > 1` or `g < 0` is reported as it is; it is never an incident.
- **Descriptive, with no outcome force:**
  - per-template and per-stratum `g`;
  - the spike's cheaper rules — value term only, the relative self logit, the oracle self-weight, the first-order
    softmax, four and six heads — computed weights-only on the fresh pairs;
  - `P1`'s `Δx3` relative error at `p_c` and `p_t`;
  - block 0's head profile;
  - the validity verdicts.

## Gates and tolerances (frozen; 022's values where they exist)

| gate | tolerance |
|---|---|
| E1, E2, E3 | exact / bit for bit |
| E4 | max abs 1e-9 |
| E5, kernel checks | `|k − d| / max(1, |d|) ≤ 1e-10` |
| E6 (`SST` one-pass against pooled two-pass), at extraction, calibration and confirm | relative 1e-10 |
| lock tables twice; I7 at confirm | bit for bit |
| block-0 algebra (`V + P`, closed-form `T`) | 1e-12 |
| I1 | max abs 1e-4 |
| I3 | relative 1e-4 |
| I4 | max abs 1e-3 nats |
| I5 | exactly 0.0 |

**Known risk.** I3 keeps 022's tolerance. Its margin over the exposed maximum is narrow: 5.6e-5 against 1e-4, about 1.8×.
At 022's confirm it was 2.4×. An I3 exceedance at confirm would be an incident and would carry no result.

## Phases and leakage boundaries

| phase | may read | may execute | writes |
|---|---|---|---|
| `validate` | frozen inputs, locks, 020/021/022 closure records and digests | nothing | nothing |
| `extract` (once) | 022's local table (digest-verified), 022's record, 020's locked states, the weights | nothing | the exposed-cells artifact (committed) |
| `freeze` | the tokenizer, the committed frame and cue sets | nothing | `confirmation-v1.json` (committed) |
| `calibrate` (once) | the committed artifact and the confirmation's counts | nothing (no model) | the candidate calibration record |
| `lock` | the committed record, 020's locked states, the weights | nothing (no forward pass) | the Y1 table, the lock, the preregistration |
| `confirm` stage 1 | the lock, the confirmation file, the weights | S1-REF and S1-VALIDITY only | the reference states, the Y2 table and its digests |
| barrier | the state and the Y2 table on disk | nothing | — |
| `confirm` stage 2 | the lock, the digested stage-1 record, the weights | every S2-TARGET key, once | measurements (before any gate), gates, scores |
| `report` | the results state and the calibration draws | nothing | the report |

No fitting or tuning happens anywhere, and nothing measured on a fresh cue prompt enters `P0` or `P1`.

## Incidents and interpretability

These follow Experiments 020–022. An implementation or protocol failure stops the phase, is recorded with its commit,
and permits no rerun at that commit. An interruption of `confirm` spends the protocol version.

- **Incidents:**
  - any E, I or barrier violation;
  - a missing or non-finite measurement;
  - a structural failure of a new frame's reference capture.
- **Interpretability:** the gap rule, per condition, on its full aggregate. A `NOT_INTERPRETABLE` result is neither an
  incident nor an exclusion.

## Interpretation (frozen)

Each condition is read on its own.

- **`PASS`.** On that population and group, the completed program recovers essentially all of the explainable upstream
  gap, within the exposed-like envelope. The weight-derived upstream program reaches the ceiling that the decoded
  downstream readout allows. A `g` above 1 means the program came slightly closer to the measured `Δc` than the
  ceiling comparator did. That is allowed and says nothing beyond "essentially all".
- **`ENVELOPE_ONLY_FAILURE`.** `g ≥ 0.90`, so "essentially all" still holds, but `g` lies below the calibrated
  envelope. This is a quantitative shift relative to exposed-like sets, never a refutation. The spike expects envelopes
  near 0.995–0.999, so this result can occur with `g ≈ 0.99`.
- **`GUARD_FAILURE`.** `g < 0.90`. The completed program does not recover essentially all of the explainable gap on that
  population and group.
- **`NOT_INTERPRETABLE`.** Too little Level-0 → ceiling gap remains. This is neither a pass nor a failure.
- **Incidents** carry no result.

**What a pass does not show:**
- that the downstream readout is complete — `R²_C` was 0.95–0.97 in 022, and the remainder is downstream;
- anything about possessive or pronoun cues, new nouns, or behavior beyond `Δc` at `p_t`;
- that a cheaper block-0 rule would suffice (the comparators are descriptive).

## Interpretation limits

- The value function is flattened `Δc` SSE/`R²`. The ceiling depends on the frozen 020 readout.
- The scope: one checkpoint, three templates, three cue strata, 24 new cues, 18 new frames and the 79 exposed nouns.

## Review record

**Spike and decisions (2026-09-24).** The reviewer chose Option A, prospective block-0 completion, and decided:
1. **Metric.** The primary statistic is `g` in its SSE form, unclipped, for four conditions (Y1/Y2 × cue-final/
   coordinated).
2. **Interpretability.** The rule is 022's aggregate gap requirement: `(SSE₀ − SSE_C)/SST ≥ 0.02` with `SST > 0`.
3. **Guard.** The meaning guard is `g ≥ 0.90`. The effective requirement is `g ≥ max(F, 0.90)` with 022's four-way
   reading. There is no absolute-`R²` gate; `R²₀`, `R²₁` and `R²_C` and a descriptive `ceiling_limited` flag are reported.
4. **Population.** Option (a): 3 strata × 8 = 24 new cues. The possessive-or-pronoun stratum is dropped. It is not
   replaced by two words or by informal forms, and the scope limitation is stated. There are 18 new frames, 6 per
   template.
5. **Calibration.** It reuses 022's verified exposed table through a compact committed sufficient-statistics artifact.
   It uses 10,000 SHA-indexed draws at 8/8/8 and 6/6/6, and no new exposed prompt.
6. **Confirmation.** Fresh `Δx3` is measured only for the ceiling and the identities; `P1` stays weight-derived and
   parameter-free.
7. **Comparators.** The cheaper rules are descriptive only; there is no necessity claim.
8. **Replication.** There is no automatic spent-set replication.

**Design approval in principle (2026-09-24, revision 1).** The reviewer approved revision 1 and required five details
to be explicit before the implementation plan; revision 2 makes them so.
1. **Sufficient statistics.** `n`, `Σy`, `Σy²` and the three SSE per pair. A draw's `SST = Q − S²/N` from the pooled
   sums, never a sum of pair-local variances. A repeated pair contributes repeatedly to every statistic.
2. **Calibration semantics.** 10,000 draws; the element `[249]` bound; `g` unclipped, with `g > 1` and `g < 0` valid;
   undefined draws when the `SST` or gap rule fails, placed against the lower tail; 250 or more undefined draws stop
   calibration for review before `lock`.
3. **The ceiling.** `C` is a frozen comparator, not a mathematical upper bound; `P1` beating it is not an incident.
4. **Scope.** New determiner-like, quantity and adjective cues only; no claim about possessive or pronoun cues or "all
   grammatical-number cue classes".
5. **The extraction boundary.** Once the artifact is verified and committed, calibration never reads 022's table
   again.

Revision 2 keeps 022's two-pass moments beside the reviewer's `SST` formula as a cross-check (E6), not as an
alternative.

## Implementation boundary (for the plan, after approval)

- A new module (planned as `src/neural_decompiler/block0_completion.py`) imports 022's module and the 017–021 programs
  as frozen dependencies, checked by git blob. Nothing in 023 edits them. `P0`, `P1`, `C` and the block-0 terms are
  022's functions.
- The runner has the phases `validate`, `extract`, `freeze`, `calibrate`, `lock`, `confirm` and `report`. There is no
  replication phase.
- The artifact, the confirmation file, the calibration record, the lock, the preregistration and the Y1 table are
  committed. `outputs/experiment-023/` holds the results state, the draws, the Y2 table (committed as closure evidence
  after a successful confirmation) and the stage-2 measurements.

## Stopping condition

The reviewer approved the design in principle and authorized the implementation plan document after this revision.
Stop after the plan, for its review. Then, each step only when authorized:
1. the implementation plan's review;
2. the implementation and its independent review;
3. tier C on the clean gated commit;
4. `extract`, then install and commit;
5. `freeze`, then commit;
6. `calibrate` once, then install, commit and the floor review;
7. `lock`, then install, commit and the independent lock review;
8. `confirm` once;
9. `report`;
10. closure.

## Revision history

- Revision 1 (2026-09-24, `68698d6`): proposal for review, after the exposed-only design spike and the reviewer's
  decisions.
- Revision 2 (2026-09-24): the five explicit details required at the approval in principle. These are the sufficient
  statistics and the pooled `SST` formula with its two-pass cross-check (E6); the exact calibration semantics; the
  ceiling as a comparator, not an upper bound; the positive scope statement; and the one-way extraction boundary. The
  artifact grows from six to eight columns (1,209,600 bytes). Nothing else changes.

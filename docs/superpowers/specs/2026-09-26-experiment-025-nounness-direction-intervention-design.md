# Experiment 025: Does Moving the Cue Embedding Along the Frozen Nounness Direction Causally Change the Frozen Readout Error and the Attention Routing?

**Date:** 2026-09-26

**Status:** Final design, revision 1. The reviewer approved it, subject to three clarifications that are incorporated
here:
- the Level-1 validity domain covers all 16 outcome-bearing primary-dose conditions;
- `A` means +nounness against −nounness, not against the baseline;
- the routing distance is written `D_attn`, not `ρ`.

It follows Experiment 024's closure (`eb5f6e8`) and three design-spike rounds (see the review record).

No code, runner, freeze, lock or prompt exists for 025. The spikes executed no prompt, fresh or spent, made no rotated
measurement and ran no pilot. They used only:
- the pinned checkpoint's embedding table, read directly;
- the tokenizer;
- committed records and local ledgers.

No nounness score was computed for any word on the new noun list. The 28 reserve picks (20 adjectives and 8
ordinary nouns) were scored at design time, descriptively, with 024's other cues. That could not affect the
selection, which follows list order mechanically.

**Intent.** This is the final experiment of the sequence. After it, on any clean result, the project stops and is
written up.

## Background: what Experiment 024 established (closed, `eb5f6e8`)

**What was shown.** On 40 fresh cues, the frozen operational weight-derived nounness score
`nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue)` prospectively predicted larger error in the frozen block-4/5
attention readout approximation.
- The primary result was ρ = `0.6108818011257036` against the locked `0.3136960600375234`.
- The association passed the E-versus-N disambiguation guard: `D_EN = 0.5680373703974243` and K = 1 of 12,870.
- The outcome was `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.

**What was not shown.** The result is predictive, not causal. The "readout error" is the error of a frozen
approximation, not rerouting observed directly. E and N differ in lexical and semantic properties besides nounness.

**What 025 adds.** 025 asks whether deliberately moving the representation along the predictor's own direction moves
two things:
- the frozen readout error;
- the directly captured attention routing.

## Question (primary)

Does a norm-preserving rotation of a fresh cue's input embedding, toward or away from the frozen 024 nounness
direction, causally change the following in the predicted direction?
- the frozen block-4/5 readout approximation error, more than equal-angle rotations that are neutral to nounness;
- the directly captured L4/L5 attention-routing summary.

## The intervention (frozen)

### Location
- **The site:** the token embedding at the cue position, `("EMBED", p_c)`, in the cue prompt itself.
- **The operation:** a REPLACE through `pm.run_patched`, whose execution check verifies:
  - the replacement landed bit for bit;
  - no other position changed.
- **Why here:**
  - it is exactly the vector the 024 score is computed from;
  - it is the earliest point of the decoded chain;
  - every downstream computation (blocks 0–3, then Δx3, then blocks 4/5) is left intact.
- **Rejected alternatives:**
  - the block-0 output, because block-0 attention mixes context into it and the direction is undefined there;
  - the block-3 residual, because it bypasses the upstream chain and would need a fitted mapping, which would be
    circular.

### The direction and the per-cue geometry
- `d = μ̂_noun − μ̂_cue`, frozen from 024's committed calibration record:
  - μ_noun is the mean of 158 noun rows;
  - μ_cue is the mean of the 139 exposed cue rows;
  - both are unit-normalized.

  Measured: cos(μ̂_noun, μ̂_cue) = 0.0784 and |d| = 1.3577.
- For cue `i`, with `E_i` the float64 embedding row and `Ê_i = E_i/|E_i|`:
  - `s₀,i = d·Ê_i` is 024's score with the full centroids;
  - `t_i = d − s₀,i·Ê_i` is the exact gradient direction of the score on the sphere;
  - `τ_i = |t_i|` and `t̂_i = t_i/τ_i`.
- Hence `d = s₀,i·Ê_i + τ_i·t̂_i`.
- Measured over every cue scored so far (226 cues): cue embedding lengths have a median of 0.714 (0.603–0.738; the
  vocabulary median is 0.709), and τ lies in 1.256–1.358.

### The nounness rotation
`R_i(±θ) = |E_i|·(cos θ·Ê_i ± sin θ·t̂_i)`: a geodesic rotation that preserves the length exactly. Its score is
`s(±θ) = s₀,i·cos θ ± τ_i·sin θ`.

- **Primary dose.** The directional (odd) score component is `τ_i·sin θ_i = ±0.32` exactly, with
  `θ_i = asin(0.32/τ_i)`: about 13.8° (13.6°–14.8° over every cue scored so far), a chord of about 24% of `|E_i|`.
  - 0.32 is the rounded difference (0.3201) between 024's E-class and N-class mean *scores*, locked before any outcome. No 024 outcome
    chose it.
  - Over every cue scored so far, the nearest other token lies at least 34.1° away (median 50°–69°). So a 13.8°
    rotation stays nearer the cue's own token than any other. The 12 new nouns are first checked at the lock, and the
    realized values are recorded there, descriptively.
- **The complete score change** is `Δs(±) = s₀,i·(cos θ_i − 1) ± 0.32`.
  - The even term `s₀,i·(cos θ_i − 1)` is the same for + and −, and the same for every equal-angle control.
  - Over every cue scored so far (|s₀| ≤ 0.515, θ up to 14.76°), it is at most about 0.017.
  - The realized values are bound at the lock.
  - **The total score change is not exactly ±0.32; only its odd component is.**
- **Half dose (secondary).** The odd component is ±0.16, about 6.9° (6.8°–7.3°).
- **Symmetric ± angles are kept.** They give the clean odd/even decomposition.

### The random controls (nounness-neutral; 7 per cue)
- **The directions `u_ij`** for j = 1..7:
  - drawn deterministically from a SHA-256 counter-mode stream (tag `025|control|{token_id}|{j}`) turned into Gaussians
    in float64 by Box–Muller, independent of any library's random-number generator;
  - then projected off `Ê_i` and `t̂_i` twice and unit-normalized.
- **The rotation:** `R_ij(±θ_i) = |E_i|·(cos θ_i·Ê_i ± sin θ_i·u_ij)`, at the **same angle `θ_i`** as the nounness
  rotation.
- **Why they are neutral to nounness.** `u_ij ⟂ Ê_i` and `u_ij ⟂ t̂_i`, and `d = s₀,i·Ê_i + τ_i·t̂_i`, so
  `d·u_ij = 0`. Hence `s_ij(+θ_i) = s_ij(−θ_i) = s₀,i·cos θ_i`.
  - **The controls move the embedding by the same angle but do not move the frozen nounness score directionally.**
  - Their only score change is the same even term.
  - This is verified for every cue and control before any prompt (I7′).

### The plurality control (secondary)
- The plurality direction `p̂` is the unit mean difference of the 79 scorable nouns' plural and singular rows.
- Its tangent component at `Ê_i` is orthogonalized to `t̂_i`, normalized to `p̂′_i`, and rotated by the same `θ_i`. It is
  therefore also nounness-neutral. I7′ checks its orthogonality to `Ê_i` and `t̂_i`, and its ±θ score equality.
- Weights-only: cos(d̂, p̂) = −0.066, and at 13.8° the nounness rotation moves `Ê·p̂` by 0.015, against 0.20 for a real
  singular-to-plural change.

## Measurements (per cue × frame × condition)

Each run is one forward pass of the cue prompt with `EMBED@p_c` replaced by the condition's float32 vector. It
captures:
- 024's measurement sites, from which Δx1, Δx3 and Δc are taken against the locked 020 reference state
  (`ul.measure_prompt`'s definitions);
- `ATTN_PATTERN.L4@p_t` and `ATTN_PATTERN.L5@p_t`.

`C = ul.contrast_of(progs, state, Δx3)` is 020's frozen Level-0 readout. It freezes:
- the whole block-4 attention output (pattern and values);
- the block-5 attention pattern at `p_t` (the reference rows times the value changes).

**`ℓ` (the frozen-readout discrepancy).** For a cue and condition, `ℓ = log(Σ SSE_C / Σ n)`, pooled over the 108
exposed frames × 79 scorable nouns, where `SSE_C = Σ_nouns (Δc − C)²`. This is 024's primary response, measured under
each condition.
- **What it means rests on the Level-1 identity** (gated below). By that identity, `Δc − C` is the error of the frozen
  block-4/5 attention: the change in block 4's attention output, plus the change in block 5's attention pattern.

**`D_attn` (the routing summary).** For each run and head, take the total-variation distance
`½·Σ_keys |a − a_ref|` between the captured row at query `p_t` and the locked 020 reference row (`rows4` for block 4,
`rows5` for block 5).
- `D_attn` is the equal mean over the frozen 16 heads (blocks 4 and 5), then the mean over the 108 frames.
- **This is a routing summary fixed before any outcome.**
  - Equal weighting avoids choosing heads after seeing effects.
  - Irrelevant heads, or heads that respond in the opposite direction, can dilute or oppose the aggregate signal, so a
    failure of `G` is conservative with respect to this particular routing summary.
  - No head is selected using 024 or 025 outcomes.

## The outcome-bearing statistics (frozen)

Per cue `i` (40 cues), at the primary dose:

| | per-cue value | PASS |
|---|---|---|
| **A** (causal direction) | `A_i = ½·[ℓ_i(+θ_i) − ℓ_i(−θ_i)]` | at least 27 of 40 strictly positive |
| **B** (specificity) | `B_i = A_i − (1/7)·Σ_j \|A_ij\|`, with `A_ij = ½·[ℓ_ij(+θ_i) − ℓ_ij(−θ_i)]` for the 7 random controls | at least 27 of 40 strictly positive |
| **G** (routing) | `G_i = ½·[D_attn,i(+θ_i) − D_attn,i(−θ_i)]` | at least 27 of 40 strictly positive |

**Ties and zeros count against PASS**, in all three.

**The criterion.** These are **pre-registered sign-count criteria with an exact Binomial(40, 0.5) reference tail
under H0: P(positive) ≤ 0.5 for independent cue units.**
- `P(X ≥ 27) = 0.01923865414210013`; `P(X ≥ 26) = 0.040345…` would exceed 0.025.
- They are **not** unconditional, design-based randomization tests: the 40 lexical items are selected mechanically, not
  sampled at random from a defined population.
- **The causal reading comes from the within-cue intervention**, since both `+θ` and `−θ` are actually run for each
  cue. The sign count measures consistency across the frozen 40-cue test population.

**What `A` means.** `A_i > 0` means the +nounness intervention produced greater frozen-readout approximation error than
the matched −nounness intervention.
- `A` alone does **not** establish that `+θ` rises above the unperturbed baseline, nor that `−θ` falls below it.
- The baseline and the even component `½·[ℓ(+) + ℓ(−)] − ℓ(0)` are descriptive only. They cannot strengthen or alter
  the outcome.
- The preregistration and the report must keep this distinction.

**Why `B` uses absolute values.**
- The random controls' distribution is symmetric under `u → −u`, and `A(−u) = −A(u)`. So the *signed* mean of `A_ij`
  has expectation exactly 0, and subtracting it would retest `A`.
- `B` instead asks whether the nounness direction's effect in the predicted sign exceeds the typical size of an
  equal-angle, nounness-neutral directional effect.
- It does not assume that `t̂_i` is exchangeable with the controls: `t̂_i` stays a fixed, special direction, and the
  controls only estimate each cue's typical sensitivity.

**Why 7 controls.** They are the only estimate of the comparator. With fewer, the comparator is noisier and `B` loses
power in both directions; they cost 14 runs per pair.

**What `G` requires.** `G` is directional: the routing summary moves in the predicted manner. Specificity is carried by
`B` at the readout level. `G`'s own comparison against the random directions is reported descriptively.

## Outcome (frozen)

| condition | label |
|---|---|
| any failed validity gate or incident (no result is outcome-bearing) | `NOT_INTERPRETABLE` |
| A fails | `CAUSAL_EFFECT_NOT_ESTABLISHED` |
| A passes, B fails | `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC` |
| A and B pass, G fails | `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED` |
| A, B and G pass | `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR` |

- **The strongest claim, and nothing more.** Controlled movement of the cue embedding along the frozen weight-derived
  nounness-related direction causally changes the frozen block-4/5 readout approximation error in the predicted
  direction (+ against −). It does so more strongly than matched rotations that are neutral to nounness, and it
  produces a directional change in the pre-registered L4/L5 attention-routing summary.
- **`READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED`** makes only the readout-error claim. It claims no direct rerouting.
- **Not established by any label:**
  - semantic nounhood as the model's variable;
  - a general routing controller;
  - algorithm selection in general;
  - generalization beyond this checkpoint and this mechanism.
- **Stratum counts (descriptive).** The positive counts of `A`, `B` and `G` are reported separately for the 20
  adjectives and the 20 ordinary singular nouns. They cannot change the label. If a global PASS is strongly
  concentrated in one stratum, the final claim must say so and must not imply the effect holds across both cue types.
  As a reporting trigger only, the prose names any stratum whose positive count is below 14 of 20, the per-stratum
  analogue of 27 of 40.

## Validity gates (frozen)

Any failure is an incident: no statistic is outcome-bearing, and the outcome is `NOT_INTERPRETABLE`.

### I7′: geometry, before any prompt
The following are recomputed from the weights and compared bit for bit with the locked values and digests:
- d, and for every cue `s₀,i`, `t̂_i`, `τ_i`, and `θ_i` for both doses;
- the 280 control vectors;
- the plurality vectors;
- every patched float32 vector.

Then, for every cue and every control:

| check | float64 | patched float32 vector |
|---|---|---|
| `\|t̂_i\| = 1`, `\|u_ij\| = 1` | ≤ 1e-12 | — |
| `Ê_i·u_ij`, `t̂_i·u_ij`, `Ê_i·t̂_i` | ≤ 1e-12 | — |
| `d·u_ij` | ≤ 1e-12 | — |
| nounness neutrality: `\|s_ij(+θ_i) − s_ij(−θ_i)\|` | ≤ 1e-12 | ≤ 1e-6 |
| the nounness rotation's odd component: `\|½·[s(+θ_i) − s(−θ_i)] − 0.32\|` (and 0.16 for the half dose) | ≤ 1e-12 | ≤ 1e-6 |
| length preserved, relative | ≤ 1e-12 | ≤ 1e-6 |
| angle to the original `= θ_i`, for the nounness rotation and every control | ≤ 1e-12 rad | ≤ 1e-6 rad |
| the plurality control `p̂′_i`: unit norm, `⟂ Ê_i`, `⟂ t̂_i`, ±θ score equality | ≤ 1e-12 | ≤ 1e-6 |
| `R_i(0)` cast to float32 equals the model's embedding row | — | bit for bit |

**Where the 1e-6 comes from.**
- A float32 rounding error of at most 2⁻²⁴ per component moves s by at most `τ·2⁻²⁴`, about 8e-8, per vector, so the ±
  difference is at most about 1.6e-7.
- The angle error is at most about `2⁻²⁴/sin θ`: 2.5e-7 at 13.8° and 5.0e-7 at 6.8°.

The bounds come from implementation precision, not from data.

### Runtime gates (confirm)

| gate | tolerance |
|---|---|
| **patch fidelity**, every run: `run_patched`'s check (the replacement landed bit for bit; no other change; the declared source) | exact |
| **the patch path, before the first fresh prompt, on 4 spent prompts frozen by key at the lock:** (a) a patched θ = 0 run equals a plain capture bit for bit; (b) the embedding hook equals the block-0 residual input. This is how the approved runtime identity "the θ = 0 run equals a plain capture" is met without adding fresh runs. Only equality results and digests are recorded, in a separate record outside `executed_prompt_keys`, the 90,720-key manifest and the accounting. No scientific quantity is computed. A failure is an incident before any fresh prompt (`NOT_INTERPRETABLE`). **This needs the reviewer's explicit sign-off; see the review record.** | bit for bit |
| **vector I1** (023's block-0 identity with `d_emb = v − W_E[ref]` and `delta_e = MLP0(LN2(v)) − lexicon(ref)` for the patched vector `v`), then I3 and I4, on **every run** (all 21 conditions) | 1e-4, relative 1e-4, 1e-3 |
| `C` recomputed from the saved Δx3 | bit for bit |
| **the Level-1 identity** `max \|L1 − Δc\|`, on **every** cue × frame for **all 16 outcome-bearing primary-dose conditions**: nounness `±θ` and all 7 random controls at `±θ` | 2e-2 nats |
| the accounting: every cue × frame × condition exactly once, under condition-tagged keys | exact |
| every measurement finite; the 020–024 committed inputs re-verified | exact |

**Why the Level-1 domain covers all 16 conditions.** `B` compares nounness-direction readout error with random-direction
readout error. Every quantity that enters `A` or `B` must lie inside the same validated approximation regime. The half
dose and the plurality control are secondary: they carry the Level-1 identity only descriptively.

## Population (frozen): 40 fresh cues × 024's 108 exposed frames

**Eligibility rules.** 024's rules are kept unchanged:
- a single token with a leading space;
- never used as a cue by 005–024;
- not an exposed-frame token;
- for nouns, both singular and plural eligible, a regular plural, no adjective or modal use, and not a container or
  measure.

**One tightening, the prior-noun rule.** Exclude every form of the 80 pool nouns (161 ids), and also of the 24 nouns of
020's committed confirmation list (48 ids), which 021 read as fresh target nouns. The committed 020 file shows that
use; 021's noun ledger does not record it. This excludes `statue` and `barrel`.

| stratum | the rule | the picks |
|---|---|---|
| **20 adjectives** | the first 20 eligible of 024's 23 adjective reserves, in list order | anxious, cheerful, curious, jealous, lonely, nasty, careful, careless, famous, friendly, gorgeous, hungry, weary, wicked, ugly, vivid, vague, rapid, rigid, clever |
| **8 ordinary nouns (singular)** | 024's ordinary reserves, in list order, under the tightened rule (`statue` excluded) | soldier, sailor, priest, knight, onion, carrot, pirate, tourist |
| **12 new ordinary nouns (singular)** | the first 12 eligible entries of the frozen new list, in its textual order | author, bishop, dancer, duck, goat, guitar, hunter, lawyer, monk, nurse, painter, prince |

**The new candidate list, frozen in alphabetical textual order** before any score of its words was computed:

> author baker bishop camel carpet clerk dancer dolphin donkey duck eagle frog goat guitar hammer hunter ladder
> lawyer mirror monk nurse owl painter parrot pencil pillow planet prince queen robot rocket shark singer snake tiger
> tractor violin whale wizard

Of the 39, 17 are eligible: 19 are multi-token and 3 are pool target-noun forms. **No score was computed for any word
on this list.** The picks follow list order mechanically, so no score can have selected or reordered them.

**The list's provenance, for transparency.**
- **How the 39 were chosen.** They were composed from an earlier illustrative 50-noun screen. That screen used the
  tokenizer only and printed only aggregate counts.
- **The 11 words dropped by semantic judgment:**
  - `wolf`: an irregular plural;
  - `giant`, `blanket`, `pilot`: an adjective or adjective-like use;
  - `bear`, `judge`, `guard`: strong verb polysemy;
  - `candle`: one of 020's confirmation nouns;
  - `fox`, `engine`, `comet`: omitted during composition, with no recorded reason.
- **The effect on the picks.** 024's stated criteria do not require dropping verb-polysemous words. Had `judge` and
  `guard` been kept, they would have been picked instead of `painter` and `prince`.
- **No score or outcome was involved at any point.** No per-word eligibility was even printed before the list was
  frozen. The reviewer approved the frozen list as written.

## Conditions, manifest and budget

**The conditions per pair (21):**

| condition | runs |
|---|---|
| `base` (θ = 0) | 1 |
| `noun±0.32` | 2 |
| `noun±0.16` | 2 |
| `rand{j}±0.32`, j = 1..7 | 14 |
| `plur±0.32` | 2 |

- **Outcome-bearing:** the 16 primary-dose conditions, nounness `±0.32` and the random controls at `±0.32`.
- **Descriptive:** `base`, the half dose and the plurality control.
- **Keys:** `f"{frame_id}|{word}|{token_id}|{condition}"`. The ledger cannot otherwise tell a patched run from a clean
  one.
- **The manifest:** 40 × 108 × 21 = **90,720** condition-tagged keys.
- **Isolation:** the untagged keys must not collide with any spent key. The spent set is 020's ledger, 020's
  confirmation set, and the 022, 023 and 024 manifests.
- **Runtime:** about 2 hours, including the gates and the Level-1 chain on the 69,120 outcome-bearing runs. 024's
  4,320 prompts took 294 s.
- **Storage:** about 0.9 GB of local, gitignored measurements, with recorded hashes. Per run, every run:
  - the raw float32 captures from which Δx1 and Δx3 derive exactly;
  - Δc and C;
  - the two attention rows.

## Secondary analyses (descriptive; no authority over the outcome)

- the magnitudes: mean and median of `A`, `B` and `G`;
- the even components and the baseline comparisons (ℓ(+) against ℓ(0) against ℓ(−));
- the half-dose `A` (dose-response);
- the plurality-control `A`;
- `A` on normalized MSE (`Σ SSE_C / Σ Q`, with `Q = Σ Δc²`);
- the ladder: the pure block-5 pattern term (C5 − C) and block 4 (L1 − C5);
- `G` against the random directions: `G_i − mean_j |G_ij|`;
- the per-template breakdown;
- the stratum counts;
- the causal effect as a fraction of 024's associational slope.

## Phases and leakage boundaries

| phase | may read | may execute | writes |
|---|---|---|---|
| `validate` | the frozen inputs and the 020–024 committed records | nothing | nothing |
| `freeze` | the tokenizer, the committed lists, noun sets and frames | nothing | `confirmation-v1.json` (the 40 cues and the 90,720-key manifest) |
| `lock` | the committed freeze, 024's calibration record, the embedding matrix | nothing (no forward pass) | the lock (the geometry, the vectors and their digests, the thresholds, the semantics) and the preregistration |
| `confirm` (once) | the lock, the freeze, the weights, 020's locked states | the 90,720 runs, once each, plus the spent patch-path identity check | measurements (before any gate), gates, results |
| `report` | the results state | nothing | the report |

**No calibration phase.** Every threshold is a fixed count (27 of 40), the direction is frozen from 024's record, and
the doses come from 024's locked scores. Nothing needs empirical calibration.

## Engineering tests (spent prompts only; no measured pilot)

**Allowed during implementation:**
- patch fidelity, and the equality of the patched embedding hook and the block-0 residual input;
- the θ = 0 patched path against a plain capture, bit for bit;
- vector I1 against the token-id I1 at θ = 0;
- the geometry and I7′, from the weights only;
- saving, accounting and condition-tagged ledger keys, on the fake world plus real spent prompts at θ = 0.

**Prohibited:**
- Rotated runs on spent prompts may capture **only** the embedding site, to prove the patch landed.
- `ℓ`, `A`, `B`, `G`, `D_attn` and Level-1 values are never computed, logged or inspected for any rotated run before
  confirm.
- Nothing from these tests may change the dose, a statistic, the number of random controls (7), the 27-of-40
  threshold or the population.

## Power (spent data only; approximate)

**The benchmarks:**
- 024's cross-sectional slopes are 1.30 (exposed) and 1.77 (fresh E–N).
- Within a lemma (plural minus singular, 024), Δscore is +0.11 and Δlog MSE is +0.076, with 12 of 16 positive. That
  gives a least-squares slope through the origin of about 0.28: the ratio of the means is 0.69, and with an intercept
  the slope is −1.0. So this benchmark is noisy.

So the causal within-item slope may be far below the cross-sectional one.

**Power of `A`** is P(Bin(40, q) ≥ 27), with q = Φ(μ/sd) and μ = f × 1.3 × 0.32:

| sd of A_i \ f | 0.25 | 0.5 | 1 |
|---|---|---|---|
| 0.1 | 1.00 | 1.00 | 1.00 |
| 0.2 | 0.70 | 1.00 | 1.00 |
| 0.3 | 0.37 | 0.91 | 1.00 |

- **80% power needs about 72% of cues in the predicted sign.**
- **`B`'s power** falls as the typical random-direction effect `c` approaches `μ`. For example, at sd 0.2 and f = 0.5:
  1.00, 0.96, 0.70 and 0.22 for c = 0, 0.25μ, 0.5μ and 0.75μ. That is by design: `B` should fail if random directions
  are nearly as effective.
- **`G`'s power** cannot be estimated without measurement. That is the cost of running no pilot.

## Risks and limits

- **The rotated embeddings were never seen by the model.** The rotation is modest (13.8°, against at least 34.1° to
  the nearest other token in every cue scored so far), and the controls share it exactly.
- **`d` mixes "toward nouns" and "away from the exposed cues"** (cos 0.68 each). The design does not separate them.
- **Block 4's frozen part includes values**, so `ℓ` is not a pure pattern quantity. `G` and the ladder address this.
- **A small causal slope** would lower power (see above).
- **The strict Level-1 gate.** It covers 69,120 outcome-bearing runs; 024's largest value was 0.0099 at the baseline.
  A rotation that leaves the validated regime makes the result `NOT_INTERPRETABLE`, by design.
- **Vocabulary.** 025 consumes 20 of the 23 remaining curated fresh adjectives. That is another reason it is final.
- **`B` measures specificity only against isotropic random tangent directions.** It is not measured against every
  structured alternative direction. The plurality control is the one structured comparison, and it is secondary.

## Review record

- **Spike 1** (norm-preserving tangent rotation at `EMBED@p_c`; frozen-readout error as the primary): the direction was
  approved.
  - The reviewer rejected the uniform-rank specificity null, because `t̂` is not exchangeable with the controls.
  - The reviewer asked for a sign test instead of Wilcoxon.
  - The reviewer asked to decide between readout-error and routing claims.
- **Spike 2:**
  - the sign-count criteria (27 of 40);
  - `B` built from the absolute random effects, because the signed mean is 0 by symmetry;
  - Option B, the direct L4/L5 routing guard `G`;
  - the corrected dose wording;
  - the population, selected mechanically with the tightened prior-noun rule, and the new list frozen.

  Approved in substance.
- **Spike 3:**
  - the terminology ("not an unconditional design-based randomization test");
  - the random-control nounness-neutrality gate in I7′;
  - the corrected `G` wording (a failure is conservative; no head selection);
  - the stratum counts, descriptive only.

  Approved.
- **The pre-commit independent verification** returned PASS WITH NOTES, with no blockers:
  - the 40 cues reconstruct mechanically;
  - no 025 prompt or rotated measurement exists;
  - no score of a new-list word exists.

  Its factual and wording corrections are applied: the even-term bound, the half-dose angle, the tolerance naming,
  the benchmark slope, the list's provenance, the gate domains and the runtime.
- **For the reviewer's explicit sign-off:** the confirm-time patch-path check on 4 spent prompts, frozen by key at
  the lock (the runtime-gates table). It meets the approved runtime identity "the θ = 0 run equals a plain
  capture" without adding fresh runs, and computes no scientific quantity.
  - **If it is declined,** that identity is established only by the implementation-time engineering tests and
    tier C.
  - **Either way,** the 90,720 fresh runs and the outcome rules are unchanged.
- **This revision** adds the three final clarifications: the Level-1 domain over all 16 outcome-bearing primary-dose
  conditions; `A` meaning + against −, not against the baseline; `D_attn` notation.

## Implementation boundary (for the plan)

- **New code:**
  - a new module (planned as `src/neural_decompiler/cue_rotation.py`);
  - a runner `experiments/025-nounness-direction-intervention/run.py`.

  The pinned 020–024 modules are not edited; they are called.
- **The code it needs:**
  - a vector variant of `pm.lexicon_vector` for vector I1;
  - a patched measurement: `pm.run_patched` with the measurement sites and the L4/L5 pattern sites;
  - the geometry, and the deterministic control generator.

  It needs no general intervention framework.
- **Tests:**
  - tier A for the geometry, the gates and the sign-count rules;
  - tier B for a fake-world runner, including the write ordering;
  - tier C for spent-only real-model contracts under the engineering restrictions.

## Stopping condition

This design is committed alone, then the implementation plan is written and committed separately, and work stops for
review. Then, each step only when authorized:
1. implementation and its independent review;
2. `freeze`, the review and the commit;
3. `lock`, the review, the install and the commit;
4. `confirm` once, then the independent confirmation review;
5. `report`;
6. closure.

**After 025, on any clean result, the experiments stop, and the project is written up.**

## Revision history

- Revision 1 (2026-09-26): the final design after three spike rounds, with the reviewer's three final clarifications
  (the Level-1 domain, the meaning of `A`, `D_attn`).

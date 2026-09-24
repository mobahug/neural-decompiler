# Experiment 024: Does an Operational Nounness Score Predict When the Frozen Downstream Routing Stops Holding?

**Date:** 2026-09-24

**Status:** Revision 1, for design review. It follows:
- Experiment 023's closure (`b45a646`);
- the post-closure diagnostic of 023's spent data;
- the exposed-only design spike for 024;
- the reviewer's decisions: Option C as the primary test and Option A as a secondary discriminating analysis, with the
  seven pre-design points settled.

No code, runner, freeze, calibration or fresh prompt exists yet. The diagnostic and the spike measured no new cue. They
used spent data, the weights and the tokenizer only; every forward pass was refused, and nothing was written into the
repository.

## Background: what Experiment 023 settled (confirmed)

Experiment 023 closed with four preregistered conditions:
- **Three PASS and one ENVELOPE_ONLY_FAILURE** (Y1 cue-final: `g = 0.9955204329648195` against its floor
  `F = 0.9968974947302105`).
- **No aggregate label**; each condition is read on its own.
- **Upstream is effectively reconstructed.** The fully weight-derived upstream program recovered essentially all of
  the explainable gap between Level 0 and the measured-`Δx3` ceiling on 24 new cues in the exposed and in new frames:
  `g ≥ 0.9955` in every condition.
- **The remaining error sits downstream.** The frozen Experiment 020 readout, fed the measured `Δx3`, explained
  `R²_C` 0.95–0.96 of `Δc` on those units.

Experiment 024 asks about that readout.

## Post-hoc background (hypothesis generation only; not Experiment 024 evidence)

Everything in this section is post hoc, computed on already-spent data. It motivates the design and is never evidence
for its hypothesis.

1. **The upstream residual is `R`.** The excess error of 023's `P1` over the ceiling is entirely 022's `R`: the
   reduced layers 1–2 in place of the exact ones.
   - Restoring the exact layers removes 100 % of `R²_C − R²₁`.
   - The excess is small: 0.00038–0.00131 of the variance.
2. **The readout residual is the frozen attention of blocks 4 and 5.** The frozen readout holds block-4 attention at
   the reference and block-5 attention at the reference pattern.
   - 020's exact Level-1 chain reproduces the measured `Δc`: the largest difference is 1.1e-2 nats, against a
     readout-residual RMS of 0.24–0.30 and a maximum of 3.8.
   - Making block-5 attention exact removes 31–61 % of the readout residual; making block-4 attention exact removes
     the rest.
3. **Where the fresh readout error concentrated.** On 023's fresh cues, the extra readout error was concentrated in
   the "quantity" stratum:
   - 1.6–2.3× the error of the exposed quantity cues, in absolute and in normalized terms.
   - Six of its eight words are plural measure nouns (tons, piles, masses, stacks, scores, batches), while the exposed
     quantity cues are quantity adjectives and determiners (least, total, whole, plenty…). gross and net were just as
     high.
   - Frames and nouns did not explain it.
   - Other drops in `R²_C` came from a smaller spread of `Δc`, not from larger error.
4. **Nounness had the strongest association.** Across 163 measured cues, the operational nounness score (defined
   below) had the strongest association with per-cue readout error: Spearman +0.53 over all cues, +0.50 within the 139
   exposed cues alone.
   - Weaker: distance from the calibration distribution (+0.35 / +0.29), token id as a frequency proxy (+0.25 /
     +0.19), the number direction (plural morphology; +0.16 / +0.12). Embedding norm was about 0.
5. **It predicts within classes and out of fit.**
   - Within classes: determiner-like +0.26, quantity +0.31, adjective +0.43.
   - The 36 pronoun cues, never in the fit: +0.42.
   - 023's 24 spent cues over all 108 frames: +0.50, with the exposed-fitted line's median |log error| 0.10.
   - Leave-one-cue-out median |log error|: 0.15.
6. **Power, from simulations using the expected picks' scores** (not an outcome), with the exposed residual spread:

   | if | probability that the fresh Spearman ≥ 0.315 |
   |---|---|
   | the exposed nounness relation holds | 0.999 (median ρ 0.65) |
   | only plural morphology matters (only B and C elevated) | 0.983 (median ρ 0.52) |
   | only measure semantics matters (only B and D elevated) | 0.090 |
   | there is no effect | 0.028 |

   **The primary test cannot separate nounness from plurality.** Plural forms score higher than their own singulars:
   on average +0.10 across the B/D lemma pairs and +0.12 across the C/E pairs. Only the lemma-paired secondary
   contrasts separate the two: the plurality effect, and D/E against N.

7. **Ordinary plural nouns have never been measured as cues.** The crucial control is genuinely prospective.

## The frozen objects

**The downstream comparator `C`.** Experiment 020's Level-0 readout (pinned module `readout_decompilation.py`), fed the
measured `Δx3` at `p_c`, and at `p_t` in coordinated frames:
- **Block 3:** every head recomputed by the exact layer program on the changed state; the MLP at the frame's operating
  point.
- **Block 4:** attention frozen at the reference, so no attention change; the MLP at the operating point.
- **Block 5:** heads as the frame's reference rows times the value changes (frozen pattern); the MLP at the operating
  point.
- **Output:** exact `LN_final`, then the noun read `Δc(n) = ⟨ΔLN_final, Δw(n)⟩` over the 79 scorable exposed nouns
  (`ul.contrast_of`).

`C` is a comparator. Its two frozen attention approximations are exactly what 024 studies.

**The readout error of a pair:** `e(cue, frame, n) = Δc_measured − C`, over the 79 nouns.

**The operational nounness score (frozen):**

`nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue)`

- `E_w` is the float64 input-embedding row of the single token `' ' + w` in the pinned checkpoint
  (`EleutherAI/pythia-70m-deduped`, revision `e93a9faa…`).
- `μ_noun` is the mean of the 158 singular and plural embedding rows of the 79 scorable exposed target nouns.
- `μ_cue` is the mean of the embedding rows of the 139 exposed cues of the three calibration strata.
- For an exposed calibration cue, `μ_cue` is leave-one-out (computed without that cue). A fresh cue is never in
  `μ_cue`.
- Cosine: `⟨a, b⟩ / (‖a‖ ‖b‖)`, float64.

It is an **operational, weight-derived score relative to the target-noun and exposed-cue embedding populations**, not
a general linguistic nounhood oracle. It uses no measured outcome, and it never selects, rejects, reorders or classifies
a candidate.

## Question (primary)

Does the frozen operational nounness score prospectively predict the per-cue error of the frozen block-4/5 attention
readout on 40 never-executed cue words, in the 108 exposed frames?

The hypothesis is **predictive and associational**: higher operational nounness predicts larger readout error. The
working mechanistic reading, that noun-like cue representations redistribute downstream attention more so that the
frozen reference routing no longer holds, is not what is tested. 024 does not claim that nounness causes attention
redistribution.

## Populations and fresh units

- **Frames:** the 108 exposed frames: 36 cardinal, 36 quantifier, 36 coordinated-adjective; 72 cue-final and 36
  coordinated. No new frame enters 024.
- **Nouns:** the 79 scorable exposed nouns.
- **Fresh cues:** 40, as five classes × 8.

| class | content | form |
|---|---|---|
| **N** | non-noun controls: adjectives and determiner-like words with no common noun use | single words |
| **B** | plural measure and quantity nouns | plural of the B/D lemmas |
| **D** | singular measure and quantity nouns | singular of the same B/D lemmas |
| **C** | ordinary plural nouns | plural of the C/E lemmas |
| **E** | ordinary singular nouns | singular of the same C/E lemmas |

Lemma pairing makes the plurality contrast within-lemma.

**Eligibility** (tokenizer and text only):
- `' ' + w` is a single token;
- its id was never used as a cue by Experiments 005–023: Experiment 023's exclusion (327 ids) plus 023's 24 cues and
  its frames' cue ids;
- it is not a form of a target noun;
- neither form of the lemma (the word itself, for N) occurs as a token anywhere in the 108 exposed frames. This is the
  lesson of 023's `" shiny"`.
- For B/D and C/E, a lemma counts only when both its singular and its plural are eligible.

**Class rules**, curated text, applied when the lists were written and frozen with them:
- **N:** no word with a common noun use. Excluded, for example: light, major, minor, gross, net, calm, noble, hollow,
  crisp, stiff, tender, following.
- **Noun classes:** ordinary verb uses are allowed (crowd, bundle, cluster). Common adjective, determiner or modal
  senses are excluded (for example can).
- **C/E:** no container or measure nouns (cup, box, bag, jar, bottle, basket).
- **Ambiguous words:** adjective/noun items such as gross and net belong to no primary class.

**Ordered candidate lists (frozen).** `freeze` takes the first 8 eligible entries per class mechanically. A shortfall
writes nothing and stops for review. Design-time tokenizer eligibility is marked.

| list | ordered candidates |
|---|---|
| N | eager ✗, fierce ✗, **honest, polite, rude, sleepy, wise, lucky, merry, nervous**, anxious, cheerful, clumsy ✗, curious, grumpy ✗, jealous, lonely, nasty, careful, careless, famous, friendly, gorgeous, hungry, thirsty ✗, weary, wicked, ugly, vivid, vague, rapid, rigid, clever, fuzzy, latest, earliest, brave ✗ |
| B/D lemmas | **gallon, ounce, acre**, pint ✗, quart ✗, **herd**, flock ✗, swarm ✗, **crowd, bundle**, bunch ✗, heap ✗, mound ✗, **cluster**, handful ✗, **litre**, liter ✗, dozen ✗, barrel, bucket, sack, crate ✗, basket ✗, carton ✗ |
| C/E lemmas | **apple, horse, doctor, king**, teacher ✗ (frame token), **rabbit, poet**, tiger ✗, castle ✗, pencil ✗, **dragon, lion**, farmer ✗ (frame token), soldier, sailor, priest, queen ✗, baker ✗, knight, lemon ✗, banana ✗, onion, carrot, violin ✗, wizard ✗, pirate, tourist, statue |

✗ marks an entry ineligible at design time: multi-token, an id used before, or a frame token.

**Expected picks:**
- **N:** honest, polite, rude, sleepy, wise, lucky, merry, nervous.
- **B / D:** gallons / gallon, ounces / ounce, acres / acre, herds / herd, crowds / crowd, bundles / bundle, clusters /
  cluster, litres / litre.
- **C / E:** apples / apple, horses / horse, doctors / doctor, kings / king, rabbits / rabbit, poets / poet, dragons /
  dragon, lions / lion.

**Reserves**, in order: N from anxious; B/D barrel, bucket, sack; C/E soldier, sailor, priest, knight, onion, carrot,
pirate, tourist, statue.

The manifest is 40 cues × 108 frames = **4,320** target prompts. There is no stage 1: every frame is exposed, and its
reference state is 020's locked state.

## The primary statistic (frozen)

**Per-cue response:** `MSE(cue) = (1 / (108 · 79)) · Σ_frames Σ_nouns (Δc_measured − C)²`
- over all 108 exposed frames and all 79 nouns;
- every frame × noun cell weighs the same, so the battery keeps its 72:36 cue-final to coordinated composition.

**Primary statistic:** the Spearman correlation `ρ` across the 40 fresh cues between `nounness(cue)` and `MSE(cue)`.
- **Frozen implementation:** ranks by ascending value, ties at the average rank, then the Pearson correlation of the
  two rank vectors, in float64.
- A log transform of MSE leaves the ranks, and so `ρ`, unchanged.
- `ρ` is undefined when either rank vector is constant.

**Results**, in precedence order, for the single primary statistic:
1. `NOT_INTERPRETABLE`: `ρ` cannot be defined.
2. `GUARD_FAILURE`: `ρ < null₉₇.₅`, no predictive information beyond chance.
3. `ENVELOPE_ONLY_FAILURE`: `null₉₇.₅ ≤ ρ < F_ρ`, predictive beyond chance but weaker than the exposed-like
   relationship.
4. `PASS`: `ρ ≥ max(F_ρ, null₉₇.₅)`.

The two thresholds serve different purposes, and both come from calibration, never chosen after fresh outcomes:
- `F_ρ` asks that the fresh result is not unusually weak compared with the frozen exposed relationship.
- `null₉₇.₅` requires predictive information beyond chance.

The preregistered guard is the one-sided 97.5 % null. The 99.5 % value is never used.

## Calibration (exposed only, once; no forward pass)

**Population:** the 139 exposed cues of the three strata (determiner-like 45, quantity 45, adjective 49), the same
cues that define `μ_cue`. The 36 pronoun cues stay outside calibration; they are a spent-data, out-of-fit diagnostic
only.

**Per-cue MSE of a calibration cue:** `Σ SSE_C / Σ n` over its 108 exposed pairs. It is read from 023's committed
exposed-cells artifact (`d2ee71e5…` / `62818147…`, verified against its index and content digests), the same formula
as the fresh response. Calibration never opens Experiment 022's table.

**Nounness of a calibration cue:** with its leave-one-out `μ_cue`, from the pinned embedding matrix. Only the weights
are read.

**`F_ρ`:**
- 10,000 SHA-indexed draws of 40 cues, with replacement, from the 139:
  `int.from_bytes(sha256(f"024|primary|{b}|{slot}")[:8], "big") % 139`.
- Each draw's Spearman uses the frozen implementation.
- `F_ρ` is the 250th ascending value (1-based), zero-based element `[249]`.
- An undefined draw is placed at −∞. 250 or more undefined draws stop calibration for review (none is expected).
- **Direction check:** `F_ρ ≤` the median of the draws.

**`null₉₇.₅`:**
- 100,000 deterministic SHA-indexed permutations of 40 distinct ranks, tag `024|null`. Each permutation is built by a
  Fisher–Yates shuffle whose swap indices come from `sha256(f"024|null|{p}|{i}")`.
- Each permutation's Spearman against the identity ranking.
- `null₉₇.₅` is the 97,500th ascending value (1-based), zero-based element `[97499]`.

**The secondary numerical predictor:** OLS of `log MSE` on the leave-one-out nounness over the 139 cues, giving the
slope, intercept and residual standard deviation.
- Design diagnostics give 1.299, −2.481 and 0.257. The exact values are recomputed at calibration and frozen in the
  record; the rounded prose values are never copied.
- It is an exposed-data-fitted, prospectively frozen predictor. The primary statistic uses no fitted coefficient.

**Also recorded, descriptive only:**
- the calibration Spearman over the 139 cues (design diagnostic +0.530);
- the within-class Spearman;
- the pronoun out-of-fit check;
- the draw distribution's median and tails.

The design diagnostics give `F_ρ ≈ 0.257` and `null₉₇.₅ ≈ 0.315`, so the null guard binds; the exact values come from
calibration. The candidate record is installed byte-identically, committed and reviewed before `lock`.

## Freeze (tokenizer only; no model output)

- Applies the eligibility and class rules to the ordered lists.
- Writes `confirmation-v1.json`:
  - the 40 cues with class, lemma and token id;
  - the exclusion and its sources;
  - the frame-token rule's inputs;
  - the manifest: 4,320 S2-TARGET keys, and no stage 1.
- Nounness is not computed here.
- The file is committed by hand and independently reviewed.

## Lock (weights only; no forward pass)

The lock binds:
- each fresh cue's nounness score (float64, with a digest), computed from the pinned embedding matrix;
- the secondary line's prediction for each cue;
- `F_ρ` and `null₉₇.₅` from the committed record;
- the semantics, the manifest, the exposed-state digest, the module blobs, the design and plan, and the confirmation
  file and calibration record digests.

It also renders the preregistration. Every installed file is committed, byte-identical, and independently reviewed.
There is no prediction table: the outcome needs measurement, and the predictor is the score.

## Confirm (once; never resumed)

1. `validate_lock`: tracked files, a clean tree, no scientific change since the lock. Then the runtime check. The
   isolation check: the ledger must hold none of 020's ledger, 021's spent set, 022's manifest or 023's manifest.
2. The 4,320 target keys are recorded in the ledger. Then each prompt runs once, through 023's measurement function:
   `Δc` over the 79 nouns, and `Δx3` and `Δx1` at the changed positions.
3. Every measurement is saved before any gate. The prompt accounting must hold: expected manifest = executed prompts =
   ledger, with no extra, missing or duplicate key.
4. **Gates**, inherited from 022 and 023. A failure is an incident, and an incident carries no result:
   - I1: the block-0 decomposition against the measured `Δx1`, 1e-4;
   - I3: 022's full composition against the measured `Δx3`, relative 1e-4;
   - I4: the full composition's `Δĉ` against `C`, 1e-3 nats.
5. `C` from the saved `Δx3`, then the per-cue MSE, the primary `ρ` and its result. The results are written, together
   with the completed phase, before anything descriptive runs.
6. The secondary analyses and the descriptive records.

## Secondary analyses (frozen; no authority to redefine the primary result)

Reported for every cue and class:
- Spearman on cue-final frames only (per-cue MSE over the 72 cue-final frames) and on coordinated frames only (36).
  Disagreement between the two never overturns the primary result.
- Normalized MSE (`MSE / mean Δc²`) and its Spearman with nounness.
- `Var(Δc)` per cue, and `R²_C` per cue (pooled over its pairs).
- Bias, `mean(Δc − C)`, and slope, the OLS of `Δc` on `C` per cue.
- The frozen line's numerical prediction against the observed `log MSE`: median |log error|, mean signed error.

**Lexical contrasts** on the per-cue `log MSE`, with class means `m_K`:

| contrast | definition |
|---|---|
| noun effect | `mean(m_B, m_C, m_D, m_E) − m_N` |
| measure effect among nouns | `mean(m_B, m_D) − mean(m_C, m_E)` |
| plurality effect among nouns | `mean(m_B, m_C) − mean(m_D, m_E)` |
| measure × plurality interaction | `(m_B − m_D) − (m_C − m_E)`, descriptive |

- Each is reported with a deterministic cue-level bootstrap 95 % interval: 10,000 SHA-indexed resamples of cues within
  class, tag `024|contrast`, elements `[249]` and `[9750]`.
- **Frozen qualitative readings.** They are descriptive, no explanation is declared a winner, and more than one reading
  may apply:
  - a positive noun effect, with measure and plurality effects near 0, is consistent with the nounness account;
  - a measure effect with little plurality effect is consistent with measure semantics;
  - a plurality effect with little measure effect is consistent with plural morphology.

**The descriptive block-4/5 ladder**, per class: the share of `Σ (Δc − C)²` removed by making block-5 attention
exact, then block-4 attention (020's Level-1 chain). The Level-1 identity, `max |Δĉ_L1 − Δc|` against 020's readout
identity tolerance of 2e-2 nats, is reported. It is not a gate.

## Gates and tolerances

| check | tolerance | on failure |
|---|---|---|
| I1, I3, I4 | 1e-4, relative 1e-4, 1e-3 nats | incident |
| `C` recomputed from the saved `Δx3` against the confirm-time `C` | bit for bit | incident |
| the prompt accounting | exact | incident |
| the exposed-cells artifact at calibration (index, content and data digests) | exact | refused |
| the Spearman implementation against a direct recomputation | 1e-12 | incident |
| the Level-1 identity | 2e-2 nats | descriptive only |

## Phases and leakage boundaries

| phase | may read | may execute | writes |
|---|---|---|---|
| `validate` | frozen inputs, 020–023 closure records and digests | nothing | nothing |
| `freeze` | the tokenizer, the committed cue and frame sets, the exposed frames' token ids | nothing | `confirmation-v1.json` (committed) |
| `calibrate` (once) | 023's committed exposed-cells artifact, the embedding matrix | nothing (no forward pass) | the candidate calibration record |
| `lock` | the committed record and confirmation, the embedding matrix | nothing (no forward pass) | the lock and the preregistration |
| `confirm` (once) | the lock, the confirmation file, the weights, 020's locked states | the 4,320 S2-TARGET prompts, once each | measurements (before any gate), gates, results |
| `report` | the results state | nothing | the report |

Like 023, the frozen-input loader also reads the hash-pinned 020 results state, which is archived in
`experiments/023-block0-completion/evidence/archive/`, for integrity and isolation checks.

## Incidents and interpretability

- **Hard identities** (I1, I3, I4, the `C` recomputation, the accounting): an incident, with no result.
- **Stops for review, not incidents:** a freeze shortfall, and a calibration stop (250 or more undefined draws, or a
  reversed direction check).
- **`NOT_INTERPRETABLE`:** only if `ρ` is undefined.
- An interruption of `confirm` spends the protocol version.
- Nothing is retried at an incident's commit.

## Interpretation (frozen)

- **`PASS`:** higher operational nounness prospectively predicted larger error of the frozen block-4/5 attention readout
  on unseen cue words. The prediction was at least as strong as the exposed-like relationship and beyond chance.
- **`ENVELOPE_ONLY_FAILURE`:** the score carries predictive information beyond chance, but less than the exposed
  relationship would lead one to expect.
- **`GUARD_FAILURE`:** no evidence that the score predicts the readout error on fresh cues beyond chance.
- **`NOT_INTERPRETABLE`:** no result.
- **What a `PASS` does not show:**
  - that nounness causes the attention change;
  - that the score measures linguistic nounhood;
  - which lexical property drives the error, and in particular nounness as against plurality (the secondary contrasts
    inform this, with no winner);
  - generality beyond the 108 exposed frames, the 79 nouns and this checkpoint.

## Interpretation limits

- The score is operational, relative to the 79 target nouns and the 139 exposed cues.
- **The primary association cannot separate nounness from plurality.** The score is higher for plural forms than for
  the singulars of the same lemmas, and a plural-morphology-only world would pass the primary test with probability
  0.98 in the design simulation. The lemma-paired contrasts are the discriminating analysis: a positive plurality
  effect, or D/E at the level of N. A `PASS` alone says that the score predicts the error, not which of the two
  properties does.
- The classes are curated by hand from ordered lists, frozen before any measurement.
- There are 8 cues per class, one checkpoint and the exposed frames only.
- The comparator is 020's frozen readout.
- The 36:72 group composition is fixed by the battery.

## Review record

**Reviewer decisions (2026-09-24):**
1. **Option C primary, A secondary.** The primary test is the prospective Spearman between the frozen nounness score
   and per-cue readout MSE. The secondary analysis is the fresh lexical contrasts, preregistered in factorial form.
2. **Populations.** Fresh N, B, C, D and E, 8 each, for 40. The exposed quantity adjectives are calibration only,
   never the control group. There is no gross/net class. The ordered lists, the expected first-eight picks and the
   reserves are approved. Verb uses are allowed in noun classes; adjective, determiner and modal senses are not; C/E
   has no container or measure nouns. The frame-token rule stands. Class assignment is text-based only.
3. **The score.** Called an operational weight-derived nounness score. The calibration population is 139 cues, not
   175.
4. **The line.** The OLS line is an exposed-data-fitted, prospectively frozen predictor, never "parameter-free". It
   is recomputed exactly at calibration.
5. **Response.** Absolute per-cue MSE. Normalized MSE, `Var(Δc)`, `R²_C`, bias, slope and the line's prediction are
   secondary.
6. **Pass rule.** PASS iff `ρ ≥ max(F_ρ, null₉₇.₅)`, with the four-way semantics. The 99.5 % threshold is never used.
7. **Frames.** The exposed frames only, all 108 pooled for the primary MSE. Cue-final and coordinated Spearman are
   secondary.
8. **No measured spike.** No cue is measured during design.
9. **Background.** The post-hoc material stays hypothesis-generation background only.

## Implementation boundary (for the plan, after approval)

- **New module:** planned as `src/neural_decompiler/readout_routing.py`.
- **Reused frozen dependencies, pinned by git blob:**
  - 022's module and the ten modules it pins, including `readout_decompilation.py` (`C`, Level 1);
  - 023's module (`block0_completion.py`), for the exposed-cells artifact reader and its measurement function.
- **Runner phases:** `validate`, `freeze`, `calibrate`, `lock`, `confirm` and `report`. There is no extraction:
  023's committed artifact is the calibration source.
- **Committed:** the confirmation file, the calibration record, the lock and the preregistration. The results state and
  the measurements stay in `outputs/experiment-024/`, and closure commits the evidence.
- **Tests:** a tier-B fake-world runner test, tier-A pure tests of the score, the Spearman implementation, the draws
  and the null, and tier-C contracts: the real-tokenizer freeze picks and the real score values.

## Stopping condition

Stop after this revision for design review. Then, each step only when authorized:
1. the implementation plan and its review;
2. the implementation and its independent review;
3. tier C on the clean gated commit;
4. `freeze`, then commit and the freeze review;
5. `calibrate` once, then install, commit and the floor review;
6. `lock`, then install, commit and the independent lock review;
7. `confirm` once, then the independent confirmation review;
8. `report`;
9. closure.

## Revision history

- Revision 1 (2026-09-24): after 023's closure, the post-closure diagnostic, the exposed-only design spike and the
  reviewer's choice of Option C primary with Option A secondary.

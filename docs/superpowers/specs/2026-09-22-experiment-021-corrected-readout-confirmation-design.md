# Experiment 021: Corrected Readout-Decompilation Confirmation — the Experiment 020 Question with Floors from a Preregistered Exposed-Only Calibration

**Date:** 2026-09-22

**Status:** Revision 1 — design proposal for review. Nothing is implemented: no Experiment 021 directory, code,
results state, calibration record, lock or model run exists. This document freezes the calibration procedure and the
rule that turns its distributions into floors; **the floors themselves have not been computed** and are computed
exactly once, after approval, by the committed `calibrate` phase, whose output is reviewed before any lock. No
fresh prompt, fresh-frame state or fresh-noun quantity was computed or inspected for this design; the only
Experiment 020 values it quotes are the corrected exposed aggregates already recorded at its closure.

**Kind:** Prospective, zero-parameter, mechanism (readout). Same question, same program, same model and same untouched
confirmation set as Experiment 020; only the floors are re-derived, mechanically, from the corrected exposed record.
It is not a new research direction.

## Why a new protocol version

Experiment 020 closed at Tier A
([README](../../../experiments/020-readout-decompilation/README.md), `closure.json`, commit `7f4a4f8`). Its floors had
been calibrated on a scratch design pass that fed block 5's MLP the post-attention residual; Pythia is a
parallel-residual model, the production implementation is correct, and the official exploration at `ce3766b`
produced corrected exposed values far from the scratch ones — flattened `R²` 0.689 against 0.848, token means 0.680
against 0.942, pair means 0.577 against 0.858, frame means 0.433 against 0.702, coordinated-adjective 0.273 against
0.702, median per-noun `R²` 0.610 against 0.829 — with six of the twelve floor conditions below threshold on the
exposed pool itself. A confirm under 020 would have tested floors inherited from an implementation known to be wrong,
so its labels could not be read cleanly; 020 therefore did not spend its confirmation set, kept its floors unchanged
and assigned no label.

Experiment 021 asks the same question with floors that are (i) derived from the corrected exposed record, (ii) by a
rule frozen here before its output is known, (iii) applied identically to every condition, whether that raises or
lowers the Experiment 020 value. The 020 floor values play no role in the rule.

## Question (unchanged)

Given the change entering the post-`L03` readout path as the decoded chain *predicts* it (Experiments 011/012/017,
never measured), can the resulting change `Δc` of the model's own singular-versus-plural logit contrast at `p_t` be
predicted prospectively for unseen cues in the exposed frames (Y1), for unseen cues in unseen frames conditional on
each frame's own reference state (Y2), and for unseen nouns (Y3)?

Under calibrated floors the operative reading is: *are the fresh units predicted at least as well as exposed units of
the same strata are, allowing for the sampling variability of a set of the confirmation's size, and with positive
skill?*

## Preserved unchanged (each asserted by the implementation)

| element | frozen value | how it is asserted |
|---|---|---|
| model, checkpoint, runtime | pinned `EleutherAI/pythia-70m-deduped` `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; CPU float32; 4 threads; seeds `20260916` / `20260924`; TransformerBridge without compatibility mode, `use_attn_result=True`; torch 2.14.0, transformers 5.17.0, transformer-lens 3.9.0, huggingface-hub 1.31.0, Python 3.12.13 | runtime and versions equal to Experiment 020's recorded explore runtime at every scientific phase |
| the Level 0 program | `src/neural_decompiler/readout_decompilation.py` exactly as at `ce3766b` (git blob `caa73b40192f4c910dc63371bd19db75a3258339`): the corrected parallel-residual block 5, the retained layer-5 frozen-row term, layer-4 attention frozen, MLPs at the frame's operating point, `LN_final` exactly, the weight-only noun read | the file's blob hash checked by a test and by every phase; 021 code calls it and reimplements none of it |
| inherited objects | Experiment 011 lock `769bfeac…`, 012 lock `830abc3b…`, 017 lock `b4fc9014…`, read verbatim, never refitted | digests, as in 020 |
| exposed pool | `build_pool_020`: 108 frames, 279 cues, 80 nouns (79 scorable; `peach` not single token) | counts asserted |
| confirmation set | `experiments/020-readout-decompilation/confirmation-v1.json`, content sha256 `e098e2b44a1702d2b35c20db2ce111358897ea867996024c3e51a79f379c309d`, read in place as Experiment 007 read Experiment 006's — never copied, rewritten, filtered, reordered or selected from: 24 cues, 18 frames, 24 nouns, manifest S1-REF 18 / S1-VALIDITY 18 / S2-TARGET 2592 + 432. No file-format change is needed, so the digest stays | digest; Experiment 020's closure record and its ledger (0 of 3060 keys executed) verified before every phase |
| populations | Y1 = 24 fresh cues × 108 exposed frames × 79 scorable exposed nouns; Y2 = 24 fresh cues × valid fresh frames × 79 exposed nouns; Y3 = 24 fresh nouns on the Y1 pairs; joint = fresh × fresh × fresh, descriptive | as in 020 |
| procedure | the two-stage confirmation, the barrier, the frozen validity rule, the prompt manifest, one execution per key | 020's functions, unchanged |
| preconditions | ≥ 16 scored cues and ≥ 60 valid exposed frames (Y1); ≥ 12 valid fresh frames with ≥ 4 coordinated (Y2); ≥ 18 scorable fresh nouns (Y3); ≥ 3 valid frames per scored cue | as in 020 |
| tolerances | readout 2e-2, logit 2e-2, additive 1e-4, `E₁` 7e-3, inherited 017 1e-6, locked rows exact, stage-1 digest and exact rows, provenance 1e-6 | as in 020 |
| outcome concepts and labels | `CONTRAST_PREDICTED_TOKENS` / `CONTRAST_NOT_PREDICTED_TOKENS` / `PRECONDITION_FAILED_TOKENS`; `…_FRAMES_CONDITIONAL`; `NOUN_READOUT_FIXED` / `NOUN_READOUT_NOT_ESTABLISHED` / `PRECONDITION_FAILED_NOUNS`; outcome `Y1 \| Y2 \| Y3` | as in 020 |
| comparators | ΔT-only (exposed-disfavoured baseline), template-base MLPs (exposed-disfavoured operating-point comparator), no-layer-5-head (nested simplification comparator), rank-1 nouns (descriptive), with 020's frozen objects | as in 020 |

Experiment 020's results state (`outputs/experiment-020/results.json`, state sha256 `2e5485dc…`, file sha256
`da63b8c2…`) is read only; every phase verifies both digests against the committed closure record and re-verifies
them after it runs.

## What changes (the complete list)

1. **The twelve numeric floors** `F1…F12` are derived by the calibration below.
2. **The share conditions are written as the equivalent order statistics.** For a fixed share `s` of `n` units,
   "at least a share `s` meet criterion `c`" is identical to "the `k`-th best unit meets `c`" with
   `k = ⌈s·n⌉`. The shares 80 % (cues), 75 % (frames) and 90 % (nouns) stay; what is calibrated is the per-unit
   threshold `c`, never the share. (Calibrating the share instead would put a knife-edge at 100 % wherever every
   exposed unit already meets the old criterion, as with per-cue MAE, slope and bias.)
3. **The slope condition becomes `|slope − 1|`**, the symmetric deviation from the ideal; its asymmetric 020 band
   `[0.75, 1.15]` was itself read off the scratch calibration. This is the only change in a condition's shape.
4. **Added, descriptive:** the downstream ceiling (the same program fed the measured `Δx3`) on both fresh sets, with
   the error split. Experiment 020's design promised it, but its stage 2 never computed it — a gap found while
   writing this design; 020 never reached stage 2, so nothing was affected.
5. **Experiment 021's own results state, calibration record, lock and report**, and Experiment 020's runner refusing
   `lock` and `confirm` (its closure record, already committed).

## The calibration data: the exposed table must be re-materialized

The review's premise was that the completed 020 exposed measurements and predictions suffice without any model run.
They suffice only in part. Experiment 020's exploration record keeps per-unit **summaries** — per-cue `R²` and MAE
(279), per-frame `R²` and MAE (108), per-template `R²`, per-noun `R²`, slope and bias (79) and the aggregates — but not
the per-pair × per-noun table (30,024 × 79 measured and predicted `Δc`), and every summary is taken over all exposed
cues rather than over a 24-cue set:

| statistic on a pseudo-fresh set | recomputable from the recorded summaries? |
|---|---|
| Y1 per-cue MAE order statistic; Y1 pooled MAE | yes (per-cue MAE over 108 frames × 79 nouns is exactly what Y1 uses) |
| Y1 token-mean `R²`, pair-level noun-mean `R²` | no — needs per-cue and per-pair means |
| Y2 frame-mean `R²`, per-frame `R²`, both splits on 24 cues × 18 frames | no — the recorded per-frame values average over all 278 cues |
| Y3 median, order statistics, slope and bias on 24 cues | no — the recorded per-noun values average over all cues |

Ten of the twelve statistics need the per-pair table, and no summary-only substitute measures the quantity that
Y1–Y3 actually score. The table is therefore **re-materialized from the exposed ledger and verified against the
recorded record** before anything is drawn from it:

1. **Only Experiment 020's exposed ledger runs.** `calibrate` executes the 108 exposed reference prompts and the
   30,024 exposed cue prompts — as a set, exactly the 30,132 keys of 020's ledger — in 020's explore order. Before
   each frame it asserts that every key it is about to run is in 020's ledger and outside the confirmation manifest;
   its own ledger is asserted equal to 020's at the end. No fresh noun is constructed (`NounSet` over the exposed
   nouns only), and 021's record is checked with 020's `assert_fresh_nouns_absent`.
2. **The environment check first.** Each re-captured reference state is compared with 020's locked state (`x1…x5` at
   every position, `h6`, the layer-4/5 rows, `c_ref`) at `1e-9`, the locked-state tolerance of Experiments 013–019.
3. **The same computation as 020's explore**, with 020's functions: `measure_pair` with the same capture sites,
   `predict_pair` with 020's recorded template bases, `ceiling_prediction` from the measured `Δx3`, and every
   identity on every pair enforced at its frozen 020 tolerance.
4. **The reproduction gate.** From the re-materialized table, 020's `pair_statistics` and `noun_statistics` and the
   comparator values are recomputed and compared with 020's recorded exploration record, every numeric leaf — all
   aggregates, all 279 per-cue, 108 per-frame and 3 per-template entries, all 79 per-noun entries, the ceiling,
   no-layer-5-head, template-base and ΔT-only values — at an absolute tolerance of `1e-9`
   (`RECONSTRUCTION_TOLERANCE`). A failure is an incident: nothing is drawn and no floor is computed.
5. The table (measured, Level 0, ceiling; float64) is written to `outputs/experiment-021/` and digested in the
   results state.

Cost ≈ one Experiment 020 explore (1 h 09 m on this machine). It introduces no new information: identical prompts,
identical model, verified identical outputs.

## The pseudo-confirmation draws (frozen)

**Each draw is one complete pseudo-confirmation set** of the confirmation's exact size and composition, and the
twelve statistics are computed on it exactly as `confirm` will compute them on the fresh set:

| unit | fresh set | pseudo-fresh pool (by provenance, fixed before any calibration output) | pool size | draw |
|---|---|---|---|---|
| cue | 6 each: `determiner-like`, `quantity`, `possessive-or-pronoun`, `adjective` | exposed cues of those four frozen classes that entered the project as the fresh cues of the Experiment 011–019 confirmations (the five-class scheme of the same candidate-list construction; the 006 cohort's four `quantity` words come from a different class scheme and are excluded) | 45 / 45 / 36 / 49 = 175 | 6 per class |
| frame (Y2) | 6 per template | exposed frames first confirmed in Experiments 017, 018 and 019 — the frames that entered **no** fit the program inherits (011's axes used 8 per template, 012's bases 10, 017's layer-3 bases 22) — screened by the frozen stage-1 validity rule, evaluated from the re-materialized table (each exposed frame's reference state and its template's plural-cue measurement, both in 020's ledger) | 14 per template = 42 before screening | 6 per template |
| frame (Y1, Y3) | the 108 exposed frames | the same 108 frames (the Y1 population *is* the exposed frames) | 108 | all, fixed |
| noun (Y3) | 8 per rule class | the 79 scorable exposed nouns by rule class; no noun enters any object the program uses (its noun side is the weight difference `Δw(n)`) | 40 / 19 / 20 | 8 per class |
| noun (Y1, Y2) | the 79 exposed nouns | the same 79 | 79 | all, fixed |

In draw `b`: Y1 is scored on the drawn cues × 108 frames × 79 nouns; Y2 on the drawn cues × the drawn frames × 79
nouns; Y3 on the drawn nouns over the drawn cues × 108 frames. The class and template counts of the fresh set (6/6/6/6,
6/6/6, 8/8/8) are read from the confirmation file's tokenizer-level metadata only; nothing else of it is read.

**Resampled draws, not disjoint folds.** Because the program fits nothing on the exposed pool, "holding out" a fold
changes nothing in the predictor; what the floors need is the sampling distribution of each statistic for a *new*
set of the confirmation's size. Disjoint folds at those sizes are too few for any quantile rule — 6 for cues (36
pronouns / 6), 2 for frames (14 / 6), 2 for nouns (19 sibilant / 8) — and subsets drawn *without* replacement from
pools this small understate that distribution by the finite-population factor `(N − n)/(N − 1)` (0.62 for frames,
0.61–0.82 for nouns, 0.86–0.90 for cues). The draws are therefore a stratified bootstrap: within each stratum, units
are drawn **with replacement**, which approximates drawing new units from the population the pool represents. The
historical cue cohorts are kept as a descriptive disjoint-fold view (below).

**Draw construction.** `B = 10 000` draws. Within a stratum, units are ordered by a frozen key — cues by token id,
frames by `frame_id`, nouns by `lexical_key` — and slot `i` of stratum `s` in draw `b` takes the unit at index

```
int.from_bytes(sha256(f"021|{variant}|{b}|{s}|{i}".encode("utf-8")).digest()[:8], "big") % n_s
```

with `variant` = `primary` for the floors, `s` written as `cue/<class>`, `frame/<template>` or `noun/<rule class>`,
`b` in `0…9999`, `i` in `0…q_s − 1`. No random generator is involved, so the draws are identical across library
versions; the index arrays are digested into the calibration record.

## The statistics (frozen)

The definitions are Experiment 020's (`pair_statistics`, `noun_statistics`, `_r2`, `_mae`, `_slope`, the even-`n`
median as the mean of the two middle values); `k` uses integer arithmetic, `k = ⌈p·n/100⌉` with `p` ∈ {80, 75, 90}.

| | statistic | direction | `R²`-type | definition |
|---|---|---|---|---|
| S1 | `token_mean_r2` | ≥ | yes | `R²` between per-cue measured and predicted means (each over the cue's pairs × nouns), over the scored cues |
| S2 | `pair_mean_r2` | ≥ | yes | `R²` between per-pair noun-means, over all pairs |
| S3 | `cue_mae_k80` | ≤ | no | the `k`-th smallest per-cue MAE, `k = ⌈80n/100⌉` over the `n` scored cues (20 of 24) |
| S4 | `pooled_mae` | ≤ | no | MAE over all pairs × nouns |
| S5 | `frame_mean_r2` | ≥ | yes | `R²` between per-frame means, over the frames |
| S6 | `frame_r2_k75` | ≥ | yes | the `k`-th largest per-frame `R²` (each over the frame's pairs × nouns), `k = ⌈75n/100⌉` over the `n` valid frames (14 of 18) |
| S7 | `cue_final_r2` | ≥ | yes | flattened `R²` over the cardinal and quantifier pairs × nouns |
| S8 | `coordinated_r2` | ≥ | yes | flattened `R²` over the coordinated-adjective pairs × nouns |
| S9 | `noun_median_r2` | ≥ | yes | the median per-noun `R²` (each over the population's pairs) |
| S10 | `noun_r2_k90` | ≥ | yes | the `k`-th largest per-noun `R²`, `k = ⌈90n/100⌉` over the `n` scorable nouns (22 of 24) |
| S11 | `noun_slope_dev_k90` | ≤ | no | the `k`-th smallest per-noun `|slope − 1|`, `slope = Σ ŷy / Σ y²` |
| S12 | `noun_bias_k90` | ≤ | no | the `k`-th smallest per-noun `|mean(ŷ − y)|` |

A statistic that is undefined in a draw (a zero-variance denominator) takes the worst value (`−∞` for ≥, `+∞` for ≤)
and the draw is kept; at `confirm`, as in 020, an undefined value fails its condition.

## The floor rule (frozen)

With the `B = 10 000` values of statistic `S_j` over the primary draws:

- for a **≥** statistic, `F_j` is the **250th smallest** value (`k = ⌈0.025·B⌉`);
- for a **≤** statistic, `F_j` is the **250th largest** value;
- for an **`R²`-type** statistic, `F_j = max(F_j, 0.0)` — the **zero-skill clamp**: no calibration may certify as a
  prediction a performance that explains none of the fresh units' variance;
- `F_j` is recorded at full float64 precision; nothing is rounded.

Property: if the fresh units are exchangeable with the pools' units within strata, each condition fails with
probability about 0.025, and each outcome (four conditions) with probability at most 0.10 by the union bound — less in
practice, since the four statistics are positively correlated. The empirical joint pass rate of each outcome over the
draws, with the final floors, is recorded; when the clamp binds, that rate may fall below 0.90, and it is recorded, not
corrected.

**What the rule is not.** It is not the exposed values minus a margin, it does not compare with or cap at the 020
floors, it does not treat any condition differently, and it is applied once. A floor may come out above its 020 value
(the error ceilings in particular may tighten) or below it; either stands. Nothing is recalibrated after the floors
are seen; a floor that proves uninformative is reported as such.

## Descriptive records of the calibration (no outcome force)

- Per statistic: the median and the 2.5 / 97.5 % points of its draw distribution; per outcome and over all three, the
  joint pass rate with the 021 floors; the pass rate each **Experiment 020 condition, in its original form and with its
  original floor,** would have had over the same draws — the size of the calibration mismatch that closed 020.
- The validity screen of the 42 frames (each frame's verdict and values).
- The downstream ceiling's flattened `R²` distribution on the Y1 and Y2 draws, so the fresh ceiling can be placed.
- Three sensitivity variants, computed by the same rule and recorded only: (a) Y2 frames drawn from all 108 exposed
  frames (validity-screened, 36 per template); (b) cues drawn from the Experiment 017–019 cohorts only, the same
  candidate-list lineage as the fresh cues (15 / 15 / 12 / 18); (c) the nine historical Experiment 011–019 cue cohorts,
  restricted to the four classes (19–20 cues each), each scored as one disjoint Y1 set.

None of these can replace a primary floor.

## Outcomes (frozen form; the numeric floors come from the calibration record)

- **Y1:** `CONTRAST_PREDICTED_TOKENS` iff `S1 ≥ F1`, `S2 ≥ F2`, `S3 ≤ F3` and `S4 ≤ F4` on the Y1 population; else
  `CONTRAST_NOT_PREDICTED_TOKENS`; `PRECONDITION_FAILED_TOKENS` as in 020.
- **Y2:** `CONTRAST_PREDICTED_FRAMES_CONDITIONAL` iff `S5 ≥ F5`, `S6 ≥ F6`, `S7 ≥ F7` and `S8 ≥ F8` on the valid fresh
  frames; else `CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL`; `PRECONDITION_FAILED_FRAMES` as in 020.
- **Y3:** `NOUN_READOUT_FIXED` iff `S9 ≥ F9`, `S10 ≥ F10`, `S11 ≤ F11` and `S12 ≤ F12` on the 24 fresh nouns over the
  Y1 pairs; else `NOUN_READOUT_NOT_ESTABLISHED` (with 020's wording: this fixed-unembedding predictor was not
  established on this sample); `PRECONDITION_FAILED_NOUNS` as in 020.
- **Outcome label** `Y1 | Y2 | Y3`.

**Reading.** A positive label means the fresh units are predicted within the lower 2.5 % envelope of exposed-like
sets of the same size and strata on every condition of that outcome, with positive skill — prospective
generalization at the stated absolute levels, which the report prints beside each fresh value, the exposed draw
median and the fresh value's percentile within the draw distribution. It is not a fixed standard of "good
prediction": where the exposed program is weak (the coordinated template), the floor is correspondingly low, and the
report says so. A negative label means at least one condition falls outside the envelope — a generalization gap on
that dimension beyond sampling variability, or a systematic difference between the fresh units and their strata.

## Comparators and descriptive additions

The four 020 comparators keep their standing and are recomputed on both fresh sets. Added: the downstream ceiling on
both fresh sets (from the same stage-2 measurements; no extra prompt), the error split
`(1 − R²_Level0) − (1 − R²_ceiling)` against `1 − R²_ceiling`, and the fresh ceiling's percentile within its exposed
draw distribution. If the ceiling stays inside its envelope while Level 0 does not, a failure is placed upstream, in
the inherited `Δ̂x3` — the reading Experiment 020's exposed record suggests.

## Phases and leakage boundaries

| phase | may read | may execute | writes | must not |
|---|---|---|---|---|
| `validate` | frozen inputs, the 011/012/017 locks, the 020 confirmation file, 020's closure record, evidence extract and results-state digests | nothing | nothing | load a model |
| `calibrate` (once) | the above, 020's results state (read only), the weights | exactly the 30,132 keys of 020's exposed ledger | 021's results state (ledger, identity maxima, reproduction gate, table digest, draw digests, calibration record); the table under `outputs/experiment-021/` | run a key outside 020's ledger or inside the manifest; build a fresh-noun quantity; write 020's state |
| *(commit)* | | | the calibration record installed byte-identical in `experiments/021-…/`; **stop for review of the floors** | |
| `lock` | the committed calibration record, 020's locked reference states, the locks, the weights | nothing (no forward pass, as in 020) | the candidate lock: the 2592 Y1 rows with the fresh-noun columns and every comparator column, `F1…F12`, the calibration record's digest, the confirmation digest and manifest; the predictions artifact | read any fresh measured quantity |
| `confirm` (once) | the installed lock, the confirmation file, the weights | stage 1: S1-REF and S1-VALIDITY; after the barrier, stage 2: S2-TARGET of the Y1 block and of the valid fresh frames | 020's stage-1 record and scoring shape, with the 021 floors and the descriptive ceiling | execute a key twice or outside the manifest; score a fresh noun into Y1/Y2 |
| `report` | the results state | nothing | the report | |

Before `lock` and before `confirm`, both Experiment 020's ledger and 021's own are asserted to contain no confirmation
key.

## Incidents and preconditions

As in Experiment 020: scientific invalidity (an invalid fresh frame, an unscored cue, a non-scorable noun) feeds the
`PRECONDITION_FAILED_*` counts and never stops a phase; implementation and protocol failures are incidents that stop
the phase, are recorded with their commit and permit no rerun at that commit. `calibrate` adds four incident
conditions: a re-captured reference state beyond `1e-9`, a reproduction-gate difference beyond `1e-9`, a key outside
020's exposed ledger or inside the manifest, and a 020 digest mismatch before or after the phase.

## Interpretation limits

- The floors are relative to the program's own exposed performance. A positive label certifies the absence of a
  generalization gap beyond sampling variability, at the absolute levels printed in the report.
- The program's structural choices were made after exposed data were seen, and this rule was written after the
  corrected exposed aggregates were known (the per-draw statistics were not). Exposed-like draws may therefore be
  somewhat optimistic, which makes the floors, if anything, harder to meet.
- The bootstrap represents the unit populations by 175 cues, 42 frames and 79 nouns; heterogeneity absent from them is
  absent from the floors. Systematic lexical novelty in the fresh units is exactly what a negative can reveal, but
  the strata cannot adjust for it.
- At `confirm`, fewer than 18 valid fresh frames make the Y2 statistics noisier than the calibration assumed.
- Y2 remains conditional on each fresh frame's stage-1 reference state; nothing predicts a frame state from text.
- The claim concerns `Δc` at `p_t` for single-token regular-plural noun pairs under the frozen templates; one checkpoint,
  three templates, four cue classes, three noun rule classes, 18 new frames.

## Decisions this proposal makes that the review did not

1. **Re-materializing the exposed table** (≈ 70 min of exposed-only forwards, verified against 020's record at
   `1e-9`) — the one departure from "no model run": ten of the twelve statistics need per-pair data 020 did not
   persist.
2. **Stratified bootstrap draws** in place of disjoint folds, with the cue cohorts kept as a descriptive disjoint view.
3. **Provenance-matched pools**: the four fresh cue classes from the 011–019 cohorts; for Y2 only the 42 frames that
   entered no inherited fit, validity-screened; all 79 exposed nouns.
4. **`α = 0.025` per condition** (≤ 0.10 per outcome), `B = 10 000`, and the zero-skill clamp on `R²`-type floors.
5. **Order-statistic form** of the share conditions, and `|slope − 1|` for the slope band.
6. **The downstream ceiling on the fresh sets**, which 020's design promised and its code omitted.

## Implementation boundary (for the plan, after approval)

`src/neural_decompiler/readout_calibration.py` (the re-materialization, the reproduction gate, the draws, the
statistics, the floor rule, the 021 scoring, the stage-2 ceiling, the lock), `experiments/021-corrected-readout-confirmation/run.py`
with phases `validate`, `calibrate`, `lock`, `confirm`, `report`, and `tests/test_readout_calibration.py` +
`tests/test_experiment_021_runner.py`, with `CURRENT_EXPERIMENT = "021"`. `readout_decompilation.py` is not edited.
Unit tests cover, on synthetic tables: the order-statistic equivalence with 020's share conditions for every `n`
(including the integer `k`), the floor rule and its clamp, the draw index derivation and its digest, and the
reproduction gate's refusal on a planted `1e-8` difference.

## Stopping condition

Stop after this proposal. After approval: plan → implementation and tests (tier B) → independent review → tier C on
the clean gated commit → `calibrate` once → install and commit the calibration record → **stop for review of the
floors** → `lock` (no forward pass) → the user's lock commit and the reviewer's sign-off → `confirm` once → report →
closure.

## Revision history

- **Revision 1** (2026-09-22): proposal, written after Experiment 020's closure at `7f4a4f8`. No model run, no fresh
  quantity and no per-draw statistic was computed for it.

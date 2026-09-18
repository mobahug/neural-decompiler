# Experiment 006: Low-Rank Decompilation of the Cue Encoding — Prospective Prediction for Arbitrary Cue Tokens

**Date:** 2026-09-18

**Status:** Draft for review. Not approved. No Experiment 006 directory,
confirmation manifest, lock, or model run exists. Experiment 005 is closed and
is not amended by this document.

**Scope:** the count-cued singular/plural contrast in pinned
`EleutherAI/pythia-70m-deduped`, restricted to the circuit Experiment 005
localized (E = `L00.MLP` at the cue position, T = `L03.H04` at the noun
position, R = `L04.MLP` and `L05.MLP` at the noun position), with every
Experiment 005 prompt, noun, and cue token treated as exploratory input and a
genuinely fresh confirmation set frozen before any Experiment 006 model run.

## Purpose

Experiment 005 ended with two recorded negatives — `NO_COMPACT_MECHANISM`
(protocol v1) and `BEHAVIOR_NOT_REPLICATED` (protocol v2) — and one sharp
positive lead. Across development, holdout, reserve nouns, and six unseen
frames, the cue-induced computation was stable: exact replacement of the k = 3
set recovered 0.977–0.985 of the contrast shift, isolating it retained
0.78–0.82, the transport head carried about 0.82 of the coordinated shift, and
the E → T → R chain mediated 0.99 of the readout change. What failed were
threshold-crossing gates (the reserve nouns often did not cross the
singular/plural decision boundary although the cue shift itself was present in
120/120 pairs) and the *one-dimensional* program: `L00.MLP` replacement
reproduced the behavioral shift of every unseen cue word almost exactly, but a
single scalar axis fitted on four cue tokens could not predict those shifts
(`all`, `some`, `both`, `few`, `many` moved the contrast by −3.8 to −4.8 nats
with scalar values within ±0.48; `every` moved it by about zero with −0.61).

Experiment 006 therefore asks the question Experiment 005 exposed:

> Can the token-local representation that `L00.MLP` writes for an arbitrary
> cue token be decompiled into a small multidimensional variable — a rank-`r`
> projection with `r ≤ 4` — followed by frozen low-rank transport and readout
> maps and the model's exact final LayerNorm, such that the resulting program
> prospectively predicts how unseen cue tokens, in unseen frames, with unseen
> nouns, are transported through `L03.H04` and converted by the layer-4 and
> layer-5 MLPs into noun-number logit shifts?

The circuit itself is re-tested with split-invariant criteria, but the
experiment's primary deliverable is the program.

## What Experiment 005 established and what is now exposed

- Circuit (exploratory and calibration evidence; claim C002 at `LOCALIZED`):
  `L00.MLP` at the cue position is a pure token function and carries 0.535 of
  the transport-input number axis; `L03.H04` attends to the cue (0.747) and
  alone recovers 0.816 of the coordinated shift with a blocked fraction of
  0.865; `L04.MLP` and `L05.MLP` read the transported signal (`m_R` 0.99).
- Behavior: the clean model's decision-boundary crossings depend on the noun
  set (development 87/120 flips, holdout 98/120, reserve 75/120, new frames
  65/120) while the cue-induced shift `d_full` was positive in 120/120 pairs
  on every set; 44 of 45 reserve failures were plural conditions in which the
  number variable still matched the cue.
- Cue words: the E-patch (replace `L00.MLP` at the cue position with the
  word's own output) reproduced each word's behavioral shift within 0.35
  nats for all twelve extension words; the scalar lexicon did not predict
  the magnitudes; `every` and `a` produced near-zero shifts, `the` −2.6 nats,
  numerals −4.5 nats, and the quantifiers `all`, `some`, `both`, `few`,
  `many` −3.8 to −4.8 nats.
- Exposed data (all exploratory for Experiment 006): the twelve manifest
  prompts, the six extension frames, the sixteen cue tokens (`one`, `two`,
  `each`, `several`, `a`, `the`, `three`, `four`, `five`, `ten`, `many`,
  `few`, `some`, `all`, `both`, `every`), and all sixty nouns.
- Never executed: the eight remaining tokens of the Experiment 005 cue-word
  list (`any`, `no`, `another`, `single`, `multiple`, `numerous`, `twelve`,
  `hundred`).

## Fixed coordinates

- Model, revision, runtime, instrumentation, contrast `c(x)`, `d_full`,
  aligned patched shift, recovery, denominator floors, pair-centered
  neutralization, freeze, and the exact final LayerNorm (population variance,
  `ε = 1e-5`) are inherited unchanged from the Experiment 005 design
  (revision 5). All quantities are computed within the instrumented forward
  path, whose numerics differ from the plain forward by up to ~5e-3 nats.
- The circuit is fixed a priori: E = `L00.MLP` (cue position), T = `L03.H04`
  (noun position), R = {`L04.MLP`, `L05.MLP`} (noun position). Experiment 006
  performs no component search. Runtime seed `20260916`; control and bootstrap
  seed `20260919`.

## Data: exploratory pool and frozen confirmation set

### Exploratory pool

Everything Experiment 005 executed: 12 manifest prompts + 12 extension-frame
prompts (four cue tokens), 72 cue-word prompts (twelve words in the six
manifest frames), and the 60 nouns. Every fit, rank selection, calibration,
and tolerance in Experiment 006 uses only this pool.

### Frozen confirmation set — `experiments/006-low-rank-cue-decompilation/confirmation-v1.json`

Built by tokenizer-only rules and committed before the first Experiment 006
model run; never executed before the `confirm` phase. It contains:

1. **Fresh nouns:** the first twenty tokenizer-eligible entries (singular and
   regular plural both single tokens with a leading space) of a frozen ordered
   pool disjoint from all sixty Experiment 005 nouns, stratified 10 simple, 5
   sibilant-`es`, 5 consonant-`y`. Candidate pool, in order: simple — `stone`,
   `field`, `road`, `door`, `window`, `bridge`, `tower`, `planet`, `letter`,
   `wheel`, `candle`, `branch`; sibilant — `patch`, `coach`, `church`, `tax`,
   `boss`, `lens`, `arch`, `flash`; consonant-`y` — `body`, `copy`, `duty`,
   `spy`, `study`, `memory`, `theory`, `entry`, `colony`, `galaxy`. Every
   entry was checked to be tokenizer-eligible while drafting; disjointness
   from the sixty Experiment 005 nouns is enforced by the builder.
2. **Fresh frames:** two per template, fixed literals: cardinal — `The crate
   holds {cue}`, `The gallery shows {cue}`; quantifier — `The index lists
   {cue}`, `The archive keeps {cue}`; coordinated-adjective — `Ravi and Elena
   sorted {cue} plain`, `Nora and Felix stacked {cue} heavy`. Cue and adjective
   must be single tokens; positions are derived per prompt.
3. **Fresh cue tokens:** the eight never-executed Experiment 005 tokens plus
   the first sixteen tokenizer-eligible entries of a frozen ordered list, for
   twenty-four in total, grouped for reporting only: numerals — `six`, `seven`,
   `eight`, `nine`, `twenty`, `fifty`, `thousand`; quantity words — `dozen`,
   `countless`, `various`, `fewer`, `more`, `most`, `enough`, `plenty`;
   determiners — `this`, `that`, `these`, `those`, `either`, `neither`, `an`,
   `my`, `our`, `their`; non-cue controls — `big`, `red`, `old`, `fresh`,
   `only`, `very`. The program's prediction is unconditional: it is made for
   every token in the cue slot, control or not.

Confirmation prompts are every fresh cue token in every fresh frame (24 × 6 =
144 prompts) plus the original four cue tokens in the fresh frames (12
prompts), all scored with the twenty fresh nouns. Fresh nouns are also scored
on the twelve manifest prompts for the noun-generalization families.

## The program: low-rank decompilation of E

The decompiled computation is a weight-only program with these frozen parts:

- `E(w) = MLP_0(ln2_0(embed[w]))`, the exact token-local output of the layer-0
  MLP for token `w` (weights only, as in Experiment 005).
- `z(w) = U_rᵀ (E(w) − μ_E) ∈ ℝ^r`, with `μ_E` the mean of `E` over the
  exposed cue tokens and `U_r` the top-`r` left singular vectors of the
  centered matrix of exposed `E(w)` vectors; `r ∈ {1, 2, 3, 4}` is selected as
  below and frozen.
- Transport and readout as one linear map per template from `z` to the
  residual increment at the noun position: `δ_T(w) = V_T z(w)`, with `V_T ∈
  ℝ^{d_model × r}` fitted by least squares to the measured increment of the
  summed T ∪ R outputs (coordinated frames) or E ∪ T ∪ R outputs (cue-final
  frames) at the noun position, relative to the frame context, over the
  exposed cue tokens and frames of that template.
- Readout: `ĉ_N(w; frame) = −u_N · LN(ρ + δ_T(w))`, exact LayerNorm; `ρ` is
  the frozen context residual of a known frame or the template mean for a
  fresh frame; shifts are always relative to the template's singular reference
  cue (`one` or `each`).
- E-patch prediction: replacing `E` alone in the reference prompt by `E(w)`
  is predicted to shift the contrast by `−u_N · [LN(ρ + V_T z(w)) − LN(ρ +
  V_T z(ref))]`.

The rank-1 program of Experiment 005 is the special case `r = 1` with `U_1`
the mean-difference direction; Experiment 006 reports it as a baseline.

### Rank selection (Tier A, exploratory, frozen before confirmation)

Leave-one-cue-out over the sixteen exposed cue tokens: for each `r` and each
held-out token, fit `μ_E`, `U_r`, and `V_T` on the other fifteen (all exposed
frames) and predict the held-out token's E-patch shifts and behavioral shifts
in every exposed frame. Score each `r` by the mean absolute error of the
E-patch prediction over held-out tokens, frames, and nouns. Select the
smallest `r` whose error is within one standard error of the best `r`; ties
resolve to the smaller rank. The selection rule, the scores for every `r`,
and the chosen `r` are recorded; the final parameters are then fitted on all
sixteen tokens and frozen in the lock before any fresh token is executed.

## Split-invariant criteria

Experiment 005 showed that absolute threshold-crossing counts depend on which
nouns a split contains. Experiment 006 replaces them:

- **Cue-effect gate (precondition, replaces B1):** on the fresh nouns and the
  original cue pairs, `d_full > 0` in at least 114/120 pairs on the manifest
  frames and at least 108/120 on the fresh frames, with every template
  denominator above the floors. Absolute correctness (primary accuracy and
  flips) is reported descriptively and gates nothing.
- **Fidelity instead of label retention (P3):** the isolated circuit must
  reproduce the clean model's contrasts, including its mistakes:
  `sign(c_iso(x)) = sign(c_clean(x))` in at least 108/120 conditions per cue
  and the correlation between `c_iso` and `c_clean` across conditions at
  least 0.90, in addition to `F ≥ 0.50` overall and `≥ 0.40` per template.
- **Circuit families kept:** P1 (recovery 0.70 / 0.60 / 0.60), P4 (E-only at
  the cue position, coordinated, ≥ 0.50), P5 (T alone ≥ 0.50; blocked fraction
  ≥ 0.50), P7 (cross-frame control), P9 (chain: `m_T`, `m_R` ≥ 0.50,
  `m_R|T ≤ 0.5·m_R`), each evaluated on the fresh nouns over the manifest
  frames and, separately, over the fresh frames. P2, P6, and P8 are dropped
  as already established for this circuit; their Experiment 005 values are
  cited.

## Prediction families (decompilation axis)

Floors are fixed here; tolerances come from the leave-one-cue-out residuals
(τ = max(0.5 nats, 3 × the LOCO root-mean-square error of the selected rank)).

| ID | Family | Prompts | Floor |
|---|---|---|---|
| Y1 | E-patch prediction for fresh cue tokens | 24 tokens × 6 fresh frames × 20 nouns; replace E alone in the frame's reference prompt with `E(w)` | Spearman ≥ 0.80 between predicted and measured per-token mean shifts; mean absolute error ≤ τ; sign agreement in ≥ 5/6 frames for every token whose predicted magnitude is ≥ 1.0 nat |
| Y2 | Behavioral shift prediction for fresh cue tokens | the same prompts, no intervention | Spearman ≥ 0.70; mean absolute error ≤ 1.5 τ (the difference from Y1 is the non-E route, reported as residual) |
| Y3 | Frame invariance of the pair shift | four original cues × 6 fresh frames × 20 nouns | predicted mean `d_full` per template within τ of the measured mean; `d_full > 0` in ≥ 108/120 pairs |
| Y4 | Rank-1 baseline comparison | Y1 and Y2 recomputed with the `r = 1` program | reported; the selected rank must beat the rank-1 baseline on Y1's error by at least 20% or the selected rank is 1 |
| Y5 | Residual accounting | all | reported, never gated |

Y1–Y3 are primary; a `PROGRAM_PASS` requires all three floors. Bands: each
per-token prediction carries `± τ`; the program hits its bands when at least
18 of the 24 tokens fall inside in at least 5 of 6 frames on Y1.

## Tiers and execution boundary

- **Tier A (exploratory):** on the exploratory pool, verify the fixed circuit
  once (P1, P3-fidelity, P4, P5, P9 on the 60 nouns and the 12 frames), fit
  the program, run the rank selection, and record the rank-1 baseline. No
  fresh noun, frame, or cue token is executed.
- **Tier B (calibration):** LOCO residuals of the selected rank define τ and
  the per-token bands; the fixed-circuit families' bands are the Experiment
  005 calibration bands carried over, because the circuit is unchanged.
- **Lock:** `experiments/006-low-rank-cue-decompilation/preregistration-lock.json`
  records the confirmation-set digest, the circuit, `r`, `μ_E`, `U_r`, every
  `V_T`, the context residuals, the LOCO scores, τ, and the program's
  predictions for every confirmation prompt computed before execution; it is
  installed and committed by hand.
- **Tier C (confirmation, once):** the fresh set is executed exactly once
  after the lock validates (digests, unchanged scientific paths, clean tree,
  execution ledger without any confirmation prompt or fresh noun).

## Outcome rule

- Precondition: the cue-effect gate on fresh nouns; failure → `CUE_EFFECT_NOT_REPLICATED`.
- Circuit axis: `CIRCUIT_PASS` if P1, P3-fidelity, P4, P5, P9 pass on the
  fresh nouns over the manifest frames and over the fresh frames; the two
  evaluations are reported separately and both are required.
- Program axis: `PROGRAM_PASS` if Y1–Y3 pass.
- `DECOMPILED` — both axes pass and the program hits its bands.
- `DECOMPILED_MISCALIBRATED` — both axes pass; bands missed.
- `CIRCUIT_ONLY` — circuit passes, program fails (the failing Y families name where: encoding rank, transport/readout linearity, or frame context).
- `CIRCUIT_NOT_GENERALIZED` — circuit passes on manifest frames but not on fresh frames.
- `NOT_SUPPORTED` — circuit fails on the manifest frames.

No scientific retries; software defects follow the incident rule.

## Claims and prior work

A `DECOMPILED` outcome would support a new claim (C003) at `CAUSAL_EVIDENCE`
with Level 6 gates reviewed individually; `CIRCUIT_ONLY` would promote C002's
circuit evidence to `CAUSAL_EVIDENCE` if the fresh-frame families pass, and
leave the program at the identified encoding failure. The prior-work boundary
of Experiment 005 applies unchanged; the new element is a prospective,
weight-only, low-rank account of how an arbitrary token in the cue slot is
encoded and read out, which Experiment 005's audit found absent from the close
literature.

## Interpretation limits

- The circuit is inherited, not rediscovered; Experiment 006 cannot find a
  different mechanism.
- The program is linear from `z` to the residual increment; nonlinearity in
  the transport head or the readout MLPs appears as residual, and the rank cap
  of 4 is a compactness choice, not a claim about the representation's true
  dimension.
- Cue-slot tokens that are not quantity words test the program's
  unconditional predictions; their behavioral meaning for the model is not
  interpreted beyond the measured shift.
- Nothing generalizes beyond the pinned checkpoint, the three template
  families, and single-token regular nouns.

## Minimal implementation boundary

Reuse the Experiment 005 module (`plural_mechanism.py`) for capture, exact
replacement, contrasts, recovery, isolation, chain measurements, exact
LayerNorm, weight export, lock validation, and reporting; add one module for
the low-rank program (rank selection, fitting, predictions) that, like
`mechanism_program.py`, runs without the network; one confirmation-set builder
(tokenizer-only); one runner with phases `validate`, `freeze-confirmation`,
`explore`, `calibrate`, `lock`, `confirm`, `report`; focused offline tests.
No new hook kinds, no component search, no other model, and no prompts outside
the exploratory pool and the frozen confirmation set.

## Approval and stopping condition

This design is complete when it has been reviewed and explicitly approved.
Only then are the plan and the experiment directory created; the
confirmation set is frozen before `explore`; Tier C runs once after the lock
commit. The experiment ends after the single confirmation, the outcome rule,
the claim review, and the evidence copies.

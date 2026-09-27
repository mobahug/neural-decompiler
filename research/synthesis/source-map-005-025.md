# Source map: Experiments 005–025

**Step 1 of the [synthesis](README.md).** This is a factual inventory of what each experiment asked, froze, ran and
found. It interprets nothing, ranks nothing and proposes no new experiment. The synthesis (step 2) is built on it only
after it has been checked.

## How to read this map

- **Sources.** The committed records at `d7c5e6279e171ef0d8d8bc8aa4b8b512356d962e`, the closure of Experiment 025:
  - each experiment's README, evidence directory, preregistration and lock files;
  - the design and plan documents they link;
  - the claim and anomaly registries;
  - git history.

  Where a summary here disagrees with its source, the source wins.
- **Citations.** `(023:142)` means line 142 of `experiments/023-*/README.md` at `d7c5e62`. `(path:L)` means line L of
  that file at `d7c5e62`. Later edits (the 025 correction and the root README entries made alongside this map) do
  not move these citations.
- **Quotes.** Text in curly quotes is verbatim, with Markdown emphasis and block-quote markers dropped and line
  breaks joined. Everything else is paraphrase.
- **Checks.** `checks/check_source_map.py` checks every quote against its cited lines and parses every citation. It
  also checks that every cited file and line, commit hash and outcome label exists. It does not check paraphrases,
  numbers outside quotes or classifications; the independent review sampled those
  (`checks/independent-review-2026-09-27.md`).
- **Latest authoritative reading.** Where a conclusion was corrected, narrowed or restated after it was first
  written, this map gives the latest authoritative wording. The original wording and the correction are recorded
  under [Record caveats](#record-caveats).
- **The rule for the synthesis.** No synthesis sentence may use stronger epistemic language than the strongest
  source record supporting it.
  - Designs, plans and the root README never raise a finding above the strength its experiment README records.
  - The evidence-status categories below make that strength explicit.

## Evidence-status categories

Each finding carries exactly one of these five categories. The methodology itself defines only exploratory and
confirmatory results (docs/RESEARCH_METHODOLOGY.md:165–171). The five categories refine that split for this map, and
are proposed for review.

| category | meaning in this map |
|---|---|
| **prospective confirmation** | a criterion committed in version control before the data, which passed in the single preregistered `confirm` run on never-executed data |
| **locked prediction** | a prediction committed before the data and recorded against the measurement, but with no outcome-bearing criterion attached (for example the per-cue calls of 011–014) |
| **exploratory** | formed or measured on data already inspected: `explore`/Tier A phases, discovery-only runs, design spikes, post-hoc diagnostics |
| **descriptive / post hoc** | reported beside a preregistered result with no outcome force, or computed after the result was written |
| **unestablished** | a preregistered criterion failed or was not met, or the confirmatory test was never run |

**Tie-break.** The data a quantity was computed on decides between the two middle categories:
- on the confirm run's fresh data, after the result was written: *descriptive / post hoc*;
- on data already spent before the confirm (explore phases, spikes, diagnostics of spent data): *exploratory*.

Three qualifiers travel with these categories.
- **Conditional.** From 013 on, every result on new frames is prospective but conditional on those frames' measured
  reference states, whether or not its label says so. “Y2 is therefore a prospective prediction of unseen cues on
  previously untested frames, conditional on the frame's reference/base state — not a prediction on completely unseen
  frames.” (docs/superpowers/specs/2026-09-19-experiment-013-attention-paths-frozen-pattern-design.md:253–255) This
  covers:
  - the `*_FRAMES_CONDITIONAL` labels;
  - 022's and 023's Y2 conditions, each frame “at its own stage-1 reference state”
    (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:254;
    docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md:214);
  - the new-frame share of statistics pooled over both sets: 013–016 Y3, 018 Y3, and 019 Y3–Y5.
- **Given the measured input.** The head read-outs (009 Y1, 011 Y2) model the head's output from its measured input:
  “P1–P3 decompose the head's output change given its *measured* input change; they say how the head transforms what
  it receives, not why upstream layers place each token where they do.” (docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:223–224)
- **Label-level versus criterion-level.** A criterion can pass while the frozen outcome rule still assigns a negative
  label, as in 007. The map reports both levels, and the synthesis must not lift a passing criterion into the label's
  place.

## Overview

| exp | subject | kind of test, as recorded | frozen outcome (verbatim) | status of the outcome | closed in |
|---|---|---|---|---|---|
| screen | behavior candidate screen | candidate selection; the compactness probe is “an exploratory triage heuristic” (screening/behavior-candidates/report-2026-09-17.md:7) | `ONE_PROPOSED_TARGET` (`regular-plural`) | exploratory | `1f043d9` |
| 005 | count-cued noun number: prospective mechanism | Tier A discovery on development data, then one confirm (protocol v2) | v1 `NO_COMPACT_MECHANISM`; v2 `BEHAVIOR_NOT_REPLICATED` | v1 exploratory; v2 unestablished (behavioral precondition failed) | `13600cd`, `6bc2d6a` |
| 006 | low-rank decompilation of the cue encoding | Tier A only; no lock | `QUALITY_GATE_FAILED` | exploratory; the confirmatory test was never run | `10c8faf` |
| 007 | response-supervised low-rank cue subspace | prospective, one confirm | `PROGRAM_NOT_SUPPORTED` (Y1 failed) | unestablished; the precondition, Y2 and Y3 passed as criteria | `6340a4c` |
| 008 | where the number signal of `this`-like cues is suppressed | “Discovery-only” (008:7) | `MIXED` | exploratory | `40265b5` |
| 009 | the transport rule of `L03.H04` | prospective, one confirm | `HEAD_MECHANISM_CONFIRMED_P1 \| TRANSPORT_RULE_FAILED \| NOT_LOCKED` | mechanism (given the measured head input): prospective confirmation; transport rule: unestablished; contrast rule: never locked | `27083c6` |
| 010 | how the head-readable number signal is assembled | “Discovery-only exact attribution” (010:7) | `MIXED` | exploratory | `16c6491` |
| 011 | encoding read predicts `L03.H04` transport | “Prospective and zero-parameter” (011:7) | `ENCODING_READ_PREDICTS_TRANSPORT \| HEAD_P1_REPLICATED` | prospective confirmation (both) | `0358c8e` |
| 012 | layers-1–2 correction as a token-local MLP computation | “Prospective and zero-parameter” (012:7) | `LAYER_CORRECTION_TOKEN_LOCAL_MLP \| COMPOSITE_PREDICTS_P1` | prospective confirmation (both); claim restated after closure | `fc442f6` (`1b1edff`) |
| 013 | what layers 1–2's attention adds | “Prospective, zero-parameter, residual-predicting” (013:7) | `RESIDUAL_PREDICTED_TOKENS \| RESIDUAL_PREDICTED_FRAMES_CONDITIONAL \| ATTRIBUTION_PREDICTED` | prospective confirmation (Y2 and the new-frame share of Y3 conditional) | `9b656e2` |
| 014 | what drives block-2 neuron 1987 | “Prospective, zero-parameter, feature-reconstructing” (014:7) | `NEURON_FEATURE_PREDICTED_TOKENS \| NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL \| AXIS_ONLY_REJECTED` | prospective confirmation (Y2 conditional); the axis-only alternative rejected on both sets pooled | `56a0fc5` |
| 015 | the cue-induced attention-pattern change | “Prospective, zero-parameter, program-completing” (015:7) | `PATTERN_CHANGE_PREDICTED_TOKENS \| PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL \| AXIS_ONLY_REJECTED` | Y1 prospective confirmation; Y2 unestablished; the axis-only alternative rejected on both sets pooled | `08f4ca4` |
| 016 | the frame-conditioned token-local Q/K model | “Prospective, zero-parameter, decomposition-completing” (016:7) | `FRAME_CHANNELS_PREDICTED_TOKENS \| FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL \| SCALE_ONLY_REJECTED` | prospective confirmation (Y2 conditional); the scale-only alternative rejected on both sets pooled | `de61ead` |
| 017 | the transport head's own pattern change | prospective, one two-stage confirm | `HEAD_PATTERN_PREDICTED_TOKENS \| HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL \| FROZEN_PATTERN_NOT_REJECTED` | Y1, Y2 (conditional) prospective confirmation; Y3: the necessity of the pattern term unestablished | `c583405` |
| 018 | block 2's operating-point dependence (channel D) | prospective, one two-stage confirm | `CHANNEL_D_CONCENTRATED_TOKENS \| CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL \| SINGLE_NEURON_REJECTED \| PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES` | Y1 prospective confirmation; Y2 unestablished (no-harm guard, one frame); single-neuron account rejected (pooled, pair-weighted); Y4 unestablished | `5132174` |
| 019 | frame-conditioned routing of channel D | prospective, one two-stage confirm | `ROUTING_PREDICTED_TOKENS \| ROUTING_PREDICTED_FRAMES_CONDITIONAL \| OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT \| TEMPLATE_FAMILY_INSUFFICIENT \| MEMBERSHIP_PREDICTED` | prospective confirmation (Y2 and the new-frame parts of Y3–Y5 conditional); aggregate, not per frame | `a0d73c8` |
| 020 | readout decompilation to the logit contrast | Tier A only; the confirm intentionally not run | none assigned | exploratory; the confirmatory test was never run | `7f4a4f8` (erratum `2320345`) |
| 021 | corrected readout-decompilation confirmation | prospective, calibrated floors, one confirm | `CONTRAST_PREDICTED_TOKENS \| CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL \| NOUN_READOUT_FIXED` | Y1, Y3 prospective confirmation; Y2 unestablished (a zero-clamped floor; the erratum names a wiring defect); all three labels test 020's `p_t`-wired program | `86f7f8c` (erratum `2320345`) |
| 022 | localization of the upstream `Δx3` error | prospective, one confirm; eight conditions | all eight `PASS`; no aggregate label | prospective confirmation, per condition (Y2 conditional) | `0818c4c` |
| 023 | prospective block-0 completion | prospective, one confirm; four conditions | three `PASS`, Y1/cue_final `ENVELOPE_ONLY_FAILURE`; no aggregate label | three prospective confirmations (Y2 conditional); one unestablished at its envelope (guard met) | `b45a646` |
| 023-d | post-closure diagnostic of 023's spent data | post hoc, recorded only in 024's design | none | exploratory | — |
| 024 | an operational nounness score predicts readout error | “predictive, not causal” (024:373) | `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS` | prospective confirmation of a predictive association; the guard's formal scope is two simple alternatives | `eb5f6e8` |
| 025 | nounness-direction intervention on the cue embedding | an intervention: a norm-preserving rotation of the cue embedding | `CAUSAL_EFFECT_NOT_ESTABLISHED` | unestablished; the end of the sequence | `d7c5e62` |

## Evidence status, by experiment

Each block sorts one experiment's findings into the five categories. A category that is absent has no finding. Where a
criterion passed but the frozen outcome rule still assigns a negative label, the block says so at both levels. The
per-experiment records below give the numbers and the full citations.

- **Screen.** Exploratory: the choice of `regular-plural` and the compactness probe
  (screening/behavior-candidates/report-2026-09-17.md:7). The behavioral gates were frozen in protocol v1 before the
  run; the map keeps the whole screen exploratory, the safe direction.
- **005.**
  - Prospective confirmation: none. “By the preregistered rule no other family is confirmatory.” (005:128–129)
  - Exploratory:
    - the Tier A circuit, E `L00.MLP`, T `L03.H04`, R `L04.MLP` + `L05.MLP`, with recovery 0.982 and isolation 0.783;
    - the v1 program-floor failure (005:88–101).

    C002 records the same origin: “Origin: exploratory (mechanism found on development data in Experiment 005 Tier A”
    (research/claims/C002-count-cued-noun-number-circuit.md:9). Protocol v2's rule amendment came after Tier A (see the
    005 record).
  - Descriptive / post hoc: the reserve-noun and new-frame P and X values, recorded as non-confirmatory. Several of them
    missed their floors: P3's retention count (71/120), X2's P3 retention (62/120), X3, X4, and B2's template bands
    (005:130–140).
  - Unestablished: protocol v2's outcome, `BEHAVIOR_NOT_REPLICATED`: the behavioral precondition B1 failed (005:125–127).
- **006.**
  - Exploratory:
    - the pre-lock quality gate: Spearman 0.800 passed, normalized RMSE 0.623 failed;
    - the post-hoc rank diagnostic;
    - the carried-forward circuit result, P3's isolation fidelity: 708/708 paired signs, but a correlation of 0.756
      against the 0.90 criterion (006:58–59, 67–89).
  - Unestablished: every confirmatory family. No lock could be written, so none was tested (006:71).
- **007.**
  - Prospective confirmation, at criterion level only: the precondition, Y1's MAE, Y2 and Y3 passed (007:85–93).
  - Exploratory: the Tier A quality gate and the rank selection (007:125–130).
  - Descriptive / post hoc, reported and not in the outcome (007:95–100):
    - the circuit families: P1, P4, P5, P7, P8 and P9 passed, and P3's correlation floor failed on both fresh sets;
    - the `Ridge-full`, `E005-scalar` and `PCA-006` baselines.
  - Unestablished, at label level: `PROGRAM_NOT_SUPPORTED`. Y1's Spearman 0.785 missed 0.80, and the confident-token
    sign failed for `another` and `this` (007:87–94).
- **008.** Exploratory: everything. “Discovery-only: no lock, no confirmation, no claim promotion.” (008:7)
- **009.**
  - Prospective confirmation: the precondition, and Y1, the head's P1 read-out given its measured input (Spearman
    0.957, MAE 0.066), on the 23 fresh cues in the 6 fresh frames (009:61–64, 148–149).
  - Exploratory: the Tier A gates and the fitted rules, including the 40 exposed cues and 18 exposed frames (009:93–124).
  - Descriptive / post hoc: the non-locked P2 and P3 levels on the fresh pairs, and the non-additivity stages
    (009:64–65, 80–81).
  - Unestablished: Y2, the transport rule from `E` (009:69–75); Y3, never locked (009:77).
- **010.** Exploratory: everything, including the `MIXED` summary, the relay-bound failure, the encoding-borne
  opposition and the `g_E` candidate (010:7–8, 62–67, 85–96, 110–112).
- **011.**
  - Prospective confirmation: Y1, the encoding read (Spearman 0.848, MAE 0.077), and Y2, P1 replicated given the
    measured head input (0.974, 0.076) (011:75–76, 87–89).
  - Locked prediction: the per-token predictions, for example `an` predicted 0.18 and measured 0.30 (011:76–83).
  - Descriptive / post hoc: Experiment 009's fitted rank-1 rule on the same token means, which “would also have cleared
    the Y1 floors on this set” (011:91–92); and the residual's structure (011:83–86).
  - Exploratory: the Tier A calibration (011:116–121).
- **012.**
  - Prospective confirmation: Y1 (0.864, 0.061, R² 0.795) and Y2 (0.922, 0.062) (012:81–83, 91–93).
  - Locked prediction: the per-token calls (012:83–89). `thy` shows their limit: its total was predicted exactly, but
    the split between its MLP and attention parts was wrong (012:99–102).
  - Descriptive / post hoc: the pair level and the block-2 neuron ledger (012:89–90, 104–107).
  - Exploratory: the Tier A calibration (012:144–158).
- **013.**
  - Prospective confirmation: Y1; Y2 (conditional); Y3, the attribution, pooled over both sets, with the new-frame share
    conditional (013:88–108).
  - Locked prediction: the per-token committed calls (013:91–97).
  - Descriptive / post hoc: the compensation cases and the ladder (013:109–115).
  - Exploratory: the Tier A calibration (013:141–157).
- **014.**
  - Prospective confirmation: Y1; Y2 (conditional); Y3, the rejection of the axis-only alternative on both sets pooled
    (014:96–105).
  - Locked prediction: the strong firings, the eighteen vetoes and the borderline cases (014:107–114).
  - Descriptive / post hoc: the firing set, “interpreted only now” (014:125).
  - Exploratory: Tier A (014:158–176).
- **015.**
  - Prospective confirmation: Y1; Y3, the rejection of the axis-only alternative on both sets pooled (015:105–107,
    123–124).
  - At criterion level, Y2's Spearman (0.857) and layer-1 entry R² passed (015:109–111).
  - Descriptive / post hoc: the per-frame Y2 values; the five-frame R² 0.82, which “changes nothing” (015:121–122); the
    comparator margins; and the ladder (015:111–140).
  - Exploratory: Tier A (015:83–86).
  - Unestablished: Y2 (conditional). The token-mean R² 0.490 and the layer-2 entry R² 0.423 missed 0.50 (015:109–111).
- **016.**
  - Prospective confirmation: Y1; Y2 (conditional), with the per-frame guard; Y3, the rejection of the scale-only
    alternative on both sets pooled (016:106–119).
  - Descriptive / post hoc: the ablation ladder, the comparator and the head statistics (016:121–135).
  - Exploratory: Tier A (016:81–84).
- **017.**
  - Prospective confirmation: Y1, and Y2 (conditional) (017:116–124).
  - Descriptive / post hoc: the per-template frozen-pattern values, the coordinated split, the rungs and the weight
    movement (017:132–157).
  - Exploratory: Tier A (017:88–94).
  - Unestablished: Y3, on the cue-final pairs of both sets. The frozen-pattern alternative was not rejected, so the
    necessity of modelling the pattern term is not established (017:126–131).
- **018.**
  - Prospective confirmation: Y1, the concentration on unseen cues; Y3, the rejection of the single-neuron account,
    pooled over both sets and pair-weighted (018:102–104, 118–121).
  - At criterion level, Y2's pooled closures (0.843 and 0.798) and both split guards passed (018:107–110).
  - Descriptive / post hoc: the rung ladder, the per-frame values and the ranking of neuron 1102 in the new quantifier
    frames (018:111–142).
  - Exploratory: Tier A (018:69–76).
  - Unestablished:
    - Y2 (conditional): the no-harm guard failed in one new frame;
    - Y4: the within-frame pattern term (018:107–128).
- **019.**
  - Prospective confirmation: Y1 to Y5, with Y2 and the new-frame parts of Y3–Y5 conditional (019:120–151). The result
    is aggregate: “the result is an aggregate routing result under the preregistered majority and family guards, not a
    rule that holds in every frame.” (019:164–165)
  - Descriptive / post hoc: the headroom witness, the absolute values, and a predeclared expectation that missed on Y2
    (019:152–174).
  - Exploratory: Tier A (019:81–92).
- **020.**
  - Exploratory: everything (020:110–132, 190–193). That includes:
    - the corrected exposed statistics;
    - the ceiling 0.9760;
    - the reading that the inherited `Δx3` prediction carries most of the error. The erratum qualifies the
      coordinated-frame part (020:13–15).
  - Unestablished: Y1, Y2 and Y3. The confirmation was never run (020:8–11, 182).
- **021.** All three labels test 020's `p_t`-wired program, which the erratum keeps as “a valid test of that program”
  (021:14–16).
  - Prospective confirmation: Y1 and Y3 (021:116–119, 124–127).
  - At criterion level, Y2's other three conditions passed (021:120–122).
  - Descriptive / post hoc: the ceiling (0.9683 and 0.9652), the error split and the calibration pass rates
    (021:140–149).
  - Unestablished: Y2 (conditional). `coordinated_r2` −0.153795 failed against a floor clamped to 0.0 (021:123).
    - The design's frozen reading of a negative label is “a generalization gap on that dimension beyond sampling
      variability, or a systematic difference between the fresh units and their strata.” (docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md:336–337)
    - The README's narrower reading, that it shows only that positive skill was not established (021:154–156), was
      first committed at closure (see [Record caveats](#record-caveats)).
    - The erratum attributes the failure plausibly to the wiring defect (021:16–17).
- **022.**
  - Prospective confirmation: all eight conditions, each on its own, with Y2 conditional (022:187–194).
  - Descriptive / post hoc: C4's magnitude, the shares beyond the condition statistics, and the ladder
    (022:197–237, 249).
  - Exploratory: `replicate-021`, “not a prospective confirmation” (022:261).
- **023.**
  - Prospective confirmation: Y1/coordinated, by a margin of +7.979e-5 (023:156–157); Y2/cue_final and Y2/coordinated,
    conditional (023:140–145).
  - Descriptive / post hoc: the joint rate 0.9178, the `" shiny"` exclusion, the strata and the cheaper block-0 rules
    (023:149–181).
  - Unestablished: Y1/cue_final at its envelope, an `ENVELOPE_ONLY_FAILURE` with the 0.90 guard met (023:152–155).
- **023 diagnostic.** Exploratory. It is recorded only in 024's design prose
  (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:33–36).
- **024.**
  - Prospective confirmation: the primary association, and the E–N guard (024:274–279). The result is associational:
    “The result is predictive, not causal.” (024:373) The guard's formal scope is two preregistered simple
    alternatives (024:295–297).
  - Descriptive / post hoc: the class means and contrasts, the subsets, and block 4's share of the error (024:309–326).
  - Exploratory: the calibration line, from spent exposed data (024:179–182).
- **025.**
  - Unestablished: `CAUSAL_EFFECT_NOT_ESTABLISHED`, with A = 26/40 and B = 23/40 against the threshold of 27
    (025:244–245). “It is not a finding of no causal effect.” (025:249)
  - Descriptive / post hoc: G = 37/40, the adjective/noun split, the causal fraction 0.341, the ladder, and the related
    correlations (025:252–262).

## Per-experiment records

Each record gives the question, the protocol, the data, the frozen outcome space where one exists, the results with
their evidence status, the limits the record itself states, and the hand-off. *Preregistered, one confirm run* marks
results decided by a locked criterion in the single confirmation run; the [evidence status](#evidence-status-by-experiment)
section sorts them into the five categories.
### Screening — context, before 005

- **Outcome.** “Final status: `ONE_PROPOSED_TARGET` — `regular-plural` (count-cued English noun number selection in pinned Pythia-70M).” (screening/behavior-candidates/README.md:39–40) `ordinal-suffix` was eliminated (screening/behavior-candidates/README.md:36–37).
- **Status.** “This is candidate selection under the frozen protocol v1. The compactness probe is an exploratory triage heuristic, not a mechanism claim.” (screening/behavior-candidates/report-2026-09-17.md:7) Run once at `e610c5f`, on pinned `EleutherAI/pythia-70m-deduped` revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (screening/behavior-candidates/README.md:27–29). Recorded in `1f043d9`.

### 005 — Count-cued noun number selection: prospective mechanism

- **Question.** “Design revision 5 (approved 2026-09-18) continues as protocol v2 with `PROGRAM_CAPPED`; its prospective question is whether the E → T → R circuit survives untouched nouns and unseen frames.” (005:105–107) The design's broader question asks whether the cue-to-noun-number computation can be recovered as “an explicit, compact, human-readable causal mechanism” (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:142–146).
- **Protocol.** Two protocol versions. v1: Tier A `discover` at `83d1ae4` (attempt 2; attempt 1 stopped on numerics before any conclusion) (005:82–83). v2: `continue` twice (M2 superseded, M3 adopted), `calibrate`, lock in `30e8b84`, `confirm` once on `30e8b84` (005:119–122). “The experiment is complete; a second scientific attempt requires a new protocol version.” (005:142) Design revision 5 “is not a preregistered amendment; it is a continuation motivated by a stopping-rule design issue, recorded as such.” (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:7–8)
- **A rule amended after Tier A.** The M3 eligibility amendment “was made after inspecting Tier A results and therefore is not preregistered with respect to development data. It was made before any Tier B holdout, reserve, or extension execution.” (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:120–122)
- **Data.** Tier A: development data only (005:88). Confirm: “runs the reserve nouns on the twelve manifest prompts, the six new frames, and the 72 cue-word prompts exactly once” (005:57–58).
- **Results.**
  - v1 — exploratory (Tier A development data): the program under-predicted the quantifier template beyond the 1.0-nat floor, “so version M1 is `rejected` and the recorded outcome is `NO_COMPACT_MECHANISM`” (005:98–101).
  - v2 — preregistered, one confirm run: “Outcome: `BEHAVIOR_NOT_REPLICATED` (axes `CIRCUIT_FAIL` via P3's retention count, `PROGRAM_FAIL` by cap, `NOT_GENERALIZED` via P3's retention count on the new frames).” (005:125–126) The behavioral precondition B1 failed: 98/120 primary correct against 103, 75/120 flips against 96 (005:126–127). “By the preregistered rule no other family is confirmatory.” (005:128–129)
  - Recorded, non-confirmatory values on the reserve nouns: P1 0.985, P3 0.795, P4 0.873, P5 0.817 / 0.869, P8 0.826, P9 0.987 / 0.900 / 0.074; new frames X1 120/120 positive pairs (005:130–140). Several missed their floors: P3's retention count (71/120), X2's P3 retention (62/120), X3 (lexicon Spearman 0.699 against 0.70), X4, and B2's template means, which fell below their holdout bands (005:130–140).
- **Tier A findings — exploratory** (005:88–97):
  - “`L03.H04` alone recovers 0.816 of the coordinated shift” (005:90–91);
  - “`L00.MLP` carries 0.535 of the transport-input axis, the raw embedding 0.009.” (005:94);
  - the k = 3 set (E `L00.MLP`, T `L03.H04`, R `L04.MLP` + `L05.MLP`) “recovers 0.982 overall (every template and rule class ≥ 0.975) and the isolation test retains 0.783 of the shift” (005:95–97).
- **Claim.** C002 was created at `LOCALIZED` after this run (005:141–142); see [Registries](#registries).
- **Post-closure notes.** “Archival notes (2026-09-18, after closure; no artifact rewritten)” (005:144), commit `6bc2d6a`: the report's `report: not_started` is “a runner cosmetic, not a missing phase” (005:146–148).
- **Hand-off.** “Experiment 005 is closed. Both outcomes stand as recorded; the next experiment is designed separately as Experiment 006 and treats all Experiment 005 data, including the reserve nouns and the extension prompts, as exploratory input.” (005:153–155)
- **Evidence.** `evidence/discovery-report-2026-09-18.md`, `evidence/final-report-2026-09-18.md`, `preregistration-lock.json`, `extension-v1.json`; raw outputs local only (005:77–78). Closed in `13600cd` and `6bc2d6a`.

### 006 — Low-rank decompilation of the cue encoding

- **Question** (design). “Can the token-local representation that `L00.MLP` writes for an arbitrary cue token be decompiled into a small multidimensional variable — a rank-`r` projection with `r ≤ 4` — followed by frozen low-rank transport and readout maps and the model's exact final LayerNorm, such that the resulting program prospectively predicts how unseen cue tokens, in unseen frames, with unseen nouns, are transported through `L03.H04` and converted by the layer-4 and layer-5 MLPs into noun-number logit shifts?” (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:66–72)
- **Closure reading**, narrowed in the closure commit (see [Record caveats](#record-caveats)). “Stated narrowly: Experiment 006 falsified the hypothesis that the cue-to-readout computation can be captured by the leading ≤ 4 *unsupervised* PCA directions of the `L00.MLP` cue representations.” (006:77–79)
- **Protocol.** Confirmation set frozen before any model output (`c5d32d1`); `explore` once at `adea65f` (006:52–53). “Consequently no lock may be written and the experiment ends at Tier A.” (006:71) Calibrate, lock and confirm: NOT_RUN.
- **Result — exploratory** (Tier A, on the pool exposed by 005). The pre-lock quality gate: Spearman 0.800 against ≥ 0.70 (pass), normalized leave-one-cue-out RMSE 0.623 against ≤ 0.50 (fail); no rank beat rank 1 by 20 % (006:67–69). “Outcome recorded as `QUALITY_GATE_FAILED`; no confirmation, no claim change (C002 stays `LOCALIZED`).” (006:71–72)
- **Post-hoc diagnostic — exploratory.** Cue-level MAE 2.35 (r = 1) falling to 0.40 (r = 15) (006:74–77). “The diagnostic supports, but does not prove, the explanation that those variance-leading directions are poorly aligned with the downstream-relevant directions; whether a *supervised* rank ≤ 4 subspace works is an open question for the next experiment, not a conclusion of this one.” (006:79–81)
- **Carried forward, exploratory.** Rank 4 was only about 9 % better than rank 1, against the 20 % requirement. The fixed circuit's isolation fidelity (P3) reached 708/708 paired signs but a correlation of 0.756, against the unchanged 0.90 criterion (006:58–59, 86–89).
- **Label provenance.** `QUALITY_GATE_FAILED` appears in the README but not in the design's outcome list (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:341–353).
- **Hand-off.** The never-executed confirmation set is inherited “byte-for-byte as the untouched holdout of Experiment 007” (006:86–92).
- **Evidence.** `evidence/exploration-report-2026-09-18.md`, `confirmation-v1.json`. Closed in `10c8faf`.

### 007 — A response-supervised low-rank cue subspace

- **Question** (design). “Does a small response-supervised subspace of the token-local `L00.MLP` representation — directions chosen to predict the measured downstream causal response rather than to explain the variance of `E` — prospectively predict the E-patch response and the behavioral shift of unseen cue tokens, in unseen frames, with unseen nouns?” (docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md:32–36)
- **Protocol.** `explore` once at `cc56014`; `calibrate` and `lock` at the same commit; lock committed in `ede0f53`; `confirm` once on a clean tree (007:79–81, 117, 139). “The outcome is final and is not amended or rerun.” (007:83)
- **Data.** Fresh: 006's never-executed set, 24 tokens × 6 frames × 20 nouns (007:13–15, 140–141).
- **Tier A — exploratory.** Quality gate Spearman 0.765 and normalized LOCO RMSE 0.320, both passing; rank 1 selected; τ = 3.972 nats (007:125–130).
- **Result — preregistered, one confirm run.** “Outcome rule: a Y floor failed → `PROGRAM_NOT_SUPPORTED`, failing family Y1, located at the encoding subspace.” (007:94)
  - precondition 120/120 against 108 — pass (007:85–86);
  - Y1 Spearman 0.785 against ≥ 0.80 — fail; Y1 MAE 1.071 nats against τ 3.972 — pass; confident-token sign `another` 3/6 and `this` 3/6 against 5/6 — fail (007:87–89; docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md:236–237);
  - Y2 Spearman 0.801, MAE 1.195 — pass; Y3 — pass (007:91–93).
- **Descriptive.** The baselines on the same fresh data: `Ridge-full` Y1 Spearman 0.796, `E005-scalar` 0.781, `PCA-006` (rank 1) 0.610; “`Ridge-full` and `E005-scalar` fail the same two confident tokens (`another`, `this`).” (007:95–97)
- **Limitation.** “This is a statement about the two estimators tested, not about every possible linear map” (007:109). This narrowing came after closure, in `5f2464a` (not labelled an erratum; no value changed).
- **Circuit families (reported; not in the outcome).** P1, P4, P5, P7, P8 and P9 passed; P3's correlation floor failed on both fresh sets, 0.696 and 0.840 against 0.90 (007:98–100). “C002 is not eligible for review and stays at `LOCALIZED`.” (007:100)
- **Hand-off.** “The 24 confirmation tokens, 6 frames, and 20 nouns are now exposed and may join the exploratory pool of a future experiment; they are no longer a holdout.” (007:112–113) This sentence was added after closure, in `5f2464a`.
- **Evidence.** `evidence/exploration-report-2026-09-18.md`, `evidence/final-report-2026-09-18.md`, `preregistration-lock.json`. Closed in `6340a4c`.

### 008 — Where the number signal of `this`-like cues is suppressed

- **Question** (design). “Why do cues such as `this` and `another` produce near-zero E-patch effects although their token-local `L00.MLP` representation contains a strong plural-side projection? Where along the causal path is that apparent number signal suppressed, transformed, or cancelled?” (docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md:24–26)
- **Protocol.** “Discovery-only: no lock, no confirmation, no claim promotion.” (008:7) `explore` once at `d324755` (008:40–41); the rules were frozen before the run (docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md:190). All data spent: 18 frames, 40 cue tokens, 80 nouns, all exposed by 005–007 (008:12–14).
- **Result — exploratory.** “Summary label: `MIXED` (joint stage–class consensus not reached). H2 is rejected for the suppressed tokens; H0 holds for `every` only.” (008:65–66)
  - suppressed stratum `a`, `this`, `another`, `every`; `every` classified `AXIS_ARTIFACT` (008:52–56);
  - `a`, `this`, `another`: `NONLINEAR_GATING`; the collapse stage differs per token, with no 75 % consensus; “`HEAD_DROPPED` for all three” (008:57–61).
- **Descriptive, within the exploratory run.** The head's output carries “almost none of the number signal for `this`, `a`, `another` (`T/R1` 0.07–0.29)” (008:86–89) while its attention is normal, and more than its input's axis projection for `these` and `those`.
- **Limitation.** “Nothing here is a confirmed mechanism; Experiment 008 is closed as a discovery run.” (008:115–116)
- **Hand-off to 009.** “The apparent number signal of `this`, `a`, `another` is dropped at transport” (008:106–115); a prospective test must freeze new determiners and frames and predict the head-output fraction `q_T` (008:106–115).
- **Evidence.** `evidence/exploration-report-2026-09-18.md`. Closed in `40265b5`.

### 009 — The transport rule of `L03.H04`

- **Question** (design; the README states none of its own). “Q2 (prediction). Can a rule derived from the forty exposed cue tokens predict, before any of them is run, how completely `L03.H04` transports the number signal of new determiners and numerals in new frames — and can the linear-from-`E` decompiler, refitted on forty tokens, predict their behavioral contrast shift?” (docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:31–33) Q1 asks whether the head's output change is an exact linear read-out of its cue-position input (docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:26–29).
- **Protocol.** Design revision 2 `fa62ab9`, written before the freeze `197d8e6` and Tier A (009:150). `explore` once at `b4c281a`; lock installed at `877221f`; a pre-confirm integrity check at `78d71c2`; `confirm` once at `78d71c2` (009:55–57, 95–96, 140). Closed in `27083c6`.
- **Data.** Fitting pool: 008's 40 cue tokens, 18 frames, 80 nouns (009:10). Fresh: 23 cue tokens in six fresh frames, 138 token prompts (009:13–17, 148–149).
- **Result — preregistered, one confirm run.** “Status — 2026-09-18: complete; outcome `HEAD_MECHANISM_CONFIRMED_P1 | TRANSPORT_RULE_FAILED | NOT_LOCKED`” (009:53)
  - precondition: fresh-frame cue effect 473/474 against 427 — pass (009:61);
  - Y1, head mechanism at the locked level P1 over 138 fresh pairs: Spearman 0.957 against ≥ 0.90, MAE 0.066 against τ_M 0.321 — pass (009:62–64);
  - Y2, the rank-1 transport rule from `E` over 23 fresh token means: Spearman 0.765 against ≥ 0.80 — fail, with category floors also missed (009:69–75). The `plural_predicted` failure was stated before confirm (009:132–133);
  - Y3, the contrast rule: `NOT_LOCKED`, “its gate failed at Tier A” (009:77).
- **The README's statement of the confirmed part.** “The head-level mechanism is now a prospectively confirmed statement: the selective transport of number by `L03.H04` is a fixed-normalization linear read-out of components already present in the cue residual it receives, for 40 exposed and 23 never-seen cues in 24 frames.” (009:83–85) The confirmatory part is the 23 fresh cues in the 6 fresh frames, 138 pairs (009:148–149); the 40 exposed cues and 18 exposed frames are Tier A.
- **Limit** (design). “P1–P3 decompose the head's output change given its *measured* input change; they say how the head transforms what it receives, not why upstream layers place each token where they do.” (docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:223–224)
- **Tier A — exploratory.** The P1 gate: R² 0.909, MAE 0.078 (009:102–104); the transport-rule gate passed (LOCO Spearman 0.776); the contrast-rule gate failed (Spearman 0.643 against 0.70) (009:116–124).
- **Descriptive.** The non-locked levels on the fresh pairs: “P2 (exact cue-position LayerNorm): 0.963 / 0.055; P3 (patched attention): 0.998 / 0.006” (009:64–65).
- **Open, per the README.** “What remains open is *upstream*: which components of the token-local encoding, transformed by layers 0–2, land on the head's read direction.” (009:85–86)
- **Hand-off.** “a next experiment should fit the head's read direction back through layers 0–2 to `E` rather than a single response-weighted direction.” (009:90–91)
- **Claim.** “C002 is unchanged (out of scope here).” (009:91)
- **Evidence.** `evidence/exploration-report-2026-09-18.md`, `evidence/candidate-predictions-2026-09-18.md`, `evidence/final-report-2026-09-18.md`, `preregistration-lock.json`, `predictions.md`, `confirmation-v1.json`.

### 010 — How the head-readable number signal is assembled

- **Question** (design). “Where, between the token-local encoding and the head's input, are the components that `L03.H04` reads as number assembled and opposed?” (docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md:25–26)
- **Protocol.** “Discovery-only exact attribution: nothing is fitted; every number is a measured or weight-only quantity under Experiment 009's confirmed P1 read functional, normalized by the template's plural cue.” (010:7–8) The classification rules were frozen before measurement (docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md:155); `explore` once at `0a6836b` (010:49–50). No freeze, lock or confirm by design. Closed in `16c6491`.
- **Data.** All spent: 63 cue tokens, 24 frames, 80 nouns, exposed by 005–009 (010:10–13).
- **Result — exploratory.** Every low-stratum token classed `MIXED`, so the summary is `MIXED`; no consistent opposer or supporter; `CONCENTRATED_ENCODING` absent (010:62–63). The relay bound failed everywhere: “no token in the pool has `G` below 0.29 (median 0.44, max 0.71)” (010:64–66), and “The bound was frozen without a baseline for this quantity and is too strict for this network; that is recorded as the reason, not corrected after the fact.” (010:66–67)
- **Findings within the exploratory run.**
  - “The opposition is already inside the encoding.” (010:85) For `this`, `a`, `another`, `every`, the on-axis part of `ΔE` reads +0.37 / +0.46 / +0.40 / +0.17 and the rest −0.17 / −0.35 / −0.27 / −0.24, negative in 24 of 24 frames for each; layers 1–2 change the read by at most 0.07 net (010:85–89).
  - `L02.MLP` adds +0.16 on average for the high stratum and about 0 for `a`, `this`, `another` (010:93–96).
  - `n_80` is 320–700 of 2048 (010:63).
- **Hand-off.** “The candidate zero-parameter, weight-only rule for a prospective test is `g_E(w, T)`: the encoding read through the head's direction, normalized by the template plural cue (Spearman 0.925 across the 63 exposed tokens here).” (010:110–112)
- **Record inconsistencies.** The README cites design revision 2 (010:5) but also “Design revision 3 (pre-measurement)” (010:41). Its results-state digest differs from the one two downstream extracts record for the same run (experiments/011-encoding-read-prospective/inherited/experiment-010-transport-fractions.json:1).
- **Evidence.** `evidence/exploration-report-2026-09-18.md`.

### 011 — Does the encoding read predict `L03.H04`'s transport of new cues?

- **Question.** “Experiment 011: Does the Encoding Read Predict `L03.H04`'s Transport of New Cues?” (011:1) “Prospective and zero-parameter: the quantity under test is a weight-defined read functional applied to the cue's layer-0 encoding” (011:7–8).
- **Protocol.** Design revision 2 `b5f59b9` (docs/superpowers/plans/2026-09-19-experiment-011-encoding-read-prospective-plan.md:5). Freeze `8c1c6c5`; `explore` once at `69c7a14`; the lock installed at `85a6c04`; `confirm` once at `85a6c04` (011:60–65, 109–110). Closed in `0358c8e`.
- **Data.** Calibration only on 010's 63 exposed tokens and 24 frames; fresh: 24 cue tokens from five lexical classes, six frames, 142 token prompts (011:13–22).
- **Result — preregistered, one confirm run.** “Status — 2026-09-19: complete; outcome `ENCODING_READ_PREDICTS_TRANSPORT | HEAD_P1_REPLICATED`” (011:58)
  - Y1, `ḡ_E` against the measured transport over 24 token means: Spearman 0.848 against ≥ 0.80, MAE 0.077 against τ_g 0.248 — pass (011:75–76);
  - Y2, the P1 read-out from the measured head input against the measured transport over 142 fresh pairs: Spearman 0.974 against ≥ 0.90, MAE 0.076 against τ_M 0.313 — pass (011:87–89).
- **Descriptive.** Experiment 009's fitted rank-1 rule on the same token means reaches Spearman 0.816 and MAE 0.091: “it would also have cleared the Y1 floors on this set.” (011:91–92) “The zero-parameter read is better on both figures, but the margin is modest” (011:92–93).
- **Locked prediction.** `an`, the lowest measured transport (0.30), was predicted 0.18 (011:78).
- **Scope.** “The claim is proportional to the set: the 24 cues were drawn by quota from five lexical classes (determiner-like, numeral, quantity, possessive/pronoun, adjective), so the support is generalization across this preregistered heterogeneous cue set, not to arbitrary English words.” (011:100–102)
- **Residual, per the README.** “The systematic part of the residual is the net contribution of layers 1–2, which the encoding read omits by construction and which is positive on average (+0.18); where layers add more, the read under-predicts more.” (011:102–104)
- **Claim.** “C002 is unchanged (out of scope here).” (011:104–105)
- **Evidence.** `evidence/exploration-report-2026-09-19.md`, `evidence/candidate-predictions-2026-09-19.md`, `evidence/final-report-2026-09-19.md`, the lock and predictions.

### 012 — Is the layers-1–2 correction a token-local MLP computation?

- **Question.** “Experiment 012: Is the Layers-1–2 Correction to the Encoding Read a Token-Local MLP Computation?” (012:1) The prediction evaluates the block-1 and block-2 MLPs at the cue position on `ΔE`, with attention held at the reference (012:7–9).
- **Protocol.** Design revision 2 `19183a7` (docs/superpowers/plans/2026-09-19-experiment-012-layer-correction-token-local-plan.md:5). Freeze `2a7c3d3`; `explore` once at `ddbbbd3`; the lock installed at `5cea6c4`; `confirm` once at `5cea6c4` (012:65–71, 130–131). Closed in `fc442f6`; the closing claim was restated after closure in `1b1edff`.
- **Data.** Calibration on 87 exposed cue tokens and 30 frames; fresh: 24 cue tokens, six frames, 144 token prompts (012:15–26).
- **Result — preregistered, one confirm run.** “Status — 2026-09-19: complete; outcome `LAYER_CORRECTION_TOKEN_LOCAL_MLP | COMPOSITE_PREDICTS_P1`” (012:63)
  - Y1, predicted against measured net layers-1–2 correction over 24 token means: Spearman 0.864 against ≥ 0.80, MAE 0.061 against τ_c 0.207, R² 0.795 against ≥ 0.50 — pass (012:81–83);
  - Y2, the composite against the P1 read: Spearman 0.922 against ≥ 0.90, MAE 0.062 against τ_P 0.130 — pass (012:91–93).
- **The claim at its stated strength.** “Stated at its safe strength: a frozen token-local MLP model, derived before any exposure to the test prompts, prospectively predicts most of the net layers-1–2 correction on unseen cues in unseen frames (ordering, absolute error and magnitude)” (012:110–112) “the design does not claim to have decoded the layer-1–2 MLP correction itself, only to have predicted the net correction and its effect on the head's input.” (012:119–120)
- **Descriptive.** Pair level: Spearman 0.859, MAE 0.098, R² 0.65 over 144 pairs (012:89–90). “neuron 1987 is again the top contributor in every template (mean absolute mass 0.088, next 0.049)” (012:105–106).
- **Post-closure restatement** (`1b1edff`, after the closure `fc442f6`; not labelled an erratum, no value changed). It replaced the sentence “The composite "encoding read + token-local MLP correction" now accounts for what the head reads” (012@fc442f6:117) with the safe-strength statement above.
- **Hand-off.** “The omitted attention paths are the natural next question.” (012:125) This sentence was added after closure, in `1b1edff`.
- **Claim.** “C002 is unchanged (out of scope here).” (012:125)
- **Evidence.** `evidence/exploration-report-2026-09-19.md`, `evidence/candidate-predictions-2026-09-19.md`, `evidence/final-report-2026-09-19.md`, the lock and predictions.

### 013 — What layers 1–2's attention adds

- **Question.** “Prospective, zero-parameter, residual-predicting: Experiment 012's frozen token-local MLP model is taken as given, and the candidate for the residual it left is the computation it omitted — the frame's own base state and the sixteen heads of layers 1–2 at the cue position with their attention patterns held at the reference.” (013:7–9)
- **Protocol.** Design revision 2 `617f549` (docs/superpowers/plans/2026-09-19-experiment-013-attention-paths-frozen-pattern-plan.md:5). Freeze `12090c7`; `explore` once at `cc14911`; the lock installed at `c01126f`; the reviewer signed off; the single two-stage `confirm` at `c01126f` (013:70–82, 134–135). Closed in `9b656e2`.
- **Data.** Exposed: 111 cue tokens, 36 frames. Fresh: 24 cue tokens in the 36 exposed frames (Y1, 864 prompts) and in 6 fresh frames (Y2, 144) (013:16–28).
- **Result — preregistered, one confirm run.** “Status — 2026-09-19: complete; outcome `RESIDUAL_PREDICTED_TOKENS | RESIDUAL_PREDICTED_FRAMES_CONDITIONAL | ATTRIBUTION_PREDICTED`” (013:68)
  - Y1: Spearman 0.877 against ≥ 0.80, MAE 0.021 against τ_r 0.070, R² 0.668 against ≥ 0.50 (013:88–90);
  - Y2, conditional on the new frames' reference states: 0.919, 0.022, R² 0.747 (013:100–101);
  - Y3, pooled over the 1,008 pairs of both sets (the new-frame share conditional): `ĉ_H` against `c_H` Spearman 0.838 against ≥ 0.70, MAE 0.030 against τ_A 0.106; `ĉ_M` against `c_M` 0.986 against ≥ 0.90 (013:106–108).
- **Descriptive.** Compensation cases: 236 fresh pairs meet the frozen rule, and the measured signs match the prediction in 95 % (225) (013:109–112).
- **Limits.** “Limits: the residual is a small quantity and R² was 0.67 / 0.75, not higher — the numerals' over-shoot was predicted in direction but over-estimated in size, and near-zero residuals had unreliable signs; the reference patterns are taken from the reference run, not explained; five lexical classes, three templates, this checkpoint. C002 is unchanged (out of scope here).” (013:126–129)
- **Hand-off.** None stated as such. The remaining error is the attention-pattern-change term, |.| ≈ 0.03 per pair (013:123–126); 014 and 015 take up the pieces.
- **Evidence.** `evidence/candidate-predictions-2026-09-19.md`, `evidence/exploration-report-2026-09-19.md`, `evidence/final-report-2026-09-19.md`, the lock and predictions. The Y2 prediction table was written to the results state with its digest, not committed (013:54–55).

### 014 — What drives block-2 neuron 1987

- **Question** (design). “Which input feature controls block-2 neuron 1987, and does a compact input-side predictor — one weight-defined direction, one reference operating point per frame, the frozen-pattern arriving change of Experiment 013 — prospectively predict the neuron's activation change and its firing for cue words and frames never measured on it? Is the simple number-axis account rejected on the same fresh data?” (docs/superpowers/specs/2026-09-19-experiment-014-neuron-1987-feature-design.md:61–64)
- **Protocol.** Design revision 2 `bea0203`. Freeze `3013a3e`; `explore` once at `bcb0bfe`; the lock installed at `d008dae`; the reviewer signed off; the single two-stage `confirm` at `d008dae` (014:74–85, 148–149). Closed in `56a0fc5`.
- **Data.** Exposed: 135 cue tokens, 42 frames. Fresh: 24 cue tokens in the 42 exposed frames (Y1, 1,008) and in 6 fresh frames (Y2, 144) (014:15–27).
- **Result — preregistered, one confirm run.** “Status — 2026-09-19: complete; outcome `NEURON_FEATURE_PREDICTED_TOKENS | NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`” (014:72)
  - Y1: Spearman 0.950, R² 0.970, balanced firing accuracy 0.923 against 0.80 / 0.50 / 0.90 (014:96–98);
  - Y2, conditional on the new frames' reference states: 0.910, 0.957, 0.927 (014:101–102);
  - Y3, pooled over both sets: the axis-only alternative reaches R² −1.41 and balanced accuracy 0.601, below its rejection bounds 0.30 and 0.70, so it is rejected (014:104–105).
- **Locked predictions without outcome force.** The committed per-cue calls: the strong firings of `dozens`, `hundreds` and `thousands`, eighteen vetoes with 0 firing frames of 756, and the borderline cases `infinite`, `millions`, `billions` (014:107–114).
- **Descriptive.** The firing set is interpreted only after scoring (014:125–131). “The claim is about the computation, not a label: the firing set is described afterwards and is consistent with a count-magnitude reading, but the experiment tests the mechanism (excitation along the number axis, veto from off-axis structure, one neuron), not the gloss.” (014:138–140)
- **Limits.** “Limits: the predictor under-shoots the boundary cases, most in the coordinated-adjective template; it inherits Experiment 013's approximations (the pattern-change remainder is |.| 0.17 in preactivation units); one neuron, five lexical classes, three templates, this checkpoint. C002 is unchanged (out of scope here).” (014:140–143)
- **Evidence.** `evidence/candidate-predictions-2026-09-19.md`, `evidence/exploration-report-2026-09-19.md`, `evidence/final-report-2026-09-19.md`, the lock and predictions. The Y2 prediction table was recorded by digest only (014:53–54).

### 015 — The cue-induced attention-pattern change from the decoded query/key computation

- **Question.** The pattern-change term Experiment 013 measured is written as a weight-only program and evaluated at two levels kept apart (015:7–10). Level 0 is “the token-local cue-change model conditional on the reference frame state” (015:13).
- **Protocol.** Design revision 2 `4babcce`. Freeze `5d2a8f3`; `explore` once at `407dd40`; the lock installed at `5ed51c9`; the reviewer cleared the install; the single two-stage `confirm` at `5ed51c9` (015:83–97). Closed in `08f4ca4`.
- **Data.** Exposed: 159 cue tokens, 48 frames. Fresh: 24 cue tokens in the 48 exposed frames (Y1, 1,152) and in 6 fresh frames (Y2, 144) (015:21–34).
- **Result — preregistered, one confirm run.** “Status — 2026-09-20: complete; outcome `PATTERN_CHANGE_PREDICTED_TOKENS | PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`” (015:81)
  - Y1: Spearman 0.986, R² 0.895, pooled entry R² 0.908 (layer 1) and 0.700 (layer 2), all passing (015:105–107);
  - Y2: Spearman 0.857 passes; R² 0.490 and layer-2 entry R² 0.423 fail the 0.50 floor (015:109–111);
  - Y3, pooled over both sets: the axis-only alternative reaches R² −1.16 and pooled entry R² 0.040, below its bounds, so it is rejected (015:123–124).
- **Descriptive.** One new frame, `quantifier-015-2` (`The agency recruits`), accounts for most of the Y2 shortfall; “The frame is valid by every frozen criterion and Level 1 is exact on it, so the miss is the reduced description's” (015:116–118). Over the other five frames the token means would give R² 0.82, “that is not the frozen criterion and changes nothing” (015:121–122).
- **Limits.** “This is not "attention predicted from the token": the frame still supplies its reference keys, values and logit rows; the experiment explains the change, not the attention state.” (015:157–158) “Y2's failure is a frozen result of this protocol version and any better reduced description (for example a frame-conditional base) is a new experiment.” (015:159–160)
- **Hand-off.** The frame-conditional base, taken up by 016 (015:159–160).
- **Evidence.** `evidence/candidate-predictions-2026-09-20.md`, `evidence/final-report-2026-09-20.md` (it holds the Tier A section; there is no separate exploration report), the lock and predictions.

### 016 — What the frame contributes: the frame-conditioned token-local Q/K model

- **Question** (design). “Through which channels does a frame's cue-position state enter the cue-induced attention-pattern change, and does the frame-conditioned token-local model — the cue's query, key and value changes token-local up to the frame's normalization scale, the self-logit formed with the frame's own reference operands, block 1's MLP at the frame's own operating point, everything else at the template means — prospectively predict the change for cue words and frames never measured on it? Is the scale-only account rejected on the same fresh data?” (docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:69–73)
- **Protocol.** Design revision 2 `498e604`. Freeze `e73a3cf`; `explore` once at `6343f23`; the lock installed at `dc4e2a7`; the single two-stage `confirm` at `dc4e2a7` (016:81–97). No reviewer sign-off is recorded in the README. Closed in `de61ead`.
- **Data.** Exposed: 183 cue tokens, 54 frames. Fresh: 24 cue tokens in the 54 exposed frames (Y1, 1,296) and in 12 fresh frames (Y2, 288) (016:22–37).
- **Result — preregistered, one confirm run.** “Status — 2026-09-20: complete; outcome `FRAME_CHANNELS_PREDICTED_TOKENS | FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL | SCALE_ONLY_REJECTED`” (016:79)
  - Y1: Spearman 0.998, R² 0.994, pooled entry R² 0.994 (layer 1) and 0.988 (layer 2), against 0.90 / 0.85 (016:106–108);
  - Y2, conditional on the new frames' reference states: 1.000, 0.996, 0.995, 0.986, and every new frame above the 0.80 per-frame guard, lowest 0.933 (016:111–115; docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:238–239);
  - Y3, pooled over both sets: the scale-only alternative reaches 0.672, 0.316 below Level 0-F, so it is rejected (016:118–119).
- **Descriptive.** The ablation ladder: “without the reference operands 0.709 (−0.278), without block 1's operating point 0.888 (−0.100), without the scale 0.946 (−0.042) — the predeclared ordering operands > operating point > scale holds” (016:121–123).
- **Limits.** “This is *not* the claim that three numbers explain the frame, nor that attention is predicted from a frame's text: the channels are reference-run quantities, and the experiment explains the cue-induced change given the frame's reference state.” (016:145–146) “Limits: one checkpoint, three templates, five lexical classes, sixteen heads, twelve new frames.” (016:150–151)
- **Hand-off.** None stated as such. “With Experiments 015 and 016 together, the cue-position computation through layers 1–2 is written end to end as a weight-only program whose frame inputs are named: the reference keys, values and logit rows, the reference self-logit operands, the residual's scale, and block 1's operating point.” (016:147–150)
- **Evidence.** `evidence/candidate-predictions-2026-09-20.md`, `evidence/final-report-2026-09-20.md` (with the Tier A section), the lock and predictions. The Y2 prediction table was recorded by digest only (016:64–65).

### 017 — The transport head's own pattern change

- **Question.** “This experiment predicts the head's own attention-row change at the transport position `p_t` and its output change `ΔT` along the locked axis `d̂_T` from the decoded chain alone” (017:9–10). The rival account is the value term `F`, “the frozen-pattern account, the alternative committed for rejection on the cue-final pairs” (017:18–20).
- **Protocol.** Design revision 2 `4a2e409` (docs/superpowers/plans/2026-09-20-experiment-017-transport-head-pattern-plan.md:5). Freeze `01cdc69`; `explore` once at `ebc2206`; the lock installed at `4b6f766`; the single two-stage `confirm` at `4b6f766` (017:88–106). Closed in `c583405`.
- **Data.** Exposed pool: 207 cue tokens, 66 frames, 80 nouns. Fresh: 24 cue tokens, in the 66 exposed frames (Y1, 1,584 prompts) and in 12 fresh frames (Y2, 288) (017:25–40).
- **Result — preregistered, one confirm run.** “Status — 2026-09-20: complete; outcome `HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_NOT_REJECTED`” (017:86)
  - Y1, all five floors: head-row entry R² 0.997; `Π` token means Spearman 0.998 and R² 1.000; `ΔT` pairs 0.999 and token means 1.000 (017:116–119);
  - Y2, conditional, all six floors: row entry 0.998; `Π` 1.000 / 1.000; `ΔT` 1.000 / 1.000; every fresh frame above the 0.90 guard, lowest 0.986 (017:121–124; docs/superpowers/specs/2026-09-20-experiment-017-transport-head-pattern-design.md:330–331);
  - Y3: the frozen pattern reaches `ΔT` R² 0.958 on the 1,248 pooled cue-final pairs, with a gap of 0.042 to Level 0, so neither rejection clause (below 0.95, gap at least 0.05) holds (017:126–129).
- **The frozen reading of Y3.** “the head's pattern change was predicted, but the fresh set did not establish that modelling it explicitly was necessary relative to the frozen-pattern alternative” (017:129–131).
- **Descriptive (no label).** Per template over both sets, the frozen pattern reaches cardinal 0.907 and quantifier 0.883; in the new frames alone, 0.590 and 0.415 (017:132–137). “A per-template criterion was not preregistered and none is applied.” (017:138) On the 624 coordinated pairs the criterion would have rejected the frozen pattern (0.945, gap 0.053), “recorded, entering no label” (017:138–140).
- **Not claimed.** “This is not the claim that the frame is a few numbers (the channels' dimensionality is recorded in the design), nor that block 2's frame dependence is concentrated (Experiment 018), nor anything about behaviour.” (017:170–172)
- **Hand-off.** Block 2's frame dependence goes to 018 (017:170–172).
- **Record inconsistency.** One passage says no frame fell below row-entry R² 0.985 (017:163–164); the exposed per-frame minimum is 0.970 (017:117). 0.985 holds only for the twelve new frames.
- **Evidence.** `evidence/candidate-predictions-2026-09-20.md`, `evidence/final-report-2026-09-20.md`, the lock and predictions.

### 018 — Block 2's operating-point dependence: is channel D carried by a locked subset?

- **Question.** “Experiment 018: Block 2's Operating-Point Dependence — Is Channel D Carried by a Locked Subset of 256 Neurons?” (018:1) The statistic is the closure fraction κ of a masked subset `S` (018:17–18).
- **Protocol.** Design revision 2 `ab9e53a` (docs/superpowers/plans/2026-09-21-experiment-018-block2-concentration-plan.md:5). Freeze `e994202`; `explore` once at `9b629bf`; the lock installed at `6fa4c48`; the reviewer signed off; the single two-stage `confirm` at `6fa4c48` (018:69–91). Closed in `5132174`.
- **Data.** Exposed pool: 231 cue tokens, 78 frames. Fresh: 24 cue tokens, 12 new frames; 2,160 fresh pairs (018:23–33, 97).
- **Result — preregistered, one confirm run.** “Status — 2026-09-21: complete; outcome `CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED | PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES`” (018:67)
  - Y1: `κ_{c_L}(S_256)` 0.863 against ≥ 0.70 and `κ_Π(S_256)` 0.724 against ≥ 0.50 — pass (018:102–104);
  - Y2: pooled 0.843 / 0.798 and both split guards pass, but the no-harm guard fails in one new quantifier frame, `quantifier-018-4` (−0.102 against the −0.05 margin), so Y2 is negative (018:107–115);
  - Y3, pooled over both sets and pair-weighted: the single neuron `S_1 = {1987}` reaches 0.219 against 0.860 for `S_256` (gap 0.641), so the single-neuron account is rejected (018:118–121);
  - Y4: three cardinal frames reach frozen-pattern `ΔT` R² at or above 0.90, so the within-frame pattern term is not established (018:123–126).
- **The README's reading of Y2.** “Read as the design fixed it: a failure of full frame-conditional robustness of the locked subset, not an absence of channel-D concentration on new frames.” (018:115–117) The design text gives no such fixed reading for a guard failure (see [Record caveats](#record-caveats)).
- **Descriptive.** The eleven other new frames improve on the template base; “in the four new quantifier frames neuron 1102 is first (mean read-unit effect 0.0246) and 1987 fourth” (018:139–140). “Neither observation carries a label.” (018:142)
- **Not claimed.** “This is not the claim that channel D lives in 256 neurons (`S_256` is a recovery subset chosen by one read-unit rule, not a uniquely defined circuit), nor that the head's row dependence is compact (it is diffuse, as predeclared), nor anything about behaviour” (018:156–158)
- **Hand-off.** “which members of the population carry the computation appears to depend on the frame, which is recorded here descriptively and is the natural question for Experiment 019.” (018:158–159)
- **Evidence.** `evidence/candidate-predictions-2026-09-21.md`, `evidence/final-report-2026-09-21.md`, the lock and predictions.

### 019 — Frame-conditioned routing of channel D

- **Question.** “Experiment 019: Frame-Conditioned Routing of Channel D — Is the Frame-Specific Membership of Block 2's Recovery Subset Predicted from the Frame's Reference State?” (019:1)
- **Protocol.** Design revision 3 `b130b50`, wording `efe0623` (019:4–5). Freeze `4ab1110`; `explore` once at `2dbfdd1`; the lock installed at `9033041`; the reviewer signed off; the single two-stage `confirm` at `9033041` (019:81–111). Closed in `a0d73c8`.
- **Data.** Exposed pool: 255 cue tokens, 90 frames, 11,796 licensed pairs. Fresh: 24 cue tokens and 18 new frames; 2,592 fresh pairs (019:26–43, 116).
- **Result — preregistered, one confirm run.** “Status — 2026-09-21: complete; outcome `ROUTING_PREDICTED_TOKENS | ROUTING_PREDICTED_FRAMES_CONDITIONAL | OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT | TEMPLATE_FAMILY_INSUFFICIENT | MEMBERSHIP_PREDICTED`” (019:79)
  - Y1: `Δκ_64(E)` 0.090 against ≥ 0.05, above the global rule in 64 of 90 frames, split gains 0.070 / 0.159 (019:120–123);
  - Y2: 0.144, in 14 of 18 frames, split gains 0.147 / 0.133 (019:130–133);
  - Y3: `ρ_64` 0.862 (Y1), 1.136 (Y2, conditional), 0.913 pooled, against ≥ 0.50 per set and ≥ 0.60 pooled (019:137–139);
  - Y4: the template family falls short by 0.068 pooled and 0.080 on Y2, against the 0.05 margin (019:145–146);
  - Y5: predicted membership 0.867 against `S'` 0.410 (Y1) and 0.375 (Y2) (019:149–151).
- **Interpretation, from the frozen table.** “The frozen interpretation table gives Account 2, compact form” (019:176).
- **Descriptive.** Not a per-frame rule: `E_64` loses more than 0.05 R² to the global rule in 7 of 90 Y1 frames and 2 of 18 Y2 frames (019:161–164); “the result is an aggregate routing result under the preregistered majority and family guards, not a rule that holds in every frame.” (019:164–165) One predeclared expectation failed on Y2 by 0.0008 (019:156–157).
- **Not claimed.** “This is not the claim that 64 neurons suffice (every selector's absolute `κ` stands beside its gain), nor that either oracle is the best possible subset, nor that a frame's state is predicted from its text, nor anything about behaviour.” (019:186–188)
- **Hand-off.** None stated.
- **Evidence.** `evidence/candidate-predictions-2026-09-21.md`, `evidence/final-report-2026-09-21.md`, the lock, predictions and `build_inherited_extract.py`.

### 020 — Readout decompilation to the model's own logit contrast

- **Question.** “Given the change entering the post-`L03` readout path as the already-decoded chain *predicts* it (Experiments 011/012/017, never measured), can the resulting change of the singular-versus-plural logit contrast `Δc(f, w, n) = c(frame f, cue w, noun n) − c(frame f, reference cue, noun n)` at the prediction position be predicted prospectively for unseen cues (Y1), unseen frames conditional on their reference state (Y2) and unseen nouns (Y3)?” (020:21–24)
- **Status.** “Status: closed at Tier A on 2026-09-22.” (020:8) The confirmation was “intentionally not run because the corrected block-5 parallel-residual implementation materially changed the exposed calibration on which the preregistered outcome floors had been based.” (020:8–10) Lock and confirm: NOT_RUN. “The Experiment 020 floors are not changed retrospectively, and no Y1/Y2/Y3 confirmation label is assigned.” (020:182)
- **Protocol.** Freeze at `ba2cdc2`. Explore attempt 1 at `abf3df9` stopped as an incident on the Level-1 identity (4.593e-03 against 1e-3); after a diagnostic, an authorized tolerance amendment (`ce3766b`, design revision 3) raised it to 7e-3, and attempt 2 completed at `ce3766b` (020:69–86). Closed in `7f4a4f8`.
- **Data.** Exposed only: 108 frames, 279 cue tokens, 80 nouns; 30,024 exposed pairs; the 3,060-prompt confirmation set was never executed (020:32–39, 164–166).
- **Findings — exploratory** (exposed data; the README calls them “calibration record, never evidence for a label” (020:110)):
  - corrected flattened `R²` 0.6891; by template cardinal 0.8535, quantifier 0.8427, coordinated-adjective 0.2729 (020:114–120);
  - the downstream ceiling, the same program fed the measured `Δx3`: 0.9760 (020:132);
  - six of the twelve frozen floor conditions sit below their thresholds on the exposed pool itself (020:160);
  - “What Experiment 020 establishes, descriptively and on exposed data only: the downstream readout from `Δx3` to the contrast is decoded almost exactly (ceiling 0.976 with the measured input); the weakness of the end-to-end prediction is the inherited upstream prediction of `Δx3`, above all in coordinated-adjective frames” (020:190–193).
- **Erratum** (`2320345`, 2026-09-23). “In coordinated-adjective frames this program's Level 0 builds the layer-1/2 reference rows through `p_t` instead of the cue position `p_c` used by the chain validated in 017–019, which changes the computation (exposed coordinated `Δc` `R²` −0.146 as wired here against 0.555 as 017 wired it).” (020:13–15) It changes no recorded number, digest or closure record (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:3–4). Cue-final values are unaffected, and the ceiling stands (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:55–57).
- **Claim.** C002 unchanged at `LOCALIZED` (020:188).
- **Hand-off.** “The confirmation set `e098e2b4…` has never produced a model output and is reserved, byte-for-byte and read in place, for the next protocol version” (020:184–186), which is 021.
- **Evidence.** `evidence/exploration-report-2026-09-22.md`, `evidence/exploration-record-2026-09-22.json`, the erratum, `closure.json`.

### 021 — Corrected readout-decompilation confirmation

- **Question.** “The Experiment 020 question — does the decoded mechanism predict the model's own singular-versus-plural logit contrast for unseen cues, unseen frames and unseen nouns — with the same program, model, populations, procedure, labels and untouched confirmation set, and floors from a preregistered exposed-only calibration.” (021:9–11)
- **Protocol.** Design revision 3 `0ac46aa`, plan `fb10d44` (021:3–7). Calibrate at `c49c16d` (record committed alone in `ad22671`), lock at `ad22671` (committed in `b7e8861`), confirm once at `b7e8861` (021:67–84). The floors come from 10,000 SHA-indexed exposed-only draws, with a zero-skill clamp (021:47–48, 146–148). Closed in `86f7f8c`.
- **Data.** Fresh: 020's reserved set, 24 cues, 18 frames, 24 nouns, 3,060 prompts, executed for the first time (021:85–95).
- **Result — preregistered, one confirm run.** “Status — 2026-09-23: complete; outcome `CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL | NOUN_READOUT_FIXED`” (021:59)
  - Y1: all four conditions pass; `token_mean_r2` 0.016018 against a floor clamped to 0.0 (021:116–119);
  - Y2: three of four pass; `coordinated_r2` −0.153795 fails against a floor clamped to 0.0 (021:120–123);
  - Y3: all four pass (021:124–127).
- **The design's frozen reading** of a negative label: “a generalization gap on that dimension beyond sampling variability, or a systematic difference between the fresh units and their strata.” (docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md:336–337)
- **The README's narrower reading of a clamp-bound failure**, first committed at closure (`86f7f8c`), after the confirm: “A failed condition whose floor is clamped to 0.0 establishes only that positive predictive skill was not established under that criterion; it is evidence of a generalization gap only if the fresh value also falls below the raw calibration tail.” (021:154–156) The README heads it “agreed at the floor and lock reviews” (021:151); no committed review record shows that. The value sits at the 45.8th percentile of the exposed-like draws (021:159–160).
- **Descriptive.** The ceiling: “flattened `R²` 0.9683 on Y1 (75.6th percentile of its row's draws) and 0.9652 on Y2 (37.3rd)” (021:140–141); the error split: “0.352 of 0.383 (Y1) and 0.385 of 0.420 (Y2) is inherited from the upstream prediction of `Δx3`, and 0.032 / 0.035 is downstream” (021:141–142).
- **Erratum** (`2320345`). “The frozen results, labels, floors and digests below are unchanged and remain a valid test of that program” (021:14–16), meaning 020's program, which builds the coordinated reference rows through `p_t`. All three labels therefore test that wiring, not the chain as validated in 017: Y1 includes the 36 coordinated exposed frames, and the two wirings agree only in cue-final frames (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:48–49). “the single coordinated Y2 failure must not be read as evidence about the correctly wired 017 upstream chain; it is plausibly explained by the wiring defect.” (021:16–17)
- **Not claimed.** “This is not a claim of good prediction everywhere (the floors are relative to the program's own exposed performance, and the token-mean margin is small), nor that a frame's state is predicted from its text, nor anything about behaviour beyond `Δc` at `p_t`.” (021:171–173)
- **Claim.** C002 unchanged at `LOCALIZED` (021:174–176).
- **Evidence.** `calibration-v1.json`, `preregistration-lock.json`, `predictions.md`, `evidence/final-report-2026-09-23.md`, `evidence/confirmation-record-2026-09-23.json`, the erratum.

### 022 — Prospective localization of the upstream `Δx3` error

- **Question.** “The question is where the committed upstream `Δx3` prediction loses the gap between Level 0 and the `Δc` ceiling.” (022:18) Four claims: “C1: block-0 attention dominates the cue-final gap” (022:28–31), “C2: the reductions contribute little to the cue-final gap” (022:28–31), “C3: the reductions contribute little to the coordinated gap” (022:28–31) and “C4: block-0 cue→target attention contributes positively to the coordinated gap.” (022:28–31)
- **Protocol.** Design revision 4 `219cdc5`, plan revision 3 `e6d8299` (022:3–7). Revision 4 adopted 017's wiring after the spike was recomputed: “Revision 3's C3 ("the coordinated gap is split") rested on the defect and was withdrawn.” (022:35–39) Freeze at `dd50231` (committed `0de0fb8`), calibrate (committed `f947305`), lock (committed `4f8f31d`), confirm once at `4f8f31d`, report, then the exploratory `replicate-021` (022:122–169). Closed in `0818c4c`.
- **Data.** Fresh: 24 cues and 18 frames, 3,060 manifest keys; stage 2 ran 3,024 target prompts once each; the 79 exposed nouns (022:122–160, 256).
- **Frozen outcome space.** Per condition NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS; “Aggregate: none: the eight condition results are the result, each reported and read on its own; Y1 and Y2 are never pooled; no all-pass requirement.” (experiments/022-upstream-error-localization/preregistration.md:37)
- **Result — preregistered, one confirm run.** “Closed (2026-09-24). The single confirmation ran once at `4f8f31d`, with no incident, and all eight preregistered conditions passed individually, each against its frozen envelope and its meaning guard; there is no aggregate label.” (022:9–10)
  - Y1/C1 σ Bv + σ Bp, cue-final: 0.94789 against ≥ 0.901075; Y2/C1: 0.93808 against ≥ 0.888785 (022:187, 191);
  - Y1/C2 σ R, cue-final: −0.0000118 against ≤ 0.005595; Y2/C2: 0.0038511 against ≤ 0.008226 (022:188, 192);
  - Y1/C3 σ R, coordinated: 0.0059509 against ≤ 0.006904; Y2/C3: 0.0069294 against ≤ 0.014172 (022:189, 193);
  - Y1/C4 σ T, coordinated: 0.40814 against ≥ 0.327059; Y2/C4: 0.38338 against ≥ 0.207165 (022:190, 194).
  - The four Y2 conditions are frame-conditional, each frame at its own stage-1 reference state (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:254).
- **Descriptive.** C4's magnitude: “its measured share, 0.408 (Y1) and 0.383 (Y2), is descriptive” (022:249); the ladder from Level 0 to the full composition, 0.8005 → 0.9607 (Y1 cue-final) and 0.6277 → 0.9652 (Y1 coordinated) (022:210–219); the tightest margin is Y1/C3 (022:197–198).
- **`replicate-021` — exploratory.** On 021's stored measurements, after the report, “not a prospective confirmation” (022:261); descriptively, all eight corresponding statistics lie inside the 022 envelopes, a comparison that “is not recorded by the runner and is not a result” (022:284–287).
- **Not shown.** “What this does not show (frozen): that any corrected program would predict well — 022 builds none, and the full composition is the identity endpoint only — or which of 016/017's reductions carries σ R.” (022:252–253)
- **Claim.** C002 unchanged at `LOCALIZED` (022:257–259).
- **Hand-off.** “Whether the localized block-0 pathways can be written as a compact weight-derived program is left to a later experiment.” (022:256–257)
- **Evidence.** `preregistration.md`, the lock, `calibration-v1.json`, `confirmation-v1.json`, `evidence/final-report-2026-09-24.md`, `evidence/confirmation-record-2026-09-24.json`, the Y2 table, the `replicate-021` record and report.

### 023 — Prospective block-0 completion

- **Question.** “The question: does the completed, fully weight-derived program `P1` prospectively recover essentially all of the explainable gap between Level 0 and the measured-`Δx3` ceiling, on new cues in the exposed frames (Y1) and in new frames (Y2), in cue-final and coordinated frames?” (023:12–14)
- **Protocol.** Design revision 2 `5b38aba`, plan revision 1 `3a795fb` (023:6–10). Extract at `2111271`, freeze at `89536b0`, calibrate at `4e5deeb`, lock at `13b8d39` (installed in `f2294ab`), each once; “The single `confirm` ran at `c7efec7` (17:38:14–17:49:23Z, exit 0).” (023:138). Closed in `b45a646` (2026-09-24).
- **Data.** Fresh: 24 new cues (8 determiner-like, 8 quantity, 8 adjective) and 18 new frames (023:47); “All 3,060 frozen prompts executed exactly once” (023:159). Spent: 022's local table, extracted once into the committed exposed-cells artifact (023:56–58, 91).
- **Frozen outcome space.** Per condition, “decided in the order NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS” (experiments/023-block0-completion/preregistration.md:19); “Aggregate: none: the four condition results are the result, each read on its own; Y1 and Y2 are never pooled; no all-pass requirement.” (experiments/023-block0-completion/preregistration.md:27). The statistic is `g = (SSE0 − SSE1) / (SSE0 − SSEC)`, and a PASS needs `g ≥ max(F, 0.90)` (023:32, 39).
- **Result — preregistered, one confirm run** (023:140–145):

  | condition | g | F (locked) | result |
  |---|---|---|---|
  | Y1/cue_final | 0.9955204329648195 | 0.9968974947302105 | `ENVELOPE_ONLY_FAILURE` |
  | Y1/coordinated | 0.9987141803462269 | 0.9986343903419131 | `PASS` |
  | Y2/cue_final | 0.9974185656823554 | 0.9950528086546427 | `PASS` |
  | Y2/coordinated | 0.9965628370581808 | 0.9951864564481918 | `PASS` |

  “Primary result: three PASS and one ENVELOPE_ONLY_FAILURE.” (023:152) The README reads the Y1/cue_final result as “neither simply a failure nor "basically a pass"” (023:155). Y1/coordinated passed by +7.979e-5 in exact arithmetic (023:156–157). The Y2 conditions are frame-conditional, each frame at its own stage-1 reference state (docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md:214).
- **Descriptive (no outcome force).**
  - “There is no aggregate PASS/FAIL criterion, and the 0.9178 joint pass rate is descriptive only.” (023:149–150)
  - The `" shiny"` exclusion: “Its preplanned descriptive exclusion gives `g = 0.9987161` (863 pairs) and does not change the PASS.” (023:164–165)
  - “Y1 cue-final by stratum: adjective 0.9926, determiner-like 0.9959, quantity 0.9974; both templates about 0.9955.” (023:179)
  - “Cheaper block-0 rules recover far less in cue-final frames: value-only about 0.50, six heads 0.91–0.94, linear response about 0.76. So the exact block-0 rule is what completes the program.” (023:180–181)
  - The gap and R² columns of the result table carry no PASS rule; R²_C is 0.9531–0.9613 (023:140–145). The README calls only the `ceiling_limited` flag descriptive explicitly (023:42).
- **Scope and not claimed.** “The claim covers new determiner-like, quantity and adjective cues and new frames only. There is no claim about possessive or pronoun cues, and none that the downstream readout is complete (`R²_C` 0.95–0.96). No claim file was changed.” (023:183–185)
- **Errata.** None. **Hand-off.** None stated in 023's README; 024's design cites 023's closure and a post-closure diagnostic (next entry).
- **Evidence.** `evidence/` (extract, freeze, calibrate, lock, confirm with `prompts.jsonl`, the reviews, report, closure), `final-report-2026-09-24.md`, `confirmation-record-2026-09-24.json`; the stage-2 measurements are local only (`.gitignore`: `outputs/experiment-023/*`).

### 023 post-closure diagnostic — context, exploratory

- **Where it is recorded.** Only in 024's design, `docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md`, section “Post-hoc background (hypothesis generation only; not Experiment 024 evidence)” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:33). No committed artifact holds its numbers; they exist only in that prose.
- **Its status.** “Everything in this section is post hoc, computed on already-spent data. It motivates the design and is never evidence for its hypothesis.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:35–36)
- **What it reports** (all exploratory):
  - “The upstream residual is `R`.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:38) Restoring the exact layers 1–2 removes 100 % of `R²_C − R²₁`, and “The excess is small: 0.00038–0.00131 of the variance.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:38–41)
  - “The readout residual is the frozen attention of blocks 4 and 5.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:42) “Making block-5 attention exact removes 31–61 % of the readout residual; making block-4 attention exact removes the rest.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:46–47)
  - On 023's fresh cues the extra readout error concentrated in the quantity stratum, “1.6–2.3× the error of the exposed quantity cues, in absolute and in normalized terms.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:50)
  - “Nounness had the strongest association.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:56) Across 163 measured cues: “Spearman +0.53 over all cues, +0.50 within the 139 exposed cues alone.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:56–58)
- **Note.** The design does not separate which background items come from the diagnostic and which from the exposed-only design spike (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:5–8, 15).

### 024 — Readout error predicted by an operational nounness score

- **Question.** “The question: does a frozen, weight-derived operational nounness score prospectively predict, on 40 never-executed cue words, how much the frozen block-4/5 attention readout errs in the 108 exposed frames?” (024:64–65)
- **Protocol.** Design revision 2 `9d03dee`, plan revision 1 `608088c` (024:58–62). Freeze at `44c3c2b`, calibrate at `bf0049c`, lock at `be74d23` (installed in `3affe55`), confirm at `af160ce`, report at `b80409b`, each once; “No incident occurred and nothing was retried.” (024:255). Closed in `eb5f6e8`.
- **Data.** Fresh: 40 never-executed cues in five classes of 8, “4,320 keys (40 cues × 108 exposed frames), sha256 `fcc437fc…`, with 0 overlap with the 39,312 spent keys” (024:147–148). Spent: the calibration used 023's committed exposed cells and the weights, over 139 exposed cues (024:172, 181).
- **Frozen outcome space** (experiments/024-readout-routing-nounness/preregistration.md:33–39): an incident writes no result; `NOT_INTERPRETABLE`; `NOUNNESS_PREDICTION_NOT_ESTABLISHED`; `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED`; `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.
- **Result — preregistered, one confirm run** (024:274–279):
  - primary Spearman ρ, nounness against fresh per-cue MSE: `0.6108818011257036` (= 1628/2665) against ρ ≥ `0.3136960600375234` — PASS;
  - E–N guard: `D_EN = 0.5680373703974243`; K = 1 of 12,870 against K ≤ 321 — PASS;
  - frozen outcome `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.
- **The README's reading of the PASS.** “The frozen rule was ρ ≥ max(F_ρ, null₉₇.₅), with F_ρ = `0.24411074612857814` and null₉₇.₅ = `0.3136960600375234`.” (024:282–283) “The stored reading "at least as strongly as the exposed-like relationship" is not the formal interpretation.” (024:284–285)
- **Descriptive (no outcome force)** (024:309–326):
  - “MSE by class (mean): N 0.0714, B 0.1189, D 0.1127, C 0.1340, E 0.1250.” (024:310)
  - class contrasts on log MSE: noun effect +0.538 [+0.395, +0.677]; measure effect among nouns −0.126, interval spanning zero; plurality effect among nouns +0.076, interval spanning zero (024:311–318);
  - “Subsets: ρ is 0.586 on cue-final frames and 0.698 on coordinated frames; on normalized MSE it is 0.592.” (024:319)
  - “Block-4 attention's share of the readout error: 0.58–0.73 by class; block 5 carries the rest.” (024:324)
  - “The extrapolation flags: 23 of 40 fresh cues lie beyond the calibration's nounness maximum (0 N, 6 B, 4 D, 8 C, 5 E).” (024:325–326)
- **Calibration — exploratory, spent exposed data.** The exposed line “log MSE = `-2.481260357420486` + `1.2992017464639374` · nounness” (024:179–182), over 139 cues; “The exposed Spearman is `0.5296617364493499`.” (024:179–182)
- **Limitations and not claimed.** “The result is predictive, not causal.” (024:373) “What was measured is the error of a frozen approximation, not directly observed rerouting.” (024:374) “The guard's formal scope is limited to those two preregistered simple alternatives. E and N differ in other lexical and semantic properties besides nounness: all eight N cues are adjectives, and the E cues are concrete, mostly animate nouns.” (024:295–297) “23 of the 40 cues lie beyond the calibration's nounness range, so the test is partly an extrapolation.” (024:300–301) Not claimed: “that nounness causes rerouting, or that rerouting was measured directly;” (024:303) “that the model detects nouns;” (024:304) “that nounness is a general linguistic representation;” (024:305) “that E-vs-N rules out all lexical or semantic confounds;” (024:306) “that 024 explains all routing-regime changes, or establishes general algorithm choice.” (024:307)
- **Errata.** None after closure. **Hand-off.** None stated in 024's README; 025's design follows 024's closure (025:63–65).
- **Planned but not done at closure.** 024's plan lists “The root `README.md` registry entry, at closure only.” (docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md:616). The entry was missing until this step; see [Record caveats](#record-caveats).
- **Evidence.** `evidence/` (freeze, calibrate, lock, post-install review, confirm with `confirm_prompts.jsonl`, the reviews, report, closure), `final-report-2026-09-25.md`, `confirmation-record-2026-09-25.json`; the measurements stay local (024:354–357).

### 025 — Nounness-direction intervention on the cue embedding

- **Question.** “The question: Experiment 024 showed that a frozen, weight-derived nounness score *predicts* the frozen block-4/5 readout error. Does *moving* a fresh cue's input embedding along that score's own direction *cause* the error, and the directly captured L4/L5 routing, to change in the predicted direction?” (025:63–65)
- **Protocol.** Design revision 1 `c0885e5` (corrected in `26c9925`), plan revision 1 `7d90d28` (025:55–59). Freeze at `c765148` (installed in `57f1f98`), lock at `d3ecbd6` (installed in `5ee7a74`), confirm at `6f72d7d`, report at `c8d9fad`, each once. There is no calibration phase: “Every threshold is a fixed count (27 of 40), the direction is frozen from 024's record, and the doses come from 024's locked scores.” (docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md:360–361). Closed in `d7c5e62`.
- **Data.** Fresh: “40 cues × 108 exposed frames × 21 conditions = 90,720 runs, each under a condition-tagged key `frame_id|word|token_id|condition`.” (025:78–79) 20 adjectives and 20 ordinary singular nouns, picked mechanically (025:99–104). None of the 43,632 earlier spent keys was executed (025:221–222).
- **Frozen outcome space** (experiments/025-nounness-direction-intervention/preregistration.md:29–35): `NOT_INTERPRETABLE`; `CAUSAL_EFFECT_NOT_ESTABLISHED` (A fails); `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC` (A passes, B fails); `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED` (A and B pass, G fails); `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR` (all pass). Each statistic passes at ≥ 27 of 40 strictly positive.
- **Result — preregistered, one confirm run.** “The official result is `CAUSAL_EFFECT_NOT_ESTABLISHED`: A = 26/40, B = 23/40, G = 37/40, threshold = 27, reference tail = 0.01923865414210013.” (025:244–245) The preregistered reading: “the preregistered Experiment 025 test did not establish that moving fresh cue embeddings in the +nounness direction consistently causes greater frozen block-4/5 readout-approximation error than the matched −nounness intervention across the frozen 40-cue population.” (025:23–25) “The label records that this test did not establish the effect. It is not a finding of no causal effect.” (025:249) “A failed, so the frozen hierarchy assigns this label whatever B and G are. G's 37/40 cannot change it.” (025:250)
- **Descriptive (no outcome force)** (025:252–262):
  - “the adjective/noun stratum split: A 20/20 and 6/20, B 20/20 and 3/20, G 20/20 and 17/20. The stratum counts cannot change the label, and no stratum-level claim is made;” (025:254–255)
  - the causal fraction 0.341; the block-4/5 ladder; “the related correlations, such as Spearman(A, s₀) = −0.83 across the 40 cues, reported descriptively in the independent confirmation review;” (025:257–260)
- **Limitations and not claimed.** The population was “40 fresh cues picked mechanically (not sampled at random)” (025:265–266). “The intervention is narrow: one norm-preserving rotation of the cue's input embedding, along one frozen weight-derived direction, at two doses.” (025:267–268) “What was measured is the error of a frozen approximation, and one frozen L4/L5 routing summary. Neither is directly observed rerouting.” (025:269–270) “The nounness direction is partly aligned with plurality (cos +0.235).” (025:271)
- **Provenance note.** The exit-127 shell event before the launch of record: “It was not a scientific attempt.” (025:238)
- **Hand-off.** “The endpoint. Experiment 025 is the endpoint of this sequence, as its design fixed in advance. Confirm is not rerun, and no Experiment 026 follows.” (025:279–280)
- **Evidence.** `evidence/` (freeze, lock, confirm with `production_runs.jsonl`, the reviews, report, closure), `final-report-2026-09-26.md`, `confirmation-record-2026-09-26.json`; the stage-2 measurements stay local (025:227–230).

## Record caveats

None of these items changes a frozen result unless its entry says so. They are recorded here so that none of them
disappears during the synthesis.

### Errata

| exp | commit | what it corrects | what stands |
|---|---|---|---|
| 020 | `2320345` | In coordinated frames the program builds the layer-1/2 reference rows through `p_t` instead of the cue position `p_c` validated in 017–019: exposed coordinated `Δc` `R²` −0.146 as wired, against 0.555 as 017 wired it (020:13–15) | “Every recorded Experiment 020 number, digest, report and the closure decision.” (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:55) The ceiling and all cue-final values also stand (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:56–57) |
| 021 | `2320345` | The same wiring. The single coordinated Y2 failure “must not be read as evidence about the correctly wired 017 upstream chain; it is plausibly explained by the wiring defect.” (021:16–17) | The frozen results, labels, floors and digests (021:14–16) |

022's design revision 4 adopted 017's wiring, and withdrew revision 3's C3 because it “rested on the defect” (022:35–39).

### Conclusions narrowed or restated

The map uses the latest wording. The earlier wording is kept here.

| exp | when | earlier wording | latest wording |
|---|---|---|---|
| 006 | in the closure commit `10c8faf` | “so a rank ≤ 4 PCA projection cannot represent the cue-to-readout map, whatever the fitting procedure” (006@9566372:78–79) | “Stated narrowly: Experiment 006 falsified the hypothesis that the cue-to-readout computation can be captured by the leading ≤ 4 *unsupervised* PCA directions of the `L00.MLP` cue representations.” (006:77–79). The root README still carries the earlier wording (see Registries). |
| 007 | after closure, `5f2464a` | “What fails, for every linear map from the token-local `E(w)`, is a small set of singular-selecting determiners (`this`, `another`)” (007@6340a4c:105–107) | “This is a statement about the two estimators tested, not about every possible linear map” (007:109) |
| 012 | after closure, `1b1edff` | “The composite "encoding read + token-local MLP correction" now accounts for what the head reads” (012@fc442f6:117) | “the design does not claim to have decoded the layer-1–2 MLP correction itself, only to have predicted the net correction and its effect on the head's input.” (012:119–120) |
| 020, 021 | after closure, `2320345` | the closure readings that the coordinated template carries most of the shortfall (020:179, 021:168–169), left unedited in the body | qualified by the errata above |
| 021 | first committed at closure, `86f7f8c`, after the confirm; the README heads it as agreed at the floor and lock reviews, which no committed record shows | the design's reading that a negative label means a generalization gap (docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md:335–337) | “too strong for failures caused only by the clamp, and for them it is superseded here” (021:156–158) |
| 024 | at closure | the renderer's PASS prose, “at least as strongly as the exposed-like relationship” (experiments/024-readout-routing-nounness/preregistration.md:18) | “The stored reading "at least as strongly as the exposed-like relationship" is not the formal interpretation.” (024:284–285) |

### Digest mismatches, investigated

Both are explained exactly. `checks/digest_investigation.py` rebuilds the earlier states from the local, gitignored
final states and reproduces every recorded digest bit for bit. In both cases the mismatch is between two states of
the same run. No data differs.

| exp | README records | other records carry | explanation |
|---|---|---|---|
| 010 | `a9f8d703…` (010:49–50) | `b8986dd3…` in the 011 and 012 inherited extracts (`source.state_sha256`) | The README gives the state at the end of `explore`, as the explore log prints it. Thirteen seconds later, `report` rewrote the state with `phases.report` complete. The extracts were built from that final state. |
| 012 | `069a6306…`, called the Tier A state (012:130–131) | `3eb1db00…` in the lock's `tier_a_results_sha256` and the explore log | The README's value is the state after the `lock` phase, which wrote `state["lock"]` and `phases.lock`. The end-of-explore state is `3eb1db00…`. The README mislabels a later state of the same run. |

### Inconsistencies in the records

Each item has one of four classes:
- **provenance** — it bears on when or how something was recorded;
- **outcome** — it bears on how a frozen outcome is named or read;
- **interpretation** — a descriptive statement that the committed evidence does not bear out;
- **cosmetic** — wording, labels, dates or formatting.

| exp | class | item |
|---|---|---|
| 005 | provenance | C002 was created after the confirm, at `LOCALIZED` and with origin exploratory (research/claims/C002-count-cued-noun-number-circuit.md:5–9). The design had planned it at the lock, at `PROPOSED` and with origin confirmatory (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:955–956). |
| 005 | cosmetic | The final report names a third axis, new frames `NOT_GENERALIZED` (experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md:131). The design's outcome section defines two axes (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:859–863), and revision 5 defines the related label `CIRCUIT_NOT_GENERALIZED` (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:87, 879–881). |
| 005 | cosmetic | The calibration population is called 114 single-token holdout nouns (005:49), elsewhere 114 cases (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:623) and 19 nouns (research/claims/C002-count-cued-noun-number-circuit.md:29). The design's status line is stale. |
| 006 | outcome | The label `QUALITY_GATE_FAILED` (006:72) is not in the design's outcome list (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:341–353). |
| 006 | interpretation | The root README kept the pre-narrowing wording and said the confirmation set “remains unexecuted” (README.md:21), although 007 executed it (007:112–113). Fixed in the root README after this map was first committed (`5fc98f6`). |
| 006, 007 | cosmetic | The designs' status lines are stale (they predate the experiments). |
| 008 | cosmetic | Frozen Rule 4 defines the head-dropped decision (docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md:216–218); only the label string `HEAD_DROPPED` (008:61) comes from the code. The README says the frozen rules “did not anticipate this pattern” (008:68). |
| 009 | interpretation | “`R3` for the numerals” (009:80), while the final report shows stage `T` for `fourteen` and `sixteen` (experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md:84, 95). |
| 009, 016, 017 | provenance | No reviewer sign-off before the confirm is recorded, although the design order includes one (for example docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:292–294). 013, 014, 018 and 019 record it; 015 records that the reviewer cleared the install. |
| 010 | interpretation | Three statements do not match the committed exploration report. `n_80` is “320–700 of 2048 everywhere” (010:63), but the report shows 726 (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:39). The layer-1 heads and MLP are said to contribute at most a few hundredths (010:94–95), but the report shows `L01.MLP` 0.102 for `a` (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:105). The neuron 1133 and 148 statements (010:100–102) do not match the report's lists (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:120–141). |
| 010 | cosmetic | The README cites design revision 2 (010:5) and revision 3 (010:41); the root README says revision 3. |
| 011, 012, 013 | provenance | Some README figures are not in the committed reports: identity maxima, residual correlations, per-template means and ledger values (for example 011:71–74, 84; 012:89–90, 104–107). |
| 011, 012, 013 | cosmetic | The READMEs' cue-effect notation, for example “79/80 nouns” (011:72), differs from the reports', for example “cue effect 79/72” (experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md:28). |
| 012 | outcome | The design read a Y2 pass as closing “the account up to the head” (docs/superpowers/specs/2026-09-19-experiment-012-layer-correction-token-local-design.md:242). The README's latest wording is weaker (see Conclusions narrowed). |
| 013 | interpretation | The Y1 sign-miss list names `last` (013:96–97), but the report shows `last` with the same sign on both sides and `limited` with opposite signs (experiments/013-attention-paths-frozen-pattern/evidence/final-report-2026-09-19.md:69, 71). |
| 013–016 | provenance | The Y2 prediction tables were written to the results state with their digest, not committed before scoring (013:54–55, 014:53–54, 015:61–62, 016:64–65). |
| 014 | interpretation | “The three strong predictions fired in every frame” (014:107), but `hundreds` fired in 39 of 42 (014:108). |
| 015 | provenance | The README paragraph “Scoring conventions fixed before Tier A” (015:76–79) first appears in the closure commit `08f4ca4`, after the confirm. The conventions themselves are in the design at `4babcce` (docs/superpowers/specs/2026-09-19-experiment-015-attention-pattern-token-local-design.md:320–324). The Tier A text was also first committed at closure, and there is no exploration report in evidence. |
| 015 | cosmetic | The ladder and rungs are labelled “on the fresh pairs” (015:134, 138), but match the Y1 set alone (experiments/015-attention-pattern-token-local/evidence/final-report-2026-09-20.md:55–63). |
| 017 | interpretation | “no frame falling below row entry R² 0.985” (017:163–164), while the exposed per-frame minimum is 0.970 (017:117); 0.985 holds for the twelve new frames only. |
| 018 | outcome | The README reads Y2 “as the design fixed it” (018:115–117), but the design fixes no reading for a Y2 guard failure (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:359–368). |
| 018 | interpretation | “neuron 1987 ranks first everywhere” (018:150) stands beside “in the four new quantifier frames neuron 1102 is first” (018:139); the README does not say whether the two statements use the same ranking. |
| 018 | provenance | “only the two exempt files changed since `9b629bf`” (018:81), but git shows a README commit, `b956d1e`, in between. |
| 019 | interpretation | The design still calls the witness a “ceiling” (docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md:29–31); the README says it is never a ceiling or upper bound (019:165–166). |
| 019 | cosmetic | The status date (019:79) precedes the closing commit's date; the top-neuron counts cover 86 of 90 frames (019:87–88); the Y2 membership margin is not printed but follows from 0.867 and 0.375. |
| 020 | cosmetic | Two coordinated Level-0 values, 0.2729 (020:119) and −0.146 (020:15), both describe 020's wiring, on different pair sets; line 15 does not name its population. |
| 021 | cosmetic | The report's line on “33192 keys” (experiments/021-corrected-readout-confirmation/evidence/final-report-2026-09-23.md:17) and its list of clamped Y3 conditions (experiments/021-corrected-readout-confirmation/evidence/final-report-2026-09-23.md:20) differ in form from the README. |
| 021 | provenance | The README's interpretation section, headed “agreed at the floor and lock reviews” (021:151), supersedes the design's frozen reading for clamp-bound failures (docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md:335–337). It first appears in the closure commit `86f7f8c`, after the confirm at `b7e8861`, and no committed review record shows the earlier agreement. |
| 022 | cosmetic | The Shapley shares sit under the heading of the primary evidence although only the four condition statistics carry outcome force; the summary rounds; the envelope indexing is 0-based in the README (022:181) and 1-based in the preregistration. |
| 024 | interpretation | 024's design heads its account of 023 “what Experiment 023 settled (confirmed)” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:19) and says “Upstream is effectively reconstructed.” (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:25–27), with `g ≥ 0.9955` in every condition. 023's own README records one `ENVELOPE_ONLY_FAILURE` and reads it as “neither simply a failure nor "basically a pass"” (023:155). The map follows 023's README. |
| 024 | provenance | “The repository tracks no binary measurement artifact” (024:356) holds for 024's own directory only. At `eb5f6e8` the repository already tracked 022's and 023's binary tables, among them 023's `exposed-cells.f64`, which is derived from 022's exposed measurements. |
| 024 | cosmetic | The README's outcome list (024:83–86) omits the `NOT_INTERPRETABLE` and incident rows of the preregistration's table (experiments/024-readout-routing-nounness/preregistration.md:33–39). |
| 025 | provenance | “The repository tracks no binary measurement artifact” (025:229–230) is wrong: six binary tensor files from the smoke test and the reviews are tracked in 025's evidence, three of them holding values derived from the measurements. It is corrected after closure (see Post-closure documentation corrections). |
| 007, 012 | provenance | Two hand-off sentences were added after closure: 007's (007:112–113) in `5f2464a`, and 012's (012:125) in `1b1edff`. |
| 025 | cosmetic | The report's evidence files are dated 2026-09-26, the day of the confirm, although the report ran on 2026-09-27; 024 followed the same convention. |
| 005–025 | cosmetic | The copied final reports print `report: not_started`, because the renderer runs before the phase completes. This is documented in 005 (005:146–148) and 023 (023:167–171). |

### Misleading commit subjects

Six lock commits carry the subject “Refactor code structure for improved readability and maintainability” (message of c01126f). Each adds
only `predictions.md` and `preregistration-lock.json`, and each README identifies it as that experiment's lock
commit:

| exp | commit |
|---|---|
| 013 | `c01126f` |
| 014 | `d008dae` |
| 015 | `5ed51c9` |
| 016 | `dc4e2a7` |
| 017 | `4b6f766` |
| 018 | `6fa4c48` |

The pushed history is not rewritten; the mismatch is recorded here instead.

### Post-closure documentation corrections

| exp | commit | kind |
|---|---|---|
| 005 | `6bc2d6a` | archival notes; no artifact rewritten (005:144) |
| 007 | `5f2464a` | the interpretation narrowed to the estimators tested |
| 012 | `1b1edff` | the claim restated at its safe strength |
| 020, 021 | `2320345` | errata: the coordinated reference-row wiring |
| 025 | `e34f16d`, `8fe3ee9` | the tracked-binary-files clause corrected, with an erratum file; the original wording stays visible, struck through; the erratum's own scope wording made exact in the second commit |
| root README | `4f1f97d` | entries for 024 and 025 added |
| root README | the commit after `5fc98f6` | the 006 sentence updated to the narrowed wording, with the confirmation set's later execution in 007 |

### Registries

- **Claims.** C002 is `LOCALIZED` and `ACTIVE`, last reviewed 2026-09-18, with Experiment 005 as its only related
  experiment (research/claims/C002-count-cued-noun-number-circuit.md:5–10). Its status history has one row
  (research/claims/C002-count-cued-noun-number-circuit.md:144). No experiment from 006 to 025 is recorded as evidence.
  It is deliberately not updated in this step: the claims registry follows the checked synthesis. C001 concerns
  Experiments 001–004, outside this map.
- **Anomalies.** “No anomalies have been registered.” (research/anomalies/README.md:31)
- **Root README.** Its experiment entries ended at 023 (README.md:21), although 024's plan required an entry at
  closure: “The root `README.md` registry entry, at closure only.” (docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md:616)
  Entries for 024 and 025 were added in `4f1f97d`. Its 006 sentence kept the pre-narrowing wording and said the
  confirmation set “remains unexecuted” (README.md:21). In the commit after `5fc98f6`, it was updated to the narrowed
  wording and now notes that 007 executed the set; 006's own README and artifacts are unchanged.

## What this map does not do

- **It interprets nothing.** What the sequence established, and whether a genuine unresolved mechanistic gap remains,
  is step 2.
- **It does not change a claim.** C002 stays as it is until the checked synthesis says whether the evidence from
  006–025 strengthens it, narrows it, supersedes it or leaves it alone.
- **It proposes no experiment.** Experiment 025's preregistered stopping rule ended the sequence.
- **It does not rewrite history.** Commit messages, closed READMEs and frozen artifacts stay as they are.
  Corrections are recorded as corrections.

## Checks

- **`checks/check_source_map.py`** (output in `checks/check_source_map.out`):
  - every quote against its cited lines, at `d7c5e62` or at the earlier commit its citation names;
  - every citation parsed;
  - every cited file and line, every commit hash, and every outcome label shown to exist;
  - in the overview, every label found in its experiment's own README.

  It fails on a quote without a citation and on straight-quoted prose. It does not check paraphrases, numbers outside
  quotes, or classifications.
- **`checks/digest_investigation.py`** (output in `checks/digest_investigation.out`) rebuilds the earlier results
  states of 010 and 012 from their local final states. It reproduces every recorded digest, including the ones the
  explore logs print.
- **`checks/independent-review-2026-09-27.md`** is the independent, read-only review of this map, of the 025
  correction and of the root README entries. Its verdict was PASS WITH NOTES, with no blocker. The file lists how
  each finding was addressed.
- **A later correction to this map (2026-09-27).** Two lines called 006's 708/708 paired signs and its correlation of
  0.756 an E-patch result. They are the fixed circuit's isolation fidelity, P3 (006:58–59, 86–89), and now say so.
  The independent review of the C002 claim review found the mislabel (`../claims/reviews/`).

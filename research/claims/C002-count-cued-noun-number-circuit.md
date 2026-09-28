# C002: A three-stage circuit (L00.MLP → L03.H04 → L04/L05 MLPs) localizes count-cued noun number selection in pinned Pythia-70M

## Metadata

- Current supported maturity: `LOCALIZED`
- Disposition: `ACTIVE`
- Created: 2026-09-18
- Last reviewed: 2026-09-27
- Origin: exploratory (mechanism found on development data in Experiment 005 Tier A; predictions preregistered in the protocol v2 lock before the single confirmation run)
- Related experiments: Experiment 005 (behavior candidate screen protocol v1 as its precursor); Experiments 006–025, reviewed in [the 2026-09-27 claim review](reviews/C002-review-2026-09-27.md)
- Related claims: [C003](C003-frame-conditioned-decoded-program.md), registered 2026-09-28, a separate claim describing the decoded predictive program; it sits beside this claim and does not supersede it

## Hypothesis

In pinned `EleutherAI/pythia-70m-deduped`, the preference between a singular and a
plural noun form after a numeral or quantifier cue is carried by: the layer-0 MLP
at the cue position (E, a pure token function), attention head `L03.H04` at the
noun-prediction position, which transports the cue's number information when the
cue is not the final token (T), and the layer-4 and layer-5 MLPs at the
noun-prediction position (R), which convert it into the singular-versus-plural
logit contrast. Exact counterfactual replacement of these components moves the
contrast toward the counterfactual, holding them fixed while neutralizing every
other head and MLP output retains most of the contrast shift, and the E → T → R
chain mediates the effect (dedicated transport, hypothesis H1 of the design).

## Scope

- Model: `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32, TransformerBridge without compatibility mode, instrumented forwards with `use_attn_result=True`.
- Behavior: fixed-orientation contrast `c(x) = log P(singular) − log P(plural)` for matched cue pairs (`one`/`two`, `each`/`several`) in the three screening templates (twelve manifest prompts) and, prospectively, in six new frames and with twelve new cue words; single-token nouns of three regular rule classes.
- Data: development (20 nouns), holdout (19 single-token nouns), reserve (20 nouns), all from `screening/behavior-candidates/manifest-v1.json`; extension set `experiments/005-regular-plural-mechanism/extension-v1.json`.
- Interventions: exact replacement (counterfactual, freeze, cross-frame and cross-cue resample), pair-centered neutralization, isolation of the declared set, random component sets.
- Not in scope: other checkpoints or models, other templates, multi-token nouns, full-vocabulary top-1 behavior, the compact executable program (capped in protocol v2; see C002's "Not supported").

## Operational definitions

- Recovery `R(S)`: mean aligned bidirectional patched shift divided by mean full contrast shift over a stratum (design section "Fixed coordinates").
- Isolation faithfulness `F(S)`: retained contrast difference when every other head/MLP output at the noun position (and the cue position) is replaced by its pair-centered midpoint, as a fraction of the full shift.
- Blocked fraction, `m_T`, `m_R`, `m_R|T`, compensation ratio: as defined in the design (revision 5).
- Primary condition, flips, positive pairs: as in the screen.

## Predictions

Preregistered in `experiments/005-regular-plural-mechanism/preregistration-lock.json` (content sha256 `ec2a8879bab35fd6e4d45484321595aed565fb40f19bf608daf2574ea50ccc97`, commit `30e8b84`): floors and bands for families B1–B2, P1–P9, S1–S3, and X1–X4, with the decompilation axis capped from the outset.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | Experiment 005 Tier A (`discover`, development data, exploratory) | supporting | `experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md` | H1 unambiguous (`L03.H04` alone 0.816, blocked 0.865, `m_T` 0.90, `m_R` 0.99); E = `L00.MLP` (0.535 of the transport-input axis); k = 3 recovery 0.982, isolation 0.783; program under-predicts the quantifier mean by 1.44 nats (protocol v1 `NO_COMPACT_MECHANISM`) | code `83d1ae4`; results state `d5977a11…7795b` |
| E002 | Experiment 005 Tier B (`calibrate`, 114 single-token holdout cases; readout generalization to new nouns) | supporting | `experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md` (calibration section) | every primary circuit floor passed: P1 0.977, P3 0.782 (retention 91/114), P4 0.882, P5 0.816 / 0.865, P6 0.854 / 0.070, P7 0.976 / −0.006, P8 loss 0.823, P9 0.987 / 0.900 / 0.074; S1 (shared axis cosine 0.292) failed; program RMSE_B 0.609 | code `009607d`; results state `fe3833ed…47f0` |
| E003 | Experiment 005 Tier C (`confirm`, reserve nouns and extension prompts, single run under the lock) | null (precondition failed) | `experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md` (confirmation section) | outcome `BEHAVIOR_NOT_REPLICATED`: B1 98/120 primary correct (floor 103) and 75/120 flips (floor 96); by the preregistered rule no other family is confirmatory. Recorded non-confirmatory values: P1 0.985, P3 0.795 (0.962 / 0.760 / 0.672), P4 0.873, P5 0.817 / 0.869, P6 0.840 / 0.074, P7 0.984 / −0.007, P8 0.826, P9 0.987 / 0.900 / 0.074, every P band hit except P3's retention count (71/120) and the secondary P8 compensation band; new frames X1 passed (120/120 positive pairs, 12/12 number variables) with P1 0.980, P3 0.824, P4 0.948, P5 0.822 / 0.872, P8 0.796, P9 0.937 / 1.079 / 0.286; cue words: E-patch shifts track behavioral shifts for every word, lexicon Spearman 0.699 (floor 0.70) | code `30e8b84`; lock `ec2a8879…cc97`; results state `5a0c6836…f6bf` |
| E004 | Instrumentation contract and numerics | methodological | `experiments/005-regular-plural-mechanism/README.md` (numerical note); `outputs/experiment-005/results.json` `discovery.a0_contract_test`, `a1` | pinned Bridge contract test passed; A1 matched the screen per case to 0.0; instrumented-path readouts differ from plain forwards by up to 4.5e-3 nats; direct-effect decomposition identity within 1e-4 nats | code `83d1ae4` |
| E005 | Experiment 006 (Tier A only; no lock, no confirmation) | null | `experiments/006-low-rank-cue-decompilation/evidence/exploration-report-2026-09-18.md`; the 006 design | Kind C, a route to this claim's Level 3 that never ran. 006 tied this claim's review to a `CIRCUIT_ONLY` outcome, which needs `CIRCUIT_PASS`: P1, P3-fidelity, P4, P5, P7, P8 and P9 all passing on fresh nouns and on fresh frames (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:344–347, 351, 360–362). 006 stopped at its quality gate and wrote no lock (006:71). At Tier A on the exposed pool [exploratory]: P1 0.978, P3 F 0.804 with 708/708 paired signs and correlation 0.756, P4 0.915, P5 0.819 / 0.870, P8 0.813 (006:58–60). | explore `adea65f`; closed `10c8faf` |
| E006 | Experiment 007, circuit families (reported outside its outcome) | supporting | `experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md` | Kind C, route not met, and Level-2 replication; criterion-level passes with no outcome force. On fresh nouns over the manifest frames and on fresh frames: P1 0.981 / 0.983; P3 F 0.781 / 0.773 with 120/120 paired signs; P4 0.887 / 0.913; P5 0.825 / 0.793 alone, blocked 0.865 / 0.845; P8 0.822 / 0.873; P7 and P9 passed (experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md:109, 115; 007:98–100). The fresh-noun set records this claim's 98 %, 78 % and 82 % again on new nouns in its own frames; the fresh frames lie beyond its declared frames. 007's rule reviews this claim only if every family passes on both sets (docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md:251–252); P3 failed (E007), so this route did not establish Level 3. | lock and confirm at `ede0f53`, results state `118dffa0…`; closed `6340a4c` |
| E007 | Experiment 007, P3's correlation floor | null | `experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md` | Kind C; a failed floor. P3's pair-level correlation was 0.696 (fresh nouns) and 0.840 (fresh frames) against 0.90 (007:98–100). The isolated circuit keeps its mean retention (F ≈ 0.78) and every sign, but not the pair-by-pair magnitude, a stronger property than this claim states. It is why 007's route to Level 3 was not met. 006's Tier A had shown the same pattern on the exposed pool: 708/708 signs, correlation 0.756 [exploratory] (006:58–59). | as E006 |
| E008 | Experiment 007, program axis | methodological (scope-limiting) | `experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md` | Bears on generalization beyond this claim's declared cue scope, not on the claim as written: 007's 24 cue words are new words in new frames, none of them among this claim's twelve (006:16–18). Kind M, P. On them the E-patch's token-mean shift is within 0.2 nats of the real cue swap's for 21; the other three differ by 0.58 (`any`), 0.37 (`no`) and 0.60 (`single`), computed in the 2026-09-27 review from the report's table (experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md:72–97) [descriptive]. Y1, the rank-1 program's prediction of those shifts, failed with `PROGRAM_NOT_SUPPORTED` (007:87–94) [unestablished]; Y2 and Y3 passed at criterion level. | as E006 |
| E009 | Experiment 008 (discovery only) | supporting | `experiments/008-cue-suppression-localization/evidence/exploration-report-2026-09-18.md` | Kind M; exploratory. For `this`, `a` and `another`, `L03.H04` attends to the cue as much as in the reference prompt but passes on 0.07–0.29 of the number signal, so the selection sits in the head's value/output pathway (008:86–91). The axis part of their encoding alone drives 0.50–0.64 of the plural cue's effect, and the whole encoding about none: `NONLINEAR_GATING` (008:57–59). Bears on T, and on the third competing explanation. | explore `d324755`; closed `40265b5` |
| E010 | Experiment 009 | supporting | `experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md` | Kind M, P; Y1 prospective, given the measured head input; Y2 `TRANSPORT_RULE_FAILED` [unestablished]; Y3 `NOT_LOCKED` (009:69–79). The head's selective transport is “a fixed-normalization linear read-out of components already present in the cue residual it receives” (009:84–85), including 23 never-seen cues. Bears on T. | confirm `78d71c2`; closed `27083c6` |
| E011 | Experiment 010 (discovery only) | supporting | `experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md` | Kind M; exploratory (summary `MIXED`). For `this`, `a`, `another` and `every`, the encoding carries an on-axis number signal and an opposing off-axis part; the signal is spread over 320–700 of 2048 neurons (010:85–89, 97). “Layers 1–2 amplify strong number signals but do not create the suppression.” (010:93) For this claim's plural cues, `two` and `several` (the `cardinal:pl` and `quantifier:pl` rows), layers 1–2 add a net +0.081 and +0.187 of the plural cue's head change to encoding reads of 0.910 and 0.770 (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:87, 91). Bears on E, on the link from E to T, and on the third competing explanation. | explore `0a6836b`; closed `16c6491` |
| E012 | Experiment 011 | supporting | `experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md` | Kind P, M; Y1 prospective, from the weights alone; Y2 prospective, given the measured head input. The zero-parameter encoding read predicts the head's transport of 24 never-seen cues in six new frames: Spearman 0.848, MAE 0.077 (011:75–76). The head's linear read-out replicates: 0.974 / 0.076 (011:87–89). The read omits the layers-1–2 contribution, which is “positive on average (+0.18)” (011:104). Bears on T, and on the link from E to T. | confirm `85a6c04`; closed `0358c8e` |
| E013 | Experiments 012, 013, 015, 016 | supporting | `evidence/final-report-2026-09-19.md` of 012 and 013; `evidence/final-report-2026-09-20.md` of 015 and 016 | Kind M, P; prospective; 013 and 016 Y2 conditional on the new frames' reference states; 015 Y2 unestablished (015:109–110). Between the encoding and the head, layers 1–2 at the cue position add a predicted correction: token-local MLPs (012), the heads' value paths (013), and the heads' pattern change with the frame's reference channels (015, 016). The MLPs respond to the off-axis part of the encoding, 0.28 against 0.03 [descriptive] (012:103–104). This decomposes the link from E to T as this claim measured it, with layers 1–2 recomputing. | confirms `5cea6c4`, `c01126f`, `5ed51c9`, `dc4e2a7`; closed `fc442f6`, `9b656e2`, `08f4ca4`, `de61ead` |
| E014 | Experiments 014, 018, 019 | supporting | `evidence/final-report-2026-09-19.md` of 014; `evidence/final-report-2026-09-21.md` of 018 and 019 | Kind M, P; prospective; Y2 conditional; 018 Y2 and Y4 unestablished (018:106, 122); 019 an aggregate result, not a rule for every frame (019:164–165). Block 2's part of that correction depends on the frame. 014 reconstructed block-2 neuron 1987's activation change for new cues at the frame's reference operating point (014:133–137). Channel D is “the frame's operating-point dependence of block 2's MLP as it reaches the transport read” (018:144–145): a locked 256-neuron subset carries 0.86 of its contribution to the decoded read on new cues, while neuron 1987 alone carries 0.22 (018:146–151). 019 predicted how the frame routes that population (019:120, 129–130). Also inside the link from E to T. | confirms `d008dae`, `6fa4c48`, `9033041`; closed `56a0fc5`, `5132174`, `a0d73c8` |
| E015 | Experiment 017 | supporting | `experiments/017-transport-head-pattern/evidence/final-report-2026-09-20.md` | Kind M, P; Y1 prospective; Y2 conditional; Y3 `FROZEN_PATTERN_NOT_REJECTED` (017:126–131). `L03.H04`'s own attention-row change and output change are predicted by a chain whose “only cue-specific input is `ΔE`” (017:14–15): row entries R² 0.997 (Y1) and 0.998 (Y2), `ΔT` pairs 0.999 and 1.000 (017:116–123). Whether the pattern change needs to be modelled at all was not established (Y3). Bears on T. | confirm `4b6f766`; closed `c583405` |
| E016 | Experiments 020, 021, 024 | supporting | `experiments/020-readout-decompilation/evidence/exploration-report-2026-09-22.md`; `experiments/021-corrected-readout-confirmation/evidence/final-report-2026-09-23.md`; `experiments/024-readout-routing-nounness/evidence/final-report-2026-09-25.md` | Kind M, P. 020 exploratory, never confirmed, its ceiling exploratory; 021 Y1 and Y3 prospective, Y2 unestablished, all three qualified by E019, its ceilings descriptive; 024 prospective, predictive not causal (024:373). 020's readout program starts from the residual change `Δx3`. It runs block 3's heads and MLP; block 4 “with attention frozen at the reference and its MLP at the frame's operating point” (020:44–45); and “block 5's heads as the frame's reference rows times the predicted value changes at every key position” (020:45–46), with block 5's MLP. Fed the measured `Δx3`, it reaches R² 0.976 on exposed pairs (020:132) and 0.968 / 0.965 on fresh ones (021:140–141). The downstream shortfall is 0.024 (020:140) and 0.032 / 0.035 (021:141–142). Post-hoc analysis on spent data, cited as background in 024's design, places it in the frozen attention of blocks 4 and 5 (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:33–47) [exploratory]. On 40 new cues its size follows a nounness score, ρ 0.611 against 0.314 (024:277). In 020's design spike, `L04.MLP` and `L05.MLP` contribute about 61 % of `Δc` directly (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:45–49) [exploratory]. Bears on R. The block-5 head term covers every block-5 head, over the value changes at `p_c` and `p_t` (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:112–115). 020 calls it “a small frozen-row second-transport term” (020:43), because of the path from `p_c` in coordinated frames; it bears on T (see *Competing explanations*). | 020 explore `ce3766b`, closed `7f4a4f8`; 021 confirm `b7e8861`, closed `86f7f8c`; 024 confirm `af160ce`, closed `eb5f6e8` |
| E017 | Experiments 022, 023 | methodological (scope-limiting) | `experiments/022-upstream-error-localization/evidence/final-report-2026-09-24.md`; `experiments/023-block0-completion/evidence/final-report-2026-09-24.md` | Bears on generalization beyond this claim's declared cue scope, not on the claim as written. Kind M, P; 022 prospective, all eight conditions pass, Y2 conditional, its coordinated `T` share descriptive; 023 prospective, three passes and one `ENVELOPE_ONLY_FAILURE`, Y2 conditional, its R² values descriptive. The cues are new determiner-like, quantity and adjective cues (023:47), such as `general`, `tons` and `humble`, not numerals or quantifiers (experiments/023-block0-completion/confirmation-v1.json:1). On them the decoded E-only chain explains pair-level `Δc` R² 0.8182 / 0.8146 in cue-final frames and 0.5420 / 0.5719 in coordinated frames (Y1 / Y2), against ceilings of 0.95–0.96; adding block 0 brings it to 0.95–0.96 (023:142–145). Block-0 attention at the cue position accounts for 0.948 / 0.938 of the cue-final gap (022:245–247). In coordinated frames “block 0's change at the target position (`T`, the cue→target pathway) contributes positively” (022:248–249): 0.138 / 0.157 of R² (022:206, 208). So `L00.MLP` alone is not the whole cue stage for these cue classes at the pair level; this is recorded under *Not supported*. It does not measure this claim's pairs. | 022 confirm `4f8f31d`, closed `0818c4c`; 023 confirm `c7efec7`, closed `b45a646` |
| E018 | Experiment 025 | null | `experiments/025-nounness-direction-intervention/evidence/final-report-2026-09-26.md` | No kind: a different intervention; unestablished. “The official result is `CAUSAL_EFFECT_NOT_ESTABLISHED`” (025:244), for a rotation of new cues' input embeddings along 024's nounness direction. It does not test this claim's intervention chain. | confirm `6f72d7d`; closed `d7c5e62` |
| E019 | Post-closure errata of Experiments 020 and 021 | methodological | `evidence/erratum-2026-09-23-coordinated-reference-rows.md` of 020 and of 021 | No kind: an erratum. 020's program built the coordinated layer-1/2 reference rows through `p_t` instead of `p_c` (020:13–15). This qualifies every coordinated Level-0 value of 020, including the coordinated part of the pooled +0.089 comparator (the block-5 head term, E016). All three 021 labels remain a valid test of 020's program as wired (021:13–17), and 021's Y1 pools the coordinated frames, so none of them tests the correctly wired chain there. The downstream readout and its ceiling stand (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:54–57). | `2320345` |

E005–E019 were added by [the 2026-09-27 claim review](reviews/C002-review-2026-09-27.md), which gives the reasoning for each. By the 2026-09-27 decision on that review, E008 and E017 are typed methodological (scope-limiting), not as the review proposed (supporting and contradicting, each beyond the declared cue scope), because their relevance depends on extending this claim beyond its declared cue scope. For the same reason, the review's failed-test line for E017 is recorded under *Not supported*. Citations such as `(007:98–100)` are lines of `experiments/007-*/README.md` at `d7c5e62`. Kind M marks evidence on the mechanistic decomposition, P evidence on the prediction of intervention effects, and C evidence that could satisfy the Level-3 necessity, sufficiency and control gates.

## Evidence gates

### Level 0 — Measurement validity

- [x] Quantities and success criteria operationally defined. Evidence: design revision 5; E004.
- [x] Inputs, tokenization, positions, and targets recorded. Evidence: manifest and extension digests in the lock; E003.
- [x] Reconstructed outputs agree with model ground truth. Evidence: E004 (A1 per-case match to the screen; decomposition identity).
- [x] Numerical validation, tolerances, and failure behavior documented. Evidence: E004.
- [x] Model revision, dependencies, runtime, and artifact locations recorded. Evidence: results state provenance; E001–E003.
- [x] Reproducible from version-controlled instructions. Evidence: `experiments/005-regular-plural-mechanism/README.md`.

### Level 1 — Observation

- [x] Phenomenon measured directly with explicit sample and scope. Evidence: E001, E002.
- [x] Variation across cases quantified. Evidence: E002 (strata and bands), E003.
- [x] Result labeled exploratory or confirmatory. Evidence: E001 exploratory; E002 calibration; E003 confirmatory attempt whose precondition failed.
- [x] Conclusion avoids causal interpretation at this level. Evidence: this record.

### Level 2 — Localization

- [x] Narrowed to defined components and positions. Evidence: E001 (A2 position map, k = 3 set).
- [x] Plausible alternative locations compared. Evidence: E001 (54-component ranking at both positions; random sets), E002 (P2).
- [x] Obvious artifacts and confounders tested. Evidence: E001 (structural no-ops asserted; cross-cue and cross-frame controls), E004.
- [x] Localization replicates within its declared scope. Evidence: E002 (19 holdout nouns never used for ranking).
  - 2026-09-27: E006, Experiment 007's circuit families on fresh nouns in the manifest frames, at criterion level; E007, P3's pair-level correlation failed on that set (0.696).
- [x] Not presented as proof of causality. Evidence: this record.

### Level 3 — Causal evidence

- [x] Intervention and predicted outcome documented before inspection. Evidence: the lock (commit `30e8b84`).
- [ ] Necessity-oriented intervention supports the prediction on confirmatory data. Not established: E003's precondition failed, so its P8 value is recorded but not confirmatory.
  - 2026-09-27: Experiment 007's P8 passed at criterion level on fresh data (E006), but 007's composite route was not met, because P3 failed (E007). Still not established.
- [ ] Sufficiency-oriented intervention supports the prediction on confirmatory data. Not established for the same reason (E003 P1/P4/P5 recorded, non-confirmatory).
  - 2026-09-27: Experiment 007's P1, P4 and P5 passed at criterion level on fresh data (E006), but its composite route was not met (E007). E010–E015 predict the consequences of the E-patch and are not sufficiency evidence; E016's results are predictive, not causal. Still not established.
- [ ] Matched, negative, and other controls on confirmatory data. Not established (E003 P2/P6/P7 recorded, non-confirmatory).
  - 2026-09-27: Experiment 007's P7 passed at criterion level on fresh data (E006), but its composite route was not met (E007). Still not established.
- [x] Effect sizes and uncertainty reported. Evidence: E002 bands; E003 values.
- [x] Plausible alternative causal explanations considered. Evidence: design hypotheses H2/H3 and their discriminating quantities (E001).
  - 2026-09-27: later evidence on H2 and on the third competing explanation is noted under *Competing explanations* (E005, E008–E013, E016, E017, E019).

### Level 4 — Falsification tested

- [ ] Not reached.

### Level 5 — Generalized

- [ ] Not claimed. The new-frame and cue-word measurements (E003) are recorded but non-confirmatory under the outcome rule.

### Level 6 — Algorithmic explanation

- [ ] Not claimed. The compact program failed the protocol v1 floor (E001) and was capped in protocol v2; the cue-word results (E003) show the one-dimensional lexicon variable does not predict unseen cue words even though the E route does.

## Contradicting evidence and failed tests

- Protocol v1 program floor: the token-local program reproduced every development pair sign but under-predicted the quantifier template mean by 1.44 nats (E001).
- Reserve behavior (E003): 98/120 primary correct and 75/120 flips, below the preregistered 103 and 96; 44 of the 45 wrong conditions are plural conditions, spread over 13 of the 20 reserve nouns; the number variable still matched the cue in all 45 (S3).
- P3's absolute retention count (84/120) was unattainable on the reserve (clean flips 75/120) and on the new frames (65/120), the same split dependence noted for development in design revision 5.
- Cue words (E003): `all`, `some`, `both`, `few`, `many` shift the contrast by −3.8 to −4.8 nats although their lexicon values are within ±0.48; `every` (n_c −0.61) shifts it by about zero; `the` shifts it by −2.6. The E-patch reproduces each word's behavioral shift closely, so the cue-word information travels through E, but not along the single development-derived axis.
- S1: the readout-input number axes of the three templates have a minimum pairwise cosine of 0.292 (floor 0.70) on holdout and reserve.
- Experiment 007's P3 (E007): the pair-level fidelity correlation failed on both fresh sets (0.696 and 0.840 against 0.90), although the mean retention (F 0.781 / 0.773) and the 120/120 paired signs held (E006).
- Later program-level results not established: 007 Y1; 009 Y2 and Y3; 015 Y2; 017 Y3; 018 Y2 and Y4; 021 Y2; 023 Y1/cue_final; and 025. They concern decoded programs or a different intervention, not this claim's components (E008, E010, E013–E018).

## Competing explanations

- H2 (distributed transport): disfavored on development (top head 0.816 alone; freezing it blocks 0.865) and not contested by the recorded reserve values.
  - 2026-09-27: the later record names second routes to the target position by layer, not by head. 020's block-5 head term covers every block-5 head, over the value changes at the cue and target positions; removing it costs 0.089 of pooled exposed R² [exploratory], with no corrected template split, and in cue-final frames it is position-local (E016, E019). No record attributes it to `L05.H07`. 022 finds block 0's cue→target change contributing positively in coordinated frames for new cue classes, 0.138 / 0.157 of R² [descriptive] (E017). On this claim's pairs the second routes are small: `L05.H07` alone 0.122, blocking 0.128, `L00.H05` alone 0.028, and routes that bypass the cue-position residual after block 0 about 0.035 (E001). `L03.H04` alone carries 0.825 and 0.793 on Experiment 007's fresh sets (E006). H2 stays disfavored within scope. For other cue classes block 0's route is positive, of unestablished relative size; the later record names second routes and does not show a distributed transport.
- H3 (late computation): disfavored on development (`m_R` 0.99 with E alone patched; R alone 0.606).
- A higher-dimensional or nonlinear cue encoding in `L00.MLP`: supported by the cue-word results and not distinguished by this experiment's one-dimensional variables.
  - 2026-09-27: its higher-dimensional part is supported further. 006 falsified a description by the leading ≤ 4 unsupervised PCA directions (006:77–79) [exploratory] (E005); on cue words beyond this claim's declared scope, 007's rank-1 program and its full-dimensional ridge baseline both failed `this` and `another` (007:102–109) [Y1 unestablished] (E008); 010 finds an opposing off-axis part inside a distributed encoding [exploratory] (E011); 012's MLPs respond to that off-axis part [descriptive] (E013). Its nonlinear part points downstream of `L00.MLP`: the head reads its input linearly (E010, E012); at the head's output the axis and off-axis parts of `this`, `a` and `another` cancel mostly linearly, and the non-additivity that 008 classed `NONLINEAR_GATING` (E009) grows downstream of the head (009:110–115) [exploratory]; layers 1–2's MLPs show a small interaction for numerals (012:103–104) [descriptive] (E013).

## Relationship to prior work

See the finalist audit in `screening/behavior-candidates/audit-2026-09-17.json` and the design's prior-work section: number representations and agreement circuits are already mapped; this claim concerns the cue-to-noun-inflection pathway and is not a novelty claim about number representations.

## Potential novelty if confirmed

A component-level, position-resolved account of cue-to-noun-inflection transport in an unmodified Pythia-70M, with prospectively locked intervention predictions. Confirmation was not achieved under the preregistered rule.

## Supported conclusion

Within pinned Pythia-70M, the three screening templates, single-token regular nouns, and the tested replacement distributions, the count-cued singular/plural contrast is localized to `L00.MLP` at the cue position, `L03.H04` at the noun position, and the layer-4 and layer-5 MLPs at the noun position: on development data and on nineteen held-out nouns, exact replacement of these components recovers about 98% of the contrast shift, isolating them retains about 78% of it, and the head carries about 82% of the coordinated-template shift on its own. The same magnitudes were recorded on the reserve nouns and on six unseen frames, but that run does not count as confirmatory because the reserve nouns failed the preregistered behavioral precondition.

## Not supported

- Causal or algorithmic status of the mechanism (Levels 3–6).
- Generalization beyond the tested templates, frames, cue words, nouns, checkpoint, or model.
- The compact executable program: its one-dimensional lexicon does not predict unseen cue words, and it under-predicted the quantifier template on development data.
- That reserve-noun behavior matches holdout behavior: it does not (E003).
- That `L00.MLP` alone is the cue stage, at the pair level, for cue words beyond the tested ones: for new determiner-like, quantity and adjective cues, block-0 attention at the cue position also carries part of the cue's effect (E017). This makes explicit an exclusion that the generalization line above already states.

## Limitations

- Every split shares the same twelve prompts; noun-level replication tests readout generalization only. Prompt-side generalization rests on six new frames and twelve cue words whose results are recorded but non-confirmatory.
- The preregistered absolute counts (B1 flips, P3 retention) depend on the split's clean behavior; the reserve split is behaviorally weaker than the holdout split (75/120 versus 98/120 clean flips), which drove both failures.
- Instrumented forwards differ numerically from plain forwards by up to ~5e-3 nats; all Experiment 005 quantities are consistent within the instrumented path.
- Necessity and isolation conclusions are scoped to pair-centered neutralization.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-18 | — | `LOCALIZED` | — | `ACTIVE` | Created after Experiment 005's single confirmation run; Level 0–2 gates satisfied by E001, E002, E004; Level 3 not established because E003's behavioral precondition failed | E001–E004 |
| 2026-09-27 | `LOCALIZED` | `LOCALIZED` | `ACTIVE` | `ACTIVE` | Reviewed against Experiments 006–025 ([review](reviews/C002-review-2026-09-27.md)). The recorded measurements stand, and 007 recorded the same magnitudes on new nouns and frames at criterion level. Level 3 not established: 007's composite route was not met (P3's correlation floor); 009–019 are predictive and mechanistic evidence on the consequences of the E-patch, not sufficiency evidence; 025 is recorded as null and does not test this chain. The contributors the later record names are small on C002's pairs (E001). Beyond the tested cue words, at the pair level, `L00.MLP` alone is not the whole cue stage (E017); this is recorded under Not supported. | E005–E019 |
| 2026-09-28 | `LOCALIZED` | `LOCALIZED` | `ACTIVE` | `ACTIVE` | C003 was registered beside this claim. It describes the decoded predictive program and does not supersede this claim, whose evidence, gates and wording are unchanged. | — |

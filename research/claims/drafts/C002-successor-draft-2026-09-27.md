# Draft successor to C002: frame-conditioned decoded programs prospectively predict how a cue change reaches noun-number selection in pinned Pythia-70M

**A draft for review. It is not in the claim registry.**
- It has no claim identifier. Adoption would assign the next free one.
- It is not listed in [`../README.md`](../README.md), and no claim links to it.
- It sits beside [C002](../C002-count-cued-noun-number-circuit.md) and does not replace or supersede it.
- It was authorized on 2026-09-27 as a draft for review only; see
  [the decision on the C002 review](../reviews/C002-review-2026-09-27.md#decision-2026-09-27).

## Metadata

- Proposed maturity on adoption: `LOCALIZED`. Prospectively predictive; causal gates unestablished.
- Proposed disposition on adoption: `ACTIVE`
- Drafted: 2026-09-27
- Last reviewed: 2026-09-28, by three independent read-only reviews, of the first draft and of revisions 2 and 3
  (this is revision 4)
- Origin: each stage was formulated on exposed data and tested once, in its experiment's single confirmation run
  under its own lock (009, 011–019, 021–024); not every test passed. The composite hypothesis below was assembled
  afterwards, in this draft, from those runs and from exploratory work (005, 008, 010, 020).
- Related experiments: Experiments 005–025
- Related claim: C002, a localization claim (see the next section)

## Relationship to C002

- **C002 localizes.** It names three components: `L00.MLP` at the cue position, `L03.H04` at the prediction position,
  and the layer-4 and layer-5 MLPs there. It measures what fraction of the contrast shift they recover, retain and
  mediate, for two cue pairs.
- **This draft describes decoded programs.** They are written from the weights, each frame's measured reference
  state and quantities locked from exposed data, such as 012's template bases (012:8–9), 017's layer-3 bases
  (017:13–14) and 018's neuron subset (018:12–13). By rules frozen before the target prompts were executed, they predicted the consequences of a cue change:
  stage by stage for the E-patch, and end to end for a real cue swap.
- **The two coexist.** C002's measurements stand. Its 2026-09-27 review records Experiments 006–025 against C002's own
  components (C002, E005–E019). This draft starts from C002's localization and does not restate it.
- The evidence identifiers below are local to this draft.

## Hypothesis

In pinned `EleutherAI/pythia-70m-deduped`, frozen programs written from the weights, each frame's measured reference
state and quantities locked from exposed data predict, by rules frozen before the target prompts are executed, the consequences of changing the
count cue before a noun. Two interventions were predicted:
- **The E-patch** replaces the `L00.MLP` output at the cue position with another cue's encoding. For it, the programs
  predicted the intermediate states, stage by stage (009–019).
- **A real cue swap** replaces the cue token. For it, only the contrast change `Δc` at the prediction position was
  scored prospectively, together with where the E-only chain's remaining error lies (021–023). The programs'
  reconstruction of the intermediate states under a real swap is descriptive (022:225–229).

In the model's order, the program's stages are:
1. **The cue's block-0 state.** The layer-0 MLP's encoding, `MLP₀(LN₂(W_E[w]))`. Under a real swap, also the embedding
   change and block-0 attention's value and pattern terms at the cue position, and in coordinated frames block 0's
   update at the prediction position (023:16–20).
2. **Layers 1–2.** At the cue position, and in coordinated frames at the prediction position after one propagation
   step, a frame-conditioned correction of that state
   (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:74–81; 018:9–12):
   - the token-local MLPs at the frame's operating point;
   - the heads' value paths and their pattern change;
   - block 2's operating-point dependence (channel D). A locked subset of block-2 neurons largely recovers it, and the
     frame's routing of that population is predicted in aggregate. This is not a uniquely defined circuit
     (018:156–157; 019:164–165, 186–187).
3. **`L03.H04`.**
   - Given its measured input, a fixed-normalization linear read-out predicts its output (009, 011).
   - A weight-only read of the encoding predicts its transport of new cues (011).
   - Its own attention-row change is predicted from the encoding change (017).
   - End to end, the head is computed exactly as part of block 3, in the readout
     (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:70–72).
4. **The readout.** Blocks 3 and 4 run at the cue and prediction positions, and block 5 at the prediction position,
   with MLPs at the frame's operating point
   (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:106–115).
   - Block-4 attention is held at the reference.
   - Block 5's heads keep their reference rows over the predicted value changes. That includes a small
     second-transport path from the cue (020:43).
   - The residual the readout leaves is small: exploratory on exposed data (020:132), descriptive on fresh data
     (021:140–142; 023:143–145). 023 makes no claim that the readout is complete (023:183–184). A frozen,
     weight-derived nounness score predicts the residual's size across new cues, as a rank association that is
     predictive, not causal.

The claim is predictive. It does not claim that the programs' variables are the model's causal variables.

## Scope

- **Model:** `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. The model runs in
  float32 on CPU (020:28; 019:101); 020's program works in float64
  (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:77).
- **Behavior:** the contrast change `Δc(f, w, n) = c(frame f, cue w, noun n) − c(frame f, reference cue, noun n)` at
  the prediction position (020:23), in the frame families of the three screening templates: cardinal, quantifier and
  coordinated-adjective.
- **Cues:** the exposed pool, and fresh sets of 24 cues in most experiments (23 in 009, 40 in 024 and 025), drawn by
  quota from lexical classes. In 011 these were determiner-like, numeral, quantity, possessive/pronoun and adjective
  (011:100–102); in 023, determiner-like, quantity and adjective (023:47).
- **Frames:** up to 108 exposed frames (021:129), and 6 to 18 new frames per experiment in 009–023; 024 and 025 used
  exposed frames only. New frames enter through their
  measured reference states: “Nothing here predicts a frame's attention state from its text alone.”
  (docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:262–263)
- **Nouns:** the 79 scorable exposed nouns (021:129), and 24 fresh nouns in 021's Y3 (021:138).
- **Interventions:** the E-patch (009–019), full cue substitutions (020–024), and one rotation of the cue embedding
  (025).
- **Not in scope:** other checkpoints or models; other templates; multi-token nouns; cue classes not tested; frames
  whose reference state was not measured; the causal status of the decoded variables.

## Operational definitions

- **The E-patch:** replacing the `L00.MLP` output at the cue position of the frame's reference prompt with
  `MLP₀(LN₂(W_E[w]))`. A real cue swap also changes the embedding and block-0 attention
  (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:83–86).
- **The program.**
  - Its current version is 023's `P1` (023:16–20). This is 022's committed upstream chain, the 017 chain at `LEVEL0`
    built from the 011, 012 and 017 locks
    (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:74–81), with block 0's
    terms added. It is fed through 020's frozen readout
    (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:102–116).
  - Earlier versions were tested stage by stage, and one end to end:
    - 009's P1 read and 011's encoding read;
    - 012's token-local MLP model, on exposed base states (012:7–9);
    - 013's attention value paths and 014's neuron predictor;
    - 015's token-local and 016's frame-conditioned pattern models;
    - 017's chain to the head, 018's masked channel D and 019's routing rule;
    - 020's end-to-end program on its `p_t` wiring, tested in 021.
- **`Δx3`:** the change of the residual entering layer 3.
- **The P1 read:** the fixed-normalization linear `W_V W_O` read-out of the residual arriving at `L03.H04` (009:62–67).
- **Channel D:** “the frame's operating-point dependence of block 2's MLP as it reaches the transport read”
  (018:144–145).
- **The ceiling:** the readout program fed the measured `Δx3` (020:132).
- **`g`:** the share of the gap between the E-only chain (`P0`) and the ceiling (`C`) that the completed program
  (`P1`) recovers, `(SSE0 − SSE1) / (SSE0 − SSEC)` (023:16–21, 32). `R²₀`, `R²₁` and `R²_C` are the three programs'
  flattened `Δc` R² over pairs and the 79 nouns (023:42;
  docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md:159, 491).
- **Conditional:** a result on new frames is prospective but conditional on the frames' measured reference states
  (research/synthesis/source-map-005-025.md@b722e9c:52–54).

## Predictions

Each experiment's lock committed its families' criteria, their prediction rules, and every prediction that needs
neither a new frame's reference state nor a measured input, before the experiment's single confirmation run (E027).
- Frame-conditional tables were computed at stage 1 of that run and held by digest before any target prompt: 013–019
  (experiments/013-attention-paths-frozen-pattern/predictions.md:5; experiments/019-block2-routing/predictions.md:7),
  021 (021:86–89) and 022–023 (experiments/022-upstream-error-localization/preregistration.md:7;
  experiments/023-block0-completion/preregistration.md:8). None of these tables was committed before scoring. The
  source map records this for 013–016 (research/synthesis/source-map-005-025.md@b722e9c:678), and 022 committed its
  table after the confirmation, as closure evidence (022:100–104).
- The head read-outs, 009's Y1 and 011's Y2, apply a locked rule to the measured head input
  (docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:223–224).

| exp | lock installed or committed | families |
|---|---|---|
| 009 | `877221f` | Y1 the head's P1 read-out; Y2 a fitted rank-1 transport rule; Y3 a contrast rule, not locked |
| 011 | `85a6c04` | Y1 the zero-parameter encoding read of the head's transport; Y2 the P1 read-out on new frames |
| 012 | `5cea6c4` | Y1 the net layers-1–2 correction; Y2 the composite at the head's read-out |
| 013 | `c01126f` | Y1, Y2 the residual 012 left; Y3 the attribution between the heads and the MLPs |
| 014 | `d008dae` | Y1, Y2 neuron 1987's activation change; Y3 the axis-only alternative |
| 015 | `5ed51c9` | Y1, Y2 the layers-1–2 attention-pattern change; Y3 the axis-only alternative |
| 016 | `dc4e2a7` | Y1, Y2 the frame-conditioned pattern model; Y3 the scale-only alternative |
| 017 | `4b6f766` | Y1, Y2 the head's row change and output change; Y3 the frozen-pattern alternative |
| 018 | `6fa4c48` | Y1, Y2 channel D in a locked subset; Y3 the single-neuron alternative; Y4 the pattern term |
| 019 | `9033041` | Y1, Y2 routing; Y3 the operating-point-plus-drive rule; Y4 the template-family alternative; Y5 membership |
| 021 | `b7e8861` | `Δc` for new cues in exposed frames (Y1), in new frames (Y2) and for new nouns (Y3) |
| 022 | `4f8f31d` | eight conditions localizing the E-only chain's remaining error |
| 023 | `f2294ab` | four conditions of `g` for the completed program |
| 024 | `3affe55` | the rank association of nounness with readout error, and the E–N guard |
| 025 | `5ee7a74` | the effect of a nounness-direction rotation on readout error |

020 wrote no lock; its confirmation was never run.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | Experiment 005, Tier A and calibration | supporting | `experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md`; `experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md` | Kind M; exploratory and calibration. The localization the programs start from, recorded in C002 (its E001–E002): `L00.MLP` carries 0.535 of the transport-input axis, and the three-component set recovers 0.982 (005:94–97). Isolation and its pair-level fidelity are recorded against C002, not here (C002, E006–E007). 005's single confirmation ended `BEHAVIOR_NOT_REPLICATED` (005:125–129), recorded in C002 as its E003, and 006's circuit families never ran (C002, E005), so the starting localization rests on exploratory and calibration data. | code `83d1ae4`, `009607d` |
| E002 | Experiments 005 (protocol v1), 006 and 007: their compact programs | null | 005's discovery report; `experiments/006-low-rank-cue-decompilation/evidence/exploration-report-2026-09-18.md`; `experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md` | Kind M; 005 v1 and 006 exploratory, 007 unestablished. Compact programs failed. 005's token-local program missed its floor, `NO_COMPACT_MECHANISM` (005:98–101). 006 falsified the leading ≤ 4 unsupervised PCA directions (006:77–79). 007's supervised rank-1 program failed Y1, `PROGRAM_NOT_SUPPORTED`, though its Y2 and Y3 passed as criteria (007:87–94). The programs read the encoding in full, from the weights. | 005 `83d1ae4`; 006 explore `adea65f`; 007 confirm `ede0f53` |
| E003 | Experiments 008 and 010 (discovery only) | supporting | `evidence/exploration-report-2026-09-18.md` of 008 and of 010 | Kind M; exploratory; both summaries `MIXED` (008:65; 010:62). For `this`, `a` and `another`, `L03.H04` attends normally but passes on 0.07–0.29 of the number signal (008:86–91). For `this`, `a`, `another` and `every`, the encoding carries an on-axis number signal and an opposing off-axis part (010:85–89). 010's relay bound failed: layers 1–2 move the read by at least 0.29 in gross even when their net effect is near zero (010:64–67). 010 attributes this to a bound frozen without a baseline, and reads layers 1–2 as mostly relaying the signal (010:66–67, 110); this draft reads the churn as consistent with stage 2. The signal is spread over hundreds of neurons: 320–700 of 2048 by the README (010:63), up to 726 in its report (research/synthesis/source-map-005-025.md@b722e9c:672). | 008 explore `d324755`; 010 explore `0a6836b` |
| E004 | Experiment 009, Y1 | supporting | `experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md` | Kind M, P; prospective, given the measured head input. `HEAD_MECHANISM_CONFIRMED_P1`: on 138 fresh (token, frame) pairs, the P1 read predicts the head's output with Spearman 0.957 and MAE 0.066 (009:62–64). The head's transport is “a fixed-normalization linear read-out of components already present in the cue residual it receives” (009:84–85). | confirm `78d71c2` |
| E005 | Experiment 009, Y2 and Y3 | null | as E004 | Kind P; unestablished. The fitted rank-1 transport rule failed, `TRANSPORT_RULE_FAILED`: Spearman 0.765 against 0.80 (009:69–70). The contrast rule was `NOT_LOCKED` (009:77–79). A compact transport rule is not established; the stage-level programs use the P1 read. | confirm `78d71c2` |
| E006 | Experiment 011 | supporting | `experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md` | Kind P, M; Y1 prospective from the weights alone; Y2 prospective, given the measured head input. `ENCODING_READ_PREDICTS_TRANSPORT`: Spearman 0.848, MAE 0.077 on 24 never-seen cues in six new frames (011:75–76). `HEAD_P1_REPLICATED`: 0.974 / 0.076 (011:87–89). `an` was predicted at 0.18 and measured at 0.30, the lowest of the 24 [locked prediction] (011:77–79). 009's fitted rule would also have cleared the Y1 floors [descriptive] (011:91–92). | confirm `85a6c04` |
| E007 | Experiments 012 and 013 | supporting | `evidence/final-report-2026-09-19.md` of 012 and of 013 | Kind M, P; prospective; 013 Y2 conditional; 013 Y3 pooled over both sets, so its new-frame share is conditional. `LAYER_CORRECTION_TOKEN_LOCAL_MLP`: the net layers-1–2 correction, Spearman 0.864, MAE 0.061, R² 0.795 (012:81–83). `COMPOSITE_PREDICTS_P1`: 0.922 / 0.062 (012:91–93). 012 predicted the net correction, not its internal split (012:110–112, 119–120). `RESIDUAL_PREDICTED_TOKENS`, `RESIDUAL_PREDICTED_FRAMES_CONDITIONAL` and `ATTRIBUTION_PREDICTED`: Y1 0.877 with R² 0.668; Y2 0.919 with R² 0.747; the predicted direct head part, Spearman 0.838, and the MLP part, 0.986 (013:88–90, 100–101, 106–108). | confirms `5cea6c4`, `c01126f` |
| E008 | Experiment 014 | supporting | `experiments/014-neuron-1987-feature/evidence/final-report-2026-09-19.md` | Kind M, P; prospective; Y2 conditional; Y3 pooled over both sets, so its new-frame share is conditional. `NEURON_FEATURE_PREDICTED_TOKENS`: Spearman 0.950, R² 0.970 (014:96–97). `NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL`: 0.910, R² 0.957 (014:101–102). `AXIS_ONLY_REJECTED`: R² −1.41 (014:104). | confirm `d008dae` |
| E009 | Experiment 015 Y1 and Y3, and Experiment 016 | supporting | `evidence/final-report-2026-09-20.md` of 015 and of 016 | Kind M, P; prospective; 016 Y2 conditional; the Y3 alternatives pooled over both sets, so their new-frame shares are conditional. `PATTERN_CHANGE_PREDICTED_TOKENS`: Spearman 0.986, R² 0.895 (015:105–106); the axis-only alternative rejected, R² −1.16 (015:123–124). `FRAME_CHANNELS_PREDICTED_TOKENS`: 0.998, R² 0.994 (016:106–107). `FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL`: 1.000, R² 0.996 (016:111–112). `SCALE_ONLY_REJECTED`: 0.672 (016:118–119). | confirms `5ed51c9`, `dc4e2a7` |
| E010 | Experiment 015, Y2 | contradicting | `experiments/015-attention-pattern-token-local/evidence/final-report-2026-09-20.md` | Kind P; unestablished. `PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL`: R² 0.490 against 0.50, and layer-2 entry R² 0.423 (015:109–110). The token-local pattern model failed in new frames. 016's frame-conditioned model passed in its own twelve new frames, where 015's model would reach 0.668 (016:111–117). | confirm `5ed51c9` |
| E011 | Experiment 017, Y1 and Y2 | supporting | `experiments/017-transport-head-pattern/evidence/final-report-2026-09-20.md` | Kind M, P; Y1 prospective; Y2 conditional. `HEAD_PATTERN_PREDICTED_TOKENS` and `HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL`: row entries R² 0.997 and 0.998, `ΔT` pairs 0.999 and 1.000 (017:116–123), from a chain whose “only cue-specific input is `ΔE`” (017:14–15). | confirm `4b6f766` |
| E012 | Experiment 017 Y3, and Experiment 018 Y4 | null | as E011; `experiments/018-block2-concentration/evidence/final-report-2026-09-21.md` | Kind M; unestablished; program-internal necessity tests of `L03.H04`'s pattern term. `FROZEN_PATTERN_NOT_REJECTED` (017:126–131) and `PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES` (018:122–126). | confirms `4b6f766`, `6fa4c48` |
| E013 | Experiment 018, Y1 and Y3 | supporting | `experiments/018-block2-concentration/evidence/final-report-2026-09-21.md` | Kind M, P; prospective; Y3 pooled over both sets, so its new-frame share is conditional. `CHANNEL_D_CONCENTRATED_TOKENS`: a locked 256-neuron subset carries 0.863 of channel D's contribution to the decoded read and 0.724 of its contribution to the pattern term (018:102–104, 146–147). `SINGLE_NEURON_REJECTED`: neuron 1987 alone 0.219 (018:118–119). | confirm `6fa4c48` |
| E014 | Experiment 018, Y2 | contradicting | as E013 | Kind P; unestablished. `CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL`: in one new quantifier frame the locked subset fell below the template base, 0.745 against 0.847, though the pooled closure held, 0.84 and 0.80 (018:151–155). | confirm `6fa4c48` |
| E015 | Experiment 019 | supporting | `experiments/019-block2-routing/evidence/final-report-2026-09-21.md` | Kind M, P; prospective; Y2 conditional; Y3 scored per set and pooled, Y4 on the new frames and pooled, Y5 per set (019:75–76), their new-frame parts conditional; aggregate, not per frame. `ROUTING_PREDICTED_TOKENS`, `ROUTING_PREDICTED_FRAMES_CONDITIONAL`, `OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT` (ρ 0.862 / 1.136, pooled 0.913), `TEMPLATE_FAMILY_INSUFFICIENT` and `MEMBERSHIP_PREDICTED` (0.867 on both sets) (019:120, 129–130, 137–138, 145, 149–151). It is “not a rule that holds in every frame” (019:164–165). | confirm `9033041` |
| E016 | Experiment 020: the readout at its ceiling | supporting | `experiments/020-readout-decompilation/evidence/exploration-report-2026-09-22.md` | Kind M; exploratory; its confirmation was never run. The readout program (020:43–47), fed the measured `Δx3`, reaches R² 0.976 (020:132); the downstream shortfall is about 0.024 (020:140). In 020's design spike, `L04.MLP` and `L05.MLP` contribute about 61 % of `Δc` directly, and block-4 attention −0.000 (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:45–49). | explore `ce3766b`; closed `7f4a4f8` |
| E017 | Experiment 020: its end-to-end program | contradicting | as E016 | Kind P; exploratory. On the exposed pool, the corrected program reached flattened R² 0.689, where the scratch pass had 0.848. It sat below six of its twelve frozen floor conditions, the coordinated template carrying most of the shortfall (020:160, 177–179). These exposed values are a calibration record, never evidence for a label (020:110). The confirmation was not run because the floors had been calibrated on a scratch pass with a block-5 implementation error, so no label could have been read cleanly (020:8–11, 175–181). Its coordinated values are qualified by E025. | explore `ce3766b` |
| E018 | Experiment 021, Y1 and Y3 | supporting | `experiments/021-corrected-readout-confirmation/evidence/final-report-2026-09-23.md` | Kind P; prospective; qualified by E025. `CONTRAST_PREDICTED_TOKENS`, with a small margin: token-mean R² 0.016018 against a floor clamped to 0.0 (021:116, 129–132), and coordinated R² −0.014 [descriptive] (021:132–133). `NOUN_READOUT_FIXED` (021:138). 021 says this is “not a claim of good prediction everywhere” (021:171–172). The ceilings, 0.9683 and 0.9652, are descriptive (021:140–141). | confirm `b7e8861` |
| E019 | Experiment 021, Y2 | contradicting | as E018 | Kind P; unestablished. `CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL`: coordinated R² −0.1538 against its clamped floor 0.0 (021:134–137). By the erratum it “must not be read as evidence about the correctly wired 017 upstream chain” (021:16–17). | confirm `b7e8861` |
| E020 | Experiment 022 | supporting | `experiments/022-upstream-error-localization/evidence/final-report-2026-09-24.md` | Kind M, P; prospective; Y2 conditional; the coordinated share descriptive. All eight conditions pass (022:115). The error is localized primarily to block-0 attention and the state it constructs (022:243–245): block-0 attention at the cue position accounts for 0.948 / 0.938 of the cue-final gap. The correctly wired layer-1–2 reductions account for at most 0.007 of either gap, and block 0's change at the prediction position contributes positively in coordinated frames (022:243–250). Under a real swap, the E-only chain's `Δx3` has a median relative error of about 0.50–0.66; block 0's terms make it nearly exact at the changed positions [descriptive] (022:225–229). | confirm `4f8f31d` |
| E021 | Experiment 023: Y1/coordinated, Y2/cue_final, Y2/coordinated | supporting | `experiments/023-block0-completion/evidence/final-report-2026-09-24.md` | Kind P; prospective; Y2 conditional; no aggregate label. With block 0 completed, `g` is 0.9987, 0.9974 and 0.9966, each above its floor. `R²₁` reaches 0.9574, 0.9610 and 0.9518, against ceilings of 0.9579, 0.9613 and 0.9531 [descriptive] (023:143–145). Y1/coordinated passed by +7.979e-5 (023:156–157). The claim covers new determiner-like, quantity and adjective cues only (023:183–184). | confirm `c7efec7` |
| E022 | Experiment 023, Y1/cue_final | contradicting | as E021 | Kind P; unestablished at its envelope; the 0.90 guard was met. `ENVELOPE_ONLY_FAILURE`: `g` 0.99552 against `F` 0.99690 (023:142). Under the frozen reading this is “a small quantitative shift, not a refutation” (023:154). | confirm `c7efec7` |
| E023 | Experiment 024 | supporting | `experiments/024-readout-routing-nounness/evidence/final-report-2026-09-25.md` | Kind P; prospective; “predictive, not causal” (024:373). `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`: ρ 0.611 against 0.314, and the E–N guard K = 1 of 12,870, on 40 new cues in exposed frames (024:277–279). The guard's formal scope is two simple alternatives (024:295–297). 23 of the 40 cues lie beyond the calibration's nounness range (024:300–301). What was measured is the error of a frozen approximation (024:298–299). | confirm `af160ce` |
| E024 | Experiment 025 | null | `experiments/025-nounness-direction-intervention/evidence/final-report-2026-09-26.md` | No kind: an intervention on a decoded direction; unestablished. `CAUSAL_EFFECT_NOT_ESTABLISHED` (025:244): rotating new cues' embeddings along the nounness direction did not establish its predicted effect on the readout error. | confirm `6f72d7d` |
| E025 | Post-closure errata of Experiments 020 and 021 | methodological | `evidence/erratum-2026-09-23-coordinated-reference-rows.md` of 020 and of 021 | No kind: an erratum. 020's program built the coordinated layer-1/2 reference rows through the prediction position instead of the cue position (020:13–15). This qualifies 020's coordinated Level-0 values and all three 021 labels. The downstream readout and its ceiling stand (experiments/020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md:54–57). 022 and 023 use the correct wiring. | `2320345` |
| E026 | Measurement-validity records of Experiments 009–025 | methodological | each experiment's README and evidence directory | No kind. The frozen prompt manifests and ledgers, the exact identities, the runtime records and the runners. For example, 022's confirmation gates I1, I3 and I4 held at 1.64e-5, 4.23e-5 and 7.69e-5 (022:161–163), and their values are held under `confirmation.gates` in its committed `experiments/022-upstream-error-localization/evidence/confirmation-record-2026-09-24.json` (022:171–174). 020's Level-1 identity is bounded by a diagnosed float32 precision floor (020:69–81, 190–194). 023 executed its 3,060 frozen prompts exactly once (023:158–160). | the confirm commits of E004–E024; 020's explore commit `ce3766b` |
| E027 | The preregistration locks | methodological | each experiment's lock and design | No kind. Each lock committed its families' criteria, their prediction rules, and every prediction that needs neither a new frame's reference state nor a measured input. Frame-conditional tables were computed at stage 1 and held by digest before any target prompt; the source map records that 013–016's were not committed before scoring (research/synthesis/source-map-005-025.md@b722e9c:678). See *Predictions*. | the lock commits under *Predictions* |

Kind M marks evidence on the mechanistic decomposition, P evidence on the prediction of intervention effects. No
entry satisfies the Level-3 necessity, sufficiency or control gates for the decoded variables. The one preregistered
intervention on a decoded direction did not establish its predicted effect (E024).

The types follow one rule. Each tested version of the program counts: a pass is supporting and a failure
contradicting. Compact alternatives, meaning fitted or low-rank rules outside the zero-parameter program class (005's
lexicon program, 006's PCA directions, 007's rank-1 program, and 009's rank-1 rule and contrast rule), and
program-internal necessity tests that were not established are null.
Exploratory records are typed by what they show: E001, E003 and E016 support the decomposition, and E017 records an
end-to-end shortfall. E024, an intervention whose effect was not established, is null.

## Evidence gates

### Level 0 — Measurement validity

- [x] Quantities and success criteria operationally defined. Evidence: E027; the definitions above.
- [x] Inputs, tokenization, positions and targets recorded. Evidence: E026.
- [x] Reconstructed outputs agree with model ground truth. Evidence: E026 (022's confirmation gates; 020's Level-1
  identity within its diagnosed float32 floor).
- [x] Numerical validation, tolerances and failure behavior documented. Evidence: E026 (020's identity incident and
  its tolerance amendment); E025.
- [x] Model revision, dependencies, runtime and artifact locations recorded. Evidence: E026.
- [x] Reproducible from version-controlled instructions. Evidence: E026.

### Level 1 — Observation

- [x] Measured directly, with explicit sample and scope. Evidence: E003–E024.
- [x] Variation across cases quantified. Evidence: per-frame and per-condition values, for example E014, E015 and
  E021.
- [x] Repeated or related cases examined. Evidence: fresh cues in exposed frames, in new frames, or both. 009, 011
  and 012 used new frames only (research/synthesis/synthesis-005-025.md@3d87d3c:272), 024 and 025 exposed frames only
  (024:300), and the others both (E004–E015, E018–E024). 011 replicated 009's read-out on new cues and frames (E004,
  E006).
- [x] Results labeled exploratory or confirmatory. Evidence: each entry's status, E001–E024.
- [x] The conclusion avoids causal interpretation. Evidence: each entry states its result as a prediction or an
  observation, with its status (E001–E024); this record.

### Level 2 — Localization

- [x] Narrowed to defined layers, positions and components. The programs narrow the cue's effect to named terms at
  the only positions a cue changes: the cue position and, in coordinated frames, the prediction position
  (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:76–77, 113–115). Evidence:
  - by construction, the E-patch's only cue-specific input is the encoding change, and a real swap adds block 0's
    terms (E011, E021); the empirical narrowing rests on the next bullets;
  - the E-only chain's remaining error lies primarily in block-0 attention, with the layer-1–2 reductions at most
    0.007 of the gap (E020);
  - channel D concentrates in a locked subset on new cues (E013), though not in one new frame (E014);
  - block-4 attention can be held at the reference [exploratory] (E016); its share of the frozen readout's own error
    is 0.58–0.73 by class [descriptive] (024:324);
  - the starting point is C002's localization (E001).
- [x] Plausible alternative locations compared. Evidence: 005's position map (E001); 022's attribution across
  block-0 attention, the embedding and the layer-1–2 reductions (E020); 018's random and lowest-ranked subsets (E013).
- [x] Obvious artifacts and confounders tested. Evidence: the exact identities (E026); the wiring defect found and
  recorded (E025); E023's guard against two simple confounds, plurality and measure class, the one preregistered
  confounder test (024:278, 295–297).
- [x] Replicates within its declared scope. Evidence: prospective confirmations on new cues and nouns, and on new
  frames given their reference states (E004, E006–E009, E011, E013, E015, E018, E020, E021), with the failures E010,
  E014, E019 and E022.
- [x] Not presented as proof of causality. Evidence: each entry's status (E001–E024); E023, stated as predictive, not
  causal; E024, the one causal test, null; this record.

### Level 3 — Causal evidence

- [x] Intervention and predicted outcome documented before inspection. Evidence: E027.
- [ ] Necessity-oriented intervention supports the prediction on confirmatory data. Not established. The
  program-internal necessity tests of `L03.H04`'s pattern term were preregistered and not established (E012). They
  are variants of the program, not interventions in the model. No preregistered, outcome-bearing necessity test in the
  model targeted a decoded variable.
- [ ] Sufficiency-oriented intervention supports the prediction on confirmatory data. Not established. The
  confirmations predict the consequences of the E-patch or of a real cue swap. No decoded variable was itself
  manipulated and shown sufficient in a preregistered, outcome-bearing test. The one preregistered intervention on a
  decoded direction did not establish its predicted effect (E024). The decision on the C002 review applies the same
  reading here.
- [ ] Matched, negative and other controls on confirmatory data. Not established. The preregistered alternatives
  (E008, E009, E012, E013, E015) are variants of the program, not interventions in the model.
- [x] Effect sizes and uncertainty reported. Evidence: floors from exposed-only calibration (E018), and per-frame and
  per-condition values (E014, E015, E021).
- [x] Plausible alternative causal explanations considered. Evidence: the preregistered alternatives (E008, E009,
  E012, E013, E015); *Competing explanations*.

As in C002, the preregistration gate and the two reporting gates (effect sizes; alternatives considered) are checked.
No necessity, sufficiency or control gate is checked.

### Level 4 — Falsification tested

- [ ] Not reached, because Level 3 is not. Discriminating predictions against alternatives were preregistered:
  - the axis-only (E008, E009), scale-only (E009), single-neuron (E013) and template-family (E015) alternatives were
    rejected or found insufficient;
  - the frozen-pattern alternative was not rejected (E012);
  - failed predictions are retained (E010, E014, E019, E022), and so is 020's exploratory end-to-end shortfall
    (E017).

### Level 5 — Generalized

- [ ] Not claimed. The axes that changed are new cues from several lexical classes, new frames given their reference
  states, and new nouns (E018). The checkpoint, the three templates and single-token regular nouns stayed fixed.
  Generalization failed in E010, E014 and E019, and at its envelope in E022.

### Level 6 — Algorithmic explanation

- [ ] Not claimed. The record bears on several of this level's criteria:
  - it predicts unseen model behavior (E018, E021);
  - it predicts the outcomes of the same intervention as each Tier A, the E-patch, on new cues and frames by locked
    rules, which only partly meets the criterion of new interventions (E004, E006–E009,
    E011, E013, E015);
  - residual unexplained behavior is quantified (E016, E021);
  - known failures are documented (E010, E014, E017, E019, E022).

  No compact, human-readable mechanism for the whole is established (E002, E005). Compact rules hold for parts only:
  the P1 read (E004), the zero-parameter encoding read (E006), neuron 1987's predictor (E008) and the routing rule,
  predicted in aggregate (E015). Levels 3–5 are not met.

## Contradicting evidence and failed tests

- E010: 015's token-local pattern model failed in new frames.
- E014: 018's locked channel-D subset failed the no-harm guard in one new quantifier frame.
- E017: 020's end-to-end program sat below six of its twelve frozen floors on exposed data. The floors had been
  calibrated on a scratch pass with a block-5 error, and 020 calls the exposed values a calibration record, never
  evidence for a label; its confirmation was never run (020:110, 143, 175–181).
- E019: 021's coordinated new-frame condition failed. It is plausibly explained by the wiring defect (E025).
- E022: 023's Y1/cue_final fell below its envelope.
- Null results:
  - 005's single confirmation, `BEHAVIOR_NOT_REPLICATED` (E001; C002, E003);
  - the compact programs (E002), a compact transport rule and the contrast rule (E005);
  - `L03.H04`'s pattern term (E012);
  - the nounness-direction intervention (E024).
- Exploratory frozen-rule outcomes: 008's and 010's `MIXED` summaries, and 010's failed relay bound (E003).

## Competing explanations

Each is stated with the test that would distinguish it.

- **The decoded variables predict without being the model's causal variables.** Not distinguished.
  - Interventions on decoded parts exist outside any outcome: the axis-only and orthogonal-only E-patches, exploratory
    in 008 (008:57–59) and in 009's Tier A (009:31), and descriptive on 009's fresh tokens (009:80–81).
  - The one preregistered intervention on a decoded direction did not establish its effect (E024).
  - The test: preregistered, outcome-bearing interventions on the decoded variables themselves, with necessity,
    sufficiency and control logic.
- **Much of the accuracy comes from recomputing exact layers given the measured reference state.**
  - Block 3's heads are computed by the exact layer program
    (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:106–109), and block 0 by an
    exact row rule (023:17–18). If so, the confirmations mainly test whether the reductions elsewhere are adequate.
  - Partial evidence for it at block 0: 023's cheaper block-0 rules recovered far less than the exact rule
    [descriptive] (023:180–181).
  - The test: each stage's reduced rule scored against the exact computation it replaces, on fresh data.
- **A more compact program.** Not supported for the encoding (E002) or the transport rule (E005); compact rules hold
  for parts, the routing rule in aggregate (E004, E006, E008, E015). The test: a preregistered compact program for a
  whole stage.
- **Frames predicted from their text.** Not attempted; new frames enter through their measured reference states. The
  test: predicting a frame's reference state from its text.
- **`L03.H04`'s pattern term is unnecessary.** Not rejected (E012). The test: the frozen-pattern alternative on cues
  and frames where the pattern term is larger.
- **The readout residual reflects rerouting rather than approximation error.** Not distinguished. 024 measured “the
  error of a frozen approximation, not directly observed rerouting” (024:374). The test: rerouting measured directly.

## Relationship to prior work

Experiment 005's prior-work boundary applies
(docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:976–988), with the finalist
audit as its authoritative statement (screening/behavior-candidates/audit-2026-09-17.json:3). Plural and inflection
representations and agreement circuits in Pythia are already mapped; this is not a novelty claim about number
representations. The nearest methods are those the same section names: program extraction from small trained
networks, formal guarantees for small transformers, solver-checkable circuit explanations, and synthesized programs
that approximate attention heads
(docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:990–999).

## Potential novelty if confirmed

In an unmodified Pythia-70M, programs written from the weights and each frame's reference state predicted, by rules
frozen before the target prompts were executed, how a new cue changes the internal states under the E-patch and the noun-number contrast under a real
swap. They did so under preregistered single-run tests, with every failure listed above retained. This is subject to
systematic literature review and comparison (docs/RESEARCH_METHODOLOGY.md:396).

## Supported conclusion (proposed)

Within pinned Pythia-70M, the three screening templates, single-token regular nouns, the tested cue classes and the
tested interventions, frozen programs derived from the weights and each frame's measured reference state predicted,
by rules frozen before the target prompts were executed:
- **for the E-patch, stage by stage:**
  - the head's transport of new cues from the weights alone [prospective] (E006), and its read-out from the measured
    input [prospective, given the measured input] (E004);
  - the net layers-1–2 correction, on new cues in new frames from exposed base states [prospective] (E007);
  - the attention value paths (013), the attention-pattern change (015, 016), neuron 1987 (014), block 2's operating
    point and routing (018, 019), and the head's pattern change (017), for new cues [prospective], and in new frames
    for most stages [prospective, conditional; 015 and 018 Y2 unestablished] (E007–E015);
- **for a real cue swap:**
  - the contrast change on 020's wiring, for new cues and nouns in exposed frames, pooled over templates, with a small
    margin [prospective]. In coordinated frames it had no positive skill, in exposed frames (−0.014 [descriptive]) or
    new ones (Y2 unestablished), plausibly because of the wiring defect (E018, E019, E025);
  - the localization of the E-only chain's remaining error primarily to block-0 attention [prospective; Y2
    conditional] (E020);
  - with block 0 completed, the contrast change relative to the ceiling for new determiner-like, quantity and
    adjective cues. Three conditions pass, and Y1/cue_final falls below its envelope (`ENVELOPE_ONLY_FAILURE`, with
    the guard met); there is no aggregate label
    [prospective; Y2 conditional] (E021, E022);
- **for the readout:** that a frozen nounness score predicts the size of the readout's error on new cues in exposed
  frames [prospective; a rank association, predictive, not causal] (E023).

The programs are prospectively predictive; their causal gates are unestablished.

## Not supported

- That the programs' variables are the model's causal variables (Level 3).
- That the programs predicted the intermediate states under a real cue swap. Only `Δc` was scored; the reconstruction
  of `Δx3` is descriptive (022:225–229; 023:130).
- That a frame's internal state can be predicted from its text.
- A compact program for the whole mechanism.
- The necessity of `L03.H04`'s pattern term.
- A causal role of the nounness direction (E024). Also not claimed by 024: that nounness causes rerouting, that the
  model detects nouns, that nounness is a general linguistic representation, or that the E–N guard rules out all
  lexical or semantic confounds (024:302–306).
- Generalization beyond the tested cue classes, templates, checkpoint and model, or to multi-token nouns.
- Anything about the localization of the three components beyond what C002 supports.

## Limitations

- One checkpoint, three templates, single-token regular nouns, 23 to 40 fresh cues per experiment, and 6 to 18 new
  frames per experiment in 009–023.
- From 013 on, new-frame results are conditional on the frames' measured reference states.
- The E-patch leaves out parts of a real cue swap; 022 and 023 address them, for `Δc` only.
- 021's labels test 020's wiring (E025).
- Several floors were calibrated against the program's own exposed performance, and some were clamped to zero skill
  (021:146–149).
- 024's association is on exposed frames only, and 23 of its 40 cues lie beyond the calibration range (E023).
- The source map's record caveats apply
  ([`../../synthesis/source-map-005-025.md`](../../synthesis/source-map-005-025.md#record-caveats)).

## Status history

None. This is a draft; adoption would add the first row.

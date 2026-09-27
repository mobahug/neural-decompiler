# Synthesis: what Experiments 005–025 established about count-cued noun number in Pythia-70M

**Step 2 of the [synthesis](README.md).** This document answers the sequence's original question from the checked
[source map](source-map-005-025.md). It adds no measurement, changes no claim and proposes no experiment.

**How to read it:**
- **Evidence status.** Every finding carries its status from the map, in brackets: **[prospective]**,
  **[prospective, conditional]**, **[locked prediction]**, **[exploratory]**, **[descriptive]** or
  **[unestablished]**. The categories are defined in the source map.
  - *Conditional* means: on new frames, given each frame's measured reference state.
  - A statistic *pooled* over both sets has its new-frame share conditional.
- **Citations** use the source map's conventions: `(023:142)` is `experiments/023-*/README.md` line 142 at
  `d7c5e62`. `checks/check_source_map.py` verifies the quotes.
- **The rule.** No sentence here is stronger than the strongest record supporting it. Where the chain is described as
  a whole, each link keeps its own status.

## 1. The question

The sequence began with this question: “Can the computation by which an unmodified pretrained transformer converts an
external count/quantifier cue into the selection of a singular versus plural noun form be recovered as an explicit,
compact, human-readable causal mechanism that is specified before, and then tested on, untouched examples and
interventions it was not built from?” (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:142–146)

- **The model:** pinned Pythia-70M, deduped, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
- **The behavior:** `regular-plural`, chosen by the behavior screen [exploratory]. It is studied in three templates
  (cardinal, quantifier, coordinated-adjective) with single-token regular nouns.
- **005's own prospective test** ended `BEHAVIOR_NOT_REPLICATED`: the reserve nouns failed the behavioral
  precondition, so no circuit family counted as confirmatory (005:125–129) [unestablished].
- **The measured quantities** are differences between two forward passes, of three kinds:
  - *E-patch runs*, from 006 to 019. The cue's layer-0 encoding alone is put into the frame's reference prompt, an
    intervention on one component, and the downstream changes are measured. In 006's words: “replace E alone in the
    frame's reference prompt with `E(w)`” (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:308).
    019 inherits the same E-patch (docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md:317).
  - *Full cue substitutions.* These are the behavioral shifts of 006's and 007's Y2, “the same prompts, no
    intervention” (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:309), and every
    measurement from 020 to 024. At the output: “`Δc(f, w, n) = c(frame f, cue w, noun n) − c(frame f, reference cue,
    noun n)`” (020:21–24).
  - *An embedding rotation* in 025.
- **What the E-patch omits.** A real cue swap “also changes three things the chain does not carry: the embedding
  itself at `p_c`; block 0's attention output at `p_c`; and, in coordinated frames, block 0's attention output at
  `p_t`.” (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:85–86) 022 and 023
  address that gap (§3.2).

## 2. The answer, attribute by attribute

- **Explicit, specified before and tested on untouched examples: largely yes, stage by stage.**
  - From 009 on, frozen programs specified before the fresh data predicted the cue-induced changes at every decoded
    stage. On new cues, every decoded stage passed at least one preregistered criterion: from 013 on in the exposed
    frames, and for 009, 011 and 012 in new frames [prospective].
  - On new frames most stages passed, given each frame's measured reference state [prospective, conditional].
  - Every result that was not established is listed in §4.
  - Nothing predicts a new frame's internal state from its text.
- **Compact: established for parts of the mechanism, not for the whole.**
  - No compact program for the whole mechanism was established:
    - 005's program missed its floor [exploratory; `NO_COMPACT_MECHANISM`];
    - 006's leading ≤ 4 unsupervised PCA directions failed the quality gate [exploratory];
    - 007's supervised rank-1 program failed Y1 [unestablished].
  - Some parts have small prospective rules, in the records' own terms:
    - the head's “fixed-normalization linear read-out” (009:83–85) [prospective, given the measured input];
    - the zero-parameter encoding read [prospective];
    - 014's predictor of neuron 1987, “A Compact Input-Side Predictor” (014:1) [prospective; new frames conditional];
    - 019's routing rule, “Account 2, compact form” (019:176), which carries most of the routing gain [prospective;
      new frames conditional; aggregate, not per frame].
  - Other parts are distributed or high-dimensional. Channel D sits in “a distributed-but-concentrated population”
    (018:149). The frame enters through “three structured classes of reference-state information” (016:137–138), one
    of them the self-logit operands at 1024 numbers per layer.
- **Human-readable: in parts.**
  - The head reads its input linearly [prospective, given the measured input].
  - For `this`, `a`, `another` and `every`, the encoding carries an on-axis number signal
    and an opposing off-axis part [exploratory]. For `an` in 011, the encoding read predicted a low transport of 0.18;
    the measured value, 0.30, was the lowest of the 24 cues [locked prediction].
  - Neuron 1987 combines excitation along the number axis with a veto from off-axis structure [prospective; new
    frames conditional]; its
    count-magnitude gloss is [descriptive].
- **A causal mechanism, tested by interventions it was not built from: not established at the level the question
  asks.**
  - From 009 to 019 the confirmations predict the consequences of one intervention, the E-patch on the cue's layer-0
    encoding [prospective].
    - Some READMEs describe the computation itself: 009's read-out, 014's “The claim is about the computation, not a
      label” (014:138), 016's weight-only program, 019's routing rule.
    - None claims a causal status for its decoded variables established by intervening on them.
    - Whether predictions of the E-patch's consequences meet any of C002's Level-3 gates is for the claim review.
  - Four preregistered routes to causal status were designed; three ran, and none established it (§5):
    - 005's confirmation run;
    - 006's circuit families, never run (006 wrote no lock);
    - 007's circuit families, where P3 failed;
    - 025's rotation.
  - Exact identities show only that a program is implemented correctly, “not as evidence”
    (docs/superpowers/specs/2026-09-19-experiment-015-attention-pattern-token-local-design.md:356–357).

## 3. The chain

This is the order in which the records decode the computation, from the cue token to the logit contrast:

```text
cue token w
  → E(w): the layer-0 MLP's token-local encoding at the cue position p_c
  → layers 0–2 at p_c: block-0 attention; the layer-1–2 MLP correction; the layer-1–2 heads' value paths and
    pattern change; block 2's operating point (channel D, neuron 1987, frame-routed neurons)
  → Δx3: the change of the residual entering layer 3
  → L03.H04: transport from p_c to the prediction position p_t
  → layers 4–5 at p_t: the readout
  → Δc: the singular-versus-plural logit contrast
```

### 3.1 The cue representation and the encoded number signal

- **Where the mechanism sits.** C002 localizes the contrast “to `L00.MLP` at the cue position, `L03.H04` at the noun
  position, and the layer-4 and layer-5 MLPs at the noun position” (research/claims/C002-count-cued-noun-number-circuit.md:124).
  - It rests on development data and on nineteen held-out nouns, “never used for ranking”
    (research/claims/C002-count-cued-noun-number-circuit.md:76).
  - Status: C002 `LOCALIZED`, with an exploratory origin. Its Level-3 necessity, sufficiency and control gates are
    unchecked (research/claims/C002-count-cued-noun-number-circuit.md:79–86).
- **The encoding is weight-only.** “`E(w) = MLP₀(ln2₀(W_E[w]))`, weight-only since Experiment 005” (011:7–9).
- **The two tested low-rank descriptions failed.**
  - 006: the leading ≤ 4 unsupervised PCA directions failed the quality gate [exploratory] (006:77–79).
  - 007: a response-supervised rank-1 direction failed Y1, on the determiners `this` and `another` especially
    [unestablished] (007:87–94). Its narrowed reading: “This is a statement about the two estimators tested, not about
    every possible linear map” (007:109–110).
  - The full-dimensional ridge baseline failed the same two tokens [descriptive] (007:95–97).
- **Suppression and opposition** [exploratory]:
  - For `this`, `a` and `another`, the number signal is dropped at transport although the head attends normally
    (008:86–89).
  - For `this`, `a`, `another` and `every`, the on-axis part of the encoding change reads positive through the head's
    direction and the rest negative. The rest is negative “in 24 of 24 frames for each of the four” (010:85–89), and
    layers 1–2 change the read by at most 0.07 net.
  - The pattern is not general. For `these` the rest of the encoding supports the signal (+0.32), and for the
    numerals it is about zero (010:78–83).
- **The encoding read predicts transport.**
  - The zero-parameter encoding read `g_E` predicted how strongly the head transports the E-patched encodings of 24
    never-seen cues in six never-seen frames, Spearman 0.848, MAE 0.077 [prospective] (011:75–76).
  - Its per-token call for `an` (predicted 0.18, measured 0.30) includes “the suppression of `an` by the opposition
    inside its own encoding” (011:99–100) [locked prediction].
  - The support is “generalization across this preregistered heterogeneous cue set, not to arbitrary English words”
    (011:100–102).
  - Experiment 009's fitted rank-1 rule, reported beside it, “would also have cleared the Y1 floors on this set”
    (011:91–92) [descriptive].

### 3.2 Layers 0–2: what is added, opposed and routed before transport

- **The layers-1–2 MLP correction** (012).
  - A frozen token-local MLP model “prospectively predicts most of the net layers-1–2 correction on unseen cues in
    unseen frames” (012:110–112) [prospective].
  - “the design does not claim to have decoded the layer-1–2 MLP correction itself” (012:119–120).
  - `thy` shows the limit. Its total was predicted exactly [locked prediction], but the split between its MLP and
    attention parts, from the own-base ladder, was wrong (012:99–102) [descriptive].
- **Attention value paths** (013).
  - With the heads' patterns held at the reference, the frame's base state and the sixteen heads' value paths explain
    “a large part of the residual the token-local 012 model left” (013:118–121) [prospective; new frames conditional].
  - The MLP/heads attribution was predicted at the pair level [prospective; pooled] (013:106–108). The compensation
    cases carried the predicted signs in 95 % [descriptive] (013:109–112).
  - The remaining error is “the attention-pattern-change term” (013:123–126) [descriptive].
- **The attention-pattern change** (015, 016).
  - 015: a token-local query/key model predicted the pattern change for new cues [prospective]. In new frames it missed
    two floors, the token-mean R² 0.490 and the layer-2 entry R² 0.423, both against 0.50 [unestablished]
    (015:105–111).
  - 016 added three reference-state channels (016:137–141) and predicted the change “nearly exactly” (016:137–144)
    for new cues and twelve new frames [prospective; new frames conditional]:
    - the self-logit operands, 1024 numbers per layer;
    - the LayerNorm scale, two scalars per layer;
    - block 1's MLP at the frame's own cue residual.
  - 016 rejected the scale-only alternative [prospective; pooled] (016:118–119).
  - “This is *not* the claim that three numbers explain the frame, nor that attention is predicted from a frame's
    text” (016:145–146).
- **Block 2's operating point, channel D** (017, 018).
  - The gap 016 left “was block 2's operating point” (017:146–147) [descriptive].
  - A fixed 256-neuron subset carries 0.86 of channel D's contribution to the decoded read, and 0.72 of its
    contribution to the pattern term, on unseen cues [prospective] (018:144–148).
  - “The exact concentration is not fully invariant across new frames” (018:151): one new frame failed the no-harm
    guard [unestablished].
  - The single-neuron account (neuron 1987 alone) is rejected [prospective; pooled and pair-weighted] (018:118–121).
- **Frame-dependent routing** (019).
  - Which neurons carry the correction is “prospectively predictable from the frame's reference state” (019:181–182),
    by the full prospective evaluation [prospective; new frames conditional].
  - A compact operating-point-plus-drive rule “carries most of that routing gain” (019:183) [prospective; new frames
    conditional; aggregate]. It “does not exhaust the additional selectable headroom” (019:185–186), a statement that
    rests on the descriptive witness [descriptive].
  - The rule is not per frame: “not a rule that holds in every frame” (019:164–165).
- **Neuron 1987** (014).
  - Its cue-induced change is reconstructed from its own weight-defined input direction at the frame's reference
    operating point. A number-axis model fails because “off-axis structure in that predicted change suppresses the
    activation for most number- and plural-associated cues” (014:133–138) [prospective; new frames conditional].
  - Its committed per-cue firing calls held for the strong firings and the vetoes [locked prediction] (014:107–114).
- **Block-0 attention** (022, 023). This is part of what the E-patch leaves out of a real cue swap.
  - The remaining upstream error of the 017 chain is localized “primarily to block-0 attention and the state it
    constructs” (022:244–245) [prospective, per condition; Y2 conditional].
  - 023's completed program adds the embedding change, block 0's value and pattern terms at the cue position, and
    block 0's update at the target position (023:17–20).
  - It gave “three PASS and one ENVELOPE_ONLY_FAILURE” (023:152), with no aggregate label.
    - [prospective; Y2 conditional]: Y1/coordinated, Y2/cue_final and Y2/coordinated.
    - [unestablished at its envelope, guard met]: Y1/cue_final, “neither simply a failure nor "basically a pass"”
      (023:155).
  - Its scope is three cue classes (023:183–185). Cheaper block-0 rules recover far less [descriptive] (023:180–181).

### 3.3 Transport by `L03.H04`

- **The head's read-out.** “The head-level mechanism is now a prospectively confirmed statement: the selective
  transport of number by `L03.H04` is a fixed-normalization linear read-out of components already present in the cue
  residual it receives” (009:83–85).
  - Status: [prospective, given the measured input]. The confirmatory part is 23 fresh cues in 6 fresh frames.
  - It was replicated on 142 new pairs, Spearman 0.974 [prospective, given the measured input] (011:87–89).
- **Rules from the encoding alone.**
  - 009's fitted rank-1 transport rule failed Y2, and its contrast rule never passed its Tier A gate [unestablished]
    (009:69–77).
  - The zero-parameter encoding read passed in 011 [prospective]. The fitted rule would also have passed there
    [descriptive] (§3.1).
- **The head's own pattern change.** The attention-row change at `p_t` and the end-to-end `ΔT` were predicted for new
  cues and new frames [prospective; new frames conditional] (017:116–124).
- **Whether the pattern term is necessary.** Not established.
  - 017 Y3: the frozen-pattern alternative reached `ΔT` R² 0.958 on the pooled cue-final pairs (gap 0.042), so it was
    not rejected [unestablished] (017:126–129).
  - Descriptively, with no criterion attached [descriptive], the per-template values are lower: 0.907 (cardinal) and 0.883 (quantifier) over both
    sets, and 0.590 and 0.415 in the new frames alone. On the coordinated pairs, 0.945 with gap 0.053 would have met
    the rejection criterion (017:132–140).
  - 018 Y4, `PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES`: three cardinal frames reached 0.90 or more [unestablished]
    (018:122–128). Its fixed reading keeps “the pattern mechanism (Experiment 017) is not in question” (018:127–128).

### 3.4 The readout to the logit contrast

- **The readout given the measured input.**
  - Given the measured `Δx3`, the frozen readout reproduces `Δc` with ceiling R² 0.9760 on exposed data [exploratory]
    (020:132).
  - On 021's fresh data the ceiling was 0.9683 (Y1) and 0.9652 (Y2) [descriptive] (021:140–141).
  - 023 makes no claim “that the downstream readout is complete” (023:184).
- **End to end, from the decoded upstream program.**
  - 021 tests 020's `p_t` wiring:
    - new cues and new nouns [prospective] (021:116–127), with the caveat that this is “not a claim of good prediction
      everywhere (the floors are relative to the program's own exposed performance, and the token-mean margin is
      small)” (021:171–173);
    - Y1's token-mean R² was 0.016, against a floor clamped to 0.0 (021:116);
    - new frames [unestablished]: the coordinated condition failed against a floor clamped to 0.0 (021:123), and the
      erratum attributes that plausibly to the wiring defect (021:16–17).
  - 023 tests the correctly wired, completed chain. Its `g` is an end-to-end statistic on `Δc`, relative to the ceiling.
    Three conditions passed, among them both new-frame conditions [prospective; Y2 conditional], and one is an
    `ENVELOPE_ONLY_FAILURE`. There is no aggregate label.
    The completed program's R² on `Δc` was 0.9518–0.9610 [descriptive] (023:140–145).
- **Where the readout errs.** A frozen, weight-derived nounness score's rank association with each fresh cue's readout
  error was ρ 0.6108818011257036, against the locked threshold 0.3136960600375234 [prospective] (024:274–279).
  - “The result is predictive, not causal.” (024:373)
  - Its guard covers two simple alternatives only (024:295–297).
  - 23 of the 40 cues lie beyond the calibration range (024:300–301).
  - In spent data, the readout residual is the frozen attention of blocks 4 and 5 [exploratory]
    (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:42–47).

### 3.5 The causal test at the end of the chain

025 rotated fresh cue embeddings along 024's nounness direction.
- **The result.** “The official result is `CAUSAL_EFFECT_NOT_ESTABLISHED`: A = 26/40, B = 23/40, G = 37/40, threshold =
  27, reference tail = 0.01923865414210013.” (025:244–245) [unestablished]
- **What it does not mean.** “It is not a finding of no causal effect.” (025:249)
- **What is descriptive only.** The adjective/noun split (A 20/20 and 6/20), G = 37/40, the causal fraction 0.341, the
  ladder and the related correlations [descriptive] (025:252–262).

## 4. Where prediction succeeds

| stage | what the frozen program predicts | new cues, exposed frames | new frames |
|---|---|---|---|
| encoding → transport | how strongly `L03.H04` transports a cue (011) | — | [prospective]; zero-parameter; `an` [locked prediction] |
| head read-out | the head's output from its measured input (009, 011) | — | [prospective, given the measured input] |
| layers-1–2 MLP correction | the net correction (012) | — | [prospective] |
| layers-1–2 attention values | the residual 012 left, and the MLP/heads split (013) | [prospective] | [prospective, conditional]; the split pooled |
| layers-1–2 pattern change | the attention rows' change (015, 016) | [prospective] | 015 [unestablished]; 016 [prospective, conditional] |
| neuron 1987 | activation change and firing (014) | [prospective] | [prospective, conditional] |
| channel D subset | concentration in 256 neurons (018) | [prospective] | [unestablished] (one frame) |
| routing | frame-specific membership (019) | [prospective]; aggregate | [prospective, conditional]; aggregate |
| head pattern | the attention row and `ΔT` (017) | [prospective] | [prospective, conditional] |
| upstream error | its localization to block-0 attention (022) | [prospective] | [prospective, conditional] |
| end to end, 020's `p_t` wiring | `Δc` (021) | [prospective] for new cues and nouns; small margin | [unestablished] (coordinated) |
| end to end, the completed chain | `Δc`, relative to the ceiling (023) | Y1/coordinated [prospective]; Y1/cue_final [unestablished] at the envelope | [prospective, conditional] |
| readout error | its rank association with nounness (024) | [prospective]; predictive, not causal | — |

009, 011 and 012 tested fresh cues in new frames only, so their exposed-frame cells are empty.

**Every result not established, across the sequence:**
- 005: `BEHAVIOR_NOT_REPLICATED`, because the precondition failed.
- 006: every confirmatory family; the confirmation was never run.
- 007: Y1 (`PROGRAM_NOT_SUPPORTED`), and the C002 review rule, where P3 failed.
- 009: Y2, the fitted transport rule; and Y3, never locked.
- 015: Y2.
- 017: Y3, the pattern term's necessity.
- 018: Y2 and Y4.
- 020: the confirmation was never run.
- 021: Y2.
- 023: Y1/cue_final, at its envelope.
- 025: A and B.

## 5. Where causal establishment stops

- **What the confirmations predict.**
  - From 009 to 019, the frozen programs predict the downstream consequences of the E-patch, an intervention on the
    cue's layer-0 encoding. They predict:
    - how the head transports it (009, 011);
    - what layers 1–2 add (012–016);
    - how block 2 responds (014, 018, 019);
    - the head's own pattern change (017).
  - From 021 to 024, the confirmations use full cue substitutions: 021 predicts `Δc`, 022 localizes the upstream
    error, 023 predicts `Δc` relative to the ceiling, and 024 predicts where the frozen readout errs. 020's
    confirmation was never run.
  - The preregistered alternatives are variants of the program, not interventions in the model. They include the
    axis-only, scale-only, frozen-pattern, single-neuron, subset and selector alternatives.
- **What the records claim for them.**
  - The READMEs state their results “at its safe strength” (013:118).
  - None claims a causal status for its decoded variables established by intervening on them.
  - C002 has not been reviewed against these results. Whether predictions of the E-patch's consequences meet any of
    its Level-3 gates, the necessity- and sufficiency-oriented interventions on confirmatory data, is for the claim
    review. It is not settled here.
- **The preregistered routes to causal status did not establish it.**
  - 005's confirmation run, recorded in C002 as E003: the interventions behind C002's Level-3 necessity, sufficiency
    and control gates were preregistered in 005's lock, but “Not established: E003's precondition failed”
    (research/claims/C002-count-cued-noun-number-circuit.md:82) [unestablished].
  - 006: its circuit families were designed to supply “the confirmatory sufficiency (P1, P4, P5), necessity (P8),
    control (P7), and chain (P9) evidence that C002's Level 3 gates currently lack”
    (docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md:361–363). They were never
    run, because 006 stopped at its quality gate [unestablished].
  - 007 ran them once on fresh data. P1, P4, P5, P7, P8 and P9 passed, but P3's correlation floor failed on both fresh
    sets, 0.696 and 0.840 against 0.90 (007:98–100). By 007's rule, “If they pass on both sets, C002 is reviewed for
    promotion citing them; otherwise C002 is unchanged.” (docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md:251–252)
    - Status: the criterion-level passes are recorded with no outcome force; the review rule was not met
      [unestablished].
  - 025: rotating the cue's input embedding along the nounness direction did not establish its predicted effect on
    the readout error [unestablished] (025:244–249).
- **Interventions on the decoded variables themselves are exploratory or descriptive.**
  - Examples are the axis-only and orthogonal-only E-patches (008, and 009's fresh tokens, reported descriptively).
  - No preregistered, outcome-bearing test intervened on the number axis, the head's read direction, channel D's
    subset, the routing membership or block-0 attention's pathways.
  - 025's intervention on the input embedding is the one exception, and its effect was not established.
- **New frames enter through their measured reference states.** “Nothing here predicts a frame's attention state
  from its text alone.” (docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:262–263)
- **So the boundary is here:**
  - On new cues, every decoded stage passed at least one preregistered criterion: from 013 on in the exposed frames,
    and for 009, 011 and 012 in new frames [prospective]. The end-to-end test on 020's wiring passed with a small
    margin [prospective].
  - On new frames, given their reference states, most stages passed [prospective, conditional]. 015 Y2, 018 Y2 and
    021 Y2 did not [unestablished].
  - The correctly wired end-to-end test, 023's, passed in both new-frame conditions relative to the ceiling
    [prospective, conditional]. The readout's own accuracy on new frames is [descriptive].
  - Much of this predicts the consequences of one intervention, on the encoding.
  - The causal role of the decoded internal variables beyond that is [unestablished].
  - This locates where the evidence changes status. It is not a negative finding about the mechanism.
- **Does a genuine unresolved mechanistic gap remain?** Yes, at the causal level.
  - The decoded variables' causal roles have not been established by preregistered interventions of their own.
  - The one preregistered intervention on a decoded direction, 025's, did not establish its effect.
  - Within prediction, the remaining gaps are the results not established, listed in §4.
  - Whether any of this justifies a new sequence is step 3.

## 6. Limits of the whole account

- **The scale.**
  - One checkpoint; three templates; two to five lexical cue classes (025: 20 adjectives and 20 nouns); single-token
    regular nouns.
  - 24 fresh cues in most experiments (23 in 009, 40 in 024 and 025).
  - 6 to 18 new frames.
  - An exposed noun set that reached 80 nouns, 79 of them scorable, from 008 on (008:12–14).
- **The data.**
  - The E-patch leaves out parts of a real cue swap (§1). 022 and 023 address them.
  - From 013 on, new-frame results are conditional on the frames' measured reference states.
  - 021's labels test 020's `p_t` wiring. 022 adopted 017's wiring after the erratum (022:35–39).
  - Several floors were calibrated relative to the program's own exposed performance, and some were clamped to zero
    skill (021:146–149).
- **The records.** The source map's [record caveats](source-map-005-025.md#record-caveats) apply to every statement
  here. None changes a frozen result.

## 7. Open questions the records name

These are listed as the records state them. None is proposed here as a next experiment; that is step 3.
- **The pattern term's necessity**, 017 Y3 and 018 Y4. It was not established [unestablished]. The descriptive
  values differ between the two [descriptive]:
  - 017's new frames: 0.590 (cardinal) and 0.415 (quantifier), with per-frame values from 0.83 down to −0.46
    (017:132–137);
  - 018's new cue-final frames: 0.771–0.951, three cardinal frames at or above 0.90 (018:122–126).
- **018:** the locked channel-D subset is not robust in every new frame [unestablished] (018:151–155).
- **021:** the correctly wired chain was evaluated on 021's spent set only exploratorily, in 022's `replicate-021`,
  which records numbers and no result (022:261–270). 021's own conditions were not re-scored. 022 and 023 tested the
  correctly wired chain prospectively, 023 end to end relative to the ceiling.
- **022:** “Whether the localized block-0 pathways can be written as a compact weight-derived program is left to a
  later experiment.” (022:256–257) 023 completed them with the exact block-0 rule: three PASS and one `ENVELOPE_ONLY_FAILURE`, Y2 conditional. Whether
  a more compact rule suffices is not established; cheaper rules recovered far less [descriptive] (023:180–181).
- **024 and 025:** the causal role of the nounness direction was not established. The adjective/noun split is
  descriptive only (025:254–255).
- **C002:** its maturity and disposition have not been reviewed against the evidence from 006–025, including 007's
  circuit families. The claim review follows this synthesis.

## 8. What this synthesis does not decide

- It does not change C002. What the synthesis implies for the claim is for a separate claim review.
- It does not propose an experiment. Whether a scientifically justified new sequence exists is step 3.

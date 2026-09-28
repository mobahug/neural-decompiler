# Paper-readiness assessment, 2026-09-28

**Status: for decision; input to step 3 of the synthesis (the decision).** This is revision 3. The first draft's
independent review returned FAIL, and revision 2's re-review PASS WITH NOTES; both are archived, with what was done,
in [`checks/paper-readiness-review-2026-09-28.md`](checks/paper-readiness-review-2026-09-28.md).
The assessment answers four questions the project's reviewer set after the claim work:
1. What is the strongest defensible paper claim?
2. What is the actual novelty, relative to existing mechanistic-interpretability work?
3. Does 005–025 provide enough evidence for a coherent paper now?
4. Is any missing causal evidence important enough to justify a new experimental sequence before writing?

It proposes no experiment to run. Under question 4 it names one primary test, and one secondary candidate, that would
most change the paper, so that the decision can be made. It is built on the two registered claims,
[C002](../claims/C002-count-cued-noun-number-circuit.md) and [C003](../claims/C003-frame-conditioned-decoded-program.md),
and on the [synthesis](synthesis-005-025.md) and the [source map](source-map-005-025.md). Where a statement here
disagrees with them, they win.

## The answers in brief

1. **The claim.** A two-level result with a stated boundary:
   - a localized mechanism (C002);
   - frame-conditioned decoded programs, each stage tested once, prospectively, under its own lock; not every test
     passed, and the composite was assembled afterwards (C003);
   - and an explicit limit: the decoded variables are not established as the model's causal variables.

   The protocol that produced it is part of the paper.
2. **The novelty.** Candidates only. For the behavior, the recorded boundary supports a behavior-specific gap. For the
   method, the candidates are the prospective protocol and its application to a naturally learned behavior in a
   pretrained model. Programs that approximate attention heads in pretrained language models already exist. A
   systematic literature review must confirm any novelty statement.
3. **Enough for a paper now.** Yes, for the two-level claim at its boundary. The gaps a reviewer would raise are named
   in §3; none requires new data to state honestly.
4. **Causal evidence before writing.** Not needed for a coherent paper at the predictive level. It would be needed for
   the strongest original claim. Four preregistered routes to causal status were designed, three ran, and none
   established it. The recommendation is to draft first, and to decide afterwards whether to run one targeted causal
   test (§4).

## 1. The strongest defensible paper claim

**The two registered levels.**
- **C002** (`LOCALIZED`, `ACTIVE`) localizes count-cued noun-number selection to `L00.MLP` at the cue position,
  `L03.H04`, and the layer-4 and layer-5 MLPs. Within its scope, exact replacement of these components recovers about
  98 % of the contrast shift, isolating them retains about 78 %, and the head alone carries about 82 % of the
  coordinated-template shift (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:148–150).
- **C003** (`LOCALIZED`, `ACTIVE`) records the decoded programs. Its boundary: “The programs are prospectively
  predictive; their causal gates are unestablished.”
  (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:391)

**A headline at the boundary.** In an unmodified Pythia-70M, programs written from the weights, each frame's measured
reference state and quantities locked from exposed data were decoded stage by stage. Each stage was tested once, on
data it had not seen:
- **Under the E-patch,** they predicted the consequences stage by stage, from the cue's encoding to the head's output
  (009–019). This held for new cues, and in new frames for most stages, conditional on the frames' measured reference
  states; 015 Y2 and 018 Y2 were not established.
- **Under real cue swaps,** they predicted the contrast change `Δc` end to end:
  - on 020's wiring, for new cues and nouns in exposed frames, pooled over templates, by a small margin; without
    positive skill in coordinated frames; and not in new frames, where its label was negative (021);
  - once block 0 was completed, relative to the ceiling, for new determiner-like, quantity and adjective cues. Three
    conditions pass and Y1/cue_final is an `ENVELOPE_ONLY_FAILURE`; there is no aggregate label (023).
- No decoded program predicted the E-patch's own contrast shift. 009's contrast rule was never locked, and 007's
  rank-1 program failed its Y1 (C003, E002 and E005).
- Every failure is kept, and whether the decoded variables are the model's causal variables was not established.

**What must travel with the headline.** Each item is recorded in C002 or C003:
- **Failures:**
  - 015 Y2, 018 Y2 and 021 Y2;
  - 023 Y1/cue_final, at its envelope;
  - 020's exploratory end-to-end shortfall.
- **Null results:**
  - 005's `BEHAVIOR_NOT_REPLICATED`;
  - the compact programs of 005, 006, 007 and 009;
  - `L03.H04`'s pattern term;
  - 025's `CAUSAL_EFFECT_NOT_ESTABLISHED`.
- **Exploratory negatives:** 008's and 010's `MIXED` summaries, and 010's failed relay bound.
- **C002's basis and scope:**
  - C002 rests on development data and nineteen held-out nouns, which are exploratory and calibration data.
  - 007's P3 pair-level correlation failed: 0.696 and 0.840 against 0.90.
  - It covers two cue pairs. Beyond the tested cue words, `L00.MLP` alone is not the whole cue stage at the pair
    level.
  - Its own failed tests also travel: the templates' readout axes have a minimum cosine of 0.292 against 0.70 (S1),
    P3's retention counts missed on the reserve and new frames, and the cue-word lexicon reached 0.699 against 0.70
    (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:122–129).
- **How the predictions were held.** The locks committed the criteria, the prediction rules, and every prediction that
  needs neither a new frame's reference state nor a measured input.
  - Frame-conditional tables were computed at stage 1 of the single run and held by digest before any target prompt.
    None of them was committed before scoring.
  - The head read-outs apply a locked rule to the measured input.

  All of this is recorded in C003 (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:130–139).
- **Qualifiers:**
  - From 013 on, new-frame results are conditional on the frames' measured reference states.
  - The head read-outs are given the measured input.
  - Under a real swap only `Δc` was scored.
  - 023's scored statistic is `g`; its R² values are descriptive.
- **Margins and floors:**
  - 021's Y1 token-mean R² was 0.016018, against a floor clamped to 0.0 (021:116).
  - 023's Y1/coordinated passed by +7.979e-5 (023:156–157).
  - Several floors were set against the program's own exposed performance, and some were clamped to zero skill
    (021:146–149).
  - 021's labels test 020's wiring, as its erratum records.
- **Scope:** one checkpoint, three templates, single-token regular nouns, the tested cue classes, and the tested
  interventions.
- **The source map's record caveats.**

**What the paper must not claim:**
- a causal mechanism;
- a compact program for the whole computation;
- that the programs predicted the intermediate states under a real cue swap;
- prediction of a frame's state from its text;
- generality beyond the tested scope;
- the necessity of `L03.H04`'s pattern term;
- a causal role of the nounness direction;
- anything about the localization of the three components beyond what C002 supports;
- anything 024 does not claim: that nounness causes rerouting, or that rerouting was measured directly; that the
  model detects nouns; that nounness is a general linguistic representation; or that its guard rules out all lexical
  or semantic confounds (024:302–306).

**The method as a contribution.** The sequence is an application of what 005's design called its protected idea:
“"Prospective neural decompilation" is the protected idea”
(docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:1014).
- A stage is decoded on exposed data and frozen as a program.
- Its lock commits the criteria, the prediction rules and every prediction that needs no new frame's reference state
  and no measured input.
- Frame-conditional predictions are computed at the start of the single confirmation run and held by digest before
  any target prompt.
- Failures, errata and nulls are kept and reported, and claims are graded against an explicit evidence ladder.

The design's full chain (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:1005–1011)
was only partly realized:
- **Realized:**
  - explicit programs decoded on exposed data;
  - numeric predictions for unseen cues, frames and nouns, committed or, for frame-conditional ones, held by digest
    (§1);
  - one execution each;
  - residual accounting against measured ceilings.
- **Not realized:**
  - A program that runs without the network: the programs need each frame's measured reference state, and the head
    read-outs need the measured input.
  - Confirmatory self-repair bounds: 005's confirmation did not count, and 007 met its necessity family only at
    criterion level.
  - Predictions of new interventions: the stage programs predicted the E-patch, the same intervention used in each
    exploration run, on new cues and frames, which C003 records as only partly meeting that criterion
    (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:295–297).

A paper can present the protocol and its first application together.

## 2. Novelty relative to existing work

**The recorded boundary for the behavior.**
- The finalist audit, searched through 2026-09-17, lists close work on grammatical number:
  - inflection subspaces across models including Pythia;
  - subject–verb agreement circuits in Pythia-70M;
  - causal agreement variables across Pythia scales;
  - circuit, neuron and head accounts of agreement in other models.
- Its stated gap is specific to this behavior. It includes that “none predicts held-out multi-component interventions
  for that behavior prospectively” (screening/behavior-candidates/audit-2026-09-17.json:30).
- So the representation and the agreement circuits of grammatical number are not new. The count-cued noun-inflection
  route is the behavior-specific candidate.
- The record's outcome-bearing prospective tests are of the E-patch, a single-component intervention, and of `Δc`
  under real cue swaps. Its multi-component predictions, 005's and 007's circuit families, did not reach outcome force:
  005's precondition failed, and 007 met them at criterion level only, with P3 failing. So the record does not yet fill
  the audit's multi-component gap.

**The recorded boundary for the method.** 005's design names the neighbourhood
(docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:990–1003):
- program extraction from small RNNs on synthetic tasks;
- compact formal guarantees for small transformers;
- solver-checkable circuit explanations for small or post-hoc-modified transformers;
- attention heads in GPT-2 and Llama-scale models approximated by synthesized programs;
- self-repair after ablation as a documented confound.

**Candidate novelty,** each still to be confirmed by a systematic literature review
(docs/RESEARCH_METHODOLOGY.md:396):
1. **The protocol.**
   - A long sequence of preregistered, single-run prospective tests of a mechanistic decomposition, built as
     described in §1.
   - Whether published work runs such a sequence is for the literature review. The brief search below was not
     designed to establish absence.
2. **The application.** For a naturally learned behavior in a pretrained model, weight-derived, frame-conditioned stage
   programs predicted, under the E-patch, the internal states for new cue words from several lexical classes. In new
   frames most stages did so, given the frames' reference states; 015 Y2 and 018 Y2 were not established. Relative
   to the recorded method boundary, which already
   includes head-level programs in pretrained models, the candidate is the whole chain, tested prospectively.
3. **Specific findings,** each at its status:
   - `L03.H04`'s selective transport is a fixed-normalization linear read-out of components already present in the cue
     residual it receives, given the measured input (009:84–85) [prospective].
   - A weight-only read of the encoding predicts that transport (011) [prospective].
   - For `this`, `a`, `another` and `every`, the encoding carries an opposing off-axis part (010:85–89) [exploratory].
   - Channel D concentrates in a locked subset of block-2 neurons on new cues (018) [prospective]; it failed in one new
     frame (018 Y2) [unestablished].
   - The frame's routing of that population is predicted (019) [prospective; new frames conditional; aggregate, not
     per frame].
   - Block 0's terms, the embedding change and block-0 attention, complete the upstream program for three cue classes
     (023) [prospective; Y2 conditional; three conditions pass, one `ENVELOPE_ONLY_FAILURE`].
   - A nounness score predicts the readout's error (024) [predictive, not causal]; the intervention on its direction
     did not establish an effect (025) [unestablished].
4. **The registered boundary.** C003 records the separation of prospective prediction from causal status as a
   registered limit. By its title, the lead 2605.22488 below concerns a related distinction, between what is
   represented and what is computed.

**What is not new:**
- that number is represented;
- that agreement uses attention heads and MLPs;
- activation patching and mediation analysis;
- reading OV and QK circuits from the weights;
- programs approximating attention heads in pretrained models.

**Leads for the review.**
- A brief web search on 2026-09-28 returned titles and search summaries only. None of these papers has been read for
  this assessment, and none is characterized here beyond its title:
  - [Discovering Interpretable Algorithms by Decompiling Transformers to RASP](https://arxiv.org/abs/2602.08857);
  - [Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer](https://arxiv.org/abs/2601.05770);
  - [Represented Is Not Computed: A Causal Test of Candidate Algorithmic Intermediates in a Transformer](https://arxiv.org/abs/2605.22488);
  - [Hypothesis Testing the Circuit Hypothesis in LLMs](https://arxiv.org/abs/2410.13032);
  - [Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](https://arxiv.org/abs/2602.16823);
  - [Certified Circuits: Stability Guarantees for Mechanistic Circuits](https://arxiv.org/abs/2602.22968);
  - [Preregistration for Experiments with AI Agents](https://arxiv.org/abs/2606.11217).
- Method families the review should also cover:
  - causal abstraction and interchange-intervention evaluation;
  - causal scrubbing;
  - automated circuit discovery;
  - attribution graphs and replacement models;
  - compiled and learned transformer programs;
  - the interpretability-illusion critique of subspace patching.
- The close work that 005's design and the audit cite is also to be read in full.

## 3. Is 005–025 enough for a coherent paper now?

**Yes, for the two-level claim at its boundary, together with the protocol.**
- **The record is mapped and audited, with its caveats recorded:**
  - a source map of every experiment;
  - a reviewed synthesis;
  - two registered claims with independent reviews;
  - errata recorded as errata.
- **Locked predictions were tested once each** in 005, 007, 009, 011–019 and 021–025. 006, 008, 010 and 020 wrote no
  lock. The failures are informative, and they are kept.
- **The boundary is sharp and registered:** C003's Level 3 is open.

**The gaps a reviewer would raise, and what the record says:**
1. **Causal status of the decoded variables.** Not established; it is the stated boundary (§4).
2. **One checkpoint, three templates, single-token nouns.** This is a scope limit, stated in both claims.
3. **New frames enter through their measured reference states.** Stated; nothing predicts a frame's state from its
   text.
4. **How the predictions were held.** The frame-conditional tables were held by digest, not committed, before
   scoring. The paper must say so.
5. **The recomputation alternative.** Much of the programs' accuracy may come from exact layer computation given the
   measured reference state. C003 lists this alternative with its test. Evidence bearing on it:
   - At block 0, 023's cheaper rules recovered far less than the exact rule in cue-final frames (023:180–181)
     [descriptive].
   - In block 2, 018's closure ladder measures how much of the exact operating-point computation a neuron subset
     recovers (018:14–18, 130–131). The locked 256-neuron rung on exposed frames is prospective (018:102–104); the
     other rungs are descriptive, and the new frames' result is unestablished.

   The paper should state the alternative. Any new analysis of spent data would be a new computation, labelled
   exploratory.
6. **Baselines.** Descriptive baselines exist on the same fresh quantities:
   - 007's Y4, “reported, never selected against” (007:95);
   - 011's comparison with 009's fitted rule (011:91–94);
   - 023's cheaper block-0 rules (023:180–181).

   The record also has preregistered rival programs (C003's axis-only, scale-only, frozen-pattern, single-neuron and
   template-family alternatives), 018's random and lowest-ranked controls (018:130–134), and 021's descriptive 020
   comparators (021:142–145).

   What is missing is a comparison with standard pipelines for the stage programs of 012–023 and for end-to-end
   `Δc`, such as probe-, patching- or subspace-based predictors. The paper should either add a descriptive comparison
   or say plainly that none was run.
7. **Narrative load.** The paper must compress 21 experiments into one chain, one status map and a short timeline.
8. **Reproducibility.** Code, designs, locks, confirmation sets, reports and extracts are committed. Some measurements
   are kept local by policy and recorded by digest. A release decision is needed.

**Figures that tell the story.** The data would come from committed reports and extracts, and a figure-data audit
comes first. Figures for 023 and 024 would use their committed confirmation records.
- **F1: the chain.** From the cue's encoding to the contrast, each stage with its status.
- **F2: the prediction map.** Stage by population (new cues in exposed frames; new frames), colored by status, as in
  the synthesis's table, with the failures.
- **F3: the head.**
  - 009's read-out, given the measured input.
  - 011's weight-only encoding read, with `an`'s locked prediction marked.
- **F4: the opposition ledger.** The on-axis and off-axis parts for `this`, `a`, `another` and `every` [exploratory]
  (010).
- **F5: channel D's concentration ladder** (018:102–104, 118, 130–131):
  - Y1: `S_4` 0.60, `S_64` 0.72, `S_256` 0.86, `S_1024` 0.98; only `S_256` is prospective, the other rungs descriptive.
  - Y2: 0.50, 0.68, 0.84 and 0.99 [descriptive]; the new-frame label is unestablished, with one frame failing its
    guard.
  - The single neuron: 0.24 on Y1 [descriptive], and 0.219 pooled for its label [prospective].
- **F6: end to end.**
  - 023's scored `g` against its floor `F` per condition, beside the descriptive R² values (023:142–145).
  - 021's descriptive R² on 020's wiring, 0.6167 and 0.5797 (021:132–137).
- **F7: the sequence.** A timeline of 005–025, each experiment with its outcome and status, including the failures and
  025's null.
- **F8: the boundary.** What is predicted, against what is causal, shown as C002's and C003's gates.

## 4. Is missing causal evidence important enough to justify a new sequence before writing?

**What is missing.**
- No preregistered, outcome-bearing test has satisfied necessity, sufficiency or control for any decoded variable
  (C003, Level 3). The one such intervention on a decoded direction, 025's rotation, did not establish its effect.
  C002's Level 3 is open too.
- Four preregistered routes to causal status were designed, three ran, and none established it
  (research/synthesis/synthesis-005-025.md@3d87d3c:83–87):
  - 005's confirmation run;
  - 006's circuit families, never run;
  - 007's circuit families, where P3 failed;
  - 025's rotation.
- Apart from 025, manipulations of decoded parts exist only as exploratory or descriptive records (008, 009).

**Why it matters.**
- The project set out to test whether transformer computations can be turned into explanations that are “increasingly
  complete, causal, and human-understandable” (README.md:3).
- Compactness and human-readability were reached in parts (research/synthesis/synthesis-005-025.md@3d87d3c:53–74).
  Completeness was quantified as residuals against measured ceilings, with no claim that the readout is complete
  (023:183–184). Causality was not reached.
- A decoded variable can predict well without being the one the model uses. The paper's boundary exists for that
  reason.

**What a causal test could show.**
- **Established.** A preregistered test of the central decoded variable, meeting necessity, sufficiency and control,
  would give that stage causal evidence. A claim scoped to that stage could then be reviewed for higher maturity.
- **Not established.** This is the outcome every causal route that ran in the record has reached. It would leave the
  boundary where it is. As 025 says of its own label: “It is not a finding of no causal effect.” (025:249)
- Only a test built with an equivalence bound could show that an effect is below a preset margin. Even that would
  bound an effect; it would not show that a variable goes unused.

**Cost and risk.**
- A new test needs its own design, freeze, lock, a single confirmation and reviews. 021–024 also calibrated floors on
  exposed data; 025 had no calibration phase (025:116).
- The record's history is the main risk: none of the four causal routes established its effect.
- Any new sequence must meet the synthesis's criterion: “a clear, new, falsifiable question”
  (research/synthesis/README.md@3d87d3c:13–14).

**The primary test that would most change the paper,** named for the decision and not proposed to run:
- **The variable.** The component of the residual entering layer 3 at the cue position that the P1 read uses. That is
  the transported number signal as `L03.H04` reads it.
- **Where.** In coordinated frames only, where the head transports across positions. In cue-final frames the cue and
  prediction positions coincide
  (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:76), so a change there acts on every
  downstream component. An alternative is to intervene on the head's value input only.
- **The arms:**
  - *necessity:* remove that component in real cue swaps;
  - *sufficiency:* install it in reference prompts;
  - *control:* an equal-norm direction orthogonal to the read, and the off-read remainder.
- **The predictions.** `Δc` for each arm, from the program, including block 5's path from the cue position (020:43).
- **The requirements:** fresh cues and frames; one confirmation; predictions committed before any target prompt.
  That means either committing the frame-conditional tables at stage 1, before any target prompt, which would also
  close gap 4, or restricting the test to exposed frames.
- **As a question,** against the criterion above: does removing, installing or replacing the component that
  `L03.H04` reads change `Δc` in coordinated frames as the program predicts, beyond an equal-norm control?

**A secondary candidate.** Ablating or installing the locked channel-D subset at the frame's operating point, with the
program's predicted effect. That subset was not robust in every new frame (018 Y2), and its routing is predicted only
in aggregate (019).

**Recommendation.**
1. Do not start a new experimental sequence before writing.
2. Draft the paper at the predictive level, with the boundary explicit and the protocol as a contribution.
3. Run the systematic literature review alongside the draft, before any novelty statement is final.
4. After the draft, decide whether to run the primary test. It is worth running if the paper's main claim, or the
   intended venue, needs causal evidence for the transport stage. It is not worth running only to add to the
   experiment count.

## What this assessment does not do

- It changes no claim, and C002 and C003 stay as registered.
- It runs and proposes no experiment; §4 names one primary test and one secondary candidate for the decision.
- It does not make the novelty statement final; that needs the literature review.
- It does not choose a venue or a paper format.

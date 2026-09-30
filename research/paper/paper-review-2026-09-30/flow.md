# Draft v2: narrative flow and redundancy review

Read-only review of `research/paper/draft-v2-2026-09-30.md`, read as a paper for an ML interpretability audience. Line
numbers are the draft's. Word counts are of rendered text, with HTML comments, image links and the internal header
removed, computed by a script in this scratchpad. Nothing in the repository was changed, and nothing under `outputs/` was
read. Findings already resolved in `reviews.md` are not repeated. Where a proposal moves wording that an earlier review
put in place, the finding says so.

## Overall assessment

Accurate and carefully bounded, but organized as an audited record rather than an argument, so the core contribution is
hard to find. By the end of §1 a mechanistic-interpretability reader knows the task and the contribution labels, but not
why the work matters, what the two interventions are, what a stage program looks like or what was found. The abstract
gives its only numbers to the exploratory localization, and the caveat that decides how to read every prediction (exact
computation given the measured reference state) is asserted five times and explained once, in a §8 bullet. Results
appear in up to five displays, captions restate the text, no figure is used to make an argument, and 64 % of the body is
bullet lists. Every fix below is a move, a cut or a definition, and none changes a number or a claim. Target: about
6,100 rendered words from abstract to conclusion (now about 7,500), with F2, F7 and F8 out of the body.

## Findings

### MAJOR

**1. MAJOR: §1 is a list of definitions, not an introduction (l.92–146).**
- **What is wrong.**
  - Nothing says why prospective tests of a decoded mechanism matter, or where they stand in the field. That material
    exists, but in §7 (l.594–600) and §9 (l.696–709, 727–734), 450 or more lines later. By then, §1 has already framed
    the contributions against it: “found precedents for every ingredient” (l.128–129).
  - The two interventions that organize every result, the E-patch and the real swap, are defined only in §2
    (l.199–206). The E-patch is used before that, at l.113.
  - Contribution 3 is a pointer (l.146). The four findings are listed only in §9 (l.743–750).
  - The question is a 55-word verbatim quotation from an internal design specification (l.117–121). It repeats the
    abstract's opening (l.49–51).
  - The reader of §1 therefore learns what was done only in part, why it matters not at all, and what is claimed only
    as labels.
- **Fix.** Rewrite §1 as an introduction of about 800 words in five paragraphs, built only from statements the draft
  already makes:
  1. **The question, and why it matters.** Take the abstract's question. Add the cautions now at l.594–597 and
     l.702–709: prediction and decoding do not establish causal variables, interventional success can mislead, and many
     circuits can explain one behavior. Add the precedents for frozen predictions at l.730–731. End with the draft's
     hedge, *we are not aware of…*. A sketch assembled from those lines:

     > Circuits in language models are found by path patching and its automated variants, and evaluated by causal
     > abstraction, interchange interventions and causal scrubbing. These evaluations have known limits. Prediction and
     > decoding do not establish a model's causal variables. Interventional success can mislead unless it acts on the
     > aligned variable itself, under stated assumptions. In small networks, many circuits and algorithms can explain one
     > behavior. Recent studies of model internals freeze their predictions or criteria in advance and keep their
     > failures. We apply that discipline to one naturally learned behavior across a long sequence: each tested stage of
     > a decoded mechanism was frozen as a program and tested once, on cues and frames it had never been run on, with
     > the exceptions named in §3.

     The sketch keeps *under stated assumptions*, which the v1 review restored, and *each tested stage … with the
     exceptions named*, which v0's blocker 4 requires. §3 is the protocol section of the proposed outline. The
     citations are those of the lines the sketch comes from.

  2. **What was done.** Use l.53–56, the two interventions from l.199–206, and a schematic of the chain (finding 6).
  3. **The answer, attribute by attribute.** Move the full version from the abstract (l.78–85); the abstract keeps one
     sentence (finding 2).
  4. **The contributions.** Give each its hedged novelty sentence from §9 (l.737–742), and list the four findings of
     l.743–750 under contribution 3.
  5. **A roadmap.**

  Move the task definitions to a *Task, model and interventions* section: the contrast `c(x)`, frames and families, cues,
  the reference cue, `Δc`, exposed and new, and the model. Replace the verbatim question with the abstract's paraphrase.

**2. MAJOR: the abstract is 514 words in five labelled blocks, and its emphasis is inverted (l.47–90).**
- **What is wrong.**
  - Its only numbers, about 98 % and about 78 % (l.60–61), go to the localization. That result is exploratory, and its
    one confirmation failed. The core contribution gets no number.
  - 021 tested a wiring later found defective. It gets 52 words and five qualifiers (l.71–73), against 34 words for
    023's result (l.74–76).
  - It is 54 % bullets, and it cites §9 (l.90).
  - Its heads *First level* and *Second level* (l.58, 65) collide with the Levels 0–6 of the evidence ladder (F8; §2,
    l.209–214).
- **Fix.** Write one paragraph. The sketch below reuses only the draft's approved wording and keeps every qualifier
  earlier reviews required:
  - C002's basis and two-pair scope;
  - C003's origin qualifiers and the tested lexical classes;
  - the attribute-by-attribute answer, in one sentence;
  - the split of the causal routes into components and a direction (v0 blocker 1);
  - 021's scope (v0 re-review S1);
  - `ENVELOPE_ONLY_FAILURE`, *no aggregate label* and the exact-computation caveat.

  It runs to 353 words, or 309 without the bracketed sentence on 021. For a limit of 250 words, cut that sentence first
  and then the attribute sentence. That decision is the authors', since S1 asked for 021's scope.

  > We ask whether the computation by which a pretrained transformer turns a count cue into the choice between a
  > singular and a plural noun can be recovered as an explicit, compact, human-readable causal mechanism, specified
  > before it is tested on examples and interventions it was not built from. In pinned Pythia-70M-deduped we ran 21
  > experiments: stages were formulated on exposed data, frozen as programs and tested once each, in single
  > preregistered confirmation runs; every failure was kept, and the composite account was assembled afterwards. On
  > exploratory and calibration data, for two cue pairs, the singular-versus-plural contrast is localized to the
  > layer-0 MLP at the cue position, the attention head `L03.H04` and the layer-4 and layer-5 MLPs; its one
  > preregistered confirmation failed its behavioral precondition. Stage programs written from three inputs (the
  > weights, each frame's measured reference state, and quantities locked from exposed data) then predicted, by rules
  > frozen before the target prompts were executed, the consequences of an intervention on the cue's layer-0 encoding,
  > stage by stage up to the head's output, for new cues from the tested lexical classes, and in new frames for most
  > stages, given their measured reference states. Under real cue swaps, with block 0 completed, three conditions for
  > new determiner-like, quantity and adjective cues passed relative to the readout's own ceiling (new frames
  > conditional), and one was an `ENVELOPE_ONLY_FAILURE`; there is no aggregate label. [On an earlier wiring, later
  > found defective in coordinated frames, the end-to-end prediction passed by a small margin, pooled over templates,
  > for new cues in exposed frames with exposed nouns; it had no positive skill in coordinated frames, and its
  > new-frame label was negative.] Much of this accuracy may come from exact computation given the measured reference
  > state. Against the question: explicit and specified before testing, largely; compact and human-readable, in parts
  > only; tested on interventions it was not built from, only partly; runnable without the network, no; causal, not
  > established. Of four preregistered routes to causal status, three concerned the localized components and one a
  > decoded direction; three ran, and none established it. The programs are prospectively predictive; their causal
  > gates are unestablished.

  To fix the emphasis, either give contribution 2 one anchor number from §5, chosen by the authors (for example the one
  used in finding 3's worked example), or drop the localization's two numbers.

**3. MAJOR: no stage program is ever shown (§4, l.292–364).**
- **What is wrong.**
  - The core contribution is programs, but the paper never shows one. For no stage does it give the input, the locked
    quantities, the formula, the output and the scored target.
  - §4 describes the stages by experiment history: “012…; 013 added…; 015 and 016 added…; 017 evaluated…”
    (l.339–346).
  - It describes the completed program by lineage: “022's committed upstream chain, the 017 chain built from the 011,
    012 and 017 locks … fed through 020's frozen readout” (l.358–359).
  - The paper's cleanest instance is row 2 of the §5 table. 011's encoding read has these properties:
    - it has zero parameters and depends on the weights alone, apart from a frozen axis (l.411–412, 467);
    - it is not frame-conditional (l.385);
    - its per-cue predictions were committed in the lock (l.191–192), including the `an` call (l.449–451);
    - Figure 3B plots it.
- **Fix.** Add a boxed worked example of about 150 words in §4, on the encoding read, with the formula from 011's
  design. It would follow the program from start to finish:
  1. the input, the cue's encoding `E(w)`, from the weights;
  2. the locked axis, 010's exploratory candidate (l.392);
  3. the prediction, the transport for each new cue;
  4. the target, the measured mean transport under the E-patch;
  5. the lock, committed before the run;
  6. the result: Spearman 0.848, MAE 0.077 on 24 new cues, with `an` predicted at 0.18 and measured at 0.30, the lowest.

  Set Figure 3B beside it. Put l.450–451 in the box too: 009's fitted rule would also have cleared the floors
  [descriptive]. That is the honest comparator. Then describe the other stages by what they compute, with the experiment
  in parentheses.

**4. MAJOR: the caveat that decides how to read contribution 2 is asserted five times and explained once, in §8
(l.88–89, 144, 623–629, 741, 773).**
- **What is wrong.**
  - In the abstract and §1, *much of their accuracy may come from exact computation given the measured reference state*
    reaches the reader without content.
  - The content is a limitation bullet (l.625–629). Block 3's heads are computed by the exact layer program and block 0
    by an exact row rule, so the confirmations mainly test the reductions elsewhere. There is partial, descriptive
    evidence for this at block 0.
  - The facts a reader needs to weigh it are scattered through the paper:
    - block 2's MLP enters “in full, at the frame's operating point” (l.362);
    - the readout runs its MLPs at the operating point and holds its attention at the reference (l.351–354);
    - *`L03.H04` is computed exactly inside block 3* is filed as a “simplification of the diagram” (l.309–311);
    - the head read-outs are tagged *given the measured input* (l.373, 375), but the draft never says what the read-out
      fixes and what it computes.
- **Fix.** In §4, after the worked example, add a short paragraph or a three-column table: the stage, what it computes
  exactly from the measured state, and what it reduces or locks. Assemble it from l.339–355, l.357–364, l.625–629 and the
  C003 lines cited at l.627, and add no new statement. Say there what *given the measured input* leaves to prediction in
  009 and 011. §5 and §6 can then say, at their openings, which rows test reductions and which mostly recompute. §8
  keeps a one-line pointer, and §9 drops its copy (l.740–741). This is how the reviewer's decision to state the caveat
  clearly is carried out: it is explained where the programs are introduced.

**5. MAJOR: each result is displayed up to five times, and three of the displays are tables drawn as images.**
- **What is wrong.**
  - The §5 table (l.371–388) is contained in Figure 2 (l.396–403). F2 also holds the rivals (l.424–435) and the failed
    compact programs (l.453–457).
  - 023's table (l.510–515) is the same as F6 panels A and B, and the bullets at l.517–521 are repeated in the caption
    (l.533–535).
  - Appendix A (l.923–958) is Figure 7; l.951 says so (“Figure 7 draws the same rows”).
  - F1 shows the statuses of §5 and §6 again, stage by stage.
  - F8 is a grid of 25 rows by 2 columns whose columns are identical except for one cell. Its message is F1's causal band
    plus one sentence of §7 (l.574–575). It is also mostly green, 18–19 *checked* cells per column, so it signals
    *checked* in the figure meant to show *not established*.
  - The figures' README (l.194–196) says that in print the text of F1, F2 and F7 falls to about 3 pt.
- **Fix: one home per result.**
  - **§5.** Keep one table in the text. Extend it with F2's rival rows and failed-compact rows, and add a status column.
    Move F2 to the appendix, or drop it. The rival and compact-failure bullets become table rows plus a two-line note.
    The reason for the empty cells, now given only in F2's caption (l.400–402), moves to the table note. The negative
    results stay in the body, as the v0 review required.
  - **§6.** Keep 023's table. Of F6, keep only panel A, whose unique content is the calibration draws' range. Drop panel
    B, which repeats the table's R² columns, and panel C (finding 6).
  - **The sequence.** Keep either Appendix A or F7 in full, not both. A compact sequence table goes into the protocol
    section (finding 8).
  - **F8.** Move it to the appendix. The boundary stays as C003's sentence in §7 and as F1's causal band.

  This saves about 400 rendered words and three full-page images from the body.

**6. MAJOR: the figures are attached to the text, not used by it (l.312, 390–394, 427, 528, 577–578).**
- **What is wrong.**
  - Each figure is cited once. Only Figure 5 is cited in support of a statement (l.427). The others are cited in
    sentences that paraphrase their captions.
  - F2, F3 and F4 are stacked back to back with no text between them (l.396–422), and so are F7 and F8 (l.580–592).
  - F1 sits in §4, before any result. Its badges either depend on §5 and §6 (Y1, Y2, *rival*) or are never defined in
    the text: C1–C4 (§6 never names 022's conditions, l.500–505), c-f and co, `R²_C`, and 021's Y3.
  - F4 illustrates stage 1 of §4, which is exploratory, but it sits in §5 among the prospective results.
  - F5's ladder is explained only in its caption: the nested subsets, the random and bottom-256 controls, and the oracle.
    Its axis symbol `c_L` is the text's *decoded read* (l.382).
  - F6-C plots 021's *flattened* R², 0.6167 and 0.5797. The text reports 021's *token-mean* R², 0.016018, and a negative
    label in new frames (l.490, 494–495). Nowhere does the text say that two R² statistics are involved, and 023's table
    does not say that its R² is the flattened one. A reader will compare 0.016 with 0.96.
  - Every image repeats its caption's title, and much of its text, in a note inside the image (F1, F2, F4, F5, F6, F7,
    F8).
- **Fix: one sentence per figure where its claim is made, with presentation changes only.**

| figure | now | move to | what the text should say, using statements the draft already makes |
|---|---|---|---|
| F1 | §4, before the results | Split it. The top row, with the positions from the code block, becomes the schematic in §4 and replaces the code block (l.297–307). The two lanes become the summary figure at the end of §6 or the start of §7, with a print layout split by lane, as its README suggests. | §5 and §6 each open by naming their lane. §7 points to the causal band. |
| F2 | §5, stacked | the appendix; the extended table replaces it | — |
| F3 | §5, stacked | beside the worked example and the 009/011 rows | Panel B shows the locked predictions against the measurements. `an` was predicted low, at 0.18, and measured 0.30, the lowest of the 24 (l.449–450). Consider dropping A1, which is descriptive and whose scored pairs are not committed (l.407–409). A2 shows the same read-out prospectively. |
| F4 | §5 | stage 1 of §4, or the appendix | The finding stated at l.327–334. |
| F5 | §5, after the rivals | the 018 row | Introduce the ladder in the text. Then: the locked 256-neuron rung closes 0.863 [prospective], 0.843 in new frames, where the label is unestablished; the single neuron closes 0.219, rejected. |
| F6 | §6 | panel A only, beside 023's table | Y1/cue-final falls below its floor but above the 0.90 guard. Name the R² variant at l.490 and in the table header. |
| F7 | §7 | the protocol section, as a compact table, or the appendix | Name it as the key to the experiment numbers. |
| F8 | §7 | the appendix | — |

  Remove the titles and notes inside the images, so that each caption is the single source.

**7. MAJOR: the paper reads as an audited record rather than an argument.**
- **What is wrong.**
  - 64 % of the rendered body is in bullet lists: §2 96 %, §3 84 %, §8 100 %, §9 92 %, §10 89 % and the abstract 54 %.
    The connectives that carry an argument, such as *because*, *so* and *however*, are mostly missing.
  - The body quotes the project's own records 23 times.
  - Several phrases point at the record rather than at the model: *The stages as the records describe them* (l.326),
    *the record compares* (l.633), *C003's evidence E004–E015* (l.369), and *before this draft* and *for this draft*
    (l.605, 636).
  - Experiments are the grammatical subjects. There are 170 experiment-number mentions in 7,544 words, about one every 44
    words; l.358–361 mention seven experiments 13 times in four sentences.
  - There are 46 inline status tags (finding 13).
- **Fix.**
  - Write the abstract, §2, §3, §6, §9 and §10 as prose paragraphs. Bullets suit §8 and short enumerations.
  - Name each result by what it is, with the experiment in parentheses, for example *the head's own pattern change
    (017)*.
  - Keep a verbatim quotation only where the exact wording carries weight:
    - C003's boundary sentence (l.575);
    - 025's *not a finding of no causal effect* (l.561);
    - 023's *not a protocol outcome* (l.521);
    - 024's *predictive, not causal* (l.548);
    - 016's *Nothing here predicts…* (l.619).
  - State the rest in the paper's voice, and keep the trace in the comments, as v1 already did for one quotation.

**8. MAJOR: contribution 1 is described but never shown at work (§2, l.148–197; §6, l.475–526; §10, l.779–780).**
- **What is wrong.**
  - §2 gives six steps and seven bullets of exceptions at the level of detail of internal documentation, for example
    “Committed review records start at 023” (l.174) and “Some README sections first appeared at closure”
    (l.185–186). The exceptions also repeat Appendix A's *kind of test* column.
  - The paper's clearest instance of the protocol at work is told in §6 as a plain chronology, with no framing:
    1. 020's confirmation was not run, because its floors came from a scratch pass with a block-5 error (l.483–484);
    2. the wiring defect was found after closure and recorded as an erratum (l.182–183, 485–486);
    3. 021 was kept as a failure of that program (l.496–497);
    4. 022 localized the remaining error prospectively (l.500–505);
    5. 023 completed block 0 prospectively (l.507–526).
  - §10 mentions the protocol only in its last sentence.
- **Fix.**
  - Open the real-swap section with a framing paragraph that presents 020 to 023 as the protocol's worked instance and
    states only those recorded facts. Point to it from the protocol section and from the conclusion.
  - Condense §2's exceptions to about 80 words: *typically, with the named exceptions of Table X*. Put a table of
    deviations in the appendix, or add columns to Appendix A.
  - Keep every exception that v0's blocker 4 put there. Move the details of review records and restated readings
    (l.171–177, 184–186) to that appendix.

### SHOULD

**9. SHOULD: the arc and the handoffs between sections.**

| section | earns its place? | handoff |
|---|---|---|
| Abstract | yes, at 60–70 % of its length (finding 2) | — |
| §1 | as setup, yes; as introduction, no (finding 1) | no bridge to §2 |
| §2 | yes, as contribution 1, but its exceptions and notation belong elsewhere (findings 8, 10) | ends on the no-harm guard (l.240–241), with no bridge to §3 |
| §3 | as a subsection of the chain (finding 15) | adequate (l.294) |
| §4 | yes; it should carry the worked example and the split between exact and reduced parts (findings 3, 4) | adequate |
| §5 | yes, the core | the fact that motivates §6, that no decoded program predicted the E-patch's own contrast shift (l.463–464), is not used as the bridge |
| §6 | yes | opens with no motivation (l.473). It spends about 300 words on 020 and 021 before its headline, 023. The link from 024 to 025 appears only in §7 (l.560). |
| §7 | yes, the boundary; F7 is misplaced here (finding 6), and l.594–597 repeat §9 (finding 16) | adequate |
| §8 | yes; three bullets are relocated explanations (findings 4, 18, 19) | — |
| §9 | yes, at half its length; the positioning belongs in §1 as well (finding 16) | — |
| §10 | yes, once restructured (finding 17) | — |

- **Fix.**
  - Open §6 with the approved summary of the real-swap results (l.70–76), and with the reason the sequence moved from the
    E-patch to real swaps, if the record states it.
  - Then give 023's table and F6-A, then the path from 020 to 022 (finding 8), and close with 024. Add one sentence
    saying that 025 intervened along 024's direction (§7).
  - Rename §5 *E-patch predictions, stage by stage* and §6 *Real-swap predictions, end to end*, to match F1's lanes.
    *Untouched* (l.366) is one synonym too many for *new*.

**10. SHOULD: notation and conventions are front-loaded, far from where they are used (l.16–45, 216–241).**
- **What is wrong.**
  - 467 words of internal header and conventions precede the abstract, including a legend of 12 items.
  - §2 defines about 20 symbols before any result. Several are used far from their definition:
    - `g`, `F` and `ENVELOPE_ONLY_FAILURE` only in §6, about 270 lines later;
    - `κ`, *rung* and *decoded read* only in §5 and F5;
    - P3, `m_T` and `m_R` once, in §3;
    - channel D is defined by forward reference, “(§4)” (l.233), and then again in §4 (l.343–344);
    - the P1 read is defined at l.221 and again by quotation at l.347–349.
  - The v0 review asked for definitions, and the notation block supplied them. The problem now is where they are, not
    whether they exist.
- **Fix.**
  - Define each term at first use:
    - P3, `m_T` and `m_R` in §3;
    - channel D and the P1 read in §4;
    - `κ`, rung, decoded read and the no-harm guard at the 018 row;
    - `Δx3`, the E-only chain, the ceiling, `g` and `F` in §6.
  - Keep Y1, Y2 and the names of the data splits in the setup.
  - Put a glossary for lookup in an appendix.
  - Move the status legend into the protocol section, next to the evidence ladder.
  - Move the audit trail, the checking and the figures' provenance (l.20–25, 42–45; Appendix B) to a reproducibility
    appendix.

**11. SHOULD: terms that are undefined, appear only in figures, or are named inconsistently.**
- **Undefined in the text:**
  - *cue-final*: first at l.321, then l.412, 434, 502 and 512–515. Name it at l.101: the first two families are
    cue-final, with `p_c = p_t`, and the third is coordinated.
  - `ΔT` (l.381, 434);
  - neuron 1987, which first appears as a table row (l.378) and is never introduced in §4;
  - *channels* (l.342), and *channel D*, which implies channels A–C that the paper never mentions;
  - *E–N guard* (l.545);
  - *reserve nouns* (l.272), *development data* (l.60, 252) and *Tier A* (l.167);
  - *calibration tier* (l.37);
  - *licensed* frames (l.412) and *informative* frames (l.419);
  - *interpretable condition* (l.238);
  - *frozen reading* (l.524);
  - *zero-parameter* (l.467);
  - *pair-centered neutralization* (l.284, 646);
  - `LN₂` (l.201);
  - *two-stage confirm* (Appendix A only);
  - *pairs*, the unit of analysis (l.267, 373, 409, 434);
  - *flattened* versus *token-mean* R² (l.490, 535–537).
- **Only in figures:** `c_L` and `Π`, which the text calls the *decoded read* and the *pattern term*; `R²_C` and C1–C4
  (F1); `T₆₄` and `A` (F2); `S₂₅₆` (F2, F5); `q_T` and `ḡ_E` (F3).
- **Several names for one thing:**
  - *template bases*, *template base states* and *exposed base states* (l.231, 339, 360, 376);
  - three different *reads*: the P1 read (l.221), the encoding read (l.219) and the decoded read (l.234);
  - *E* in six senses: the component E (l.246), `E(w)`, the E-patch, the E-only chain, `E_64` and E–N.
- **Fix.** Add one paragraph on the data splits to the setup: exposed or development data, the confirmation set, new or
  fresh items, the reserve, calibration draws and spent data. Use one name per concept, and define or drop each symbol
  that appears only in a figure.

**12. SHOULD: Y1 and Y2 change meaning (l.217–220), and the §5 table mixes the meanings.**
- **What is wrong.**
  - In 009, Y1 is the head read-out. In 011, Y1 is the encoding read and Y2 the read-out. From 013 on, Y1 and Y2 are
    populations.
  - In the §5 table, *011 Y2* sits in the *new frames* column but means the read-out (l.375). *012 Y1, Y2* are two
    statistics, not two populations (l.376).
- **Fix.** In the body, use Y1 and Y2 only in their sense from 013 on, the sense the figures and the condition names of
  022 and 023 use. Label the rows for 009, 011 and 012 by what they test: *009, head read-out*; *011, encoding read*;
  *011, read-out replicated*; *012, net correction; composite*.

**13. SHOULD: the status tags are a heavy load on the reader (l.26–41 and throughout).**
- **What is wrong.**
  - There are 46 inline tags in 13 variants, and the legend gives 5 statuses and 7 qualifiers.
  - Four of the legend's qualifiers are used once, each in a sentence that already explains it: *calibration* (l.245),
    *exposed base states* (l.376), *aggregate* (l.383) and *at its envelope* (l.519).
  - Three variants are not in the legend: *on spent data* (l.417), *predictive, not causal* (l.542) and *0.90 guard met*
    (l.519).
  - [locked prediction] is used once (l.450).
  - *rejected [prospective]* (l.445) reads as if the rejection had a status of its own.
- **Fix.**
  - State a default for each section, for example: *Unless tagged, results in §5 are [prospective]; every new-frame
    result is conditional*. Then tag only the exceptions.
  - Keep two qualifiers, *conditional* and *given the measured input*.
  - Let tables and figures carry the status as a column or a colour.
  - Say in the legend that the rejection of a rival is prospective when its rejection criterion passed.
  - Aim for 20 inline tags or fewer.

**14. SHOULD: *P1* names two different things (l.221, 357).** One is 009's read; the other is *023's P1*, the
completed program. The note that tells them apart is the last line of Appendix A (l.958), and l.357 is the only place
in the body where the completed program is called P1. **Fix.** Call it *the completed program* there.

**15. SHOULD: §3, §4 and §7 overlap, so merge §3 into the section on the chain.**
- **What is wrong.**
  - E, T and R are components in §3 (l.246–249) and stages 1, 4 and 5 in §4. §3's scope bullet (l.286–288) anticipates
    §4's block-0 stage and §6.
  - 005's failed precondition and 007's failed P3 are told in §3 (l.270–275) and again in §7 (l.556–559), with the same
    citations.
  - 006's early stop is told in §2 (l.160), §5 (l.455), §7 (l.557) and F2's caption (l.402).
  - *About 82 %* (l.253) and *0.816* (l.259) are the same result.
- **Fix.** Make one section, *Where and how*, with four subsections:
  1. the localization, about 250 words: the components, recovery and isolation, the dedicated but not exclusive
     transport, the replication, C002's own failed tests and scope, and its status;
  2. the chain, with F4 at stage 1;
  3. the worked example and the split between exact and reduced parts (findings 3 and 4);
  4. the completed program.

  The failures of the causal routes are told only in §7; this section keeps a pointer to them.

**16. SHOULD: §9 is 1,170 words, 15 % of the body and 92 % bullets, and it partly repeats other sections
(l.662–760).**
- **What is wrong.**
  - *What this work adds* (l.736–750) repeats §1 (l.128–146) and the abstract (l.87–90).
  - Every item of *What it does not add* (l.752–760) is already cited in its cluster above.
  - The principle that prediction does not establish causation is cited in §7 (l.594–597) and again in §9 (l.702–709,
    759–760).
  - Méloux et al. (2026) appear in both §8 (l.657–658) and §9 (l.707–708).
  - The paragraph on the screening audit (l.688–693) answers an internal document the reader has never seen.
  - The opening paragraph (l.664–667) describes process.
- **Fix.**
  - Write each cluster as prose that ends with one sentence on how this work differs. Fold each *does not add* item
    into the cluster that cites it, so that every precedent is named once, with its relation to this work. This
    reverses the lead sentence added in v1, so move that sentence to §1, where the novelty statements go.
  - Move the substance of the screening-audit paragraph to §8 as a limitation, without naming the audit: the
    outcome-bearing tests were of one component, and the multi-component predictions did not reach outcome force.
  - Keep in §7 only its last two sentences (l.597–600), which make this paper's own point.
  - Target about 600 words. Keep §9 late, but put the positioning paragraph in §1 (finding 1): the contributions are
    defined against precedents, and the reader needs those precedents first.

**17. SHOULD: the conclusion does not follow the contributions (l.762–780).**
- **What is wrong.** It runs *Localized*, *Predicted*, *Not established*, and then gives the protocol one sentence.
  Contribution 3's findings are missing, and the *Predicted* bullet repeats the abstract and §1 word for word
  (l.767–771).
- **Fix.** Write three short paragraphs in the order of the contributions:
  1. the protocol, with the instance from 020 to 023;
  2. the programs and their caveats;
  3. the findings, each at its status, with C002 at `LOCALIZED`.

  Then give the boundary sentence and the one test that would move it (l.602–604). About 220 words, in prose.

**18. SHOULD: an inventory of the repetitions.** Estimated savings are in rendered words.

| # | item | where it occurs (lines) | keep | cut | ≈ saved |
|---|---|---|---|---|---|
| R1 | the programs' 30-word formula and its two bullets | 65–76, 136–142, 767–771 | the abstract (condensed); the full definition in §1 or §4 | §10 becomes one sentence | 60 |
| R2 | the exact-computation caveat | 88–89, 144, 623–629, 741, 773 | a clause each in the abstract, §1 and §10; the explanation in §4 (finding 4) | §8 becomes a pointer; §9 l.740–741 | 50 |
| R3 | the four causal routes | 84–85, 270–275, 553–562, 774–777, F8 | §7 in full; one sentence each in the abstract and §10 | §3's two bullets become a pointer | 70 |
| R4 | 006 stopped without a lock | 160, 402, 455, 557, App. A, F7 | §7 and the §5 table row | the §2 bullet (to the deviations table); F2's caption | 20 |
| R5 | the research question | 49–51, 117–121 | the abstract's paraphrase | §1's verbatim quotation | 50 |
| R6 | the opposition ledger (this, a, another, every; either; these; `MIXED`; the relay bound) | 327–334; 417–422 and F4's own note; 744–747 | §4's stage 1, with F4 placed there under a caption that says what is plotted | the caption's restatement; the note in the image | 50 |
| R7 | the chain diagram and its two simplifications | 297–311; F1's top row; its caption 322–324 | one schematic, with the two notes in its caption | the code block; l.309–311 | 110 |
| R8 | the P1 read | 221–222, 347–349, F3's caption | §4's quotation, at first use | the §2 notation entry | 20 |
| R9 | channel D | 233, 343–344 | §4 | the §2 entry and its forward reference | 15 |
| R10 | tables held by digest | 192–194, 621–622 | the protocol section | the §8 bullet | 20 |
| R11 | the head alone | 253 (about 82 %), 259 (0.816) | l.259 | the *about 82 %* clause | 12 |
| R12 | the §5 table and F2, with the rivals (424–435) and the compact failures (453–457) | — | one extended table | F2 (to the appendix); the bullets become rows | 200, and a page |
| R13 | 023's table and F6 A/B; the bullets at 517–521 and the caption at 532–540 | — | the table, and F6-A with a caption of about 50 words | F6-B, F6-C and the restatement in the caption | 100, and two panels |
| R14 | Appendix A and F7 | 923–958, 580–585 | one of them (finding 5) | the other | 75–95 in the body |
| R15 | the header's conventions and Appendix B, against the captions | 16–45, 960–979 | a reproducibility note of three sentences, in an appendix | the table; the header | about 250 of front matter |
| R16 | the literature on prediction versus causation | 594–597, 702–709, 759–760 | §9, with its citations; in §7, l.597–600 | §7's first two sentences; §9's *does not add* bullet | 60 |
| R17 | Méloux et al. (2026) | 657–658, 707–708 | §8 | the §9 sub-bullet | 30 |
| R18 | *What this work adds* and *What it does not add* | 736–760, against 87–90, 128–146 and 669–734 | §1's contributions, with their hedges | both lists (finding 16) | 250 |
| R19 | *no aggregate label* | 75–76, 505, 521, 535; F1, F6's own note, F7, App. A | the abstract; §6 once, with the quotation | F6's caption and its note | 15 |
| R20 | new frames are conditional | 34, 69, 220, the tags at 377–383, 618–620; the captions of F1, F2 and F7 | a default stated once, and §8's quotation from 016 | the tags on each row | 40 |
| R21 | §2's exceptions and Appendix A's *kind of test* column | 159–177, 925–947 | a table of deviations | the bullets, condensed to about 80 words | 140 |
| R22 | the notes inside the images and the captions | F1, F2, F4–F8 | the captions | the titles and notes inside the images | vertical space |

The total is about 1,400 words from the body and about 250 from the front matter.

**19. SHOULD: the setup leaves out facts that a reader needs to follow the chain.**
- **What is wrong.**
  - **The architecture.** The paper never gives the number of layers and heads, the naming convention behind `L03.H04`,
    or the fact that attention and the MLP in a Pythia block read the same input. The draft's “runs in parallel” (l.309)
    is filed as a simplification of the diagram. F5's `S₂₀₄₈` implies the width of the MLP.
  - **The scale of the data.** The sizes of the fresh sets appear only as a limitation (l.614–617), and the unit of
    analysis, *pairs*, is never defined.
- **Fix.** Add about 100 words to the setup, as the record's designs state these facts. §8 keeps a pointer.

**20. SHOULD: the submission build (the comments, the header, the links and anonymity).**
- **The comments.** Comments open on 173 lines, some of them inside table cells and list items that span lines.
  - None begins a line.
  - Pandoc's LaTeX writer drops raw HTML, so they would vanish in that path. Its HTML output keeps them in the page
    source unless comments are stripped. A LaTeX port by hand would probably turn them into `%` comments, and arXiv
    publishes the source.
  - Ten of them name internal paths (`docs/superpowers/specs/…`), and many name commits.
  - I could not test a build: pandoc is not installed here.
- **Anonymity.** The rendered references name the repository together with its owner's account (l.786), which breaks
  double-blind review.
- **The rest.** The header (l.3–45) is internal. The body links to repository files by relative path (l.659–660, 664).
- **Fix.** Generate the submission from a stripped copy, by a script that does the following:
  - removes the comments and the header;
  - turns the relative links into appendix references;
  - anonymizes the link to the repository.

  Keep the traced source in the repository as the canonical version, and run the checker only on that source.

### MINOR

21. **The lists of lexical classes render without labels (l.108–110).** The experiments they belong to (011, 023, 025)
    exist only in hidden comments. Label each list in the text.
22. **Two orphan sentences in §1.** *The behavior's origin* (l.126) stands alone: say in one sentence, from the
    screening record, what the screen chose among, or cut it. *(005–025)* (l.53) starts at 005 with no explanation; one
    clause is enough.
23. **The contributions' lead hedges before the list (l.128–129).** Put the hedge after the list, or in each item as
    finding 16 proposes.
24. **The programs' formula can be misparsed (l.65–66, 136–138, 767–768).** *Written from the weights, each frame's
    measured reference state and quantities locked from exposed data* can be read with *each frame's measured reference
    state* as an apposition. Write *written from three inputs: the weights, each frame's measured reference state, and
    quantities locked from exposed data*.
25. **§7 says one thing twice (l.564–566):** a paraphrase, then a quotation of the same sentence. Keep the quotation.
26. **Process voice (l.605, 636, 640).** *It was deliberately not run before this draft*, *None was run for this draft*
    and *A descriptive comparison would be a new, exploratory computation on spent data* should become *We leave it to
    future work* and *We did not run such a comparison*.
27. **The §5 table mixes separators:** *0.957 / 0.066* in some rows, *0.877, R² 0.668* in others. Its note does not say
    why the first four rows have *—* under *exposed frames*.
28. **The evidence ladder has two sets of names.** §2's names for its rungs (l.209–210) differ from F8's level titles,
    and the text never ties `LOCALIZED` (l.290) to a level number. Name the levels once, as F8 does.

## Proposed outline of the revised paper

The budgets are rendered words and include captions and tables. They total about 6,150 from the abstract to the
conclusion, against about 7,500 now. For a 9-page NeurIPS or ICLR format, reach about 5,000 by moving these to the
appendix: §8's detailed bullets, most of §9, the rival and compact rows of §5, and the detail of the localization and of
the path from 020 to 022.

| # | section | ≈ words | content, and where it comes from |
|---|---|---|---|
| — | Abstract | 310–350 | one paragraph (finding 2) |
| 1 | Introduction | 800 | the question and why it matters; what was done, with the chain schematic (F1's top row); the answer, attribute by attribute; three contributions with their hedges, and the four findings under contribution 3; a roadmap (finding 1) |
| 2 | Task, model and interventions | 450 | `c(x)`, frames and families (naming cue-final and coordinated), cues, the reference cue, `Δc`, the data splits and their sizes, the architecture facts; the E-patch, the real swap and the rotation (from the current §1 and §2; findings 11, 19) |
| 3 | The prospective protocol | 550 | the six steps; what the locks held; the exceptions in brief, with the table in the appendix; the evidence ladder and the status legend with its defaults; a compact sequence table (findings 8, 10, 13) |
| 4 | Where and how: components and the decoded chain | 950 | the localization (C002) as a subsection; the stages, with F4 at stage 1; the worked example (the encoding read, F3B); exact versus reduced; the completed program (findings 3, 4, 15) |
| 5 | E-patch predictions, stage by stage | 800 | a summary sentence first; the extended table with its status column; F3 and F5 used in the text; the rival not rejected; *What these predictions are not*; a bridge to §6 (findings 5, 6, 12) |
| 6 | Real-swap predictions, end to end | 650 | the summary and the reason for real swaps; 023's table and F6-A; how the program was reached, 020 to 022, as the protocol at work; 024, and a bridge to 025 (findings 8, 9) |
| 7 | Causal tests and the boundary | 400 | the four routes, told once; the manipulations outside any outcome; C003's boundary sentence, with l.597–600; the test that would move it; F1's lanes as the summary figure (here or at the end of §6) |
| 8 | Limitations | 450 | one checkpoint and uniqueness; the small fresh sets (a pointer); the baselines; self-repair; subspace interventions; validation by prediction; the multi-component gap (moved from §9) |
| 9 | Related work | 600 | four clusters in prose, each ending with how this work differs (finding 16) |
| 10 | Conclusion | 220 | the contributions in order; the boundary; the one test that would move it (finding 17) |

**Appendices.**
- **A.** The sequence: Appendix A's verbatim outcomes and statuses, or F7, with the table of protocol deviations from
  §2.
- **B.** A glossary of symbols and data splits.
- **C.** Supplementary figures: F2 if it is kept, F8, and F1 at full size in its print layout.
- **D.** Reproducibility and the audit trail: the pins, the checker, the figures' provenance and the record caveats,
  from the current header, Appendix B and l.659–660.

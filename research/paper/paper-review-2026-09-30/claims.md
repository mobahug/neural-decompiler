# Claim-strength review: `research/paper/draft-v2-2026-09-30.md` (uncommitted)

Read-only. Nothing in the repository was modified; nothing under `outputs/` was read. The checker
(`research/synthesis/checks/check_source_map.py`) exits 0: 32 quotes, 183 citation groups, 2 commit hashes, 41 labels,
PASS. Every one of the 183 citation groups was also dumped against its cited lines and read.

## Verdict: PASS WITH NOTES

No blocker: no status tag contradicts the source map, no statement makes the decoded variables causal, and every
known trap is handled. Six SHOULD items, all wording or trace fixes. The most substantive is finding 1: the frozen
reading of 023's failed condition is quoted with an adjective, "small", that the frozen reading does not contain.

## Findings

### 1. SHOULD: line 524. "small" is attributed to 023's frozen reading

- **Sentence:** "Under the frozen reading, Y1/cue-final's failure is “a small quantitative shift, not a refutation”".
- **Record:**
  - The frozen reading says only "a quantitative shift, never a refutation"
    (experiments/023-block0-completion/preregistration.md@d7c5e62:23). The design says the same, relative to
    exposed-like sets (docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md@d7c5e62:476–477).
  - "small" first appears in the README's closure note (023:154), written after the result.
  - Descriptively, g 0.995520 lies below 9,999 of the 10,000 exposed-like calibration draws: minimum 0.995470, CDF
    0.0001 (experiments/023-block0-completion/evidence/final-report-2026-09-24.md@d7c5e62:19, 30). It is small in
    absolute g, but not small relative to the envelope. The adjective softens a failed condition.
  - The same wording is in C003 E022 (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:186). Whether
    C003 gets a documentation-only note is a separate decision.
- **Corrected:** "Under the frozen reading, it is “a quantitative shift, never a refutation”<!--
  (experiments/023-block0-completion/preregistration.md:23) -->. 023's closure note calls the shift small<!-- (023:154)
  -->." Optionally add: "Its `g` lies below 9,999 of the 10,000 exposed-like calibration draws<!--
  (experiments/023-block0-completion/evidence/final-report-2026-09-24.md:19, 30) --> [descriptive]."

### 2. SHOULD: lines 744–750. §9's findings carry no statuses, and two read as model-level facts

- **Sentences:**
  - "a frame-routed population of block-2 MLP neurons carrying an operating-point dependence;"
  - "block-0 attention completing the upstream program;"
  - "a nounness-associated readout error."
  - The list is headed "each at its recorded status", but only the off-axis part carries a tag.
- **Record:**
  - "Carrying" is a closure fraction of the program's own masked chain, not an intervention in the model
    (018:10–18).
    - It is prospective only for new cues in exposed frames. In new frames the label is negative (018:106–117;
      research/synthesis/source-map-005-025.md@b722e9c:196–198).
    - It is "not a uniquely defined circuit" (018:156–157).
    - The routing is "not a rule that holds in every frame" (019:164–165).
  - "Completing" reads as a success, yet 023 records three PASS, one `ENVELOPE_ONLY_FAILURE` and no aggregate label
    (023:150–155; research/synthesis/source-map-005-025.md@b722e9c:92).
  - 024 is "predictive, not causal" (024:373).
  - As worded, the population bullet edges toward the causal reading that §7 disclaims.
- **Corrected:**
  - "a population of block-2 MLP neurons whose operating-point dependence a locked subset largely recovers for new
    cues in exposed frames [prospective; new frames unestablished], with frame-dependent membership predicted
    [prospective; Y2 conditional; aggregate, not per frame]; not a uniquely defined circuit;"
  - "the E-only chain's remaining upstream error, localized primarily to block-0 attention [prospective; Y2
    conditional]; with block 0 added, three conditions passed and one was an `ENVELOPE_ONLY_FAILURE`, with no
    aggregate label;"
  - "a nounness-associated readout error [prospective; predictive, not causal]."
  - Also tag the first bullet's read-out [prospective, given the measured input].

### 3. SHOULD: lines 53, 87 and 130. "the mechanism" presupposes what the answer leaves unestablished

- **Sentences:**
  - "Stages of the mechanism were formulated on exposed data, frozen as programs, and tested once each" (53).
  - "a long sequence of single-run prospective tests of one mechanism" (87).
  - "Twenty-one experiments on one mechanism" (130).
- **Record:**
  - The abstract defines "the mechanism" as the causal mechanism asked for, and its answer is that causal status is
    not established (lines 49–51, 84–85).
  - The literature review's candidate 1 is "tests of a mechanistic decomposition"
    (research/paper/literature-review/review-2026-09-28.md@70caf7d:12).
  - C003 speaks of the program's stages (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:9–12,
    46).
- **Corrected:**
  - "Stages of a candidate mechanism were formulated on exposed data, frozen as programs, and tested in single
    preregistered confirmation runs."
  - "a long sequence of single-run prospective tests of one mechanistic decomposition"
  - "Twenty-one experiments on one mechanistic decomposition"
  - §9's "decodes one mechanism stage by stage" (line 737) is the review's own narrowing wording and can stay.

### 4. SHOULD: line 58. The abstract's localization sentence drops C002's scope

- **Sentence:** "The singular-versus-plural contrast is localized to three components: the layer-0 MLP at the cue
  position, the attention head `L03.H04`, and the layer-4 and layer-5 MLPs." Only the bullet beneath narrows the
  numbers to two cue pairs.
- **Record:**
  - C002's supported conclusion holds "Within pinned Pythia-70M, the three screening templates, single-token regular
    nouns, and the tested replacement distributions"
    (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:150).
  - C002 excludes `L00.MLP` alone as the cue stage for other cue words
    (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:158).
  - §3 (line 286) and §10 (lines 764–766) carry this scope; the abstract does not.
- **Corrected:** "For two cue pairs in three screening templates, the singular-versus-plural contrast is localized to
  three components: the layer-0 MLP at the cue position, the attention head `L03.H04` and the layer-4 and layer-5
  MLPs, both at the prediction position." Then trim "For two cue pairs," from the first bullet.

### 5. SHOULD: line 738. The protocol's precedent is narrower than the review found

- **Sentence:** "Each practice is established, and one preregistered study combines several of them (Xun, 2026)."
- **Record:**
  - The review's verdict: "several 2026 studies of model internals combine several of them (D1–D3, C1, C13; Buehler,
    under C)" (research/paper/literature-review/review-2026-09-28.md@70caf7d:23–25).
  - Among them, D3 has "preregistered kill criteria" (review:109), and C13's arXiv record "calls the framework
    pre-registered" (review:960).
  - The sentence leaves one precedent where the review found several, which slightly overstates candidate 1.
- **Corrected:** "Each practice is established, and several recent studies of model internals combine several of them
  (Xun, 2026; Zhu, 2026; Dong, 2026; McCann, 2026; Somani, 2026; Buehler, 2026); the closest found is a preregistered
  one (Xun, 2026)."

### 6. SHOULD: line 409. A caption statement is traced to lines that do not state it

- **Sentence (F3 caption):** "Y1 was scored over its 138 pairs, which are not committed<!-- (009:62–64) -->".
- **Record:**
  - 009:62–64 give the 138 pairs and the statistic, not their commit status.
  - The source is the figure-data audit (research/paper/figures/figure-data-audit-2026-09-30.md@e65c683:154–157;
    also research/paper/figures/README.md@70caf7d:190).
  - The header promises captions "traced like the body", and the checker cannot catch a paraphrase whose trace does
    not support it.
- **Corrected:** "Y1 was scored over its 138 pairs<!-- (009:62–64) -->, which are not committed<!--
  (research/paper/figures/figure-data-audit-2026-09-30.md@e65c683:154–157) -->."

### 7. NOTE: line 345. The channel-D sentence in §4 is unscoped

- **Sentence:** "A locked subset of block-2 neurons largely recovers channel D, and the frame's routing of that
  population is predicted in aggregate."
- **Record:** this is C003's hypothesis wording (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:55–57),
  so it is allowed. But 0.863 holds only for new cues in exposed frames; in new frames the no-harm guard failed
  (018:102–117).
- **Suggested:** "... largely recovers channel D for new cues in exposed frames [prospective], though not in one new
  frame [unestablished], ..."

### 8. NOTE: line 328. `every`'s on-axis part is weak, and 008 classed it an artifact

- **Sentence:** "For `this`, `a`, `another` and `every`, the encoding carries an on-axis number signal and an opposing
  off-axis part".
- **Record:**
  - 010 supports the sentence, but `every`'s axis part is only +0.17 (010:86).
  - 008's frozen rule classed `every` `AXIS_ARTIFACT`: it "does **not** carry the mechanism-axis signal" (008:54–56).
- **Suggested:** add "; for `every` the on-axis part is small (+0.17), and 008's frozen rule classed it an axis
  artifact<!-- (008:54–56) -->."

### 9. NOTE: line 258. A section heading asserts H1 without its tag

- **Sentence:** "**Transport is dedicated but not exclusive.**" The [exploratory] tag sits only at the end of the second
  bullet.
- **Record:** H1 was chosen on development data (005:90–93), and H2 is "disfavored on development"
  (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:134).
- **Suggested:** "**On development data, transport runs mainly through one head, but not only through it**
  [exploratory]."

### 10. NOTE: line 263. The 007 replication carries no tag

- **Sentence:** "**Replication at criterion level.** Experiment 007 recorded the same magnitudes, without outcome
  force, on twenty fresh nouns in the original frames and on fresh frames".
- **Record:**
  - The source map classes these values descriptive (research/synthesis/source-map-005-025.md@b722e9c:129–130).
  - The fresh frames "lie beyond [C002's] declared frames"
    (research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:53).
- **Suggested:** add [descriptive], and "the fresh frames lie beyond C002's declared frames".

### 11. NOTE: line 382. The closure fraction is described imprecisely

- **Cell:** "closure fraction 0.863 of the decoded read, 0.724 of the pattern term".
- **Record:** κ is the share of channel D's *contribution* that the subset closes, not a share of the read itself
  (018:17–18, 102–104; research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:177).
- **Suggested:** "κ 0.863 of channel D's contribution to the decoded read, 0.724 of its contribution to the pattern
  term".

### 12. NOTE: line 564. 025 is framed as an intervention on the chain's variables

- **Sentence:** "Apart from 025's rotation, no preregistered, outcome-bearing test intervened on the chain's decoded
  variables."
- **Record:**
  - 025 rotated the input embedding along 024's nounness direction and scored the frozen readout's approximation
    error (025:63–69, 80–88). The direction is not a stage variable of the chain.
  - C003 says "a decoded direction" (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:194–195).
- **Suggested:** "Apart from 025's rotation along 024's nounness direction, no preregistered, outcome-bearing test
  intervened on a decoded variable."

### 13. NOTE: line 690. The list of outcome-bearing tests is incomplete

- **Sentence:** "Its outcome-bearing prospective tests are of the E-patch, a single-component intervention, and of
  `Δc` under real swaps."
- **Record:** the list omits 024's association and 025's rotation (Appendix A, lines 946–947).
- **Suggested:** "... of `Δc` under real swaps, of 024's association and of 025's single-direction rotation."

### 14. NOTE: line 169. 024's binding threshold was not an exposed-draws floor

- **Sentence:** "In 021–024 they were calibrated on exposed-only draws."
- **Record:** 024's rule was ρ ≥ max(F_ρ, null₉₇.₅), and "the chance-level null was the binding threshold"
  (024:282–286, 360–363).
- **Suggested:** "In 021–023 they were calibrated on exposed-only draws; in 024 a permutation null was the binding
  threshold."

### 15. NOTE: lines 72–73. The abstract's negative new-frame label invites a broader reading

- **Sentence:** "It had no positive skill in coordinated frames, and in new frames its label was negative."
- **Record:** the label "comes solely from `coordinated_r2`"; three of four conditions passed (021:134–137).
- **Suggested:** "... and in new frames its label was negative, from the coordinated condition alone."

### 16. NOTE: the captions omit qualifiers that the figures' own notes carry

- **F1 (lines 317–324):**
  - It does not define the Y3 badge, 021's new nouns over the Y1 pairs.
  - It does not describe the causal-status band.
  - Its *rival* wording lacks "its new-frame share is conditional" (research/paper/figures/README.md@70caf7d:70–80).
- **F5 (line 445):** the star's "rejected [prospective]" lacks the pooled qualifier
  (research/paper/figures/README.md@70caf7d:122–123). Write "[prospective; pooled]".
- **F7 (lines 582–585):**
  - It omits that 009's and 011's head read-outs are given the measured input, which their plain-prospective colour
    does not show.
  - It omits that the never-run chips of 006 and 020 are not labels
    (research/paper/figures/README.md@70caf7d:142–147).

### 17. NOTE: line 313. F1 does not draw every result

- **Sentence:** "with the recorded status of every result at its stage".
- **Record:** F1 does not draw 013's Y3 or 019's Y3 and Y5 (Appendix A, lines 935 and 941).
- **Suggested:** "of each result it draws".

### 18. NOTE: lines 539–540. The caption gives only one reason why Panel C is not comparable

- **Sentence:** "... through `p_t` instead of the cue position `p_c`, so Panel C is not comparable with A or B."
- **Record:** the audit names three reasons: the wiring, the cues, and 021's pooled templates against 023's split
  (research/paper/figures/figure-data-audit-2026-09-30.md@e65c683:307–308).
- **Suggested:** "The wiring and the cues differ, and 021 pools its three templates where 023 splits them, so ...".

### 19. NOTE: C003's maturity is never stated

- **Gap:** the draft gives C002's maturity (line 290) but never C003's `LOCALIZED`
  (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:5).
- **Suggested:** add "C003 is registered at `LOCALIZED`" beside the boundary quote in §7 or in §10.

### 20. NOTE: minor precision

- **Line 424:** "Four preregistered rivals" is followed by five tests, since 014 and 015 are counted together. Write
  "Four kinds of preregistered rival, in five tests".
- **Line 68:** "given their measured reference states" applies from 013 on. For 009's and 011's read-outs the
  qualifier is the measured input, and 011 Y1 and 012 are not conditional (line 385;
  research/synthesis/source-map-005-025.md@b722e9c:52–64).
- **Line 964:** "byte-identical from run to run" holds with matplotlib 3.11.2
  (research/paper/figures/README.md@70caf7d:28–29).

## Checked and found correct

- **Numbers:** every number in §3–§6, the §5 table, the 023 table and the captions matches its cited lines:
  - 009, 011–019 and 021–025 against their READMEs at `d7c5e62`;
  - 005's, 007's, 009's, 010's and 018's evidence reports;
  - C002 and C003 at `eeff9ff` and `454327d`;
  - the source map at `b722e9c` and the synthesis at `3d87d3c`.
- **Appendix A:** 20 of its 21 rows are byte-identical to the source map's overview (@b722e9c:75–92, 94–95). 005's row
  differs only by the dropped closed-in column.
- **Status tags and qualifiers:** every tag in §3–§7 and in the captions matches the source map's evidence-status
  blocks (@b722e9c:97–248). That covers:
  - given the measured input (009 Y1, 011 Y2);
  - exposed base states (012);
  - Y2 conditional from 013 on;
  - aggregate (019);
  - at its envelope (023 Y1/cue_final);
  - pooled (the rivals);
  - [descriptive] for 021's −0.014, 023's R² values and the cheaper rules;
  - [unestablished] for 005 v2, 007, 009 Y2, 015 Y2, 018 Y2 and Y4, 017 Y3 and 021 Y2.
- **Traps:**
  - 021's Y1 is "with exposed nouns" and Y3 holds the new nouns (lines 71–72, 489–491, F6); no "new cues and nouns"
    wording survives.
  - The off-axis part is limited to `this`, `a`, `another` and `every`, with `either` positive and `these` supporting
    (328–331, 419–422, 744–747).
  - 023 has no aggregate label (75–76, 521, 535, F7), and "three of four" appears only in 023's own negating quote.
  - 022's C4 is given as positivity only, with no share (504).
  - 018 Y2 and the pattern-term necessity are unestablished (382, 433–435, F2, F5).
  - 025 is never "no causal effect", and no A count or "near miss" appears (560–562, 775).
  - The gates of C002 and C003 appear as recorded (290, 766, 775–777, F8).
- **Causal boundary:** stated in §1 (144–145), §4 (294–295), §7 (574–600) and §10. The routes are split three for
  C002's components and one for a decoded direction, as the first review required.
- **Literature:** the §7 and §9 comparisons match the review's entries A1–A13, B1–B12, C1–C13 and D1–D14 and its
  "partly new" verdicts. The wording is "we are not aware"; "novel" and "first" never appear.
- **New v2 text:** the header, the Figures convention and Appendix B were checked against the F1–F8 PNGs, the figures
  README and the figure-data audit. Every plotted value and badge status matches the record, for example:
  - F2's cells;
  - F5's 0.863, 0.843 and 0.219 against 0.860;
  - F6's g, F, draw ranges and R²;
  - F7's chips;
  - F8's gate states.

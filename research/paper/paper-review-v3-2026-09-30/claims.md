# Claim-strength review: `research/paper/draft-v3-2026-09-30.md` (uncommitted)

**Verdict: FAIL.** There is one blocker. The core claim lost v2 revision 1's qualifiers in three places: new cues,
most stages in new frames, and conditional on the frames' reference states. Everything else is at SHOULD or NOTE
level, and every item has a short wording fix.

This review was read-only. Nothing in the repository was changed, and nothing under `outputs/` was read. The draft was
compared line by line with v2 revision 1, as committed at `2bc13a9`.

## Checks

- **Checker:** `check_source_map.py` exits 0, with 26 quotes, 209 citation groups, 2 commit hashes and 44 labels.
- **Rendering:** checked with CommonMark (`markdown_it`). All 208 comment blocks stay hidden; the only escaped marker is
  the intended code span in Appendix D.
- **Traces:** every new or changed citation group was dumped against its cited lines, 24 new groups in all.
- **Table A1:** byte-identical to v2 revision 1's Appendix A, which matches the source map's overview.
- **References:** every in-text citation has a reference.
- **Figures:** the new F1a and F6a and the retitled F1, F2, F3, F4, F5 and F7 were checked against their captions.

## Findings

### 1. BLOCKER: the core claim dropped its scope and qualifiers

**Where.**
- Line 60, the introduction: "The programs predicted the consequences of two interventions: the **E-patch**, … stage
  by stage; and a **real cue swap**, which replaces the cue token, end to end."
- Lines 92–93, contribution 2: "the programs predicted the internal consequences of the E-patch stage by stage, and the
  contrast change under real cue swaps within the margins and failures of §6."
- Lines 583–585, the conclusion: "… predicted, by rules frozen in advance, the consequences of an encoding intervention
  stage by stage, and the contrast change under real cue swaps …".

**What the record says.**
- C003's supported conclusion holds "for new cues [prospective], and in new frames for most stages [prospective,
  conditional; 015 and 018 Y2 unestablished]" (C003@eeff9ff:375–377).
- Under a real swap, "only the contrast change `Δc` at the prediction position was scored prospectively"
  (C003@eeff9ff:42–44).
- v2 revision 1 carried the qualifier in contribution 2 and in §10 (draft-v2@2bc13a9:141–147). v3's own abstract still
  carries it (lines 21–22).
- As written, the core contribution and the conclusion say the E-patch consequences were predicted without exception,
  and the introduction says a real swap's "consequences" were predicted. The coordinator's list names this qualifier
  ("conditional new frames") as one that must survive the cuts.

**Corrected wording.**
- Contribution 2: "the programs predicted the internal consequences of the E-patch stage by stage, for new cues from
  the tested lexical classes and, in new frames, for most stages, conditional on the frames' measured reference states;
  and the contrast change under real cue swaps within the margins and failures of §6."
- §10: "… the consequences of an encoding intervention stage by stage, for new cues from the tested lexical classes and,
  in new frames, for most stages, given those frames' measured reference states; and the contrast change …".
- Introduction: "The programs predicted the consequences of the **E-patch** …, stage by stage, and, under a **real cue
  swap** …, the contrast change end to end (§5, §6)."

### 2. SHOULD: the localization has lost its scope outside the abstract

**Where.**
- Lines 57–58: "we localized the behavior to three components".
- Line 200 (§4.1): "On exploratory and calibration data, the contrast is localized to three components".
- Line 581 (§10): "For two cue pairs, on exploratory and calibration data, the contrast is localized to three
  components".

**What the record says.** C002's supported conclusion holds "Within pinned Pythia-70M, the three screening templates,
single-token regular nouns, and the tested replacement distributions" (C002@eeff9ff:150). C002 does not support
generalization beyond them (C002@eeff9ff:155). v2 revision 1's §3 bullet "Its scope" and its §10 carried this scope;
v3 keeps it only in the abstract.

**Corrected wording.**
- Introduction: "we localized the singular-versus-plural contrast, for two cue pairs in three screening templates, to
  three components".
- §4.1: restore "It covers two cue pairs in its three screening templates and tested replacement distributions."
- §10: "For two cue pairs in three screening templates, …".

### 3. SHOULD: "a decoded mechanism" undoes the earlier fix to "the mechanism"

- **Where:** lines 49–50, "each tested stage of a decoded mechanism".
- **What the record says:** revision 1 changed "the mechanism" to "a candidate mechanism" because causal status is not
  established (C003@eeff9ff:5). *Decoded mechanism* asserts more than *candidate*.
- **Corrected wording:** "each tested stage of a candidate mechanism".

### 4. SHOULD: not every test used new frames

- **Where:** line 50, "tested once, on cues and frames it had never been run on".
- **What the record says:**
  - From 013 on, each Y1 is new cues in *exposed* frames (source map@b722e9c:52–57).
  - 021–023 also scored exposed frames.
  - "024 and 025 used exposed frames only" (C003@eeff9ff:88–89), as v3's own §2 says (lines 129–130).
- **Corrected wording:** "tested once, on new cues, in exposed frames and, from 009 to 023, in new frames".

### 5. SHOULD: statuses inside Box 1 are wrong

**Where.** Lines 278–281: "*Result* [prospective]. Over 24 new cues, Spearman 0.848 and MAE 0.077; the committed low
call for `an`, 0.18, came out lowest, at 0.30 … The read omits by construction the net contribution of layers 1–2,
which the residuals track."

**What the record says.**
- `an`'s call is a locked prediction (source map@b722e9c:147). v2 tagged it [locked prediction]. Inside "Result
  [prospective]" it reads as prospective, a lifted status.
- The residuals' structure is descriptive (source map@b722e9c:148–149), but the sentence is untagged.
- §4.3 states no default status.

**Corrected wording.** "Over 24 new cues in six new frames, Spearman 0.848 and MAE 0.077 [prospective]; the committed
low call for `an`, 0.18, came out lowest, at 0.30 [locked prediction] … which the residuals track [descriptive]."

Two minor points in the Box:
- "floors on Spearman (0.80) and MAE (0.248)" should read "Spearman at least 0.80 and MAE at most τ_g = 0.248"
  (docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md@d7c5e62:143–144).
- The six new frames should be named (011:16–22).

### 6. SHOULD: Table 2 leaves block 1's exact MLP out and lists a superseded reduction

**Where.** Line 304, the layers 1–2 row.
- The exact column names only block 2's MLP.
- The reduced column lists "012's token-local MLP model at template bases".

**What the record says.**
- From 016 on, block 1's MLP is evaluated "at the frame's operating point" (docs/superpowers/specs/
  2026-09-20-experiment-016-frame-conditioned-qk-design.md@d7c5e62:108–110).
- The current chain passes `ΔE` through "block 1's MLP at the frame's operating point, and block 2's MLP (channel D)"
  (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md@d7c5e62:80).
- Its reductions in layers 1–2 are the heads' rows at "template-base normalization, the frame's operands and scale"
  (same design:77–78) and, in coordinated frames, "the exact one-step propagation to `p_t`, which assumes `x1(p_t)` is
  unchanged" (same design:81).
- As drawn, the table understates the exact recomputation that §4.4's caveat is about.

**Corrected wording.**
- Exact column: add "block 1's MLP at the frame's operating point, from 016 on<!-- (016 design:108–110; 022
  design:80) -->".
- Reduced column: "the heads' rows at template-base normalization, with the frame's operands and scale, and their value
  paths (013, 016)<!-- (022 design:77–78) -->; in coordinated frames, one propagation step to `p_t`, assuming `x1(p_t)`
  unchanged<!-- (022 design:81) -->; in 012–015, the MLPs at template bases; in the E-patch tests of 018–019, channel
  D's locked subset and routing".

### 7. SHOULD: §6 uses the phrasing that 023's record says is not an outcome

- **Where:** lines 392–393, "three of 023's four conditions passed relative to the readout's own ceiling". The same
  paragraph then quotes “"Three of four conditions passed" is not a protocol outcome” (line 420).
- **What the record says:** 023:150–151.
- **Corrected wording:** "With block 0 completed, three conditions passed relative to the readout's own ceiling (new
  frames conditional), and the fourth, Y1/cue_final, was an `ENVELOPE_ONLY_FAILURE`; there is no aggregate label."

### 8. SHOULD: 021's passes lost their status tags

- **Where:** lines 428–430, "For new cues in exposed frames, with exposed nouns, the contrast was predicted, pooled over
  templates, by a small margin … New nouns, tested separately over the same pairs, passed." Both are untagged, and §6
  states no default.
- **What the record says:** 021's Y1 and Y3 are prospective (source map@b722e9c:215). v2 revision 1 tagged both.
- **Corrected wording:** add "[prospective]" after each. Alternatively, open §6 with a default, as §5 does.

### 9. SHOULD: "decompiled" is stronger than the record

- **Where:** line 423, "The readout was decompiled on exposed data (020) [exploratory]".
- **What the record says:**
  - 020's confirmation was never run (020:8–11).
  - 023 claims nothing "that the downstream readout is complete" (023:183–184).
  - 024 measured the readout's residual error (024:374).
- **Corrected wording:** "A readout program was written and explored on exposed data (020) [exploratory]".

### 10. SHOULD: Table 1 files 009 under discovery

- **Where:** line 190, "008–010 | where the number signal of determiner-like cues is carried (discovery)".
- **What the record says:** 009 was "prospective, one confirm" (Table A1, line 741; source map@b722e9c:78). The next
  row lists it again as a stage program.
- **Corrected wording:** "008, 010".

### 11. SHOULD: "found" was dropped from the superlatives

- **Where:**
  - line 93, "The closest precedents also condition on a measured state";
  - line 536, "The closest work on a cue and a noun's number finds";
  - lines 563–564, "like the closest precedents".
- **What the record says:** the literature review hedges each: B6 is "the closest analogue in the literature read"; A3
  is "the closest prior study found" (research/paper/literature-review/review-2026-09-28.md@70caf7d). v2 revision 1 had
  "found" in all three places (reviews.md, draft v1 note "superlatives hedged with *found*").
- **Corrected wording:** "The closest precedents found …", "The closest work found …", "like the closest precedents
  found".

### 12. NOTE: qualifiers and records lost in the cuts

- **Figure 5 (lines 412–415).** The image no longer says there is no aggregate label. Add to the caption: "There is no
  aggregate label, and Y1 and Y2 are never pooled."
- **Figure 6 (lines 456–457).** *rival* lost "whose new-frame share is conditional"
  (research/paper/figures/README.md@70caf7d:73–74).
- **Figure C1 (lines 822–827).** The image lost its note. The caption should add that the new-frame share of pooled
  statistics is conditional, that 019's rival was also scored on the new frames alone, and that Tier-A results are
  drawn only for 005 and 006 (figures/README.md@70caf7d:88–94).
- **Figure 6's introduction (lines 449 and 453).** "each result" should read "each result it draws", as revision 1
  fixed for the same figure.
- **The `either` sentence in §4.2 (lines 229–230)** lost its trace. Add
  `<!-- (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:45) -->`.
- **Recorded failures and caveats cut without a trace elsewhere in the body:**
  - 020 sat below six of its twelve frozen floor conditions. C003 lists this as contradicting evidence E017
    (C003@eeff9ff:181; 020:160, 177–179).
  - C002's missed P3 retention counts, and the lexicon's 0.699 against 0.70 (C002@eeff9ff:50, 126).
  - 022 has no aggregate label, and its C4 result is a positive contribution only (022:9–10, 248–250).
  - 021: "not a claim of good prediction everywhere" (021:171–172).
  - "Absence in a bounded search is not evidence of absence" (review:57–58).

  One clause each would restore them. The first matters most, because v3 says "every failure was kept".

### 13. NOTE: status defaults

- §5's default makes every untagged result in §5 correct. That covers Table 3, the ladder text and the "not tested"
  cells, which match synthesis@3d87d3c:272, 007:59–60 and 006:71.
- Two sections state no default but leave predictions untagged:
  - §4.2 (lines 246–258): 012's correction, the subset, the routing, 017's row change;
  - §6: finding 8.

  §3's rule (line 183) implies that such sections should tag their results. Add "Statuses as in Table 3" to §4.2.

### 14. NOTE: wording, one line each

- **Line 132.** "In 021–024, floors were calibrated on draws from exposed data (§3)". The pointer should be to Table
  A2, and in 024 the permutation null was binding (024:360–363).
- **Line 159 ("recorded as an erratum, not repaired") and lines 578–579 ("an erratum recorded rather than
  repaired").** 022 and 023 used the corrected wiring (022:35–39; C003@eeff9ff:189). Suggested: "recorded as an erratum,
  with the frozen results left unchanged".
- **Line 579.** "and block 0 completed, by further prospective tests" reads as an unqualified success. Suggested:
  "and block 0 added and tested prospectively".
- **Lines 101–103 and 586.**
  - "its routing is predicted": 019's routing is the frame-specific selection within the population, not the locked
    subset's (019:11–13). Suggested: "the frame-dependent routing of that population is predicted in aggregate".
  - "block 2's routed population" reads as a fact about the model. Suggested: "block 2's neuron population and its
    predicted routing".
- **Lines 68–69.** "`L03.H04` transports the change of the residual entering layer 3, `Δx3`, to `p_t`": the head reads
  a component of it (009:83–85), and in cue-final frames `p_t = p_c`. Suggested: "`L03.H04` reads `Δx3` at `p_c` and
  carries what it reads to `p_t`".
- **Lines 538–539 (Bayazit et al.).** The phrase lost "for verb agreement" (literature review A12).
- **Lines 53–54.** "Given a count cue … Pythia-70M-deduped prefers the singular or the plural form" is untraced and
  general. The reserve nouns failed the behavioral precondition (005:126–128). Suggested: "tends to prefer the form that
  agrees with the cue", with a trace.
- **Lines 13–15 ("we ran 21 experiments: stages … tested once each") and line 576 ("tested … prospectively … across 21
  experiments").** 006, 008, 010 and 020 were not prospective tests (Table A2). Suggested: "most of them single-run
  prospective tests".
- **Line 487.** "We leave it to future work." The trace supports only that the test was not run before this draft and
  is to be decided after it (paper-readiness@139755d:354–360).
- **Lines 351–352.** Table 3 puts 005's and 006's development and exposed results in the "new cues, exposed frames"
  column. The rival rows' pooled results sit there too. Suggested: a third column or a note.
- **Appendix D, lines 841–846.**
  - "Each experiment's designs, locks, confirmation sets …": 006, 008, 010 and 020 have no lock, and 008 and 010 no
    confirmation set.
  - "Every statement … keeps a trace": see the `either` sentence and line 219's untraced "registered at `LOCALIZED`".
- **README entry.** "It adds no claim and no number" is not quite true: v3 adds record numbers, such as the
  architecture sizes and Box 1's thresholds. The header's wording ("no number that the record does not state") is the
  accurate one.
- **AI statement, line 37.** "literature citations were checked against their original sources". The verification
  checked every link and bibliographic record but spot-checked only the key entries' paraphrases (review:1340–1343,
  1358–1361). The header already schedules an audit.
- **The abstract.** It no longer reports 021: the first wiring's small-margin pass and negative new-frame label. An
  earlier review asked for that (reviews.md, v0 re-review S1), and the flow review proposed cutting it. The abstract says
  nothing false; this is a decision for the reviewer.

## Checked and correct

- **New §2 model paragraph.**
  - 512 dimensions, eight heads and 2048 MLP neurons (016 design:96, 109).
  - Parallel residual and rotary-only positions (005 design:185–188).
  - E-patch, real-swap and 025 definitions.
- **§3.** The protocol, its exceptions (matching Table A2), the five statuses and the two qualifiers
  (source map@b722e9c:39–64).
- **Box 1's program, input, lock and target**, with their traces: 010:21–24, 44–45, 112–113; 011:36, 39–40, 75–79,
  84–86, 91–92, 103; 012:8–10.
- **Table 2's encoding, block-0, transport and readout rows**, and §4.4's text: C003@eeff9ff:331–336, 023:17–18,
  018:8–10, 018:16, 022 design:74, 020 design:106–115.
- **§4.5**, the completed program.
- **Table 3:** every number, status and "not tested" cell; the rival and compact-program rows; 015:109–111; 018:14–18.
- **Tables 4 and A1**, and Figure 5's values.
- **The §6 path paragraph's numbers and 021 scope** (exposed nouns; Y3 new nouns; the erratum quote).
- **§7:** the routes, 025's wording, C003's maturity and the boundary.
- **§8** and **§9**'s characterizations against the literature review (apart from finding 11 and the Bayazit note).
- **Traps that held:**
  - 021's scope;
  - no aggregate label for 023;
  - conditional new frames in the abstract, §3, §5 and §6;
  - given the measured input;
  - `either`;
  - `every` at +0.17 with `AXIS_ARTIFACT`;
  - 018 Y2 and 017's pattern term unestablished;
  - 025 never called "no causal effect";
  - the causal boundary in §1, §4.2, §7 and §10.

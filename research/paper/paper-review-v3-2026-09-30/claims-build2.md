# Claim-strength re-verification: draft v3, build 2

File: `research/paper/draft-v3-2026-09-30.md` (uncommitted; 913 lines; sha256 271308cf…, unchanged since 16:51).

**Verdict: PASS WITH NOTES.** There are no blockers. The blocker and findings 2–14 are resolved. Two SHOULD items
remain, and each is a one-phrase fix:
- **S1** is new framing in the Figure 5 draws paragraph (choice i).
- **S2** is the reviewer's own finding-4 wording, adopted as given, and itself inaccurate.

There are also four notes. The work was read-only: nothing in the repository was changed, and nothing under
`outputs/` was read. No report file was written this round; this file keeps the report as the reviewer returned it.
Line numbers are build 2's.

## Checks run

- **Checker:** `check_source_map.py`, re-run on the current file, exits 0. It reports 28 quotes, 235 citation groups,
  2 commit hashes and 43 labels, and prints PASS.
- **Build 1:** `draft-v3-build1.md` is the text the reviewer reviewed.
- **New or changed citation groups:** all 32 were dumped with their cited lines, and each one supports its statement.
  They are:
  - the header (review@57d6450:1341–1342, 1358–1360);
  - the behavior sentence (C002@eeff9ff:28; 005:126–128);
  - the Data paragraph (C002:29, 49–50);
  - the erratum (021:13–17; 022:35–37);
  - the ladder (METHODOLOGY:52…113);
  - §4.1 (C002:50, 120; C002:150; C002:5);
  - Figure 2 (010 exploration report:25–33, 45, 89);
  - the channels (016 design:94–110);
  - Box 1 (source map:142–143; 011:35–36; 011 design:143–144; 011:20–22);
  - the five groups in Table 2 (016 design:105–110; 018:16 with 022 design:74, 80; 022 design:77–78; 022 design:81;
    012:7–9 with 016 design:110);
  - Table 3's caption (019:11–13, 145–147);
  - Π (017:16–17);
  - the frozen pattern (017:10, 17–18, 126–129);
  - Figure 5 and the draws (final report:43; 13, 17–22; 19–20);
  - the §6 path (020:160, 177–179; source map:638; 020:110; 021:171–172; 022:28–31, 245–248; 022:9–10, 248–250);
  - §7 (paper-readiness@139755d:332–348).
- **019's A values:** re-checked against 019's README. Pooled 0.068 and new frames 0.080, both at or above the 0.05
  margin.
- **Rendering (CommonMark):** all 234 comments stay hidden. The only escaped marker is the intended code span in
  Appendix D. Box 1 renders as one blockquote paragraph with its 13 hidden traces.
- **References:** all 61 entries are cited, and every in-text citation has an entry. Table A1 is unchanged.
- **Tightening:** every deletion in the word diff was checked, and nothing with outcome force was lost.
  - Méloux et al. 2026 now sits in §8, "Validation by prediction".
  - The digest statement in §3 is now more precise.
  - 019's comparators are defined in Table 3's caption.
  - The glossary's `m_T`/`m_R` and E–N entries were unused.
  - Figure 6's caption lost its hatching legend, but the image has a legend for all five statuses.
- **Figures:** F6a (Figure 5) matches its caption. The label changes in figure_data.py and make_figures.py are
  cosmetic.

## Findings 1–14: resolution

1. **BLOCKER: RESOLVED.**
   - Contribution 2 (L98–102) and §10 (L635–639) now carry the full scope: new cues from the tested lexical classes,
     most stages in new frames, conditional on (or given) the frames' measured reference states, and the real-swap
     contrast "within the margins and failures".
   - The abstract keeps this scope (L24–29).
   - The introduction (L66–69) now only names the interventions.
2. **RESOLVED.** L63–65, L218, L239–241 (C002@eeff9ff:150) and L632–634.
3. **RESOLVED.** L55 now says "a candidate mechanism".
4. **APPLIED AS WORDED, BUT THE WORDING IS WRONG.** See S2.
5. **RESOLVED.**
   - The statuses are right (L310–314): [prospective], [locked prediction] and two [descriptive].
   - The floors are stated with the design trace (L305).
   - The six frames are named (L308–310).
6. **RESOLVED** (L338), with a residual gap in N2.
7. **RESOLVED** (L435–437).
8. **RESOLVED** by §6's default (L433–434). 021's Y1 and Y3 are prospective (source map@b722e9c:215).
9. **RESOLVED** (L472).
10. **RESOLVED.** L208 now reads "008, 010", and 009 is in L209.
11. **RESOLVED** (L102, L590, L615–616).
12. **RESOLVED.**
    - Figure 5's caption says there is no aggregate label (L459–460).
    - Figure 6: *rival* now says its new-frame share is conditional (L511–512), and "each result it draws" is restored
      (L503, L508).
    - Figure C1's caption is restored (L876–884).
    - The `either` statement is traced (L108–109; L260–266).
    - The cut records are restored: 020's six of twelve floor conditions (L474–477); C002's P3 counts and 0.699
      (L233–237); 022's C4 and its lack of an aggregate label (L489–491); 021's quote (L484); the bounded-search caveat
      (L91).
13. **RESOLVED** (L249–250 and L433–434), with the S1 exception.
14. **RESOLVED**, item by item:
    - L148 points to Table A2, and 024's permutation null appears in Table A2 and §8.
    - L175–177: the erratum, "with the frozen results left unchanged".
    - L630–631: "block 0's prospective test".
    - L112 and L280: the routing is "predicted in aggregate".
    - L640: "block 2's neuron population and its predicted routing".
    - L76–77: `L03.H04` "reads `Δx3` at `p_c` and carries what it reads to `p_t`".
    - L592–593: Bayazit, which matches the literature review's entry A12.
    - L58–60: "tends to prefer", now traced.
    - L18–19, L93 and L628–629: "most of them … single-run".
    - L542: "It was not run for this paper".
    - L369–370: Table 3's caption on the 005/006 rows and the rival rows.
    - L898–902: Appendix D.
    - The README now reads "no claim, and no number that the record does not state".

## Deliberate choices (a)–(j)

- **(a) Accepted.** The introduction makes no claim about results, and the scoped claim is in the abstract,
  contribution 2 and §10.
- **(b) Accepted.**
- **(c) Accepted.**
- **(d) Accepted,** with N2.
- **(e) Accepted,** except S1.
- **(f) Accepted,** with N1. The 019 A values are verified.
- **(g) Accepted.** ΔT and the no-harm guard first appear in Table 3 (L385–386) and are defined just after it (L403,
  L419). That is fine for a table.
- **(h) Accepted.** 005's reserve nouns scored 98/120 against a floor of 103 (005:126–128), so "tends to" is the right
  strength.
- **(i) Not accepted as written.** See S1.
- **(j) Accepted as deferrals.**
  - The abstract says nothing false about 021, and §6 reports 021 with its statuses.
  - The header schedules the AI-statement audit, with a correct trace.

## Remaining findings

### S1. SHOULD: the draws paragraph softens the envelope-only failure

**Where.** L462–465: "Y1/coordinated passed by +7.979e-5, small against its draws' spread, and Y1/cue_final fell below
its floor but within its draws' range<!-- (…final-report-2026-09-24.md:19–20) --> (Figure 5)."

**What the record says.**
- The sentence is literally true: g 0.995520 is above the lowest draw, 0.995470
  (experiments/023-block0-completion/evidence/final-report-2026-09-24.md@d7c5e62:19).
- But the same report puts g at CDF percentile 0.0001, labelled descriptive: only one of the 10,000 draws is at or below
  it (same file:30). Figure 5 shows the dot at the left end of its bar.
- "But within its draws' range" invites a reading as ordinary sampling variation, which the record does not give. The
  record's own words, which the paragraph already quotes, are "a small quantitative shift, not a refutation" and
  "neither simply a failure nor 'basically a pass'" (023:154–155).
- Both comparisons with the draws are also descriptive, but under §6's default (L433–434) they read as [prospective].

**Corrected wording.** "…: Y1/coordinated passed by +7.979e-5<!-- (023:156–157) -->, small against its draws' spread,
and Y1/cue_final fell below its floor, with only one of the 10,000 draws at or below it<!--
(experiments/023-block0-completion/evidence/final-report-2026-09-24.md:19–20, 30) --> [descriptive] (Figure 5)."

### S2. SHOULD: the introduction says every stage was tested in exposed frames

**Where.** L54–56: "each tested stage of a candidate mechanism was frozen as a program and tested once, on new cues, in
exposed frames and, from 009 to 023, in new frames, with the exceptions named in §3."

**What the record says.**
- 009, 011 and 012 tested new cues in new frames only: "009, 011 and 012 tested fresh cues in new frames only, so
  their exposed-frame cells are empty" (research/synthesis/synthesis-005-025.md@3d87d3c:272).
- 007 ran its Y families "on the six fresh frames" (007:59–60).
- The draft itself says so: Table 3 has "not tested" at L377–380 and L394–395, and Figure C1's caption says it at
  L878–879.
- §3's exceptions (L179–184) do not cover this.
- "From 009 to 023" also leaves out 007's test in new frames.

**Corrected wording.** "…tested once, on new cues, in exposed frames, new frames or both (Table 3), with the exceptions
named in §3."

A longer alternative: "…in new frames in 007–012, in exposed and new frames in 013–023, and in exposed frames in 024
and 025, …".

### N1. NOTE: Table 3's status cells that hold only a qualifier

**Where.**
- The caption (L368) says "an empty status cell means [prospective]".
- Cells hold only a qualifier at L377 and L379 ("[given the measured input]"), L380 ("[exposed base states]"), L387
  ("[aggregate, not per frame]") and L388–391 ("[pooled]").
- At L383 and L386, "[new frames unestablished]" leaves the status of the exposed-frame cell implicit.
- At L391, 019's new-frames-alone value (A 0.080) is not pooled. It is conditional: Table A1 says "the new-frame parts
  of Y3–Y5 conditional".

**Corrected wording.**
- Add to the caption: "a cell holding only a qualifier means [prospective] with that qualifier, and where only the
  new-frame status is given, the exposed-frame result is [prospective]".
- L391: "[pooled; new frames alone conditional]".

### N2. NOTE: Table 2 omits 016's block-2 reduction

**Where.** L338, the reduced column: "in 012–015, the MLPs at template bases".

**What the record says.** In 016, block 2 was still at the template base: "Experiment 016's decoded-`c_L` convention
(block 2 at the template base)" (018:141). The exact column already says "block 2's from 017 on".

**Corrected wording.** "in 012–015 both MLPs, and in 016 block 2's, at template bases<!-- (012:7–9;
docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md:110; 018:141) -->".

### N3. NOTE: the review records named by the header and README do not exist yet

**Where.**
- The draft header (L4–5) says it "applies the reviews of its first build ([reviews](reviews.md))". But reviews.md has
  no draft-v3 section yet; its last section covers draft v2.
- research/paper/README.md (L19–24) links to `paper-review-v3-2026-09-30/README.md`, which does not exist.

**What to do.** Add both before commit.

### N4. NOTE (optional): "each from three inputs"

**Where.** L65–67.

**What the record says.**
- C003's wording (C003@eeff9ff:27–28, 37–38) has no "each".
- The head read-outs also take the measured head input (C003@eeff9ff:138; §3 L172–173).
- 011's encoding read uses the weights and a frozen axis only (010:44–45).

**Corrected wording.** Drop "each", or add: "and, for the head read-outs, the head's measured input (§3)".

## Checked and correct

- **Scope and qualifiers:** the scope of the abstract, contribution 2 and §10; the "conditional" and "given the measured
  input" qualifiers; the §4.2 and §6 defaults.
- **Box 1:** its statuses, traces and floors.
- **Table 2:** the exact column (016 design:105–110; 018:8–10, 16; 022 design:74, 80).
- **Table 3:** every value, including 019's A values.
- **Tables 4 and A1**, and the Figure 5 and Figure 6 captions.
- **§6's path paragraph**, with the 020/021/022 records restored.
- **§7's boundary sentence**, and 025 is never called "no causal effect".
- **The known traps still hold:**
  - 021's scope;
  - no aggregate label for 023, and "Three of four conditions passed" appears only as a quote;
  - the opposing off-axis part is limited to `this`/`a`/`another`/`every`;
  - `every` at +0.17 with `AXIS_ARTIFACT`;
  - 022's C4 is descriptive;
  - 018's Y2 and 017's pattern term are unestablished.

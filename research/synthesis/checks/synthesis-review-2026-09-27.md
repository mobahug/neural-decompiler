# Independent reviews of the synthesis, 2026-09-27

Two separate agents reviewed the synthesis, read-only. Each loaded no model and ran no phase, and changed nothing:
- the first review read the draft;
- the second, a focused re-review, read the revision.

## The first review: PASS WITH NOTES, no blocker

Its findings are condensed below, each with what was done. Line numbers refer to the draft.

### Should-fix

1. **The preregistered routes to C002's Level 3 were incomplete.** The draft named only 005's confirm and 025.
   - 006's design planned its circuit families as the confirmatory Level-3 evidence, but they were never run.
   - 007 ran them. P1, P4, P5, P7, P8 and P9 passed; P3's correlation floor failed, so C002 was not reviewed.
   - The draft also omitted that the localization rests on nineteen held-out nouns as well as development data.

   *Done:* §2, §3.1, §4 and §5 list all four routes with their status, and §3.1 cites the held-out nouns.
2. **§5's boundary dropped each link's status and misplaced the end-to-end evidence.**
   - 021 Y2 is unestablished.
   - The readout's accuracy on new frames is descriptive.
   - 021 carries "not a claim of good prediction everywhere".
   - 023's `g` is the correctly wired end-to-end prospective test, relative to the ceiling.
   - §7 wrongly said the correct wiring was tested only exploratorily.

   *Done:* the boundary is tagged link by link. 023 is added to §3.4, §4 and §5. 021's caveat and its small margin are
   quoted, and §7 is restated.
3. **The measurement taxonomy was inaccurate.** 019 does use the E-patch (019 design:317, 341, 445). 006's and 007's
   Y2 are full substitutions. 025 is an embedding rotation. *Done:* §1 gives the three kinds, with 022's statement of
   what the E-patch omits, and §2 and §5 read "009 to 019".
4. **Some tags lifted sub-results.** 013's compensation cases are descriptive, and `an` is a locked prediction; the
   locked-prediction tag was unused. *Done:* both were re-tagged, and the tag is now used for 011, 012 and 014.
5. **"Low-rank descriptions failed" overgeneralized.** *Done:* now "The two tested low-rank descriptions failed", with
   007's narrowed reading quoted.
6. **The opposition was generalized to "the cue's encoding".** *Done:* it is restricted to `this`, `a`, `another` and
   `every`, is exploratory, and `these` and the numerals are shown as counter-cases.
7. **011's descriptive baseline was missing.** *Done:* "would also have cleared the Y1 floors on this set" is quoted,
   in §3.1 and §3.3.
8. **018 Y4 was missing.** *Done:* added to §3.3, §4 and §7.
9. **015 Y2 failed on two floors, not only "at layer 2".** *Done:* the token-mean R² 0.490 and the layer-2 entry R²
   0.423.
10. **The list of failures was incomplete.** *Done:* §4 lists every result not established.

### Notes, all addressed

11. **Compactness:** 006 added; 014's "A Compact Input-Side Predictor"; 018's "distributed-but-concentrated"; the 1024
    numbers identified as one channel.
12. **018:** 0.86 of the decoded read and 0.72 of the pattern term; "not fully invariant across new frames".
13. **019:** predictability from the full evaluation; G "carries most of that routing gain" and does not exhaust the
    headroom; aggregate carried into §2 and §4.
14. **023:** no "three of four passed" wording; P1's additions (023:17–20); three cue classes.
15. **017:** the pooled 0.958, and the coordinated 0.945 with gap 0.053.
16. **Wording:** no "prospectively" paired with a descriptive tag.
17. **024:** stated as a rank association, ρ 0.611 against 0.314; 23 of 40 cues extrapolated.
18. **010:** "24 of 24" applies to the off-axis part, and "the four" is named.
19. **Pooled statistics:** tagged as pooled, with the new-frame share conditional.
20. **C002's Level 3:** the three unchecked gates named; E003 glossed as 005's confirmation run.
21. **The §4 footnote:** names 009, 011 and 012.
22. **Safe-strength example and computation statements:** the example is now 013's original wording; the READMEs
    that state computations are named (009, 014, 016, 019).
23. **Program variants:** the preregistered alternatives are stated to be program-level variants, not model
    interventions.
24. **Wording:** "did not establish"; "its effect was not established"; `BEHAVIOR_NOT_REPLICATED` named in §1; the
    scale limits made exact.
25. **Source provenance:** every fact from outside the source map is now cited to its record.

The verified-correct items of the first review stand: 025's exact wording; C002 unchanged; the record caveats
respected; no criterion-level pass lifted; the qualifiers; and the claim that no preregistered test intervened on the
decoded variables themselves, which is true for model interventions.

## The re-review: PASS WITH NOTES, no blocker

It confirmed that all ten earlier should-fix items were resolved and the notes covered. Its own findings and what was
done:

1. **§7 misdescribed `replicate-021`.** It records numbers and no result, and 021's conditions were never re-scored.
   *Done:* restated as the re-review proposed (022:261–270).
2. **§7 called the new-frame frozen-pattern values "low" for 017 and 018 alike.** That holds for 017 only; 018's
   cue-final frames reached 0.771–0.951. *Done:* both are stated separately, as descriptive.
3. **§2 and §5 said every stage passed on new cues "in the exposed frames".** 009, 011 and 012 have no exposed-frame
   test. *Done:* "from 013 on in the exposed frames, and for 009, 011 and 012 in new frames".
4. **Routes wording.** *Done:* "four routes were designed; three ran; none established it", and "the interventions
   behind C002's Level-3 gates were preregistered in 005's lock".
5. **Status tags.** *Done:*
   - 019's headroom is tagged descriptive, and its gain clause conditional;
   - neuron 1987 is tagged "new frames conditional";
   - "[no criterion]" became [descriptive];
   - the 020-wiring link is tagged [prospective];
   - 013's remaining error is tagged descriptive;
   - 011 Y2 reads "given the measured input";
   - `thy`'s total is a locked prediction, and its split descriptive.
6. **E-patch and substitution wording.** *Done:* "part of what the E-patch leaves out"; "address them"; §5 names
   what each of 021–024 predicts, and that 020 was never confirmed.
7. **`every` is not a suppressed token** (008 classes it `AXIS_ARTIFACT`). *Done:* "suppressed" dropped.
8. **Compactness wording.** *Done:* the records' own terms are used — 009's "fixed-normalization linear read-out", the
   zero-parameter read, 014's title, 019's "Account 2, compact form" — and "distributed or high-dimensional", with
   016's "structured classes".
9. **023's completion carried no status.** *Done:* three PASS and one `ENVELOPE_ONLY_FAILURE`, Y2 conditional; whether
   a more compact rule suffices is not established.
10. **"Three to five lexical cue classes" missed 025.** *Done:* "two to five".
11. **Six citations passed only through the checker's one-line slack.** *Done:* all six ranges made exact.
12. **The fenced code block shifted the checker's backtick pairing,** so it checked 2 of 6 labels. *Done:* the
    checker now removes fenced blocks before its backtick checks, and checks all 6.
13. **Step 2's remit asks whether a genuine unresolved mechanistic gap remains.** *Done:* §5 now answers it
    explicitly — yes, at the causal level — without proposing an experiment.

After these changes the checker passes on the synthesis: 49 quotes, 100 citation groups, 2 commit hashes and 6
labels (`check_synthesis.out`). It still passes on the source map (`check_source_map.out`).

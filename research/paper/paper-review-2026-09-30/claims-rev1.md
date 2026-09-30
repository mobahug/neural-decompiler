# Verification of draft v2, revision 1, against the claim-strength review

This check is read-only. Nothing in the repository was modified, and nothing under `outputs/` was read. It compared
`research/paper/draft-v2-2026-09-30.md` (revision 1) against the pre-revision copy with `diff -u`. That copy is
identical to the text first reviewed. The diff has 13 hunks, and every one is an intended change. The check also read
the changes to `research/paper/reviews.md` and `research/paper/README.md`, and the new
`research/paper/paper-review-2026-09-30/`.

## Verdict: PASS WITH NOTES

All six SHOULD items and all fourteen notes are applied, and each new trace was checked against its cited lines. No
edit introduces an overclaim, a wrong status or a statement the record contradicts. Six minor notes remain, four of
them on text added by the revision.

## Checks

- **Checker:** `check_source_map.py` exits 0, with 32 quotes, 196 citation groups (183 before), 2 commit hashes and 43
  labels.
- **Rendering:** checked with CommonMark (`markdown_it`). There are 195 hidden comment blocks, up from 182. The tables
  (4) and lists (70) are unchanged. No comment marker leaks into the text; the only escaped one is the intended code
  span in *Conventions*.
- **The review record:** `paper-review-2026-09-30/claims.md` is byte-identical to the review report.

## The six SHOULD items: all applied correctly

1. **S1 (lines 534–536).** The frozen reading is now quoted from the preregistration, "a quantitative shift, never a
   refutation" (experiments/023-block0-completion/preregistration.md@d7c5e62:23), and the quote is verbatim. *small* is
   now attributed to 023's closure notes (023:154), which sit under 023's *Closure notes* heading (023:147).
   - The optional descriptive sentence on the draws was not added. That is acceptable, since Figure 6's panel A shows
     the draws' range.
2. **S2 (lines 762–774).** All four findings now carry statuses:
   - the read-out: [prospective, given the measured input];
   - the off-axis part: [exploratory];
   - the block-2 population: [prospective; new frames unestablished], with the routing [prospective; Y2 conditional;
     aggregate, not per frame], and *not a uniquely defined circuit* (018:102–117, 156–157; 019:164–165);
   - block 0: 022's localization [prospective; Y2 conditional], then 023's three passes, one `ENVELOPE_ONLY_FAILURE` and
     no aggregate label (022:243–245; 023:140–155);
   - nounness: [prospective; predictive, not causal] (024:373).

   *Carrying* and *completing* are gone.
3. **S3.** Four passages changed:
   - line 56: "Stages of a candidate mechanism";
   - line 91: "one mechanistic decomposition";
   - line 134: "one mechanistic decomposition";
   - §9 line 754: "one candidate mechanism", slightly narrower than the review's wording, which is fine.
4. **S4 (lines 61–65).** "For two cue pairs in three screening templates" matches C002's scope (C002@eeff9ff:150), and
   the bullet no longer repeats the cue pairs.
5. **S5 (lines 755–756).** "several 2026 studies of model internals combine several of them (Xun, 2026; Zhu, 2026;
   Dong, 2026; Somani, 2026; McCann, 2026; Buehler, 2026)" matches the review's D1–D3, C1, C13 and Buehler
   (literature review@70caf7d:23–25). All six are in the references.
6. **S6 (lines 418–419).** "not committed" is now traced to figure-data-audit-2026-09-30.md@e65c683:154–157, and the
   pair count keeps 009:62–64.

## The notes: all applied correctly

- **7. Channel D's subset (lines 353–355).** It is scoped to new cues in exposed frames, with the one new frame that
  failed the no-harm guard (018:102–117).
- **8. `every` (lines 336–337).** Its on-axis part is +0.17 (010:86), and 008 classed it `AXIS_ARTIFACT` (008:54–55).
  Both are exact.
- **9. The transport heading (line 263).** It carries [exploratory].
- **10. 007's replication (lines 268–271).** It carries [descriptive], and the fresh frames are said to lie beyond
  C002's declared frames (C002@eeff9ff:53). Both are exact.
- **11. κ (line 391).** The table now reads "κ 0.863 of channel D's contribution to the decoded read, 0.724 to the
  pattern term" (018:17, 102–104).
- **12. 025 (line 577).** It reads "rotation along a decoded direction … intervened on a decoded variable", as C003
  E024 has it.
- **13. The outcome-bearing tests (lines 707–708).** 024 and 025 are added, traced to 024:274–279 and 025:244.
- **14. 024's threshold (lines 173–174).** "a chance-level permutation null proved the binding threshold" (024:360–363).
  The null is a 100,000-permutation order statistic (024:76, 184–185).
- **15. The abstract (line 77).** "from the coordinated condition alone" (021:136).
- **16. Captions:**
  - F1 (lines 327–330) defines Y3, gives *rival*'s conditional new-frame share, and names the causal-status band;
  - F5 (lines 455–456) reads "`S_1` over both sets pooled and pair-weighted, rejected [prospective; pooled]" (018:118);
  - F7 (lines 600–601) covers the read-outs given the measured input and says the never-run chips are not labels.
- **17. Figure 1's introduction (line 319).** "each result it draws".
- **18. Figure 6's caption (lines 551–553).** It gives the three reasons from audit@e65c683:307–308. This is applied,
  despite the covering note's "not applied".
- **19. C003's maturity (lines 587–588).** "C003 is registered at maturity `LOCALIZED`" (C003@eeff9ff:5).
- **20. Minor fixes:**
  - line 434: "five tests of four accounts", which matches the five bullets;
  - line 72: "from 013 on, those results are conditional";
  - line 988: "with matplotlib 3.11.2" (figures/README.md@70caf7d:29).

## Remaining notes, all minor

1. **Figure 6's caption, lines 549–550 (added text).**
   - **Sentence:** "It is not the token-mean R² on which 021's Y1 was scored<!-- (021:116) -->."
   - **Problem:** it is accurate but partial. Y1 was scored on four statistics (021:116–119), and panel C's Y2 value is
     likewise none of Y2's four (021:120–123).
   - **Suggested:** "Neither value is a frozen condition of 021's Y1 or Y2<!-- (021:114–127) -->; Y1's token-mean R² is
     one of those conditions<!-- (021:116) -->." This is the audit's own wording.
2. **The third §9 finding, lines 771–773.**
   - **Problem:** "with block 0 added, three conditions passed" does not say what passed (`g`, relative to the
     ceiling). The 023 part also carries no tags.
   - **Suggested:** "… three conditions of `g`, relative to the ceiling, passed [prospective; Y2 conditional] and one
     was an `ENVELOPE_ONLY_FAILURE` [unestablished at its envelope], with no aggregate label".
3. **`every`, lines 336–337.**
   - **Problem:** read together, "carries an on-axis number signal" and "classed it an `AXIS_ARTIFACT`" can look
     contradictory. 008 glosses the class as "does **not** carry the mechanism-axis signal", but it measured a
     different quantity (008:54–55) from 010's read through the head's direction (010:86).
   - **Suggested:** "008's frozen rule, on a different axis measure, classed it an `AXIS_ARTIFACT`".
4. **The header, lines 14–15.**
   - **Problem:** "adds one number, the on-axis part of `every`'s encoding" is essentially true, but the revision also
     adds the count "five tests of four accounts" and the matplotlib version.
   - **Suggested:** "adds one value from the record".
5. **Cosmetic, line 764.** The line ends early at "its" after the reflow. It renders correctly.
6. **reviews.md, *Draft v2*, format part.**
   - **Sentence:** "venues with 8–9 pages need cuts of about 40 %".
   - **Problem:** the format review gives about 40 % for NeurIPS, ICML and ICLR, but about 45 % for ARR's 8 pages
     (paper-review-2026-09-30/format.md:24, 26).
   - **Suggested:** "about 40–45 %".

## Also checked and correct

- **reviews.md, *Draft v2*.** The claim-strength summary matches the review: six SHOULD items and fourteen notes, each
  listed correctly. The checker counts are right. The flow counts (eight MAJOR findings, about 6,150 words against
  about 7,500) match flow.md:18, 22–238 and 495.
- **The paper README.** Both new entries are accurate.
- **The review folder's README.** It is accurate, including that the reports' line numbers refer to the pre-revision
  draft.
- **Unchanged text.** No other paragraph changed. Every number, status tag and qualifier verified in the first review
  still stands.

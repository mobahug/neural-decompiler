# Independent review of the source map, 2026-09-27

A separate agent reviewed the draft source map read-only. It also reviewed commit `e34f16d` (the 025 post-closure
documentation correction) and the draft root-README entries for 024 and 025.
- **What it ran.** It reran both check scripts. It loaded no model and ran no phase.
- **What it changed.** Nothing: `git status` was unchanged afterwards.
- **Verdict:** PASS WITH NOTES, with no blocker.

The report is kept below in condensed form. Every finding and its substance are kept; some lists of line numbers are
shortened, and the line numbers refer to the draft it reviewed. The [disposition](#disposition) lists what was done
with each finding.

---

**Verdict: PASS WITH NOTES.** I found no blocker. Every frozen label is exact, and no failed or untested criterion is
promoted to a confirmation. Nine should-fix items remain. Most are limits or provenance facts the map leaves out; one
is the checker's coverage claim.

### SHOULD-FIX

1. **021: the design's reading was replaced after the confirm, and the map states the timing as fact.** At map:175–177,
   460 and 585, the "when" cell says "at the floor and lock reviews, before the confirm".
   - The design's frozen reading (021 design:335–337) says a negative label means a generalization gap beyond sampling
     variability.
   - The section that replaces it, "Interpretation (agreed at the floor and lock reviews…)" (021:151–161), first
     appears in the closure commit `86f7f8c`, after the confirm at `b7e8861`. The README at `ad22671` and `b7e8861`
     has no such text.
   - No 021 review record is committed; its evidence holds only the final report, the confirmation record and the
     erratum.

   Fix: state the timing as recorded, add a provenance row, and put the design's frozen reading beside 021's Y2
   status.
2. **The "conditional" qualifier is tied to label names, so it misses frame-conditional results without such a
   label.**
   - Both designs define Y2 as "(frame-conditional) … each frame at its own stage-1 reference state" (022 design:254,
     023 design:214). 022's four Y2 conditions and 023's two Y2 conditions carry no qualifier in the map.
   - Several rejection and routing results pool the new frames in: 013 Y3, 014 Y3, 015 Y3, 016 Y3, 018 Y3 (pooled and
     pair-weighted), and 019 Y3–Y5.

   Fix: define the qualifier by each design's frame-conditional scope, and tag those conditions and the new-frame share
   of the pooled statistics.
3. **024: the limits that narrow the label are dropped.**
   - The guard's formal scope is the two preregistered simple alternatives. All eight N cues are adjectives; the E cues
     are concrete, mostly animate nouns (024:295–297, 375–376).
   - Not claimed: "that nounness is a general linguistic representation" and "that E-vs-N rules out all lexical or
     semantic confounds" (024:305–306).
   - The test is "partly an extrapolation" (024:300–301, 379).

   These bear directly on reading 025's descriptive adjective/noun split.
4. **021: the erratum's scope is dropped.** 021:14–16 says the frozen results "remain a valid test of that program",
   meaning 020's `p_t`-wired one. Y1 includes the 36 coordinated exposed frames, and the wirings agree only in
   cue-final frames (020 erratum:48–49). Fix: quote the clause, and say all three 021 labels test 020's wiring.
5. **007 and 006: README-stated circuit results are left out.**
   - 007: P3's correlation floor failed on both fresh sets (0.696 and 0.840 against 0.90), while P1, P4, P5, P7, P8 and
     P9 passed (007:98–100).
   - 006: the second carried-forward result, 708/708 paired signs but correlation 0.756 against 0.90 (006:86–89).

   Both bear on C002. Fix: add them as reported results with no outcome force.
6. **011: a stated limit is missing.** 009's fitted rule "would also have cleared the Y1 floors on this set … the
   margin is modest" (011:91–94).
7. **A record inconsistency is missing: 024's design overstates 023.** 024 design:19–29 is headed "what Experiment 023
   settled (confirmed)" and says "Upstream is effectively reconstructed … `g ≥ 0.9955` in every condition". 023's own
   reading is one `ENVELOPE_ONLY_FAILURE`, "neither simply a failure nor 'basically a pass'" (023:152–155). Fix: add
   an interpretation row, and state in the synthesis rule that designs and the root README never raise strength above
   the experiment README.
8. **The checker covers less than the map claims** (map:20–21, 695–697). Every skipped item was checked by hand and is
   accurate.
   - It checks only quotes followed directly by a citation, and silently skipped 14 quotes and one straight-quoted
     text.
   - It silently skipped multi-file or prefixed citation groups.
   - It exempted 13–40-character hex strings from the commit check.
   - It checked only that labels with "_" exist somewhere, not which experiment they belong to.

   Fix: make it fail on unmatched quotes and on citation-like parentheses it cannot parse, and replace the length
   exemption with an allowlist — or narrow the claim.
9. **The categories are wrongly attributed to the methodology** (map:30–32). docs/RESEARCH_METHODOLOGY.md:165–171
   defines only exploratory and confirmatory results. The exploratory and descriptive definitions also overlap, so
   015's five-frame R² 0.82, which the README calls "Descriptive only, after the fact", was classed exploratory. Fix:
   correct the attribution and add a tie-break rule.

### NOTE

10. map:667 calls the 025 correction "in this step"; it is committed as `e34f16d`.
11. Some inconsistency classes do not match the map's definitions.
    - 614: "Revision 3 (commit `f64a609` reviewed)" means revision 3 answers the review of `f64a609`; it is not a
      misattributed hash.
    - 615: design Rule 4 defines the head-dropped decision; only the label string comes from the code.
    - 622 should be *outcome*, 628 *interpretation*, and 627 and 634 *cosmetic*.
12. Two hand-offs were added after closure, 007:112–113 in `5f2464a` and 012:125 in `1b1edff`. Row 582 paraphrases a
    commit message although the original wording is available at 007@6340a4c:105–107.
13. Two internal references do not resolve: "the cross-cutting section" and "the record notes".
14. 005 omissions:
    - protocol v2 "is not a preregistered amendment" (design:6–8);
    - the P3 retention counts, the X2 and X4 failures and the B2 band misses are not marked as failed;
    - row 610 should note that design revision 5 defines `CIRCUIT_NOT_GENERALIZED`.
15. The 009 quote folds the Tier A exposed cues and frames into "prospectively confirmed". The confirmatory part is 23
    cues × 6 frames, 138 pairs.
16. Missing qualifiers:
    - 018 Y3 is pooled and pair-weighted;
    - the `thy` example needs the caveat that its attribution was wrong (012:100–102);
    - 023 Y1/coordinated passed by +7.979e-5;
    - 025:250 is not a descriptive record;
    - 025's cues were not sampled at random.
17. Some thresholds come from designs but are cited to README lines: 5/6 (007 design:236–237), 0.80 (016
    design:238–239), 0.90 (017 design:330–331). The R²_C range is in 023:142–145, not 023:42.
18. Criterion-level passes are listed for 007 but not for 015, 018 and 021. 006's "Question" is actually the narrowed
    conclusion.
19. Screen:
    - `b561bc6`, a same-day qualification of the screening README, is not listed;
    - its held-out behavioral gates would meet the map's definition of prospective confirmation, so classing the
      screen as exploratory is weaker than needed (the safe direction).
20. Scripts:
    - `check_source_map.out` said "2 file(s)" where the documented command gives 1;
    - `digest_investigation.py` hard-coded the explore-log digests instead of reading the logs. The explanation holds
      either way.
21. `e34f16d`:
    - the erratum and the commit message say no evidence file changed, although the commit adds one and edits the
      evidence index (the erratum discloses both under "What changes");
    - "True for 024" holds only per experiment. At `eb5f6e8` the repository already tracked 022's and 023's `.f64`
      files, including 023's `exposed-cells.f64`, derived from 022's exposed measurements.
22. Root README:
    - the 024 entry omits the implementation review and the guard's formal scope;
    - the Synthesis link called the map "checked" before this review.

### Verified and found correct

- **Scripts.** The checker printed PASS. The digest investigation reproduced its recorded output byte for byte.
- **The digest explanations.** Each reconstruction undoes only what later phases write, and every recorded digest
  matches bit for bit.
- **Labels.** All 21 overview label sets match each README's status line exactly (scripted).
- **Results.** Results, numbers and statuses were spot-checked, at least two per experiment for 005–025. All were
  correct apart from the three design-sourced thresholds in note 17.
- **Record caveats.**
  - The inconsistency rows are factually right.
  - Exactly six lock commits carry the misleading subject, and each adds only `predictions.md` and
    `preregistration-lock.json`.
  - The only post-closure commits are `5f2464a`, `1b1edff`, `2320345`, `6bc2d6a` and `e34f16d`, and there are no
    other errata files.
  - The C002 and anomaly-registry statements are correct.
- **`e34f16d`.**
  - The six files' sizes, sha256 values and adding commits match, and there is no other binary file under 025.
  - The original clause stays struck through and is quoted verbatim.
  - Only the three 025 paths changed.
- **Root README.** Only line 21 grows and one bullet is added. The labels and numbers are exact, the 025 wording
  matches the fixed form, 024 stays predictive, not causal, and all link targets exist.

---

## Disposition

Every should-fix item and every note was addressed before commit, except the two parts of note 19 marked below.

| finding | what was done |
|---|---|
| 1 | The narrowed-conclusions row now gives the recorded timing: first committed at closure `86f7f8c`, after the confirm, with no committed review record of the earlier agreement. A provenance row was added. The design's frozen reading sits beside 021's Y2 in the record and in the evidence status. |
| 2 | The *conditional* qualifier is redefined by the designs' frame-conditional scope. 022's and 023's Y2 conditions and the new-frame share of the pooled 013–016 Y3, 018 Y3 and 019 Y3–Y5 statistics are tagged. A *given the measured input* qualifier was added for 009 Y1 and 011 Y2. |
| 3 | The 024 record now quotes the guard's formal scope, the lexical and semantic differences between E and N, the two further not-claimed items, and the extrapolation. The overview row names the guard's scope. |
| 4 | The 021 record quotes the erratum's "remain a valid test of that program" and says all three labels test 020's `p_t` wiring. The overview row says so too. |
| 5 | 007's circuit families (P3's correlation floor failed on both fresh sets) and 006's carried-forward 0.756 result were added, with no outcome force. |
| 6 | 011's "would also have cleared the Y1 floors on this set" and "the margin is modest" are quoted. |
| 7 | An interpretation row records 024's design overstating 023. The synthesis rule now says designs, plans and the root README never raise a finding above its README's strength. |
| 8 | The checker was rewritten. It now fails on a quote without a parseable citation, on straight-quoted prose, and on a citation-like group it cannot parse. It parses multi-file groups and `(message of <hash>)`. It replaces the length exemption with a one-entry allowlist (the Pythia model revision). It checks each overview row's labels against that experiment's own README. The 15 unmatched quotes were cited or rephrased, and the map's description of the checker states exactly what it checks. |
| 9 | The attribution now says the methodology defines only exploratory and confirmatory results. A tie-break rule by data provenance was added. 015's five-frame R² 0.82 is now descriptive / post hoc. |
| 10 | The 025 correction is cited as `e34f16d` and `8fe3ee9`. |
| 11 | The 007 `f64a609` item was removed. The 008 row was reworded and made cosmetic. 012's design-reading row is now *outcome*, 017's *interpretation*, and 015's ladder and 020's two values *cosmetic*. |
| 12 | Both hand-offs are marked as added after closure, and a provenance row records them. The 007 row quotes the original wording at 007@6340a4c:105–107. |
| 13 | Both references now link to [Registries](../source-map-005-025.md#registries) and [Record caveats](../source-map-005-025.md#record-caveats). |
| 14 | 005's record quotes "is not a preregistered amendment" and marks the missed floors: P3 retention, X2, X3, X4 and the B2 bands. The row about the third axis notes revision 5's `CIRCUIT_NOT_GENERALIZED` and is now cosmetic. |
| 15 | The 009 record says the confirmatory part is 23 fresh cues × 6 fresh frames (138 pairs), and that the exposed cues and frames are Tier A. |
| 16 | All five qualifiers were added. |
| 17 | The 5/6, 0.80 and 0.90 thresholds now cite their designs. The R²_C range cites 023:140–145. |
| 18 | Criterion-level passes were added for 015, 018 and 021. 006's record now gives the design question, and the narrowed statement is labelled the closure reading. |
| 19 | **Not changed.** The screen stays exploratory as a whole, the safe direction, and the evidence status notes that its behavioral gates were frozen before the run. `b561bc6` is left out because the map covers the screen as context only. |
| 20 | `check_source_map.out` was regenerated with the documented command. `digest_investigation.py` now reads the explore logs, and its output was regenerated. |
| 21 | The erratum's wording was made exact in `8fe3ee9`. Commit `e34f16d`'s message is left as it is, since history is not rewritten. The map records that 024's own sentence (024:356) holds only for its own directory. |
| 22 | The 024 entry names the implementation review and the guard's formal scope (`4f1f97d`). The Synthesis link no longer calls the map "checked". |

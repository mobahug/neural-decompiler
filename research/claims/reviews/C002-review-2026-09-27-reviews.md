# Independent reviews of the C002 claim review, 2026-09-27

Two separate agents reviewed [the C002 claim review](C002-review-2026-09-27.md), read-only. Each loaded no model, ran
no phase, and changed nothing:
- the first review read the first draft;
- the second, a re-review, read revision 2.

Revision 2 answered the first review, and revision 3 the re-review. What was done about each finding is listed below.

## The first review: FAIL, two blockers

Its findings are condensed below, each with what was done. Line numbers refer to the first draft.

### Blockers

1. **The readout point misdescribed 020's program.**
   - 020 freezes block-4 attention at the reference, and runs block 5's heads with their reference rows
     (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:42–43).
   - Its layer-5 head term is a second transport from the cue position, which bears on T and H2, not on R. C002's own
     E001 already measured that head, `L05.H07`.
   - Block-4/5 attention changes are only the small readout residual: 0.024 on exposed pairs, 0.032 and 0.035 on fresh
     ones.

   *Done:* E016 describes the program as 020 records it. The layer-5 value path moved to §4's paragraph on second
   routes, with E001's `L05.H07` figures. The readout residual is described as small, with its size predicted by
   nounness, and the readout is no longer called material. The re-review refined the attribution to `L05.H07` (its
   finding 1, below).
2. **The words incomplete and material went beyond C002's text, scope and quantity.**
   - C002 claims most of the shift, and its chain metrics were measured with layers 1–2 recomputing.
   - Within scope the named contributors are small: E alone recovers 0.885; the cue-position residual after block 0
     recovers 0.965; layers 1–2 add +0.081 and +0.187 for `two` and `several`; 007's E-patch reproduces the real cue
     swap closely on fresh cue words.
   - 023's R² values concern other cue classes at the pair level, and 022's coordinated share is descriptive.

   *Done:*
   - §1 lists what C002's own record bounds.
   - E013 and E014 are retyped as supporting (M, P) evidence that decomposes the link from E to T.
   - E017 is typed as contradicting only beyond C002's declared cue scope.
   - The verdict is restated within scope. Not `CONTESTED` now rests on the absence of a material conflict within
     scope.
   - The successor is argued on the difference in kind, and recommended beside C002 rather than superseding it.

### Should-fix

3. **007's replication was omitted and isolation overstated.** *Done:* E006 gives 007's values for P1, P3's F and signs,
   P4, P5 and P8, and lists P7 and P9. E007 says that only the pair-level correlation failed, and adds 006's 0.756.
4. **C002's third competing explanation was not addressed.** *Done:* a paragraph in §4, and an edit in §5 citing E005,
   E008, E009, E011 and E013.
5. **Unestablished results were missing.** *Done:* the status cells carry 007 Y1, 009 Y2 and Y3, 015 Y2, 017 Y3, 018 Y2
   and Y4, 021 Y2, 023 Y1/cue_final and 025, and §5 lists them.
6. **The erratum's effect was understated.** *Done:* E019 qualifies the coordinated part of the pooled +0.089
   comparator and all three 021 labels.
7. **Option B's interim state conflicted with the reasoning against `CONTESTED`.** *Done:* that reasoning now rests on
   the absence of a material conflict within scope. The status row uses the wording of rulings 2 and 4, and no longer
   announces `SUPERSEDED`.
8. **The entries lacked required fields and mixed types.** *Done:* a provenance table gives each entry's artifact paths
   and revisions. Each entry has one type. 017 has its own entry (E015), and E013's source reads 012, 013, 015, 016.
9. **§5 did not list every edit to C002.** *Done:* §5 item 4 lists every edit, and names the sections that do not
   change.

### Notes

10. **T's qualifier was dropped.** *Done:* §1 quotes T in full, with when the cue is not the final token.
11. **Given the measured head input was applied too widely.** *Done:* §4 tags 009 and 011's Y2 that way, 011's Y1 as
    from the weights alone, and 017 as from `ΔE` alone.
12. **Ruling 2 was restated too broadly.** *Done:* it now says that no preregistered, outcome-bearing test manipulated a
    decoded variable, and names 008's exploratory axis-only E-patch.
13. **025's kind and its place in the documented-beforehand row.** *Done:* 025 (now E018) has no kind, and is not in
    that row.
14. **E005 paraphrased 006's design.** *Done:* E005 quotes the `CIRCUIT_PASS` condition and the sentence on what
    `CIRCUIT_ONLY` supplies.
15. **023's Y2 values, the source of 024's mechanistic content, and channel D.** *Done:* E017 gives the Y1 and Y2
    values. E016 attributes the residual's location to the post-hoc analysis in 024's design. E014 quotes 018's
    definition of channel D.
16. **The ladder statement's assumption.** *Done:* §5 item 6 states it as an assumption for the successor's own review.
17. **The source map's mislabel of 006's 0.756** as an E-patch result. *Done:* both lines of the map now call it P3's
    isolation fidelity, with a note in the map's Checks section. The map's checker still passes.

The first review's verified-correct items stand: C002 unchanged; §1's account of C002; the methodology quotes; 005's
position map; 006's and 007's design quotes; the application of rulings 1, 2 and 4 in §3; 022's and 023's figures; and
the ladder claim.

## The re-review: PASS WITH NOTES, no blocker

It found both blockers resolved, and findings 3–17 of the first review resolved. Its own findings are condensed below,
each with what was done. Line numbers refer to revision 2.

### Should-fix

1. **020's layer-5 head term was attributed to `L05.H07`, and its whole pooled value counted as transport.**
   - The term covers every block-5 head, over the value changes at `p_c` and `p_t`
     (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:112–115).
   - No record after 005 names `L05.H07`.
   - The 0.089 is pooled over all templates, and in cue-final frames, where `p_t = p_c`, the term is position-local
     (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:76–77).
   - 022's frozen reading licenses a positive contribution for block 0's route, not a small one: 0.138 and 0.157 of
     R² (022:206, 208).

   *Done:*
   - E016 and §4 describe the term as covering every block-5 head, and name the second routes by layer.
   - They say the pooled 0.089 has no corrected template split, that the term is position-local in cue-final frames,
     and that no record attributes it to `L05.H07`. E001's `L05.H07` figures stay as C002's own.
   - Small is scoped to C002's pairs. 022's route is given as 0.138 and 0.157 of R², and called positive.
   - That R names most of the conversion now rests on 020's design spike, where `L04.MLP` and `L05.MLP` create about
     61 % of `Δc` [exploratory]
     (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:45–49), next to C002's R alone,
     0.606.
   - §5 notes that attaching C002's head names to the later program would repeat, on a small scale, what ruling 3
     advises against.
2. **The status-history row overgeneralized.** C002's own isolation retains 0.783, 0.927 / 0.760 / 0.664 by template
   (005:96–97), so about a fifth of the shift lies outside the three components
   (experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md:180).

   *Done:* the row now says that the contributors the later record names are small on C002's pairs (E001). §1 and the
   verdict add C002's own isolation residual.
3. **Scope was applied asymmetrically.** 007's 24 cue words are new words in new frames, none of them among C002's
   twelve (006:16–18), yet they sat under the within-scope heading, while E017 was set aside as beyond scope. 007's
   fresh frames also lie beyond C002's declared frames.

   *Done:*
   - E008 is typed as supporting beyond C002's declared cue scope, as E017 is typed as contradicting beyond it.
   - §4 splits the cue stage into what lies within scope and what lies beyond it.
     - Within scope: P4, and C002's twelve cue words, where the E-patch is within 0.35 nats of the real swap for
       every word, computed from 005's table
       (experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md:162–175).
     - Beyond scope: E008 at the token-mean level and E017 at the pair level, named as measured differently.
   - The Level-2 replication gate cites only E006's fresh-noun set.
4. **§5 item 5 lifted the status of the evidence.** 020 was never confirmed; 015, 018 and 021 Y2 are unestablished;
   new-frame results are conditional; and 020–024 concern the real cue swap, not the E-patch.

   *Done:* item 5 now says the program predicts the consequences of a cue change for new cues [prospective], and in
   new frames, given their reference states, for most stages [prospective, conditional; 015, 018 and 021 Y2
   unestablished; 020 never confirmed]. The E-patch is named only for 009–019, as in ruling 2 and the synthesis
   (research/synthesis/synthesis-005-025.md@3d87d3c:290–295). 021–024 are described by what they predict or localize.

### Notes

5. **E017's type.** *Done:* §2's preface names E008 and E017 as typed against a reading of C002 beyond its declared
   cue scope. E017 names its cue classes, which are not numerals or quantifiers. The *Not supported* line and the
   status row say at the pair level. §4 says the line makes explicit an exclusion C002 already states, and narrows
   nothing C002 claimed.
6. **Ruling 2 was stretched to E016.** *Done:* §3 and §5 say that E010–E015 predict the consequences of the E-patch
   (ruling 2), and that E016's results are predictive, not causal (024:373).
7. **§3 and §5 disagreed on the alternative-explanations gate.** *Done:* both say it stays checked, with a pointer
   appended.
8. **Status tags.** *Done:*
   - 020's ceiling is tagged exploratory, and 021's descriptive.
   - §4 tags 0.024 as exploratory, and 0.032 and 0.035 as descriptive.
   - E014 carries 019's qualifier: an aggregate result, not a rule for every frame (019:164–165).
   - E017 tags 023's passes as prospective, with Y2 conditional.
9. **Which part of C002's third competing explanation is supported.** *Done:* §4 says the higher-dimensional part is
   supported further (006, 007, 010, 012). The nonlinear part points downstream of `L00.MLP`:
   - the head reads its input linearly (009:84–85);
   - at the head's output, the axis and off-axis parts cancel mostly linearly, and the non-additivity grows downstream
     of the head (009:110–115) [exploratory];
   - layers 1–2's MLPs show a small interaction for numerals (012:103–104) [descriptive].
10. **Wording.** *Done:*
    - Ruling 3 says that narrowing or superseding is considered.
    - The successor's drafting needs its own authorization.
    - Isolation keeps its mean retention (F ≈ 0.78).
    - §6 names both computed comparisons and the figures derived as one minus a recorded value.
    - These dispositions are archived here.
    - P4's remainder includes block 0's attention at the target position.
    - Ruling 2 names the axis-only patches of 009's Tier A and its descriptive fresh-token patches (009:31, 80–81).

The re-review's verified-correct items stand:
- C002 byte-identical to HEAD and to `d7c5e62`;
- every artifact path and commit of the provenance table;
- more than 30 figures checked against their sources, including a recomputation of 007's table and of the P4 range;
- rulings 1, 2 and 4 applied, with no Level-3 gate checked or implied;
- the ladder statement.

## Checks

After revision 3, the checker (`../../synthesis/checks/check_source_map.py`) passes on the review: 34 quotes, 108
citation groups, 42 commit hashes and 9 labels. Every quote also matches its cited lines exactly, with no line of
slack. The checker passes on the corrected source map: 259 quotes, 565 citation groups, 123 commit hashes and 58 labels
(`../../synthesis/checks/check_source_map.out`).

# Independent reviews of the successor draft, 2026-09-27

Two separate agents reviewed [the successor draft](C002-successor-draft-2026-09-27.md), read-only. Each loaded no model,
ran no phase, and changed nothing:
- the first review read the first draft;
- the second, a focused re-review, read revision 2.

Revision 2 answered the first review, and revision 3 the re-review.

## The first review: PASS WITH NOTES, no blocker

It found the authorization constraints met. The draft is not registered, has no claim identifier, does not supersede
C002, proposes `LOCALIZED` with the required note, and invents no maturity level. Every figure it checked matched its
source. Its findings are condensed below, each with what was done in revision 2. Line numbers refer to the first
draft.

### Should-fix

1. **The stage-by-stage predictions were stated as predictions of a real cue swap.** The records made them for the
   E-patch (009–019). Under a real swap only `Δc` was scored prospectively, and the reconstruction of `Δx3` is
   descriptive (022:225–229; 023:130).

   *Done:*
   - The hypothesis now names the two interventions, the E-patch and the real cue swap, and says what was predicted
     for each.
   - A *Not supported* line says the intermediate states under a real swap were not predicted prospectively.
   - The conclusion's scope names the tested interventions, and the new-frame qualifier reads *for most stages*, with
     015 and 018 Y2 unestablished.
2. **The program was not defined, and the stages did not match the program tested end to end.** *Done:*
   - An operational definition names the current version, 023's `P1`, and the earlier versions tested stage by stage.
   - Stage 2 includes the prediction position in coordinated frames.
   - Stage 3 lists what 009, 011 and 017 found about `L03.H04`, and says that end to end the head is computed exactly
     inside block 3.
   - Stage 4 includes block 5's second-transport path.
3. **The conclusion dropped its sources' caveats.** *Done:* it now carries 021's small margin and its own statement, “not a claim
   of good prediction everywhere” (021:171–172), in E018, 012's net correction, 023's three cue classes and its lack of an
   aggregate label.
4. **The first Level-2 gate cited circular evidence and misplaced positions.** *Done:* it states what the programs
   narrow and cites their own evidence: E011, E021, E020, E013, E014 and E016, with E001 as the starting point. The
   positions follow 020's design.
5. **Checked gates lacked evidence identifiers, and two ground-truth examples did not fit.** *Done:*
   - Two methodological entries were added: E026, the measurement-validity records, and E027, the locks.
   - Every checked gate cites an identifier.
   - The ground-truth gate cites 019's exact-chain identities and 020's Level-1 identity with its float32 floor.
   - The identifier ranges were narrowed.
6. **020's floor shortfall was omitted.** *Done:* a new contradicting entry, E017, records that the end-to-end
   program sat below six of its twelve frozen floors on exposed data. It is listed under *Contradicting evidence*.
7. **The nounness wording overstated 024.** *Done:* the draft now calls it a rank association, predictive, not causal.
   E023 and *Limitations* carry the guard's scope, the 23 of 40 cues beyond the calibration range, and the frozen
   approximation. *Not supported* lists 024's items that are not claimed.

### Notes

8. **Typing rule and wording.** *Done:* the rule is stated under the table. E010, E019 and E022 use the sources'
   wording, and E022 records that the guard was met.
9. **Context of three entries.** *Done:* E003 gains the `MIXED` summaries, the relay bound and the `n_80` caveat. E001
   points to C002 for isolation. E002 gains 005's `NO_COMPACT_MECHANISM` and 007's criterion-level Y2 and Y3.
10. **Statistics and statuses.** *Done:* E007 gains 013's direct-head statistic, 0.838. The pooled statistics are
    marked conditional for their new-frame shares. E006's baseline is tagged descriptive. The R² definition is
    flattened over pairs and nouns.
11. **Level-3 wording.** *Done:* the first gate cites E027, which records the digested Y2 tables of 013–016. The
    necessity note separates program-internal tests from interventions in the model. The closing note names E024.
12. **Competing explanations.** *Done:* added the recomputation alternative and the exploratory interventions. Each
    alternative names its distinguishing test, and the pattern term is named as `L03.H04`'s.
13. **Prior work, novelty and predictions.** *Done:* the prior-work boundary and the nearest methods are cited from
    005's design and the finalist audit. The novelty is subject to systematic literature review. The predictions
    table names the families per lock.
14. **Minor.** *Done:* float32 for the model and float64 for the programs, the Origin field, and the wrap width.

## The re-review: PASS WITH NOTES, no blocker

It found findings 1, 2, 4, 6 and 7 of the first review resolved, and 3 and 5 partly resolved. It found no factual
error of substance in the figures. Its findings, with what was done in revision 3; line numbers refer to revision 2:

### Should-fix

1. **E017 misstated why 020's confirmation was not run.** It was not run because the floors had been calibrated on a
   scratch pass with a block-5 implementation error, so no label could be read cleanly (020:8–11, 175–181). The
   exposed values are a calibration record, never evidence for a label (020:110). *Done:* E017 says so, and Level 4
   no longer counts it as a failed prediction.
2. **The Predictions preface and E027 overstated what the locks contain.** Frame-conditional tables were computed at
   stage 1 of each run and held by digest, and the head read-outs apply a locked rule to the measured input. *Done:*
   both now say what each lock committed, with the stage-1 tables and the read-outs described separately.
3. **The conclusion omitted 021's coordinated result,** which had no positive skill in exposed frames either
   (021:132–133, 168–169). *Done:* the conclusion now states it, with the wiring defect as the plausible explanation.
4. **The new ground-truth example was also absent from the committed reports.** *Done:* E026 and the Level-0 gate cite
   022's confirmation gates, whose values are in its committed confirmation record (022:161–163, 171–174).

### Notes

5. **Three Level-3 boxes are checked.** *Done:* a line records that, as in C002, the two reporting gates are checked
   and no causal gate is.
6. **022's localization is primary, not complete.** *Done:* primarily is restored in E020, the Level-2 gate and the
   conclusion (022:243–245).
7. **The prediction timing clashed with the measured inputs.** *Done:* the draft now says the rules were frozen before
   the target prompts were executed.
8. **Origin said confirmed once.** *Done:* it says tested once, and that not every test passed.
9. **023's margins.** *Done:* E021 records Y1/coordinated's margin of +7.979e-5 (023:156–157), and the conclusion names
   `ENVELOPE_ONLY_FAILURE`, with the guard met.
10. **The routing rule holds in aggregate.** *Done:* said wherever compact rules are listed.
11. **Citation precision.** *Done:* the readout positions cite 020's design (106–115), the changed positions add
    113–115, E002's label cites 007:87–94, and E015 states how Y3–Y5 were scored (019:75–76).
12. **Gate evidence and ranges.** *Done:* the causal-interpretation gate cites the entries' statuses, and the
    labelling gate cites E001–E024.
13. **New frames.** *Done:* 6 to 18 new frames applies to 009–023; 024 and 025 used exposed frames only.
14. **The typing rule.** *Done:* it now covers the exploratory entries and E024.
15. **Minor.** *Done:* 009:80–81 is tagged descriptive; the partial evidence is said to favour the recomputation
    alternative at block 0; float64 is attributed to 020's program; and the earlier versions include one end-to-end
    test.

Its verified-correct items stand:
- every figure and citation in E001–E025 matches its cited lines;
- every artifact path, lock commit and confirm commit resolves and matches its README;
- the definitions of `P1`, `g` and the flattened R² are correct;
- the draft is not registered: the registry README and C002 are unchanged, and it has no claim identifier;
- the proposed maturity is `LOCALIZED`, with the note, and necessity, sufficiency and controls stay unchecked.

## Checks

After revision 3, the checker (`../../synthesis/checks/check_source_map.py`) passes on the draft: 10 quotes, 124
citation groups, 29 commit hashes and 38 labels. Every quote also matches its cited lines exactly.

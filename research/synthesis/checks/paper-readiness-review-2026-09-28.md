# Independent reviews of the paper-readiness assessment, 2026-09-28

Two separate agents reviewed [the paper-readiness assessment](../paper-readiness-2026-09-28.md), read-only. Each
loaded no model, ran no phase, and changed nothing:
- the first review read the first draft;
- the second, a focused re-review, read revision 2.

Revision 2 answered the first review, and revision 3 the re-review.

## The first review: FAIL, two blockers

It found every citation, quote and stated figure correct, and the checker passing. Its findings are condensed below,
each with what was done in revision 2.

### Blockers

1. **The headline claimed more than C003.**
   - The E-patch predictions stop at the head's output (009–019), and in new frames only most stages were predicted.
   - No decoded program predicted the E-patch's own contrast shift.
   - The real-swap `Δc` passed by a small margin on 020's wiring, without positive skill in coordinated frames. Once
     block 0 was completed, it held relative to the ceiling in three of four conditions, for three cue classes.

   *Done:* the headline and the brief answer carry C003's qualifiers.
2. **The protocol was misdescribed as locking every prediction.** The locks held criteria, rules and the predictions
   needing neither a new frame's reference state nor a measured input. Frame-conditional tables were computed at stage
   1 and held by digest, and the head read-outs apply a locked rule to the measured input. *Done:* §1 describes this,
   and §3 lists it as a gap to state.

### Should-fix

3. **The list of locked tests omitted 005 and 007.** *Done.*
4. **The must-travel list was incomplete.** *Done:* it now carries:
   - C002's basis and scope;
   - the erratum, the floors and the margins;
   - `g` as the scored statistic;
   - the stage-1 tables and the exploratory negatives;
   - the source-map caveats.
5. **The baseline gap misstated the record.** Descriptive baselines exist (007's Y4; 011 against 009's rule; 023's
   cheaper rules). *Done:* they are cited, and the gap is narrowed to what is missing.
6. **First full application.** *Done:* now first application, with the design chain's realized and unrealized links.
7. **The novelty section went beyond its sources.** *Done:*
   - the recorded boundary's head-level programs in GPT-2 and Llama-scale models are kept;
   - the unsourced sentence and the sentence about the leads' content are removed;
   - under the E-patch is added;
   - the audit's gap is scoped to the behavior.
8. **A conflated finding and missing statuses.** *Done:* the findings are split, each with its status.
9. **§4's reasoning.** *Done:*
   - the claim that either outcome strengthens the paper is replaced by the possible outcomes, with 025's own reading;
   - the four causal routes are listed;
   - 025's lack of a calibration phase is corrected;
   - the new-sequence criterion is cited;
   - the named test is made specific to coordinated frames and to block 5's path.

### Notes

The eight notes were answered in revision 2:
- aligning the step label;
- one primary test and one secondary candidate;
- the record as mapped and audited;
- 018's ladder as evidence on the recomputation alternative;
- dropping an unsourced constraint;
- corrections to F3, F5 and F6;
- a longer list of what must not be claimed;
- rewording the fourth novelty candidate and the project aim.

## The re-review: PASS WITH NOTES, no blocker

It found both blockers resolved, and the should-fix items resolved except the novelty section, which was partly
resolved. Its own findings, with what was done in revision 3; line numbers refer to revision 2:

### Should-fix

1. **The what-is-missing block omitted 025,** the one preregistered intervention on a decoded direction. *Done:* §4
   names it, and the exploratory manipulations are described as those apart from 025.
2. **018's ladder was tagged prospective.** Only the locked 256-neuron rung on exposed frames, 0.863 and 0.724, and
   the pooled single-neuron value, 0.219, are prospective. The other rungs are descriptive, and the new frames are
   unestablished. *Done:* gap 5 and F5 carry these statuses, and F5 cites 018:118.
3. **Completeness was said to be reached in parts, citing a section on compactness.** *Done:* compactness and
   human-readability were reached in parts. Completeness was quantified as residuals against measured ceilings, with
   no claim of a complete readout (023:183–184).
4. **The application candidate dropped most stages for new frames.** *Done:* it now names 015 Y2 and 018 Y2.
5. **The audit's multi-component gap was paired with tests that did not meet it.** *Done:* the novelty section says
   the outcome-bearing tests were of the E-patch and of `Δc`. It says the multi-component predictions did not reach
   outcome force, so the record does not yet fill that gap.

### Notes

1. **021's scope and its negative new-frame label.** *Done.*
2. **C002's failures.** *Done:*
   - at the pair level is restored;
   - C002's own failed tests, S1, P3's retention counts and the lexicon's 0.699, now travel with the headline.
3. **The list of what must not be claimed.** *Done:*
   - it gains the pattern term's necessity, the nounness direction and localization beyond C002;
   - 024's exact wording is restored.
4. **The design's phrase is the protected idea.** *Done.*
5. **The frame-conditional predictions were held by digest,** and the new-interventions link was only partly met.
   *Done.*
6. **Three status tags.** *Done:*
   - 019 is conditional in new frames;
   - 023 names the embedding change, three cue classes and Y2 conditional;
   - 025 is [unestablished].
7. **Reading content into a title.** *Done:* the lead is said to concern a related distinction.
8. **023's cheaper rules hold in cue-final frames.** *Done.*
9. **The other existing comparisons.** *Done:* the rival programs, 018's controls and 021's 020 comparators are listed.
10. **Wording in §4.** *Done:*
    - maturity is reviewed for a claim scoped to a stage;
    - not established is the outcome of every route that ran;
    - an equivalence bound bounds an effect.
11. **The named test's predictions and its question.** *Done:* the frame-conditional tables would be committed before
    any target prompt, or the test restricted to exposed frames. The test is stated as a question.
12. **A citation the checker skipped.** *Done:* now in parentheses.
13. **The first review was not archived.** *Done:* this file.

Its verified-correct items stand:
- every figure and citation it checked;
- the four routes, word for word as the synthesis;
- the named test's variable, which matches C003's P1 read;
- the realized and unrealized chain links, checked against 005's design;
- the committed data behind the figures.

## Checks

After revision 3, the checker (`check_source_map.py`) passes on the assessment: 7 quotes and 36 citation groups. Every
quote also matches its cited lines exactly, with no line of slack.

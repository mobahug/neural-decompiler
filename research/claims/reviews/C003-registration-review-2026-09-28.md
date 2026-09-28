# C003 registration review, 2026-09-28

**Status: registration.** This file records the decision to register the frozen successor draft as C003, and every
change made in doing so. An independent read-only check compares the registered claim with the frozen draft; its
result is archived at the end.

- The registered claim: [`../C003-frame-conditioned-decoded-program.md`](../C003-frame-conditioned-decoded-program.md).
- The source: the draft
  [`../drafts/C002-successor-draft-2026-09-27.md`](../drafts/C002-successor-draft-2026-09-27.md), frozen at `b1f6247`.
  Its three independent read-only reviews are archived in
  [`../drafts/C002-successor-draft-2026-09-27-reviews.md`](../drafts/C002-successor-draft-2026-09-27-reviews.md).

## The decision being recorded

The project's reviewer asked, on 2026-09-28, to register the frozen draft as C003, with these terms:
- confirm that the draft deserves a formal claim;
- assign `C003`, and keep the maturity at `LOCALIZED`;
- freeze the exact hypothesis and supported conclusion;
- move only the reviewed evidence entries into the registry;
- preserve the boundary, *prospectively predictive; causal gates unestablished*.

Registration must not make the claim stronger, broader or less caveated than the reviewed draft.

## Does the draft deserve a formal claim?

1. **A scoped, testable statement that no existing claim makes.**
   - C002 localizes. It names three components and measures what fraction of the shift they recover, retain and
     mediate, for two cue pairs.
   - The draft states something else. Frozen decoded programs predicted, by locked rules, the consequences of the
     E-patch stage by stage, and the contrast change under a real cue swap. Its failures and competing explanations
     are stated with the tests that would decide them.
   - The C002 review recommended a successor beside C002 on exactly this difference in kind (its §5, item 5).
2. **Evidence entries that meet the methodology's requirements.** Each of E001–E027 records a source, one type, an
   artifact, a result and a revision, as “Each evidence entry records:” (docs/RESEARCH_METHODOLOGY.md:153) requires.
   The draft's reviews checked these entries against their sources.
3. **Gates satisfied through the proposed maturity.**
   - The methodology requires that “All required gates through a maturity level must be explicitly satisfied”
     (docs/RESEARCH_METHODOLOGY.md:50).
   - Levels 0–2 are checked, each gate citing evidence identifiers.
   - At Level 3, the necessity, sufficiency and control gates are unchecked. `LOCALIZED` is therefore the highest
     level supported.
4. **A scoped conclusion with the nearby claims it does not support.** The draft has a supported conclusion, a *Not
   supported* section and limitations, in the form docs/RESEARCH_METHODOLOGY.md:308–314 describes.
5. **Failures and null results retained.** Methodology rule 1, “Add new evidence without deleting inconvenient prior
   evidence.” (docs/RESEARCH_METHODOLOGY.md:298), is met. The draft lists every failure and null result the three
   reviews checked for.
6. **Independent review.** Three independent read-only reviews, of the first draft and of revisions 2 and 3, returned
   PASS WITH NOTES with no blocker. Revision 4 answered every should-fix item of the third. No review read revision 4
   itself; the independent check of this registration confirmed that its answers are present and spot-checked their
   citations (see *The independent check*).

**Finding: the draft deserves registration as a `LOCALIZED` claim.** It records something the registry does not yet
hold. The project has decoded a program whose stages prospectively predicted the E-patch's consequences for new cues,
in new frames conditional on their measured reference states. Its completed version predicted the contrast change
under a real cue swap, with the failures C003 lists. It has not established that the decoded variables are the causal
mechanism.

## What registration changed

The registered file was produced from the draft at `b1f6247` by scripted edits. `diff` showed 43 changed lines at
`0810a75`, and shows 49 after the independent check's fixes. All of them are listed here:
- **Title.** *Draft successor to C002:* becomes *C003:*, with the first word capitalized; the rest is unchanged.
- **The draft's header block** is removed. It said that the draft is not in the registry, has no identifier, sits
  beside C002 and was authorized as a draft. That it sits beside C002 is restated in the metadata and the status
  history. The header's two links, to C002 and to the decision on the C002 review, are restored in the claim: at
  *Related claim* and in the Level-3 sufficiency gate.
- **Metadata.**
  - *Proposed maturity on adoption* becomes *Current supported maturity*. The value and the note are unchanged:
    `LOCALIZED`, *prospectively predictive; causal gates unestablished*.
  - *Proposed disposition on adoption* becomes *Disposition*, `ACTIVE`.
  - *Drafted: 2026-09-27* becomes *Created: 2026-09-28 (drafted 2026-09-27)*.
  - *Last reviewed* points to this registration review.
  - *Origin* opens with the template's term, *exploratory composite of confirmatory stage tests*. This adds a
    qualifier and removes none. It also names the draft as the place the composite hypothesis was assembled.
  - *Related claim* adds that C003 sits beside C002 and does not replace or supersede it.
  - A *Registered from* line names the draft, `b1f6247` and its revision number, and the review archive. It says which
    revisions the three reviews read, and that this registration's check confirmed revision 4's answers.
- **Self-references.** *This draft* becomes *this claim* in four places: three in *Relationship to C002*, and one in
  E003's reading of 010's relay bound.
- **Supported conclusion.** The heading drops *(proposed)*. The text is unchanged.
- **Limitations.** One relative link now resolves from the new location.
- **Status history.** The first row is added: `LOCALIZED`, `ACTIVE`, for the reason stated there, with E001–E027. It
  states the same review coverage as the *Registered from* line.

Nothing else changed. These sections are word for word as in the reviewed draft:
- the hypothesis, the scope, the operational definitions and the predictions;
- every evidence entry, E001–E027, apart from the one self-reference in E003, and the typing rule;
- every gate and its evidence, apart from the restored link in the Level-3 sufficiency gate;
- the contradicting evidence and failed tests, and the competing explanations;
- the prior work and the potential novelty;
- the text of the supported conclusion, *Not supported*, and the limitations apart from the one relative link.

## Evidence migration

All 27 entries move into the registry. Each was checked in the draft's reviews. Revision 4's changes to E001, E003 and
E026 were checked by this registration's independent check. None is added, removed or retyped. The E008/E017 decision about C002's own evidence does not apply here. C003's entries are typed against C003's
claim, and the reviews checked that typing.

## Registry edits

- `research/claims/README.md` gains one line, for C003.
- C002 is unchanged. It remains the active `LOCALIZED` claim.
- The C002 review anticipated this step: “If the successor is adopted, the row adds that it sits beside C002.”
  (research/claims/reviews/C002-review-2026-09-27.md@2318659:385) A cross-reference from C002 to C003 is therefore an
  open item. It needs its own decision, and it is not made here.
- The draft and its review archive are unchanged. They stay frozen at `b1f6247` as the reviewed source. The draft's
  header still describes its state at `b1f6247`: not in the registry, and no claim linking to it. The freeze keeps
  that header as it was.

## The independent check

An independent read-only check compared the registered file (`0810a75`) with the draft at `b1f6247`. It returned
**PASS WITH NOTES, with no blocker**. Its verdict: C003 is the draft with registration edits only; it is neither
stronger nor broader; and no scientific caveat was lost. Its findings, with what was done in the commit after
`0810a75`:

1. **Should-fix: the review coverage was stated more broadly than the record shows.** The three reviews read the first
   draft and revisions 2 and 3. Revision 4, whose answers changed about 57 lines, was read by none of them. *Done:* the
   *Registered from* line, the status-history row and this review now say which revisions the reviews read. They also
   say that this check confirmed revision 4's answers: all four final-review should-fix items are present, and the
   revision-4 citations it spot-checked match their sources.
2. **Should-fix: this review's finding overstated the record.** It said the stages predicted behavior on untouched
   data. The stages predicted the E-patch's consequences, the completed program predicted `Δc`, and new frames are
   conditional. *Done:* the finding now says so.
3. **Note: the word-for-word statements had unnamed exceptions.** *Done:* the one relative link and the restored link
   are named. `0810a75`'s commit message also says “word for word as reviewed” (message of `0810a75`) of all 27 entries. That is inexact:
   E003's self-reference changed, and revision 4's text was checked here, not in the three reviews. The commit is
   kept as checked, and the correction is recorded here.
4. **Note: two links were lost with the header.** *Done:* C002 is linked at *Related claim*, and the decision on the
   C002 review in the Level-3 sufficiency gate.
5. **Note: the C002 review's anticipated cross-reference was not cited.** *Done:* cited under *Registry edits*, as an
   open item.
6. **Note: Origin did not use the template's term.** *Done:* it opens with *exploratory composite of confirmatory stage
   tests*, which adds a qualifier.
7. **Note: the frozen draft's header is out of date.** *Done:* recorded under *Registry edits*. The draft is not edited.

Its verified-correct items stand:
- the diff of `0810a75` against the draft shows 43 changed lines, every one declared;
- no strengthening: the boundary appears in the metadata, the conclusion and the status history;
- the claim follows the template;
- the gates cite evidence, and `LOCALIZED` follows from the ladder;
- the registry line matches the title;
- C002, the draft and the other records are unchanged, and every link resolves;
- the checker passes.

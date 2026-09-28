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
6. **Independent review.** Three independent read-only reviews returned PASS WITH NOTES with no blocker, and every
   should-fix item is resolved in revision 4.

**Finding: the draft deserves registration as a `LOCALIZED` claim.** It records something the registry does not yet
hold. The project has decoded a substantial program whose stages prospectively predict behavior and the consequences
of an intervention on untouched data. It has not established that the decoded variables are the causal mechanism.

## What registration changed

The registered file was produced from the draft at `b1f6247` by scripted edits. `diff` shows 43 changed lines, all
listed here:
- **Title.** *Draft successor to C002:* becomes *C003:*, with the first word capitalized; the rest is unchanged.
- **The draft's header block** is removed. It said that the draft is not in the registry, has no identifier and was
  authorized as a draft.
- **Metadata.**
  - *Proposed maturity on adoption* becomes *Current supported maturity*. The value and the note are unchanged:
    `LOCALIZED`, *prospectively predictive; causal gates unestablished*.
  - *Proposed disposition on adoption* becomes *Disposition*, `ACTIVE`.
  - *Drafted: 2026-09-27* becomes *Created: 2026-09-28 (drafted 2026-09-27)*.
  - *Last reviewed* points to this registration review.
  - *Origin* names the draft as the place the composite hypothesis was assembled.
  - *Related claim* adds that C003 sits beside C002 and does not replace or supersede it.
  - A *Registered from* line names the draft, `b1f6247` and the review archive.
- **Self-references.** *This draft* becomes *this claim* in four places: three in *Relationship to C002*, and one in
  E003's reading of 010's relay bound.
- **Supported conclusion.** The heading drops *(proposed)*. The text is unchanged.
- **Limitations.** One relative link now resolves from the new location.
- **Status history.** The first row is added: `LOCALIZED`, `ACTIVE`, for the reason stated there, with E001–E027.

Nothing else changed. These sections are word for word as in the reviewed draft:
- the hypothesis, the scope, the operational definitions and the predictions;
- every evidence entry, E001–E027, apart from the one self-reference in E003, and the typing rule;
- every gate and its evidence;
- the contradicting evidence and failed tests, and the competing explanations;
- the prior work and the potential novelty;
- the text of the supported conclusion, *Not supported* and the limitations.

## Evidence migration

All 27 entries move into the registry. Each was reviewed in the draft's reviews, and none is added, removed or
retyped. The E008/E017 decision about C002's own evidence does not apply here. C003's entries are typed against C003's
claim, and the reviews checked that typing.

## Registry edits

- `research/claims/README.md` gains one line, for C003.
- C002 is unchanged. It remains the active `LOCALIZED` claim. A cross-reference from C002 to C003 would be a separate
  decision, and it is not made here.
- The draft and its review archive are unchanged. They stay frozen at `b1f6247` as the reviewed source.

## The independent check

(Recorded below once it has run.)

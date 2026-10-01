# Reviews of the submission-build cleanup, 2026-10-01

On 2026-10-01 the reviewer asked for a submission-build cleanup of the [TMLR build](../README.md): fix the two
overflowing table cells, convert the bibliography to BibTeX, make F1, F2 and F7 print-friendly, rebuild the anonymous
PDF, and run one final claim, citation and layout review. Independent read-only reviewers did the three reviews, then
re-checked the fixes. Each report is kept as returned, apart from the changes noted below.

| report | what it checked | verdict | what was done |
|---|---|---|---|
| [claims.md](claims.md) | the conversion and the new §2 sentence against the draft and the record | PASS WITH NOTES | all three SHOULDs and Notes 1, 2 and 4 applied; Note 3 is open |
| [citations.md](citations.md) | every bibliography entry against its source record | PASS WITH NOTES | S1–S6, N2 and N4 applied; the optional N3, N6, N7 and N9 not applied |
| [layout.md](layout.md) | the PDF's layout, the print variants and the build's claims | FAIL, three blockers | all three blockers and the notes applied |
| [citations-recheck.md](citations-recheck.md) | the applied bibliography findings | PASS | — |
| [claims-recheck.md](claims-recheck.md) | the applied claims findings | PASS WITH NOTES | the SHOULD and the NOTE applied |
| [layout-recheck.md](layout-recheck.md) | the applied layout findings | FAIL, two small items | both applied, and the F5 nit |
| [final-check.md](final-check.md) | the last fixes | PASS WITH NOTES | S1 and N1–N5 applied; N6 is the open item |

**The open item.** Claims Note 3, and the final check's N6, concern the statement on AI tools. It refers to the
author in the singular, which the anonymous build should not, and its sentence on citations claims more than the
literature review checked. Both are left to the reviewer, as the draft's header records.

**What the re-checks led to.**
- **Claims.** Appendix D's anonymous paragraph on the record's caveats no longer says that every caveat bearing on
  outcomes or provenance is in §3 and Table A2. It now names the kinds of caveat the record lists and says which are
  stated in the paper. The draft's reference-list lead-in and the build README now say that the venues, volumes and
  pages, and Grant et al.'s year, were updated from the publishers' records.
- **Layout.** Figure C1 is now set on the page of the Appendix C heading. The print variant of F2 is spaced more
  tightly so that it fits there at the full text width. In Figure 3, the key to the circled `an` points moved from
  inside panel A2 to beneath its note.
- **Final check.** The `float` package, which sets Figure C1 in place, had been loaded after `hyperref`. That
  duplicated every float's PDF anchor, added space between full-width figures and their captions, and pushed Figure
  C3's caption below the text block; `float` now loads first. Appendix D now says *the wiring erratum* and
  *inconsistencies in its records*. The figures README states the print sizes more exactly and points to these
  reviews. The reference list gives two missing page ranges, and both provenance sentences mention the changed
  links.

**Changes to the reports.**
- The first-person wording of [citations.md](citations.md), [claims-recheck.md](claims-recheck.md) and
  [layout-recheck.md](layout-recheck.md) was put in the third person, and the two re-checks were wrapped at 120
  columns.
- `<SCRATCH>` stands for the session's scratchpad. The renders, crops and scripts the reviewers left there are not
  kept.

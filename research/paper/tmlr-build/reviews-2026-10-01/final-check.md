# Final check of the last TMLR-build fixes (read-only)

The reviewer could not write this report to `<SCRATCH>/finalcheck/report.md`, because the harness blocks subagents from
writing report files; it is kept here as returned. `<SCRATCH>` is the session's scratchpad, which is not kept. Its
scratch builds, renders and scripts were in `<SCRATCH>/finalcheck/`. What was done about each finding is in the
[README](README.md).

**Verdict: PASS WITH NOTES.** All three fixes do what was intended:
- Figure C1 shares page 24 with the Appendix C heading.
- Figure 3's key sits beneath panel A2's note.
- Appendix D's caveats paragraph no longer overclaims.

The provenance statements, the READMEs, regeneration, anonymity and both checkers also hold.

One regression, S1, should be fixed before committing. Fix 1 loads `float` after `hyperref`, and that has three effects:
- every float's PDF anchor is duplicated;
- each full-width figure gains about 12 pt between image and caption;
- Figure C3 is 4.7 pt taller than the text block.

Moving one line fixes all three. A scratch build verified the fix.

The check was read-only. No repository file changed (git status was identical before and after), nothing under
`outputs/` was read, and every build, regeneration and tool ran in `<SCRATCH>/finalcheck/`. Two tracked files changed
during the check, research/paper/README.md and reviews.md (10:26). They were only skimmed. Their figures (27 pages,
about 12.2 pages, no overfull box) match the build and stay true after the S1 fix.

## SHOULD

**S1. `\usepackage{float}` comes after `\usepackage{hyperref}`** (md2tex.py:385 after :379; main.tex:10).
- **Cause.** float.sty's `\caption` replaces hyperref's. It steps the counter with hyperref's anchoring
  `\refstepcounter`, so every float gets a second anchor. After a full-width image, the end-of-line space and that
  anchor start a new line.
- **How it was isolated.** Scratch builds of main.tex compared three variants: as built, without `float`, and with
  `float` moved above `hyperref`. A rebuild of the previous round's main.tex served as a baseline.
- **Duplicate anchors.** build.log:8–21 lists 14 duplicate destinations ("Object @figure.1 already defined"), for 9
  figures and 5 tables. The previous round's build had none.
- **Extra space.** Every full-width captioned figure is 12 pt taller, and the extra space falls between image and
  caption. Float heights with and without `float`, in pt:

  | Figure 1 | Figure 3 | Figure 4 | A1 | C2 | C3 |
  |---|---|---|---|---|---|
  | 126.2 / 114.2 | 259.8 / 247.8 | 324.7 / 312.7 | 475.7 / 463.7 | 504.4 / 492.4 | 655.2 / 643.2 |

  Figures 2 and 5, set at 0.75 and 0.8 of the line, and all tables are unchanged. Pages 2–11 reflow, but the main text
  still ends at 219 pt on page 13.
- **Overflow.** Figure C3 now exceeds `\textheight` by 4.72 pt (main.log:502, "Float too large for page"). Its
  caption's last line, "chip.", ends at 734.5 pt on page 26, about 2 pt below the text block's 732.3 pt. No other page
  leaves the block.
- **Underfull vbox.** The underfull vbox on page 3 (badness 3281) also comes from this change, so it is not
  pre-existing. The previous round's build had one on page 8 instead (badness 4036).
- **Fix.** In md2tex.py's preamble, load `float` before `hyperref`. A scratch build with only that line moved gave:
  - no duplicate destinations and no "Float too large";
  - float heights as without `float`;
  - C1 still on the heading's page 24, its caption ending at 712.7 pt (17 pt more slack than now);
  - C2 on page 25, C3 on page 26 (ending at 726.3 pt) and D on page 27;
  - References on page 13 at 3.30 in, and 27 pages;
  - one log item: the earlier underfull vbox (4036) on page 8.

  No README needs to change.

## NOTES

**N1. Appendix D, "Record caveats": two wording points** (md2tex.py:197–203; main.tex:429; p. 27). Neither is an
overclaim.
- **"The erratum" follows "errata."** The record has two errata: 020/021's wiring (`2320345`) and 025's
  tracked-binary-files clause (`e34f16d`, a post-closure correction). §3 states only the first. Fix: "The wiring
  erratum".
- **"Inconsistencies between its records."** The source map heads the section "Inconsistencies in the records", and a
  few of its entries sit within one record: 018:139/150 and 020:15/119. Fix: "inconsistencies in its records".

**N2. Figures README: two loose ranges.**
- **README.md:35, "about 3–4.7 pt."** At md2tex's widths, the smallest text of the seven figures prints at 2.8–4.2 pt:
  F1 2.81, F7 2.84, F3 3.27, F2 3.41, F1a 3.49, F5 4.18 (at 0.85) and F8 4.24. The stated range contains these values,
  but no figure comes near 4.7. Fix: "about 3–4.2 pt".
- **README.md:229, "text of 6–8 pt."** The screen figures' text actually runs from 5.8 pt (F2's statuses) to 9.5 pt
  (the F3 and F6 titles). Fix: "6–9.5 pt".

**N3. The readable reference list is missing two page ranges the PDF prints** (draft-v3:673–674, 742–743).
- Geiger et al. 2024 and Syed et al. 2024 have no pages in the list, while references.bib gives 160–187 and 407–416.
- Neither provenance sentence mentions the 2026-10-01 link changes: Chan's Alignment Forum link, McDougall's ACL
  Anthology link, and five ACL `url` fields in the bib. These are draft-v3:625–627 and the tmlr-build README:52–54.
  Neither sentence claims to be complete.
- Optional fix: add the two page ranges, and "and some links" to both sentences.

**N4. The figures README's Reviews section stops at draft v3** (README.md:234–267). It does not point to the
2026-10-01 reviews of the print variants. Optional fix: add a bullet linking `../tmlr-build/reviews-2026-10-01/`.

**N5. reviews-2026-10-01/README.md links a file that does not exist yet.** The README was written at 10:25, during
this check, and links `final-check.md`. Fix: save this report there before committing.

**N6. The first-page footnote still names a single author** (p. 1; main.tex:21). It says "approved by the author" and
"the author takes full responsibility". This is the known open item.

## Checked and correct

- **Appendix C and D layout.** Evidence: main.aux, renders of all 27 pages at about 200 dpi, and pages 6 and 24 at about
  576 dpi.
  - The C heading, Figure C1 and its full caption share page 24, and the caption ends at 729.8 pt. C2 is on page 25,
    C3 on page 26 and D on page 27.
  - No page holds only a heading.
  - All ink stays within 72–540 pt horizontally. The one exception is link borders at 540.9 pt on page 17, which are
    non-printing and older than this round. Vertically, only page 26 leaves the block (S1).
- **Figure 3, page 6.** Redrawn in scratch, with ink measured at 600 dpi.
  - The key is a third line under A2's note: 0.7 pt below the note's ink, its left edge within 0.12 pt of the note's.
  - Its text sits level with the third lines of A1's and B's notes, and the three notes' tops are equal.
  - It overlaps no other ink and lies wholly below the axes. The four `an` points remain inside the limits.
  - The screen F3 is byte-identical to 3d47ee9. The print texts equal the screen texts, apart from panel B's earlier
    "; " join.
- **F2-print against F2.** All text artists were compared in order.
  - Column heads, the 19 row labels, 29 cell texts and 29 statuses are equal. The only difference is "under- predicted"
    → "under-predicted".
  - The 29 fills and hatches match in order, and so does the legend.
  - An ink check at 400 dpi found no two texts touching. The tightest gap between a cell's text and its status is 1.44
    pt (in the 011-replicated and 007 cells).
  - All text sits at least 1.04 pt inside its cell box, and none crosses a box or background edge. Row labels clear the
    first column by 7.9 pt, and the legend sits 8.5 pt above the column heads.
- **Log and length.**
  - main.log has 0 overfull boxes, no underfull hbox and one underfull vbox (3281, page 3; see S1). There are no
    undefined references or citations, and 81 `\citealp` calls.
  - findref puts the References on page 13 at 3.30 in, as in the previous build. The main text ends at 219.2 pt, 21 % of
    the 84–732 pt block, so it runs about 12.2 pages. The PDF has 27 pages.
  - Running md2tex.py on the live draft reproduces build/main.tex byte for byte, and build/figures equals
    research/paper/figures.
- **Regeneration.** make_figures.py ran twice in scratch, with figure_data.py's ROOT pointed at the repository for
  `git show`.
  - All 36 files are byte-identical between the two runs and equal to the repository's.
  - The 20 screen files equal their 3d47ee9 blobs.
  - figure_data.py, run read-only, exits 0: "every source checked: F1 chips 27, F2 cells 29, …".
- **Appendix D against the record and the draft.**
  - Each kind of caveat the paragraph names is a section of the source map at b722e9c:
    - Errata, 620–627;
    - Conclusions narrowed or restated, 629–640, dated "in the closure commit", "at closure" or "after closure";
    - Digest mismatches, 642–651: "Both are explained exactly";
    - Inconsistencies, 653–698;
    - Misleading commit subjects, 700–715.
  - Its sentence on frozen results renders line 617 faithfully.
  - The four items said to be in the paper are there. The erratum and the uncommitted frame-conditional tables are in
    §3 (draft:168–173). The review-record gaps and the restatements are in Table A2's "before 023" and "several" rows
    (824–825).
  - The paragraphs on the record, the audit trail and the figures match the Markdown, minus the link, hash and paths.
- **Provenance and READMEs.**
  - references.bib has 63 ASCII entries:
    - 13 from the ACL Anthology;
    - 9 ICML entries, each with its PMLR volume and pages;
    - Geiger 2025 as JMLR 26(83), 1–64;
    - Grant as ICLR 2026, per OpenReview (citations.md S1);
    - Chan from the AI Alignment Forum, with "Goldowsky-Dill" spelled correctly;
    - the two Transformer Circuits entries, and TransformerLens from GitHub.
  - citations.md checked all 63 author lists, titles and years against their sources, and citations-recheck.md
    confirmed the applied changes.
  - The draft's lead-in is internal only, because md2tex drops the Markdown list.
  - The figure mapping matches md2tex's `PRINT_VARIANTS`.
  - Print variants are saved at 6.25–6.55 in, with text at 4.9–7.2 pt; only the ∥ glyph is larger, at 8–9 pt. F4
    prints at 5.2–5.9 pt and F6a at 4.9–5.9 pt.
  - The float-placement note matches md2tex: `!tb` in the body, `tbp` in the appendices, a `\FloatBarrier` per
    appendix, and `\clearpage` plus `[H]`.
  - The results row and the Limits section match the build; "Hao & Linzen, 2023" appears on page 12.
- **Anonymity.**
  - The PDF metadata holds only CreationDate, Creator ("LaTeX with hyperref") and Producer ("xdvipdfmx (0.1)"), with no
    Author or Title.
  - The PDF text and main.tex contain no author or repository name, repository path, link or commit hash. The only hex
    string is the public model revision e93a9faa….
- **Checkers.**
  - check_source_map.py passes with exit 0 on all three files: the draft (27 quotes, 235 citation groups, 2 hashes, 43
    labels), the figures README (0, 10, 6 and 5) and the tmlr-build README (0, 0, 1 and 0).
  - check_bib.py reports "checked 63 references and 63 in-text forms: PASS" and exits 0.

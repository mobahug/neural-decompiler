# TMLR page-count build of draft v3

This build measures how many pages of main content, before the references, [draft v3](../draft-v3-2026-09-30.md) runs
to in the mandatory TMLR template. The reviewer asked for this on 2026-09-30: the page count, not the word count,
decides whether to cut. It is not the submission build.

## What it does

`build.sh` runs three steps.

1. **It fetches the TMLR style files.** These are `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` and `math_commands.tex`,
   from the repository that TMLR's author guidelines link, JmlrOrg/tmlr-style-file, at that repository's commit
   7bf90efe3a0d (Apache 2.0).
2. **It converts the draft with `md2tex.py`.** This produces the anonymous version:
   - the internal header and every hidden trace are dropped;
   - the statement on AI tools becomes an unmarked footnote on the first page;
   - the repository link is removed;
   - the figures are the committed PNGs at text width, except Figure 2 at 0.75, Figure 4 at 0.85 and Figure 5 at 0.8
     of it;
   - tables are set in `\footnotesize`;
   - every body figure and table is placed before the References heading, with `\FloatBarrier`;
   - the appendices start on a new page.
3. **It compiles with Tectonic** (`brew install tectonic`) and prints the page of each figure and table, where the
   references begin, and the last page. Tectonic is XeTeX-based, while the official template targets pdfLaTeX. Both
   use the template's Latin Modern fonts, so line breaks should match closely.

The output goes to `build/`, which git ignores.

## Results, 2026-09-30

| draft | main text ends | References begin | last page |
|---|---|---|---|
| v3 as committed at `298769e`, with Figure 6 in §7 | about a third of the way down page 14, so about 13.4 pages | page 14 | 23 |
| v3 with Figure 6 moved to Appendix C as Figure C3 | about a third of the way down page 13, so about 12.3 pages | page 13 | 23 |

In the first build, Figure 6 stood alone on a float page, page 11. The reviewer's rule of 2026-09-30:
- at 12 pages or fewer, keep the draft;
- slightly over 12, first move Figure 6 to Appendix C;
- still materially over 12, consider moving Figure 4 as well.

Figure 6 was therefore moved. At about 12.3 pages, Figure 4 stays.

## Limits

- **References** are a plain list rather than BibTeX with `tmlr.bst`, and the in-text citations are typed. This does
  not change the main-text count.
- **Two table cells overflow slightly:** Table 3's header and Table 4's `ENVELOPE_ONLY_FAILURE`.
- **The figures are drawn for the screen,** and their print profile is still pending.
- **The submission build** will need BibTeX citations, the print figures and a final layout pass. It should then be
  measured again.

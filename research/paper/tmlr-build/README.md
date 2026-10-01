# TMLR build of draft v3

This folder builds [draft v3](../draft-v3-2026-09-30.md) in the mandatory TMLR template, as an anonymous PDF. The
reviewer asked for the build on 2026-09-30, to let the page count rather than the word count decide the cuts. It was
then cleaned up toward a submission build on 2026-10-01. The Markdown draft stays the canonical text, and this folder
only converts it.

## What it does

`build.sh` runs four steps.

1. **It fetches the TMLR style files.** These are `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` and `math_commands.tex`,
   from the repository that TMLR's author guidelines link, JmlrOrg/tmlr-style-file, at that repository's commit
   7bf90efe3a0d (Apache 2.0).
2. **It checks the bibliography with `check_bib.py`.** Every entry of the draft's reference list must have one entry
   in `references.bib` with the same year, title and author surnames in order. Every in-text form in
   `citekeys.json` must name a key.
3. **It converts the draft with `md2tex.py`.** This produces the anonymous version:
   - the internal header and every hidden trace are dropped;
   - the statement on AI tools becomes an unmarked footnote on the first page;
   - Appendix D is written without the repository, its commits and its paths;
   - each in-text citation becomes a natbib `\citealp`, inside the draft's own parentheses;
   - the references come from `references.bib`, set by the template's `tmlr.bst`.

   Figures and tables are handled as follows:
   - Figures 1, 3 and 4 and Figures A1, C1, C2 and C3 use the print variants drawn at the text width (see the
     [figures' README](../figures/README.md)). Figure A1 comes in two parts.
   - Figures 2 and 5 use the committed PNGs.
   - Tables are set in `\footnotesize`. No column is narrower than its longest word, and long identifiers may break
     after their underscores.
   - Body floats sit at the top or bottom of a text page. Appendix floats stay in their own appendix. An appendix
     that opens with a figure starts a new page, and that figure is set on the heading's page.
4. **It compiles with Tectonic** (`brew install tectonic`), which runs BibTeX as needed. It then prints the page of
   each figure and table, where the references begin, and the last page. Tectonic is XeTeX-based, while the official
   template targets pdfLaTeX; both use the template's Latin Modern fonts.

The output goes to `build/`, which git ignores.

## The bibliography

`references.bib` holds the draft's 63 references: the 61 of the literature review, plus Pythia and TransformerLens,
which draft v3 cites in §2. Each entry was built from its publisher's record and checked against the draft's
verified list:
- the arXiv API for authors' full names;
- the ACL Anthology's BibTeX for its 13 papers, five of them linked through arXiv in the list as it then stood;
- PMLR for Bertinetto et al. (2021), Geiger et al. (2024), Pythia and every ICML entry's volume and pages;
- JMLR for Geiger et al. (2025), and OpenReview for Grant et al.'s ICLR 2026 paper;
- the citation blocks of the two Transformer Circuits articles;
- the AI Alignment Forum's record for causal scrubbing, whose own BibTeX misspells Goldowsky-Dill;
- TransformerLens's README for the library.

Titles follow the draft's list, where the literature review verified them. On 2026-10-01 the venues, volumes and pages,
Grant et al.'s year of publication and some links were updated from those records in the list, `references.bib` or
both, after an independent check of every entry against its source
([reviews-2026-10-01/](reviews-2026-10-01/README.md)). Non-ASCII names are written as LaTeX accents.

## Results

| build | main text ends | References begin | last page |
|---|---|---|---|
| 2026-09-30, v3 as committed at `298769e`, with Figure 6 in §7 | about a third of the way down page 14, so about 13.4 pages | page 14 | 23 |
| 2026-09-30, Figure 6 moved to Appendix C as Figure C3 | about a third of the way down page 13, so about 12.3 pages | page 13 | 23 |
| 2026-10-01, cleanup: BibTeX, the print variants, table widths, the anonymous Appendix D | about a fifth of the way down page 13, so about 12.2 pages | page 13 | 28 |
| 2026-10-01, after the final layout review: Figure C1 on the heading's page of Appendix C | about a fifth of the way down page 13, so about 12.2 pages | page 13 | 27 |

In the first build, Figure 6 stood alone on a float page, page 11. The reviewer's rule of 2026-09-30:
- at 12 pages or fewer, keep the draft;
- slightly over 12, first move Figure 6 to Appendix C;
- still materially over 12, consider moving Figure 4 as well.

Figure 6 was therefore moved, and Figure 4 stays. The cleanup changed the appendices' length, mostly because the
print variants and the full author lists are longer, but not the main text's.

## Limits

- **It is not yet the submission build.** The statement on AI tools still needs its audit against the project's
  history, and its wording in the anonymous build is open: it names the author in the singular.
- **Tectonic is not pdfLaTeX.** The final PDF should be checked once with pdfLaTeX, the engine the template targets.
- **Two-author citations** follow the template's style, with an ampersand (*Hao & Linzen, 2023*), where the draft
  writes *and*.

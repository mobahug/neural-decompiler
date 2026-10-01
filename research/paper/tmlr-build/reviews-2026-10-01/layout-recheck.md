# Confirmation check of the layout fixes (read-only)

The layout reviewer's verification of its first review ([layout.md](layout.md)), kept as returned, with its first-person
wording put in the third person. `<SCRATCH>` is the session's scratchpad, which is not kept.

**Verdict: FAIL, on two small items. Everything else is fixed.**
- One earlier blocker is not fixed in effect.
- One fix introduced a new overlap.

The supplied `main.pdf` and `main.tex` are byte-identical to the live build.

## Still to fix

1. **Blocker 3 is not fixed in effect: the Appendix C heading now stands alone on page 24.**
   - The `\clearpage` moved the heading to the top of page 24. Nothing else is on that page, which is 98 % blank.
   - Figures C1–C3 still go to float pages 25–27.
   - Cause: Figure C1, F2-print at `\linewidth`, is about 7.3 in tall, plus a caption of about 1.35 in. Heading, figure
     and caption do not fit in the 9 in text block, and as a `[tbp]` float it cannot sit below the heading.
   - **Fix, either of:**
     - Set C1 at about 0.95 `\linewidth` with `[!h]` or `[H]`, so that the heading and C1 share page 24.
     - Place the shorter Figure C2 under the heading, and renumber.
2. **New: in Figure 3, panel A2's legend now overlaps** (F3-print, page 6).
   - At "lower right" the one-line legend is wider than the narrow panel. It extends left past the y-axis.
   - Its marker covers the "0.0" tick label.
   - The identity line crosses "an,".
   - The lowest data point (≈0.23, −0.04) sits on "4 licensed", and two more points sit just above the text.
   - **Fix, either of:**
     - Use a two-line, right-aligned label in the empty lower-right area (x > 0.5, y < 0.5).
     - Move the legend into the note below the panel.

## Confirmed fixed

- **Blocker 1, the status list.** Page 4 now has bullets and the bracketed tags. Every page's ink stays
  within 72–540 pt.
- **Blocker 2, the nested quote.** Page 9 reads "‘basically a pass’”" in the right order. The opening "“‘Three" is
  correct but still has no thin space, which is acceptable.
- **Note 4, F8.** F8-print draws exactly what F8 draws:
  - all 93 texts and all 50 gate fills (face, hatch, line style, including the dashed "no such item") are identical;
  - rows are paired by the shared `f8_rows`;
  - text prints at 5.3–6.8 pt, and Figure C2 uses it.
- **Note 5, F1-print.** Both dashed frames now lie inside the canvas, at 0.015 in and 6.485 in. The link runs from the
  encoding head's cell down a gutter to the 011 chip.
- **Note 7, F2-print.** The cell now reads "under-predicted"; that is the only text difference.
- **Note 8, the bibliography.**
  - `{An}` and `{Makelov}` are braced, and the opening quote is now a backtick.
  - Bertinetto's pages are dropped.
  - TransformerLens now reads "GitHub repository" plus a URL.
  - JMLR 26(83):1–64 is in.
  - Grant et al. is 2026 consistently in the draft, the list, `citekeys.json` and the `.bib`.
  - `check_bib.py` prints PASS.
- **Note 9, Table A1.** The status column is 0.249 of the line, against 0.148 for "kind of test". The caption runs the
  full width, and the table ends on page 20.
- **Note 11.** The hash breaks every eight characters, and the page-3 underfull hbox is gone.
- **Note 13, the READMEs.**
  - "About a fifth … 12.2 pages" matches the reviewer's measurement: the text ends at 219 pt, 12.21 pages.
  - "F4 and F6a 4.9–5.9 pt" matches.
  - Nit: "smallest text about 3–4.7 pt" understates F5. Its screen version would print at 4.9 pt at full width (4.2 pt
    at the former 0.85 width).
- **Build, regeneration and anonymity.**
  - The log shows 0 overfull boxes, and only the harmless underfull vbox remains.
  - References begin on page 13, and the last page is 28.
  - All 36 figure files are byte-identical on a scratch rerun, and the screen files equal 3d47ee9.
  - `figure_data.py` exits 0, and the checker passes on the three READMEs.
  - No name, repository name, hash or path is left in `main.tex`. The page-1 footnote's "the author" remains the known
    pending item.

# Final layout review: print variants and the TMLR PDF (read-only)

**Verdict: FAIL.**
- **Not ready.** Three defects in the PDF must be fixed before submission. Each is a small fix in `md2tex.py`.
- **Print variants:** they draw exactly what their screen figures draw.
- **Reproducibility:** all 34 figure files reproduce byte for byte.
- **Build claims:** 0 overfull boxes, References on page 13, last page 28.

Renders and crops are in `<SCRATCH>/figreview/` (`pdfpages/`, `pdfsheets/`, `zoom/`, `printcrops/`). The signature scripts are in `print/`.

## Blockers

1. **The status list on page 4 sits in the margin and loses its brackets** (main.tex:68–72).
   - md2tex turns the draft's `- [prospective]: …` into `\item [prospective]: …`. LaTeX reads `[prospective]` as the item's optional label.
   - The five items therefore lose their bullets and the brackets of the paper's status tags.
   - The labels are set right-aligned into the left margin. "locked prediction" begins 28.9 pt from the page edge, 43 pt into the 72 pt margin.
   - Page 4 is the only page with ink outside 72–540 pt.
   - TeX reports no overfull box for an overhanging item label, so the log's "0 overfull" misses it.
   - **Fix:** protect a leading bracket, e.g. `\item{} [prospective]: …` or `\item {[}prospective{]}: …`.
2. **A nested quotation closes in the wrong order on page 9** (§6, main.tex:233).
   - The text reads "nor ‘basically a pass”’". The source `` `basically a pass''' `` becomes ” then ’ through TeX's ligatures.
   - The opening case two lines later, `` ```Three ``, renders in the right order “‘ but with no space between the marks.
   - **Fix:** when a single and a double quote meet, emit `'\,''` and ``` ``\,` ```.
3. **The heading of Appendix C is orphaned at the bottom of page 24.**
   - Below it, 552–732 pt of the page is blank. Its three figures sit on float pages 25, 26 and 27.
   - Cause: the `\FloatBarrier` before the section, together with `[tbp]` floats too tall for the rest of page 24.
   - **Fix:** `\clearpage` before Appendix C, which holds only figures.

## Notes

4. **Figure C2 (F8) prints below 4.5 pt, and no README covers it.**
   - It is the committed PNG, 9.5 in wide, set at `\linewidth` (scale 0.684).
   - Printed sizes: C002's differing wording, in grey, at 4.2 pt; gate cells 4.65 pt; row titles 4.9 pt.
   - No other figure goes below 4.9 pt.
   - The print-size statements in both READMEs omit F8. The draft's header says the build uses print variants for "the figures whose text would print too small".
   - **Fix:** add an F8 print variant at 6.5 in, or state it as a limit.
5. **Figure C3 (F1-print): both dashed frames are clipped at the canvas edges.**
   - The parallel box around the encoding and block 0 loses its left side: `box(ax, -0.04, …)` lies outside the 0–6.5 in canvas.
   - The real-swap frame around transport and the readout loses its right side; its edge sits at 6.535 in.
   - The encoding-read link line runs along the parallel box's right edge, the gutter at 0.99 in. It also starts at the E-patch lane's edge, beside the 010 chip, rather than at the encoding head.
   - **Fix:** keep both frames inside the canvas, and move the gutter off the box edge.
6. **Figure 3 (F3-print): the identity line runs through A2's legend.** The legend sits upper left; the line crosses "cue-final frames". Move it back to lower right, or wrap it.
7. **Figure C1 (F2-print): "under-" stands alone on a line in the 005 cell.**
   - The screen text's forced break, "under-\npredicted", survives the rewrap. The cell reads "quantifier template's mean / under- / predicted by 1.44 nats, / against 1.0".
   - **Fix:** join the hyphenated word before wrapping.
8. **Bibliography rendering** (`references.bib`, sentence-cased by tmlr.bst):
   - **Wu et al. 2024** renders as "A reply to makelov et al. (2023)'s ’interpretability illusion’ arguments". "Makelov" is lowercased, and the opening quote prints as a closing one. **Fix:** `{Makelov}`, and `` `Interpretability Illusion' `` with a backtick.
   - **Makelov et al. 2024** renders as "…looking for? an interpretability illusion…". **Fix:** `{An}`.
   - **Bertinetto et al. 2021** shows "pp. i–i", from `pages = {i--i}`.
   - **TransformerLens:** the entry's line is stretched (underfull hbox, main.bbl:438–441). Harmless.
   - McCann's "Pythia-70m" is braced on purpose and is correct.
9. **Table A1: the status column is starved** (pages 19–21).
   - The status column holds the most text (77 characters on average) but gets 0.143 of the line. "Kind of test" (39 on average) gets 0.227.
   - Cause: `min_fraction` treats hyphenated words such as "decomposition-completing”" as unbreakable, but TeX breaks after hyphens.
   - Result: rows up to nine lines tall, and a table over two pages.
   - **Fix:** split words at hyphens in `min_fraction`.
   - Separately, the longtable caption is set at the default 4 in `\LTcapwidth`, unlike the other captions.
10. **Appendix white space.**
    - Page 21 is 38 % blank between the text and the bottom float, Table A2, because of the barrier before Appendix B.
    - Pages 22 and 23 are float pages with Figure A1's two parts centred, 30–36 % blank.
    - This is acceptable. Optionally, put Table A2 before Figure A1 in the source.
    - Pages 18 and 28 are partial end pages, as expected.
11. **Page 3: the first line of "The model." is visibly stretched.** This is the underfull hbox at main.tex:56, caused by the monospace model name and the 40-character hash.
12. **The page-1 footnote renders correctly.**
    - It is unmarked, with the footnote rule. The hyperref "empty anchor" warning is harmless.
    - Its "the author takes full responsibility" still reveals a single author. This is the pending audit item the draft's header already lists.
13. **README accuracy.**
    - **tmlr-build README results row:** it says the main text ends about a quarter of the way down page 13, about 12.3 pages. Measured: the text block is 84–732 pt, and the main text ends at 219 pt on page 13. That is 21 %, so about 12.2 pages.
    - **Same README:** "not the main text's" also slightly understates a small change.
    - **figures README, "text would print at 3–4 pt":** true for F1, F1a, F3 and F7. F2 would print at 3.4–5.0 pt. F5 would print at 4.2–5.0 pt at its former 0.85 width, or 4.9–5.9 pt at full width.
    - **figures README, "F4 and F6a print at 4.6–6 pt":** measured 5.2–5.9 and 4.9–5.9 pt. F8 is omitted (see note 4).
    - Everything else checked matches the code and the PDF (see below).

## Checked and found correct

- **Print against screen.** The structural signatures cover texts with sizes, status fills, hatches and hatch colours, scatter data, lines, limits and legends.
  - F1a, F1, F2 and F5: texts, status patches, legends and data are identical.
  - F3: identical, except panel B's note joined with "; ", as announced, and fewer ticks.
  - F7: the two parts together equal F7 plus the repeated header and legend, split 005–015 and 016–025.
  - The patterns stay visible at print size.
- **Smallest printed text per figure** (only C2 is below 4.5 pt):

  | figure | variant | smallest text |
  |---|---|---|
  | Figure 1 | F1a-print | 5.4 pt |
  | Figure 2 | F4 PNG | 5.2 pt |
  | Figure 3 | F3-print | 5.2 pt |
  | Figure 4 | F5-print | 5.5 pt |
  | Figure 5 | F6a PNG | 4.9 pt |
  | Figure A1 | F7-print | 5.2 pt |
  | Figure C1 | F2-print | 4.9 pt |
  | Figure C2 | F8 PNG | 4.2 pt |
  | Figure C3 | F1-print | 5.3 pt |
- **Reproducibility.**
  - All 34 files, screen and print, regenerated in the scratchpad, are byte-identical.
  - The screen PNGs and SVGs equal 3d47ee9, and `figure_data.py` is unchanged.
  - `figure_data.py` exits 0.
  - `check_source_map.py` prints PASS, exit 0, on all three READMEs.
- **The PDF.**
  - The supplied `main.pdf` and `main.tex` are identical to the live build.
  - The log shows 0 overfull boxes and 3 underfull ones.
  - References begin on page 13; the last page is 28.
  - Every body float sits at the top of its page, with no body float pages.
  - Numbering: Figures 1–5, Tables 1–4, Tables A1–A2, Figure A1 (one caption plus a "continued" line on the next page) and Figures C1–C3. Every reference resolves, with no stale "Figure 6".
  - 81 `\citealp` commands cover all 63 keys, with no unconverted citation. The ampersand style is disclosed.
- **The tmlr-build README:**
  - the figure mapping;
  - 63 entries, 13 ACL papers with 5 linked through arXiv;
  - float placement, and the build steps.

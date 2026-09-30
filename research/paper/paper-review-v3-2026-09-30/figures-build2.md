# Check of the label changes and the draft-v3 captions (read-only)

**Verdict: PASS WITH NOTES.**
- Only the announced label strings changed.
- The new labels fit.
- Regeneration is byte-identical.
- Every audit caption-must item stands in the captions, or, where I allowed it before, next to the figure.
- Each caption matches its image.
- My three earlier fix-worthy notes are resolved: A defined, the "no titles or notes" wording, and "weight-only" in Figure 3's caption.
- There are two new minor notes, both about wording.

## Checks

1. **Nothing but labels changed: confirmed.**
   - **Method.** I drew the current `make_figures.py` and `figure_data.py` into the scratchpad, with `save` replaced. I compared their structural signatures against the ones I recorded for the version checked last round (`<SCRATCH>/figreview/v3b/sig_cur.json` against `v3/sig_new.json`). The signatures cover every patch with its colour, hatch and hatch colour, and every marker, line, text, axis, tick and legend.
   - **F2:** exactly six texts differ. The five row labels lose "Y1"/"Y2" (011 ×2, 012 ×2, 009's transport rule), and the 007 cell loses "Y1: ".
   - **F3:** exactly three texts differ: A1's inset, and A2's and B's panel titles.
   - **Everything else**, in all ten figures, is identical: geometry, values, colours, patterns and positions. The two `figure_data.py` hunks change only `Cell` label strings; the values, statuses, anchors and check words are unchanged.
2. **The new labels fit, with no clipping or overlap.**
   - F2's longest row label, "encoding read + correction vs P1 (012)", sits inside the left margin.
   - F3's titles end inside their panels. A1's three-line inset clears the points and the identity line.
3. **Regeneration: confirmed.**
   - My scratch run (`<SCRATCH>/figreview/regen5`) is byte-identical to the repository on all 20 files (sha256).
   - `figure_data.py` prints "every source checked" and exits 0.
   - `check_source_map.py` on both READMEs prints PASS, exit 0 (10 citation groups, 6 commit hashes, 5 labels).
4. **Audit caption-must items still stand.**
   - **F1a/F1** (Figures 1 and 6):
     - Figure 1 has the parallel encoding and block 0 (the dashed box and ∥), and L03.H04 inside block 3 as part of the readout.
     - Figure 6 has the dashed frame and the conditional Y2 and rivals.
     - The lane subtitle "only Δc was scored" and the pattern legend are drawn in the image.
   - **F4** (Figure 2): exploratory on spent data, `MIXED`, relay bound failed, the four tokens with 24 of 24, `either` and `these`.
   - **F3** (Figure 3):
     - `an` is a mean over its four licensed cue-final frames.
     - The read is weight-only apart from the frozen axis; this is now in the caption.
     - A1's pair-level result stays next to the figure, in the inset: 0.957 / 0.066 over 138 pairs, not shown.
   - **F5** (Figure 4): no curve; the pooled star, "not Y1's one-neuron rung"; the controls one by one; the series labels in the legend.
   - **F6a** (Figure 5): axis zoomed to 0.985–1.017, the 0.90 guard met, the statuses. "There is no aggregate label, and Y1 and Y2 are never pooled" is now in the caption.
   - **F7** (Figure A1): verbatim, ordinal axis. The screen and 023-d omission stays at Table A1's caption, as allowed before.
   - **F2** (Figure C1):
     - the empty-cell rule, criterion-level passes not shown, and ρ;
     - A = κ(E₆₄) − κ(T₆₄), now in the caption;
     - exploratory results drawn only for 005 and 006;
     - the conditional new-frame share;
     - the qualifiers inside the cells.
   - **F8** (Figure C2): the claims' own words, and 025 bearing on C003 only.
5. **Captions match their images, and the new citations hold.**
   - **Figure 2** defines informative frames as those in which the token is not the reference cue. 010's design lines 106–107 support this. It matches F4's axis label, "means over informative frames".
   - **Figure 3's** A2 and B wording matches the new panel titles.
   - **Figure 5's** "Y1 and Y2 as in Table 4" is supported: Table 4's caption defines them (lines 443–445).
   - **Figure 6's** "C1–C4 … (§6)" is supported: §6 names C1, C2, C3 and C4 (lines 488–490). "R²_C the ceiling's R²" matches the chip "021, 023: fresh ceilings and R²_C".
   - **Figure C1's** "whose new cues were tested on new frames only" is now exact for 007 (007:59–60). This resolves my earlier optional note.
   - **The README** now says "no figure titles or footnote notes", with panel titles and in-plot annotations kept. Its F3 entry matches the new titles. The departures bullet gives the reason for dropping Y1/Y2 from panel labels: for 009 and 011 they name families, while elsewhere they name populations.
   - **Remaining 009/011/012 "Y" usages** are:
     - the glossary line in draft v3 (848), which explains them;
     - two check anchors in `figure_data.py` that are never drawn.
6. **Hygiene.**
   - My runs wrote only to the scratchpad.
   - Two non-figure files changed during this check: `research/paper/paper-review-2026-09-30/README.md` and `research/paper/reviews.md`, both at 16:54:53. That is before my scratch run started (16:54:56), and none of my code touches them.

## Findings (both NOTE)

1. **"009's test" / "009's prospective test" is ambiguous.** It appears in F3 A1's inset, in Figure 3's caption (lines 319–320) and in the README's F3 entry.
   - 009 had two prospective tests. The one scored over 138 pairs is its P1 read-out test.
   - The transport rule was scored over 23 token means and failed. F2's row "rank-1 transport rule (009)" shows it.
   - **Fix:** "009's read-out test" would be exact without reviving the family name Y1.
2. **§6 calls C4 descriptive, while Figure 6 draws its C4 badges as prospective.**
   - Figure 6's caption sends the reader to §6 for C1–C4. §6 says "The share of block 0's contribution at the prediction position to the coordinated gap (C4) is descriptive" (lines 489–490).
   - The C4 badges are solid prospective (Y1) and conditional (Y2). That is correct: the condition C4, a positive contribution, passed prospectively (022:190, 194; source-map@b722e9c:226). Only the size of the share, 0.408 and 0.383, is descriptive (022:249).
   - A reader can take the parenthetical "(C4)" as the status of the condition.
   - **Fix:** in §6, say that C4, a positive contribution, passed, and that only its share is descriptive.

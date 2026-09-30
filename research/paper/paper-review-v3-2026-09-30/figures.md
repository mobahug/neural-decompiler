# Check of the draft-v3 figure changes (read-only)

**Verdict: PASS WITH NOTES.**
- No plotted value, status colour or pattern changed.
- F1a and F6a reproduce exactly what they are cut from.
- Nothing the audit required is lost.
- Regeneration is byte-identical, and the checker passes.
- Four notes follow, all minor. The only one I would fix before submission is the undefined "A" in Figure C1 (note 1).

## Checks

1. **No value, colour or pattern changed (F1–F8): confirmed.**
   - **Method.** I drew both the committed `make_figures.py` (70caf7d) and the working-tree version into the scratchpad, with their `save` replaced. From each figure object I extracted a structural signature: every patch (geometry, face and edge colour, hatch, hatch colour, line style), every scatter collection (offsets, colours, sizes), every line, every text (string, position, colour, size, backing box), axis limits and ticks, and the legends. Scripts and signatures are in `<SCRATCH>/figreview/v3/`.
   - **What differs.** Only the removed items differ:
     - F1: the title and the note;
     - F2, F4, F5 and F8: the title and the note;
     - F3: the suptitle;
     - F6: the suptitle and the three panel notes;
     - F7: the title and the footnote.
   - **F1 and F7** match element for element after a uniform upward shift of the canvas, 0.43 in for F1 and 0.35 in for F7. That shift is the removed title's space.
   - **Unchanged:** `figure_data.py` (no diff since 70caf7d), the colours, and the hatch, stripe and dot patterns.
   - **The code diff agrees.** `chain_header` and `g_panel` are verbatim extractions, and the other changes are the removals and the F6 and F7 legend/header offsets.
2. **F1a and F6a draw only what F1's stage row and F6's panel A drew: confirmed.**
   - F1a's 8 patches and 19 texts, arrows included, equal F1's stage row after an x shift. It has no lines or collections.
   - F6a's 4 collections, 12 lines and 4 texts equal F6's panel A.
   - F6a lacks two things F6 has, both covered:
     - panel A's title, whose content Figure 5's caption states (dot, black tick, grey bar, grey tick, zoomed axis);
     - the "descriptive" legend entry, which panel A does not use.
3. **Nothing the audit required is lost: confirmed.** Every must-say item from a removed note now stands in draft v3's caption, in the v3 text next to the figure, or in the figures README.
   - **Figure 1 = F1a.** Caption lines 66–73: the encoding runs in parallel with block 0 (∥), and end to end L03.H04 is computed inside block 3 as part of the readout. Both are audit "must show" items.
   - **Figure 6 = F1.** Caption lines 453–458: badges, Y1/Y2 (conditional), C1–C4, c-f/co, Y3, *rival* ("rejected or found insufficient"), the dashed frame, dashes and patterns. The italic note inside the real-swap lane is still drawn. The ∥ is explained in Figure 1's caption, which shows the same row.
   - **Figure 2 = F4.** Caption lines 236–241: [exploratory, on spent data], MIXED, relay bound failed, the four tokens with 24 of 24, `either` and `these`. The token-means limit is in the README.
   - **Figure 3 = F3.** The pair-level result and its status, and "weight-only apart from the frozen axis", stay in the panels' own text. The caption (285–291) states `an`'s four licensed frames and the identity lines. See note 3.
   - **Figure 4 = F5.** Caption lines 370–376 cover:
     - no curve, since no monotonicity was predeclared;
     - decision values and two decimals;
     - controls and the oracle;
     - the pooled star, "not Y1's one-neuron rung".

     The new-frame share of the star is conditional; §3's general qualifier (180–181) and the README cover it.
   - **Figure 5 = F6a.** Caption lines 412–415: the axis zoomed to 0.985–1.017, the 0.90 guard met, the statuses, and Y2 conditional. "No aggregate label" is at line 394, and "comparator, not a mathematical upper bound" at 396.
   - **F6 itself is not used in v3.** Its panel B and C notes (the ceiling as comparator; the erratum, not comparable) stay in the README's F6 caption.
   - **Figure A1 = F7.** Caption lines 768–772: verbatim from Table A1, per-population badges, the conditional pooled share, given the measured input, the 006/020 chips, ordinal axis, patterns. The omitted screen and 023-d rows are stated in Table A1's caption (731–733), which Figure A1 names.
   - **Figure C1 = F2.** Caption lines 822–827 cover:
     - ρ is Spearman;
     - the empty-cell note, verbatim;
     - criterion-level passes not shown as passes;
     - the patterns.

     The rivals' pooled scoring and the 019 new-frame value are in Table 3's note and §5 (331–332, 362–364), and the conditional qualifiers in §3. See note 1 for "A".
   - **Figure C2 = F8.** Caption lines 831–835: claims' own words, level states, C002's wording beneath, 025 for C003 only. The level-name titling is in the README.
4. **Layout: no clipping or overlap.**
   - I checked every PNG in full and the top/bottom edges of F3, F4, F5 and F8.
   - The F6 legend now sits above the panel titles, and the F7 legend above the column headers, without touching them.
   - F1a and F6a are tightly cropped with margins.
5. **Regeneration: confirmed.** All 20 files (10 PNG, 10 SVG), regenerated into `<SCRATCH>/figreview/regen4`, are byte-identical (sha256) to `research/paper/figures/`.
6. **Checker: passes.**
   - `check_source_map.py research/paper/figures/README.md` prints PASS, exit 0: "0 quotes, 10 citation groups, 6 commit hashes and 5 labels in 1 file(s)". With `research/paper/README.md` added, it also prints PASS, exit 0.
   - `figure_data.py` exits 0.
   - The repository is unchanged apart from the expected modified and untracked files.
   - The README's v3 numbering matches the draft's embeds (lines 64, 234, 283, 368, 410, 451, 766, 820, 829). The body figures are first cited in order 1–6.

## Findings (all NOTE)

1. **Figure C1 (F2) shows "A 0.068" but draft v3 never defines A.**
   - The definition, A = κ(E₆₄) − κ(T₆₄), sat in F2's removed note. It now lives only in the figures README (lines 89–91).
   - Table 3's note (362–364) describes the new-frame difference in words, without the symbol and without 0.068.
   - **Fix:** add the definition to Figure C1's caption, e.g. "A: the frame-specific selection's closure minus the template ranking's (019:145)".
2. **"No titles or notes" overstates slightly** (README "Titles and notes" bullet and departure bullet; the `make_figures.py` docstring).
   - Panel titles remain in F3 (A1, A2, B) and F6 (A, B, C). So do the in-plot annotations: F1's italic real-swap-lane note, F3's text boxes, F5's three annotations.
   - Their staying is right: they carry audit items such as F3's pair-level result and "weight-only apart from the frozen axis".
   - **Fix:** say "no figure titles or footnote notes".
3. **Figure 3's caption omits "weight-only apart from the frozen axis"**, which the audit asks the caption to say (audit F3 B).
   - It stands in panel B's own text and in §4.3 (line 274), so it is not lost.
   - **Fix:** add it to the caption for the audit's letter.
4. **Some removed-note items now live only in the README or the v3 body, not in the caption.** None is an audit "caption must":
   - Figure 6 does not repeat ∥ (Figure 1 explains it), nor the rivals' conditional new-frame share (§3 covers it).
   - Figure 4 does not give 0.24, nor the star's conditional new-frame share.
   - Figure 2 does not give the token-means limit.
   - Figure C2 does not give the level-name titling.

   Acceptable as they stand.

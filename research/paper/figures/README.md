# Figures

The paper's figures F1–F8, drawn only from what the [figure-data audit](figure-data-audit-2026-09-30.md) supports.
They are internal, like the drafts, and not for circulation.

## How they are made

- **The data.** [`figure_data.py`](figure_data.py) reads every plotted value with `git show <commit>:<path>`, from a
  pinned commit, never from a working copy and never from `outputs/`. It checks each value against an anchor in its
  cited lines: the header of the table it comes from, or a string that must stand there. It computes nothing.
- **The statuses.** Every colour is a status the record gives. A label's status is checked against its experiment's
  block in the source map's evidence-status section, which lists each finding under its category
  (research/synthesis/source-map-005-025.md@b722e9c:97–248); a status the record gives only in prose, such as a
  descriptive reconstruction, is checked against cited lines that state it.
- **The drawing.** [`make_figures.py`](make_figures.py) draws the figures from those values. Its transformations are
  presentational only, as the audit allows: selecting, ordering, labelling, colouring by a recorded status, choosing
  axes, drawing identity lines and showing a recorded number at lower precision. No curve is fitted and no statistic
  is recomputed.
- **The pins.**
  - `d7c5e62`: the experiment READMEs, final reports, committed extracts and designs;
  - `b722e9c`: the source map, for the overview, the evidence-status section and the status categories;
  - `454327d`: C002's evidence gates;
  - `eeff9ff`: C003's evidence gates, stage summary and *Not supported* section;
  - `3d87d3c`: the synthesis's stage table.

  The claim and synthesis pins predate the documentation-only correction `57d6450`. It changed no line the figures
  cite; C003's lines 382–424 now sit one line lower, so its cited lines 396–397 are 397–398 at `57d6450`.
- **Titles and notes.** Since draft v3 the images carry no figure titles or footnote notes, so that a paper can number
  its figures in its own order; the captions below, and the captions in the drafts, carry them. Panel titles and
  in-plot annotations stay, since some carry what the audit requires.
- **Regenerating.** The first command checks every source and prints what it read; the second writes each figure as
  SVG and PNG. With matplotlib 3.11.2 the outputs are byte-for-byte identical from run to run.

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/paper/figures/figure_data.py
```

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/paper/figures/make_figures.py
```

## Status colours

One colour per recorded status, in every figure. The categories are the source map's
(research/synthesis/source-map-005-025.md@b722e9c:39–45), and the conditional qualifier is its first
(research/synthesis/source-map-005-025.md@b722e9c:52–61).

| status | colour | where colour alone carries the status |
|---|---|---|
| prospective | green `#009E73` | solid |
| prospective, conditional | light green `#7FCDB5` | solid |
| locked prediction | blue `#0072B2` | solid |
| exploratory | orange `#E69F00` | striped |
| descriptive | grey `#A6A6A6` | dotted |
| unestablished | vermilion `#D55E00` | hatched |

The hues are the Okabe–Ito palette's, except the conditional tint. In greyscale, prospective and unestablished have
nearly the same lightness, as do exploratory and descriptive, and exploratory lies close to the conditional tint;
under deuteranopia the conditional tint lies close to the grey. So wherever colour alone carries a status (the badges
of F1, the cells of F2 and the chips of F7), the unestablished fill is hatched, the exploratory fill striped and the
descriptive fill dotted, and F2's cells also name their status. The other figures name each status in text. Text on a
fill is dark or white, whichever contrasts more.

F8 colours the claims' gate states instead, and each cell also names its state: checked in green, not established in
vermilion, not reached in light grey, not claimed in a lighter grey, and a missing item as a white box with a dashed
border.

## The figures, with draft captions

**F1, the chain** ([PNG](F1-chain.png), [SVG](F1-chain.svg)). The decoded chain in the model's order, in two lanes.
The E-patch lane tests the programs stage by stage; the real-swap lane tests them end to end, where only the contrast
change `Δc` was scored. Each chip is one result at one stage, and each of its badges carries one recorded status.
- Y1 is new cues in exposed frames; Y2 is new cues in new frames, conditional on those frames' measured reference
  states; 021's Y3 is new nouns over the Y1 pairs.
- C1–C4 are 022's conditions, and c-f and co are 023's cue-final and coordinated conditions, one badge each.
- *rival* marks a preregistered alternative account, rejected or found insufficient on both sets pooled; its
  new-frame share is conditional.
- The encoding, the layer-0 MLP, runs in parallel with block-0 attention. End to end, `L03.H04` is computed inside
  block 3, as part of the readout, which the dashed frame in the real-swap lane marks; the transport results come
  from the E-patch lane.
- A dash marks a cell where no chip is drawn.
- The bottom band gives the causal status: C003's necessity, sufficiency and control gates are not established, and
  025's outcome is `CAUSAL_EFFECT_NOT_ESTABLISHED`.

**F2, the prediction map** ([PNG](F2-prediction-map.png), [SVG](F2-prediction-map.svg)). What the frozen programs
predicted, stage by stage and population by population, with the failures.
- Each cell names its metric and its label-level status; ρ is Spearman.
- An empty cell has no result drawn. It means not tested for the exposed-frame and pooled cells of 007, 009, 011
  and 012, whose new cues were tested on new frames only, and for every fresh-data cell of 006, whose confirmatory
  test never ran.
- The alternatives are drawn at the pooled level on which their labels are defined. 019's was also scored on the new
  frames alone: A = κ(E₆₄) − κ(T₆₄), the amount by which the frame-specific evaluation exceeds the template
  ranking, is 0.080 there, which is conditional.
- Exploratory Tier A results are drawn only for 005 and 006.
- The head read-outs of 009 and 011 are given the measured input; 012 was evaluated at exposed base states; every
  new-frame result from 013 on is conditional, including the new-frame share of statistics pooled over both sets;
  019's results are aggregate, not per frame.
- A criterion that passed under a failed label is not shown as a pass.

**F3, the head** ([PNG](F3-head.png), [SVG](F3-head.svg)). The transport head `L03.H04`.
- A1: 009's 23 token means of the P1 read-out against the measured transport [descriptive]. 009's read-out test was
  scored over 138 pairs, which are not committed: Spearman 0.957, MAE 0.066 [prospective, given the measured input].
- A2: 011's 142 scored pairs, the P1 read-out replicated, to which its Spearman 0.974 and MAE 0.076 apply
  [prospective, given the measured input]. The open circles are `an`'s pairs, one for each of its four licensed
  cue-final frames.
- B: 011's 24 locked predictions from the encoding read, taken from its predictions file, against the measured mean
  transport: Spearman 0.848, MAE 0.077 [prospective]. The read is weight-only apart from the frozen axis. `an` is a
  mean over its four licensed cue-final frames, predicted at 0.18 and measured at 0.30, the lowest.
- Dashed lines are identities.

**F4, the opposition ledger** ([PNG](F4-opposition-ledger.png), [SVG](F4-opposition-ledger.svg)). 010's split of the
encoding's read into its axis part (filled) and its rest (outlined), as fractions of the plural cue's measured head
change, means over informative frames [exploratory, on spent data]. For `this`, `a`, `another` and `every` the rest is
negative in 24 of 24 frames. The pattern is not general: `either`, also singular-selecting, has a positive rest, and for
`these` the rest supports the signal. 010's discovery summary was `MIXED` and its relay bound failed. Per-frame values
of the split are not committed, so token means are shown.

**F5, channel D's closure ladder** ([PNG](F5-closure-ladder.png), [SVG](F5-closure-ladder.svg)). κ, the closure of
channel D's contribution to `c_L`, for the nested neuron subsets of 018.
- The controls are three seeded random 256-neuron subsets and the 256 lowest-ranked neurons. The oracle, each frame's
  own subset, is a descriptive comparator.
- Circles are new cues in exposed frames, diamonds new cues in new frames.
- The locked 256-neuron rung is plotted at its decision values: 0.863 [prospective], and 0.843 in new frames, whose
  label is unestablished because the no-harm guard failed in one frame. The other rungs, the controls and the oracle
  are descriptive, at two decimals.
- The star is the single-neuron statistic, `S_1` over both sets pooled and pair-weighted: κ 0.219 against `S_256`'s
  0.860, rejected [prospective]. It is not Y1's one-neuron rung, 0.24, and its new-frame share is conditional.
- No curve is drawn, since no monotonicity was predeclared.

**F6, end to end** ([PNG](F6-end-to-end.png), [SVG](F6-end-to-end.svg)). The contrast change under real cue swaps.
- A: 023's share `g` of the gap to the ceiling recovered, per condition, against its preregistered floor `F` (black
  tick), with the calibration draws' range (grey bar) and median (grey tick), on an axis zoomed to 0.985–1.017. Three
  conditions pass, with Y2 conditional; Y1/cue_final is an `ENVELOPE_ONLY_FAILURE`, its 0.90 guard met. There is no
  aggregate label, and Y1 and Y2 are never pooled.
- B: 023's flattened `Δc` R² for the E-only chain, the completed chain and the ceiling [descriptive], offset within
  each row. The ceiling is a comparator, not a mathematical upper bound.
- C: 021's end-to-end flattened R² on 020's wiring [descriptive]. Y1 is new cues in exposed frames, with exposed nouns;
  Y2 is new cues in new frames. The erratum: 020's program built the coordinated frames' layer-1/2 reference rows
  through `p_t` instead of the cue position `p_c`. Panel C is not comparable with A or B.

**F7, the sequence** ([PNG](F7-sequence.png), [SVG](F7-sequence.svg)). Experiments 005–025 as the source map's
overview records them.
- Each row gives its subject and frozen outcome verbatim, one chip per label coloured by its recorded status, and the
  status verbatim beneath.
- 022's and 023's passes carry one badge per population, since their Y2 conditions are conditional.
- Statistics pooled over both sets, 013–016 Y3, 018 Y3 and 019 Y3–Y5, have a conditional new-frame share, which their
  colour does not show. The head read-outs, 009's mechanism and 011's `HEAD_P1_REPLICATED`, are given the measured
  input.
- The chips of 006 and 020 for the confirmatory test that was never run come from the status column. 020's *none
  assigned* is drawn as a chip because it is that row's whole outcome; 022's and 023's *no aggregate label* qualifies
  their labels and is plain text.
- The overview's screen row and its 023-d row, the post-closure diagnostic, are not plotted. The axis is ordinal: no
  durations are drawn.

**F8, the boundary** ([PNG](F8-boundary.png), [SVG](F8-boundary.svg)). The evidence gates of C002 and C003. Each
cell names its claim's own state. Levels 0–2 are checked. At Level 3 the preregistration and reporting items are
checked, and necessity, sufficiency and controls are not established. Level 4 is not reached, and Levels 5 and 6 are
not claimed. The row titles are C003's items; where C002's own wording differs, it is shown beneath. C003 has one
Level-1 item that C002 lacks. 025 bears on C003 only: C002 records that it does not test C002's chain
(research/claims/C002-count-cued-noun-number-circuit.md@454327d:66).

**F1a, the chain schematic** ([PNG](F1a-chain-schematic.png), [SVG](F1a-chain-schematic.svg)). F1's stage row alone,
drawn for the body of draft v3 as its Figure 1: the stages in the model's order, with the encoding in parallel with
block-0 attention. It plots no data.

**F6a, the recovered share per condition** ([PNG](F6a-completion.png), [SVG](F6a-completion.svg)). F6's panel A alone,
drawn for the body of draft v3 as its Figure 5, from the same data as F6: 023's `g` per condition against its floor
`F`, with the calibration draws' range and median, on a zoomed axis.

**In draft v3** the figures are numbered by first citation: Figure 1 is F1a, Figure 2 F4, Figure 3 F3, Figure 4 F5,
Figure 5 F6a and Figure 6 F1; in the appendices, Figure A1 is F7, Figure C1 F2 and Figure C2 F8. F6 is not used there.

## Where the figures go beyond the audit's plan

Each is presentational, uses only values or words the record commits, and is checked in `figure_data.py`.
- **F1.**
  - A `Δc` stage holds 021's three labels in the real-swap lane and 009's contrast rule, never locked, in the E-patch
    lane. The audit's F1 table has no row for either, but `Δc` is the last stage of the chain of §4 of the draft.
  - The badges split 022's and 023's results per condition, as the audit asks, and add *rival* badges for the
    alternatives of F2's rows 014–019.
  - The chips' wording is the figure's own; each badge's status is checked as described above.
- **F2.** Each cell names its status. The note says which empty cells mean not tested: the exposed-frame cells of
  009, 011 and 012, which the audit names, and their pooled cells, since those experiments tested new cues in new
  frames only (research/synthesis/synthesis-005-025.md@3d87d3c:272); 007's, whose Y families ran on the six fresh
  frames only (007:59–60); and 006's, whose confirmatory test never ran (006:71). It adds the conditional new-frame
  share of pooled statistics (research/synthesis/source-map-005-025.md@b722e9c:61).
- **F3.** Panel A2 marks `an`'s four pairs, the four licensed frames behind Panel B's lowest point.
- **F4.** The ledger's `either` row is added (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:45),
  as the singular-selecting case whose rest is positive; the audit's F4 section names it. The axis takes the ledger's
  own heading (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:25).
- **F5.** Colour is per rung and per population: the locked rung is prospective in exposed frames and unestablished in
  new frames; every other rung, the controls and the oracle are descriptive in both.
- **F6.** Panel B's markers are offset within each row, since the completed chain's and the ceiling's values nearly
  coincide and would hide each other.
- **F7.** The chips take the overview's outcome wording verbatim, so 005's version markers precede their labels and
  007's chip keeps *(Y1 failed)*; the status column is printed verbatim beneath each row. 022's and 023's passes carry
  one badge per population, since their Y2 conditions are conditional.
- **F8.** The row titles are C003's items, with C002's differing wording beneath; an item stated only by its state is
  titled with its level's name.
- **F1a and F6a.** F1's stage row and F6's panel A, each drawn alone for draft v3's body, with no new data.
- **All figures, since draft v3.** No figure titles or footnote notes inside the images; the captions carry them.
  Panel labels no longer use 009's and 011's own family names (Y1, Y2), which elsewhere name populations.

## Not drawn

- F6's Panel D, 020 on exposed data: not in the plan, and an annotation only if added.
- F6's `g − F` margins, which only the committed independent review of 023 states.
- F3's 138 pairs of 009, which are not committed; the near-substitutes the audit names are not used.

## Limits

- **Print size.** The figures are drawn for reading on screen, 10–15 in wide with text of 6–8 pt. Printed at a text
  width of 6.5–7 in, the text of F1, F2 and F7 would fall to about 3 pt. A print version needs its own layout, for
  example a full page each, or F1 split by lane.
- **The chips of F1 and F7** name results, not claims; the claims are C002 and C003.

## Reviews

Two independent read-only reviews checked the figures on 2026-09-30: one the data's provenance and statuses, the other
the figures, their captions and the drawing code. Each had a re-review and a final check. No plotted value was ever
found wrong, and nothing was found computed.

- **First round: both FAIL.**
  - Provenance, two blockers: F7 coloured 022's and 023's passes plain prospective though their Y2 conditions are
    conditional, and F2's note said that empty cells were not tested, which is false for several. Notes: F1 lacked
    F2's composite and rival results; some anchors were too weak to catch a wrong line, and statuses were not checked
    label by label; qualifiers and wording (*primarily*, the single-neuron statistic, the oracle, F2's metric labels);
    the pins predating `57d6450`.
  - Figures, three blockers: F1 and F2 carried status by colour alone, which fails in greyscale and partly under
    deuteranopia; the same F2 note; a dash in F1's E-patch `Δc` cell, where 009's contrast rule, never locked, belongs.
    Notes: the head inside the readout shown in text only; merged conditions; F2's 019 cell; the oracle called a
    control; F6's zoomed axis, hidden markers and erratum wording; F8's titles and fills; an incomplete list of
    departures; print size.
  - Done: every status checked against its experiment's block in the source map's evidence-status section; patterns
    for the statuses that colour alone carried; badges per condition, rival badges and the 009 chip; the F2 note
    rewritten; the wording, F5, F6, F8 and README fixes; print size stated as a limit.
- **Re-review: provenance FAIL, figures PASS WITH NOTES.** The rewritten F2 note named too few untested cells; the
  status check could not tell a conditional result from a prospective one; the rival wording did not fit 019's
  insufficient template family. Figure notes: A undefined; a computed difference in this README; exploratory close to
  the conditional tint in greyscale; white text below 4.5:1 contrast; F4 reusing the unestablished hatch; F7's badges
  missing from the departures. All applied: the note lists the untested cells with their sources; a prospective status
  may no longer rest on an anchor that names a conditional result; exploratory fills are striped; text colour follows
  the higher contrast; F4's rest bars are outlined.
- **Final check: both PASS WITH NOTES.** Both asked for one clause to be narrowed: 007's new cues, not all its tests,
  ran on new frames only. Applied. The figures regenerated byte for byte in both reviewers' scratch copies, and the
  checker passed on both READMEs.
- **Draft v3, 2026-09-30.** One reviewer checked the v3 changes (titles and notes removed, F1a and F6a added) and then
  the relabelling that followed. Both checks were PASS WITH NOTES, and every note was applied. No plotted value, colour
  or pattern changed, and the figures regenerated byte for byte in the reviewer's scratch copies. The reports are with
  the [reviews of draft v3](../paper-review-v3-2026-09-30/README.md).

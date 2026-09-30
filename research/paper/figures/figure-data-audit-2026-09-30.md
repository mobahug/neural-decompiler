# Figure-data audit, 2026-09-30

**Status: read-only.** This audit checks the figure plan of draft v1's Appendix B, F1–F8
(research/paper/draft-v1-2026-09-29.md@1a81f30:845–857), against the committed record, before any figure is drawn.
- **What it records.** For every planned panel:
  - where each value to be plotted is committed;
  - the value's evidence status;
  - which transformations the plot would need;
  - a verdict.
- **What it did not do:**
  - it computed nothing, ran no model and no experiment code;
  - it read no uncommitted output: nothing under `outputs/` was opened, and 022's calibration table was not read;
  - it changes no draft, claim or record.

  The draft corrections it found are listed for the next revision.

Three read-only agents traced the values, one for F3–F4, one for F1, F2 and F5, and one for F6–F8. Their key
sources were checked again before this document was written.

## Rules

- **Allowed.** Presentational transformations only:
  - selecting and ordering;
  - labelling;
  - colouring by a recorded status;
  - choosing axes;
  - drawing an identity line;
  - showing a recorded number at lower precision.
- **Not allowed** unless the record already states the result:
  - computing a new statistic, difference, residual, fit or curve;
  - aggregating per-item values into a new summary;
  - using an uncommitted output;
  - recomputing anything from the model.
- **Verdicts:**
  - *supported*: every value is committed, and only presentational transformations are needed;
  - *supported with limits*: the limits are stated;
  - *redesign*: the planned form needs uncommitted or computed data, and a record-supported form is given;
  - *reject*: nothing in the record supports it.
- **Citations.** Markdown sources are cited by line, in the checker's form, as they stood at `d7c5e62` unless a citation
  names another commit; `(011:75–76)` is `experiments/011-*/README.md` at `d7c5e62`. Of the files cited that way, only
  025's README has changed since: its line 244 is line 250 at `1a81f30`. The committed JSON records are single lines, so
  they are cited by file and key path. The checker cannot verify key paths; the independent review of this audit checks
  them.

## Summary

| figure | panel | verdict | colour by | main limit |
|---|---|---|---|---|
| F1, the chain | — | redesign | recorded tag per chip | no single status per stage exists |
| F2, the prediction map | — | supported with limits | label-level status | four population columns; metric labelled per cell |
| F3, the head | A, the P1 read-out | redesign | status | 009's 138 pairs are not committed |
| F3, the head | B, the encoding read | supported | status | none |
| F4, the opposition ledger | — | supported with limits | exploratory | the axis-and-rest split as token means only |
| F5, the closure ladder | — | supported with limits | status per rung | two decimals off the decision rungs; no curve |
| F6, end to end | A, `g` against `F` | supported with limits | status per condition | margins only from the committed review |
| F6, end to end | B, 023's R² | supported with limits | descriptive | not a bound |
| F6, end to end | C, 021 | supported with limits | descriptive | erratum in the caption; not compared with Panel B |
| F6, end to end | D, 020, not in the plan | supported with limits | exploratory | annotation only, if added; not a baseline |
| F7, the sequence | — | supported with limits | one chip per label | ordinal axis; verbatim labels |
| F8, the boundary | — | supported with limits | the claims' own words | no counts; no merged states |

No figure is rejected.

## F1: the chain

**Planned:** the chain of §4, with each stage's status.

**Verdict: redesign.** The record gives each stage several statuses, not one.
- §4 tags only two stages:
  - the encoding, as exploratory (research/paper/draft-v1-2026-09-29.md@1a81f30:305);
  - transport, as prospective given the measured input (research/paper/draft-v1-2026-09-29.md@1a81f30:324–325).
- Picking one label for a stage would aggregate. The record forbids that for block 0, whose four conditions carry no
  aggregate label (023:149–151).

**The record-supported form.** Each stage box carries chips, one per experiment or family, each with its recorded tag
and its citation. The chips run in two lanes:
- **the E-patch lane,** stage by stage, 009–019;
- **the real-swap lane,** end to end, 021–024, where only `Δc` was scored.

A stage-level caption appears only where the record states one:
- the two tags of §4;
- C003's stage summary (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:374–377);
- C003's readout summary (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:70–71).

| stage | chips, with their recorded status |
|---|---|
| the encoding | 010's opposition [exploratory] (010:85–89); the compact encoding programs of 005 and 006 [exploratory] and 007 [unestablished] (005:98–101; 006:67–72; 007:87–94); 011's read of transport [prospective], drawn on the encoding-to-transport link (011:75–76; research/synthesis/synthesis-005-025.md@3d87d3c:258) |
| block 0, real swap only | 022's block-0 conditions, C1 and C4 [prospective, per condition; Y2 conditional] (022:187, 190–191, 194, 246); 023's three passes [prospective; Y2 conditional] and Y1/cue-final [unestablished at its envelope; guard met], with no aggregate label (023:140–155); the cheaper block-0 rules [descriptive] (023:180–181) |
| layers 1–2 | the E-patch rows 012–019 of F2, each with its tag (research/synthesis/source-map-005-025.md@b722e9c:151–205); under a real swap, the layer-1–2 reductions at most 0.007 of the gap [prospective, per condition; Y2 conditional] (022:188–189, 192–193, 247) |
| `Δx3` | under a real swap, the reconstruction is [descriptive] (022:225–229); C003 lists it under what is not supported (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:396–397) |
| transport | the P1 read-out [prospective, given the measured input] (009:62–64, 83–85; 011:87–89); 017's head row and `ΔT` [prospective; Y2 conditional] (017:116–124); the pattern term's necessity [unestablished] (017:126–131; 018:122–126); 009's rank-1 rule [unestablished] (009:69–70) |
| the readout | the ceiling 0.976 on exposed data [exploratory] (020:132); fresh ceilings and `R²_C` [descriptive] (021:140–141; 023:140–145); 020's confirmation never ran [unestablished] (020:8–11); 024's nounness association [prospective; predictive, not causal] (024:277–279) |
| the causal band | C003's necessity, sufficiency and control gates, not established (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:257–267); 025 [unestablished] (025:244) |

**The drawing must also show:**
- **Block 0 in parallel.** The block-0 MLP runs in parallel with block-0 attention (012:9; 020:47), so the arrow
  from the encoding to block 0 must not suggest a sequence.
- **The head inside the readout.** In the end-to-end program, `L03.H04` is computed exactly inside block 3, as part
  of the readout (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:70–72;
  research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:62–63). The transport chips come from the E-patch
  lane.

## F2: the prediction map

**Planned:** the table of §5 as a status-coloured grid, failures included.

**Verdict: supported with limits.** Every number in §5 matches its README and its final report. The confirmation
JSON files of 006–019 are frozen input sets that hold no results. F2's statistics are recorded only in the READMEs and
the final reports; the per-pair extracts under the `inherited/` directories of 007–019 hold no F2 statistic. No
transformation is needed.

**Where the cells are committed.** Each row is cited in the draft's own audit trail (draft §5), and each was checked
again against its final report where one exists; 006 has none.

| row | value | committed source |
|---|---|---|
| 009 Y1 | 0.957 / 0.066 on 138 pairs | (experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md:72; 009:62) |
| 011 Y1, Y2 | 0.848 / 0.077; 0.974 / 0.076 | (experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md:34–35) |
| 012 Y1, Y2 | 0.864 / 0.061, R² 0.795; 0.922 / 0.062 | (experiments/012-layer-correction-token-local/evidence/final-report-2026-09-19.md:44–45) |
| 013 Y1, Y2 | 0.877, R² 0.668; 0.919, R² 0.747 | (experiments/013-attention-paths-frozen-pattern/evidence/final-report-2026-09-19.md:45, 48) |
| 014 Y1, Y2 | 0.950, R² 0.970; 0.910, R² 0.957 | (experiments/014-neuron-1987-feature/evidence/final-report-2026-09-19.md:46, 52) |
| 015 Y1 | 0.986, R² 0.895 | (experiments/015-attention-pattern-token-local/evidence/final-report-2026-09-20.md:51) |
| 015 Y2 | R² 0.490 and layer-2 entry 0.423 missed 0.50; Spearman 0.857 and layer-1 entry 0.888 passed | (experiments/015-attention-pattern-token-local/evidence/final-report-2026-09-20.md:64; 015:109–110) |
| 016 Y1, Y2 | 0.998, R² 0.994; 1.000, R² 0.996 | (experiments/016-frame-conditioned-qk/evidence/final-report-2026-09-20.md:52, 62) |
| 017 Y1, Y2 | row R² 0.997; `ΔT` pairs R² 0.999; row R² 0.998; `ΔT` pairs R² 1.000 | (experiments/017-transport-head-pattern/evidence/final-report-2026-09-20.md:56, 67) |
| 018 Y1 | 0.863 and 0.724 | (experiments/018-block2-concentration/evidence/final-report-2026-09-21.md:66) |
| 018 Y2 | no-harm guard failed in one frame, −0.102 against −0.05 | (experiments/018-block2-concentration/evidence/final-report-2026-09-21.md:89; 018:110–111) |
| 019 Y1, Y2 | `Δκ_64(E)` 0.090, 64 of 90 frames; 0.144, 14 of 18 frames | (019:120–123, 129–133; experiments/019-block2-routing/evidence/final-report-2026-09-21.md:62, 164) |
| rivals | −1.41; −1.16; layer-2 entry 0.672; 0.219 against 0.860; margin 0.068 pooled and 0.080 on new frames | (014:104–105; 015:123–124; 016:118–119; 018:118–121; 019:145–148) |
| frozen pattern | `ΔT` R² 0.958 on 1,248 pooled cue-final pairs | (017:126–129) |
| failed compact programs | 1.44 nats; normalized RMSE 0.623; Spearman 0.785; 0.765 | (005:98–101; 006:67–72; 007:87–94; 009:69–70) |

**Limits.**
- **Four population columns, not two:**
  - new cues in exposed frames;
  - new frames;
  - both sets pooled, for the rivals and 017's frozen-pattern value;
  - exposed or development data, for 005 and 006, which are exploratory.
- **Metrics labelled per cell.** The cells hold different metrics: Spearman and MAE, R², entry R², κ, `Δκ` and a
  margin. Colour by status, never by value.
- **Colour by label-level status.** A criterion that passed under a failed label, as in 015 Y2, 018 Y2 and 007, is an
  annotation only (research/synthesis/source-map-005-025.md@b722e9c:65–67).
- **Qualifiers:**
  - *given the measured input* for 009 Y1 and 011 Y2;
  - *exposed base states* for 012;
  - *conditional* on every new-frame cell from 013 on;
  - *aggregate, not per frame* for 019.
- **Empty cells.** The exposed-frame cells of 009, 011 and 012 mean not tested, not failed
  (research/synthesis/synthesis-005-025.md@3d87d3c:272).

## F3: the head

**Panel A, the P1 read-out given the measured input: redesign.**
- **What the plan needs and the record lacks.** The plan does not say whether it plots pairs or token means. This
  audit assumes that the panel should show 009 Y1's scored population, its 138 pairs. They are not committed; only
  009's 23 token means are, in its final report
  (experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md:75–99).
- **Near-substitutes that must not be used.** Three committed files look like 009's pairs but are not its Y1 record:
  - 009's per-pair E-patch contrast shifts, in
    `experiments/010-read-direction-assembly/inherited/experiment-009-confirmation-epatch-means.json`;
  - 010's exploratory re-measurement of the same pairs, in 011's and 012's inherited extracts of 010
    (`experiments/011-encoding-read-prospective/inherited/experiment-010-transport-fractions.json`, `q_T` only, and
    `experiments/012-layer-correction-token-local/inherited/experiment-010-component-fractions.json`), whose token
    means differ from 009's record.
- **A column to avoid.** In 009's final-report table
  (experiments/009-head-transport-rule/evidence/final-report-2026-09-18.md:75), the column `q_T predicted` is the
  failed rank-1 rule's prediction, not P1.
  Plot only the columns `P1` and `q_T measured`.
- **Status.** No criterion was scored on the P1 token means, so they carry no outcome force [descriptive].
- **011 Y2's pairs are committed,** all 142 of them:
  - where: in 012's inherited extract,
    `experiments/012-layer-correction-token-local/inherited/experiment-011-component-fractions.json`, keys
    `entries["<token>|<frame>"].f_total` (the P1 read) and `.q_T`;
  - why it is 011's state: the extract's state digest is the one 011 recorded for its confirmation state (011:65),
    and 012 replicated 011's 142 pairs exactly (012:137);
  - the key: `f_total` is Y2's predictor by construction (src/neural_decompiler/encoding_read.py:650, 692).
- **The record-supported form,** in two separate subpanels, since each uses its own frozen axis:
  - **A1:** 009's 23 token means with an identity line. The caption gives the pair-level result, 0.957 / 0.066 over
    138 pairs, not shown (009:62–64).
  - **A2:** 011 Y2's 142 pairs with an identity line. Its recorded 0.974 / 0.076 applies to exactly these points
    (011:87–89).

  The pair-level results, 009 Y1 and 011 Y2, are [prospective, given the measured input]
  (research/synthesis/source-map-005-025.md@b722e9c:62–64, 136–137, 145–146). A1 plots token means, which are
  [descriptive]; A2 plots the scored pairs themselves. No statistic may be recomputed over the token means.

**Panel B, the encoding read: supported.**
- **The data.** All 24 locked predictions and all 24 measurements are committed.
  - The predictions are in 011's predictions file, in the bold mean rows
    (experiments/011-encoding-read-prospective/predictions.md:12–104), and in its lock at
    `predictions.tokens.<w>.means.g_E`.
  - The measurements are in its final report
    (experiments/011-encoding-read-prospective/evidence/final-report-2026-09-19.md:41–64).
- **The statistic and `an`.** The statistic is 0.848 / 0.077 over these 24 points (011:75–76). `an` was predicted at
  0.18 and measured at 0.30, the lowest of the 24 (011:77–79).
- **Status.** The predictions are locked, and Y1 is prospective and not frame-conditional
  (research/synthesis/source-map-005-025.md@b722e9c:52, 145–147).
- **The caption must say:**
  - `an` is a mean over its 4 licensed cue-final frames (011:18–19);
  - the read is weight-only apart from the frozen axis (010:44–45).
- **Not allowed:** plotting residuals.

## F4: the opposition ledger

**Verdict: supported with limits.**
- **What is committed.** The split of the encoding into the axis part `f_∥` and the rest `f_⊥` is committed only as
  token means over the 24 frames, in 010's exploration report
  (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:27–33, 89), beside each token's
  measured transport `q_T` and encoding read `f_E`:

  | token | `q_T` | `f_E` | `f_∥` | `f_⊥` |
  |---|---|---|---|---|
  | `every` | −0.129 | −0.074 | 0.170 | −0.244 |
  | `this` | 0.034 | 0.199 | 0.370 | −0.171 |
  | `another` | 0.171 | 0.128 | 0.401 | −0.272 |
  | `a` | 0.195 | 0.104 | 0.456 | −0.352 |
  | `these` | 0.965 | 0.830 | 0.515 | 0.316 |

- **Per-frame values.** Per-frame `q_T` and `f_E` are committed, in
  `experiments/012-layer-correction-token-local/inherited/experiment-010-component-fractions.json`, keys
  `entries["<token>|<frame>"].q_T` and `.f_E`. Per-frame `f_∥` and `f_⊥` are not, so the split is shown as token means
  only.
- **The 24-of-24 count.** It covers the rest, `f_⊥`, of the four tokens (010:85–89). No per-frame sign of the axis
  part is recorded, and there is no per-frame count for `these`.
- **Status.** Everything is exploratory, on spent data. The discovery summary was `MIXED`, and 010's relay bound
  failed (010:62–67). The caption must say so.
- **The caption must name the four tokens.** The ledger does not support a statement about singular-selecting
  determiners in general. In 010's ledger the singular-selecting `either` has a
  positive rest, +0.156 (experiments/010-read-direction-assembly/evidence/exploration-report-2026-09-18.md:45).

## F5: channel D's closure ladder

**Verdict: supported with limits.** Every rung's κ is committed with its subset size.
- **The values.**
  - Every rung's κ is in 018's final report, at two decimals: new cues in exposed frames, new frames, and both sets
    pooled (experiments/018-block2-concentration/evidence/final-report-2026-09-21.md:69–81, 92–104, 139–150).
  - The subset sizes are in the README (018:14–16) and in the lock, at `floors.subset_sizes`.
- **What is not committed.**
  - The rungs that are not decision statistics, at more than two decimals; they sit in the uncommitted results state,
    of which only a digest is committed (018:85–86).
  - Per-frame κ for any rung other than the locked 256-neuron subset. Its per-frame κ on the twelve new frames is the
    only per-frame κ series committed, beside per-frame R² values for a few rungs and the oracle
    (experiments/018-block2-concentration/evidence/final-report-2026-09-21.md:112–123).
  - Confidence intervals: none is recorded for any rung, and 018's design and README define none.
- **Statuses:**
  - **The locked 256-neuron rung on new cues in exposed frames** (0.863 and 0.724) is prospective (018:102–104).
  - **The other rungs, the oracle and the random and lowest-ranked controls** are descriptive
    (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:217;
    research/synthesis/source-map-005-025.md@b722e9c:193).
  - **The pooled single-neuron statistic,** 0.219 against 0.860 with a gap of 0.641, is prospective: it is Y3's
    rejection (018:118–121; research/synthesis/source-map-005-025.md@b722e9c:190–191).
  - **The new frames** are unestablished at label level: the no-harm guard failed in one frame. Their pooled closures
    passed at criterion level (research/synthesis/source-map-005-025.md@b722e9c:192, 196–197).
- **The form.**
  - Rungs go on a categorical axis, since the empty subset cannot sit on a log axis.
  - No curve. No monotonicity of the ladder was predeclared
    (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:222).
  - The single-neuron statistic is a separate marker, labelled both sets pooled. It is not the one-neuron rung of Y1
    (0.24, descriptive).
  - The random controls are shown one by one from the final report, since the README gives only their ranges
    (018:130–134).
  - The exposed-frame series is captioned new cues × exposed frames, to keep it apart from the exploratory Tier-A
    record (018:75).

## F6: end to end

**Panel A, 023's `g` against its floor `F`: supported with limits.**
- **The values.** All values are in 023's README (023:142–145), in its final report
  (experiments/023-block0-completion/evidence/final-report-2026-09-24.md:28–33), and in its confirmation record at
  `confirmation.conditions["<condition>"]`, keys `.g`, `.envelope.bound` and `.result`. They agree at the displayed
  precision.
- **The draws.** The draws' minimum, median and maximum per condition are in the final report
  (experiments/023-block0-completion/evidence/final-report-2026-09-24.md:19–22).
- **Form.** One row per condition: a dot for `g`, a tick for `F`, and the draws' range behind them, on a zoomed axis.
  The 0.90 guard is stated in text.
- **Margins.** `g − F` margins may be plotted only from the committed independent review, which states all four
  (experiments/023-block0-completion/evidence/confirmation-review-2026-09-24/REVIEW.md:25–28). The README states only
  Y1/coordinated's (023:156–157).
- **Status.** Three conditions are prospective, with Y2 conditional. Y1/cue-final is unestablished at its envelope,
  with its guard met (research/synthesis/source-map-005-025.md@b722e9c:92).
- **Not allowed.** No aggregate, since there is no aggregate label (023:150–151), and no pooling of Y1 with Y2
  (experiments/023-block0-completion/evidence/final-report-2026-09-24.md:43).

**Panel B, 023's R² values: supported with limits.**
- **The values.** R²₀, R²₁ and R²_C per condition, at the same places; in the record, keys `.R2_0`, `.R2_1` and
  `.R2_C`.
- **Status.** Descriptive (023:42; research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:185).
- **Form.** The ceiling must not be drawn as a limit: it is not a mathematical upper bound (023:21–22). The axis label
  names the statistic: the flattened `Δc` R² over pairs and nouns.
- **Differences.** The only recorded difference is the gap, `(SSE0 − SSEC)/SST`, at key `.gap`
  (023:140–145; experiments/023-block0-completion/evidence/final-report-2026-09-24.md:28–33); it equals R²_C − R²₀.
  It may be drawn. No other difference between the three values is drawn.

**Panel C, 021 on 020's wiring: supported with limits.**
- **The planned values.** Appendix B follows the readiness assessment
  (research/paper/draft-v1-2026-09-29.md@1a81f30:845). The assessment names 021's descriptive R² on 020's wiring,
  0.6167 and 0.5797 (research/synthesis/paper-readiness-2026-09-28.md@139755d:289).
  - They are Y1's and Y2's end-to-end flattened R² (021:132, 137).
  - In 021's confirmation record they sit at `confirmation.Y1.statistics_020.flattened_r2` and
    `confirmation.Y2.statistics_020.flattened_r2`.
- **Status.** Descriptive: neither is a frozen condition of Y1 or Y2 (021:114–127).
- **Populations.**
  - Y1 is 24 fresh cues × 108 exposed frames, with 79 exposed nouns (021:129).
  - Y2 is 24 fresh cues × 18 fresh frames, 432 pairs (021:134), over the same 79 exposed nouns, at key
    `confirmation.Y2.statistics_020.n_nouns`.
- **Form.** The values get their own axis, labelled with the statistic. The caption carries the erratum (021:13–17).
- **Not connected to Panel B.** The wiring and the cues differ, and 021 pools its three templates where 023 splits
  cue-final from coordinated frames. Y1's 108 exposed frames are the same in both: key `exposed_frame_ids` is
  identical in `experiments/020-readout-decompilation/confirmation-v1.json`, which 021 read in place (021:28), and in
  `experiments/023-block0-completion/confirmation-v1.json`.
- **If §6's three values are added,** each keeps its own status:

  | value | what it is | status | source |
  |---|---|---|---|
  | 0.016018 | Y1's token-mean R², against a floor clamped to 0.0; a pass | prospective | (021:116) |
  | −0.014 | Y1's coordinated-adjective R² | descriptive | (021:132–133) |
  | −0.1538 | Y2's `coordinated_r2`, against a floor clamped to 0.0; the failed condition itself | unestablished | (021:123, 134–137) |

  In the record they sit at `confirmation.Y1.values.token_mean_r2`,
  `confirmation.Y1.statistics_020.per_template["coordinated-adjective"].r2` and `confirmation.Y2.values.coordinated_r2`.
- **The Y1 value's population.** 0.016018 is Y1: fresh cues in exposed frames, with 79 exposed nouns (021:129). The
  fresh nouns were Y3, which passed (021:138). Neither a caption nor the draft may say that this value covers new
  nouns.

**Panel D, 020 on exposed data: not in the plan (research/paper/draft-v1-2026-09-29.md@1a81f30:854–855); supported with
limits (annotation only), if added.**
- **The values.** The ceiling, 0.976, and the end-to-end value, 0.689, are exploratory (020:114, 132).
- **Their labels.** The phrase “calibration record, never evidence for a label” (020:110) heads only the table that
  holds 0.689. The ceiling's own label is “ceiling, never a floor” (020:132).
- **Not a baseline.** Neither value may serve as a baseline for later results.

## F7: the sequence

**Verdict: supported with limits.**
- **What each row takes.** Its outcome and its status, verbatim from the source map's overview
  (research/synthesis/source-map-005-025.md@b722e9c:74–92, 94–95). The caption says that the overview's screen row and
  its 023-d row, the post-closure diagnostic, are not plotted.
- **Chips, not row colours.** Rows 005, 006, 007, 009, 015, 017, 018, 020, 021 and 023 carry more than one status,
  so each label gets its own chip. The rows of 006 and 020 add a chip for *the confirmatory test was never run*
  [unestablished] (research/synthesis/source-map-005-025.md@b722e9c:45, 125, 212). In 007's row the second status is
  criterion-level (research/synthesis/source-map-005-025.md@b722e9c:127); as in F2, it is an annotation on the
  label's chip, not a chip of its own (research/synthesis/source-map-005-025.md@b722e9c:65–67).
- **An ordinal axis, by experiment number.**
  - Every README has a dated status line, and the dates never decrease with the experiment number.
  - The dated events differ, though (research/synthesis/source-map-005-025.md@b722e9c:687, 697). So no durations are
    drawn.
  - Dates, if shown, are quoted day labels only.
- **Appendix A is not a verbatim copy of the overview.** It differs from the overview in these places, and only the
  011 change is disclosed:
  - 007's outcome drops (Y1 failed);
  - 022 and 023 move *no aggregate label* into the status column, and 022's outcome gains the word *conditions*;
  - 023's outcome replaces the failing condition's name with *one*;
  - 005's version markers move behind their labels, in parentheses;
  - multi-label outcomes are split into separate labels joined by commas, a cosmetic change;
  - the overview's screen row and 023-d row are omitted without a note.

  The figure must take the overview's wording. The draft's appendix is to be brought into line with it.

## F8: the boundary

**Verdict: supported with limits.**
- **The source.** A matrix of the two claims' gate items, taken from their evidence-gate sections
  (research/claims/C002-count-cued-noun-number-circuit.md@454327d:71–121;
  research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:204–303).
- **Recorded states:**
  - Levels 0–2 are checked.
  - At Level 3, the preregistration and reporting items are checked. Necessity, sufficiency and controls are not
    established (research/claims/C002-count-cued-noun-number-circuit.md@454327d:101, 103, 105;
    research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:257, 261, 266).
  - Level 4 is not reached, and Levels 5 and 6 are not claimed
    (research/claims/C002-count-cued-noun-number-circuit.md@454327d:113, 117, 121;
    research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:278, 287, 293).
- **The claims' own words.** Each cell uses its claim's own wording. Neither claim file says *open*, so *not
  reached* and *not claimed* are not merged into *open* or *not established*. The figure prints no counts.
- **An extra row.** C003 has one Level-1 item that C002 lacks
  (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:222). C002's cell for it reads no such item.
- **025.** It appears only in C003's column. C002 records that 025 does not test its intervention chain
  (research/claims/C002-count-cued-noun-number-circuit.md@454327d:66).

## Draft corrections found

These are for the next revision of the draft. None changes a number. The draft is at
research/paper/draft-v1-2026-09-29.md, cited at `1a81f30`.
1. **§9's third contribution overgeneralizes.** It speaks of an opposing off-axis part in the encoding of
   singular-selecting determiners (research/paper/draft-v1-2026-09-29.md@1a81f30:637–638). The record names four
   tokens, `this`, `a`, `another` and `every` (010:85–89).
2. **The abstract and §6 over-state 021's Y1 population.** They attach 021's small-margin Y1 pass to new cues *and
   nouns* (research/paper/draft-v1-2026-09-29.md@1a81f30:66, 415). Y1 used exposed nouns (021:129); new nouns were Y3.
3. **Appendix B's F6 entry does not name its 021 values.** It speaks of 021's descriptive values
   (research/paper/draft-v1-2026-09-29.md@1a81f30:854–855). The readiness assessment names them, 0.6167 and 0.5797
   (research/synthesis/paper-readiness-2026-09-28.md@139755d:289), but the draft never does.
4. **Appendix B's F5 entry misses a prospective statistic.** It says only the locked rung is prospective
   (research/paper/draft-v1-2026-09-29.md@1a81f30:852). The pooled single-neuron statistic is prospective too, and
   *exposed frames* should read *new cues in exposed frames*.
5. **Appendix B's F3 entry drops a qualifier.** It calls 011's read weight-only
   (research/paper/draft-v1-2026-09-29.md@1a81f30:849). The record says weight-only apart from the frozen axis
   (010:44–45). 011's design requires that qualifier for the read
   (docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md:160–162, 196–197), though its
   own title omits it (docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md:1).
6. **§2 describes the no-harm guard as two-sided.** It says the subset's R² must stay within 0.05 of the template
   base's (research/paper/draft-v1-2026-09-29.md@1a81f30:235–236). The guard is one-sided: the subset's R² must be at
   least the base's minus 0.05
   (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:364).
7. **§5 says all alternatives were scored pooled.** It says so of all the alternatives
   (research/paper/draft-v1-2026-09-29.md@1a81f30:370). 019's template-family test was also scored on the new frames
   alone, at 0.080 (019:145–148).
8. **§5's 017 row does not label its metric.** The `ΔT` values there are R² values
   (research/paper/draft-v1-2026-09-29.md@1a81f30:357), and should say so.
9. **§5's 019 row gives no numbers.** It says only *predicted* (research/paper/draft-v1-2026-09-29.md@1a81f30:359);
   the record has 0.090 and 0.144 (019:120–123, 129–133).
10. **Two §4 citations are imprecise:**
    - *not through 018's locked subset* rests on 018:16 alone (research/paper/draft-v1-2026-09-29.md@1a81f30:339).
      018:16 says that the full rung is 017's Level 0; that the completed program runs 017's Level-0 chain is in
      022's design (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:74). Cite
      both;
    - `these` is traced to six table rows, 010:78–83 (research/paper/draft-v1-2026-09-29.md@1a81f30:305–308); its
      own row is 010:80.
11. **§4's chain omits two facts:**
    - the block-0 MLP runs in parallel with block-0 attention (012:9);
    - end to end, the head sits inside block 3's readout.
12. **§6 quotes 020 too widely.** The phrase *calibration record, never evidence for a label* is applied to both of
    020's values (research/paper/draft-v1-2026-09-29.md@1a81f30:408). It covers only the end-to-end table (020:110).
13. **Appendix A is not a verbatim copy.** It differs from the source map's overview in the places listed under F7,
    though its note says it copies the overview (research/paper/draft-v1-2026-09-29.md@1a81f30:839).
14. **C002's causal gates are called open.** The draft does so (research/paper/draft-v1-2026-09-29.md@1a81f30:285,
    541, 657, 667), where C002's own word is *not established*
    (research/claims/C002-count-cued-noun-number-circuit.md@454327d:101).
15. **Appendix A's rows for 008 and 021 paraphrase their subject loosely:**
    - 008 concerns where the number signal of `this`-like cues is suppressed (008:1);
    - 021's row, *the corrected readout*, takes 021's title (021:1), where *corrected* means 020's corrected block-5
      implementation (docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md:61).
      It can be read as the wiring fix, but 021 tested 020's program unchanged (021:22–23).

## Record notes

These concern committed records beyond the draft. None is changed here, and each would be a separate decision.
- **C003's wording repeats correction 2.** It speaks of new cues and nouns for 021's Y1 pass
  (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:379), while its own definitions give Y1 exposed
  nouns (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:92).
- **The literature review repeats correction 1.** Its candidate 3 speaks of singular-selecting determiners
  (research/paper/literature-review/review-2026-09-28.md@1a81f30:211–213). The phrase came from the frozen protocol's
  candidate statement (research/paper/literature-review/protocol-2026-09-28.md@1a81f30:23–25), so the review stays as
  committed.
- **Other records repeat corrections 2, 5 and 14.**
  - The synthesis and the readiness assessment speak of new cues and nouns for 021's Y1 pass
    (research/synthesis/synthesis-005-025.md@3d87d3c:268; research/synthesis/paper-readiness-2026-09-28.md@139755d:56).
  - C003 and the readiness assessment call 011's read weight-only without the qualifier, and Appendix B's F3 entry
    follows the readiness wording (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:60;
    research/synthesis/paper-readiness-2026-09-28.md@139755d:187, 279).
  - The readiness assessment calls C002's Level 3 open
    (research/synthesis/paper-readiness-2026-09-28.md@139755d:299).
- **The readiness assessment's status for 018's Y2 closure.** It tags 018's Y2 closure of 0.84 as descriptive
  (research/synthesis/paper-readiness-2026-09-28.md@139755d:284). It is the criterion statistic under an unestablished
  label (research/synthesis/source-map-005-025.md@b722e9c:192, 196–197).

## Limits of this audit

- **Precision.** Values were checked at the precision shown.
- **JSON values** were read by key path, which the checker cannot verify.
- **The confirmation files** of 006–019 hold frozen inputs, not results. The statistics rest on the READMEs and the
  final reports. Per-pair extracts of earlier results are committed under later experiments' `inherited/`
  directories; F3's Panel A2 uses one, and F4's per-frame `q_T` and `f_E` come from another.
- **Figures that need more.** A figure that needs data beyond what is listed here needs a new audit entry before it
  is drawn.

## Reviews

Two independent agents reviewed this audit, read-only. Each loaded no model, read nothing under `outputs/`, and
changed nothing in the repository.
- **The first review: FAIL, one blocker.**
  - **The blocker.** Panel C of F6 had audited three 021 values that the draft's §6 reports, but the figure plan
    follows the readiness assessment, which names 021's descriptive flattened R² values instead. The panel's redesign
    verdict and the matching draft correction therefore had no cause. *Done:* Panel C audits the planned values and
    is supported with limits; correction 3 now records that the draft never names them.
  - **Should-fix,** twelve, all done:
    - a contradiction about the status of 009's token means;
    - the source of `either`'s positive rest, which is 010's report, not 009's;
    - per-frame values committed for F4;
    - the committed per-pair extracts under `inherited/`;
    - an unsupported statement about confidence intervals;
    - two citation ranges;
    - the gap as a recorded difference;
    - the verdict name for Panel D;
    - the overview's 023-d and screen rows;
    - the rows of 006 and 020, which carry two statuses;
    - the reasoning of correction 10;
    - a missing draft line in correction 14.
  - **Notes,** fourteen, all done.
- **The re-review of the second revision: PASS WITH NOTES, no blocker.**
  - **Should-fix,** one, done: correction 5 had said that 011's design never omits the qualifier, but its own title
    does.
  - **Notes,** thirteen, all done:
    - the citation pin;
    - Panel C's populations and shared frames;
    - a third near-substitute file for 009's pairs;
    - two citations;
    - Panel D marked as not in the plan;
    - 007's criterion-level status as an annotation;
    - one more Appendix A difference;
    - further records that repeat corrections 2, 5 and 14;
    - F2's source column;
    - correction 15's reading of 021's title;
    - F4's extract named in the limits;
    - the gap's heading.

After these changes, the checker passes on this audit: 2 quotes and 154 citation groups.

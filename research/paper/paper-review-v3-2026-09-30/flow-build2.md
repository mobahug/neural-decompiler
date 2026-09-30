# Draft v3, build 2: re-check of the flow review, and length cuts

This re-check was read-only. Nothing in the repository was changed, and nothing under `outputs/` was read.

What was checked:
- `research/paper/draft-v3-2026-09-30.md`, called build 2, which is the current file;
- `review-v3/build1-to-build2.unwrapped.diff`;
- the regenerated F2 and F3 labels;
- the claims review's finding 1 and its note 12, in `paperreview/claims-v3.md`.

How to read the references:
- **Line numbers "l."** refer to the wrapped build-2 file.
- **Exact texts** are quoted from `review-v3/build2-unwrapped.md`. That file equals the current draft once whitespace is normalized and the Box's `>` continuation markers are removed.
- **A byte-exact copy** of every cut is in `paperreview/cuts/cuts-build2.json`. It lists the cuts in rank order, each with a `rank` field, and includes the optional cut 25. Applied in that order to `build2-unwrapped.md`, every string matches exactly once, or four times for rank 13, and the cuts reduce the body as stated.
- **Word counts** use my earlier method. It counts rendered words from the abstract to the conclusion, without headings, HTML comments, image lines, table pipes, list or quote markers, or the AI statement.

## Task 1: verdict

**18 of the 19 findings are resolved. Finding 9, length, is the open one, and it is task 2.** I accept all of your stated choices. Six new flow problems are all MINOR, and each is fixed by a move or a split with no change in length.

| # | finding | status in build 2 | where |
|---|---|---|---|
| 1 | MAJOR: reference-state vocabulary not defined in the body | resolved | §2 defines reference prompt, reference state, operating point and template bases (l.145–147); §1 glosses the first use (l.66–67) |
| 2 | MAJOR: scope qualifiers and negatives lost | resolved | contribution 2 (l.98–102) uses the claims review's wording; §1 ¶2 now says "were tested under" (l.67), accepted; §10 carries C002's scope, the prediction scope, "before the target prompts were executed" and the routes sentence (l.633–646); §4.1 has the P3 retention counts, the lexicon's 0.699 and the replacement distributions (l.232–241); §6 has 020's six of twelve floors and "calibration record" (l.474–476) and 021's caveat (l.484) |
| 3 | SHOULD: "three of 023's four conditions" | resolved | l.435–437 |
| 4 | SHOULD: terms defined only in the glossary | resolved | E-only chain (l.437–438); no-harm guard (l.403–405); decoded read and Π (l.398–400); P1 (l.285); development, calibration and reserve data (l.139–141); channels, as you chose (l.274–276, plus the glossary); "preregistered reading" (l.466); Levels numbered (l.186–188, 516); licensed (l.324); informative frames (l.261–262); `LN₂` (l.159); E–N dropped (l.495); "weight-defined" (l.426) |
| 5 | SHOULD: figures not used | resolved | Figure 3's A1 and A2 are cited (l.286); Figure 5's calibration range is used (l.462–465); C1–C4 appear in §6 (l.488–491) and `R²_C` in Figure 6's caption (l.511); Figure 1's caption defines coordinated frames and the dashed box (l.73–79); §4.2 points to Figure 1 (l.248–249) |
| 6 | SHOULD: the old Y1/Y2 sense | resolved | F3's panel titles and inset and F2's row labels were regenerated; Figure 3's caption (l.319–320); Table 3's 007 row (l.394); Table 4 now matches Figure 5, with Y1 and Y2 defined in its caption (l.444) |
| 7 | SHOULD: handoffs | resolved | §5's summary and Table 2 sentence (l.360–364); §6's Table 2 sentence (l.434–435); §3 points to §6 (l.176–177); §1 points to Table 1 (l.64) |
| 8 | SHOULD: repetition | resolved | ledger restatement cut from the text, with the audit-required caption kept (l.254–255); Figure 5's pass and fail list kept because note 12 requires the no-aggregate sentence there, accepted; Méloux 2026 gone from §9; "Beyond the methods of §1" (l.596); the digest sentence gone from §8; §4.2's no-harm clause now a pointer (l.279–280); Figure 6's caption clauses cut; §10's formula cut |
| 9 | SHOULD: length | **open** | 7,305 words by my count; your 7,319 carries the same +14 offset as build 1 (6,637 against 6,651); up 668 on build 1 |
| 10 | SHOULD: submission build | resolved, as a plan | the header lists stripping the comments and anonymizing the link, and flags "the author" (l.6–12) |
| 11 | MINOR: "from experiment 013 on" | resolved | l.26–27 |
| 12 | MINOR: (E)(T)(R) labels | resolved | removed |
| 13 | MINOR: §4.5 lineage | resolved | function first, lineage second (l.351–355) |
| 14 | MINOR: Box 1 | resolved | "prediction errors" (l.313); the axis's origin (l.299–302); a displayed equation was optional and not taken |
| 15 | MINOR: §4.3's title | resolved | l.294 |
| 16 | MINOR: [locked prediction] unused | resolved | l.311 |
| 17 | MINOR: Table 3's statuses; Table 1's 009 | resolved | status column bracketed; Table 1 reads "008, 010" (l.208) |
| 18 | MINOR: pointers, glossary, tag load | resolved | §2 points to Table A2 (l.148); the `m_T`/`m_R` entry removed; §6 has a default (l.433–434) |
| 19 | MINOR: two names for 023's conditions; `g` in §1 | resolved | Table 4 matches Figure 5; "`g` (§6)" (l.115–116) |

**New flow problems.** All are MINOR, and none changes the length.
- **N1. A line-initial numeral in Figure 5's caption (l.458).**
  - The rewrap put "4." at the start of l.458 ("…as in Table⏎4. The passes are…") and indented l.459–460 as list continuation. l.599 has "2025)" in the same position.
  - Both still render as paragraph text: CommonMark lets only a list starting at 1 interrupt a paragraph, and pandoc wants a blank line before a list.
  - They are fragile under unwrap and rewrap: the unwrapped copy already splits the caption into two lines.
  - Fix: write "as in Table 4;" or use `Table&nbsp;4`.
- **N2. §8's "007's Y4" (l.557–558).** It is the last undecoded outcome-family label left in the body. Fix: "a descriptive baseline in 007", or name what 007:95–97 compared.
- **N3. §6's headline comes third (l.432–437).** It follows the default and the Table 2 sentence. Fix: move "In Table 2's terms, …" after the headline sentence.
- **N4. §6's path paragraph (l.472–491) runs to about 330 words.** It is the longest paragraph, with three quotations and six required negatives. Fix: split it at "022, correctly wired, then localized…".
- **N5. §1 ¶2's third sentence (l.63–67) runs to about 65 words.** Fix: split after "(Figure 1)" with "Each program is written from three inputs: …".
- **N6. §2's Data paragraph (l.138–148) mixes four topics.** Cuts 12 and 15 below remove two of them. Optionally, give the reference-state sentence its own paragraph.

## Task 2: length

**Totals.**
- Cuts 1–24 below remove **499** words. The body goes from 7,305 to 6,806 by my count, about 6,820 by yours.
- The optional cut 25 raises the total to **586** words. The body goes to 6,719, about 6,733 by yours.

**Priority.**
- Rank 1 moves Figure 6, whose content the body already carries.
- Ranks 2–18 are pure repetition of text stated elsewhere in the body, or in the image's own legend.
- Ranks 19–22 move technical detail to Appendix A. The companion edit at the end adds one paragraph there.
- Ranks 23–24 are editorial.
- Nothing in the list touches:
  - the scoped core claim (claims finding 1);
  - any item of note 12;
  - the definitions from findings 1 and 4;
  - Box 1's result or Table 2;
  - Table 3's negative rows;
  - §6's path paragraph;
  - §8's baselines, self-repair, off-distribution or validation bullets.

**Figure 6. Yes, move it to Appendix C (cut 1).**
- At this length, the TMLR build will likely run past 12 pages.
- Figure 6 is the densest figure. Its text prints at about 3 pt at text width, per the figures README, so it would need a print redraw anyway.
- The body carries its content:
  - Table 3 carries the E-patch lane;
  - Table 4 and §6 carry the real-swap lane;
  - §7's boundary sentence carries the causal band.
- It is the last body figure, so moving it needs no renumbering.
- It saves 109 words and about a page.

Figure 4, the ladder, is the next candidate if the pages are still over. That is cut 25, which is optional.

Strings below show a newline as ⏎. Where a current text begins with a space, the space is part of the match.

### 1. Move Figure 6 to Appendix C as Figure C3 — saves 109 words (running total 109)
*Where:* §7, l.503–513; Appendix C after l.892 (wrapped); unwrapped line(s) 190, 376.
*Current text (edit 1):*
```text
Figure 6 summarizes the chain with the recorded status of each result it draws, and its bottom band gives the causal status.⏎
⏎
![Figure 6: the decoded chain with the recorded status of each result](figures/F1-chain.png)⏎
⏎
**Figure 6. The decoded chain, with the recorded status of each result it draws.** The E-patch lane tests the programs stage by stage, the real-swap lane end to end. Each badge carries one recorded status. Y1 is new cues in exposed frames and Y2 new cues in new frames, conditional; C1–C4 are 022's conditions (§6), c-f and co 023's cue-final and coordinated conditions, 021's Y3 its new nouns, and `R²_C` the ceiling's R²; *rival* marks a preregistered alternative, rejected or found insufficient on both sets pooled, whose new-frame share is conditional. The dashed frame marks `L03.H04` inside block 3 end to end, and a dash a cell with no chip.
```
*Replacement (edit 1):*
```text
Figure C3 summarizes the chain with the recorded status of each result it draws, and its bottom band gives the causal status.
```
*Current text (edit 2):*
```text
025 bears on C003 only<!-- (research/claims/C002-count-cued-noun-number-circuit.md@454327d:66) -->.
```
*Replacement (edit 2):*
```text
025 bears on C003 only<!-- (research/claims/C002-count-cued-noun-number-circuit.md@454327d:66) -->.⏎
⏎
![Figure C3: the decoded chain with the recorded status of each result](figures/F1-chain.png)⏎
⏎
**Figure C3. The decoded chain, with the recorded status of each result it draws.** The E-patch lane tests the programs stage by stage, the real-swap lane end to end. Each badge carries one recorded status. Y1 is new cues in exposed frames and Y2 new cues in new frames, conditional; C1–C4 are 022's conditions (§6), c-f and co 023's cue-final and coordinated conditions, 021's Y3 its new nouns, and `R²_C` the ceiling's R²; *rival* marks a preregistered alternative, rejected or found insufficient on both sets pooled, whose new-frame share is conditional. The dashed frame marks `L03.H04` inside block 3 end to end, and a dash a cell with no chip.
```
*Why nothing required is lost:* The caption moves verbatim to Appendix C as Figure C3, including claims note 12's *each result it draws* and the rival's *new-frame share is conditional*. §7 keeps its boundary sentence and points to C3. No renumbering is needed, since Figure 6 is the last body figure.

### 2. §1: delete the roadmap paragraph — saves 33 words (running total 142)
*Where:* §1, l.121–122 (wrapped); unwrapped line(s) 32.
*Current text:*
```text
§2 defines the task, §3 the protocol, §4 the localization and the decoded chain, §5 and §6 the prospective tests, §7 the causal tests, and §8 and §9 the limitations and related work.⏎
⏎

```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* It repeats the section headings; no qualifier is involved.

### 3. §4.4: shorten the opening sentences — saves 24 words (running total 166)
*Where:* §4.4, l.329–330 (wrapped); unwrapped line(s) 108.
*Current text:*
```text
The programs mix two kinds of step (Table 2). Some evaluate a layer as the model does, at the frame's measured operating point; others replace a computation by a reduced or locked rule.
```
*Replacement:*
```text
The programs mix exact and reduced steps (Table 2).
```
*Why nothing required is lost:* Table 2's column heads define the two kinds of step. §4.4's closing paragraph keeps the caveat and the block-0 evidence.

### 4. §5: delete the opening's last sentence — saves 22 words (running total 188)
*Where:* §5, l.365–366 (wrapped); unwrapped line(s) 128.
*Current text:*
```text
 The head read-outs are given the measured input; 011's encoding read and 012 are not frame-conditional, since 012 used exposed base states.
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* The two qualifiers stand in Table 3's status column, `[given the measured input]` and `[exposed base states]`. *Not frame-conditional* for 011 and 012 follows from the default's *from 013 on* in the sentence before, which stays (claims note 13).

### 5. §10: delete two duplicate causal statements — saves 19 words (running total 207)
*Where:* §10, l.635 and l.643 (wrapped); unwrapped line(s) 233, 235.
*Current text (edit 1):*
```text
C002 is registered at `LOCALIZED`, and its causal gates are not established.
```
*Replacement (edit 1):*
```text
C002 is registered at `LOCALIZED`.
```
*Current text (edit 2):*
```text
Whether the programs' variables are the model's causal variables is not established. 
```
*Replacement (edit 2):*
```text
(delete)
```
*Why nothing required is lost:* ¶3 keeps both the routes sentence (the v0 re-review fix) and C003's boundary sentence. *Its causal gates are not established* is said again two sentences later, and *Whether the programs' variables … is not established* restates C003's sentence.

### 6. Figure 1 caption: delete the readout-position clause — saves 16 words (running total 223)
*Where:* Figure 1 caption, l.77–78 (wrapped); unwrapped line(s) 19.
*Current text:*
```text
; the readout runs blocks 3 and 4 at `p_c` and `p_t`, and block 5 at `p_t`<!-- (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:106–115) -->
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* §4.2's *The readout* paragraph states the same positions under the same trace (020 design:106–115). The caption keeps the coordinated-frame definition, the dashed box and the note on `L03.H04` inside block 3 (flow finding 5).

### 7. §2: paraphrase 022's quotation on real swaps — saves 15 words (running total 238)
*Where:* §2, l.159–164 (wrapped); unwrapped line(s) 42.
*Current text:*
```text
A **real cue swap** replaces the cue token itself; it was used from 020 to 024. As 022's design notes, it “also changes three things the chain does not carry: the embedding itself at `p_c`; block 0's attention output at `p_c`; and, in coordinated frames, block 0's attention output at `p_t`.”<!-- (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:85–86) -->
```
*Replacement:*
```text
A **real cue swap** replaces the cue token itself, which also changes the embedding at `p_c` and block 0's attention output at `p_c` and, in coordinated frames, at `p_t`<!-- (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:85–86) -->; it was used from 020 to 024.
```
*Why nothing required is lost:* The three changes and their trace stay, as a paraphrase: the embedding at `p_c`, and block 0's attention output at `p_c` and, in coordinated frames, at `p_t`. §4.2's *Block 0* paragraph restates them under its own trace (023:16–20), and §6 defines the E-only chain as the program fed the encoding change alone. The quote count drops by one.

### 8. Figure 4 caption: delete the marker sentence — saves 14 words (running total 252)
*Where:* Figure 4 caption, l.412 (wrapped); unwrapped line(s) 158.
*Current text:*
```text
 Circles are new cues in exposed frames, and diamonds new cues in new frames.
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* The image's legend carries the circles and diamonds. All the caption's statuses stay.

### 9. §4.2: shorten the opening sentence — saves 14 words (running total 266)
*Where:* §4.2, l.248–249 (wrapped); unwrapped line(s) 80.
*Current text:*
```text
The later experiments wrote programs for the computation in and around the three components, stage by stage in the order of Figure 1.
```
*Replacement:*
```text
The stages below follow the order of Figure 1.
```
*Why nothing required is lost:* §1 ¶2 says that the programs were written for the computation in and around the components. The section default (claims note 13) and the causal-boundary clause (claims *traps that held*) stay.

### 10. §4.2: shorten the P1 sentence — saves 13 words (running total 279)
*Where:* §4.2, l.284–285 (wrapped); unwrapped line(s) 92.
*Current text:*
```text
That read-out, a linear `W_V W_O` read of the measured residual arriving at the head, is the **P1 read**
```
*Replacement:*
```text
That read-out is the **P1 read**
```
*Why nothing required is lost:* The quotation just before defines the read-out, the glossary gives `W_V W_O`, and the P1 read is still named at first use (flow finding 4).

### 11. §8: fold *Small fresh sets* into the first bullet — saves 13 words (running total 292)
*Where:* §8, l.546 and l.549 (wrapped); unwrapped line(s) 206, 207.
*Current text (edit 1):*
```text
- **One checkpoint** of one model, three frame families and single-token regular nouns.
```
*Replacement (edit 1):*
```text
- **One checkpoint** of one model, three frame families, single-token regular nouns and small fresh sets (§2).
```
*Current text (edit 2):*
```text
- **Small fresh sets:** 23 to 40 new cues and 6 to 18 new frames per experiment (§2).⏎

```
*Replacement (edit 2):*
```text
(delete)
```
*Why nothing required is lost:* The counts stay in §2 under their C003 trace, so the fresh-set counts are still cited (v0 re-review note). The limitation stays named in §8.

### 12. §2: delete the floors sentence — saves 12 words (running total 304)
*Where:* §2, l.147–148 (wrapped); unwrapped line(s) 38.
*Current text:*
```text
 In 021–024, floors were calibrated on draws from exposed data (Table A2).
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* Table A2's 021–024 row says it, §6 gives 023's calibration, and §8 keeps the floors caveat and 024's permutation null.

### 13. Table 3: drop *both sets pooled:* from the four rival cells (replace all four) — saves 12 words (running total 316)
*Where:* Table 3, l.388–391 (wrapped); unwrapped line(s) 145.
*Current text:*
```text
| both sets pooled: 
```
*Replacement:*
```text
| 
```
*Why nothing required is lost:* Table 3's caption says the rival rows were scored on both sets pooled (claims note 14's *note*), and each row's status cell reads `[pooled]`.

### 14. Figure 3 caption: delete the weight-only clause — saves 10 words (running total 326)
*Where:* Figure 3 caption, l.323–325 (wrapped); unwrapped line(s) 104.
*Current text:*
```text
; the read is weight-only apart from the frozen axis<!-- (010:44–45) -->, and `an`, licensed only in the four cue-final frames, is a mean over them
```
*Replacement:*
```text
; `an`, licensed only in the four cue-final frames, is a mean over them
```
*Why nothing required is lost:* Box 1 states it under the same trace (010:44–45), and panel B's inset states it too.

### 15. §2: merge the confirmation-set sentence — saves 9 words (running total 335)
*Where:* §2, l.141–142 (wrapped); unwrapped line(s) 38.
*Current text:*
```text
 Each confirmation set was frozen before any model output on it. Its **new** (fresh) items had never been executed
```
*Replacement:*
```text
 A confirmation set's **new** (fresh) items had never been executed
```
*Why nothing required is lost:* §3 says a confirmation set was frozen before any model output on it (l.168).

### 16. §1 ¶3: replace the exploration clause with a pointer — saves 9 words (running total 344)
*Where:* §1, l.84–86 (wrapped); unwrapped line(s) 21.
*Current text:*
```text
, since the stage predictions concern the intervention used in exploration<!-- (research/claims/C003-frame-conditioned-decoded-program.md@eeff9ff:295–297) -->;
```
*Replacement:*
```text
 (§5);
```
*Why nothing required is lost:* §5's *What these predictions are not* makes the statement under the same trace. The attribute answer keeps *only partly*.

### 17. §3: shorten the erratum sentence — saves 9 words (running total 353)
*Where:* §3, l.175–176 (wrapped); unwrapped line(s) 46.
*Current text:*
```text
was recorded as an erratum with the frozen results left unchanged: 021's labels stand as a test of that program, and later programs used the correct wiring
```
*Replacement:*
```text
was recorded as an erratum with the frozen results left unchanged, and later programs used the correct wiring
```
*Why nothing required is lost:* *Frozen results left unchanged* (claims note 14) says the same thing, and §6 says 021 tested that program unchanged and that the erratum qualifies its labels. *Later programs used the correct wiring* stays.

### 18. §8: shorten *New frames are conditional* — saves 8 words (running total 361)
*Where:* §8, l.550 (wrapped); unwrapped line(s) 208.
*Current text:*
```text
- **New frames are conditional.** From 013 on, new-frame results depend on the frames' measured reference states. As 016's design puts it,
```
*Replacement:*
```text
- **New frames are conditional** from 013 on (§3). As 016's design puts it,
```
*Why nothing required is lost:* *From 013 on* and the quotation from 016 stay, and §3 defines *conditional*.

### 19. §4.1: move 007's replication to Appendix A — saves 29 words (running total 390)
*Where:* §4.1, l.225–229 (wrapped); unwrapped line(s) 74.
*Current text:*
```text
 007 recorded the same magnitudes, without outcome force, on fresh nouns and on fresh frames beyond the claim's declared frames: recovery 0.981 and 0.983, isolation 0.781 and 0.773<!-- (experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md:109, 115; research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:53) --> [descriptive].
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* It moves to Appendix A with its [descriptive] status and its scope (fresh frames beyond C002's declared frames), as the v2 claims revision required. It is descriptive evidence without outcome force; no qualifier on C002 is lost.

### 20. Box 1: move the six frame names to Appendix A — saves 22 words (running total 412)
*Where:* Box 1, l.308–310 (wrapped); unwrapped line(s) 100.
*Current text:*
```text
, in six new frames: `The cabinet stores`, `The vendor sells`, `The survey covers` and `The monitor tracks`, cue-final, and `Leo and Maya labeled … round` and `Priya and Jonas bundled … soft`, coordinated<!-- (011:20–22) -->.
```
*Replacement:*
```text
, in six new frames, four cue-final and two coordinated (Appendix A)<!-- (011:20–22) -->.
```
*Why nothing required is lost:* The Box keeps the scope, *six new frames, four cue-final and two coordinated*, and the frames are still named, in Appendix A (claims finding 5's request). The claims reviewer should agree to the move.

### 21. §7: move the exploratory manipulations to Appendix A — saves 34 words (running total 446)
*Where:* §7, l.526–529 (wrapped); unwrapped line(s) 198.
*Current text:*
```text
Decoded parts were manipulated outside any outcome. 008's axis-only E-patches drove 0.50–0.64 of the plural cue's effect for `this`, `a` and `another`, and the whole encoding about none<!-- (008:57–59) --> [exploratory]. At the head's output, 009 recorded mostly linear cancellation between the parts, with the non-additivity growing downstream<!-- (009:110–115) --> [exploratory].
```
*Replacement:*
```text
Decoded parts were also manipulated outside any outcome, in exploratory runs (Appendix A).
```
*Why nothing required is lost:* The two exploratory results move to Appendix A with their tags and traces. §7 keeps a pointer, and C003's *no preregistered, outcome-bearing necessity test…* stays.

### 22. §4.5: move the upstream lineage to Appendix A — saves 24 words (running total 470)
*Where:* §4.5, l.354–355 (wrapped); unwrapped line(s) 124.
*Current text:*
```text
 Its upstream chain is 022's, built from the 011, 012 and 017 locks and used through the frozen 012, 015, 016 and 017 programs<!-- (020:30–31; 022:43–45) -->.
```
*Replacement:*
```text
(delete)
```
*Why nothing required is lost:* §4.5 keeps the substantive v0-blocker-3 correction: block 2's MLP enters in full, not through 018's subset. The input lineage (022:43–45) moves to Appendix A intact. Take this cut last, since a verifier may look for the lineage in §4.5.

### 23. §9: delete the RASP clause and its two references — saves 21 words (running total 491)
*Where:* §9, l.607–608 (and References) (wrapped); unwrapped line(s) 225.
*Current text:*
```text
; RASP expresses transformer computation as programs, and compiled RASP programs give ground truth (Weiss et al., 2021; Lindner et al., 2023)
```
*Replacement:*
```text
(delete)
```
*Also delete the two reference entries, each a whole line in the References:*
```text
- Lindner, Kramár, Farquhar, Rahtz, McGrath and Mikulik (2023). *Tracr: Compiled Transformers as a Laboratory for Interpretability.* NeurIPS 2023. [arXiv 2301.05062](https://arxiv.org/abs/2301.05062)
```
```text
- Weiss, Goldberg and Yahav (2021). *Thinking Like Transformers.* ICML 2021. [arXiv 2106.06981](https://arxiv.org/abs/2106.06981)
```
*Why nothing required is lost:* No review required it. This is an editorial cut: delete the two reference entries (Lindner et al., 2023; Weiss et al., 2021), which are cited nowhere else.

### 24. §1: shorten the contributions' lead — saves 8 words (running total 499)
*Where:* §1, l.89 (wrapped); unwrapped line(s) 23.
*Current text:*
```text
A literature review, run under a protocol frozen before any search, found
```
*Replacement:*
```text
A literature review found
```
*Why nothing required is lost:* It is process detail. No review required it in the body; the literature review records its own protocol.

### Companion edit for 19–22: Appendix A's new paragraph (0 words from the body)
*Where:* before `## Appendix B: Glossary`, l.845; unwrapped line 352.
*Current text:*
```text
⏎
⏎
## Appendix B: Glossary
```
*Replacement:*
```text
⏎
⏎
**Further details.** 007 recorded the localization's magnitudes, without outcome force, on fresh nouns and on fresh frames beyond C002's declared frames: recovery 0.981 and 0.983, isolation 0.781 and 0.773<!-- (experiments/007-supervised-cue-subspace/evidence/final-report-2026-09-18.md:109, 115; research/claims/C002-count-cued-noun-number-circuit.md@eeff9ff:53) --> [descriptive]. 011's six new frames were `The cabinet stores`, `The vendor sells`, `The survey covers` and `The monitor tracks`, cue-final, and `Leo and Maya labeled … round` and `Priya and Jonas bundled … soft`, coordinated<!-- (011:20–22) -->. Decoded parts were manipulated outside any outcome. 008's axis-only E-patches drove 0.50–0.64 of the plural cue's effect for `this`, `a` and `another`, and the whole encoding about none<!-- (008:57–59) --> [exploratory]. At the head's output, 009 recorded mostly linear cancellation between the parts, with the non-additivity growing downstream<!-- (009:110–115) --> [exploratory]. The completed program's upstream chain is 022's, built from the 011, 012 and 017 locks and used through the frozen 012, 015, 016 and 017 programs<!-- (020:30–31; 022:43–45) -->.⏎
⏎
## Appendix B: Glossary
```
*Why:* This carries cuts 19, 20, 21 and 22 into Appendix A, each with its trace and status. Take only the sentences for the cuts you accept.

### 25 (optional). Move Figure 4, the closure ladder, to Appendix C as Figure C4 — saves 87 words with cut 8 taken, 101 without (running total 586)
*Where:* §5, l.397 and l.408–416; §6, l.437, l.454, l.456 and l.465 (wrapped); unwrapped lines 154–158, 166, 177, 179 and 182.
*Current text (edit 1):*
```text
Channel D's result is a closure ladder (Figure 4):
```
*Replacement (edit 1):*
```text
Channel D's result is a closure ladder (Figure C4):
```
*Current text (edit 2):*
```text
![Figure 4: channel D's closure ladder](figures/F5-closure-ladder.png)⏎
⏎
**Figure 4. Channel D's closure ladder** (018). κ for nested neuron subsets and for the controls: three seeded random 256-neuron subsets and the 256 lowest-ranked neurons<!-- (018:15–16) -->. The oracle, each frame's own subset, is a descriptive comparator<!-- (018:55) -->. Circles are new cues in exposed frames, and diamonds new cues in new frames. The locked rung is plotted at its decision values, and the other rungs, the controls and the oracle, descriptive, at two decimals. The star is the single-neuron statistic over both sets pooled and pair-weighted, rejected [prospective; pooled]; it is not Y1's one-neuron rung. No curve is drawn, since no monotonicity was predeclared<!-- (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:222) -->.⏎
⏎

```
*Replacement (edit 2):*
```text
(delete)
```
If cut 8 was taken, this block no longer contains the *Circles…* sentence. Match it without that sentence.
*Current text (edit 3):*
```text
(Table 4, Figure 5)
```
*Replacement (edit 3):*
```text
(Table 4, Figure 4)
```
*Current text (edit 4):*
```text
![Figure 5: the recovered share per condition]
```
*Replacement (edit 4):*
```text
![Figure 4: the recovered share per condition]
```
*Current text (edit 5):*
```text
**Figure 5. The recovered share per condition**
```
*Replacement (edit 5):*
```text
**Figure 4. The recovered share per condition**
```
*Current text (edit 6):*
```text
(Figure 5). Under the preregistered reading
```
*Replacement (edit 6):*
```text
(Figure 4). Under the preregistered reading
```
*Current text (edit 7), the end of Figure C3's caption after cut 1:*
```text
a dash a cell with no chip.
```
*Replacement (edit 7):*
```text
a dash a cell with no chip.⏎
⏎
![Figure C4: channel D's closure ladder](figures/F5-closure-ladder.png)⏎
⏎
**Figure C4. Channel D's closure ladder** (018). κ for nested neuron subsets and for the controls: three seeded random 256-neuron subsets and the 256 lowest-ranked neurons<!-- (018:15–16) -->. The oracle, each frame's own subset, is a descriptive comparator<!-- (018:55) -->. Circles are new cues in exposed frames, and diamonds new cues in new frames. The locked rung is plotted at its decision values, and the other rungs, the controls and the oracle, descriptive, at two decimals. The star is the single-neuron statistic over both sets pooled and pair-weighted, rejected [prospective; pooled]; it is not Y1's one-neuron rung. No curve is drawn, since no monotonicity was predeclared<!-- (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:222) -->.
```
If cut 8 was taken, leave the *Circles…* sentence out of the moved caption as well.

*Why nothing required is lost:* The caption moves intact to Appendix C: its statuses, *it is not Y1's one-neuron rung*, and *no curve … predeclared*. §5's ladder paragraph keeps the definitions of rung, κ, `c_L` and Π, the locked rung's 0.863 and 0.843 with the no-harm guard, and neuron 1987's 0.219. Table 3 keeps every status. Figure 5 becomes Figure 4. Take this cut only if the full list is needed: this is the body's only picture of the channel-D result.

### Further options, not recommended

- **Move Figure 2, the ledger, to Appendix C.** It saves about 70 words net. That allows for a 15-word clause restoring `these` and `either` in §4.2, with their traces, since the caption would leave the body. It also means renumbering Figures 3–5, or 3–4 after cut 25. It is the only figure behind contribution 3's first finding.
- **Delete §1 ¶3, the attribute answer.** It saves about 45 words after cut 16. The abstract carries the answer, and §4.5, §5 and §7 carry its parts. But it is an element of the outline the reviewer adopted.

Neither is needed to reach about 600.

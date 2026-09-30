# Draft v3: flow, conformance and redundancy review

A read-only review of `research/paper/draft-v3-2026-09-30.md` (uncommitted). It is measured against my v2 outline and
the reviewer's decisions of 2026-09-30:
- the venue is TMLR;
- the target is 6,000–6,500 words;
- the outline is adopted in full;
- F2, F7 and F8 go to the appendix, and only F6's panel A stays in the body;
- the paper is written in prose;
- related work is shortened;
- there is no conference-length cut;
- the AI statement becomes a footnote.

Line numbers are v3's; "v2" means the committed v2, revision 1 (`2bc13a9`).

Word counts are of rendered text. HTML comments, image links, table pipes, list and quote markers, and the AI statement
are removed. I checked all eight body and appendix images: they carry no titles or notes, apart from the panel titles
and annotations inside F3. Nothing in the repository was changed, and nothing under `outputs/` was read.

## Verdict

**The restructure conforms. It reads as an argument, and no further restructuring is needed.** The introduction now
motivates the work and says what was found. Box 1 and Table 2 make the core contribution and its caveat concrete. Each
result has one home. Every figure and table is placed where its claim is made. The handoffs between sections work, and
the text is 13 % bullets, down from 64 %.

**Two MAJOR findings, both text fixes of about 250 words in all:**
1. The reference-state vocabulary that carries the central caveat is not defined in the body.
2. The compression removed scope qualifiers from the core claim (contribution 2 and §10), and removed negatives that
   earlier reviews required.

Eight SHOULD findings and nine MINOR findings follow.

**Length.** I count 6,637 words, or 6,701 with headings, which agrees with your 6,735. It is about 2–4 % over 6,500.
The MAJOR restorations add about 250 words and the safe cuts remove about 200, so the result stays near 6,700. That is
acceptable as "about 6,500". Check the page count in the template before cutting more text.

## 1. Conformance, section by section

| outline item (v2 report) | v3 | lines | verdict |
|---|---|---|---|
| Abstract: one paragraph, 310–350 words | 322 words, one paragraph. The 021 sentence was cut, the reviewer's call; §6 keeps 021. | 11–29 | ✓ |
| AI statement as a first-page footnote | Placed after the abstract and marked for the footnote, in the reviewer's wording | 31–38 | ✓ |
| §1, 800: why; what, with the schematic; the attribute answer; contributions with hedges and four findings; roadmap | 895 words, all present | 42–111 | ✓, but contribution 2 lost its scope (finding 2) |
| §2, 450: task, frames with cue-final and coordinated named, cues, `Δc`, data splits and sizes, architecture, interventions | 432 words, all present; lexical classes labelled; pairs defined | 115–148 | ✓, but reference prompt and state, operating point, template bases, and development, reserve and calibration data are undefined (findings 1, 4) |
| §3, 550: steps, locks, exceptions with Table A2, legend with defaults, sequence table | 471 words, all present | 152–194 | ✓; no pointer to §6's "protocol at work" (finding 7) |
| §4, 950: localization; stages with F4 at stage 1; worked example beside F3B; exact versus reduced; completed program | 1,342 words, all present | 196–322 | ✓. It is 390 words over, all from the adopted Box 1 and Table 2 plus two captions. §4.1 lost two failed tests (finding 2). |
| §5, 800: summary first; extended table with a status column; F3 and F5 used; the rival not rejected; "not" paragraph; bridge | 760 words. Table 3 carries the rival and compact rows; the ladder is introduced in text; there is a bridge to §6. | 324–387 | mostly ✓; no summary sentence and no link to Table 2 (finding 7) |
| §6, 650: summary; 023's table with F6-A; the path 020–022; 024 with a bridge to 025 | 640 words, all present | 389–445 | ✓; one wording regression (finding 3); 020's and 021's negatives thinned (finding 2) |
| §7, 400: routes told once; manipulations; boundary; the test; F1's lanes as the summary | 428 words, all present | 447–487 | ✓ |
| §8, 450; §9, 600 in prose; §10, 220 in contribution order | 446, 683 and 218 words | 489–591 | ✓ in structure; §10 lost qualifiers (finding 2) |
| Appendices A–D | Table A1, Figure A1, Table A2; glossary; C1 and C2; reproducibility | 729–855 | ✓ |
| Images without titles or notes; F1a and F6a added | ✓ | — | ✓ |

## 2. Findings

### MAJOR

**1. MAJOR: the vocabulary of the central caveat is defined only in the glossary.**
- **What is wrong.**
  - *Reference state* appears 17 times in the body: in the abstract, §1 (l.59, 79, 95), §3, Table 2's column header
    (l.300), §4.4, §4.5, §5, §8 and §10.
  - *Operating point* is used at l.249–250, 261, 304 and 306; *template base(s)* and *base states* at l.246, 304, 329
    and 339; *reference prompt* at l.142.
  - None of these is defined outside Appendix B (l.802–804). §2 defines only the reference *cue* (l.122).
  - The core claim and its caveat are stated in these words, so a reader cannot read Table 2 without the glossary.
- **Fix (about 40 words).** Put this sentence in §2, after the reference cue (l.122) or at the start of *Interventions*
  (l.142):

  > A frame's **reference prompt** carries its reference cue; its **reference state** is the model's measured state on
  > that prompt, an **operating point** is one layer's state there, and **template bases** are reference states averaged
  > over a family's exposed frames.

  The wording is the glossary's own. In §1, gloss the first use at l.59: "(the model's measured state on the frame's
  reference prompt, §2)".

**2. MAJOR: the compression removed scope qualifiers from the core claim, and negatives that earlier reviews
required.**
- **What is wrong.**
  - **Contribution 2 (l.91–93).** It now reads "the programs predicted the internal consequences of the E-patch stage
    by stage, and the contrast change under real cue swaps…". v2 (l.141–146) qualified this as follows, and v3 drops all
    of it:
    - by rules frozen before the target prompts were executed;
    - up to the head's output;
    - for new cues from the tested lexical classes;
    - in new frames for most stages, given the frames' measured reference states.

    Without these, *stage by stage* reads as every stage, in every frame. §1 ¶2 (l.60–62) repeats the unqualified form.
  - **§10 (l.581–585).** It has lost all of the following, which v2 (l.789–802) carried:
    - C002's scope: its three screening templates, its tested replacement distributions, and "for the new cue classes
      tested, `L00.MLP` alone is not the whole cue stage at the pair level";
    - *from the tested lexical classes* (a v1 review note);
    - *in new frames for most stages*;
    - the routes sentence that resolved the v0 re-review's blocker: "The one preregistered intervention on a decoded
      direction, 025's rotation, did not establish its effect…".

    It also says *by rules frozen in advance* where C003's wording is *before the target prompts were executed*. The
    frame-conditional tables were computed inside the run, so *in advance* is less exact.
  - **§4.1 (l.212–215).** It keeps one of C002's three failed tests. Dropped are "P3's absolute retention counts were
    missed on the reserve and on new frames" and "the cue-word lexicon reached 0.699, against 0.70" (v2 l.285–287; the v0
    review asked for the lexicon's own citation). *Tested replacement distributions* is gone everywhere.
  - **§6 (l.423–425).** 020 lost "sat below six of its twelve frozen floor conditions, the coordinated template carrying
    most of the shortfall", qualified by the erratum. It also lost its status as "a calibration record, never evidence
    for a label" (v2 l.489–493). The v0 review required both. 021's own caveat, "not a claim of good prediction
    everywhere" (v2 l.509), is also gone.
- **Fix (about 170 words; no new content, all taken from v2).**
  - Restore v2's contribution-2 wording, with *programs written from three inputs*.
  - Add "(within the scopes and failures of §5–§6)" to l.62.
  - Restore §10's C002 scope, the prediction scope, *before the target prompts were executed*, and the routes sentence.
  - Restore the two failed C002 tests and *tested replacement distributions* in §4.1.
  - Add one clause for 020 in §6: "below six of its twelve frozen floors, mostly in coordinated frames, a reading the
    erratum qualifies; a calibration record, never evidence for a label". Adding 021's caveat is optional.

### SHOULD

**3. SHOULD: §6 opens with the phrase the record rules out (l.392–393).**
- **What is wrong.** "three of 023's four conditions passed…" is followed 27 lines later by the quotation that
  "Three of four conditions passed" is not a protocol outcome (l.420). It also echoes v0 blocker 2.
- **Fix.** Write "three conditions passed relative to the readout's own ceiling, with the new frames conditional, and
  one was an `ENVELOPE_ONLY_FAILURE`", as the abstract does (l.22–24).

**4. SHOULD: further terms are used in the body but defined only in the glossary, or nowhere.** Each needs one clause at
its first use.
- *E-only chain* (l.104, 395, 434): add "the program fed the encoding change alone" at l.395.
- *No-harm guard* (l.252, 345, 364): add the glossary's 0.05 rule at l.252.
- *Decoded read* `c_L` (l.345, 362): add "the program's prediction of what `L03.H04` reads" at l.362.
- *P1*: it appears only in Figure 3's caption (l.285), F3's axis labels and Figure 6's 012 chip. Name it at l.256: "a
  fixed-normalization linear read-out (the P1 read)".
- *Calibration data*, the basis of C002 (l.16, 200, 581): v2's legend defined it as 005's calibration tier. It is not
  the calibration draws of 021–024 (l.132) or 024's calibration range (l.444). Define it once in §2.
- *Development data* (l.202–203, 351) and *reserve nouns* (l.213): define them in §2's Data paragraph.
- *Channels* (l.249, 304): the letter D implies channels A–C, which never appear. Gloss from 016's design, or write
  "made to depend on the frame's reference state".
- *Pattern term* (l.345, 380), `Π` in Figure C1: gloss it from 018's design.
- *Frozen reading* (l.417): write "preregistered reading".
- *Level-3* (l.460–461): §3 names the rungs of the ladder in words (l.169–170). Number them there, or write "C002's
  causal-evidence gates".
- Smaller items:
  - *licensed* (l.291) and *informative frames* (l.237);
  - `LN₂` (l.143): write "the MLP's input LayerNorm";
  - *E–N* (l.441): drop the letters, since the sentence already states the guard's scope;
  - *zero-parameter* (l.385).

**5. SHOULD: some figures and tables are not fully used by the text.**
- **Figure 3.** Panels A1–A2 are never cited; only "panel B" is (l.279). Cite them in the transport paragraph (l.256–258)
  or in Table 3's read-out rows.
- **Figure 5.** Its only content that Table 4 lacks is the calibration range, and the text never uses it. Tie the
  +7.979e-5 margin and the envelope-only failure (l.417) to it.
- **Figure 6 (l.453–458).**
  - C1–C4 are named as "022's conditions", but §6 (l.433–437) never names them.
  - `R²_C`, in the chip for 021 and 023, is undefined.
  - Either define both in §6's sentence on 022, or relabel the badges by population only, since each chip's text already
    says what it measured.
- **Figure 1 (l.66–73).**
  - The caption uses *coordinated frames* before §2 defines them.
  - Neither caption explains the dashed box around the first two stages, which recurs in Figure 6's top row.
  - §4.2 never points back to Figure 1 as it walks the stages.

**6. SHOULD: Y1 and Y2 in the 009/011 sense survive.**
- **Where.**
  - F3's panel titles, "011 Y2" and "011 Y1";
  - A1's inset: "Y1 was scored over 138 pairs";
  - the caption's "009's Y1" (l.286);
  - Table 3's 007 row, "Y1 failed", which sits in the *new frames* column (l.353);
  - Figure C1's row labels: 011 Y1/Y2, 012 Y1/Y2 and 009 Y2.
- **Why it matters.** Figure 5, Table 4 and Figure 6 use Y1 and Y2 as populations, as the glossary's first sense says.
- **Fix.** Relabel these as Table 3 does: *009*, *011, encoding read*, *011, read-out replicated*. At l.353, write
  "failed, Spearman 0.785 against 0.80".

**7. SHOULD: §4.4's handoff is not taken up, and §5 has no opening summary.**
- **What is wrong.** §4.4 says "The confirmations should be read against this table" (l.308), but neither §5
  (l.326–329) nor §6 (l.391–398) refers to Table 2. §5 opens with its scope but not its result.
- **Fix.**
  - Add one sentence at each opening, from Table 2, saying which scored steps are reductions and which are exact
    evaluations.
  - Put the approved summary first in §5: "They held for new cues from the tested lexical classes, and in new frames for
    most stages. Among the stage programs, the new-frame exceptions are 015's token-local pattern model and 018's
    locked subset, and the compact alternatives failed (Table 3)."
  - In §3 (l.158–160), point the erratum sentence forward to §6's "How the completed program was reached".
  - In §1 ¶2 (l.57–58), point *numbered 005 to 025* to Table 1. About 15 experiment numbers appear before the key.

**8. SHOULD: the remaining repetition, about 160 words.**

| item | lines | keep | cut | ≈ words |
|---|---|---|---|---|
| the opposition ledger | 226–232; caption 236–241 | the §4.2 text | the caption's restatement; keep only what is plotted | 40 |
| 023's outcome | 392–394; Table 4; caption 413–415 | the opening and Table 4 | the caption's pass and fail list | 25 |
| Méloux et al. (2026) | 525–526; 547–548 | §8 | the §9 clause | 25 |
| circuit-discovery methods | 42–44; 541–545 | §1 | §9's restatement ("Beyond the methods of §1, …") | 20 |
| tables held by digest | 155–156; 497–498 | §3 | the §8 sentence | 13 |
| 018's no-harm-guard failure | 102, 252, 345, 363–365 | §1's tag, Table 3, l.363–365 | l.252's clause | 12 |
| Figure 6's status patterns; *only `Δc` scored* | 454, 458 (also 391–392, 508) | the image's legend; §6 and §8 | the two caption clauses | 18 |
| the three-input formula | 18–19, 58–59, 582–583 | the abstract and §1 | §10's copy, once §10's scope is restored | 10 |

**9. SHOULD: length.**
- **The count.** 6,637 words without headings, 6,701 with them, of which:
  - 622 in table cells: Tables 1–4 have 84, 174, 305 and 59;
  - 518 in figure captions;
  - 65 in table captions;
  - 183 in Box 1.

  Your 6,735 agrees within 1–2 %; the difference is headings and list markers.
- **Against the outline.** §4 is 390 words over, all from Box 1, Table 2 and two captions that the reviewer adopted. §1
  is 95 over and §9 is 83 over. The rest are on budget.
- **Cuts that help, about 200 words:**
  - finding 8, about 160;
  - §9's programs cluster (l.553–560), about 40;
  - §4.5's sentence of lineage (finding 13).
- **Cuts that would hurt:**
  - Box 1, Table 2 and §4.4's paragraph: the core claim and its caveat;
  - Table 3's rival and failed-compact rows, the negatives the v0 review required;
  - §6's paragraph on the path from 020 to 022, where contribution 1 is shown at work;
  - §8's bullets on baselines, self-repair, off-distribution states and validation;
  - §5's ladder paragraph, the body's only explanation of `κ`;
  - anything the claim-strength revision added, such as `every`'s `AXIS_ARTIFACT`, 024's permutation null and C003's
    maturity.
- **Net.** Finding 1 adds about 40 words, finding 2 about 170 and finding 4 about 50. Against the cuts, the paper stays
  near 6,700.
- **Pages.** Figures and tables are outside the word count. Table 3 and Figure 6 are each likely about a page at print
  size. If the TMLR build exceeds about 12 pages, move Figure 6, the lane summary, to Appendix C: Tables 3–4 and §7's
  boundary sentence carry its content. Do that rather than cut text.

**10. SHOULD: the submission build.**
- **The comments.** They open on 189 lines, 17 of which name internal design paths.
- **Anonymity.** Appendix D names the repository and its owner's account (l.841), and TMLR reviews anonymously.
- **The header.** Its pre-submission list (l.6–7) includes neither stripping the comments nor anonymizing the link,
  though the format review lists both.
- **The AI statement.** It says "the author" (l.36, 38). For the anonymous build that reveals a single author. The
  wording is the reviewer's, so the change is theirs to decide.
- **Fix.** Add both items to the header's list, and build from a stripped, anonymized copy.

### MINOR

11. **The abstract (l.21–22).** *from 013 on* is opaque there. The claim-strength revision dated the qualifier, so
    keep it, but write *from experiment 013 on*.
12. **§4.1 (l.201–202).** The labels (E), (T) and (R) are never used again. E also names `E_64` (l.346, 356). Drop them.
13. **§4.5 (l.318–322).** One sentence names ten experiments. Lead with what the program computes (Table 2) and move the
    lineage into a parenthesis.
14. **Box 1 (l.270–281).**
    - *the residuals track* (l.281) collides with the residual stream, in the Box itself (l.271) and at l.69. If that is
      011's meaning, write *prediction errors*.
    - State where the frozen axis came from: v2 called it 010's exploratory candidate.
    - Consider setting the read as a displayed equation, from 010's and 011's definitions.
15. **§4.3's title, "A stage program, end to end".** *End to end* means real swaps everywhere else (l.62, 71, 305,
    389). Retitle it "A worked example: the encoding read".
16. **The legend defines [locked prediction] (l.174), but the body no longer uses it.** Either tag the `an` call in Box 1
    (l.278–279), as v2 did, or drop it from the legend.
17. **Table 3's status column (l.336–354)** mixes bracketed and plain statuses. **Table 1 (l.190)**: *008–010* also
    covers 009, which has its own row and is not discovery; write *008, 010*.
18. **Stray pointers and entries.**
    - §2's *(§3)* for calibrated floors (l.132) should point to Table A2.
    - The glossary's `m_T`, `m_R` entry (l.797–799) carries result numbers for terms the body no longer uses; drop it,
      or return the clause to §4.1.
    - The status-tag load is 36 inline tags, down from 46, in 11 variants. §6 carries 8: add a default sentence like
      §5's (l.327).
19. **The four conditions have two names.** Figure 5's rows read *Y1/cue_final* while Table 4 reads *new cues, exposed
    frames / cue-final*. The caption maps them (l.413–414); relabelling the figure to match Table 4 would remove the
    mapping. In §1, "three conditions of the recovered share" (l.105) needs "`g`, §6".

## 3. Readability, in numbers

| measure | v2 | v3 |
|---|---|---|
| share of words in bullets | 64 % | 13 % |
| inline status tags | 46 in 13 variants | 36 in 11 variants |
| quotations of the project's own record | 23 | 17 |
| experiment-number mentions | 170, about 1 per 44 words | 152, about 1 per 44 words |

# Submission format and readiness: draft v2 (2026-09-30)

**What this review did.** It is read-only. It read:
- the whole draft (979 lines);
- the paper README, `reviews.md` and the figures README;
- the literature review's structure and entries;
- all eight figure PNGs, and the font settings in `make_figures.py`;
- `pyproject.toml`.

Nothing in the repository was changed. The counts come from throwaway scripts; "visible text" means the draft with its HTML comments removed.

Venue rules were checked on the web on 2026-09-30, and the sources are listed at the end. Where no 2027 call has been published yet, the 2026 rule stands in for it and is marked **unconfirmed**. No venue is recommended here.

**Tags.**
- **MUST**: needed for any submission.
- **MUST at X**: needed only at the venues named.
- **SHOULD**: strongly advised.
- **OPTIONAL**: worth considering.

## Summary

| Venue type | Length limit (source) | Estimated fit of v2 | Main changes needed |
|---|---|---|---|
| ML conference main track: NeurIPS, ICML, ICLR | **NeurIPS:** 9 content pages including figures and tables. References, appendices and the checklist are not counted (NeurIPS 2026 handbook; 2027 unconfirmed). **ICML:** 8 pages. References, the impact statement and appendices are not counted (ICML 2026 call; 2027 unconfirmed). **ICLR:** 9 pages at submission, 10 afterwards (ICLR 2027 guide; its 25 Sep 2026 deadline has passed). | **No.** About 13–16 pages in these templates, so about 40 % of the main text has to go. | Cut, or move to appendices. Redraw every figure. Anonymize. Add the NeurIPS checklist, the ICML impact statement or the ICLR AI-use statement. |
| NeurIPS Evaluations & Datasets track | The 2026 call page gives no page limit (**unconfirmed**). Its scope includes evaluation methodology and negative results. Review is double-blind by default, and single-blind on request when artifacts cannot be anonymized. | Same as the main track, if its 9 pages apply. | Same as the main track. Single-blind review would keep the commit pins intact. |
| ACL family through ARR, long paper | 8 pages. The mandatory Limitations section, ethics, references and appendices are not counted. Abstract of at most 200 words. Responsible NLP checklist. A4, two columns (ARR call; ACLPUB). The next cycle closes 12 Oct 2026 and feeds NAACL and COLING 2027. | **No.** About 15–17 ACL pages, so about 45 % has to go. About 0.9 page leaves the count when §8 becomes the Limitations section. | Everything above, plus A4 two-column figures, captions below, DOIs and the checklist. |
| TACL | 10 pages, references not counted. Appendices are capped at 5 + 3 pages and are not reviewed. A deadline every month (TACL submissions page; appendix policy). | **No.** About 35–40 % has to go. | Same as ACL. The appendix cap forces a choice between Appendix A, the related work and the prompt materials. |
| TMLR | No page limit. Reviews are due in 2 weeks for up to 12 pages of main content, and 4 weeks above that. Double-blind. Supplement up to 100 MB (TMLR author and reviewer guides). | **Yes, without major cuts.** About 14–16 pages. Cutting about 3 pages would bring it under the 12-page review clock, which is optional. | Template, abstract, statements, figures, bibliography, anonymization. |
| Journal: *Computational Linguistics* | Long articles are typically 30–40 journal pages. Review is not double-blind. Abstract of about 150–250 words (CL submission guidelines). | **Yes.** About 25–30 journal pages. | Template, abstract, figures and bibliography. Nothing needs anonymizing, so the repository and its commit pins can be cited directly. |
| Interpretability workshops | **MI Workshop at ICML 2026:** short papers 4 pages in ICML format or 5 in NeurIPS format, long papers 8 or 9. **Interpretability as a Science at NeurIPS 2026:** short up to 5 pages, long up to 9. Both exclude references and appendices and are non-archival. **BlackboxNLP 2026:** 8 pages (+1), plus a 6-page track on reproducibility and reliability. All 2026 deadlines have passed; the next editions are **unconfirmed**. | **Long papers:** like a main track, about a 40 % cut. **Short papers:** about a 70 % cut, down to one contribution. | Same as the main track. |
| arXiv preprint (a release, not a venue) | None. | **Yes.** | arXiv publishes the LaTeX source, so either strip the `.tex` comments or publish them on purpose. |

## Findings

### 1. Length

1. **MUST at every page-limited venue (all except TMLR and CL). How long the draft is.**
   - **Words.** The visible text is **9,735 words**, counted as whitespace-separated tokens containing a letter or digit. The brief's 10,843 includes the comments. The parts are:
     - internal header: 447 words (l. 1–45);
     - abstract: 501 (l. 47–90);
     - body prose: 5,488, by section:

       | §1 | §2 | §3 | §4 | §5 | §6 | §7 | §8 | §9 | §10 |
       |---|---|---|---|---|---|---|---|---|---|
       | 555 | 802 | 374 | 526 | 425 | 507 | 396 | 556 | 1,130 | 217 |

     - chain diagram: 92;
     - the two body tables: 249;
     - captions: 795;
     - references: 1,307 (61 entries);
     - Appendix A: 600; Appendix B: 189.
   - **Main text.** From the abstract to the conclusion, including headings, captions and tables, it is **7,170 words**, or 46.5k characters. It is set as **237 list items**.
   - **Other elements.** 183 hidden comments; 8 figures; 4 tables, none of them numbered.

   **Page estimate** (±20 %). Solid text alone would fill:

   | Template | Solid-text pages |
   |---|---|
   | NeurIPS / ICLR | 9.1 |
   | ICML | 7.8 |
   | ACL | 10.3 |
   | TMLR | 9.3 |

   Add about 20 % for the list structure and headings, and 3.5–4.5 pages of figures and tables at a legible size. That assumes F7 is left out and F2 and F8 become tables. The totals come to about:

   | Template | Estimated pages | Limit |
   |---|---|---|
   | NeurIPS / ICLR | 14–16 | 9 |
   | ICML | 13–14 | 8 |
   | ACL | 15–17 | 8 |
   | TMLR | 14–16 | none (12 sets the review clock) |

   **Action.** Set a target once the venue is chosen. These moves cut length without changing any claim:
   - Drop the text diagram (l. 297–307); Figure 1 shows the same chain.
   - Move F7 out of the main text; it repeats Appendix A.
   - Turn F8 into a table, or move it to an appendix.
   - Keep only one of three versions of the same results: F2, the §5 table, or the rival and compact-program bullets (l. 424–457).
   - Cut §9 to about 400–500 words and put the full version in an appendix.
   - Turn nested bullets into prose.

   Every cut has to keep its qualifiers. After cutting, rerun the checker and an independent review, as for v0–v2.

### 2. What a submission build must strip or convert

2. **MUST. Remove the internal header (l. 3–45).** It holds:
   - the line "Internal; not for circulation" (l. 3);
   - the provenance paragraph, with 9 repository links (l. 3–14);
   - the Conventions block (l. 16–45).

   Keep the substance of the status legend (l. 26–41), because the body's 45 status tags depend on it. Move it into §2 as a short table of evidence statuses. The audit-trail and checking conventions (l. 20–25, 42–45) belong in the supplement's README.

3. **MUST. Drop the hidden audit trail mechanically.**
   - **What it is.** 183 comments: 170 in the body, 12 in Appendix A and 1 in the header. They trace to 21 experiment READMEs and 18 other record files, at 6 commits: `d7c5e62` by default, plus `139755d`, `3d87d3c`, `454327d`, `b722e9c` and `eeff9ff`.
   - **How they get dropped.** Pandoc reads each comment as raw HTML, which its LaTeX writer leaves out. Copying the text by hand would not drop them.
   - **What not to do.** Do not turn them into `%` comments in a `.tex` file that will be posted, because arXiv serves the source.
   - **Checking.** Add a leak check (finding 35).

4. **MUST. Remove internal pointers and process wording from the visible text.**
   - l. 44: link to the figures README.
   - l. 659–660: "Record caveats" points to a section of the source map. State the caveats in §8 or in the supplement instead.
   - l. 664: link to the literature review.
   - l. 688–689: a quotation from the screening audit's JSON file. Introduce the screen or drop the quotation.
   - l. 962–966: Appendix B's links. Drop Appendix B and move the figures' provenance to the supplement.
   - l. 369: "C003's evidence E004–E015" uses internal IDs.
   - l. 174–176 and 184–186: "the source map finds", "README wording" and "committed review records". Say what an outside reader can actually check.
   - l. 605 and 636: "this draft" should read "this paper".
   - Captions at l. 324 and 582–585, and the Appendix A note at l. 949–951: "verbatim from the source map's overview", "screen row", "023-d row".
   - **Quotations of internal documents.** The draft quotes the project's own documents 32 times ("In C003's words", "In 005's words", and 021's, 024's and 025's). Either keep each as a quotation of a record cited in the supplement, or paraphrase it. Paraphrasing removes the checker's exact-quote anchor, so the trace would then live only in the audit edition.

5. **MUST at double-blind venues (every venue listed except CL). Remove identity leaks.**
   - l. 786 gives `github.com/mobahug/…`, which includes a username.
   - Commit hashes appear at l. 22–23 and l. 787, and throughout the comments. A reviewer can look up a commit hash from a public repository, so each one points to the authors.
   - The repository is public; the literature review found it in a search (review l. 82). It therefore acts as a non-anonymous preprint. NeurIPS, ICML, ICLR, ARR and TMLR allow that, provided the submission does not point to it.
   - Also: no acknowledgments; no author in the PDF metadata (TACL asks for this explicitly); self-references in the third person; supplementary links anonymous and not tracking downloads (ARR, ICML).

6. **SHOULD. Make experiment numbers readable as names.**
   - **How often they occur.** Experiment numbers appear about 225 times on about 160 visible lines. Record IDs also appear in citation-style parentheses (l. 53, 221, 223, 245, 349, 439, 475, 488, 500, 507, 542, 766, 771). C002 and C003 are mentioned 34 times.
   - **Action.**
     - Keep the original numbers, because they index the public record and renumbering would break the trail.
     - Write them one way throughout, such as "Exp. 23".
     - Define them once, with a short key in the main text; Appendix A remains the full key.
     - Give C002 and C003 descriptive names at first use.
     - Keep record IDs out of the parentheses that also hold author–year citations.

7. **SHOULD. Typeset status tags and protocol labels consistently.**
   - **What there is.**
     - 45 bracketed status tags in 13 variants, for example "[unestablished at its envelope; 0.90 guard met]" (l. 519).
     - 49 `UPPER_SNAKE` labels on 25 lines, up to 72 characters long (`NOUNNESS_PREDICTS_…_MEASURE_CLASS`, l. 946).
   - **Action.**
     - Keep both, because they carry the evidence grading.
     - Render every status tag through one macro so they all look alike.
     - Where a table already lists a result, move its tag into a table column.
     - Make the labels breakable, either with break points after the underscores or with a key of short names. A 42–72-character label in monospace overflows a 3.03–3.25 in column.

8. **SHOULD. Define terms an outside reader meets without a definition.**
   - "Closure" means two things: an experiment's closure (l. 157, 182–185) and the closure fraction κ (l. 234–235). Rename one of them.
   - Define "Tier A" (l. 167), "spent data" (l. 417, 640), "lock" (from l. 131) and "floor", "guard" and "envelope" at first use.

### 3. Structure

9. **MUST. Rewrite the abstract.** It is 501 words: about 30 sentences, 11 bullets, 11 bold labels, code spans and a protocol label (l. 47–90). The venues ask for:
   - ACL: at most 200 words of plain text (ACLPUB);
   - NeurIPS: one paragraph (template);
   - CL: about 150–250 words;
   - ICML: its template asks for 4–6 sentences (not rechecked here).

   **Action.** Write one plain paragraph of about 150–250 words. It should give the question, the setting, the main results with their statuses, and the causal boundary, with no labels and no code.

10. **SHOULD. Separate the introduction from the definitions.**
    - **The problem.** §1 opens with definitions (l. 94–124) and puts the contributions after them (l. 128–146). The same list comes back in the abstract (l. 87–90) and in §9 (l. 736–750).
    - **Action.**
      - Split §1 into an introduction and a *Setting* subsection that holds the definitions.
      - The introduction gives the motivation, the question, a summary of the results and one contributions list, with forward references.
      - State the list once; in §9, keep only the comparison with prior work.
    - **OPTIONAL.** The NeurIPS template asks for a title in initial capitals.

11. **SHOULD. Decide where related work goes.** §9 comes last and is 1,130 words, 16 % of the main text.
    - ML venues accept either placement; ACL papers usually put related work early.
    - **Trade-off.** Placed early, it lets reviewers weigh the narrow novelty statements before reading the results. Placed late, the results come first.
    - Either way, add two or three positioning sentences to the introduction.

12. **MUST at ARR, SHOULD elsewhere. Place the limitations section.** §8 (l. 608–660) already holds the right material.
    - ARR requires an unnumbered "Limitations" section after the conclusion and before the references. It may only discuss limitations, and it does not count toward the page limit.
    - NeurIPS checklist item 2 points to it.
    - In either case, remove its internal pointer (l. 659–660).

13. **MUST at ICML; SHOULD at NeurIPS and ICLR. Add an impact or ethics statement.** The draft has none.

    | Venue | Requirement |
    |---|---|
    | ICML | Impact statement, mandatory; not counted toward the page limit |
    | NeurIPS | Checklist item 10, broader impacts |
    | ICLR | Ethics statement, encouraged, at most 1 page, not counted |
    | ARR | Ethics section, optional |
    | TMLR | Only for work with significant risk |

    A short statement is enough for basic interpretability research on a 70M model. It should add no claims.

14. **MUST. Disclose the use of AI agents in the method and in the paper.**
    - **In the protocol.** The record shows agents inside the protocol itself. 023's lock review states that "A separate agent reviewed the lock read-only" (`experiments/023-block0-completion/evidence/lock-review-2026-09-24/REVIEW.md`, l. 3). Yet §2 (l. 171–177) presents the review gates as part of contribution 1 without saying who reviewed.
    - **In the literature search.** The search behind §9 was run by four agents, which read pages through a tool that summarizes them (literature review l. 62, 72–75). §9 (l. 664–667) does not say so.
    - **In the paper's reviews.** The reviews of the drafts were done by agents (`reviews.md` l. 94).
    - **Venue rules.** ICLR requires an AI-use statement. NeurIPS requires a description when LLMs are an important or non-standard part of the method, and checklist item 16. ARR has checklist item E1.
    - **Everywhere else.** Saying who reviewed is part of describing the protocol accurately.

15. **SHOULD, and needed for the NeurIPS and ARR checklists. Add a reproducibility statement.**
    - **Already present:** the model, its revision, and float32 on CPU (l. 123–124).
    - **Missing:**
      - **code availability:** the only pointer is the link that identifies the author (l. 786);
      - **software:** TransformerLens ~3.9 and PyTorch ≥ 2.6 per `pyproject.toml`, with `uv.lock`. TransformerLens is never named in the draft;
      - **compute:** hardware and run times. Take them from the record; do not estimate them;
      - **seeds:** only the seeded control subsets are mentioned (l. 440);
      - **prompt materials:** the frames, the cue lists by class and the nouns. The paper gives only counts (l. 616–617);
      - **the lock and digest mechanism;**
      - **the code license:** AGPL-3.0 (`LICENSE`).

16. **MUST at NeurIPS and ARR. Fill in the checklist.**
    - **NeurIPS 2026** has 16 items: claims; limitations; theory; reproducibility; open access to data and code; experimental details; statistical significance; compute; code of ethics; broader impacts; safeguards; licenses; new assets; human subjects; IRB; and LLM use. It goes in the PDF after the references and appendices. A missing checklist means desk rejection.
    - **ARR** uses the Responsible NLP checklist. An incorrect or incomplete one can mean desk rejection.
    - **Items that need care:**
      - **Statistical significance.** Most statistics are single-run point estimates against frozen floors. Answer with that justification, and compute nothing new.
      - **Compute.** It is not stated anywhere in the draft.
      - **Licenses.** See finding 28.
      - **LLM use.** See finding 14.

17. **SHOULD. Follow appendix conventions.**
    - **Placement.** Appendices go after the references, in the same PDF. At NeurIPS the checklist follows them. At ARR they are two-column, except for math-heavy sections.
    - **TACL** caps appendices at 5 + 3 pages and does not review them.
    - **ICLR and ARR** say reviewers need not read appendices, so nothing the argument depends on can live only there.
    - **An appendix set built from existing material:**
      - the sequence (Appendix A), rewritten without "verbatim from the source map's overview";
      - the evidence-status definitions and the notation, from the header and §2;
      - the full related work, if §9 is cut;
      - the prompt materials, from the record;
      - the evidence ladder, as a table (from F8).

      Appendix B (l. 960–979) moves to the supplement.

### 4. Figures and tables

18. **MUST. Make the figures legible in print.**
    - **How they are drawn now.** The figures are 7.6–15.0 in wide, with 5.8–9 pt text (`make_figures.py`).
    - **Printed at full text width.** At the full text width of every template, all figures except F4 fall to **2.4–5.4 pt**. The widths are 5.5 in for NeurIPS and ICLR, 6.30 in for ACL, 6.5 in for TMLR and 6.75 in for ICML. At column width (3.03 in for ACL, 3.25 in for ICML) they fall to **1.3–3.9 pt**. F4 reaches 5.3–8.0 pt at full width.
    - **Beyond the figures README.** Its *Limits* section names F1, F2 and F7; F3, F5, F6 and F8 are affected too.
    - **Target.** A common rule of thumb is 6–8 pt at final size. ACL asks for body-text size where possible, and NeurIPS for legible artwork.
    - **Action.** Add a print profile to `make_figures.py` that draws each figure at its final width with 7–9 pt text. Then rerun `figure_data.py`'s checks and review the figures again, as in the first round.

19. **MUST. Remove the in-image titles and notes.**
    - **Titles.** Each image carries its own title ("F1. …" to "F8. …") and a block of notes, both of which repeat the caption. Remove the titles, move the notes into the captions or trim them, and write "Figure N" throughout.
    - **Internal terms in the notes.**
      - F1's footer: "the source map's evidence-status section".
      - F7's footer: "From the source map's overview; its screen row and its 023-d row …".
      - F8's footer.

20. **SHOULD. Choose a layout for each figure.** The two text-size columns show the smallest and main text size at the given width.

    | Figure | Drawn (in) | Text at 5.5 in | Text at 6.75 in | Single-column template | Two-column template |
    |---|---|---|---|---|---|
    | F1 chain | 14.4 × 10.0 | 2.4–2.7 pt | 2.9–3.3 pt | A full page, or split into two figures, one per lane. Shorten the chip text and move the notes to the caption. | Two `figure*`, one per lane. |
    | F2 prediction map | 11.3 × 11.2 | 2.8–3.3 | 3.5–4.1 | Merge with the §5 table into one table with a status column, or redraw at text width. | `table*` |
    | F3 head | 14.9 × 5.4 | 2.8–3.3 | 3.4–4.1 | Three panels at text width, 7–8 pt, with the in-panel notes moved to the caption. | `figure*`, or two panels (A2 and B) with A1, which is descriptive, moved to an appendix. |
    | F4 ledger | 7.6 × 4.0 | 5.3–6.5 | 6.5–8.0 | Redraw at half the text width. | Redraw at one column, 3.03–3.25 in. |
    | F5 closure ladder | 11.3 × 4.4 | 3.6–4.4 | 4.4–5.4 | Text width, shorter annotations, legend in the caption. | `figure*`, or one column with the annotations in the caption. |
    | F6 end to end | 15.0 × 3.7 | 2.7–3.3 | 3.3–4.1 | Two rows, or panel C moved to an appendix. | `figure*` with panels A and B; C in an appendix. |
    | F7 sequence | 14.2 × 13.0 | 2.4–2.7 | 2.9–3.3 | It repeats Appendix A, so keep only one. As a figure, it fits only on a landscape appendix page. | Same. |
    | F8 boundary | 10.3 × 9.5 | 3.3–3.9 | 4.1–4.7 | A table. 18 of its 25 rows read "checked" for both claims; show a summary per level, with the full grid in an appendix. | A one-column table with short state codes. |

21. **SHOULD. Produce figures in print-ready formats.**
    - Export vector PDF from matplotlib. LaTeX cannot include SVG without Inkscape, and the PNGs are only 200 dpi.
    - Set `pdf.fonttype` to 42 so matplotlib does not produce Type-3 fonts. ICML 2026 does not check for them but prefers vector graphics; ACL requires every font to be embedded.
    - Fix the PDF metadata so regeneration stays byte-identical.
    - Write raw identifiers as math: `f_‖` and `f_⊥` in F4, `c_L` in F5, and `cue_final` and `p_t` in F6.
    - **OPTIONAL.** Match the figure font to the body font.

22. **SHOULD. Recheck colour and greyscale after resizing.**
    - **Already done.** Where colour alone carries a status (F1, F2, F7), patterns duplicate it; elsewhere the status is written out. That meets ACL's rule not to rely on colour alone.
    - **To recheck.** Hatching, stripes and dots alias at small sizes, and green and vermilion print at nearly the same grey (figures README).

23. **MUST. Number and caption the tables.**
    - **Numbers and captions.** None of the four tables (l. 371–383, 510–515, 925–947, 968–977) has a number or a caption. NeurIPS puts a table's title above it (template); ACL puts captions below (ACLPUB).
    - **Formatting.**
      - Use booktabs.
      - Name the metrics in the column headers. In the §5 table, a cell such as "0.957 / 0.066" is identifiable as Spearman / MAE only from l. 369.
      - In a two-column template, the §5 table needs `table*`.
      - Appendix A needs `longtable` or a landscape page, with breakable labels.
      - Appendix B's table is dropped.
    - **Duplicate captions in pandoc.** Pandoc turns an image that stands alone in a paragraph into a figure and uses its alt text as the caption (l. 315, 396, 405, 415, 437, 530, 580, 587). Each figure would get a second caption. Map the bold caption paragraph to `\caption` and drop the alt text.

24. **SHOULD. Tighten the captions.**
    - **Length.** The captions total 795 words: F6 about 145, F5 about 120 and F2 about 110. For page-limited venues, move the provenance details into the supplement.
    - **F6, panel C.** It plots 021's flattened R², 0.6167 and 0.5797, while §6 reports 021's scored token-mean R², 0.016018 (l. 489–490). The caption labels panel C "flattened … [descriptive]", but it should also say plainly that this is a different statistic from the one in §6, so that readers do not compare the two.

### 5. References

25. **MUST. Complete and standardize the bibliographic records.**
    - **What is wrong.**
      - 61 entries, with authors given by surname only (l. 789). ACL requires full names, and most bibliography styles need given names.
      - Four author lists are cut short with "and colleagues": Ameisen 2025 (l. 790), Elhage 2021 (l. 818), McDougall 2024 (l. 878) and Mueller 2025 (l. 887).
      - Venue details are uneven. Only 14 entries have page ranges, and most NeurIPS, ICLR and ICML entries lack proceedings details.
      - Geiger et al. (2025) gives "JMLR 26" with no article number (l. 834–836). Bayat Makou et al. gives "TMLR" with no publication record (l. 797–798).
      - Link labels vary from entry to entry.
    - **Action.**
      - Build a `.bib` file from DBLP, ACL Anthology, PMLR and arXiv exports.
      - Check each entry against the literature review's verified record.
      - Add DOIs; ACL wants them where possible and Anthology links otherwise.
      - Cite with natbib, author–year.

26. **MUST. Remove the provenance notes from reference entries.**
    - Chan et al. (l. 809–811) says "authors and date as given by the mirror" and links a GreaterWrong mirror. Cite the original Alignment Forum post.
    - Joshi et al. (l. 855–857) and Vaccaro (l. 903–905) carry notes about their arXiv titles.

    In each case, pick the version to cite and cite it plainly; the notes stay in the literature review.

27. **SHOULD. Update the preprints.** 18 entries are arXiv-only, 13 of them from 2026. §9's closest precedents rest partly on them, for example Xun (2026). At submission time, check each for a published version.

28. **SHOULD, and needed for the NeurIPS and ARR license items. Cite the model and the library.** The paper cites neither Pythia (Biderman et al., 2023) nor TransformerLens, which the code runs on (`pyproject.toml`). Cite both, with their licenses. Citing PyTorch is optional.

29. **MUST. Replace the "project's record" paragraph (l. 784–787).** It is not a reference.
    - For review, point to the anonymized supplement instead.
    - At camera-ready, cite an archived, versioned snapshot of the repository, such as a DOI for a tagged release, as an ordinary reference.
    - The experiments are defined in Appendix A; they are not cited as publications.

30. **OPTIONAL. Consider coverage.** The literature review has verified entries that the draft does not cite:
    - two of the screening audit's seven close works: A6, Mueller, Xia and Linzen (2022), and A8, Lepori, Serre and Pavlick (2023);
    - Lakretz et al. (2019), on number units;
    - Leavitt and Morcos (2020), on falsifiable interpretability.

    Reviewers in the area may expect some of these. The choice is the authors'.

### 6. Keeping the audit trail available, and the toolchain

31. **SHOULD. Ship the audit trail as supplementary material.** The package would contain:
    1. an **audit-edition PDF**: the same text, with each trace as a numbered endnote giving the record file and line;
    2. a **static snapshot** of the roughly 39 cited files at their 6 commits, so the trail can be followed without git history;
    3. **`check_source_map.py` in snapshot mode**. It currently hard-codes an absolute path containing the username (l. 34);
    4. **`figure_data.py` and `make_figures.py` in snapshot mode**. `figure_data.py` reads with `git show <commit>:<path>` (l. 38), which an anonymized snapshot cannot serve;
    5. a **README**.

    **For double-blind review:**
    - Replace the commit pins with neutral labels, and reveal the mapping later.
    - Scrub identifying strings: 189 files under `experiments/` and 5 under `research/` contain the username or an absolute home path.
    - Scrubbing changes bytes, so the recorded digests will not verify on the anonymized copy. Say so in the README, and point to the archived original for verification after review.
    - Limits: the supplement must stay within 100 MB at NeurIPS and TMLR. ICML accepts an anonymous repository branch that is not modified after the deadline. ARR forbids links that track downloads.

    **At camera-ready, or on arXiv:** archive a tagged release with a DOI and cite it. The audit edition can go on arXiv as an ancillary file.

32. **MUST. Install a toolchain.** Neither pandoc nor any TeX engine is installed on this machine (checked). Install them or use Overleaf. The figure scripts need matplotlib 3.11.2 to reproduce the figures byte for byte (figures README).

33. **SHOULD. Convert Unicode and math.**
    - **Unicode characters in the visible text:** Δ (20), ² (30), → (7), κ (4), − (6), and ρ, ₀, ₂, ∥ and ≤.
    - **Math written in code spans:**
      - `c(x) = log P(singular) − log P(plural)` (l. 94);
      - `Δc(f, w, n) = …` (l. 112);
      - `MLP₀(LN₂(W_E[w]))` (l. 201);
      - `W_V W_O`, `m_T`, `m_R` and `Δκ_64(E)`.
    - **Action.**
      - Convert all of this to LaTeX math, with the two definitions as numbered equations, so the source compiles with pdflatex. Every template listed supports pdflatex, and arXiv processes it.
      - Write component names such as `L03.H04` in one consistent style.

34. **SHOULD. Build from a single source.**
    - **Recommended.** Keep editing the Markdown with its traces, and generate LaTeX from it with a pandoc Lua filter:
      - comments become `\trace{}`, which is empty in the submission build and printed as endnotes in the audit build;
      - status tags become `\status{}`;
      - the bold caption paragraphs become captions;
      - citations are converted.

      That way the checker keeps working on every later edit.
    - **Alternative.** Convert once, then port the checker to read `\trace{}` in the `.tex`.
    - **First, commit draft v2.** It is untracked (`??` in git status), and the submission should come from a pinned audit edition.

35. **MUST. Run release gates on every build.**
    - **Leak check.** Search the PDF text and the sources for:
      - `<!--`;
      - `@` followed by 7 hex digits;
      - `experiments/`, `research/` and `docs/`;
      - `mobahug` and `/Users/`;
      - "draft" and "Internal".
    - **PDF checks.** Check the metadata, and run `pdffonts`: every font embedded, no Type 3.
    - **Numbers.** Compare the numbers in the submission text with those in the audit edition; none should be added, lost or changed.
    - **Checker.** Run the checker on the audit edition.
    - **Review.** Have the converted PDF reviewed independently and read-only, as for v0–v2.

## Conversion plan (Markdown to LaTeX; template left open)

| # | Step, in order | Effort (rough, working days) |
|---|---|---|
| 0 | **Choose the venue type, template and target length.** The user's reviewer decides, using the summary and the trade-offs below. The two-column or single-column choice has to come before any figure work. | – |
| 1 | **Commit draft v2** as the audit edition to convert from. It is untracked now, and committing needs the user's go-ahead. | minutes |
| 2 | **Set up the toolchain.** Install pandoc and TeX, or use Overleaf, then compile the official template unchanged. | 0.5 |
| 3 | **Build the pipeline.** A pandoc Lua filter handles the comments (dropped, or turned into `\trace`), the status macro, captions, and Unicode to math. It produces a submission target and an audit target, and runs the release gates (finding 35). | 1.5–2.5 |
| 4 | **Bibliography.** Build the `.bib` for the 61 entries plus Pythia and TransformerLens; check it against the literature review; update the preprints; remove the notes. | 1 |
| 5 | **Front matter and apparatus.** Rewrite the abstract and introduction, list the contributions once, put the status legend and notation in §2, fix the internal pointers, "this draft" and the quotations. | 1–2, plus review |
| 6 | **Length reduction,** for page-limited venues only. This also decides which figures and tables stay in the main text. | 2–4, plus review of the qualifiers (TMLR: optional, 0.5–1; CL: none) |
| 7 | **Figures.** Add the print profile, redesign each figure (finding 20), export PDF with TrueType fonts, rerun `figure_data.py` and review the figures again. | 2–4 |
| 8 | **Tables.** Numbers and captions, booktabs, `table*` or `longtable`, breakable labels. | 0.5–1 |
| 9 | **Statements and checklist.** Impact or ethics, reproducibility (from the record), AI use, placement of the limitations section, and the NeurIPS or ARR checklist. | 0.5–1.5 |
| 10 | **Supplement and anonymization.** The audit edition, the record snapshot, the scripts in snapshot mode, scrubbing, the README and a size check. For CL, the repository can simply be cited. | 1–2 |
| 11 | **Release gates, and an independent read-only review of the PDF.** | 0.5–1 |

**Totals, roughly, excluding time spent waiting for reviews:**

| Target | Working days |
|---|---|
| TMLR or CL | 9–16 (CL about 1 day less, since nothing is anonymized) |
| 8–9-page conference, ARR or TACL | 11–20 |
| 4–5-page workshop paper | 7–12 |

The two largest items are length reduction and figures.

## Trade-offs for the reviewer (no venue is recommended)

- **Length versus qualifiers.**
  - Formats of 8–9 pages force cuts of about 40–45 % in a text whose reviews repeatedly restored qualifiers. Appendices are not reliably read: ICLR, ARR and TACL say so.
  - TMLR and CL keep the text nearly whole.
- **Anonymity versus the audit trail.**
  - Double-blind review means removing the commit pins and identifying strings from the supplement. That stops the digests from verifying until the authors are revealed.
  - CL is not double-blind. The NeurIPS Evaluations & Datasets track grants single-blind review on request when artifacts cannot be anonymized.
  - The public repository already acts as a non-anonymous preprint.
- **Review criteria.**
  - TMLR asks whether the claims are supported by the evidence and whether some readers would want to know the findings; novelty and significance are not criteria.
  - Main tracks weigh significance. There, one behavior in a 70M model, with its causal gates unestablished, is what will be judged.
  - The Evaluations & Datasets track welcomes evaluation methodology and negative results.
  - Workshops are non-archival: they give feedback and leave archival submission open.
- **Audience.**
  - The linguistic phenomenon suits ACL venues, TACL, CL and BlackboxNLP.
  - The protocol and the interpretability method suit ML venues, the MI and Interpretability-as-a-Science workshops and the Evaluations & Datasets track.
- **Timing, as of 30 Sep 2026.**
  - The ICLR 2027 deadline passed on 25 Sep.
  - The next ARR deadline is 12 Oct 2026, feeding NAACL and COLING 2027.
  - ICML 2027 is expected around late January 2027 and NeurIPS 2027 around May 2027; both are unconfirmed.
  - TMLR, TACL (monthly) and CL take submissions on a rolling basis.
  - The 2026 workshop deadlines have passed.
- **Paperwork and layout.**
  - Each venue adds its own paperwork: the NeurIPS checklist; the ARR checklist and Limitations section; the ICML impact statement; the ICLR AI-use statement.
  - The figures have to be redrawn for one layout: Letter single-column (NeurIPS, ICLR, TMLR), Letter two-column (ICML) or A4 two-column (ACL, TACL).

## Sources (checked 2026-09-30)

**NeurIPS**
- [NeurIPS 2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [Main track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [Paper checklist](https://neurips.cc/public/guides/PaperChecklist)
- [Official template, on Overleaf](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc)
- [Evaluations & Datasets track call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)

**ICML and ICLR**
- [ICML 2026 call for papers](https://icml.cc/Conferences/2026/CallForPapers)
- [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- [ICLR 2027 call for papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- ICML 2027: no official call yet; dates **unconfirmed**.

**ACL, ARR and TACL**
- [ARR call for papers](https://aclrollingreview.org/cfp)
- [ARR dates](https://aclrollingreview.org/dates)
- [ARR Responsible NLP checklist](https://aclrollingreview.org/responsibleNLPresearch/), seen through a search summary.
- [ACLPUB formatting](https://acl-org.github.io/ACLPUB/formatting.html)
- [TACL submissions](https://transacl.org/index.php/tacl/about/submissions)
- [TACL appendix policy](https://transacl.org/index.php/tacl/announcement/view/105)

**TMLR and CL**
- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
- [TMLR reviewer guide](https://jmlr.org/tmlr/reviewer-guide.html)
- [TMLR style file](https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/main/tmlr.sty): 6.5 in text width, single column.
- [*Computational Linguistics* submission guidelines](https://cljournal.org/submissions.html)

**Workshops**
- [MI Workshop, ICML 2026](https://mechinterpworkshop.com/cfp/)
- [Interpretability as a Science, NeurIPS 2026](https://interpscience.github.io/cfp)
- [BlackboxNLP 2026 call](https://blackboxnlp.github.io/2026/call/), seen through a search summary.

**Column and text widths**
- The NeurIPS width of 5.5 in is from its template. ICML's 6.75 in with a 0.25 in gap is from its instructions, as summarized by search. ACL's 7.7 cm columns on A4 are from ACLPUB.

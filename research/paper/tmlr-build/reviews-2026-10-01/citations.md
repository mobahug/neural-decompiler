# Citation review of the TMLR build bibliography (draft v3), 2026-10-01

The reviewer could not write this report to `scratchpad/citereview/report.md`, because the harness blocks subagents from
writing report files; it is kept here as returned. The working files (the arXiv, ACL, PMLR, OpenReview and JMLR records,
the page renders, and a hashed snapshot) stayed in the session's scratchpad, `citereview/`, which is not kept.

## Verdict: PASS WITH NOTES

There are no blockers. Every author list, title and year matches its source record, and nothing is misattributed. All
81 in-text citations go to the intended work, and `check_bib.py` passes.

The findings that need action are about venues:
- **Two works cited as arXiv preprints were actually published:**
  - Grant et al. at ICLR 2026;
  - Zhang et al. at ICML 2026.
- **Three entries lack proceedings details their sources give:** Anani et al. 2026, Huang et al. 2026 and Geiger et al.
  2025.
- **Three older ICML entries are inconsistent** with the ones that carry PMLR details.

All of these come from the literature review's records. `check_bib.py` can't catch them because it doesn't check venues.

**The files changed during the review.** `references.bib` and `citekeys.json` were rewritten at 09:48:59 and again at
09:56:14. The draft was rewritten at 09:49:10, and the PDF was rebuilt at 09:49:43 and at 09:57:39. Those edits fixed
four rendering problems the reviewer had found in the 09:33 build (see N1). Every check was re-run against one
consistent snapshot, and the build used exactly that bib:

| file | SHA-1 prefix |
|---|---|
| references.bib | 0c82373a |
| citekeys.json | 034c353a |
| draft-v3-2026-09-30.md | 61c2e847 |
| main.tex | 348dedc1 |
| main.bbl | 7ea363a9 |
| main.pdf (built 09:57:39) | 2bf1618e |

No repository file was modified.

## BLOCKER

None.
- **Authors:** all 63 lists match the source records in full names, order and accents, with nobody missing or added.
  Three use the publisher's form rather than arXiv's, and each matches that publisher's record:
  - `biderman2023pythia`: "Quentin Gregory Anthony" (PMLR v202 and the Pythia README);
  - `geiger2024finding`: "Noah Goodman" (PMLR v236);
  - `mcdougall2024copy`: "Callum Stuart McDougall" (ACL Anthology).
- **Titles:** every title matches its source.
  - `mcdougall2024copy` uses its published title, confirmed in the ACL Anthology.
  - `joshi2026causality` and `vaccaro2026preregister` use their "Position:" titles, confirmed in PMLR v306 (ICML 2026,
    online since 2026-09-29).

## SHOULD

**S1. `grant2025addressing` cites the preprint, but the paper was published at ICLR 2026.** Fields: entry type, journal,
year.
- **Source:** OpenReview `cZrTMqYVL6` (https://openreview.net/forum?id=cZrTMqYVL6).
  - It records venue "ICLR 2026 Oral" (ICLR.cc/2026/Conference), decided 2026-01-26.
  - Its BibTeX gives booktitle "The Fourteenth International Conference on Learning Representations" and year 2026.
  - Authors and title match.
- **Now:** `@article`, "arXiv preprint arXiv:2511.04638", 2025.
- **Fix:** `@inproceedings`, `booktitle = {International Conference on Learning Representations}`, `year = {2026}`.
- **Knock-on edits:**
  - §8's in-text form ("Off-distribution states") becomes "Grant et al., 2026";
  - `citekeys.json` changes to match;
  - the draft's list entry becomes "(2026) … ICLR 2026", and `check_bib.py` will enforce that year.
- This isn't a misattribution, because 2025 is correct for the arXiv version.

**S2. `zhang2026weights` is cited as an arXiv preprint but appeared at ICML 2026.**
- **Source:** https://proceedings.mlr.press/v306/zhang26d.html gives PMLR 306:154170–154189, 2026. OpenReview
  `nYzdAcLN5F` records "ICML 2026 regular", accepted 2026-04-30. Authors and title match.
- **Fix:** `@inproceedings`, with:
  - `booktitle = {International Conference on Machine Learning}`;
  - `series = {Proceedings of Machine Learning Research}`, `volume = {306}`, `pages = {154170--154189}`;
  - `publisher = {PMLR}`, `year = {2026}`.
- **Draft list:** "ICML 2026, PMLR 306, pp. 154170–154189".

**S3. `anani2026certified` has no volume or pages.**
- **Source:** https://proceedings.mlr.press/v306/anani26a.html gives PMLR 306:2650–2676.
- **Add:** `series`, `volume = {306}`, `pages = {2650--2676}`, `publisher = {PMLR}`.
- **Draft list:** "ICML 2026, PMLR 306, pp. 2650–2676". This matches how Joshi and Vaccaro, in the same volume, are
  already cited.

**S4. `huang2026discovering` has no volume or pages.**
- **Source:** https://proceedings.mlr.press/v306/huang26b.html gives PMLR 306:46309–46412. (`huang26a` is a different
  paper.)
- **Add:** `volume = {306}`, `pages = {46309--46412}`, plus `series` and `publisher` as in S3.

**S5. `geiger2025causal` has no number or pages.**
- **Source:** the JMLR BibTeX at https://jmlr.org/papers/v26/23-0058.bib gives volume 26, number 83, pages 1–64. Its
  authors, title and year match exactly.
- **Add:** `number = {83}`, `pages = {1--64}`.
- **Draft list:** "JMLR 26(83):1–64".

**S6 (lower priority). Three older ICML entries leave out their PMLR volume and pages,** while Pythia, Joshi and Vaccaro
include them. Each needs `series = {Proceedings of Machine Learning Research}` and `publisher = {PMLR}`, plus:

| key | add | source |
|---|---|---|
| `weiss2021thinking` | `volume = {139}`, `pages = {11080--11090}` | https://proceedings.mlr.press/v139/weiss21a.html |
| `rushing2024explorations` | `volume = {235}`, `pages = {42836--42855}` | https://proceedings.mlr.press/v235/rushing24a.html (also the arXiv journal_ref) |
| `mueller2025mib` | `volume = {267}`, `pages = {45069--45108}` | https://proceedings.mlr.press/v267/mueller25a.html |

## NOTE

**N1. Four rendering problems were fixed during the review.** All four now render correctly in the 09:57:39 PDF.
- **`wu2024reply`** had rendered "makelov", with a closing quote where the opening one belonged. It is now `{Makelov}`
  with `` `…' ``.
- **`makelov2024this`** had rendered "…for? an interpretability…". It is now `{An}`.
- **`bertinetto2021preface`** had rendered "pp. i–i". That field has been dropped. PMLR itself gives `i--i`, so
  `pages = {i}` would also be faithful.
- **`nanda2022transformerlens`** had rendered as a stretched line. It now uses `howpublished = {GitHub repository}` plus
  a `url` field. That differs from the README's BibTeX in form only: the authors, title, year and URL are the same.

**N2. The ICML entries aren't styled consistently.**
- `joshi2026causality` and `vaccaro2026preregister` have no `publisher = {PMLR}`, so "PMLR" doesn't print for them.
- Their booktitle is "International Conference on Machine Learning", while Pythia's is "Proceedings of the 40th
  International Conference on Machine Learning".

It would be worth choosing one form when applying S2–S4 and S6.

**N3. `hayes2026explaining`** gives "Belinda Z Li" with no period after the initial. That matches arXiv, so "Belinda Z.
Li" is optional.

**N4. Five ACL-published entries link to arXiv:** `mcdougall2024copy`, `syed2024attribution`, `hao2023verb`,
`brinkmann2025large` and `vanmiltenburg2021preregistering`. They have DOIs, so this is acceptable. For McDougall, though,
the arXiv page carries the old title ("…an Attention Head"). Consider
`url = {https://aclanthology.org/2024.blackboxnlp-1.22/}`.

**N5. `chan2022causal` is correct.**
- The Alignment Forum GraphQL record confirms the authors directly: LawrenceC (Lawrence Chan), then seven coauthors in
  the bib's order.
- It confirms the date: posted 2022-12-03.
- The post's own BibTeX misspells "Goldwosky-Dill". The bib's "Goldowsky-Dill" is right.
- The draft's list still says "authors and date as given by the mirror" and links GreaterWrong. It could link the
  Alignment Forum post instead.
- The record's title ends in "[Redwood Research]". Leaving that suffix out matches the post's own "please cite as" line.

**N6 (optional). No NeurIPS entry gives a volume or pages.** Because they're all left out, this is a style choice. The
values from proceedings.neurips.cc are below; each paper's title, authors and year match the bib.

| key | volume | pages |
|---|---|---|
| `geiger2021causal` | 34 | 9574–9586 |
| `conmy2023towards` | 36 | 16318–16352 |
| `friedman2023learning` | 36 | 49044–49067 |
| `lindner2023tracr` | 36 | 37876–37899 |
| `wu2023interpretability` | 36 | 78205–78226 |
| `gross2024compact` | 37 | 79453–79515 |
| `shi2024hypothesis` | 37 | 94539–94567 |
| `sutter2025non` | 38 | main-track proceedings not posted yet |

**N7. `bayatmakou2026many`** could link TMLR's OpenReview page, https://openreview.net/forum?id=Ur42Xx1QPv, which shows a
"Featured Certification".

**N8. Two citation forms are the template's house style.**
- "Geiger et al., 2021, 2024" renders as "Geiger et al., 2021; 2024", because `tmlr.sty` sets `yysep={;}`.
- Two-author citations use "&".

No action is needed.

**N9. `mueller2025mib`:** PMLR lists "Jaden Fried Fiotto-Kaufman", while the bib uses the arXiv form. Both are published
forms of the name, so no action is needed.

**N10. The bibliography runs to page 18,** not 17. Zhang and Zhu are on p. 18.

## In-text citations (task 3)

The Markdown body has 81 citation instances, covering 82 keys:

| section | citations |
|---|---|
| §1 | 18 |
| §2 | 2 |
| §8 | 12 |
| §9 | 49 |

All 63 keys are cited.
- **main.tex:** it contains exactly 81 `\citealp` commands, the same keys in the same order as the Markdown. `main.log`
  has no undefined citations.
- **citekeys.json:** every in-text form agrees with its bib entry on first surname, author count (one author, "X and
  Y", or "et al.") and year.
- **Every citation in context:** all 81 were read, not a sample of 25, and each cites the intended work.
  - The only multi-year group, "Geiger et al., 2021, 2024", maps to `geiger2021causal` (causal abstraction, NeurIPS
    2021) and `geiger2024finding` (DAS, CLeaR 2024).
- **Against the arXiv abstracts:** the semicolon groups' claims were checked, and they match:
  - Zhu: "All 81 predictions were frozen before their data".
  - Dong: preregistered with frozen constants.
  - Xun and McCann: pre-registered studies.
  - Gong: self-repair backups.
  - Darade: what is represented isn't computed.
  - Lin and Liu: identification assumptions.
  - Li et al.: predicting out-of-distribution behavior.
  - Hayes and Tan: programs for attention heads, and MLP surrogates.
  - Somani: verified programs at GPT-2 scale.
  - Bayat Makou: five Pythia models.

## The two new references (task 4)

**Biderman et al. (2023) is the right citation for `EleutherAI/pythia-70m-deduped`.**
- The Pythia README says "If you use the Pythia models in your research, please cite our paper". It gives the ICML 2023
  entry: PMLR, pp. 2397–2430, with "Anthony, Quentin Gregory".
- The model card links the same paper.
- The pinned revision `e93a9faa…` exists on Hugging Face, last modified 2023-07-09.

**The TransformerLens README recommends exactly "Nanda and Bloom, 2022".** Under "Please cite this library as" it gives
`@misc{nanda2022transformerlens, title={TransformerLens}, author={Neel Nanda and Joseph Bloom}, year={2022},
howpublished={\url{https://github.com/TransformerLensOrg/TransformerLens}}}`.
- The sentence now reads "TransformerLens 3.9 (Nanda and Bloom, 2022)"; the version number was added during the
  review.
- That version is consistent with the project: `uv.lock` pins `transformer-lens ~=3.9.0`, and 3.9.0 is installed in
  `.venv`.

## `check_bib.py` (task 5)

The command `.venv/bin/python research/paper/tmlr-build/check_bib.py research/paper/draft-v3-2026-09-30.md` printed:

```
checked 63 references and 63 in-text forms: PASS
```

Exit status was 0. It also passed on the 09:48:59 version.

## Rendered bibliography (task 6, main.pdf pp. 13–18)

- **Accents** are correct throughout: João, Gül, Adrià, Costa-jussà, Gállego, Kramár, Méloux, François, Vukelić, Iván,
  O'Brien.
- **Proper nouns** keep their capitals: GPT-2, Pythia, Pythia-70m, LLM, LLMs, AtP*, CausalGym, RASP, MIB, SymTorch,
  Alpaca, AI, NLP, TransformerLens. BERT doesn't appear in any title.
- **Names:** "T. Ben Thompson" matches the Circuit Tracing byline.
- **Sorting** is correct.
- **Fields:** nothing is garbled. The only remaining cosmetic items are N2 and N3.

## Entries verified, and their sources

- **arXiv API**, in one batched call for all 50 IDs. Every entry with an arXiv link matched on authors, title and year,
  or the difference is explained above.
- **ACL Anthology .bib,** 13 entries, which match field for field (author, title, booktitle, year, pages, publisher,
  address, DOI):
  - arora2024causalgym, bayazit2026crosscoding, brinkmann2025large, ferrando2024similarity, ferrando2023explaining,
    finlayson2021causal, hao2023verb, kryvosheieva2026different, lasri2022probing, li2026model, mcdougall2024copy,
    syed2024attribution, vanmiltenburg2021preregistering.
- **PMLR:**
  - these match: bertinetto2021preface (v148), biderman2023pythia (v202), geiger2024finding (v236), joshi2026causality
    and vaccaro2026preregister (v306);
  - these need additions: anani2026certified and huang2026discovering (S3, S4), zhang2026weights (S2),
    weiss2021thinking, rushing2024explorations and mueller2025mib (S6).
- **Transformer Circuits citation blocks:** ameisen2025circuit and elhage2021mathematical match verbatim.
- **Other sources:**
  - Alignment Forum GraphQL: chan2022causal matches.
  - TransformerLens README: nanda2022transformerlens matches.
  - JMLR: geiger2025causal needs S5.
- **OpenReview search, for venues:**
  - ICLR: wang 2023; hernandez, makelov and niu 2024; marks and meloux 2025; hadad 2026. Grant at ICLR 2026 is S1.
  - COLM 2024: hanna.
  - NeurIPS: shi and gross 2024; sutter 2025.
  - TMLR: bayatmakou.
  - ICML 2026: zhang.
  - Workshops only, so the arXiv citation stands: li2025interpretation, and meloux2026validating.
- **NeurIPS proceedings:** seven entries, listed in N6.
- **Still preprints.** These 16 have no archival version in OpenReview, the PMLR v306 index, or the ACL Anthology
  volumes for ACL 2026, EACL 2026, EMNLP 2025 and BlackboxNLP 2025:
  - kramar2024atp, michaud2024opening, wu2024reply, li2025interpretation, tan2026symtorch, darade2026represented,
    hayes2026explaining, lin2026mechanistic, somani2026towards, mccann2026polymorphism, meloux2026validating,
    gong2026conditional, buehler2026reading, dong2026kinetics, xun2026lagged, zhu2026reading.

## Could not reach, or unconfirmed

- **DBLP** serves a bot-check page. It was not bypassed; OpenReview, PMLR and the ACL Anthology were used instead.
- **Semantic Scholar and GreaterWrong** were avoided, as instructed.
- **OpenReview's per-note endpoint** returns 403 to anonymous requests. Its search endpoint returned the full records
  instead.
- **NeurIPS 2025 main-track proceedings** aren't posted yet. So Sutter et al.'s volume and pages are unconfirmed, though
  the venue itself is confirmed on OpenReview.
- **ICLR and COLM** have no volume or page numbers to confirm.

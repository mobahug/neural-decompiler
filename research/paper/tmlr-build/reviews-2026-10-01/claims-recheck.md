# Short check: cleanup findings applied, draft v3 TMLR build (rebuilt PDF)

The claims reviewer's verification of its first review ([claims.md](claims.md)), returned to the session on 2026-10-01
and kept as returned, with its first-person wording put in the third person.

**Verdict: PASS WITH NOTES.** Every item of the first claims review is fixed, and the later changes leave every claim
intact. One SHOULD remains, in the new anonymous Appendix D: its record-caveats sentence claims more than §3 and Table
A2 contain. The wording it took came from that review's own Note 2, which was too loose. One NOTE remains on provenance
wording.

The check was read-only. Nothing in the repository was changed, and nothing under `outputs/` was read.

## Checks
- **Review copies:** `draft.diff` equals the live diff against 3d47ee9. `main.tex` and `main.pdf` equal the live build.
  Re-running md2tex.py on the live draft reproduces `main.tex` byte for byte.
- **Checkers:** `check_source_map.py` exits 0 (27 quotes, 235 citation groups, 2 commit hashes, 43 labels).
  `check_bib.py` passes (63 references, 63 in-text forms).
- **Draft against LaTeX, word by word:** the only differences are the intended ones: the anonymous Appendix D, the
  nested quotes, Table A1's repeated header and the Figure A1 notes. The model revision hash breaks across a line but
  stays intact.
- **Old PDF against new PDF:** every changed word is explained. The changes are Appendix D, the status list, the
  references, the vector text of the F8 print variant (37 cells, plus "no such item", matching the screen figure) and
  line-break changes in Table A1 and Figure A1.

## The first review's items: all fixed
- **S1 and S2:** p. 28 is coherent. The PDF text and `main.tex` contain no hash, no repository path or link, and no
  "mobahug".
- **S3:** p. 4 now prints "• [prospective]: …" and the other four with brackets and bullets.
- **Note 1:** the PDF now closes with ’”, from the source `'{}''`.
- **Note 4:**
  - §2 reads "TransformerLens 3.9 (Nanda & Bloom, 2022)", which matches the plan's "TransformerLens 3.9.x" (plan:9).
  - The reference renders as "A reply to Makelov et al. (2023)'s ‘interpretability illusion’ arguments".

## Anonymous Appendix D
- **"The record" and "The audit trail"** paragraphs are supported. They say what the Markdown Appendix D says, without
  the hash and paths. The checker does check the quotes and traces mechanically.
- **"The figures"** paragraph is supported. The figures README records the two independent reviews, of provenance and of
  drawing, and byte-identical regeneration. The 2026-10-01 layout review confirms that the print variants draw exactly
  what their screen figures draw. The new F8 print variant came after that review; a text check shows it carries the
  same cells.

### SHOULD: "the ones that bear on outcomes or provenance are stated in §3 and Table A2" is not fair as worded
The source map classes its caveats as provenance, outcome, interpretation or cosmetic (source map@b722e9c:615–742).
Several items in its provenance and outcome classes are in neither §3 nor Table A2:
- 005, provenance: C002 was created after the confirm, at `LOCALIZED`, rather than at the lock (L663).
- 006, outcome: the label `QUALITY_GATE_FAILED` is not in the design's outcome list (L666).
- 011–013, provenance: some README figures are not in the committed reports (L674).
- 018, outcome: the design fixes no reading for a Y2 guard failure (L683).
- 018, provenance: a README commit sits between commits the README says are adjacent (L685).
- 024 and 025, provenance: their statements about binary artifacts (L693, L695).
- The two digest mismatches, which the map explains (L642–652).
- The six lock commits with misleading subjects (L700ff).

§3 and Table A2 do state the weightiest items:
- the erratum;
- the frame-conditional tables that were not committed before scoring;
- the gaps in review records;
- the restatements after closure.

The first review's Note 2 said "the caveats with outcome or provenance weight are already in the paper". That was too
broad, and it is the source of this wording.

**Corrected wording** (in md2tex's anonymous Appendix D): "**Record caveats.** The record lists its own caveats: an
erratum, readings narrowed or restated after closure, two digest mismatches it explains, inconsistencies between its
records and misleading commit subjects. In its words, none of them "changes a frozen result unless its entry says so"
(source map@b722e9c:617). The erratum, the frame-conditional tables that were not committed before scoring, the gaps in
review records and the restatements after closure are stated in §3 and Table A2; the full list is withheld with the
repository during review."

## Later changes
- **Grant et al.:** consistent throughout. §8 cites "(Makelov et al., 2024; Grant et al., 2026)". The draft list,
  `references.bib` (`grant2026addressing`, ICLR 2026) and `citekeys.json` all agree, and the PDF prints both.
- **Venue changes:** consistent between the draft list and the bib. They are:
  - Anani, Huang and Zhang: PMLR 306, with their pages;
  - Mueller: PMLR 267;
  - Rushing: PMLR 235;
  - Weiss: PMLR 139;
  - Geiger 2025: JMLR 26(83), pp. 1–64;
  - Chan: the Alignment Forum link.
- **Unchanged claims:** no body sentence depends on these venues, and the §8 and §9 claims that cite them are unchanged.

### NOTE: two provenance statements are now out of date
**Where.** These sentences no longer hold for Grant, Zhang, Chan and the new PMLR and JMLR details:
- the draft's reference-list lead-in, "the literature review holds each entry's verified record";
- the tmlr-build README's "Titles and venues follow the draft's list, where the literature review verified them",
  together with its source list ("PMLR for Bertinetto et al. (2021), Geiger et al. (2024) and Pythia").

**What the record says.** The literature review lists Grant as "(2025) … arXiv" (review-2026-09-28.md:793) and Zhang as
"arXiv preprint (revised May 2026)" (L860–861). It read Chan through the GreaterWrong mirror (L597–600).

**Fix.** Say that venues, volumes and pages, and Grant et al.'s year, were updated from the publishers' records on
2026-10-01, and extend the README's source list (OpenReview for Grant, PMLR for the ICML entries, JMLR). This is
internal only: md2tex drops the reference list's lead-in from the PDF.

**Still open, as the draft's header already records:** the AI footnote's "the author" and its sentence on citations.

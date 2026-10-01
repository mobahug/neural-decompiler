# Claim-strength check: the submission-build cleanup of draft v3

**Verdict: PASS WITH NOTES.** There are no blockers. The new §2 sentence and its trace hold. The conversion keeps every
claim, status tag, qualifier and number in the body, the captions, the tables, Box 1, the AI footnote and the
appendices, with two exceptions (S1 and S3). Three SHOULD items should be fixed before the build is submitted:
- **S1:** a sentence garbled by the conversion;
- **S2:** an anonymity exposure;
- **S3:** the status definitions lost their brackets.

Four notes follow. The review was read-only. Nothing under `outputs/` was read, and nothing in the repository was
changed. Tools and outputs stayed in the scratchpad:
- Python ran only as the prescribed `.venv` interpreter.
- Three small read-only PDFKit tools were compiled with `swiftc`, for text, metadata and links.
- md2tex.py was run once, writing only to the scratchpad.

## Checks run

- **Inputs.**
  - `draft.diff` equals `git diff 3d47ee9` of the live draft.
  - Re-running `md2tex.py` on the live draft reproduces `build/main.tex` byte for byte.
  - `build/main.tex` and `build/main.pdf` are identical to the review copies.
  - The checker exits 0 on the live draft: 27 quotes, 235 citation groups, 2 commit hashes, 43 labels.
  - CommonMark: 235 comments, all hidden, apart from the intended code span in Appendix D.
- **The new §2 sentence** holds and does not overclaim.
  - INSTRUMENTATION.md:3–6 says the instrumentation is "Pythia-first and TransformerBridge-first".
  - INSTRUMENTATION.md:153–156 says each run records the "TransformerLens and dependency versions, compatibility mode,
    `use_attn_result`".
  - The plan's line 9 names "TransformerLens 3.9.x `TransformerBridge`".
  - `git grep` at d7c5e62: every experiment from 005 to 025 imports the shared `neural_decompiler` package, whose
    models.py and capture.py use TransformerLens.
  - 005's README says that every measurement "runs through the instrumented forward".
  - The Pythia and TransformerLens entries match their published records (ICML 2023, PMLR 202, pp. 2397–2430; the
    TransformerLens repository).
- **Word-by-word comparison.** The draft's visible text, with comments, header and reference list removed, was compared
  against main.tex's text, with the citations mapped back through citekeys.json. Every difference is formatting:
  - LaTeX numbering in place of the "Figure N." and "Table N." prefixes;
  - table rules;
  - the two nested quotes;
  - the References heading;
  - the two added Figure A1 notes.
- **Numbering.** LaTeX's numbering matches the draft's (from main.aux): §1–§10, §4.1–4.5, Appendices A–D, Tables 1–4, A1
  and A2, Figures 1–5, A1 and C1–C3.
- **Glyphs.** The log shows no missing characters. ≥, ≤, −, Δ, κ, Π, ⊙, γ₃, d̂, ∥, ρ and τ are all present in the PDF
  text.
- **Pages viewed:** 1–4, 6–10, 22 and 25–27.
  - Table 3's status cells are intact, including "[pooled; new frames alone conditional]" and the cells that hold only a
    qualifier.
  - Box 1 keeps its four status tags.
  - §6's quotes and the draws sentence are intact.
  - §7's boundary sentence and 025's wording are intact.
  - The captions of Figures 3, 4 and 5 and of Figures C1–C3 are intact.
  - Figure A1 is split correctly. Part 1 holds rows 005–015 and ends "This part shows 005–015; the second part
    follows." Part 2 holds rows 016–025 and ends "Figure A1, continued: 016–025."
  - The print variants carry the screen versions' text and statuses.
- **Bibliography.** All 63 entries are present, and their venues match the draft's list entry for entry.
- **Anonymity.**
  - Page 1 reads "Anonymous authors / Paper under double-blind review".
  - No author name appears in the text, the figures or the metadata. The PDF info has no Author or Title, the figure
    PDFs carry Matplotlib metadata only, and the PNGs have no text chunks.
  - The repository link is replaced by "[repository link removed for anonymity]".
  - The link annotations point only to arXiv, ACL, PMLR, Transformer Circuits, the Alignment Forum and TransformerLens's
    GitHub.

## Findings

### S1. SHOULD: Appendix D's audit-trail sentence is garbled

**Where.** PDF p. 28 and main.tex:422, from draft L904–907.

**What the build does.**
- md2tex strips every `<!--…-->`, including the code-span example `<!-- (023:142) -->`. That leaves an empty
  `\texttt{}`.
- The PDF reads: "as HTML comments that do not render: is experiments/023-*/README.md line 142 at d7c5e62, (path:L) is
  that file at d7c5e62…".
- The sentence also describes traces that the anonymous build has removed.

**Fix.**
- Remove comments outside code spans only.
- Better, give the anonymous build its own wording: "The Markdown source of this paper keeps, for every statement that
  rests on the record, a trace to it as a hidden comment; for example `(023:142)` names line 142 of experiment 023's
  README at a pinned commit. A checker verifies the quotes and traces mechanically."

### S2. SHOULD: the anonymous build can still be traced to the repository

**Where.** PDF p. 28 (`d7c5e62`, twice; the paths `experiments/023-*/README.md` and
`research/synthesis/checks/check_source_map.py`).

**What the record says.** `git branch -r --contains d7c5e62` gives origin/main, and the origin is
github.com/mobahug/neural-decompiler. If that repository is public, GitHub's commit search resolves the hash to it. So would a code search for
the verbatim outcome labels the paper prints, such as `HEAD_MECHANISM_CONFIRMED_P1`.

**Fix.**
- Replace the hash and paths in the anonymous Appendix D with "a pinned commit" and generic descriptions.
- Keep the repository private, or provide an anonymized copy, for the review period.

### S3. SHOULD: the five status definitions lost their brackets

**Where.** §3, PDF p. 4; main.tex:68–72, from draft L187–191.

**What the build does.**
- LaTeX reads `\item [prospective]: …` as `\item[prospective]`.
- The PDF therefore prints "prospective : passed a preregistered criterion…" as hanging labels, with no brackets and no
  bullets, and "locked prediction" runs into the left margin.
- These are the definitions of the bracketed tags the paper uses throughout.

**Fix.** In md2tex's `item_lines`, emit `\item{}` (or `\item\relax`) before any item text that begins with "[".

### Note 1: one nested quote closes in the wrong order

**Where.** PDF p. 9; main.tex:233.

**What the build does.** The source has `` `basically a pass''' ``. TeX's ligatures render this as "‘basically a
pass”’", with the double closing quote before the single one.

**Fix.** Write `'{}''` or `\textquoteright\textquotedblright`. The opening sequence, "“‘Three…", is already correct.

### Note 2: the record caveats are out of reach in the anonymous PDF

**Where.**
- §8 (draft L557): "The record's own caveats apply (Appendix D)".
- Appendix D (L914), which now prints a dangling "(record caveats)".

**What the record says.** The caveats with outcome or provenance weight are already in the paper, so nothing with
outcome force is lost:
- §3 says the frame-conditional prediction tables were not committed before scoring.
- Table A2 records the review-record gaps and the restatements made after closure.

**Fix.** Say where the list will be available, such as the supplementary material or the de-anonymized repository, or
drop the pointer in the anonymous build. The same applies to "the figures' README" in Appendix D.

### Note 3: the AI footnote's two open items remain

**Where.** PDF p. 1.

**What remains.** These are unchanged, and the header still schedules them:
- "the author" reveals a single author in the anonymous build.
- "literature citations were checked against their original sources" overstates the literature review's verification,
  which checked every link and bibliographic record but only spot-checked the key paraphrases
  (review@57d6450:1341–1342, 1358–1360).

### Note 4 (optional): versions and one bibliography entry

- **§2's model sentence (draft L145–148)** could name the version, "TransformerLens 3.9's TransformerBridge" (plan:9),
  for reproducibility. The sentence is accurate as it stands.
- **Wu et al. (2024)'s title prints "makelov" in lower case**, with two right quotes around 'interpretability illusion'.
  Brace `{Makelov}` in references.bib and use TeX opening quotes.
- **Layout:** §2's first line is stretched by the revision hash (an underfull hbox), and the Appendix C heading is
  orphaned at the foot of p. 24. Both are for the layout pass and have no effect on claims.

## Checked and correct

- **Front matter:** the title, the abstract and the AI footnote are verbatim. The "(In the TMLR build…)" stage direction
  is removed, as intended.
- **Body and appendices:** §1–§10 and Appendices A–D are verbatim, apart from S1 and S3.
- **Restored scope and qualifiers:** "(§2, Table 3)" in §1; all of §6, including C4's "passed as a positive
  contribution" and Note 1's quote; §7's routes and boundary.
- **Tables 1–4, A1 and A2:** every cell. Table A1 is a longtable with a repeated header.
- **Citations:** in-text citations render, including multi-year ones ("Geiger et al., 2021; 2024"), compound surnames
  (Bayat Makou, van Miltenburg) and accents (Méloux, Kramár, Costa-jussà).
- **The build README's statements match the build:** the variant assignments, the 63 entries, page 13 for the References
  and 28 pages in all.

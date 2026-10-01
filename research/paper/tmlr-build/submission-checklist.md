# TMLR submission checklist, 2026-10-01

What the submission of [draft v3](../draft-v3-2026-09-30.md) to TMLR needs, from TMLR's FAQ, author guidelines,
submission page, editorial policies, reviewer guidelines and ethics guidelines as read on 2026-10-01. The record of
that reading is in the [AI-use audit](../ai-use-audit-2026-10-01.md), §7. Each item has a state: *done*, *to do*
(with whose act it is), or *optional*.

## The PDF

- **The official template, unmodified.** *Done.* The build fetches `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` and
  `math_commands.tex` from JmlrOrg/tmlr-style-file at a pinned commit and does not change them
  ([README](README.md)). The submission page says changes that alter the formatting, font or layout may lead to
  rejection without review.
- **Compiled with pdfLaTeX.** *Done.* The template targets pdfLaTeX. The final PDF is compiled with `ENGINE=pdflatex
  ./build.sh`, which uses TinyTeX 2026.10 (TeX Live 2026, pdfTeX 1.40.29). It matches the Tectonic build page for
  page.
- **Anonymous.** *Done.* The front matter reads *Anonymous authors* and *Paper under double-blind review*. The PDF's
  author and title fields are empty. Appendix D describes the record without the repository, its commits or its
  paths.
- **The statement on AI tools.** *Done.* It is a first-page footnote, as the FAQ requires, and it is the statement
  the [AI-use audit](../ai-use-audit-2026-10-01.md) derived from the record and the author's answers (§9). §3 says
  who reviewed at each gate.
- **Length.** *Done, at about 12.5 pages.* The main text runs to about 12.5 pages before the References. TMLR sets
  no page limit, but papers whose main body exceeds 12 pages get four weeks of review instead of two. The appendices
  come after the References, outside the main content.
- **Figures.** *Done.* The figures whose text would print too small use print variants drawn at the text width
  ([figures' README](../figures/README.md)).
- **Figure fonts.** *Optional.* The print variants embed their text as Type 3 fonts, matplotlib's default. TMLR's
  pages set no font rule. Setting matplotlib's `pdf.fonttype` to 42 would embed TrueType instead.

## Anonymity outside the PDF

- **The repository.** *To do, by the author.* The project's GitHub repository is public. The records cite its
  commits, and the paper prints outcome labels that a code search could find. It should be private, or replaced by an
  anonymized copy, for the whole review. Only the account owner can change this.
- **Preprints.** *To note.* TMLR allows a preprint, anonymous or not, but the submission must not link to a version
  that names the authors. A preprint on arXiv should use the template's *preprint* option.

## Supplementary material

- **The ZIP.** *Done; the author uploads it.* `supplement.py` builds `build/supplement/supplement.zip`, about 85 MB
  against TMLR's 100 MB limit. It holds every tracked file except 12 binary arrays, which it lists with their
  digests. It replaces identifying strings in 200 files, and the build stops if any identifying string is left.
  Supplementary material must be anonymized, as a PDF or a ZIP. Reviewers look at it at their discretion.

## OpenReview

These are the author's acts, on the OpenReview TMLR site.
- **Profiles.** Every author needs a complete OpenReview profile.
- **Declarations.** The submission form asks for human-subjects (IRB), funding and competing-interest information.
  Reviewers do not see it.
- **Action editors.** Authors suggest action editors.
- **Submission budget.** Each author must have yearly submission budget left.
- **Authorship.** No author can be added or removed after submission. TMLR's editorial policy rules out LLMs as
  authors.
- **Author contributions.** *Optional.* A statement of author contributions.

## Statements

- **Broader impact.** *Optional.* TMLR requires a broader impact statement only for work with a significant risk of
  harm, and its ethics guidelines expect a discussion of negative societal impacts where applicable. This study of
  how a 70M-parameter model computes noun number is unlikely to need one. A short statement would remove the
  question.

## After acceptance

- The de-anonymized version gives the true number of authors and the repository's address, and restores the record
  links in Appendix D.
- The camera-ready version may come with code. A tagged release of the record would serve.

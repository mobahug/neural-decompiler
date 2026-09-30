# Paper

Drafts of the write-up of Experiments 005–025. At step 3 of the [synthesis](../synthesis/README.md), on 2026-09-28,
the project's reviewer decided to write the paper now, before any new causal experiment. Whether the one named causal
test is worth running is to be decided only after a full draft exists.

- [Draft v0](draft-v0-2026-09-28.md) — internal, not for circulation. It is structured by the research question, not by
  chronology. Every number in its body carries a traceability citation to the committed records; the abstract and the
  conclusion summarize the body.
- [Draft v1](draft-v1-2026-09-29.md) — internal, not for circulation. It replaces v0's traceability markers with
  references: published work by author and year, with a bibliography, and the project's record by experiment and
  claim. The line-level trace to the record is kept in the source as HTML comments, which do not render and which the
  checker validates. Its relation to prior work is completed from the literature review, and its contributions follow
  the reviewer's decision of 2026-09-29: the prediction-versus-causation boundary is no longer a separate contribution,
  and stays in the abstract, §7 and the conclusion.
- [Draft v2](draft-v2-2026-09-30.md) — internal, not for circulation. It is draft v1, revision 2, with its figures:
  Figures 1–8 are embedded where the text first relies on them, each with a caption traced like the body, and Appendix
  B lists them. Revision 1 applies the paper-level claim-strength review.
- [Draft v3](draft-v3-2026-09-30.md) — internal, not for circulation. Draft v2, revision 1, restructured for TMLR, the
  venue the reviewer chose on 2026-09-30, along the paper-level review's outline: an introduction and a one-paragraph
  abstract, a statement on AI tools, a worked stage program, a table of exact and reduced steps, one home for each
  result, and the chronology, supplementary figures and audit trail in appendices. It adds no claim, and no number that
  the record does not state. Its first build had three independent read-only reviews, applied before it was committed
  ([reviews of draft v3](paper-review-v3-2026-09-30/README.md)).
- [Paper-level review of draft v2](paper-review-2026-09-30/README.md) — three independent read-only reviews: claim
  strength, applied in v2's revision 1; narrative flow and redundancy, whose outline the reviewer adopted for draft v3;
  and submission format, after which the reviewer chose TMLR.
- [Reviews of the drafts](reviews.md) — the independent read-only reviews and what was done about each finding.
- [Literature review](literature-review/review-2026-09-28.md) — the first step of v1 preparation: what in the draft is
  new relative to existing work, under a [protocol](literature-review/protocol-2026-09-28.md) frozen before any search.
  Its [verification record](literature-review/verification-2026-09-28.md) archives the read-only checks of every cited
  work and of the review's logic.
- [Figure-data audit](figures/figure-data-audit-2026-09-30.md) — read-only: every value the planned figures F1–F8 would
  plot, traced to committed evidence, with its status and a verdict per panel. No figure is rejected; two need a
  record-supported redesign, F1 and F3's panel A. It also lists corrections for the next draft revision.
- [Figures](figures/README.md) — F1–F8, drawn only from what the audit supports, by scripts that read every value from
  a pinned commit and check it against its source. The README gives the provenance, the status colours, a draft caption
  for each figure and where the figures go beyond the audit's plan.

No new experiment is planned.

The drafts claim nothing beyond the registered claims, [C002](../claims/C002-count-cued-noun-number-circuit.md) and
[C003](../claims/C003-frame-conditioned-decoded-program.md). Where a draft and a record disagree, the record wins.

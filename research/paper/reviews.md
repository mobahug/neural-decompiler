# Independent reviews of the paper drafts

## Draft v0

### The first review, of revision 1: FAIL, four blockers

It found every number in both tables and every experiment-README citation correct, and the checker passing. The
problems were framing, completeness and three misstatements. Its findings, condensed, with what was done in revision 2:

#### Blockers

1. **The causal routes were attributed to the decoded variables.** Three of the four preregistered routes concerned
   C002's localized components; only 025's rotation concerned a decoded direction. *Done:* the abstract, §7 and the
   conclusion split the routes that way. §7 adds that apart from 025, no preregistered, outcome-bearing test intervened
   on the chain's decoded variables, quoting C003.
2. **The abstract said three of four conditions held.** 023 has no aggregate label, and its ceiling is not an upper
   bound. *Done:* the abstract names the three cue classes, the conditional new frames, the `ENVELOPE_ONLY_FAILURE` and
   the missing aggregate label, and calls the ceiling the readout's own.
3. **The completed program was said to use 018's neuron subset.** It does not: block 2's MLP enters in full, through the
   frozen 012, 015, 016 and 017 programs. *Done:* §4 corrects its inputs (022:43–45; 018:16).
4. **The protocol was described more strongly than the records.** *Done:* §2 describes a typical sequence from 009 on,
   with named exceptions:
   - 006, 008, 010 and 020 had no lock or confirmation;
   - 005 ran two protocol versions, with an amendment after inspection;
   - the floors were frozen constants in 013–019 and calibrated in 021–024, and 025 had no calibration phase;
   - reviews gated every phase only from 021 on.

   The incident example is corrected, errata and restatements are separated, and *no confirmation run was repeated*
   replaces *no scientific retries*.

#### Should-fix and notes

- **C002 and the abstract.** C002's basis and two-pair scope are in the abstract and the conclusion. The abstract
  answers the question attribute by attribute. It carries C003's origin qualifiers, the tested lexical classes, and
  021's pooled margin and coordinated failure on the later-corrected wiring.
- **Wording.** The explicit-mechanism, decoded-inside and predict-well wordings are replaced by C003's.
- **024 and 020.** 024 is stated as a rank association in exposed frames, with its guard's scope and the frozen
  approximation. 020 carries its six-of-twelve floor shortfall, its calibration-record status, and *end-to-end program*.
- **Nulls and negatives in the body.** The `MIXED` summaries and the failed relay bound are in §4. §5 lists all four
  failed compact programs, and 018 Y4 beside 017 Y3.
- **§8.** The recomputation alternative carries C003's substance, and the baselines are scoped. The rerouting
  alternative is added.
- **§3.** It is tagged exploratory and calibration, with C002's limitations, S1's floor and the lexicon's own citation.
- **023 and Appendix A.** 023's failed condition is tagged unestablished at its envelope, paired with *neither simply a
  failure nor basically a pass*. Appendix A's statuses follow the source map's overview in full.
- **§1 and notation.** §1 defines frames, the reference cue, `Δc`, exposed and fresh cues, and the cue classes. A
  notation block is added to §2.
- **§9.** It uses the assessment's candidate wording, and keeps its sources' qualifiers and C002's *Confirmation was not
  achieved*.
- **Minor fixes:**
  - the paper-readiness citations carry a commit and lines;
  - 018's no-harm citation is extended;
  - the conditional qualifier applies from 013 on;
  - 012 is described as the net correction on exposed base states;
  - the chain's readout positions are fixed;
  - the status legend matches the tags;
  - the title no longer reads as a completed decompilation;
  - the README's traceability claim is scoped to the body, and it records who decided step 3 and when.

### The re-review, of revision 2: FAIL on one residual blocker

It found blockers 2, 3 and 4 resolved, and blocker 1 partly resolved: the abstract and §7 were fixed, but the
conclusion still tied all four routes to the programs' variables. Every other item was at the PASS WITH NOTES level.
What was done in revision 3:

- **The residual blocker.** The conclusion now says that the one preregistered intervention on a decoded direction,
  025's rotation, did not establish its effect. It also says that the three preregistered routes to C002's causal gates
  did not establish them, so C002's necessity, sufficiency and control gates are open.
- **Should-fix S1.** The abstract's 021 bullet names the pass's scope, new cues and nouns in exposed frames, and the
  negative new-frame label.
- **Should-fix S2.** Appendix A's status column is now copied from the source map's overview, with 011's measured-input
  qualifier added. The first review's disposition had said revision 2 followed the overview in full; it did not, and
  this revision corrects that.
- **Notes.** Also done:
  - the protocol's exceptions now include the missing exploration phase of 021–025 and 009–012's Tier-A tolerances;
  - the reviews bullet says that review gates are stated from 021 on and that committed records start at 023;
  - the end-to-end quantity `Δc` is distinguished from the E-patch's intermediate changes;
  - the methodology sentence allows a substituted causal test with a rationale;
  - the no-harm guard carries its 0.05 margin, and `g`'s rule applies to interpretable conditions;
  - the completed program's locked inputs are cited to 020's inherited locks;
  - 020's coordinated-shortfall reading is qualified by the erratum;
  - the fresh-set counts are cited;
  - C002's scope carries its templates, its tested replacement distributions and at the pair level;
  - there are four rivals, and the named test uses the assessment's wording and citation;
  - the related-distinction lead is noted;
  - the tag legend covers every tag used.
- The checker's per-row label check skips Appendix A's short rows. The re-review checked those labels by hand, and
  every one occurs in its own README.

After revision 3, the checker passes on the draft: 23 quotes and 140 citation groups, every quote exact.

## Draft v1

One independent agent reviewed v1, read-only, against v0, the verified literature review and its verification
record, the protocol, the claims C002 and C003, and the checker change. It loaded no model and changed nothing in the
repository.

### The review: PASS WITH NOTES, no blocker

- **Verified as correct.**
  - **Conversion.** v0's 140 citation groups became 137 comments, in order and at the same statements. Exactly the
    three §9 citations that went with the rewritten §9 were dropped, and one quote became a paraphrase with its
    citation.
  - **Content.** No number, label, status tag or qualifier in the body changed.
  - **Visible changes.** Every visible change is an attribution, the header or new material.
  - **Rendering.** Checked with a CommonMark renderer: every comment stays hidden, and the tables, the code fence and
    the nested lists are intact.
  - **References.** All entries match the literature review's verified records, and every citation resolves.
  - **Contributions.** They follow the reviewer's decisions.
  - **Checker change.** The old and new checkers agree on all 75 tracked Markdown files under `research/` and `docs/`.
- **Should-fix,** eight, all done:
  1. Contribution 2 said a pretrained model; it now says a pretrained language model, as the protocol states it.
  2. §7's principle regained the qualifier *under stated assumptions*.
  3. §8's self-repair bullet now says that C002 scopes its necessity and isolation conclusions to pair-centered
     neutralization, and that its necessity gate is open in any case.
  4. §8 no longer implies that the E-patch is exempt from off-distribution concerns. It says the E-patch is not a
     subspace intervention, and that it too builds a state that need not occur on natural inputs.
  5. Méloux and colleagues (2026) are characterized as in the review, in §8 and §9.
  6. Bayazit and colleagues' feature is described as annotated, as in the review.
  7. Weiss and colleagues are no longer credited with compiled programs.
  8. §9's list of what the work does not add gained predictors conditioned on a measured reference state.
- **Notes,** all done except where stated:
  - the header's three ways;
  - the audit-trail sentence limited to kept statements;
  - a trace for the count of experiments;
  - the exact-computation caveat in the abstract and the conclusion;
  - the conclusion's *from the tested lexical classes*;
  - the contributions' sentence on precedents;
  - §9's hedges and dropped qualifiers restored;
  - superlatives hedged with *found*;
  - §9's lead sentence for what the work adds;
  - two works on circuits with guarantees and one on agreement units added from the review;
  - traces for the new §8 statements;
  - 005's compensation ratio and its missed band added for balance;
  - the record entry, two ICML page ranges and arXiv titles, and the mirror's provenance in the references;
  - the checker's docstring made exact.

  The reviewer also suggested that the checker catch an unclosed comment and a comment that begins a line. The checker
  now does both, and its output on the 75 tracked files is unchanged.

After these changes, the checker passes on v1: 22 quotes and 142 citation groups. The five new groups are the traces
added for the count of experiments and the new §8 statements.

### Revision 2 and the record corrections: verification PASS WITH NOTES, no blocker

Revision 2 applies the fifteen wording corrections of the
[figure-data audit](figures/figure-data-audit-2026-09-30.md). A separate documentation-only commit corrects the same
021 wording in C003, the synthesis, the readiness assessment and the literature review. One independent agent verified
both, read-only.
- **Verified:**
  - All fifteen corrections are applied and match the record, including the four tokens with `either` noted as
    different, and 021's Y1 with exposed nouns and the separate Y3.
  - No existing number changed. Every added number matches its source.
  - Appendix A is identical to the source map's overview in all 21 rows, and Appendix B matches the audit's verdicts.
  - The record corrections are exact and documentation-only. C003 keeps its registered wording, quoted with provenance
    in its status history, and its maturity, disposition, evidence and gates are unchanged.
- **Should-fix,** two, both done:
  1. §9's new status tag covered the whole bullet, including the prospective read-out. It now sits on the off-axis part
     only.
  2. C003's new row named the wrong section; the exposed nouns are stated in its scope.
- **Notes,** done:
  - the diagram note now says the layer-0 MLP runs in parallel, not before attention as drawn;
  - the 019 row's notation is defined;
  - the 0.080 margin names its metric;
  - Y3 is named;
  - F6's values carry their outcomes;
  - F7 adds the never-run chips of 006 and 020;
  - Appendix A's 021 note cites the wiring's line;
  - the header mentions Appendix B;
  - the `P1` note covers every label with P1;
  - C003's quote runs to the end of the phrase;
  - the literature review's correction note is exact about what changed;
  - the paper README counts two redesigns, not three.

  One optional clarification in the synthesis was left as is, because the verification judged its wording accurate:
  it lists new cues and new nouns as the union of Y1 and Y3.

After these changes, the checker passes on every edited file.

## Draft v2

Draft v2 (2026-09-30) is draft v1, revision 2, with Figures 1–8 embedded and captioned. It had a paper-level review in
three independent, read-only parts, each by its own reviewer. The full reports are in
[`paper-review-2026-09-30/`](paper-review-2026-09-30/README.md); their line numbers refer to draft v2 before revision 1.

### Claim strength: PASS WITH NOTES, no blocker

- **Six should-fix items and fourteen notes,** all wording or trace fixes, applied in revision 1:
  - 023's frozen reading is quoted from its preregistration; the word *small* is now attributed to its closure notes;
  - §9's findings carry their statuses, and the block-2 and block-0 findings their scope;
  - *the mechanism* became *a candidate mechanism* or *one mechanistic decomposition*;
  - the abstract's localization names its two cue pairs and three templates;
  - the protocol's precedents follow the literature review: several 2026 studies, not one;
  - Figure 3's *not committed* is traced to the figure-data audit.
- **Notes, done:**
  - the locked subset's scope, new cues in exposed frames;
  - `every`'s small on-axis part and 008's `AXIS_ARTIFACT`;
  - status tags on the transport heading and on 007's replication, whose fresh frames lie beyond C002's declared frames;
  - κ described as a closure of channel D's contribution;
  - 025 as a rotation along a decoded direction;
  - the outcome-bearing tests listed with 024's and 025's;
  - 024's binding threshold, a permutation null;
  - 021's negative new-frame label, from the coordinated condition alone;
  - the captions of Figures 1, 5, 6 and 7;
  - C003's maturity, `LOCALIZED`;
  - the rivals counted as five tests of four accounts;
  - the conditional qualifier dated from 013;
  - byte-identity tied to matplotlib 3.11.2.
- After revision 1 the checker passes: 32 quotes, 196 citation groups, 43 labels.
- **Verification of revision 1: PASS WITH NOTES.** Every item was applied correctly, and the edits add no overclaim.
  Six minor notes, done:
  - Figure 6's caption says that neither plotted value is a frozen condition of 021's Y1 or Y2;
  - §9 says what passed: three conditions of `g`, relative to the ceiling;
  - 008's rule is said to use a different axis measure;
  - the header counts one added value from the record;
  - a line left short by reflowing is joined;
  - the format summary gives cuts of about 40–45 %.

### Narrative flow and redundancy: eight major findings, pending a decision

- The draft reads as an audited record rather than an argument. The proposals are moves, cuts and definitions that
  change no number and no claim:
  - an introduction that says why the question matters and what was found, with the definitions in a setup section;
  - a one-paragraph abstract;
  - a worked example of one stage program, the encoding read, and a table of what each stage computes exactly from the
    measured state and what it reduces or locks;
  - one home per result: one extended table in §5; 023's table with Figure 6's panel A; Appendix A or Figure 7; Figure
    8 in an appendix;
  - each figure where its claim is made, without the titles and notes inside the images;
  - prose in place of most bullets, and the real-swap section framed as the protocol at work, from 020 to 023;
  - §9 halved, and a conclusion that follows the contributions;
  - about 6,150 words from the abstract to the conclusion, against about 7,500 now.
- Not applied: they restructure the paper, so they wait for the reviewer's decision.

### Submission format: depends on the venue, none recommended

- TMLR and *Computational Linguistics* fit without major cuts; venues with 8–9 pages need cuts of about 40–45 %.
- Needed for any submission:
  - remove the internal header, the hidden audit trail and the internal pointers;
  - rewrite the abstract as one plain paragraph;
  - disclose the use of AI agents in the protocol's reviews, the literature search and the reviews of the drafts;
  - make the figures legible in print, without titles inside the images;
  - number and caption the tables;
  - complete the bibliographic records, and cite the model and the library;
  - replace the paragraph on the project's record with a citable snapshot, and ship the audit trail as supplementary
    material.
- The venue type, the template and the target length come first; the report gives a conversion plan for each.

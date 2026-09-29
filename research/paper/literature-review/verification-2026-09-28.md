# Verification of the literature review, 2026-09-28

The [protocol](protocol-2026-09-28.md) requires a read-only verification pass over [the review](review-2026-09-28.md).
Five separate agents ran it. Each loaded no model and changed nothing in the repository. A usage limit interrupted all
five on 2026-09-28; each was resumed on 2026-09-29 with its context intact.
- **Source checks:** four agents, one per cluster. Each fetched every linked work, checked its title, authors, year and
  venue, and spot-checked the paraphrases, numbers and relation lines of the key entries.
- **Conformance review:** one agent. It checked the review's logic against the protocol, the draft, the claim files and
  the checker.

A sixth agent then re-reviewed the second revision (see the last section).

The findings below concern the first revision (1,183 lines). Every finding below was applied in the second revision
unless it says otherwise.

## Source checks

### Cluster A

Every link resolved, and every title matched.
- **Factual errors:**
  1. A4's effect was misstated. Suppressing the two neurons clearly shifts the relative probabilities of singular and
     plural forms, but accuracy on the benchmark paradigms falls by only 2.0–5.2 points.
  2. A13 was published at ACL 2026, not only as a preprint, and its year is 2026.
  3. The first author of the screened work on interchangeable circuits is Bayat Makou, not Makou.
  4. Guerner and colleagues' author order follows their current arXiv version.
- **Imprecision:**
  - A1 names boundary-detector features, not attention, as the carriers of number.
  - A3's validation covered more than agreement.
  - A4 is reverse-direction work: it predicts the determiner from the noun.
  - A9 compares residual streams with attention outputs, and localizes nothing to heads or neurons.
  - A12 annotates one plural-quantifier feature, early in pretraining.
  - A13's model sizes and localizer scope were corrected.
  - Venues and pages were added.
  - The screened works of Tigges, Africa, and Ryu and Lewis were reworded.
  - Titles were added for five screened works: those of Arora, Bayat Makou, Guerner and Gurnee with their colleagues,
    and Africa's.
- **Not verified:**
  - A8's venue: one search summary said COLM 2024, but the agent could not confirm it, so A8 stays a preprint;
  - whether A1's boundary detectors are attention features, since the figure is an image;
  - A12's plural-quantifier feature beyond the fetch tool's search of the page;
  - the magnitudes in A4's figure, which is an image; the accuracy drops come from its table;
  - the paraphrases of entries outside the key set.

### Cluster B

Every link resolved, and every author list, year and venue matched. The numbers of B4 and B6 were confirmed.
- **Factual errors:**
  1. B7's results were misstated. Only the two compiled circuits pass all three tests: the induction circuit passes two
     and the docstring circuit one, both in toy models, and the indirect-object and greater-than circuits pass none.
     On the graded tests, removing a published circuit does not always hurt more than removing a random circuit of the
     same size.
  2. B7's p-hacking caveat is generic, and is paired with a caveat about the fixed experimental setup.
  3. B11 covers five models, not four.
- **Imprecision:**
  - B1's graded faithfulness has several instances.
  - B4's head count differs between the arXiv version and the conference abstract.
  - B5 and B8: a full title, and a fairer paraphrase of the reply.
  - B6's perturbations hold attention patterns fixed.
  - B12 calls one metric family functional.
  - The screened work of Merullo and colleagues makes a qualitative prediction of an intervention's downstream effects,
    now reflected in candidate 2's covered list.
  - Titles were added for the screened works the verdicts cite.
  - B11's relation to candidate 4 now states its point.
  - McDougall and colleagues are cited under their published title and venue, and the Biology paper under its venue.
  - The guidance for v1 names the metric families B12 finds wanting.
- **Not applied (optional):** splitting the citation for the principle behind candidate 4 by source. The reworded
  principle already separates decoding and prediction from interventional success.
- **Not verified:**
  - whether the camera-ready body of B4 keeps its figures;
  - any passage of B7 beyond its impact statement on p-hacking;
  - the model behind B6's correlation;
  - that B10's second paper makes no claim about causal role, a claim of absence;
  - B3's authors and date, taken from the mirror;
  - the paraphrases of entries outside the key set.

### Cluster C

All 29 links resolved, and every title and author list matched.
- **Factual errors:**
  1. C12 does test alternative templates, in an appendix, and applies its maps to distracted prompts. Nothing is frozen
     on one frame and scored prospectively on new frames.
  2. Elhage and colleagues cannot be called the earliest source of rules read from weights. An earlier reading of
     feed-forward value vectors is now noted.
  3. The screened work of Buehler partly overlaps candidate 1: its endpoints and success rules were frozen before their
     outputs, and a failed registered endpoint is kept.
- **Imprecision:**
  - C1: two heads replaced, program-local calibration, 1,280 prompts, and an amended gate.
  - C2's exact overlap and perplexity figures.
  - C3's linearized LayerNorm and its partial success on formal languages.
  - C4's fitting and test data.
  - C6's wording.
  - C10's bridges, and its prediction of adversarial failures.
  - C13's failure modes, its steering-vector claim, a switched predictor, and the criteria scored on the toy.
  - The terminology of the gradient approximation of Kramár and colleagues.
  - The screened works of Dunefsky, Nettasinghe and Manning-Coe.
  - A typo.
  - Candidate 2's covered list now says which predictors concern internal effects and which the output.
- **Not verified:**
  - C6's appendices;
  - the Elhage paper beyond a summary;
  - whether C2's best programs are chosen on the data on which they are scored.

### Cluster D

Every link resolved, and every title and author list matched, except the two proceedings titles below. The key entries
were read in full text.
- **Factual errors:**
  1. D9 reports two audits, not one of 40 papers: 10 papers in full text, and 30 coded from abstracts and metadata.
  2. D8 appeared at ICML 2026, in the position-paper track, under a title with the prefix *Position:*.
  3. D4 appeared at ICML 2026 under the title *Position: Preregister Experiments with AI Agents*.
  4. Item 7 of the review's guidance for v1 was updated to match.
- **Imprecision:**
  - D2's hash files are claimed by the paper but were not inspected, and its original predictions were split.
  - D3's failure log, and its prediction limited to induction.
  - D7's research outputs.
  - D10 has no archival venue, and some predictive heads suppress the rule they predict.
  - The causal-test input's reading of the literature.
  - Cacioli's small models trained from scratch.
  - A pull-request link.
  - A nuance in MAPS.
  - Søgaard and colleagues, D11's workshop, Fukushima's local records and D1's null edits were reworded.
  - Titles were added for the screened works of Lee, Thomas and Cacioli, the excluded article and two unverified
    preprints, and all 11 arXiv numbers under *Unverified* were confirmed to resolve.
- **Not verified:**
  - D1's venue: two search summaries asserted one, and none of their sources supports it;
  - D9's position-track decision;
  - D2's hash files;
  - the paraphrases of D11 and D12.

## Conformance review: FAIL, two blockers

- **Checker:** passed with 2 quotes and 11 citation groups. Both quotes were exact, and every citation pointed at the
  stated lines.
- **Blockers:**
  1. **The remains of candidate 1 were stronger than the record.** It said each stage was confirmed, where the record
     says each was tested once and not every test passed. It dropped three parts of the protocol's statement:
     preregistered, frozen confirmation sets, and errata recorded after the fact. *Done:* the protocol's statement is
     restored, with the draft's exceptions and its sentence that not every test passed.
  2. **The remains of candidate 2 widened the protocol's statement.** It named an early encoding, cue words, input
     swaps, and no behavior or model class. *Done:* the cue's layer-0 encoding, cues from several lexical classes, real
     cue swaps, and a naturally learned behavior in a pretrained language model.
- **Should-fix,** 12, all done except one request of the depth fix: the sections read in the partial entries could not
  be named, because the search did not record them. Deviation 7 says so.
  - Candidate 2's covered list cites only entries whose own relation lines support it: C3 and C6. It names C4, C5 and
    C7 as distinct, and adds D2, C13, C10 and B1.
  - Candidate 2's covered list separates internal effects (B6) from effects on the output (C12).
  - A4 is no longer called cue-to-noun work.
  - The gap's summary carries the record's own limit.
  - The draft's full limits travel with candidate 2: the `Δc` record, and the recomputation alternative.
  - Candidate 4's covered principle states the qualifier under which B1 supports it: interventional success can
    mislead unless it acts on the aligned variable itself.
  - The review recommends dropping candidate 4 as a separate candidate, citing the protocol's rule. The boundary stays
    where the draft states it.
  - The causal-test input no longer reads as a decision, and it cites the protocol and the readiness assessment instead
    of an uncommitted instruction.
  - Three deviations were added: relations left implicit, depths the protocol does not name, and two seeds read below
    full text.
  - Titles were added for the screened works the verdicts cite.
- **Notes,** 18, all done:
  - the short labels;
  - the claim-file citations;
  - what each verdict states;
  - the findings at their recorded status;
  - superlatives hedged with *found*;
  - the brief's wording for candidate 1's precedents;
  - the lead count;
  - the families list;
  - D2's hash files and D4's venue;
  - the grey-literature sentence;
  - Buehler;
  - A5 and A11;
  - C13's warning;
  - the causal-test wording;
  - the gap's other clauses;
  - the Verification section;
  - what A14 and A15 bear on;
  - two seeds read below full text.

## Re-review of the second revision

**PASS WITH NOTES,** 2026-09-29. One independent agent re-reviewed the second revision, read-only. It checked it
against:
- the protocol, the draft and the claim files;
- the full reports of the five agents above;
- the local copies of the sources they saved.

It loaded no model and changed nothing in the repository.
- **Blockers:** both resolved.
  - In the verdicts, candidate 1's remains restates every part of the protocol's statement, with the draft's named
    exceptions and its sentence that not every test passed.
  - Candidate 2's remains restates the protocol's statement without widening it.
- **Findings:** every MUST-FIX of the four source checks, and every should-fix and note of the conformance review, was
  applied as the reports proposed, except as listed below.
- **Checker:** passed with 3 quotes, 24 citation groups, 1 commit hash and 1 label, exit status 0. Each citation points
  at the lines it claims, and each quote is exact.
- **Should-fix,** six:
  1. the brief paraphrased two parts of candidate 1: the lock, and errata recorded after the fact;
  2. candidate 2's covered list placed C13's preset criteria in Pythia-70M, where they were scored only on the toy;
  3. candidate 2's limits said only that the record does not rule out recomputation, where the draft reports partial
     evidence for it at block 0;
  4. Buehler, cited in three places, had no title;
  5. C13, cited as a caution for candidate 3, did not name candidate 3, which deviation 6 reads as not relevant;
  6. this record had three faults:
     - it placed the unapplied request under the notes;
     - it said that candidate 4 was dropped, where the review recommends dropping it;
     - it omitted some applied fixes and unverified points.
- **Notes:** wording and consistency, among them:
  - the gap's statement in the brief;
  - three superlatives without *found*;
  - non-protocol relation labels in B3, B8 and B12;
  - A15's relation;
  - D8's arXiv title;
  - the limits carried by candidates 1 and 2;
  - one double blank line.

The should-fix items and notes were applied before the review was committed.

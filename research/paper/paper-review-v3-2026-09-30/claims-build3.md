# Claim-strength check: draft v3, build 3

File: `research/paper/draft-v3-2026-09-30.md` (uncommitted; 903 lines; sha256 d229364c…). It is identical to
`review-v3/draft-v3-build3.md`, and word for word identical to `build3-unwrapped.md` (11,509 tokens).

**Verdict: PASS WITH NOTES.** There are no blockers and no SHOULD items.
- Every cut keeps the qualifiers, negatives, statuses and scopes the claims findings required.
- Every moved passage keeps its tag and trace.
- No new wording overclaims.
- S1, S2, N1, N2 and N4 from build 2 are applied correctly.

Three notes remain; the first is carried over from build 2.

Nothing in the repository was changed, and nothing under `outputs/` was read. Process disclosure: in this round, two
read-only scratch analyses ran under the system `python3` rather than the prescribed interpreter. They were the
cut-list summary and the citation-group comparison. One more, the wrap comparison, ran under `.venv/bin/python -B`
without the `HF_HUB_OFFLINE=1` prefix. All three only read the draft and scratch files, and wrote only under
`scratchpad/paperreview/`.

## Checks run

- **Checker:** exits 0. It reports 27 quotes, 234 citation groups, 2 commit hashes and 43 labels, and prints PASS.
- **Citation groups, build 2 to build 3:** five groups are gone and four are new; the rest are unchanged or moved with
  their text.
  - **Gone:**
    - Figure 1's readout clause (020 design:106–115). §4.2's readout paragraph keeps the content with the same trace.
    - The introduction's C003@eeff9ff:295–297. The clause now points to §5, whose "What these predictions are not"
      paragraph (L403–409) keeps that trace.
    - Three groups replaced by longer versions of themselves.
  - **New, each checked against its cited lines:**
    - Box 1's `(011:20–22)`, for "six new frames, four cue-final and two coordinated". 011:20–22 names the six frames,
      and 011:18 names the four cue-final ones.
    - Table 2's `018:141`: "Experiment 016's decoded-`c_L` convention (block 2 at the template base)".
    - `final report:19–20, 30`. Y1/cue_final's lowest draw is 0.995470, below g = 0.995520. The CDF percentile is
      0.0001, labelled descriptive, which means exactly one of the 10,000 draws is at or below g.
    - `022:9–10, 190, 194, 248–250`.
      - All eight conditions passed individually, and there is no aggregate label.
      - Y1/C4 is 0.40814 and Y2/C4 is 0.38338. Each is above its envelope and above 0, and each is marked PASS.
      - The README reads: "its measured share … is descriptive, since the frozen reading of C4 is a positive
        contribution and never 'large' or 'substantial'".
      - The new C4 sentence therefore matches the record, Table A1 ("all eight PASS") and Figure 6's C4 chips. Build
        2's "C4 is descriptive" had understated it.
- **Moved passages:**
  - **008:57–59:** the axis component alone drives 0.64, 0.50 and 0.55 of the effect, so the range 0.50–0.64 is right.
    The whole encoding drives −0.09, −0.01 and +0.06, so "about none" is right.
  - **009:110–115:** this sits under 009's "Tier A … for the record" heading, so [exploratory] is correct.
  - **007:** keeps [descriptive] and its trace.
  - **Upstream-chain lineage:** keeps its trace.
- **Rendering (CommonMark):**
  - The source has 234 comments. 233 pass through hidden, and the last is the intended code span in Appendix D. There
    are no stray markers.
  - There are 6 tables and 1 blockquote. The blockquote is Box 1: one paragraph with 13 hidden traces.
  - The only list starts are the intended ones: the contributions, the statuses and the references.
- **References:** all 61 are cited.
- **Figures:**
  - F3's A1 inset reads "009's read-out test … [prospective, given the measured input]", which is consistent with the
    caption and the figures README.
  - The Figure 4 image carries its own legend for circles, diamonds and statuses, so the cut caption sentence loses
    nothing.
  - Figure 6's C4 chips agree with the new §6 wording.

## Build-2 findings

- **S1 resolved** (L442–446): "Against those draws [descriptive], … only one draw lies at or below Y1/cue_final's
  `g`". The tag covers both comparisons.
- **S2 resolved** (L54–56): "in exposed frames, new frames or both". See the optional note on its pointer.
- **N1 resolved** (L348–351): qualifier-only status cells are now defined. L372 reads "[pooled; new frames alone
  conditional]".
- **N2 resolved** (L321), with the added trace 018:141.
- **N3 partly resolved.** See note 1 below.
- **N4 resolved** (L65–67): "each" is dropped.

## The cuts taken (ranks 2–13 and 15–22)

Each is safe for claim strength:
- **Roadmap:** no claim.
- **§4.4 opening:** Table 2's column heads carry the distinction.
- **§5 last sentence:** the qualifiers survive elsewhere.
  - Table 3's status cells carry "given the measured input" and "exposed base states".
  - Table 3's caption now defines qualifier-only cells.
  - §3 defines the qualifiers.
  - §5's "from 013 on" keeps 011 and 012 unconditional.
- **§10 duplicates:** §10's last paragraph still says the routes to C002's causal gates "did not establish them", and
  that "their causal gates are unestablished".
- **Figure 1 readout clause:** kept in §4.2.
- **Real-swap quote:** the paraphrase is faithful to 022 design:85–86.
- **Figure 4 circles:** the image has a legend.
- **§4.2 opening:** the status default is kept.
- **P1 definition:** the glossary keeps it (009:62–67).
- **§8 fresh sets:** the numbers are in §2 (L134–137).
- **§2 floors:** Table A2 and §8 keep it, including 024's binding null.
- **Rival prefixes in Table 3:** the caption and "[pooled]" carry it.
- **Confirmation-set sentence:** §3 keeps "a confirmation set frozen before any model output on it".
- **Introduction clause:** §5 keeps it, with its trace.
- **Erratum clause:** §6 keeps "021 tested that program unchanged" and "the erratum qualifies all three labels", and so
  does Table A1's 021 row.
- **§8 new frames:** §3 defines conditional, and the 016 quote stays.
- **Ranks 19–22 (moved to Appendix A):** tags and traces are kept. §4.1 loses only descriptive support, and the Box
  and §7 point to Appendix A.

The ranks not taken (1, 14, 23, 24 and 25) raise no claim-strength objection.

## Remaining findings

### Note 1: review records named in the draft are still missing (carried over from N3)

**Where.**
- The paper README (L19–24) links to `paper-review-v3-2026-09-30/README.md`. That folder now holds six files
  (claims.md, claims-build2.md, figures.md, figures-build2.md, flow.md and flow-build2.md), but no README.md.
- The draft header (L4–5) says the draft "applies the reviews of its first build ([reviews](reviews.md))". reviews.md
  has no draft-v3 section yet.

**What to do.** Add both before commit.

### Note 2: "one in 007" undercounts 007's baselines

**Where.** §8, Baselines (L536–537): "a few descriptive baselines: one in 007".

**What the record says.** 007 reported three Y4 baselines, "reported, never selected against": Ridge-full,
E005-scalar and PCA-006 (007:95–97).

**Corrected wording.** "three in 007".

### Note 3 (optional): the introduction's pointer to §2

**Where.** L55: "in exposed frames, new frames or both (§2)".

**What the record says.** §2 (L134–137) gives the new-frame counts for 009–023 and says 024 and 025 used exposed frames
only. It does not say that 007–012 had no test in exposed frames. Table 3's "not tested" cells and Figure C1's caption
do.

**Corrected wording.** "(§2, Table 3)".

## Checked and correct

- **Unchanged and still in place:**
  - the scope of the abstract, contribution 2 and §10;
  - the localization's scope (L63–64, §4.1, §10);
  - "candidate mechanism";
  - Box 1's statuses;
  - the §4.2, §5 and §6 status defaults;
  - Table 1;
  - the "found" hedges;
  - the restored records: 020's six of twelve floors, C002's P3 counts and 0.699, 021's quote and the erratum, and the
    bounded-search caveat.
- **The causal boundary** is still stated in §1 (L83, L101), §4.2 (L235), §7 (L512) and §10 (L622–623).
- **The known traps still hold:**
  - 021's scope;
  - no aggregate label for 022 or 023, and Y1 and Y2 are never pooled;
  - the conditional and given-the-measured-input qualifiers;
  - the scopes of `either` and `every`;
  - 018's Y2 and 017's pattern term are unestablished;
  - 025 is never called "no causal effect".

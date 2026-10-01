# Review of research/paper/ai-use-audit-2026-10-01.md (uncommitted; HEAD = origin/main = 7f2cce1)

The independent read-only review of the [audit](ai-use-audit-2026-10-01.md)'s revision 1. It is kept as returned,
wrapped at 120 columns, with the three variants in B1's fix set as sub-items rather than quotes. The harness blocked the
reviewer from writing it to the scratchpad, so it came back as a message. Its line numbers refer to revision 1. What was
done about each finding is in the audit's *Review* section.

**Erratum (from the re-check, Note 12):** the two citations of *024 plan:5* below should read *024 plan:6*; line 5 is
blank, and the approval is on line 6.

**Verdict: FAIL.** One BLOCKER, in the proposed statement (§6). The counts in §2–§3 are right, apart from two definition
slips. Every cited line says what the audit says it says, apart from one line range. The source-map checker passes.
Several characterizations in §3 and §4 are stronger or weaker than the record. The audit also leaves out record evidence
that bears directly on its open questions. Most of that evidence is in commit-message bodies, which the audit did not
use.

Method: the commit numbers were recomputed from `git log
--format='%h|%ad|%an|%ae|%cn|%ce|%P|%s|%(trailers:key=Co-Authored-By,valueonly)' --name-only`. Every citation was opened
at its stated commit. The plans, designs, READMEs, evidence and the claim, synthesis and paper records were searched
with git grep, and commit-message bodies with `git log --format=%B`. The three TMLR pages were fetched. No repository
file was changed. The harness blocked writing the report to the scratchpad, so it is given here in full.

## BLOCKER

**B1. §6, lines 125–127: the sentence on the authors' acts gives the record's account of the reviewer's acts as the
authors' acts, and its AI-reviewer variant falls short of §4's own requirement.**
- **What is wrong.** Only the subject is bracketed ("[The research direction was set by the authors, who]"). The
  predicate is presented as the record's account (line 117), but the record does not assign it to the author:
  - **Design approvals,** where the record names anyone, are the reviewer's: 005's revision 5
    (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:62–63), 022
    (…022…design.md:530), 023 (…023…design.md:5), 024
    (docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md:5) and 025 (…025…design.md:5).
    Design changes are recorded as "reviewer's three changes" and "reviewer's six changes" (messages of 19183a7,
    617f549, bea0203).
  - **Claim decisions** are the reviewer's: the C002 rulings (research/claims/reviews/C002-review-2026-09-27.md:20) and
    the C003 registration request (research/claims/reviews/C003-registration-review-2026-09-28.md:15).
  - **"How to state them":** the claim texts were drafted by the assistant and reviewed by agents
    (research/claims/drafts/C002-successor-draft-2026-09-27-reviews.md:3). The reviewer set terms, not wording (C003
    review:15–20).
  - **Locks:** 025's install was authorized by the reviewer
    (experiments/025-nounness-direction-intervention/evidence/lock-review-2026-09-26/REVIEW.md:1). 021–024's lock
    commits are the assistant's, and no authorizer is named (see S3).
  - **The AI-reviewer variant.** The bracketed "[, with an AI assistant acting as reviewer in these decisions]" does not
    say what §4 (lines 80–82) says the statement must then say: that AI proposed the decisions and the author adopted
    them.
- **Why it blocks.** Adopted as written, the template can turn the reviewer's recorded decisions into the authors'
  decisions. It then understates AI's part on exactly the human-sourcing point TMLR examines (§7).
- **Exact fix.** Replace lines 125–127 with explicit variants, and add a line saying that none of them is the record's
  account until §4 is answered:
  - [If the project's reviewer is the author:] The author set the project's aim, chose among the proposed design
    options, required changes to the designs, approved each design before it was implemented, committed or authorized
    each preregistration lock before its confirmation run, and decided which claims to register.
  - [If the reviewer is an AI assistant the author consulted:] The decisions recorded as the project's reviewer's —
    approving and changing the designs, choosing among design options, signing off the locks, and registering the claims
    — were proposed by [ChatGPT], acting as reviewer, and adopted by the author. The author committed or authorized each
    preregistration lock before its confirmation run.
  - [If the reviewer is another person: settle acknowledgment or authorship under TMLR's authorship criteria first.]

## SHOULD

**S1. §3, line 61 (the repository): the push claim is contradicted by eleven plans, and commit identity is not a human
act.**
- **Pushes.** Only 005's plan holds pushes until the user decides (…005…plan.md:23, 188). The plans of 007–017 list
  "commit; push" as the executing agent's own step (007:83, 008:68, 009:61, 010:55, 011:49, 012:50, 013:49, 014:48,
  015:52, 016:40, 017:40, under docs/superpowers/plans/).
- **Commit identity.** All 364 commits are under the author's identity: 362 as mobahug/mobahug, and two authored by the
  author's GitHub noreply address with GitHub as committer, the initial commit `ef0c716` and the PR merge `ea65363`. But
  the identity does not show who made a commit:
  - 311 commits carry the Claude trailer;
  - `2e9994f` was made outside the assistant's session from the assistant's in-progress patch (message of dd50231).
- **Fix.** "Every commit is under the author's git identity, including 311 co-authored by Claude. Two, the initial
  commit and the PR merge, were made through GitHub. 005's plan holds pushes until the user decides (…:23); the plans of
  007–017 list the push as the agent's step (e.g. …009…plan.md:61)." Set the class to "§4". Do not class it as a human
  act.

**S2. §2, line 45 ("one test refactor of 022, `2e9994f`") and §4, lines 86–88: the record says otherwise.**
- `2e9994f` touches `src/neural_decompiler/upstream_localization.py` (190 lines) and 022's `run.py`. It changes C3's
  claim wording, its guard (`GUARD_C3_RANGE` → `GUARD_C3_MAX`) and the calibration stop. Its subject and its bulleted
  body describe only test changes.
- The next commit records that it holds "the bulk of the revision-4 code patch", committed "outside this session under
  the repository identity". Its content is "this session's in-progress patch" (message of dd50231).
- **Fix for line 45:** "`2e9994f` (2026-09-23): the bulk of 022's design-revision-4 code patch, written in the
  assistant's session and committed outside it under the repository identity (message of dd50231). Its subject and body
  describe only test changes."
- **Fix for §4, line 86:** add "and `2e9994f`'s subject and body". The record shows how `2e9994f` was committed, but not
  how its message was written.

**S3. §3, line 54 (locks): the record does say on whose instruction some of the later locks were committed.**
- **012:** installed "at the user's instruction" (message of 5cea6c4).
- **025:** installed after the reviewer "authorized installing the exact candidates" (025 lock-review REVIEW.md:1).
- **021:** its plan assigns the lock commit to the user
  (docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md:330–331, 340), yet the commit
  carries the trailer and records only that the lock was authorized (message of b7e8861).
- **022–024:** the plans run each step only when authorized, without naming who authorizes (023 plan:339–340; 024
  plan:661).
- **Also:** "Locks 005–019, apart from 012" implies that 006, 008 and 010 had locks. They had none (006:71–72; 008:7;
  010:7).
- **Fix.** Name the locks: 005, 007, 009, 011 and 013–019. State the above. Classes: 11 commits, a human act (inferred);
  012, the assistant at the user's instruction; 021–025, §4, with 025 authorized by the reviewer.
- **Add:** the content of every lock (candidate lock and predictions) was produced by the assistant's `lock` phase and
  installed byte for byte.

**S4. §3, line 56: "The preregistered rules, frozen at the lock, assigned every label" is too strong.**
- 006 ended at its quality gate with no lock (006:71–72), and its label is not in the design's outcome list
  (research/synthesis/source-map-005-025.md@b722e9c:666).
- 008 and 010 were discovery-only, with no lock (008:7; 010:7).
- 020 assigned no label (020:8–11).
- 005's protocol-v1 label came at Tier A, before any lock (…005…design.md:62–71).
- Some readings were agreed at reviews after the lock: 021:151 and 022:241, and source-map@b722e9c:690 finds no record
  of 021's agreement.
- **Fix:** "In the 17 locked experiments the outcome labels follow the rules frozen at the lock. 006, 008 and 010 were
  labelled by rules frozen in their designs without a lock; 020 has no label; and 021's and 022's interpretations were
  agreed at reviews."

**S5. §1 line 10, §6 line 118 and §4/§5: the record says the reviewer gave the statement under audit; the audit omits
this and misdates it.**
- "On 2026-09-30 the reviewer chose TMLR … and gave the statement on AI tools" (research/paper/reviews.md@7f2cce1:249).
  The message of 298769e calls it "the reviewer's statement on AI tools", and draft v3 calls its *the author* "the
  reviewer's wording" (draft-v3@7f2cce1:14–15).
- The statement entered the draft in 298769e (2026-09-30) and has not changed since. "the reviewer's text of 2026-10-01"
  (line 118) is therefore wrong by the record.
- Line 10 ("On 2026-10-01 the reviewer asked for an audit") cites nothing. The audit has been required by draft v3's
  header since 298769e (draft-v3@7f2cce1:10–15), and the cleanup's reviews leave the statement to the reviewer
  (research/paper/tmlr-build/reviews-2026-10-01/README.md@7f2cce1:18–20).
- **Fix:** cite these lines in §1. Change line 118 to "of 2026-09-30". Add to §4's first bullet: "This includes the
  statement on AI tools itself, which the record gives as the reviewer's."

**S6. §2 lines 39–40 and §4 lines 80–82: the reviewer's role is under-described.**
- **The reviewer is distinct from the user.** The records keep the two roles apart: "the user's lock commit → the
  reviewer's sign-off" (e.g. docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md:182),
  and "The user's reviewer decides" (research/paper/paper-review-2026-09-30/format.md:367).
- **Sign-offs.** "signs off each lock" is too strong: no sign-off before the confirm is recorded for 009, 016 and 017
  (source-map@b722e9c:671).
- **The same word for agents.** Some records also call an independent agent "the reviewer"
  (tmlr-build/reviews-2026-10-01/citations.md@7f2cce1:3, a subagent; 023 README:297; 022 README:148).
- **A third case.** §4 considers only an AI reviewer. If the reviewer is another person, TMLR's authorship criteria and
  an acknowledgment apply.
- **Fix:** add these points and citations, and add the human-other-than-author case to §4.

**S7. §4, line 83: "The designs record approvals, not origins" is too absolute.**
- **The project's aim** is in the initial commit, made through GitHub (README.md@ef0c716:1–2).
- **The behavior** was proposed by the frozen screening, designed in the Codex period
  (screening/behavior-candidates/report-2026-09-17.md:3, 182).
- **Later directions** followed exploratory diagnostics and design spikes, with the reviewer choosing among options:
  - "the reviewer's choice of Option A after the exposed-only research spike" (message of 2b20db3);
  - …024…design.md:7–10;
  - …025…design.md:422–443;
  - the synthesis followed "the reviewer's direction" (message of 3d87d3c).
- **Fix:** state this partial account. Keep open who proposed the options and 005's question.

**S8. §2, lines 28–31: the Codex-period coverage omits the methodology the paper uses.** docs/RESEARCH_METHODOLOGY.md
(the evidence ladder and gates) and research/claims/README.md were written in 37d209a and d1427c7 (2026-09-15, no
trailer). The methodology has not changed since. Draft v3 rests contribution 1's "one evidence ladder" and its evidence
statuses on it (draft-v3@7f2cce1:95, 182–186). **Fix:** add a bullet for "the research methodology and claim registry".

**S9. §6, lines 120–129: the AI-role sentences understate AI's part or leave it implicit.**
- **The checks.** "Every reported number was checked…" and "Every bibliographic record was checked…" are passive. They
  hide that AI agents and assistant-written scripts did the checking: the claims reviews, e.g.
  research/paper/paper-review-2026-09-30/claims.md:246, the figure-data script and the citation agents.
- **"Independent."** The reviewers were subagents of the same assistant. Their scripts sit in the same session
  scratchpad as the operator's (experiments/023-block0-completion/evidence/README.md:3–4; the lock-review scripts under
  `.../scratchpad/lockreview023`).
- **The review list** omits the implementations, freezes, calibrations, claim records and synthesis
  (research/synthesis/checks/*.md; research/claims/reviews/C002-review-2026-09-27-reviews.md:3).
- **"the preregistered phases of each experiment"** does not fit 006, 008, 010 or 020.
- **Literature.** It omits that the literature agents read sources through a summarizing tool
  (research/paper/literature-review/review-2026-09-28.md@7f2cce1:72–75).
- **Interpretations.** It omits the AI-drafted synthesis and interpretations (research/synthesis: 8/8 commits with the
  trailer) and the AI-run diagnostics and spikes behind later designs.
- **Exact fixes:**
  - "ran the preregistered phases of each experiment" → "ran every phase of every experiment, including each single
    preregistered confirmation run, under protocols frozen before the runs";
  - "performed the literature searches" → "performed the literature searches, reading the sources partly through
    AI-generated page summaries";
  - "drafted the experiment reports, claim records, figures and this manuscript" → "drafted the experiment reports and
    their interpretations, the claim records, the cross-experiment synthesis, the figures and this manuscript, and ran
    the exploratory diagnostics and design spikes that preceded later designs";
  - the review sentence → "Separate read-only instances of the same assistant reviewed the implementations, the freezes,
    calibrations, locks and confirmations (with committed records from Experiment 023 on), the claim records, the
    synthesis, the literature review, the figures and every draft.";
  - the two check sentences → "These reviews and assistant-written scripts checked every reported number against the
    committed experimental record and every bibliographic record against its source; the key paraphrases of prior work
    were spot-checked."

**S10. §3 class column, lines 51–59: "assistance" pre-judges the question §7 leaves open.**
- The column calls drafting the designs, writing all code, running every phase, computing every statistic, drafting the
  interpretations and running the literature searches "assistance".
- TMLR's editorial policy permits LLMs as general-purpose assistive tools. Calling AI-performed work "assistance"
  presumes the conclusion that §7 (lines 149–152) leaves to the author.
- **Fix:** use "AI-performed" for these rows, with direction and approval under §4. Keep "verification" for the agent
  reviews.

**S11. §1, line 14: commit-message bodies were not used as a source.** They hold the record's own provenance: 5cea6c4
(the user's instruction), dd50231 (`2e9994f`), 298769e (the reviewer's statement), 2b20db3 (option choice), and 19183a7,
617f549 and bea0203 (the reviewer's design changes). **Fix:** read them, and add "and their messages" to the sources.

**S12. Missing from §5/§8: the draft's method text does not say who ran the review gates.**
- Draft v3 describes "a review gate between phases" (draft-v3@7f2cce1:179) without saying that the gates were AI agents
  and the reviewer.
- The format review made this a MUST: research/paper/paper-review-2026-09-30/format.md:179–184 says to disclose agents
  in the method, not only in a footnote.
- **Fix:** add a point 7 to §5. Add a step to §8: "§3 of the draft says who reviewed at each gate."

## NOTE

- **N1.** Line 57: 90/98 counts the experiments' READMEs only. With the reports it is 91/99: `5344f44` adds 023's final
  report without touching the README. Fix: "the experiments' READMEs", or "91 of the 99".
- **N2.** Line 59: "All 10" matches research/paper/ outside literature-review/. The literal definition (drafts, figures,
  reviews, build) gives 9, and all of research/paper/ gives 12. All of them carry the trailer. Fix: "All 12 commits that
  touch research/paper/".
- **N3.** Line 58: "63" is at citations.md@7f2cce1:41 (also 176, 217), not at 9–10. Fix: cite ":9–10, 41".
- **N4.** Line 60: "The user approved 005's design." The cited lines state a requirement (1110–1113). The plan calls
  revision 4 approved (…005…plan.md:5), and revision 5 was approved by the reviewer (design:62–63). Also add 024
  (plan:5) to the reviewer's design approvals.
- **N5.** Line 57: "001–003's" reads like claims C001–C003. The five trailer-less claim commits touch only C001 and the
  claims README, for experiments 001–003, on 2026-09-15.
- **N6.** Lines 28–35 and 121: the Codex attribution is an inference from branch names. No tracked file names Codex,
  Claude or ChatGPT; only the draft says "OpenAI- and Anthropic-based".
  - Branch `codex/behavior-candidate-screening` (tip b561bc6) also carries 8 Claude-trailered commits
    (8d9c4a7..b561bc6).
  - `ef0c716` and `ea65363` were made through GitHub.
  - The 001–003 commits were made on main.
  - "[, and ChatGPT]" has no basis in the record (git grep: 0 hits). Say that it awaits the author's account (§4).
- **N7.** Lines 106–107: the source-map checker does not check numbers outside quotes
  (research/synthesis/checks/check_source_map.py:22). The claims reviews and the figure-data script checked the numbers.
- **N8.** Line 55 needs a scope:
  - agents reviewed implementations per the plans (e.g. …005…plan.md:178; 021 plan:311);
  - agents reviewed locks from 021 on (021:81–83; 022:147–148);
  - agents reviewed freezes, the extraction, 024's calibration and the confirmations from 023 on;
  - the floor reviews of 021 and 023 were the reviewer's (021 plan:326; 023
    evidence/calibrate-2026-09-24/FLOOR-REVIEW.md:19, 30);
  - before 021, the recorded lock check is the reviewer's sign-off.
- **N9.** Line 53 is correct, and can cite 023 evidence/README.md:3–4. The term "operator" appears only in 025's
  records. Earlier, all 28 commits that add experiment reports carry the Claude trailer. The plans give the confirm and
  report phases to the agent too (e.g. 011 plan:54).
- **N10.** Line 60 could add three things:
  - the paper's outline came from an AI agent's flow review, which the reviewer adopted
    (research/paper/paper-review-2026-09-30/README.md:3–4, 9–10);
  - the contributions follow the reviewer's decision (research/paper/README.md@7f2cce1:14);
  - the decision to write the paper followed an AI-drafted readiness assessment
    (research/synthesis/paper-readiness-2026-09-28.md:6, 31).
- **N11.** §7, lines 142–144 and 149–152:
  - The editorial policy targets "Low-quality contributions (be they submissions or reviews) that appear to be largely
    LLM-generated". It examines them for misconduct. The audit drops the qualifier.
  - The FAQ's human-sourcing expectation covers "results" as well as ideas and claims. The results were computed by
    AI-written code that AI agents ran, and §7 should say so.
- **N12.** Line 131: "The authors" is acceptable for double-blind and reveals no count. Restore the true number in the
  de-anonymized version.
- **N13.** Line 51: "Designs were revised after AI reviews" cites nothing. Cite the agent-driven revisions (008
  design:5; 022 design:13; 025 design:444) and the reviewer's (messages of 19183a7, 617f549, bea0203).

## Checked and correct

- **Commits.**
  - 364 commits, 2026-09-14 to 2026-10-01, by both author and committer dates.
  - 311 carry Claude trailers (85 %): 175 "Claude Opus 5", 2026-09-17 to 22, and 136 "Claude Opus 5.5", 2026-09-22 to
    10-01. There are no other trailer values.
  - 53 have no trailer. 37 of them are every commit of 2026-09-14 to 16.
  - 16 are after 2026-09-16:
    - the locks of 005, 007, 009, 011 and 013–019;
    - b0eeaf4, 44789a1, 69e4c72 and e2ef765;
    - 2e9994f.
- **Categories.** The definitions are reasonable.
  - specs/ 69/78: the 9 without the trailer are 6 Codex-period commits and 3 amendments of 09-17;
  - src/ 100/115;
  - tests/ 111/132;
  - experiments/*/*.py 56/67;
  - claims 11/16;
  - synthesis 8/8;
  - 17 lock commits, 11 without the trailer. The "rest" in the code row are exactly the commits §2 lists.
- **Codex evidence.**
  - `ea65363` is the GitHub merge of PR #1 from mobahug/codex/experiment-004, which carries 5 commits.
  - `codex/behavior-candidate-screening` is a local branch, in a worktree.
  - origin/codex/experiment-004 exists.
  - The instrumentation plan, line 719, names `codex/causal-instrumentation`.
- **Citations.** Each says what the audit says:
  - plans: 005 plan:3, 23; 011 plan:50; 009 plan:62; and line 3 of all 25 plans;
  - specs: 022 design:530; 023 design:5–6; 025 design:5; 005 design:1110–1113; 018 design:458; 020 design:478–479;
  - research/paper/README.md@7f2cce1:3–5, 20;
  - C003 review@7f2cce1:15;
  - literature review@7f2cce1:62–63; verification@7f2cce1:3–11, 6–7;
  - source-map@b722e9c:702–704;
  - the 025 lock-review PROMPT.md:1–3;
  - draft-v3@7f2cce1:39–46, including each quoted fragment.
- **§5.** Items 1, 2, 5 and 6 are accurate as far as they go. The citation sentence does claim more than the
  verification did.
- **§7.** The FAQ, editorial-policy and author-guide summaries match the pages, apart from N11. Disclosure goes in a
  first-page footnote, and the pages give no level of detail and no rule for how authors refer to themselves.
- **The checker.** `.venv/bin/python research/synthesis/checks/check_source_map.py
  research/paper/ai-use-audit-2026-10-01.md` printed "checked 5 quotes, 25 citation groups, 2 commit hashes and 0 labels
  … PASS" and exited with **0**. Unversioned citations are read at the checker's pin, d7c5e62. The cited docs/ and 025
  evidence files have not changed since then.

---

# Re-check of research/paper/ai-use-audit-2026-10-01.md, revision 2

The same reviewer's focused re-check of revision 2, kept as returned and wrapped at 120 columns. Its line numbers refer
to revision 2. Revision 3 applies its SHOULD and all thirteen notes.

**Verdict: PASS WITH NOTES.** There is no blocker. All 26 original findings are resolved. One new SHOULD: line 157
credits the Codex period with a screening that Claude ran. There are also 13 small notes.

## Checker

`.venv/bin/python research/synthesis/checks/check_source_map.py research/paper/ai-use-audit-2026-10-01.md` printed
"checked 7 quotes, 70 citation groups, 14 commit hashes and 0 labels … PASS" and exited with **0**.
- Unversioned citations are read at the pin, d7c5e62. Every cited path is unchanged between d7c5e62 and HEAD.
- All 70 groups (74 references) were opened. Every one exists, and the commit messages cited say what revision 2 says
  they say.
- The coordinator's list checks out: 024 plan:6, 008 design:5, 022 design:13, 025 design:422–443 and 444, the
  021/023/024 plan lines, the README lines of 006/008/010/020/021/022, source map 666/671/690/702–704,
  README.md@ef0c716:1–2, screening report 3/182, 023 evidence README:3–4, FLOOR-REVIEW:19, paper-readiness 6/31,
  paper-review README 3–4/9–10, format.md 179–184/367, C002 review:20, successor-draft reviews:3 and
  check_source_map.py:22.
- The cited commits check out: 5cea6c4, dd50231, b7e8861, 298769e, 2b20db3 (022's design revision 1), 3d87d3c, 19183a7,
  617f549 and bea0203.
- The original review was wrong to cite "024 plan:5": that line is blank, and the approval is on line 6. Revision 2's
  ":6" is correct.

## The sentences flagged for checking

- **"with committed records from the nineteenth experiment on":** the arithmetic is right (023 is the 19th of 005–025),
  but the paper never counts experiments this way (Note 8).
- **"All 28 commits that add experiment reports carry the Claude trailer":** true. Exactly 28 commits add
  `experiments/*/evidence/*report*.md`, and all carry the trailer. None only modifies a report, and 001–004 have no
  tracked report.
- **"Every lock's content … was written by the assistant's lock phase and installed byte for byte":** supported.
  - Byte identity is stated for 009 (README:140), 011–019 (READMEs; the message of 5cea6c4) and 021–025.
  - For 005 and 007, the committed lock's embedded `content_sha256` equals the candidate's (005:120–121 `ec2a8879…`;
    007:139–140 `de0ae866…`).
  - The plans give `lock` to the agent (e.g. 011 plan:50).
- **"006, 008 and 010 were labelled under their designs without a lock":** acceptable.
  - 008 and 010 were discovery-only under frozen design rules.
  - 006's quality gate is in its design, and revision 2 already says its label is not in the design's outcome list.
  - Optional: also cite source-map@b722e9c:669, where one 008 label string comes from the code.
- **The verification bullets' scope:** mostly right, with three gaps (Notes 2, 3 and 13).

## Remaining and new findings

**SHOULD**

**S-new. Lines 157–158 say the behavior came from "the frozen screening of the Codex period". It did not.** Only the
screening's design is from the Codex period (§2, line 50). On 2026-09-17, commits with the Claude trailer implemented,
froze, amended and ran it:
- `3655a62` "freeze behavior candidate manifest";
- `8d9c4a7`, which eliminated a candidate;
- `e610c5f`, which changed the gates after an independent pre-output review;
- `1f043d9`, which recorded the result.

As written, the line understates Claude's part in choosing the paper's behavior.
- **Fix:** "the behavior was proposed by the behavior screening, designed in the Codex period and then implemented,
  frozen, amended after an independent pre-output review, run and recorded with the Claude trailer on 2026-09-17
  (`3655a62`, `e610c5f`, `1f043d9`), as the subject of a user-approved Experiment 005
  (screening/behavior-candidates/report-2026-09-17.md:3, 182);".

**NOTES**

1. **Line 117 ("This is inferred, not recorded").** The READMEs say these locks were "committed by hand" (005:121;
   011:60; 013:70; 014:74; 015:86–87; 016:84–85; 017:94; 018:77; 019:94). The project uses the same phrase for commits
   that carry the Claude trailer: 021's lock (021:52–53, committed in `b7e8861`) and 024's freeze (024:105, `566eb6f`).
   So the phrase does not show who made the commit.
   - **Fix:** add "The READMEs call these commits 'by hand' (e.g. 011:60), a phrase they also use for commits that carry
     the Claude trailer (021:52–53 and `b7e8861`; 024:105 and `566eb6f`), so it does not show who made them."
2. **Lines 132–134 (locks before 021).** Before 021 the records also show read-only post-lock reviews and checks by an
   unnamed actor (012:67; 013:72; 014:76; 015:89; 016:88; 017:98; 018:81) and the reviewer's own post-lock review in 011
   (011:62). Also, "None is recorded for 009, 016 and 017" means no sign-off, but 016's and 017's post-lock checks are
   recorded.
   - **Fix:** "Before 021, the records show a read-only post-lock review or check (012:67; 013:72; 014:76; 015:89;
     016:88; 017:98; 018:81), the reviewer's post-lock review in 011 (011:62) and the reviewer's sign-off, which is not
     recorded for 009, 016 and 017 (research/synthesis/source-map-005-025.md@b722e9c:671)."
3. **Lines 135–136 (calibrations).** The bullet leaves out 022's independent calibration review (022:140). **Fix:**
   "agents reviewed 022's calibration (022:140) and, from 023 on, the freezes, …".
4. **Line 88.** "005's protocol-v1 label came at Tier A" cites design:62–63, but the label is stated at 70–71. **Fix:**
   ":62–71".
5. **Line 83 ("unchanged since").** The methodology is unchanged since then. The claim registry's README is not: it
   gained the C002 and C003 entries in `13600cd` and `0810a75`, both with the Claude trailer.
   - **Fix:** "Written in the Codex period, without a trailer (`37d209a`, `d1427c7`). The methodology is unchanged
     since; the registry's README gained only the C002 and C003 entries (`13600cd`, `0810a75`)."
6. **Line 48 ("its statuses").** Draft v3's five evidence statuses come from the source map, which refines the
   methodology's exploratory/confirmatory split (draft-v3@7f2cce1:186; source-map@b722e9c:36–45).
   - **Fix:** "for its evidence ladder and claim maturity (research/paper/draft-v3-2026-09-30.md@7f2cce1:95, 182–185)".
7. **Line 91.** "and so does the venue (…README.md@7f2cce1:3–5, 20)": lines 3–5 record the reviewer's decision to write
   the paper, not the venue.
   - **Fix:** "The reviewer decided to write the paper (research/paper/README.md@7f2cce1:3–5) after an assistant-drafted
     readiness assessment of questions the reviewer set (…paper-readiness…:6, 31); the contributions and the venue
     follow the reviewer's decisions (research/paper/README.md@7f2cce1:14, 20)."
8. **Line 215.** The paper numbers its experiments 005–025 (draft-v3@7f2cce1:66–67), so "the nineteenth experiment" will
   confuse readers. **Fix:** "from Experiment 023 on".
9. **Lines 159 and 213 ("run by the assistant"; "ran the exploratory diagnostics and design spikes").** No source is
   cited, and the record shows this only indirectly.
   - **Fix:** cite "(message of `219cdc5`;
     docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:451–452)". The first recomputes
     022's spike; the second names the scratch design checks.
10. **Lines 208 and 211.** §2 says Codex is identified "by inference from branch names" (line 44), yet the statement
    names Codex without a bracket. And "AI-generated page summaries" goes beyond the record's "a tool that summarizes a
    page" (review@7f2cce1:72–75).
    - **Fix:** write "OpenAI's [Codex]" for the author to confirm. Write "through a page-summarizing tool", or keep
      "AI-generated" once the author confirms it.
11. **Line 89 ("The claim texts were drafted by the assistant").** C001's text dates from the Codex period. **Fix:**
    "The texts of C002 and C003 were drafted by the assistant".
12. **The archived review** (research/paper/ai-use-audit-review-2026-10-01.md:29, 211) keeps the original's "024
    plan:5". **Fix:** add a one-line erratum to the archive: line 5 is blank, and the approval is at line 6.
13. **Lines 129–131 (implementations), optional.** The reviewer also reviewed 021's implementation and set six
    requirements for 024's (021 plan:315; 024:62). **Fix:** add "and the reviewer reviewed 021's implementation and set
    requirements for 024's (…021…plan.md:315; 024:62)".

## Resolved, per original finding

- **B1:** resolved. §6 now gives three variants (lines 221–229). None counts as the record's account until §4 is
  answered (204–205, 231–232), and variant B says AI proposed and the authors adopted (225–228).
- **S1:** resolved (line 92).
- **S2:** resolved (lines 74–76, 167–170).
- **S3:** resolved (lines 86, 114–126). See Note 1.
- **S4:** resolved (line 88). See Note 4.
- **S5:** resolved (lines 12–16, 154, 204).
- **S6:** resolved (lines 62–69, 133–134, 148–152).
- **S7:** resolved (lines 155–164). See the new SHOULD for line 157 and Note 9.
- **S8:** resolved (lines 47–48, 83). See Notes 5 and 6.
- **S9:** resolved (lines 207–219). See Notes 8–10.
- **S10:** resolved (lines 18–22 and the class column, 82–92).
- **S11:** resolved (lines 25–26). The messages are used at lines 75, 82, 89, 118, 121 and 160.
- **S12:** resolved (lines 198–200, 264).
- **N1:** resolved (line 89).
- **N2:** resolved (line 91).
- **N3:** resolved (line 90).
- **N4:** resolved (lines 95–108; 024 plan:6 at line 100).
- **N5:** resolved (line 89).
- **N6:** resolved (lines 37–38, 44–58, 208).
- **N7:** resolved (lines 188–191).
- **N8:** resolved (lines 128–143). See Notes 2, 3 and 13.
- **N9:** resolved (line 85).
- **N10:** resolved (line 91). See Note 7.
- **N11:** resolved (lines 246–249, 254–256).
- **N12:** resolved (lines 234–235, 266).
- **N13:** resolved (lines 82, 109–112).

# Audit of the statement on AI tools, 2026-10-01

**Status:** revision 3. This is a read-only audit for the statement on AI tools that TMLR requires as a first-page
footnote. It changes no record. It checks [draft v3](draft-v3-2026-09-30.md)'s statement against the project's history
and proposes a replacement. Revision 1 failed an independent read-only review on one blocker: its proposed statement
could turn decisions the record gives to the project's reviewer into the authors' own. Revision 2 applied every finding
of that review, and its re-check passed with notes, which revision 3 applies ([Review](#review)). The replacement cannot
be completed until the author answers §4, and the statement must not be frozen before then.

## 1. Purpose and method

Draft v3's statement on AI tools (research/paper/draft-v3-2026-09-30.md@7f2cce1:39–46) is the reviewer's text. On
2026-09-30 the reviewer chose TMLR and gave the statement (research/paper/reviews.md@7f2cce1:249), and it entered the
draft in `298769e`. Since then the draft's header has called for an audit of the statement against the project's history
before submission (research/paper/draft-v3-2026-09-30.md@7f2cce1:10–15). The reviews of the submission-build cleanup
left the statement open (research/paper/tmlr-build/reviews-2026-10-01/README.md@7f2cce1:18–20).

The audit asks what AI did in each activity, and classes each role in one of four ways:
- **AI-performed:** done by an AI assistant;
- **verification:** checked by AI agents;
- **human act:** done by the author, where the record shows it;
- **§4:** open, because the record cannot settle it.

Sources read:
- **The git history of `main`:** 364 commits from 2026-09-14 to 2026-10-01, with their authors, committers, co-author
  trailers and messages.
- **The tracked files,** searched for the tools' names and for the roles the records name: *the user*, *the reviewer*,
  *the operator*, *agent* and *subagent*.
- **The records themselves:** the designs and plans under `docs/superpowers/`, the experiments' READMEs and evidence,
  the claim reviews, the synthesis, the literature review's records and the paper's reviews.

Not read: anything under `outputs/`, and the conversations in which the work was directed. Those conversations are not
part of the record (§4).

## 2. The tools and roles in the record

No tracked file names Claude, Codex or ChatGPT as a tool. Draft v3's statement names only “OpenAI- and Anthropic-based
assistants” (research/paper/draft-v3-2026-09-30.md@7f2cce1:39–40). The tools are identified from the git history.
- **Anthropic's Claude, through Claude Code.** 311 of the 364 commits (85 %) carry a co-author trailer for Claude:
  - *Claude Opus 5* on 175 commits, from 2026-09-17 to 2026-09-22;
  - *Claude Opus 5.5* on 136, from 2026-09-22 to 2026-10-01.

  No commit carries any other trailer.
- **OpenAI's Codex, by inference from branch names.** The 37 commits of 2026-09-14 to 2026-09-16 carry no trailer. They
  cover:
  - Experiments 001–004, which the paper does not report;
  - the research methodology and the claim registry (`37d209a`, `d1427c7`), which draft v3 relies on for its evidence
    ladder and claim maturity (research/paper/draft-v3-2026-09-30.md@7f2cce1:95, 182–185);
  - the causal-instrumentation package that 005–025 use;
  - the design of the behavior screening.

  Their branches are named for Codex. The merged pull request `ea65363` came from `codex/experiment-004`, a local branch
  is `codex/behavior-candidate-screening`, and the instrumentation plan expects a clean tree on
  `codex/causal-instrumentation` (docs/superpowers/plans/2026-09-16-causal-instrumentation.md:719). The attribution is
  still an inference:
  - the screening branch also carries eight commits with the Claude trailer;
  - the commits for 001–003 were made on `main`;
  - the initial commit `ef0c716` and the merge `ea65363` were made through GitHub.
- **Agentic plans.** The implementation plans are written to be executed by AI agents. Each opens with an instruction
  for agentic workers to use a subagent-driven skill
  (docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md:3).
- **The user and the project's reviewer.** The records keep two roles apart, the user's acts and the reviewer's
  decisions, as in “the user's lock commit → the reviewer's sign-off”
  (docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md:182). The format review calls
  the reviewer *the user's reviewer* (research/paper/paper-review-2026-09-30/format.md@7f2cce1:367). The reviewer
  approves and changes designs, signs off locks and decides the project's course. No record says who or what the
  reviewer is. Some records also call an independent agent *the reviewer*, for example
  research/paper/tmlr-build/reviews-2026-10-01/citations.md@7f2cce1:3.

After 2026-09-16, 16 commits carry no trailer:
- the lock commits of 005, 007, 009, 011 and 013–019, eleven in all;
- three amendments to the screening design and one screening module, on 2026-09-17;
- `2e9994f`, on 2026-09-23. It holds the bulk of 022's design-revision-4 code patch. That patch was written in the
  assistant's session and committed outside it under the repository identity (message of `dd50231`). The commit's
  subject and body describe only test changes.

## 3. What AI did, activity by activity

| activity | what the record shows | class |
|---|---|---|
| designs and their decision rules | 69 of the 78 commits that touch the design specs carry the Claude trailer. The other 9 date from the Codex period or are the 2026-09-17 amendments. Designs were revised after agent reviews (008 and 022, the files cited below) and after the reviewer's changes (messages of `19183a7`, `617f549` and `bea0203`). | AI-performed; approvals: §4 |
| the methodology and the claim registry | Written in the Codex period, without a trailer (`37d209a`, `d1427c7`). The methodology is unchanged since; the registry's README gained only the C002 and C003 entries (`13600cd`, `0810a75`). | §4 |
| code, tests and runners | The Claude trailer is on 100 of the 115 commits that touch `src/`, 111 of the 132 that touch `tests/` and 56 of the 67 that touch the experiments' Python files. The rest are Codex-period commits or the trailer-less commits listed in §2. | AI-performed |
| running the experiments | The plans give every phase to the executing agent, the confirmation and the report included (docs/superpowers/plans/2026-09-19-experiment-011-encoding-read-prospective-plan.md:50, 54). All 28 commits that add experiment reports carry the Claude trailer. From 023 on, the run records and reviews were run from the assistant's session scratchpad (experiments/023-block0-completion/evidence/README.md:3–4). | AI-performed |
| preregistration locks | See the list below the table. 17 experiments have a lock; 006, 008, 010 and 020 have none (006:71–72; 008:7; 010:7; 020:8–11). Every lock's content, the candidate lock and its predictions, was written by the assistant's `lock` phase and installed byte for byte. | AI-performed; commits: human act or §4 |
| verification | See the list below the table. The agents were read-only instances of the same assistant, run from the same session scratchpad as the operator's own checks (experiments/023-block0-completion/evidence/README.md:3–4). | verification |
| analysis and outcome labels | Assistant-written code computed every statistic. In the 17 locked experiments, the outcome labels follow the rules frozen at the lock. 006, 008 and 010 were labelled under their designs without a lock, and 006's label is not in its design's outcome list (research/synthesis/source-map-005-025.md@b722e9c:666). 020 has no label. 005's protocol-v1 label came at Tier A, before any lock (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:62–71). 021's and 022's interpretations were agreed at reviews (021:151; 022:241), and no committed record shows 021's agreement (research/synthesis/source-map-005-025.md@b722e9c:690). | AI-performed; rules: §4 |
| interpretation: reports, claims and synthesis | The Claude trailer is on 90 of the 98 commits that touch the experiments' READMEs, 11 of the 16 that touch the claims and all 8 that touch the synthesis. The five trailer-less claim commits touch only C001 and the claims README, for Experiments 001–003, on 2026-09-15. The texts of C002 and C003 were drafted by the assistant and reviewed by agents (research/claims/drafts/C002-successor-draft-2026-09-27-reviews.md@7f2cce1:3). The claim decisions are the reviewer's: the rulings on C002 (research/claims/reviews/C002-review-2026-09-27.md@7f2cce1:20) and the terms of C003's registration (research/claims/reviews/C003-registration-review-2026-09-28.md@7f2cce1:15–20). The synthesis followed the reviewer's direction (message of `3d87d3c`). | AI-performed; decisions: §4 |
| literature | Four read-only agents searched, with about 135 queries (research/paper/literature-review/review-2026-09-28.md@7f2cce1:62–63). They read the sources partly through a tool that summarizes pages (research/paper/literature-review/review-2026-09-28.md@7f2cce1:72–75). Five agents verified the review, and a sixth re-reviewed it (research/paper/literature-review/verification-2026-09-28.md@7f2cce1:3–11). On 2026-10-01 an agent checked all 63 references against their source records (research/paper/tmlr-build/reviews-2026-10-01/citations.md@7f2cce1:9–10, 41). | AI-performed and verification |
| the manuscript, figures and bibliography | All 12 commits that touch `research/paper/` carry the Claude trailer. AI agents reviewed every draft and the figures. The paper's outline came from an agent's flow review, which the reviewer adopted (research/paper/paper-review-2026-09-30/README.md@7f2cce1:3–4, 9–10). The reviewer decided to write the paper (research/paper/README.md@7f2cce1:3–5) after an assistant-drafted readiness assessment of questions the reviewer set (research/synthesis/paper-readiness-2026-09-28.md@7f2cce1:6, 31). The contributions and the venue follow the reviewer's decisions (research/paper/README.md@7f2cce1:14, 20). | AI-performed; decisions: §4 |
| the repository | Every commit is under the author's git identity, including the 311 co-authored by Claude. Two, `ef0c716` and `ea65363`, were made through GitHub. 005's plan holds pushes until the user decides (docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md:23). The plans of 007–017 list the push as the agent's own step (docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md:61). | §4 |

**The design approvals and changes.**
- **Approved by the reviewer:**
  - 005's revision 5, as protocol v2
    (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:62–63);
  - 022 (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:530);
  - 023 (docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md:5–6);
  - 024 (docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md:6);
  - 025 (docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md:5).
- **The user:**
  - 005's design required the user's explicit approval
    (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:1110–1113);
  - a definition in 018 was tightened at the user's request
    (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:458);
  - 020's revision 3 was authorized by the user on the diagnostic's evidence
    (docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:478–479).
- **Changed after agent reviews:**
  - 008 (docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md:5);
  - 022 (docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md:13);
  - 025 (docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md:444).

**The lock commits.**
- **005, 007, 009, 011 and 013–019:** eleven commits without an AI trailer. That matches the plans' rule that installing
  and committing the lock is the user's act
  (docs/superpowers/plans/2026-09-18-experiment-009-head-transport-rule-plan.md:62). This is inferred, not recorded. The
  READMEs call these commits *by hand*, as 011's does (011:60). They use the same phrase for commits that carry the
  Claude trailer: 021's lock (021:52–53), committed in `b7e8861`, and 024's freeze (024:105), committed in `566eb6f`. So
  the phrase does not show who made them.
- **012:** installed by the assistant at the user's instruction (message of `5cea6c4`).
- **021:** its plan gives the lock commit to the user
  (docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md:330–331, 340). The commit
  carries the Claude trailer and records only that the lock run was authorized (message of `b7e8861`).
- **022–024:** the plans run each phase only when separately authorized, without naming who authorizes
  (docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md:339–340;
  docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md:661).
- **025:** installed after the reviewer accepted the lock review and authorized installing the candidates
  (experiments/025-nounness-direction-intervention/evidence/lock-review-2026-09-26/REVIEW.md:1).

**The reviews.**
- **Implementations:** agents reviewed each implementation against its design, as the plans require
  (docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md:178;
  docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md:311). The reviewer also
  reviewed 021's implementation and set requirements for 024's
  (docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md:315; 024:62).
- **Locks:** agents reviewed them from 021 on (021:81–83; 022:147–148). Before 021, the records show a read-only
  post-lock review or check (012:67; 013:72; 014:76; 015:89; 016:88; 017:98; 018:81), the reviewer's post-lock review in
  011 (011:62) and the reviewer's sign-off, which is not recorded for 009, 016 and 017
  (research/synthesis/source-map-005-025.md@b722e9c:671).
- **Freezes, extraction, calibrations and confirmations:** agents reviewed 022's calibration (022:140) and, from 023 on,
  the freezes, the extraction, 024's calibration and the confirmations, with records in each experiment's `evidence/`
  folder. 025's include the prompts the agents were given
  (experiments/025-nounness-direction-intervention/evidence/lock-review-2026-09-26/PROMPT.md:1–3).
- **Calibration floors:** the reviewer reviewed the floors of 021 and 023
  (docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md:326;
  experiments/023-block0-completion/evidence/calibrate-2026-09-24/FLOOR-REVIEW.md:19).
- **The records:** agents also reviewed the claim records, the synthesis, the literature review, the figures and every
  draft.

## 4. What the record cannot settle

Only the author can answer these. The statement must not be frozen before they are answered.
- **Who or what the project's reviewer is.** There are three cases.
  - **The author.** The reviewer's decisions are then the author's own.
  - **An AI assistant the author consulted.** AI then proposed the decisions recorded as the reviewer's, and the author
    adopted them, and the statement must say so.
  - **Another person.** TMLR's authorship criteria and an acknowledgment must be settled first.

  The answer also covers the statement on AI tools itself, which the record gives as the reviewer's.
- **Where the research question and the experiments' directions came from.** The record gives part of the answer:
  - the project's aim is stated in the initial commit, made through GitHub (README.md@ef0c716:1–2);
  - the behavior was proposed by the behavior screening, designed in the Codex period and then implemented, frozen,
    amended after an independent pre-output review, run and recorded with the Claude trailer on 2026-09-17 (`3655a62`,
    `e610c5f`, `1f043d9`), as the subject of a user-approved Experiment 005
    (screening/behavior-candidates/report-2026-09-17.md:3, 182);
  - later directions followed exploratory diagnostics and design spikes run by the assistant, as the message of
    `219cdc5` and 018's design record
    (docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md:451–452), with the reviewer
    choosing among options: *the reviewer's choice of Option A* in 022's design (message of `2b20db3`), 024's design
    (docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md:7–10) and 025's
    (docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md:422–443).

  It does not say who proposed the options, who posed 005's question, or who wrote the methodology.
- **What lies behind each recorded act.** The record shows the lock commits, approvals, authorizations and pushes. It
  does not show the deliberation behind them, or who authorized the lock commits of 021–024.
- **How the misleading commit messages were written.** Six lock commits read *Refactor code structure for improved
  readability and maintainability*, which does not describe them
  (research/synthesis/source-map-005-025.md@b722e9c:702–704). `2e9994f`'s subject and body describe only test changes.
  The record shows how `2e9994f` was committed, but not how any of these messages was written.

## 5. Draft v3's statement against the record

1. **The tools.** “OpenAI- and Anthropic-based assistants” (research/paper/draft-v3-2026-09-30.md@7f2cce1:39–40) matches
   the Claude trailers and the Codex branches. Any further assistant, such as one that may stand behind the reviewer, is
   not in the record.
2. **The uses.** The list holds, but it understates AI's role:
   - AI drafted the designs, preregistrations and decision rules, not only checked them;
   - AI ran every phase of the experiments, which the statement does not mention;
   - agents performed the literature searches, which the statement calls *support*;
   - AI drafted the interpretations and the synthesis, and ran the diagnostics that led to later designs.
3. **What the author determined.** The statement says that the research question, the experimental scope, the
   preregistered decision rules, the interpretation of results, the decisions to accept or reject claims and the final
   manuscript content were all “determined and approved by the author”
   (research/paper/draft-v3-2026-09-30.md@7f2cce1:44). The record gives the approvals and decisions, where it names
   anyone, to the reviewer, and some acts to the user. AI drafted all of these things. Whether the author determined
   them depends on §4.
4. **The numbers.** “Reported numerical results were verified against the committed experimental record”
   (research/paper/draft-v3-2026-09-30.md@7f2cce1:44–45) holds, as work done by AI. The claims reviews, the figure-data
   script and the other reviews checked the numbers. The source-map checker checks quotes and traces, not numbers
   outside quotes (research/synthesis/checks/check_source_map.py@7f2cce1:22).
5. **The citations.** “literature citations were checked against their original sources”
   (research/paper/draft-v3-2026-09-30.md@7f2cce1:45–46) claims too much. Every bibliographic record was checked against
   its source. The paraphrases were spot-checked for the key entries only
   (research/paper/literature-review/verification-2026-09-28.md@7f2cce1:6–7).
6. **Anonymity.** “the author takes full responsibility” (research/paper/draft-v3-2026-09-30.md@7f2cce1:46) reveals a
   single author in the anonymous build.
7. **The method text.** Draft v3's §3 describes a review gate between phases
   (research/paper/draft-v3-2026-09-30.md@7f2cce1:179) without saying who reviewed. The format review made disclosure of
   the agents in the method itself a MUST (research/paper/paper-review-2026-09-30/format.md@7f2cce1:179–184).

## 6. A proposed statement

This follows the record. The reviewer's text of 2026-09-30 understates AI's role in the ways §5 lists. The sentence on
the authors' part comes in three variants, and none of them is the record's account until the author answers §4.

> **Use of AI tools.** Generative AI tools were used extensively in this work: Anthropic's Claude, through the Claude
> Code agent, and OpenAI's [Codex][, and ChatGPT, if §4 confirms it]. AI assistants drafted the experimental designs,
> preregistrations and decision rules; wrote the analysis code and tests; ran every phase of every experiment, including
> each single preregistered confirmation run, under protocols frozen before the runs; performed the literature searches,
> reading the sources partly through a page-summarizing tool; drafted the experiment reports and their interpretations,
> the claim records, the cross-experiment synthesis, the figures and this manuscript; and ran the exploratory
> diagnostics and design spikes that preceded later designs. Separate read-only instances of the same assistant reviewed
> the implementations, the freezes, calibrations, locks and confirmations (with committed records from Experiment 023
> on), the claim records, the synthesis, the literature review, the figures and every draft. These reviews and
> assistant-written scripts checked every reported number against the committed experimental record, and every
> bibliographic record against its source; the key paraphrases of prior work were spot-checked. [The sentence on the
> authors' part, from the variants below.] The authors take full responsibility for the manuscript.

The variants:
- **A. If the project's reviewer is the author.** *The authors set the project's aim, chose among the proposed design
  options, required changes to the designs, approved each design before it was implemented, committed or authorized each
  preregistration lock before its confirmation run, and decided which claims to register.*
- **B. If the reviewer is an AI assistant the author consulted.** *The decisions recorded as the project's reviewer's,
  approving and changing the designs, choosing among design options, signing off the locks and registering the claims,
  were proposed by [ChatGPT], acting as reviewer, and adopted by the authors. The authors committed or authorized each
  preregistration lock before its confirmation run.*
- **C. If the reviewer is another person.** Acknowledgment or authorship under TMLR's criteria must be settled first.

Each variant's acts must also be confirmed. The record shows eleven lock commits by the user, by inference, and one at
the user's instruction. It gives 025's lock to the reviewer's authorization, and leaves 021–024's authorizer open.

*The authors* is the usual double-blind form, and it does not reveal how many authors there are. The de-anonymized
version should give the true number.

## 7. TMLR's rules, as of 2026-10-01

An agent read TMLR's FAQ, author guidelines, submission page, editorial policies, reviewer guidelines and ethics
guidelines on 2026-10-01.
- **Disclosure.** The FAQ allows the use of large language models, and requires authors to mention it explicitly, in a
  footnote on the first page ([FAQ](https://jmlr.org/tmlr/faq.html), *Authors*). It does not say how much detail the
  footnote must give.
- **Human-sourced work.** The FAQ states the expectation that the ideas, claims and results in a submission are
  human-sourced, and not produced by such tools.
- **Editorial policy.** LLMs may be used as general-purpose assistive tools, authors are fully responsible for the
  content, and LLMs cannot be authors. Low-quality contributions that appear largely LLM-generated will be closely
  examined for misconduct ([editorial policies](https://jmlr.org/tmlr/editorial-policies.html), *LLM use policy for
  authors*).
- **Anonymity.** Submissions and supplementary material must be anonymized, and a public version must not be linked if
  it names the authors ([author guidelines](https://www.jmlr.org/tmlr/author-guide.html)). No page says how authors
  should refer to themselves.

The statement will be read against the human-sourcing expectation, which covers results as well as ideas and claims. The
record shows that AI drafted the designs, wrote the code that computed the results, ran it, and drafted the
interpretations and the manuscript. Whether the ideas, claims and results are human-sourced in TMLR's sense depends on
the author's answers to §4, which only the author can give. The statement should describe the division of work as it
was, and the author should judge whether the submission meets that expectation.

## 8. What follows

1. The author answers §4.
2. The statement is written from §6 and those answers. It replaces draft v3's statement in the draft and the build.
3. Draft v3's §3 says who reviewed at each gate (§5, point 7).
4. An independent read-only check compares the final statement and the method text with this audit.
5. The de-anonymized version gives the true number of authors.

## Review

Revision 1 had an independent read-only review on 2026-10-01 ([report](ai-use-audit-review-2026-10-01.md)). Its verdict
was FAIL, on one blocker, B1: the proposed statement gave the record's account of the reviewer's acts as the authors'
acts, and its AI-reviewer variant fell short of §4. It also raised twelve SHOULDs and thirteen notes. All were applied
in revision 2:
- the statement's sentence on the authors' part now comes in three variants;
- the repository, `2e9994f`, the lock commits and the outcome labels are stated as the record gives them;
- the statement under audit is now traced to the reviewer, and dated 2026-09-30;
- the reviewer's role and three cases are spelled out;
- the partial account of where the directions came from is given;
- the Codex period now covers the methodology and the claim registry;
- the AI-role sentences are made explicit;
- the class column is renamed;
- commit messages are used as a source;
- the method text is added to §5 and §8;
- the counts, citations and scope notes are corrected.

The same reviewer re-checked revision 2 (the report's second part). Its verdict was PASS WITH NOTES, with no blocker:
all 26 findings were resolved. It raised one new SHOULD and thirteen notes, all applied in revision 3:
- the behavior screening was designed in the Codex period, but implemented, frozen, amended, run and recorded with the
  Claude trailer;
- the READMEs' *by hand* does not show who made a lock commit;
- the lock checks before 021, 022's calibration review and the reviewer's implementation reviews are added;
- the paper decisions, the methodology's later changes, the claim texts and the diagnostics' sources are stated exactly;
- the statement brackets Codex and says *a page-summarizing tool* until the author confirms both, and it counts
  experiments by number.

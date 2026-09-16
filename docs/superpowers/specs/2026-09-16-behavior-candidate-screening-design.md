# Behavior Candidate Screening Design

**Date:** 2026-09-16

**Status:** Awaiting user review

**Scope:** Candidate selection before Experiment 005

## Purpose

Neural Decompiler must not create Experiment 005 until it has identified a
behavior that is reliable enough to study, plausibly admits a compact causal
account, and is not already substantially reverse-engineered in the literature.
This design specifies a deliberately small candidate-selection study. It does
not specify the eventual mechanistic explanation and it does not treat candidate
screening evidence as confirmation of that explanation.

The study answers one project-management question:

> Is there one simple, naturally learned behavior in pinned Pythia-70M that
> passes fixed behavioral, nontriviality, prior-art, and exploratory compactness
> gates strongly enough to justify a separately preregistered Experiment 005?

The allowed result is zero or one selected target. A zero-target result is not a
failure of the screening study.

## Design constraints

The study follows these constraints:

1. perform lightweight prior-art triage before implementing or running a model
   screen;
2. freeze cases, templates, splits, metrics, thresholds, baselines, retry rules,
   and the Pythia-160M fallback before inspecting model outputs;
3. keep causal compactness explicitly exploratory and restrict it to discovery
   data;
4. preserve untouched behavioral cases and intervention questions for a future
   Experiment 005;
5. implement only one declarative candidate file, one runner, and one
   machine-readable plus Markdown report;
6. do not create an Experiment 005 directory, claim, or preregistration unless a
   candidate survives every selection gate and the user separately approves the
   resulting Experiment 005 design.

## Lightweight prior-art triage

### Decision meanings

- **ELIMINATE:** close causal or circuit work already maps the same behavior or
  the candidate lies directly inside a heavily occupied mechanism family.
- **RETAIN:** no close component-level causal account was located in this
  lightweight search. This is not a novelty claim.
- **RETAIN — HIGH RISK:** the phenomenon has substantial behavioral or internal
  analysis, but this pass did not locate a complete component-level causal
  account matching the proposed Pythia behavior.

The search was conducted on 2026-09-16 using primary paper pages from arXiv,
ACL Anthology, PMLR, NeurIPS, and OpenReview. Search phrases combined each
behavior name with `mechanistic interpretability`, `circuit`, `causal`,
`transformer`, `language model`, and `Pythia`. Absence from this pass means only
"not located yet." Every behavioral finalist requires a deeper audit.

### Initial candidate set

| ID | Candidate behavior | Triage | Reason |
|---|---|---|---|
| `sva-attractor` | English subject–verb number agreement across intervening nouns | **ELIMINATE** | A Gemma-2B study reports an attention head writing a causal subject-number direction that final MLP neurons read, alongside earlier causal agreement work. This is too close to the proposed mechanism family. |
| `pronoun-fidelity` | Pronoun gender/number selection with competing referents | **ELIMINATE** | A 2026 causal study reports distributed group-binding, recency, and stereotype mechanisms explaining 91–99.5% across several models. |
| `npi-licensing` | Negative-polarity-item licensing under scope changes | **ELIMINATE** | CausalGym already studies causal variables for NPI licensing across Pythia scales and analyzes its training trajectory in Pythia-1B. |
| `sequence-successor` | Weekday, month, alphabet, or ordinal sequence continuation | **ELIMINATE** | Prior circuit work identifies shared sequence-member detection and successor-prediction subcircuits in GPT-2 and Llama; the broader successor-head neighborhood is crowded. |
| `literal-copying` | Literal token-sequence copying across token-frequency bands | **ELIMINATE** | A 2026 Pythia study extracts 75 circuits from 70M–1.4B, finds a shared core recovering at least 99% of circuit performance, and validates interchangeability causally. |
| `delimiter-closure` | Quote, parenthesis, or bracket-type closure | **ELIMINATE** | Recent solver-checkable work verifies quote-closing and bracket-type circuits on bounded domains, making this an especially direct collision. |
| `regular-plural` | Context-sensitive English plural inflection, including regular spelling allomorphs | **RETAIN** | Behavioral and morphosyntactic work exists, but this pass found no close end-to-end circuit account of plural generation in an unchanged pretrained autoregressive LM. |
| `regular-past` | Context-sensitive English past-tense inflection, including regular spelling allomorphs | **RETAIN** | Tense localization and causal-control work exists, but this pass found no close circuit-level account of the complete regular past-tense transformation. |
| `degree-inflection` | Comparative/superlative adjective inflection, including regular spelling changes | **RETAIN** | No close causal circuit account was located in the lightweight search. |
| `ordinal-suffix` | English ordinal suffix selection (`st`, `nd`, `rd`, `th`) under held-out numbers | **RETAIN** | No close causal circuit account was located; tokenizer and memorization confounds make behavioral screening essential. |
| `reflexive-number` | Reflexive-pronoun number agreement across a distractor noun | **RETAIN — HIGH RISK** | GPT-2 behavioral and attention analyses study reflexive interference, but this pass did not locate a compact end-to-end causal circuit matching the proposed behavior. |
| `article-allomorphy` | Preference for `a` versus `an` in matched phrase-level minimal pairs | **RETAIN** | No close causal circuit account was located; the design must rule out purely local token-frequency explanations. |

### Triage references

- Subject–verb agreement: [Ferrando and Costa-jussà, 2024](https://aclanthology.org/2024.findings-emnlp.591/)
- Pronoun fidelity: [Trinley et al., 2026](https://arxiv.org/abs/2606.16407)
- Negative-polarity licensing in Pythia: [Arora et al., 2024](https://aclanthology.org/2024.acl-long.785/)
- Sequence continuation circuits: [Gould et al., 2024](https://aclanthology.org/2024.emnlp-main.699/)
- Literal copying in Pythia: [Bayat Makou et al., 2026](https://arxiv.org/abs/2606.06267)
- Solver-checkable quote and bracket circuits: [Somani, 2026](https://arxiv.org/abs/2605.24033)
- Reflexive interference analysis: [Ryu and Lewis, 2021](https://aclanthology.org/2021.cmcl-1.6/)
- Adjacent causal morphosyntax work: [Amini et al., 2023](https://aclanthology.org/2023.tacl-1.23/)

The six retained candidates proceed to manifest construction. A candidate is
not restored after elimination without a new version of this design and an
explicit rationale committed before any model run.

## Frozen screening protocol

### Model order

1. The primary screen uses the exact pinned Pythia-70M model and revision
   already declared by the repository.
2. Pythia-160M is run only if **zero** candidates pass every Pythia-70M
   behavioral gate.
3. The 160M fallback uses byte-identical case manifests, metrics, baselines,
   and thresholds. Failure on 70M never permits a lowered threshold or changed
   template.
4. If one or more candidates pass on 70M, the candidate-selection study does
   not run 160M.

### Candidate manifests

Each retained candidate receives exactly three predeclared template families.
All concrete text, expected alternatives, lexical items, case IDs, tokenizer
eligibility decisions, and split assignments are generated and committed
before the first scientific model run.

Every evaluated item is a matched pair or matched set with:

- a grammatical/intended alternative;
- a minimally changed counterfactual whose intended choice reverses;
- one behavior-specific trivial heuristic prediction;
- a target position and explicit target-versus-foil tokens or sequence scores;
- tokenizer metadata recorded without inspecting model probabilities.

Targets and foils must have equal token counts within a case. Single-token
alternatives are preferred. A multi-token case is allowed only when both
alternatives have the same token count and the metric sums log probabilities
over the same number of positions.

### Split sizes

For each candidate and each of its three templates:

- **selection-development:** 40 cases;
- **selection-holdout:** 40 cases using disjoint lexical items or numbers;
- **future-reserve:** 40 cases using another disjoint set.

Each candidate therefore has 120 development, 120 selection-holdout, and 120
future-reserve cases. Pair members remain in the same split. Split assignment
uses integer seed **20260916** and is stratified by the candidate's controlling
feature and surface-rule exception class.

The future-reserve manifest and digest are committed, but neither Pythia-70M
nor Pythia-160M is run on those cases during candidate selection. Future
Experiment 005 may use them only after its predictions and intervention
criteria are committed.

### Candidate-specific template families

The manifest implementation must instantiate exactly these three families per
candidate; wording variants within a family are fixed in the manifest and may
not be selected based on model output.

| Candidate | Template family 1 | Template family 2 | Template family 3 |
|---|---|---|---|
| `regular-plural` | cardinal contrast (`one NOUN`; `two NOUNS`) | quantifier contrast (`each NOUN`; `several NOUNS`) | coordinated count context with an intervening adjective |
| `regular-past` | explicit today/yesterday contrast | earlier/later narrative contrast | temporal-adverb contrast with an intervening adverb |
| `degree-inflection` | two-item comparative | three-item superlative | explicit base/comparative analogy with held-out adjectives |
| `ordinal-suffix` | bare numeral plus suffix | dated-event completion | ranked-list completion without ordered-sequence context |
| `reflexive-number` | singular antecedent plus plural distractor | plural antecedent plus singular distractor | matched no-distractor control using the same antecedent lexicon |
| `article-allomorphy` | bare noun phrase | adjective-plus-noun phrase | matched phrase embedded in a short sentence |

The article candidate is evaluated by equal-length whole-sequence minimal-pair
scores because the article precedes the conditioning word. It is not presented
as a next-token article-generation task.

### Primary measurements

For every case the runner records:

- intended-alternative log probability;
- foil-alternative log probability;
- intended-minus-foil log-probability margin;
- binary pairwise correctness (`margin > 0`);
- full-vocabulary top-1 output where a single next-token target exists;
- the matched counterfactual's corresponding measurements;
- tokenizer and provenance fields required by the existing instrumentation.

Selection is based on pairwise correctness and margins. Full-vocabulary top-1
is descriptive because several grammatical continuations may exist.

### Trivial baselines and nontriviality

Every candidate is evaluated against all applicable baselines:

1. **Majority baseline:** always select the more frequent target class in the
   frozen case manifest. Balanced manifests should make this 50%.
2. **Lexical-prior baseline:** ignore the controlling context and predict from
   the lexical item's development-set majority form. Lexical items are disjoint
   in holdout, so unseen items back off to the global development majority.
3. **Local heuristic:** a predeclared behavior-specific shallow rule listed
   below. At least half of each split consists of cases on which that heuristic
   is wrong while the intended linguistic or numeric rule remains
   deterministic.
4. **Cue-shuffle control:** deterministically exchange the controlling cue
   between matched cases while leaving the local target region unchanged.
5. **Counterfactual flip:** require the margin sign to reverse when the
   controlling feature is minimally changed.

The local heuristics are frozen as follows:

| Candidate | Shallow heuristic deliberately stress-tested |
|---|---|
| `regular-plural` | append `s`, ignoring `es`, consonant-plus-`y`, and other predeclared regular spelling classes |
| `regular-past` | append `ed`, ignoring final-`e`, consonant-plus-`y`, and predeclared consonant-doubling classes |
| `degree-inflection` | append `er`/`est`, ignoring final-`e`, consonant-plus-`y`, and predeclared consonant-doubling classes |
| `ordinal-suffix` | choose from the final digit only, deliberately failing the `11`/`12`/`13` exceptions |
| `reflexive-number` | copy the number of the nearest noun rather than the grammatical antecedent |
| `article-allomorphy` | choose from the first written letter rather than pronunciation, tested on predeclared vowel/consonant-sound exceptions |

Only productive, predeclared spelling classes are eligible for the three
inflection candidates; idiosyncratic irregular forms such as `went` or `better`
are excluded. This makes the shallow rule beatable without turning the task
into arbitrary lexical memorization.

A candidate that can only be distinguished from chance after adding a new
baseline post hoc is not selected. Newly noticed confounds are documented and
may eliminate the candidate; they do not justify repairing its screen in the
same version.

### Behavioral pass gates

A candidate passes Pythia-70M behavioral screening only if all conditions hold
on the untouched selection-holdout split:

1. overall pairwise accuracy is at least **85%**;
2. the two-sided Wilson 95% lower confidence bound (`z =
   1.959963984540054`) for overall pairwise accuracy is at least **78%**;
3. pairwise accuracy is at least **75% in each template family**;
4. the range between best and worst template accuracy is no more than **15
   percentage points**;
5. the development-to-holdout accuracy drop is no more than **10 percentage
   points**;
6. mean intended-minus-foil margin is positive in every template family;
7. counterfactual margin sign flips correctly in at least **80%** of matched
   pairs;
8. cue shuffling reduces pairwise accuracy by at least **20 percentage points**
   or reduces the mean paired margin by at least **50%**;
9. model accuracy exceeds every applicable trivial baseline by at least **15
   percentage points**;
10. no tokenizer, leakage, duplicate-item, or case-integrity validation fails.

The same gates apply unchanged to the predeclared 160M fallback.

### Retry and amendment rules

- There are **zero scientific retries** in one protocol version: no candidate,
  template, lexical item, threshold, or baseline changes after model output is
  inspected.
- A software defect permits a rerun only after an incident note identifies the
  defect, affected artifacts are marked invalid, the fix is committed, and the
  entire affected model screen is rerun for every candidate—not only the
  candidate helped by the fix.
- Resource interruption may resume from provenance-verified completed cases.
- A second scientific attempt requires a new protocol version and new cases;
  results from the first version remain reported.

## Exploratory causal compactness probe

Only candidates passing every behavioral gate enter this phase. The probe uses
selection-development data only. Selection-holdout and future-reserve data are
never used for component ranking or intervention exploration.

### Components and intervention

The probe tests the fixed Pythia component vocabulary already supported by the
instrumentation:

- every attention-head result;
- every whole-MLP output.

For matched counterfactual pairs, the runner performs exact activation
replacement at the predeclared target position and measures the change in the
behavioral margin. Singleton effects are discovery scores, not additive causal
claims.

### Discovery/validation split inside development data

The 120 selection-development cases are deterministically divided into 60
compactness-discovery and 60 compactness-validation cases, stratified by
template. Components are ranked only on compactness-discovery cases.

On compactness-validation cases, cumulative replacement interventions evaluate
the top-ranked set. A candidate passes the exploratory compactness gate if some
set of at most **12 components**:

1. produces at least **70%** of the full matched-counterfactual margin shift;
2. exceeds the median effect of 100 deterministic, size-matched random component
   sets by at least a factor of **2**; and
3. has the same effect direction in all three template families.

This gate is a triage heuristic. It does not establish completeness,
minimality, self-repair robustness, or a human-readable mechanism. The future
Experiment 005 must independently test those properties with untouched cases
and preregistered predictions.

All inspected component rankings, unsuccessful subsets, and null results are
retained. The study may not search additional component bases, SAE features,
or arbitrary directions to rescue a failing candidate.

## Finalist prior-art audit

Every candidate passing behavioral and compactness gates receives a deeper
audit before selection. The audit must include:

- the exact behavior and close variants;
- the model family and Pythia-specific work;
- component-, edge-, feature-, subspace-, and program-level explanations;
- causal validation, reconstruction, self-repair, and held-out intervention
  work;
- papers published or posted through the audit date;
- a comparison table stating which proposed Experiment 005 contribution is
  absent from each close work.

"The behavior has been studied" does not by itself eliminate a finalist.
"A substantially equivalent naturally learned mechanism has already been
causally mapped" does.

Novelty status must be one of:

- `NO_CLOSE_MECHANISM_LOCATED`;
- `PARTIAL_OVERLAP_WITH_EXPLICIT_GAP`;
- `SUBSTANTIALLY_MAPPED`.

Only the first two statuses may survive, and partial overlap requires a concrete
contribution that is more than applying the same analysis to a smaller model.

## Selection rule

The study selects at most one target. After the full audit, surviving candidates
are ordered lexicographically by:

1. novelty status (`NO_CLOSE_MECHANISM_LOCATED` before partial overlap);
2. worst-template selection-holdout accuracy, descending;
3. overall selection-holdout accuracy, descending;
4. smallest validated compact component count, ascending;
5. counterfactual flip rate, descending;
6. candidate ID, ascending, as a deterministic final tie-breaker.

The report includes every candidate and every failed gate. The winner becomes a
proposal for Experiment 005, not an Experiment 005 result.

## Minimal future implementation

After this design is reviewed, implementation is limited to:

1. one versioned declarative candidate manifest containing templates, cases,
   splits, controls, and digests;
2. one screening command supporting `validate`, `behavioral`, `compactness`, and
   `report` phases with enforced phase isolation;
3. one JSON results artifact and one Markdown candidate matrix/report;
4. focused tests for manifest immutability, split isolation, metric correctness,
   phase gates, fallback enforcement, retry refusal, provenance, and reserve-set
   non-execution.

The implementation must reuse the existing model, behavior, capture,
intervention, component, and provenance modules. It must not introduce a plugin
system, database, dashboard, generalized benchmark API, SAE framework,
automated circuit-discovery package, or Experiment 005 directory.

## Deliverables and stopping condition

Candidate selection is complete when:

- the frozen manifest and its digest exist in version control;
- the prior-art triage and any finalist audits are recorded;
- the applicable model screen has run reproducibly;
- the matrix reports each behavioral, baseline, compactness, and novelty gate;
- future-reserve cases are proven unexecuted;
- the result is explicitly zero or one proposed target.

The project then stops. Designing Experiment 005 is a new architectural task
requiring a separate design, explicit intervention predictions, user approval,
and preregistration commit before any confirmatory run.

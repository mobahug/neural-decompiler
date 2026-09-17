# Behavior Candidate Screening Design

**Date:** 2026-09-16

**Status:** Reviewed and amended after second prior-art triage; `degree-inflection` eliminated on 2026-09-17 for tokenizer infeasibility (pre-output amendment)

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

The initial search was conducted on 2026-09-16 using primary paper pages from
arXiv, ACL Anthology, PMLR, NeurIPS, and OpenReview. Search phrases combined
each behavior name with `mechanistic interpretability`, `circuit`, `causal`,
`transformer`, `language model`, and `Pythia`.

A second, synonym-based pass was completed before manifest construction. It
combined each surviving behavior with `morphology`, `inflection`, `allomorphy`,
`causal circuit`, `subnetwork`, `circuit probing`, `activation patching`,
`feature`, `subspace`, and close surface-form terms. Searches included papers
posted through 2026-09-16, with particular attention to 2025--2026 work.
Absence from either pass means only "not located yet." Every behavioral
finalist still requires a deeper audit.

### Initial candidate set

| ID | Candidate behavior | Triage | Reason |
|---|---|---|---|
| `sva-attractor` | English subject–verb number agreement across intervening nouns | **ELIMINATE** | A Gemma-2B study reports an attention head writing a causal subject-number direction that final MLP neurons read, alongside earlier causal agreement work. This is too close to the proposed mechanism family. |
| `pronoun-fidelity` | Pronoun gender/number selection with competing referents | **ELIMINATE** | A 2026 causal study reports distributed group-binding, recency, and stereotype mechanisms explaining 91–99.5% across several models. |
| `npi-licensing` | Negative-polarity-item licensing under scope changes | **ELIMINATE** | CausalGym already studies causal variables for NPI licensing across Pythia scales and analyzes its training trajectory in Pythia-1B. |
| `sequence-successor` | Weekday, month, alphabet, or ordinal sequence continuation | **ELIMINATE** | Prior circuit work identifies shared sequence-member detection and successor-prediction subcircuits in GPT-2 and Llama; the broader successor-head neighborhood is crowded. |
| `literal-copying` | Literal token-sequence copying across token-frequency bands | **ELIMINATE** | A 2026 Pythia study extracts 75 circuits from 70M–1.4B, finds a shared core recovering at least 99% of circuit performance, and validates interchangeability causally. |
| `delimiter-closure` | Quote, parenthesis, or bracket-type closure | **ELIMINATE** | Recent solver-checkable work verifies quote-closing and bracket-type circuits on bounded domains, making this an especially direct collision. |
| `regular-plural` | Context-sensitive English plural inflection, including regular spelling allomorphs | **RETAIN — HIGH RISK** | No end-to-end causal generation circuit was located, but Li and Subramani probe singular/plural features and report compact steerable inflection subspaces across modern LMs, including Pythia. The remaining gap is the naturally used generation mechanism. |
| `regular-past` | Context-sensitive English past-tense inflection, including regular spelling allomorphs | **ELIMINATE** | Zhang et al. use path patching and ablation to identify English past-tense heads, copy heads, and late FFNs in pretrained transformers; Africa independently studies a GPT-2 conjugation circuit across temporal and morphological variants. This is already inside the proposed mechanism family. |
| `degree-inflection` | Comparative/superlative adjective inflection, including regular spelling changes | **RETAIN — HIGH RISK** | No end-to-end causal generation circuit was located, but Li and Subramani probe comparative/superlative features and report compact steerable inflection subspaces across modern LMs, including Pythia. The gap is downstream generation and component-level mechanism, not representation discovery. |
| `ordinal-suffix` | English ordinal suffix selection (`st`, `nd`, `rd`, `th`) under held-out numbers | **RETAIN** | No close causal circuit account was located; tokenizer and memorization confounds make behavioral screening essential. |
| `reflexive-number` | Reflexive-pronoun number agreement across a distractor noun | **ELIMINATE** | Circuit Probing explicitly identifies and causally ablates circuits responsible for reflexive anaphora in GPT-2 Small and Medium. |
| `article-allomorphy` | Preference for `a` versus `an` in matched phrase-level minimal pairs | **ELIMINATE** | Kim and Lee identify a one-dimensional phonological feature that causally drives a/an selection and trace how the forecasted trigger feature controls article prediction. This is a direct mechanistic account of the proposed behavior. |

### Triage references

- Subject–verb agreement: [Ferrando and Costa-jussà, 2024](https://aclanthology.org/2024.findings-emnlp.591/)
- Pronoun fidelity: [Trinley et al., 2026](https://arxiv.org/abs/2606.16407)
- Negative-polarity licensing in Pythia: [Arora et al., 2024](https://aclanthology.org/2024.acl-long.785/)
- Sequence continuation circuits: [Gould et al., 2024](https://aclanthology.org/2024.emnlp-main.699/)
- Literal copying in Pythia: [Bayat Makou et al., 2026](https://arxiv.org/abs/2606.06267)
- Solver-checkable quote and bracket circuits: [Somani, 2026](https://arxiv.org/abs/2605.24033)
- Reflexive interference analysis: [Ryu and Lewis, 2021](https://aclanthology.org/2021.cmcl-1.6/)
- Adjacent causal morphosyntax work: [Amini et al., 2023](https://aclanthology.org/2023.tacl-1.23/)
- Reflexive-anaphora circuits: [Lepori et al., 2025](https://arxiv.org/abs/2311.04354)
- English article allomorphy mechanism: [Kim and Lee, 2026](https://arxiv.org/abs/2609.04708)
- English past-tense circuitry: [Zhang et al., 2024](https://arxiv.org/abs/2410.09223)
- GPT-2 conjugation circuit: [Africa, 2025](https://arxiv.org/abs/2506.22105)
- Behavioral English past-tense study: [Ma and Gao, 2022](https://aclanthology.org/2022.aacl-main.81/)
- Inflectional representations and steering: [Li and Subramani, 2026](https://aclanthology.org/2026.acl-long.720/)

The three retained candidates proceeded to manifest construction; `degree-inflection` was subsequently eliminated before any output by the tokenizer-infeasibility amendment below, leaving two candidates. The second
pass found no close end-to-end causal generation account for regular plural,
degree inflection, or ordinal suffix selection. It did find adjacent work on
plural/degree representation and classifier-based steering, so those overlaps
must be carried into any finalist audit rather than treated as novelty.

A candidate is not restored after elimination without a new version of this
design and an explicit rationale committed before any model run.

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

#### Amendment — 2026-09-17: tokenizer-eligibility pool extension

Before any manifest case was written or inspected and before any model weight,
logit, probability, or scientific output was inspected, tokenizer-only
validation of the pinned 70M and 160M revisions found only nine eligible forms
in the original `regular-plural` / `future-reserve` / `simple` literal pool.
The existing pool is therefore extended by appending, in exactly this order,
`table`, `stone`, `field`, `road`, and `door`. This is a pre-output protocol
amendment, not a substitution selected from model behavior. The rule remains
unchanged: the manifest builder selects the first ten tokenizer-eligible simple
forms in the frozen ordered pool.

#### Amendment — 2026-09-17: complete-census degree-pool extension

The complete tokenizer-only census, still before any manifest output or any
weight, logit, or probability inspection, found two deficient degree spelling-
change pools: seven eligible forms in `selection-development` and nine in
`future-reserve`. Before encoding or testing any new form, append the following
fixed literals in the listed order. For development: `gentle`, `noble`,
`humble`, `simple`, `dense`, `loose`, `sane`, `stale`, `rude`, `crude`,
`angry`, `dirty`. For future reserve: `polite`, `remote`, `mature`, `severe`,
`vague`, `acute`, `fierce`, `huge`, `lame`, `empty`, `ugly`, `sleepy`. This
does not change the first-ten-eligible selection rule, token-length policy,
templates, seed, or any threshold.

#### Amendment — 2026-09-17: corrected development eligibility census

The prior complete census correctly checked equality for every split, but did
not apply the stricter development-only one-token rule while classifying word
eligibility. Before encoding or testing these new forms, correct that predicate
and append the following fixed ordered lists. Development/simple: `black`,
`brown`, `green`, `round`, `straight`, `narrow`, `quiet`, `brisk`, `faint`,
`harsh`, `cheap`, `dear`, `full`, `hollow`, `smart`, `solid`, `strict`,
`swift`, `tight`, `tough`. Development/change: `rainy`, `windy`, `tidy`,
`crazy`, `lively`, `merry`, `cheery`, `cloudy`, `dusty`, `frosty`, `silky`,
`spicy`, `itchy`, `witty`, `silly`, `jolly`, `nasty`, `thirsty`, `filthy`,
`stormy`. These forms were fixed before encoding; the first-ten-eligible and
all other frozen rules remain unchanged.

#### Amendment — 2026-09-17: `degree-inflection` eliminated for tokenizer infeasibility

After the three pool extensions above, and still before any manifest case,
model weight, logit, probability, or scientific output was inspected, an
exhaustive tokenizer-only census of the shared pinned Pythia vocabulary showed
that the frozen rules cannot all hold for `degree-inflection`. The protocol
requires ten predeclared spelling-change adjectives per split, and every
`selection-development` alternative must be exactly one token so that the
compactness probe can intervene at one final prompt position. Under the pinned
tokenizer only four ordinary spelling-change adjectives (`large`, `easy`,
`big`, `simple`) have single-token comparative and superlative forms in every
frozen template variant; every consonant-plus-`y` comparative or superlative
(for example ` happier`, ` happiest`) is two tokens, and no further disjoint
ordinary final-`e`, consonant-plus-`y`, or consonant-doubling adjective exists
in the compatible single-token vocabulary. The alternative of defining a
multi-token, teacher-forced compactness intervention was rejected before any
output because protocol v1 deliberately has no multi-token compactness path.

`degree-inflection` is therefore **ELIMINATED — TOKENIZER INFEASIBLE** in
protocol v1. This is a pre-output amendment: no behavioral information
influenced it, and it cannot be reversed in this protocol version. The retained
candidate set is exactly `regular-plural` and `ordinal-suffix`, in that order.
Every per-candidate rule is unchanged; the committed manifest therefore
contains 720 cases (two candidates, three splits of 120). The `regular-plural`
and `ordinal-suffix` pools already satisfy every tokenizer rule without
further extension. Compactness cases remain single-token only.

The local-heuristic baseline is clarified, not changed: it reads the controlling
cue and applies the shallow surface rule, so it predicts the bare noun in a
singular context and `base + s` in a plural context, and the final-digit suffix
for ordinals. It is therefore wrong on exactly the 20 spelling-change or
`11`/`12`/`13` primary cases in each 40-case template stratum, which is what
"at least half of each split" requires.

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
| `degree-inflection` | two-item comparative | three-item superlative | explicit base/comparative analogy with held-out adjectives |
| `ordinal-suffix` | bare numeral plus suffix | dated-event completion | ranked-list completion without ordered-sequence context |

### Primary measurements

For every case, let `A` and `B` denote the two alternatives in a fixed canonical
order that never changes between an original prompt and its matched
counterfactual. Define the fixed-orientation contrast

```text
c(x) = log P(A | x) - log P(B | x)
```

and let `y(x) = +1` when `A` is intended and `y(x) = -1` when `B` is intended.
The correctness margin is

```text
m(x) = y(x) * c(x).
```

Thus `m(x) > 0` means the model prefers the intended answer in both the
original and counterfactual conditions. The runner records:

- intended-alternative log probability;
- foil-alternative log probability;
- fixed-orientation contrast `c(x)`;
- intended-minus-foil correctness margin `m(x)`;
- binary pairwise correctness (`m(x) > 0`);
- full-vocabulary top-1 output where a single next-token target exists;
- the matched counterfactual's corresponding measurements;
- tokenizer and provenance fields required by the existing instrumentation.

Selection is based on pairwise correctness and correctness margins. Any
directional comparison across original/counterfactual prompts--including sign
flips, cue shuffling, and activation-patching recovery--uses `c(x)`, never
`m(x)`. Full-vocabulary top-1 is descriptive because several grammatical
continuations may exist.

For a matched pair, write `x_A` for the prompt where `A` is intended and `x_B`
for the prompt where `B` is intended. Its full fixed-orientation contrast shift
is

```text
d_full = c(x_A) - c(x_B).
```

The counterfactual sign-flip condition is `c(x_A) > 0` and `c(x_B) < 0`.
For the cue-shuffle control, `s(x_A)` receives the controlling cue from `x_B`
and vice versa while the local target region is unchanged. Its aligned cue
effect is

```text
d_cue = 0.5 * ((c(x_A) - c(s(x_A))) + (c(s(x_B)) - c(x_B))).
```

These definitions are frozen before any model output is inspected.

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
5. **Counterfactual flip:** require the fixed-orientation contrast `c(x)` to
   reverse sign when the controlling feature is minimally changed; the
   correctness margin `m(x)` should remain positive in both conditions.

The local heuristics are frozen as follows:

| Candidate | Shallow heuristic deliberately stress-tested |
|---|---|
| `regular-plural` | append `s`, ignoring `es`, consonant-plus-`y`, and other predeclared regular spelling classes |
| `degree-inflection` | append `er`/`est`, ignoring final-`e`, consonant-plus-`y`, and predeclared consonant-doubling classes |
| `ordinal-suffix` | choose from the final digit only, deliberately failing the `11`/`12`/`13` exceptions |

Only productive, predeclared spelling classes are eligible for plural and
degree inflection; idiosyncratic irregular forms such as `children` or `better`
are excluded. This makes the shallow rule beatable without turning the task
into arbitrary lexical memorization.

A candidate that can only be distinguished from chance after adding a new
baseline post hoc is not selected. Newly noticed confounds are documented and
may eliminate the candidate; they do not justify repairing its screen in the
same version.

### Behavioral pass gates

A candidate passes Pythia-70M behavioral screening only if all conditions hold
on the untouched selection-holdout split:

1. overall pairwise accuracy is at least **85%**; with `n = 120`, the combined
   accuracy and Wilson requirements make the effective minimum **103/120
   correct (85.83%)**;
2. the two-sided Wilson 95% lower confidence bound (`z =
   1.959963984540054`) for overall pairwise accuracy is at least **78%**
   (`102/120` gives `77.5325%`, while `103/120` gives `78.4805%`);
3. pairwise accuracy is at least **75% in each template family**;
4. the range between best and worst template accuracy is no more than **15
   percentage points**;
5. the development-to-holdout accuracy drop is no more than **10 percentage
   points**;
6. mean intended-minus-foil margin is positive in every template family;
7. the fixed-orientation contrast satisfies `c(x_A) > 0` and `c(x_B) < 0` in
   at least **80%** of matched pairs;
8. cue shuffling reduces pairwise accuracy by at least **20 percentage points**
   or the mean aligned cue effect is at least **50%** of the mean full contrast
   shift (`mean(d_cue) >= 0.5 * mean(d_full)`);
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
fixed-orientation contrast. It evaluates both directions for every component
set `S`:

```text
c_A       = c(x_A)
c_B       = c(x_B)
c_A<-B(S) = c(x_A with components S replaced from the x_B run)
c_B<-A(S) = c(x_B with components S replaced from the x_A run)

d_full       = c_A - c_B
d_patch(S)   = 0.5 * ((c_A - c_A<-B(S)) + (c_B<-A(S) - c_B))
```

Positive `d_patch(S)` is therefore always movement in the matched
counterfactual direction. Singleton effects are discovery scores, not additive
causal claims.

### Discovery/validation split inside development data

The 120 selection-development cases are deterministically divided into 60
compactness-discovery and 60 compactness-validation cases, stratified by
template. Components are ranked only on compactness-discovery cases.

Components are ranked by `mean(d_patch({component}))` on discovery cases, with
ties broken by the canonical component identifier. On compactness-validation
cases, cumulative replacement interventions evaluate the fixed top-`k` sets
for `k = 1..12`.

For a validation case set `D`, recovery is the ratio of aggregate shifts, not
the mean of per-case ratios:

```text
R_D(S) = mean_{i in D}(d_patch_i(S)) / mean_{i in D}(d_full_i).
```

The denominator must be at least **0.25 nats overall** and **0.10 nats in each
template family**. A candidate with a smaller or non-positive denominator fails
compactness as an insufficiently identified causal contrast; the runner never
divides by it and never substitutes an epsilon after seeing results.

For each tested `k`, the runner also draws 100 size-`k` component sets uniformly
without replacement from the same component universe using seed **20260917**.
Let `M_k` be the median of their raw, aligned validation recoveries. Define the
random reference as `B_k = max(M_k, 0.05)`. This makes a zero or negative random
median non-vacuous rather than allowing any positive effect to pass.

A candidate passes the exploratory compactness gate if the smallest cumulative
top-`k` set, for some `k <= 12`:

1. has overall recovery `R_D(S) >= 0.70`;
2. satisfies `R_D(S) >= 2 * B_k`; and
3. has positive aggregate patched shift and positive recovery in every template
   family.

Raw recovery is reported without clipping, including values below zero or above
one. The deterministic seed, sampled component IDs, all denominators, random
recoveries, `M_k`, and `B_k` are retained in the result artifact.

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

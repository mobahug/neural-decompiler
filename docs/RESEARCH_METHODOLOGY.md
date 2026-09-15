# Neural Decompiler Research Methodology

## Purpose

Neural Decompiler investigates whether learned transformer computations can be recovered as increasingly complete, causal, and human-understandable explanations. The long-term objective is not merely to visualize activations. It is to develop and evaluate explanations that make testable predictions about model behavior and internal interventions.

No checklist establishes universal understanding of neural networks. At every maturity level, conclusions remain limited to the declared models, checkpoints, tasks, data, prompts, metrics, and interventions.

## Core principles

1. **Measurements precede interpretations.** Observed activation or logit changes are reported without assigning a mechanism prematurely.
2. **Experiments and claims are separate.** Completing an experiment records what was done. It does not automatically establish a claim.
3. **Every checked gate cites evidence.** A checkbox without a concrete evidence identifier is not satisfied.
4. **Causal language requires interventions.** Correlation, attribution, and localization alone do not establish causality.
5. **Falsification is part of the method.** Counterexamples, null results, failed hypotheses, and competing explanations are retained.
6. **Claims can weaken.** New evidence can narrow, contest, downgrade, or reject a claim.
7. **Conclusions are scoped.** The project states the strongest conclusion supported within the tested scope and no stronger.

## Records and authority

The project uses three distinct records:

- **Experiment records** describe a research question, protocol, execution, validation, artifacts, and limitations. They answer: “What did we do and observe?”
- **Evidence records** connect a concrete result or methodological validation to a claim. They answer: “What happened, and where is the supporting material?”
- **Claim records** evaluate evidence against explicit gates. They answer: “What are we currently justified in believing?”

Each file under `research/claims/` is the authoritative source for that claim's evidence gates, current supported maturity, disposition, limitations, and scoped conclusion. This methodology defines the rules and template only. It contains no live per-claim status.

Experiment and claim identifiers are independent. One experiment may bear on several claims, and one claim may accumulate evidence from several experiments.

## Experiment completion

An experiment README tracks completion independently from claim maturity. The following is a recommended baseline rather than a universal artifact schema:

- [ ] Research question documented
- [ ] Protocol documented
- [ ] Implementation and test status documented
- [ ] Real experiment execution status documented
- [ ] Measurement-validity checks documented
- [ ] Results and artifact locations documented
- [ ] Limitations documented
- [ ] Related claim IDs linked, if any

Experiment-specific outputs may differ. An HTML report, SVG plot, notebook, or any other particular format is required only when justified by that experiment's design.

## Evidence maturity ladder

The ladder describes increasing evidential strength where appropriate. A claim does not need to reach its highest level. A narrow observational claim may legitimately remain observational.

All required gates through a maturity level must be explicitly satisfied in the claim file before the claim moves to that level. A claim may add stricter, claim-specific gates. A gate marked not applicable requires a written justification and cannot bypass evidence fundamental to the requested maturity.

### Level 0 — Measurement validity

Before a measurement can support a claim:

- The measured quantities and success criteria are operationally defined.
- Inputs, tokenization, positions, targets, and other relevant parameters are recorded.
- Reconstructed outputs agree with appropriate model or system ground truth.
- Numerical validation, tolerances, and failure behavior are documented.
- Model revision, dependencies, runtime conditions, and artifact locations are recorded.
- The procedure is reproducible from version-controlled instructions.

### Level 1 — Observation

- The phenomenon is measured directly.
- The sample and scope are explicit.
- Variation across the tested cases is quantified where applicable.
- Repeated or related cases are examined when the claim concerns a pattern.
- The result is labeled exploratory or confirmatory.
- The conclusion avoids causal interpretation.

### Level 2 — Localization

- The phenomenon is narrowed to defined layers, positions, or components.
- Plausible alternative locations are compared.
- Obvious measurement artifacts and confounders are tested.
- The localization replicates within its declared scope.
- Localization is not presented as proof of causality.

### Level 3 — Causal evidence

- An intervention and its predicted outcome are documented before the relevant result is inspected.
- A necessity-oriented intervention, such as an appropriate ablation, supports the prediction where applicable.
- A sufficiency-oriented intervention, such as activation patching or reconstruction, supports the prediction where applicable.
- Matched, negative, and other appropriate controls are included.
- Effect sizes and uncertainty are reported.
- Plausible alternative causal explanations are considered.

The intervention types depend on the claim. Substituting another valid causal test requires an explicit rationale; it is not a way to omit necessity, sufficiency, or control logic when those are fundamental to the claim.

For this project, “documented before the result” means that the hypothesis, prediction, intervention, and outcome criteria exist in version control before inspecting the relevant intervention result. A Git commit is sufficient unless a later venue or collaboration requires formal external preregistration.

### Level 4 — Falsification tested

- Counterexamples are actively sought.
- Semantic paraphrases and adversarial cases are tested where relevant.
- Competing explanations are stated and tested using discriminating predictions.
- Failed predictions, null results, and rejected hypotheses are retained.
- The claim is narrowed or downgraded when the evidence requires it.

### Level 5 — Generalized

Generalization is always relative to declared axes. Relevant axes can include:

- new examples, facts, or datasets;
- new prompt structures or languages;
- new task distributions;
- another checkpoint or model scale;
- another model family.

A claim at this level states exactly which axes changed, what remained fixed, and where generalization failed. “Generalized” never means universal.

### Level 6 — Algorithmic explanation

- A compact, human-readable mechanism is specified precisely.
- The mechanism predicts unseen model behavior.
- The mechanism predicts outcomes of new interventions.
- A simplified or reconstructed circuit retains the relevant behavior where the claim requires it.
- Residual unexplained behavior is quantified.
- Known boundaries and failure cases are documented.

## Maturity and disposition

Each claim records two independent fields.

Evidence maturity is the highest level currently supported:

```text
PROPOSED
MEASUREMENT_VALIDATED
OBSERVATIONAL
LOCALIZED
CAUSAL_EVIDENCE
FALSIFICATION_TESTED
GENERALIZED
ALGORITHMIC_EXPLANATION
```

Disposition describes how the project currently treats the claim:

```text
ACTIVE
CONTESTED
NOT_SUPPORTED
SUPERSEDED
RETIRED
```

For example, a claim can have current maturity `CAUSAL_EVIDENCE` and disposition `CONTESTED`. The maturity field records the strongest level that remains supported, not a historical high-water mark. Previous states are preserved in status history.

## Evidence requirements

Evidence identifiers are local, stable references such as `E001` and `E002`. Every checked gate cites one or more evidence identifiers. Each evidence entry records:

- experiment ID or other source;
- evidence type: supporting, contradicting, null, or methodological;
- artifact path or durable external reference;
- concise result summary;
- model/data revision when relevant;
- source-code commit or revision when available;
- artifact digest when useful for generated or externally stored results.

Generated artifacts may remain outside version control when an experiment requires that policy. In that case the evidence entry records enough provenance to regenerate the artifact and, when practical, a digest identifying the exact evaluated copy.

## Exploratory and confirmatory results

An **exploratory** result is one for which the hypothesis or important outcome pattern was formulated after inspecting the relevant data. It can motivate a claim, but the same cases cannot then serve as an independent confirmation of that claim.

A **confirmatory** result tests a prediction and outcome criteria documented in version control before the relevant result is inspected. Held-out data or a genuinely new run must be used when prior results informed the prediction.

Both forms are valuable. They must not be presented as interchangeable.

## Promotion, contestation, and downgrade

Claim review follows these rules:

1. Add new evidence without deleting inconvenient prior evidence.
2. Determine which gates the evidence supports or invalidates.
3. Check or uncheck gates and cite the relevant evidence identifiers.
4. Set current maturity to the highest level whose required gates remain satisfied.
5. Set disposition to `CONTESTED` immediately when credible evidence creates a material unresolved conflict.
6. Narrow, downgrade, mark unsupported, or supersede the claim when review resolves the conflict that way.
7. Record the previous and new maturity and disposition, date, reason, and evidence in status history.

Promotion is not permanent. Downgrade is expected scientific behavior when later evidence overturns an earlier assessment.

## Scoped conclusions

Every active claim states the strongest conclusion currently justified and nearby claims that are not supported. A suitable form is:

> Within the tested models, tasks, prompts, measurements, and interventions, the evidence supports conclusion X, subject to the documented limitations.

Completing all applicable gates does not prove a universal truth or imply that the entire model is understood.

## Claim template

```markdown
# CXXX: Concise claim title

## Metadata

- Current supported maturity: PROPOSED
- Disposition: ACTIVE
- Created: YYYY-MM-DD
- Last reviewed: YYYY-MM-DD
- Origin: exploratory | confirmatory
- Related experiments: EXPERIMENT-XXX

## Hypothesis

A precise, falsifiable statement.

## Scope

Models, checkpoints, tasks, prompts, metrics, and interventions covered.

## Operational definitions

Exact meanings of terms that determine whether the claim is supported.

## Predictions

Expected observations or intervention outcomes if the claim is correct.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | EXPERIMENT-XXX | supporting | `path` | Concise measured result | Commit or digest |

## Evidence gates

### Level 0 — Measurement validity

- [ ] Gate description. Evidence: E001.

### Level 1 — Observation

- [ ] Gate description. Evidence: E001.

### Level 2 — Localization

- [ ] Gate description.

### Level 3 — Causal evidence

- [ ] Gate description.

### Level 4 — Falsification tested

- [ ] Gate description.

### Level 5 — Generalized

- [ ] Gate description.

### Level 6 — Algorithmic explanation

- [ ] Gate description.

## Contradicting evidence and failed tests

Retain negative, null, anomalous, and contradicting results.

## Competing explanations

State alternatives and tests capable of distinguishing them.

## Supported conclusion

The strongest statement currently justified within the declared scope.

## Not supported

Nearby interpretations that the evidence does not justify.

## Limitations

Known boundaries, confounders, and unresolved questions.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
```

## From experiment to conclusion

```text
research question
       ↓
hypothesis and scoped predictions
       ↓
experiment and measurement validation
       ↓
supporting, null, or contradicting evidence
       ↓
claim-level gate review
       ↓
promote, preserve, narrow, contest, or downgrade
       ↓
strongest justified scoped conclusion
```

## Publication readiness

The evidence ladder organizes claim maturity; it is not a publication checklist. A publishable result additionally requires a meaningful research question, engagement with prior work, appropriate baselines, reproducible methods, suitable statistical treatment, honest negative results and limitations, and evidence of novelty. Those requirements depend on the target contribution and publication venue.

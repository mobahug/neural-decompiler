# Anomaly Registry

This directory is Neural Decompiler's lightweight discovery inbox. It preserves unexpected measurements long enough to verify, reproduce, explain, or reject them without prematurely treating them as scientific claims.

Each `Axxx-<slug>.md` file is authoritative for that anomaly's evidence, provenance, current status, and resolution. This registry lists anomaly IDs and links only; it must not duplicate live status. No anomaly is created merely to populate the registry.

See the [research methodology](../../docs/RESEARCH_METHODOLOGY.md) for discovery, confirmation, triage, and stopping rules.

## Categories

- `DATA_BEHAVIORAL`
- `METHOD_DISAGREEMENT`
- `SCALE_TRAINING_DYNAMICS`

The methodology's [anomaly categories](../../docs/RESEARCH_METHODOLOGY.md#anomaly-categories) section is authoritative for their meanings and requirements. Categories may be added only when concrete experiments require them.

## Statuses

- `OPEN`
- `REPRODUCED`
- `EXPLAINED`
- `NOT_REPRODUCED`
- `PROMOTED_TO_HYPOTHESIS`
- `PROMOTED_TO_CLAIM`
- `RETIRED`

The methodology's [anomaly lifecycle](../../docs/RESEARCH_METHODOLOGY.md#anomaly-lifecycle) section is authoritative for status meanings. Most anomalies are expected to terminate without becoming claims.

## Anomalies

No anomalies have been registered.

## Record template

```markdown
# AXXX: Concise anomaly title

## Metadata

- Category: DATA_BEHAVIORAL | METHOD_DISAGREEMENT | SCALE_TRAINING_DYNAMICS
- Status: OPEN
- Origin: exploratory
- Last reviewed: YYYY-MM-DD
- Related claims: none

## Discovery provenance — preserve original context

- Source experiment ID: <required, or unavailable/not applicable with reason>
- Exact source artifact: <required, or unavailable/not applicable with reason>
- Artifact digest: <required, or unavailable/not applicable with reason>
- Analysis-code commit SHA: <required, or unavailable/not applicable with reason>
- Model identifier: <required, or unavailable/not applicable with reason>
- Model checkpoint/revision: <required, or unavailable/not applicable with reason>
- Detection-rule version or exact original wording: <required>
- Date first identified: YYYY-MM-DD

Do not silently rewrite this section when later analysis changes the
interpretation. Append a dated correction and preserve it in status history.
Blank provenance fields are invalid.

## Expected pattern

State the expectation or baseline and where it came from.

## Observed pattern

State the measurement, effect size, and uncertainty where available.

## Detection rule

Define what identified the anomaly and whether the rule was specified before
or after viewing the source data.

## Scope

Models, checkpoints, inputs, metrics, positions, and methods covered.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|

## Baseline or comparison set

- Comparison:
- Why it is appropriate:
- Matching variables:
- Inclusion and exclusion criteria:
- Sample-size rationale:
- Expected variance:
- Known mismatches:
- Selection or multiple-comparison concerns:

If no suitable comparison exists, explain why and keep the anomaly provisional.

## Verification and independent reproduction

Record integrity checks, new evidence, and results. Discovery data cannot serve
as independent confirmation.

## Mundane explanations

| Explanation | Discriminating test | Result | Status |
|---|---|---|---|

## Scientific hypotheses

Record possible explanations as hypotheses, not findings.

## Method disagreement

Required for METHOD_DISAGREEMENT; otherwise optional.

- Method A measures:
- Method B measures:
- Material incompatibility:
- Test capable of distinguishing the explanations:

## Confirmation boundary

- Discovery data that cannot be reused for confirmation:
- Acceptable independent evidence:

## Investigation stop rule

- Result that justifies continuing:
- Result that falsifies or substantially weakens the anomaly:
- Investigation budget or stopping condition:
- Decision when the stopping condition is reached:

## Relationship to prior work

Identify the nearest relevant methods and findings.

## Potential novelty if confirmed

State what might be new. This is not a novelty claim without systematic review.

## Triage decision

- Next action:
- Rationale:
- Reproducibility evidence:
- Mundane explanations considered:

High novelty potential alone is insufficient reason to continue.

## Resolution

Record the current or final resolution and links to resulting claims.

## Status history

| Date | Previous | New | Reason | Evidence |
|---|---|---|---|---|
```

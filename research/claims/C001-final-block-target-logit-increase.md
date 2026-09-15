# C001: Largest target-logit increase occurs across the final block for four tested capital prompts

## Metadata

- Current supported maturity: `OBSERVATIONAL`
- Disposition: `ACTIVE`
- Created: 2026-09-15
- Last reviewed: 2026-09-15
- Origin: exploratory
- Related experiments: Experiment 001

## Hypothesis

For each of the four capital-recall prompts measured in Experiment 001 with `EleutherAI/pythia-70m-deduped`, the intended target token has its largest positive adjacent-stage logit change from `after_layer_4` to `after_layer_5` under the experiment's logit-lens projection.

This hypothesis was formulated after inspecting the Experiment 001 measurements. The four cases are supporting exploratory observations, not an independent confirmatory test.

## Scope

- Model: `EleutherAI/pythia-70m-deduped`
- Resolved model revision: `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`
- Device and dtype: CPU, float32
- Task form: next-token projection for four English capital-recall prompts
- Prompts and targets: France/`" Paris"`, Germany/`" Berlin"`, Italy/`" Rome"`, and Finland/`" Helsinki"`
- Metric: change in the target token's raw logit between adjacent normalized accumulated-residual projections
- Stages: embedding and the output of each of Pythia-70M's six transformer blocks
- Interventions: none

The claim does not extend to other facts, prompts, languages, checkpoints, models, model families, or causal mechanisms.

## Operational definitions

- **Target logit** means the raw logit assigned to the specified single-token target after applying the model's final normalization and complete unembedding to an accumulated residual-stream state.
- **Adjacent-stage logit change** means the later stage's target logit minus the immediately preceding stage's target logit.
- **Largest positive change** means the numerically greatest adjacent-stage target-logit delta within one prompt's measured trajectory.
- **Final block transition** means `after_layer_4` to `after_layer_5`.

Absolute logit magnitude is not interpreted independently of the vocabulary distribution. The projected softmax values are not treated as actual intermediate model predictions.

## Predictions

Because this claim is exploratory, the existing four cases are observations from which the hypothesis was formed. A future confirmatory test should document a held-out prompt set and success criteria before inspecting its trajectories. The claim predicts that each held-out case meeting the declared inclusion criteria will have its largest positive adjacent-stage target-logit change across the final block transition.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | Experiment 001 | methodological | [`results.json`](../../outputs/experiment-001/results.json) | For all four prompts, projected final logits exactly matched actual logits; target rank and top-1 also matched. | Code commit `dffce3126cd78465f9708b593f12187ab6d19ea0`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E002 | Experiment 001, France | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Paris"`: largest increase was `after_layer_4` → `after_layer_5`, +714.766296; final rank 50; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E003 | Experiment 001, Germany | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Berlin"`: largest increase was `after_layer_4` → `after_layer_5`, +718.587402; final rank 60; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E004 | Experiment 001, Italy | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Rome"`: largest increase was `after_layer_4` → `after_layer_5`, +721.740479; final rank 120; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E005 | Experiment 001, Finland | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Helsinki"`: largest increase was `after_layer_4` → `after_layer_5`, +702.356262; final rank 185; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E006 | Experiment 001 documentation | methodological | [Experiment README](../../experiments/001-capital-recall/README.md) | Documents the one-command reproduction procedure, locked environment, generated artifacts, and stop-on-parity-failure behavior. | Code commit `dffce3126cd78465f9708b593f12187ab6d19ea0` |

Across these four prompts, the final-block target-logit increase ranged from +702.356262 to +721.740479. All four actual final predictions were the token `" the"`, not their intended capital tokens.

The output directory is generated and gitignored by Experiment 001. E001 records the exact evaluated artifact's digest; the artifact can be regenerated using the experiment's documented command and locked dependencies.

## Evidence gates

### Level 0 — Measurement validity

- [x] The target metric, stages, token positions, and single-token targets are operationally defined. Evidence: E001.
- [x] The final residual projection agrees with actual final logits for all four prompts with maximum absolute logit difference 0.0. Evidence: E001.
- [x] Final target rank and top-1 token agree between projection and actual output for all four prompts. Evidence: E001.
- [x] Model revision, runtime dependencies, prompt tokens, token IDs, tolerances, and measurements are recorded. Evidence: E001.
- [x] Numerical failure behavior stops the experiment on target-rank or top-1 disagreement instead of weakening validation. Evidence: E006.
- [x] A version-controlled one-command procedure and locked dependencies can regenerate the artifact. Evidence: E001, E006.

### Level 1 — Observation

- [x] The adjacent-stage trajectory is measured directly for every scoped prompt. Evidence: E002, E003, E004, E005.
- [x] All four scoped cases exhibit the stated final-block maximum. Evidence: E002, E003, E004, E005.
- [x] Per-case effect magnitudes and final ranks are reported rather than replacing variation with only an aggregate. Evidence: E002, E003, E004, E005.
- [x] The claim is labeled exploratory because it was formulated after inspecting these results. Evidence: E002, E003, E004, E005.
- [x] The supported conclusion is observational and makes no component-level or causal assertion. Evidence: E001.

### Level 2 — Localization

- [ ] The pattern has been evaluated with a localization protocol and appropriate artifact controls.
- [ ] Localization has replicated within a declared scope.

### Level 3 — Causal evidence

- [ ] A version-controlled intervention prediction and outcome criteria exist before inspecting intervention results.
- [ ] Appropriate necessity, sufficiency, and control interventions support the hypothesis.

### Level 4 — Falsification tested

- [ ] Held-out counterexamples, paraphrases, adversarial cases, and competing explanations have been tested as appropriate.
- [ ] Failed predictions and boundary conditions have been incorporated into the claim.

### Level 5 — Generalized

- [ ] Generalization has been demonstrated along explicitly declared axes outside the current four prompts and one model revision.

### Level 6 — Algorithmic explanation

- [ ] A compact mechanism predicts unseen behavior and intervention outcomes within a declared scope.
- [ ] Relevant retained behavior and residual unexplained behavior have been quantified.

Levels 2–6 are not required destinations for this narrow descriptive claim. They remain visible so that any attempted promotion is evaluated explicitly.

## Contradicting evidence and failed tests

No contradicting experiment is currently attached. This means only that none has yet been recorded, not that the claim has survived active falsification.

The intended capital token was not the actual top-1 prediction for any of the four prompts. This does not contradict the logit-change claim, but it sharply limits any interpretation of the pattern as successful factual recall.

## Competing explanations

The shared final-block increase could reflect a broad transformation affecting many vocabulary logits rather than capital-specific processing. It could also depend on the common prompt template, token frequency, target-token properties, final normalization, or another uncontrolled feature of this four-item sample.

Distinguishing these possibilities requires predefined controls and held-out prompts. Experiment 001 does not do so.

## Supported conclusion

Within the four prompts, one Pythia-70M checkpoint, and the specified logit-lens measurement, each intended capital token's largest positive adjacent-stage logit change occurred from `after_layer_4` to `after_layer_5`. The four increases ranged from +702.356262 to +721.740479.

## Not supported

The evidence does not establish:

- that the model successfully recalled any of the four capitals as its top-1 next token;
- that the final block stores or retrieves capital facts;
- that the change is specific to factual recall or capital tokens;
- that an attention head, MLP, neuron, or other component explains the change;
- that any component causally produced the target-logit increase;
- that the pattern generalizes beyond the four inspected prompts and model revision.

## Limitations

- The claim was formed after seeing the data and has no held-out confirmation.
- The sample contains only four prompts sharing one English template.
- All four intended targets failed to become the actual top-1 next token.
- Raw-logit changes can reflect broad vocabulary-wide effects; Experiment 001 does not provide a matched baseline or selectivity metric.
- The result covers one small model checkpoint and one projection method.
- The generated result artifact is reproducible but intentionally not committed to the repository.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-15 | — | `OBSERVATIONAL` | — | `ACTIVE` | Created from the measured Experiment 001 artifact; measurement validity passed and the four scoped exploratory observations agree. | E001–E006 |

# C001: Largest target-logit increase occurs across the final block for four tested capital prompts

## Metadata

- Current supported maturity: `OBSERVATIONAL`
- Disposition: `ACTIVE`
- Created: 2026-09-15
- Last reviewed: 2026-09-15
- Origin: exploratory
- Related experiments: Experiment 001, Experiment 002, Experiment 003

## Hypothesis

For each of the four capital-recall prompts measured in Experiment 001 with `EleutherAI/pythia-70m-deduped`, the intended target token has its largest positive adjacent-stage logit change from `after_layer_4` to `after_layer_5` under the experiment's logit-lens projection.

This hypothesis was formulated after inspecting the Experiment 001 measurements. The four cases are supporting exploratory observations, not an independent confirmatory test.

## Scope

- Model: `EleutherAI/pythia-70m-deduped`
- Resolved model revision: `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`
- Device and dtype: CPU, float32
- Task form: next-token projection for four English capital-recall prompts
- Prompts and targets: France/`" Paris"`, Germany/`" Berlin"`, Italy/`" Rome"`, and Finland/`" Helsinki"`
- Replication evidence: 24 held-out single-token country–capital pairs under the canonical template, with a second fixed paraphrase reported as a supplementary robustness condition
- Cross-scale evidence: the same 24 pairs and two templates measured with pinned Pythia-70M and Pythia-160M checkpoints through one Experiment 003 pipeline
- Metric: change in the target token's raw logit between adjacent normalized accumulated-residual projections
- Stages: embedding and the output of each of Pythia-70M's six transformer blocks
- Interventions: none

The original claim remains scoped to Pythia-70M. Experiment 003 supplies related evidence for an analogous pattern at one Pythia-160M checkpoint; it does not establish other tasks, languages, model families, checkpoints, or causal mechanisms.

## Operational definitions

- **Target logit** means the raw logit assigned to the specified single-token target after applying the model's final normalization and complete unembedding to an accumulated residual-stream state.
- **Adjacent-stage logit change** means the later stage's target logit minus the immediately preceding stage's target logit.
- **Largest positive change** means the numerically greatest adjacent-stage target-logit delta within one prompt's measured trajectory.
- **Final block transition** means `after_layer_4` to `after_layer_5`.

Absolute logit magnitude is not interpreted independently of the vocabulary distribution. The projected softmax values are not treated as actual intermediate model predictions.

## Predictions

Because this claim is exploratory, the original four cases are observations from which the hypothesis was formed. Experiment 002 subsequently committed a held-out prompt set and success criterion before inspecting its trajectories. Its canonical-template criterion required at least 18 of 24 held-out targets to have their largest positive adjacent-stage target-logit change across the final block transition.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | Experiment 001 | methodological | [`results.json`](../../outputs/experiment-001/results.json) | For all four prompts, projected final logits exactly matched actual logits; target rank and top-1 also matched. | Code commit `dffce3126cd78465f9708b593f12187ab6d19ea0`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E002 | Experiment 001, France | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Paris"`: largest increase was `after_layer_4` → `after_layer_5`, +714.766296; final rank 50; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E003 | Experiment 001, Germany | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Berlin"`: largest increase was `after_layer_4` → `after_layer_5`, +718.587402; final rank 60; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E004 | Experiment 001, Italy | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Rome"`: largest increase was `after_layer_4` → `after_layer_5`, +721.740479; final rank 120; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E005 | Experiment 001, Finland | supporting | [`results.json`](../../outputs/experiment-001/results.json) | `" Helsinki"`: largest increase was `after_layer_4` → `after_layer_5`, +702.356262; final rank 185; not top-1. | Code `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `6e90dc1962b2e31f48fc2da18544c351917817db5097a488a67dc49ad55ba0e2` |
| E006 | Experiment 001 documentation | methodological | [Experiment README](../../experiments/001-capital-recall/README.md) | Documents the one-command reproduction procedure, locked environment, generated artifacts, and stop-on-parity-failure behavior. | Code commit `dffce3126cd78465f9708b593f12187ab6d19ea0` |
| E007 | Experiment 001 reproducibility run | methodological | [`results.json`](../../outputs/experiment-001/results.json) | A fresh run reproduced every recorded prompt token, target ID, stage metric, extremum, final prediction, and parity result. Removing only `run_timestamp_utc` produced the same canonicalized result hash before and after the run. This repeats the discovery cases and is not held-out confirmation. | Run at repository commit `d1427c7c02fb12c2f331c98a2c4804f13f0924da`; analysis code last changed at `dffce3126cd78465f9708b593f12187ab6d19ea0`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; run timestamp `2026-09-15T11:28:55.024492+00:00`; results SHA-256 `47b631692ced2a2cfbab3df7f6b0ab64275dbba3245f00f9e863b18c1027e0a7`; canonicalized SHA-256 `d4afde384e8b4f2c529643ef9bf90870d78b185d879de3fa5f6bd832d317719b` |
| E008 | Experiment 002, canonical held-out replication | supporting | [`results.json`](../../outputs/experiment-002/results.json) | All 24 canonical held-out targets had their largest positive adjacent-stage raw-logit change across the final block, exceeding the preregistered threshold of 18/24. All final projections matched actual logits exactly. | Protocol commit `7fa9f558b039f752c553017746ee78672aea56c2`; analysis commit `94cf529dfc00a927a6fc16b44729a4558419d000`; model `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; results SHA-256 `5526c33f1dcef31790e50ba75aa35c2869a17f24d2d7886d8bcdabf6d85d1d31` |
| E009 | Experiment 002, balanced-control selectivity | null | [`results.json`](../../outputs/experiment-002/results.json) | The stronger preregistered conjunction was not supported: median final-block selectivity changes were −12.074371 canonical and −10.439687 paraphrase, and all 48 case-level changes were negative. This limits target-selective interpretations but does not contradict the raw-logit claim. | Same protocol, analysis, model, artifact, and digest as E008 |
| E010 | Experiment 003, cross-scale measurement validity | methodological | [`results.json`](../../outputs/experiment-003/results.json) | The fresh 70M cases matched Experiment 002 exactly. All 96 final projections across 70M and 160M matched actual logits, target ranks, and top-1 IDs exactly. | Protocol/analysis commit `5837c3058eb72a44fcca0321c2871c87af988101`; 160M revision `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`; results SHA-256 `30969fb8d8aaba85882d9275b8b0c0f79b24c54a711c6ebb4b04979174c03642` |
| E011 | Experiment 003, final-block raw-logit pattern | supporting | [`results.json`](../../outputs/experiment-003/results.json) | The largest positive adjacent-stage target-logit change occurred across the final block in 24/24 canonical and 24/24 paraphrase cases for both the fresh 70M and 160M runs. | Same protocol, analysis, revisions, artifact, and digest as E010 |
| E012 | Experiment 003, behavior and final-block selectivity | null | [`results.json`](../../outputs/experiment-003/results.json) | Neither model produced an intended capital as top-1 in any of 96 cases. Final-block correct-minus-control selectivity change was negative in all 48 cases for each model. | Same protocol, analysis, revisions, artifact, and digest as E010 |

Across these four prompts, the final-block target-logit increase ranged from +702.356262 to +721.740479. All four actual final predictions were the token `" the"`, not their intended capital tokens.

The output directory is generated and gitignored by Experiment 001. E001 preserves the original evaluated artifact's digest even though a later run regenerated the same path. E007 identifies the current artifact and records that its measurement payload reproduced exactly after excluding the new run timestamp.

## Evidence gates

### Level 0 — Measurement validity

- [x] The target metric, stages, token positions, and single-token targets are operationally defined. Evidence: E001.
- [x] The final residual projection agrees with actual final logits for all four prompts with maximum absolute logit difference 0.0. Evidence: E001.
- [x] Final target rank and top-1 token agree between projection and actual output for all four prompts. Evidence: E001.
- [x] Model revision, runtime dependencies, prompt tokens, token IDs, tolerances, and measurements are recorded. Evidence: E001.
- [x] Numerical failure behavior stops the experiment on target-rank or top-1 disagreement instead of weakening validation. Evidence: E006.
- [x] At the 2026-09-15 audit state, the version-controlled one-command procedure and locked dependencies regenerated the measurement payload. The resolved model revision was recorded but is not pinned by the loader. Evidence: E001, E006, E007.
- [x] Experiment 002 pinned the recorded model revision and obtained exact final-logit, target-rank, and top-1 parity in all 48 held-out prompt cases. Evidence: E008.

### Level 1 — Observation

- [x] The adjacent-stage trajectory is measured directly for every scoped prompt. Evidence: E002, E003, E004, E005.
- [x] All four scoped cases exhibit the stated final-block maximum. Evidence: E002, E003, E004, E005.
- [x] Per-case effect magnitudes and final ranks are reported rather than replacing variation with only an aggregate. Evidence: E002, E003, E004, E005.
- [x] The claim is labeled exploratory because it was formulated after inspecting these results. Evidence: E002, E003, E004, E005.
- [x] The supported conclusion is observational and makes no component-level or causal assertion. Evidence: E001.
- [x] A preregistered test on 24 held-out canonical prompts met its 18/24 replication criterion, with 24/24 observed replications. Evidence: E008.
- [x] A predeclared cross-scale comparison observed the analogous raw-logit extremum in 24/24 canonical and 24/24 paraphrase cases at both tested Pythia scales. Evidence: E010, E011.

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

Experiment 003 supplies evidence along one additional same-family model-scale axis, but the claim is not promoted: Levels 2–4 remain unsatisfied, only one additional scale was tested, and task/model-family generalization remains unknown. Evidence: E011, E012.

### Level 6 — Algorithmic explanation

- [ ] A compact mechanism predicts unseen behavior and intervention outcomes within a declared scope.
- [ ] Relevant retained behavior and residual unexplained behavior have been quantified.

Levels 2–6 are not required destinations for this narrow descriptive claim. They remain visible so that any attempted promotion is evaluated explicitly.

## Contradicting evidence and failed tests

No experiment currently contradicts the narrow raw-logit claim. Experiment 002 did, however, contradict a stronger nearby interpretation: selectivity for the correct target did not increase across the final block. That null result is retained as E009 rather than being treated as support.

The intended capital token was not the actual top-1 prediction for any of the four Experiment 001 prompts, 48 Experiment 002 prompt cases, or 96 Experiment 003 model–prompt cases. This does not contradict the logit-change claim, but it sharply limits any interpretation of the pattern as successful factual recall.

## Competing explanations

The shared final-block increase could reflect a broad transformation affecting many vocabulary logits rather than capital-specific processing. Experiment 002 strengthened this explanation: the correct targets' raw logits rose most across the final block in every held-out case, while their correct-minus-control margins fell in every case because the balanced control logits rose still more on average.

Experiment 002 balanced target-token reuse within the tested candidate set and added a second template. Experiment 003 found the same combination at 160M: the raw target logit rose most across the final block while correct-minus-control selectivity fell in every case. Training-corpus frequency, non-capital vocabulary behavior, normalization-specific effects, and other model-wide transformations remain possible explanations.

## Relationship to prior work

This claim has not yet been compared with a systematic review of prior logit-lens, factual-recall, or residual-stream studies. Logit-lens projection is an existing observational method, so neither use of the method nor a late-layer change is presumed novel. Any later novelty assessment must identify the nearest comparable results and test the same operational claim against them.

## Potential novelty if confirmed

No novelty is established at the current scope. Experiment 002 did not support the proposed target-selective final-block interpretation: every measured final-block correct-minus-control change was negative. Any future novelty would therefore need to come from a different, preregistered explanation that survives appropriate controls and prior-work comparison, not from the raw-logit extremum alone.

## Supported conclusion

Within the four original prompts, one Pythia-70M checkpoint, and the specified logit-lens measurement, each intended capital token's largest positive adjacent-stage logit change occurred from `after_layer_4` to `after_layer_5`. A preregistered test found the same raw-logit extremum in 24 of 24 held-out canonical prompts. A later predeclared comparison observed the analogous final-block extremum in all 24 canonical and all 24 paraphrase cases for both Pythia-70M and Pythia-160M. This is observational cross-scale evidence, not evidence that the final block selectively amplifies, stores, or retrieves a capital.

## Not supported

The evidence does not establish:

- that the model successfully recalled any of the four capitals as its top-1 next token;
- that the final block stores or retrieves capital facts;
- that the change is specific to factual recall or capital tokens;
- that an attention head, MLP, neuron, or other component explains the change;
- that any component causally produced the target-logit increase;
- that the pattern generalizes beyond the 28 measured country–capital pairs, two exact templates, two tested Pythia checkpoints, and projection metric;
- that the pattern generalizes to another model family, task, language, or tokenization regime;
- that the final block increases the correct target's support relative to balanced capital-token controls; Experiments 002 and 003 measured the opposite direction in every tested case;
- that either tested model successfully produced an intended capital as top-1 in Experiment 003.

## Limitations

- The claim was formed after seeing the Experiment 001 data; Experiment 002 provides held-out confirmation only for the operational raw-logit extremum.
- The discovery sample contains only four prompts sharing one English template. Experiment 002 adds 24 tokenization-constrained held-out pairs and one fixed paraphrase, not a representative population sample.
- None of the four Experiment 001 targets, 48 Experiment 002 prompt cases, or 96 Experiment 003 model–prompt cases produced the intended capital as the actual top-1 next token.
- Raw-logit changes can reflect broad vocabulary-wide effects; Experiment 001 does not provide a matched baseline or selectivity metric.
- Target-token corpus frequency was not measured. Token IDs are not frequency measurements, so frequency and token-identity effects remain uncontrolled.
- The runner records the resolved model revision but loads the default upstream `main` revision rather than pinning the recorded commit. The 2026-09-15 rerun reproduced exactly, but the one-command procedure does not guarantee the same checkpoint if upstream `main` changes.
- Raw final parity was exactly 0.0 in the recorded runs. However, the validation policy would only warn about a raw-logit mismatch that preserves target rank and top-1; future claims about raw-logit deltas require exact/all-close parity or shift-invariant measurements.
- The formal claim concerns one small model checkpoint; Experiment 003 adds related same-family evidence at one larger checkpoint using the same projection method.
- The generated result artifact is reproducible but intentionally not committed to the repository.
- Experiment 002's countries are a tokenization-constrained convenience sample rather than a random population sample.
- Balanced capital-token controls test relative discrimination within the selected candidate set; they do not establish selectivity against the full vocabulary or successful unconstrained completion.
- Experiment 003 changes model size, layer count, width, and learned weights together; it cannot attribute differences specifically to layer count or scale.
- Equal normalized depths are descriptive plotting coordinates and do not establish functionally corresponding layers.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-15 | — | `OBSERVATIONAL` | — | `ACTIVE` | Created from the measured Experiment 001 artifact; measurement validity passed and the four scoped exploratory observations agree. | E001–E006 |
| 2026-09-15 | `OBSERVATIONAL` | `OBSERVATIONAL` | `ACTIVE` | `ACTIVE` | A fresh execution reproduced the complete measurement payload but reused the four discovery prompts, so it does not provide held-out confirmation or justify promotion. | E007 |
| 2026-09-15 | `OBSERVATIONAL` | `OBSERVATIONAL` | `ACTIVE` | `ACTIVE` | Experiment 002 met the preregistered held-out raw-logit replication criterion, while its balanced-control result rejected a stronger target-selective final-block interpretation. The narrow claim remains observational and non-causal. | E008, E009 |
| 2026-09-15 | `OBSERVATIONAL` | `OBSERVATIONAL` | `ACTIVE` | `ACTIVE` | Experiment 003 observed the analogous final-block raw-logit extremum at Pythia-160M, but neither model completed an intended capital and final-block selectivity fell in every case. Maturity and disposition remain unchanged. | E010–E012 |

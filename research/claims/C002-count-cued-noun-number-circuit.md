# C002: A three-stage circuit (L00.MLP → L03.H04 → L04/L05 MLPs) localizes count-cued noun number selection in pinned Pythia-70M

## Metadata

- Current supported maturity: `LOCALIZED`
- Disposition: `ACTIVE`
- Created: 2026-09-18
- Last reviewed: 2026-09-18
- Origin: exploratory (mechanism found on development data in Experiment 005 Tier A; predictions preregistered in the protocol v2 lock before the single confirmation run)
- Related experiments: Experiment 005 (behavior candidate screen protocol v1 as its precursor)

## Hypothesis

In pinned `EleutherAI/pythia-70m-deduped`, the preference between a singular and a
plural noun form after a numeral or quantifier cue is carried by: the layer-0 MLP
at the cue position (E, a pure token function), attention head `L03.H04` at the
noun-prediction position, which transports the cue's number information when the
cue is not the final token (T), and the layer-4 and layer-5 MLPs at the
noun-prediction position (R), which convert it into the singular-versus-plural
logit contrast. Exact counterfactual replacement of these components moves the
contrast toward the counterfactual, holding them fixed while neutralizing every
other head and MLP output retains most of the contrast shift, and the E → T → R
chain mediates the effect (dedicated transport, hypothesis H1 of the design).

## Scope

- Model: `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32, TransformerBridge without compatibility mode, instrumented forwards with `use_attn_result=True`.
- Behavior: fixed-orientation contrast `c(x) = log P(singular) − log P(plural)` for matched cue pairs (`one`/`two`, `each`/`several`) in the three screening templates (twelve manifest prompts) and, prospectively, in six new frames and with twelve new cue words; single-token nouns of three regular rule classes.
- Data: development (20 nouns), holdout (19 single-token nouns), reserve (20 nouns), all from `screening/behavior-candidates/manifest-v1.json`; extension set `experiments/005-regular-plural-mechanism/extension-v1.json`.
- Interventions: exact replacement (counterfactual, freeze, cross-frame and cross-cue resample), pair-centered neutralization, isolation of the declared set, random component sets.
- Not in scope: other checkpoints or models, other templates, multi-token nouns, full-vocabulary top-1 behavior, the compact executable program (capped in protocol v2; see C002's "Not supported").

## Operational definitions

- Recovery `R(S)`: mean aligned bidirectional patched shift divided by mean full contrast shift over a stratum (design section "Fixed coordinates").
- Isolation faithfulness `F(S)`: retained contrast difference when every other head/MLP output at the noun position (and the cue position) is replaced by its pair-centered midpoint, as a fraction of the full shift.
- Blocked fraction, `m_T`, `m_R`, `m_R|T`, compensation ratio: as defined in the design (revision 5).
- Primary condition, flips, positive pairs: as in the screen.

## Predictions

Preregistered in `experiments/005-regular-plural-mechanism/preregistration-lock.json` (content sha256 `ec2a8879bab35fd6e4d45484321595aed565fb40f19bf608daf2574ea50ccc97`, commit `30e8b84`): floors and bands for families B1–B2, P1–P9, S1–S3, and X1–X4, with the decompilation axis capped from the outset.

## Evidence

| ID | Source | Type | Artifact | Result | Revision/digest |
|---|---|---|---|---|---|
| E001 | Experiment 005 Tier A (`discover`, development data, exploratory) | supporting | `experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md` | H1 unambiguous (`L03.H04` alone 0.816, blocked 0.865, `m_T` 0.90, `m_R` 0.99); E = `L00.MLP` (0.535 of the transport-input axis); k = 3 recovery 0.982, isolation 0.783; program under-predicts the quantifier mean by 1.44 nats (protocol v1 `NO_COMPACT_MECHANISM`) | code `83d1ae4`; results state `d5977a11…7795b` |
| E002 | Experiment 005 Tier B (`calibrate`, 114 single-token holdout cases; readout generalization to new nouns) | supporting | `experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md` (calibration section) | every primary circuit floor passed: P1 0.977, P3 0.782 (retention 91/114), P4 0.882, P5 0.816 / 0.865, P6 0.854 / 0.070, P7 0.976 / −0.006, P8 loss 0.823, P9 0.987 / 0.900 / 0.074; S1 (shared axis cosine 0.292) failed; program RMSE_B 0.609 | code `009607d`; results state `fe3833ed…47f0` |
| E003 | Experiment 005 Tier C (`confirm`, reserve nouns and extension prompts, single run under the lock) | null (precondition failed) | `experiments/005-regular-plural-mechanism/evidence/final-report-2026-09-18.md` (confirmation section) | outcome `BEHAVIOR_NOT_REPLICATED`: B1 98/120 primary correct (floor 103) and 75/120 flips (floor 96); by the preregistered rule no other family is confirmatory. Recorded non-confirmatory values: P1 0.985, P3 0.795 (0.962 / 0.760 / 0.672), P4 0.873, P5 0.817 / 0.869, P6 0.840 / 0.074, P7 0.984 / −0.007, P8 0.826, P9 0.987 / 0.900 / 0.074, every P band hit except P3's retention count (71/120) and the secondary P8 compensation band; new frames X1 passed (120/120 positive pairs, 12/12 number variables) with P1 0.980, P3 0.824, P4 0.948, P5 0.822 / 0.872, P8 0.796, P9 0.937 / 1.079 / 0.286; cue words: E-patch shifts track behavioral shifts for every word, lexicon Spearman 0.699 (floor 0.70) | code `30e8b84`; lock `ec2a8879…cc97`; results state `5a0c6836…f6bf` |
| E004 | Instrumentation contract and numerics | methodological | `experiments/005-regular-plural-mechanism/README.md` (numerical note); `outputs/experiment-005/results.json` `discovery.a0_contract_test`, `a1` | pinned Bridge contract test passed; A1 matched the screen per case to 0.0; instrumented-path readouts differ from plain forwards by up to 4.5e-3 nats; direct-effect decomposition identity within 1e-4 nats | code `83d1ae4` |

## Evidence gates

### Level 0 — Measurement validity

- [x] Quantities and success criteria operationally defined. Evidence: design revision 5; E004.
- [x] Inputs, tokenization, positions, and targets recorded. Evidence: manifest and extension digests in the lock; E003.
- [x] Reconstructed outputs agree with model ground truth. Evidence: E004 (A1 per-case match to the screen; decomposition identity).
- [x] Numerical validation, tolerances, and failure behavior documented. Evidence: E004.
- [x] Model revision, dependencies, runtime, and artifact locations recorded. Evidence: results state provenance; E001–E003.
- [x] Reproducible from version-controlled instructions. Evidence: `experiments/005-regular-plural-mechanism/README.md`.

### Level 1 — Observation

- [x] Phenomenon measured directly with explicit sample and scope. Evidence: E001, E002.
- [x] Variation across cases quantified. Evidence: E002 (strata and bands), E003.
- [x] Result labeled exploratory or confirmatory. Evidence: E001 exploratory; E002 calibration; E003 confirmatory attempt whose precondition failed.
- [x] Conclusion avoids causal interpretation at this level. Evidence: this record.

### Level 2 — Localization

- [x] Narrowed to defined components and positions. Evidence: E001 (A2 position map, k = 3 set).
- [x] Plausible alternative locations compared. Evidence: E001 (54-component ranking at both positions; random sets), E002 (P2).
- [x] Obvious artifacts and confounders tested. Evidence: E001 (structural no-ops asserted; cross-cue and cross-frame controls), E004.
- [x] Localization replicates within its declared scope. Evidence: E002 (19 holdout nouns never used for ranking).
- [x] Not presented as proof of causality. Evidence: this record.

### Level 3 — Causal evidence

- [x] Intervention and predicted outcome documented before inspection. Evidence: the lock (commit `30e8b84`).
- [ ] Necessity-oriented intervention supports the prediction on confirmatory data. Not established: E003's precondition failed, so its P8 value is recorded but not confirmatory.
- [ ] Sufficiency-oriented intervention supports the prediction on confirmatory data. Not established for the same reason (E003 P1/P4/P5 recorded, non-confirmatory).
- [ ] Matched, negative, and other controls on confirmatory data. Not established (E003 P2/P6/P7 recorded, non-confirmatory).
- [x] Effect sizes and uncertainty reported. Evidence: E002 bands; E003 values.
- [x] Plausible alternative causal explanations considered. Evidence: design hypotheses H2/H3 and their discriminating quantities (E001).

### Level 4 — Falsification tested

- [ ] Not reached.

### Level 5 — Generalized

- [ ] Not claimed. The new-frame and cue-word measurements (E003) are recorded but non-confirmatory under the outcome rule.

### Level 6 — Algorithmic explanation

- [ ] Not claimed. The compact program failed the protocol v1 floor (E001) and was capped in protocol v2; the cue-word results (E003) show the one-dimensional lexicon variable does not predict unseen cue words even though the E route does.

## Contradicting evidence and failed tests

- Protocol v1 program floor: the token-local program reproduced every development pair sign but under-predicted the quantifier template mean by 1.44 nats (E001).
- Reserve behavior (E003): 98/120 primary correct and 75/120 flips, below the preregistered 103 and 96; 44 of the 45 wrong conditions are plural conditions, spread over 13 of the 20 reserve nouns; the number variable still matched the cue in all 45 (S3).
- P3's absolute retention count (84/120) was unattainable on the reserve (clean flips 75/120) and on the new frames (65/120), the same split dependence noted for development in design revision 5.
- Cue words (E003): `all`, `some`, `both`, `few`, `many` shift the contrast by −3.8 to −4.8 nats although their lexicon values are within ±0.48; `every` (n_c −0.61) shifts it by about zero; `the` shifts it by −2.6. The E-patch reproduces each word's behavioral shift closely, so the cue-word information travels through E, but not along the single development-derived axis.
- S1: the readout-input number axes of the three templates have a minimum pairwise cosine of 0.292 (floor 0.70) on holdout and reserve.

## Competing explanations

- H2 (distributed transport): disfavored on development (top head 0.816 alone; freezing it blocks 0.865) and not contested by the recorded reserve values.
- H3 (late computation): disfavored on development (`m_R` 0.99 with E alone patched; R alone 0.606).
- A higher-dimensional or nonlinear cue encoding in `L00.MLP`: supported by the cue-word results and not distinguished by this experiment's one-dimensional variables.

## Relationship to prior work

See the finalist audit in `screening/behavior-candidates/audit-2026-09-17.json` and the design's prior-work section: number representations and agreement circuits are already mapped; this claim concerns the cue-to-noun-inflection pathway and is not a novelty claim about number representations.

## Potential novelty if confirmed

A component-level, position-resolved account of cue-to-noun-inflection transport in an unmodified Pythia-70M, with prospectively locked intervention predictions. Confirmation was not achieved under the preregistered rule.

## Supported conclusion

Within pinned Pythia-70M, the three screening templates, single-token regular nouns, and the tested replacement distributions, the count-cued singular/plural contrast is localized to `L00.MLP` at the cue position, `L03.H04` at the noun position, and the layer-4 and layer-5 MLPs at the noun position: on development data and on nineteen held-out nouns, exact replacement of these components recovers about 98% of the contrast shift, isolating them retains about 78% of it, and the head carries about 82% of the coordinated-template shift on its own. The same magnitudes were recorded on the reserve nouns and on six unseen frames, but that run does not count as confirmatory because the reserve nouns failed the preregistered behavioral precondition.

## Not supported

- Causal or algorithmic status of the mechanism (Levels 3–6).
- Generalization beyond the tested templates, frames, cue words, nouns, checkpoint, or model.
- The compact executable program: its one-dimensional lexicon does not predict unseen cue words, and it under-predicted the quantifier template on development data.
- That reserve-noun behavior matches holdout behavior: it does not (E003).

## Limitations

- Every split shares the same twelve prompts; noun-level replication tests readout generalization only. Prompt-side generalization rests on six new frames and twelve cue words whose results are recorded but non-confirmatory.
- The preregistered absolute counts (B1 flips, P3 retention) depend on the split's clean behavior; the reserve split is behaviorally weaker than the holdout split (75/120 versus 98/120 clean flips), which drove both failures.
- Instrumented forwards differ numerically from plain forwards by up to ~5e-3 nats; all Experiment 005 quantities are consistent within the instrumented path.
- Necessity and isolation conclusions are scoped to pair-centered neutralization.

## Status history

| Date | Previous maturity | New maturity | Previous disposition | New disposition | Reason | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-18 | — | `LOCALIZED` | — | `ACTIVE` | Created after Experiment 005's single confirmation run; Level 0–2 gates satisfied by E001, E002, E004; Level 3 not established because E003's behavioral precondition failed | E001–E004 |

# Experiment 021 Report

- Run ID: `9725893f39907cbd`
- Confirmation set: `e098e2b44a1702d2b35c20db2ce111358897ea867996024c3e51a79f379c309d` (Experiment 020's, read in place)
- Program blob: `caa73b40192f4c910dc63371bd19db75a3258339`; protocol/code commit at calibrate: `c49c16d8675b18b444e474181cd93d2324333dd2`

## Phases

- `calibrate`: `complete`
- `confirm`: `complete`
- `lock`: `complete`
- `report`: `not_started`
- `validate`: `not_started`

## Calibration (exposed only)

- Re-materialized 30024 pairs × 79 nouns from Experiment 020's ledger (33192 keys); environment drift 0.0; reproduction gate max difference 0.0 (tolerance 1e-09)
- Identities: additive 4.1e-06, inherited_017 0.0e+00, level1 4.6e-03, logit 3.3e-03, readout 5.2e-03, reference_component_sum 1.2e-06
- Validity screen and precondition: {'counts': {'cardinal': 14, 'coordinated-adjective': 14, 'quantifier': 14}, 'minimum': 6, 'ok': True}
- Calibration record `f939a84df3ff3dcf55f95877c951059c5157857d72d6be8ef036e10f7923044d`: {'clamped': {'Y1': ['pair_mean_r2', 'token_mean_r2'], 'Y2': ['coordinated_r2', 'frame_mean_r2', 'frame_r2_k75'], 'Y3': ['noun_r2_k90']}, 'joint_pass_rate_all_outcomes': 0.1662, 'rows': {'Y1': 1, 'Y2': 64, 'Y3': 84}}

## Confirmation — stage 1

- Table rows 432; digest `8af7c00056cdbc48c6f4386725b63d7945509d32fd5d1810fa28d25a0104a1cd`; Y2 selection [6, 6, 6] → row 6/6/6 (digest `fda5f93644b46f2f77208ee4f273922437129c2501f446749968eadd98a40c37`)

## Confirmation — stage 2 — `CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL | NOUN_READOUT_FIXED`

- Y1: **CONTRAST_PREDICTED_TOKENS**; row all; precondition {'n_scored_tokens': 24, 'n_valid_frames': 108, 'ok': True, 'scored_tokens': True, 'valid_frames': True}
  - token_mean_r2: fresh 0.0160 against floor 0.0000 (r2) → pass; exposed-like draw median -0.1451, fresh percentile among the draws 0.657
  - pair_mean_r2: fresh 0.3421 against floor 0.0000 (r2) → pass; exposed-like draw median 0.2388, fresh percentile among the draws 0.772
  - cue_mae_k80: fresh 0.7620 against floor 1.0258 (error) → pass; exposed-like draw median 0.8770, fresh percentile among the draws 0.992
  - pooled_mae: fresh 0.6478 against floor 0.7977 (error) → pass; exposed-like draw median 0.7268, fresh percentile among the draws 0.988
- Y2: **CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL**; row 6/6/6; precondition {'n_valid_coordinated': 6, 'n_valid_fresh_frames': 18, 'ok': True, 'valid_coordinated': True, 'valid_fresh_frames': True}
  - frame_mean_r2: fresh 0.2807 against floor 0.0000 (r2) → pass; exposed-like draw median 0.2642, fresh percentile among the draws 0.516
  - frame_r2_k75: fresh 0.4496 against floor 0.0000 (r2) → pass; exposed-like draw median -0.0605, fresh percentile among the draws 0.929
  - cue_final_r2: fresh 0.8445 against floor 0.7422 (r2) → pass; exposed-like draw median 0.8170, fresh percentile among the draws 0.822
  - coordinated_r2: fresh -0.1538 against floor 0.0000 (r2) → fail; exposed-like draw median -0.1066, fresh percentile among the draws 0.458
- Y3: **NOUN_READOUT_FIXED**; row 8/8/8; precondition {'n_scorable_nouns': 24, 'ok': True, 'scorable_nouns': True}
  - noun_median_r2: fresh 0.4467 against floor 0.2553 (r2) → pass; exposed-like draw median 0.4259, fresh percentile among the draws 0.597
  - noun_r2_k90: fresh 0.2859 against floor 0.0356 (r2) → pass; exposed-like draw median 0.2579, fresh percentile among the draws 0.601
  - noun_slope_dev_k90: fresh 0.1588 against floor 0.2240 (error) → pass; exposed-like draw median 0.1696, fresh percentile among the draws 0.736
  - noun_bias_k90: fresh 0.4939 against floor 0.7673 (error) → pass; exposed-like draw median 0.6017, fresh percentile among the draws 0.928
- Y1 ceiling (measured Δx₃): flattened R² 0.9683 against Level 0 0.6167; split {'downstream': 0.03168516218052875, 'inherited_upstream': 0.3515770713710509, 'total_unexplained': 0.3832622335515796}; the fresh ceiling's percentile among the row's exposed-like draws 0.756
- Y2 ceiling (measured Δx₃): flattened R² 0.9652 against Level 0 0.5797; split {'downstream': 0.034833178393910535, 'inherited_upstream': 0.3854885368102039, 'total_unexplained': 0.42032171520411443}; the fresh ceiling's percentile among the row's exposed-like draws 0.373
- Y1 comparators: dT_only -0.5382 (exposed-disfavoured baseline), no_l5_heads 0.4848 (nested simplification comparator), rank1_nouns 0.2000 (descriptive comparator), template_base_mlps 0.5115 (exposed-disfavoured operating-point comparator)
- Y2 comparators: dT_only -0.2027 (exposed-disfavoured baseline), no_l5_heads 0.4781 (nested simplification comparator), rank1_nouns 0.2158 (descriptive comparator), template_base_mlps 0.4838 (exposed-disfavoured operating-point comparator)
- Joint diagnostic (fresh cues × fresh frames × fresh nouns; no threshold): flattened R² 0.5858

Interpretation limit (frozen): the floors are relative to the program's own exposed performance; Y2 is conditional on each fresh frame's stage-1 reference state.

## Execution ledger

- Executed prompt keys: 33192
- Executed noun keys: 80

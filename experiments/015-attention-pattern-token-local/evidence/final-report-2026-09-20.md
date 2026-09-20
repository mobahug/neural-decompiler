# Experiment 015 Report

- Run ID: `eba024c09e58950a`
- Confirmation sha256: `66afcba4cddb872c91a99611a1d0d726e0b2d88e2e2c4e240bfec393fa1ce633`
- Experiment 014 lock sha256: `34b97174ebd68617f7a13a7d876f405761a8faaf42c6675ac8fa42de8c805d01`
- Protocol/code commit at explore: `407dd40cf290a8dfe3fabe198fb50b666ac91a69`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication: Experiment 014 ledger 4884 pairs, max deviation 0.00e+00; Experiment 013 patterns 3732 pairs, max deviation 2.78e-17
- Identities (Level 1, checks only): I1_level1_rows 1.8e-05, I1_patched_rows 1.7e-05, I1_reference_rows 9.8e-06, I2_chain 3.1e-06, I3_head_split 9.4e-07, neuron_sum 2.3e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Program: layer 1: d_head 64, rotary_dim 16, base 10000, layer 2: d_head 64, rotary_dim 16, base 10000
- Token means (159): ĉ_ΔA vs c_ΔA — Spearman 0.963, R² 0.787, MAE 0.0077, bias -0.0072 (n 159); c_ΔA spread sd 0.0203; per layer: Spearman 0.996, R² 0.969, MAE 0.0029, bias -0.0026 (n 159) | Spearman 0.948, R² 0.811, MAE 0.0052, bias -0.0045 (n 159)
- Level 0 entries (4884 pairs): layer 1 entry R² 0.919 (TV ratio 0.240), layer 2 entry R² 0.720 (TV ratio 0.455), both 0.854; self weights pooled Spearman 0.883, R² 0.821, MAE 0.0505, bias 0.0007 (n 78144); pairs ĉ_ΔA vs c_ΔA Spearman 0.891, R² 0.771, MAE 0.0131, bias -0.0067 (n 4884)
- Axis-only: token means Spearman 0.297, R² -2.082, MAE 0.0313, bias 0.0300 (n 159); entries layer 1 entry R² 0.166 (TV ratio 0.915), layer 2 entry R² 0.037 (TV ratio 0.976), both 0.123
- Diagonal-proportional (Level-0 self weight): token means Spearman 0.900, R² 0.668, MAE 0.0101, bias 0.0080 (n 159); entries layer 1 entry R² 0.684 (TV ratio 0.568), layer 2 entry R² 0.530 (TV ratio 0.708), both 0.634; margin of Level 0 over it layer 1 0.235, layer 2 0.190 → interaction beyond the self logit: True
- Ladder of decoded c_L: `c_012` pairs R² 0.735 / token means R² 0.803; `c_013_own` pairs R² 0.950 / token means R² 0.971; `c_L_level0` pairs R² 0.808 / token means R² 0.930; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level-0 c_M R² 0.827, c_H R² 0.916; Level 1's residual max 0.000003
- Rungs (own state, exact arriving change; descriptive):
  - rung `level1`: layer 1 entry R² 1.000 (TV 0.000), layer 2 1.000 (TV 0.000); read vs c_ΔA Spearman 1.000, R² 1.000, MAE 0.0000, bias 0.0000 (n 4884)
  - rung `diag_oracle`: layer 1 entry R² 0.762 (TV 0.428), layer 2 0.793 (TV 0.432); read vs c_ΔA Spearman 0.775, R² 0.565, MAE 0.0192, bias 0.0109 (n 4884)
  - rung `query_only`: layer 1 entry R² 0.246 (TV 0.788), layer 2 -0.196 (TV 0.861); read vs c_ΔA Spearman 0.523, R² 0.138, MAE 0.0265, bias 0.0041 (n 4884)
  - rung `key_only`: layer 1 entry R² 0.269 (TV 0.874), layer 2 -0.647 (TV 1.092); read vs c_ΔA Spearman 0.430, R² -0.631, MAE 0.0381, bias 0.0293 (n 4884)
  - rung `additive`: layer 1 entry R² 0.301 (TV 0.773), layer 2 -0.705 (TV 0.862); read vs c_ΔA Spearman 0.682, R² 0.103, MAE 0.0280, bias 0.0111 (n 4884)
  - rung `linearised`: layer 1 entry R² -0.572 (TV 1.333), layer 2 -1.489 (TV 1.651); read vs c_ΔA Spearman 0.495, R² -1.553, MAE 0.0457, bias 0.0068 (n 4884)
  - rung `frozen_arrival`: layer 1 entry R² 1.000 (TV 0.000), layer 2 0.860 (TV 0.337); read vs c_ΔA Spearman 0.971, R² 0.948, MAE 0.0064, bias -0.0010 (n 4884)
  - diagonal logit change rms: layer 1 ⟨Δq,k⟩ 4.59, ⟨q,Δk⟩ 1.07, ⟨Δq,Δk⟩ 1.14; layer 2 6.71 / 1.68 / 2.46; ‖Δx‖/‖x−μ‖ 0.96 / 0.81; measured TV per pair 1.66 / 1.07; max |ΔA_pc| 0.843

## Lock

- Candidate lock sha256 `6987704901dccf8c3638c7f770f64d77044320af5a1726e5c517130bc7423ebb`; predictions sha256 `35efa1e9b8d2575cf69fcb4f5a73baaf0e84ad3548edff188d25d6f933929fe3`

## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)

- Table rows 144; digest `2dcc809e5cfabbd2648414b9471835c4201d4f2a6f5f862db0c7c66d960c359e`; commit `5ed51c97d1af4993fa1fa501ae9bad33e32a6497`
  - cardinal-015-1: valid (plural head change 1.383, cue effect 79/72; p_c 3)
  - cardinal-015-2: valid (plural head change 1.630, cue effect 79/72; p_c 3)
  - coordinated-adjective-015-1: valid (plural head change 2.071, cue effect 79/72; p_c 5)
  - coordinated-adjective-015-2: valid (plural head change 2.264, cue effect 79/72; p_c 4)
  - quantifier-015-1: valid (plural head change 2.990, cue effect 79/72; p_c 3)
  - quantifier-015-2: valid (plural head change 1.995, cue effect 79/72; p_c 3)

## Confirmation — stage 2 — `PATTERN_CHANGE_PREDICTED_TOKENS | PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`

- Y1 (strict prospective: fresh cues × exposed frames): scored tokens 24 (precondition ok); token means ĉ_ΔA vs c_ΔA — Spearman 0.986, R² 0.895, MAE 0.0070, bias -0.0070 (n 24); entry R² layer 1 0.908, layer 2 0.700 → pass 
  - Level 0 entries: layer 1 entry R² 0.908 (TV ratio 0.257), layer 2 entry R² 0.700 (TV ratio 0.470), both 0.833; self weights pooled Spearman 0.870, R² 0.788, MAE 0.0565, bias 0.0011 (n 18432); pairs ĉ_ΔA vs c_ΔA Spearman 0.883, R² 0.749, MAE 0.0154, bias -0.0070 (n 1152); per layer token means Spearman 1.000, R² 0.978, MAE 0.0029, bias -0.0028 (n 24) | Spearman 0.983, R² 0.928, MAE 0.0043, bias -0.0043 (n 24); c_ΔA spread sd 0.0249
  - axis-only on this set: token means Spearman 0.008, R² -2.244, MAE 0.0412, bias 0.0367 (n 24); entries layer 1 entry R² 0.072 (TV ratio 0.972), layer 2 entry R² -0.000 (TV ratio 0.995), both 0.046
  - diagonal-proportional (Level-0 self weight) on this set: token means Spearman 0.948, R² 0.847, MAE 0.0071, bias 0.0064 (n 24); entries layer 1 entry R² 0.649 (TV ratio 0.599), layer 2 entry R² 0.527 (TV ratio 0.710), both 0.605; margin layer 1 0.259, layer 2 0.173
  - Ladder of decoded c_L: `c_012` pairs R² 0.725 / token means R² 0.843; `c_013_own` pairs R² 0.938 / token means R² 0.912; `c_L_level0` pairs R² 0.779 / token means R² 0.939; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level-0 c_M R² 0.820, c_H R² 0.893; Level 1's residual max 0.000002
  - rung `level1`: layer 1 entry R² 1.000 (TV 0.000), layer 2 1.000 (TV 0.000); read vs c_ΔA Spearman 1.000, R² 1.000, MAE 0.0000, bias 0.0000 (n 1152)
  - rung `diag_oracle`: layer 1 entry R² 0.735 (TV 0.452), layer 2 0.811 (TV 0.416); read vs c_ΔA Spearman 0.831, R² 0.676, MAE 0.0181, bias 0.0081 (n 1152)
  - rung `query_only`: layer 1 entry R² 0.284 (TV 0.787), layer 2 -0.001 (TV 0.820); read vs c_ΔA Spearman 0.572, R² 0.031, MAE 0.0329, bias 0.0238 (n 1152)
  - rung `key_only`: layer 1 entry R² 0.339 (TV 0.827), layer 2 -0.275 (TV 0.994); read vs c_ΔA Spearman 0.585, R² -0.185, MAE 0.0366, bias 0.0308 (n 1152)
  - rung `additive`: layer 1 entry R² 0.372 (TV 0.740), layer 2 -0.257 (TV 0.768); read vs c_ΔA Spearman 0.749, R² 0.303, MAE 0.0275, bias 0.0211 (n 1152)
  - rung `linearised`: layer 1 entry R² -0.506 (TV 1.295), layer 2 -1.318 (TV 1.611); read vs c_ΔA Spearman 0.704, R² -0.571, MAE 0.0409, bias -0.0095 (n 1152)
  - rung `frozen_arrival`: layer 1 entry R² 1.000 (TV 0.000), layer 2 0.857 (TV 0.331); read vs c_ΔA Spearman 0.979, R² 0.961, MAE 0.0063, bias 0.0015 (n 1152)
  - diagonal logit change rms: layer 1 ⟨Δq,k⟩ 4.90, ⟨q,Δk⟩ 0.99, ⟨Δq,Δk⟩ 1.14; layer 2 7.32 / 1.75 / 2.67; ‖Δx‖/‖x−μ‖ 0.99 / 0.84; measured TV per pair 1.71 / 1.16; max |ΔA_pc| 0.825
- Y2 (frame-conditional prospective, aggregate over the valid fresh frames: fresh cues × new frames): scored tokens 24 (precondition ok); token means ĉ_ΔA vs c_ΔA — Spearman 0.857, R² 0.490, MAE 0.0107, bias -0.0082 (n 24); entry R² layer 1 0.888, layer 2 0.423 → FAIL ['r2', 'entry_r2_layer2']
  - Level 0 entries: layer 1 entry R² 0.888 (TV ratio 0.287), layer 2 entry R² 0.423 (TV ratio 0.598), both 0.756; self weights pooled Spearman 0.823, R² 0.695, MAE 0.0621, bias 0.0040 (n 2304); pairs ĉ_ΔA vs c_ΔA Spearman 0.879, R² 0.735, MAE 0.0155, bias -0.0082 (n 144); per layer token means Spearman 0.938, R² 0.905, MAE 0.0035, bias -0.0019 (n 24) | Spearman 0.911, R² 0.501, MAE 0.0086, bias -0.0063 (n 24); c_ΔA spread sd 0.0193
  - axis-only on this set: token means Spearman 0.120, R² -0.370, MAE 0.0197, bias 0.0119 (n 24); entries layer 1 entry R² -0.004 (TV ratio 1.019), layer 2 entry R² -0.015 (TV ratio 0.996), both -0.007
  - diagonal-proportional (Level-0 self weight) on this set: token means Spearman 0.583, R² -0.287, MAE 0.0181, bias -0.0072 (n 24); entries layer 1 entry R² 0.667 (TV ratio 0.584), layer 2 entry R² 0.273 (TV ratio 0.793), both 0.555; margin layer 1 0.221, layer 2 0.151
  - Ladder of decoded c_L: `c_012` pairs R² 0.727 / token means R² 0.766; `c_013_own` pairs R² 0.940 / token means R² 0.900; `c_L_level0` pairs R² 0.686 / token means R² 0.719; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level-0 c_M R² 0.733, c_H R² 0.894; Level 1's residual max 0.000002
  - rung `level1`: layer 1 entry R² 1.000 (TV 0.000), layer 2 1.000 (TV 0.000); read vs c_ΔA Spearman 1.000, R² 1.000, MAE 0.0000, bias -0.0000 (n 144)
  - rung `diag_oracle`: layer 1 entry R² 0.768 (TV 0.416), layer 2 0.807 (TV 0.410); read vs c_ΔA Spearman 0.837, R² 0.692, MAE 0.0196, bias -0.0021 (n 144)
  - rung `query_only`: layer 1 entry R² 0.276 (TV 0.827), layer 2 -0.401 (TV 0.937); read vs c_ΔA Spearman 0.673, R² 0.267, MAE 0.0316, bias 0.0155 (n 144)
  - rung `key_only`: layer 1 entry R² 0.311 (TV 0.837), layer 2 -0.652 (TV 1.120); read vs c_ΔA Spearman 0.495, R² 0.204, MAE 0.0316, bias 0.0124 (n 144)
  - rung `additive`: layer 1 entry R² 0.370 (TV 0.762), layer 2 -0.950 (TV 0.936); read vs c_ΔA Spearman 0.812, R² 0.524, MAE 0.0243, bias 0.0157 (n 144)
  - rung `linearised`: layer 1 entry R² -0.499 (TV 1.324), layer 2 -1.478 (TV 1.630); read vs c_ΔA Spearman 0.785, R² -0.182, MAE 0.0366, bias -0.0056 (n 144)
  - rung `frozen_arrival`: layer 1 entry R² 1.000 (TV 0.000), layer 2 0.812 (TV 0.352); read vs c_ΔA Spearman 0.977, R² 0.948, MAE 0.0069, bias 0.0049 (n 144)
  - diagonal logit change rms: layer 1 ⟨Δq,k⟩ 4.89, ⟨q,Δk⟩ 1.03, ⟨Δq,Δk⟩ 1.16; layer 2 7.02 / 1.50 / 2.34; ‖Δx‖/‖x−μ‖ 0.99 / 0.80; measured TV per pair 1.70 / 1.00; max |ΔA_pc| 0.680
  - frame cardinal-015-1: 24 pairs; Level 0 entry R² layer 1 0.717 (TV 0.448), layer 2 0.474 (TV 0.599); ĉ_ΔA vs c_ΔA Spearman 0.684, R² 0.622, MAE 0.0120, bias 0.0012 (n 24); axis-only entry R² -0.073 / -0.006
  - frame cardinal-015-2: 24 pairs; Level 0 entry R² layer 1 0.941 (TV 0.227), layer 2 0.618 (TV 0.468); ĉ_ΔA vs c_ΔA Spearman 0.785, R² 0.773, MAE 0.0111, bias -0.0015 (n 24); axis-only entry R² 0.102 / -0.029
  - frame coordinated-adjective-015-1: 24 pairs; Level 0 entry R² layer 1 0.935 (TV 0.232), layer 2 0.666 (TV 0.594); ĉ_ΔA vs c_ΔA Spearman 0.873, R² 0.700, MAE 0.0120, bias -0.0053 (n 24); axis-only entry R² 0.003 / 0.014
  - frame coordinated-adjective-015-2: 24 pairs; Level 0 entry R² layer 1 0.942 (TV 0.218), layer 2 0.743 (TV 0.389); ĉ_ΔA vs c_ΔA Spearman 0.976, R² 0.958, MAE 0.0056, bias -0.0006 (n 24); axis-only entry R² -0.233 / 0.054
  - frame quantifier-015-1: 24 pairs; Level 0 entry R² layer 1 0.963 (TV 0.185), layer 2 0.664 (TV 0.508); ĉ_ΔA vs c_ΔA Spearman 0.877, R² 0.638, MAE 0.0174, bias -0.0150 (n 24); axis-only entry R² 0.091 / -0.041
  - frame quantifier-015-2: 24 pairs; Level 0 entry R² layer 1 0.789 (TV 0.425), layer 2 -0.266 (TV 1.045); ĉ_ΔA vs c_ΔA Spearman 0.678, R² -0.141, MAE 0.0349, bias -0.0278 (n 24); axis-only entry R² -0.006 / -0.049
- Y3 (axis-only alternative over both sets, 48 token means / 1296 pairs): token means Spearman -0.324, R² -1.163, MAE 0.0305, bias 0.0243 (n 48) (rejected iff R² < 0.3); pooled entry R² 0.040 (rejected iff < 0.3) → REJECTED; Level 0 on the same data: token means Spearman 0.939, R² 0.781, MAE 0.0089, bias -0.0076 (n 48), entries layer 1 entry R² 0.906 (TV ratio 0.261), layer 2 entry R² 0.676 (TV ratio 0.483), both 0.824
- Comparator rule over both sets (1296 pairs): Level 0 layer 1 entry R² 0.906 (TV ratio 0.261), layer 2 entry R² 0.676 (TV ratio 0.483), both 0.824; Level-0 diagonal-proportional layer 1 entry R² 0.651 (TV ratio 0.598), layer 2 entry R² 0.505 (TV ratio 0.718), both 0.599; margins layer 1 0.255, layer 2 0.171 (required ≥ 0.1 at both layers) → Level 0 decodes the query/key interaction beyond the self-logit effect

| token | class | Y1 frames | Y1 ĉ_ΔA | Y1 c_ΔA | Y2 frames | Y2 ĉ_ΔA | Y2 c_ΔA | axis ĉ (Y1) | diag-prop ĉ (Y1) | arrival read (Y1) |
|---|---|---|---|---|---|---|---|---|---|---|
| particular | determiner-like | 48 | 0.0399 | 0.0415 | 6 | 0.0429 | 0.0432 | 0.0203 | 0.0457 | 0.401 |
| ourselves | possessive-or-pronoun | 48 | 0.0313 | 0.0324 | 6 | 0.0163 | 0.0234 | 0.0182 | 0.0383 | 0.312 |
| itself | possessive-or-pronoun | 48 | 0.0304 | 0.0333 | 6 | 0.0112 | 0.0168 | 0.0151 | 0.0330 | 0.158 |
| moderate | quantity | 48 | 0.0030 | 0.0099 | 6 | 0.0452 | 0.0338 | 0.0272 | 0.0191 | 0.632 |
| scant | quantity | 48 | -0.0027 | 0.0066 | 6 | 0.0327 | 0.0555 | 0.0241 | 0.0152 | 0.654 |
| previous | determiner-like | 48 | -0.0042 | -0.0018 | 6 | -0.0010 | -0.0121 | 0.0274 | 0.0165 | 0.646 |
| modern | adjective | 48 | -0.0089 | 0.0007 | 6 | 0.0038 | 0.0109 | 0.0253 | -0.0018 | 0.458 |
| enormous | quantity | 48 | -0.0118 | 0.0009 | 6 | 0.0110 | 0.0103 | 0.0311 | 0.0016 | 0.610 |
| frozen | adjective | 48 | -0.0160 | -0.0116 | 6 | 0.0110 | 0.0150 | 0.0214 | 0.0041 | 0.611 |
| substantial | quantity | 48 | -0.0164 | -0.0072 | 6 | -0.0018 | -0.0049 | 0.0289 | 0.0002 | 0.569 |
| ancient | adjective | 48 | -0.0204 | -0.0128 | 6 | 0.0154 | 0.0111 | 0.0291 | -0.0144 | 0.577 |
| final | determiner-like | 48 | -0.0229 | -0.0169 | 6 | -0.0145 | -0.0046 | 0.0202 | -0.0003 | 0.317 |
| himself | possessive-or-pronoun | 48 | -0.0257 | -0.0176 | 6 | -0.0271 | -0.0210 | 0.0164 | -0.0188 | 0.131 |
| fourth | determiner-like | 48 | -0.0267 | -0.0224 | 6 | -0.0069 | 0.0029 | 0.0245 | -0.0198 | 0.281 |
| adequate | quantity | 48 | -0.0273 | -0.0179 | 6 | -0.0033 | -0.0004 | 0.0302 | 0.0087 | 0.538 |
| herself | possessive-or-pronoun | 48 | -0.0289 | -0.0222 | 6 | -0.0317 | -0.0253 | 0.0173 | -0.0215 | 0.122 |
| eighth | ordinal-or-numeral | 48 | -0.0360 | -0.0293 | 6 | -0.0112 | 0.0061 | 0.0234 | -0.0263 | 0.325 |
| sixth | ordinal-or-numeral | 48 | -0.0376 | -0.0330 | 6 | -0.0186 | -0.0062 | 0.0243 | -0.0226 | 0.248 |
| seventh | ordinal-or-numeral | 48 | -0.0395 | -0.0369 | 6 | -0.0149 | -0.0099 | 0.0224 | -0.0238 | 0.253 |
| ninth | ordinal-or-numeral | 48 | -0.0456 | -0.0392 | 6 | -0.0236 | -0.0062 | 0.0207 | -0.0298 | 0.235 |
| fifth | determiner-like | 48 | -0.0498 | -0.0450 | 6 | -0.0252 | -0.0170 | 0.0217 | -0.0390 | 0.207 |
| loud | adjective | 48 | -0.0531 | -0.0396 | 6 | -0.0080 | 0.0117 | 0.0246 | -0.0361 | 0.645 |
| plastic | adjective | 48 | -0.0550 | -0.0374 | 6 | -0.0265 | 0.0051 | 0.0263 | -0.0386 | 0.540 |
| tenth | ordinal-or-numeral | 48 | -0.0661 | -0.0552 | 6 | -0.0449 | -0.0121 | 0.0206 | -0.0564 | 0.211 |

Measured / predicted self-attention weight change ΔA_h(p_c, p_c), token means over the Y1 frames:

| token | L01.H00 | L01.H01 | L01.H02 | L01.H03 | L01.H04 | L01.H05 | L01.H06 | L01.H07 | L02.H00 | L02.H01 | L02.H02 | L02.H03 | L02.H04 | L02.H05 | L02.H06 | L02.H07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| particular | -0.05 / -0.04 | -0.00 / 0.01 | -0.07 / -0.06 | 0.06 / 0.07 | 0.09 / 0.09 | 0.04 / 0.04 | 0.20 / 0.19 | -0.23 / -0.24 | 0.02 / 0.03 | 0.01 / 0.01 | -0.08 / -0.08 | -0.00 / -0.03 | -0.02 / -0.02 | 0.09 / 0.08 | -0.11 / -0.13 | -0.00 / -0.00 |
| ourselves | -0.05 / -0.02 | -0.17 / -0.17 | -0.16 / -0.18 | 0.10 / 0.11 | -0.02 / -0.03 | 0.02 / 0.03 | 0.04 / 0.04 | -0.21 / -0.21 | 0.11 / 0.11 | -0.01 / -0.02 | -0.01 / -0.03 | 0.28 / 0.29 | -0.00 / -0.00 | 0.14 / 0.14 | -0.10 / -0.18 | 0.04 / 0.04 |
| itself | -0.10 / -0.08 | -0.11 / -0.11 | -0.25 / -0.26 | 0.09 / 0.10 | 0.04 / 0.04 | 0.02 / 0.04 | 0.01 / 0.03 | -0.32 / -0.32 | 0.19 / 0.20 | -0.02 / -0.02 | -0.04 / -0.05 | 0.19 / 0.20 | -0.03 / -0.02 | 0.11 / 0.11 | -0.06 / -0.10 | 0.01 / 0.01 |
| moderate | -0.21 / -0.21 | 0.10 / 0.10 | -0.28 / -0.28 | 0.04 / 0.05 | -0.05 / -0.05 | -0.05 / -0.05 | -0.05 / -0.04 | 0.09 / 0.09 | 0.08 / 0.08 | 0.01 / 0.00 | -0.19 / -0.19 | -0.03 / -0.04 | 0.05 / 0.01 | 0.03 / 0.04 | -0.30 / -0.24 | -0.03 / -0.03 |
| scant | -0.05 / -0.06 | 0.25 / 0.24 | -0.31 / -0.31 | 0.11 / 0.12 | 0.21 / 0.22 | 0.12 / 0.12 | -0.03 / -0.03 | 0.12 / 0.10 | 0.08 / 0.06 | -0.01 / -0.02 | 0.01 / 0.05 | 0.03 / 0.03 | 0.04 / -0.01 | -0.08 / -0.12 | 0.23 / 0.26 | -0.04 / -0.05 |
| previous | -0.02 / 0.00 | 0.25 / 0.26 | 0.06 / 0.07 | 0.06 / 0.06 | -0.01 / -0.01 | -0.06 / -0.06 | -0.13 / -0.12 | -0.06 / -0.07 | 0.11 / 0.12 | 0.02 / 0.01 | -0.03 / -0.03 | 0.04 / 0.02 | 0.07 / 0.06 | 0.11 / 0.09 | -0.09 / -0.07 | 0.00 / -0.01 |
| modern | -0.20 / -0.19 | 0.17 / 0.18 | -0.22 / -0.21 | -0.01 / -0.01 | -0.02 / -0.03 | -0.05 / -0.05 | -0.00 / 0.00 | -0.04 / -0.04 | 0.04 / 0.04 | -0.00 / -0.01 | -0.13 / -0.13 | -0.04 / -0.05 | -0.01 / -0.02 | -0.09 / -0.10 | -0.25 / -0.19 | -0.03 / -0.03 |
| enormous | 0.04 / 0.05 | 0.29 / 0.30 | -0.09 / -0.09 | 0.08 / 0.08 | 0.03 / 0.02 | -0.09 / -0.09 | 0.26 / 0.25 | -0.13 / -0.14 | 0.03 / 0.03 | 0.03 / 0.02 | -0.09 / -0.09 | -0.01 / -0.02 | -0.05 / -0.05 | 0.01 / 0.00 | -0.31 / -0.24 | -0.01 / -0.01 |
| frozen | -0.20 / -0.20 | 0.39 / 0.40 | -0.21 / -0.20 | -0.01 / -0.01 | 0.08 / 0.06 | -0.21 / -0.21 | -0.03 / -0.04 | 0.03 / 0.02 | 0.08 / 0.07 | 0.06 / 0.05 | -0.13 / -0.12 | 0.14 / 0.14 | 0.35 / 0.30 | -0.12 / -0.14 | -0.38 / -0.33 | 0.01 / -0.00 |
| substantial | -0.18 / -0.18 | 0.22 / 0.23 | -0.28 / -0.28 | 0.03 / 0.04 | 0.01 / 0.02 | -0.08 / -0.08 | 0.02 / 0.03 | -0.13 / -0.14 | 0.07 / 0.07 | 0.01 / -0.00 | -0.09 / -0.09 | -0.08 / -0.11 | -0.04 / -0.05 | 0.03 / 0.04 | -0.19 / -0.12 | -0.01 / -0.02 |
| ancient | -0.19 / -0.19 | 0.26 / 0.28 | -0.23 / -0.23 | 0.02 / 0.01 | 0.10 / 0.09 | -0.10 / -0.10 | 0.20 / 0.22 | -0.04 / -0.04 | 0.12 / 0.13 | 0.02 / 0.02 | -0.13 / -0.13 | 0.04 / 0.03 | 0.08 / 0.08 | -0.17 / -0.18 | -0.32 / -0.26 | 0.00 / -0.00 |
| final | 0.11 / 0.15 | 0.18 / 0.19 | 0.00 / 0.02 | -0.02 / -0.03 | 0.05 / 0.05 | -0.06 / -0.05 | -0.05 / -0.06 | -0.13 / -0.14 | 0.04 / 0.06 | -0.01 / -0.02 | -0.10 / -0.09 | -0.09 / -0.10 | -0.01 / -0.03 | -0.08 / -0.12 | -0.05 / -0.03 | -0.04 / -0.04 |
| himself | -0.04 / -0.02 | -0.11 / -0.11 | -0.21 / -0.23 | 0.06 / 0.07 | -0.04 / -0.05 | -0.03 / -0.02 | 0.03 / 0.04 | -0.10 / -0.09 | 0.13 / 0.12 | -0.02 / -0.03 | -0.01 / -0.02 | 0.35 / 0.37 | -0.01 / -0.01 | 0.07 / 0.05 | -0.20 / -0.23 | 0.02 / 0.02 |
| fourth | 0.04 / 0.08 | 0.32 / 0.33 | -0.04 / -0.02 | -0.01 / -0.01 | 0.13 / 0.14 | 0.00 / 0.02 | -0.12 / -0.13 | 0.14 / 0.14 | 0.13 / 0.14 | 0.00 / -0.01 | -0.05 / -0.04 | 0.06 / 0.04 | 0.05 / 0.02 | -0.16 / -0.20 | 0.04 / 0.11 | -0.04 / -0.05 |
| adequate | -0.04 / -0.03 | 0.15 / 0.15 | -0.17 / -0.17 | 0.01 / 0.01 | 0.16 / 0.15 | -0.11 / -0.12 | 0.09 / 0.09 | -0.07 / -0.08 | 0.11 / 0.11 | 0.03 / 0.03 | -0.10 / -0.10 | -0.08 / -0.10 | -0.00 / -0.01 | 0.07 / 0.06 | -0.23 / -0.19 | -0.01 / -0.02 |
| herself | -0.04 / -0.02 | -0.07 / -0.07 | -0.24 / -0.25 | 0.08 / 0.09 | -0.03 / -0.04 | -0.09 / -0.09 | -0.05 / -0.06 | -0.13 / -0.12 | 0.14 / 0.12 | -0.00 / -0.01 | 0.02 / -0.00 | 0.37 / 0.38 | -0.00 / -0.00 | 0.10 / 0.08 | -0.25 / -0.28 | 0.02 / 0.01 |
| eighth | 0.04 / 0.07 | 0.39 / 0.41 | -0.08 / -0.07 | 0.01 / 0.01 | 0.14 / 0.14 | -0.03 / -0.02 | -0.14 / -0.14 | 0.11 / 0.11 | 0.11 / 0.12 | 0.01 / -0.00 | -0.08 / -0.06 | 0.06 / 0.04 | 0.04 / 0.01 | -0.05 / -0.10 | -0.01 / 0.07 | -0.04 / -0.05 |
| sixth | 0.10 / 0.13 | 0.30 / 0.30 | 0.04 / 0.06 | -0.02 / -0.02 | 0.16 / 0.16 | -0.01 / 0.00 | -0.09 / -0.09 | -0.05 / -0.05 | 0.08 / 0.08 | 0.01 / -0.01 | -0.09 / -0.08 | 0.05 / 0.03 | 0.09 / 0.07 | -0.17 / -0.22 | 0.07 / 0.15 | -0.05 / -0.05 |
| seventh | 0.16 / 0.19 | 0.38 / 0.40 | 0.05 / 0.07 | 0.02 / 0.02 | 0.13 / 0.14 | -0.06 / -0.06 | -0.08 / -0.09 | 0.03 / 0.03 | 0.07 / 0.08 | 0.01 / 0.00 | -0.06 / -0.04 | 0.07 / 0.05 | 0.03 / 0.00 | -0.06 / -0.11 | 0.30 / 0.35 | -0.04 / -0.05 |
| ninth | 0.09 / 0.12 | 0.33 / 0.34 | 0.04 / 0.05 | 0.01 / 0.01 | 0.16 / 0.16 | -0.02 / -0.01 | -0.12 / -0.13 | 0.07 / 0.08 | 0.12 / 0.13 | 0.01 / -0.01 | -0.11 / -0.09 | 0.03 / 0.01 | 0.05 / 0.02 | -0.06 / -0.11 | 0.00 / 0.09 | -0.04 / -0.05 |
| fifth | 0.08 / 0.12 | 0.35 / 0.37 | -0.01 / 0.02 | 0.02 / 0.01 | 0.14 / 0.15 | -0.04 / -0.03 | -0.03 / -0.04 | 0.14 / 0.13 | 0.15 / 0.16 | 0.01 / 0.00 | 0.04 / 0.08 | 0.08 / 0.06 | 0.07 / 0.04 | -0.12 / -0.16 | 0.17 / 0.25 | -0.04 / -0.05 |
| loud | -0.01 / -0.00 | 0.17 / 0.16 | -0.17 / -0.17 | -0.02 / -0.02 | 0.01 / -0.00 | -0.27 / -0.28 | 0.32 / 0.35 | 0.05 / 0.04 | 0.06 / 0.06 | 0.03 / 0.02 | -0.10 / -0.08 | 0.08 / 0.08 | 0.24 / 0.21 | 0.03 / 0.02 | -0.24 / -0.20 | -0.01 / -0.01 |
| plastic | 0.20 / 0.21 | 0.28 / 0.29 | -0.24 / -0.25 | 0.03 / 0.03 | 0.12 / 0.11 | -0.23 / -0.24 | 0.25 / 0.26 | 0.17 / 0.16 | 0.08 / 0.07 | 0.03 / 0.02 | -0.04 / 0.00 | 0.18 / 0.18 | 0.12 / 0.09 | -0.14 / -0.15 | -0.30 / -0.22 | -0.03 / -0.04 |
| tenth | 0.14 / 0.17 | 0.46 / 0.47 | -0.18 / -0.18 | 0.04 / 0.04 | 0.05 / 0.06 | 0.09 / 0.10 | -0.13 / -0.13 | 0.12 / 0.12 | 0.11 / 0.11 | -0.01 / -0.01 | -0.10 / -0.08 | 0.14 / 0.14 | 0.13 / 0.08 | -0.07 / -0.12 | -0.20 / -0.09 | -0.04 / -0.05 |

## Execution ledger

- Executed prompt keys: 1458
- Executed noun keys: 80

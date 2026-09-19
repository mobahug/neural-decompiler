# Experiment 014 Report

- Run ID: `00eef3b8bb95c8b8`
- Confirmation sha256: `a28edf831413489b2c8ffd4710bba4d742d4e542348120b0cab90c4f3ab3796d`
- Experiment 013 lock sha256: `ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`
- Protocol/code commit at explore: `bcb0bfe3e94ee24a2740ee22b4e9228789a7436d`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 013: 3732 pairs, max deviation 0.00e+00; identities neuron_sum 2.0e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Neuron 2/1987: cos(γ₂ ⊙ W_in, d̂_E) 0.241; r(W_out) 0.333; r(W_out)/denominator per template {cardinal: 0.204, coordinated-adjective: 0.204, quantifier: 0.222}; b_in -0.659
- Operating points: pre_ref -1.60 to 0.49; reference activation max 0.337 (below the firing threshold 0.5)
- Token means (135): Δâ vs Δa — Spearman 0.985, R² 0.989, MAE 0.049, bias 0.020 (n 135); Δa spread sd 0.825; axis-only Δâ vs Δa — Spearman 0.562, R² 0.166, MAE 0.686, bias 0.454 (n 135)
- Pairs (3732): Δâ vs Δa — Spearman 0.976, R² 0.963, MAE 0.094, bias 0.015 (n 3732); Δp̂re vs Δpre — Spearman 0.980, R² 0.968, MAE 0.175, bias 0.025 (n 3732); Jacobian form vs Δpre — Spearman 0.963, R² 0.913, MAE 0.309, bias 0.203 (n 3732) (vs the exact form R² 0.952)
- Classification (pairs): predictor balanced accuracy 0.989 (sensitivity 0.991, specificity 0.987; raw 0.988, majority baseline 0.689; measured firing 1160/3732); axis-only balanced accuracy 0.611 (sensitivity 0.938, specificity 0.284; raw 0.487, majority baseline 0.689; measured firing 1160/3732)
- Jacobian form (descriptive): Δâ_J vs Δa token means — Spearman 0.952, R² 0.913, MAE 0.161, bias 0.154 (n 135); pairs — Spearman 0.947, R² 0.868, MAE 0.191, bias 0.151 (n 3732); balanced accuracy 0.979 (sensitivity 0.993, specificity 0.965; raw 0.974, majority baseline 0.689; measured firing 1160/3732)
- Amplitude among firing pairs: error 0.219 on a mean Δa of 1.66; MAE all pairs 0.094
- Accounting (mean |Jacobian contribution|): E 1.08, MLP₁ 0.33, heads₁ 0.41 (negative in 0.94 of pairs); axis 1.35, off-axis 1.14; remainders: pattern change |.| 0.175, LayerNorm linearization |.| 0.229
- Neuron's share of the block-2 read change among firing pairs 1.63; mean term firing 0.351, not firing -0.016
- Measured firing set (≥ half of frames), by class: extension: few, five, four, many, ten, three; inherited: hundred, multiple, numerous, twelve; numeral: billion, eight, eighteen, eighty, fifty, forty, million, nine, nineteen, ninety, seven, seventy, six, sixty, thirty, thousand, trillion, twenty; original-cue: cardinal:pl, quantifier:pl; plural-numeral: eleven, fifteen, fourteen, seventeen, sixteen, thirteen; quantity: countless, dozen, fewer, myriad, various
- Predicted firing set: extension: few, five, four, many, ten, three; inherited: hundred, multiple, numerous, twelve; numeral: billion, eight, eighteen, eighty, fifty, forty, million, nine, nineteen, ninety, seven, seventy, six, sixty, thirty, thousand, trillion, twenty; original-cue: cardinal:pl, quantifier:pl; plural-numeral: eleven, fifteen, fourteen, seventeen, sixteen, thirteen; quantity: countless, dozen, fewer, myriad, various

## Lock

- Candidate lock sha256 `34b97174ebd68617f7a13a7d876f405761a8faaf42c6675ac8fa42de8c805d01`; predictions sha256 `a0b7448a4c42faee538b180794f95abf6fbf5e3049ab8358ba83875ce05b091e`

## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)

- Table rows 144; predicted to fire in 22 pairs; digest `102103ed6f746e5b24698c6c07441e7965b7fa04cf9a6a67206ed738522e217a`; commit `d008dae9ab8338dc5c6a30f13c69689ba4a74fe4`
  - cardinal-014-1: valid (plural head change 1.930, cue effect 79/72; pre_ref -0.36)
  - cardinal-014-2: valid (plural head change 1.291, cue effect 79/72; pre_ref -0.47)
  - coordinated-adjective-014-1: valid (plural head change 2.421, cue effect 79/72; pre_ref 0.03)
  - coordinated-adjective-014-2: valid (plural head change 2.908, cue effect 79/72; pre_ref -0.02)
  - quantifier-014-1: valid (plural head change 1.461, cue effect 79/72; pre_ref -1.55)
  - quantifier-014-2: valid (plural head change 1.295, cue effect 79/72; pre_ref -2.36)

## Confirmation — stage 2 — `NEURON_FEATURE_PREDICTED_TOKENS | NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED`

- Y1 (strict: new tokens × exposed frames): scored tokens 24 (precondition ok); token means Δâ vs Δa — Spearman 0.950, R² 0.970, MAE 0.040, bias 0.002 (n 24); balanced accuracy 0.923 (sensitivity 0.852, specificity 0.994; raw 0.964, majority baseline 0.792; measured firing 210/1008) → pass 
  - pairs: Δâ vs Δa Spearman 0.979, R² 0.940, MAE 0.060, bias 0.002 (n 1008); Δp̂re vs Δpre Spearman 0.971, R² 0.954, MAE 0.165, bias 0.021 (n 1008); Jacobian form vs Δpre Spearman 0.941, R² 0.861, MAE 0.298, bias 0.185 (n 1008); amplitude error among firing pairs 0.197 (mean Δa firing 0.99); MAE 0.060; Δa spread sd 0.451
  - accounting |.|: E 0.65, MLP₁ 0.30, heads₁ 0.47, axis 1.18, off-axis 1.29; remainders pattern change 0.165, LayerNorm 0.214; neuron's share of the block-2 read change among firing pairs 1.41
  - axis-only on this set: Spearman 0.414, R² -1.742, MAE 0.658, bias 0.653 (n 24); balanced accuracy 0.592 (sensitivity 0.876, specificity 0.308; raw 0.427, majority baseline 0.792; measured firing 210/1008)
  - Jacobian form on this set (descriptive): token means Spearman 0.648, R² 0.747, MAE 0.117, bias 0.114 (n 24); balanced accuracy 0.955 (sensitivity 0.938, specificity 0.972; raw 0.965, majority baseline 0.792; measured firing 210/1008)
  - measured firing set: numeral: billions, dozens, hundreds, millions, thousands; quantity: infinite | predicted: numeral: dozens, hundreds, millions, thousands; quantity: infinite
- Y2 (frame-conditional: new tokens × fresh frames): scored tokens 24 (precondition ok); token means Δâ vs Δa — Spearman 0.910, R² 0.957, MAE 0.043, bias -0.009 (n 24); balanced accuracy 0.927 (sensitivity 0.870, specificity 0.983; raw 0.965, majority baseline 0.840; measured firing 23/144) → pass 
  - pairs: Δâ vs Δa Spearman 0.957, R² 0.937, MAE 0.053, bias -0.009 (n 144); Δp̂re vs Δpre Spearman 0.972, R² 0.960, MAE 0.166, bias 0.022 (n 144); Jacobian form vs Δpre Spearman 0.940, R² 0.877, MAE 0.299, bias 0.030 (n 144); amplitude error among firing pairs 0.188 (mean Δa firing 0.98); MAE 0.053; Δa spread sd 0.358
  - accounting |.|: E 0.61, MLP₁ 0.38, heads₁ 0.48, axis 1.19, off-axis 1.31; remainders pattern change 0.166, LayerNorm 0.262; neuron's share of the block-2 read change among firing pairs 1.17
  - axis-only on this set: Spearman 0.478, R² -0.892, MAE 0.438, bias 0.406 (n 24); balanced accuracy 0.653 (sensitivity 0.826, specificity 0.479; raw 0.535, majority baseline 0.840; measured firing 23/144)
  - Jacobian form on this set (descriptive): token means Spearman 0.874, R² 0.754, MAE 0.079, bias 0.075 (n 24); balanced accuracy 0.966 (sensitivity 0.957, specificity 0.975; raw 0.972, majority baseline 0.840; measured firing 23/144)
  - measured firing set: numeral: billions, dozens, hundreds, thousands; quantity: infinite | predicted: numeral: dozens, hundreds, thousands; quantity: infinite
- Y3 (axis-only alternative over both sets, 48 token means / 1152 pairs): Spearman 0.183, R² -1.414, MAE 0.548, bias 0.530 (n 48) (rejected iff R² < 0.3); balanced accuracy 0.601 (sensitivity 0.871, specificity 0.331; raw 0.440, majority baseline 0.798; measured firing 233/1152) (rejected iff < 0.7) → REJECTED; the predictor on the same token means Spearman 0.977, R² 0.965, MAE 0.042, bias -0.004 (n 48)

| token | class | Y1 frames | Y1 Δâ | Y1 Δa | Y1 fires ĉ/meas | Y2 frames | Y2 Δâ | Y2 Δa | Y2 fires ĉ/meas | axis Δâ (Y1) | pre_ref (Y2) | term (Y1 meas) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dozens | numeral | 42 | 1.490 | 1.267 | 1.00/1.00 | 6 | 1.255 | 1.034 | 1.00/0.83 | 1.226 | -0.79 | 0.267 |
| hundreds | numeral | 42 | 1.185 | 0.988 | 1.00/0.93 | 6 | 0.949 | 0.817 | 0.83/0.83 | 1.063 | -0.79 | 0.208 |
| thousands | numeral | 42 | 1.087 | 1.155 | 0.98/1.00 | 6 | 0.842 | 0.970 | 0.83/0.83 | 1.129 | -0.79 | 0.243 |
| infinite | quantity | 42 | 0.573 | 0.626 | 0.57/0.69 | 6 | 0.547 | 0.670 | 0.67/0.50 | 0.833 | -0.79 | 0.132 |
| millions | numeral | 42 | 0.533 | 0.696 | 0.50/0.74 | 6 | 0.270 | 0.381 | 0.17/0.33 | 0.934 | -0.79 | 0.147 |
| billions | numeral | 42 | 0.444 | 0.590 | 0.33/0.64 | 6 | 0.310 | 0.450 | 0.17/0.50 | 1.015 | -0.79 | 0.124 |
| rough | adjective | 42 | -0.089 | -0.064 | 0.00/0.00 | 6 | -0.027 | -0.006 | 0.00/0.00 | 1.006 | -0.79 | -0.013 |
| excess | quantity | 42 | -0.094 | -0.100 | 0.00/0.00 | 6 | -0.089 | -0.095 | 0.00/0.00 | 0.647 | -0.79 | -0.020 |
| smooth | adjective | 42 | -0.097 | -0.089 | 0.00/0.00 | 6 | -0.037 | -0.024 | 0.00/0.00 | 0.814 | -0.79 | -0.018 |
| everybody | possessive-or-pronoun | 42 | -0.099 | -0.100 | 0.00/0.00 | 6 | -0.021 | -0.024 | 0.00/0.00 | 0.490 | -0.79 | -0.021 |
| golden | adjective | 42 | -0.099 | -0.101 | 0.00/0.00 | 6 | -0.030 | -0.019 | 0.00/0.00 | 0.769 | -0.79 | -0.021 |
| only | determiner-like | 42 | -0.101 | -0.107 | 0.00/0.00 | 6 | -0.026 | -0.022 | 0.00/0.00 | 0.543 | -0.79 | -0.022 |
| lesser | quantity | 42 | -0.105 | -0.107 | 0.00/0.00 | 6 | -0.081 | -0.078 | 0.00/0.00 | 0.811 | -0.79 | -0.022 |
| second | determiner-like | 42 | -0.106 | -0.117 | 0.00/0.00 | 6 | -0.038 | -0.050 | 0.00/0.00 | 0.921 | -0.79 | -0.024 |
| former | determiner-like | 42 | -0.109 | -0.117 | 0.00/0.00 | 6 | -0.051 | -0.071 | 0.00/0.00 | 0.787 | -0.79 | -0.024 |
| anybody | possessive-or-pronoun | 42 | -0.109 | -0.115 | 0.00/0.00 | 6 | -0.066 | -0.056 | 0.00/0.00 | 0.322 | -0.79 | -0.024 |
| third | determiner-like | 42 | -0.109 | -0.112 | 0.00/0.00 | 6 | -0.058 | -0.061 | 0.00/0.00 | 0.778 | -0.79 | -0.023 |
| narrow | adjective | 42 | -0.110 | -0.122 | 0.00/0.00 | 6 | -0.079 | -0.090 | 0.00/0.00 | 0.809 | -0.79 | -0.025 |
| minimal | quantity | 42 | -0.110 | -0.121 | 0.00/0.00 | 6 | -0.065 | -0.064 | 0.00/0.00 | 0.643 | -0.79 | -0.025 |
| anyone | possessive-or-pronoun | 42 | -0.112 | -0.114 | 0.00/0.00 | 6 | -0.067 | -0.057 | 0.00/0.00 | 0.276 | -0.79 | -0.023 |
| latter | determiner-like | 42 | -0.115 | -0.121 | 0.00/0.00 | 6 | -0.087 | -0.062 | 0.00/0.00 | 0.521 | -0.79 | -0.025 |
| somebody | possessive-or-pronoun | 42 | -0.117 | -0.118 | 0.00/0.00 | 6 | -0.077 | -0.074 | 0.00/0.00 | 0.543 | -0.79 | -0.024 |
| sharp | adjective | 42 | -0.118 | -0.119 | 0.00/0.00 | 6 | -0.080 | -0.054 | 0.00/0.00 | 0.890 | -0.79 | -0.025 |
| considerable | quantity | 42 | -0.121 | -0.125 | 0.00/0.00 | 6 | -0.079 | -0.075 | 0.00/0.00 | 1.248 | -0.79 | -0.026 |

## Execution ledger

- Executed prompt keys: 1296
- Executed noun keys: 80

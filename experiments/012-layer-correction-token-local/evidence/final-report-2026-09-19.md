# Experiment 012 Report

- Run ID: `cec6c3908508f723`
- Confirmation sha256: `32892d23d8f853f6af71218a0562a6ffc7c375ae0fae81fc9566a79866964e07`
- Experiment 011 lock sha256: `769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`
- Protocol/code commit at explore: `ddbbbd3a53977aaa5eb3cc6fdd1056c05783112d`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration only)

- Replication: Experiment 010 1512 pairs, max deviation 0.00e+00; Experiment 011 142 pairs, max deviation 0.00e+00
- Identities (max relative errors): neuron_sum 2.0e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Weight-only denominators {'cardinal': 1.629910036913704, 'coordinated-adjective': 1.629910036913704, 'quantifier': 1.4960738062126115}; modelled plural totals D̂_T {cardinal: 1.745, coordinated-adjective: 1.851, quantifier: 2.213}; σ_r 0.793; defined ['cardinal', 'quantifier', 'coordinated-adjective']
- τ_c 0.207, τ_P 0.130 (leave-one-frame-out token means, 87 tokens)
- Y1 calibration (ĉ̄ vs c̄_L, leave-one-frame-out): Spearman 0.944, MAE 0.053, R² 0.773, bias 0.035; all-frames base 0.944/0.053/0.770; own base 0.948/0.039/0.890; c̄_L spread sd 0.145
- R²_LOFO over 24-token subsamples: percentiles {1: 0.450, 10: 0.629, 5: 0.580, 50: 0.770} (frozen floor 0.5)
- Y2 calibration (q̂̄ vs P̄1, leave-one-frame-out): Spearman 0.982, MAE 0.035, R² 0.979; g_E vs P̄1 0.903/0.091; g_E vs q̄_T 0.928/0.065; q̂ vs q̄_T 0.933/0.073
- Pairs (leave-one-frame-out): Y1 0.881/0.089; per template {cardinal: 0.777/0.082, coordinated-adjective: 0.823/0.079, quantifier: 0.811/0.107}
- Heads' share mean|c_H|/mean|c_M| 0.239; ladder means: base-point (own − LOFO) -0.041 (|.| 0.041), attention-input 0.020 (|.| 0.053)
- Class means (c̄_L, ĉ̄_LOFO): {adjective: 0.141/0.138, bare-adjective: 0.080/0.076, control: 0.132/0.142, determiner: 0.010/-0.013, determiner-like: 0.110/0.123, extension: 0.133/0.205, inherited: 0.188/0.219, number-neutral: 0.044/0.050, numeral: 0.361/0.454, original-cue: 0.117/0.146, plural-numeral: 0.353/0.437, plural-quantity: 0.142/0.183, possessive-or-pronoun: 0.059/0.070, quantity: 0.232/0.265, singular-selecting: 0.039/-0.010}

- `L01.MLP` top-20 neurons by mean |term|: cardinal: [1429, 473, 102, 1350, 490, 1511, 768, 839, 1845, 931]…; coordinated-adjective: [473, 1429, 1413, 102, 334, 1350, 490, 231, 839, 918]…; quantifier: [762, 912, 315, 389, 918, 839, 1845, 1742, 1097, 79]…; template overlap Jaccard {cardinal|coordinated-adjective: 0.43, cardinal|quantifier: 0.18, quantifier|coordinated-adjective: 0.11}
- `L02.MLP` top-20 neurons by mean |term|: cardinal: [1987, 1726, 1558, 1461, 434, 1102, 129, 1311, 1310, 1073]…; coordinated-adjective: [1987, 1726, 1461, 1558, 1562, 1102, 434, 1680, 1859, 1104]…; quantifier: [1987, 1726, 129, 1102, 815, 791, 1484, 1310, 322, 554]…; template overlap Jaccard {cardinal|coordinated-adjective: 0.54, cardinal|quantifier: 0.29, quantifier|coordinated-adjective: 0.25}

## Lock

- Candidate lock sha256 `830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb`; predictions sha256 `f32ce4da97544f395280cfccb800b90f2c1a1cf23ec524ea2eba3dc7069a811f`

## Confirmation — `LAYER_CORRECTION_TOKEN_LOCAL_MLP | COMPOSITE_PREDICTS_P1`

- Valid frames 6/6 (min 4); scored tokens 24 (min 16)
  - cardinal-012-1: valid (plural head change 1.855, cue effect 77/72; own-base plural total 1.798 vs measured 1.680)
  - cardinal-012-2: valid (plural head change 1.306, cue effect 79/72; own-base plural total 1.855 vs measured 1.723)
  - coordinated-adjective-012-1: valid (plural head change 2.706, cue effect 79/72; own-base plural total 1.831 vs measured 1.846)
  - coordinated-adjective-012-2: valid (plural head change 2.526, cue effect 79/72; own-base plural total 1.732 vs measured 1.879)
  - quantifier-012-1: valid (plural head change 1.715, cue effect 79/72; own-base plural total 1.896 vs measured 1.915)
  - quantifier-012-2: valid (plural head change 2.205, cue effect 79/72; own-base plural total 1.963 vs measured 1.979)
- Y1 (ĉ̄ vs c̄_L over 24 scored tokens): Spearman 0.864 (≥ 0.8), MAE 0.061 (≤ τ_c 0.207), R² 0.795 (≥ 0.5), bias -0.002 → pass 
- Y2 (q̂̄ vs P̄1 over 24 scored tokens): Spearman 0.922 (≥ 0.9), MAE 0.062 (≤ τ_P 0.130), R² 0.917 → pass 
- Descriptive: ĉ̄ vs c̄_M 0.850/0.079/0.599; own-base ĉ̄ vs c̄_L 0.950/0.051/0.872; heads' share 0.227 (exposed 0.239); ladder means base-point -0.037 (|.| 0.047), attention-input 0.051 (|.| 0.065), heads -0.013
- Reported beside (never judged): g_E vs P̄1 0.856/0.127; g_E vs q̄_T 0.883/0.068; q̂ vs q̄_T 0.956/0.079; q̂' vs P1' 0.922/0.061
- Class means (c̄_L, locked ĉ̄): {adjective: 0.163/0.134, determiner-like: 0.097/0.071, numeral: 0.429/0.469, possessive-or-pronoun: 0.157/0.108, quantity: 0.223/0.271}
- Block-2 top-20 neuron overlap exposed vs fresh (Jaccard): {cardinal: 0.33, coordinated-adjective: 0.54, quantifier: 0.67}

| token | class | scored | valid frames | predicted ĉ | measured c_L | c_M | c_H | own-base ĉ | base-point | attention-input | ĉ(∥) | ĉ(⊥) | g_E | q̂ | P1 | q_T | contrast | behavior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thy | possessive-or-pronoun | True | 6 | -0.077 | -0.079 | 0.018 | -0.097 | -0.108 | -0.031 | 0.126 | 0.024 | 0.036 | 0.579 | 0.394 | 0.422 | 0.396 | -2.184 | -2.550 |
| which | determiner-like | True | 6 | -0.055 | 0.066 | 0.092 | -0.026 | -0.043 | 0.011 | 0.135 | 0.009 | 0.143 | 0.464 | 0.317 | 0.449 | 0.358 | -2.800 | -2.957 |
| what | determiner-like | True | 6 | 0.027 | 0.145 | 0.169 | -0.024 | 0.059 | 0.032 | 0.110 | 0.010 | 0.228 | 0.564 | 0.473 | 0.606 | 0.568 | -3.680 | -3.982 |
| whatever | determiner-like | True | 6 | 0.049 | 0.126 | 0.117 | 0.008 | 0.055 | 0.006 | 0.062 | -0.004 | 0.166 | 0.458 | 0.401 | 0.500 | 0.439 | -2.816 | -2.429 |
| least | quantity | True | 6 | 0.049 | 0.111 | 0.081 | 0.029 | 0.063 | 0.014 | 0.019 | 0.009 | 0.171 | 0.418 | 0.362 | 0.448 | 0.470 | -2.791 | -3.442 |
| empty | adjective | True | 6 | 0.079 | 0.160 | 0.194 | -0.034 | 0.097 | 0.019 | 0.097 | 0.019 | 0.264 | 0.551 | 0.504 | 0.610 | 0.609 | -3.150 | -3.898 |
| scarce | quantity | True | 6 | 0.099 | 0.035 | 0.008 | 0.027 | 0.018 | -0.081 | -0.010 | 0.028 | 0.238 | 0.770 | 0.693 | 0.686 | 0.667 | -3.701 | -4.181 |
| theirs | possessive-or-pronoun | True | 6 | 0.099 | 0.152 | 0.162 | -0.010 | 0.087 | -0.012 | 0.075 | 0.011 | 0.169 | 0.419 | 0.408 | 0.489 | 0.440 | -3.672 | -3.816 |
| young | adjective | True | 6 | 0.108 | 0.104 | 0.184 | -0.080 | 0.046 | -0.062 | 0.138 | 0.019 | 0.269 | 0.640 | 0.592 | 0.633 | 0.628 | -3.139 | -3.545 |
| black | adjective | True | 6 | 0.129 | 0.121 | 0.171 | -0.050 | 0.126 | -0.003 | 0.045 | 0.048 | 0.268 | 0.610 | 0.589 | 0.628 | 0.605 | -2.933 | -3.396 |
| half | determiner-like | True | 6 | 0.138 | 0.012 | 0.027 | -0.016 | 0.060 | -0.079 | -0.032 | 0.021 | 0.193 | 0.470 | 0.481 | 0.403 | 0.505 | -2.699 | -3.546 |
| white | adjective | True | 6 | 0.140 | 0.186 | 0.196 | -0.010 | 0.152 | 0.012 | 0.044 | 0.030 | 0.275 | 0.508 | 0.517 | 0.595 | 0.589 | -2.852 | -3.398 |
| ours | possessive-or-pronoun | True | 6 | 0.179 | 0.241 | 0.263 | -0.022 | 0.172 | -0.007 | 0.091 | -0.001 | 0.308 | 0.312 | 0.386 | 0.470 | 0.406 | -3.441 | -3.516 |
| whichever | determiner-like | True | 6 | 0.194 | 0.136 | 0.196 | -0.060 | 0.146 | -0.048 | 0.050 | 0.002 | 0.282 | 0.377 | 0.450 | 0.437 | 0.327 | -2.042 | -2.128 |
| wooden | adjective | True | 6 | 0.212 | 0.246 | 0.263 | -0.017 | 0.176 | -0.036 | 0.087 | 0.055 | 0.423 | 0.549 | 0.611 | 0.684 | 0.641 | -3.502 | -4.012 |
| hers | possessive-or-pronoun | True | 6 | 0.232 | 0.314 | 0.384 | -0.071 | 0.264 | 0.033 | 0.120 | -0.001 | 0.398 | 0.206 | 0.343 | 0.442 | 0.364 | -3.103 | -3.132 |
| sparse | quantity | True | 6 | 0.243 | 0.192 | 0.146 | 0.046 | 0.147 | -0.095 | -0.002 | 0.040 | 0.305 | 0.659 | 0.729 | 0.729 | 0.695 | -3.437 | -3.898 |
| ninety | numeral | True | 6 | 0.379 | 0.354 | 0.348 | 0.006 | 0.314 | -0.064 | 0.034 | 0.062 | 0.217 | 0.857 | 0.998 | 1.038 | 0.864 | -4.784 | -4.848 |
| manifold | quantity | True | 6 | 0.379 | 0.219 | 0.237 | -0.018 | 0.295 | -0.084 | -0.058 | 0.027 | 0.412 | 0.607 | 0.793 | 0.707 | 0.702 | -4.128 | -4.087 |
| sixty | numeral | True | 6 | 0.429 | 0.337 | 0.389 | -0.052 | 0.351 | -0.078 | 0.038 | 0.076 | 0.247 | 0.828 | 1.014 | 0.997 | 0.826 | -4.870 | -4.951 |
| eighty | numeral | True | 6 | 0.445 | 0.457 | 0.422 | 0.034 | 0.376 | -0.069 | 0.046 | 0.070 | 0.309 | 0.830 | 1.030 | 1.102 | 0.841 | -4.819 | -4.853 |
| fifty | numeral | True | 6 | 0.536 | 0.510 | 0.536 | -0.026 | 0.471 | -0.065 | 0.065 | 0.085 | 0.371 | 0.797 | 1.079 | 1.119 | 0.893 | -4.916 | -4.960 |
| seventy | numeral | True | 6 | 0.558 | 0.489 | 0.488 | 0.002 | 0.479 | -0.079 | 0.008 | 0.082 | 0.368 | 0.823 | 1.117 | 1.123 | 0.855 | -4.810 | -4.738 |
| myriad | quantity | True | 6 | 0.586 | 0.560 | 0.415 | 0.144 | 0.475 | -0.111 | -0.059 | 0.046 | 0.579 | 0.804 | 1.122 | 1.166 | 1.094 | -4.939 | -4.938 |

## Execution ledger

- Executed prompt keys: 252
- Executed noun keys: 80

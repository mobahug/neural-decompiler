# Experiment 007 Report

- Run ID: `81e578c5e6d41814`
- Confirmation sha256 (inherited from Experiment 006): `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`
- Protocol/code commit at explore: `cc560140f632dd9a5fab4b648852be12db3e438c`

## Phases

- `calibrate`: `complete`
- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — inherited responses

- Recomputed E-patch mean shifts match Experiment 006 on 192 (token, frame) pairs; max deviation 0.00e+00 (tolerance 1e-06)

## Rank selection (leave-one-cue-out, supervised cross-moment SVD)

- rank 1: error 1.0793 ± 0.1979; PCA-006 2.3049 (ratio 0.468, prediction met)
- rank 2: error 1.1560 ± 0.1948; PCA-006 2.4000 (ratio 0.482, prediction met)
- rank 3: error 1.1758 ± 0.1999; PCA-006 2.3577 (ratio 0.499, prediction met)
- rank 4: error 1.0614 ± 0.2045; PCA-006 2.0986 (ratio 0.506, prediction met)
- eligible [1], best 1, threshold 1.2773, **selected r = 1**
- Ridge-full LOCO error 0.8602 (ratio to selected 0.797); E005-scalar cue-level error 1.410
- Ridge multipliers by fold: {'a': 0.01, 'all': 0.01, 'both': 0.01, 'cardinal:pl': 0.01, 'cardinal:sg': 0.01, 'every': 0.01, 'few': 0.01, 'five': 0.01, 'four': 0.01, 'many': 0.01, 'quantifier:pl': 0.01, 'quantifier:sg': 0.01, 'some': 0.01, 'ten': 0.01, 'the': 0.01, 'three': 0.01}

- Quality gate: passed (Spearman 0.765, normalized RMSE 0.320, improvement True); τ = 3.972
- Outcome at explore: gate passed (assigned by confirm)

- Final fit: rank(C) 15, gaps {'1': 0.4543681495111287, '2': 0.3630810744502309, '3': 0.07783731824150424, '4': 0.006110401802123474}, rank(Z_T) {'cardinal': 1, 'coordinated-adjective': 1, 'quantifier': 1}

| token | measured | supervised (LOCO) | PCA-006 (LOCO) | Ridge-full (LOCO) | E005-scalar |
|---|---|---|---|---|---|
| a | 0.301 | -3.135 | -0.568 | -2.265 | -2.175 |
| all | -4.416 | -2.583 | 0.112 | -2.998 | -1.888 |
| both | -4.295 | -2.729 | -0.650 | -4.075 | -2.662 |
| cardinal:pl | -4.908 | -4.020 | -3.008 | -4.253 | -4.194 |
| cardinal:sg | -0.510 | -0.081 | -0.975 | -0.953 | 0.028 |
| every | 0.409 | -1.327 | 1.170 | -1.208 | -0.816 |
| few | -5.061 | -4.029 | -1.532 | -4.907 | -3.027 |
| five | -4.939 | -4.684 | -3.720 | -4.787 | -3.678 |
| four | -4.916 | -4.732 | -3.645 | -5.138 | -3.952 |
| many | -4.924 | -3.921 | -1.067 | -4.188 | -2.779 |
| quantifier:pl | -4.762 | -4.670 | -1.834 | -4.618 | -3.817 |
| quantifier:sg | 0.110 | 0.173 | 1.085 | -0.186 | -0.055 |
| some | -3.779 | -3.122 | -0.500 | -3.416 | -2.205 |
| ten | -4.949 | -4.263 | -3.111 | -4.346 | -3.144 |
| the | -2.266 | -3.233 | -0.407 | -0.893 | -2.303 |
| three | -4.926 | -4.546 | -3.737 | -4.972 | -4.072 |

## Lock

- Candidate lock sha256 `de0ae866a5625811ef8fcf01f60c0b563aeebb03f7c3f80e058a0bf82216dd96` (rank 1, τ 3.972)

## Confirmation — outcome `PROGRAM_NOT_SUPPORTED`

- Program `PROGRAM_FAIL` (failures ['Y1']; where ['encoding subspace']); bands hit True; fresh cue effect True

### Program (Y families)

- Y1 FAIL: Spearman 0.785, MAE 1.071 (τ 3.972), confident-sign ok False
- Y2 pass: Spearman 0.801, MAE 1.195
- Y3 pass: within τ True, positive 120/108
- Bands: 24/24 tokens inside ± τ in ≥ 5/6 frames

- Y4 e005-scalar: Y1 Spearman 0.781, MAE 1.744; Y2 Spearman 0.793, MAE 1.854
- Y4 pca-006: Y1 Spearman 0.610, MAE 2.708; Y2 Spearman 0.611, MAE 2.813
- Y4 ridge-full: Y1 Spearman 0.796, MAE 1.049; Y2 Spearman 0.795, MAE 1.182

| token | category | E-patch measured | behavior measured | selected | PCA-006 | Ridge-full | E005-scalar |
|---|---|---|---|---|---|---|---|
| another | inherited | -0.319 | -0.254 | -2.758 | -0.518 | -1.255 | -2.068 |
| any | inherited | -2.852 | -3.435 | -2.416 | 0.779 | -1.652 | -1.362 |
| big | control | -4.640 | -4.715 | -3.537 | -0.765 | -2.856 | -2.239 |
| countless | quantity | -5.049 | -5.089 | -4.723 | -1.231 | -4.388 | -3.234 |
| dozen | quantity | -4.841 | -4.746 | -4.369 | -2.450 | -4.840 | -3.070 |
| eight | numeral | -5.139 | -5.233 | -5.321 | -4.376 | -5.405 | -3.993 |
| fewer | quantity | -5.332 | -5.338 | -3.943 | -0.866 | -3.861 | -2.593 |
| fresh | control | -3.852 | -3.942 | -3.270 | -0.955 | -2.481 | -2.131 |
| hundred | inherited | -5.038 | -4.898 | -4.213 | -2.403 | -4.346 | -2.743 |
| multiple | inherited | -4.919 | -5.021 | -4.307 | -1.812 | -4.081 | -3.459 |
| nine | numeral | -5.105 | -5.222 | -5.374 | -4.430 | -5.525 | -3.956 |
| no | inherited | -3.604 | -3.975 | -3.152 | -0.016 | -2.219 | -1.950 |
| numerous | inherited | -4.769 | -4.704 | -4.659 | -1.418 | -4.310 | -3.323 |
| old | control | -3.824 | -4.016 | -3.253 | -0.553 | -3.076 | -2.153 |
| red | control | -3.947 | -4.026 | -3.523 | -1.415 | -2.898 | -2.199 |
| seven | numeral | -5.225 | -5.262 | -5.508 | -4.667 | -5.481 | -4.070 |
| single | inherited | -3.035 | -3.635 | -2.269 | -0.903 | -2.339 | -1.554 |
| six | numeral | -5.089 | -5.196 | -5.287 | -4.428 | -5.253 | -4.036 |
| that | determiner | -2.706 | -2.652 | -3.106 | -0.224 | -2.231 | -2.027 |
| these | determiner | -5.185 | -5.323 | -3.743 | -0.496 | -3.982 | -2.646 |
| this | determiner | 0.007 | 0.024 | -2.795 | 0.020 | -1.586 | -1.914 |
| those | determiner | -5.093 | -4.989 | -3.407 | -0.528 | -3.876 | -2.288 |
| twelve | inherited | -4.990 | -5.040 | -5.019 | -4.029 | -5.225 | -3.827 |
| various | quantity | -4.664 | -4.703 | -4.718 | -1.381 | -4.412 | -3.525 |

### Circuit families (reported; not in the outcome)

- Fresh nouns on manifest frames: FAIL (failures ['P3'])
- Fresh frames: FAIL (failures ['P3'])
- C002 review eligible: False

#### Fresh nouns on manifest frames

- Cue effect 120/120 (floor 114)
- P1 pass, P3 FAIL, P4 pass, P5 pass, P7 pass, P8 pass, P9 pass
- P1 0.981; P3 F 0.781, signs 120/114, corr 0.696; P4 0.887; P5 0.825/0.865; P7 -0.010/0.984; P8 0.822; P9 0.900/0.987/0.074

#### Fresh frames

- Cue effect 120/120 (floor 108)
- P1 pass, P3 FAIL, P4 pass, P5 pass, P7 pass, P8 pass, P9 pass
- P1 0.983; P3 F 0.773, signs 120/114, corr 0.840; P4 0.913; P5 0.793/0.845; P7 -0.000/0.983; P8 0.873; P9 0.934/1.193/-0.623

## Execution ledger

- Executed prompt keys: 198
- Executed noun keys: 80

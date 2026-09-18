# Experiment 007 Report

- Run ID: `81e578c5e6d41814`
- Confirmation sha256 (inherited from Experiment 006): `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`
- Protocol/code commit at explore: `cc560140f632dd9a5fab4b648852be12db3e438c`

## Phases

- `calibrate`: `complete`
- `confirm`: `not_started`
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

## Execution ledger

- Executed prompt keys: 36
- Executed noun keys: 60

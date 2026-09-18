# Experiment 006 Report

- Run ID: `b09a022e021ba19c`
- Confirmation sha256: `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521`
- Protocol/code commit at explore: `adea65f0d36b6121b16ba29241505c5e9dcfffdd`

## Phases

- `calibrate`: `not_started`
- `confirm`: `not_started`
- `explore`: `complete`
- `lock`: `not_started`
- `report`: `not_started`

## Tier A — circuit on the exposed pool

- Cue effect: 708/708 positive (floor 673)
- P1 pass, P3 FAIL, P4 pass, P5 pass, P7 FAIL, P8 pass, P9 pass
- P1 0.978; P3 F 0.804, signs 708/673, corr 0.756; P4 0.915; P5 0.819/0.870; P7 —/—; P8 0.813; P9 0.917/1.037/0.190

## Rank selection (leave-one-cue-out)

- rank 1: error 2.3049 ± 0.2962
- rank 2: error 2.4000 ± 0.3160
- rank 3: error 2.3577 ± 0.2745
- rank 4: error 2.0986 ± 0.3021
- eligible [1], best 1, threshold 2.6011, **selected r = 1**
- E005-scalar cue-level error: 1.4100

- Quality gate: FAILED (Spearman 0.800, normalized RMSE 0.623, improvement True); τ = 7.724

| token | measured | predicted (LOCO) | E005-scalar |
|---|---|---|---|
| a | 0.301 | -0.568 | -2.175 |
| all | -4.416 | 0.112 | -1.888 |
| both | -4.295 | -0.650 | -2.662 |
| cardinal:pl | -4.908 | -3.008 | -4.194 |
| cardinal:sg | -0.510 | -0.975 | 0.028 |
| every | 0.409 | 1.170 | -0.816 |
| few | -5.061 | -1.532 | -3.027 |
| five | -4.939 | -3.720 | -3.678 |
| four | -4.916 | -3.645 | -3.952 |
| many | -4.924 | -1.067 | -2.779 |
| quantifier:pl | -4.762 | -1.834 | -3.817 |
| quantifier:sg | 0.110 | 1.085 | -0.055 |
| some | -3.779 | -0.500 | -2.205 |
| ten | -4.949 | -3.111 | -3.144 |
| the | -2.266 | -0.407 | -2.303 |
| three | -4.926 | -3.737 | -4.072 |

## Execution ledger

- Executed prompt keys: 36
- Executed noun keys: 60

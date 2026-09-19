# Experiment 013 — preregistered predictions (frozen-pattern attention paths; residual of the Experiment 012 model)

- Lock run `db79dab2e9f63182` at commit `cc14911f7efe27f7e60c26b132555273a4bf64bb`; confirmation set sha256 `25f891e0a52da6cf60cd428f191010bf5cdc02d279b495b699f8cecc3e6c8f06`; Experiment 012 lock sha256 `830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb`
- Frozen tolerances τ_r 0.07 (Y1/Y2 MAE ceiling, token means), τ_A 0.106 (Y3, pairs); floors Spearman ≥ 0.8 and R² ≥ 0.5 (Y1/Y2), Spearman ≥ 0.7 (c_H) and ≥ 0.9 (c_M) (Y3); c_H degenerate-spread threshold 0.017
- Y1 table: 864 rows (fresh tokens × exposed frames); Y2 rows are computed at confirm stage 1 from each fresh frame's reference state and digested before any fresh cue prompt

| token | class | frames | ĉ_012 | base-point | frozen-attention | **predicted residual r̂** | ĉ_M | ĉ_H | ĉ_L | g_E |
|---|---|---|---|---|---|---|---|---|---|---|
| same | determiner-like | 36 | 0.113 | -0.017 | 0.026 | **0.009** | 0.170 | -0.048 | 0.122 | 0.324 |
| own | determiner-like | 36 | 0.049 | -0.037 | 0.022 | **-0.015** | 0.064 | -0.029 | 0.034 | 0.540 |
| last | determiner-like | 36 | -0.007 | -0.004 | -0.010 | **-0.014** | 0.010 | -0.030 | -0.020 | 0.314 |
| next | determiner-like | 36 | -0.016 | -0.005 | 0.038 | **0.033** | 0.048 | -0.031 | 0.017 | 0.244 |
| first | determiner-like | 36 | -0.057 | 0.003 | 0.005 | **0.008** | -0.047 | -0.002 | -0.049 | 0.326 |
| zero | numeral | 36 | 0.173 | -0.035 | 0.020 | **-0.015** | 0.165 | -0.007 | 0.158 | 0.481 |
| thousand | numeral | 36 | 0.483 | -0.068 | -0.001 | **-0.069** | 0.423 | -0.009 | 0.414 | 0.627 |
| million | numeral | 36 | 0.504 | -0.067 | -0.051 | **-0.118** | 0.415 | -0.029 | 0.386 | 0.531 |
| billion | numeral | 36 | 0.432 | -0.059 | -0.030 | **-0.089** | 0.355 | -0.012 | 0.343 | 0.583 |
| trillion | numeral | 36 | 0.408 | -0.059 | -0.025 | **-0.084** | 0.323 | 0.001 | 0.324 | 0.662 |
| limited | quantity | 36 | 0.058 | -0.053 | 0.058 | **0.006** | 0.017 | 0.047 | 0.063 | 0.637 |
| surplus | quantity | 36 | 0.346 | -0.066 | -0.072 | **-0.137** | 0.186 | 0.022 | 0.208 | 0.551 |
| endless | quantity | 36 | 0.111 | -0.049 | 0.009 | **-0.040** | 0.044 | 0.027 | 0.071 | 0.749 |
| plenty | quantity | 36 | 0.383 | -0.051 | -0.020 | **-0.071** | 0.270 | 0.041 | 0.311 | 0.451 |
| vast | quantity | 36 | 0.216 | -0.054 | 0.078 | **0.023** | 0.192 | 0.047 | 0.240 | 0.704 |
| whom | possessive-or-pronoun | 36 | 0.160 | -0.019 | 0.068 | **0.049** | 0.258 | -0.050 | 0.209 | 0.408 |
| someone | possessive-or-pronoun | 36 | 0.074 | -0.010 | -0.016 | **-0.027** | 0.135 | -0.088 | 0.048 | 0.299 |
| nobody | possessive-or-pronoun | 36 | 0.123 | -0.015 | 0.030 | **0.014** | 0.134 | 0.004 | 0.138 | 0.228 |
| everyone | possessive-or-pronoun | 36 | 0.085 | -0.016 | 0.010 | **-0.006** | 0.058 | 0.020 | 0.079 | 0.284 |
| tall | adjective | 36 | 0.283 | -0.056 | 0.017 | **-0.039** | 0.263 | -0.020 | 0.244 | 0.605 |
| thick | adjective | 36 | 0.166 | -0.032 | 0.095 | **0.063** | 0.224 | 0.005 | 0.229 | 0.509 |
| quiet | adjective | 36 | 0.165 | -0.037 | 0.099 | **0.062** | 0.218 | 0.009 | 0.227 | 0.489 |
| broken | adjective | 36 | 0.125 | -0.027 | 0.060 | **0.033** | 0.203 | -0.045 | 0.158 | 0.507 |
| wide | adjective | 36 | 0.208 | -0.048 | 0.063 | **0.015** | 0.176 | 0.047 | 0.223 | 0.573 |

Per-template ĉ_012 (the frozen Experiment 012 model): same: cardinal -0.036, quantifier 0.239, coordinated-adjective 0.135; own: cardinal -0.104, quantifier 0.305, coordinated-adjective -0.054; last: cardinal -0.142, quantifier 0.232, coordinated-adjective -0.110; next: cardinal -0.116, quantifier 0.198, coordinated-adjective -0.131; first: cardinal -0.157, quantifier 0.111, coordinated-adjective -0.125; zero: cardinal -0.008, quantifier 0.394, coordinated-adjective 0.134; thousand: cardinal 0.358, quantifier 0.674, coordinated-adjective 0.417; million: cardinal 0.387, quantifier 0.741, coordinated-adjective 0.384; billion: cardinal 0.300, quantifier 0.641, coordinated-adjective 0.356; trillion: cardinal 0.264, quantifier 0.602, coordinated-adjective 0.358; limited: cardinal -0.186, quantifier 0.342, coordinated-adjective 0.017; surplus: cardinal 0.128, quantifier 0.687, coordinated-adjective 0.222; endless: cardinal -0.040, quantifier 0.342, coordinated-adjective 0.032; plenty: cardinal 0.251, quantifier 0.618, coordinated-adjective 0.278; vast: cardinal 0.050, quantifier 0.512, coordinated-adjective 0.086; whom: cardinal 0.014, quantifier 0.378, coordinated-adjective 0.088; someone: cardinal -0.037, quantifier 0.224, coordinated-adjective 0.035; nobody: cardinal -0.023, quantifier 0.248, coordinated-adjective 0.146; everyone: cardinal -0.056, quantifier 0.317, coordinated-adjective -0.008; tall: cardinal 0.141, quantifier 0.486, coordinated-adjective 0.221; thick: cardinal 0.049, quantifier 0.354, coordinated-adjective 0.096; quiet: cardinal 0.005, quantifier 0.415, coordinated-adjective 0.075; broken: cardinal -0.033, quantifier 0.262, coordinated-adjective 0.147; wide: cardinal 0.049, quantifier 0.468, coordinated-adjective 0.108

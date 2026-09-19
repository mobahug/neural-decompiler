# Experiment 014 — preregistered predictions (block-2 neuron 1987; exact LayerNorm and input weights at the frozen-pattern predicted state)

- Lock run `00eef3b8bb95c8b8` at commit `bcb0bfe3e94ee24a2740ee22b4e9228789a7436d`; confirmation set sha256 `a28edf831413489b2c8ffd4710bba4d742d4e542348120b0cab90c4f3ab3796d`; Experiment 013 lock sha256 `ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`
- Floors: Y1/Y2 Spearman ≥ 0.8, R² ≥ 0.5 (token means), balanced firing accuracy ≥ 0.9 (pairs, threshold Δa ≥ 0.5); Y3 axis-only rejected iff R² < 0.3 and balanced accuracy < 0.7
- Y1 table: 1008 rows (fresh tokens × exposed frames); predicted to fire in 184 pairs (axis-only alternative: 736); Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt
- Neuron: cos(γ₂ ⊙ W_in, d̂_E) 0.241; r(W_out) / r(E(pl) − E(ref)) per template {cardinal: 0.204, coordinated-adjective: 0.204, quantifier: 0.222}

| token | class | frames | pre_ref | **predicted Δâ** | fires (fraction of frames) | Jacobian Δp̂re | axis part | off-axis part | E | MLP₁ | heads₁ | axis-only Δâ | predicted term |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dozens | numeral | 42 | -0.31 | **1.490** | 1.00 | 2.33 | 1.65 | 0.67 | 2.09 | 0.24 | -0.01 | 1.226 | 0.314 |
| hundreds | numeral | 42 | -0.31 | **1.185** | 1.00 | 2.03 | 1.48 | 0.55 | 1.84 | 0.35 | -0.16 | 1.063 | 0.250 |
| thousands | numeral | 42 | -0.31 | **1.087** | 0.98 | 1.90 | 1.55 | 0.35 | 1.63 | 0.40 | -0.14 | 1.129 | 0.229 |
| infinite | quantity | 42 | -0.31 | **0.573** | 0.57 | 1.24 | 1.24 | 0.01 | 0.95 | 0.48 | -0.18 | 0.833 | 0.121 |
| millions | numeral | 42 | -0.31 | **0.533** | 0.50 | 1.23 | 1.35 | -0.12 | 1.28 | 0.20 | -0.26 | 0.934 | 0.113 |
| billions | numeral | 42 | -0.31 | **0.444** | 0.33 | 1.16 | 1.43 | -0.27 | 1.31 | 0.22 | -0.36 | 1.015 | 0.094 |
| rough | adjective | 42 | -0.31 | **-0.089** | 0.00 | -0.95 | 1.43 | -2.38 | -0.74 | 0.45 | -0.66 | 1.006 | -0.018 |
| excess | quantity | 42 | -0.31 | **-0.094** | 0.00 | -0.19 | 1.04 | -1.23 | 0.04 | 0.32 | -0.55 | 0.647 | -0.019 |
| smooth | adjective | 42 | -0.31 | **-0.097** | 0.00 | -0.75 | 1.22 | -1.97 | -0.20 | 0.13 | -0.68 | 0.814 | -0.020 |
| everybody | possessive-or-pronoun | 42 | -0.31 | **-0.099** | 0.00 | -0.84 | 0.84 | -1.69 | -0.28 | -0.16 | -0.40 | 0.490 | -0.020 |
| golden | adjective | 42 | -0.31 | **-0.099** | 0.00 | -0.84 | 1.17 | -2.01 | -0.42 | 0.19 | -0.61 | 0.769 | -0.020 |
| only | determiner-like | 42 | -0.31 | **-0.101** | 0.00 | -0.16 | 0.92 | -1.08 | 0.40 | -0.12 | -0.43 | 0.543 | -0.021 |
| lesser | quantity | 42 | -0.31 | **-0.105** | 0.00 | -0.14 | 1.21 | -1.35 | 0.49 | -0.00 | -0.63 | 0.811 | -0.022 |
| second | determiner-like | 42 | -0.31 | **-0.106** | 0.00 | -0.71 | 1.33 | -2.04 | 0.18 | -0.47 | -0.42 | 0.921 | -0.022 |
| former | determiner-like | 42 | -0.31 | **-0.109** | 0.00 | -0.75 | 1.19 | -1.93 | 0.28 | -0.09 | -0.94 | 0.787 | -0.022 |
| anybody | possessive-or-pronoun | 42 | -0.31 | **-0.109** | 0.00 | -0.38 | 0.62 | -1.00 | -0.07 | 0.07 | -0.39 | 0.322 | -0.022 |
| third | determiner-like | 42 | -0.31 | **-0.109** | 0.00 | -0.44 | 1.18 | -1.63 | 0.24 | -0.30 | -0.39 | 0.778 | -0.022 |
| narrow | adjective | 42 | -0.31 | **-0.110** | 0.00 | -0.18 | 1.21 | -1.40 | 0.45 | -0.01 | -0.62 | 0.809 | -0.022 |
| minimal | quantity | 42 | -0.31 | **-0.110** | 0.00 | -0.51 | 1.03 | -1.55 | -0.59 | 0.61 | -0.53 | 0.643 | -0.022 |
| anyone | possessive-or-pronoun | 42 | -0.31 | **-0.112** | 0.00 | -0.36 | 0.55 | -0.91 | -0.19 | 0.29 | -0.45 | 0.276 | -0.023 |
| latter | determiner-like | 42 | -0.31 | **-0.115** | 0.00 | -0.31 | 0.89 | -1.20 | 0.27 | -0.08 | -0.49 | 0.521 | -0.023 |
| somebody | possessive-or-pronoun | 42 | -0.31 | **-0.117** | 0.00 | -0.39 | 0.90 | -1.29 | -0.03 | 0.17 | -0.53 | 0.543 | -0.024 |
| sharp | adjective | 42 | -0.31 | **-0.118** | 0.00 | -0.41 | 1.30 | -1.71 | 0.02 | 0.31 | -0.74 | 0.890 | -0.024 |
| considerable | quantity | 42 | -0.31 | **-0.121** | 0.00 | -0.31 | 1.67 | -1.99 | 0.00 | 0.28 | -0.60 | 1.248 | -0.025 |

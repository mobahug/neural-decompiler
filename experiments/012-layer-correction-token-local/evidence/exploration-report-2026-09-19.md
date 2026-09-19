# Experiment 012 Report

- Run ID: `cec6c3908508f723`
- Confirmation sha256: `32892d23d8f853f6af71218a0562a6ffc7c375ae0fae81fc9566a79866964e07`
- Experiment 011 lock sha256: `769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`
- Protocol/code commit at explore: `ddbbbd3a53977aaa5eb3cc6fdd1056c05783112d`

## Phases

- `confirm`: `not_started`
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

## Execution ledger

- Executed prompt keys: 90
- Executed noun keys: 80
